import socket
import sys
from typing import NamedTuple, Optional, Callable
from collections import namedtuple
from threading import Thread

# Third-party imports
import whisperx as whisper
import ffmpeg
import numpy as np

# GLaDos module imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GLaDosEnums import LoggingEnums, SystemEnums, STTEnums, MQTTEnums
from glados_modules.MqttClient import MQTTClient, SttMessageBuilder


class AudioServerTx:
    """Sends audio data as a byte stream to a remote server.

    This class creates a TCP client that connects to a broker server and sends
    audio data in byte stream format using a specified chunk size.
    """

    # Create a named tuple for broker configuration (IP and port)
    broker_tuple = namedtuple(
        SystemEnums.BROKER.value,
        [SystemEnums.BROKER_IP.value, SystemEnums.BROKER_PORT.value]
    )

    def __init__(self, broker: NamedTuple, buffer: int = 4096) -> None:
        """Initialize an instance of AudioServerTx.

        Args:
            broker (NamedTuple): Broker configuration containing 'ip' and 'port'.
            buffer (int, optional): Chunk size in bytes for sending data. Defaults to 4096.
        """
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(
            name=self.__name__,
            console_logging=LoggingEnums.LOG_LEVEL_DEBUG.value
        )
        self.broker: NamedTuple = broker
        self.buffer: int = buffer

    def send_bytes(self, byte_stream: bytes) -> None:
        """Send a byte stream to the remote server in chunks.

        This method creates a TCP socket connection to the broker server and sends
        the provided byte stream in chunks defined by the buffer size.

        Args:
            byte_stream (bytes): The audio data as a byte stream.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            try:
                client_socket.connect((self.broker.ip, self.broker.port))
                self.logger.info(f"Connected to {self.broker.ip}:{self.broker.port}")

                total_sent: int = 0
                while total_sent < len(byte_stream):
                    chunk: bytes = byte_stream[total_sent: total_sent + self.buffer]
                    client_socket.sendall(chunk)
                    total_sent += len(chunk)

                self.logger.info("Byte stream sent successfully.")
            except Exception as e:
                self.logger.error(f"Error sending byte stream: {e}")


class AudioServerRX(Thread):
    """Receives an audio byte stream from a remote client.

    This class sets up a TCP server that listens for incoming connections and
    collects data from clients into a byte stream for further processing.
    """

    # Create a named tuple for broker configuration (IP and port)
    broker_tuple = namedtuple("broker", ["ip", "port"])

    def __init__(
        self,
        broker: NamedTuple,
        buffer: int = 4096,
        callback: Optional[Callable[[bytes], None]] = None
    ) -> None:
        """Initialize an instance of AudioServerRX.

        Args:
            broker (NamedTuple): Broker configuration containing 'ip' and 'port'.
            buffer (int, optional): Buffer size in bytes for receiving data. Defaults to 4096.
            callback (Optional[Callable[[bytes], None]], optional): Callback function to process
                the received byte stream. Defaults to None, which uses a do-nothing function.
        """
        Thread.__init__(self)
        Thread.daemon = True
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(
            name=self.__name__,
            console_logging=LoggingEnums.LOG_LEVEL_DEBUG.value
        )
        if callback is None:
            callback = AudioServerRX.do_nothing
        self.callback: Callable[[bytes], None] = callback
        self.broker: NamedTuple = broker
        self.buffer: int = buffer

    def start_server(self) -> None:
        """Start the TCP server to listen for and receive a byte stream.

        The server runs in an infinite loop, accepting client connections and
        handling each connection to receive the audio byte stream.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.broker.ip, self.broker.port))
            server_socket.listen(5)
            self.logger.info(f"Server listening on {self.broker.ip}:{self.broker.port}")

            while True:
                try:
                    conn, addr = server_socket.accept()
                    self.logger.info(f"Connection established with {addr}")
                    self.handle_client(conn)
                except Exception as e:
                    self.logger.error(f"Unexpected server error: {e}")

    def run(self) -> None:
        """
        Start the TCP server Thread
        """
        self.start_server()

    @staticmethod
    def do_nothing(data: bytes) -> None:
        """A callback function that does nothing.

        Args:
            data (bytes): The received byte stream.
        """
        pass

    def handle_client(self, conn: socket.socket) -> None:
        """Handle a client connection by receiving the byte stream.

        This method continuously reads data from the client connection until no
        more data is received. The received data is accumulated into a bytearray
        and then processed by the callback function.

        Args:
            conn (socket.socket): The client socket connection.
        """
        try:
            received_bytes: bytearray = bytearray()
            while True:
                data: bytes = conn.recv(self.buffer)
                if not data:
                    break
                received_bytes.extend(data)

            self.logger.info(f"Received byte stream of length: {len(received_bytes)}")
            self.callback(received_bytes)
        except Exception as e:
            self.logger.error(f"Error while receiving byte stream: {e}")
        finally:
            conn.close()
            self.logger.info("Connection closed.")


class LocalSTT(MQTTClient):
    """Processes local speech-to-text and sends results via MQTT.

    Inherits from MQTTClient to receive audio byte streams, transcribe them using a Whisper model,
    and publish the results.
    """

    def __init__(self, broker: NamedTuple) -> None:
        """Initialize an instance of LocalSTT.

        Args:
            broker (NamedTuple): Broker configuration containing 'ip' and 'port'.
        """
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(
            name=self.__name__,
            console_logging=LoggingEnums.LOG_LEVEL_DEBUG.value
        )
        self.model = whisper.load_model(whisper_arch="large-v2", device="cuda", compute_type="float16")
        super().__init__(ip=broker.ip, port=broker.port)

    def process_audio(self, byte_stream: bytes) -> None:
        """Process an audio byte stream, perform speech-to-text transcription, and publish results.

        Args:
            byte_stream (bytes): The audio data as a byte stream.
        """
        audio: np.ndarray = LocalSTT.load_audio_from_bytes(byte_stream)
        results = self.model.transcribe(audio, batch_size=16)
        self.logger.debug(f"Detected language: {results['language']}")
        text = " ".join(results['segments'])
        self.logger.debug(f"Detected text: {text}")
        rsp = {
            STTEnums.STT_TEXT_KEY.value: text,
            STTEnums.STT_LANGUAGE_KEY.value: results["language"],
        }
        self.logger.debug(f"Detected Language is {rsp[STTEnums.STT_LANGUAGE_KEY.value]}")
        self.send_command(
            SttMessageBuilder.send_speech_to_text_message(rsp),
            MQTTEnums.STT_RESULTS_MQTT_TOPIC.value,
        )

    @staticmethod
    def load_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
        """Load audio data from a byte stream using ffmpeg.

        This method pipes the audio bytes into ffmpeg to convert them to WAV format,
        and then converts the resulting bytes to a normalized NumPy array.

        Args:
            audio_bytes (bytes): The raw audio data in bytes.

        Returns:
            np.ndarray: The audio data as a NumPy array of float32 values.
        """
        process = (
            ffmpeg.input("pipe:0")
            .output("pipe:1", format="wav", ac=1, ar="16000")
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        out, _ = process.communicate(input=audio_bytes)
        audio_np: np.ndarray = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
        return audio_np


if __name__ == "__main__":
    # Test stub
    import time
    broker = AudioServerRX.broker_tuple
    server_broker = broker("127.0.0.1", 5000)
    mqtt_broker = broker("192.168.86.28", 1883)
    lstt = LocalSTT(mqtt_broker)
    rx = AudioServerRX(server_broker, callback=lstt.process_audio)
    rx.start()
    tx = AudioServerTx(server_broker)
    with open(sys.argv[1], "rb") as f:
        tx.send_bytes(f.read())
    time.sleep(5)