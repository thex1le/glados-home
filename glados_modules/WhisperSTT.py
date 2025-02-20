import multiprocessing
import socket
import logging
from pathlib import Path

# 3rd Party imports
import whisper


class AudioServerTx:
    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        # Client settings
        self.SERVER_IP: str = "192.168.1.100"  # Change to server's IP
        self.PORT: int = 5000
        self.BUFFER_SIZE: int = 4096  # Chunk size

    def send_file(self, file_path: str) -> None:
        """Sends an audio file to the server."""
        file = Path(file_path)
        if not file.exists():
            logging.error(f"File not found: {file_path}")
            return

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            try:
                client_socket.connect((self.SERVER_IP, self.PORT))
                logging.info(f"Connected to {self.SERVER_IP}:{self.PORT}")
                # Send the file name first
                client_socket.sendall(file.name.encode() + b"\n")

                # Send file in chunks
                with file.open("rb") as f:
                    while chunk := f.read(self.BUFFER_SIZE):
                        client_socket.sendall(chunk)
                logging.info(f"File '{file.name}' sent successfully.")
            except Exception as e:
                logging.error(f"Error sending file: {e}")


class AudioServerRX:
    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.HOST: str = "0.0.0.0"  # Listen on all interfaces
        self.PORT: int = 5000
        self.SAVE_DIR: Path = Path("received_files")  # Directory to save received files
        self.BUFFER_SIZE: int = 4096  # Chunk size

    def start_server(self) -> None:
        """Starts the TCP server to receive audio files."""
        self.SAVE_DIR.mkdir(exist_ok=True)  # Ensure save directory exists

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow quick restarts
            server_socket.bind((self.HOST, self.PORT))
            server_socket.listen(5)  # Allow up to 5 queued connections
            logging.info(f"Server listening on {self.HOST}:{self.PORT}")

            while True:
                try:
                    conn, addr = server_socket.accept()
                    logging.info(f"Connection established with {addr}")

                    # Handle file reception
                    self.handle_client(conn)

                except Exception as e:
                    logging.error(f"Unexpected server error: {e}")

    def handle_client(self, conn: socket.socket) -> None:
        """Handles receiving the file from a connected client."""
        try:
            # Receive the file name first
            file_name: bytes = conn.recv(1024).strip()
            if not file_name:
                logging.warning("No file name received.")
                return

            file_path = self.SAVE_DIR / file_name.decode()
            logging.info(f"Receiving file: {file_path}")

            # Receive the file in chunks
            with file_path.open("wb") as file:
                while True:
                    data = conn.recv(self.BUFFER_SIZE)
                    if not data:
                        break  # End of file transfer
                    file.write(data)

            logging.info(f"File saved successfully: {file_path}")

        except Exception as e:
            logging.error(f"Error while receiving file: {e}")

        finally:
            conn.close()
            logging.info("Connection closed.")


class LocalSTT:
    def __init__(self):
        self.model = whisper.load_model("turbo")

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio("audio.mp3")
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).to(self.model.device)

        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)

        # print the recognized text
        print(result.text)