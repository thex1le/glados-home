from threading import Thread
from time import sleep
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Union

# 3rd party
import regex as re

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GLaDosEnums import LoggingEnums, SystemEnums, STTEnums
from glados_modules.WhisperSTT import LocalSTTrx, AudioServerTx


class GladosSTT(Thread):
    """GLaDOS Speech-To-Text processing thread.

    This class continuously records audio input, sends it for transcription,
    and processes recognized commands.
    """

    def __init__(self, config: Dict[str, Any], glocal: Any) -> None:
        """Initialize the GladosSTT thread with configuration and global context.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing system and STT settings.
            glocal (Any): Global context object providing response methods and command handling.
        """
        super().__init__()
        self.daemon = True
        ip: str = config[SystemEnums.CONFIG_HEAD_MQTT.value][SystemEnums.MQTT_SERVER_IP.value]
        port: int = int(config[SystemEnums.CONFIG_HEAD_MQTT.value][SystemEnums.MQTT_PORT.value])
        # grab one of the broker tuples
        broker = AudioServerTx.broker_tuple
        self.mqtt_broker = broker(ip=ip, port=port)
        stt_config = config[STTEnums.CONFIG_HEAD_STT.value]
        self.audioServer_broker = broker(
            ip=stt_config[STTEnums.STT_SERVER_IP.value],
            port=stt_config[STTEnums.STT_SERVER_PORT.value]
        )
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.text: Optional[str] = None
        self.glocal = glocal
        self.mplist: List[Optional[str]] = list()
        # Local STT transaction and audio transmission; will be initialized later.
        self.localsttrx: Optional[LocalSTTrx] = None
        self.audioTx: Optional[AudioServerTx] = None
        self.proc: Optional[mp.Process] = None

    def get_text(self) -> Optional[str]:
        """Retrieve the last transcribed text from the shared list.

        Returns:
            Optional[str]: The transcribed text if available; otherwise, None.
        """
        try:
            text: Optional[str] = self.mplist.pop()
        except IndexError:
            text = None
        return text

    @staticmethod
    def parse_command(user_prompt: str) -> Dict[str, Union[Optional[str], bool]]:
        """Parse the user prompt to extract greeting and command.

        The function uses a regex pattern to detect the greeting "hey glados" (with error tolerance)
        and separates it from the remaining command.

        Args:
            user_prompt (str): The transcribed user prompt.

        Returns:
            Dict[str, Union[Optional[str], bool]]: A dictionary containing:
                - "greeting": The greeting portion if detected, else None.
                - "has_extra_command": True if additional command text exists, else False.
                - "command": The command text if present, else None.
        """
        glados_pattern: str = r'(hey glados){e<=3}'
        glados_match = re.search(glados_pattern, user_prompt, re.IGNORECASE | re.BESTMATCH)
        if glados_match:
            split_index: int = glados_match.end()
            greeting: str = user_prompt[:split_index].strip()
            command: str = user_prompt[split_index:].strip()
            has_extra_command: bool = bool(command)
            return {"greeting": greeting, "has_extra_command": has_extra_command, "command": command}
        else:
            return {"greeting": None, "has_extra_command": False, "command": None}

    def __get_audio(self, sr: Any, duration: float = 0.5, no_limit: bool = False, noise: bool = False) -> str:
        """Record audio from a microphone, send it for transcription, and return the resulting text.

        Args:
            sr (Any): The speech recognition module (typically the 'speech_recognition' module).
            duration (float, optional): Time in seconds to adjust for ambient noise. Defaults to 0.5.
            no_limit (bool, optional): If True, record without a phrase time limit. Defaults to False.
            noise (bool, optional): If True, adjust for ambient noise. Defaults to False.

        Returns:
            str: The transcribed audio as text.
        """
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            if noise:
                self.logger.debug("Adjusting for noise")
                recognizer.adjust_for_ambient_noise(source, duration)
            self.logger.debug("Recording audio")
            if no_limit:
                audio = recognizer.listen(source)
            else:
                self.logger.debug("Time limit removed on audio recording")
                source.pause_threshold = 1
                audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
            self.logger.debug("Recording audio complete & sending audio for transcription")
            self.audioTx.send_bytes(audio)
            audio_text = self.localsttrx.get_text(block=True)
            self.logger.debug(f"Transcription complete, Audio: {audio_text.lower()}")
            return audio_text

    def record(self, mp_list: List[Optional[str]]) -> None:
        """Continuously record audio, process transcription, and extract commands.

        This method initializes the local speech-to-text and audio transmission systems,
        listens for audio input, sends the audio data for transcription, and then parses the result.
        Recognized commands are appended to the shared multiprocessing list.

        Args:
            mp_list (List[Optional[str]]): A multiprocessing-managed list to store recognized commands.
        """
        # Initialize local STT and audio transmission after mp fork.
        self.localsttrx = LocalSTTrx(broker=self.mqtt_broker)
        self.audioTx = AudioServerTx(self.audioServer_broker)
        # Import the speech recognition module after process fork.
        import speech_recognition as sr
        while True:
            msg: str = "Say 'Hey GLaDOS' to start recording your question"
            self.logger.info(msg)
            # TODO: Investigate why the mic doesn't always open and audio may not be transmitted via send_bytes.
            try:
                transcription = self.__get_audio(sr, noise=True)
                pcommand: Dict[str, Union[Optional[str], bool]] = GladosSTT.parse_command(transcription)
                self.logger.debug(f"Parse command is {pcommand}")
                if pcommand["greeting"] is not None:
                    # Pause and take a longer recording for the command if necessary.
                    # TODO: Reconsider how this works with multithreading.
                    self.logger.debug(pcommand)
                    if not pcommand["has_extra_command"]:
                        greet: str = self.glocal.random_greeting(True)
                        rq: str = self.glocal.random_question(True)
                        self.glocal.speak(f"{greet}. {rq}")
                        transcription = self.__get_audio(sr, no_limit=True)
                        self.logger.debug("Good user_prompt")
                        # Check for a cancel command in the transcription.
                        # TODO: Work out how the cancel command works.
                        if self.glocal._gladosLocal__check_local_command(
                            transcription.lower(),
                            re.compile(r'cancel?')
                        ):
                            self.logger.debug("Cancel command issued")
                            self.glocal.random_cancel_response()
                            continue
                    self.logger.debug(transcription)
                    mp_list.append(pcommand["command"])
            except Exception as e:
                self.logger.error(f"An unknown error occurred: {e}")

    def run(self) -> None:
        """Run the speech-to-text processing loop in a separate process.

        This method sets up a multiprocessing manager and process to continuously run the recording
        and transcription functionality.
        """
        with mp.Manager() as manager:
            self.mplist = manager.list()
            self.proc = mp.Process(target=self.record, args=(self.mplist,))
            self.proc.start()
            while True:
                sleep(10)


if __name__ == "__main__":
    # for module debug
    from glados_modules.GLaDOSGpt import GLaDOSGpt
    from argparse import ArgumentParser
    from configparser import ConfigParser
    from GLaDOS import GladosLocal
    from os import path

    parser = ArgumentParser(description='Camera streaming via RTSP')
    parser.add_argument('-config', type=str, default=None, dest='conf', help='Config File')
    args = parser.parse_args()
    if not args.conf or not path.isfile(args.conf):
        raise Exception(f"Invalid config file: {args.conf}")
    config_p = ConfigParser()
    config_p.read(args.conf)
    gl = GladosLocal(config_p, GLaDOSGpt)
    gstt = GladosSTT(config_p, gl)
    gstt.start()
