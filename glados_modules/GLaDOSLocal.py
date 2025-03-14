import io
import base64
import random
from threading import Thread, Lock
import time
from os import path, getcwd
import multiprocessing as mp
from queue import Queue
from ctypes import *
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Pattern, Union

# 3rd party imports
import requests
from pydub import AudioSegment
from pydub.playback import play
from alsaaudio import Mixer
import regex as re

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosHomeAssistant import HomeAssistantLink
from glados_modules.EggTimer import EggTimer
from glados_modules.MqttClient import MQTTClient, LEDMessageBuilder
from glados_modules.GLaDosEnums import SystemEnums, MQTTEnums, LoggingEnums, LEDHead, STTEnums
from glados_modules.WhisperSTT import AudioServerTx, LocalSTTrx


# silence some errors on the terminal
def py_error_handler(filename, line, function, err, fmt):
    pass


# silence some errors on the terminal
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)


@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary(SystemEnums.LIB_ASOUND.value)
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)


class GladosException(Exception):
    pass


class GladosLocal(Thread, MQTTClient):
    """Local instance of Glados integrating threading, MQTT communication,
    audio processing, and home assistant features.

    This class processes incoming commands, interacts with a remote LLM,
    plays audio responses, and handles timers and other interactions.

    Attributes:
        logger: Logger instance for logging.
        cmd_topic: MQTT topic for vision results.
        intensity_topic: MQTT topic for system intensity.
        topic_handler: Dictionary mapping topics to their respective handlers.
        llm: Remote LLM instance.
        last_greeting: Last greeting response issued.
        last_insult: Last insult response issued.
        last_process: Last processing response issued.
        last_question: Last question issued.
        last_qresponse: Last question response issued.
        last_fresponse: Last "fuck off" response issued.
        last_cresponse: Last cancel response issued.
        timers: Queue holding active EggTimer instances.
        configFile: Configuration file as a dictionary.
        voiceurl: URL used for text-to-speech.
        configp: Local speak configuration.
        greetings: List of greeting phrases.
        processing: List of processing phrases.
        insults: List of insult phrases.
        questions: List of questions.
        qresponse: List of question responses.
        cancel: List of cancel responses.
        vision_confidence: Confidence threshold for vision detection.
        fuck: List of "fuck off" phrases.
        mixer: Audio mixer instance.
        current_vol: Current volume level.
        sight_results: Dictionary shared between processes for vision data.
        stop: Flag to indicate when the thread should stop running.
        homeass: Home Assistant link instance.
        mp_lock: Multiprocessing lock.
        seen: Latest processed sight result.
        last_seen_human: Timestamp of the last detected human.
    """

    def __init__(self, config_file: Dict[str, Any], remote_llm: Any) -> None:
        """Initialize a new GladosLocal instance.

        Args:
            config_file: Configuration dictionary containing various settings.
            remote_llm: Remote large language model instance.
        """
        Thread.__init__(self)
        self.daemon = True  # Set this thread as a daemon
        conf_mqtt = config_file[SystemEnums.CONFIG_HEAD_MQTT.value]
        conf_STT = config_file[STTEnums.CONFIG_HEAD_STT.value]
        ip = conf_mqtt[SystemEnums.MQTT_SERVER_IP.value]
        port = int(conf_mqtt[SystemEnums.MQTT_PORT.value])
        mqtt_broker = MQTTClient.broker_tuple(ip, port)
        audio_broker = MQTTClient.broker_tuple(conf_STT[STTEnums.STT_SERVER_IP.value],
                                               int(conf_STT[STTEnums.STT_SERVER_PORT.value]))
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(
            name=self.__name__,
            console_logging=LoggingEnums.LOG_LEVEL_DEBUG.value
        )
        MQTTClient.__init__(self, ip=ip, port=port)
        self.cmd_topic: str = MQTTEnums.VISION_RESULTS_MQTT_TOPIC.value
        self.intensity_topic: str = MQTTEnums.SYSTEM_INTENSITY_TOPIC.value
        self.topic_handler: Dict[str, Callable] = {
            self.intensity_topic: self.handle_intensity
        }
        self.llm = remote_llm
        self.last_greeting: Optional[str] = None
        self.last_insult: Optional[str] = None
        self.last_process: Optional[str] = None
        self.last_question: Optional[str] = None
        self.last_qresponse: Optional[str] = None
        self.last_fresponse: Optional[str] = None
        self.last_cresponse: Optional[str] = None
        self.timers: Queue = Queue()
        self.configFile = config_file
        # TODO finish converting all of this into enums
        self.voiceurl: str = config_file[SystemEnums.CONFIG_HEAD_DEFAULT.value][
            SystemEnums.VOICE_URL.value
        ]
        self.configp: Dict[str, Any] = config_file[
            SystemEnums.CONFIG_HEAD_LOCALSPEAK.value
        ]
        root_path: str = self.configp.get("localpath", "./txt_responses")
        self.greetings: List[str] = self.llp(self.configp.get("greetings", []),
                                             root_path)
        self.processing: List[str] = self.llp(self.configp.get("processing", []),
                                             root_path)
        self.insults: List[str] = self.llp(self.configp.get("insults", []),
                                          root_path)
        self.questions: List[str] = self.llp(self.configp.get("questions", []),
                                            root_path)
        self.qresponse: List[str] = self.llp(self.configp.get("qresponses", []),
                                            root_path)
        self.cancel: List[str] = self.llp(self.configp.get("cancel", []),
                                         root_path)
        self.vision_confidence: float = float(
            self.configp.get("VisionConfidence", 0.0)
        )
        self.fuck: List[str] = self.llp(self.configp.get("fuck", []), root_path)
        self.mixer = Mixer("Master")
        self.__change_volume(
            int(config_file[SystemEnums.CONFIG_HEAD_DEFAULT.value]["VolumeLevel"])
        )
        self.current_vol: int = int(self.mixer.getvolume()[0])
        self.sight_results = mp.Manager().dict()
        self.stop: bool = False
        self.homeass = HomeAssistantLink(config_file)
        # self.homeass.get_temp()
        self.mp_lock = mp.Lock()
        self.seen: Optional[str] = None
        self.last_seen_human: float = time.time()
        # add in support to get timing maps for played audio
        self.audioTx = AudioServerTx(broker=audio_broker)
        self.localsttrx = LocalSTTrx(broker=mqtt_broker)
        # TODO setup LEFT LCD

    def handle_intensity(self) -> None:
        """Handle intensity MQTT topic messages.

        TODO: Implement the actual intensity handling.
        """
        pass

    def __random_audio(self, choice: str, last: Optional[str],
                       options_list: List[str], last_attr_name: str,
                       just_text: bool = False) -> str:
        """Play a random audio response ensuring it is not the same as the last.

        Args:
            choice: Initial chosen response.
            last: The last response used.
            options_list: List of possible responses.
            last_attr_name: Name of the attribute to update with the new response.
            just_text: If True, only process text without audio playback.

        Returns:
            The selected response string.
        """
        proc: str = self.__dedupe(choice, last, options_list)
        if not just_text:
            self.speak(proc)
        if hasattr(self, last_attr_name):
            setattr(self, last_attr_name, proc)
        return proc

    def random_response(self, category: str, last_response: Optional[str],
                        responses: List[str], last_response_attr: str,
                        just_text: bool = False) -> str:
        """Select and optionally play a random response from a category.

        Args:
            category: The name of the response category.
            last_response: The last response used in this category.
            responses: List of possible responses.
            last_response_attr: The attribute name to update with the new response.
            just_text: If True, only return text without playing audio.

        Returns:
            The selected response string.
        """
        response = self.__random_audio(
            random.choice(responses), last_response,
            responses, last_response_attr, just_text
        )
        self.logger.debug(f"Random {category}: {response}")
        return response

    def random_cancel_response(self, just_text: bool = False) -> str:
        """Generate a random cancel command response.

        Args:
            just_text: If True, only return text.

        Returns:
            The cancel response string.
        """
        return self.random_response('Cancel Command response', self.last_cresponse,
                                    self.cancel, 'last_cresponse', just_text)

    def random_question_response(self, just_text: bool = False) -> str:
        """Generate a random response for a question command.

        Args:
            just_text: If True, only return text.

        Returns:
            The question response string.
        """
        return self.random_response('Question Response', self.last_qresponse,
                                    self.qresponse, 'last_qresponse', just_text)

    def random_question(self, just_text: bool = False) -> str:
        """Generate a random question.

        Args:
            just_text: If True, only return text.

        Returns:
            The question string.
        """
        return self.random_response('Question', self.last_question,
                                    self.questions, 'last_question', just_text)

    def random_insult(self, just_text: bool = False) -> str:
        """Generate a random insult response.

        Args:
            just_text: If True, only return text.

        Returns:
            The insult string.
        """
        return self.random_response('Insult', self.last_insult,
                                    self.insults, 'last_insult', just_text)

    def random_processing(self, just_text: bool = False) -> str:
        """Generate a random processing response.

        Args:
            just_text: If True, only return text.

        Returns:
            The processing response string.
        """
        return self.random_response('Processing', self.last_process,
                                    self.processing, 'last_process', just_text)

    def random_fuck_response(self, just_text: bool = False) -> str:
        """Generate a random 'fuck off' response.

        Args:
            just_text: If True, only return text.

        Returns:
            The 'fuck off' response string.
        """
        return self.random_response('Fuck Off Response', self.last_fresponse,
                                    self.fuck, 'last_fresponse', just_text)

    def random_greeting(self, just_text: bool = False) -> str:
        """Generate a random greeting response.

        Args:
            just_text: If True, only return text.

        Returns:
            The greeting string.
        """
        return self.random_response('Greeting', self.last_greeting,
                                    self.greetings, 'last_greeting', just_text)

    def __dedupe(self, current: str, last: Optional[str],
                 options: List[str]) -> str:
        """Ensure the current choice is different from the last used.

        Args:
            current: The current candidate string.
            last: The last used string.
            options: List of possible candidate strings.

        Returns:
            A deduplicated string different from 'last'.
        """
        while current == last:
            current = random.choice(options)
        return current

    def llp(self, filename: str, root_path: str) -> List[str]:
        """Load local phrases from a file.

        Args:
            filename: The file name to load.
            root_path: The base directory path.

        Returns:
            A list of strings loaded from the file.

        Raises:
            GladosException: If the file does not exist.
        """
        file_path: str = path.abspath(path.join(root_path, filename))
        if path.isfile(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
            # Clean the strings by stripping whitespace.
            return [line.strip() for line in lines]
        else:
            msg: str = f"Unable to load file {file_path}"
            self.logger.error(msg)
            raise GladosException(msg)

    def check_local_command(self, user_prompt: str,
                              command: Union[str, Pattern]) -> bool:
        """Check if a user prompt matches a given command pattern.

        Args:
            user_prompt: The input string from the user.
            command: A regex pattern or string to search for.

        Returns:
            True if the command pattern is found in the prompt, False otherwise.
        """
        if not isinstance(command, re.Pattern):
            command = re.escape(command)
        match = re.search(command, user_prompt)
        self.logger.debug(f"Found match of {match} for prompt {user_prompt}")
        return bool(match)

    def get_seen_prompt(self) -> Optional[str]:
        """Retrieve the last seen prompt.

        Returns:
            The last seen prompt as a string, or None if not set.
        """
        return self.seen

    def play_portal1song(self) -> None:
        """Play the Portal 1 theme song."""
        self.__play_local_wav('./wav/portal_still_alive.wav')

    def play_portal2song(self) -> None:
        """Play the Portal 2 theme song."""
        self.__play_local_wav('./wav/portal2_want_you_gone.wav')

    def get_temp(self, user_prompt: str) -> bool:
        """Check for a temperature query in the user prompt and speak the
        temperature if found.

        Args:
            user_prompt: The input command from the user.

        Returns:
            True if the temperature command was detected, False otherwise.
        """
        c_str: str = r"what(?:'?s| is) the (current )?(outside )?(temp(erature)?)( outside)?\??"
        check: bool = self.__check_local_command(
            user_prompt.lower(), re.compile(c_str)
        )
        if check:
            self.speak(self.homeass.get_temp())
        return check

    def fuck_you(self, user_prompt: str) -> bool:
        """Check for a 'fuck you' command in the user prompt and respond.

        Args:
            user_prompt: The input command from the user.

        Returns:
            True if the command was detected, False otherwise.
        """
        check: bool = self.__check_local_command(user_prompt.lower(), "fuck you")
        if check:
            self.random_fuck_response()
        return check

    def translate_time(self, user_prompt: str) -> Dict[str, int]:
        """Translate a time string from the user prompt into a dictionary
        containing time units and total seconds.

        Args:
            user_prompt: The input string containing time specifications.

        Returns:
            A dictionary with keys 'hours', 'minutes', 'seconds', and
            'total_seconds'.
        """
        pattern: str = r'(\d+)\s*(hour|minute|second)s?'
        matches = re.findall(pattern, user_prompt)
        time_dict: Dict[str, int] = {f'{time_unit}s': int(value)
                                     for value, time_unit in matches}
        total_seconds: int = (
            time_dict.get('seconds', 0) +
            time_dict.get('minutes', 0) * 60 +
            time_dict.get('hours', 0) * 3600
        )
        time_dict['total_seconds'] = total_seconds
        self.logger.debug(f"User requested time: {time_dict}")
        return time_dict

    def timer(self, user_prompt: str) -> bool:
        """Set or cancel a timer based on the user prompt.

        If a timer is set, it creates an EggTimer; if a stop timer command
        is detected, it stops the timer.

        Args:
            user_prompt: The input command from the user.

        Returns:
            True if a timer command was processed, False otherwise.
        """
        user_prompt_lower: str = user_prompt.lower()
        check: bool = self.__check_local_command(
            user_prompt_lower, re.compile(r'set\s+(a\s+|the\s+)?timer')
        )
        if check:
            time_dict: Dict[str, int] = self.translate_time(user_prompt)
            egg = EggTimer(time_dict['total_seconds'], self.speak)
            egg.start()
            self.timers.put(egg)
            time_units: List[str] = []
            if 'hours' in time_dict:
                time_units.append(f"{time_dict['hours']} hours")
            if 'minutes' in time_dict:
                time_units.append(f"{time_dict['minutes']} minutes")
            if 'seconds' in time_dict:
                time_units.append(f"{time_dict['seconds']} seconds")
            time_string: str = ', '.join(time_units)
            if ',' in time_string:
                # Replace the last comma with ' and '
                parts = time_string.rsplit(', ', 1)
                time_string = ' and '.join(parts)
            self.speak(time_string)
        else:
            check = self.__check_local_command(
                user_prompt_lower, re.compile(r'(stop|cancel)\s+(the\s+|a\s+)?timer')
            )
            if check:
                if self.timers.empty():
                    msg: str = "You have no running Timers"
                    self.logger.debug(msg)
                    self.speak(msg)
                else:
                    # TODO: When stopping timers, track which one we stop...
                    t = self.timers.get()
                    t.stop()
                    t.join()
        return check

    def run(self) -> None:
        """Main loop to process sight results and handle human detection.

        Continuously processes sight data and updates the last seen human
        time accordingly.
        """
        self.last_seen_human = time.time()
        scan_room: int = 0
        while not self.stop:
            self.seen = self.process_sight(self.sight_results)
            if self.sight_results.get("person", None) is None:
                # TODO: Use human detector millimeter wave.
                # TODO: Set scan config time and number of times from config file.
                # TODO: Consider scanning for other objects.
                if (time.time() - self.last_seen_human) < 120 and scan_room <= 2:
                    # PUT SCANNING FUNCTION HERE...
                    scan_room += 1
                else:
                    time.sleep(5)
            else:
                self.last_seen_human = time.time()
                time.sleep(1)

    def __adjust_count(self, obj: List[Dict[str, Any]]) -> int:
        """Count the number of objects exceeding the vision confidence.

        Args:
            obj: A list of dictionaries containing a 'confidence' key.

        Returns:
            The count of objects with confidence greater than or equal to
            the vision threshold.
        """
        count: int = 0
        for o in obj:
            if o['confidence'] >= self.vision_confidence:
                count += 1
        return count

    def process_sight(self, seen: Dict[str, Any]) -> str:
        """Process sight results into a readable string.

        Args:
            seen: A dictionary where keys are object types and values are
                details (including detected objects and their confidence).

        Returns:
            A formatted string summarizing what is seen.
        """
        context: List[str] = ["You can see the following things in the room"]
        for item in seen.keys():
            count: int = self.__adjust_count(seen[item]["objects"])
            if count == 0:
                continue
            context.append(f"{count} {item}")
        return ", ".join(context)

    def __get_audio(self, response: str) -> Union[bytes, int]:
        """Get audio data for a text response using a remote voice service.

        Args:
            response: The text to be converted into audio.

        Returns:
            Audio content in bytes if successful, or -1 on failure.
        """
        response_text: str = ", , " + response
        rsp = base64.b64encode(response_text.encode("utf8"))
        url: str = f'{self.voiceurl}{str(rsp, "utf8")}'
        response_obj = requests.get(url)
        if response_obj.status_code == 200:
            return response_obj.content
        else:
            msg: str = "Failed to translate text"
            self.logger.debug(msg)
            return -1

    def __play_local_wav(self, wav_file: str, led: bool = True) -> None:
        """Play a local WAV file.

        Args:
            wav_file: Relative path to the WAV file.
        """
        wav_path: str = path.abspath(path.join(getcwd(), wav_file))
        self.logger.debug(f"Playing {wav_path}")
        with open(wav_path, 'rb') as wav:
            self.__play_audio(wav.read(), led=led)

    def play_ding_up(self) -> None:
        """Play the 'ding on' sound."""
        self.__play_local_wav("./wav/ding_on.wav", led=False)

    def play_ding_down(self) -> None:
        """Play the 'ding off' sound."""
        self.__play_local_wav("./wav/ding_off.wav", led=False)

    def __play_audio(self, data: bytes, led: bool = True) -> None:
        """Play audio from byte data.

        Args:
            data: Audio data in bytes.
        """
        if led is True:
            # send auto out to get converted and time mapped
            self.logger.debug("Sending audio bytes off to be processed")
            self.audioTx.send_bytes(data)
            time_map = self.localsttrx.get_segment_map(block=True)
            self.logger.debug(f"Timing map is {time_map}")
            # send timing map to led
            led_msg = {LEDHead.MSG_COMMAND_KEY.value: LEDHead.LED_LOCATION.value,
                       LEDHead.LED_LOCATION.value: {
                           LEDHead.MSG_COMMAND_KEY.value: LEDHead.ANIMATION_SPEECH_EYE_KEY.value,
                           LEDHead.MSG_COMMAND_ARGUMENTS_KEY.value: time_map}
                       }
            self.logger.debug(f"LED Message is {led_msg}")
            LEDMessageBuilder.send_led_animation(led_msg)
        self.logger.debug("Playing audio file")
        play(AudioSegment.from_file(io.BytesIO(data)))

    def speak(self, text: str) -> None:
        """Convert text to speech and play the resulting audio.

        Args:
            text: The text to be spoken.
        """
        audio_data = self.__get_audio(text)
        # Only play if valid audio data was returned.
        if isinstance(audio_data, bytes):
            self.__play_audio(audio_data)

    def __change_volume(self, level: int) -> None:
        """Change the system volume.

        Args:
            level: The new volume level to set.
        """
        self.mixer.setvolume(int(level))

    def set_volume(self, user_prompt: str) -> bool:
        """Set the volume based on a user prompt.

        This method checks for volume change commands and updates the volume
        accordingly.

        Args:
            user_prompt: The input command from the user.

        Returns:
            True if a volume command was processed, False otherwise.
        """
        check: bool = False
        pc_command: List[str] = ["set volume", "change volume"]
        user_prompt_lower: str = user_prompt.lower()
        for pc in pc_command:
            check = self.__check_local_command(user_prompt_lower, pc)
            if check:
                break
        scheck: bool = self.__check_local_command(user_prompt_lower, re.compile(r'%'))
        if scheck:
            level_matches = re.findall(r'\b\d+\b', user_prompt)
            if level_matches:
                level = int(level_matches[0])
                self.__change_volume(level)
                self.current_vol = level
                msg: str = f"I have set the volume to {level} percent"
                self.logger.debug(msg)
                self.speak(msg)
        return check

if __name__ == "__main__":
    import sys
    import configparser
    from os import path
    from glados_modules.GLaDOSGpt import GladosGPT
    configp = configparser.ConfigParser()
    if path.isfile(sys.argv[1]) is True:
        configp.read(sys.argv[1])
    else:
        raise GladosException("Unable to load file {}".format(sys.argv[1]))
    gl = GladosLocal(configp, GladosGPT)
    gl.start()
    gl.speak("Oh Its you! , , Its been a long time...")
    gl.play_ding_up()