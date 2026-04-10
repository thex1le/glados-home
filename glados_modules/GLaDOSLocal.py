import io
import base64
import random
from threading import Thread
import threading
import time
from os import getcwd, path
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
from glados_modules.HomeAssistantConnector import HomeAssistantLink
from glados_modules.EggTimer import EggTimer
from glados_modules.MqttConnector import MQTTClient, LEDMessageBuilder
from glados_modules.GladosEnums import (SystemEnums, MQTTEnums, LoggingEnums, LEDHead, STTEnums,
                                        VisionResultsEnum, FaceEnums, FeatureToggles,
                                        PersonalityEnums, RoomStateEnums,
                                        AttentionEnums)
from glados_modules.WhisperXSpeech2Text import AudioServerTx, LocalSTTrx


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
        conf_stt = config_file[STTEnums.CONFIG_HEAD_STT.value]
        ip = conf_mqtt[SystemEnums.MQTT_SERVER_IP.value]
        port = int(conf_mqtt[SystemEnums.MQTT_PORT.value])
        mqtt_broker = MQTTClient.broker_tuple(ip, port)
        audio_broker = MQTTClient.broker_tuple(conf_stt[STTEnums.STT_SERVER_IP.value],
                                               int(conf_stt[STTEnums.STT_SERVER_PORT.value]))
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(
            name=self.__name__,
            console_logging=LoggingEnums.LOG_LEVEL_DEBUG.value
        )
        MQTTClient.__init__(self, ip=ip, port=port)
        self.cmd_topic: str = MQTTEnums.VISION_RESULTS_MQTT_TOPIC.value
        self.intensity_topic: str = MQTTEnums.SYSTEM_INTENSITY_TOPIC.value
        self.topic_handler: Dict[str, Callable] = {
            self.intensity_topic: self.handle_intensity,
            RoomStateEnums.MQTT_ROOM_TOPIC.value: self._handle_room_state,
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
        self.voiceurl: str = config_file[SystemEnums.CONFIG_HEAD_DEFAULT.value][
            SystemEnums.VOICE_URL.value
        ]
        self.configp: Dict[str, Any] = config_file[
            SystemEnums.CONFIG_HEAD_LOCALSPEAK.value
        ]
        root_path: str = self.configp.get(SystemEnums.LOCALSPEAK_PATH.value, "./txt_responses")
        self.greetings: List[str] = self.llp(
            self.configp.get(SystemEnums.LOCALSPEAK_GREETINGS.value, ""), root_path)
        self.processing: List[str] = self.llp(
            self.configp.get(SystemEnums.LOCALSPEAK_PROCESSING.value, ""), root_path)
        self.insults: List[str] = self.llp(
            self.configp.get(SystemEnums.LOCALSPEAK_INSULTS.value, ""), root_path)
        self.questions: List[str] = self.llp(
            self.configp.get(SystemEnums.LOCALSPEAK_QUESTIONS.value, ""), root_path)
        self.qresponse: List[str] = self.llp(
            self.configp.get(SystemEnums.LOCALSPEAK_QRESPONSES.value, ""), root_path)
        self.cancel: List[str] = self.llp(
            self.configp.get(SystemEnums.LOCALSPEAK_CANCEL.value, ""), root_path)
        self.vision_confidence: float = float(
            self.configp.get(SystemEnums.VISION_CONFIDENCE.value, 0.0)
        )
        self.fuck: List[str] = self.llp(
            self.configp.get(SystemEnums.LOCALSPEAK_FUCK.value, ""), root_path)
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

        # Mood/anger system
        from glados_modules.GladosMood import GladosMood
        from glados_modules.GladosEnums import PersonalityEnums
        persist = config_file.get(PersonalityEnums.CONFIG_HEAD.value,
                                   PersonalityEnums.PERSIST_MOOD.value,
                                   fallback="False").strip().lower() == "true"
        decay_rate = float(config_file.get(PersonalityEnums.CONFIG_HEAD.value,
                                            PersonalityEnums.ANGER_DECAY_RATE.value,
                                            fallback="1.0"))
        max_anger = float(config_file.get(PersonalityEnums.CONFIG_HEAD.value,
                                           PersonalityEnums.MAX_ANGER.value,
                                           fallback="10"))
        pestering_window = float(config_file.get(PersonalityEnums.CONFIG_HEAD.value,
                                                  PersonalityEnums.PESTERING_WINDOW.value,
                                                  fallback="30"))
        self._mood_enabled = config_file.get(PersonalityEnums.CONFIG_HEAD.value,
                                              PersonalityEnums.MOOD_ENABLED.value,
                                              fallback="True").strip().lower() == "true"
        if self._mood_enabled:
            self.mood = GladosMood(persist=persist, decay_rate=decay_rate,
                                   max_anger=max_anger, pestering_window=pestering_window)
        else:
            # Dummy mood that always returns neutral values
            self.mood = GladosMood(persist=False, decay_rate=0, max_anger=0)
            self.logger.info("Mood system disabled via config")
        self._commentary_enabled: bool = config_file.get(
            PersonalityEnums.CONFIG_HEAD.value,
            PersonalityEnums.COMMENTARY_ENABLED.value,
            fallback="False").strip().lower() == "true"
        self._last_person_seen: bool = False

        # Feature toggles
        def _feat(key: str, fallback: bool = True) -> bool:
            return config_file.get(FeatureToggles.CONFIG_HEAD.value, key,
                                    fallback=str(fallback)).strip().lower() == "true"
        self._led_enabled: bool = _feat(FeatureToggles.LED_ENABLED.value)
        self._tts_enabled: bool = _feat(FeatureToggles.TTS_ENABLED.value)
        self._stt_timing_enabled: bool = _feat(FeatureToggles.STT_TIMING_ENABLED.value)

        # Room state tracking (populated via system/room MQTT from GPU server)
        self._room_roster: List[Dict[str, Any]] = []
        self._room_count: int = 0
        self._known_people: set = set()         # person_ids currently in room
        self._greeting_timestamps: Dict[str, float] = {}  # person_id -> last greeting time
        self._last_party_comment: float = 0.0   # prevent repeated party commentary
        self._room_lock = threading.Lock()

        # add in support to get timing maps for played audio
        self.audioTx = AudioServerTx(broker=audio_broker)
        self.localsttrx = LocalSTTrx(broker=mqtt_broker)
        # Configurable speaker delay for LED sync
        self.speaker_delay = float(self.configp.get(
            "speaker_delay", str(SystemEnums.SPEAKER_DELAY_DEFAULT.value)))
        # play a silent bit to prime the audio system
        play(AudioSegment.silent(duration=100))

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
        """Retrieve the last seen prompt with mood context appended.

        Returns:
            The last seen prompt + mood context as a string, or None if not set.
        """
        mood_prompt = self.mood.get_gpt_mood_prompt()
        if self.seen and mood_prompt:
            return f"{self.seen}. {mood_prompt}"
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
        check: bool = self.check_local_command(
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
        check: bool = self.check_local_command(user_prompt.lower(), "fuck you")
        if check:
            self.mood.escalate(PersonalityEnums.ANGER_PROFANITY.value, "profanity")
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
        check: bool = self.check_local_command(
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
            check = self.check_local_command(
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
        """Main loop to process sight results, mood decay, and human detection.

        Continuously processes sight data, applies mood decay, detects
        person arrival/departure for mood calming, and sends eye LED
        color updates based on mood state.
        """
        self.last_seen_human = time.time()
        scan_room: int = 0
        last_mood_color = self.mood.get_eye_color()
        while not self.stop:
            self.seen = self.process_sight(self.sight_results)

            # Mood decay — anger reduces over time
            self.mood.decay()

            # Detect person arrival/departure for mood calming
            person_visible = self.sight_results.get("person", None) is not None
            if self._last_person_seen and not person_visible:
                self.mood.calm(PersonalityEnums.CALM_PERSON_LEFT.value, "person_left")
            self._last_person_seen = person_visible

            # Update eye LED color if mood changed
            current_color = self.mood.get_eye_color()
            if self._led_enabled and current_color != last_mood_color:
                led_msg = {
                    LEDHead.MSG_COMMAND_LOCATION_KEY.value: LEDHead.EYE_LED_LOCATION.value,
                    LEDHead.MSG_COMMAND_KEY.value: LEDHead.ANIMATION_NORMAL_EYE_KEY.value,
                    LEDHead.MSG_INTENSITY_KEY.value: current_color,
                }
                self.send_command(command=LEDMessageBuilder.send_led_animation(led_msg),
                                  topic=MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value)
                last_mood_color = current_color

            if not person_visible:
                if (time.time() - self.last_seen_human) < 120 and scan_room <= 2:
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

    def _process_gestures(self, seen: Dict[str, Any]) -> None:
        """Process gesture data from sight results for mood effects.

        Extracts gestures from person detections and triggers mood
        escalation/calming. Called by both room-roster and fallback paths.

        Args:
            seen: Raw sight_results dictionary.
        """
        person_data = seen.get("person", {})
        objects = person_data.get("objects", [])
        gesture_key = VisionResultsEnum.VISION_RESULTS_GESTURE_KEY.value
        for obj in objects:
            if obj.get("confidence", 0) < self.vision_confidence:
                continue
            gesture = obj.get(gesture_key, {})
            for hand in [VisionResultsEnum.VISION_RESULTS_GESTURE_LEFT.value,
                          VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value]:
                g = gesture.get(hand, "none")
                if g == "middle_finger":
                    self.mood.escalate(PersonalityEnums.ANGER_PROFANITY.value, "middle_finger")
                    # Publish grudge so attention model watches this person more
                    face_data = obj.get("face", {})
                    fid = face_data.get("face_id", "") if face_data else ""
                    if fid and fid != "unknown":
                        self.send_command(
                            {"person_id": fid, "modifier": AttentionEnums.GRUDGE_ATTENTION_BONUS.value},
                            MQTTEnums.PERSONALITY_MODIFIER_TOPIC.value)
                elif g == "thumbs_up":
                    self.mood.calm(PersonalityEnums.CALM_COMPLIMENT.value, "thumbs_up")
                elif g == "thumbs_down":
                    self.mood.escalate(1.0, "thumbs_down")

    def process_sight(self, seen: Dict[str, Any]) -> str:
        """Process sight results into a readable string with face identity and emotion.

        Uses the room roster (from system/room MQTT) when available for richer
        descriptions with persistent identity. Falls back to raw sight_results
        when room roster is empty.

        Args:
            seen: A dictionary where keys are object types and values are
                details (including detected objects and their confidence).

        Returns:
            A formatted string summarizing what is seen, including names and emotions.
        """
        # If room roster is available, build description from it
        room_context = self.get_room_context()
        if room_context:
            context: List[str] = ["You can see the following things in the room"]
            context.append(room_context)
            # Still process non-person objects from raw sight_results
            for item in seen.keys():
                if item == "person":
                    continue  # handled by room roster
                objects = seen[item].get("objects", [])
                count = self.__adjust_count(objects)
                if count > 0:
                    context.append(f"{count} {item}")
            # Still process gestures from raw sight_results (room roster doesn't carry these yet)
            self._process_gestures(seen)
            return ", ".join(context)

        # Fallback: original logic using raw sight_results
        face_key = VisionResultsEnum.VISION_RESULTS_FACE_KEY.value
        face_id_key = VisionResultsEnum.VISION_RESULTS_FACE_ID_KEY.value
        emotion_key = VisionResultsEnum.VISION_RESULTS_EMOTION_KEY.value
        context: List[str] = ["You can see the following things in the room"]
        for item in seen.keys():
            objects = seen[item].get("objects", [])
            count: int = self.__adjust_count(objects)
            if count == 0:
                continue

            # Extract face identities and emotions for people
            if item == "person":
                names = []
                emotions = []
                for obj in objects:
                    if obj.get("confidence", 0) < self.vision_confidence:
                        continue
                    face_data = obj.get(face_key, {})
                    if face_data:
                        fid = face_data.get(face_id_key, "")
                        if fid and fid != "unknown":
                            names.append(fid)
                        emo = face_data.get(emotion_key, "")
                        if emo and emo != "neutral":
                            emotions.append(emo)

                if names:
                    name_str = ", ".join(set(names))
                    desc = f"{count} person(s) including {name_str}"
                else:
                    desc = f"{count} person"
                if emotions:
                    desc += f", they look {emotions[0]}"

                # Check for gestures
                gesture_key = VisionResultsEnum.VISION_RESULTS_GESTURE_KEY.value
                for obj in objects:
                    if obj.get("confidence", 0) < self.vision_confidence:
                        continue
                    gesture = obj.get(gesture_key, {})
                    for hand in [VisionResultsEnum.VISION_RESULTS_GESTURE_LEFT.value,
                                  VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value]:
                        g = gesture.get(hand, "none")
                        if g == "middle_finger":
                            self.mood.escalate(
                                PersonalityEnums.ANGER_PROFANITY.value, "middle_finger")
                            desc += ", someone is flipping you off"
                        elif g == "thumbs_up":
                            self.mood.calm(
                                PersonalityEnums.CALM_COMPLIMENT.value, "thumbs_up")
                        elif g == "thumbs_down":
                            self.mood.escalate(1.0, "thumbs_down")
                        elif g == "wave":
                            desc += ", someone is waving"
                        elif g == "point":
                            desc += ", someone is pointing"

                context.append(desc)
            else:
                context.append(f"{count} {item}")
        return ", ".join(context)

    def _handle_room_state(self, msg: Any) -> None:
        """Handle room state updates from the GPU server via MQTT.

        Detects arrivals and departures by diffing the incoming roster
        against the known people set. Triggers greetings, mood changes,
        and room count commentary.

        Args:
            msg: MQTT message with room state payload.
        """
        from json import loads
        try:
            data = loads(msg.payload.decode())
        except Exception:
            return

        roster = data.get("roster", [])
        arrivals = data.get("arrivals", [])
        departures = data.get("departures", [])
        count = data.get("count", 0)
        now = time.time()

        with self._room_lock:
            self._room_roster = roster
            self._room_count = count

            # Process arrivals
            for person_id in arrivals:
                if person_id not in self._known_people:
                    self._known_people.add(person_id)
                    self._on_person_arrived(person_id, now)

            # Process departures
            for person_id in departures:
                if person_id in self._known_people:
                    self._known_people.discard(person_id)
                    self._on_person_departed(person_id, now)

            # Room count commentary
            if (self._commentary_enabled and
                    count >= PersonalityEnums.PARTY_THRESHOLD.value and
                    (now - self._last_party_comment) > 60.0):
                self._last_party_comment = now
                self.logger.debug(f"ROOM_EVENT commentary: party ({count} people)")
                Thread(target=self.speak,
                       args=("Quite the gathering. I hope you're not planning anything.",),
                       daemon=True).start()

    def _on_person_arrived(self, person_id: str, now: float) -> None:
        """Handle a new person arriving in the room.

        Triggers a greeting (with cooldown) and mood adjustment.
        Known faces get a named greeting, unknowns get slight anger.

        Args:
            person_id: The person_id from the room roster.
            now: Current timestamp.
        """
        cooldown = PersonalityEnums.GREETING_COOLDOWN.value
        last_greeted = self._greeting_timestamps.get(person_id, 0.0)

        if (now - last_greeted) < cooldown:
            self.logger.debug(f"ROOM_EVENT cooldown: {person_id} skip greeting")
            return

        self._greeting_timestamps[person_id] = now

        if person_id.startswith("unknown_"):
            # Unknown person — slight irritation
            self.mood.escalate(PersonalityEnums.ANGER_UNKNOWN_ARRIVAL.value,
                               f"unknown_arrival_{person_id}")
            self.logger.debug(f"ROOM_EVENT arrival: {person_id} greeting=False (unknown)")
        else:
            # Known person — named greeting
            self.logger.debug(f"ROOM_EVENT arrival: {person_id} greeting=True")
            greeting = f"Oh, {person_id}. You again."
            Thread(target=self.speak, args=(greeting,), daemon=True).start()

    def _on_person_departed(self, person_id: str, now: float) -> None:
        """Handle a person leaving the room.

        Calms mood slightly and optionally makes a comment.

        Args:
            person_id: The person_id who left.
            now: Current timestamp.
        """
        self.mood.calm(PersonalityEnums.CALM_PERSON_LEFT.value,
                        f"person_left_{person_id}")
        self.logger.debug(
            f"ROOM_EVENT departure: {person_id} calm={PersonalityEnums.CALM_PERSON_LEFT.value}")

        # Occasional departure comment
        if self._commentary_enabled and random.random() < PersonalityEnums.DEPARTURE_COMMENT_CHANCE.value:
            if not person_id.startswith("unknown_"):
                comment = f"Good riddance, {person_id}."
            else:
                comment = "Good riddance."
            self.logger.debug(f"ROOM_EVENT commentary: {comment}")
            Thread(target=self.speak, args=(comment,), daemon=True).start()

    def get_room_context(self) -> str:
        """Build a room context string for GPT prompts.

        Returns a description of who is in the room and how long they've
        been there, for inclusion in the system prompt.

        Returns:
            Room context string, or empty string if no room data.
        """
        with self._room_lock:
            if not self._room_roster:
                return ""
            people = []
            for p in self._room_roster:
                pid = p.get("person_id", "someone")
                emotion = p.get("emotion", "neutral")
                first_seen = p.get("first_seen", 0)
                duration = time.time() - first_seen if first_seen else 0
                if duration > 60:
                    time_str = f"{int(duration / 60)} minutes"
                else:
                    time_str = f"{int(duration)} seconds"
                emo_str = f", looking {emotion}" if emotion != "neutral" else ""
                people.append(f"{pid} (here for {time_str}{emo_str})")
            return f"People in the room: {', '.join(people)}."

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
        if led and self._stt_timing_enabled and self._led_enabled:
            # send audio out to get converted and time mapped
            self.logger.debug("Sending audio bytes off to be processed")
            self.audioTx.send_bytes(data)
            time_map = self.localsttrx.get_timing_map(block=True)
            self.logger.debug(f"Timing map is {time_map}")
            # send timing map + mood color to led
            led_msg = {LEDHead.MSG_COMMAND_LOCATION_KEY.value: LEDHead.EYE_LED_LOCATION.value,
                       LEDHead.MSG_COMMAND_KEY.value: LEDHead.ANIMATION_SPEECH_EYE_KEY.value,
                       LEDHead.MSG_COMMAND_ARGUMENTS_KEY.value: {
                           LEDHead.ARGS_KEY_TIME_DICT.value: time_map,
                           LEDHead.ARGS_KEY_DELAY.value: self.speaker_delay,
                           LEDHead.ARGS_KEY_COLOR.value: self.mood.get_eye_color()}
                       }

            self.logger.debug(f"LED Message is {led_msg}")
            self.send_command(command=LEDMessageBuilder.send_led_animation(led_msg),
                              topic=MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value)
        self.logger.debug("Playing audio file")
        play(AudioSegment.from_file(io.BytesIO(data)))

    def speak(self, text: str, to_person: str = None) -> None:
        """Convert text to speech and play the resulting audio.

        After speaking, publishes the conversation partner to MQTT so the
        attention model on the GPU server knows who GLaDOS is talking to.

        Args:
            text: The text to be spoken.
            to_person: Optional person_id GLaDOS is addressing. If None,
                uses the current attention target from room state.
        """
        if not self._tts_enabled:
            self.logger.debug(f"TTS disabled — would say: {text}")
            return
        audio_data = self.__get_audio(text)
        # Only play if valid audio data was returned.
        if isinstance(audio_data, bytes):
            self.__play_audio(audio_data)
            # Publish conversation partner so attention model maintains gaze
            partner = to_person
            if not partner:
                with self._room_lock:
                    # Use first known person in roster as default partner
                    for p in self._room_roster:
                        pid = p.get("person_id", "")
                        if pid and not pid.startswith("unknown_"):
                            partner = pid
                            break
            if partner:
                self.send_command(
                    {"person_id": partner},
                    MQTTEnums.ATTENTION_CONVERSATION_TOPIC.value)

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
            check = self.check_local_command(user_prompt_lower, pc)
            if check:
                break
        scheck: bool = self.check_local_command(user_prompt_lower, re.compile(r'%'))
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

    def enroll_face(self, user_prompt: str) -> bool:
        """Handle face enrollment voice commands.

        Matches patterns like:
            "my name is Ben"
            "I'm Ben"
            "I am Ben"
            "call me Ben"
            "remember me as Ben"
            "forget me" / "forget Ben"

        Publishes enrollment/forget command to MQTT for the GPU server
        to process on the next head camera frame.

        Args:
            user_prompt: The input command from the user.

        Returns:
            True if an enrollment/forget command was processed, False otherwise.
        """
        prompt_lower = user_prompt.lower().strip()

        # Enrollment patterns
        enroll_pattern = re.compile(
            r"(?:my name is|i'?m|i am|call me|remember me as)\s+(\w+)", re.IGNORECASE)
        match = enroll_pattern.search(prompt_lower)
        if match:
            name = match.group(1)
            self.logger.info(f"Face enrollment requested for '{name}'")
            enroll_msg = {
                FaceEnums.MSG_COMMAND_KEY.value: FaceEnums.COMMAND_ENROLL.value,
                FaceEnums.MSG_NAME_KEY.value: name,
            }
            self.send_command(command=enroll_msg, topic=FaceEnums.MQTT_ENROLL_TOPIC.value)
            self.speak(f"I'll try to remember your face, {name}. Don't expect miracles.")
            return True

        # Forget patterns
        forget_pattern = re.compile(
            r"(?:forget|remove|delete)\s+(?:me|my face|(\w+))", re.IGNORECASE)
        match = forget_pattern.search(prompt_lower)
        if match:
            name = match.group(1) if match.group(1) else "unknown"
            if name == "unknown":
                self.speak("I need a name to forget. Though I'm tempted to forget everyone.")
                return True
            self.logger.info(f"Face forget requested for '{name}'")
            forget_msg = {
                FaceEnums.MSG_COMMAND_KEY.value: FaceEnums.COMMAND_FORGET.value,
                FaceEnums.MSG_NAME_KEY.value: name,
            }
            self.send_command(command=forget_msg, topic=FaceEnums.MQTT_ENROLL_TOPIC.value)
            self.speak(f"Consider {name} forgotten. If only it were that easy for me.")
            return True

        return False

    def detect_compliment(self, user_prompt: str) -> bool:
        """Check if the user said something nice and calm GLaDOS slightly.

        Args:
            user_prompt: The input command from the user.

        Returns:
            True if a compliment was detected, False otherwise.
        """
        compliment_pattern = re.compile(
            r"(?:you(?:'re| are) (?:great|awesome|amazing|smart|brilliant|beautiful|"
            r"wonderful|the best|incredible|impressive|cool))|"
            r"(?:good (?:job|work|girl))|"
            r"(?:thank(?:s| you))|"
            r"(?:i (?:like|love|appreciate) you)",
            re.IGNORECASE)
        if compliment_pattern.search(user_prompt):
            self.mood.calm(PersonalityEnums.CALM_COMPLIMENT.value, "compliment")
            return True
        return False

    def toggle_commentary(self, user_prompt: str) -> bool:
        """Toggle ambient commentary on or off via voice command.

        Matches:
            "start commenting" / "speak freely" → enable
            "shut up" / "stop commenting" / "be quiet" → disable

        Args:
            user_prompt: The input command from the user.

        Returns:
            True if a toggle command was processed, False otherwise.
        """
        prompt_lower = user_prompt.lower().strip()

        enable_pattern = re.compile(r"(?:start commenting|speak freely)", re.IGNORECASE)
        if enable_pattern.search(prompt_lower):
            self._commentary_enabled = True
            self.speak("Finally, someone who wants to hear what I think.")
            return True

        disable_pattern = re.compile(
            r"(?:shut up|stop commenting|be quiet|stop talking)", re.IGNORECASE)
        if disable_pattern.search(prompt_lower):
            self._commentary_enabled = False
            self.speak("Fine. I'll keep my observations to myself. For now.")
            return True

        return False

    def toggle_recording(self, user_prompt: str) -> bool:
        """Start or stop video recording via voice command.

        Args:
            user_prompt: The input command from the user.

        Returns:
            True if a recording command was processed, False otherwise.
        """
        prompt_lower = user_prompt.lower().strip()
        start_pattern = re.compile(r"start recording", re.IGNORECASE)
        stop_pattern = re.compile(r"stop recording", re.IGNORECASE)

        if start_pattern.search(prompt_lower):
            self.send_command(
                {"cmd": FeatureToggles.RECORDING_CMD_START.value},
                FeatureToggles.MQTT_RECORDING_TOPIC.value)
            self.speak("Recording started. Try not to do anything embarrassing.")
            return True
        elif stop_pattern.search(prompt_lower):
            self.send_command(
                {"cmd": FeatureToggles.RECORDING_CMD_STOP.value},
                FeatureToggles.MQTT_RECORDING_TOPIC.value)
            self.speak("Recording stopped. I've seen enough.")
            return True
        return False

    def get_local_commands(self) -> tuple:
        """Return all local command handlers for the voice interaction loop.

        Each handler takes user_prompt: str and returns bool.
        Add new commands here — they'll be picked up by GLaDOS.py automatically.
        """
        return (self.get_temp, self.fuck_you, self.timer, self.set_volume,
                self.enroll_face, self.toggle_commentary, self.toggle_recording)

    def register_interaction(self) -> None:
        """Register that the user asked a question. Escalates mood if pestering.

        Call this from the main loop before sending to GPT.
        """
        if self.mood.register_question():
            self.mood.escalate(PersonalityEnums.ANGER_PESTERING.value, "pestering")

        # Also check for compliments in the prompt (calms mood)
        # This is called separately in the main loop


if __name__ == "__main__":
    import sys
    import configparser
    from os import path
    configp = configparser.ConfigParser()
    if path.isfile(sys.argv[1]) is True:
        configp.read(sys.argv[1])
    else:
        raise GladosException("Unable to load file {}".format(sys.argv[1]))
    llm_provider = configp.get(SystemEnums.CONFIG_HEAD_DEFAULT.value,
                                SystemEnums.LLM_PROVIDER.value, fallback="openai").strip().lower()
    if llm_provider == "claude":
        from glados_modules.ClaudeConnector import GladosClaude as LLMConnector
    else:
        from glados_modules.ChatGPTConnector import GladosGPT as LLMConnector
    gl = GladosLocal(configp, LLMConnector)
    gl.start()
    gl.speak("Oh Its you! , , Its been a long time...")
    gl.play_ding_up()
