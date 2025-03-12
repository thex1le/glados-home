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
from typing import Dict, Callable

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
from glados_modules.MqttClient import MQTTClient
from glados_modules.GLaDosEnums import CameraEnum, SystemEnums, MQTTEnums, LoggingEnums

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
    def __init__(self, config_file, remote_llm):
        Thread.__init__(self)
        Thread.daemon = True
        conf_mqtt = config_file[SystemEnums.CONFIG_HEAD_MQTT.value]
        ip = conf_mqtt[SystemEnums.MQTT_SERVER_IP.value]
        port = int(conf_mqtt[SystemEnums.MQTT_PORT.value])
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        MQTTClient.__init__(self, ip=ip, port=port)
        self.cmd_topic: str = MQTTEnums.VISION_RESULTS_MQTT_TOPIC.value
        self.intensity_topic: str = MQTTEnums.SYSTEM_INTENSITY_TOPIC.value
        # TODO consider how we want to handle all 3 cameras...
        self.main_camera = CameraEnum.CAMERA_HEAD.value
        self.topic_handler: Dict[str, Callable] = {self.intensity_topic: self.handle_intensity}
        self.llm = remote_llm
        self.last_greeting = None
        self.last_insult = None
        self.last_process = None
        self.last_question = None
        self.last_qresponse = None
        self.last_fresponse = None
        self.last_cresponse = None
        self.timers = Queue()
        self.configFile = config_file
        # TODO finish converting all of this into enums
        self.voiceurl = config_file[SystemEnums.CONFIG_HEAD_DEFAULT.value][SystemEnums.VOICE_URL.value]
        self.configp = config_file[SystemEnums.CONFIG_HEAD_LOCALSPEAK.value]
        root_path = self.configp.get("localpath", "./txt_responses")
        self.greetings = self.llp(self.configp.get("greetings", list()), root_path)
        self.processing = self.llp(self.configp.get("processing", list()), root_path)
        self.insults = self.llp(self.configp.get("insults", list()), root_path)
        self.questions = self.llp(self.configp.get("questions", list()), root_path)
        self.qresponse = self.llp(self.configp.get("qresponses", list()), root_path)
        self.cancel = self.llp(self.configp.get('cancel', list()), root_path)
        self.vision_confidence = float(self.configp.get("VisionConfidence", 0.0))
        self.fuck = self.llp(self.configp.get("fuck", list()), root_path)
        self.mixer = Mixer("Master")
        self.__change_volume(int(config_file[SystemEnums.CONFIG_HEAD_DEFAULT.value]["VolumeLevel"]))
        self.current_vol = int(self.mixer.getvolume()[0])
        self.sight_results = mp.Manager().dict()
        self.stop = False
        self.homeass = HomeAssistantLink(config_file)
        # self.homeass.get_temp()
        # TODO figure out how to implement the songs
        # self.portal1song()
        self.mp_lock = mp.Lock()
        self.seen = None
        self.last_seen_human = time.time()
        # TODO setup LEFT LCD

    def handle_intensity(self):
        # TODO FIGURE OUT WHAT TO DO HERE
        pass

    def __random_audio(self, choice, last, options_list, last_attr_name, just_text=False):
        proc = self.__dedupe(choice, last, options_list)
        if just_text is False:
            self.speak(proc)
        if hasattr(self, last_attr_name):
            setattr(self, last_attr_name, proc)
        return proc

    def random_response(self, category: str, last_response: str, responses: list, last_response_attr: str,
                        just_text: bool = False) -> str:
        response = self.__random_audio(random.choice(responses), last_response,
                                       responses, last_response_attr, just_text)
        self.logger.debug(f"Random {category}: {response}")
        return response

    def random_cancel_response(self, just_text: bool = False) -> str:
        return self.random_response('Cancel Command response', self.last_cresponse,
                                    self.cancel, 'last_cresponse', just_text)

    def random_question_response(self, just_text: bool = False) -> str:
        return self.random_response('Question Response', self.last_qresponse,
                                    self.qresponse, 'last_qresponse', just_text)

    def random_question(self, just_text: bool = False) -> str:
        return self.random_response('Question', self.last_question,
                                    self.questions, 'last_question', just_text)

    def random_insult(self, just_text: bool = False) -> str:
        return self.random_response('Insult', self.last_insult,
                                    self.insults, 'last_insult', just_text)

    def random_processing(self, just_text: bool = False) -> str:
        return self.random_response('Processing', self.last_process,
                                    self.processing, 'last_process', just_text)

    def random_fuck_response(self, just_text: bool = False) -> str:
        return self.random_response('Fuck Off Response', self.last_fresponse,
                                    self.fuck, 'last_fresponse', just_text)

    def random_greeting(self, just_text: bool = False) -> str:
        return self.random_response('Greeting', self.last_greeting,
                                    self.greetings, 'last_greeting', just_text)

    def __dedupe(self, current, last, options):
        while current == last:
            current = random.choice(options)
        return current

    def llp(self, file, root_path):
        file = path.abspath(path.join(root_path, file))
        if path.isfile(file) is True:
            with open(file, 'r') as f:
                lines = f.readlines()
            # clean the strings
            clines = list()
            for i in lines:
                clines.append(i.strip())
            return clines
        else:
            msg = f"Unable to load file {file}"
            self.logger.error(msg)
            raise GladosException(msg)
        # load local phrases

    def __check_local_command(self, user_prompt, command):
        if type(command) is not re.Pattern:
            command = re.escape(command)
        match = re.search(command, user_prompt)
        return bool(match)

    def get_seen_prompt(self):
        return self.seen

    def play_portal1song(self):
        self.__play_local_wav('./wav/portal_still_alive.wav')

    def play_portal2song(self):
        self.__play_local_wav('./wav/portal2_want_you_gone.wav')

    def get_temp(self, user_prompt):
        c_str = r"what(?:'?s| is) the (current )?(outside )?(temp(erature)?)( outside)?\??"
        check = self.__check_local_command(user_prompt.lower(), re.compile(c_str))
        if check is True:
            self.speak(self.homeass.get_temp())
        return check

    def fuck_you(self, user_prompt):
        check = self.__check_local_command(user_prompt.lower(), "fuck you")
        if check is True:
            self.random_fuck_response()
        return check

    def translate_time(self, user_prompt: str) -> dict:
        pattern = r'(\d+)\s*(hour|minute|second)s?'
        matches = re.findall(pattern, user_prompt)
        time_dict = {f'{time_unit}s': int(value) for value, time_unit in matches}
        total_seconds = time_dict.get('seconds', 0) \
                        + time_dict.get('minutes', 0) * 60 \
                        + time_dict.get('hours', 0) * 3600
        time_dict['total_seconds'] = total_seconds
        self.logger.debug(f"User requested time: {time_dict}")
        return time_dict

    def timer(self, user_prompt):
        user_prompt = user_prompt.lower()
        check = self.__check_local_command(user_prompt, re.compile(r'set\s+(a\s+|the\s+)?timer'))
        if check is True:
            time_dict = self.translate_time(user_prompt)
            egg = EggTimer(time_dict['total_seconds'], self.speak)
            egg.start()
            self.timers.put(egg)
            time_units = list()
            if 'hours' in time_dict:
                time_units.append(f"{time_dict['hours']} hours")
            if 'minutes' in time_dict:
                time_units.append(f"{time_dict['minutes']} minutes")
            if 'seconds' in time_dict:
                time_units.append(f"{time_dict['seconds']} seconds")
            time_string = ', '.join(time_units)
            if ',' in time_string:
                time_string = time_string.rsplit(', ', 1)
                time_string = ' and '.join(time_string)
            self.speak(time_string)
        else:
            check = self.__check_local_command(user_prompt, re.compile(r'(stop|cancel)\s+(the\s+|a\s+)?timer'))
            if check is True:
                if self.timers.empty() is True:
                    msg = "You have no running Timers"
                    self.logger.debug(msg)
                    self.speak(msg)
                else:
                    # TODO when stopping timers track which one we stop...
                    t = self.timers.get()
                    t.stop()
                    t.join()
        return check

    def run(self):
        self.last_seen_human = time.time()
        scan_room = 0
        while self.stop is False:
            self.seen = self.process_sight(self.sight_results)
            if self.sight_results.get("person", None) is None:
                # TODO this where you will do human detector millimeter wave
                # TODO set scan config time and number of times to look in conf file
                # TODO consider scanning for other things?
                if (time.time() - self.last_seen_human) < 120 and scan_room <= 2:
                    # PUT SCANNING FUNCTION HERE...
                    scan_room += 1
                else:
                    time.sleep(5)
            else:
                self.last_seen_human = time.time()
                time.sleep(1)

    def __adjust_count(self, obj):
        count = 0
        for o in obj:
            if o['confidence'] >= self.vision_confidence:
                count += 1
        return count

    def process_sight(self, seen):
        context = ["You can see the following things in the room"]
        for item in seen.keys():
            count = self.__adjust_count(seen[item]["objects"])
            if count == 0:
                continue
            context.append(f"{count} {item}")
        return ", ".join(context)

    def __get_audio(self, response):
        response = ", , " + response
        rsp = base64.b64encode(response.encode("utf8"))
        url = '{}{}'.format(self.voiceurl, str(rsp, 'utf8'))
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            msg = "Failed to translate text"
            self.logger.debug(msg)
            return -1

    def __play_local_wav(self, wav_file):
        wav_path = path.abspath(path.join(getcwd(), f"{wav_file}"))
        self.logger.debug(f"Playing {wav_path}")
        with open(wav_path, 'rb') as wav:
            self.__play_audio(wav.read())

    def play_ding_up(self):
        self.__play_local_wav("./wav/ding_on.wav")

    def play_ding_down(self):
        self.__play_local_wav("./wav/ding_down.wav")

    def __play_audio(self, data):
        play(AudioSegment.from_file(io.BytesIO(data)))

    def speak(self, text):
        self.__play_audio(self.__get_audio(text))

    def __change_volume(self, level):
        # Set the volume
        self.mixer.setvolume(int(level))

    def set_volume(self, user_prompt):
        check = False
        pc_command = ["set volume", "change volume"]
        user_prompt = user_prompt.lower()
        for pc in pc_command:
            check = self.__check_local_command(user_prompt, pc)
            if check is True:
                break
        scheck = self.__check_local_command(user_prompt, re.compile(r'%'))
        if scheck is True:
            level = re.findall(r'\b\d+\b', user_prompt)
            # TODO figure out why vol level doesn't set correctly
            self.__change_volume(int(level[0]))
            self.current_vol = level[0]
            msg = f"I have set the volume to {level[0]} percent"
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