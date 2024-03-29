import io
import base64
import random
from threading import Thread
import time
from os import path
import argparse
import sys
import configparser
from ctypes import *
from contextlib import contextmanager
import multiprocessing as mp
from queue import Queue

# 3rd party imports
import requests
import pyaudio
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from alsaaudio import Mixer
import regex as re
from homeassistant_api import Client

# glados imports
from GLaDOSSenses import camera as gleyes
from GLaDOSBody import GBody


class GladosException(Exception):
    pass


# silence some errors on the terminal
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)


def py_error_handler(filename, line, function, err, fmt):
    pass


c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)


@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)


with noalsaerr():
    p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=1)
stream = p.open(format=pyaudio.paFloat32, channels=2, rate=44100, output=1)


class GladosSTT(Thread):
    # glados speach to text
    def __init__(self, glocal):
        Thread.__init__(self)
        Thread.daemon = True
        self.text = None
        self.glocal = glocal
        self.mplist = list()

    def get_text(self):
        # return the text and sent it back to none for next question
        try:
            text = self.mplist.pop()
        except IndexError:
            text = None
        return text

    def parse_command(self, user_prompt):
        glados_pattern = r'(hey glados){e<=3}'
        glados_match = re.search(glados_pattern, user_prompt, re.IGNORECASE | re.BESTMATCH)
        if glados_match:
            split_index = glados_match.end()
            greeting = user_prompt[:split_index].strip()
            command = user_prompt[split_index:].strip()
            has_extra_command = bool(command)
            return {"greeting": greeting, "has_extra_command": has_extra_command, "command": command}
        else:
            return {"greeting": None, "has_extra_command": False, "command": None}

    def record(self, mp_list):
        # TODO, how do we keep things local so were not hitting google all the time...
        while True:
            print("Say 'Hey GLaDOS' to start recording your question")
            with sr.Microphone() as source:
                recognizer = sr.Recognizer()
                print("Adjusting for noise")
                recognizer.adjust_for_ambient_noise(source, .5)
                print("getting audio")
                audio = recognizer.listen(source)
                print("audio done")
                try:
                    transcription = recognizer.recognize_google(audio)
                    print("transcribe done")
                    print(transcription.lower())
                    pcommand = self.parse_command(transcription)
                    print(f'parse command is {pcommand}')
                    if pcommand["greeting"] is not None:
                        # here is where we should pause and take a longer recording
                        # for the command we need to trigger glados to talk here...
                        # TODO reconsider how this works with multithreading
                        print(pcommand)
                        if pcommand["has_extra_command"] is False:
                            greet = self.glocal.random_greeting(True)
                            rq = self.glocal.random_question(True)
                            self.glocal.speak(f"{greet}. {rq}")
                            with sr.Microphone() as source:
                                recognizer = sr.Recognizer()
                                source.pause_threshold = 1
                                audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                                transcription = recognizer.recognize_google(audio)
                            print("good user_prompt")
                            # transcript audio to test
                            # check for cancel command
                            # TODO work out how the cancel command works
                            if self.glocal._gladosLocal__check_local_command(transcription.lower(),
                                                                             re.compile(r'cancel?')) is True:
                                print('cancel true')
                                self.glocal.random_cancel_response()
                                continue
                        print(transcription)
                        mp_list.append(pcommand['command'])
                        print("list appended")
                except Exception as e:
                    print("An unknown error occurred : {}".format(e))

    def run(self):
        # use manager to run management loop
        with mp.Manager() as manager:
            self.mplist = manager.list()
            self.proc = mp.Process(target=self.record, args=(self.mplist,))
            self.proc.start()
            while True:
                time.sleep(10)


class HomeAssistantLink:
    def __init__(self, config_file):
        base = config_file['HOMEASSISTANT']
        self.token = base['token']
        self.api = base['api']
        self.weather_entity_id = base['weather_entity']

    def __get_weather(self) -> dict:
        client = Client(self.api, self.token)
        data = None
        try:
            # Fetch the state of the weather entity
            weather_data = client.get_entity(entity_id=self.weather_entity_id)
            if weather_data:
                data = weather_data
        except Exception as e:
            print("An error occurred:", e)
        return data      
    
    def get_temp(self) -> dict:
        """
        Return current temp highs and low's as a string
        """
        wdata = self.__get_weather()
        watt = wdata.state.attributes
        return "The current temperature is {}".format(watt['temperature'])


class GladosLocal(Thread):
    def __init__(self, config_file, remote_llm):
        Thread.__init__(self)
        Thread.daemon = True
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
        self.voiceurl = config_file["DEFAULT"]["VoiceUrl"]
        self.configp = config_file["LOCALSPEAK"]
        self.greetings = self.llp(self.configp.get("greetings", list()))
        self.processing = self.llp(self.configp.get("processing", list()))
        self.insults = self.llp(self.configp.get("insults", list()))
        self.questions = self.llp(self.configp.get("questions", list()))
        self.qresponse = self.llp(self.configp.get("qresponses", list()))
        self.cancel = self.llp(self.configp.get('cancel', list()))
        self.vision_confidence = float(self.configp.get("VisionConfidence", 0.0))
        self.fuck = self.llp(self.configp.get("fuck", list()))
        self.mixer = Mixer("Speaker")
        self.__change_volume(int(config_file["DEFAULT"]["VolumeLevel"]))
        self.current_vol = int(self.mixer.getvolume()[0])
        self.eyes = gleyes(config_file)
        self.eyes.start()
        self.sight_results = mp.Manager.dict()
        self.stop = False
        self.homeass = HomeAssistantLink(config_file)
        self.homeass.get_temp()
        # TODO figure out how to implement the songs
        #self.portal1song()
        self.mp_lock = mp.Lock()
        self.seen = None
        # TODO, get camera size from config file and update all libs to use that
        self.glados_body = GBody(640, 640, self.sight_results, self.mp_lock)
        self.glados_body.start()

    def __random_audio(self, choice, last, options_list, last_attr_name, just_text=False):
        proc = self.__dedupe(choice, last, options_list)
        if just_text is False:
            self.speak(proc)
        if hasattr(self, last_attr_name):
            setattr(self, last_attr_name, proc)
        return proc

    def random_cancel_response(self, just_text=False):
        return self.__random_audio(random.choice(self.cancel), 
                                   self.last_cresponse, self.cancel, 'last_cresponse', just_text)
    
    def random_question_response(self, just_text=False):
        return self.__random_audio(random.choice(self.qresponse), 
                                   self.last_qresponse, self.qresponse,'last_qresponse', just_text)

    def random_question(self, just_text=False):
        return self.__random_audio(random.choice(self.questions), 
                                   self.last_question, self.questions, 'last_question', just_text)

    def random_insult(self, just_text=False):
        return self.__random_audio(random.choice(self.insults), 
                                   self.last_insult, self.insults, 'last_insult', just_text)
    
    def random_processing(self, just_text = False):
        return self.__random_audio(random.choice(self.processing), 
                                   self.last_process, self.processing, 'last_process', just_text)
    
    def random_fuck_response(self, just_text = False):
        return self.__random_audio(random.choice(self.fuck), 
                                   self.last_fresponse, self.fuck, 'last_fresponse', just_text)
    
    def random_greeting(self, just_text = False):
        return self.__random_audio(random.choice(self.greetings),
                                   self.last_greeting, self.greetings, 'last_greeting', just_text)
    
    def __dedupe(self, current, last, options):
        while current == last:
            current = random.choice(options)
        return current

    def llp(self, file):
        if path.isfile(file) is True:
            with open(file, 'r') as f:
                lines = f.readlines()
            # clean the strings
            clines = list()
            for i in lines:
                clines.append(i.strip())
            return clines
        else:
            raise GladosException("Unable to load file {}".format(file))
        # load local phrases

    def __check_local_command(self, user_prompt, command):
        if type(command) is not re.Pattern:
            command = re.escape(command)
        match = re.search(command, user_prompt)
        return bool(match)
    
    def get_seen_prompt(self):
        return self.seen
    
    def portal1song(self):
        with open('./wav/portal_still_alive.wav', 'rb') as wav:
            self.__play_audio(wav.read())
    
    def portal2song(self):
        with open('./wav/portal2_want_you_gone.wav', 'rb') as wav:
            self.__play_audio(wav.read())
    
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
        print(time_dict)
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
                    self.speak("You have no running Timers")
                else:
                    # TODO when stopping timers track which one we stop...
                    t = self.timers.get()
                    t.stop()
                    t.join()
        return check

    def run(self):
        while self.stop is False:
            with self.mp_lock:
                self.sight_results = self.eyes.get_results()
            self.seen = self.process_sight(self.sight_results)
            if self.sight_results.get("person", None) is None:
                # TODO this where you will do human detector millimeter wave
                time.sleep(5)
            else:
                # update at 30fps for now
                time.sleep(2)
    
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
            print("Failed to translate text")
            return -1

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
            self.__change_volume(int(level[0]))
            self.current_vol = level[0]
            self.speak("I have set the volume to {} percent".format(level[0]))
        return check


class EggTimer(Thread):
    def __init__(self, duration_in_seconds, speak):
        Thread.__init__(self)
        Thread.daemon = True
        self.duration = duration_in_seconds
        self.start_time = None
        self.is_running = False
        self.speak = speak

    def timer_start(self):
        if not self.is_running:
            self.start_time = time.time()
            self.is_running = True
            print("Egg timer started for {} seconds.".format(self.duration))

    def stop(self):
        if self.is_running:
            elapsed_time = time.time() - self.start_time
            remaining_time = max(0, self.duration - elapsed_time)
            self.is_running = False
            self.speak("Timer stopped. Remaining time: {:.2f} seconds.".format(remaining_time))

    def check_remaining_time(self):
        rtn = {"remain": 0, "complete":False}
        if self.is_running:
            elapsed_time = time.time() - self.start_time
            remaining_time = max(0, self.duration - elapsed_time)
            rtn["remain"] = remaining_time
            if remaining_time == 0:
                rtn["remian"] = 0
                rtn["complete"] = True
                self.speak("Your Timer is complete")
        else:
            rtn["remain"] = 0
            rtn["complete"] = True
        return rtn
    
    def run(self):
        self.timer_start()
        while True:    
            r = self.check_remaining_time()
            print(r)
            if r["complete"] is True:
                break
            time.sleep(.2)
            

class GladosGPT(Thread):
    def __init__(self, configp, prompt):
        Thread.__init__(self)
        Thread.daemon = True
        self.real_response = None
        self.prompt = prompt
        self.configp = configp["OPENAI"]
        self.model = self.configp["model"]
        self.api_key = self.configp["apikey"]
        self.api_endpoint = self.configp["endpoint"]
        self.content = self.configp["user_prompt"]
        self.updated_content = None

    def add_prompt(self, content):
        # allow extra info to be added to the user_prompt
        self.updated_content = content

    def generate_text(self):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",}
        if self.updated_content is None:
            local_content = self.content
        else:
            local_content = f"{self.content}. {self.updated_content}"
        print(local_content)
        data = {
                "model": self.model,
                "messages": [{"role": "system", "content": local_content},
                    {"role": "user", "content": self.prompt}],
                "max_tokens": 1500}
        response = requests.post(self.api_endpoint, headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()
            self.real_response = response_json['choices'][0]['message']['content'].strip()
            print("done")
        else:
            print(f"Failed to call the API. Status code: {response.status_code}")
            print(response.text)

    def run(self):
        self.generate_text()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evil Home AI')
    parser.add_argument('-config', type=str, default=1, dest='conf', nargs=1, help='Config File')
    try: 
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    configp = configparser.ConfigParser()
    if path.isfile(args.conf[0]) is True:
        configp.read(args.conf[0])
    else:
        raise GladosException("Unable to load file {}".format(args.conf[0]))
    gl = GladosLocal(configp, GladosGPT)
    gl.start()
    gstt = GladosSTT(gl)
    gstt.start()
    local_commands = (gl.get_temp, gl.fuck_you, gl.timer, gl.set_volume)
    while True:
        prompt = gstt.get_text()
        if prompt is not None:
            cmd_bool = False
            # check for local commands
            # TODO load commands from config?
            print(f"user_prompt is in main {prompt}")
            for cmd in local_commands:
                cmd_bool = cmd(user_prompt=prompt)
                if cmd_bool is True:
                    # break the for loop
                    break
            if cmd_bool is True:
                # skip the rest on the while loop
                continue
            gladosgpt = GladosGPT(configp, prompt)
            gladosgpt.add_prompt(gl.get_seen_prompt())
            gladosgpt.start()
            time.sleep(0.2)
            while gladosgpt.real_response is None:
                gl.random_processing()
                time.sleep(0.3)
                rfunc = random.choice((gl.random_processing,
                                       gl.random_insult))
                rfunc()
            time.sleep(0.2)
            gl.speak(gladosgpt.real_response)
