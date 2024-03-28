import requests
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
from json import loads

#3rd party imports
import pyaudio
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from alsaaudio import Mixer
import regex as re
#glados imports
from GLaDOSSenses import camera as gleyes
from homeassistant_api import Client

class GLaDOS_Exception(Exception):
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
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=1)


class gladosSTT(Thread):
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

    def parse_command(self, prompt):
        glados_pattern = r'(hey glados){e<=3}'
        glados_match = re.search(glados_pattern, prompt, re.IGNORECASE | re.BESTMATCH)
        if glados_match:
            split_index = glados_match.end()
            greeting = prompt[:split_index].strip()
            command = prompt[split_index:].strip()
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
                    # use private method in this case to reduce code
                    #if self.glocal._gladosLocal__check_local_command(transcription.lower(), re.compile(r'glad??')) is True:
                    pcommand = self.parse_command(transcription)
                    print(f'parse command is {pcommand}')
                    if pcommand["greeting"] is not None:
                        # here is where we should pause and take a longer recording for the command we need to trigger glados to talk here...
                        # reconsider how this works with multithreading
                        print(pcommand)
                        if pcommand["has_extra_command"] is False:
                            greet = self.glocal.random_greeting(True)
                            rq = self.glocal.random_question(True)
                            self.glocal.speak(f"{greet}. {rq}")
                            with sr.Microphone() as source:
                                recognizer = sr.Recognizer()
                                source.pause_threshold = 1
                                audio = recognizer.listen(source,phrase_time_limit=None, timeout=None)
                                transcription = recognizer.recognize_google(audio)
                            print("good prompt")
                            #transcript audio to test 
                            # check for cancel command
                            # TODO work out how the cancel command works
                            if self.glocal._gladosLocal__check_local_command(transcription.lower(), re.compile(r'cancel?')) is True:
                                print('cancel true')
                                self.glocal.random_cancel_response()
                                continue
                        print(transcription)
                        #mp_list.append(transcription)
                        mp_list.append(pcommand['command'])
                        print("list appended")
                except Exception as e:
                 print("An error ocurred : {}".format(e))

    def run(self):
        # use manager to run managment loop
        with mp.Manager() as manager:
            self.mplist = manager.list()
            self.proc = mp.Process(target=self.record, args=(self.mplist,))
            self.proc.start()
            while True:
                time.sleep(10)

class HomeAssistantLink:
    def __init__(self, configFile):
        base = configFile['HOMEASSISTANT']
        self.token = base['token']
        self.api = base['api']
        self.weather_entity_id = base['weather_entity']

    def __get_weather(self) -> dict:
        client = Client(self.api, self.token)
        data = None
        try:
            # Fetch the state of the weather entity
            weather_data = client.get_entity(entity_id = self.weather_entity_id)
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
        return str(f"The current temperature is {watt['temperature']}")


class gladosLocal(Thread):
    def __init__(self, configFile, remoteLLM):
        Thread.__init__(self)
        Thread.daemon = True
        self.llm = remoteLLM
        self.last_greeting = None
        self.last_insult = None
        self.last_process = None
        self.last_question = None
        self.last_qresponse = None
        self.last_fresponse = None
        self.last_cresponse = None
        self.timers = Queue()
        self.configFile = configFile
        self.voiceurl = configFile["DEFAULT"]["VoiceUrl"]
        self.configp = configFile["LOCALSPEAK"]
        self.greetings = self.llp(self.configp.get("greetings", list()))
        self.processing = self.llp(self.configp.get("processing", list()))
        self.insults = self.llp(self.configp.get("insults", list()))
        self.questions = self.llp(self.configp.get("questions", list()))
        self.qresponse = self.llp(self.configp.get("qresponses", list()))
        self.cancel = self.llp(self.configp.get('cancel', list()))
        self.vision_confidence = float(self.configp.get("VisionConfidence", 0.0))
        self.fuck = self.llp(self.configp.get("fuck", list()))
        self.mixer = Mixer("Speaker")
        self.__change_volume(int(configFile["DEFAULT"]["VolumeLevel"]))
        self.currentvol = int(self.mixer.getvolume()[0])
        self.eyes = gleyes(configFile)
        self.eyes.start()
        self.sight_results = None
        self.stop = False
        self.seen = None
        self.homeass = HomeAssistantLink(configFile)
        self.homeass.get_temp()
        #self.portal1song()

    def __gen_random_short_greeting(self) -> list:
        # TODO make this work, there is multiple problems getting back predictable responses in a parseable format
        #llmi  = self.llm(self.configFile, "Generate 20 short greetings, do not number them and return them in a csv formated string")
        llmi  = self.llm(self.configFile, "Generate 20 short greetings, do not number them and return them as a base64 encoded json string")
        llmi.start()
        while llmi.real_response is None:
            time.sleep(.1)
        import pdb
        pdb.set_trace()
        #llmi.real_response.replace('"','').split('.')
        # run json loads on it twice, not sure why but it strips the escapes and works
        sgreeting = loads(loads(llmi.real_response)).values()
        # more hackery since list() on the dict_values throws an error but this works to get the item
        for i in sgreeting:
            y = list(i)

    def __random_audio(self, choice, last, options_list, just_text = False):
        proc = self.__dedupe(choice, last, options_list)
        if just_text is False:
            self.speak(proc)
        last = proc
        return proc

    def random_cancel_response(self, just_text = False):
        return self.__random_audio(random.choice(self.cancel), 
                                   self.last_cresponse, self.cancel, just_text)
    
    def random_question_response(self, just_text = False):
        return self.__random_audio(random.choice(self.qresponse), 
                                   self.last_qresponse, self.qresponse, just_text)

    def random_question(self, just_text = False):
        return self.__random_audio(random.choice(self.questions), 
                                   self.last_question, self.questions, just_text)

    def random_insult(self, just_text = False):
        return self.__random_audio(random.choice(self.insults), 
                                   self.last_insult, self.insults, just_text)
    
    def random_processing(self, just_text = False):
        return self.__random_audio(random.choice(self.processing), 
                                   self.last_process, self.processing, just_text)
    
    def random_fuck_response(self, just_text = False):
        return self.__random_audio(random.choice(self.fuck), 
                                   self.last_fresponse, self.fuck, just_text)
    
    def random_greeting(self, just_text = False):
        return self.__random_audio(random.choice(self.greetings),
                                   self.last_greeting, self.greetings, just_text)
    
    def __dedupe(self, current, last, options):
        while current == last:
            current = random.choice(options)
        last = current
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
            raise GLaDOS_Exception("Unable to load file {}".format(file))
        # load local phrases

    def __check_local_command(self, prompt, command):
        if type(command) != re.Pattern:
            command = re.escape(command)
        match = re.search(command, prompt)
        return bool(match)
    
    def get_seen_prompt(self):
        return self.seen
    
    def portal1song(self):
        with open('./wav/portal_still_alive.wav', 'rb') as wav:
            self.__play_audio(wav.read())
    
    def portal2song(self):
        with open('./wav/portal2_want_you_gone.wav', 'rb') as wav:
            self.__play_audio(wav.read())
    
    def get_temp(self, prompt):
        check = self.__check_local_command(prompt.lower(), re.compile(r"what(?:'?s| is) the (current )?(outside )?(temp(erature)?)( outside)?\??"))
        if check is True:
            self.speak(self.homeass.get_temp())
        return check

    def fuckyou(self, prompt):
        check = self.__check_local_command(prompt.lower(), "fuck you")
        if check is True:
            self.random_fuck_response()
        return check
    
    def translate_time(self, prompt: str) -> dict:
        pattern = r'(\d+)\s*(hour|minute|second)s?'
        matches = re.findall(pattern, prompt)
        time_dict = {f'{time_unit}s': int(value) for value, time_unit in matches}
        total_seconds = time_dict.get('seconds', 0) \
                    + time_dict.get('minutes', 0) * 60 \
                    + time_dict.get('hours', 0) * 3600
        time_dict['total_seconds'] = total_seconds
        print(time_dict)
        return time_dict

    def timer(self, prompt):
        prompt = prompt.lower()
        check = self.__check_local_command(prompt, re.compile(r'set\s+(a\s+|the\s+)?timer'))
        if check is True:
            time_dict = self.translate_time(prompt)
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
            check = self.__check_local_command(prompt, re.compile(r'(stop|cancel)\s+(the\s+|a\s+)?timer'))
            if check is True:
                if self.timers.empty() is True:
                    self.speak("You have no running Timers")
                else:
                    #TODO when stopping timers track which one we stop...
                    t = self.timers.get()
                    t.stop()
                    t.join()
        return check

    def run(self):
        while self.stop is False:
            self.sight_results = self.eyes.get_results()
            self.seen = self.process_sight(self.sight_results)
            if self.sight_results.get("person", None) is None:
                print("sleeping 10")
                time.sleep(10)
            else:
                print("sleeping 1")
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
            #count = seen[item]['count']
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
    
    def set_volume(self, prompt):
        pccommand = ["set volume", "change volume"]
        prompt = prompt.lower()
        for pc in pccommand:
            check = self.__check_local_command(prompt, pc)
            if check is True:
                break
        scheck = self.__check_local_command(prompt, re.compile(r'%'))
        if scheck is True:
            level = re.findall(r'\b\d+\b', prompt)
            self.__change_volume(int(level[0]))
            self.currentvol = level[0]
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

    def tstart(self):
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
        self.tstart()
        while True:    
            r = self.check_remaining_time()
            print(r)
            if r["complete"] is True:
                break
            time.sleep(.2)
            

class gladosGPT(Thread):
    def __init__(self, configp, prompt):
        Thread.__init__(self)
        Thread.daemon = True
        self.real_response = None
        self.prompt = prompt
        self.configp = configp["OPENAI"]
        self.model = self.configp["model"]
        self.api_key = self.configp["apikey"]
        self.api_endpoint = self.configp["endpoint"]
        self.content = self.configp["prompt"]
        self.updated_content = None

    def add_prompt(self, content):
        # allow extra info to be added to the prompt
        self.updated_content = content

    def generate_text(self):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",}
        if self.updated_content is None:
            localcontent = self.content
        else:
            localcontent = f"{self.content}. {self.updated_content}"
        print(localcontent)
        data = {
                "model": self.model,
                "messages": [{"role": "system", "content": localcontent},
                    {"role": "user", "content": self.prompt}],
            "max_tokens": 1500,
        }
        
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
    except:
        parser.print_help()
        sys.exit(0)
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    configp = configparser.ConfigParser()
    if path.isfile(args.conf[0]) is True:
        configp.read(args.conf[0])
    else:
        raise GLaDOS_Exception("Unable to load file {}".format(configFile))
    gl = gladosLocal(configp, gladosGPT)
    gl.start()
    gstt = gladosSTT(gl)
    gstt.start()
    local_commands = (gl.get_temp, gl.fuckyou, gl.timer, gl.set_volume)
    while True:
        prompt = gstt.get_text()
        #prompt = "what's the temperature outside"
        if prompt is not None:
            # check for local commands
            # TODO load commands from config?
            print(f"prompt is in main {prompt}")
            for cmd in local_commands:
                cmdbool = cmd(prompt)
                if cmdbool is True:
                    # break the for loop
                    break
            if cmdbool is True:
                # skip the rest on the while loop
                continue
            gladosgpt = gladosGPT(configp, prompt)
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
