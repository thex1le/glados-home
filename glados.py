import requests
import urllib
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
import re
import multiprocessing as mp

#3rd party imports
import pyaudio
import speech_recognition as sr
import openai
from pydub import AudioSegment
from pydub.playback import play
from alsaaudio import Mixer

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

    def transcribe_audio_to_test(self, filename):
        recogizer=sr.Recognizer()
        with sr.AudioFile(filename)as source:
            audio=recogizer.record(source) 
        try:
            return recogizer.recognize_google(audio)
        except:
            print("skipping unkown error")

    def get_text(self):
        # return the text and sent it back to none for next question
        try:
            text = self.mplist.pop()
        except IndexError:
            text = None
        return text

    def record(self, mp_list):
        while True:
            print("Say 'Hey GLaDOS' to start recording your question")
            with sr.Microphone() as source:
                recognizer = sr.Recognizer()
                print("getting audio")
                audio = recognizer.listen(source)
                print("audio done")
                try:
                    transcription = recognizer.recognize_google(audio)
                    print(transcription.lower())
                    if transcription.lower() in ["hey glados", "hey gladys", "glados", "egg glados", "play glados"]:
                        #record audio
                        filename = "input.wav"
                        self.glocal.random_greeting()
                        print("ask question")
                        self.glocal.random_question()
                        with sr.Microphone() as source:
                            recognizer = sr.Recognizer()
                            source.pause_threshold = 1
                            audio = recognizer.listen(source,phrase_time_limit=None, timeout=None)
                            with open(filename,"wb")as f:
                                f.write(audio.get_wav_data())
                        #transcript audio to test 
                        self.glocal.random_question_response()
                        mp_list.append(self.transcribe_audio_to_test(filename))
                except Exception as e:
                 print("An error ocurred : {}".format(e))

    def run(self):
        # use manager to run managment loop
        with mp.Manager() as manager:
            self.mplist = manager.list()
            self.proc = mp.Process(target=self.record, args=(self.mplist,))
            self.proc.start()
            while True:
                time.sleep(1)


class gladosLocal(Thread):
    def __init__(self, configFile):
        Thread.__init__(self)
        Thread.daemon = True
        self.greetings = list()
        self.processing = list()
        self.insults = list()
        self.questions = list()
        self.qresponses = list()
        self.fuck = list()
        self.last_greeting = None
        self.last_insult = None
        self.last_process = None
        self.last_question = None
        self.last_qresponse = None
        self.last_fresponse = None
        self.timers = list()
        self.voiceurl = configFile["DEFAULT"]["VoiceUrl"]
        self.configp = configFile["LOCALSPEAK"]
        self.greetings = self.llp(self.configp["greetings"])
        self.processing = self.llp(self.configp["processing"])
        self.insults = self.llp(self.configp["insults"])
        self.questions = self.llp(self.configp["questions"])
        self.qresponse = self.llp(self.configp["qresponses"])
        self.fuck = self.llp(self.configp["fuck"])
        self.mixer = Mixer("Speaker")
        self.__change_volume(int(configFile["DEFAULT"]["VolumeLevel"]))
        self.currentvol = int(self.mixer.getvolume()[0])

    def __random_audio(self, choice, last, options_list):
        proc = self.__dedupe(choice, last, options_list)
        self.speak(proc)
        last = proc

    def random_question_response(self):
        self.__random_audio(random.choice(self.qresponse), self.last_qresponse, self.qresponse)

    def random_question(self):
        self.__random_audio(random.choice(self.questions), self.last_question, self.questions)

    def random_insult(self):
        self.__random_audio(random.choice(self.insults), self.last_insult, self.insults)
    
    def random_processing(self):
        self.__random_audio(random.choice(self.processing), self.last_process, self.processing)
    
    def random_fuck_response(self):
        self.__random_audio(random.choice(self.fuck), self.last_fresponse, self.fuck)
    
    def random_greeting(self):
        self.__random_audio(random.choice(self.greetings), self.last_greeting, self.greetings)
    
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
         
    def fuckyou(self, prompt):
        check = self.__check_local_command(prompt.lower(), "fuck you")
        if check is True:
            self.random_fuck_response()
        return check
    
    def timer(self, prompt):
        pcommand = ["set timer", "set a timer"]
        prompt = prompt.lower()
        for pc in pcommand:
            check = self.__check_local_command(prompt, pc)
            if check is True:
                break
        # figure amount of time
        ttype = [{"type":"minutes", "re":re.compile(r'minutes?'), "mul":60}, 
              {"type":"seconds", "re":re.compile(r'seconds?'), "mul": 1}, 
              {"type":"hours", "re":re.compile(r'hours?'), "mul": 3600}]
        for t in ttype:
            scheck = self.__check_local_command(prompt, t["re"])
            if scheck is True:
                ti = re.findall(r'\b\d+\b', prompt)
                num = int(ti[0])
                seconds = num * t['mul']
                egg = EggTimer(seconds, self.speak)
                egg.start()
                self.timers.append(egg)
                self.speak("I have Set a Timer for {}, {}".format(num, t['type']))
                break
        #if check is False:
        #    # note, need to find number of timer...
        #    pcommand = ["stop timer", "stop a timer"]
        #    for pc in pcommand:
        #        check = self.__check_local_command(prompt, pc)
        #        if check is True:
        #            break
        #        # need to fix number there
        #        self.timmers[0].stop()
        return check

    def run(self):
        pass

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
            print("Egg timer stopped. Remaining time: {:.2f} seconds.".format(remaining_time))

    def check_remaining_time(self):
        rtn = {"remain": 0, "complete":False}
        if self.is_running:
            elapsed_time = time.time() - self.start_time
            remaining_time = max(0, self.duration - elapsed_time)
            rtn["remain"] = remaining_time
            if remaining_time == 0:
                rtn["remian"] = 0
                rtn["complete"] = True
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
                self.speak("Your Timer is complete")
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

    def generate_text(self):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",}
        data = {
                "model": self.model,
                "messages": [{"role": "system", "content": self.content},
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
    gl = gladosLocal(configp)
    gstt = gladosSTT(gl)
    gstt.start()
    local_commands = [gl.fuckyou, gl.timer, gl.set_volume]
    while True:
        prompt = gstt.get_text()
        if prompt is not None:
            # check for local commands
            # TODO load commands from config?
            for cmd in local_commands:
                cmdbool = cmd(prompt)
                if cmdbool is True:
                    # break the for loop
                    break
            if cmdbool is True:
                # skip the rest on the while loop
                continue
            gladosgpt = gladosGPT(configp, prompt)
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
