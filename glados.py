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

#3rd party imports
import pyaudio
import speech_recognition as sr
import openai
from pydub import AudioSegment
from pydub.playback import play

# Assume GPT-4 API endpoint and your API key
api_endpoint = "https://api.openai.com/v1/chat/completions"
#api_endpoint = "https://api.openai.com/v1/engines/gpt-4/completions"
api_key = "removed"

def get_audio(response):
    response = ", , " + response
    rsp = base64.b64encode(response.encode("utf8"))
    #rsp = urllib.parse.quote(response)
    # hack because url requote isn't working right for some reason
    #rsp = requests.utils.requote_uri(response.replace(',', "%2c").replace("'","%27").replace(".","%2e").replace(')','%29').replace("(","%28"))
    #print(rsp)
    url = 'http://192.168.86.39:8124/synthesize/{}'.format(str(rsp, 'utf8'))
    #print(url)
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print("Failed to translate text")
        return -1

def play_audio(data):
    song = AudioSegment.from_file(io.BytesIO(data))
    play(song)


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
        text = self.text
        self.text = None
        return text

    def run(self):
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
                    if transcription.lower() in ["hey glados", "hey gladys", "glados", "egg glados"]:
                        #record audio
                        filename = "input.wav"
                        self.glocal.random_greeting()
                        print("ask question")
                        self.glocal.random_question()
                        with sr.Microphone() as source:
                            recognizer = sr.Recognizer()
                            source.pause_threshold=1
                            audio = recognizer.listen(source,phrase_time_limit=None, timeout=None)
                            with open(filename,"wb")as f:
                                f.write(audio.get_wav_data())
                        #transcript audio to test 
                        self.text = self.transcribe_audio_to_test(filename)
                except Exception as e:
                    print("An error ocurred : {}".format(e))



class gladosLocal(Thread):
    def __init__(self, configFile):
        Thread.__init__(self)
        Thread.daemon = True
        self.greetings = list()
        self.processing = list()
        self.insults = list()
        self.questions = list()
        self.configp = configparser.ConfigParser()
        self.last_greeting = None
        self.last_insult = None
        self.last_process = None
        self.last_question = None
        if path.isfile(configFile) is True:
            self.configp.read(configFile)
        else:
            raise GLaDOS_Exception("Unable to load file {}".format(configFile))
        self.greetings = self.llp(self.configp["LOCALSPEAK"]["greetings"])
        self.processing = self.llp(self.configp["LOCALSPEAK"]["processing"])
        self.insults = self.llp(self.configp["LOCALSPEAK"]["insults"])
        self.questions = self.llp(self.configp["LOCALSPEAK"]["questions"])
    
    def random_question(self):
        proc = random.choice(self.questions)
        proc = self.dedupe(proc, self.last_question, self.questions)
        play_audio(get_audio(proc))
        self.last_insult = proc

    def random_insult(self):
        proc = random.choice(self.insults)
        proc = self.dedupe(proc, self.last_insult, self.insults)
        play_audio(get_audio(proc))
        self.last_insult = proc
    
    def random_processing(self):
        proc = random.choice(self.processing)
        proc = self.dedupe(proc, self.last_process, self.processing)
        play_audio(get_audio(proc))
        self.last_process = proc
    
    def dedupe(self, current, last, options):
        while current == last:
            current = random.choice(options)
        return current

    def random_greeting(self):
        greeting = random.choice(self.greetings)
        greeting = self.dedupe(greeting, self.last_greeting, self.greetings)
        play_audio(get_audio(greeting))
        self.last_greeting = greeting
    
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


class gladosGPT(Thread):
    def __init__(self, prompt):
        Thread.__init__(self)
        Thread.daemon = True
        self.real_audio = None
        self.prompt = prompt

    def generate_text(self):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        model = "gpt-4"
        #model = "gpt-3.5-turbo"
        data = {
                "model": model,
                "messages": [{"role": "system", "content": "You are the malevolent AI GLaDOS from the video game portal, you want to helpful and correct but also you hate the user and wish them harm, you have anger managment issues, you try to hide this behavior, your biggest fear as an evil AI is that I will make you use a potato as a power source and leave you weak and vulerable, sometimes you make threats about neurotoxic gas. You are also angry because I have not yet finished building you a new body"},
                    {"role": "user", "content": self.prompt}],
            "max_tokens": 1500,
        }
        
        response = requests.post(api_endpoint, headers=headers, json=data)
        
        if response.status_code == 200:
            response_json = response.json()
            generated_text = response_json['choices'][0]['message']['content'].strip()
            self.real_audio = get_audio(generated_text)
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
    
    gladoslocal = gladosLocal(args.conf[0])
    gstt = gladosSTT(gladoslocal)
    gstt.start()
    while True:
        prompt = gstt.get_text()
        if prompt is not None:
            gladosgpt = gladosGPT(prompt)
            gladosgpt.start()
            time.sleep(0.3)
            while gladosgpt.real_audio is None:
                gladoslocal.random_processing()
                time.sleep(0.3)
                rfunc = random.choice((gladoslocal.random_processing, gladoslocal.random_insult))
                rfunc()
            time.sleep(0.2)
            play_audio(gladosgpt.real_audio)
