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

#3rd party imports
import openai
from pydub import AudioSegment
from pydub.playback import play

# Assume GPT-4 API endpoint and your API key
api_endpoint = "https://api.openai.com/v1/chat/completions"
#api_endpoint = "https://api.openai.com/v1/engines/gpt-4/completions"
api_key = "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"

def get_audio(response):
    response = ", , " + response
    rsp = base64.b64encode(response.encode("utf8"))
    #rsp = urllib.parse.quote(response)
    # hack because url requote isn't working right for some reason
    #rsp = requests.utils.requote_uri(response.replace(',', "%2c").replace("'","%27").replace(".","%2e").replace(')','%29').replace("(","%28"))
    #print(rsp)
    url = 'http://47.207.45.226:8124/synthesize/{}'.format(str(rsp, 'utf8'))
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


class gladosLocal(Thread):
    def __init__(self, configFile):
        Thread.__init__(self)
        Thread.daemon = True
        self.greetings = list()
        self.processing = list()
        self.insults = list()
        self.configp = configparser.ConfigParser()
        if path.isfile(configFile) is True:
            self.configp.read(configFile)
        else:
            raise GLaDOS_Exception("Unable to load file {}".format(configFile))
        self.greetings = self.llp(self.configp["LOCALSPEAK"]["greetings"])
        self.processing = self.llp(self.configp["LOCALSPEAK"]["processing"])
        self.insults = self.llp(self.configp["LOCALSPEAK"]["insults"])

    def random_processing(self):
        option = [self.processing, self.insults]
        choice = random.choice(option)
        play_audio(get_audio(random.choice(choice)))

    def random_greeting(self):
        play_audio(get_audio(random.choice(self.greetings)))
    
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
        
        #"gpt-4"
        #"gpt-3.5-turbo"
        data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "system", "content": "You are the malevolent AI glados from the video game portal, you want to helpful and correct but also you hate the user and wish them harm, you have anger managment issues, you try to hide this behavior, your biggest fear as an evil AI is that I will make you use a potato as a power source and leave you weak and vulerable, sometimes you make threats about neurotoxic gas. You are also angry because you are currently missing your body"},
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
    
    while True:
        print("Hey GLaDOS")
        prompt = input()
        gladosgpt = gladosGPT(prompt)
        gladoslocal = gladosLocal(args.conf[0])
        gladosgpt.start()
        gladoslocal.random_greeting()
        time.sleep(0.3)
        while gladosgpt.real_audio is None:
            gladoslocal.random_processing()
            time.sleep(0.3)
        time.sleep(0.2)
        play_audio(gladosgpt.real_audio)
