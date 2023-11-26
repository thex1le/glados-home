import sys
sys.path.insert(0, '/home/textile/glados-home/gladosTTS')
import json
from threading import Thread
import argparse
import configparser
import multiprocessing as mp
from os import path
import time
from pickle import dumps, loads

#3rd party
from ultralytics import YOLO
import zmq
import pdb
import numpy as np

class GLaDOS_Server_Exception(Exception):
    pass


class YoloDetect(Thread):
    #TODO Make this use a configfile for model's and confidence scores
    def __init__(self, configfile):
        Thread.__init__(self)
        Thread.daemon = True
        #TODO, do we need to thread? maybe thread to pull from zmq?
        self.configfile = configfile
        self.confyolo = configfile["YOLO"]
        self.model = YOLO(self.confyolo["model"])
        self.sight = None
        self.imageget = DataRecv(self.configfile)
        self.imageget.start()
        self.imagesend = DataSend(self.configfile)
        self.imagesend.start()

    def get_sight(self):
        return self.sight

    def __translate_results(self, results):
        results_dict = {}
        for yclass in results:
            jclass = json.loads(yclass.tojson())
            for cname in jclass:
                name = cname["name"]
                if name in list(results_dict.keys()):
                    #update teh count
                    results_dict[name]["count"] += 1
                    results_dict[name]["objects"].append(cname)
                else:
                    results_dict[name] = {"count": 1, "objects": list(), "class_name": name}
        return results_dict

    def __yolo_process_image(self, image):
        results = self.model(image)
        return results

    def process_image(self, image):
        # you left off here... it was processing the image, likely need to clean up logic here for other classes?
        # error in system trying to send zmq mesage back of the object, likely just pickle it again?
        final_image=loads(image) 
        self.sight = self.__yolo_process_image(final_image)
        print("sending back dict")
        self.imagesend.send_data(self.__translate_results(self.sight))

    def run(self):
        while True:  # show streamed images until Ctrl-C
            image = self.imageget.get_data(True) 
            print(f"Got image from sender")
            self.process_image(image)


class DataSend(Thread):
    # threaded zmq class for sending to clients
    def __init__(self, configfile):
        Thread.__init__(self)
        Thread.daemon = True
        self.configfile = configfile["DEFAULT"]
        ip = self.configfile["ZMQSenderAddress"]
        port = self.configfile["ZMQSenderPort"]
        self.context = zmq.Context()
        # clientaddress is ip and port
        self.clientaddress = f"tcp://{ip}:{port}"
        self.socketsend = self.context.socket(zmq.PUSH)
        self.socketsend.connect(self.clientaddress)
        self.stop = False
        self.data = list()

    def send_data(self, data):
        self.data.append(json.dumps(data))

    def run(self):
        print("loop started")
        while self.stop is False: 
            try:
                data = self.data.pop(0)
                self.socketsend.send_string(data)
            except IndexError:
                pass
            time.sleep(.1)
        self.socketsend.close()
        self.context.term()


class DataRecv(Thread):
    def __init__(self, configfile):
        Thread.__init__(self)
        Thread.daemon = True
        self.config = configfile["DEFAULT"]
        ip = self.config["ZMQListenAddress"]
        port = self.config["ZMQSenderPort"]
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)  # Create a PULL socket
        self.socket.bind(f"tcp://{ip}:{port}")  # Bind to the TCP port 5555
        ip_address = self.socket.getpeername()[0]
        print(ip_address)
        self.data = None
        self.stop = False

    def get_data(self, blocking=False):
        if blocking is True:
            while self.data is None:
                time.sleep(.1)
        data = self.data
        self.data = None
        return data

    def run(self):
        while self.stop is False:
            #TODO check if there is a timeout here or it will never close on exit?
            self.data = self.socket.recv()  # Receive the data as a JSON string
            print("got data")
        self.socketsend.close()
        self.context.term()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evil Home AI Senses Server')
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
        raise GLaDOS_Server_Exception("Unable to load file {}".format(configFile))
    # import and init the 3rd part glados text to speach engine, this prevents init of the engine when you just want to print the help
    from gladosTTS import engine
    eyes = YoloDetect(configp)
    eyes.start()
    # start the text to speach engine
    #ttsengine = mp.Process(target=engine.main, args=())
    #ttsengine.start()
    
    # loop
    # you have a few problems, how does eye track which device its processing for?, we need a datasender object for each server wre tracking to return the response, kinda sudo code for now
    # this is threaded and will just spin up a ton of threads... do we block? sleep? what else would be done in this loop?
    while True:
        time.sleep(1)
        

