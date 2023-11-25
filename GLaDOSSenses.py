import sys
import json
from threading import Thread
import argparse
import configparser
"""
import multiprocessing as mp
    with mp.Manager() as manager:
            self.mplist = manager.list()
            self.proc = mp.Process(target=self.record, args=(self.mplist,))
            self.proc.start()
            while True:
                time.sleep(1)
"""

#3rd party
from ultralytics import YOLO
import imagezmq
import zmq


class GLaDOS_Server_Exception(Exception):
    pass


class YoloDetect(Thread):
    #TODO Make this use a configfile for model's and confidence scores
    def __init__(self, model="yolov8m.pt"):
        Thread.__init__(self)
        Thread.daemon = True
        #TODO, do we need to thread? maybe thread to pull from zmq?
        self.model = YOLO(model)
        self.image_hub = imagezmq.ImageHub()
        self.sight = None

    def get_sight(self):
        return self.sight

    def __translate_results(self, results):
        results_list = list()
        for yclass in res:
            results_list.append(json.loads(yclass.tojson()))
        return results_list

    def __find_people(self, res, confidence=0.6):
        # look for people in the yolo results
        people = list()
        for yclass in res:
            dictclass = json.loads(yclass.tojson())
            for found in dictclass:
                if found['name'] == 'person' and found['confidence'] >= confidence:
                    people.append(found)
        return {"people": people}

    def __yolo_process_image(self, image):
        results = self.model(image)
        return results

    def process_image(self, image)
        found_items = dict()
        item_search = (self.__find_people, )
        for item in item_search:
            found_items.update(item(image))
        self.sight = found_items

    def run(self)
        while True:  # show streamed images until Ctrl-C
            sys_name, image = self.image_hub.recv_image()
            print(f"Got image from {sys_name}")
            self.image_hub.send_reply(b'OK')
            self.process_image(image)


class DataSend(Thread):
    # threaded zmq class for sending to clients
    def __init__(self, client_name, data):
        Thread.__init__(self)
        Thread.daemon = True
        self.client_name = client_name
        self.context = zmq.Context()
        # clientaddress is ip and port
        self.clientaddress = "tcp://locahost:5555"
        self.socketsend = self.context.socket(zmq.PUSH)
        self.data = data

    def run(self):
        self.socketsend.connect(self.clientaddress)
        self.socketsend.send_string(json.dumps(self.data))
        self.socketsend.close()
        self.context.term()


class DataRecv(Thread):
    def __init(self, client_name):
        Thread.__init__(self)
        Thread.daemon = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)  # Create a PULL socket
        self.socket.bind("tcp://*:5555")  # Bind to the TCP port 5555
        self.data = None
        self.stop = False

    def get_data(self, blocking=False)
        if blocking is True:
            while self.data is None:
                time.sleep(.1)
        data = self.data
        self.data = None
        return data

    def run():
        while self.stop is False:
            #TODO check if there is a timeout here or it will never close on exit?
            json_data = socket.recv_string()  # Receive the data as a JSON string
            self.data = json.loads(json_data)


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
    
    eyes = YoloDetect()
    eyes.start()
    # loop
    # you have a few problems, how does eye track which device its processing for?, we need a datasender object for each server wre tracking to return the response, kinda sudo code for now
    while True:
        # this is threaded and will just spin up a ton of threads... do we block? sleep? what else would be done in this loop?
        sender = DataSend("gclient", eyes.get_sight())
        sender.start()
        

