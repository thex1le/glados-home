import sys
sys.path.insert(0, '/home/textile/glados-home/gladosTTS')
import json
from threading import Thread
import argparse
import configparser
from os import path
from time import sleep
from pickle import dumps, loads

#3rd party
import zmq
import cv2

class GLaDOS_Server_Exception(Exception):
    pass


class Camera(Thread):
    def __init__(self, configfile):
        from picamera2 import Picamera2
        Thread.__init__(self)
        Thread.daemon = True
        self.config = configfile['DEFAULT']
        cam_res = self.config['camera_resolution'].split(',')
        self.cam_res_x = int(cam_res[0])
        self.cam_res_y = int(cam_res[1])
        # use json lib to convert string bool to bool
        self.picam = json.loads(self.config['picam'].lower())
        if self.picam is True:
            # pi cam
            self.cap = Picamera2()
            self.cap.configure(self.cap.create_preview_configuration({"size": (self.cam_res_x, self.cam_res_y), 'format': 'RGB888'}))
            self.cap.start()
        else:
            # usb webcam
            camera = int(self.config["Camera"])
            self.cap = cv2.VideoCapture(camera)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.image = None
        self.stop = False
        self.results = dict()
        self.imageget = DataRecv(configfile)
        self.imageget.start()
        self.imagesend = DataSend(configfile)
        self.imagesend.start()

    def get_image(self, blocking=True):
        while blocking is True and self.image is None:
            sleep(.1)
        return self.image

    def run(self):
        while self.stop is False:
            if self.picam is True:
                frame = self.cap.capture_array()
            else:
                ret, frame = self.cap.read()
            self.image = frame
            cv2.imwrite('raw.jpg', frame)
            sleep(.02)

    def get_results(self):
        self.imagesend.send_data(self.get_image(), jsonsend=False)
        data = self.imageget.get_data()
        if data is not None:
            self.results = json.loads(data)
        return self.results


class YoloDetect(Thread):
    def __init__(self, configfile):
        #internal libs, import here so its deps are not needed on other devices
        import GLaDOSVision
        Thread.__init__(self)
        Thread.daemon = True
        self.configfile = configfile
        cam_res = self.configfile['DEFAULT']['camera_resolution'].split(',')
        self.cam_res_x = int(cam_res[0])
        self.cam_res_y = int(cam_res[1])
        self.factory = self.configfile['DEFAULT']['rtsp_factory']
        self.rtsp_port = int(self.configfile['DEFAULT']['rtsp_port'])
        self.rtsp_server_ip = self.configfile['DEFAULT']['rtsp_server_ip']
        self.confyolo = configfile["YOLO"]
        self.model = YOLO(self.confyolo["model"])
        self.sight = None
        self.imageget = DataRecv(self.configfile)
        self.imageget.start()
        self.imagesend = DataSend(self.configfile)
        self.imagesend.start()
        # TODO add this to config file
        print("Starting the RTSP server on rtsp://{}:{}/{}".format(self.rtsp_server_ip, self.rtsp_port, self.factory))
        self.rtsp = GLaDOSVision.RTSPServer(cam_x=self.cam_res_x, cam_y=self.cam_res_y,
                                            port=self.rtsp_port, factory=self.factory)

    def get_sight(self):
        return self.sight

    def __translate_results(self, results):
        results_dict = {}
        for yclass in results:
            print(yclass)
            print(type(yclass))
            if yclass is None:
                continue
            jclass = json.loads(yclass.tojson())
            for cname in jclass:
                name = cname["name"]
                if name in list(results_dict.keys()):
                    results_dict[name]["count"] += 1
                    results_dict[name]["objects"].append(cname)
                else:
                    results_dict[name] = {"count": 1, "objects": [cname], "class_name": name}
        print(results_dict)
        return results_dict

    def __yolo_process_image(self, image):
        # pass image to rtsp..
        results = self.model(image)
        annotator = Annotator(image)
        for r in results:
            annotator = Annotator(image)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])
        a_image = annotator.result()
        self.rtsp.send_data(a_image)
        return results

    def process_image(self, image):
        final_image=loads(image) 
        cv2.imwrite('raw_rx.jpg', final_image)
        self.sight = self.__yolo_process_image(final_image)
        print("sending back dict")
        self.imagesend.send_data(self.__translate_results(self.sight))

    def run(self):
        while True:
            image = self.imageget.get_data(True) 
            print(f"Got image from sender")
            try:
                self.process_image(image)
            except Exception:
                print("Image Error")


class DataSend(Thread):
    # threaded zmq class for sending to clients
    def __init__(self, configfile):
        Thread.__init__(self)
        Thread.daemon = True
        self.configfile = configfile["DEFAULT"]
        ip = self.configfile["ZMQSenderAddress"]
        port = self.configfile["ZMQSenderPort"]
        self.context = zmq.Context()
        self.clientaddress = f"tcp://{ip}:{port}"
        self.socketsend = self.context.socket(zmq.PUSH)
        self.socketsend.connect(self.clientaddress)
        self.stop = False
        self.data = list()

    def send_data(self, data, jsonsend=True):
        if jsonsend is True:
            self.data.append(json.dumps(data))
        if jsonsend is False:
            self.data.append(dumps(data))

    def run(self):
        print("loop started")
        while self.stop is False: 
            try:
                data = self.data.pop(0)
                if type(data) is str:
                    self.socketsend.send_string(data)
                else:
                    self.socketsend.send(data)
            except IndexError:
                pass
            sleep(.1)
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
        self.data = None
        self.stop = False

    def get_data(self, blocking=False):
        if blocking is True:
            while self.data is None:
                sleep(.1)
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
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator
    eyes = YoloDetect(configp)
    eyes.start()
    # start the text to speach engine
    engine.main()
    #ttsengine = mp.Process(target=engine.main, args=())
    #ttsengine.start()
    # loop
    # do we keep looping hear? what blocks on main?
    #while True:
    #    time.sleep(1)
        

