import sys
sys.path.insert(0, '/home/textile/glados-home/gladosTTS')
import json
from threading import Thread
import argparse
import configparser
from os import path
from time import sleep, time
from pickle import dumps, loads

#3rd party
import zmq
import cv2

# glados imports
from glog_conifig import setup_logger


class GLaDOSServerException(Exception):
    pass


class Camera(Thread):
    def __init__(self, callback, configfile):
        from picamera2 import Picamera2
        Thread.__init__(self)
        Thread.daemon = True
        self.logger = setup_logger(name=self.__name__)
        self.config = configfile['DEFAULT']
        cam_res = self.config['camera_resolution'].split(',')
        self.cam_res_x = int(cam_res[0])
        self.cam_res_y = int(cam_res[1])
        self.logger.debug(f"Camera resoltion of {self.cam_res_x} x {self.cam_res_y}")
        # use json lib to convert string bool to bool
        self.picam = json.loads(self.config['picam'].lower())
        if self.picam is True:
            self.logger.debug("Using PiCam")
            # pi cam
            self.cap = Picamera2()
            self.cap.configure(self.cap.create_preview_configuration({"size": (self.cam_res_x, self.cam_res_y),
                                                                      'format': 'RGB888'}))
            self.cap.start()
        else:
            # usb webcam
            self.logger.debug("Using USB Webcam")
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
        self.scan_callback = callback
        self.skip_run = False

    def __capture_image(self):
        if self.picam is True:
            frame = self.cap.capture_array()
        else:
            ret, frame = self.cap.read()
        return frame

    # TODO this is where we can use mqtt to signal servos to stop moving?
    # TODO what were you thinking here? why is target_scan under camera class? how will this work with 3 cameras?
    # how will we de conflict zmq images with multiple camera's, how do we tag images to keep them straight?
    # main pi 5 wil have 2 and second pi 2 will have one...
    # more of a body function thing? how do we link the camera
    def target_scan(self, target="person", search_time=90, confidence=.70):
        self.logger.debug(f"Camera Scanning for target: {target}")
        target_found = False
        t = time()
        while (time() - t) < search_time and target_found is False:
            if target in self.results.keys():
                for p in self.results[target]['objects']:
                    if p['confidence'] >= confidence:
                        # found the target in the timeframe
                        target_found = True
                        self.logger.debug(f"Camera Found target: {target} , {p}")
                        self.scan_callback()
                        break

    def __process_image(self):
        results = dict()
        self.logger.debug("Sending image for processing")
        self.imagesend.send_data(self.get_image(), jsonsend=False)
        data = self.imageget.get_data()
        self.logger.debug(f"Got image back: {data}")
        if data is not None:
            results = json.loads(data)
        return results

    def get_image(self, blocking=True):
        while blocking is True and self.image is None:
            sleep(.1)
        return self.image

    def run(self):
        while self.stop is False:
            self.image = self.__capture_image()
            self.results = self.__process_image()
            # cv2.imwrite('raw.jpg', self.image)
            sleep(.02)

    def get_results(self):
        return self.results


# TODO YOU LEFT OFF CONNECTING IN MQQT CLIENT, it needs to send data to a debug channel
# DATASEND can be replaced by mqtt to send text back to main running program
# CONSIDER setting up a listener to take commands if needed? maybe not this portion kind of runs alone by its self

class YoloDetect(Thread):
    def __init__(self, configfile):
        # internal libs, import here so its deps are not needed on other devices
        import GLaDOSRTSP
        Thread.__init__(self)
        Thread.daemon = True
        self.logger = setup_logger(name=self.__name__)
        self.configfile = configfile
        cam_res = self.configfile['DEFAULT']['camera_resolution'].split(',')
        self.cam_res_x = int(cam_res[0])
        self.cam_res_y = int(cam_res[1])
        self.factory = self.configfile['DEFAULT']['rtsp_factory']
        self.rtsp_port = int(self.configfile['DEFAULT']['rtsp_port'])
        self.rtsp_server_ip = self.configfile['DEFAULT']['rtsp_server_ip']
        self.confyolo = configfile["YOLO"]
        self.logger.debug(f"YOLO model started with {self.confyolo["model"]}")
        self.model = YOLO(self.confyolo["model"])
        self.sight = None
        self.imageget = DataRecv(self.configfile)
        self.imageget.start()
        self.imagesend = DataSend(self.configfile)
        self.imagesend.start()
        # TODO add this to config file
        msg = f"Starting the RTSP server on rtsp://{self.rtsp_server_ip}:{self.rtsp_port}/{self.factory}"
        self.logger.info(msg)
        print(msg)
        self.rtsp = GLaDOSRTSP.RTSPServer(cam_x=self.cam_res_x, cam_y=self.cam_res_y,
                                          port=self.rtsp_port, factory=self.factory)

    def get_sight(self):
        return self.sight

    def __translate_results(self, results):
        results_dict = {}
        for yclass in results:
            if yclass is None:
                continue
            self.logger.debug(f"Translating {yclass} with type {type(yclass)}")
            jclass = json.loads(yclass.tojson())
            for cname in jclass:
                name = cname["name"]
                if name in list(results_dict.keys()):
                    results_dict[name]["count"] += 1
                    results_dict[name]["objects"].append(cname)
                else:
                    results_dict[name] = {"count": 1, "objects": [cname], "class_name": name}
        self.logger.debug(results_dict)
        print(results_dict)

    def __yolo_process_image(self, image):
        # pass image to rtsp...
        results = self.model(image)
        annotator = Annotator(image)
        for r in results:
            annotator = Annotator(image)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])
                self.logger.debug(f"Labeled image with, {self.model[int(c)]}")
        a_image = annotator.result()
        self.rtsp.send_data(a_image)
        return results

    def process_image(self, image, debug_file_name='raw_rx.jpg'):
        final_image = loads(image)
        cv2.imwrite(debug_file_name, final_image)
        self.logger.debug(f"Wrote out sample debug image to {debug_file_name}")
        self.sight = self.__yolo_process_image(final_image)
        self.logger.debug("Sending back process dict of seen data")
        self.imagesend.send_data(self.__translate_results(self.sight))

    def run(self):
        while True:
            image = self.imageget.get_data(True) 
            msg = "Got image from sender"
            print(msg)
            self.logger.debug(msg)
            try:
                self.process_image(image)
            except Exception:
                msg = "Image Error"
                print(msg)
                self.logger.error(msg)


class DataSend(Thread):
    # threaded zmq class for sending to clients
    def __init__(self, configfile):
        Thread.__init__(self)
        Thread.daemon = True
        self.logger = setup_logger(name=self.__name__)
        self.configfile = configfile["DEFAULT"]
        ip = self.configfile["ZMQSenderAddress"]
        port = self.configfile["ZMQSenderPort"]
        self.context = zmq.Context()
        self.clientaddress = f"tcp://{ip}:{port}"
        self.logger.debug(f"Data Sender listening on {self.clientaddress}")
        self.socketsend = self.context.socket(zmq.PUSH)
        self.socketsend.connect(self.clientaddress)
        self.stop = False
        self.data = list()

    def stop_thread(self):
        self.logger.debug("ZMQ Sending Thread Stop Called")
        self.stop = True

    def send_data(self, data, jsonsend=True):
        if jsonsend is True:
            self.logger.debug("Sending JSON data")
            self.data.append(json.dumps(data))
        if jsonsend is False:
            self.logger.debug("Sending PICKLE Data")
            self.data.append(dumps(data))

    def run(self):
        msg = "Data Sending Loop Started"
        print(msg)
        self.logger.debug(msg)
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
        self.logger = setup_logger(self.__name__)
        self.config = configfile["DEFAULT"]
        ip = self.config["ZMQListenAddress"]
        port = self.config["ZMQSenderPort"]
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)  # Create a PULL socket
        self.clientaddress = f"tcp://{ip}:{port}"
        self.socket.bind(self.clientaddress)  # Bind to the TCP port 5555
        self.logger.debug(f"Data Sender listening on {self.clientaddress}")
        self.data = None
        self.stop = False

    def get_data(self, blocking=False):
        if blocking is True:
            while self.data is None:
                sleep(.1)
        data = self.data
        self.data = None
        self.logger.debug(f"Data from zmq returned f{data}")
        return data

    def stop_thread(self):
        self.logger.debug("ZMQ Receiving Thread Stop Called")
        self.stop = True

    def run(self):
        while self.stop is False:
            # TODO check if there is a timeout here or it will never close on exit?
            self.data = self.socket.recv()  # Receive the data as a JSON string
            msg = "Got data from ZMQ Listener"
            print(msg)
            self.logger.debug(msg)
        self.socketsend.close()
        self.context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evil Home AI Senses Server')
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
        raise GLaDOSServerException("Unable to load file {}".format(args.conf[0]))
    # import and init the 3rd part glados text to speach engine,
    # this prevents init of the engine when you just want to print the help
    from gladosTTS import engine
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator
    eyes = YoloDetect(configp)
    eyes.start()
    # start the text to speech engine
    engine.main()
    # ttsengine = mp.Process(target=engine.main, args=())
    # t tsengine.start()
    # loop
    # do we keep looping hear? what blocks on main?
    # while True:
    #    time.sleep(1)
