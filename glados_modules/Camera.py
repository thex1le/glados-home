# built in
from threading import Thread
from time import sleep, time
from json import loads
# 3rd party
import cv2
from picamera2 import Picamera2

# glados config
from glados_modules.GlogConfig import setup_logger
from glados_modules import RxTx


class GLaDOSServerException(Exception):
    pass


class Camera(Thread):
    def __init__(self, callback, configfile, location):
        # TODO need to fix this up for new config file
        # TODO need to figure out how we are going to sync the camera "scan / hunt" function for new people...
        Thread.__init__(self)
        Thread.daemon = True
        self.location = location
        self.logger = setup_logger(name=f"{self.__name__}_location")
        self.config = configfile['DEFAULT']
        cam_res = self.config['camera_resolution'].split(',')
        self.cam_res_x = int(cam_res[0])
        self.cam_res_y = int(cam_res[1])
        self.logger.debug(f"Camera resolution of {self.cam_res_x} x {self.cam_res_y}")
        self.picam = bool(self.config['picam'].lower())
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
        self.image_get = RxTx.DataRecv(configfile)
        self.image_get.start()
        self.image_send = RxTx.DataSend(configfile)
        self.image_send.start()
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
        self.image_send.send_data(self.get_image(), json_send=False)
        data = self.image_get.get_data()
        self.logger.debug(f"Got image back: {data}")
        if data is not None:
            results = loads(data)
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


# TODO YOU LEFT OFF CONNECTING IN MQTT CLIENT, it needs to send data to a debug channel
# DATASEND can be replaced by mqtt to send text back to main running program
# CONSIDER setting up a listener to take commands if needed? maybe not this portion kind of runs alone by its self
