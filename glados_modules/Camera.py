# built in
from multiprocessing import Process
from time import sleep, time
from json import loads
# 3rd party
import cv2
from picamera2 import Picamera2

# glados config
from glados_modules.GlogConfig import setup_logger
from glados_modules import RxTx
from glados_modules.MqttClient import MQTTClient


class GLaDOSServerException(Exception):
    pass


class Camera(Process, MQTTClient):
    def __init__(self, configfile, location):
        Process.__init__(self)
        self.daemon = True
        self.location = location
        self.__name__ = f"{self.__class__.__name__}_{location}"
        self.logger = setup_logger(name=self.__name__)
        self.config = configfile
        broker = self.config['MQTT']['mqtt_server_ip']
        port = self.config['MQTT']['mqtt_port']
        MQTTClient.__init__(self, broker, port)
        resolution = f"{self.location}_Resolution"
        picam = f"{self.location}_Picam"
        cam_res = self.config['CAMERAS'][resolution].split(',')
        self.cam_res_x = int(cam_res[0])
        self.cam_res_y = int(cam_res[1])
        self.logger.debug(f"Camera resolution of {self.cam_res_x} x {self.cam_res_y}")
        self.picam = bool(self.config['CAMERAS'][picam].lower())
        self.camera_num = int(self.config["CAMERAS"][self.location])
        self.image = None

    def __init_camera(self):
        # allow us to init the camera inside the multiprocess thread
        if self.picam is True:
            self.logger.debug(f"Using PiCam for {self.location}")
            # pi cam
            self.cap = Picamera2(self.camera_num)
            self.cap.configure(self.cap.create_still_configuration({"size": (self.cam_res_x, self.cam_res_y),
                                                                      'format': 'RGB888'}))
            self.cap.start()
        else:
            # usb webcam
            self.logger.debug(f"Using USB Webcam for {self.location}")
            self.cap = cv2.VideoCapture(self.camera_num)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.image = None
        self.stop = False
        self.results = dict()
        self.image_send = RxTx.DataSend(self.config, self.location)
        self.image_send.start()
        self.client.publish("status", f"Camera {self.location} Started")

    def __capture_image(self):
        if self.picam is True:
            frame = self.cap.capture_array()
        else:
            ret, frame = self.cap.read()
        return frame

    def get_image(self, blocking=True):
        while blocking is True and self.image is None:
            sleep(.1)
        return self.image

    def run(self):
        self.__init_camera()
        import time
        count = 0
        t = time.time()
        while self.stop is False:
            self.image = self.__capture_image()
            self.logger.debug("Sending image for processing")
            image_dict = {"camera": f"/{self.location}", "raw": self.get_image()}
            self.image_send.send_data(image_dict, json_send=False)
            #sleep(.02)
            count += 1
            if count >= 60:
                print(time.time() - t)




