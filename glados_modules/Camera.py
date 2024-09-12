# built in
from multiprocessing import Process, Queue
from time import sleep, time
# 3rd party
import cv2
from picamera2 import Picamera2

# glados config
from glados_modules.GlogConfig import setup_logger
from glados_modules import RxTx
from glados_modules.MqttClient import MQTTClient, CameraMessageBuilder
from glados_modules.GLaDosEnums import CameraEnum


class GLaDOSServerException(Exception):
    pass


class Camera(Process, MQTTClient):
    def __init__(self, configfile, location):
        Process.__init__(self)
        self.location = location
        self.__name__ = f"{self.__class__.__name__}_{location}"
        self.logger = setup_logger(name=self.__name__)
        self.config = configfile
        broker = self.config['MQTT']['mqtt_server_ip']
        port = self.config['MQTT']['mqtt_port']
        MQTTClient.__init__(self, broker, port)
        resolution = f"{self.location}_Resolution"
        self.fps = int(self.config[CameraEnum.CONFIG_HEAD.value][f"{self.location}_{CameraEnum.MSG_FPS.value}"])
        picam = f"{self.location}_Picam"
        cam_res = self.config[CameraEnum.CONFIG_HEAD.value][resolution].split(',')
        self.cam_res_x = int(cam_res[0])
        self.cam_res_y = int(cam_res[1])
        self.picam = bool(self.config[CameraEnum.CONFIG_HEAD.value][picam].lower())
        self.camera_num = int(self.config[CameraEnum.CONFIG_HEAD.value][self.location])
        self.image = None
        self.status_topic = CameraEnum.MQTT_STATUS_TOPIC.value
        status = CameraMessageBuilder.send_status(self.location, f"Camera {self.location} Started")
        self.send_command(status, CameraEnum.MQTT_STATUS_TOPIC.value)

    def __init_camera(self):
        # allow us to init the camera inside the multiprocess thread
        self.logger.debug(f"Starting sub init for {self.location}")
        if self.picam is True:
            self.logger.debug(f"Using PiCam for {self.location}")
            # pi cam
            self.cap = Picamera2(self.camera_num)
            self.logger.debug(f"Camera resolution of {self.cam_res_x} x {self.cam_res_y}")
            self.logger.debug(f" Camera FPS set to {self.fps} and RBG888 and YUV420 Modes")
            self.cam_config = self.cap.create_video_configuration(
                main={"format": "RGB888", "size": (self.cam_res_x, self.cam_res_y)},
                lores={"format": "YUV420", "size": (self.cam_res_x, self.cam_res_y)},
                display="lores",
                controls={"FrameRate": self.fps}
            )
            self.cap.configure(self.cam_config)
            self.cap.start()
            self.__capture_image = self.__capture_image_pi
        else:
            # usb webcam
            self.logger.debug(f"Using USB Webcam for {self.location}")
            self.cap = cv2.VideoCapture(self.camera_num)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.__capture_image = self.__capture_image_cv2
        self.image = None
        self.stop = False
        self.results = dict()
        self.queue = Queue()
        self.image_send = RxTx.DataSend(self.config, self.location, mpqueue=self.queue)
        self.image_send.start()
        self.logger.debug(f"Sub init for {self.location} Complete")

    def get_camera_config(self):
        """
        Return the camera config
        """
        return self.config

    def set_camera_conf(self, config) -> None:
        """
        Set config, no check is done to make sure it's correct
        """
        #TODO FIGURE OUT HOW TO SIGNAL RTSP SERVER TO UPDATE THIS
        self.cap.configure(config)

    def __capture_image_pi(self):
        return self.cap.capture_buffer("lores")

    def __capture_image_cv2(self):
        ret, frame = self.cap.read()
        return frame

    def get_image(self, blocking=True):
        """
        Allow external call of the object to return an image
        """
        while blocking is True and self.image is None:
            sleep(.02)
        return self.image

    def run(self):
        # sub init of the camera object after the multiprocess memory copy
        self.__init_camera()
        while self.stop is False:
            self.image = self.__capture_image()
            self.logger.debug(f"Sending image from {self.location} for processing")
            image_dict = {CameraEnum.MSG_LOCATION_KEY.value: f"/{self.location}",
                          CameraEnum.MSG_RAW_IMAGE: self.get_image()}
            # load the shared TX object sending queue,
            self.queue.put(image_dict)


if __name__ == "__main__":
    # imports only for testing
    from argparse import ArgumentParser
    import sys
    from configparser import ConfigParser
    from os import path
    parser = ArgumentParser(description='Evil Home AI Senses Server')
    parser.add_argument('-config', type=str, default=1, dest='conf', nargs=1, help='Config File')
    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    config_p = ConfigParser()
    if path.isfile(args.conf[0]) is True:
        config_p.read(args.conf[0])
    else:
        raise GLaDOSServerException("Unable to load file {}".format(args.conf[0]))
    head_camera_location = config_p["CAMERAS"]["Camera_Head_Factory"]
    head_camera = Camera(configfile=config_p, location=head_camera_location)




