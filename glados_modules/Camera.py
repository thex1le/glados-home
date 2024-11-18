# built in
from multiprocessing import Process, Queue
from time import sleep, time
import socket
# 3rd party
import cv2
from picamera2.outputs import FileOutput
from picamera2 import Picamera2, MappedArray

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
        self.server_ip = self.config[CameraEnum.CONFIG_HEAD.value][CameraEnum.CAMERA_AI_SERVER_RX.value]
        self.server_port = int(self.config[CameraEnum.CONFIG_HEAD.value][f"{self.location}_port"])
        self.reconnect_timeout = int(
            self.config[CameraEnum.CONFIG_HEAD.value][CameraEnum.CAMERA_AI_SERVER_RX_TIMEOUT.value])
        self.socket = None
        status = CameraMessageBuilder.send_status(self.location, f"Camera {self.location} Started")
        MQTTClient.__init__(self, broker, port)
        self.send_command(status, CameraEnum.MQTT_STATUS_TOPIC.value)
# TODO YOU LEFT OFF DOING THE CONFIG AND SETTING UP TO PUSH H264 video to a different port on the AI server to do streaming video in real time
    def __init_camera(self, server_ip, server_port, reconnect_timeout=5):
        """
        Initialize the camera and stream to a remote server.

        :param server_ip: IP address of the remote server
        :param server_port: Port of the remote server
        :param reconnect_timeout: Timeout (in seconds) to wait before reconnecting
        """
        self.logger.debug(f"Starting sub init for {self.location}")
        self.logger.debug(f"Using PiCam for {self.location}")
        self.cap = Picamera2()
        self.logger.debug(f"Camera resolution of {self.cam_res_x} x {self.cam_res_y}")
        self.logger.debug(f"Camera FPS set to 8 and H.264 streaming to remote server")

        # Configure the camera for H.264 video streaming
        self.cam_config = self.cap.create_video_configuration(
            main={"format": "RGB888", "size": (self.cam_res_x, self.cam_res_y)},
            controls={
                "FrameRate": 8,  # Set FPS to 8
                "Brightness": 0.5,
                "Sharpness": 1.0,
                "Contrast": 1.0,
                "Saturation": 1.0,
                "NoiseReductionMode": 2
            }
        )
        self.cap.configure(self.cam_config)

        # Attempt to connect to the remote server
        self.server_ip = server_ip
        self.server_port = server_port
        self.reconnect_timeout = reconnect_timeout
        self.socket = None
        self.__connect_to_server()

        # Use the socket as the output for the camera
        self.output = self.socket.makefile('wb')
        self.cap.start_recording(FileOutput(self.output), codec="h264")
        self.logger.debug(f"Streaming H.264 video to {self.server_ip}:{self.server_port}")
        self.logger.debug(f"Sub init for {self.location} Complete")

    def __connect_to_server(self):
        """
        Connect to the remote server with retry logic.
        """
        while True:
            try:
                self.logger.debug(f"Attempting to connect to {self.server_ip}:{self.server_port}")
                self.socket = socket.create_connection((self.server_ip, self.server_port), timeout=10)
                self.logger.debug(f"Connected to {self.server_ip}:{self.server_port}")
                break
            except (socket.timeout, socket.error) as e:
                self.logger.warning(f"Connection failed: {e}. Retrying in {self.reconnect_timeout} seconds...")
                sleep(self.reconnect_timeout)

    def set_camera_conf(self, config) -> None:
        """
        Set config, no check is done to make sure it's correct
        """
        #TODO FIGURE OUT HOW TO SIGNAL RTSP SERVER TO UPDATE THIS
        self.cap.configure(config)

    def __capture_image_pi(self):
        return self.cap.capture_buffer("main")

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
                          CameraEnum.MSG_RAW_IMAGE.value: self.get_image(),
                          CameraEnum.MSG_RESOLUTION.value: (self.cam_res_x, self.cam_res_y)}
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




