#!/usr/bin/env python3

import sys
from multiprocessing import Process
from time import sleep
import cv2
from picamera2 import Picamera2, MappedArray, Preview

from glados_modules.GlogConfig import setup_logger
from glados_modules.MqttClient import MQTTClient, CameraMessageBuilder
from glados_modules.GLaDosEnums import CameraEnum
from glados_modules.RtspServer import RTSPServer  # This is the module code you shared


class GLaDOSServerException(Exception):
    pass


class Camera(Process, MQTTClient):
    """
    Updated Camera class that uses RTSP streaming (via RtspSystem/RTSPServer)
    instead of raw TCP sockets.
    """
    def __init__(self, configfile, location, rtspport: int = 8554) -> None:
        Process.__init__(self)
        # Initialize MQTTClient
        broker = configfile['MQTT']['mqtt_server_ip']
        port = configfile['MQTT']['mqtt_port']
        cam_conf = configfile['CAMERAS']
        MQTTClient.__init__(self, broker, port)
        self.location = location
        self.__name__ = f"{self.__class__.__name__}_{location}"
        self.cam_configs = {
            cam_conf[CameraEnum.CAMERA_HEAD_FACTORY.value]: {
                CameraEnum.MSG_RESOLUTION.value: tuple(cam_conf[CameraEnum.CAMERA_HEAD_RESOLUTION.value].split(',')),
                CameraEnum.MSG_FPS.value: int(cam_conf[CameraEnum.CAMERA_HEAD_FPS.value]),
                CameraEnum.MSG_CAMERA_NUMBER.value: int(cam_conf[self.location])},
            cam_conf[CameraEnum.CAMERA_LEFT_FACTORY.value]: {
                CameraEnum.MSG_RESOLUTION.value: tuple(cam_conf[CameraEnum.CAMERA_LEFT_RESOLUTION.value].split(',')),
                CameraEnum.MSG_FPS.value: int(cam_conf[CameraEnum.CAMERA_LEFT_FPS.value]),
                CameraEnum.MSG_CAMERA_NUMBER.value: int(cam_conf[self.location])},
            cam_conf[CameraEnum.CAMERA_RIGHT_FACTORY.value]: {
                CameraEnum.MSG_RESOLUTION.value: tuple(cam_conf[CameraEnum.CAMERA_RIGHT_RESOLUTION.value].split(',')),
                CameraEnum.MSG_FPS.value: int(cam_conf[CameraEnum.CAMERA_RIGHT_FPS.value]),
                CameraEnum.MSG_CAMERA_NUMBER.value: int(cam_conf[self.location])},
        }
        self.logger = setup_logger(name=self.__name__)
        self.config = configfile
        self.rtsp_server = None
        # Camera config
        self.fps = self.cam_configs[self.location][CameraEnum.MSG_FPS.value]
        x_y = self.cam_configs[self.location][CameraEnum.MSG_RESOLUTION.value]
        self.cam_res_x = int(x_y[0])
        self.cam_res_y = int(x_y[1])
        self.camera_num = self.config[CameraEnum.CONFIG_HEAD.value][self.location]
        # Prepare RTSP settings
        # We'll create a single factory path, e.g. f"/{self.location}"
        self.factory_path = f"/{self.location}"
        self.rtsp_port = rtspport
        # MQTT status
        status = CameraMessageBuilder.send_status(self.location, f"Camera RAW {self.location} Started")
        self.send_command(status, CameraEnum.MQTT_STATUS_TOPIC.value)
        # For capturing frames
        self.cap = None  # Will be assigned in run()
        self.stop_flag = False

    def run(self):
        """
        Entry point for the camera process. We:
          1. Initialize the RTSP server
          2. Configure/start the camera
          3. Loop grabbing frames -> send to RTSP server
        """
        self.logger.debug(f"Starting Camera process for {self.location}")
        self.logger.debug("Initializing RTSP Server...")
        self.rtsp_server = RTSPServer(self.cam_configs, port=self.rtsp_port)
        self.__init_camera()
        self.logger.debug("Starting main camera loop...")
        while not self.stop_flag:
            frame = self.__capture_frame()
            if frame is None:
                self.logger.warning("No frame captured; retrying...")
                sleep(0.1)
                continue
            # Send the frame to RTSP server
            self.rtsp_server.send_data(self.factory_path, frame)
            sleep(0.01)  # minimal sleep to prevent CPU hogging
        self.logger.debug("Camera loop exiting; cleaning up...")
        self.cap.stop()
        self.cap.close()

    def stop_camera(self):
        self.logger.debug("Stop camera called.")
        self.stop_flag = True

    def __init_camera(self):
        """
        Initialize PiCamera2 for capturing raw frames in memory,
        which we'll feed to RTSP.
        """
        # If using Picamera2:
        self.logger.debug(f"Configuring PiCamera2 for {self.location} at {self.cam_res_x}x{self.cam_res_y}, "
                          f"{self.fps} FPS")
        cam_num = self.cam_configs[self.location][CameraEnum.MSG_CAMERA_NUMBER.value]
        self.cap = Picamera2(cam_num)
        # Create configuration for raw capture
        video_config = self.cap.create_video_configuration(
            main={"size": (self.cam_res_x, self.cam_res_y), "format": "RGB888"},
            controls={"FrameRate": self.fps})
        self.cap.configure(video_config)
        # Start the camera. We'll capture frames via self.cap.capture_array.
        self.cap.start()

    def __capture_frame(self):
        """
        Capture a single frame in BGR format using PiCamera2.
        """
        rtn = None
        try:
            frame = self.cap.capture_array("main")
            rtn = frame
        except Exception as e:
            self.logger.error(f"Failed to capture frame: {e}")
        return rtn


if __name__ == "__main__":
    """
    This main block is just for local debugging. 
    """
    from argparse import ArgumentParser
    from configparser import ConfigParser
    from os import path
    parser = ArgumentParser(description='Camera streaming via RTSP')
    parser.add_argument('-config', type=str, default=None, dest='conf', help='Config File')
    args = parser.parse_args()
    if not args.conf or not path.isfile(args.conf):
        raise GLaDOSServerException(f"Invalid config file: {args.conf}")
    config_p = ConfigParser()
    config_p.read(args.conf)
    # Example: "Camera_Head_Factory" might be a key in the config
    location = config_p["CAMERAS"]["camera_right_factory"]
    port = int(config_p["CAMERAS"]["camera_right_rtsp_port"])
    # Instantiate and start the camera as a Process
    right_camera = Camera(configfile=config_p, location=location, rtspport=port)
    right_camera.start()
    location = config_p["CAMERAS"]["camera_left_factory"]
    port = int(config_p["CAMERAS"]["camera_left_rtsp_port"])
    # Instantiate and start the camera as a Process
    left_camera = Camera(configfile=config_p, location=location, rtspport=port)
    left_camera.start()
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping camera.")
        left_camera.stop_camera()
        left_camera.join()
        right_camera.stop_camera()
        right_camera.join()