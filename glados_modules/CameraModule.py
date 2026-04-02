#!/usr/bin/env python3
from multiprocessing import Process
from time import sleep, time

# 3rd party import
import cv2
from picamera2 import Picamera2, MappedArray, Preview

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.MqttConnector import MQTTClient, CameraMessageBuilder
from glados_modules.GladosEnums import CameraEnum, SystemEnums
from glados_modules.RtspServer import RTSPServer


class GLaDOSServerException(Exception):
    pass


class Camera(Process):
    """Camera process that captures frames and serves them via RTSP.

    Extends Process (not Thread) because Picamera2 and GStreamer hold kernel
    resources that don't survive fork(). All hardware init happens in run()
    after the fork, not in __init__.

    MQTT is initialized in run() for the same reason -- socket connections
    created before fork() become stale in the child process.
    """

    def __init__(self, configfile, location, rtspport: int = 8554) -> None:
        Process.__init__(self)
        self.daemon = True
        self.location = location
        self.__name__ = f"{self.__class__.__name__}_{location}"

        # Store config as plain data (no hardware handles, no sockets)
        cam_conf = configfile[CameraEnum.CONFIG_HEAD.value]
        self._broker_ip = configfile[SystemEnums.CONFIG_HEAD_MQTT.value][SystemEnums.MQTT_SERVER_IP.value]
        self._broker_port = configfile[SystemEnums.CONFIG_HEAD_MQTT.value][SystemEnums.MQTT_PORT.value]
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
        self.fps = self.cam_configs[self.location][CameraEnum.MSG_FPS.value]
        x_y = self.cam_configs[self.location][CameraEnum.MSG_RESOLUTION.value]
        self.cam_res_x = int(x_y[0])
        self.cam_res_y = int(x_y[1])
        self.factory_path = f"/{self.location}"
        self.rtsp_port = rtspport
        self.cap = None
        self.stop_flag = False
        self.rtsp_server = None
        # Per-camera 180 degree flip (for upside-down mounted cameras)
        flip_key = f"{self.location}_flip"
        self._flip_180 = cam_conf.get(flip_key, "False").strip().lower() == "true"
        self.logger = setup_logger(name=self.__name__)

    def run(self):
        """Entry point for the camera process (runs after fork).

        All hardware init happens here: MQTT connection, RTSP server,
        Picamera2. Includes automatic camera restart on failure.

        Wrapped in try/finally so Picamera2 always releases the camera
        device on crash — prevents "Device or resource busy" on respawn.
        """
        # MQTT must be initialized after fork -- parent's socket is stale
        self._mqtt = MQTTClient(self._broker_ip, self._broker_port)
        status = CameraMessageBuilder.send_status(self.location, f"Camera RAW {self.location} Started")
        self._mqtt.send_command(status, CameraEnum.MQTT_STATUS_TOPIC.value)

        self.logger.debug(f"Starting Camera process for {self.location}")
        self.logger.debug("Initializing RTSP Server...")
        self.rtsp_server = RTSPServer(self.cam_configs, port=self.rtsp_port)

        consecutive_failures = 0
        max_failures = 50

        try:
            self.__init_camera()
            self.logger.debug("Starting main camera loop...")

            frame_count = 0
            last_log_time = time()
            LOG_INTERVAL = 10.0  # log stats every 10 seconds

            while not self.stop_flag:
                try:
                    t0 = time()
                    frame = self.__capture_frame()
                    t_capture = time() - t0

                    if frame is None:
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            self.logger.warning(f"{consecutive_failures} consecutive capture failures, restarting camera...")
                            self.__restart_camera()
                            consecutive_failures = 0
                        else:
                            sleep(0.1)
                        continue
                    consecutive_failures = 0

                    t1 = time()
                    self.rtsp_server.send_data(self.factory_path, frame)
                    t_send = time() - t1

                    # Flag if either call took suspiciously long
                    if t_capture > 1.0:
                        self.logger.warning(f"capture_array blocked for {t_capture:.2f}s")
                    if t_send > 1.0:
                        self.logger.warning(f"rtsp send_data blocked for {t_send:.2f}s")

                    frame_count += 1
                    now = time()
                    if now - last_log_time >= LOG_INTERVAL:
                        fps = frame_count / (now - last_log_time)
                        self.logger.info(f"{self.location}: {fps:.1f} FPS, "
                                         f"last capture={t_capture*1000:.0f}ms send={t_send*1000:.0f}ms")
                        frame_count = 0
                        last_log_time = now

                    sleep(0.01)
                except Exception as e:
                    self.logger.error(f"Camera loop error: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        self.logger.warning("Too many errors, restarting camera...")
                        self.__restart_camera()
                        consecutive_failures = 0
                    sleep(0.1)
        finally:
            self.logger.info(f"Camera {self.location} releasing hardware...")
            self._release_camera()

    def _release_camera(self) -> None:
        """Release Picamera2 resources so the device is freed for respawn."""
        if self.cap:
            try:
                self.cap.stop()
            except Exception:
                pass
            try:
                self.cap.close()
            except Exception:
                pass
            self.cap = None
            self.logger.info(f"Camera {self.location} released")

    def stop_camera(self):
        self.logger.debug("Stop camera called.")
        self.stop_flag = True

    def respawn(self) -> 'Camera':
        """Create a new Camera process with the same config after this one dies.

        Returns:
            A new Camera instance ready to start().
        """
        new_cam = Camera.__new__(Camera)
        # Copy all config state from the dead process (set in __init__, safe to reuse)
        new_cam.daemon = True
        new_cam.location = self.location
        new_cam.__name__ = self.__name__
        new_cam._broker_ip = self._broker_ip
        new_cam._broker_port = self._broker_port
        new_cam.cam_configs = self.cam_configs
        new_cam.fps = self.fps
        new_cam.cam_res_x = self.cam_res_x
        new_cam.cam_res_y = self.cam_res_y
        new_cam.factory_path = self.factory_path
        new_cam.rtsp_port = self.rtsp_port
        new_cam.cap = None
        new_cam.stop_flag = False
        new_cam.rtsp_server = None
        new_cam._flip_180 = self._flip_180
        new_cam.logger = self.logger
        Process.__init__(new_cam)
        new_cam.daemon = True
        return new_cam

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

    def __restart_camera(self):
        """Tear down and reinitialize the camera after failures."""
        self.logger.info(f"Restarting camera {self.location}...")
        try:
            if self.cap:
                self.cap.stop()
                self.cap.close()
        except Exception as e:
            self.logger.error(f"Error stopping camera during restart: {e}")
        self.cap = None
        sleep(2.0)
        try:
            self.__init_camera()
            self.logger.info(f"Camera {self.location} restarted successfully")
        except Exception as e:
            self.logger.error(f"Failed to restart camera: {e}")
            sleep(5.0)

    def __capture_frame(self):
        """
        Capture a single frame in BGR format using PiCamera2.
        """
        rtn = None
        try:
            frame = self.cap.capture_array("main")
            if self._flip_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
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
    cam_conf = config_p[CameraEnum.CONFIG_HEAD.value]
    location = cam_conf[CameraEnum.CAMERA_RIGHT_FACTORY.value]
    port = int(cam_conf[CameraEnum.CAMERA_RIGHT_PORT.value])
    # Instantiate and start the camera as a Process
    right_camera = Camera(configfile=config_p, location=location, rtspport=port)
    right_camera.start()
    location = cam_conf[CameraEnum.CAMERA_LEFT_FACTORY.value]
    port = int(cam_conf[CameraEnum.CAMERA_LEFT_PORT.value])
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
