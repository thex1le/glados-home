#!/usr/bin/env python3
import os
import subprocess
from multiprocessing import Process
from threading import Thread
from time import sleep, time

# 3rd party import
import cv2
from picamera2 import Picamera2, MappedArray, Preview

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.MqttConnector import MQTTClient, CameraMessageBuilder
from glados_modules.GladosEnums import CameraEnum, SystemEnums
from glados_modules.RtspServer import RTSPServer

# Timeout for capture_array — if Picamera2 blocks longer than this,
# the camera driver is assumed stuck
CAPTURE_TIMEOUT = 5.0


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
            try:
                self.__init_camera()
            except (RuntimeError, IndexError):
                # RuntimeError: camera device stuck from a previous crash
                # IndexError: camera hardware not found (wrong cam_num)
                self.logger.warning("Camera init failed — resetting kernel module and retrying...")
                self._reset_camera_kernel_module()
                sleep(3.0)
                self.__init_camera()
            self.logger.debug("Starting main camera loop...")

            frame_count = 0
            last_log_time = time()
            last_successful_capture = time()
            LOG_INTERVAL = 10.0  # log stats every 10 seconds

            # Single watchdog thread that detects capture hangs.
            # Escalation: kernel module reset -> hard process exit.
            # cap.stop()/cap.close() can themselves hang when the driver is
            # stuck, so we don't attempt soft cleanup — go straight to the
            # kernel module reset which forces the driver to release.
            self._watchdog_last_frame = time()

            def _watchdog():
                attempts = 0
                while not self.stop_flag:
                    sleep(1.0)
                    stall_time = time() - self._watchdog_last_frame
                    if stall_time <= CAPTURE_TIMEOUT:
                        attempts = 0
                        continue

                    attempts += 1
                    self.logger.warning(
                        f"Watchdog: no frame for {stall_time:.1f}s "
                        f"(attempt {attempts})")

                    if attempts <= 2:
                        # Kernel module reset — forces the driver to release
                        # even if cap.stop() would hang
                        self.logger.info("Watchdog: resetting kernel camera module...")
                        self._reset_camera_kernel_module()
                        sleep(3.0)
                        # Reinit camera (old cap is dead, just overwrite it)
                        self.cap = None
                        try:
                            self.__init_camera()
                            self._watchdog_last_frame = time()
                            self.logger.info("Watchdog: camera restarted after kernel reset")
                        except Exception as e:
                            self.logger.error(f"Watchdog: reinit failed: {e}")
                    else:
                        # Kernel reset didn't help, hard exit for respawn
                        self.logger.error("Watchdog: all restarts failed, forcing process exit")
                        os._exit(1)

            wd = Thread(target=_watchdog, daemon=True)
            wd.start()

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
                            self._watchdog_last_frame = time()
                        else:
                            sleep(0.1)
                        continue
                    consecutive_failures = 0
                    self._watchdog_last_frame = time()

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
        # Process.__init__ must be called first to set up internal state
        Process.__init__(new_cam)
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
            self.logger.info("Attempting kernel camera module reset...")
            self._reset_camera_kernel_module()
            sleep(3.0)
            try:
                self.__init_camera()
                self.logger.info(f"Camera {self.location} restarted after kernel reset")
            except Exception as e2:
                self.logger.error(f"Camera restart failed even after kernel reset: {e2}")
                sleep(5.0)

    @staticmethod
    def _reset_camera_kernel_module() -> None:
        """Attempt to reset the kernel camera module to free stuck devices.

        Kills any processes holding /dev/video* and /dev/media*, then
        unloads and reloads the camera kernel module. Requires root.
        """
        import logging
        logger = logging.getLogger("CameraModuleReset")

        # Step 1: Kill any processes holding camera device files
        for dev_pattern in ["/dev/video*", "/dev/media*"]:
            try:
                result = subprocess.run(
                    f"fuser -k {dev_pattern}",
                    shell=True, capture_output=True, timeout=5)
                if result.stdout:
                    logger.info(f"Killed processes holding {dev_pattern}")
            except Exception:
                pass
        sleep(1.0)

        # Step 2: Unload and reload the camera kernel module
        # Pi4 uses bcm2835_unicam, Pi5 uses rp1_cfe
        for module in ["bcm2835_unicam", "rp1_cfe"]:
            try:
                r_remove = subprocess.run(
                    ["modprobe", "-r", module],
                    capture_output=True, timeout=5, text=True)
                if r_remove.returncode == 0:
                    logger.info(f"Unloaded kernel module {module}")
                    sleep(0.5)
                    r_load = subprocess.run(
                        ["modprobe", module],
                        capture_output=True, timeout=5, text=True)
                    if r_load.returncode == 0:
                        logger.info(f"Reloaded kernel module {module}")
                    else:
                        logger.error(f"Failed to reload {module}: {r_load.stderr}")
                    return
                else:
                    logger.warning(f"Could not unload {module}: {r_remove.stderr.strip()}")
            except Exception as e:
                logger.warning(f"Module reset failed for {module}: {e}")
                continue

    def __capture_frame(self):
        """Capture a single frame using PiCamera2.

        Detects hangs by checking if time since last successful capture
        exceeds CAPTURE_TIMEOUT. The actual timeout detection and restart
        is handled in the main loop.
        """
        try:
            frame = self.cap.capture_array("main")
            if frame is not None and self._flip_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            return frame
        except Exception as e:
            self.logger.error(f"Failed to capture frame: {e}")
            return None


class CameraWatchdog(Thread):
    """Daemon thread that monitors Camera processes and respawns them if they die.

    Checks each registered camera every second. If a camera process has exited,
    waits a short delay (for the kernel to release the device), respawns it,
    and re-registers it with the HealthMonitor.

    Usage:
        watchdog = CameraWatchdog(health_monitor=health)
        watchdog.add_camera("head_camera", head_camera)
        watchdog.add_camera("left_camera", left_camera)
        watchdog.start()
    """

    RESPAWN_DELAY = 5    # initial seconds to wait before respawning
    MAX_RESPAWN_DELAY = 300  # max backoff (5 minutes) for repeatedly failing cameras

    def __init__(self, health_monitor=None) -> None:
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__)
        self._health = health_monitor
        self._cameras: dict = {}
        self._fail_counts: dict = {}

    def add_camera(self, name: str, camera: Camera) -> None:
        """Register a camera process to be monitored.

        Args:
            name: Health monitor registration name for this camera.
            camera: The Camera process instance.
        """
        self._cameras[name] = camera
        self._fail_counts[name] = 0

    def shutdown(self) -> None:
        """Terminate all managed camera processes."""
        for name, camera in self._cameras.items():
            if camera.is_alive():
                camera.terminate()
        for name, camera in self._cameras.items():
            camera.join(timeout=2)
            if camera.is_alive():
                camera.kill()

    def run(self) -> None:
        """Monitor loop — checks all cameras every second.

        Uses exponential backoff for cameras that keep failing (e.g., hardware
        not connected). Resets backoff after a successful run of 30+ seconds.
        """
        last_alive_time: dict = {}
        while True:
            sleep(1)
            for name in list(self._cameras.keys()):
                camera = self._cameras[name]
                if camera.is_alive():
                    last_alive_time[name] = time()
                    # Reset fail count if camera has been alive for 30+ seconds
                    if self._fail_counts[name] > 0:
                        alive_duration = time() - last_alive_time.get(name, time())
                        if alive_duration > 30:
                            self._fail_counts[name] = 0
                    continue

                self._fail_counts[name] += 1
                delay = min(self.RESPAWN_DELAY * (2 ** (self._fail_counts[name] - 1)),
                            self.MAX_RESPAWN_DELAY)
                self.logger.warning(
                    f"Camera {name} died (exit code {camera.exitcode}), "
                    f"respawn attempt {self._fail_counts[name]}, "
                    f"waiting {delay}s...")
                camera.join(timeout=2)
                sleep(delay)
                new_camera = camera.respawn()
                new_camera.start()
                self._cameras[name] = new_camera
                last_alive_time[name] = time()
                if self._health:
                    self._health.register(name, new_camera)
                self.logger.info(f"Camera {name} respawned")


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
