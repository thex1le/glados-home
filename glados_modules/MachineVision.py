from json import loads as json_loads
from threading import Thread
from time import time
from datetime import datetime
from typing import Dict, Callable

# 3rd party
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from paho.mqtt.client import MQTTMessage

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.RxTx import DataRecv
from glados_modules.RtspServer import RTSPServer
from glados_modules.MqttClient import MQTTClient, CameraMessageBuilder
from glados_modules.GLaDosEnums import CameraEnum, VisionResultsEnum, TrackingEnums


class GLaDOSServerException(Exception):
    pass


class YoloDetect(Thread, MQTTClient):
    def __init__(self, configfile):
        # Internal initialization
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = "yolo_detector"
        self.logger = setup_logger(name=self.__name__)
        self.configfile = configfile
        broker = self.configfile['MQTT']['mqtt_server_ip']
        port = self.configfile['MQTT']['mqtt_port']
        self.cmd_topic: str = CameraEnum.MQTT_RESPONSE_TOPIC.value
        self.status_topic: str = CameraEnum.MQTT_STATUS_TOPIC.value
        self.movent_topic: str = TrackingEnums.MQTT_MOVEMENT_ALERT.value
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd}
        MQTTClient.__init__(self, broker, port)
        cam_conf = self.configfile['CAMERAS']
        self.skip_image: bool = False

        # Camera configurations for each camera
        self.cam_configs = {
            f"/{cam_conf[CameraEnum.CAMERA_HEAD_FACTORY.value]}": {
                CameraEnum.MSG_RESOLUTION.value: tuple(cam_conf[CameraEnum.CAMERA_HEAD_RESOLUTION.value].split(',')),
                CameraEnum.MSG_FPS.value: int(cam_conf[CameraEnum.CAMERA_HEAD_FPS.value]),
                "tracker_thread": None},
            f"/{cam_conf[CameraEnum.CAMERA_LEFT_FACTORY.value]}": {
                CameraEnum.MSG_RESOLUTION.value: tuple(cam_conf[CameraEnum.CAMERA_LEFT_RESOLUTION.value].split(',')),
                CameraEnum.MSG_FPS.value: int(cam_conf[CameraEnum.CAMERA_LEFT_FPS.value]),
                "tracker_thread": None},
            f"/{cam_conf[CameraEnum.CAMERA_RIGHT_FACTORY.value]}": {
                CameraEnum.MSG_RESOLUTION.value: tuple(cam_conf[CameraEnum.CAMERA_RIGHT_RESOLUTION.value].split(',')),
                CameraEnum.MSG_FPS.value: int(cam_conf[CameraEnum.CAMERA_RIGHT_FPS.value]),
                "tracker_thread": None}
        }

        self.rtsp_port = int(self.configfile['RTSP']['rtsp_port'])
        self.rtsp_server_ip = self.configfile['RTSP']['rtsp_server_ip']
        model = configfile["YOLO"]["model"]
        self.tracker_yaml = configfile["YOLO"]["tracker"]

        self.logger.debug(f"YOLOv8 model started with {model}")
        self.model = YOLO(model)

        # Image receiver setup
        self.image_get = DataRecv(configfile=self.configfile, location=f"{self.__name__}_zmq_rx")
        self.image_get.start()

        # Start RTSP servers
        for key in self.cam_configs.keys():
            msg = {"status": f"Starting the RTSP server on rtsp://{self.rtsp_server_ip}:{self.rtsp_port}{key}"}
            status = CameraMessageBuilder.send_status(key, msg)
            self.send_command(status, self.status_topic)
            self.logger.info(msg)
        self.rtsp = RTSPServer(self.cam_configs)

    def handle_cmd(self, msg: MQTTMessage):
        j_msg = json_loads(msg.payload.decode())
        if j_msg.get(TrackingEnums.MSG_COMMAND_KEY.value) == TrackingEnums.MSG_MOVEMENT_KEY.value:
            movement = j_msg.get(TrackingEnums.MSG_MOVEMENT_KEY.value)
            if movement == TrackingEnums.MSG_MOVING_TRUE.value:
                with self._lock:
                    # robot is moving set the skip image to true
                    self.skip_image = True
            elif movement == TrackingEnums.MSG_MOVING_FALSE.value:
                with self._lock:
                    self.skip_image = False

    def __translate_results(self, results):
        name_key = VisionResultsEnum.YOLO_CLASS_NAME_KEY.value
        count_key = VisionResultsEnum.VISION_RESULTS_COUNT_KEY.value
        class_name_key = VisionResultsEnum.VISION_RESULTS_CLASS_KEY.value
        objects_key = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
        ts_key = VisionResultsEnum.VISION_RESULTS_TS_KEY.value
        results_dict = {}
        for y_class in results:
            if y_class is None:
                continue
            self.logger.debug(f"Translating {y_class} with type {type(y_class)}")
            y_class_data = json_loads(y_class.tojson())
            for cname in y_class_data:
                class_name = cname.get(name_key)
                if not class_name:
                    self.logger.warning(f"No '{name_key}' found in {cname}")
                    continue
                if class_name in results_dict:
                    results_dict[class_name][count_key] += 1
                    results_dict[class_name][ts_key] = time()
                    results_dict[class_name][objects_key].append(cname)
                else:
                    results_dict[class_name] = {
                        count_key: 1,
                        objects_key: [cname],
                        class_name_key: class_name,
                        ts_key: time()
                    }
        self.logger.debug(f"Translated results: {results_dict}")
        return results_dict

    def run_tracker_for_camera(self, camera_key):
        """
        Each camera has its own YOLO tracker running in a separate thread.
        """
        self.logger.info(f"Starting tracker for camera {camera_key}")

        # Run the tracker for this camera
        while True:
            try:
                image_dict = self.image_get.get_data_from_queue(True)
                camera_location = image_dict[CameraEnum.MSG_LOCATION_KEY.value]
                # Ensure the image is from this camera
                if camera_location != camera_key:
                    continue
                self.logger.debug(f"Processing image from {camera_location}")
                # Process the image and track objects
                camera_location = image_dict[CameraEnum.MSG_LOCATION_KEY.value]
                if camera_location == CameraEnum.CAMERA_HEAD.value and self.skip_image is True:
                    # skip this image as robot head is moving and image is blured
                    continue
                sight = self.__yolo_process_image(image_dict)
                # the string slice strips the / off the front of the camera_location
                results = CameraMessageBuilder.send_results(camera_location[1:], self.__translate_results(sight))
                self.send_command(results, self.cmd_topic, qos=0)

            except Exception as e:
                self.logger.error(f"Error in tracker for camera {camera_key}: {e}")

    def start_tracking_threads(self):
        """
        Start a separate tracking thread for each camera.
        """
        for camera_key in self.cam_configs.keys():
            thread = Thread(target=self.run_tracker_for_camera, args=(camera_key,), daemon=True)
            thread.start()
            self.cam_configs[camera_key]["tracker_thread"] = thread

    def __yolo_process_image(self, image_dict):
        camera_location = image_dict[CameraEnum.MSG_LOCATION_KEY.value]
        raw = image_dict[CameraEnum.MSG_RAW_IMAGE.value]
        width, height = image_dict[CameraEnum.MSG_RESOLUTION.value]
        image = raw.reshape((height, width, 3))  # RGB888 format has 3 channels

        # Process the image using YOLO tracking
        results = self.model.track(source=image, device="cuda", tracker=self.tracker_yaml)
        self.logger.debug(f"Yolo processed image for camera {camera_location}")

        # Annotating and sending the processed image
        annotator = Annotator(image)
        image_center = (width // 2, height // 2)
        cv2.line(image, (image_center[0] - 10, image_center[1]), (image_center[0] + 10, image_center[1]), (0, 255, 0),
                 2)
        cv2.line(image, (image_center[0], image_center[1] - 10), (image_center[0], image_center[1] + 10), (0, 255, 0),
                 2)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                x1, y1, x2, y2 = map(int, b.tolist())
                c = box.cls
                conf = box.conf.item()
                label = f"{self.model.names[int(c)]} {conf:.2f}"
                annotator.box_label(b, label)
                self.logger.debug(f"Labeled image with {label}")
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                object_center = (center_x, center_y)
                color_current = (0, 0, 255)
                cv2.circle(image, object_center, radius=5, color=color_current, thickness=-1)
                self.logger.debug(f"Drew circle at {object_center} (center of bounding box).")
                cv2.arrowedLine(image, object_center, image_center, color=(255, 0, 0), thickness=2)
                self.logger.debug(f"Drew arrow from {object_center} to {image_center}.")

        a_image = annotator.result()
        timestamp = datetime.now().isoformat(timespec='seconds')
        yellow_orange_color = (0, 140, 255)
        position = (10, a_image.shape[0] - 10)
        cv2.putText(a_image, timestamp, position, cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=yellow_orange_color, thickness=2)
        self.logger.debug(f"Sending image to RTSP server factory: {image_dict[CameraEnum.MSG_LOCATION_KEY.value]}")
        self.rtsp.send_data(image_dict["camera"], a_image)
        return results

    def run(self):
        status = CameraMessageBuilder.send_status(self.__name__, "Machine Vision Started")
        self.send_command(status, self.status_topic)

        # Start tracking for each camera in separate threads
        self.start_tracking_threads()
        self.logger.info("Started YOLO tracking threads for all cameras")
