from json import loads as json_loads
from threading import Thread
from time import time
from datetime import datetime

#3rd party
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.RxTx import DataRecv
from glados_modules.RtspServer import RTSPServer
from glados_modules.MqttClient import MQTTClient, CameraMessageBuilder
from glados_modules.GLaDosEnums import CameraEnum, VisionResultsEnum


class GLaDOSServerException(Exception):
    pass


class YoloDetect(Thread, MQTTClient):
    def __init__(self, configfile):
        # internal libs, import here so its deps are not needed on other devices
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = "yolo_detector"
        self.logger = setup_logger(name=self.__name__)
        self.configfile = configfile
        broker = self.configfile['MQTT']['mqtt_server_ip']
        port = self.configfile['MQTT']['mqtt_port']
        MQTTClient.__init__(self, broker, port)
        self.cmd_topic: str = CameraEnum.MQTT_RESPONSE_TOPIC.value
        self.status_topic: str = CameraEnum.MQTT_STATUS_TOPIC.value
        cam_conf = self.configfile['CAMERAS']
        self.cam_configs = {
            f"/{cam_conf[CameraEnum.CAMERA_HEAD_FACTORY.value]}": {
                CameraEnum.MSG_RESOLUTION.value: tuple(cam_conf[CameraEnum.CAMERA_HEAD_RESOLUTION.value].split(',')),
                CameraEnum.MSG_FPS.value: int(cam_conf[CameraEnum.CAMERA_HEAD_FPS.value])},
            f"/{cam_conf[CameraEnum.CAMERA_LEFT_FACTORY.value]}": {
                CameraEnum.MSG_RESOLUTION.value: tuple(cam_conf[CameraEnum.CAMERA_LEFT_RESOLUTION.value].split(',')),
                CameraEnum.MSG_FPS.value: int(cam_conf[CameraEnum.CAMERA_LEFT_FPS.value])},
            f"/{cam_conf[CameraEnum.CAMERA_RIGHT_FACTORY.value]}": {
                CameraEnum.MSG_RESOLUTION.value: tuple(cam_conf[CameraEnum.CAMERA_RIGHT_RESOLUTION.value].split(',')),
                CameraEnum.MSG_FPS.value: int(cam_conf[CameraEnum.CAMERA_RIGHT_FPS.value])}}
        rtsp_port = int(self.configfile['RTSP']['rtsp_port'])
        rtsp_server_ip = self.configfile['RTSP']['rtsp_server_ip']
        model = configfile["YOLO"]["model"]
        tracker_config = configfile["YOLO"]["tracker"]

        self.logger.debug(f"YOLOv8 model started with {model} using {tracker_config}")
        # Initialize YOLOv8 model with tracking
        self.model = YOLO(model)
        self.tracker_config = tracker_config

        self.sight = None
        self.image_get = DataRecv(configfile=self.configfile, location=f"{self.__name__}_zmq_rx")
        self.image_get.start()
        for key in self.cam_configs.keys():
            msg = {"status": f"Starting the RTSP server on rtsp://{rtsp_server_ip}:{rtsp_port}{key}"}
            status = CameraMessageBuilder.send_status(key, msg)
            self.send_command(status, self.status_topic)
            self.logger.info(msg)
        self.rtsp = RTSPServer(self.cam_configs)

    def get_sight(self):
        return self.sight

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

    def __yolo_process_image(self, image_dict):
        raw = image_dict[CameraEnum.MSG_RAW_IMAGE.value]
        width, height = image_dict[CameraEnum.MSG_RESOLUTION.value]
        image = raw.reshape((height, width, 3))  # RGB888 format has 3 channels
        results = self.model.track(source=image, device="cuda", tracker=self.tracker_config)
        self.logger.debug(f"Yolo has processed raw image with tracking")
        annotator = Annotator(image)
        image_center = (width // 2, height // 2)
        color_target = (0, 255, 0)
        thickness = 2
        plus_size = 10
        cv2.line(image, (image_center[0] - plus_size, image_center[1]),
                 (image_center[0] + plus_size, image_center[1]), color_target, thickness)
        cv2.line(image, (image_center[0], image_center[1] - plus_size),
                 (image_center[0], image_center[1] + plus_size), color_target, thickness)
        self.logger.debug(f"Drew plus sign at {image_center} (center of the image).")

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

    def process_image(self, image):
        self.sight = self.__yolo_process_image(image)
        self.logger.debug(
            f"Sending back process dict of seen data for camera {image[CameraEnum.MSG_LOCATION_KEY.value]}")
        cam_name = image[CameraEnum.MSG_LOCATION_KEY.value][1:]
        results = CameraMessageBuilder.send_results(cam_name, self.__translate_results(self.sight))
        self.send_command(results, self.cmd_topic, qos=0)

    def run(self):
        status = CameraMessageBuilder.send_status(self.__name__, "Machine Vision Started")
        self.send_command(status, self.status_topic)
        while True:
            image_dict = self.image_get.get_data_from_queue(True)
            self.logger.debug(f"Got image from sender {image_dict.get(CameraEnum.MSG_LOCATION_KEY.value, 'None')}")
            try:
                self.process_image(image_dict)
            except Exception as e:
                self.logger.error(f"Image Error: {e}")
