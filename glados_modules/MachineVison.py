from json import loads as json_loads
from json import dumps as json_dumps
from threading import Thread

#3rd party
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.RxTx import DataRecv
from glados_modules.RtspServer import RTSPServer
from glados_modules.MqttClient import MQTTClient


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
        self.cmd_topic: str = "vision/camera_response"
        cam_conf = self.configfile['CAMERAS']
        self.cam_configs = {
            f"/{cam_conf['Camera_Head_Factory']}": tuple(cam_conf["Camera_Head_Resolution"].split(',')),
            f"/{cam_conf['Camera_Left_Factory']}": tuple(cam_conf["Camera_Left_Resolution"].split(',')),
            f"/{cam_conf['Camera_Right_Factory']}": tuple(cam_conf["Camera_Right_Resolution"].split(','))
        }
        rtsp_port = int(self.configfile['RTSP']['rtsp_port'])
        rtsp_server_ip = self.configfile['RTSP']['rtsp_server_ip']
        model = configfile["YOLO"]["model"]
        self.logger.debug(f"YOLO model started with {model}")
        self.model = YOLO(model)
        self.sight = None
        self.image_get = DataRecv(configfile=self.configfile, location=f"{self.__name__}_zmq_rx" )
        self.image_get.start()
        for key in self.cam_configs.keys():
            msg = f"Starting the RTSP server on rtsp://{rtsp_server_ip}:{rtsp_port}{key}"
            self.client.publish("status", msg)
            self.logger.info(msg)
        self.rtsp = RTSPServer(self.cam_configs)

    def get_sight(self):
        return self.sight

    def __translate_results(self, results):
        results_dict = {}
        for y_class in results:
            if y_class is None:
                continue
            self.logger.debug(f"Translating {y_class} with type {type(y_class)}")
            for cname in json_loads(y_class.tojson()):
                name = cname["name"]
                if name in list(results_dict.keys()):
                    results_dict[name]["count"] += 1
                    results_dict[name]["objects"].append(cname)
                else:
                    results_dict[name] = {"count": 1, "objects": [cname], "class_name": name}
        self.logger.debug(results_dict)

    def __yolo_process_image(self, image_dict):
        # pass image to rtsp...
        raw = image_dict["raw"]
        width, height = self.cam_configs[image_dict["camera"]]
        yuv420_data = raw.reshape((int(height) * 3) // 2, int(width))
        image = cv2.cvtColor(yuv420_data, cv2.COLOR_YUV420p2BGR)
        #image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        results = self.model(image)
        self.logger.debug(f"Yolo has processed raw image")
        annotator = Annotator(image)
        for r in results:
            annotator = Annotator(image)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])
                self.logger.debug(f"Labeled image with, {self.model.names[int(c)]}")
        a_image = annotator.result()
        self.logger.debug(f"Sending image to rstp server factory: {image_dict['camera']}")
        self.rtsp.send_data(image_dict["camera"], a_image)
        return results

    def process_image(self, image, debug_file_name='raw_rx.jpg'):
        name = f"{debug_file_name}"
        #cv2.imwrite(name, image)
        self.logger.debug(f"Wrote out sample debug image to {name}")
        self.sight = self.__yolo_process_image(image)
        self.logger.debug(f"Sending back process dict of seen data for camera {image['camera']}")
        self.client.publish(self.cmd_topic, json_dumps({image['camera']: self.__translate_results(self.sight)}))

    def run(self):
        self.client.publish("status", "Machine Vision Started")
        while True:
            image_dict = self.image_get.get_data(True)
            msg = f"Got image from sender {image_dict.get('camera', 'None')}"
            self.logger.debug(msg)
            try:
                self.process_image(image_dict)
            except Exception as e:
                msg = f"Image Error: {e}"
                self.logger.error(msg)

