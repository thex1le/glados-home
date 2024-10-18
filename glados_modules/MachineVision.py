from json import loads as json_loads
from threading import Thread
from time import time
from datetime import datetime
from typing import List, Dict, Any

# 3rd party
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
# get this here, https://github.com/Tau-J/rtmlib/tree/main
from rtmlib import Wholebody, draw_skeleton
import numpy as np

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.RxTx import DataRecv
from glados_modules.RtspServer import RTSPServer
from glados_modules.MqttClient import MQTTClient, CameraMessageBuilder
from glados_modules.GLaDosEnums import CameraEnum, VisionResultsEnum


class GLaDOSServerException(Exception):
    pass


class MLDetect(Thread, MQTTClient):
    def __init__(self, configfile):
        # Internal initialization
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
        # pose model
        self.coco_key_points = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
                                "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
                                "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
                                "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]

        # rtsp
        self.rtsp_port = int(self.configfile['RTSP']['rtsp_port'])
        self.rtsp_server_ip = self.configfile['RTSP']['rtsp_server_ip']
        # yolo
        self.model_config = configfile["YOLO"]["model"]
        self.tracker_yaml = configfile["YOLO"]["tracker"]

        self.logger.debug(f"YOLOv8 model started with {self.model_config}")

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

    def run_tracker_for_camera(self, camera_key, d_model, p_model):
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
                sight = self.__yolo_process_image(image_dict, d_model, p_model)
                # the string slice strips the / off the front of the camera_location
                results = CameraMessageBuilder.send_results(camera_location[1:], self.__translate_results(sight))
                self.send_command(results, self.cmd_topic, qos=0)

            except Exception as e:
                self.logger.error(f"Error in tracker for camera {camera_key}: {e}")

            except KeyboardInterrupt:
                break

    def start_tracking_threads(self):
        """
        Start a separate tracking thread for each camera.
        """
        self.logger.debug(f"YOLOv8 model started with {self.model_config}")
        pose_model = None
        for camera_key in self.cam_configs.keys():
            detection_model = YOLO(self.model_config)
            if camera_key[1:] == CameraEnum.CAMERA_HEAD.value:
                # only build a pose model for the camera head
                self.logger.debug("Creating Pose model")
                openpose_skeleton = False  # True for openpose-style, False for mmpose-style
                backend = 'onnxruntime'  # opencv, onnxruntime, openvino
                pose_model = Wholebody(to_openpose=openpose_skeleton, mode='balanced', backend=backend, device='cuda')
            thread = Thread(target=self.run_tracker_for_camera, args=(camera_key,
                                                                      detection_model, pose_model), daemon=True)
            thread.start()
            self.cam_configs[camera_key]["tracker_thread"] = thread

    def __yolo_process_image(self, image_dict, d_model, p_model) -> dict:
        raw = image_dict[CameraEnum.MSG_RAW_IMAGE.value]
        width, height = image_dict[CameraEnum.MSG_RESOLUTION.value]
        image = raw.reshape((height, width, 3))  # RGB888 format has 3 channels

        camera_location = image_dict[CameraEnum.MSG_LOCATION_KEY.value]

        # Process the image using YOLO tracking
        results = d_model.track(source=image, device="cuda", tracker=self.tracker_yaml)
        self.logger.debug(f"Yolo processed image for camera {camera_location}")
        # Annotating and sending the processed image
        annotator = Annotator(image)
        image_center = (width // 2, height // 2)
        cv2.line(image, (image_center[0] - 10, image_center[1]),
                 (image_center[0] + 10, image_center[1]), (0, 255, 0),2)
        cv2.line(image, (image_center[0], image_center[1] - 10), (image_center[0], image_center[1] + 10),
                 (0, 255, 0),2)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                x1, y1, x2, y2 = map(int, b.tolist())
                c = box.cls
                conf = box.conf.item()
                label = f"{d_model.names[int(c)]} {conf:.2f}"
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
        position = (10, 10)
        cv2.putText(a_image, camera_location, position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=yellow_orange_color, thickness=2)
        if p_model is not None:
            # we have a model, must be head camera, run model and draw points
            key_points, scores = p_model(image)
            # TODO read key point threshold from enum or config file
            a_image = draw_skeleton(a_image, key_points, scores, kpt_thr=0.5)
            self.assign_key_points_to_response(results, key_points, scores)
        self.logger.debug(f"Sending image to RTSP server factory: {image_dict[CameraEnum.MSG_LOCATION_KEY.value]}")
        self.rtsp.send_data(image_dict["camera"], a_image)
        return results

    def merge_keypoints_to_dict(self, coords_list: np.ndarray, scores_list: np.ndarray) -> List[List[Dict[str, float]]]:
        """
        Merges coordinates and confidence scores into a list of dictionaries for each person,
        while tracking the index of each keypoint.

        Args:
            coords_list (np.ndarray): A list of arrays with shape (N, M, 2), where N is the number of people,
                                      M is the number of key points, and 2 represents the (x, y) coordinates.
            scores_list (np.ndarray): A list of arrays with shape (N, M), where N is the number of people
                                      and M is the number of key points, representing confidence scores.
        Returns:
            List[List[Dict[str, float]]]: A list where each element corresponds to a person and contains a list of
                                          dictionaries with "x", "y", "confidence", and "index" keys for each keypoint.
        """
        merged_data = []
        # Iterate through each person's set of coordinates and scores
        for coords, scores in zip(coords_list, scores_list):
            person_data = []
            # Loop through each keypoint, using enumerate to track the index
            for index, ((x, y), score) in enumerate(zip(coords, scores)):
                keypoint = {
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(score),
                    "location": VisionResultsEnum.VISION_POSE_KEY_POINTS_COCO_WHOLE_BODY.value[index]}
                person_data.append(keypoint)
            merged_data.append(person_data)
        return merged_data

    def assign_key_points_to_response(self, response: Dict[str, Any],
                                      coords_list: np.ndarray, scores_list: np.ndarray) -> Dict[str, Any]:
        """
        Assigns keypoints to the appropriate person in the response dictionary if all keypoints
        lie within the defined bounding box.

        Args:
            response (Dict[str, Any]): The response dictionary containing objects with bounding boxes.
            coords_list (np.ndarray): A list of arrays with shape (N, M, 2) for coordinates.
            scores_list (np.ndarray): A list of arrays with shape (N, M) for confidence scores.

        Returns:
            Dict[str, Any]: The updated response dictionary with assigned key points.
        """
        # Merge the key points and scores into a list of dictionaries
        merged_key_points = self.merge_keypoints_to_dict(coords_list, scores_list)

        # Iterate over the response dictionary to find matching objects
        count = 0
        for person_data in response.get('person', {}).get('objects', []):
            # Get the bounding box coordinates
            box = person_data.get('box', {})
            x1, y1 = box.get('x1', 0), box.get('y1', 0)
            x2, y2 = box.get('x2', 0), box.get('y2', 0)
            # Filter key points that fit inside the bounding box
            filtered_key_points = []
            for key_points in merged_key_points:
                # Check if all key points lie within the bounding box
                if all(x1 <= kp['x'] <= x2 and y1 <= kp['y'] <= y2 for kp in key_points):
                    filtered_key_points = key_points
                    break  # Assign the first matching set of key points

            # Assign the filtered key points to the person data
            if filtered_key_points:
                response['person']['objects'][count]['pose'] = filtered_key_points
            count += 1
        return response

    def run(self):
        status = CameraMessageBuilder.send_status(self.__name__, "Machine Vision Started")
        self.send_command(status, self.status_topic)

        # Start tracking for each camera in separate threads
        self.start_tracking_threads()
        self.logger.info("Started YOLO tracking threads for all cameras")
