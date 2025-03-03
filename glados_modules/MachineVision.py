from json import loads as json_loads
from threading import Thread
from time import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# 3rd party
import cv2
from torch.serialization import add_safe_globals
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from torch.serialization import safe_globals
from torch.nn.modules.container import Sequential
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv
add_safe_globals([DetectionModel, Sequential, Conv])

# because safe globals is being fucking stupid, monkey patch back the old way
import torch
# Save the original torch.load function.
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    # Force weights_only to be False.
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# get this here, https://github.com/Tau-J/rtmlib/tree/main
from rtmlib import Wholebody, draw_skeleton
import numpy as np

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.Rtsp_Rx import RtspConsumer
from glados_modules.RtspServer import RTSPServer
from glados_modules.MqttClient import MQTTClient, CameraMessageBuilder
from glados_modules.GLaDosEnums import CameraEnum, VisionResultsEnum, SystemEnums, LoggingEnums
from glados_modules.GladosData import ServoLocation
from glados_modules.Tracker import MotionTrack


class GLaDOSServerException(Exception):
    """
    Custom exception class used for GLaDOS server errors.
    """
    pass


class MLDetect(Thread, MQTTClient):
    """
    Handles YOLOv8 object detection and RTM (Whole body) pose estimation.
    Manages reading frames from RTSP, running detection and pose estimation,
    sending results via MQTT, and streaming annotated frames via an RTSP server.
    """

    def __init__(self, configfile: Dict[str, Any]) -> None:
        """
        Initialize the MLDetect class with a given configuration.

        Args:
            configfile (Dict[str, Any]): The configuration dictionary containing
                                         MQTT settings, camera settings, YOLO model settings, etc.
        """
        # Internal initialization
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = "vision_detector"
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.configfile = configfile
        r_x, r_y = self.configfile[CameraEnum.CONFIG_HEAD.value][
            f"{CameraEnum.CAMERA_HEAD.value}_{CameraEnum.MSG_RESOLUTION.value}"].split(',')
        head_cam_resolution = MotionTrack.camera_tuple(int(r_x), int(r_y))
        mh = SystemEnums.CONFIG_HEAD_MQTT.value
        broker = self.configfile[mh][SystemEnums.MQTT_SERVER_IP.value]
        port = self.configfile[mh][SystemEnums.MQTT_PORT.value]
        MQTTClient.__init__(self, broker, port)
        self.cmd_topic: str = CameraEnum.MQTT_RESPONSE_TOPIC.value
        self.status_topic: str = CameraEnum.MQTT_STATUS_TOPIC.value
        cam_conf = self.configfile[CameraEnum.CONFIG_HEAD.value]
        # track servo movement, only process images from head camera when we're not moving
        bt = MQTTClient.broker_tuple(broker, port)
        self.servos = ServoLocation(bt)
        self.motion_tracking = MotionTrack(broker=bt, camera_resolution=head_cam_resolution)
        # Camera configurations for each camera
        self.cam_configs = {
            cam_conf[CameraEnum.CAMERA_HEAD_FACTORY.value]: {
                CameraEnum.MSG_RESOLUTION.value: tuple(cam_conf[CameraEnum.CAMERA_HEAD_RESOLUTION.value].split(',')),
                CameraEnum.MSG_FPS.value: int(cam_conf[CameraEnum.CAMERA_HEAD_FPS.value]),
                CameraEnum.MSG_RTSP_URI.value: f"rtsp://{cam_conf[CameraEnum.CAMERA_HEAD_RTSP_IP.value]}:"
                                               f"{cam_conf[CameraEnum.CAMERA_HEAD_PORT.value]}/"
                                               f"{cam_conf[CameraEnum.CAMERA_HEAD_FACTORY.value]}",
                "tracker_thread" : None},
            cam_conf[CameraEnum.CAMERA_LEFT_FACTORY.value]: {
                CameraEnum.MSG_RESOLUTION.value: tuple(cam_conf[CameraEnum.CAMERA_LEFT_RESOLUTION.value].split(',')),
                CameraEnum.MSG_FPS.value: int(cam_conf[CameraEnum.CAMERA_LEFT_FPS.value]),
                CameraEnum.MSG_RTSP_URI.value: f"rtsp://{cam_conf[CameraEnum.CAMERA_LEFT_RTSP_IP.value]}:"
                                               f"{cam_conf[CameraEnum.CAMERA_LEFT_PORT.value]}/"
                                               f"{cam_conf[CameraEnum.CAMERA_LEFT_FACTORY.value]}",
                "tracker_thread": None},
            cam_conf[CameraEnum.CAMERA_RIGHT_FACTORY.value]: {
                CameraEnum.MSG_RESOLUTION.value: tuple(cam_conf[CameraEnum.CAMERA_RIGHT_RESOLUTION.value].split(',')),
                CameraEnum.MSG_FPS.value: int(cam_conf[CameraEnum.CAMERA_RIGHT_FPS.value]),
                CameraEnum.MSG_RTSP_URI.value: f"rtsp://{cam_conf[CameraEnum.CAMERA_RIGHT_RTSP_IP.value]}:"
                                               f"{cam_conf[CameraEnum.CAMERA_RIGHT_PORT.value]}/"
                                               f"{cam_conf[CameraEnum.CAMERA_RIGHT_FACTORY.value]}",
                "tracker_thread": None}
        }

        # rtsp
        self.rtsp_port = int(self.configfile['RTSP']['rtsp_port'])
        self.rtsp_server_ip = self.configfile['RTSP']['rtsp_server_ip']
        # yolo
        self.model_config = configfile["YOLO"]["model"]
        self.tracker_yaml = configfile["YOLO"]["tracker"]

        self.logger.debug(f"YOLOv8 model started with {self.model_config}")

        # Start RTSP servers
        for key in self.cam_configs.keys():
            msg = {"status": f"Starting the RTSP server on rtsp://{self.rtsp_server_ip}:{self.rtsp_port}/{key}"}
            status = CameraMessageBuilder.send_status(key, msg)
            self.send_command(status, self.status_topic)
            self.logger.info(msg)

        self.rtsp = RTSPServer(self.cam_configs)

    def __translate_results(self, results: List[Any]) -> Dict[str, Any]:
        """
        Convert YOLO detection/tracking results into a dictionary.

        Args:
            results (List[Any]): YOLO detection results, which are objects that can be
                                 converted to JSON format for each class.

        Returns:
            Dict[str, Any]: A dictionary containing translated results with counts, timestamps, and objects.
        """
        name_key = VisionResultsEnum.YOLO_CLASS_NAME_KEY.value
        count_key = VisionResultsEnum.VISION_RESULTS_COUNT_KEY.value
        class_name_key = VisionResultsEnum.VISION_RESULTS_CLASS_KEY.value
        objects_key = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
        ts_key = VisionResultsEnum.VISION_RESULTS_TS_KEY.value
        results_dict: Dict[str, Any] = {}

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

    def run_tracker_for_camera(self, camera_key: str, d_model: YOLO, p_model: Optional[Wholebody]) -> None:
        """
        Run the YOLO tracker for a specific camera in a loop.

        Args:
            camera_key (str): The identifier for the camera (factory name).
            d_model (YOLO): The YOLO detection model.
            p_model (Optional[Whole body]): The Whole body pose model or None if not used.
        """
        self.logger.info(f"Starting tracker for camera {camera_key}")
        # Run the tracker for this camera
        image_get = RtspConsumer(location=camera_key,
                                 uri=self.cam_configs[camera_key][CameraEnum.MSG_RTSP_URI.value])
        while True:
            try:
                image_dict = image_get.get_frame()
                # check if we are moving
                if camera_key == CameraEnum.CAMERA_HEAD.value:
                    # camera head thread, don't process image if we are moving to reduce noise
                    if self.servos.check_movement() is True:
                        continue

                self.logger.debug(f"Processing image from {camera_key}")
                # Process the image and track objects
                sight = self.__process_image(image_dict, d_model, p_model)
                results = CameraMessageBuilder.send_results(camera_key, sight)
                self.send_command(results, self.cmd_topic, qos=0)

            except Exception as e:
                self.logger.error(f"Error in tracker for camera {camera_key}: {e}")

            except KeyboardInterrupt:
                break

    def start_tracking_threads(self) -> None:
        """
        Start a separate tracking thread for each camera.
        """
        self.logger.debug(f"YOLOv8 model started with {self.model_config}")
        pose_model: Optional[Wholebody] = None
        for camera_key in self.cam_configs.keys():
            with safe_globals([DetectionModel, Sequential, Conv]):
                detection_model = YOLO(self.model_config)
            if camera_key == CameraEnum.CAMERA_HEAD.value:
                # only build a pose model for the camera head
                self.logger.debug("Creating Pose model")
                openpose_skeleton = False  # True for openpose-style, False for mmpose-style
                backend = 'onnxruntime'  # opencv, onnxruntime, openvino
                pose_model = Wholebody(to_openpose=openpose_skeleton, mode='balanced', backend=backend, device='cuda')
            thread = Thread(target=self.run_tracker_for_camera, args=(camera_key,
                                                                       detection_model, pose_model), daemon=True)
            thread.start()
            self.cam_configs[camera_key]["tracker_thread"] = thread

    def __process_image(self, image_dict: Dict[str, Any], d_model: YOLO,
                        p_model: Optional[Wholebody]) -> Dict[str, Any]:
        """
        Process a single image with YOLO detection and optional RTM pose estimation.
        Annotate the image, draw bounding boxes and pose skeleton, and stream via RTSP.

        Args:
            image_dict (Dict[str, Any]): Dictionary containing the raw image and metadata like
                                         resolution and camera location.
            d_model (YOLO): The YOLO detection model.
            p_model (Optional[Whole body]): The Whole body pose model, None if no pose detection.

        Returns:
            Dict[str, Any]: The translated results containing detections (and pose assignments if pose model is used).
        """
        image = image_dict[CameraEnum.MSG_RAW_IMAGE.value]
        width, height = image_dict[CameraEnum.MSG_RESOLUTION.value]
        camera_location = image_dict[CameraEnum.MSG_LOCATION_KEY.value]
        # Process the image using YOLO tracking
        results = d_model.track(source=image, device="cuda", tracker=self.tracker_yaml)
        self.logger.debug(f"Yolo processed image for camera {camera_location}")
        # Annotating and sending the processed image
        annotator = Annotator(image)
        image_center = (int(width // 2), int(height // 2))
        cv2.line(image, (image_center[0] - 10, image_center[1]),
                 (image_center[0] + 10, image_center[1]), (0, 255, 0), 2)
        cv2.line(image, (image_center[0], image_center[1] - 10),
                 (image_center[0], image_center[1] + 10), (0, 255, 0), 2)

        # loop over all the results and draw and log the discovered items on the image
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
        # place a yellow orange time stamp at the bottom of left of the frame
        cv2.putText(a_image, timestamp, position, cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=yellow_orange_color, thickness=2)
        position = (10, 10)
        cv2.putText(a_image, camera_location, position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=yellow_orange_color, thickness=2)
        t_results = self.__translate_results(results)

        # if a pose model exists, plot the pose on the camera
        if p_model is not None:
            # we have a model, must be head camera, run model and draw points
            key_points, scores = p_model(image)
            # TODO read key point threshold from enum or config file
            a_image = draw_skeleton(a_image, key_points, scores, kpt_thr=0.5)
            # assign results to bounding boxes
            t_results = self.assign_key_points_to_response(t_results, key_points, scores)

        self.logger.debug(f"Sending image to RTSP server factory: {image_dict[CameraEnum.MSG_LOCATION_KEY.value]}")
        self.rtsp.send_data(image_dict["camera"], a_image)
        return t_results

    @staticmethod
    def merge_key_points_to_dict(cords_list: np.ndarray, scores_list: np.ndarray) -> List[List[Dict[str, float]]]:
        """
        Merge coordinates and confidence scores into a list of dictionaries for each person,
        while tracking the index of each keypoint.

        Args:
            cords_list (np.ndarray): A list of arrays with shape (N, M, 2), where N is the number of people,
                                     M is the number of key points, and 2 represents the (x, y) coordinates.
            scores_list (np.ndarray): A list of arrays with shape (N, M), where N is the number of people
                                      and M is the number of key points, representing confidence scores.

        Returns:
            List[List[Dict[str, float]]]: A list where each element corresponds to a person and contains a list of
                                          dictionaries with "x", "y",
                                          "confidence", and "location" keys for each keypoint.
        """
        merged_data: List[List[Dict[str, float]]] = []
        # Iterate through each person's set of coordinates and scores
        for cords, scores in zip(cords_list, scores_list):
            person_data: List[Dict[str, float]] = []
            # Loop through each keypoint, using enumerate to track the index
            for index, ((x, y), score) in enumerate(zip(cords, scores)):
                keypoint = {
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(score),
                    "location": VisionResultsEnum.VISION_POSE_KEY_POINTS_COCO_WHOLE_BODY.value[index]
                }
                person_data.append(keypoint)
            merged_data.append(person_data)
        return merged_data

    def assign_key_points_to_response(self,
                                      response: Dict[str, Any],
                                      cords_list: np.ndarray,
                                      scores_list: np.ndarray,
                                      percentage_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Assign key points to the appropriate person in the response dictionary if at least a certain percentage
        of key points lie within the defined bounding box.

        Args:
            response (Dict[str, Any]): The response dictionary containing objects with bounding boxes.
            cords_list (np.ndarray): A list of arrays with shape (N, M, 2) for coordinates.
            scores_list (np.ndarray): A list of arrays with shape (N, M) for confidence scores.
            percentage_threshold (float): The required percentage (0.0 to 1.0) of key points that must be
                                            within the bounding box to assign them.

        Returns:
            Dict[str, Any]: The updated response dictionary with assigned key points.
        """
        # Merge the key points and scores into a list of dictionaries
        merged_key_points = MLDetect.merge_key_points_to_dict(cords_list, scores_list)

        # Iterate over the response dictionary to find matching objects
        count = 0
        for person_data in response.get('person', {}).get('objects', []):
            # Get the bounding box coordinates
            box = person_data.get('box', {})
            x1, y1 = box.get('x1', 0), box.get('y1', 0)
            x2, y2 = box.get('x2', 0), box.get('y2', 0)

            # Filter key points that fit inside the bounding box based on the given threshold
            filtered_key_points = []
            for key_points in merged_key_points:
                total_points = len(key_points)
                if total_points == 0:
                    continue  # Avoid division by zero if no key points are present

                # Count the number of key points inside the bounding box
                points_in_box = sum(
                    1 for kp in key_points if x1 <= kp['x'] <= x2 and y1 <= kp['y'] <= y2
                )
                # If the percentage of points inside the box meets or exceeds the threshold, assign them.
                if (points_in_box / total_points) >= percentage_threshold:
                    filtered_key_points = key_points
                    self.logger.debug(f"Linking {key_points} to {person_data}")
                    break  # Use the first matching set of key points

            # Assign the filtered key points to the person data if any were found
            if filtered_key_points:
                pose_dict = {kp["location"]: kp for kp in filtered_key_points}
                response['person']['objects'][count]['pose'] = pose_dict
            count += 1

        return response

    def run(self) -> None:
        """
        Entry point for the MLDetect thread. Publishes a startup status message
        and then starts tracking threads for each camera.
        """
        status = CameraMessageBuilder.send_status(self.__name__, "Machine Vision Started")
        self.send_command(status, self.status_topic)
        # Start tracking for each camera in separate threads
        self.start_tracking_threads()
        self.logger.info("Started YOLO tracking threads for all cameras")
