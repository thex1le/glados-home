from json import loads as json_loads
from threading import Thread
from time import time, sleep
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import traceback

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
from glados_modules.RTSPClient import RtspConsumer
from glados_modules.RtspServer import RTSPServer
from glados_modules.MqttConnector import MQTTClient, CameraMessageBuilder
from glados_modules.GladosEnums import CameraEnum, VisionResultsEnum, SystemEnums, LoggingEnums, TraceEnums, FaceEnums, FeatureToggles
from glados_modules.VisionTracker import MotionTrack
from glados_modules.TraceLog import TraceLog
from glados_modules.VideoRecorder import RecordingSession, RecordingGate


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
        # Pre-populate topic_handler before MQTT connects (subscribes on_connect)
        self.topic_handler = {
            FeatureToggles.MQTT_RECORDING_TOPIC.value: self._handle_recording_command,
        }
        MQTTClient.__init__(self, broker, port)
        self.cmd_topic: str = CameraEnum.MQTT_RESPONSE_TOPIC.value
        self.status_topic: str = CameraEnum.MQTT_STATUS_TOPIC.value
        cam_conf = self.configfile[CameraEnum.CONFIG_HEAD.value]
        # track servo movement, only process images from head camera when we're not moving
        bt = MQTTClient.broker_tuple(broker, port)
        self.motion_tracking = MotionTrack(broker=bt, camera_resolution=head_cam_resolution)
        # Reuse MotionTrack's ServoLocation — avoids duplicate MQTT subscriptions
        self.servos = self.motion_tracking.servo_status
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
        self.kpt_threshold = float(configfile.get("YOLO", "keypoint_threshold",
                                                   fallback=str(VisionResultsEnum.KEYPOINT_DRAW_THRESHOLD.value)))

        self.logger.debug(f"YOLOv8 model started with {self.model_config}")

        # Start RTSP servers
        for key in self.cam_configs.keys():
            msg = {"status": f"Starting the RTSP server on rtsp://{self.rtsp_server_ip}:{self.rtsp_port}/{key}"}
            status = CameraMessageBuilder.send_status(key, msg)
            self.send_command(status, self.status_topic)
            self.logger.info(msg)

        self.rtsp = RTSPServer(self.cam_configs)

        # Face recognition (head camera only, toggleable via config)
        face_enabled = configfile.get(FaceEnums.CONFIG_HEAD.value,
                                       FaceEnums.ENABLED.value,
                                       fallback="True").strip().lower() == "true"
        if face_enabled:
            try:
                from glados_modules.FaceRecognition import GladosFaceRecognition
                face_db = configfile.get(FaceEnums.CONFIG_HEAD.value,
                                         FaceEnums.FACE_DB_PATH.value,
                                         fallback=FaceEnums.FACE_DB_PATH.value)
                emotion_enabled = configfile.get(FaceEnums.CONFIG_HEAD.value,
                                                  FaceEnums.EMOTION_ENABLED.value,
                                                  fallback="True").strip().lower() == "true"
                self.face_recognition = GladosFaceRecognition(
                    db_path=face_db, enable_emotion=emotion_enabled)
                # Add face enrollment MQTT handler (subscribe after connect)
                self.topic_handler[FaceEnums.MQTT_ENROLL_TOPIC.value] = self._handle_face_command
                self.client.subscribe(FaceEnums.MQTT_ENROLL_TOPIC.value, qos=1)
                self.logger.info("Subscribed to face enrollment commands")
            except Exception:
                self.logger.warning("Face recognition unavailable — continuing without it")
                self.face_recognition = None
        else:
            self.logger.info("Face recognition disabled via config")
            self.face_recognition = None

        # Gesture recognition from hand keypoints (head camera only)
        gesture_enabled = configfile.get(
            FeatureToggles.CONFIG_HEAD.value,
            FeatureToggles.GESTURE_RECOGNITION.value,
            fallback="True").strip().lower() == "true"
        if gesture_enabled:
            from glados_modules.GestureRecognition import GestureClassifier
            self.gesture_classifier = GestureClassifier()
            self.logger.info("Gesture recognition enabled")
        else:
            self.gesture_classifier = None

        # Video recording
        self._record_enabled = configfile.get(
            FeatureToggles.CONFIG_HEAD.value,
            FeatureToggles.RECORD_VIDEO.value,
            fallback="False").strip().lower() == "true"
        self._recording_path = configfile.get(
            FeatureToggles.CONFIG_HEAD.value,
            FeatureToggles.RECORDING_PATH.value,
            fallback="./recordings")
        self._recording_timeout = float(configfile.get(
            FeatureToggles.CONFIG_HEAD.value,
            FeatureToggles.RECORDING_READY_TIMEOUT.value,
            fallback="120"))
        self._jpeg_quality = int(configfile.get(
            FeatureToggles.CONFIG_HEAD.value,
            FeatureToggles.RECORDING_JPEG_QUALITY.value,
            fallback="85"))
        self._recording_session: RecordingSession = None
        self._recording_gate: RecordingGate = None
        self._replay_session: str = None  # set via --replay CLI flag

        if self._record_enabled:
            cam_names = list(self.cam_configs.keys())
            self._recording_gate = RecordingGate(cam_names, self._recording_timeout)

        # Face recognition rate limiting — DeepFace is expensive, run at most once per second
        self._face_last_run: float = 0.0
        self._face_interval: float = 1.0
        self._face_cache: list = []

        # Pipeline tracing
        self.tracer = TraceLog()
        self._frame_times: Dict[str, float] = {}  # per-camera FPS tracking
        self._frame_counts: Dict[str, int] = {}

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
            y_class_data = json_loads(y_class.to_json())
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
        # Select frame source: replay from disk or live RTSP
        if self._replay_session:
            from glados_modules.RTSPClient import VideoFileSource
            image_get = VideoFileSource(self._replay_session, camera_key, realtime=False)
            self.logger.info(f"Replaying from {self._replay_session} for {camera_key}")
        else:
            image_get = RtspConsumer(location=camera_key,
                                     uri=self.cam_configs[camera_key][CameraEnum.MSG_RTSP_URI.value])
        skip_counter = 0
        consecutive_errors = 0
        MAX_BACKOFF = 30  # seconds
        while True:
            try:
                image_dict = image_get.get_frame()
                wall_time = time()

                # Signal recording gate that this camera is producing frames
                if self._recording_gate and not self._recording_session:
                    self._recording_gate.camera_ready(camera_key)
                    if self._recording_gate.should_start():
                        ready_cams = self._recording_gate.get_ready_cameras()
                        self._recording_session = RecordingSession(
                            base_path=self._recording_path,
                            cameras=ready_cams,
                            jpeg_quality=self._jpeg_quality,
                            config_snapshot={"cam_configs": {
                                c: {"resolution": self.cam_configs[c].get(
                                    CameraEnum.MSG_RESOLUTION.value)}
                                for c in ready_cams}})

                # Record raw frame before YOLO processing
                if self._recording_session and self._recording_session.is_active:
                    raw_frame = image_dict.get(CameraEnum.MSG_RAW_IMAGE.value)
                    if raw_frame is not None:
                        self._recording_session.record_frame(
                            camera_key, raw_frame, wall_time)

                # During movement, process every 3rd frame instead of skipping entirely.
                # This keeps the world-angle estimate warm on the tracking side.
                if camera_key == CameraEnum.CAMERA_HEAD.value:
                    if self.servos.check_movement() is True:
                        skip_counter += 1
                        if skip_counter % 3 != 0:
                            continue
                    else:
                        skip_counter = 0

                self.logger.debug(f"Processing image from {camera_key}")
                # Generate trace ID for this frame
                trace_id = TraceLog.new_id()
                ts_vision = time()
                # Process the image and track objects
                sight = self.__process_image(image_dict, d_model, p_model)
                # Stamp trace ID and vision timestamp on results
                sight[TraceEnums.TRACE_ID.value] = trace_id
                sight[TraceEnums.TS_VISION.value] = ts_vision
                results = CameraMessageBuilder.send_results(camera_key, sight)
                self.send_command(results, self.cmd_topic, qos=0)
                # Start trace record
                self.tracer.start_trace(trace_id, camera=camera_key, ts_vision=ts_vision)

                # Record tracking data alongside the frame
                if self._recording_session and self._recording_session.is_active:
                    self._recording_session.record_frame(
                        camera_key, None, wall_time, tracking_data=sight)

                consecutive_errors = 0  # reset on success

            except (KeyboardInterrupt, StopIteration):
                self.logger.info(f"Tracker for {camera_key} finished")
                break
            except Exception:
                consecutive_errors += 1
                backoff = min(MAX_BACKOFF, 0.5 * (2 ** min(consecutive_errors, 6)))
                self.logger.error(f"Error in tracker for {camera_key} "
                                  f"(attempt {consecutive_errors}, backoff {backoff:.1f}s):\n"
                                  f"{traceback.format_exc()}")
                sleep(backoff)

    def start_tracking_threads(self) -> None:
        """
        Start a separate tracking thread for each camera.
        """
        self.logger.debug(f"YOLOv8 model started with {self.model_config}")
        pose_enabled = self.configfile.get(FeatureToggles.CONFIG_HEAD.value,
                                            FeatureToggles.POSE_MODEL_ENABLED.value,
                                            fallback="True").strip().lower() == "true"
        for camera_key in self.cam_configs.keys():
            with safe_globals([DetectionModel, Sequential, Conv]):
                detection_model = YOLO(self.model_config)
            pose_model = None
            # Pose estimation only needed on head camera (gesture recognition, nose-point tracking).
            # Side cameras only use YOLO bounding boxes for world-angle estimation.
            if pose_enabled and camera_key == CameraEnum.CAMERA_HEAD.value:
                self.logger.debug("Creating Pose model for head camera")
                openpose_skeleton = False  # True for openpose-style, False for mmpose-style
                backend = 'onnxruntime'  # opencv, onnxruntime, openvino
                pose_model = Wholebody(to_openpose=openpose_skeleton, mode='balanced', backend=backend, device='cuda')
            elif not pose_enabled:
                self.logger.info(f"Pose model disabled via config for {camera_key}")
            thread = Thread(target=self.run_tracker_for_camera, args=(camera_key,
                                                                       detection_model, pose_model), daemon=True)
            thread.start()
            self.cam_configs[camera_key]["tracker_thread"] = thread

    def __process_image(self, image_dict: Dict[str, Any], d_model: YOLO,
                        p_model: Wholebody) -> Dict[str, Any]:
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
        results = d_model.track(source=image, device="cuda", tracker=self.tracker_yaml, verbose=False)
        self.logger.debug(f"Yolo processed image for camera {camera_location}")
        t_results = self.__translate_results(results)

        # Full annotation only for head camera — side cameras just need YOLO results,
        # not expensive per-detection drawing (saves ~10% CPU across 2 side camera streams).
        is_head = camera_location == CameraEnum.CAMERA_HEAD.value
        if is_head:
            annotator = Annotator(image)
            image_center = (int(width // 2), int(height // 2))
            cv2.line(image, (image_center[0] - 10, image_center[1]),
                     (image_center[0] + 10, image_center[1]), (0, 255, 0), 2)
            cv2.line(image, (image_center[0], image_center[1] - 10),
                     (image_center[0], image_center[1] + 10), (0, 255, 0), 2)

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
        else:
            a_image = image

        # Face recognition + emotion detection (head camera only, rate-limited)
        if camera_location == CameraEnum.CAMERA_HEAD.value and self.face_recognition:
            person_key = VisionResultsEnum.VISION_RESULTS_PERSON_KEY.value
            objects_key = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
            if person_key in t_results and objects_key in t_results[person_key]:
                now = time()
                if now - self._face_last_run >= self._face_interval:
                    self._face_last_run = now
                    self._face_cache = self.face_recognition.detect_and_analyze(image)
                faces = self._face_cache
                self.face_recognition.match_faces_to_persons(
                    faces, t_results[person_key][objects_key])
                # Annotate face bounding boxes and labels on RTSP stream
                face_key = VisionResultsEnum.VISION_RESULTS_FACE_KEY.value
                for person in t_results[person_key][objects_key]:
                    if face_key in person:
                        face_data = person[face_key]
                        fb = face_data.get(VisionResultsEnum.VISION_RESULTS_FACE_BOX_KEY.value, {})
                        fx1 = fb.get(VisionResultsEnum.BOX_X1.value, 0)
                        fy1 = fb.get(VisionResultsEnum.BOX_Y1.value, 0)
                        fx2 = fb.get(VisionResultsEnum.BOX_X2.value, 0)
                        fy2 = fb.get(VisionResultsEnum.BOX_Y2.value, 0)
                        fid = face_data.get(VisionResultsEnum.VISION_RESULTS_FACE_ID_KEY.value, "?")
                        emo = face_data.get(VisionResultsEnum.VISION_RESULTS_EMOTION_KEY.value, "?")
                        cv2.rectangle(a_image, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                        label = f"{fid} ({emo})"
                        cv2.putText(a_image, label, (fx1, fy1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # if a pose model exists, plot the pose on the camera
        if p_model is not None:
            # we have a model, must be head camera, run model and draw points
            key_points, scores = p_model(image)
            a_image = draw_skeleton(a_image, key_points, scores,
                                    kpt_thr=self.kpt_threshold)
            # assign results to bounding boxes
            t_results = self.assign_key_points_to_response(t_results, key_points, scores)

        # Gesture recognition from hand keypoints (head camera only)
        gesture_key = VisionResultsEnum.VISION_RESULTS_GESTURE_KEY.value
        pose_key = VisionResultsEnum.VISION_RESULTS_POSE_KEY.value
        if self.gesture_classifier and camera_location == CameraEnum.CAMERA_HEAD.value:
            person_key = VisionResultsEnum.VISION_RESULTS_PERSON_KEY.value
            objects_key = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
            for person in t_results.get(person_key, {}).get(objects_key, []):
                if pose_key in person:
                    gesture = self.gesture_classifier.classify(person[pose_key])
                    person[gesture_key] = gesture
                    # Annotate gesture on RTSP frame
                    for hand_key in [VisionResultsEnum.VISION_RESULTS_GESTURE_LEFT.value,
                                      VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value]:
                        g = gesture.get(hand_key, "none")
                        if g != "none":
                            prefix = "Left_Hand" if "left" in hand_key else "Right_Hand"
                            wrist = person[pose_key].get(f"{prefix}_Wrist", {})
                            wx = int(wrist.get("x", 0))
                            wy = int(wrist.get("y", 0))
                            color = (0, 0, 255) if g == "middle_finger" else (0, 200, 255)
                            cv2.putText(a_image, g, (wx, wy - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Tracking state overlay removed from video — now served via /api/tracking
        # on the web dashboard sidebar instead.

        self.logger.debug(f"Sending image to RTSP server factory: {image_dict[CameraEnum.MSG_LOCATION_KEY.value]}")
        self.rtsp.send_data(image_dict[CameraEnum.MSG_LOCATION_KEY.value], a_image)
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
                    VisionResultsEnum.KEYPOINT_X.value: float(x),
                    VisionResultsEnum.KEYPOINT_Y.value: float(y),
                    VisionResultsEnum.KEYPOINT_CONFIDENCE.value: float(score),
                    VisionResultsEnum.KEYPOINT_LOCATION.value:
                        VisionResultsEnum.VISION_POSE_KEY_POINTS_COCO_WHOLE_BODY.value[index]
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

        # Enum shortcuts
        person_key = VisionResultsEnum.VISION_RESULTS_PERSON_KEY.value
        objects_key = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
        box_key = VisionResultsEnum.VISION_RESULTS_BOX_KEY.value
        pose_key = VisionResultsEnum.VISION_RESULTS_POSE_KEY.value
        kp_x = VisionResultsEnum.KEYPOINT_X.value
        kp_y = VisionResultsEnum.KEYPOINT_Y.value
        kp_loc = VisionResultsEnum.KEYPOINT_LOCATION.value
        bx1 = VisionResultsEnum.BOX_X1.value
        by1 = VisionResultsEnum.BOX_Y1.value
        bx2 = VisionResultsEnum.BOX_X2.value
        by2 = VisionResultsEnum.BOX_Y2.value

        # Track which pose indices get matched to YOLO boxes
        matched_pose_indices = set()
        source_key = VisionResultsEnum.VISION_RESULTS_SOURCE_KEY.value

        # Iterate over the response dictionary to find matching objects
        count = 0
        for person_data in response.get(person_key, {}).get(objects_key, []):
            # Get the bounding box coordinates
            box = person_data.get(box_key, {})
            x1, y1 = box.get(bx1, 0), box.get(by1, 0)
            x2, y2 = box.get(bx2, 0), box.get(by2, 0)

            # Filter key points that fit inside the bounding box based on the given threshold
            filtered_key_points = []
            for pose_idx, key_points in enumerate(merged_key_points):
                total_points = len(key_points)
                if total_points == 0:
                    continue  # Avoid division by zero if no key points are present

                # Count the number of key points inside the bounding box
                points_in_box = sum(
                    1 for kp in key_points if x1 <= kp[kp_x] <= x2 and y1 <= kp[kp_y] <= y2
                )
                # If the percentage of points inside the box meets or exceeds the threshold, assign them.
                if (points_in_box / total_points) >= percentage_threshold:
                    filtered_key_points = key_points
                    matched_pose_indices.add(pose_idx)
                    break  # Use the first matching set of key points

            # Assign the filtered key points to the person data if any were found
            if filtered_key_points:
                pose_dict = {kp[kp_loc]: kp for kp in filtered_key_points}
                response[person_key][objects_key][count][pose_key] = pose_dict
            count += 1

        # Create synthetic person detections from unmatched pose results
        min_conf = VisionResultsEnum.POSE_PERSON_MIN_CONFIDENCE.value
        box_padding = VisionResultsEnum.POSE_PERSON_BOX_PADDING.value
        kp_min_score = VisionResultsEnum.POSE_KEYPOINT_MIN_SCORE.value
        kp_conf = VisionResultsEnum.KEYPOINT_CONFIDENCE.value
        source_pose = VisionResultsEnum.VISION_RESULTS_SOURCE_POSE.value
        name_key = VisionResultsEnum.YOLO_CLASS_NAME_KEY.value
        conf_key = VisionResultsEnum.VISION_RESULTS_CONFIDENCE_KEY.value

        for pose_idx, key_points in enumerate(merged_key_points):
            if pose_idx in matched_pose_indices:
                continue
            if not key_points:
                continue

            # Filter to keypoints with meaningful confidence
            valid_kps = [kp for kp in key_points if kp.get(kp_conf, 0) > kp_min_score]
            if not valid_kps:
                continue

            # Compute average confidence
            avg_conf = sum(kp[kp_conf] for kp in valid_kps) / len(valid_kps)
            if avg_conf < min_conf:
                self.logger.debug(
                    f"POSE_FUSION: skipped pose idx={pose_idx} conf={avg_conf:.2f} below threshold")
                continue

            # Compute bounding box from keypoint extremes with padding
            xs = [kp[kp_x] for kp in valid_kps]
            ys = [kp[kp_y] for kp in valid_kps]
            raw_x1, raw_y1 = min(xs), min(ys)
            raw_x2, raw_y2 = max(xs), max(ys)
            w = raw_x2 - raw_x1
            h = raw_y2 - raw_y1
            pad_x = w * box_padding
            pad_y = h * box_padding
            syn_x1 = max(0, raw_x1 - pad_x)
            syn_y1 = max(0, raw_y1 - pad_y)
            syn_x2 = raw_x2 + pad_x
            syn_y2 = raw_y2 + pad_y

            # Build synthetic person detection
            pose_dict = {kp[kp_loc]: kp for kp in key_points}
            synthetic = {
                name_key: person_key,
                conf_key: round(avg_conf, 3),
                box_key: {bx1: syn_x1, by1: syn_y1, bx2: syn_x2, by2: syn_y2},
                pose_key: pose_dict,
                source_key: source_pose,
            }

            # Ensure person key exists in response
            if person_key not in response:
                response[person_key] = {objects_key: [], "count": 0}
            response[person_key][objects_key].append(synthetic)
            response[person_key]["count"] = len(response[person_key][objects_key])

            self.logger.debug(
                f"POSE_FUSION: created synthetic person from pose idx={pose_idx} "
                f"conf={avg_conf:.2f} box=({syn_x1:.0f},{syn_y1:.0f},{syn_x2:.0f},{syn_y2:.0f})")

        return response

    def run(self) -> None:
        """
        Entry point for the MLDetect thread. Publishes a startup status message
        and then starts tracking threads for each camera.
        """
        status = CameraMessageBuilder.send_status(self.__name__, "Machine Vision Started")
        self.send_command(status, self.status_topic)
        # MQTT topics are subscribed in __init__ via topic_handler (before connect)

        # Start tracking for each camera in separate threads
        self.start_tracking_threads()
        self.logger.info("Started YOLO tracking threads for all cameras")
        # Keep this thread alive so HealthMonitor reports it as running.
        # The actual work happens in the tracker threads spawned above.
        while True:
            sleep(1)

    def _handle_recording_command(self, msg: Any) -> None:
        """Handle start/stop recording commands from MQTT.

        Args:
            msg: MQTT message with cmd field.
        """
        from json import loads
        try:
            j_msg = loads(msg.payload.decode())
        except Exception:
            return
        cmd = j_msg.get("cmd", "")
        if cmd == FeatureToggles.RECORDING_CMD_START.value:
            if self._recording_session and self._recording_session.is_active:
                self.logger.warning("Recording already active")
                return
            cam_names = list(self.cam_configs.keys())
            self._recording_gate = RecordingGate(cam_names, self._recording_timeout)
            self._record_enabled = True
            self.logger.info("Recording requested via MQTT — waiting for cameras")
        elif cmd == FeatureToggles.RECORDING_CMD_STOP.value:
            if self._recording_session and self._recording_session.is_active:
                self._recording_session.stop()
                self._recording_session = None
                self.logger.info("Recording stopped via MQTT")
            self._record_enabled = False

    def _handle_face_command(self, msg: Any) -> None:
        """Handle face enrollment/forget commands from MQTT.

        Args:
            msg: MQTT message with cmd and name fields.
        """
        from json import loads
        try:
            j_msg = loads(msg.payload.decode())
        except Exception:
            return
        cmd = j_msg.get(FaceEnums.MSG_COMMAND_KEY.value, "")
        name = j_msg.get(FaceEnums.MSG_NAME_KEY.value, "")
        if not name:
            return

        if cmd == FaceEnums.COMMAND_ENROLL.value:
            self.logger.info(f"Face enrollment requested for '{name}'")
            self.face_recognition.start_enrollment(name)
            status = {FaceEnums.MSG_COMMAND_KEY.value: FaceEnums.COMMAND_STATUS_ENROLLED.value,
                      FaceEnums.MSG_NAME_KEY.value: name,
                      "status": "pending"}
            self.send_command(status, FaceEnums.MQTT_STATUS_TOPIC.value)
        elif cmd == FaceEnums.COMMAND_FORGET.value:
            success = self.face_recognition.forget(name)
            status_msg = "forgotten" if success else "not_found"
            status = {FaceEnums.MSG_COMMAND_KEY.value: status_msg,
                      FaceEnums.MSG_NAME_KEY.value: name}
            self.send_command(status, FaceEnums.MQTT_STATUS_TOPIC.value)
