import time
from typing import Dict, Callable, Tuple, NamedTuple, Any
from json import loads
from collections import namedtuple
from math import sqrt, radians, tan, atan, degrees

# 3rd party imports
from paho.mqtt.client import MQTTMessage
import numpy as np

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.MqttConnector import MQTTClient, ServoMessageBuilder
from glados_modules.MqttConsumerModules import ServoLocation, VisionTracker
from glados_modules.Kalman2D import KalmanFilter2D
from glados_modules.GladosEnums import (CameraEnum, ServoEnum, SystemEnums,
                                        TrackingEnums, VisionResultsEnum, LoggingEnums)


class MotionTrack(MQTTClient):
    """
    A class for motion tracking on a target. Inherits from MQTTClient to utilize MQTT functionality
    for receiving commands and sending servo movement instructions.
    """
    broker_tuple = MQTTClient.broker_tuple
    camera_tuple = namedtuple("cam_resolution", ['x', 'y'])

    def __init__(
        self,
        broker: NamedTuple,
        camera_resolution: NamedTuple,
        target: str = "person",
        # set pose_target to nose
        pose_target: str = VisionResultsEnum.VISION_POSE_KEY_POINTS_COCO_WHOLE_BODY.value[0],
        confidence: float = 0.65,
        move_fudge_factor: int = 3
    ) -> None:
        """
        Initialize the MotionTrack class with the given parameters.

        :param broker: NamedTuple containing the MQTT broker details (ip, port).
        :param camera_resolution: NamedTuple containing camera resolution (x, y).
        :param target: Target label/string to track (default: "person").
        :param confidence: Minimum confidence level to consider a valid detection (default: 0.65).
        :param move_fudge_factor: Dead zone factor for servo movement filtering (default: 3).
        """
        self.__name__ = self.__class__.__name__
        self.location = self.__name__
        self.logger = setup_logger(self.__name__, console_logging=LoggingEnums.LOG_LEVEL_DEBUG.value)

        # MQTT topics and triggers
        self.cmd_topic: str = TrackingEnums.MQTT_COMMAND_TOPIC.value
        self.cmd_trigger: str = TrackingEnums.MSG_COMMAND_KEY.value
        self.intensity_topic: str = SystemEnums.MQTT_INTENSITY_TOPIC.value
        self.count = VisionResultsEnum.VISION_RESULTS_COUNT_KEY.value
        self.intensity: Tuple[float, float] = (.1, .1)

        # Dictionary to store history of bounding boxes for smoothing
        self._bbox_history: dict = {}

        # Dictionary that will map topic names to handler functions
        self.topic_handler: Dict[str, Callable] = {
            self.cmd_topic: self.handle_cmd,
            self.intensity_topic: self.handle_intensity
        }

        # Head camera resolution (width -> cam_x, height -> cam_y)
        # This works for now but future modifications may be needed for multiple cameras.
        self.cam_x = int(camera_resolution.x)
        self.cam_y = int(camera_resolution.y)

        # Camera references
        self.main_camera = CameraEnum.CAMERA_HEAD.value
        self.left_camera = CameraEnum.CAMERA_LEFT.value
        self.right_camera = CameraEnum.CAMERA_RIGHT.value

        # Dead zone factor for servo movement
        self.dead_zone_factor = move_fudge_factor

        # Namedtuple for servo objects
        servo = namedtuple("servo", ["name", "move"])

        # Servo names
        self.head_LR = servo(ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value, ServoMessageBuilder.head_left_right)
        self.head_UD = servo(ServoEnum.LOCATION_HEAD_UP_DOWN.value, ServoMessageBuilder.head_up_down)
        self.body_LR = servo(ServoEnum.LOCATION_BODY_LEFT_RIGHT.value, ServoMessageBuilder.body_left_right)
        self.body_UD = servo(ServoEnum.LOCATION_BODY_UP_DOWN.value, ServoMessageBuilder.body_up_down)

        # Tracking-related attributes
        self.target = target
        self.pose_target = pose_target
        self.confidence = confidence
        self.servos: Dict[str, object] = dict()
        self.dms: int = 3  # default movement speed
        self.peripheral_hunt = True  # if movement on side cameras is allowed
        self.head_tracking = False  # if the head is currently tracking something
        self.head_UD_last_angle = None  # last angle we tried to set the head to
        self.hysteresis_threshold: int = 5  # hysteresis threshold

        super().__init__(ip=broker.ip, port=broker.port)
        self.side_camera_count: int = 0

        # Create Servo Location Tracker to get and send servo angles
        self.servo_status = ServoLocation(broker)

        # Vision seen Tracker
        self.objects = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
        self.vision_tracker = VisionTracker(broker=broker, target=self.target,
                                            confidence=self.confidence, tracker_callback=self.track_loop)

        # Hanging tracker: indicates if the robot is in "hang around" mode
        self.hanging = False

        # Store the last move time to enforce rate limiting
        self._last_move_time = time.time()

        # create kalman filter
        self.kf = KalmanFilter2D(dt=0.1)
        self.kf_initialized = False
        # Lock for thread-safety (inherited from MQTTClient)
        # NOTE: Self._lock is defined in MQTTClient, ensuring concurrency control
        # across method calls, especially servo movements.

    def predict_target_bbox(self, bbox: dict) -> dict:
        """
        Use the Kalman filter to predict the new position of the target.

        :param bbox: Detected bounding box with keys 'x1', 'y1', 'x2', 'y2' and confidence.
        :return: A new bounding box dictionary with the predicted center.
        """
        # Compute the detected center
        center_x = (bbox['x1'] + bbox['x2']) / 2
        center_y = (bbox['y1'] + bbox['y2']) / 2
        measurement = np.array([[center_x], [center_y]])

        # If this is the first measurement, initialize the filter’s state.
        if not self.kf_initialized:
            self.kf.x[0, 0] = center_x
            self.kf.x[1, 0] = center_y
            self.kf_initialized = True

        # Update the filter with the new measurement
        self.kf.update(measurement)

        # Predict the next state
        predicted_state = self.kf.predict()
        predicted_center_x = predicted_state[0, 0]
        predicted_center_y = predicted_state[1, 0]

        # Optionally, you can use self.kf.x[2] and self.kf.x[3] for velocity info
        self.logger.debug(f"Predicted position: ({predicted_center_x:.2f}, {predicted_center_y:.2f}), "
                          f"Velocity: ({self.kf.x[2, 0]:.2f}, {self.kf.x[3, 0]:.2f})")

        # Preserve the detected bounding box size
        width = bbox['x2'] - bbox['x1']
        height = bbox['y2'] - bbox['y1']

        # Reconstruct the bounding box using the predicted center
        new_bbox = {
            'x1': predicted_center_x - width / 2,
            'y1': predicted_center_y - height / 2,
            'x2': predicted_center_x + width / 2,
            'y2': predicted_center_y + height / 2,
            TrackingEnums.KEY_CONFIDENCE.value: bbox[TrackingEnums.KEY_CONFIDENCE.value]
        }

        return new_bbox

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """
        Handle incoming command messages from MQTT. If a START command is received,
        trigger the tracking loop for the specified camera.

        :param msg: MQTTMessage containing the payload with the command.
        """
        j_msg = loads(msg.payload.decode())
        print(f"*** TRACKING FIRED", {self.cmd_trigger}, {TrackingEnums.MSG_COMMAND_START.value})
        print(j_msg, j_msg.get(self.cmd_trigger))
        if j_msg.get(self.cmd_trigger, "") == TrackingEnums.MSG_COMMAND_START.value:
            self.logger.debug(f"Tracking Command Received, {msg.topic}, {j_msg}")
            trigger_camera = j_msg.get(TrackingEnums.MSG_CAMERA_KEY.value, "")
            self.logger.debug(f"Running Track loop for {trigger_camera}")
            self.track_loop(trigger_camera)
            self.logger.debug(f"Tracking complete for {trigger_camera}")

    def __dead_zone_check(
        self,
        servo: NamedTuple,
        new_angle: int,
        degree_diff: int = 2,
        confidence: float = 0.8,
        depth: float = 1.0
    ) -> int:
        """
        Adaptive dead zone check for filtering out minor or jittery servo movements.
        This method calculates a threshold (dynamic_diff) based on the confidence
        and depth of the detected object, and checks if the absolute difference
        between the current servo angle and the new angle exceeds that threshold.

        :param servo: Servo object that contains current angle, location, etc.
        :param new_angle: The proposed new angle for the servo movement.
        :param degree_diff: Base threshold for ignoring small movements.
        :param confidence: The confidence level of the detection (0 - 1).
        :param depth: Distance or depth measure to adjust the threshold.
        :return: 1 (True) if movement should happen, 0 (False) otherwise.
        """
        # Use a higher dead zone for vertical movement (Y-axis).
        if servo.location == ServoEnum.LOCATION_HEAD_UP_DOWN.value:
            degree_diff = 5  # Increase to stabilize head nodding

        # Adjust dead zone dynamically based on confidence
        dynamic_diff = (degree_diff * (1 - confidence) + 1)

        # Increase tolerance for medium confidence (0.6-0.7)
        if 0.6 <= confidence <= 0.7:
            dynamic_diff += 1.5  # More aggressive filtering

        # Modify by distance factor
        distance_factor = max(0.5, min(2.0, depth))
        dynamic_diff *= distance_factor

        # Apply a hard minimum threshold
        dynamic_diff = max(dynamic_diff, 3)  # Ensure no movement for <3° differences

        current_angle = servo.current
        move = abs(new_angle - current_angle) > dynamic_diff
        self.logger.debug(
            f"{servo.location}: Angle {new_angle}, Current {current_angle}, "
            f"Diff {abs(new_angle - current_angle)}, Threshold {dynamic_diff}, Move: {move}"
        )
        return move

    def track_loop(self, camera: str) -> None:
        """
        Main tracking loop that examines the vision data from a specified camera,
        finds the target bounding box, and moves servos accordingly.

        :param camera: String representing the camera that triggered this tracking loop.
        """
        # Don't double call if head_tracking is True; skip this detection to prevent overlap.
        if self.__check_tracking() is True:
            self.logger.debug("Getting Vision Map")
            vision_map = self.vision_tracker.get_vision_map()
            self.logger.debug("Looping though vision map")
            self.servo_status.update_servo_status()
            if camera in vision_map.keys():
                if vision_map[camera][self.target].get(self.count, 0) != 0:
                    # If we have detections for the target, find the bounding box with the highest confidence
                    best_target = self.__find_target(vision_map[camera][self.target][self.objects])
                    target_ts = vision_map[camera].get(VisionResultsEnum.VISION_RESULTS_TS_KEY.value, None)
                    # create just a bounding box and confidence object
                    target_bounding = best_target[TrackingEnums.KEY_BOX.value]
                    target_bounding[
                        TrackingEnums.KEY_CONFIDENCE.value] = best_target[
                        VisionResultsEnum.VISION_RESULTS_CONFIDENCE_KEY.value]
                    # If the camera is the head camera, move all servos
                    if camera == TrackingEnums.BODY_HEAD_CAMERA.value:
                        self.logger.debug(
                            f"Ready to move all servos for target {self.target} "
                            f"message times stamp {target_ts} for {camera}"
                        )
                        # track pose data if we have it
                        current_pose_target = None
                        if TrackingEnums.KEY_POSE.value in best_target.keys():
                            pose_data = best_target[TrackingEnums.KEY_POSE.value]
                            if self.pose_target in pose_data.keys():
                                current_pose_target = pose_data[self.pose_target]
                        # Attempt to smooth the bounding box for visual noise
                        target_bounding = self.smooth_bounding_box(target_bounding)
                        # Apply predictive Kalman filter
                        #target_bounding = self.predict_target_bbox(target_bounding)
                        # maybe also pass the point for the center point of a pose target or pose targets?
                        # Move head and body servos based on the bounding box
                        if current_pose_target is not None:
                            self.move_all_servos(current_pose_target, camera, pose=True)
                        else:
                            self.move_all_servos(target_bounding, camera)

                        with self._lock:
                            self.side_camera_count = 0
                            self.hanging = False

                        self.logger.debug(
                            f"Movement complete for target {self.target} and message times stamp {target_ts}"
                        )

                    # If the camera is one of the side cameras, rotate the body towards the target
                    elif camera in (TrackingEnums.BODY_LEFT_CAMERA.value, TrackingEnums.BODY_RIGHT_CAMERA.value):
                        if self.side_camera_count <= 5:
                            self.logger.debug(f"Rotating Body to face target {self.target}")
                            self.rotate_body(target=target_bounding, camera=camera, flip=False)
                            with self._lock:
                                self.side_camera_count += 1

                            # Hold for a while to let the main camera capture the target
                            time.sleep(3)
                        else:
                            # Move to just hang around if we've tried 5 times already
                            if self.hanging is False:
                                self.hang_around()
                                self.logger.debug("Couldn't get target on head camera in 5 tries skipping for now")
                            else:
                                self.logger.debug("Already hanging out")

            with self._lock:
                self.head_tracking = False

    def __check_tracking(self) -> bool:
        """
        Check if the system is currently tracking. If not, set head_tracking to True
        to allow movement. Otherwise, return False to skip new tracking requests.

        :return: True if tracking is now allowed, False if we are already tracking.
        """
        with self._lock:
            if self.head_tracking is False:
                self.head_tracking = True
                self.logger.debug(f"Tracking is allowed, Moving To track {self.target}")
                rtn = True
            else:
                self.logger.debug("Tracking requested but already currently moving to track target")
                rtn = False
        return rtn

    def rotate_body(
        self,
        target: dict,
        camera: str,
        flip: bool = False,
        return_message: bool = False,
        pose: bool = False
    ) -> None | tuple:
        """
        Rotate the body servo to face the detected target, primarily used by side cameras.
        Optionally return the MQTT message to be sent instead of sending it directly.

        :param target: Dictionary with the bounding box info and confidence for the detected object.
        :param camera: String indicating which camera sees the target ("LEFT" or "RIGHT").
        :param flip: Boolean indicating if the angle should be mirrored (default: False).
        :param return_message: If True, returns a tuple of (movement_dict, mqtt_message_list) instead of sending.
        :param pose: Boolean indication if we are working with pose point data
        :return: None or tuple of (dict, list) depending on return_message.
        """
        self.logger.debug("Moving servos getting angle map")
        self.servos = self.servo_status.get_angle_map()
        self.logger.debug("Calculating movement for servos")

        mv_list = list()
        body_movement = dict()

        if target != {}:
            body_lr = self.__calc_servo(self.servos[self.body_LR.name], target, camera=camera, point=pose)
            # Check if movement is needed based on dead zone
            if self.__dead_zone_check(
                self.servos[self.body_LR.name],
                body_lr,
                degree_diff=self.dead_zone_factor,
                confidence=target[TrackingEnums.KEY_CONFIDENCE.value]
            ):
                if flip is True:
                    body_lr = MotionTrack.mirror_calc(body_lr)
                mv_list.append(self.body_LR.move(body_lr))
                if mv_list:
                    self.logger.debug("Sending Move commands for Head and Neck")
                    body_movement = {self.body_LR.name: body_lr}
                    if return_message is False:
                        # Send the MQTT command immediately
                        self.servo_status.send_command(mv_list, ServoEnum.MQTT_COMMAND_TOPIC.value)
                        #self.__block_for_update(body_movement)

        if return_message is True:
            return body_movement, mv_list

    def hang_around(self) -> None:
        """
        Rotate to a neutral center point and remain in a 'hang around' position,
        typically used when the system no longer detects or is not actively tracking a target.
        """
        with self._lock:
            self.hanging = True

        self.logger.debug("Deciding to hang out")
        msg_list = [ServoMessageBuilder.body_left_right(angle=90, speed=1)]
        msg_list.extend([ServoMessageBuilder.head_left_right(angle=92, speed=1)])
        msg_list.extend([ServoMessageBuilder.head_up_down(angle=125, speed=1)])
        msg_list.append(ServoMessageBuilder.body_up_down(angle=180, speed=1))

        # Send the hang-around servo commands
        self.servo_status.send_command(msg_list, ServoEnum.MQTT_COMMAND_TOPIC.value)

    def move_all_servos(self, target: dict, camera: str, pose: bool=False) -> None:
        """
        Move the head and body servos based on the location of the target in the frame.

        :param target: Dictionary containing the target bounding box data.
        :param camera: String representing which camera sees the target (e.g., "HEAD").
        """
        self.logger.debug("Moving servos getting angle map")
        self.servos = self.servo_status.get_angle_map()
        mv_list = list()
        self.logger.debug(f"Calculating movement for servos, moving based on pose detection is {pose}")

        if target != {}:
            # Enforce rate limit checks
            if hasattr(self, '_last_move_time') and not self.rate_limited_update(self._last_move_time, interval=.4):
                self.logger.debug("Rate limited, moved too soon")
                return

            # Update the last move time
            self._last_move_time = time.time()

            # Calculate angles for head left-right (X-axis) and up-down (Y-axis)
            head_lr = self.__calc_servo(self.servos[self.head_LR.name], target, camera=camera, point=pose)
            head_ud = self.__calc_servo(self.servos[self.head_UD.name], target, camera=camera, point=pose)
            print(f"head lr down angle {head_lr}")
            print(f"head up down angle {head_ud}")

            # Initialize last commanded head up/down angle if necessary
            if self.head_UD_last_angle is None:
                self.head_UD_last_angle = self.servos[self.head_UD.name].current

            # Apply hysteresis for head up/down:
            if abs(head_ud - self.head_UD_last_angle) > self.hysteresis_threshold:
                # Only send a move if the new angle is also outside the dead zone
                if self.__dead_zone_check(
                        self.servos[self.head_UD.name],
                        head_ud,
                        degree_diff=self.dead_zone_factor,
                        confidence=target[TrackingEnums.KEY_CONFIDENCE.value]
                ):
                    mv_list.append(self.head_UD.move(head_ud))
                    self.logger.debug(f"Head up/down move commanded: from {self.head_UD_last_angle} to {head_ud}")
                    self.head_UD_last_angle = head_ud
                else:
                    self.logger.debug("Head up/down within dead zone after hysteresis check; no move.")
            else:
                self.logger.debug("Head up/down change within hysteresis band; no move commanded.")

            # Check if the LR movement is outside the dead zone
            if self.__dead_zone_check(self.servos[self.head_LR.name], head_lr, self.dead_zone_factor):
                mv_list.append(self.head_LR.move(head_lr))
            else:
                head_lr = self.servos[self.head_LR.name].current

            # Send servo commands if we have any movement
            if mv_list:
                self.logger.debug("Sending Move commands for Head and Neck")
                self.servo_status.send_command(mv_list, ServoEnum.MQTT_COMMAND_TOPIC.value)
                head_movement = {self.head_LR.name: head_lr, self.head_UD.name: head_ud}
                #self.__block_for_update(head_movement)

            # Check if head movement has reached physical limits
            if self.__reached_limit(self.servos[self.head_LR.name]):
                self.logger.debug("Head reached left/right limit, rotating body to extend range")
                self.__rotate_body_to_extend_range()

            if self.__reached_limit(self.servos[self.head_UD.name]):
                self.logger.debug("Head reached up/down limit, bending body to extend range")
                self.__bend_body_to_extend_range()

            # Rotate body to face the target, then attempt to level the head and body
            body_movement, mv_list = self.rotate_body(target, camera, return_message=True, pose=pose)

            # Level the head left-right to its middle
            middle = self.servos[self.head_LR.name].middle
            if self.__dead_zone_check(
                self.servos[self.head_LR.name],
                middle,
                degree_diff=self.dead_zone_factor,
                confidence=target[TrackingEnums.KEY_CONFIDENCE.value]
            ):
                mv_list.append(self.head_LR.move(middle))

            # Send any new servo commands collected
            self.servo_status.send_command(mv_list, ServoEnum.MQTT_COMMAND_TOPIC.value)

            # Level the body up-down with the head
            servo_3, servo_4 = self.__level_servos(self.head_UD, self.body_UD)
            self.logger.debug("Leveling out body")

            body_level = {self.head_UD.name: servo_3, self.body_UD.name: servo_4}
            body_level.update(body_movement)

            #self.__block_for_update(body_level)
            self.logger.debug("Leveling out body complete")

    def __block_for_update(self, target_positions: Dict[str, int]) -> None:
        """
        Blocks execution until all specified servos reach their target positions.
        This polls for servo updates at intervals, and re-requests an update if it
        takes too long.

        :param target_positions: Dictionary of servo names to the angles we expect them to reach.
        """
        count = 0
        m_count = 0
        no_move_servo = False
        self.logger.debug(f"Waiting for updates on {len(target_positions.keys())}")
        while True:
            self.servos = self.servo_status.get_angle_map()
            all_reached = True
            for name, target in target_positions.items():
                if self.servos[name].moving is False:
                    m_count += 1
                    if m_count >= 2:
                        # checked servo twice and its movement is complete, angle not met for some reason
                        no_move_servo = True
                        break
                else:
                    m_count = 0
                if self.servos[name].current != target:
                    all_reached = False
                    self.logger.debug(f"{name} servo is currently blocking attempting to get to {target}")
                    break
                else:
                    self.logger.debug(f"{name} servo has updated and reached {target}")

            if all_reached or no_move_servo is True:
                break

            time.sleep(0.2)
            count += 1

            if count >= 15:
                # If servos are taking too long to update, request a status update again
                count = 0
                self.servo_status.update_servo_status()

        self.logger.debug("Blocking Updates Complete")

    def __reached_limit(self, servo: NamedTuple) -> bool:
        """
        Check if the servo has reached one of its movement limits (min or max angle).

        :param servo: Servo object that contains current angle, min, max, etc.
        :return: True if the servo is at its min or max position, False otherwise.
        """
        return servo.current == servo.min or servo.current == servo.max

    def __rotate_body_to_extend_range(self) -> None:
        """
        Rotate the body left-right servo to compensate for the head servo reaching its horizontal limit.
        """
        self.logger.debug("Rotating body to extend range of neck")
        diff = self.servos[self.head_LR.name].current - self.servos[self.head_LR.name].middle
        new_body_angle = self.servos[self.body_LR.name].current + diff

        # Clamp the new angle within the body's servo range
        new_body_angle = max(
            min(new_body_angle, self.servos[self.body_LR.name].max),
            self.servos[self.body_LR.name].min
        )

        # Send movement command
        self.servo_status.send_command(
            [self.body_LR.move(new_body_angle)],
            ServoEnum.MQTT_COMMAND_TOPIC.value
        )
        # Block until movement completes
        #self.__block_for_update({self.body_LR.name: new_body_angle})

    def __bend_body_to_extend_range(self) -> None:
        """
        Bend the body up-down servo to compensate for the head servo reaching its vertical limit.
        """
        self.logger.debug("Bending body to extend range of head")
        diff = self.servos[self.head_UD.name].current - self.servos[self.head_UD.name].middle
        new_body_angle = self.servos[self.body_UD.name].current + diff

        # Clamp the new angle within the body's servo range
        new_body_angle = max(
            min(new_body_angle, self.servos[self.body_UD.name].max),
            self.servos[self.body_UD.name].min
        )

        # Send movement command
        self.servo_status.send_command(
            [self.body_UD.move(new_body_angle)],
            ServoEnum.MQTT_COMMAND_TOPIC.value
        )
        # Block until movement completes
        #self.__block_for_update({self.body_UD.name: new_body_angle})

    def __find_target(self, seen_data: dict) -> dict:
        """
        From the list/dict of detection data, return the bounding box of the target
        with the highest confidence.

        :param seen_data: List or iterable containing detection data. Each element
                          includes a confidence value and bounding box info.
        :return: A dictionary with the bounding box coordinates of the highest confidence target.
        """
        confidence = VisionResultsEnum.VISION_RESULTS_CONFIDENCE_KEY.value
        rtn = {}
        highest_confidence = 0
        for p in seen_data:
            if p[confidence] > highest_confidence:
                highest_confidence = p[confidence]
                rtn = p
        self.logger.debug(f"Confidence box found {rtn} with confidence score of {highest_confidence}")
        return rtn

    @staticmethod
    def fisheye_correction(offset_proportion: float, fov: float) -> float:
        """
        Apply a fisheye lens correction to the offset proportion based on the Field of View (FOV).

        :param offset_proportion: Normalized offset from the image center, ranging -1 to 1.
        :param fov: The Field of View of the camera in degrees.
        :return: The corrected offset proportion after applying radial distortion compensation.
        """
        if fov >= 160:
            # Apply a non-linear correction based on radial distortion
            k1 = 0.2  # Example distortion coefficient for fisheye
            radial_distance = sqrt(offset_proportion ** 2)
            corrected_proportion = offset_proportion * (1 + k1 * (radial_distance ** 2))
            return corrected_proportion
        return offset_proportion

    def rate_limited_update(self, last_update_time: float, interval: float = 0.2) -> bool:
        """
        Check if enough time has passed since the last update to avoid excessive servo commands.

        :param last_update_time: Timestamp of the last movement or update.
        :param interval: Minimum interval required between updates (in seconds).
        :return: True if it's okay to proceed with an update, False otherwise.
        """
        current_time = time.time()
        if current_time - last_update_time < interval:
            self.logger.debug("Rate limiting: Skipping update.")
            return False
        return True

    def __calc_servo(
            self,
            servo: Any,
            bbox: Dict[str, float],
            camera: str,
            point: bool = False
    ) -> int:
        """
        Calculate the new servo angle using an arctan-based mapping from the
        target's offset from the image center to an angular correction.

        This method computes the effective focal length based on the camera's FOV,
        then calculates the angle as:

            angle_offset = arctan(offset_from_center / focal_length)

        The new servo angle is then given by:

            new_servo_angle = current + (direction_factor * angle_offset_degrees)
                              + mounting_angle

        where the mounting angle and direction factor adjust for different camera
        and servo configurations.

        :param servo: The servo object (with attributes such as current, min, max,
                      axis, and location).
        :param bbox: Dictionary containing bounding box coordinates or a single point.
                     For a bounding box, keys 'x1', 'x2' or 'y1', 'y2' are expected;
                     if point is True, then keys 'x' or 'y' are used.
        :param camera: Identifier for the camera (e.g. "HEAD", "LEFT", "RIGHT").
        :param point: If True, the bbox contains a single point rather than a full box.
        :return: The new servo angle as an integer, clamped within [servo.min, servo.max].
        """
        # Determine the size of the axis (in pixels)
        if servo.axis == ServoEnum.X_AXIS.value:
            axis_size: float = float(self.cam_x)
        else:
            axis_size: float = float(self.cam_y)

        # Calculate the target's center on this axis.
        if not point:
            if servo.axis == ServoEnum.X_AXIS.value:
                center_of_bbox: float = (bbox['x1'] + bbox['x2']) / 2
            else:
                center_of_bbox: float = (bbox['y1'] + bbox['y2']) / 2
        else:
            center_of_bbox = bbox['x'] if servo.axis == ServoEnum.X_AXIS.value else bbox['y']

        # Compute the offset in pixels from the center of the image.
        # A positive offset means the target is left of center (assuming a reversed convention).
        offset_from_center: float = (axis_size / 2) - center_of_bbox

        # Default field of view (in degrees) and mounting angle.
        fov: float = 54.0
        mounting_angle: float = 0.0
        current: float = servo.current

        # Adjust the field of view and mounting angle based on the camera.
        if camera == CameraEnum.CAMERA_HEAD.value:
            if servo.axis == ServoEnum.X_AXIS.value:
                fov = CameraEnum.CAMERA_HEAD_FOV_X.value
            else:
                fov = CameraEnum.CAMERA_HEAD_FOV_Y.value
        elif camera == CameraEnum.CAMERA_RIGHT.value:
            fov = CameraEnum.CAMERA_RIGHT_FOV.value
            if servo.axis == ServoEnum.X_AXIS.value and servo.location == ServoEnum.LOCATION_BODY_LEFT_RIGHT.value:
                mounting_angle = 55.0
                current = 90.0
        elif camera == CameraEnum.CAMERA_LEFT.value:
            fov = CameraEnum.CAMERA_LEFT_FOV.value
            if servo.axis == ServoEnum.X_AXIS.value and servo.location == ServoEnum.LOCATION_BODY_LEFT_RIGHT.value:
                mounting_angle = -55.0
                current = 90.0

        # Compute the focal length in pixels from the FOV.
        # f = (axis_size/2) / tan(FOV/2), with FOV converted to radians.
        fov_rad: float = radians(fov)
        focal_length: float = (axis_size / 2) / tan(fov_rad / 2)

        # Compute the angular offset in radians using arctan.
        angle_offset_rad: float = atan(offset_from_center / focal_length)
        angle_offset_deg: float = degrees(angle_offset_rad)

        # Determine the direction factor based on servo location.
        # For head servos, we use 1; for others (e.g. body servos) we invert the response.
        direction_factor: int = 1 if servo.location in (
            ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value,
            ServoEnum.LOCATION_HEAD_UP_DOWN.value
        ) else -1

        # Compute the new servo angle.
        new_servo_angle: float = current + direction_factor * angle_offset_deg + mounting_angle

        # Clamp the new angle within the servo's allowed range.
        new_servo_angle = max(min(new_servo_angle, servo.max), servo.min)

        return round(new_servo_angle)

    def __level_servos(self, servo1: object, servo2: object) -> tuple:
        """
        Attempt to bring servo1 to its midpoint by moving servo2 in a mirrored fashion.
        This keeps the head and body in sync.

        :param servo1: The 'head' servo (or main servo).
        :param servo2: The 'body' servo (or secondary servo).
        :return: A tuple of (servo1_target_angle, servo2_target_angle).
        """
        self.logger.debug(f"Leveling Servos {self.servos[servo1.name].location} & {self.servos[servo2.name].location}")

        # Ensure servos are on the same axis
        if self.servos[servo1.name].axis != self.servos[servo2.name].axis:
            msg = "Servos are not on same axis"
            self.logger.error(msg)
            raise Exception(msg)

        current = self.servos[servo1.name].current

        # If servo1 is one of the head servos, then only adjust the body angle
        # if the head's angle is between 64° and 126°.
        if servo1.name in (
                ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value,
                ServoEnum.LOCATION_HEAD_UP_DOWN.value
        ):
            head_angle = current  # Use the current head servo angle.
            if 64 <= head_angle <= 126:
                # Define a helper function for piecewise linear interpolation.
                def calc_body_angle_from_head(h_angle):
                    # Data points in the form (head_angle, body_angle)
                    data = [
                        (64, 30),
                        (66, 40),
                        (68, 50),
                        (70, 60),
                        (72, 70),
                        (79, 80),
                        (83, 90),
                        (92, 100),
                        (97, 110),
                        (104, 120),
                        (114, 130),
                        (121, 140),
                        (126, 150)
                    ]
                    # If the head angle is below or above the data range, clamp it.
                    if h_angle <= data[0][0]:
                        return data[0][1]
                    if h_angle >= data[-1][0]:
                        return data[-1][1]
                    # Otherwise, find the correct interval and interpolate.
                    for i in range(len(data) - 1):
                        H1, B1 = data[i]
                        H2, B2 = data[i + 1]
                        if H1 <= h_angle <= H2:
                            ratio = (h_angle - H1) / (H2 - H1)
                            return round(B1 + ratio * (B2 - B1))
                    # Fallback (should not happen)
                    return self.servos[servo2.name].current

                # Calculate the new body angle using the head angle.
                angle = calc_body_angle_from_head(head_angle)
            else:
                self.logger.debug("Head angle is out of interpolation range (64-126). No angle changes made.")
                # If head angle is out of range, return the current positions (no changes).
                return self.servos[servo1.name].current, self.servos[servo2.name].current
        else:
            # For non-head servos, just use the current value.
            angle = current

        self.logger.debug(f"Servo {servo2.name} is at {self.servos[servo2.name].current} before leveling")

        # Clamp the computed angle for servo2 to its allowed range.
        angle = max(min(angle, self.servos[servo2.name].max), self.servos[servo2.name].min)

        # Decide on final movement based on a dead-zone check.
        if abs(self.servos[servo1.name].middle - angle) > self.dead_zone_factor:
            servo1_move = self.servos[servo1.name].middle
            servo2_move = angle
        else:
            servo1_move = self.servos[servo1.name].current
            servo2_move = self.servos[servo2.name].current

        # Build the movement commands.
        mv_list = [
            servo2.move(angle),
            servo1.move(self.servos[servo1.name].middle)
        ]

        # Send the commands.
        self.servo_status.send_command(mv_list, ServoEnum.MQTT_COMMAND_TOPIC.value)

        return servo1_move, servo2_move

    def smooth_bounding_box(
        self,
        bbox: dict,
        alpha_x: float = 0.8,
        alpha_y: float = 0.9
    ) -> dict:
        """
        Smooth a single bounding box using an exponential moving average,
        applying separate smoothing factors for X and Y axes.

        :param bbox: Dictionary with bounding box coordinates {'x1', 'y1', 'x2', 'y2'}.
        :param alpha_x: Smoothing factor for horizontal (X) movement (range 0-1).
        :param alpha_y: Smoothing factor for vertical (Y) movement (range 0-1).
        :return: Dictionary with smoothed bounding box coordinates.
        """
        # Initialize history on the first call if needed
        if not hasattr(self, '_bbox_history'):
            self._bbox_history = bbox.copy()

        # Smooth X-axis movements (left/right)
        for key in ['x1', 'x2']:
            prev_value = self._bbox_history.get(key, bbox[key])
            bbox[key] = alpha_x * prev_value + (1 - alpha_x) * bbox[key]

        # Smooth Y-axis movements (up/down)
        for key in ['y1', 'y2']:
            prev_value = self._bbox_history.get(key, bbox[key])
            bbox[key] = alpha_y * prev_value + (1 - alpha_y) * bbox[key]

        # Update history
        self._bbox_history = bbox

        # Ensure confidence remains a float
        bbox[TrackingEnums.KEY_CONFIDENCE.value] = float(bbox[TrackingEnums.KEY_CONFIDENCE.value])

        self.logger.debug(f"Smoothed bounding box: {bbox}")
        return bbox

    @staticmethod
    def mirror_calc(servo_angle: int) -> int:
        """
        Mirror the given servo angle around the midpoint (usually 90 or 180).
        Useful for servos that need to align in opposite directions.

        :param servo_angle: The current or target angle of one servo.
        :return: The mirrored angle.
        """
        return 180 - servo_angle

    def handle_intensity(self, msg: MQTTMessage) -> None:
        """
        Handle intensity messages (e.g., LED updates) for this location.

        :param msg: MQTTMessage containing intensity information in its payload.
        """
        j_msg = loads(msg.payload.decode())
        if j_msg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {j_msg}")
            self.intensity = j_msg["intensity"]
