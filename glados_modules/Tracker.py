import time
from typing import Dict, Callable, Tuple, NamedTuple
from json import loads
from collections import namedtuple
from math import sqrt

# 3rd party imports
from paho.mqtt.client import MQTTMessage

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.MqttClient import MQTTClient, ServoMessageBuilder
from glados_modules.GladosData import ServoLocation, VisionTracker
from glados_modules.GLaDosEnums import CameraEnum, ServoEnum, SystemEnums, TrackingEnums, VisionResultsEnum


class MotionTrack(MQTTClient):
    # class for motion tracking on a target
    # TODO figure out if we want this here, or in teh Gbody class in the body server?
    def __init__(self, broker: NamedTuple,  camera_resolution: NamedTuple, target: str = "person",
                 confidence: float = 0.65, move_fudge_factor: int = 3):
        self.__name__ = self.__class__.__name__
        self.location = self.__name__
        self.logger = setup_logger(self.__name__)
        self.cmd_topic: str = TrackingEnums.MQTT_COMMAND_TOPIC.value
        self.cmd_trigger: str = TrackingEnums.MSG_COMMAND_KEY.value
        self.intensity_topic: str = SystemEnums.MQTT_INTENSITY_TOPIC.value
        self.count = VisionResultsEnum.VISION_RESULTS_COUNT_KEY.value
        self.intensity: Tuple[float, float] = (.1, .1)
        self._bbox_history: dict = {}
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd,
                                                   self.intensity_topic: self.handle_intensity}
        # head camera resolution
        # TODO this will work for now but need to get all camera resolution to account for side cameras
        self.cam_x = int(camera_resolution.x)
        self.cam_y = int(camera_resolution.y)
        self.main_camera = CameraEnum.CAMERA_HEAD.value
        self.left_camera = CameraEnum.CAMERA_LEFT.value
        self.right_camera = CameraEnum.CAMERA_RIGHT.value
        self.dead_zone_factor = move_fudge_factor
        servo = namedtuple("servo", ["name", "move"])
        # servo names
        self.head_LR = servo(ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value, ServoMessageBuilder.head_left_right)
        self.head_UD = servo(ServoEnum.LOCATION_HEAD_UP_DOWN.value, ServoMessageBuilder.head_up_down)
        self.body_LR = servo(ServoEnum.LOCATION_BODY_LEFT_RIGHT.value, ServoMessageBuilder.body_left_right)
        self.body_UD = servo(ServoEnum.LOCATION_BODY_UP_DOWN.value, ServoMessageBuilder.body_up_down)
        self.target = target
        self.confidence = confidence
        self.servos = dict()
        # default movement speed
        self.dms: int = 3
        # bool if movement on left or right cameras, true we move, false we dont
        self.peripheral_hunt = True
        # bool if the head is currently tracking something
        self.head_tracking = False
        super().__init__(ip=broker.ip, port=broker.port)
        self.side_camera_count: int = 0
        # Create Servo Location Tracker
        self.servo_status = ServoLocation(broker)
        # Vision seen Tracker
        self.objects = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
        self.vision_tracker = VisionTracker(broker, self.target, self.confidence, self.track_loop)
        #hanging tracker
        self.hanging = False
        self._last_move_time = time.time()  # Update the last move time
        # TODO do we need these there? are we sending signals? maybe trigger LED events? Maybe pulse eye down?
        # TODO figure out how we are going to track anger intensity over various body parts

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """
        Trigger the loop that hunts and locks onto target...
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

    def __dead_zone_check(self, servo, new_angle: int, degree_diff: int = 2,
                          confidence: float = 0.8, depth: float = 1.0) -> int:
        """Adaptive dead zone for filtering jittery movements."""
        # Use a higher dead zone for vertical movement (Y-axis)
        if servo.location == ServoEnum.LOCATION_HEAD_UP_DOWN.value:
            degree_diff = 5  # Increase to stabilize head nodding
        # Adjust dead zone dynamically based on confidence
        dynamic_diff = (degree_diff * (1 - confidence) + 1)
        # Increase tolerance for medium confidence (0.6-0.7)
        if 0.6 <= confidence <= 0.7:
            dynamic_diff += 1.5  # More aggressive filtering
        # Modify by distance
        distance_factor = max(0.5, min(2.0, depth))
        dynamic_diff *= distance_factor
        # Apply a hard minimum threshold
        dynamic_diff = max(dynamic_diff, 3)  # Ensure no movement for <3° differences
        current_angle = servo.current
        move = abs(new_angle - current_angle) > dynamic_diff
        self.logger.debug(f"{servo.location}: Angle {new_angle}, Current {current_angle}, "
                          f"Diff {abs(new_angle - current_angle)}, Threshold {dynamic_diff}, Move: {move}")
        return move

    def track_loop(self, camera: str) -> None:
        # main tracking loop
        # find target
        # don't double call if head_tracking is True, just skip this detection
        if self.__check_tracking() is True:
            self.logger.debug("Getting Vision Map")
            vision_map = self.vision_tracker.get_vision_map()
            self.logger.debug("Looping though vision map")
            if camera in vision_map.keys():
                if vision_map[camera][self.target].get(self.count, 0) != 0:
                    target_bounding = self.__find_target(vision_map[camera][self.target][self.objects])
                    target_ts = vision_map[camera].get(VisionResultsEnum.VISION_RESULTS_TS_KEY.value, None)
                    if camera == TrackingEnums.BODY_HEAD_CAMERA.value:
                        self.logger.debug(f"Ready to move all servos for " +
                                          f"target {self.target} message times stamp {target_ts}" +
                                          f"for {camera}")
                        # attempt to smooth the bounding box for visual noise
                        target_bounding = self.smooth_bounding_box(target_bounding)
                        self.move_all_servos(target_bounding, camera)
                        with self._lock:
                            self.side_camera_count = 0
                            self.hanging = False
                        self.logger.debug(f"Movement complete for target {self.target} and message times stamp {target_ts}")
                    elif camera in (TrackingEnums.BODY_LEFT_CAMERA.value, TrackingEnums.BODY_RIGHT_CAMERA.value):
                        if self.side_camera_count <= 5:
                            self.logger.debug(f"Rotating Body to face target {self.target}")
                            self.rotate_body(target=target_bounding, camera=camera, flip=False)
                            with self._lock:
                                self.side_camera_count += 1
                            # hold for a while to let main camera capture targets
                            time.sleep(3)
                        else:
                            # move to just hang around
                            if self.hanging is False:
                                self.hang_around()
                                self.logger.debug("Couldn't get target on head camera in 5 tries skipping for now")
                            else:
                                self.logger.debug("Already hanging out")
            with self._lock:
                self.head_tracking = False

    def __check_tracking(self) -> bool:
        """
        Check if we are tracking and return a bool, if we are not set the master bool to true
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

    def rotate_body(self, target: dict, camera: str, flip=False, return_message=False) -> None | tuple:
        # Get current servo position
        self.logger.debug("Moving servos getting angle map")
        self.servos = self.servo_status.get_angle_map()
        self.logger.debug("Calculating movement for servos")
        mv_list = list()
        body_movement = dict()
        if target != {}:
            # account for left right swap
            body_lr = self.__calc_servo(self.servos[self.body_LR.name], target, camera=camera)
            if self.__dead_zone_check(self.servos[self.body_LR.name], body_lr, degree_diff=self.dead_zone_factor,
                                      confidence=target[TrackingEnums.KEY_CONFIDENCE.value]):
                if flip is True:
                    body_lr = self.__mirror_calc(body_lr)
                mv_list.append(self.body_LR.move(body_lr))
                if mv_list:
                    self.logger.debug("Sending Move commands for Head and Neck")
                    body_movement = {self.body_LR.name: body_lr}
                    if return_message is False:
                        self.servo_status.send_command(mv_list, ServoEnum.MQTT_COMMAND_TOPIC.value)
                        self.__block_for_update(body_movement)
        if return_message is True:
            return body_movement, mv_list

    def hang_around(self) -> None:
        # rotate to the center point and then hang with head slightly picked up
        # this is expected to get called when there is nothing else to do so not waiting or blocking for movement
        with self._lock:
            self.hanging = True
        self.logger.debug("Deciding to hang out")
        msglist = [ServoMessageBuilder.body_left_right(angle=90, speed=1)]
        msglist.extend([ServoMessageBuilder.head_left_right(angle=92, speed=1)])
        msglist.extend([ServoMessageBuilder.head_up_down(angle=125, speed=1)])
        msglist.append(ServoMessageBuilder.body_up_down(angle=180, speed=1))
        self.servo_status.send_command(msglist, ServoEnum.MQTT_COMMAND_TOPIC.value)

    def move_all_servos(self, target: dict, camera: str) -> None:
        # Get current servo position
        self.logger.debug("Moving servos getting angle map")
        self.servos = self.servo_status.get_angle_map()
        mv_list = list()
        self.logger.debug("Calculating movement for servos")
        if target != {}:
            if hasattr(self, '_last_move_time') and not self.rate_limited_update(self._last_move_time, interval=.4):
                self.logger.debug("Rate limited, moved too soon")
                return
            self._last_move_time = time.time()  # Update the last move time

            # Move head left-right and up-down first
            head_lr = self.__calc_servo(self.servos[self.head_LR.name], target, camera=camera)
            head_ud = self.__calc_servo(self.servos[self.head_UD.name], target, camera=camera)

            # TODO stop breaking the x axis
            head_lr = self.servos[self.head_LR.name].current

            if self.__dead_zone_check(self.servos[self.head_LR.name], head_lr, self.dead_zone_factor):
                mv_list.append(self.head_LR.move(head_lr))
            else:
                # don't try small movements just set it to current
                head_lr = self.servos[self.head_LR.name].current
            # get rid of smaller movements on the head
            if self.__dead_zone_check(self.servos[self.head_UD.name], head_ud,
                                      degree_diff=self.dead_zone_factor,
                                      confidence=target[TrackingEnums.KEY_CONFIDENCE.value]):
                mv_list.append(self.head_UD.move(head_ud))
            else:
                head_ud = self.servos[self.head_UD.name].current
            if mv_list:
                self.logger.debug("Sending Move commands for Head and Neck")
                self.servo_status.send_command(mv_list, ServoEnum.MQTT_COMMAND_TOPIC.value)
                head_movement = {self.head_LR.name: head_lr, self.head_UD.name: head_ud}
                self.__block_for_update(head_movement)
            # Check if head movement reached its limit and compensate with body movement
            if self.__reached_limit(self.servos[self.head_LR.name]):
                self.logger.debug("Head reached left/right limit, rotating body to extend range")
                self.__rotate_body_to_extend_range()
            if self.__reached_limit(self.servos[self.head_UD.name]):
                self.logger.debug("Head reached up/down limit, bending body to extend range")
                self.__bend_body_to_extend_range()
            # Level the head with the body after movement
            # servo_1, servo_2 = self.__level_servos(self.head_LR, self.body_LR)
            body_movement, mv_list = self.rotate_body(target, camera, return_message=True)
            # level the head
            middle = self.servos[self.head_LR.name].middle
            if self.__dead_zone_check(self.servos[self.head_LR.name], middle,
                                      degree_diff=self.dead_zone_factor,
                                      confidence=target[TrackingEnums.KEY_CONFIDENCE.value]):
                mv_list.append(self.head_LR.move(middle))
            self.servo_status.send_command(mv_list, ServoEnum.MQTT_COMMAND_TOPIC.value)
            # level the body
            servo_3, servo_4 = self.__level_servos(self.head_UD, self.body_UD)
            self.logger.debug("Leveling out body")
            body_level = {self.head_UD.name: servo_3, self.body_UD.name: servo_4}
            body_level.update(body_movement)
            self.__block_for_update(body_level)
            # Add a small delay to make the movement seem more deliberate
            self.logger.debug("Leveling out body complete")

    def __block_for_update(self, target_positions: Dict[str, int]) -> None:
        # Loop until all servos reach their target positions
        count = 0
        self.logger.debug(f"Waiting for updates on {len(target_positions.keys())}")
        while True:
            self.servos = self.servo_status.get_angle_map()
            all_reached = True
            for name, target in target_positions.items():
                if self.servos[name].current != target:
                    all_reached = False
                    self.logger.debug(f"{name} servo is currently blocking attempting to get to {target}")
                    break
                else:
                    self.logger.debug(f"{name} servo has updated and reached {target}")
            if all_reached:
                break
            time.sleep(0.2)
            count += 1
            if count >= 15:
                # hard block for some reason, trigger servo updates
                count = 0
                self.servo_status.update_servo_status()
        self.logger.debug(f"Blocking Updates Complete")

    def __reached_limit(self, servo) -> bool:
        """
        Check if the servo has reached its movement limit.
        """
        return servo.current == servo.min or servo.current == servo.max

    def __rotate_body_to_extend_range(self):
        self.logger.debug("Rotating body to extend range of neck")
        # Calculate the difference between head's current position and middle
        diff = self.servos[self.head_LR.name].current - self.servos[self.head_LR.name].middle
        # Adjust body servo in the same direction
        new_body_angle = self.servos[self.body_LR.name].current + diff
        # Clamp the new angle within body's allowed range
        new_body_angle = max(min(new_body_angle, self.servos[self.body_LR.name].max),
                             self.servos[self.body_LR.name].min)
        # Send movement command
        self.servo_status.send_command(
            [self.body_LR.move(new_body_angle)],
            ServoEnum.MQTT_COMMAND_TOPIC.value)
        # Block until the movement is completed
        self.__block_for_update({self.body_LR.name: new_body_angle})

    def __bend_body_to_extend_range(self):
        self.logger.debug("Bending body to extend range of head")
        # Calculate the difference between head's current position and middle
        diff = self.servos[self.head_UD.name].current - self.servos[self.head_UD.name].middle
        # Adjust body servo in the same direction
        new_body_angle = self.servos[self.body_UD.name].current + diff
        # Clamp the new angle within body's allowed range
        new_body_angle = max(min(new_body_angle, self.servos[self.body_UD.name].max),
                             self.servos[self.body_UD.name].min)
        # Send movement command
        self.servo_status.send_command(
            [self.body_UD.move(new_body_angle)],
            ServoEnum.MQTT_COMMAND_TOPIC.value)
        # Block until the movement is completed
        self.__block_for_update({self.body_UD.name: new_body_angle})

    def __find_target(self, seen_data) -> dict:
        """
        Find the highest confidence target and return their bounding box from current data set
        """
        confidence = VisionResultsEnum.VISION_RESULTS_CONFIDENCE_KEY.value
        bbox = VisionResultsEnum.VISION_RESULTS_BOX_KEY.value
        rtn = dict()
        highest_confidence = 0
        for p in seen_data:
            if p[confidence] > highest_confidence:
                highest_confidence = p[confidence]
                rtn = p[bbox]
            rtn[TrackingEnums.KEY_CONFIDENCE.value] = p[confidence]
        self.logger.debug(f"Confidence box found {rtn} with confidence score of {highest_confidence}")
        return rtn

    @staticmethod
    def fisheye_correction(offset_proportion, fov):
        if fov >= 160:
            # Apply a non-linear correction based on radial distortion
            k1 = 0.2  # Example distortion coefficient for fisheye
            radial_distance = sqrt(offset_proportion ** 2)
            corrected_proportion = offset_proportion * (1 + k1 * (radial_distance ** 2))
            return corrected_proportion
        return offset_proportion

    def rate_limited_update(self, last_update_time: float, interval: float = 0.2) -> bool:
        """
        Check if enough time has passed since the last update.
        :return: bool
        """

        current_time = time.time()
        if current_time - last_update_time < interval:
            self.logger.debug("Rate limiting: Skipping update.")
            return False
        return True

    def __calc_servo(self, servo, bbox: dict, camera: str) -> int:
        # Determine axis and image dimensions
        if servo.axis == ServoEnum.X_AXIS.value:
            bbox_edge_1 = bbox['x1']
            bbox_edge_2 = bbox['x2']
            axis_size = self.cam_x
            # Determine direction factor based on servo location
            if servo.location == ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value:
                direction_factor = 1  # direct servo drive
            else:
                direction_factor = -1  # inverse because we use a 2 gear drive
        else:
            bbox_edge_1 = bbox['y1']
            bbox_edge_2 = bbox['y2']
            axis_size = self.cam_y
            # Determine direction factor based on servo location
            if servo.location == ServoEnum.LOCATION_HEAD_UP_DOWN.value:
                direction_factor = 1  # head up down direct drive
            else:
                direction_factor = -1  #
        # Calculate the center of the bounding box on the axis
        center_of_bbox = (bbox_edge_1 + bbox_edge_2) / 2
        # Calculate the offset from the image center (in pixels)
        offset_from_center = (axis_size / 2) - center_of_bbox  # Reverse due to camera movement
        # Calculate the proportion of the offset relative to the image size
        offset_proportion = offset_from_center / (axis_size / 2)  # Normalize between -1 and 1
        # Calculate angle adjustment based on camera field of view (FOV)
        # get the right focal from ENUMS
        fov = 54
        mounting_angle = 0
        current = servo.current
        if camera == CameraEnum.CAMERA_HEAD.value:
            if servo.axis == ServoEnum.X_AXIS.value:
                fov = CameraEnum.CAMERA_HEAD_FOV_X.value  # Camera's field of view in degrees
            else:
                fov = CameraEnum.CAMERA_HEAD_FOV_Y.value
        if camera == CameraEnum.CAMERA_RIGHT.value:
            fov = CameraEnum.CAMERA_RIGHT_FOV.value
            if servo.axis == ServoEnum.X_AXIS.value and servo.location == ServoEnum.LOCATION_BODY_LEFT_RIGHT.value:
                mounting_angle = 55
                current = 90
                #direction_factor = 1
                # make calculations off 90

            # account for fisheye
            offset_proportion = MotionTrack.fisheye_correction(offset_proportion=offset_proportion, fov=fov)
        if camera == CameraEnum.CAMERA_LEFT.value:
            fov = CameraEnum.CAMERA_LEFT_FOV.value
            if servo.axis == ServoEnum.X_AXIS.value and servo.location == ServoEnum.LOCATION_BODY_LEFT_RIGHT.value:
                mounting_angle = -55
                current = 90
                #direction_factor = 1
                # make calcuations off 90
            # account for fisheye
            offset_proportion = MotionTrack.fisheye_correction(offset_proportion=offset_proportion, fov=fov)
        angle_adjustment = direction_factor * offset_proportion * (fov / 2)  # Adjust for FOV
        # Determine the new servo angle based on the current position, and camera that saw it
        if camera in (CameraEnum.CAMERA_LEFT.value, CameraEnum.CAMERA_RIGHT.value):
            self.logger.debug(f"Side camera calc is currently at {current} with an adjustment of {angle_adjustment} " +
                              f"before mounting correction of {mounting_angle} and " +
                              f"a direction angle of {direction_factor}")
        new_servo_angle = current + angle_adjustment + mounting_angle
        # Clamp the new angle within servo's min and max
        new_servo_angle = max(min(new_servo_angle, servo.max), servo.min)
        # Round to the nearest whole number
        return round(new_servo_angle)

# we are over rotating because of leveling 52 on a head.. is not the same as 52 on the rotation of the body...
# body needs to calculate rotation distance to track correctly

    def __level_servos(self, servo1, servo2) -> tuple:
        # bring servo1 to midpoint by moving servo2
        # ensure servos are on the same axis
        self.logger.debug(f"Leveling Servos {self.servos[servo1.name].location} & {self.servos[servo2.name].location}")
        if self.servos[servo1.name].axis != self.servos[servo2.name].axis:
            msg = "Servos are not on same axis"
            self.logger.error(msg)
            raise Exception(msg)
        current = self.servos[servo1.name].current
        if servo1.name in (ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value, ServoEnum.LOCATION_HEAD_UP_DOWN.value):
            angle = self.__mirror_calc(current)
        else:
            angle = current
        # clamp to max min of travel
        self.logger.debug(f"Servo {servo2.name} is at {self.servos[servo2.name].current} before leveling")
        angle = max(min(angle, self.servos[servo2.name].max), self.servos[servo2.name].min)
        mv_list = [servo2.move(angle),
                   servo1.move(self.servos[servo1.name].middle)]
        self.servo_status.send_command(mv_list, ServoEnum.MQTT_COMMAND_TOPIC.value)
        # servo 1, servo 2
        if abs(self.servos[servo1.name].middle - angle) > self.dead_zone_factor:
            servo1_move = self.servos[servo1.name].middle
            servo2_move = angle
        else:
            servo1_move = self.servos[servo1.name].current
            servo2_move = self.servos[servo2.name].current
        return servo1_move, servo2_move

    def smooth_bounding_box(self, bbox: dict, alpha_x: float = 0.8, alpha_y: float=0.9) -> dict:
        """
        Smooth a single bounding box using an exponential moving average,
        applying separate smoothing factors for X and Y axes.

        :param bbox: Dictionary with bounding box coordinates {'x1', 'y1', 'x2', 'y2'}.
        :param alpha_x: Smoothing factor for horizontal (X) movement.
        :param alpha_y: Smoothing factor for vertical (Y) movement.
        :return: Dictionary with smoothed bounding box coordinates.
        """
        if not hasattr(self, '_bbox_history'):
            # Initialize history on the first call
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

        # Preserve confidence value
        bbox[TrackingEnums.KEY_CONFIDENCE.value] = float(bbox[TrackingEnums.KEY_CONFIDENCE.value])

        self.logger.debug(f"Smoothed bounding box: {bbox}")
        return bbox

    def __mirror_calc(self, servo_angle) -> int:
        """
        Figure out degree on other side when we have servos that need to align and their left and right's are flipped
        """
        return 180 - servo_angle

    def handle_intensity(self, msg: MQTTMessage) -> None:
        # TODO figure out update commands
        j_msg = loads(msg.payload.decode())
        if j_msg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {j_msg}")
            self.intensity = j_msg["intensity"]

