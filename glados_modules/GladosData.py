from typing import Dict, Callable, NamedTuple
from json import loads, JSONDecodeError
from time import time, sleep
from collections import namedtuple

# 3rd party
from paho.mqtt.client import MQTTMessage
from cachetools import TTLCache

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.MqttClient import MQTTClient, TargetMessageBuilder, ServoMessageBuilder
from glados_modules.GLaDosEnums import ServoEnum, CameraEnum, VisionResultsEnum, TrackingEnums, LoggingEnums


class ServoLocation(MQTTClient):
    """
    Keep track of all the angles based on MQTT status updates.
    """
    def __init__(self, broker: NamedTuple) -> None:
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        # Initialize shared resources before calling the superclass constructor
        self.cmd_topic = ServoEnum.MQTT_STATUS_TOPIC.value
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd}
        self.body_map = dict()
        self.min = ServoEnum.MSG_MIN.value
        self.max = ServoEnum.MSG_MAX.value
        self.current_angle = ServoEnum.MSG_CURRENT_ANGLE.value
        self.middle = ServoEnum.MSG_MIDDLE.value
        self.moving = ServoEnum.MSG_MOVING.value
        self.axis = ServoEnum.MSG_AXIS.value
        self.last_angle = ServoEnum.MSG_LAST_ANGLE.value
        self.ServoTuple = namedtuple(ServoEnum.MSG_LOCATION_KEY.value, [
            self.current_angle, self.max, self.min, self.middle, self.axis,
            self.moving, ServoEnum.MSG_LOCATION.value, self.last_angle
        ])
        self.servo_list = (
            ServoEnum.LOCATION_BODY_UP_DOWN.value,
            ServoEnum.LOCATION_HEAD_UP_DOWN.value,
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value,
            ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
        )
        # Call the superclass constructor
        super().__init__(ip=broker.ip, port=broker.port)

    def update_servo_status(self):
        """
        Trigger servo message status update.
        """
        self.logger.debug("Updating Servo angle status")
        msg = []
        for servo_location in self.servo_list:
            msg.append(ServoMessageBuilder.get_status(servo_location))
        # Send the status request commands
        self.send_command(msg, ServoEnum.MQTT_COMMAND_TOPIC.value)
        # Block and don't return until all the servos populate
        fail_count = 0
        while True:
            with self._lock:
                current_servo_count = len(self.body_map.keys())
            # fail count 600 is 2min
            if current_servo_count >= len(self.servo_list) or fail_count >= 600:
                break  # All servos have reported their status
            else:
                # Possible block here...
                # Keep sending request until we get them all
                for servo in self.servo_list:
                    with self._lock:
                        if servo not in self.body_map:
                            self.send_command(
                                ServoMessageBuilder.get_status(servo),
                                ServoEnum.MQTT_COMMAND_TOPIC.value)
                sleep(0.1)
                fail_count += 1
                self.logger.debug("Waiting for servo statuses to update...")

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """
        Command Handler for incoming servo status messages.
        """
        try:
            j_msg = loads(msg.payload.decode())
        except JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON message: {e}")
            return

        if ServoEnum.MSG_LOCATION_KEY.value in j_msg:
            # Found a servo status, update the dict
            location = j_msg.get(ServoEnum.MSG_LOCATION_KEY.value)
            results = j_msg.get(ServoEnum.MSG_RESULTS.value, {})
            self.logger.debug(f"Received servo status for {location}: {results}")
            with self._lock:
                self.body_map[location] = self.ServoTuple(
                    results.get(self.current_angle),
                    results.get(self.max),
                    results.get(self.min),
                    results.get(self.middle),
                    results.get(self.axis),
                    results.get(self.moving),
                    location,
                    results.get(self.last_angle))

    def get_angle_map(self) -> dict:
        """
        Return a copy of the angle map.
        """
        if not self.body_map or len(self.body_map) != len(self.servo_list):
            # Empty or not fully populated map, trigger a status update
            self.logger.debug("Servo map incomplete, updating servo statuses.")
            self.update_servo_status()
        # Return a copy to prevent external modifications
        angle_map_copy = self.body_map.copy()
        return angle_map_copy

    def check_movement(self) -> bool:
        """
        Check if any servos are moving and return true or false
        :return: return bool true if moving, false if not
        """
        if not self.body_map or len(self.body_map) != len(self.servo_list):
            # Empty or not fully populated map, trigger a status update
            self.logger.debug("Servo map incomplete, updating servo statuses.")
            self.update_servo_status()
        ret = False
        for s in self.servo_list:
            if s not in self.body_map.keys():
                # no status from servo yet so skip it
                continue
            if self.body_map[s].moving is True:
                ret = True
                break
        return ret


class VisionTracker(MQTTClient):
    """
    Keep track of all the vision results based on mqtt status updates
    """
    def __init__(self, broker: NamedTuple, target: str, confidence: float, tracker_callback) -> None:
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.target = target
        self.tracker_callback = tracker_callback
        self.confidence_score = confidence
        # TODO get side confidence from config file
        self.side_confidence_score = .5
        self.cmd_topic = CameraEnum.MQTT_RESPONSE_TOPIC.value
        self.main_camera = CameraEnum.CONFIG_HEAD.value
        self.left_camera = CameraEnum.CAMERA_LEFT.value
        self.right_camera = CameraEnum.CAMERA_RIGHT.value
        self.cam_key = CameraEnum.MSG_LOCATION_KEY.value
        self.results_key = CameraEnum.MSG_RESULTS.value
        self.ts_key = VisionResultsEnum.VISION_RESULTS_TS_KEY.value
        self.count = VisionResultsEnum.VISION_RESULTS_COUNT_KEY.value
        self.objects_key = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
        self.confidence_key = VisionResultsEnum.VISION_RESULTS_CONFIDENCE_KEY.value
        # Initialize the topic handler before calling the superclass constructor
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd}
        # Call the superclass constructor
        super().__init__(ip=broker.ip, port=broker.port)
        # Use a time cache and expire any vision tracking objects after 10 seconds
        self.response_cache = TTLCache(maxsize=200, ttl=10)
        self.response_map = dict()
        # Tracking variables
        self.head_target = False
        self.left_target = False
        self.right_target = False

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """
        Command Handler
        """
        try:
            j_msg = loads(msg.payload.decode())
        except JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON message: {e}")
            return

        if self.cam_key in j_msg:
            self.logger.debug(f"Camera message received, {msg.topic}, {j_msg}")
            # Protect shared resources in parse_camera
            self.parse_camera(msg=j_msg)

    def parse_camera(self, msg: dict):
        """
        Parse a camera message and add it to the cache and currently seen objects
        """
        camera = msg.get(self.cam_key, "")
        sight_results = msg.get(self.results_key, {})
        if self.target in sight_results:
            with self._lock:
                for p in sight_results[self.target][self.objects_key]:
                    c = p.get(self.confidence_key, 0.0)
                    if camera == CameraEnum.CAMERA_HEAD.value:
                        cf_score = self.confidence_score
                    elif camera in (CameraEnum.CAMERA_RIGHT.value, CameraEnum.CAMERA_LEFT.value):
                        cf_score = self.side_confidence_score
                    if float(c) >= cf_score:
                        self.logger.debug(f"Confidence of {c} found for {self.target}")
                        # Update response_map
                        self.response_map[camera] = sight_results
                        # Update counts and timestamps
                        current_time = time()
                        last_ts = self.response_map[camera].get(self.ts_key, 0)
                        if current_time - last_ts <= 0.5:
                            self.response_map[camera][self.count] = self.response_map[camera].get(self.count, 0) + 1
                        else:
                            self.response_map[camera][self.count] = max(
                                0, self.response_map[camera].get(self.count, 1) - 1)
                        self.response_map[camera][self.ts_key] = current_time
                        # Store sight results in response_cache
                        if camera in self.response_cache:
                            # Add to an existing cache
                            self.response_cache[camera][current_time] = sight_results
                        else:
                            # Add a new camera to the cache
                            self.response_cache[camera] = {current_time: sight_results}
                        self.logger.debug(f"Sending Start command to track object {self.target} with a score of {c}")
                        # Send the tracking command
                        if camera == TrackingEnums.BODY_HEAD_CAMERA.value:
                            self.send_command(
                                TargetMessageBuilder.send_track_command_start(camera),
                                TrackingEnums.MQTT_COMMAND_TOPIC.value)
                        elif camera in (TrackingEnums.BODY_LEFT_CAMERA.value, TrackingEnums.BODY_RIGHT_CAMERA.value):
                            # head has not seen any target in 60 seconds or more, see response_cache creation to verify
                            # use side cameras to try and find target
                            if TrackingEnums.BODY_HEAD_CAMERA.value not in self.response_cache.keys():
                                self.send_command(
                                    TargetMessageBuilder.send_track_command_start(camera),
                                    TrackingEnums.MQTT_COMMAND_TOPIC.value)

    def get_vision_map(self) -> dict:
        """
        Return just the last vision response messages seen
        """
        with self._lock:
            # Return a copy of the response_map
            return self.response_map.copy()

    def get_vision_cache(self) -> dict:
        """
        Return the vision results cache
        """
        with self._lock:
            # Return a copy of the response_cache
            return dict(self.response_cache)


if __name__ == "__main__":
    b = namedtuple("broker", ["ip", "port"])
    broker = b('192.168.86.23', 1883)
    # Assuming broker is a NamedTuple with 'ip' and 'port' attributes
    servo_location_tracker = ServoLocation(broker)

    # Retrieve the current servo angles
    angle_map = servo_location_tracker.get_angle_map()

    # Access servo data
    for servo_name, servo_data in angle_map.items():
        print(f"Servo {servo_name}:")
        print(f"  Current Angle: {servo_data.current}")
        print(f"  Min Angle: {servo_data.min}")
        print(f"  Max Angle: {servo_data.max}")
        print(f"  Middle Angle: {servo_data.middle}")
        print(f"  Axis: {servo_data.axis}")
