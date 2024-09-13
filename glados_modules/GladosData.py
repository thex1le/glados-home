from threading import Lock
from typing import Dict, Callable, NamedTuple
from json import loads
from time import time
from collections import namedtuple

# 3rd party
from paho.mqtt.client import MQTTMessage
from cachetools import TTLCache

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.MqttClient import MQTTClient, TargetMessageBuilder
from glados_modules.GLaDosEnums import ServoEnum, CameraEnum, VisionResultsEnum, TrackingEnums


class ServoLocation(MQTTClient):
    """
    Keep track of all the angles based on mqtt status updates
    """
    def __init__(self, broker: NamedTuple) -> None:
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__)
        MQTTClient.__init__(self, broker=broker.ip, port=broker.port)
        self.cmd_topic = ServoEnum.MQTT_STATUS_TOPIC.value
        self.topic_handler: Dict[ServoEnum, Callable] = {self.cmd_topic: self.handle_cmd}
        self.body_map = dict()
        self.lock = Lock()
        self.min = ServoEnum.MSG_MIN.value
        self.max = ServoEnum.MSG_MAX.value
        self.current_angle = ServoEnum.MSG_CURRENT_ANGLE.value
        self.middle = ServoEnum.MSG_MIDDLE.value
        self.axis = ServoEnum.MSG_AXIS.value
        self.ServoTuple = namedtuple('servo', [self.current_angle, self.max, self.min, self.middle,
                                     self.axis, "location"])

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """
        Command Handler
        """
        j_msg = loads(msg.payload.decode())
        if ServoEnum.MSG_LOCATION_KEY.value in j_msg.keys():
            # found a servo status, update the dict
            location = j_msg.get(ServoEnum.MSG_LOCATION_KEY.value)
            # you left off here populating the named tuple for the servo data
            servo_map = {location: self.ServoTuple(j_msg.get(self.current_angle),
                                                  j_msg.get(self.max), j_msg.get(self.min), j_msg.get(self.middle),
                                                  j_msg.get(self.axis), location)}
            with self.lock:
                self.body_map = servo_map

    def get_angle_map(self) -> dict:
        """
        Return angle map
        """
        with self.lock:
            return self.body_map


class VisionTracker(MQTTClient):
    """
    Keep track of all the vision results based on mqtt status updates
    """
    def __init__(self, broker: NamedTuple, target: str, confidence: float) -> None:
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__)
        MQTTClient.__init__(self, broker=broker.ip, port=broker.port)
        self.target = target
        self.lock = Lock()
        self.confidence_score = confidence
        self.cmd_topic = CameraEnum.MQTT_RESPONSE_TOPIC.value
        self.main_camera = CameraEnum.CONFIG_HEAD.value
        self.left_camera = CameraEnum.CAMERA_LEFT.value
        self.right_camera = CameraEnum.CAMERA_RIGHT.value
        self.cam_key = CameraEnum.MSG_LOCATION_KEY.value
        self.results_key = CameraEnum.MSG_RESULTS.value
        self.ts_key = VisionResultsEnum.VISION_RESULTS_TS_KEY.value
        self.topic_handler: Dict[CameraEnum, Callable] = {self.cmd_topic: self.handle_cmd}
        # Use a time cache and expire any vision tracking objects after 1min and 1000 objects
        # should only need 720 ( 3 cam 4 a second = 12 * 60 = 720) but leave some wiggle room, will need to adjust this
        # if we up the output frame rate
        self.response_cache = TTLCache(maxsize=1000, ttl=60)
        self.response_map = dict()
        self.count = VisionResultsEnum.VISION_RESULTS_COUNT_KEY.value
        self.objects_key = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
        self.confidence_key = VisionResultsEnum.VISION_RESULTS_CONFIDENCE_KEY.value
        # if we have people on both sides? how do we decide which way to turn? count based? last time seen? random?
        # need bools for left and right camera to mark if we currently see high confidence on target or not
        # when do we check the cache?
        # currently tracking object location bool's
        self.head_target = False
        self.left_target = False
        self.right_target = False

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """
        Command Handler
        """
        j_msg = loads(msg.payload.decode())
        if self.cam_key in j_msg.keys():
            # found a servo status, update the dict
            with self.lock:
                self.logger.debug(f"Camera message received, {msg.topic}, {j_msg}")
                self.parse_camera(msg=j_msg)

    def parse_camera(self, msg: dict):
        """
        Parse a camera message and add it to the cache and currently seen objects
        """
        camera = msg.get(self.cam_key, "")
        sight_results = msg.get(self.results_key)
        if self.target in sight_results.keys():
            for p in sight_results[self.target][self.objects_key]:
                if float(p[self.confidence_key]) >= self.confidence_score:
                    self.response_map[camera] = sight_results
                    # track how many high confidence in last 2 seconds
                    # create a timer tracker and + or - it depending on how many confidence hits in last .5 seconds
                    if self.ts_key not in self.response_map[camera].keys():
                        self.response_map[camera][self.ts_key] = time()
                        self.response_map[camera][self.count] = 1
                    else:
                        if time() - self.response_map[camera][self.ts_key] <= .5:
                            self.response_map[camera][self.count] += 1
                        else:
                            self.response_map[camera][self.count] -= 1
                    # store sight results
                    if camera in self.response_cache.keys():
                        # add to an existing cache
                        self.response_cache[camera].update({sight_results[self.ts_key]: sight_results})
                        # add a new camera to the cache
                    else:
                        self.response_cache[camera] = {sight_results[self.ts_key]: sight_results}
                    self.send_command(TargetMessageBuilder.send_track_command_start(), TrackingEnums.MQTT_COMMAND_TOPIC)

    def get_vision_map(self) -> dict:
        """
        Return just the last vision response messages seen
        """
        with self.lock:
            # return a copy of the cache
            return self.response_map

    def get_vision_cache(self) -> dict:
        """
        Return 5 min cache of things seen
        """
        with self.lock:
            # return a copy of the cache
            return dict(self.response_cache)
