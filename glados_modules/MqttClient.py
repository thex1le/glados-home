# native imports
from typing import Dict, Callable, NamedTuple
from json import dumps, loads
from uuid import uuid4
from time import time
from threading import Lock
from collections import namedtuple

# 3rd party imports
import paho.mqtt.client as mqtt
from cachetools import TTLCache

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GLaDosEnums import ServoEnum, CameraEnum, TrackingEnums


class MQTTClient:
    broker_tuple = namedtuple('broker', ["ip", "port"])

    #TODO convert this to named tuple
    def __init__(self, ip: str = 'localhost', port: int = 1883) -> None:
        self.ip = ip
        self.port = int(port)
        if not hasattr(self, 'topic_handler'):
            self.topic_handler: Dict[str, Callable] = {}
        self.uuid_cache = TTLCache(maxsize=100, ttl=60)
        try:
            self.logger = setup_logger(name=f"{self.__name__}")
        except AttributeError:
            self.logger = setup_logger(name=f"{self.__class__.__name__}")
            self.__name__ = self.__class__.__name__
        self._lock = Lock()
        self.client: mqtt.Client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.ip, self.port, 60)
        self.client.loop_start()

    def on_connect(self, client: mqtt.Client, userdata: object, flags: dict, rc: int) -> None:
        self.logger.debug(f"Connecting to {self.ip}:{self.port}")
        with self._lock:
            for topic in self.topic_handler:
                self.client.subscribe(topic, qos=1)

    def on_message(self, client: mqtt.Client, userdata: object, msg: mqtt.MQTTMessage) -> None:
        j_msg = loads(msg.payload.decode())
        uuid = j_msg.get("uuid", None)
        if uuid is None:
            self.logger.error("NO UUID IN MESSAGE")
            return
        with self._lock:
            if uuid not in self.uuid_cache.keys():
                self.uuid_cache[uuid] = time()
                handler = self.topic_handler.get(msg.topic)
        if handler:
            handler(msg)

    def send_command(self, command: dict | list | tuple, topic, qos: int = 1) -> None:
        """
        Generic mqtt sending function for single or multiple messages
        """
        if not isinstance(command, (tuple, list)):
            # make it an object we can iterate on
            command = (command, )
        for m in command:
            # add in uuid for message tracking and debugging
            m["uuid"] = str(uuid4())
            self.logger.debug(f"{self.__name__} sending {m} command")
            self.client.publish(topic, dumps(m), qos=qos)


# TODO flesh out message classes for easy update in one place
class TargetMessageBuilder:
    @staticmethod
    def send_track_command_start(camera):
        return {TrackingEnums.MSG_COMMAND_KEY.value: TrackingEnums.MSG_COMMAND_START.value,
                TrackingEnums.MSG_CAMERA_KEY.value: camera}


class CameraMessageBuilder:
    @staticmethod
    def send_status(location, status):
        return {CameraEnum.MSG_LOCATION_KEY.value: location,
                CameraEnum.MSG_COMMAND_KEY.value: CameraEnum.MSG_COMMAND_STATUS.value,
                CameraEnum.MSG_RESULTS.value: status}

    @staticmethod
    def send_results(location, results):
        return {CameraEnum.MSG_LOCATION_KEY.value: location,
                CameraEnum.MSG_RESULTS.value: results}


class ServoMessageBuilder:
    """
    Build and return servo messages based on enums
    """
    @staticmethod
    def move(location, angle, speed):
        return {ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_MOVE.value,
                ServoEnum.MSG_LOCATION_KEY.value: location,
                ServoEnum.MSG_ANGLE.value: angle, ServoEnum.MSG_SPEED.value: speed}

    @staticmethod
    def head_up_down(angle: int, speed: int = ServoEnum.SERVO_DEFAULT_SPEED.value) -> dict:
        return ServoMessageBuilder.move(ServoEnum.LOCATION_HEAD_UP_DOWN.value, angle, speed)

    @staticmethod
    def body_left_right(angle: int, speed=ServoEnum.SERVO_DEFAULT_SPEED.value) -> dict:
        return ServoMessageBuilder.move(ServoEnum.LOCATION_BODY_LEFT_RIGHT.value, angle, speed)

    @staticmethod
    def body_up_down(angle: int, speed=ServoEnum.SERVO_DEFAULT_SPEED.value) -> dict:
        return ServoMessageBuilder.move(ServoEnum.LOCATION_BODY_UP_DOWN.value, angle, speed)

    @staticmethod
    def head_left_right(angle: int, speed=ServoEnum.SERVO_DEFAULT_SPEED.value) -> dict:
        return ServoMessageBuilder.move(ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value, angle, speed)

    @staticmethod
    def get_status(location):
        return {ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_STATUS.value,
                ServoEnum.MSG_LOCATION_KEY.value: location}

    @staticmethod
    def send_status(location, results):
        return {ServoEnum.MSG_LOCATION_KEY.value: location,
                ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_STATUS.value,
                ServoEnum.MSG_RESULTS.value: results}
