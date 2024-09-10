from threading import Lock
from typing import Dict, Callable, NamedTuple
from json import loads

# 3rd party
from paho.mqtt.client import MQTTMessage

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.MqttClient import MQTTClient
from glados_modules.GLaDosEnums import ServoEnum, CameraEnum


class ServoLocation(MQTTClient):
    """
    Keep track of all the angles based on mqtt status updates
    """
    def __init__(self, broker: NamedTuple) -> None:
        self.__name__ = self.__class__.__name__
        MQTTClient.__init__(self, broker=broker.ip, port=broker.port)
        self.logger = setup_logger(name=self.__name__)
        self.cmd_topic = ServoEnum.MQTT_STATUS_TOPIC.value
        self.topic_handler: Dict[ServoEnum, Callable] = {self.cmd_topic: self.handle_cmd}
        self.body_map = dict()

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """
        Command Handler
        """
        j_msg = loads(msg.payload.decode())
        if ServoEnum.MSG_LOCATION_KEY.value in j_msg.keys():
            # found a servo status, update the dict
            with Lock:
                self.body_map = j_msg

    def get_angle_map(self) -> dict:
        """
        Return angle map
        """
        with Lock:
            return self.body_map


class VisionResults(MQTTClient):
    """
    Keep track of all the vision results based on mqtt status updates
    """
    def __init__(self, broker: NamedTuple) -> None:
        self.__name__ = self.__class__.__name__
        MQTTClient.__init__(self, broker=broker.ip, port=broker.port)
        self.logger = setup_logger(name=self.__name__)
        self.cmd_topic = CameraEnum.MQTT_RESPONSE_TOPIC.value
        self.topic_handler: Dict[CameraEnum, Callable] = {self.cmd_topic: self.handle_cmd}
        self.response_map = dict()

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """
        Command Handler
        """
        j_msg = loads(msg.payload.decode())
        if CameraEnum.MSG_LOCATION_KEY.value in j_msg.keys():
            # found a servo status, update the dict
            with Lock:
                self.response_map = j_msg

    def get_vision_map(self) -> dict:
        """
        Return angle map
        """
        with Lock:
            return self.response_map
