# native imports
from typing import Dict, Callable
from json import dumps

# 3rd party imports
import paho.mqtt.client as mqtt

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GLaDosEnums import ServoEnum


class MQTTClient:
    def __init__(self, broker: str = 'localhost', port: int = 1883) -> None:
        self.broker = broker
        self.port = int(port)
        self.client: mqtt.Client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.topic_handler: Dict[str, Callable] = {}
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()
        self.logger = setup_logger(name=f"{self.__name__}")

    def on_connect(self, client: mqtt.Client, userdata: object, flags: dict, rc: int) -> None:
        self.logger.debug(f"Connecting to {self.broker}:{self.port}")
        for topic in self.topic_handler:
            self.client.subscribe(topic)

    def on_message(self, client: mqtt.Client, userdata: object, msg: mqtt.MQTTMessage) -> None:
        if msg.topic in self.topic_handler:
            self.topic_handler[msg.topic](msg)

    def send_command(self, command: dict| list| tuple, topic) -> None:
        """
        Generic mqtt sending function for single or multiple messages
        """
        if type(command) not in (tuple, list):
            # make it an object we can iterate on
            command = tuple(command)
        for m in command:
            self.logger.debug(f"{self.__name__} sending {m} command")
            self.client.publish(topic, dumps(m))

# TODO flesh out message classes for easy update in one place
# should thees be enums?


class ServoMessageBuilder:
    """
    Build and return servo messages based on enums
    """
    @staticmethod
    def head_up_down(angle: int, speed: int = 1) -> dict:
        return {ServoEnum.MSG_LOCATION_KEY: ServoEnum.LOCATION_HEAD_UP_DOWN,
                ServoEnum.MSG_ANGLE: angle, ServoEnum.MSG_SPEED: speed}

    @staticmethod
    def body_left_right(angle: int, speed: int = 1) -> dict:
        return {ServoEnum.MSG_LOCATION_KEY: ServoEnum.LOCATION_BODY_LEFT_RIGHT,
                ServoEnum.MSG_ANGLE: angle, ServoEnum.MSG_SPEED: speed}

    @staticmethod
    def body_up_down(angle: int, speed: int = 1) -> dict:
        return {ServoEnum.MSG_LOCATION_KEY: ServoEnum.LOCATION_BODY_UP_DOWN,
                ServoEnum.MSG_ANGLE: angle, ServoEnum.MSG_SPEED: speed}

    @staticmethod
    def head_left_right(angle: int, speed: int = 1) -> dict:
        return {ServoEnum.MSG_LOCATION_KEY: ServoEnum.LOCATION_HEAD_LEFT_RIGHT,
                ServoEnum.MSG_ANGLE: angle, ServoEnum.MSG_SPEED: speed}

    @staticmethod
    def send_status(location, results):
        return  {ServoEnum.MSG_LOCATION_KEY: location,
                 ServoEnum.MSG_COMMAND_KEY: ServoEnum.MSG_COMMAND_STATUS,
                 ServoEnum.MSG_RESULTS: results}
