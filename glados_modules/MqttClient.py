import paho.mqtt.client as mqtt
from typing import Dict, Callable
from glados_modules.GlogConfig import setup_logger


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
