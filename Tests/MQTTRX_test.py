from paho.mqtt.client import MQTTMessage
from collections import namedtuple
from typing import NamedTuple, Dict, Callable
from glados_modules.MqttClient import MQTTClient
from glados_modules.GLaDosEnums import TrackingEnums
from glados_modules.GlogConfig import setup_logger
from json import loads
from time import sleep


class TestMqtt(MQTTClient):
    def __init__(self, broker: NamedTuple) -> None:
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__)
        self.cmd_topic: str = TrackingEnums.MQTT_COMMAND_TOPIC.value
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd}
        MQTTClient.__init__(self, broker=broker.ip, port=broker.port)

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """
        Trigger the loop that hunts and locks onto target...
        """
        j_msg = loads(msg.payload.decode())
        print("************* TRACKING FIRED WTF!?!?!?!")


if __name__ == "__main__":
    b = namedtuple("broker", ('ip', 'port'))
    broker = b("192.168.86.23", 1883)
    x = TestMqtt(broker)
    while True:
        sleep(1)
