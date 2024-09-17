import time

import paho.mqtt.client as mqtt
import json

from glados_modules.MqttClient import TargetMessageBuilder
from glados_modules.GLaDosEnums import TrackingEnums


class MQTTClient:
    def __init__(self, broker, port, topic):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client()
        #self.client.on_connect = self.on_connect

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        json_message = json.dumps({"key": "value", "number": 123})
        self.client.publish(self.topic, json_message)

    def connect(self):
        self.client.connect(self.broker, self.port, 60)
        #self.client.loop_forever()


if __name__ == "__main__":
    broker = "192.168.86.52"
    port = 1883
    topic = TrackingEnums.MQTT_COMMAND_TOPIC.value

    mqtt_client = MQTTClient(broker, port, topic)
    mqtt_client.connect()
    msglist = [TargetMessageBuilder.send_track_command_start()]
    for m in msglist:
        print(f"sending {m}")
        mqtt_client.client.publish(topic, json.dumps(m))
