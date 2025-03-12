import time

import paho.mqtt.client as mqtt
import json

from glados_modules.MqttClient import ServoMessageBuilder, MQTTClient

if __name__ == "__main__":
    broker = "192.168.1.39"
    port = 1883
    topic = "body/servo"

    mqtt_client = MQTTClient(broker, port)
    # low neck 52
    msglist = [ServoMessageBuilder.body_left_right(angle=90, speed=1)]
    msglist.append(ServoMessageBuilder.body_up_down(angle=90, speed=1))
    msglist.append(ServoMessageBuilder.head_left_right(angle=90, speed=1))
    msglist.append(ServoMessageBuilder.head_up_down(angle=83, speed=1))
    mqtt_client.send_command(msglist, topic)
