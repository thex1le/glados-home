import time

import paho.mqtt.client as mqtt
import json

from glados_modules.MqttClient import ServoMessageBuilder, MQTTClient

if __name__ == "__main__":
    broker = "192.168.86.23"
    port = 1883
    topic = "body/servo"

    mqtt_client = MQTTClient(broker, port)
    #msglist = [{"servo": "body_up_down", "angle":0, "speed": 1}]
    # low neck 52
    #msglist = [ServoMessageBuilder.body_left_right(angle=30, speed=1)]
    #msglist.extend([ServoMessageBuilder.head_left_right(angle=60, speed=1)])
    #msglist = list()
    msglist = [{'cmd': 'move', 'servo': 'head_left_right', 'angle': 83, 'speed': 1, 'uuid': '171309dc-0d2f-43f9-b525-41c314b78284'},]
    #msglist.extend([ServoMessageBuilder.head_up_down(angle=50, speed=1)])
    msglist.append({'cmd': 'move', "servo": "body_left_right", "angle": 131, "speed": 1})
    #msglist = [{"servo": "head_up_down", "angle": 6, "speed": 1}]
    #msglist = [{"servo": "body_left_right", "angle": 180, "speed": 1}, {"servo": "body_up_down", "angle": 180,
    #                                                                    "speed": 1}, {"servo": "head_up_down", "angle": 180, "speed": 1}, {"servo": "head_left_right", "angle":180, "speed": 1}]
    mqtt_client.send_command(msglist, topic)
    #time.sleep(10)
    #msglist = [{"servo": "body_left_right", "angle": 0, "speed": 1}, {"servo": "body_up_down", "angle": 0,
    #           "speed": 1}, {"servo": "head_up_down", "angle": 0, "speed": 1}, {"servo": "head_left_right", "angle":0, "speed": 1}]
    #for m in msglist:
    #    print(f"sending {m}")
    #    mqtt_client.client.publish(topic, json.dumps(m))
