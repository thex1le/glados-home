import time

import paho.mqtt.client as mqtt
import json

from glados_modules.MqttClient import ServoMessageBuilder, MQTTClient

if __name__ == "__main__":
    broker = "192.168.86.23"
    port = 1883
    topic = "body/servo"

    mqtt_client = MQTTClient(broker, port)
    # low neck 52
    #msglist = [ServoMessageBuilder.body_left_right(angle=145, speed=1)]
    #msglist.extend([ServoMessageBuilder.head_left_right(angle=92, speed=1)])
    msglist = [ServoMessageBuilder.head_up_down(angle=6, speed=1)]
    #msglist.append(ServoMessageBuilder.body_up_down(angle=180, speed=1))
    #                                                                    "speed": 1}, {"servo": "head_up_down", "angle": 180, "speed": 1}, {"servo": "head_left_right", "angle":180, "speed": 1}]
    mqtt_client.send_command(msglist, topic)
    #time.sleep(10)
    #msglist = [{"servo": "body_left_right", "angle": 0, "speed": 1}, {"servo": "body_up_down", "angle": 0,
    #           "speed": 1}, {"servo": "head_up_down", "angle": 0, "speed": 1}, {"servo": "head_left_right", "angle":0, "speed": 1}]
    #for m in msglist:
    #    print(f"sending {m}")
    #    mqtt_client.client.publish(topic, json.dumps(m))
