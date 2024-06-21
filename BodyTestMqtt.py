import paho.mqtt.client as mqtt
import json


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
        self.client.loop_forever()


if __name__ == "__main__":
    broker = "192.168.86.52"
    port = 1883
    topic = "body/servo"

    mqtt_client = MQTTClient(broker, port, topic)
    mqtt_client.connect()
    msglist = [{"servo": "body_left_right", "angle": 90, "speed": 10}]
    for m in msglist:
        mqtt_client.client.publish(json.dumps(m))
