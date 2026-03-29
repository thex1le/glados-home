"""Standalone MQTT session recorder.

Taps into the MQTT broker and records all vision, servo, and tracking
messages with timestamps. This captures the raw data stream for full
session replay without needing any cameras or servos.

Usage:
    python Tests/mqtt_session_recorder.py --broker 192.168.1.39 --port 1883
    python Tests/mqtt_session_recorder.py --broker 192.168.1.39 --output recordings/test_session.jsonl
    # Press Ctrl+C to stop recording
"""

import json
import time
import argparse
import os
import sys
from datetime import datetime

import paho.mqtt.client as mqtt

# Topics to record
TOPICS = [
    "vision/camera/results",       # YOLO detection results
    "body/servo",                  # Servo commands (move, move_all)
    "body/servo/status",           # Servo status (position, velocity)
    "body/imu/status",             # IMU sensor data
    "system/track",                # Tracking commands
    "body/tof/status",             # TOF distance
    "body/th/status",              # Temperature/humidity
    "intensity",                   # System intensity
]


class MQTTSessionRecorder:
    def __init__(self, broker: str, port: int, output_path: str):
        self.broker = broker
        self.port = port
        self.output_path = output_path
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.start_time = time.time()
        self.message_count = 0

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        self.file = open(output_path, 'w')

        # Write header
        header = {
            "type": "mqtt_session_header",
            "broker": broker,
            "port": port,
            "topics": TOPICS,
            "start_time": self.start_time,
            "start_iso": datetime.now().isoformat(),
        }
        self.file.write(json.dumps(header) + "\n")
        self.file.flush()

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to {self.broker}:{self.port} (rc={rc})")
        for topic in TOPICS:
            client.subscribe(topic, qos=1)
            print(f"  Subscribed: {topic}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = msg.payload.decode(errors='replace')

        record = {
            "type": "mqtt_message",
            "topic": msg.topic,
            "payload": payload,
            "wall_time": time.time(),
            "session_time": time.time() - self.start_time,
            "qos": msg.qos,
        }
        self.file.write(json.dumps(record) + "\n")
        self.message_count += 1

        if self.message_count % 100 == 0:
            self.file.flush()
            elapsed = time.time() - self.start_time
            rate = self.message_count / elapsed if elapsed > 0 else 0
            print(f"  {self.message_count} messages recorded ({rate:.1f} msg/s)")

    def start(self):
        print(f"Recording MQTT session to: {self.output_path}")
        print("Press Ctrl+C to stop\n")
        self.client.connect(self.broker, self.port, 60)
        try:
            self.client.loop_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        elapsed = time.time() - self.start_time
        footer = {
            "type": "mqtt_session_footer",
            "total_messages": self.message_count,
            "duration_seconds": round(elapsed, 2),
            "avg_rate": round(self.message_count / elapsed, 2) if elapsed > 0 else 0,
        }
        self.file.write(json.dumps(footer) + "\n")
        self.file.flush()
        self.file.close()
        self.client.disconnect()
        print(f"\nRecording stopped: {self.message_count} messages in {elapsed:.1f}s")
        print(f"Saved to: {self.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record MQTT session for replay testing")
    parser.add_argument("--broker", default="192.168.1.39", help="MQTT broker IP")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"recordings/mqtt_session_{ts}.jsonl"

    recorder = MQTTSessionRecorder(args.broker, args.port, args.output)
    recorder.start()
