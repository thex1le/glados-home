"""Minimal IMU publisher -- runs on Pi5 without the full GLaDOS stack.

Starts just the IMU sensor and publishes to MQTT at 10Hz.
Use this when you need IMU data but don't want to start cameras, voice, etc.

Usage:
    python3 Tests/imu_only.py -config dont_commit_glados.conf
"""

# builtin
import argparse
import sys
import os
from configparser import ConfigParser
from collections import namedtuple
from time import sleep

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# glados imports
from glados_modules.BodyControlModules import IMU
from glados_modules.GladosEnums import SystemEnums


def main(config_path: str) -> None:
    config = ConfigParser()
    config.read(config_path)

    broker_ip = config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_SERVER_IP.value)
    broker_port = int(config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_PORT.value))

    broker = namedtuple("broker", ["ip", "port"])(broker_ip, broker_port)

    print(f"Starting IMU publisher -> MQTT broker at {broker_ip}:{broker_port}")
    imu = IMU(broker=broker)
    imu.daemon = True
    imu.start()
    print("IMU running. Ctrl+C to stop.")

    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal IMU-only publisher for Pi5")
    parser.add_argument("-config", required=True, help="Path to glog.conf")
    args = parser.parse_args()
    main(args.config)
