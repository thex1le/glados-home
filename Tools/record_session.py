#!/usr/bin/env python3
"""Start/stop a recording session via MQTT.

Starts both video recording (JPEG frames) and motion recording (tracking
math JSONL) on the GPU server. All three systems must be online.

Usage:
    # Start recording, walk around, press Enter to stop
    python3 Tools/record_session.py --broker 192.168.1.2 --session golden_walk

    # Start recording with auto-stop after 30 seconds
    python3 Tools/record_session.py --broker 192.168.1.2 --session golden_walk --duration 30

    # Just stop any active recording
    python3 Tools/record_session.py --broker 192.168.1.2 --stop
"""

import argparse
import json
import sys
import time
from uuid import uuid4

import paho.mqtt.client as mqtt


def main() -> None:
    parser = argparse.ArgumentParser(description="Start/stop GLaDOS recording session")
    parser.add_argument("--broker", required=True, help="MQTT broker IP address")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--session", default=None,
                        help="Session name (auto-generated if omitted)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Auto-stop after N seconds (default: manual stop)")
    parser.add_argument("--stop", action="store_true",
                        help="Just send a stop command")
    args = parser.parse_args()

    topic = "system/recording"

    client = mqtt.Client()
    try:
        client.connect(args.broker, args.port)
    except Exception as e:
        print(f"Cannot connect to MQTT broker at {args.broker}:{args.port}: {e}")
        sys.exit(1)

    client.loop_start()

    if args.stop:
        client.publish(topic, json.dumps({"cmd": "stop_recording", "uuid": str(uuid4())}))
        print("Stop command sent.")
        client.loop_stop()
        client.disconnect()
        return

    # Start recording
    start_msg = {"cmd": "start_recording", "uuid": str(uuid4())}
    if args.session:
        start_msg["session"] = args.session
    client.publish(topic, json.dumps(start_msg))

    session_label = args.session or "(auto-named)"
    print(f"Recording started: {session_label}")
    print(f"  Video frames -> recordings/<session>/")
    print(f"  Motion data  -> recordings/<session>.jsonl")

    if args.duration:
        print(f"\nAuto-stopping in {args.duration} seconds...")
        try:
            for remaining in range(args.duration, 0, -1):
                print(f"  {remaining}s remaining", end="\r")
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        print()
    else:
        print("\nPress Enter to stop recording (or Ctrl+C)...")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            print()

    client.publish(topic, json.dumps({"cmd": "stop_recording", "uuid": str(uuid4())}))
    print("Recording stopped.")
    print(f"\nCopy to dev machine:")
    print(f"  scp gpu-server:~/glados-home/recordings/{args.session or '*'}* ./recordings/")

    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
