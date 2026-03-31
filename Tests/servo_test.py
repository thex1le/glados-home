"""Servo range-of-motion test via MQTT.

Moves each servo through min -> center -> max -> center, one at a time.
Prints which servo is being tested and its current target.

Usage:
    # Test all servos
    python3 Tests/servo_test.py -config glog.conf

    # Test a single servo
    python3 Tests/servo_test.py -config glog.conf --servo body_left_right

    # Test multiple servos
    python3 Tests/servo_test.py -config glog.conf --servo body_left_right head_up_down

    # Faster or slower movement (speed 1-5, default 2)
    python3 Tests/servo_test.py -config glog.conf --speed 1

    # Custom hold time at each position (seconds, default 2)
    python3 Tests/servo_test.py -config glog.conf --hold 3

Servo numbers:
    0 = body_left_right (GS3508MG)
    1 = body_up_down (MG92B)
    2 = head_left_right (MG92B)
    3 = head_up_down (MG90D)
"""

# builtin
import argparse
import sys
import os
import time
from configparser import ConfigParser
from json import dumps

# 3rd party
import paho.mqtt.client as mqtt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from glados_modules.GladosEnums import ServoEnum, SystemEnums


ALL_SERVOS = [
    ServoEnum.LOCATION_BODY_LEFT_RIGHT.value,
    ServoEnum.LOCATION_BODY_UP_DOWN.value,
    ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value,
    ServoEnum.LOCATION_HEAD_UP_DOWN.value,
]

SERVO_LABELS = {
    ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: "0: Body Left/Right (GS3508MG)",
    ServoEnum.LOCATION_BODY_UP_DOWN.value: "1: Body Up/Down (MG92B)",
    ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: "2: Head Left/Right (MG92B)",
    ServoEnum.LOCATION_HEAD_UP_DOWN.value: "3: Head Up/Down (MG90D)",
}

# Map channel numbers to servo names
SERVO_BY_NUMBER = {str(i): name for i, name in enumerate(ALL_SERVOS)}


def parse_csv(raw: str) -> list:
    """Parse comma-separated config value, stripping inline comments."""
    return [v.split('#')[0].strip() for v in raw.split(',')]


def get_servo_config(config: ConfigParser) -> dict:
    """Build servo min/max/center from config file."""
    head_vals = parse_csv(config.get(ServoEnum.CONFIG_HEAD.value, ServoEnum.HEAD_MIN_MAX_CENTER.value))
    neck_vals = parse_csv(config.get(ServoEnum.CONFIG_HEAD.value, ServoEnum.NECK_MIN_MAX_CENTER.value))
    default_vals = parse_csv(config.get(ServoEnum.CONFIG_HEAD.value, ServoEnum.DEFAULT_MAX_MIN_CENTER.value))

    body_lr = ServoEnum.LOCATION_BODY_LEFT_RIGHT.value
    body_ud = ServoEnum.LOCATION_BODY_UP_DOWN.value
    head_lr = ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
    head_ud = ServoEnum.LOCATION_HEAD_UP_DOWN.value

    return {
        body_lr: {"max": float(default_vals[0]), "min": float(default_vals[1]), "center": float(default_vals[2])},
        body_ud: {"max": float(neck_vals[0]), "min": float(neck_vals[1]), "center": float(neck_vals[2])},
        head_lr: {"max": float(head_vals[0]), "min": float(head_vals[1]), "center": float(head_vals[2])},
        head_ud: {"max": float(head_vals[0]), "min": float(head_vals[1]), "center": float(head_vals[2])},
    }


def send_move(client: mqtt.Client, servo_name: str, angle: float, speed: int) -> None:
    """Send a single servo move command via MQTT."""
    msg = {
        ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_MOVE.value,
        ServoEnum.MSG_LOCATION_KEY.value: servo_name,
        ServoEnum.MSG_ANGLE.value: round(angle),
        ServoEnum.MSG_SPEED.value: speed,
        "uuid": f"servo_test_{time.time()}",
    }
    client.publish(ServoEnum.MQTT_COMMAND_TOPIC.value, dumps(msg))


def center_all(client: mqtt.Client, servo_config: dict, speed: int) -> None:
    """Send all servos to their center position."""
    targets = {}
    for name, cfg in servo_config.items():
        targets[name] = {
            ServoEnum.MSG_ANGLE.value: round(cfg["center"]),
            ServoEnum.MSG_SPEED.value: speed,
        }
    msg = {
        ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_MOVE_ALL.value,
        ServoEnum.MSG_TARGETS.value: targets,
        "uuid": f"servo_test_{time.time()}",
    }
    client.publish(ServoEnum.MQTT_COMMAND_TOPIC.value, dumps(msg))


def run_servo_test(client: mqtt.Client, name: str, cfg: dict, speed: int, hold: float) -> None:
    """Run a single servo through its range: min -> center -> max -> center."""
    label = SERVO_LABELS.get(name, name)
    print(f"\n{'=' * 60}")
    print(f"Testing: {label}")
    print(f"  Range: {cfg['min']:.0f} -> {cfg['center']:.0f} -> {cfg['max']:.0f}")
    print(f"  Speed: {speed}  Hold: {hold}s")
    print(f"{'=' * 60}")

    steps = [
        ("center", cfg["center"]),
        ("minimum", cfg["min"]),
        ("center", cfg["center"]),
        ("maximum", cfg["max"]),
        ("center", cfg["center"]),
    ]

    for desc, angle in steps:
        print(f"  -> Moving to {desc}: {angle:.0f}°", end="", flush=True)
        send_move(client, name, angle, speed)
        time.sleep(hold)
        print(" ✓")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test servo range of motion via MQTT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available servos: " + ", ".join(ALL_SERVOS),
    )
    parser.add_argument("-config", required=True, help="Path to glog.conf")
    parser.add_argument("--servo", nargs="+", default=None,
                        help="Servo(s) to test by number (0-3) or name (default: all)")
    parser.add_argument("--speed", type=int, default=2,
                        help="Movement speed 1-5 (default: 2)")
    parser.add_argument("--hold", type=float, default=2.0,
                        help="Seconds to hold each position (default: 2)")
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)
    broker_ip = config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_SERVER_IP.value)
    broker_port = int(config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_PORT.value))
    servo_config = get_servo_config(config)

    # Resolve servo selection (accept numbers or names)
    if args.servo:
        servos_to_test = []
        for s in args.servo:
            if s in SERVO_BY_NUMBER:
                servos_to_test.append(SERVO_BY_NUMBER[s])
            elif s in ALL_SERVOS:
                servos_to_test.append(s)
            else:
                print(f"ERROR: Unknown servo '{s}'")
                print(f"Use numbers 0-3 or names: {', '.join(ALL_SERVOS)}")
                sys.exit(1)
    else:
        servos_to_test = ALL_SERVOS

    speed = max(1, min(5, args.speed))

    # Connect to MQTT
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(broker_ip, broker_port)
    client.loop_start()
    print(f"Connected to MQTT broker at {broker_ip}:{broker_port}")

    print(f"\nTesting {len(servos_to_test)} servo(s): {', '.join(servos_to_test)}")
    print(f"Speed: {speed}  Hold: {args.hold}s per position")

    try:
        # Center all first
        print("\nCentering all servos...")
        center_all(client, servo_config, speed)
        time.sleep(2.0)

        # Test each selected servo
        for name in servos_to_test:
            run_servo_test(client, name, servo_config[name], speed, args.hold)

        # Return to center
        print("\nReturning all servos to center...")
        center_all(client, servo_config, speed)
        time.sleep(1.0)
        print("\nDone.")

    except KeyboardInterrupt:
        print("\n\nInterrupted -- centering all servos...")
        center_all(client, servo_config, speed)
        time.sleep(1.0)
        print("Stopped.")

    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
