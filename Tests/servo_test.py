"""Servo range-of-motion test.

Moves each servo through min -> center -> max -> center, one at a time.
Supports two modes:
  - MQTT mode (default): sends commands to BodyServer via MQTT
  - Direct mode (--direct): drives servos over I2C on the local Pi, no BodyServer needed

Usage:
    # MQTT mode - run from GPU server while BodyServer is running on Pi4
    python3 Tests/servo_test.py -config glog.conf
    python3 Tests/servo_test.py -config glog.conf --servo 0 2

    # Direct mode - run on Pi4, no BodyServer needed
    python3 Tests/servo_test.py -config glog.conf --direct
    python3 Tests/servo_test.py -config glog.conf --direct --servo 1

    # Options
    python3 Tests/servo_test.py -config glog.conf --speed 1 --hold 3

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

# Pulse width ranges per motor type (from glog.conf)
PULSE_KEYS = {
    ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: ServoEnum.SERVO_GS3508MG_PULSE.value,
    ServoEnum.LOCATION_BODY_UP_DOWN.value: ServoEnum.SERVO_MG92B_PULSE.value,
    ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: ServoEnum.SERVO_MG92B_PULSE.value,
    ServoEnum.LOCATION_HEAD_UP_DOWN.value: ServoEnum.SERVO_MG90D_PULSE.value,
}


def parse_csv(raw: str) -> list:
    """Parse comma-separated config value, stripping inline comments."""
    return [v.split('#')[0].strip() for v in raw.split(',')]


def get_servo_config(config: ConfigParser) -> dict:
    """Build servo min/max/center and board assignment from config file."""
    sech = config[ServoEnum.CONFIG_HEAD.value]
    head_vals = parse_csv(sech.get(ServoEnum.HEAD_MIN_MAX_CENTER.value))
    neck_vals = parse_csv(sech.get(ServoEnum.NECK_MIN_MAX_CENTER.value))
    default_vals = parse_csv(sech.get(ServoEnum.DEFAULT_MAX_MIN_CENTER.value))

    # Parse board addresses
    pca_addrs = [a.strip() for a in sech.get(
        ServoEnum.PCA9685_ADDRESSES.value, ServoEnum.PCA9685_DEFAULT_ADDRESSES.value).split(',')]

    # Parse per-servo board,channel assignments
    board_channel_keys = {
        ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: (ServoEnum.BODY_LR_BOARD_CHANNEL.value, "0,0"),
        ServoEnum.LOCATION_BODY_UP_DOWN.value: (ServoEnum.BODY_UD_BOARD_CHANNEL.value, "0,1"),
        ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: (ServoEnum.HEAD_LR_BOARD_CHANNEL.value, "0,2"),
        ServoEnum.LOCATION_HEAD_UP_DOWN.value: (ServoEnum.HEAD_UD_BOARD_CHANNEL.value, "0,3"),
    }

    def get_board_info(servo_name: str) -> dict:
        key, fallback = board_channel_keys[servo_name]
        raw = sech.get(key, fallback)
        parts = raw.split(',')
        board_idx = int(parts[0])
        channel = int(parts[1])
        addr = pca_addrs[board_idx] if board_idx < len(pca_addrs) else "0x40"
        return {"board": board_idx, "channel": channel, "address": addr}

    # Parse pulse widths for direct mode
    pulse_ranges = {}
    for servo_name, pulse_key in PULSE_KEYS.items():
        pulse_vals = parse_csv(sech.get(pulse_key, "2665,610"))
        pulse_ranges[servo_name] = (int(pulse_vals[0]), int(pulse_vals[1]))

    body_lr = ServoEnum.LOCATION_BODY_LEFT_RIGHT.value
    body_ud = ServoEnum.LOCATION_BODY_UP_DOWN.value
    head_lr = ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
    head_ud = ServoEnum.LOCATION_HEAD_UP_DOWN.value

    return {
        body_lr: {"max": float(default_vals[0]), "min": float(default_vals[1]), "center": float(default_vals[2]),
                  "pulse": pulse_ranges[body_lr], **get_board_info(body_lr)},
        body_ud: {"max": float(neck_vals[0]), "min": float(neck_vals[1]), "center": float(neck_vals[2]),
                  "pulse": pulse_ranges[body_ud], **get_board_info(body_ud)},
        head_lr: {"max": float(head_vals[0]), "min": float(head_vals[1]), "center": float(head_vals[2]),
                  "pulse": pulse_ranges[head_lr], **get_board_info(head_lr)},
        head_ud: {"max": float(head_vals[0]), "min": float(head_vals[1]), "center": float(head_vals[2]),
                  "pulse": pulse_ranges[head_ud], **get_board_info(head_ud)},
    }


# --- MQTT mode functions ---

def mqtt_send_move(client, servo_name: str, angle: float, speed: int) -> None:
    """Send a single servo move command via MQTT."""
    msg = {
        ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_MOVE.value,
        ServoEnum.MSG_LOCATION_KEY.value: servo_name,
        ServoEnum.MSG_ANGLE.value: round(angle),
        ServoEnum.MSG_SPEED.value: speed,
        "uuid": f"servo_test_{time.time()}",
    }
    client.publish(ServoEnum.MQTT_COMMAND_TOPIC.value, dumps(msg))


def mqtt_center_all(client, servo_config: dict, speed: int) -> None:
    """Send all servos to their center position via MQTT."""
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


# --- Direct mode functions ---

def direct_init_kits(servo_config: dict) -> dict:
    """Create ServoKit instances for each unique board address."""
    from adafruit_servokit import ServoKit

    # Collect unique addresses
    addresses = {}
    for name, cfg in servo_config.items():
        addr = int(cfg["address"], 0)
        if addr not in addresses:
            addresses[addr] = None

    # Create a ServoKit for each
    kits = {}
    for addr in addresses:
        print(f"  Initializing PCA9685 board at {hex(addr)}")
        kits[addr] = ServoKit(channels=16, address=addr)

    # Set pulse width ranges for each servo
    for name, cfg in servo_config.items():
        addr = int(cfg["address"], 0)
        channel = cfg["channel"]
        pulse_max, pulse_min = cfg["pulse"]
        kits[addr].servo[channel].set_pulse_width_range(min_pulse=pulse_min, max_pulse=pulse_max)
        print(f"  {SERVO_LABELS.get(name, name)}  ->  {hex(addr)} ch {channel}  "
              f"pulse {pulse_min}-{pulse_max}us")

    return kits


def direct_move(kits: dict, servo_name: str, cfg: dict, angle: float) -> None:
    """Move a servo directly via I2C."""
    addr = int(cfg["address"], 0)
    channel = cfg["channel"]
    clamped = max(cfg["min"], min(cfg["max"], angle))
    kits[addr].servo[channel].angle = clamped


def direct_center_all(kits: dict, servo_config: dict) -> None:
    """Center all servos directly via I2C."""
    for name, cfg in servo_config.items():
        direct_move(kits, name, cfg, cfg["center"])


# --- Shared test runner ---

def run_servo_test(move_fn, name: str, cfg: dict, hold: float) -> None:
    """Run a single servo through its range: min -> center -> max -> center."""
    label = SERVO_LABELS.get(name, name)
    print(f"\n{'=' * 60}")
    print(f"Testing: {label}")
    print(f"  Board: {cfg['board']} (address {cfg['address']})  Channel: {cfg['channel']}")
    print(f"  Range: {cfg['min']:.0f} -> {cfg['center']:.0f} -> {cfg['max']:.0f}")
    print(f"  Pulse: {cfg['pulse'][1]}-{cfg['pulse'][0]}us")
    print(f"  Hold: {hold}s")
    print(f"{'=' * 60}")

    steps = [
        ("center", cfg["center"]),
        ("minimum", cfg["min"]),
        ("center", cfg["center"]),
        ("maximum", cfg["max"]),
        ("center", cfg["center"]),
    ]

    for desc, angle in steps:
        print(f"  -> Moving to {desc}: {angle:.0f}\u00b0", end="", flush=True)
        move_fn(name, angle)
        time.sleep(hold)
        print(" done")


def resolve_servos(args_servo: list) -> list:
    """Resolve servo selection from CLI args (numbers or names)."""
    if not args_servo:
        return list(ALL_SERVOS)
    servos = []
    for s in args_servo:
        if s in SERVO_BY_NUMBER:
            servos.append(SERVO_BY_NUMBER[s])
        elif s in ALL_SERVOS:
            servos.append(s)
        else:
            print(f"ERROR: Unknown servo '{s}'")
            print(f"Use numbers 0-3 or names: {', '.join(ALL_SERVOS)}")
            sys.exit(1)
    return servos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test servo range of motion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Servos: 0=body_LR  1=body_UD  2=head_LR  3=head_UD",
    )
    parser.add_argument("-config", required=True, help="Path to glog.conf")
    parser.add_argument("--servo", nargs="+", default=None,
                        help="Servo(s) to test by number (0-3) or name (default: all)")
    parser.add_argument("--speed", type=int, default=2,
                        help="Movement speed 1-5, MQTT mode only (default: 2)")
    parser.add_argument("--hold", type=float, default=2.0,
                        help="Seconds to hold each position (default: 2)")
    parser.add_argument("--direct", action="store_true",
                        help="Direct I2C mode -- drive servos locally, no MQTT/BodyServer needed")
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)
    servo_config = get_servo_config(config)
    servos_to_test = resolve_servos(args.servo)
    speed = max(1, min(5, args.speed))

    # Print servo summary
    mode = "DIRECT (I2C)" if args.direct else "MQTT"
    print(f"\nMode: {mode}")
    print(f"Testing {len(servos_to_test)} servo(s):")
    for name in servos_to_test:
        cfg = servo_config[name]
        label = SERVO_LABELS.get(name, name)
        print(f"  {label}  ->  board {cfg['board']} ({cfg['address']}) ch {cfg['channel']}  "
              f"range [{cfg['min']:.0f}-{cfg['max']:.0f}]  pulse {cfg['pulse'][1]}-{cfg['pulse'][0]}us")
    print(f"Hold: {args.hold}s per position")

    if args.direct:
        # Direct I2C mode
        print("\nInitializing I2C boards...")
        kits = direct_init_kits(servo_config)

        def move_fn(name, angle):
            direct_move(kits, name, servo_config[name], angle)

        def center_fn():
            direct_center_all(kits, servo_config)

        try:
            print("\nCentering all servos...")
            center_fn()
            time.sleep(1.0)

            for name in servos_to_test:
                run_servo_test(move_fn, name, servo_config[name], args.hold)

            print("\nReturning all servos to center...")
            center_fn()
            time.sleep(1.0)
            print("\nDone.")

        except KeyboardInterrupt:
            print("\n\nInterrupted -- centering all servos...")
            center_fn()
            time.sleep(1.0)
            print("Stopped.")

    else:
        # MQTT mode
        import paho.mqtt.client as mqtt
        broker_ip = config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_SERVER_IP.value)
        broker_port = int(config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_PORT.value))

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect(broker_ip, broker_port)
        client.loop_start()
        print(f"\nConnected to MQTT broker at {broker_ip}:{broker_port}")
        print(f"Speed: {speed}")

        def move_fn(name, angle):
            mqtt_send_move(client, name, angle, speed)

        def center_fn():
            mqtt_center_all(client, servo_config, speed)

        try:
            print("\nCentering all servos...")
            center_fn()
            time.sleep(2.0)

            for name in servos_to_test:
                run_servo_test(move_fn, name, servo_config[name], args.hold)

            print("\nReturning all servos to center...")
            center_fn()
            time.sleep(1.0)
            print("\nDone.")

        except KeyboardInterrupt:
            print("\n\nInterrupted -- centering all servos...")
            center_fn()
            time.sleep(1.0)
            print("Stopped.")

        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
