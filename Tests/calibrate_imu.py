"""IMU auto-calibration using servo movements.

Moves the servos through a sequence of positions to give the BNO055
enough orientation variety to calibrate its accel, gyro, and magnetometer.
Saves the calibration offsets to imu_calibration.json once complete.

Two modes:
  - MQTT mode (default): sends servo commands via MQTT, reads IMU via MQTT.
    Run from GPU server while BodyServer (Pi4) and GLaDOS/imu_only (Pi5) are running.
  - Direct mode (--direct): drives servos over I2C and reads IMU directly.
    Run on the Pi4/Pi5 with no other services needed.

Usage:
    # MQTT mode (from GPU server)
    python3 Tests/calibrate_imu.py -config glog.conf

    # Direct mode (on Pi4, with IMU on local I2C)
    python3 Tests/calibrate_imu.py -config glog.conf --direct
"""

# builtin
import argparse
import os
import sys
import time
from configparser import ConfigParser
from json import loads, dumps
from threading import Lock, Thread

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from glados_modules.GladosEnums import (
    ServoEnum, IMUEnums, MQTTEnums, SystemEnums
)


def parse_csv(raw: str) -> list:
    """Parse comma-separated config value, stripping inline comments."""
    return [v.split('#')[0].strip() for v in raw.split(',')]


def get_servo_ranges(config: ConfigParser) -> tuple:
    """Parse servo middles/mins/maxs and board config from config file."""
    sech = config[ServoEnum.CONFIG_HEAD.value]
    head_vals = parse_csv(sech.get(ServoEnum.HEAD_MIN_MAX_CENTER.value))
    neck_vals = parse_csv(sech.get(ServoEnum.NECK_MIN_MAX_CENTER.value))
    default_vals = parse_csv(sech.get(ServoEnum.DEFAULT_MAX_MIN_CENTER.value))

    body_lr = ServoEnum.LOCATION_BODY_LEFT_RIGHT.value
    body_ud = ServoEnum.LOCATION_BODY_UP_DOWN.value
    head_lr = ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
    head_ud = ServoEnum.LOCATION_HEAD_UP_DOWN.value

    middles = {
        body_lr: float(default_vals[2]), body_ud: float(neck_vals[2]),
        head_lr: float(head_vals[2]), head_ud: float(head_vals[2]),
    }
    mins = {
        body_lr: float(default_vals[1]), body_ud: float(neck_vals[1]),
        head_lr: float(head_vals[1]), head_ud: float(head_vals[1]),
    }
    maxs = {
        body_lr: float(default_vals[0]), body_ud: float(neck_vals[0]),
        head_lr: float(head_vals[0]), head_ud: float(head_vals[0]),
    }

    # Board config for direct mode
    pca_addrs = [a.strip() for a in sech.get(
        ServoEnum.PCA9685_ADDRESSES.value, ServoEnum.PCA9685_DEFAULT_ADDRESSES.value).split(',')]

    board_channel_keys = {
        body_lr: (ServoEnum.BODY_LR_BOARD_CHANNEL.value, "0,0"),
        body_ud: (ServoEnum.BODY_UD_BOARD_CHANNEL.value, "0,1"),
        head_lr: (ServoEnum.HEAD_LR_BOARD_CHANNEL.value, "0,2"),
        head_ud: (ServoEnum.HEAD_UD_BOARD_CHANNEL.value, "0,3"),
    }

    board_map = {}
    for servo_name, (key, fallback) in board_channel_keys.items():
        raw = sech.get(key, fallback)
        parts = raw.split(',')
        board_idx = int(parts[0])
        channel = int(parts[1])
        addr = pca_addrs[board_idx] if board_idx < len(pca_addrs) else "0x40"
        board_map[servo_name] = {"board": board_idx, "channel": channel, "address": addr}

    # Pulse widths for direct mode
    pulse_keys = {
        body_lr: ServoEnum.SERVO_GS3508MG_PULSE.value,
        body_ud: ServoEnum.SERVO_MG92B_PULSE.value,
        head_lr: ServoEnum.SERVO_MG92B_PULSE.value,
        head_ud: ServoEnum.SERVO_MG90D_PULSE.value,
    }
    pulse_map = {}
    for servo_name, pulse_key in pulse_keys.items():
        pv = parse_csv(sech.get(pulse_key, "2665,610"))
        pulse_map[servo_name] = (int(pv[0]), int(pv[1]))

    return middles, mins, maxs, board_map, pulse_map


# --- MQTT backend ---

class MQTTBackend:
    """Controls servos via MQTT, reads IMU via MQTT."""

    def __init__(self, broker_ip: str, broker_port: int) -> None:
        import paho.mqtt.client as mqtt
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._broker_ip = broker_ip
        self._broker_port = broker_port
        self._imu_data: dict = {}
        self._lock = Lock()
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def start(self) -> None:
        self._client.connect(self._broker_ip, self._broker_port)
        self._client.loop_start()
        print(f"Connected to MQTT broker at {self._broker_ip}:{self._broker_port}")

    def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        client.subscribe(MQTTEnums.IMU_STATUS_TOPIC.value)

    def _on_message(self, client, userdata, msg) -> None:
        try:
            payload = loads(msg.payload.decode())
        except Exception:
            return
        if msg.topic == MQTTEnums.IMU_STATUS_TOPIC.value:
            imu_status = payload.get(IMUEnums.IMU_STATUS_KEY.value, {})
            if imu_status:
                with self._lock:
                    self._imu_data = imu_status

    def get_euler(self) -> tuple:
        with self._lock:
            euler = self._imu_data.get(IMUEnums.EULER_KEY.value, (0, 0, 0))
            return tuple(euler) if euler else (0, 0, 0)

    def is_euler_live(self) -> bool:
        euler = self.get_euler()
        return euler is not None and any(abs(v) > 0.1 for v in euler)

    def move_all(self, angles: dict, speed: int = 2) -> None:
        targets = {}
        for name, angle in angles.items():
            targets[name] = {
                ServoEnum.MSG_ANGLE.value: round(angle),
                ServoEnum.MSG_SPEED.value: speed,
            }
        msg = {
            ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_MOVE_ALL.value,
            ServoEnum.MSG_TARGETS.value: targets,
            "uuid": f"imu_cal_{time.time()}",
        }
        self._client.publish(ServoEnum.MQTT_COMMAND_TOPIC.value, dumps(msg))

    def wait_for_imu(self, timeout: float = 10.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if self._imu_data:
                    return True
            time.sleep(0.2)
        return False

    def get_calibration_status(self) -> str:
        return "LIVE" if self.is_euler_live() else "waiting..."

    def save_calibration(self) -> None:
        print("  (Calibration auto-saves via IMU class on Pi5)")


# --- Direct I2C backend ---

class DirectBackend:
    """Drives servos directly over I2C, reads IMU via MQTT from Pi5.

    Run on Pi4 with imu_only.py running on Pi5. No BodyServer needed.
    """

    def __init__(self, broker_ip: str, broker_port: int,
                 board_map: dict, pulse_map: dict) -> None:
        self._board_map = board_map
        self._pulse_map = pulse_map
        self._kits = {}
        # IMU still comes via MQTT from Pi5
        self._mqtt_imu = MQTTBackend(broker_ip, broker_port)

    def start(self) -> None:
        from adafruit_servokit import ServoKit

        # Init PCA9685 boards
        addresses = {}
        for name, info in self._board_map.items():
            addr = int(info["address"], 0)
            addresses[addr] = True
        for addr in addresses:
            print(f"  Initializing PCA9685 board at {hex(addr)}")
            self._kits[addr] = ServoKit(channels=16, address=addr)

        # Set pulse widths
        for name, info in self._board_map.items():
            addr = int(info["address"], 0)
            channel = info["channel"]
            pulse_max, pulse_min = self._pulse_map[name]
            self._kits[addr].servo[channel].set_pulse_width_range(
                min_pulse=pulse_min, max_pulse=pulse_max)
            print(f"  {name}: board {hex(addr)} ch {channel} pulse {pulse_min}-{pulse_max}us")

        # Connect to MQTT for IMU data from Pi5
        print("  Connecting to MQTT for IMU data...")
        self._mqtt_imu.start()

    def stop(self) -> None:
        self._mqtt_imu.stop()

    def get_euler(self) -> tuple:
        return self._mqtt_imu.get_euler()

    def is_euler_live(self) -> bool:
        return self._mqtt_imu.is_euler_live()

    def move_all(self, angles: dict, speed: int = 2) -> None:
        """Move servos directly via I2C. Speed param is ignored (instant move)."""
        for name, angle in angles.items():
            if name in self._board_map:
                info = self._board_map[name]
                addr = int(info["address"], 0)
                channel = info["channel"]
                self._kits[addr].servo[channel].angle = round(angle)

    def wait_for_imu(self, timeout: float = 10.0) -> bool:
        return self._mqtt_imu.wait_for_imu(timeout)

    def get_calibration_status(self) -> str:
        return self._mqtt_imu.get_calibration_status()

    def save_calibration(self) -> None:
        print("  (Calibration auto-saves via IMU class on Pi5)")


# --- Calibration sequence (shared between backends) ---

def run_calibration(backend, middles: dict, mins: dict, maxs: dict) -> None:
    """Run the IMU calibration dance using servo movements."""
    body_lr = ServoEnum.LOCATION_BODY_LEFT_RIGHT.value
    body_ud = ServoEnum.LOCATION_BODY_UP_DOWN.value
    head_lr = ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
    head_ud = ServoEnum.LOCATION_HEAD_UP_DOWN.value

    print("\n=== IMU Auto-Calibration ===\n")

    # Wait for IMU data
    print("Waiting for IMU data...")
    if not backend.wait_for_imu():
        print("ERROR: No IMU data received.")
        return
    euler = backend.get_euler()
    cal_status = backend.get_calibration_status()
    print(f"IMU connected. Euler: ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f})  [{cal_status}]\n")

    # --- Phase 1: Gyro calibration (hold still) ---
    print("--- Phase 1: Gyro calibration (hold still 5s) ---")
    backend.move_all(middles, speed=1)
    time.sleep(3.0)
    print("Holding still...")
    time.sleep(5.0)
    euler = backend.get_euler()
    print(f"  Euler: ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f})  [{backend.get_calibration_status()}]")

    # --- Phase 2: Accel calibration (tilt to different orientations) ---
    print("\n--- Phase 2: Accel calibration (tilting through positions) ---")

    tilt_positions = [
        ("Center",            middles),
        ("Body UD min",       {**middles, body_ud: mins[body_ud] + 5}),
        ("Body UD max",       {**middles, body_ud: maxs[body_ud] - 5}),
        ("Head UD min",       {**middles, head_ud: mins[head_ud] + 10}),
        ("Head UD max",       {**middles, head_ud: maxs[head_ud] - 10}),
        ("Both UD low",       {**middles, body_ud: mins[body_ud] + 5, head_ud: mins[head_ud] + 10}),
        ("Both UD high",      {**middles, body_ud: maxs[body_ud] - 5, head_ud: maxs[head_ud] - 10}),
        ("Head LR + UD low",  {**middles, head_lr: mins[head_lr] + 10, head_ud: mins[head_ud] + 10}),
        ("Head LR + UD high", {**middles, head_lr: maxs[head_lr] - 10, head_ud: maxs[head_ud] - 10}),
    ]

    for i, (desc, angles) in enumerate(tilt_positions):
        print(f"  Position {i + 1}/{len(tilt_positions)}: {desc}")
        backend.move_all(angles, speed=2)
        time.sleep(3.0)
        euler = backend.get_euler()
        print(f"    Euler: ({euler[0]:6.1f}, {euler[1]:5.1f}, {euler[2]:5.1f})  [{backend.get_calibration_status()}]")

    backend.move_all(middles, speed=2)
    time.sleep(2.0)

    # --- Phase 3: Magnetometer calibration (rotate body LR through full range) ---
    print("\n--- Phase 3: Magnetometer calibration (rotating body LR) ---")
    print("  Sweeping body left-right through full range...")

    lr_min = mins[body_lr] + 10
    lr_max = maxs[body_lr] - 10
    lr_steps = 12
    step_size = (lr_max - lr_min) / lr_steps

    for i in range(lr_steps + 1):
        angle = lr_min + i * step_size
        backend.move_all({**middles, body_lr: angle}, speed=3)
        time.sleep(1.5)
        euler = backend.get_euler()
        bar = "#" * (i + 1) + "." * (lr_steps - i)
        print(f"    [{bar}] LR={angle:5.1f}  euler=({euler[0]:6.1f}, {euler[1]:5.1f}, {euler[2]:5.1f})  "
              f"[{backend.get_calibration_status()}]")

    print("  Sweeping back...")
    for i in range(lr_steps, -1, -1):
        angle = lr_min + i * step_size
        backend.move_all({**middles, body_lr: angle}, speed=3)
        time.sleep(1.0)

    backend.move_all(middles, speed=2)
    time.sleep(2.0)

    # --- Phase 4: Combined movements ---
    print("\n--- Phase 4: Combined movements for final calibration ---")

    combo_positions = [
        ("LR left + UD low",    {**middles, body_lr: lr_min, body_ud: mins[body_ud] + 5}),
        ("LR right + UD high",  {**middles, body_lr: lr_max, body_ud: maxs[body_ud] - 5}),
        ("LR left + head tilt", {**middles, body_lr: lr_min, head_ud: maxs[head_ud] - 10}),
        ("LR right + head tilt", {**middles, body_lr: lr_max, head_ud: mins[head_ud] + 10}),
        ("All middle",          middles),
    ]

    for i, (desc, angles) in enumerate(combo_positions):
        print(f"  Position {i + 1}/{len(combo_positions)}: {desc}")
        backend.move_all(angles, speed=2)
        time.sleep(3.0)
        euler = backend.get_euler()
        print(f"    Euler: ({euler[0]:6.1f}, {euler[1]:5.1f}, {euler[2]:5.1f})  [{backend.get_calibration_status()}]")

    # --- Done ---
    print("\n--- Calibration sequence complete ---")
    backend.move_all(middles, speed=1)
    time.sleep(2.0)

    euler = backend.get_euler()
    live = backend.is_euler_live()

    print(f"\nFinal euler: ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f})  [{backend.get_calibration_status()}]")

    if live:
        print("IMU is producing euler data.")
        backend.save_calibration()
    else:
        print("\nWARNING: Euler angles are still zero.")
        print("The BNO055 may need more movement variety. Things to try:")
        print("  - Run this script again (calibration persists across runs)")
        print("  - Power cycle the IMU and try again")
        print("  - Check BNO055 I2C connection")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-calibrate BNO055 IMU using servo movements")
    parser.add_argument("-config", required=True, help="Path to glog.conf")
    parser.add_argument("--direct", action="store_true",
                        help="Direct I2C mode -- drive servos and read IMU locally, no MQTT needed")
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)
    middles, mins, maxs, board_map, pulse_map = get_servo_ranges(config)

    mode = "DIRECT (I2C)" if args.direct else "MQTT"
    print(f"Mode: {mode}")

    broker_ip = config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_SERVER_IP.value)
    broker_port = int(config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_PORT.value))

    if args.direct:
        print("Servos: direct I2C (local)")
        print("IMU: MQTT from Pi5 (make sure imu_only.py is running on Pi5)")
        backend = DirectBackend(broker_ip, broker_port, board_map, pulse_map)
        print("Initializing hardware...")
        backend.start()
    else:
        print("Servos: MQTT -> Pi4 (make sure BodyServer is running)")
        print("IMU: MQTT <- Pi5 (make sure GLaDOS or imu_only.py is running)")
        backend = MQTTBackend(broker_ip, broker_port)
        backend.start()

    try:
        run_calibration(backend, middles, mins, maxs)
    except KeyboardInterrupt:
        print("\n\nInterrupted -- centering servos...")
        backend.move_all(middles, speed=1)
        time.sleep(1.0)
        print("Stopped.")
    finally:
        backend.stop()


if __name__ == "__main__":
    main()
