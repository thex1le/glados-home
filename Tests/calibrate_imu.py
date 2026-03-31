"""IMU auto-calibration using servo movements.

Moves the servos through a sequence of positions to give the BNO055
enough orientation variety to calibrate its accel, gyro, and magnetometer.
Saves the calibration offsets to imu_calibration.json once complete.

Run on the GPU server while BodyServer (Pi4) and GLaDOS (Pi5) are running.

Usage:
    python Tests/calibrate_imu.py -config glog.conf
"""

# builtin
import argparse
import sys
import time
from configparser import ConfigParser
from json import loads, dumps
from threading import Lock

# 3rd party
import paho.mqtt.client as mqtt

# glados imports
sys.path.insert(0, '.')
from glados_modules.GladosEnums import (
    ServoEnum, IMUEnums, MQTTEnums, SystemEnums
)


class IMUCalibrationClient:
    """MQTT client that commands servos and monitors IMU calibration status."""

    def __init__(self, broker_ip: str, broker_port: int) -> None:
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._broker_ip = broker_ip
        self._broker_port = broker_port
        self._imu_data: dict = {}
        self._lock = Lock()

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def connect(self) -> None:
        """Connect to MQTT broker."""
        self._client.connect(self._broker_ip, self._broker_port)
        self._client.loop_start()
        print(f"Connected to MQTT broker at {self._broker_ip}:{self._broker_port}")

    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
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

    def get_imu(self) -> dict:
        with self._lock:
            return dict(self._imu_data)

    def get_euler(self) -> tuple:
        with self._lock:
            euler = self._imu_data.get(IMUEnums.EULER_KEY.value, (0, 0, 0))
            return tuple(euler) if euler else (0, 0, 0)

    def get_calibration_status(self) -> tuple:
        """Get BNO055 calibration status from the IMU data.

        The BNO055 doesn't send calibration_status via MQTT by default,
        so we infer readiness from whether euler angles are non-zero.

        Returns:
            Tuple of euler angles -- if all zeros, not yet calibrated.
        """
        return self.get_euler()

    def is_euler_live(self) -> bool:
        """Check if euler angles are producing non-zero data."""
        euler = self.get_euler()
        return euler is not None and any(abs(v) > 0.1 for v in euler)

    def send_move_all(self, angles: dict, speed: int = 2) -> None:
        """Command all servos simultaneously."""
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
        """Wait until IMU data arrives."""
        start = time.time()
        while time.time() - start < timeout:
            if self.get_imu():
                return True
            time.sleep(0.2)
        return False


def run_calibration(config_path: str) -> None:
    """Run the IMU calibration dance using servo movements."""
    config = ConfigParser()
    config.read(config_path)
    broker_ip = config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_SERVER_IP.value)
    broker_port = int(config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_PORT.value))

    # Parse servo ranges from config (strip inline comments like "92# rads/s")
    def parse_csv(raw: str) -> list:
        return [v.split('#')[0].strip() for v in raw.split(',')]

    head_vals = parse_csv(config.get(ServoEnum.CONFIG_HEAD.value, ServoEnum.HEAD_MIN_MAX_CENTER.value))
    neck_vals = parse_csv(config.get(ServoEnum.CONFIG_HEAD.value, ServoEnum.NECK_MIN_MAX_CENTER.value))
    default_vals = parse_csv(config.get(ServoEnum.CONFIG_HEAD.value, ServoEnum.DEFAULT_MAX_MIN_CENTER.value))

    body_lr = ServoEnum.LOCATION_BODY_LEFT_RIGHT.value
    body_ud = ServoEnum.LOCATION_BODY_UP_DOWN.value
    head_lr = ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
    head_ud = ServoEnum.LOCATION_HEAD_UP_DOWN.value

    middles = {
        body_lr: float(default_vals[2]),
        body_ud: float(neck_vals[2]),
        head_lr: float(head_vals[2]),
        head_ud: float(head_vals[2]),
    }
    mins = {
        body_lr: float(default_vals[1]),
        body_ud: float(neck_vals[1]),
        head_lr: float(head_vals[1]),
        head_ud: float(head_vals[1]),
    }
    maxs = {
        body_lr: float(default_vals[0]),
        body_ud: float(neck_vals[0]),
        head_lr: float(head_vals[0]),
        head_ud: float(head_vals[0]),
    }

    cal = IMUCalibrationClient(broker_ip, broker_port)
    cal.connect()

    print("\n=== IMU Auto-Calibration ===")
    print("This will move all servos through their range to calibrate the BNO055.")
    print("Make sure BodyServer (Pi4) and GLaDOS (Pi5) are running.\n")

    # Wait for IMU data
    print("Waiting for IMU data from Pi5...")
    if not cal.wait_for_imu():
        print("ERROR: No IMU data received. Is GLaDOS.py running on Pi5?")
        cal.disconnect()
        return
    print("IMU data received.\n")

    # --- Phase 1: Gyro calibration (hold still) ---
    print("--- Phase 1: Gyro calibration (hold still 5s) ---")
    cal.send_move_all(middles, speed=1)
    time.sleep(3.0)  # let servos settle
    print("Holding still...")
    time.sleep(5.0)
    euler = cal.get_euler()
    print(f"  Euler: heading={euler[0]:.1f}  roll={euler[1]:.1f}  pitch={euler[2]:.1f}")

    # --- Phase 2: Accel calibration (tilt to different orientations) ---
    print("\n--- Phase 2: Accel calibration (tilting through positions) ---")

    # Build a sequence of positions that tilt the head to various orientations
    # Each position is held for 3 seconds for the BNO055 to sample gravity
    tilt_positions = [
        # (description, {servo: angle})
        ("Center",           middles),
        ("Body UD min",      {**middles, body_ud: mins[body_ud] + 5}),
        ("Body UD max",      {**middles, body_ud: maxs[body_ud] - 5}),
        ("Head UD min",      {**middles, head_ud: mins[head_ud] + 10}),
        ("Head UD max",      {**middles, head_ud: maxs[head_ud] - 10}),
        ("Both UD low",      {**middles, body_ud: mins[body_ud] + 5, head_ud: mins[head_ud] + 10}),
        ("Both UD high",     {**middles, body_ud: maxs[body_ud] - 5, head_ud: maxs[head_ud] - 10}),
        ("Head LR + UD low", {**middles, head_lr: mins[head_lr] + 10, head_ud: mins[head_ud] + 10}),
        ("Head LR + UD high",{**middles, head_lr: maxs[head_lr] - 10, head_ud: maxs[head_ud] - 10}),
    ]

    for i, (desc, angles) in enumerate(tilt_positions):
        print(f"  Position {i + 1}/{len(tilt_positions)}: {desc}")
        cal.send_move_all(angles, speed=2)
        time.sleep(3.0)
        euler = cal.get_euler()
        live = cal.is_euler_live()
        status = "LIVE" if live else "waiting..."
        print(f"    Euler: heading={euler[0]:.1f}  roll={euler[1]:.1f}  pitch={euler[2]:.1f}  [{status}]")

    # Return to center
    cal.send_move_all(middles, speed=2)
    time.sleep(2.0)

    # --- Phase 3: Magnetometer calibration (rotate body LR through full range) ---
    print("\n--- Phase 3: Magnetometer calibration (rotating body LR) ---")
    print("  Sweeping body left-right through full range...")

    # Sweep LR in steps, pausing briefly at each
    lr_min = mins[body_lr] + 10
    lr_max = maxs[body_lr] - 10
    lr_steps = 12
    step_size = (lr_max - lr_min) / lr_steps

    for i in range(lr_steps + 1):
        angle = lr_min + i * step_size
        cal.send_move_all({**middles, body_lr: angle}, speed=3)
        time.sleep(1.5)
        euler = cal.get_euler()
        live = cal.is_euler_live()
        bar = "#" * (i + 1) + "." * (lr_steps - i)
        print(f"    [{bar}] LR={angle:5.1f}  euler=({euler[0]:6.1f}, {euler[1]:5.1f}, {euler[2]:5.1f})  {'LIVE' if live else '...'}")

    # Sweep back
    print("  Sweeping back...")
    for i in range(lr_steps, -1, -1):
        angle = lr_min + i * step_size
        cal.send_move_all({**middles, body_lr: angle}, speed=3)
        time.sleep(1.0)

    # Return to center
    cal.send_move_all(middles, speed=2)
    time.sleep(2.0)

    # --- Phase 4: Combined figure-8 style movements ---
    print("\n--- Phase 4: Combined movements for final calibration ---")

    combo_positions = [
        ("LR left + UD low",   {**middles, body_lr: lr_min, body_ud: mins[body_ud] + 5}),
        ("LR right + UD high", {**middles, body_lr: lr_max, body_ud: maxs[body_ud] - 5}),
        ("LR left + head tilt",{**middles, body_lr: lr_min, head_ud: maxs[head_ud] - 10}),
        ("LR right + head tilt",{**middles, body_lr: lr_max, head_ud: mins[head_ud] + 10}),
        ("All middle",         middles),
    ]

    for i, (desc, angles) in enumerate(combo_positions):
        print(f"  Position {i + 1}/{len(combo_positions)}: {desc}")
        cal.send_move_all(angles, speed=2)
        time.sleep(3.0)
        euler = cal.get_euler()
        live = cal.is_euler_live()
        print(f"    Euler: heading={euler[0]:.1f}  roll={euler[1]:.1f}  pitch={euler[2]:.1f}  [{'LIVE' if live else 'waiting...'}]")

    # --- Done ---
    print("\n--- Calibration sequence complete ---")
    cal.send_move_all(middles, speed=1)
    time.sleep(2.0)

    euler = cal.get_euler()
    live = cal.is_euler_live()

    if live:
        print(f"\nIMU is producing euler data: ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f})")
        print("The IMU class will auto-save calibration offsets to imu_calibration.json")
        print("once the BNO055 reports all calibration statuses at 3.")
        print("\nIf euler is still (0, 0, 0), the magnetometer may need more rotation.")
        print("Try running the script again or manually rotating the robot if possible.")
    else:
        print("\nWARNING: Euler angles are still zero.")
        print("The BNO055 may need more movement variety. Things to try:")
        print("  - Run this script again (calibration persists across runs)")
        print("  - Power cycle the IMU and try again")
        print("  - Check BNO055 I2C connection on Pi5")

    print(f"\nFinal euler: heading={euler[0]:.1f}  roll={euler[1]:.1f}  pitch={euler[2]:.1f}")
    cal.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-calibrate BNO055 IMU using servo movements")
    parser.add_argument("-config", required=True, help="Path to glog.conf")
    args = parser.parse_args()
    run_calibration(args.config)
