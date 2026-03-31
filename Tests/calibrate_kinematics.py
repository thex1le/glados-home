"""Live calibration tool for RobotKinematics sign multipliers and camera axis.

Run on the GPU server while all 3 systems are up. Connects to MQTT, commands
each servo one at a time, reads IMU euler angles, and compares against FK
predictions to determine the correct sign multiplier for each joint.

Usage:
    python Tests/calibrate_kinematics.py -config glog.conf

Steps:
    1. Centers all servos and records baseline IMU orientation
    2. For each servo: moves +15 degrees, records IMU change, compares with FK
    3. Reports which signs need flipping
    4. Optionally writes corrected signs back to GladosEnums.py
"""

# builtin
import argparse
import sys
import time
from collections import namedtuple
from configparser import ConfigParser
from json import loads, dumps
from threading import Lock

# 3rd party
import paho.mqtt.client as mqtt

# glados imports
sys.path.insert(0, '.')
from glados_modules.GladosEnums import (
    ServoEnum, IMUEnums, MQTTEnums, SystemEnums, MotionProfile, KinematicsEnums
)
from glados_modules.RobotKinematics import RobotKinematics


class CalibrationClient:
    """MQTT client that commands servos and reads IMU for calibration."""

    def __init__(self, broker_ip: str, broker_port: int) -> None:
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._broker_ip = broker_ip
        self._broker_port = broker_port

        # Latest data
        self._imu_data: dict = {}
        self._servo_status: dict = {}
        self._imu_lock = Lock()
        self._servo_lock = Lock()

        # Servo names
        self.servo_names = [
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value,
            ServoEnum.LOCATION_BODY_UP_DOWN.value,
            ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value,
            ServoEnum.LOCATION_HEAD_UP_DOWN.value,
        ]

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def connect(self) -> None:
        """Connect to MQTT broker and start listening."""
        self._client.connect(self._broker_ip, self._broker_port)
        self._client.loop_start()
        print(f"Connected to MQTT broker at {self._broker_ip}:{self._broker_port}")

    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        self._client.loop_stop()
        self._client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        """Subscribe to IMU and servo status topics."""
        client.subscribe(MQTTEnums.IMU_STATUS_TOPIC.value)
        client.subscribe(ServoEnum.MQTT_STATUS_TOPIC.value)
        print(f"Subscribed to IMU and servo status topics")

    def _on_message(self, client, userdata, msg) -> None:
        """Handle incoming MQTT messages."""
        try:
            payload = loads(msg.payload.decode())
        except Exception:
            return

        if msg.topic == MQTTEnums.IMU_STATUS_TOPIC.value:
            imu_status = payload.get(IMUEnums.IMU_STATUS_KEY.value, {})
            if imu_status:
                with self._imu_lock:
                    self._imu_data = imu_status
        elif msg.topic == ServoEnum.MQTT_STATUS_TOPIC.value:
            results = payload.get(ServoEnum.MSG_RESULTS.value, {})
            location = results.get(ServoEnum.MSG_LOCATION.value, "")
            if location:
                with self._servo_lock:
                    self._servo_status[location] = results

    def get_imu_euler(self) -> tuple:
        """Get current IMU euler angles (heading, roll, pitch) in degrees."""
        with self._imu_lock:
            euler = self._imu_data.get(IMUEnums.EULER_KEY.value, (0, 0, 0))
            return tuple(euler) if euler else (0, 0, 0)

    def get_servo_angles(self) -> dict:
        """Get current servo angles from status messages."""
        with self._servo_lock:
            angles = {}
            for name in self.servo_names:
                if name in self._servo_status:
                    angles[name] = float(self._servo_status[name].get(
                        ServoEnum.MSG_CURRENT_ANGLE.value, 0))
            return angles

    def send_move(self, servo_name: str, angle: float, speed: int = 2) -> None:
        """Command a single servo to move."""
        msg = {
            ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_MOVE.value,
            ServoEnum.MSG_LOCATION_KEY.value: servo_name,
            ServoEnum.MSG_ANGLE.value: round(angle),
            ServoEnum.MSG_SPEED.value: speed,
            "uuid": f"cal_{time.time()}",
        }
        self._client.publish(ServoEnum.MQTT_COMMAND_TOPIC.value, dumps(msg))

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
            "uuid": f"cal_{time.time()}",
        }
        self._client.publish(ServoEnum.MQTT_COMMAND_TOPIC.value, dumps(msg))

    def request_status(self) -> None:
        """Request status from all servos."""
        for name in self.servo_names:
            msg = {
                ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_STATUS.value,
                ServoEnum.MSG_LOCATION_KEY.value: name,
                "uuid": f"status_{time.time()}",
            }
            self._client.publish(ServoEnum.MQTT_COMMAND_TOPIC.value, dumps(msg))

    def wait_for_imu(self, timeout: float = 5.0) -> bool:
        """Wait until IMU data is available."""
        start = time.time()
        while time.time() - start < timeout:
            with self._imu_lock:
                if self._imu_data:
                    return True
            time.sleep(0.1)
        return False

    def wait_for_servos(self, timeout: float = 5.0) -> bool:
        """Wait until all 4 servo statuses are available."""
        start = time.time()
        while time.time() - start < timeout:
            angles = self.get_servo_angles()
            if len(angles) == 4:
                return True
            self.request_status()
            time.sleep(0.5)
        return False

    def wait_settle(self, duration: float = 2.0) -> None:
        """Wait for servos to settle after a move command."""
        time.sleep(duration)

    def sample_imu(self, samples: int = 10, interval: float = 0.1) -> tuple:
        """Take multiple IMU readings and return the average euler angles."""
        readings = []
        for _ in range(samples):
            readings.append(self.get_imu_euler())
            time.sleep(interval)
        avg = tuple(sum(r[i] for r in readings) / len(readings) for i in range(3))
        return avg


def run_calibration(config_path: str) -> None:
    """Run the full sign calibration procedure."""
    # Parse config for MQTT broker
    config = ConfigParser()
    config.read(config_path)
    broker_ip = config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_SERVER_IP.value)
    broker_port = int(config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_PORT.value))

    # Parse servo config (strip inline comments like "92# rads/s")
    def parse_csv(raw: str) -> list:
        return [v.split('#')[0].strip() for v in raw.split(',')]

    head_vals = parse_csv(config.get(ServoEnum.CONFIG_HEAD.value, ServoEnum.HEAD_MIN_MAX_CENTER.value))
    neck_vals = parse_csv(config.get(ServoEnum.CONFIG_HEAD.value, ServoEnum.NECK_MIN_MAX_CENTER.value))
    default_vals = parse_csv(config.get(ServoEnum.CONFIG_HEAD.value, ServoEnum.DEFAULT_MAX_MIN_CENTER.value))

    servo_middles = {
        ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: float(default_vals[2]),
        ServoEnum.LOCATION_BODY_UP_DOWN.value: float(neck_vals[2]),
        ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: float(head_vals[2]),
        ServoEnum.LOCATION_HEAD_UP_DOWN.value: float(head_vals[2]),
    }
    servo_mins = {
        ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: float(default_vals[1]),
        ServoEnum.LOCATION_BODY_UP_DOWN.value: float(neck_vals[1]),
        ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: float(head_vals[1]),
        ServoEnum.LOCATION_HEAD_UP_DOWN.value: float(head_vals[1]),
    }
    servo_maxs = {
        ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: float(default_vals[0]),
        ServoEnum.LOCATION_BODY_UP_DOWN.value: float(neck_vals[0]),
        ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: float(head_vals[0]),
        ServoEnum.LOCATION_HEAD_UP_DOWN.value: float(head_vals[0]),
    }

    cal = CalibrationClient(broker_ip, broker_port)
    cal.connect()

    print("\n=== GLaDOS Kinematics Calibration ===")
    print("Make sure all 3 systems are running (BodyServer, AiServer, GLaDOS)")
    print()

    # Wait for data
    print("Waiting for IMU data...")
    if not cal.wait_for_imu():
        print("ERROR: No IMU data received. Is Pi5 running?")
        cal.disconnect()
        return

    print("Waiting for servo status...")
    if not cal.wait_for_servos():
        print("WARNING: Not all servo statuses received. Proceeding with middles as fallback.")

    # Step 1: Center all servos
    print("\n--- Step 1: Centering all servos ---")
    cal.send_move_all(servo_middles, speed=2)
    cal.wait_settle(3.0)

    baseline_imu = cal.sample_imu(samples=20)
    baseline_servos = cal.get_servo_angles()
    if len(baseline_servos) < 4:
        baseline_servos = dict(servo_middles)

    print(f"Baseline IMU euler: heading={baseline_imu[0]:.1f}  roll={baseline_imu[1]:.1f}  pitch={baseline_imu[2]:.1f}")
    print(f"Baseline servos: {baseline_servos}")

    # Create kinematics with current signs
    kin = RobotKinematics(servo_middles, servo_mins, servo_maxs)
    baseline_yaw, baseline_pitch = kin.forward_kinematics(baseline_servos)
    print(f"FK baseline: yaw={baseline_yaw:.2f}  pitch={baseline_pitch:.2f}")

    # Step 2: Test each servo
    test_delta = 15.0  # degrees to move each servo
    results = {}

    joint_labels = [
        ("body_left_right", "Body LR", 0),
        ("body_up_down", "Body UD", 1),
        ("head_left_right", "Head LR", 2),
        ("head_up_down", "Head UD", 3),
    ]

    for servo_name, label, joint_idx in joint_labels:
        print(f"\n--- Step 2.{joint_idx + 1}: Testing {label} ---")

        # Compute safe test angle (don't exceed limits)
        middle = servo_middles[servo_name]
        test_angle = middle + test_delta
        if test_angle > servo_maxs[servo_name]:
            test_angle = middle - test_delta
            actual_delta = test_angle - middle
        else:
            actual_delta = test_delta

        print(f"  Moving {label} from {middle:.0f} to {test_angle:.0f} (delta={actual_delta:+.0f})")

        # Move this servo only
        cal.send_move(servo_name, test_angle, speed=2)
        cal.wait_settle(3.0)

        # Read IMU
        test_imu = cal.sample_imu(samples=20)
        test_servos = cal.get_servo_angles()
        if len(test_servos) < 4:
            test_servos = dict(servo_middles)
            test_servos[servo_name] = test_angle

        # FK prediction
        test_yaw, test_pitch = kin.forward_kinematics(test_servos)

        # Compute deltas
        imu_d_heading = test_imu[0] - baseline_imu[0]
        imu_d_roll = test_imu[1] - baseline_imu[1]
        imu_d_pitch = test_imu[2] - baseline_imu[2]
        fk_d_yaw = test_yaw - baseline_yaw
        fk_d_pitch = test_pitch - baseline_pitch

        print(f"  IMU delta:  heading={imu_d_heading:+.2f}  roll={imu_d_roll:+.2f}  pitch={imu_d_pitch:+.2f}")
        print(f"  FK delta:   yaw={fk_d_yaw:+.2f}  pitch={fk_d_pitch:+.2f}")

        results[servo_name] = {
            "label": label,
            "delta_deg": actual_delta,
            "imu_heading_delta": imu_d_heading,
            "imu_roll_delta": imu_d_roll,
            "imu_pitch_delta": imu_d_pitch,
            "fk_yaw_delta": fk_d_yaw,
            "fk_pitch_delta": fk_d_pitch,
        }

        # Return to center
        cal.send_move(servo_name, middle, speed=2)
        cal.wait_settle(2.0)

    # Step 3: Analysis
    print("\n\n=== CALIBRATION RESULTS ===")
    print()
    print("Compare FK predictions vs IMU measurements for each joint.")
    print("If FK and IMU deltas have OPPOSITE signs, flip that joint's sign multiplier.")
    print()
    print("NOTE: IMU euler (heading, roll, pitch) may not map directly to FK (yaw, pitch).")
    print("The IMU is mounted on the head, so its frame depends on the mounting orientation.")
    print("Look at which IMU axis responds most strongly to each servo and compare signs.")
    print()

    print(f"{'Joint':<20} {'Servo Delta':>12} {'IMU heading':>12} {'IMU roll':>10} {'IMU pitch':>10} {'FK yaw':>10} {'FK pitch':>10}")
    print("-" * 94)
    for servo_name, data in results.items():
        print(f"{data['label']:<20} {data['delta_deg']:>+12.1f} "
              f"{data['imu_heading_delta']:>+12.2f} {data['imu_roll_delta']:>+10.2f} "
              f"{data['imu_pitch_delta']:>+10.2f} {data['fk_yaw_delta']:>+10.2f} "
              f"{data['fk_pitch_delta']:>+10.2f}")

    print()
    print("Current sign multipliers:")
    signs = [
        KinematicsEnums.SIGN_BODY_LR.value,
        KinematicsEnums.SIGN_BODY_UD.value,
        KinematicsEnums.SIGN_HEAD_LR.value,
        KinematicsEnums.SIGN_HEAD_UD.value,
    ]
    for (servo_name, label, idx), sign in zip(joint_labels, signs):
        print(f"  {label}: {sign:+.0f}")

    print()
    print("To fix a sign, edit KinematicsEnums in GladosEnums.py:")
    print("  SIGN_BODY_LR, SIGN_BODY_UD, SIGN_HEAD_LR, SIGN_HEAD_UD")
    print()
    print("Re-run this script after changing signs to verify.")

    cal.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate GLaDOS kinematics sign multipliers")
    parser.add_argument("-config", required=True, help="Path to glog.conf")
    args = parser.parse_args()
    run_calibration(args.config)
