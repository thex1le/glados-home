#!/usr/bin/env python3
"""Servo-IMU calibration sweep.

Moves each servo through its full range while logging IMU orientation.
Compares FK predicted direction vs IMU measured direction to find
discrepancies: pushrod nonlinearity, dead zones, sign errors, cross-coupling.

Requires: Pi4 (BodyServer) running for servos, Pi5 (GLaDOS) running for IMU.
Stop AiServer first so tracking doesn't override servo commands.

Usage:
    python Tools/servo_imu_sweep.py -config glog.conf
    python Tools/servo_imu_sweep.py -config glog.conf -step 2 -settle 1.5
    python Tools/servo_imu_sweep.py -config glog.conf -servo head_up_down
"""
import argparse
import json
import os
import sys
import threading
import time
from configparser import ConfigParser
from uuid import uuid4

import paho.mqtt.client as mqtt

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Servo-IMU calibration sweep")
    parser.add_argument("-config", required=True, help="Path to glog.conf")
    parser.add_argument("-step", type=float, default=5.0, help="Degrees per step (default 5)")
    parser.add_argument("-settle", type=float, default=2.0, help="Seconds to wait at each position (default 2)")
    parser.add_argument("-samples", type=int, default=5, help="IMU samples to average per position (default 5)")
    parser.add_argument("-servo", type=str, default=None,
                        help="Sweep only this servo (e.g. head_up_down). Default: sweep all 4.")
    parser.add_argument("-output", type=str, default=None, help="Output JSON file (default: auto-named)")
    parser.add_argument("-speed", type=int, default=2, help="Servo move speed 1-5 (default 2)")
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    broker_ip = config.get("MQTT", "mqtt_server_ip")
    broker_port = config.getint("MQTT", "mqtt_port")

    # Servo definitions from config
    # body_lr and body_ud use default_max_min_center (0-180)
    # head_lr uses neck_min_max_center (52-120)
    # head_ud uses head_min_max_center (6-125)
    try:
        default_vals = config.get("SERVOS", "default_max_min_center").split(",")
        head_vals = config.get("SERVOS", "head_min_max_center").split(",")
        neck_vals = config.get("SERVOS", "neck_min_max_center").split(",")
        servo_defs = {
            "body_left_right": {"max": int(default_vals[0]), "min": int(default_vals[1]),
                                "center": int(default_vals[2])},
            "body_up_down": {"max": int(default_vals[0]), "min": int(default_vals[1]),
                             "center": int(default_vals[2])},
            "head_left_right": {"max": int(neck_vals[0]), "min": int(neck_vals[1]),
                                "center": int(neck_vals[2])},
            "head_up_down": {"max": int(head_vals[0]), "min": int(head_vals[1]),
                             "center": int(head_vals[2])},
        }
        print(f"Servo ranges from config:")
        for s, d in servo_defs.items():
            print(f"  {s}: {d['min']}-{d['max']}, center={d['center']}")
    except Exception as e:
        print(f"Warning: could not read servo config: {e}, using defaults")
        servo_defs = {
            "body_left_right": {"min": 0, "max": 180, "center": 90},
            "body_up_down": {"min": 0, "max": 180, "center": 90},
            "head_left_right": {"min": 52, "max": 120, "center": 92},
            "head_up_down": {"min": 6, "max": 125, "center": 83},
        }

    # FK
    try:
        from glados_modules.RobotKinematics import RobotKinematics
        middles = {s: servo_defs[s]["center"] for s in servo_defs}
        mins = {s: servo_defs[s]["min"] for s in servo_defs}
        maxs = {s: servo_defs[s]["max"] for s in servo_defs}
        kin = RobotKinematics(middles, mins, maxs)
        print("FK computation ready")
    except Exception as e:
        print(f"FK unavailable: {e}")
        kin = None

    # IMU state
    imu_data = {"euler": None, "quaternion": None, "gyroscope": None, "ts": 0}
    imu_lock = threading.Lock()
    imu_received = threading.Event()

    # MQTT
    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            # IMU publishes as {"imu_status": {"euler": [...], ...}}
            imu_payload = data.get("imu_status", data)
            if "euler" in imu_payload:
                with imu_lock:
                    imu_data["euler"] = imu_payload.get("euler")
                    imu_data["quaternion"] = imu_payload.get("quaternion")
                    imu_data["gyroscope"] = imu_payload.get("gyroscope")
                    imu_data["ts"] = time.time()
                imu_received.set()
        except (json.JSONDecodeError, KeyError):
            pass

    # Also track health status for IMU thread
    imu_health = {"alive": None, "checked": False}
    health_lock = threading.Lock()

    def on_health_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            threads = data.get("threads", {})
            if "IMU" in threads:
                with health_lock:
                    imu_health["alive"] = threads["IMU"].get("alive", False)
                    imu_health["checked"] = True
        except (json.JSONDecodeError, KeyError):
            pass

    client = mqtt.Client()
    client.on_message = on_message
    print(f"Connecting to MQTT broker at {broker_ip}:{broker_port}")
    client.connect(broker_ip, broker_port)
    client.subscribe("body/imu/status")
    client.subscribe("system/health/#")
    # Route health messages to separate handler
    client.message_callback_add("system/health/#", on_health_message)
    client.loop_start()

    # Wait for health check first (up to 8 seconds for heartbeat)
    print("Checking IMU health status...")
    health_deadline = time.time() + 8
    while time.time() < health_deadline:
        with health_lock:
            if imu_health["checked"]:
                break
        time.sleep(0.5)

    with health_lock:
        if imu_health["checked"] and not imu_health["alive"]:
            print("ERROR: IMU thread is DEAD on the Pi5.")
            print("The BNO055 IMU has crashed (likely I2C bus error).")
            print("Restart GLaDOS.py on the Pi5 to recover the IMU.")
            client.loop_stop()
            sys.exit(1)
        elif imu_health["checked"] and imu_health["alive"]:
            print("  IMU thread is ALIVE")
        else:
            print("  WARNING: No health heartbeat received — IMU status unknown")
            print("  Continuing anyway, will check for IMU data directly...")

    # Wait for actual IMU data
    print("Waiting for IMU data on body/imu/status...")
    if not imu_received.wait(timeout=10):
        print("ERROR: No IMU data received in 10 seconds.")
        print("Possible causes:")
        print("  - GLaDOS.py (Pi5) is not running")
        print("  - IMU thread crashed (I2C bus error on BNO055)")
        print("  - MQTT broker is not forwarding messages")
        print("  - IMU sensor is not connected (check I2C: sudo i2cdetect -y 1)")
        client.loop_stop()
        sys.exit(1)

    # Verify IMU data is updating with valid values
    valid_euler = None
    for attempt in range(10):
        imu_received.clear()
        if imu_received.wait(timeout=2):
            with imu_lock:
                euler = imu_data["euler"]
                if euler and len(euler) >= 3 and all(v is not None for v in euler):
                    valid_euler = [float(v) for v in euler]
                    break
        print(f"  IMU: waiting for valid data (attempt {attempt + 1})...")

    if not valid_euler:
        print("ERROR: IMU is publishing but euler values are None.")
        print("The BNO055 sensor may need recalibration or power cycle.")
        client.loop_stop()
        sys.exit(1)

    print(f"  IMU data flowing: euler={[round(e,1) for e in valid_euler]}")
    print("  IMU OK")

    def send_servo(servo_name, angle, speed):
        msg = {"cmd": "move", "servo": servo_name, "angle": int(round(angle)),
               "speed": speed, "uuid": str(uuid4())}
        client.publish("body/servo", json.dumps(msg))

    def get_imu_samples(n=5, interval=0.15):
        """Collect n valid IMU samples and return averaged euler + raw quaternions."""
        samples = []
        attempts = 0
        max_attempts = n * 3  # retry up to 3x to get n valid samples
        while len(samples) < n and attempts < max_attempts:
            attempts += 1
            imu_received.clear()
            if imu_received.wait(timeout=1):
                with imu_lock:
                    euler = imu_data["euler"]
                    if euler and len(euler) >= 3 and all(v is not None for v in euler):
                        samples.append({
                            "euler": [float(v) for v in euler],
                            "quaternion": [float(v) for v in imu_data["quaternion"]] if imu_data["quaternion"] and all(v is not None for v in imu_data["quaternion"]) else None,
                            "gyroscope": [float(v) for v in imu_data["gyroscope"]] if imu_data["gyroscope"] and all(v is not None for v in imu_data["gyroscope"]) else None,
                        })
                    else:
                        print(f"    IMU: corrupt sample (attempt {attempts}), retrying...")
            time.sleep(interval)

        if not samples:
            return None

        # Average euler angles
        avg_euler = [
            sum(s["euler"][i] for s in samples) / len(samples)
            for i in range(3)
        ]
        # Average gyro magnitude (should be near zero when settled)
        gyro_mags = []
        for s in samples:
            if s["gyroscope"]:
                mag = sum(g ** 2 for g in s["gyroscope"]) ** 0.5
                gyro_mags.append(mag)
        avg_gyro_mag = sum(gyro_mags) / len(gyro_mags) if gyro_mags else 0

        return {
            "euler_avg": [round(e, 3) for e in avg_euler],
            "gyro_magnitude": round(avg_gyro_mag, 4),
            "n_samples": len(samples),
            "quaternion_last": samples[-1]["quaternion"],
        }

    # Determine which servos to sweep
    if args.servo:
        if args.servo not in servo_defs:
            print(f"Unknown servo: {args.servo}")
            print(f"Options: {list(servo_defs.keys())}")
            sys.exit(1)
        sweep_servos = [args.servo]
    else:
        sweep_servos = list(servo_defs.keys())

    # Center all servos first
    print("\nCentering all servos...")
    for servo_name, sdef in servo_defs.items():
        send_servo(servo_name, sdef["center"], args.speed)
    time.sleep(3)

    # Read IMU at center position
    center_imu = get_imu_samples(args.samples)
    print(f"Center IMU euler: {center_imu['euler_avg']}")

    # Compute FK at center
    center_fk = None
    if kin:
        center_angles = {s: servo_defs[s]["center"] for s in servo_defs}
        yaw, pitch = kin.forward_kinematics(center_angles)
        center_fk = {"yaw": round(yaw, 2), "pitch": round(pitch, 2)}
        print(f"Center FK: yaw={yaw:.1f} pitch={pitch:.1f}")

    # Output path — set early so incremental saves work
    if args.output:
        output_path = args.output
    else:
        output_path = f"servo_imu_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"

    def save_progress():
        """Incremental save after each section so data isn't lost on error."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [saved progress to {output_path}]")

    # Results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "step_deg": args.step,
            "settle_time": args.settle,
            "samples_per_position": args.samples,
            "speed": args.speed,
        },
        "servo_defs": servo_defs,
        "center_imu": center_imu,
        "center_fk": center_fk,
        "sweeps": {},
        "interactions": [],
    }

    def _gen_sweep_angles(sdef, step):
        """Generate angles: center → max → min → center."""
        angles = []
        a = float(sdef["center"])
        while a <= sdef["max"]:
            angles.append(a)
            a += step
        if angles[-1] != sdef["max"]:
            angles.append(float(sdef["max"]))
        a = sdef["max"] - step
        while a >= sdef["min"]:
            angles.append(a)
            a -= step
        if angles[-1] != sdef["min"]:
            angles.append(float(sdef["min"]))
        a = sdef["min"] + step
        while a < sdef["center"]:
            angles.append(a)
            a += step
        angles.append(float(sdef["center"]))
        return angles

    def _sweep_and_log(sweep_servo, fixed_servos, step):
        """Sweep one servo while others are held at specified angles. Returns list of entries."""
        sdef = servo_defs[sweep_servo]
        data = []

        # Set fixed servos
        for s, angle in fixed_servos.items():
            send_servo(s, angle, args.speed)
        time.sleep(1)

        angles = _gen_sweep_angles(sdef, step)
        total = len(angles)
        for i, angle in enumerate(angles):
            send_servo(sweep_servo, angle, args.speed)
            time.sleep(args.settle)

            imu_sample = get_imu_samples(args.samples)
            if not imu_sample:
                print(f"  WARNING: No IMU data at {sweep_servo}={angle}")
                continue

            fk_result = None
            if kin:
                fk_angles = dict(fixed_servos)
                fk_angles[sweep_servo] = angle
                yaw, pitch = kin.forward_kinematics(fk_angles)
                fk_result = {"yaw": round(yaw, 2), "pitch": round(pitch, 2)}

            entry = {
                "sweep_servo": sweep_servo,
                "angle": angle,
                "fixed_servos": dict(fixed_servos),
                "imu": imu_sample,
                "fk": fk_result,
            }
            data.append(entry)

            settled = "OK" if imu_sample["gyro_magnitude"] < 0.1 else f"MOVING({imu_sample['gyro_magnitude']:.2f})"
            fk_str = f"FK yaw={fk_result['yaw']:+.1f} pitch={fk_result['pitch']:+.1f}" if fk_result else "no FK"
            euler = imu_sample["euler_avg"]
            fixed_str = " ".join(f"{s.split('_')[0]}={a:.0f}" for s, a in fixed_servos.items()
                                  if a != servo_defs[s]["center"])
            if fixed_str:
                fixed_str = f" [{fixed_str}]"
            print(f"  [{i+1:>3}/{total}] {sweep_servo}={angle:>6.1f}{fixed_str}  "
                  f"IMU: H={euler[0]:>7.1f} R={euler[1]:>7.1f} P={euler[2]:>7.1f}  "
                  f"{fk_str}  {settled}")

        return data

    # ============================================================
    # PHASE 1: Individual servo sweeps (others at center)
    # ============================================================
    for servo_name in sweep_servos:
        sdef = servo_defs[servo_name]
        print(f"\n{'='*60}")
        print(f"  PHASE 1: INDIVIDUAL SWEEP — {servo_name}")
        print(f"  Range: {sdef['min']} to {sdef['max']}, center={sdef['center']}, step={args.step}")
        print(f"{'='*60}")

        fixed = {s: servo_defs[s]["center"] for s in servo_defs if s != servo_name}
        sweep_data = _sweep_and_log(servo_name, fixed, args.step)
        results["sweeps"][servo_name] = sweep_data

        send_servo(servo_name, sdef["center"], args.speed)
        time.sleep(1)
        save_progress()

    # ============================================================
    # PHASE 2: Interaction sweeps (test cross-coupling)
    # ============================================================
    # Use coarser step for interactions to keep time reasonable
    interaction_step = args.step * 2

    # Key interactions: body_ud affects head tracking most
    interaction_pairs = [
        # (sweep_servo, context_servo, context_angles)
        ("head_up_down", "body_up_down",
         [servo_defs["body_up_down"]["min"],
          servo_defs["body_up_down"]["center"],
          servo_defs["body_up_down"]["max"]]),
        ("head_left_right", "body_left_right",
         [servo_defs["body_left_right"]["center"] - 30,
          servo_defs["body_left_right"]["center"],
          servo_defs["body_left_right"]["center"] + 30]),
        ("head_up_down", "head_left_right",
         [servo_defs["head_left_right"]["min"] + 20,
          servo_defs["head_left_right"]["center"],
          servo_defs["head_left_right"]["max"] - 20]),
    ]

    # Only run interactions if sweeping all servos
    if args.servo is None:
        for sweep_servo, ctx_servo, ctx_angles in interaction_pairs:
            for ctx_angle in ctx_angles:
                print(f"\n{'='*60}")
                print(f"  PHASE 2: INTERACTION — sweep {sweep_servo} with {ctx_servo}={ctx_angle}")
                print(f"{'='*60}")

                fixed = {s: servo_defs[s]["center"] for s in servo_defs
                         if s != sweep_servo}
                fixed[ctx_servo] = ctx_angle

                interaction_data = _sweep_and_log(sweep_servo, fixed, interaction_step)
                results["interactions"].append({
                    "sweep_servo": sweep_servo,
                    "context_servo": ctx_servo,
                    "context_angle": ctx_angle,
                    "data": interaction_data,
                })

                send_servo(sweep_servo, servo_defs[sweep_servo]["center"], args.speed)
                send_servo(ctx_servo, servo_defs[ctx_servo]["center"], args.speed)
                time.sleep(1)
                save_progress()
    else:
        print("\n  Skipping interaction sweeps (single-servo mode)")
        print("  Run without -servo to include cross-coupling tests")

    # Center all servos at end
    print("\nReturning all servos to center...")
    for servo_name, sdef in servo_defs.items():
        send_servo(servo_name, sdef["center"], args.speed)

    save_progress()

    # Print summary
    print(f"\n{'='*60}")
    print("  SWEEP SUMMARY")
    print(f"{'='*60}")
    for servo_name, sweep_data in results["sweeps"].items():
        if not sweep_data:
            continue
        # Compare IMU euler change vs FK prediction change
        print(f"\n  {servo_name}:")
        print(f"  {'Angle':>8} {'IMU_H':>8} {'IMU_R':>8} {'IMU_P':>8} {'FK_Yaw':>8} {'FK_Pitch':>8} {'Gyro':>6}")
        for entry in sweep_data:
            e = entry["imu"]["euler_avg"]
            fk = entry.get("fk")
            fy = f"{fk['yaw']:+.1f}" if fk else "  N/A"
            fp = f"{fk['pitch']:+.1f}" if fk else "  N/A"
            gm = entry["imu"]["gyro_magnitude"]
            print(f"  {entry['angle']:>8.1f} {e[0]:>8.1f} {e[1]:>8.1f} {e[2]:>8.1f} {fy:>8} {fp:>8} {gm:>6.3f}")

    # ============================================================
    # PHASE 3: ANALYSIS — Auto-detect axis mapping, compute corrections
    # ============================================================
    print(f"\n{'='*60}")
    print("  PHASE 3: ANALYSIS")
    print(f"{'='*60}")

    analysis = {}

    # IMU euler axes: [0]=Heading, [1]=Roll, [2]=Pitch
    # FK axes: yaw, pitch
    # The IMU is mounted at an arbitrary orientation — we need to find
    # which IMU axis corresponds to FK yaw and which to FK pitch.
    # We do this by correlating: for each servo sweep, which IMU axis
    # has the strongest delta correlation with each FK axis?
    imu_axis_names = ["Heading", "Roll", "Pitch"]

    def _compute_deltas(sweep_data, servo_name):
        """Compute IMU and FK deltas from center for a sweep."""
        center_entry = None
        for e in sweep_data:
            if abs(e["angle"] - servo_defs[servo_name]["center"]) < 1:
                center_entry = e
                break
        if not center_entry:
            return None

        ref_euler = center_entry["imu"]["euler_avg"]
        fk_center = center_entry["fk"]
        deltas = []
        for entry in sweep_data:
            imu = entry["imu"]["euler_avg"]
            fk = entry["fk"]
            imu_d = [imu[i] - ref_euler[i] for i in range(3)]
            # Heading wrap-around
            if imu_d[0] > 180: imu_d[0] -= 360
            elif imu_d[0] < -180: imu_d[0] += 360
            fk_d = [fk["yaw"] - fk_center["yaw"], fk["pitch"] - fk_center["pitch"]]
            deltas.append({
                "angle": entry["angle"],
                "imu_delta": imu_d,     # [heading, roll, pitch]
                "fk_delta": fk_d,        # [yaw, pitch]
            })
        return deltas

    # --- 3a. Auto-detect IMU axis mapping ---
    print("\n  Auto-detecting IMU axis mapping...")
    # Use all individual sweeps to find the best correlation
    all_imu_deltas = {ax: [] for ax in range(3)}  # per IMU axis
    all_fk_deltas = {ax: [] for ax in range(2)}   # per FK axis (0=yaw, 1=pitch)

    for servo_name, sweep_data in results["sweeps"].items():
        if not sweep_data or not sweep_data[0].get("fk"):
            continue
        deltas = _compute_deltas(sweep_data, servo_name)
        if not deltas:
            continue
        for d in deltas:
            for i in range(3):
                all_imu_deltas[i].append(d["imu_delta"][i])
            for i in range(2):
                all_fk_deltas[i].append(d["fk_delta"][i])

    # Find best IMU axis for each FK axis using correlation
    try:
        import numpy as np

        axis_map = {}  # fk_axis_idx -> (imu_axis_idx, sign, correlation)
        fk_axis_labels = ["yaw", "pitch"]

        for fk_idx in range(2):
            fk_arr = np.array(all_fk_deltas[fk_idx])
            best_corr = 0
            best_imu_idx = 0
            best_sign = 1

            for imu_idx in range(3):
                imu_arr = np.array(all_imu_deltas[imu_idx])
                if len(fk_arr) < 3 or np.std(fk_arr) < 0.1 or np.std(imu_arr) < 0.1:
                    continue
                corr = np.corrcoef(fk_arr, imu_arr)[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_imu_idx = imu_idx
                    best_sign = 1 if corr > 0 else -1

            axis_map[fk_idx] = {
                "imu_axis": best_imu_idx,
                "imu_axis_name": imu_axis_names[best_imu_idx],
                "sign": best_sign,
                "correlation": round(best_corr, 4),
            }
            print(f"    FK {fk_axis_labels[fk_idx]} -> IMU {imu_axis_names[best_imu_idx]} "
                  f"(sign={'+' if best_sign > 0 else '-'}, r={best_corr:.3f})")

        analysis["axis_mapping"] = axis_map
    except ImportError:
        print("    numpy not available — skipping axis mapping")
        axis_map = {0: {"imu_axis": 0, "sign": 1}, 1: {"imu_axis": 1, "sign": 1}}

    # --- 3b. EYE_UD_OFFSET calibration (using correct axis mapping) ---
    if center_imu and center_fk and axis_map:
        pitch_imu_idx = axis_map[1]["imu_axis"]
        pitch_sign = axis_map[1]["sign"]
        imu_pitch_mapped = pitch_sign * center_imu["euler_avg"][pitch_imu_idx]
        fk_pitch = center_fk["pitch"]
        eye_offset = fk_pitch - imu_pitch_mapped
        analysis["eye_ud_offset"] = {
            "imu_axis_used": imu_axis_names[pitch_imu_idx],
            "imu_raw_value": round(center_imu["euler_avg"][pitch_imu_idx], 2),
            "imu_mapped_value": round(imu_pitch_mapped, 2),
            "fk_pitch_at_center": round(fk_pitch, 2),
            "measured_offset": round(eye_offset, 2),
            "current_config": 2.0,
        }
        print(f"\n  EYE_UD_OFFSET Calibration (using IMU {imu_axis_names[pitch_imu_idx]}):")
        print(f"    IMU {imu_axis_names[pitch_imu_idx]} at center: {center_imu['euler_avg'][pitch_imu_idx]:+.1f}°")
        print(f"    IMU mapped pitch:     {imu_pitch_mapped:+.1f}°")
        print(f"    FK pitch at center:   {fk_pitch:+.1f}°")
        print(f"    Measured offset:      {eye_offset:+.1f}°")

    # --- 3c. Per-servo FK error with correct axis mapping ---
    for servo_name, sweep_data in results["sweeps"].items():
        if not sweep_data or not sweep_data[0].get("fk"):
            continue
        deltas = _compute_deltas(sweep_data, servo_name)
        if not deltas:
            continue

        errors = []
        ascending = []
        descending = []
        prev_angle = deltas[0]["angle"]

        for d in deltas:
            # Map IMU deltas to FK frame using auto-detected mapping
            imu_yaw = axis_map[0]["sign"] * d["imu_delta"][axis_map[0]["imu_axis"]]
            imu_pitch = axis_map[1]["sign"] * d["imu_delta"][axis_map[1]["imu_axis"]]

            error_entry = {
                "angle": d["angle"],
                "imu_yaw": round(imu_yaw, 2),
                "imu_pitch": round(imu_pitch, 2),
                "fk_yaw": round(d["fk_delta"][0], 2),
                "fk_pitch": round(d["fk_delta"][1], 2),
                "error_yaw": round(d["fk_delta"][0] - imu_yaw, 2),
                "error_pitch": round(d["fk_delta"][1] - imu_pitch, 2),
                "imu_raw": [round(v, 2) for v in d["imu_delta"]],
            }
            errors.append(error_entry)

            if d["angle"] >= prev_angle:
                ascending.append(error_entry)
            else:
                descending.append(error_entry)
            prev_angle = d["angle"]

        # Backlash
        backlash_points = []
        asc_by_angle = {e["angle"]: e for e in ascending}
        desc_by_angle = {e["angle"]: e for e in descending}
        for angle in asc_by_angle:
            if angle in desc_by_angle:
                a = asc_by_angle[angle]
                dd = desc_by_angle[angle]
                bl = {
                    "angle": angle,
                    "yaw_backlash": round(abs(a["imu_yaw"] - dd["imu_yaw"]), 2),
                    "pitch_backlash": round(abs(a["imu_pitch"] - dd["imu_pitch"]), 2),
                }
                backlash_points.append(bl)

        max_yaw_err = max(abs(e["error_yaw"]) for e in errors) if errors else 0
        max_pitch_err = max(abs(e["error_pitch"]) for e in errors) if errors else 0
        mean_yaw_err = sum(abs(e["error_yaw"]) for e in errors) / len(errors) if errors else 0
        mean_pitch_err = sum(abs(e["error_pitch"]) for e in errors) / len(errors) if errors else 0
        max_backlash = max((bl["yaw_backlash"] + bl["pitch_backlash"])
                          for bl in backlash_points) if backlash_points else 0

        servo_analysis = {
            "errors": errors,
            "backlash": backlash_points,
            "max_yaw_error": round(max_yaw_err, 2),
            "max_pitch_error": round(max_pitch_err, 2),
            "mean_yaw_error": round(mean_yaw_err, 2),
            "mean_pitch_error": round(mean_pitch_err, 2),
            "max_backlash": round(max_backlash, 2),
        }

        # Polynomial fit with correct axis mapping
        try:
            angles_arr = np.array([e["angle"] for e in errors])
            yaw_errors = np.array([e["error_yaw"] for e in errors])
            pitch_errors = np.array([e["error_pitch"] for e in errors])

            center = servo_defs[servo_name]["center"]
            half_range = max(servo_defs[servo_name]["max"] - center,
                            center - servo_defs[servo_name]["min"])
            angles_norm = (angles_arr - center) / half_range

            yaw_poly = np.polyfit(angles_norm, yaw_errors, 3)
            pitch_poly = np.polyfit(angles_norm, pitch_errors, 3)

            yaw_pred = np.polyval(yaw_poly, angles_norm)
            pitch_pred = np.polyval(pitch_poly, angles_norm)
            yaw_ss_res = np.sum((yaw_errors - yaw_pred) ** 2)
            yaw_ss_tot = np.sum((yaw_errors - np.mean(yaw_errors)) ** 2)
            pitch_ss_res = np.sum((pitch_errors - pitch_pred) ** 2)
            pitch_ss_tot = np.sum((pitch_errors - np.mean(pitch_errors)) ** 2)
            yaw_r2 = 1 - yaw_ss_res / yaw_ss_tot if yaw_ss_tot > 0 else 0
            pitch_r2 = 1 - pitch_ss_res / pitch_ss_tot if pitch_ss_tot > 0 else 0

            servo_analysis["correction_poly"] = {
                "yaw_coefficients": [round(float(c), 6) for c in yaw_poly.tolist()],
                "pitch_coefficients": [round(float(c), 6) for c in pitch_poly.tolist()],
                "center": center,
                "half_range": half_range,
                "yaw_r_squared": round(float(yaw_r2), 4),
                "pitch_r_squared": round(float(pitch_r2), 4),
            }
            print(f"\n  {servo_name} correction polynomial:")
            print(f"    Yaw:   R²={yaw_r2:.3f}")
            print(f"    Pitch: R²={pitch_r2:.3f}")
        except Exception as e:
            print(f"\n  {servo_name}: polynomial fit failed: {e}")

        analysis[servo_name] = servo_analysis

        print(f"\n  {servo_name} FK Error (mapped axes):")
        print(f"    Max yaw error:   {max_yaw_err:.1f}°")
        print(f"    Max pitch error: {max_pitch_err:.1f}°")
        print(f"    Mean yaw error:  {mean_yaw_err:.1f}°")
        print(f"    Mean pitch error:{mean_pitch_err:.1f}°")
        print(f"    Max backlash:    {max_backlash:.1f}°")

        # Print detailed error table
        print(f"    {'Angle':>8} {'FK_Yaw':>8} {'IMU_Yaw':>8} {'Err_Y':>7} {'FK_Pit':>8} {'IMU_Pit':>8} {'Err_P':>7}")
        for e in errors:
            print(f"    {e['angle']:>8.1f} {e['fk_yaw']:>+8.1f} {e['imu_yaw']:>+8.1f} {e['error_yaw']:>+7.1f} "
                  f"{e['fk_pitch']:>+8.1f} {e['imu_pitch']:>+8.1f} {e['error_pitch']:>+7.1f}")

    # --- 3d. Cross-coupling analysis (using mapped axes) ---
    if results["interactions"]:
        print(f"\n  Cross-Coupling Analysis:")
        coupling_summary = []
        for interaction in results["interactions"]:
            sweep_s = interaction["sweep_servo"]
            ctx_s = interaction["context_servo"]
            ctx_a = interaction["context_angle"]
            ctx_center = servo_defs[ctx_s]["center"]
            data = interaction["data"]

            if not data:
                continue

            center_sweep = None
            for other in results["interactions"]:
                if (other["sweep_servo"] == sweep_s and
                        other["context_servo"] == ctx_s and
                        abs(other["context_angle"] - ctx_center) < 1):
                    center_sweep = other["data"]
                    break

            if center_sweep and ctx_a != ctx_center:
                max_shift_yaw = 0
                max_shift_pitch = 0
                for entry in data:
                    for centry in center_sweep:
                        if abs(centry["angle"] - entry["angle"]) < 1:
                            # Use mapped axes for comparison
                            yaw_idx = axis_map[0]["imu_axis"]
                            pitch_idx = axis_map[1]["imu_axis"]
                            dy = abs(entry["imu"]["euler_avg"][yaw_idx] - centry["imu"]["euler_avg"][yaw_idx])
                            dp = abs(entry["imu"]["euler_avg"][pitch_idx] - centry["imu"]["euler_avg"][pitch_idx])
                            if dy > 180: dy = 360 - dy
                            max_shift_yaw = max(max_shift_yaw, dy)
                            max_shift_pitch = max(max_shift_pitch, dp)
                            break

                coupling_entry = {
                    "sweep": sweep_s, "context": ctx_s,
                    "context_angle": ctx_a,
                    "context_offset": ctx_a - ctx_center,
                    "max_yaw_shift": round(max_shift_yaw, 2),
                    "max_pitch_shift": round(max_shift_pitch, 2),
                }
                coupling_summary.append(coupling_entry)
                print(f"    {sweep_s} + {ctx_s}={ctx_a} (offset {ctx_a-ctx_center:+.0f}): "
                      f"yaw_shift={max_shift_yaw:.1f}° pitch_shift={max_shift_pitch:.1f}°")

        analysis["cross_coupling"] = coupling_summary

    # Save analysis
    results["analysis"] = analysis
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results + analysis saved to: {output_path}")

    # --- 3e. Generate Python correction code ---
    correction_file = output_path.replace(".json", "_corrections.py")
    with open(correction_file, "w") as f:
        f.write('"""Auto-generated servo FK correction from IMU sweep data.\n\n')
        f.write(f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Source: {output_path}\n')
        f.write(f'IMU axis mapping: FK_yaw -> IMU_{axis_map[0]["imu_axis_name"]} (sign={axis_map[0]["sign"]})\n')
        f.write(f'                  FK_pitch -> IMU_{axis_map[1]["imu_axis_name"]} (sign={axis_map[1]["sign"]})\n')
        f.write('"""\n\n')

        f.write('# IMU to FK axis mapping\n')
        f.write(f'IMU_YAW_AXIS = {axis_map[0]["imu_axis"]}  # IMU euler index for FK yaw\n')
        f.write(f'IMU_YAW_SIGN = {axis_map[0]["sign"]}\n')
        f.write(f'IMU_PITCH_AXIS = {axis_map[1]["imu_axis"]}  # IMU euler index for FK pitch\n')
        f.write(f'IMU_PITCH_SIGN = {axis_map[1]["sign"]}\n\n')

        if "eye_ud_offset" in analysis:
            f.write(f'EYE_UD_OFFSET_MEASURED = {analysis["eye_ud_offset"]["measured_offset"]}\n\n')

        f.write('def correct_fk(servo_name: str, servo_angle: float,\n')
        f.write('               fk_yaw: float, fk_pitch: float) -> tuple:\n')
        f.write('    """Apply IMU-calibrated correction to FK output."""\n')

        for sname, sa in analysis.items():
            if not isinstance(sa, dict) or "correction_poly" not in sa:
                continue
            poly = sa["correction_poly"]
            center = poly["center"]
            half_range = poly["half_range"]
            yc = poly["yaw_coefficients"]
            pc = poly["pitch_coefficients"]
            f.write(f'    if servo_name == "{sname}":\n')
            f.write(f'        t = (servo_angle - {center}) / {half_range}\n')
            f.write(f'        yaw_err = {yc[0]}*t**3 + {yc[1]}*t**2 + {yc[2]}*t + {yc[3]}\n')
            f.write(f'        pitch_err = {pc[0]}*t**3 + {pc[1]}*t**2 + {pc[2]}*t + {pc[3]}\n')
            f.write(f'        return fk_yaw - yaw_err, fk_pitch - pitch_err\n')

        f.write('    return fk_yaw, fk_pitch\n\n')

        for sname, sa in analysis.items():
            if not isinstance(sa, dict) or "max_backlash" not in sa:
                continue
            f.write(f'# {sname}: yaw_err={sa["mean_yaw_error"]:.1f}° (max {sa["max_yaw_error"]:.1f}°) '
                    f'pitch_err={sa["mean_pitch_error"]:.1f}° (max {sa["max_pitch_error"]:.1f}°) '
                    f'backlash={sa["max_backlash"]:.1f}°\n')

    print(f"  Correction code saved to: {correction_file}")

    client.loop_stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
