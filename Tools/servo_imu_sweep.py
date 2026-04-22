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

    # Verify IMU data is updating (not stale from a previous session)
    imu_received.clear()
    if not imu_received.wait(timeout=2):
        print("ERROR: IMU data is stale — received once but not updating.")
        print("The IMU thread may have crashed after startup.")
        client.loop_stop()
        sys.exit(1)

    with imu_lock:
        euler = imu_data["euler"]
    print(f"  IMU data flowing: euler={[round(e,1) for e in euler]}")
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
    # PHASE 3: ANALYSIS — Generate correction data
    # ============================================================
    print(f"\n{'='*60}")
    print("  PHASE 3: ANALYSIS")
    print(f"{'='*60}")

    analysis = {}

    # --- 3a. EYE_UD_OFFSET calibration ---
    # At center position, the FK says pitch = -22.5° (camera tilt).
    # The IMU gives the actual head orientation. The difference between
    # FK pitch and IMU pitch at center IS the true camera mounting offset.
    if center_imu and center_fk:
        imu_pitch = center_imu["euler_avg"][1]  # BNO055 pitch
        fk_pitch = center_fk["pitch"]
        eye_offset = fk_pitch - imu_pitch
        analysis["eye_ud_offset"] = {
            "imu_pitch_at_center": round(imu_pitch, 2),
            "fk_pitch_at_center": round(fk_pitch, 2),
            "measured_offset": round(eye_offset, 2),
            "current_config": 2.0,
        }
        print(f"\n  EYE_UD_OFFSET Calibration:")
        print(f"    IMU pitch at center:  {imu_pitch:+.1f}°")
        print(f"    FK pitch at center:   {fk_pitch:+.1f}°")
        print(f"    Measured offset:      {eye_offset:+.1f}°")
        print(f"    Current config:       2.0°")

    # --- 3b. Per-servo pushrod nonlinearity + FK error ---
    for servo_name, sweep_data in results["sweeps"].items():
        if not sweep_data or not sweep_data[0].get("fk"):
            continue

        # Reference IMU at center (first point should be near center)
        center_entry = None
        for e in sweep_data:
            if abs(e["angle"] - servo_defs[servo_name]["center"]) < 1:
                center_entry = e
                break
        if not center_entry:
            continue

        ref_euler = center_entry["imu"]["euler_avg"]

        # Compute FK error at each position
        errors = []
        # Separate ascending (center→max) from descending (max→min) for backlash
        ascending = []
        descending = []
        prev_angle = sweep_data[0]["angle"]
        for entry in sweep_data:
            imu_euler = entry["imu"]["euler_avg"]
            fk = entry["fk"]

            # IMU delta from center (heading and pitch)
            imu_delta_h = imu_euler[0] - ref_euler[0]
            imu_delta_p = imu_euler[1] - ref_euler[1]
            imu_delta_r = imu_euler[2] - ref_euler[2]

            # Handle heading wrap-around
            if imu_delta_h > 180:
                imu_delta_h -= 360
            elif imu_delta_h < -180:
                imu_delta_h += 360

            # FK delta from center
            fk_center = center_entry["fk"]
            fk_delta_yaw = fk["yaw"] - fk_center["yaw"]
            fk_delta_pitch = fk["pitch"] - fk_center["pitch"]

            error_entry = {
                "angle": entry["angle"],
                "imu_delta_heading": round(imu_delta_h, 2),
                "imu_delta_pitch": round(imu_delta_p, 2),
                "imu_delta_roll": round(imu_delta_r, 2),
                "fk_delta_yaw": round(fk_delta_yaw, 2),
                "fk_delta_pitch": round(fk_delta_pitch, 2),
                "error_yaw": round(fk_delta_yaw - imu_delta_h, 2),
                "error_pitch": round(fk_delta_pitch - imu_delta_p, 2),
            }
            errors.append(error_entry)

            # Track direction for backlash
            if entry["angle"] >= prev_angle:
                ascending.append(error_entry)
            else:
                descending.append(error_entry)
            prev_angle = entry["angle"]

        # Backlash: find overlapping angles in ascending vs descending
        backlash_points = []
        asc_by_angle = {e["angle"]: e for e in ascending}
        desc_by_angle = {e["angle"]: e for e in descending}
        for angle in asc_by_angle:
            if angle in desc_by_angle:
                a = asc_by_angle[angle]
                d = desc_by_angle[angle]
                bl = {
                    "angle": angle,
                    "heading_backlash": round(abs(a["imu_delta_heading"] - d["imu_delta_heading"]), 2),
                    "pitch_backlash": round(abs(a["imu_delta_pitch"] - d["imu_delta_pitch"]), 2),
                }
                backlash_points.append(bl)

        # Max FK error
        max_yaw_err = max(abs(e["error_yaw"]) for e in errors) if errors else 0
        max_pitch_err = max(abs(e["error_pitch"]) for e in errors) if errors else 0
        mean_yaw_err = sum(abs(e["error_yaw"]) for e in errors) / len(errors) if errors else 0
        mean_pitch_err = sum(abs(e["error_pitch"]) for e in errors) / len(errors) if errors else 0

        # Backlash summary
        max_backlash = max((bl["heading_backlash"] + bl["pitch_backlash"])
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

        # Try polynomial fit: servo_angle → FK_error
        # This gives us a correction function
        try:
            import numpy as np
            angles_arr = np.array([e["angle"] for e in errors])
            yaw_errors = np.array([e["error_yaw"] for e in errors])
            pitch_errors = np.array([e["error_pitch"] for e in errors])

            # Fit 3rd-degree polynomial: correction = poly(servo_angle)
            # Normalize angles to [-1, 1] for numerical stability
            center = servo_defs[servo_name]["center"]
            half_range = max(servo_defs[servo_name]["max"] - center,
                            center - servo_defs[servo_name]["min"])
            angles_norm = (angles_arr - center) / half_range

            yaw_poly = np.polyfit(angles_norm, yaw_errors, 3)
            pitch_poly = np.polyfit(angles_norm, pitch_errors, 3)

            # Compute R² to see how well the poly fits
            yaw_pred = np.polyval(yaw_poly, angles_norm)
            pitch_pred = np.polyval(pitch_poly, angles_norm)
            yaw_ss_res = np.sum((yaw_errors - yaw_pred) ** 2)
            yaw_ss_tot = np.sum((yaw_errors - np.mean(yaw_errors)) ** 2)
            pitch_ss_res = np.sum((pitch_errors - pitch_pred) ** 2)
            pitch_ss_tot = np.sum((pitch_errors - np.mean(pitch_errors)) ** 2)
            yaw_r2 = 1 - yaw_ss_res / yaw_ss_tot if yaw_ss_tot > 0 else 0
            pitch_r2 = 1 - pitch_ss_res / pitch_ss_tot if pitch_ss_tot > 0 else 0

            servo_analysis["correction_poly"] = {
                "yaw_coefficients": [round(c, 6) for c in yaw_poly.tolist()],
                "pitch_coefficients": [round(c, 6) for c in pitch_poly.tolist()],
                "center": center,
                "half_range": half_range,
                "yaw_r_squared": round(yaw_r2, 4),
                "pitch_r_squared": round(pitch_r2, 4),
                "usage": "correction = poly(((servo_angle - center) / half_range))",
            }
            print(f"\n  {servo_name} correction polynomial:")
            print(f"    Yaw:   coeffs={[round(c,3) for c in yaw_poly]} R²={yaw_r2:.3f}")
            print(f"    Pitch: coeffs={[round(c,3) for c in pitch_poly]} R²={pitch_r2:.3f}")
        except ImportError:
            print(f"\n  {servo_name}: numpy not available, skipping polynomial fit")

        analysis[servo_name] = servo_analysis

        print(f"\n  {servo_name} FK Error:")
        print(f"    Max yaw error:   {max_yaw_err:+.1f}°")
        print(f"    Max pitch error: {max_pitch_err:+.1f}°")
        print(f"    Mean yaw error:  {mean_yaw_err:.1f}°")
        print(f"    Mean pitch error:{mean_pitch_err:.1f}°")
        print(f"    Max backlash:    {max_backlash:.1f}°")

    # --- 3c. Cross-coupling analysis ---
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

            # Find the same sweep at context=center for comparison
            center_sweep = None
            for other in results["interactions"]:
                if (other["sweep_servo"] == sweep_s and
                        other["context_servo"] == ctx_s and
                        abs(other["context_angle"] - ctx_center) < 1):
                    center_sweep = other["data"]
                    break

            if center_sweep and ctx_a != ctx_center:
                # Compare: how much did changing the context servo shift the IMU?
                max_shift_h = 0
                max_shift_p = 0
                for entry in data:
                    # Find matching angle in center sweep
                    for centry in center_sweep:
                        if abs(centry["angle"] - entry["angle"]) < 1:
                            dh = abs(entry["imu"]["euler_avg"][0] - centry["imu"]["euler_avg"][0])
                            dp = abs(entry["imu"]["euler_avg"][1] - centry["imu"]["euler_avg"][1])
                            if dh > 180:
                                dh = 360 - dh
                            max_shift_h = max(max_shift_h, dh)
                            max_shift_p = max(max_shift_p, dp)
                            break

                coupling_entry = {
                    "sweep": sweep_s,
                    "context": ctx_s,
                    "context_angle": ctx_a,
                    "context_offset_from_center": ctx_a - ctx_center,
                    "max_heading_shift": round(max_shift_h, 2),
                    "max_pitch_shift": round(max_shift_p, 2),
                }
                coupling_summary.append(coupling_entry)
                print(f"    {sweep_s} + {ctx_s}={ctx_a} (offset {ctx_a-ctx_center:+.0f}): "
                      f"max shift H={max_shift_h:.1f}° P={max_shift_p:.1f}°")

        analysis["cross_coupling"] = coupling_summary

    # Save analysis alongside raw data
    results["analysis"] = analysis

    # Re-save with analysis
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results + analysis saved to: {output_path}")

    # --- 3d. Generate Python correction code ---
    correction_file = output_path.replace(".json", "_corrections.py")
    with open(correction_file, "w") as f:
        f.write('"""Auto-generated servo FK correction from IMU sweep data.\n\n')
        f.write(f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Source: {output_path}\n')
        f.write('"""\n\n')
        f.write('def correct_fk(servo_name: str, servo_angle: float,\n')
        f.write('               fk_yaw: float, fk_pitch: float) -> tuple:\n')
        f.write('    """Apply IMU-calibrated correction to FK output.\n\n')
        f.write('    Args:\n')
        f.write('        servo_name: Which servo is being evaluated.\n')
        f.write('        servo_angle: Current servo angle in degrees.\n')
        f.write('        fk_yaw: FK-computed yaw.\n')
        f.write('        fk_pitch: FK-computed pitch.\n\n')
        f.write('    Returns:\n')
        f.write('        Corrected (yaw, pitch).\n')
        f.write('    """\n')

        for servo_name, sa in analysis.items():
            if not isinstance(sa, dict) or "correction_poly" not in sa:
                continue
            poly = sa["correction_poly"]
            center = poly["center"]
            half_range = poly["half_range"]
            yc = poly["yaw_coefficients"]
            pc = poly["pitch_coefficients"]
            f.write(f'    if servo_name == "{servo_name}":\n')
            f.write(f'        t = (servo_angle - {center}) / {half_range}\n')
            f.write(f'        yaw_err = {yc[0]}*t**3 + {yc[1]}*t**2 + {yc[2]}*t + {yc[3]}\n')
            f.write(f'        pitch_err = {pc[0]}*t**3 + {pc[1]}*t**2 + {pc[2]}*t + {pc[3]}\n')
            f.write(f'        return fk_yaw - yaw_err, fk_pitch - pitch_err\n')

        f.write('    return fk_yaw, fk_pitch\n')
        f.write('\n\n')

        # Also output summary constants
        f.write('# Summary\n')
        if "eye_ud_offset" in analysis:
            f.write(f'EYE_UD_OFFSET_MEASURED = {analysis["eye_ud_offset"]["measured_offset"]}\n')
        for servo_name, sa in analysis.items():
            if not isinstance(sa, dict) or "max_backlash" not in sa:
                continue
            f.write(f'# {servo_name}: max_yaw_err={sa["max_yaw_error"]}° '
                    f'max_pitch_err={sa["max_pitch_error"]}° '
                    f'backlash={sa["max_backlash"]}°\n')

    print(f"  Correction code saved to: {correction_file}")

    client.loop_stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
