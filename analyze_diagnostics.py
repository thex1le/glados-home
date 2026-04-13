"""Deep diagnostic analysis of GLaDOS tracking recordings."""
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RECORDINGS = {
    "session_214451 (diagnostics)": Path("recordings/session_20260412_214451/camera_head/tracking.jsonl"),
    "session_211251": Path("recordings/session_20260412_211251/camera_head/tracking.jsonl"),
    "session_175514 (oldest)": Path("recordings/session_20260412_175514/camera_head/tracking.jsonl"),
}

def load_frames(path):
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames

def get_persons(frame):
    tracking = frame.get("tracking", {})
    persons = tracking.get("person", {}).get("objects", [])
    return persons

def get_diagnostics(frame):
    return frame.get("tracking", {}).get("diagnostics", None)

def gyro_magnitude(gyro):
    if not gyro or len(gyro) < 3:
        return None
    return math.sqrt(sum(g*g for g in gyro))

def bbox_center(box):
    return ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)

# ============================================================
# Part 1: Deep diagnostic analysis (latest recording only)
# ============================================================
def analyze_diagnostics(frames):
    print("=" * 80)
    print("PART 1: DIAGNOSTIC ANALYSIS (session_214451)")
    print("=" * 80)
    print(f"Total frames: {len(frames)}")

    # Collect diagnostic data
    gate_on_gyros = []     # gyro mag when head_settling=True
    gate_off_gyros = []    # gyro mag when head_settling=False
    false_negatives = 0    # high gyro but gate OFF
    false_positives = 0    # low gyro but gate ON

    est_hlr_vels = []      # estimator head_left_right velocity
    imu_gyro_y_degs = []   # IMU gyro Y in deg/s

    low_conf_gyros = []    # gyro mag when person confidence < 0.3
    high_conf_gyros = []   # gyro mag when person confidence > 0.7
    no_person_gyros = []   # gyro mag when no person detected

    fusion_states = []
    fusion_transitions = 0
    prev_fusion = None

    attention_targets = []
    attention_reasons = []
    attention_switches = 0
    prev_attention = None
    attention_hold_durations = []
    current_hold_start = 0

    calibrations = {"sys": [], "gyro": [], "accel": [], "mag": []}

    settling_frames = 0

    for i, frame in enumerate(frames):
        diag = get_diagnostics(frame)
        if not diag:
            continue

        imu = diag.get("imu", {})
        gyro = imu.get("gyroscope", [0, 0, 0])
        gmag = gyro_magnitude(gyro)

        head_settling = diag.get("head_settling", False)

        # 1. Motion gate accuracy
        if head_settling:
            settling_frames += 1
            if gmag is not None:
                gate_on_gyros.append(gmag)
                if gmag < 0.1:
                    false_positives += 1
        else:
            if gmag is not None:
                gate_off_gyros.append(gmag)
                if gmag > 0.5:
                    false_negatives += 1

        # 2. Estimator vs IMU
        estimators = diag.get("estimators", {})
        hlr = estimators.get("head_left_right", {})
        if "velocity" in hlr and gyro and len(gyro) >= 2:
            est_hlr_vels.append(hlr["velocity"])
            imu_gyro_y_degs.append(gyro[1] * 57.2958)  # rad/s to deg/s

        # 3. IMU during blurred frames
        persons = get_persons(frame)
        if persons:
            max_conf = max(p.get("confidence", 0) for p in persons)
            if max_conf < 0.3 and gmag is not None:
                low_conf_gyros.append(gmag)
            if max_conf > 0.7 and gmag is not None:
                high_conf_gyros.append(gmag)
        else:
            if gmag is not None:
                no_person_gyros.append(gmag)

        # 4. Fusion state
        fs = diag.get("fusion_state", "unknown")
        fusion_states.append(fs)
        if prev_fusion is not None and fs != prev_fusion:
            fusion_transitions += 1
        prev_fusion = fs

        # 5. Attention
        att = diag.get("attention", {})
        target = att.get("attention_target")
        reason = att.get("attention_reason", "")
        attention_targets.append(target)
        if reason:
            attention_reasons.append(reason)

        if prev_attention is not None and target != prev_attention:
            attention_switches += 1
            if current_hold_start > 0:
                hold = i - current_hold_start
                attention_hold_durations.append(hold)
            current_hold_start = i
        elif prev_attention is None and target is not None:
            current_hold_start = i
        prev_attention = target

        # 6. Calibration
        cal = imu.get("calibration", [0, 0, 0, 0])
        if cal and len(cal) >= 4:
            calibrations["sys"].append(cal[0])
            calibrations["gyro"].append(cal[1])
            calibrations["accel"].append(cal[2])
            calibrations["mag"].append(cal[3])

    # Close last attention hold
    if current_hold_start > 0:
        attention_hold_durations.append(len(frames) - current_hold_start)

    # --- Report ---

    print("\n--- 1. MOTION GATE ACCURACY ---")
    print(f"Frames with head_settling=True:  {settling_frames} ({100*settling_frames/len(frames):.1f}%)")
    print(f"Frames with head_settling=False: {len(frames)-settling_frames} ({100*(len(frames)-settling_frames)/len(frames):.1f}%)")
    if gate_on_gyros:
        print(f"\nGyro magnitude when gate ON  (settling): mean={statistics.mean(gate_on_gyros):.3f}, "
              f"median={statistics.median(gate_on_gyros):.3f}, max={max(gate_on_gyros):.3f} rad/s")
    if gate_off_gyros:
        print(f"Gyro magnitude when gate OFF (tracking): mean={statistics.mean(gate_off_gyros):.3f}, "
              f"median={statistics.median(gate_off_gyros):.3f}, max={max(gate_off_gyros):.3f} rad/s")
    print(f"\nFalse negatives (gyro > 0.5 rad/s but gate OFF): {false_negatives}")
    print(f"False positives (gyro < 0.1 rad/s but gate ON):  {false_positives}")
    total_gate_decisions = len(gate_on_gyros) + len(gate_off_gyros)
    if total_gate_decisions:
        accuracy = 100 * (1 - (false_negatives + false_positives) / total_gate_decisions)
        print(f"Gate accuracy: {accuracy:.1f}%")

    print("\n--- 2. ESTIMATOR vs IMU CORRELATION ---")
    if est_hlr_vels and imu_gyro_y_degs:
        n = len(est_hlr_vels)
        mean_est = statistics.mean(est_hlr_vels)
        mean_imu = statistics.mean(imu_gyro_y_degs)

        # Pearson correlation
        if n > 2:
            std_est = statistics.stdev(est_hlr_vels)
            std_imu = statistics.stdev(imu_gyro_y_degs)
            if std_est > 0 and std_imu > 0:
                cov = sum((e - mean_est) * (i - mean_imu) for e, i in zip(est_hlr_vels, imu_gyro_y_degs)) / (n - 1)
                r = cov / (std_est * std_imu)
                print(f"Pearson correlation (estimator vel vs IMU gyro Y): r = {r:.4f}")
            else:
                print("One signal has zero variance, cannot compute correlation")

        print(f"Estimator velocity: mean={mean_est:.2f}, stdev={statistics.stdev(est_hlr_vels):.2f} deg/s")
        print(f"IMU gyro Y (deg/s): mean={mean_imu:.2f}, stdev={statistics.stdev(imu_gyro_y_degs):.2f} deg/s")

        # Disagreement analysis: frames where signs differ by a lot
        big_disagree = 0
        for ev, iv in zip(est_hlr_vels, imu_gyro_y_degs):
            if abs(ev) > 20 and abs(iv) > 20 and (ev * iv < 0):  # opposite signs, both significant
                big_disagree += 1
        print(f"Frames with significant sign disagreement (|v|>20, opposite signs): {big_disagree}/{n}")

    print("\n--- 3. IMU DURING BLURRED FRAMES ---")
    print(f"Low confidence (<0.3) frames:  {len(low_conf_gyros)}")
    if low_conf_gyros:
        print(f"  Gyro magnitude: mean={statistics.mean(low_conf_gyros):.3f}, "
              f"median={statistics.median(low_conf_gyros):.3f}, max={max(low_conf_gyros):.3f} rad/s")
    print(f"High confidence (>0.7) frames: {len(high_conf_gyros)}")
    if high_conf_gyros:
        print(f"  Gyro magnitude: mean={statistics.mean(high_conf_gyros):.3f}, "
              f"median={statistics.median(high_conf_gyros):.3f}, max={max(high_conf_gyros):.3f} rad/s")
    print(f"No-person frames:              {len(no_person_gyros)}")
    if no_person_gyros:
        print(f"  Gyro magnitude: mean={statistics.mean(no_person_gyros):.3f}, "
              f"median={statistics.median(no_person_gyros):.3f}, max={max(no_person_gyros):.3f} rad/s")

    if low_conf_gyros and high_conf_gyros:
        ratio = statistics.mean(low_conf_gyros) / statistics.mean(high_conf_gyros) if statistics.mean(high_conf_gyros) > 0 else float('inf')
        print(f"\n  Low-conf/High-conf gyro ratio: {ratio:.2f}x")
        if ratio > 2:
            print("  --> Low-confidence detections strongly correlate with head motion (blur)")
        elif ratio > 1.3:
            print("  --> Moderate correlation between low confidence and head motion")
        else:
            print("  --> Low confidence does NOT correlate strongly with head motion")

    print("\n--- 4. FUSION STATE TIMELINE ---")
    fs_counts = Counter(fusion_states)
    total_fs = len(fusion_states)
    for state, count in sorted(fs_counts.items(), key=lambda x: -x[1]):
        print(f"  {state:25s}: {count:4d} frames ({100*count/total_fs:.1f}%)")
    print(f"  State transitions: {fusion_transitions}")

    # Show timeline summary (first 20 transitions)
    print("\n  Fusion state transitions (first 20):")
    prev = None
    trans_count = 0
    for i, fs in enumerate(fusion_states):
        if fs != prev:
            if prev is not None:
                trans_count += 1
                if trans_count <= 20:
                    print(f"    Frame {i:4d}: {prev} -> {fs}")
            prev = fs

    print("\n--- 5. ATTENTION TARGET STABILITY ---")
    target_counts = Counter(attention_targets)
    print(f"  Unique targets: {len(target_counts)}")
    for target, count in sorted(target_counts.items(), key=lambda x: -x[1]):
        print(f"    {str(target):20s}: {count:4d} frames ({100*count/total_fs:.1f}%)")
    print(f"  Attention switches: {attention_switches}")
    if attention_hold_durations:
        print(f"  Hold durations (frames): mean={statistics.mean(attention_hold_durations):.1f}, "
              f"median={statistics.median(attention_hold_durations):.1f}, "
              f"min={min(attention_hold_durations)}, max={max(attention_hold_durations)}")
    reason_counts = Counter(attention_reasons)
    if reason_counts:
        print(f"  Attention reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason:30s}: {count}")

    print("\n--- 6. IMU CALIBRATION QUALITY ---")
    for axis, vals in calibrations.items():
        if vals:
            print(f"  {axis:6s}: mean={statistics.mean(vals):.2f}, min={min(vals)}, max={max(vals)}, "
                  f"at_zero={vals.count(0)} frames ({100*vals.count(0)/len(vals):.1f}%)")

    # Extra: estimator state summary
    print("\n--- BONUS: ESTIMATOR STATE SUMMARY ---")
    for servo_name in ["head_left_right", "head_up_down", "body_left_right", "body_up_down"]:
        positions = []
        velocities = []
        targets = []
        for frame in frames:
            diag = get_diagnostics(frame)
            if not diag:
                continue
            est = diag.get("estimators", {}).get(servo_name, {})
            if "position" in est:
                positions.append(est["position"])
                velocities.append(est["velocity"])
                targets.append(est["target"])
        if positions:
            print(f"\n  {servo_name}:")
            print(f"    Position: mean={statistics.mean(positions):.1f}, range=[{min(positions):.1f}, {max(positions):.1f}]")
            print(f"    Velocity: mean={statistics.mean(velocities):.1f}, stdev={statistics.stdev(velocities):.1f}, "
                  f"max_abs={max(abs(v) for v in velocities):.1f} deg/s")
            print(f"    Target:   mean={statistics.mean(targets):.1f}, range=[{min(targets):.1f}, {max(targets):.1f}]")
            # How much time spent at high velocity
            high_vel_frames = sum(1 for v in velocities if abs(v) > 30)
            print(f"    High velocity (>30 deg/s): {high_vel_frames} frames ({100*high_vel_frames/len(velocities):.1f}%)")


# ============================================================
# Part 2: 3-session comparison
# ============================================================
def compute_basic_metrics(frames):
    metrics = {}
    total = len(frames)

    # Dropout rate
    no_person = sum(1 for f in frames if not get_persons(f))
    metrics["total_frames"] = total
    metrics["dropout_rate"] = 100 * no_person / total if total else 0

    # Mean confidence
    confidences = []
    for f in frames:
        for p in get_persons(f):
            confidences.append(p.get("confidence", 0))
    metrics["mean_confidence"] = statistics.mean(confidences) if confidences else 0

    # Bbox jumps > 100px
    prev_centers = {}
    big_jumps = 0
    stable_frames = 0
    total_with_person = 0

    track_ids = set()
    prev_ids = set()
    id_switches = 0
    phantom_count = 0

    for f in frames:
        persons = get_persons(f)
        if not persons:
            prev_centers = {}
            prev_ids = set()
            continue

        total_with_person += 1
        current_ids = set()

        for p in persons:
            # Track phantom objects (person with very low confidence)
            if p.get("confidence", 0) < 0.2:
                phantom_count += 1

            box = p.get("box", {})
            if not box:
                continue

            cx, cy = bbox_center(box)
            name = p.get("name", "person")
            source = p.get("source", "")
            pid = f"{name}_{source}"
            current_ids.add(pid)
            track_ids.add(pid)

            if pid in prev_centers:
                px, py = prev_centers[pid]
                dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                if dist > 100:
                    big_jumps += 1
                if dist < 20:
                    stable_frames += 1

            prev_centers[pid] = (cx, cy)

        # Track ID switches
        if prev_ids and current_ids != prev_ids:
            id_switches += 1
        prev_ids = current_ids

    metrics["bbox_jumps_100px"] = big_jumps
    metrics["track_id_switches"] = id_switches
    metrics["stability_pct"] = 100 * stable_frames / total_with_person if total_with_person else 0
    metrics["phantom_count"] = phantom_count

    return metrics


def print_comparison(all_metrics):
    print("\n" + "=" * 80)
    print("PART 2: 3-SESSION COMPARISON")
    print("=" * 80)

    sessions = list(all_metrics.keys())

    # Header
    print(f"\n{'Metric':<30s}", end="")
    for s in sessions:
        label = s.split("_")[1][:6] if "_" in s else s[:15]
        print(f"  {label:>15s}", end="")
    print()
    print("-" * (30 + 17 * len(sessions)))

    metric_labels = [
        ("total_frames", "Total frames", "d"),
        ("dropout_rate", "Dropout rate (%)", ".1f"),
        ("mean_confidence", "Mean confidence", ".3f"),
        ("bbox_jumps_100px", "Bbox jumps >100px", "d"),
        ("track_id_switches", "Track ID switches", "d"),
        ("stability_pct", "Stability % (<20px)", ".1f"),
        ("phantom_count", "Phantom objects (<0.2)", "d"),
    ]

    for key, label, fmt in metric_labels:
        print(f"{label:<30s}", end="")
        for s in sessions:
            val = all_metrics[s].get(key, 0)
            print(f"  {val:>15{fmt}}", end="")
        print()


# ============================================================
# Part 3: Key findings
# ============================================================
def print_findings(frames, all_metrics):
    print("\n" + "=" * 80)
    print("PART 3: KEY FINDINGS")
    print("=" * 80)

    diag = get_diagnostics(frames[0])
    has_diag = diag is not None

    if not has_diag:
        print("No diagnostic data available.")
        return

    # Gather summary stats for findings
    gate_on_high_gyro = 0
    gate_off_high_gyro = 0
    gate_on_total = 0
    gate_off_total = 0

    low_conf_high_gyro = 0
    low_conf_total = 0

    for frame in frames:
        d = get_diagnostics(frame)
        if not d:
            continue
        imu = d.get("imu", {})
        gyro = imu.get("gyroscope", [0, 0, 0])
        gmag = gyro_magnitude(gyro)
        settling = d.get("head_settling", False)

        if settling:
            gate_on_total += 1
            if gmag and gmag > 0.3:
                gate_on_high_gyro += 1
        else:
            gate_off_total += 1
            if gmag and gmag > 0.3:
                gate_off_high_gyro += 1

        persons = get_persons(frame)
        if persons:
            max_conf = max(p.get("confidence", 0) for p in persons)
            if max_conf < 0.3:
                low_conf_total += 1
                if gmag and gmag > 0.3:
                    low_conf_high_gyro += 1

    print("""
MOTION GATE ASSESSMENT:""")
    if gate_on_total:
        pct = 100 * gate_on_high_gyro / gate_on_total
        print(f"  When gate ON (settling):  {pct:.0f}% of frames had significant gyro (>0.3 rad/s)")
    if gate_off_total:
        pct = 100 * gate_off_high_gyro / gate_off_total
        print(f"  When gate OFF (tracking): {pct:.0f}% of frames had significant gyro (>0.3 rad/s)")
        if pct > 20:
            print("  --> WARNING: Many frames are being processed while head is still moving.")
            print("     Consider raising the settling velocity threshold or adding IMU as a gate input.")
        else:
            print("  --> Gate is firing at appropriate times.")

    print("""
BLUR/CONFIDENCE ASSESSMENT:""")
    if low_conf_total:
        pct = 100 * low_conf_high_gyro / low_conf_total
        print(f"  {pct:.0f}% of low-confidence detections occur during significant head motion")
        if pct > 50:
            print("  --> Motion blur is the primary cause of low-confidence detections.")
            print("     The motion gate should suppress these frames before they reach tracking.")
        else:
            print("  --> Low confidence is NOT primarily caused by motion blur. Other factors involved.")

    print("""
CROSS-SESSION TRENDS:""")
    sessions = list(all_metrics.keys())
    dropouts = [all_metrics[s]["dropout_rate"] for s in sessions]
    confs = [all_metrics[s]["mean_confidence"] for s in sessions]
    stab = [all_metrics[s]["stability_pct"] for s in sessions]

    print(f"  Dropout rate trend:  {' -> '.join(f'{d:.1f}%' for d in dropouts)}")
    print(f"  Mean confidence:     {' -> '.join(f'{c:.3f}' for c in confs)}")
    print(f"  Stability:           {' -> '.join(f'{s:.1f}%' for s in stab)}")

    if dropouts[-1] < dropouts[0]:
        print("  --> Dropout rate improved across sessions")
    if stab[-1] > stab[0]:
        print("  --> Tracking stability improved across sessions")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Load latest recording for diagnostic analysis
    latest_path = RECORDINGS["session_214451 (diagnostics)"]
    latest_frames = load_frames(latest_path)

    # Part 1
    analyze_diagnostics(latest_frames)

    # Part 2: Load all sessions
    all_metrics = {}
    for name, path in RECORDINGS.items():
        frames = load_frames(path)
        all_metrics[name] = compute_basic_metrics(frames)

    print_comparison(all_metrics)

    # Part 3
    print_findings(latest_frames, all_metrics)
