"""Compare three GLaDOS recording sessions: threshold=7, threshold=15, pre-gate baseline."""

import json
import math
import os
import statistics

SESSIONS = {
    "pre-gate (211251)": "C:/Users/Ben/PycharmProjects/glados-home/recordings/session_20260412_211251/camera_head/tracking.jsonl",
    "threshold=15 (214451)": "C:/Users/Ben/PycharmProjects/glados-home/recordings/session_20260412_214451/camera_head/tracking.jsonl",
    "threshold=7 (231151)": "C:/Users/Ben/PycharmProjects/glados-home/recordings/session_20260412_231151/camera_head/tracking.jsonl",
}

FRAMES_DIR = "C:/Users/Ben/PycharmProjects/glados-home/recordings/session_20260412_231151/camera_head/frames"


def load_jsonl(path):
    frames = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def get_persons(frame):
    tracking = frame.get("tracking", {})
    person_data = tracking.get("person", {})
    return person_data.get("objects", [])


def get_diag(frame):
    return frame.get("tracking", {}).get("diagnostics")


def gyro_magnitude(imu):
    if not imu:
        return None
    g = imu.get("gyroscope")
    if not g or len(g) < 3:
        return None
    return math.sqrt(g[0]**2 + g[1]**2 + g[2]**2)


def bbox_center(box):
    return ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)


def bbox_distance(b1, b2):
    c1 = bbox_center(b1)
    c2 = bbox_center(b2)
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def analyze_session(name, frames):
    """Basic metrics for all sessions."""
    total = len(frames)
    if total < 2:
        return {"total": total}

    # Time span & FPS
    t0 = frames[0]["wall_time"]
    t1 = frames[-1]["wall_time"]
    duration = t1 - t0
    fps = (total - 1) / duration if duration > 0 else 0

    # Person detections
    dropout = 0
    confidences = []
    bbox_jumps_50 = 0
    bbox_jumps_100 = 0
    stable_frames = 0
    track_ids = set()
    prev_box = None
    phantom_count = 0

    for frame in frames:
        persons = get_persons(frame)
        tracking = frame.get("tracking", {})

        # Count non-person objects
        for key, val in tracking.items():
            if key in ("trace_id", "ts_vision", "diagnostics"):
                continue
            if key != "person" and isinstance(val, dict):
                phantom_count += val.get("count", 0)

        if not persons:
            dropout += 1
            prev_box = None
            continue

        # Use highest confidence person
        best = max(persons, key=lambda p: p.get("confidence", 0))
        confidences.append(best.get("confidence", 0))

        if "track_id" in best:
            track_ids.add(best["track_id"])

        box = best.get("box")
        if box and prev_box:
            dist = bbox_distance(box, prev_box)
            if dist > 50:
                bbox_jumps_50 += 1
            if dist > 100:
                bbox_jumps_100 += 1
            if dist < 20:
                stable_frames += 1
        elif box:
            stable_frames += 1  # first frame with box counts as stable

        prev_box = box

    frames_with_person = total - dropout
    result = {
        "total": total,
        "duration_s": round(duration, 1),
        "fps": round(fps, 1),
        "dropout_pct": round(100 * dropout / total, 1),
        "mean_confidence": round(statistics.mean(confidences), 4) if confidences else 0,
        "bbox_jumps_50": bbox_jumps_50,
        "bbox_jumps_100": bbox_jumps_100,
        "stability_pct": round(100 * stable_frames / total, 1),
        "track_id_switches": max(0, len(track_ids) - 1),
        "phantom_objects": phantom_count,
    }
    return result


def analyze_diagnostics(name, frames):
    """Diagnostic-only metrics (sessions with diagnostics field)."""
    settling_on = 0
    settling_off = 0
    fusion_states = {}
    attention_targets = []
    attention_switches = 0
    attention_reasons = {}
    prev_target = None

    head_lr_velocities = []
    imu_gyro_gate_on = []
    imu_gyro_gate_off = []
    all_gyro = []
    all_head_vel = []

    for frame in frames:
        diag = get_diag(frame)
        if not diag:
            continue

        # Settling
        settling = diag.get("head_settling")
        if settling:
            settling_on += 1
        else:
            settling_off += 1

        # Fusion state
        fs = diag.get("fusion_state", "unknown")
        fusion_states[fs] = fusion_states.get(fs, 0) + 1

        # Attention
        att = diag.get("attention", {})
        target = att.get("attention_target")
        reason = att.get("attention_reason", "")
        if target != prev_target:
            if prev_target is not None:
                attention_switches += 1
            prev_target = target
        if target:
            attention_targets.append(target)
        if reason:
            attention_reasons[reason] = attention_reasons.get(reason, 0) + 1

        # Estimator velocities
        estimators = diag.get("estimators", {})
        head_lr = estimators.get("head_left_right", {})
        vel = head_lr.get("velocity")
        if vel is not None:
            head_lr_velocities.append(abs(vel))
            all_head_vel.append(abs(vel))

        # IMU gyro
        imu = diag.get("imu", {})
        gm = gyro_magnitude(imu)
        if gm is not None:
            all_gyro.append(gm)
            if settling:
                imu_gyro_gate_on.append(gm)
            else:
                imu_gyro_gate_off.append(gm)

    total_diag = settling_on + settling_off
    result = {
        "settling_on_pct": round(100 * settling_on / total_diag, 1) if total_diag else 0,
        "settling_off_pct": round(100 * settling_off / total_diag, 1) if total_diag else 0,
        "settling_on_count": settling_on,
        "settling_off_count": settling_off,
        "fusion_states": fusion_states,
        "attention_targets_unique": list(set(attention_targets)),
        "attention_switches": attention_switches,
        "attention_reasons": attention_reasons,
        "head_lr_velocities": head_lr_velocities,
        "imu_gyro_gate_on": imu_gyro_gate_on,
        "imu_gyro_gate_off": imu_gyro_gate_off,
        "all_gyro": all_gyro,
        "all_head_vel": all_head_vel,
    }
    return result


def velocity_histogram(velocities, label):
    if not velocities:
        print(f"  No velocity data for {label}")
        return
    total = len(velocities)
    above_7 = sum(1 for v in velocities if v > 7)
    above_15 = sum(1 for v in velocities if v > 15)
    above_30 = sum(1 for v in velocities if v > 30)
    below_3 = sum(1 for v in velocities if v < 3)
    between_3_7 = sum(1 for v in velocities if 3 <= v <= 7)
    between_7_15 = sum(1 for v in velocities if 7 < v <= 15)
    between_15_30 = sum(1 for v in velocities if 15 < v <= 30)

    print(f"  {label} velocity distribution (n={total}):")
    print(f"    <3 deg/s:     {below_3:4d} ({100*below_3/total:5.1f}%)  -- essentially still")
    print(f"    3-7 deg/s:    {between_3_7:4d} ({100*between_3_7/total:5.1f}%)  -- settling")
    print(f"    7-15 deg/s:   {between_7_15:4d} ({100*between_7_15/total:5.1f}%)  -- moderate motion")
    print(f"    15-30 deg/s:  {between_15_30:4d} ({100*between_15_30/total:5.1f}%)  -- fast motion")
    print(f"    >30 deg/s:    {above_30:4d} ({100*above_30/total:5.1f}%)  -- very fast")
    print(f"    Mean: {statistics.mean(velocities):.1f}  Median: {statistics.median(velocities):.1f}  Max: {max(velocities):.1f}")


def gyro_stats(values, label):
    if not values:
        print(f"  {label}: no data")
        return
    print(f"  {label} (n={len(values)}):")
    print(f"    Mean: {statistics.mean(values):.3f} rad/s  Median: {statistics.median(values):.3f}  Max: {max(values):.3f}")
    above_1 = sum(1 for v in values if v > 1.0)
    above_05 = sum(1 for v in values if v > 0.5)
    print(f"    >0.5 rad/s: {above_05} ({100*above_05/len(values):.1f}%)  >1.0 rad/s: {above_1} ({100*above_1/len(values):.1f}%)")


def correlation(xs, ys):
    """Pearson correlation coefficient."""
    if len(xs) < 3 or len(ys) < 3:
        return None
    n = min(len(xs), len(ys))
    xs, ys = xs[:n], ys[:n]
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx)**2 for x in xs))
    dy = math.sqrt(sum((y - my)**2 for y in ys))
    if dx == 0 or dy == 0:
        return 0
    return num / (dx * dy)


def main():
    all_data = {}
    all_diag = {}

    for name, path in SESSIONS.items():
        frames = load_jsonl(path)
        all_data[name] = (frames, analyze_session(name, frames))
        # Check if diagnostics exist
        has_diag = any(get_diag(f) for f in frames)
        if has_diag:
            all_diag[name] = analyze_diagnostics(name, frames)

    # ========== PART 2: 3-session comparison table ==========
    print("=" * 100)
    print("PART 2: 3-SESSION COMPARISON TABLE")
    print("=" * 100)

    headers = list(SESSIONS.keys())
    metrics = [
        ("Total frames", "total"),
        ("Duration (s)", "duration_s"),
        ("Effective FPS", "fps"),
        ("Dropout rate (%)", "dropout_pct"),
        ("Mean person confidence", "mean_confidence"),
        ("Bbox jumps >50px", "bbox_jumps_50"),
        ("Bbox jumps >100px", "bbox_jumps_100"),
        ("Stability (<20px) %", "stability_pct"),
        ("Track ID switches", "track_id_switches"),
        ("Phantom objects", "phantom_objects"),
    ]

    col_w = 25
    print(f"\n{'Metric':<30s}", end="")
    for h in headers:
        print(f"{h:>{col_w}s}", end="")
    print()
    print("-" * (30 + col_w * len(headers)))

    for label, key in metrics:
        print(f"{label:<30s}", end="")
        for h in headers:
            _, stats = all_data[h]
            val = stats.get(key, "N/A")
            print(f"{str(val):>{col_w}s}", end="")
        print()

    # Motion gate and fusion (diag-only sessions)
    print(f"{'Motion gate ON %':<30s}", end="")
    for h in headers:
        if h in all_diag:
            print(f"{all_diag[h]['settling_on_pct']:>{col_w}.1f}", end="")
        else:
            print(f"{'N/A':>{col_w}s}", end="")
    print()

    print(f"{'Fusion: head_tracking %':<30s}", end="")
    for h in headers:
        if h in all_diag:
            fs = all_diag[h]["fusion_states"]
            total_fs = sum(fs.values())
            ht = fs.get("head_tracking", 0)
            pct = round(100 * ht / total_fs, 1) if total_fs else 0
            print(f"{pct:>{col_w}.1f}", end="")
        else:
            print(f"{'N/A':>{col_w}s}", end="")
    print()
    print()

    # ========== PART 1: Diagnostic comparison ==========
    print("=" * 100)
    print("PART 1: DIAGNOSTIC COMPARISON (threshold=7 vs threshold=15)")
    print("=" * 100)

    for name in ["threshold=15 (214451)", "threshold=7 (231151)"]:
        if name not in all_diag:
            continue
        d = all_diag[name]
        print(f"\n--- {name} ---")

        # 1. Motion gate ON ratio
        print(f"\n  1. Motion gate ON ratio: {d['settling_on_pct']:.1f}% "
              f"({d['settling_on_count']}/{d['settling_on_count']+d['settling_off_count']} frames)")

        # 2. Fusion state distribution
        print(f"\n  2. Fusion state distribution:")
        fs = d["fusion_states"]
        total_fs = sum(fs.values())
        for state, count in sorted(fs.items(), key=lambda x: -x[1]):
            print(f"    {state:<25s} {count:4d} ({100*count/total_fs:5.1f}%)")

        # 3. Attention target
        print(f"\n  3. Attention:")
        print(f"    Unique targets: {d['attention_targets_unique'] if d['attention_targets_unique'] else 'None'}")
        print(f"    Switches: {d['attention_switches']}")
        print(f"    Reasons: {d['attention_reasons'] if d['attention_reasons'] else 'None'}")

        # 4. Head servo velocity
        print(f"\n  4. Head LR servo velocity:")
        velocity_histogram(d["head_lr_velocities"], "Head LR")

        # 5. IMU gyro during gate-OFF
        print(f"\n  5. IMU gyro when gate OFF (frame passed through):")
        gyro_stats(d["imu_gyro_gate_off"], "Gate OFF gyro")

        # 6. IMU gyro during gate-ON
        print(f"\n  6. IMU gyro when gate ON (frame suppressed):")
        gyro_stats(d["imu_gyro_gate_on"], "Gate ON gyro")

    # 7. Estimator vs IMU correlation
    print(f"\n  7. Estimator vs IMU correlation:")
    for name in ["threshold=15 (214451)", "threshold=7 (231151)"]:
        if name not in all_diag:
            continue
        d = all_diag[name]
        # Build paired samples from frames
        frames_list = all_data[name][0]
        paired_vel = []
        paired_gyro = []
        for frame in frames_list:
            diag = get_diag(frame)
            if not diag:
                continue
            est = diag.get("estimators", {}).get("head_left_right", {})
            vel = est.get("velocity")
            imu = diag.get("imu", {})
            gm = gyro_magnitude(imu)
            if vel is not None and gm is not None:
                paired_vel.append(abs(vel))
                paired_gyro.append(gm)
        corr = correlation(paired_vel, paired_gyro)
        print(f"    {name}: r = {corr:.4f} (n={len(paired_vel)})" if corr is not None else f"    {name}: insufficient data")

    # ========== PART 3: Frame quality check ==========
    print()
    print("=" * 100)
    print("PART 3: FRAME QUALITY CHECK (session 231151)")
    print("=" * 100)

    frames_231 = all_data["threshold=7 (231151)"][0]
    check_indices = [50, 150, 250, 350]

    print(f"\n{'Index':>6s} {'FileSize':>10s} {'Settling':>10s} {'IMU Gyro':>12s} {'Person Conf':>12s} {'Persons':>8s}")
    print("-" * 70)

    for idx in check_indices:
        # Frame file
        fname = f"{idx:06d}.jpg"
        fpath = os.path.join(FRAMES_DIR, fname)
        if os.path.exists(fpath):
            fsize = os.path.getsize(fpath)
            fsize_str = f"{fsize:,d} B"
        else:
            fsize_str = "MISSING"

        # Diagnostic data
        if idx < len(frames_231):
            frame = frames_231[idx]
            diag = get_diag(frame)
            persons = get_persons(frame)
            n_persons = len(persons)
            best_conf = max((p.get("confidence", 0) for p in persons), default=0)

            if diag:
                settling = diag.get("head_settling", "?")
                imu = diag.get("imu", {})
                gm = gyro_magnitude(imu)
                gm_str = f"{gm:.3f}" if gm is not None else "N/A"
            else:
                settling = "no diag"
                gm_str = "N/A"

            print(f"{idx:>6d} {fsize_str:>10s} {str(settling):>10s} {gm_str:>12s} {best_conf:>12.4f} {n_persons:>8d}")
        else:
            print(f"{idx:>6d} {fsize_str:>10s} {'N/A':>10s} {'N/A':>12s} {'N/A':>12s} {'N/A':>8s}")

    # ========== SUMMARY ==========
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Compare the two diagnostic sessions
    d7 = all_diag.get("threshold=7 (231151)")
    d15 = all_diag.get("threshold=15 (214451)")
    s7 = all_data["threshold=7 (231151)"][1]
    s15 = all_data["threshold=15 (214451)"][1]
    s_base = all_data["pre-gate (211251)"][1]

    print("\nWhat IMPROVED (threshold=7 vs threshold=15):")
    if d7 and d15:
        delta_gate = d7["settling_on_pct"] - d15["settling_on_pct"]
        print(f"  - Motion gate ON: {d15['settling_on_pct']:.1f}% -> {d7['settling_on_pct']:.1f}% (delta {delta_gate:+.1f}pp)")

        # Check if more frames in head_tracking
        ht15 = d15["fusion_states"].get("head_tracking", 0)
        ht7 = d7["fusion_states"].get("head_tracking", 0)
        tot15 = sum(d15["fusion_states"].values())
        tot7 = sum(d7["fusion_states"].values())
        pct15 = 100 * ht15 / tot15 if tot15 else 0
        pct7 = 100 * ht7 / tot7 if tot7 else 0
        print(f"  - Head tracking fusion: {pct15:.1f}% -> {pct7:.1f}% (delta {pct7-pct15:+.1f}pp)")

    print(f"  - Dropout: {s15['dropout_pct']:.1f}% -> {s7['dropout_pct']:.1f}% (baseline: {s_base['dropout_pct']:.1f}%)")
    print(f"  - Stability: {s15['stability_pct']:.1f}% -> {s7['stability_pct']:.1f}% (baseline: {s_base['stability_pct']:.1f}%)")

    print("\nWhat got WORSE or stayed same:")
    print(f"  - Mean confidence: {s15['mean_confidence']:.4f} -> {s7['mean_confidence']:.4f} (baseline: {s_base['mean_confidence']:.4f})")
    print(f"  - Bbox jumps >50px: {s15['bbox_jumps_50']} -> {s7['bbox_jumps_50']} (baseline: {s_base['bbox_jumps_50']})")
    print(f"  - Bbox jumps >100px: {s15['bbox_jumps_100']} -> {s7['bbox_jumps_100']} (baseline: {s_base['bbox_jumps_100']})")

    print("\nWhat to try next:")
    if d7:
        gate_on = d7["settling_on_pct"]
        gyro_on = d7["imu_gyro_gate_on"]
        gyro_off = d7["imu_gyro_gate_off"]
        mean_gyro_on = statistics.mean(gyro_on) if gyro_on else 0
        mean_gyro_off = statistics.mean(gyro_off) if gyro_off else 0

        if gate_on > 50:
            print(f"  - Gate is still ON {gate_on:.0f}% of the time. Consider:")
            print(f"    * Lower threshold further (try 5 deg/s)")
            print(f"    * Use IMU gyro directly instead of estimator velocity")
            print(f"    * Mean gyro when gated: {mean_gyro_on:.3f} rad/s -- many of these frames may be settled")
        if mean_gyro_on < 0.3:
            print(f"  - Mean IMU gyro during suppressed frames is only {mean_gyro_on:.3f} rad/s")
            print(f"    This suggests over-suppression -- the head IS settled but estimator velocity lags")
            print(f"    * Consider using IMU gyro < 0.3 rad/s as the settling criterion instead")
        if gyro_off:
            high_gyro_passthrough = sum(1 for g in gyro_off if g > 1.0)
            if high_gyro_passthrough > 0:
                print(f"  - {high_gyro_passthrough} frames passed through gate with gyro > 1.0 rad/s (blurry risk)")
                print(f"    * These are frames where estimator said 'settled' but head was still moving")


if __name__ == "__main__":
    main()
