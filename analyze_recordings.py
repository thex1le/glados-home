"""
Comprehensive recording analysis: latest (UD fix + saccade cooldown) vs baseline.
Session 20260413_004951 (latest) vs session 20260412_175514 (baseline).
"""
import json
import os
import math
from pathlib import Path
from collections import defaultdict

BASE = Path(r"C:\Users\Ben\PycharmProjects\glados-home\recordings")
LATEST = "session_20260413_004951"
BASELINE = "session_20260412_175514"

CAMERAS = ["camera_head", "camera_left", "camera_right"]


def load_tracking(session, camera):
    path = BASE / session / camera / "tracking.jsonl"
    frames = []
    if not path.exists():
        return frames
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def bbox_center(box):
    return ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)


def _std(vals):
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def analyze_camera(frames, label=""):
    """Part 1 metrics for a single camera recording."""
    total = len(frames)
    if total == 0:
        return {"label": label, "total": 0, "duration_s": 0, "fps": 0,
                "person_frames": 0, "pct_person": 0, "mean_conf": 0,
                "unique_track_ids": 0, "pct_none_track": 0, "pct_stable_20px": 0,
                "jumps_50px": 0, "jumps_100px": 0, "phantom_count": 0}

    t0 = frames[0].get("wall_time", 0)
    t1 = frames[-1].get("wall_time", 0)
    duration = t1 - t0 if t1 > t0 else 0
    fps = (total - 1) / duration if duration > 0 else 0

    person_frames = 0
    person_confs = []
    all_track_ids = set()
    none_track_ids = 0
    phantom_count = 0
    prev_center = None
    stable_count = 0
    jump_50 = 0
    jump_100 = 0
    center_movement_frames = 0

    for frame in frames:
        tracking = frame.get("tracking", {})
        has_person = False
        best_person_conf = 0
        best_person_center = None
        best_person_track_id = None

        for class_name, class_data in tracking.items():
            if class_name in ("trace_id", "ts_vision", "diagnostics"):
                continue
            if not isinstance(class_data, dict):
                continue
            objects = class_data.get("objects", [])
            for obj in objects:
                obj_name = obj.get("name", "")
                is_person = (obj_name == "person")

                if is_person:
                    has_person = True
                    conf = obj.get("confidence", 0)
                    if conf > best_person_conf:
                        best_person_conf = conf
                        best_person_center = bbox_center(obj["box"])
                        best_person_track_id = obj.get("track_id")
                else:
                    phantom_count += 1

        if has_person:
            person_frames += 1
            person_confs.append(best_person_conf)
            if best_person_track_id is not None:
                all_track_ids.add(best_person_track_id)
            else:
                none_track_ids += 1

            if prev_center is not None and best_person_center is not None:
                dx = best_person_center[0] - prev_center[0]
                dy = best_person_center[1] - prev_center[1]
                dist = math.sqrt(dx * dx + dy * dy)
                center_movement_frames += 1
                if dist < 20:
                    stable_count += 1
                if dist > 50:
                    jump_50 += 1
                if dist > 100:
                    jump_100 += 1
            prev_center = best_person_center
        else:
            prev_center = None

    pct_person = (person_frames / total * 100) if total > 0 else 0
    mean_conf = (sum(person_confs) / len(person_confs)) if person_confs else 0
    pct_none_track = (none_track_ids / person_frames * 100) if person_frames > 0 else 0
    pct_stable = (stable_count / center_movement_frames * 100) if center_movement_frames > 0 else 0

    return {
        "label": label,
        "total": total,
        "duration_s": round(duration, 1),
        "fps": round(fps, 1),
        "person_frames": person_frames,
        "pct_person": round(pct_person, 1),
        "mean_conf": round(mean_conf, 4),
        "unique_track_ids": len(all_track_ids),
        "pct_none_track": round(pct_none_track, 1),
        "pct_stable_20px": round(pct_stable, 1),
        "jumps_50px": jump_50,
        "jumps_100px": jump_100,
        "phantom_count": phantom_count,
    }


def print_table(rows, title=""):
    if title:
        print(f"\n{'='*90}")
        print(f"  {title}")
        print(f"{'='*90}")
    if not rows:
        print("  (no data)")
        return
    keys = list(rows[0].keys())
    widths = {}
    for k in keys:
        widths[k] = max(len(str(k)), max(len(str(r.get(k, ""))) for r in rows))
    header = " | ".join(str(k).rjust(widths[k]) for k in keys)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for r in rows:
        line = " | ".join(str(r.get(k, "")).rjust(widths[k]) for k in keys)
        print(f"  {line}")


def print_delta(b, l, keys):
    print(f"\n  DELTA (LATEST - BASELINE):")
    for k in keys:
        bv, lv = b.get(k, 0), l.get(k, 0)
        if isinstance(bv, float):
            delta = round(lv - bv, 2)
            sign = "+" if delta >= 0 else ""
            print(f"    {k}: {sign}{delta}")
        else:
            delta = lv - bv
            sign = "+" if delta >= 0 else ""
            print(f"    {k}: {sign}{delta}")


# ============================================================================
# PART 1
# ============================================================================
def part1():
    print("\n" + "#" * 90)
    print("# PART 1: ALL 3 CAMERAS COMPARISON (LATEST vs BASELINE)")
    print("#" * 90)

    delta_keys = ["pct_person", "mean_conf", "pct_none_track", "pct_stable_20px",
                  "jumps_50px", "jumps_100px", "phantom_count"]

    for cam in CAMERAS:
        rows = []
        for session, tag in [(BASELINE, "BASELINE"), (LATEST, "LATEST")]:
            frames = load_tracking(session, cam)
            m = analyze_camera(frames, label=tag)
            rows.append(m)
        print_table(rows, title=cam)
        if len(rows) == 2 and rows[0]["total"] > 0 and rows[1]["total"] > 0:
            print_delta(rows[0], rows[1], delta_keys)


# ============================================================================
# PART 2
# ============================================================================
def part2():
    print("\n" + "#" * 90)
    print("# PART 2: DIAGNOSTIC DEEP-DIVE (LATEST HEAD CAMERA)")
    print("#" * 90)

    frames = load_tracking(LATEST, "camera_head")
    total = len(frames)

    # 2.1 Diagnostics check
    print(f"\n--- 2.1 Diagnostics Fields ---")
    non_empty_diag = 0
    diag_keys_found = set()
    sample_diag = None
    for f in frames:
        d = f.get("tracking", {}).get("diagnostics", {})
        if d:
            non_empty_diag += 1
            diag_keys_found.update(d.keys())
            if sample_diag is None:
                sample_diag = d
    print(f"  Total frames: {total}")
    print(f"  Frames with non-empty diagnostics: {non_empty_diag} "
          f"({round(non_empty_diag/total*100,1) if total else 0}%)")
    if diag_keys_found:
        print(f"  Diagnostic keys found: {sorted(diag_keys_found)}")
        print(f"  Sample: {json.dumps(sample_diag, indent=4)[:1000]}")
    else:
        print(f"  ALL diagnostics are empty dicts.")
        print(f"  Saccade cooldown / head_settling / fusion_state are NOT recorded.")
        print(f"  The recorder does not capture internal tracker state -- only detection results.")

    # 2.2 Estimator health -- inferred from bbox movement
    print(f"\n--- 2.2 Bbox-Inferred Motion Health ---")
    person_centers_x = []
    person_centers_y = []
    velocities_x = []
    velocities_y = []
    prev_cx, prev_cy, prev_t = None, None, None
    for f in frames:
        tracking = f.get("tracking", {})
        t = f.get("wall_time", 0)
        if "person" in tracking:
            objs = tracking["person"].get("objects", [])
            if objs:
                box = objs[0]["box"]
                cx, cy = bbox_center(box)
                person_centers_x.append(cx)
                person_centers_y.append(cy)
                if prev_cx is not None and prev_t is not None:
                    dt = t - prev_t
                    if dt > 0:
                        velocities_x.append((cx - prev_cx) / dt)
                        velocities_y.append((cy - prev_cy) / dt)
                prev_cx, prev_cy, prev_t = cx, cy, t
            else:
                prev_cx, prev_cy, prev_t = None, None, None
        else:
            prev_cx, prev_cy, prev_t = None, None, None

    if person_centers_x:
        print(f"  Person center X: min={min(person_centers_x):.1f}, max={max(person_centers_x):.1f}, "
              f"mean={sum(person_centers_x)/len(person_centers_x):.1f}, std={_std(person_centers_x):.1f}")
        print(f"  Person center Y: min={min(person_centers_y):.1f}, max={max(person_centers_y):.1f}, "
              f"mean={sum(person_centers_y)/len(person_centers_y):.1f}, std={_std(person_centers_y):.1f}")
    if velocities_x:
        abs_vx = [abs(v) for v in velocities_x]
        abs_vy = [abs(v) for v in velocities_y]
        speeds = [math.sqrt(vx*vx+vy*vy) for vx, vy in zip(velocities_x, velocities_y)]
        print(f"  Speed (px/s): min={min(speeds):.1f}, max={max(speeds):.1f}, "
              f"mean={sum(speeds)/len(speeds):.1f}, median={sorted(speeds)[len(speeds)//2]:.1f}")
        extreme_v = sum(1 for s in speeds if s > 500)
        print(f"  Frames with speed > 500 px/s: {extreme_v} ({round(extreme_v/len(speeds)*100,1)}%)")
        extreme_v2 = sum(1 for s in speeds if s > 200)
        print(f"  Frames with speed > 200 px/s: {extreme_v2} ({round(extreme_v2/len(speeds)*100,1)}%)")

    # 2.3 Detection presence per camera (proxy for fusion)
    print(f"\n--- 2.3 Detection Presence Per Camera (proxy for fusion state) ---")
    for cam in CAMERAS:
        cam_frames = load_tracking(LATEST, cam)
        person_count = sum(1 for f in cam_frames if "person" in f.get("tracking", {}))
        total_cam = len(cam_frames)
        pct = round(person_count / total_cam * 100, 1) if total_cam else 0
        state = "head_tracking possible" if cam == "camera_head" and pct > 50 else ""
        if cam != "camera_head" and pct > 50:
            state = "SIDE_DRIVE possible"
        if pct < 10:
            state = "camera rarely sees person"
        print(f"  {cam}: {person_count}/{total_cam} ({pct}%) {state}")

    # 2.4 UD oscillation analysis
    print(f"\n--- 2.4 UD (Vertical) Oscillation Analysis ---")
    for session, tag in [(BASELINE, "BASELINE"), (LATEST, "LATEST")]:
        bf = load_tracking(session, "camera_head")
        by_list = []
        for f in bf:
            tracking = f.get("tracking", {})
            if "person" in tracking:
                objs = tracking["person"].get("objects", [])
                if objs:
                    by_list.append(bbox_center(objs[0]["box"])[1])

        if not by_list:
            print(f"  {tag}: no person detections")
            continue

        # Direction changes
        dc = 0
        ld = 0
        for i in range(1, len(by_list)):
            dy = by_list[i] - by_list[i - 1]
            if abs(dy) < 1:
                continue
            cd = 1 if dy > 0 else -1
            if ld != 0 and cd != ld:
                dc += 1
            ld = cd

        # Windowed Y std
        window = 30
        y_stds = []
        for i in range(0, len(by_list) - window, window):
            chunk = by_list[i:i + window]
            y_stds.append(_std(chunk))

        high_osc = sum(1 for s in y_stds if s > 20) if y_stds else 0
        mean_ystd = sum(y_stds) / len(y_stds) if y_stds else 0

        print(f"  {tag}:")
        print(f"    Person frames: {len(by_list)}")
        print(f"    Y overall std: {_std(by_list):.1f} px")
        print(f"    Y direction changes: {dc} (rate: {round(dc / len(by_list) * 100, 1)}%)")
        print(f"    Y std per {window}-frame window: mean={mean_ystd:.1f}, "
              f"max={max(y_stds):.1f}" if y_stds else "    (not enough data for windowed analysis)")
        if y_stds:
            print(f"    High-oscillation windows (std>20px): {high_osc}/{len(y_stds)}")

    # 2.5 IMU data
    print(f"\n--- 2.5 IMU Data ---")
    print(f"  IMU data is published on body/imu/status, NOT recorded in tracking.jsonl.")
    print(f"  No IMU data available in these recording files.")

    # 2.6 Attention / track ID stability
    print(f"\n--- 2.6 Track ID Stability (proxy for attention) ---")
    for session, tag in [(BASELINE, "BASELINE"), (LATEST, "LATEST")]:
        hf = load_tracking(session, "camera_head")
        track_ids = []
        for f in hf:
            tracking = f.get("tracking", {})
            if "person" in tracking:
                objs = tracking["person"].get("objects", [])
                if objs:
                    track_ids.append(objs[0].get("track_id"))
        unique = set(t for t in track_ids if t is not None)
        none_count = sum(1 for t in track_ids if t is None)
        switches = 0
        prev_id = None
        for tid in track_ids:
            if tid is not None and prev_id is not None and tid != prev_id:
                switches += 1
            if tid is not None:
                prev_id = tid
        print(f"  {tag}: unique_ids={unique}, none_count={none_count}, switches={switches}")

    # Check gesture data
    print(f"\n--- 2.7 Gesture Data (latest head only) ---")
    gesture_counts = defaultdict(int)
    gesture_frames = 0
    for f in frames:
        tracking = f.get("tracking", {})
        if "person" in tracking:
            objs = tracking["person"].get("objects", [])
            for obj in objs:
                g = obj.get("gesture")
                if g:
                    gesture_frames += 1
                    if isinstance(g, dict):
                        for gname, gval in g.items():
                            gesture_counts[gname] += 1
                    elif isinstance(g, str):
                        gesture_counts[g] += 1
    print(f"  Frames with gesture data: {gesture_frames}")
    if gesture_counts:
        for gname, count in sorted(gesture_counts.items(), key=lambda x: -x[1]):
            print(f"    {gname}: {count}")
    else:
        print(f"  No gesture detections found.")

    # Check pose source
    print(f"\n--- 2.8 Pose Source Distribution ---")
    pose_sources = defaultdict(int)
    pose_frames = 0
    no_pose_frames = 0
    for f in frames:
        tracking = f.get("tracking", {})
        if "person" in tracking:
            objs = tracking["person"].get("objects", [])
            for obj in objs:
                if "pose" in obj:
                    pose_frames += 1
                    src = obj.get("source", "unknown")
                    pose_sources[src] += 1
                else:
                    no_pose_frames += 1
    print(f"  Frames with pose: {pose_frames}, without pose: {no_pose_frames}")
    for src, count in sorted(pose_sources.items(), key=lambda x: -x[1]):
        print(f"    source={src}: {count}")


# ============================================================================
# PART 3
# ============================================================================
def part3():
    print("\n" + "#" * 90)
    print("# PART 3: VISUAL QUALITY CHECK (HEAD CAMERA FRAMES)")
    print("#" * 90)

    frames_dir = BASE / LATEST / "camera_head" / "frames"
    tracking_frames = load_tracking(LATEST, "camera_head")
    indices = [100, 300, 500, 700]

    rows = []
    for idx in indices:
        fname = f"{idx:06d}.jpg"
        fpath = frames_dir / fname
        if fpath.exists():
            size = os.path.getsize(fpath)
            size_str = f"{size:,d}"
        else:
            size_str = "MISSING"

        conf = "-"
        has_person = "no"
        classes_str = "-"
        track_id = "-"

        if idx < len(tracking_frames):
            tf = tracking_frames[idx]
            tracking = tf.get("tracking", {})
            det_classes = [k for k in tracking if k not in ("trace_id", "ts_vision", "diagnostics")]
            classes_str = ", ".join(det_classes) if det_classes else "none"
            if "person" in tracking:
                objs = tracking["person"].get("objects", [])
                if objs:
                    conf = f"{objs[0].get('confidence', 0):.4f}"
                    has_person = "YES"
                    track_id = str(objs[0].get("track_id", "None"))

        rows.append({
            "idx": idx,
            "file": fname,
            "size_bytes": size_str,
            "person": has_person,
            "conf": conf,
            "track_id": track_id,
            "classes": classes_str,
        })

    print_table(rows, title="Frame Quality Samples (latest head camera)")

    # Compare frame sizes between sessions
    print(f"\n  --- Frame Size Comparison (average over first 100 frames) ---")
    for session, tag in [(BASELINE, "BASELINE"), (LATEST, "LATEST")]:
        fdir = BASE / session / "camera_head" / "frames"
        sizes = []
        for i in range(100):
            fp = fdir / f"{i:06d}.jpg"
            if fp.exists():
                sizes.append(os.path.getsize(fp))
        if sizes:
            print(f"  {tag}: mean={sum(sizes)/len(sizes):,.0f} bytes, "
                  f"min={min(sizes):,d}, max={max(sizes):,d}, count={len(sizes)}")
        else:
            print(f"  {tag}: no frames found")


# ============================================================================
# PART 4
# ============================================================================
def part4():
    print("\n" + "#" * 90)
    print("# PART 4: SIDE CAMERA COMPARISON (LATEST vs BASELINE)")
    print("#" * 90)

    for cam in ["camera_left", "camera_right"]:
        rows = []
        for session, tag in [(BASELINE, "BASELINE"), (LATEST, "LATEST")]:
            frames = load_tracking(session, cam)
            m = analyze_camera(frames, label=tag)
            rows.append(m)
        print_table(rows, title=cam)

        if len(rows) == 2 and rows[0]["total"] > 0 and rows[1]["total"] > 0:
            b, l = rows[0], rows[1]
            print(f"\n  Reliability Assessment:")
            for tag_name, r in [("BASELINE", b), ("LATEST", l)]:
                det_ok = r["pct_person"] > 90
                conf_ok = r["mean_conf"] > 0.7
                status = "RELIABLE" if det_ok and conf_ok else "UNRELIABLE"
                det_verdict = "PASS" if det_ok else "FAIL"
                conf_verdict = "PASS" if conf_ok else "FAIL"
                print(f"    {tag_name}: detect={r['pct_person']}% (>90%: {det_verdict}), "
                      f"conf={r['mean_conf']} (>0.7: {conf_verdict}) => {status}")
            print(f"  Phantom delta: {l['phantom_count'] - b['phantom_count']:+d} "
                  f"(BASELINE={b['phantom_count']}, LATEST={l['phantom_count']})")


# ============================================================================
# SUMMARY
# ============================================================================
def summary():
    print("\n" + "#" * 90)
    print("# BRUTAL SUMMARY: DID THE UD FIX + SACCADE COOLDOWN ACTUALLY HELP?")
    print("#" * 90)

    results = {}
    for cam in CAMERAS:
        for session, tag in [(BASELINE, "BASELINE"), (LATEST, "LATEST")]:
            frames = load_tracking(session, cam)
            results[(cam, tag)] = analyze_camera(frames, label=tag)

    bh = results[("camera_head", "BASELINE")]
    lh = results[("camera_head", "LATEST")]

    print(f"\n  HEAD CAMERA METRICS:")
    metrics = [
        ("Detection rate", "pct_person", "%", True),
        ("Mean confidence", "mean_conf", "", True),
        ("Bbox stability <20px", "pct_stable_20px", "%", True),
        ("Jumps >50px", "jumps_50px", "", False),
        ("Jumps >100px", "jumps_100px", "", False),
        ("Phantoms", "phantom_count", "", False),
        ("Track ID count", "unique_track_ids", "", False),
    ]

    improvements = 0
    regressions = 0
    for label, key, unit, higher_better in metrics:
        bv = bh[key]
        lv = lh[key]
        if isinstance(bv, float):
            delta = lv - bv
            delta_str = f"{delta:+.2f}"
        else:
            delta = lv - bv
            delta_str = f"{delta:+d}"
        if higher_better:
            improved = delta > 0
        else:
            improved = delta < 0
        status = "BETTER" if improved else ("WORSE" if delta != 0 else "SAME")
        if improved:
            improvements += 1
        elif delta != 0:
            regressions += 1
        print(f"    {label:<25s}: {bv}{unit} -> {lv}{unit} ({delta_str}) [{status}]")

    # UD oscillation
    print(f"\n  UD OSCILLATION:")
    for session, tag in [(BASELINE, "BASELINE"), (LATEST, "LATEST")]:
        bf = load_tracking(session, "camera_head")
        by_list = []
        for f in bf:
            tracking = f.get("tracking", {})
            if "person" in tracking:
                objs = tracking["person"].get("objects", [])
                if objs:
                    by_list.append(bbox_center(objs[0]["box"])[1])
        if by_list:
            dc = 0
            ld = 0
            for i in range(1, len(by_list)):
                dy = by_list[i] - by_list[i - 1]
                if abs(dy) < 1:
                    continue
                cd = 1 if dy > 0 else -1
                if ld != 0 and cd != ld:
                    dc += 1
                ld = cd
            print(f"    {tag}: Y_std={_std(by_list):.1f}px, direction_changes={dc} "
                  f"({round(dc / len(by_list) * 100, 1)}%)")

    print(f"\n  SIDE CAMERAS:")
    for cam in ["camera_left", "camera_right"]:
        bc = results[(cam, "BASELINE")]
        lc = results[(cam, "LATEST")]
        det_delta = lc["pct_person"] - bc["pct_person"]
        conf_delta = lc["mean_conf"] - bc["mean_conf"]
        phantom_delta = lc["phantom_count"] - bc["phantom_count"]
        print(f"    {cam}: detect {bc['pct_person']}%->{lc['pct_person']}% ({det_delta:+.1f}), "
              f"conf {bc['mean_conf']}->{lc['mean_conf']} ({conf_delta:+.4f}), "
              f"phantoms {phantom_delta:+d}")

    print(f"\n  OVERALL VERDICT:")
    print(f"    Head camera: {improvements} metrics improved, {regressions} regressed")
    if improvements > regressions:
        print(f"    ==> Net IMPROVEMENT")
    elif regressions > improvements:
        print(f"    ==> Net REGRESSION")
    else:
        print(f"    ==> NEUTRAL / MIXED RESULTS")

    print(f"\n  IMPORTANT CAVEATS:")
    print(f"    - Diagnostics are EMPTY in both recordings. Cannot directly measure:")
    print(f"      * Saccade cooldown activation count/frequency")
    print(f"      * Fusion state (head_tracking vs SIDE_DRIVE)")
    print(f"      * Estimator position/velocity divergence")
    print(f"      * IMU readings during tracking")
    print(f"    - All analysis is from bbox/detection data only.")
    print(f"    - To get diagnostics, the MotionRecorder needs to capture")
    print(f"      VisionTracker internal state (cooldown, estimator, fusion).")
    print(f"    - Different recording durations/conditions may skew comparison.")


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
    summary()
