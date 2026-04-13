"""Analyze side camera (left/right) tracking data from GLaDOS recordings."""
import json
import math
import os
from collections import defaultdict
from statistics import mean, median

SESSIONS = [
    ("session_20260412_211251", "recordings/session_20260412_211251"),
    ("session_20260412_214451", "recordings/session_20260412_214451"),
]

BASE = os.path.dirname(os.path.abspath(__file__))


def load_tracking(path: str) -> list:
    frames = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def bbox_center(box: dict) -> tuple:
    cx = (box["x1"] + box["x2"]) / 2.0
    cy = (box["y1"] + box["y2"]) / 2.0
    return cx, cy


def extract_detections(frame: dict) -> list:
    """Return list of all detection dicts from a frame's tracking data."""
    dets = []
    tracking = frame.get("tracking", {})
    for key, val in tracking.items():
        if key in ("trace_id", "ts_vision"):
            continue
        if isinstance(val, dict) and "objects" in val:
            for obj in val["objects"]:
                dets.append(obj)
    return dets


def extract_persons(frame: dict) -> list:
    tracking = frame.get("tracking", {})
    person_data = tracking.get("person", {})
    return person_data.get("objects", [])


def analyze_camera(frames: list, camera_name: str) -> dict:
    """Compute all stats for a single camera."""
    results = {}

    # 1. Basic stats
    total_frames = len(frames)
    if total_frames < 2:
        print(f"  {camera_name}: only {total_frames} frames, skipping")
        return None

    t_start = frames[0]["wall_time"]
    t_end = frames[-1]["wall_time"]
    duration = t_end - t_start
    fps = (total_frames - 1) / duration if duration > 0 else 0

    results["total_frames"] = total_frames
    results["duration_s"] = duration
    results["fps"] = fps

    # 2. Person count distribution
    person_counts = defaultdict(int)
    for f in frames:
        persons = extract_persons(f)
        n = len(persons)
        bucket = min(n, 3)  # 3+ grouped
        person_counts[bucket] += 1

    results["person_counts"] = dict(person_counts)

    # 3. Confidence stats (person detections only)
    person_confs = []
    for f in frames:
        for p in extract_persons(f):
            person_confs.append(p["confidence"])

    if person_confs:
        results["conf_mean"] = mean(person_confs)
        results["conf_median"] = median(person_confs)
        results["conf_min"] = min(person_confs)
        results["conf_max"] = max(person_confs)
        results["conf_below_040"] = sum(1 for c in person_confs if c < 0.40) / len(person_confs) * 100
        results["total_person_dets"] = len(person_confs)
    else:
        results["conf_mean"] = 0
        results["conf_median"] = 0
        results["conf_min"] = 0
        results["conf_max"] = 0
        results["conf_below_040"] = 0
        results["total_person_dets"] = 0

    # 4. Track ID stability
    track_ids = set()
    none_track_count = 0
    prev_track_id = None
    switches = 0
    for f in frames:
        persons = extract_persons(f)
        if persons:
            # Use highest confidence person
            best = max(persons, key=lambda p: p["confidence"])
            tid = best.get("track_id")
            if tid is None:
                none_track_count += 1
            else:
                track_ids.add(tid)
                if prev_track_id is not None and tid != prev_track_id:
                    switches += 1
                prev_track_id = tid

    results["unique_track_ids"] = len(track_ids)
    results["track_switches"] = switches
    results["none_track_count"] = none_track_count

    # 5. Bbox stability (highest-confidence person, frame-to-frame jumps)
    centers = []
    for f in frames:
        persons = extract_persons(f)
        if persons:
            best = max(persons, key=lambda p: p["confidence"])
            centers.append(bbox_center(best["box"]))
        else:
            centers.append(None)

    jumps = []
    for i in range(1, len(centers)):
        if centers[i] is not None and centers[i - 1] is not None:
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            jumps.append(math.sqrt(dx * dx + dy * dy))

    if jumps:
        results["bbox_jump_mean"] = mean(jumps)
        results["bbox_jump_median"] = median(jumps)
        results["bbox_jump_max"] = max(jumps)
        results["bbox_stability_pct"] = sum(1 for j in jumps if j < 20) / len(jumps) * 100
    else:
        results["bbox_jump_mean"] = 0
        results["bbox_jump_median"] = 0
        results["bbox_jump_max"] = 0
        results["bbox_stability_pct"] = 0

    # 6. Dropout rate
    zero_person_frames = sum(1 for f in frames if len(extract_persons(f)) == 0)
    results["dropout_pct"] = zero_person_frames / total_frames * 100
    results["dropout_frames"] = zero_person_frames

    # 7. Phantom objects (non-person detections)
    phantom_counts = defaultdict(int)
    for f in frames:
        for det in extract_detections(f):
            if det["name"] != "person":
                phantom_counts[det["name"]] += 1

    results["phantoms"] = dict(sorted(phantom_counts.items(), key=lambda x: -x[1])[:5])
    results["phantom_total"] = sum(phantom_counts.values())

    # 9. Timeline (10-second windows)
    window_size = 10.0
    windows = []
    w_start = t_start
    while w_start < t_end:
        w_end = w_start + window_size
        w_frames = [f for f in frames if w_start <= f["wall_time"] < w_end]
        if w_frames:
            w_persons = []
            w_confs = []
            for f in w_frames:
                ps = extract_persons(f)
                w_persons.append(len(ps))
                for p in ps:
                    w_confs.append(p["confidence"])
            windows.append({
                "start_offset": w_start - t_start,
                "n_frames": len(w_frames),
                "mean_persons": mean(w_persons),
                "mean_conf": mean(w_confs) if w_confs else 0,
                "zero_person_pct": sum(1 for c in w_persons if c == 0) / len(w_persons) * 100,
            })
        w_start = w_end

    results["windows"] = windows
    return results


def print_table(headers: list, rows: list, col_widths: list = None):
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(str(h))
            for r in rows:
                w = max(w, len(str(r[i])))
            col_widths.append(w + 2)

    header_line = "".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * sum(col_widths))
    for row in rows:
        print("".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))))


def analyze_session(session_name: str, session_path: str):
    print("=" * 80)
    print(f"SESSION: {session_name}")
    print("=" * 80)

    cameras = {}
    for cam in ["camera_left", "camera_right"]:
        path = os.path.join(BASE, session_path, cam, "tracking.jsonl")
        if not os.path.exists(path):
            print(f"  {cam}: file not found, skipping")
            continue
        frames = load_tracking(path)
        result = analyze_camera(frames, cam)
        if result:
            cameras[cam] = (frames, result)

    if not cameras:
        print("  No camera data found.\n")
        return

    # === SECTION 1: Basic Stats ===
    print("\n--- 1. BASIC STATS ---")
    headers = ["Camera", "Frames", "Duration(s)", "FPS"]
    rows = []
    for cam, (_, r) in cameras.items():
        rows.append([cam, r["total_frames"], f"{r['duration_s']:.1f}", f"{r['fps']:.2f}"])
    print_table(headers, rows)

    # === SECTION 2: Person Count Distribution ===
    print("\n--- 2. PERSON COUNT DISTRIBUTION ---")
    headers = ["Camera", "0 persons", "1 person", "2 persons", "3+ persons"]
    rows = []
    for cam, (_, r) in cameras.items():
        pc = r["person_counts"]
        rows.append([cam, pc.get(0, 0), pc.get(1, 0), pc.get(2, 0), pc.get(3, 0)])
    print_table(headers, rows)

    # === SECTION 3: Confidence Stats ===
    print("\n--- 3. PERSON CONFIDENCE STATS ---")
    headers = ["Camera", "Detections", "Mean", "Median", "Min", "Max", "% < 0.40"]
    rows = []
    for cam, (_, r) in cameras.items():
        rows.append([
            cam, r["total_person_dets"],
            f"{r['conf_mean']:.4f}", f"{r['conf_median']:.4f}",
            f"{r['conf_min']:.4f}", f"{r['conf_max']:.4f}",
            f"{r['conf_below_040']:.1f}%"
        ])
    print_table(headers, rows)

    # === SECTION 4: Track ID Stability ===
    print("\n--- 4. TRACK ID STABILITY ---")
    headers = ["Camera", "Unique IDs", "Switches", "None track_id"]
    rows = []
    for cam, (_, r) in cameras.items():
        rows.append([cam, r["unique_track_ids"], r["track_switches"], r["none_track_count"]])
    print_table(headers, rows)

    # === SECTION 5: Bbox Stability ===
    print("\n--- 5. BBOX STABILITY (person, frame-to-frame center jump in px) ---")
    headers = ["Camera", "Mean Jump", "Median Jump", "Max Jump", "% < 20px"]
    rows = []
    for cam, (_, r) in cameras.items():
        rows.append([
            cam,
            f"{r['bbox_jump_mean']:.2f}",
            f"{r['bbox_jump_median']:.2f}",
            f"{r['bbox_jump_max']:.2f}",
            f"{r['bbox_stability_pct']:.1f}%"
        ])
    print_table(headers, rows)

    # === SECTION 6: Dropout Rate ===
    print("\n--- 6. DROPOUT RATE (frames with 0 person detections) ---")
    headers = ["Camera", "Dropout Frames", "Total Frames", "Dropout %"]
    rows = []
    for cam, (_, r) in cameras.items():
        rows.append([cam, r["dropout_frames"], r["total_frames"], f"{r['dropout_pct']:.1f}%"])
    print_table(headers, rows)

    # === SECTION 7: Phantom Objects ===
    print("\n--- 7. PHANTOM OBJECTS (non-person detections, top 5) ---")
    for cam, (_, r) in cameras.items():
        print(f"\n  {cam} (total non-person: {r['phantom_total']}):")
        if r["phantoms"]:
            for name, count in r["phantoms"].items():
                print(f"    {name:20s} {count:5d}")
        else:
            print("    (none)")

    # === SECTION 8: Overlap Analysis ===
    print("\n--- 8. OVERLAP ANALYSIS (both cameras detect person in same frame) ---")
    cam_names = list(cameras.keys())
    if len(cam_names) == 2:
        frames_l, res_l = cameras[cam_names[0]]
        frames_r, res_r = cameras[cam_names[1]]

        # Index by frame_number
        left_by_frame = {f["frame_number"]: f for f in frames_l}
        right_by_frame = {f["frame_number"]: f for f in frames_r}

        common_frames = set(left_by_frame.keys()) & set(right_by_frame.keys())
        both_person = 0
        only_left = 0
        only_right = 0
        neither = 0
        center_diffs = []

        for fn in sorted(common_frames):
            lp = extract_persons(left_by_frame[fn])
            rp = extract_persons(right_by_frame[fn])
            has_l = len(lp) > 0
            has_r = len(rp) > 0

            if has_l and has_r:
                both_person += 1
                best_l = max(lp, key=lambda p: p["confidence"])
                best_r = max(rp, key=lambda p: p["confidence"])
                cl = bbox_center(best_l["box"])
                cr = bbox_center(best_r["box"])
                # These are different cameras so absolute pixel diff isn't meaningful
                # but we can note if both see detections consistently
                center_diffs.append((cl, cr))
            elif has_l:
                only_left += 1
            elif has_r:
                only_right += 1
            else:
                neither += 1

        total_common = len(common_frames)
        print(f"  Common frame numbers:     {total_common}")
        print(f"  Both detect person:       {both_person} ({both_person/total_common*100:.1f}%)" if total_common else "")
        print(f"  Only {cam_names[0]}:  {only_left} ({only_left/total_common*100:.1f}%)" if total_common else "")
        print(f"  Only {cam_names[1]}: {only_right} ({only_right/total_common*100:.1f}%)" if total_common else "")
        print(f"  Neither:                  {neither} ({neither/total_common*100:.1f}%)" if total_common else "")

        if center_diffs:
            # For each camera, check if bbox center is consistent across time
            l_xs = [c[0][0] for c in center_diffs]
            r_xs = [c[1][0] for c in center_diffs]
            print(f"\n  Left camera person X range:  {min(l_xs):.0f} - {max(l_xs):.0f} px (spread: {max(l_xs)-min(l_xs):.0f})")
            print(f"  Right camera person X range: {min(r_xs):.0f} - {max(r_xs):.0f} px (spread: {max(r_xs)-min(r_xs):.0f})")
    else:
        print("  Need exactly 2 cameras for overlap analysis.")

    # === SECTION 9: Timeline ===
    print("\n--- 9. DETECTION CONSISTENCY TIMELINE (10-second windows) ---")
    for cam, (_, r) in cameras.items():
        print(f"\n  {cam}:")
        headers = ["Window(s)", "Frames", "Mean Persons", "Mean Conf", "Zero %"]
        rows = []
        for w in r["windows"]:
            rows.append([
                f"{w['start_offset']:.0f}-{w['start_offset']+10:.0f}",
                w["n_frames"],
                f"{w['mean_persons']:.2f}",
                f"{w['mean_conf']:.4f}" if w["mean_conf"] > 0 else "N/A",
                f"{w['zero_person_pct']:.0f}%"
            ])
        print_table(["  " + headers[0]] + headers[1:], [["  " + r[0]] + r[1:] for r in rows])

    # Check for synchronized dropouts
    print("\n  SYNCHRONIZED DROPOUT CHECK:")
    if len(cam_names) == 2:
        _, res_l = cameras[cam_names[0]]
        _, res_r = cameras[cam_names[1]]
        both_windows = min(len(res_l["windows"]), len(res_r["windows"]))
        sync_drops = 0
        for i in range(both_windows):
            wl = res_l["windows"][i]
            wr = res_r["windows"][i]
            if wl["zero_person_pct"] > 50 and wr["zero_person_pct"] > 50:
                sync_drops += 1
                print(f"    Window {wl['start_offset']:.0f}-{wl['start_offset']+10:.0f}s: BOTH cameras > 50% dropout")
        if sync_drops == 0:
            print("    No synchronized dropout windows found.")

    print()


def main():
    for session_name, session_path in SESSIONS:
        analyze_session(session_name, session_path)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key questions answered:
- Are the side cameras providing stable person detections?
- How often do they lose track (dropout)?
- Are phantom detections a problem?
- Do track IDs stay consistent or churn?
- How much does the bbox jitter frame-to-frame?
""")


if __name__ == "__main__":
    main()
