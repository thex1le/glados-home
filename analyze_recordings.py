"""
Comprehensive recording analysis: 3-way comparison.
  - Latest (daytime, UD sweep range fix + saccade cooldown scoping + CUDA RTSP): session_20260413_164234
  - Baseline (pre-all-changes, daytime-ish): session_20260412_175514
  - Nighttime (before UD range fix): session_20260413_004951
"""
# builtin
import json
import os
import math
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path(r"C:\Users\Ben\PycharmProjects\glados-home\recordings")
LATEST = "session_20260413_164234"
BASELINE = "session_20260412_175514"
NIGHTTIME = "session_20260413_004951"

CAMERAS = ["camera_head", "camera_left", "camera_right"]

SESSIONS = [
    (BASELINE, "Baseline"),
    (NIGHTTIME, "Nighttime"),
    (LATEST, "Latest"),
]


def load_tracking(session: str, camera: str) -> list:
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


def bbox_center(box: dict) -> tuple:
    return ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)


def std_dev(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def analyze_camera(frames: list) -> dict:
    """Compute all per-camera metrics."""
    total = len(frames)
    if total == 0:
        return {"total_frames": 0, "duration_s": 0, "fps": 0,
                "detection_rate": 0, "mean_conf": 0, "conf_std": 0,
                "mean_conf_delta": 0, "stability_pct": 0,
                "jumps_50": 0, "jumps_100": 0,
                "unique_track_ids": 0, "track_switches": 0,
                "track_none_pct": 0, "phantom_total": 0,
                "top3_phantoms": [], "person_counts": {},
                "max_dropout_streak": 0, "person_frames": 0}

    t0 = frames[0].get("wall_time", 0)
    t1 = frames[-1].get("wall_time", 0)
    duration = t1 - t0 if t1 > t0 else 0
    fps = (total - 1) / duration if duration > 0 else 0

    person_frame_count = 0
    confidences = []
    prev_center = None
    stable_count = 0
    total_consecutive = 0
    jumps_50 = 0
    jumps_100 = 0
    all_track_ids = set()
    track_id_sequence = []
    none_track_count = 0
    phantom_classes = Counter()
    person_counts = Counter()
    max_dropout = 0
    current_dropout = 0

    for frame in frames:
        tracking = frame.get("tracking", {})
        # Count phantoms (non-person, non-metadata)
        for cls_name, cls_data in tracking.items():
            if cls_name in ("trace_id", "ts_vision", "diagnostics", "person"):
                continue
            if isinstance(cls_data, dict) and "count" in cls_data:
                cnt = cls_data["count"]
                if cnt > 0:
                    phantom_classes[cls_name] += cnt

        person_data = tracking.get("person", {})
        if not isinstance(person_data, dict):
            person_counts[0] += 1
            current_dropout += 1
            max_dropout = max(max_dropout, current_dropout)
            prev_center = None
            continue

        count = person_data.get("count", 0)
        objects = person_data.get("objects", [])

        bucket = min(count, 3)
        if bucket == 3:
            person_counts["3+"] += 1
        else:
            person_counts[bucket] += 1

        if count > 0 and objects:
            current_dropout = 0
            person_frame_count += 1
            obj = objects[0]
            conf = obj.get("confidence", 0)
            confidences.append(conf)
            box = obj["box"]
            cx, cy = bbox_center(box)

            tid = obj.get("track_id")
            if tid is not None:
                all_track_ids.add(tid)
            else:
                none_track_count += 1
            track_id_sequence.append(tid)

            if prev_center is not None:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                dist = math.sqrt(dx * dx + dy * dy)
                total_consecutive += 1
                if dist < 20:
                    stable_count += 1
                if dist > 50:
                    jumps_50 += 1
                if dist > 100:
                    jumps_100 += 1
            prev_center = (cx, cy)
        else:
            current_dropout += 1
            max_dropout = max(max_dropout, current_dropout)
            prev_center = None

    detection_rate = (person_frame_count / total * 100) if total > 0 else 0
    mean_conf = (sum(confidences) / len(confidences)) if confidences else 0
    conf_std_val = std_dev(confidences) if len(confidences) > 1 else 0
    conf_deltas = [abs(confidences[i] - confidences[i - 1]) for i in range(1, len(confidences))]
    mean_conf_delta = sum(conf_deltas) / len(conf_deltas) if conf_deltas else 0
    stability_pct = (stable_count / total_consecutive * 100) if total_consecutive > 0 else 0
    track_none_pct = (none_track_count / person_frame_count * 100) if person_frame_count > 0 else 0

    # Track switches
    switches = 0
    prev_id = None
    for tid in track_id_sequence:
        if tid is not None and prev_id is not None and tid != prev_id:
            switches += 1
        if tid is not None:
            prev_id = tid

    return {
        "total_frames": total,
        "duration_s": round(duration, 1),
        "fps": round(fps, 2),
        "detection_rate": round(detection_rate, 1),
        "mean_conf": round(mean_conf, 4),
        "conf_std": round(conf_std_val, 4),
        "mean_conf_delta": round(mean_conf_delta, 4),
        "stability_pct": round(stability_pct, 1),
        "jumps_50": jumps_50,
        "jumps_100": jumps_100,
        "unique_track_ids": len(all_track_ids),
        "track_switches": switches,
        "track_none_pct": round(track_none_pct, 1),
        "phantom_total": sum(phantom_classes.values()),
        "top3_phantoms": phantom_classes.most_common(3),
        "person_counts": dict(person_counts),
        "max_dropout_streak": max_dropout,
        "person_frames": person_frame_count,
    }


def fmt_row(cols: list, widths: list) -> str:
    parts = []
    for val, w in zip(cols, widths):
        parts.append(str(val).rjust(w))
    return "  " + " | ".join(parts)


def print_comparison_table(cam_label: str, latest: dict, baseline: dict) -> None:
    """Print per-camera latest vs baseline table."""
    print(f"\n{'='*72}")
    print(f"  {cam_label}: LATEST vs BASELINE")
    print(f"{'='*72}")

    rows = [
        ("Total frames", baseline["total_frames"], latest["total_frames"]),
        ("Duration (s)", baseline["duration_s"], latest["duration_s"]),
        ("FPS", baseline["fps"], latest["fps"]),
        ("Person detect rate (%)", baseline["detection_rate"], latest["detection_rate"]),
        ("Mean person confidence", baseline["mean_conf"], latest["mean_conf"]),
        ("Bbox stability <20px (%)", baseline["stability_pct"], latest["stability_pct"]),
        ("Jumps >50px", baseline["jumps_50"], latest["jumps_50"]),
        ("Jumps >100px", baseline["jumps_100"], latest["jumps_100"]),
        ("Unique track IDs", baseline["unique_track_ids"], latest["unique_track_ids"]),
        ("Track ID switches", baseline["track_switches"], latest["track_switches"]),
        ("Track ID = None (%)", baseline["track_none_pct"], latest["track_none_pct"]),
        ("Phantom object count", baseline["phantom_total"], latest["phantom_total"]),
    ]

    w = [30, 12, 12, 10]
    print(fmt_row(["Metric", "Baseline", "Latest", "Delta"], w))
    print(f"  {'-'*68}")
    for name, bv, lv in rows:
        if isinstance(bv, float):
            delta = round(lv - bv, 4)
            ds = f"{delta:+.4f}" if "conf" in name.lower() else f"{delta:+.1f}"
        else:
            delta = lv - bv
            ds = f"{delta:+d}"
        print(fmt_row([name, bv, lv, ds], w))

    # Phantoms detail
    for label, data in [("Baseline", baseline), ("Latest", latest)]:
        phantoms = data.get("top3_phantoms", [])
        if phantoms:
            pstr = ", ".join(f"{n}({c})" for n, c in phantoms)
            print(f"  Top 3 phantoms ({label}): {pstr}")


def head_deep_dive(data: dict) -> None:
    """Metrics 8-10: head camera deep dive on latest."""
    print(f"\n{'='*72}")
    print(f"  HEAD CAMERA DEEP DIVE (Latest Only)")
    print(f"{'='*72}")

    # 8. Person count distribution
    pc = data.get("person_counts", {})
    total = data["total_frames"]
    print(f"\n  [8] Person Count Distribution:")
    for k in [0, 1, 2, "3+"]:
        cnt = pc.get(k, 0)
        pct = cnt / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"      {str(k):>3s} persons: {cnt:>5d} frames ({pct:5.1f}%) {bar}")

    # 9. Longest detection dropout streak
    print(f"\n  [9] Longest detection dropout streak: {data['max_dropout_streak']} frames")
    if data["fps"] > 0 and data["max_dropout_streak"] > 0:
        dropout_sec = data["max_dropout_streak"] / data["fps"]
        print(f"      = ~{dropout_sec:.1f} seconds at {data['fps']} fps")

    # 10. Confidence variance
    print(f"\n  [10] Confidence Stability:")
    print(f"       Std dev:              {data['conf_std']:.4f}")
    print(f"       Mean frame-to-frame:  {data['mean_conf_delta']:.4f}")
    if data["mean_conf_delta"] < 0.02:
        assessment = "STABLE (delta < 0.02)"
    elif data["mean_conf_delta"] > 0.05:
        assessment = "FLICKERY (delta > 0.05)"
    else:
        assessment = "MODERATE"
    print(f"       Assessment:           {assessment}")


def check_frame_files(session: str, indices: list) -> None:
    """Metric 11: frame file sizes."""
    print(f"\n{'='*72}")
    print(f"  HEAD CAMERA FRAME FILE SIZES (Latest)")
    print(f"{'='*72}")
    frames_dir = BASE / session / "camera_head" / "frames"
    w = [10, 15, 12, 10]
    print(fmt_row(["Index", "Filename", "Size (bytes)", "Size (KB)"], w))
    print(f"  {'-'*52}")
    for idx in indices:
        fname = f"{idx:06d}.jpg"
        fpath = frames_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(fmt_row([idx, fname, f"{size:,d}", f"{size/1024:.1f}"], w))
        else:
            print(fmt_row([idx, fname, "NOT FOUND", "-"], w))


def three_way_head_table(latest: dict, baseline: dict, nighttime: dict) -> None:
    """Metric 12: 3-way head camera comparison."""
    print(f"\n{'='*72}")
    print(f"  3-WAY HEAD CAMERA COMPARISON")
    print(f"{'='*72}")

    metrics = [
        ("Total frames", "total_frames"),
        ("Duration (s)", "duration_s"),
        ("FPS", "fps"),
        ("Detection rate (%)", "detection_rate"),
        ("Mean confidence", "mean_conf"),
        ("Confidence std", "conf_std"),
        ("Conf frame delta", "mean_conf_delta"),
        ("Bbox stability (%)", "stability_pct"),
        ("Jumps >50px", "jumps_50"),
        ("Jumps >100px", "jumps_100"),
        ("Unique track IDs", "unique_track_ids"),
        ("Track switches", "track_switches"),
        ("Track None (%)", "track_none_pct"),
        ("Phantom objects", "phantom_total"),
        ("Max dropout streak", "max_dropout_streak"),
    ]

    w = [25, 12, 12, 12]
    print(fmt_row(["Metric", "Baseline", "Nighttime", "Latest"], w))
    print(f"  {'-'*65}")
    for name, key in metrics:
        bv = baseline.get(key, 0)
        nv = nighttime.get(key, 0)
        lv = latest.get(key, 0)
        print(fmt_row([name, bv, nv, lv], w))

    # Phantoms per session
    print()
    for label, data in [("Baseline", baseline), ("Nighttime", nighttime), ("Latest", latest)]:
        phantoms = data.get("top3_phantoms", [])
        if phantoms:
            pstr = ", ".join(f"{n}({c})" for n, c in phantoms)
        else:
            pstr = "none"
        print(f"  Top phantoms ({label}): {pstr}")


def verdict(latest_h: dict, baseline_h: dict, nighttime_h: dict) -> None:
    """Final assessment."""
    print(f"\n{'='*72}")
    print(f"  VERDICT: Did the fixes help?")
    print(f"{'='*72}")

    def compare(name: str, key: str, higher_better: bool, b: dict, l: dict, label: str) -> str:
        bv = b.get(key, 0)
        lv = l.get(key, 0)
        delta = lv - bv
        if isinstance(bv, float):
            ds = f"{delta:+.4f}"
        else:
            ds = f"{delta:+d}"
        improved = (delta > 0) if higher_better else (delta < 0)
        neutral_threshold = 0.01 if isinstance(bv, float) else 1
        if abs(delta) < neutral_threshold:
            tag = "SAME"
        elif improved:
            tag = "BETTER"
        else:
            tag = "WORSE"
        return f"    {name:<28s} {label}: {ds} [{tag}]"

    key_metrics = [
        ("Detection rate", "detection_rate", True),
        ("Mean confidence", "mean_conf", True),
        ("Bbox stability", "stability_pct", True),
        ("Jumps >50px", "jumps_50", False),
        ("Jumps >100px", "jumps_100", False),
        ("Track switches", "track_switches", False),
        ("Phantom objects", "phantom_total", False),
        ("Max dropout streak", "max_dropout_streak", False),
    ]

    print(f"\n  Latest vs Baseline (head camera):")
    for name, key, hb in key_metrics:
        print(compare(name, key, hb, baseline_h, latest_h, "vs baseline"))

    print(f"\n  Latest vs Nighttime (head camera):")
    for name, key, hb in key_metrics:
        print(compare(name, key, hb, nighttime_h, latest_h, "vs night"))

    # UD oscillation comparison
    print(f"\n  UD (vertical) oscillation comparison:")
    for session, tag in SESSIONS:
        hf = load_tracking(session, "camera_head")
        by_list = []
        for f in hf:
            tracking = f.get("tracking", {})
            person_data = tracking.get("person", {})
            if isinstance(person_data, dict):
                objs = person_data.get("objects", [])
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
            rate = round(dc / len(by_list) * 100, 1)
            print(f"    {tag:<12s}: Y_std={std_dev(by_list):.1f}px, direction_changes={dc} ({rate}%)")


def main() -> None:
    print("=" * 72)
    print("  GLaDOS Recording Comparison Report")
    print(f"  Baseline:  {BASELINE} (pre-all-changes, daytime)")
    print(f"  Nighttime: {NIGHTTIME} (before UD range fix, night)")
    print(f"  Latest:    {LATEST} (UD fix + saccade cooldown + CUDA RTSP)")
    print("=" * 72)

    # Load all data
    results = {}
    for session, tag in SESSIONS:
        results[tag] = {}
        for cam in CAMERAS:
            results[tag][cam] = analyze_camera(load_tracking(session, cam))

    # Metrics 1-7: Per camera, latest vs baseline
    for cam in CAMERAS:
        cam_label = cam.replace("camera_", "").upper() + " CAMERA"
        print_comparison_table(cam_label, results["Latest"][cam], results["Baseline"][cam])

    # Metrics 8-10: Head deep dive (latest only)
    head_deep_dive(results["Latest"]["camera_head"])

    # Metric 11: Frame file sizes
    check_frame_files(LATEST, [100, 300, 500, 700])

    # Metric 12: 3-way head comparison
    three_way_head_table(
        results["Latest"]["camera_head"],
        results["Baseline"]["camera_head"],
        results["Nighttime"].get("camera_head", {"total_frames": 0}),
    )

    # Final verdict
    verdict(
        results["Latest"]["camera_head"],
        results["Baseline"]["camera_head"],
        results["Nighttime"].get("camera_head", {"total_frames": 0}),
    )


if __name__ == "__main__":
    main()
