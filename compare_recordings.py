"""Compare two GLaDOS tracking recordings side-by-side."""
import json
import statistics
import os
from collections import Counter, defaultdict


def load_frames(path):
    frames = []
    with open(path) as f:
        for line in f:
            frames.append(json.loads(line))
    return frames


def analyze(frames):
    result = {}

    # 1. Total frames and FPS
    result["total_frames"] = len(frames)
    if len(frames) > 1:
        duration = frames[-1]["wall_time"] - frames[0]["wall_time"]
        result["duration_s"] = round(duration, 2)
        result["fps"] = round(len(frames) / duration, 2) if duration > 0 else 0
    else:
        result["duration_s"] = 0
        result["fps"] = 0

    # Collect per-frame person detections and all detections
    person_counts = []
    all_confidences = []
    all_track_ids = []
    best_per_frame = []  # (cx, cy) of highest-conf person per frame
    phantom_classes = Counter()
    prev_track_id = None
    track_switches = 0
    none_track_count = 0
    unique_track_ids = set()

    for frame in frames:
        tracking = frame.get("tracking", {})

        # Person detections
        person_data = tracking.get("person", {})
        person_objs = person_data.get("objects", [])
        person_counts.append(len(person_objs))

        # Confidence and track IDs
        best_conf = -1
        best_box = None
        for obj in person_objs:
            conf = obj.get("confidence", 0)
            all_confidences.append(conf)
            tid = obj.get("track_id")
            if tid is None:
                none_track_count += 1
            else:
                unique_track_ids.add(tid)
            if conf > best_conf:
                best_conf = conf
                best_box = obj.get("box")
                best_tid = tid

        if best_box:
            cx = (best_box["x1"] + best_box["x2"]) / 2
            cy = (best_box["y1"] + best_box["y2"]) / 2
            best_per_frame.append((cx, cy, best_tid))
        else:
            best_per_frame.append(None)

        # Track ID switches
        if person_objs:
            current_tid = person_objs[0].get("track_id") if len(person_objs) == 1 else max(person_objs, key=lambda o: o.get("confidence", 0)).get("track_id")
            if prev_track_id is not None and current_tid is not None and current_tid != prev_track_id:
                track_switches += 1
            prev_track_id = current_tid

        # Phantom (non-person) detections
        for class_name, class_data in tracking.items():
            if class_name in ("trace_id", "ts_vision", "person"):
                continue
            if isinstance(class_data, dict) and "objects" in class_data:
                count = len(class_data["objects"])
                phantom_classes[class_name] += count

    # 2. Person count distribution
    count_dist = Counter(person_counts)
    result["person_dist"] = {k: count_dist.get(k, 0) for k in range(4)}
    result["person_dist"]["3+"] = sum(v for k, v in count_dist.items() if k >= 3)

    # 3. Dropout rate
    zero_frames = count_dist.get(0, 0)
    result["dropout_rate"] = round(100 * zero_frames / len(frames), 2) if frames else 0
    result["dropout_frames"] = zero_frames

    # 4. Confidence stats
    if all_confidences:
        result["conf_mean"] = round(statistics.mean(all_confidences), 4)
        result["conf_median"] = round(statistics.median(all_confidences), 4)
        result["conf_min"] = round(min(all_confidences), 4)
        result["conf_max"] = round(max(all_confidences), 4)
        result["conf_below_040"] = round(100 * sum(1 for c in all_confidences if c < 0.40) / len(all_confidences), 2)
        result["total_detections"] = len(all_confidences)
    else:
        result["conf_mean"] = result["conf_median"] = result["conf_min"] = result["conf_max"] = 0
        result["conf_below_040"] = 0
        result["total_detections"] = 0

    # 5. Track ID stability
    result["unique_track_ids"] = len(unique_track_ids)
    result["none_track_count"] = none_track_count
    result["track_switches"] = track_switches

    # 6. Bbox stability
    jumps = []
    for i in range(1, len(best_per_frame)):
        if best_per_frame[i] is not None and best_per_frame[i - 1] is not None:
            dx = best_per_frame[i][0] - best_per_frame[i - 1][0]
            dy = best_per_frame[i][1] - best_per_frame[i - 1][1]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            jumps.append(dist)

    if jumps:
        result["bbox_mean_jump"] = round(statistics.mean(jumps), 2)
        result["bbox_median_jump"] = round(statistics.median(jumps), 2)
        result["bbox_max_jump"] = round(max(jumps), 2)
        result["bbox_jumps_gt50"] = sum(1 for j in jumps if j > 50)
        result["bbox_jumps_gt100"] = sum(1 for j in jumps if j > 100)
    else:
        result["bbox_mean_jump"] = result["bbox_median_jump"] = result["bbox_max_jump"] = 0
        result["bbox_jumps_gt50"] = result["bbox_jumps_gt100"] = 0

    # 7. Phantom objects
    result["phantom_total"] = sum(phantom_classes.values())
    result["phantom_top5"] = phantom_classes.most_common(5)

    # 8. Longest dropout streak
    max_streak = 0
    current_streak = 0
    for pc in person_counts:
        if pc == 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    result["longest_dropout_streak"] = max_streak

    # 9. Stability score (< 20px movement)
    stable_count = sum(1 for j in jumps if j < 20)
    result["stability_score"] = round(100 * stable_count / len(jumps), 2) if jumps else 0

    return result


def fmt_change(old_val, new_val, lower_is_better=True):
    """Format a change indicator."""
    if isinstance(old_val, str) or isinstance(new_val, str):
        return ""
    diff = new_val - old_val
    if abs(diff) < 0.005:
        return "  ~"
    if lower_is_better:
        arrow = " v" if diff < 0 else " ^"
        color = "OK" if diff < 0 else "!!"
    else:
        arrow = " ^" if diff > 0 else " v"
        color = "OK" if diff > 0 else "!!"
    if isinstance(diff, float):
        return f"{color} {diff:+.2f}{arrow}"
    return f"{color} {diff:+d}{arrow}"


def print_table(old, new):
    rows = [
        ("Total frames",           old["total_frames"],        new["total_frames"],        False),
        ("Duration (s)",            old["duration_s"],          new["duration_s"],          False),
        ("FPS",                     old["fps"],                 new["fps"],                 False),
        ("",                        "",                         "",                         None),
        ("-- Person Distribution --", "",                       "",                         None),
        ("  0 persons (frames)",    old["person_dist"][0],      new["person_dist"][0],      True),
        ("  1 person  (frames)",    old["person_dist"][1],      new["person_dist"][1],      False),
        ("  2 persons (frames)",    old["person_dist"][2],      new["person_dist"][2],      False),
        ("  3+ persons (frames)",   old["person_dist"]["3+"],   new["person_dist"]["3+"],   None),
        ("",                        "",                         "",                         None),
        ("-- Dropouts --",          "",                         "",                         None),
        ("Dropout rate (%)",        old["dropout_rate"],        new["dropout_rate"],        True),
        ("Dropout frames",          old["dropout_frames"],      new["dropout_frames"],      True),
        ("Longest dropout streak",  old["longest_dropout_streak"], new["longest_dropout_streak"], True),
        ("",                        "",                         "",                         None),
        ("-- Confidence --",        "",                         "",                         None),
        ("Total person detections", old["total_detections"],    new["total_detections"],    False),
        ("Conf mean",               old["conf_mean"],           new["conf_mean"],           False),
        ("Conf median",             old["conf_median"],         new["conf_median"],         False),
        ("Conf min",                old["conf_min"],            new["conf_min"],            False),
        ("Conf max",                old["conf_max"],            new["conf_max"],            False),
        ("Conf < 0.40 (%)",         old["conf_below_040"],      new["conf_below_040"],      True),
        ("",                        "",                         "",                         None),
        ("-- Track ID Stability --","",                         "",                         None),
        ("Unique track IDs",        old["unique_track_ids"],    new["unique_track_ids"],    True),
        ("Detections w/ track=None",old["none_track_count"],    new["none_track_count"],    True),
        ("Track ID switches",       old["track_switches"],      new["track_switches"],      True),
        ("",                        "",                         "",                         None),
        ("-- Bbox Stability --",    "",                         "",                         None),
        ("Mean bbox jump (px)",     old["bbox_mean_jump"],      new["bbox_mean_jump"],      True),
        ("Median bbox jump (px)",   old["bbox_median_jump"],    new["bbox_median_jump"],    True),
        ("Max bbox jump (px)",      old["bbox_max_jump"],       new["bbox_max_jump"],       True),
        ("Jumps > 50px",            old["bbox_jumps_gt50"],     new["bbox_jumps_gt50"],     True),
        ("Jumps > 100px",           old["bbox_jumps_gt100"],    new["bbox_jumps_gt100"],    True),
        ("Stability score (%)",     old["stability_score"],     new["stability_score"],     False),
        ("",                        "",                         "",                         None),
        ("-- Phantom Objects --",   "",                         "",                         None),
        ("Total phantom detections",old["phantom_total"],       new["phantom_total"],       True),
    ]

    print(f"{'Metric':<30} | {'OLD (175514)':>14} | {'NEW (211251)':>14} | {'Change':>14}")
    print(f"{'-'*30}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")

    for row in rows:
        metric, oval, nval, lib = row
        if lib is None or oval == "" or isinstance(oval, str):
            print(f"{metric:<30} | {'':>14} | {'':>14} | {'':>14}")
            continue
        change = fmt_change(oval, nval, lib)
        if isinstance(oval, float):
            print(f"{metric:<30} | {oval:>14.4f} | {nval:>14.4f} | {change:>14}")
        else:
            print(f"{metric:<30} | {oval:>14} | {nval:>14} | {change:>14}")

    # Phantom top 5
    print(f"\nTop 5 phantom classes (OLD): {old['phantom_top5']}")
    print(f"Top 5 phantom classes (NEW): {new['phantom_top5']}")


def check_frames(session_dir, indices):
    """Check if frame images exist and report file sizes (proxy for blur)."""
    frames_dir = os.path.join(session_dir, "camera_head", "frames")
    print(f"\n{'='*70}")
    print(f"Frame image check (NEW recording)")
    print(f"{'='*70}")
    for idx in indices:
        fname = f"{idx:06d}.jpg"
        fpath = os.path.join(frames_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  Frame {idx:>4}: {fname} exists, {size_kb:.1f} KB")
            # Note: motion blur typically reduces JPEG file size due to less detail
        else:
            print(f"  Frame {idx:>4}: {fname} NOT FOUND")


def identify_problems(new_result):
    """Identify biggest remaining problems in the NEW recording."""
    problems = []

    if new_result["dropout_rate"] > 5:
        problems.append(f"HIGH DROPOUT RATE: {new_result['dropout_rate']}% of frames have 0 person detections")
    elif new_result["dropout_rate"] > 1:
        problems.append(f"MODERATE DROPOUT RATE: {new_result['dropout_rate']}% of frames have 0 person detections")

    if new_result["longest_dropout_streak"] > 10:
        problems.append(f"LONG DROPOUT STREAK: {new_result['longest_dropout_streak']} consecutive frames with no detections")

    if new_result["conf_below_040"] > 10:
        problems.append(f"LOW CONFIDENCE: {new_result['conf_below_040']}% of detections below 0.40 confidence floor")

    if new_result["track_switches"] > 20:
        problems.append(f"TRACK INSTABILITY: {new_result['track_switches']} track ID switches (target jumping between people)")

    if new_result["bbox_jumps_gt100"] > 10:
        problems.append(f"LARGE BBOX JUMPS: {new_result['bbox_jumps_gt100']} frames with >100px bbox center jump")

    if new_result["stability_score"] < 70:
        problems.append(f"LOW STABILITY: Only {new_result['stability_score']}% of frames have <20px movement (want >70%)")

    if new_result["phantom_total"] > 100:
        problems.append(f"PHANTOM OBJECTS: {new_result['phantom_total']} non-person detections still appearing")

    if new_result["unique_track_ids"] > 10:
        problems.append(f"TRACK FRAGMENTATION: {new_result['unique_track_ids']} unique track IDs for 2 people (expect ~2-4)")

    if new_result["none_track_count"] > 50:
        problems.append(f"UNTRACKED DETECTIONS: {new_result['none_track_count']} detections with no track ID")

    return problems


def main():
    old_path = r"C:\Users\Ben\PycharmProjects\glados-home\recordings\session_20260412_175514\camera_head\tracking.jsonl"
    new_path = r"C:\Users\Ben\PycharmProjects\glados-home\recordings\session_20260412_211251\camera_head\tracking.jsonl"
    new_session = r"C:\Users\Ben\PycharmProjects\glados-home\recordings\session_20260412_211251"

    print("Loading OLD recording...")
    old_frames = load_frames(old_path)
    print("Loading NEW recording...")
    new_frames = load_frames(new_path)

    print("Analyzing OLD recording...")
    old_result = analyze(old_frames)
    print("Analyzing NEW recording...")
    new_result = analyze(new_frames)

    print(f"\n{'='*70}")
    print(f"  GLaDOS Tracking Recording Comparison")
    print(f"  OLD: session_20260412_175514 (before fixes)")
    print(f"  NEW: session_20260412_211251 (motion gate + confidence fixes)")
    print(f"{'='*70}\n")

    print_table(old_result, new_result)

    # Check frame images
    check_frames(new_session, [100, 300, 500, 700])

    # Identify problems
    problems = identify_problems(new_result)
    print(f"\n{'='*70}")
    print(f"  BIGGEST REMAINING PROBLEMS (NEW recording)")
    print(f"{'='*70}")
    if problems:
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
    else:
        print("  No major problems detected!")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    dropout_change = new_result["dropout_rate"] - old_result["dropout_rate"]
    conf_change = new_result["conf_mean"] - old_result["conf_mean"]
    stability_change = new_result["stability_score"] - old_result["stability_score"]
    phantom_change = new_result["phantom_total"] - old_result["phantom_total"]
    switch_change = new_result["track_switches"] - old_result["track_switches"]

    print(f"  Dropout rate:     {old_result['dropout_rate']:>6.2f}% -> {new_result['dropout_rate']:>6.2f}% ({dropout_change:+.2f}pp)")
    print(f"  Mean confidence:  {old_result['conf_mean']:>6.4f} -> {new_result['conf_mean']:>6.4f} ({conf_change:+.4f})")
    print(f"  Stability score:  {old_result['stability_score']:>6.2f}% -> {new_result['stability_score']:>6.2f}% ({stability_change:+.2f}pp)")
    print(f"  Phantom objects:  {old_result['phantom_total']:>6d} -> {new_result['phantom_total']:>6d} ({phantom_change:+d})")
    print(f"  Track switches:   {old_result['track_switches']:>6d} -> {new_result['track_switches']:>6d} ({switch_change:+d})")
    print(f"  Bbox mean jump:   {old_result['bbox_mean_jump']:>6.2f} -> {new_result['bbox_mean_jump']:>6.2f}px")


if __name__ == "__main__":
    main()
