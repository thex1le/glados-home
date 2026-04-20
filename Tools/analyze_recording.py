#!/usr/bin/env python3
"""Comprehensive recording analysis for GLaDOS tracking sessions.

Usage:
    python Tools/analyze_recording.py recordings/session_20260417_161625
    python Tools/analyze_recording.py  # analyzes most recent session
"""
import json
import os
import sys
from collections import Counter, defaultdict


def load_jsonl(path: str) -> list:
    frames = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def pct(n: int, total: int) -> str:
    return f"{n}/{total} ({n/total*100:.1f}%)" if total > 0 else "0/0"


def find_latest_session(base: str = "recordings") -> str:
    sessions = sorted([d for d in os.listdir(base)
                       if d.startswith("session_") and os.path.isdir(os.path.join(base, d))])
    if not sessions:
        print("No sessions found in recordings/")
        sys.exit(1)
    return os.path.join(base, sessions[-1])


def analyze(session_dir: str) -> None:
    print(f"\n{'='*70}")
    print(f"  RECORDING ANALYSIS: {os.path.basename(session_dir)}")
    print(f"{'='*70}")

    # Load data
    cameras = {}
    for cam in ["camera_head", "camera_left", "camera_right"]:
        path = os.path.join(session_dir, cam, "tracking.jsonl")
        if os.path.exists(path):
            cameras[cam] = load_jsonl(path)
        else:
            cameras[cam] = []

    # ============================================================
    # 1. FRAME COUNTS & FPS
    # ============================================================
    print(f"\n--- 1. FRAME COUNTS ---")
    for cam, frames in cameras.items():
        if not frames:
            print(f"  {cam}: NO DATA")
            continue
        duration = frames[-1]["wall_time"] - frames[0]["wall_time"] if len(frames) > 1 else 0
        fps = len(frames) / duration if duration > 0 else 0
        print(f"  {cam}: {len(frames)} frames, {duration:.1f}s, {fps:.1f} FPS")

    head = cameras.get("camera_head", [])
    right = cameras.get("camera_right", [])
    left = cameras.get("camera_left", [])

    # ============================================================
    # 2. PERSON DETECTION & COMPOSITE SCORES
    # ============================================================
    print(f"\n--- 2. PERSON DETECTION & COMPOSITE ---")
    for cam_name, frames in [("HEAD", head), ("RIGHT", right), ("LEFT", left)]:
        if not frames:
            continue
        person_frames = 0
        yolo_scores = []
        composite_scores = []
        pose_bonus = 0
        track_ids_present = 0
        total_dets = 0

        for fr in frames:
            t = fr.get("tracking", {})
            person = t.get("person", {})
            objs = person.get("objects", [])
            if objs:
                person_frames += 1
            for obj in objs:
                total_dets += 1
                yolo = obj.get("confidence", 0)
                comp = obj.get("composite_score", yolo)
                yolo_scores.append(yolo)
                composite_scores.append(comp)
                if comp > yolo + 0.01:
                    pose_bonus += 1
                if obj.get("track_id") is not None:
                    track_ids_present += 1

        print(f"\n  {cam_name}:")
        print(f"    Person frames:   {pct(person_frames, len(frames))}")
        print(f"    Total detections: {total_dets}")
        if yolo_scores:
            print(f"    YOLO mean:       {sum(yolo_scores)/len(yolo_scores):.3f}  "
                  f"(>0.50: {pct(sum(1 for y in yolo_scores if y >= 0.5), len(yolo_scores))})")
        if composite_scores:
            print(f"    Composite mean:  {sum(composite_scores)/len(composite_scores):.3f}  "
                  f"(>0.50: {pct(sum(1 for c in composite_scores if c >= 0.5), len(composite_scores))})")
            print(f"    Pose bonus:      {pct(pose_bonus, total_dets)}")
            print(f"    Track IDs:       {pct(track_ids_present, total_dets)}")

    # ============================================================
    # 3. FUSION STATE
    # ============================================================
    print(f"\n--- 3. FUSION STATE ---")
    diag_frames = []
    for fr in head:
        d = fr.get("tracking", {}).get("diagnostics", {})
        if d:
            diag_frames.append(d)

    if diag_frames:
        states = [d.get("fusion_state", "unknown") for d in diag_frames]
        state_counts = Counter(states)
        total = len(states)
        for state, count in state_counts.most_common():
            print(f"    {state}: {pct(count, total)}")

        # Transitions
        transitions = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
        print(f"    Transitions: {transitions}")

        # Lock durations (head_tracking streaks)
        streaks = []
        current = 0
        for s in states:
            if s == "head_tracking":
                current += 1
            else:
                if current > 0:
                    streaks.append(current)
                current = 0
        if current > 0:
            streaks.append(current)
        if streaks:
            fps_est = len(head) / max(1, head[-1]["wall_time"] - head[0]["wall_time"]) if len(head) > 1 else 15
            longest = max(streaks)
            print(f"    Head_tracking streaks: {len(streaks)}, "
                  f"longest={longest} frames ({longest/fps_est:.1f}s), "
                  f"mean={sum(streaks)/len(streaks):.0f} frames")

    # ============================================================
    # 4. WORLD ANGLES
    # ============================================================
    print(f"\n--- 4. WORLD ANGLES ---")
    world_lrs = [d.get("world_lr") for d in diag_frames if d.get("world_lr") is not None]
    world_uds = [d.get("world_ud") for d in diag_frames if d.get("world_ud") is not None]

    if world_lrs:
        print(f"    World LR: min={min(world_lrs):.1f}, max={max(world_lrs):.1f}, "
              f"mean={sum(world_lrs)/len(world_lrs):.1f}, unique={len(set(round(w,1) for w in world_lrs))}")
    if world_uds:
        print(f"    World UD: min={min(world_uds):.1f}, max={max(world_uds):.1f}, "
              f"mean={sum(world_uds)/len(world_uds):.1f}, unique={len(set(round(w,1) for w in world_uds))}")

    # Frozen check
    if world_lrs and len(set(round(w, 1) for w in world_lrs)) <= 3:
        print(f"    *** WARNING: World LR appears FROZEN ***")
    if world_uds and len(set(round(w, 1) for w in world_uds)) <= 3:
        print(f"    *** WARNING: World UD appears FROZEN ***")

    # ============================================================
    # 5. SERVO TARGETS
    # ============================================================
    print(f"\n--- 5. SERVO TARGETS ---")
    servo_names = ["head_left_right", "head_up_down", "body_left_right", "body_up_down"]
    for sname in servo_names:
        targets = [d["estimators"][sname]["target"] for d in diag_frames
                   if "estimators" in d and sname in d.get("estimators", {})]
        if targets:
            deltas = [abs(targets[i] - targets[i-1]) for i in range(1, len(targets))]
            big_deltas = sum(1 for d in deltas if d > 10)
            limit_hits = sum(1 for t in targets if t <= 1 or t >= 179)
            print(f"    {sname}: [{min(targets):.1f}, {max(targets):.1f}], "
                  f"mean_delta={sum(deltas)/len(deltas):.1f}, "
                  f"big_jumps(>10)={big_deltas}, limit_hits={limit_hits}")

    # ============================================================
    # 6. ATTENTION
    # ============================================================
    print(f"\n--- 6. ATTENTION ---")
    attention_targets = []
    for d in diag_frames:
        att = d.get("attention", {})
        target = att.get("attention_target")
        attention_targets.append(target)

    if attention_targets:
        target_counts = Counter(t for t in attention_targets if t is not None)
        none_count = sum(1 for t in attention_targets if t is None)
        for target, count in target_counts.most_common():
            print(f"    {target}: {pct(count, len(attention_targets))}")
        if none_count:
            print(f"    None: {pct(none_count, len(attention_targets))}")
        switches = sum(1 for i in range(1, len(attention_targets))
                       if attention_targets[i] != attention_targets[i-1])
        print(f"    Switches: {switches}")

    # ============================================================
    # 7. OSCILLATION
    # ============================================================
    print(f"\n--- 7. OSCILLATION ---")
    for axis_name, values in [("World LR", world_lrs), ("World UD", world_uds)]:
        if len(values) < 3:
            continue
        reversals = 0
        for i in range(2, len(values)):
            d1 = values[i-1] - values[i-2]
            d2 = values[i] - values[i-1]
            if d1 * d2 < 0 and abs(d1) > 2 and abs(d2) > 2:
                reversals += 1
        print(f"    {axis_name} reversals (>2 deg): {reversals}")

    # ============================================================
    # 8. SACCADE / COOLDOWN
    # ============================================================
    print(f"\n--- 8. SACCADE COOLDOWN ---")
    cooldowns = [d.get("head_cooldown_remaining", 0) for d in diag_frames]
    active = sum(1 for c in cooldowns if c > 0)
    print(f"    Active: {pct(active, len(cooldowns))}")
    if cooldowns:
        print(f"    Max: {max(cooldowns):.2f}s")

    # ============================================================
    # 9. IMU
    # ============================================================
    print(f"\n--- 9. IMU ---")
    gyro_mags = []
    for d in diag_frames:
        imu = d.get("imu", {})
        gyro = imu.get("gyroscope", [])
        if gyro and len(gyro) >= 3:
            from math import sqrt, degrees
            mag = sqrt(sum(g**2 for g in gyro))
            gyro_mags.append(degrees(mag))
    if gyro_mags:
        print(f"    Gyro magnitude (deg/s): mean={sum(gyro_mags)/len(gyro_mags):.1f}, "
              f"max={max(gyro_mags):.1f}, >100={sum(1 for g in gyro_mags if g > 100)}")

    # ============================================================
    # 10. TIMELINE (10-second windows)
    # ============================================================
    print(f"\n--- 10. TIMELINE (10s windows) ---")
    if head and diag_frames:
        t0 = head[0]["wall_time"]
        duration = head[-1]["wall_time"] - t0
        window = 10.0
        for win_start in range(0, int(duration), int(window)):
            win_end = win_start + window
            win_diags = [d for i, d in enumerate(diag_frames)
                         if i < len(head) and win_start <= head[i]["wall_time"] - t0 < win_end]
            if not win_diags:
                continue

            win_states = Counter(d.get("fusion_state", "?") for d in win_diags)
            dominant = win_states.most_common(1)[0][0] if win_states else "?"
            ht_pct = win_states.get("head_tracking", 0) / len(win_diags) * 100

            win_wud = [d.get("world_ud", 0) for d in win_diags if d.get("world_ud") is not None]
            wud_mean = sum(win_wud) / len(win_wud) if win_wud else 0

            # Person detection in this window
            win_head = [fr for fr in head if win_start <= fr["wall_time"] - t0 < win_end]
            person_count = sum(1 for fr in win_head
                              if fr.get("tracking", {}).get("person", {}).get("objects"))

            print(f"    {win_start:>3}-{win_end:>3}s: "
                  f"HT={ht_pct:4.0f}% "
                  f"person={pct(person_count, len(win_head)):>12s} "
                  f"wud={wud_mean:+6.1f} "
                  f"dominant={dominant}")

    # ============================================================
    # 11. SIDE CAMERA UD ESTIMATES
    # ============================================================
    print(f"\n--- 11. SIDE CAMERA UD ---")
    for cam_name, frames in [("RIGHT", right), ("LEFT", left)]:
        uds = []
        for fr in frames:
            t = fr.get("tracking", {})
            person = t.get("person", {})
            objs = person.get("objects", [])
            for obj in objs:
                box = obj.get("box", {})
                if box and "y1" in box and "y2" in box:
                    cy = (box["y1"] + box["y2"]) / 2
                    uds.append(cy)
        if uds:
            print(f"    {cam_name} person Y center: mean={sum(uds)/len(uds):.0f}/480, "
                  f"range=[{min(uds):.0f}, {max(uds):.0f}]")
        else:
            print(f"    {cam_name}: no person detections")

    print(f"\n{'='*70}")
    print(f"  END ANALYSIS")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        session = sys.argv[1]
    else:
        session = find_latest_session()
    if not os.path.isdir(session):
        print(f"Not a directory: {session}")
        sys.exit(1)
    analyze(session)
