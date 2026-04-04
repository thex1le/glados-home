#!/usr/bin/env python3
"""Replay recorded video sessions through the GLaDOS tracking pipeline.

Modes:
  Full replay:  Feed recorded frames through YOLO + tracking (requires GPU)
  Math-only:    Re-run tracking math with different parameters (no GPU needed)
  Compare:      Show per-frame diff against original tracking data

Usage:
    # Full replay with YOLO (requires GPU)
    python3 Tools/replay_video.py --session recordings/session_.../ --config glog.conf

    # Math-only with different smoothing alpha
    python3 Tools/replay_video.py --session recordings/session_.../ --math-only --alpha 0.5

    # Compare replay against original
    python3 Tools/replay_video.py --session recordings/session_.../ --math-only --compare

    # Full YOLO replay + compare
    python3 Tools/replay_video.py --session recordings/session_.../ --config glog.conf --compare

    # Parameter sweep
    for alpha in 0.3 0.5 0.7 0.9; do
        python3 Tools/replay_video.py --session ... --math-only --alpha $alpha --compare
    done
"""

import argparse
import json
import os
import sys
import time
import traceback

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_tracking_data(session_path: str, camera: str) -> list:
    """Load tracking JSONL records for a camera."""
    tracking_path = os.path.join(session_path, camera, "tracking.jsonl")
    records = []
    if os.path.exists(tracking_path):
        with open(tracking_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_metadata(session_path: str) -> dict:
    """Load session metadata."""
    meta_path = os.path.join(session_path, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}


def get_camera_frame_count(session_path: str, camera: str) -> int:
    """Count JPEG frames in a camera's frames directory."""
    frames_dir = os.path.join(session_path, camera, "frames")
    if not os.path.isdir(frames_dir):
        return 0
    return len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])


def full_yolo_replay(session_path: str, config_path: str,
                     output_path: str = None) -> dict:
    """Feed recorded frames through YOLO + pose detection on GPU.

    Loads each camera's frames from disk, runs YOLO detection and optional
    pose estimation, and collects the detection results per frame.

    Args:
        session_path: Path to recording session directory.
        config_path: Path to glog.conf for model config.
        output_path: Optional path to write replay results JSONL.

    Returns:
        Dict of camera -> list of per-frame detection dicts.
    """
    from configparser import ConfigParser
    import cv2
    import numpy as np

    config = ConfigParser()
    config.read(config_path)

    # Load YOLO model
    print("Loading YOLO model...")
    from glados_modules.GladosEnums import FeatureToggles, VisionResultsEnum
    from ultralytics import YOLO
    from ultralytics.nn.modules import Conv
    from ultralytics.nn.tasks import DetectionModel
    from torch.nn import Sequential
    from torch.serialization import safe_globals

    model_config = config["YOLO"]["model"]
    tracker_yaml = config["YOLO"]["tracker"]
    with safe_globals([DetectionModel, Sequential, Conv]):
        yolo_model = YOLO(model_config)
    print(f"YOLO model loaded: {model_config}")

    # Optional pose model
    pose_enabled = config.get(FeatureToggles.CONFIG_HEAD.value,
                               FeatureToggles.POSE_MODEL_ENABLED.value,
                               fallback="True").strip().lower() == "true"
    pose_model = None
    if pose_enabled:
        print("Loading pose model...")
        from rtmlib import Wholebody
        pose_model = Wholebody(to_openpose=False, mode='balanced',
                                backend='onnxruntime', device='cuda')
        print("Pose model loaded")

    metadata = load_metadata(session_path)
    cameras = metadata.get("cameras", {})
    results = {}

    for camera in cameras:
        n_frames = get_camera_frame_count(session_path, camera)
        if n_frames == 0:
            print(f"\n{camera}: No frames, skipping")
            continue

        print(f"\n--- Processing {camera} ({n_frames} frames) ---")
        frames_dir = os.path.join(session_path, camera, "frames")
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

        cam_results = []
        t_start = time.time()

        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"  WARNING: Could not read {frame_file}")
                cam_results.append({"frame": i, "error": "unreadable"})
                continue

            # Run YOLO detection
            try:
                yolo_results = yolo_model.track(
                    source=frame, device="cuda", tracker=tracker_yaml,
                    verbose=False)
            except Exception as e:
                print(f"  WARNING: YOLO failed on frame {i}: {e}")
                cam_results.append({"frame": i, "error": str(e)})
                continue

            # Translate YOLO results to standard format
            frame_detections = {}
            for r in yolo_results:
                if r is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = r.names[cls_id]
                    conf = round(box.conf.item(), 3)
                    b = box.xyxy[0].tolist()
                    det = {
                        "class_name": cls_name,
                        "confidence": conf,
                        "box": {
                            "x1": int(b[0]), "y1": int(b[1]),
                            "x2": int(b[2]), "y2": int(b[3])
                        }
                    }

                    if cls_name not in frame_detections:
                        frame_detections[cls_name] = {
                            "count": 0,
                            "objects": [],
                            "class_name": cls_name,
                        }
                    frame_detections[cls_name]["count"] += 1
                    frame_detections[cls_name]["objects"].append(det)

            # Optional pose estimation on person detections
            if pose_model and "person" in frame_detections:
                try:
                    key_points, scores = pose_model(frame)
                    frame_detections["_pose_keypoints"] = len(key_points) if key_points is not None else 0
                except Exception:
                    pass

            cam_results.append({
                "frame": i,
                "detections": frame_detections,
                "persons": frame_detections.get("person", {}).get("count", 0),
            })

            # Progress
            if (i + 1) % 50 == 0 or i == len(frame_files) - 1:
                elapsed = time.time() - t_start
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Frame {i+1}/{n_frames} ({fps:.1f} FPS)")

        results[camera] = cam_results
        elapsed = time.time() - t_start
        print(f"  Done: {n_frames} frames in {elapsed:.1f}s ({n_frames/elapsed:.1f} FPS)")

        # Write per-camera replay results
        if output_path:
            cam_output = os.path.join(output_path, f"{camera}_replay.jsonl")
            os.makedirs(output_path, exist_ok=True)
            with open(cam_output, 'w') as f:
                for r in cam_results:
                    f.write(json.dumps(r) + "\n")
            print(f"  Results written to {cam_output}")

    return results


def math_only_replay(session_path: str, alpha: float = 0.7,
                     speed: int = 3) -> dict:
    """Re-run tracking math with different parameters.

    Uses detection data from original tracking.jsonl (no YOLO re-run).
    Re-runs pixel-to-world conversion, EMA smoothing, and IK.

    Args:
        session_path: Path to recording session directory.
        alpha: World-space EMA smoothing alpha.
        speed: MotionProfile speed level (1-5).

    Returns:
        Dict with per-camera replay results.
    """
    from glados_modules.MotionRecorder import MotionReplay

    metadata = load_metadata(session_path)
    cameras = metadata.get("cameras", {})
    results = {}

    for camera in cameras:
        tracking_path = os.path.join(session_path, camera, "tracking.jsonl")
        if not os.path.exists(tracking_path):
            continue

        records = load_tracking_data(session_path, camera)
        if not records:
            continue

        print(f"\n--- Replaying {camera} ({len(records)} frames, alpha={alpha}, speed={speed}) ---")

        try:
            replay_results = MotionReplay.replay(tracking_path,
                                                  world_smooth_alpha=alpha)
            results[camera] = replay_results
            print(f"  Replayed {len(replay_results)} frames")
        except Exception as e:
            print(f"  Replay failed: {e}")
            traceback.print_exc()
            results[camera] = []

    return results


def compare_detections(session_path: str, replay_results: dict) -> None:
    """Compare YOLO replay detections against original tracking data.

    Shows per-frame detection count differences and missing/extra detections.

    Args:
        session_path: Path to recording session directory.
        replay_results: Dict of camera -> list of per-frame detection dicts.
    """
    for camera, replay_frames in replay_results.items():
        original = load_tracking_data(session_path, camera)
        if not original or not replay_frames:
            print(f"\n{camera}: No data to compare")
            continue

        n = min(len(original), len(replay_frames))
        person_match = 0
        person_mismatch = 0
        total_orig_persons = 0
        total_replay_persons = 0

        for i in range(n):
            orig_tracking = original[i].get("tracking", {})
            replay_frame = replay_frames[i]

            orig_persons = orig_tracking.get("person", {}).get("count", 0)
            replay_persons = replay_frame.get("persons", 0)

            total_orig_persons += orig_persons
            total_replay_persons += replay_persons

            if orig_persons == replay_persons:
                person_match += 1
            else:
                person_mismatch += 1

        print(f"\n{camera}: {n} frames compared")
        print(f"  Person count match: {person_match}/{n} ({100*person_match/n:.0f}%)")
        print(f"  Person count mismatch: {person_mismatch}/{n}")
        print(f"  Total persons: original={total_orig_persons}, replay={total_replay_persons}")


def compare_results(session_path: str, replay_results: dict,
                    tolerance: float = 1.0) -> None:
    """Compare math-only replay results against original tracking data.

    Args:
        session_path: Path to recording session directory.
        replay_results: Dict of camera -> list of replay frame results.
        tolerance: Maximum acceptable difference in degrees.
    """
    for camera, replay_frames in replay_results.items():
        original = load_tracking_data(session_path, camera)
        if not original or not replay_frames:
            print(f"\n{camera}: No data to compare")
            continue

        n = min(len(original), len(replay_frames))
        diffs = []
        regressions = 0

        for i in range(n):
            orig_tracking = original[i].get("tracking", {})
            replay_frame = replay_frames[i] if i < len(replay_frames) else {}

            for servo in ["head_left_right", "head_up_down",
                          "body_left_right", "body_up_down"]:
                orig_val = orig_tracking.get(servo, {}).get("angle")
                replay_val = replay_frame.get(f"{servo}_target")
                if orig_val is not None and replay_val is not None:
                    diff = abs(orig_val - replay_val)
                    diffs.append(diff)
                    if diff > tolerance:
                        regressions += 1

        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            max_diff = max(diffs)
            print(f"\n{camera}: {n} frames compared")
            print(f"  avg diff: {avg_diff:.2f}°")
            print(f"  max diff: {max_diff:.2f}°")
            print(f"  regressions (>{tolerance}°): {regressions}/{len(diffs)}")
        else:
            print(f"\n{camera}: No comparable servo target data")


def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded video sessions through GLaDOS tracking pipeline")
    parser.add_argument("--session", required=True,
                        help="Path to recording session directory")
    parser.add_argument("--config", default=None,
                        help="Path to glog.conf (required for full replay)")
    parser.add_argument("--math-only", action="store_true",
                        help="Re-run tracking math only (no YOLO, no GPU)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare replay results against original")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="World-space EMA smoothing alpha (default 0.7)")
    parser.add_argument("--speed", type=int, default=3,
                        help="MotionProfile speed level 1-5 (default 3)")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="Regression tolerance in degrees (default 1.0)")
    parser.add_argument("--output", default=None,
                        help="Output directory for replay results")
    args = parser.parse_args()

    if not os.path.isdir(args.session):
        print(f"Session directory not found: {args.session}")
        sys.exit(1)

    metadata = load_metadata(args.session)
    print(f"Session: {metadata.get('session_id', 'unknown')}")
    print(f"Duration: {metadata.get('duration_s', '?')}s")
    cameras = metadata.get("cameras", {})
    for cam, info in cameras.items():
        frames = info.get("total_frames", 0)
        disk_frames = get_camera_frame_count(args.session, cam)
        print(f"  {cam}: {frames} recorded, {disk_frames} on disk")

    if args.math_only:
        results = math_only_replay(args.session, alpha=args.alpha, speed=args.speed)
        if args.compare:
            compare_results(args.session, results, tolerance=args.tolerance)
    else:
        if not args.config:
            print("\nFull replay requires --config glog.conf")
            sys.exit(1)

        print(f"\n=== Full YOLO Replay ===")
        output = args.output or os.path.join(args.session, "replay_output")
        results = full_yolo_replay(args.session, args.config, output_path=output)

        if args.compare:
            print(f"\n=== Detection Comparison ===")
            compare_detections(args.session, results)

    print("\nDone.")


if __name__ == "__main__":
    main()
