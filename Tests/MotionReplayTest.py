"""Replay recorded motion sessions and compare results.

Usage:
    # Replay a recording and compare against the original outputs
    python Tests/MotionReplayTest.py replay recordings/session_20240115.jsonl

    # Compare two replays (e.g., before/after a code change)
    python Tests/MotionReplayTest.py compare recordings/session_A.jsonl recordings/session_B.jsonl

    # Print smoothness metrics for a recording
    python Tests/MotionReplayTest.py smoothness recordings/session_20240115.jsonl

    # Replay with different smoothing alpha to test parameter changes
    python Tests/MotionReplayTest.py replay recordings/session_20240115.jsonl --alpha 0.5
"""

import sys
import json
import argparse
from os import path

sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))

from glados_modules.MotionRecorder import MotionReplay


def cmd_replay(args):
    """Replay a session and compare against original recording."""
    print(f"Replaying: {args.recording}")
    results = MotionReplay.replay(args.recording, world_smooth_alpha=args.alpha)
    print(f"Replayed {len(results)} frames")

    report = MotionReplay.compare(args.recording, results, tolerance=args.tolerance)

    print(f"\n--- Comparison Report ---")
    print(f"Total frames:     {report['total_frames']}")
    print(f"Max diff:         {report['max_diff_degrees']}deg")
    print(f"Avg diff:         {report['avg_diff_degrees']}deg")
    print(f"Regressions:      {report['regressions']} ({report['regression_rate']*100:.1f}%)")
    print(f"Tolerance:        {report['tolerance']}deg")
    print(f"\n--- Smoothness ---")
    for key, val in report['smoothness'].items():
        print(f"  {key}: {val}")

    if report['regressions'] > 0:
        print(f"\n--- Regression Frames ---")
        for diff in report['frame_diffs']:
            if not diff['pass']:
                print(f"  Frame {diff['frame_number']}: max_diff={diff['max_diff']}deg")

    # Save full report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to: {args.output}")

    # Exit code: 1 if regressions detected
    return 1 if report['regressions'] > 0 else 0


def cmd_compare(args):
    """Compare two recordings replayed through current code."""
    print(f"Replaying A: {args.recording_a}")
    results_a = MotionReplay.replay(args.recording_a, world_smooth_alpha=args.alpha)
    print(f"  -> {len(results_a)} frames")

    print(f"Replaying B: {args.recording_b}")
    results_b = MotionReplay.replay(args.recording_b, world_smooth_alpha=args.alpha)
    print(f"  -> {len(results_b)} frames")

    report = MotionReplay.compare_two_replays(results_a, results_b, label_a="A", label_b="B")

    print(f"\n--- Smoothness A ---")
    for key, val in report['smoothness_A'].items():
        print(f"  {key}: {val}")

    print(f"\n--- Smoothness B ---")
    for key, val in report['smoothness_B'].items():
        print(f"  {key}: {val}")

    print(f"\n--- Improvement (A - B, positive = B is smoother) ---")
    for key, val in report['smoothness_improvement'].items():
        direction = "better" if val > 0 else "worse" if val < 0 else "same"
        print(f"  {key}: {val:+.4f} ({direction})")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to: {args.output}")

    return 0


def cmd_smoothness(args):
    """Print smoothness metrics for a single recording."""
    print(f"Replaying: {args.recording}")
    results = MotionReplay.replay(args.recording, world_smooth_alpha=args.alpha)
    print(f"Replayed {len(results)} frames\n")

    servo_keys = ["head_lr_target", "head_ud_target", "body_lr_target", "body_ud_target"]
    from glados_modules.MotionRecorder import _compute_smoothness
    metrics = _compute_smoothness(results, servo_keys)

    print("--- Smoothness Metrics (lower = smoother) ---")
    for key in sorted(metrics.keys()):
        print(f"  {key}: {metrics[key]}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion tracking replay and regression testing")
    subparsers = parser.add_subparsers(dest="command")

    # Replay command
    p_replay = subparsers.add_parser("replay", help="Replay recording and compare against original")
    p_replay.add_argument("recording", help="Path to JSONL recording file")
    p_replay.add_argument("--alpha", type=float, default=0.7, help="World-space smoothing alpha (default: 0.7)")
    p_replay.add_argument("--tolerance", type=float, default=1.0, help="Max acceptable diff in degrees (default: 1.0)")
    p_replay.add_argument("--output", "-o", help="Save full JSON report to file")

    # Compare command
    p_compare = subparsers.add_parser("compare", help="Compare two recordings")
    p_compare.add_argument("recording_a", help="First recording")
    p_compare.add_argument("recording_b", help="Second recording")
    p_compare.add_argument("--alpha", type=float, default=0.7, help="World-space smoothing alpha")
    p_compare.add_argument("--output", "-o", help="Save full JSON report to file")

    # Smoothness command
    p_smooth = subparsers.add_parser("smoothness", help="Print smoothness metrics")
    p_smooth.add_argument("recording", help="Path to JSONL recording file")
    p_smooth.add_argument("--alpha", type=float, default=0.7, help="World-space smoothing alpha")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "replay":
        sys.exit(cmd_replay(args))
    elif args.command == "compare":
        sys.exit(cmd_compare(args))
    elif args.command == "smoothness":
        sys.exit(cmd_smoothness(args))
