"""Motion calculation recording and replay for regression testing.

Records every tracking frame's inputs (detection data, estimator state) and
outputs (world angles, servo targets) to a JSONL file. These recordings can
be replayed offline through the calculation code to detect regressions or
measure improvements without needing cameras, servos, or GPUs.

Usage - Recording (happens automatically when enabled in MotionTrack):
    recorder = MotionRecorder("session_2024_01_15")
    recorder.log_frame(frame_data)
    recorder.close()

Usage - Replay (offline, no hardware needed):
    results = MotionReplay.replay("recordings/session_2024_01_15.jsonl")
    MotionReplay.compare(results_a, results_b)
"""

import json
import time
import os
from typing import Dict, Any, List, Optional
from math import radians, tan, atan, degrees
from datetime import datetime


class MotionRecorder:
    """Records tracking frame data to a JSONL file for later replay.

    Each line is a self-contained JSON object with all inputs needed to
    reproduce the calculation, plus the outputs for comparison.
    """

    def __init__(self, session_name: str = None, output_dir: str = "./recordings") -> None:
        os.makedirs(output_dir, exist_ok=True)
        if session_name is None:
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.filepath = os.path.join(output_dir, f"{session_name}.jsonl")
        self._file = open(self.filepath, 'a')
        self._frame_count = 0
        self._start_time = time.time()
        # Write header line
        self._file.write(json.dumps({
            "type": "header",
            "session": session_name,
            "start_time": self._start_time,
            "start_iso": datetime.now().isoformat(),
        }) + "\n")
        self._file.flush()

    def log_frame(self, frame_data: dict) -> None:
        """Log a single tracking frame. Called from MotionTrack.track_loop()."""
        frame_data["type"] = "frame"
        frame_data["frame_number"] = self._frame_count
        frame_data["wall_time"] = time.time()
        frame_data["session_time"] = time.time() - self._start_time
        self._file.write(json.dumps(frame_data) + "\n")
        self._frame_count += 1
        # Flush periodically (every 10 frames)
        if self._frame_count % 10 == 0:
            self._file.flush()

    def close(self) -> None:
        """Write footer and close the file."""
        self._file.write(json.dumps({
            "type": "footer",
            "total_frames": self._frame_count,
            "duration": time.time() - self._start_time,
        }) + "\n")
        self._file.flush()
        self._file.close()

    @property
    def frame_count(self) -> int:
        return self._frame_count


def build_frame_record(
    camera: str,
    detection: dict,
    use_point: bool,
    estimator_state: Dict[str, Dict[str, float]],
    servo_middles: Dict[str, float],
    servo_mins: Dict[str, float],
    servo_maxs: Dict[str, float],
    cam_resolution: tuple,
    raw_world_lr: float,
    raw_world_ud: float,
    smoothed_world_lr: float,
    smoothed_world_ud: float,
    output_targets: Dict[str, dict],
) -> dict:
    """Build a complete frame record from MotionTrack's internal state.

    This captures everything needed to replay the calculation offline:
    - Input: detection bbox/pose, camera, estimator positions, servo config
    - Intermediate: raw and smoothed world angles
    - Output: final servo targets
    """
    return {
        # Inputs
        "camera": camera,
        "detection": detection,
        "use_point": use_point,
        "cam_resolution": list(cam_resolution),
        "estimator_state": estimator_state,
        "servo_middles": servo_middles,
        "servo_mins": servo_mins,
        "servo_maxs": servo_maxs,
        # Intermediate calculations
        "raw_world_lr": round(raw_world_lr, 4),
        "raw_world_ud": round(raw_world_ud, 4),
        "smoothed_world_lr": round(smoothed_world_lr, 4),
        "smoothed_world_ud": round(smoothed_world_ud, 4),
        # Outputs
        "output_targets": output_targets,
    }


class MotionReplay:
    """Replays recorded sessions through calculation code offline.

    No MQTT, no hardware, no GPU needed. Feeds recorded inputs through
    the same math and produces outputs that can be compared against the
    original recording or against other code versions.
    """

    @staticmethod
    def load_session(filepath: str) -> List[dict]:
        """Load a recorded session and return only frame records."""
        frames = []
        with open(filepath, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                if record.get("type") == "frame":
                    frames.append(record)
        return frames

    @staticmethod
    def replay(filepath: str, world_smooth_alpha: float = 0.7) -> List[dict]:
        """Replay a recorded session through the calculation pipeline.

        Uses the recorded inputs (detection, estimator state, servo config)
        and re-runs the world-angle and target calculations using forward
        kinematics and Jacobian IK. Returns a list of result records that
        can be compared against the original recording or other code versions.

        Args:
            filepath: Path to the JSONL recording file.
            world_smooth_alpha: EMA smoothing factor for world-space angles.
        """
        from glados_modules.GladosEnums import ServoEnum
        from glados_modules.RobotKinematics import RobotKinematics

        frames = MotionReplay.load_session(filepath)
        results = []

        smoothed_lr = None
        smoothed_ud = None

        for frame in frames:
            camera = frame["camera"]
            detection = frame["detection"]
            use_point = frame["use_point"]
            cam_x, cam_y = frame["cam_resolution"]
            est_state = frame["estimator_state"]
            middles = frame["servo_middles"]
            mins = frame["servo_mins"]
            maxs = frame["servo_maxs"]

            # Re-run pixel_to_world_angle for both axes (now using FK)
            world_lr = _calc_world_angle(
                detection, camera, ServoEnum.X_AXIS.value, use_point,
                cam_x, cam_y, est_state, middles, mins, maxs)
            world_ud = _calc_world_angle(
                detection, camera, ServoEnum.Y_AXIS.value, use_point,
                cam_x, cam_y, est_state, middles, mins, maxs)

            # Smooth
            if smoothed_lr is None:
                smoothed_lr = world_lr
                smoothed_ud = world_ud
            else:
                smoothed_lr = world_smooth_alpha * smoothed_lr + (1 - world_smooth_alpha) * world_lr
                smoothed_ud = world_smooth_alpha * smoothed_ud + (1 - world_smooth_alpha) * world_ud

            # Compute targets using Jacobian IK
            head_lr_name = ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
            head_ud_name = ServoEnum.LOCATION_HEAD_UP_DOWN.value
            body_lr_name = ServoEnum.LOCATION_BODY_LEFT_RIGHT.value
            body_ud_name = ServoEnum.LOCATION_BODY_UP_DOWN.value

            kin = RobotKinematics(middles, mins, maxs)

            # Head IK: given current body, find head angles for target
            current_body = {
                body_lr_name: est_state[body_lr_name]["position"],
                body_ud_name: est_state[body_ud_name]["position"],
            }
            head_targets = kin.inverse_kinematics_head(smoothed_lr, smoothed_ud, current_body)

            # Body IK: find where body should go (head at middle)
            body_targets = kin.inverse_kinematics_body(smoothed_lr, smoothed_ud)

            # Clamp
            head_lr_target = max(mins[head_lr_name], min(maxs[head_lr_name], head_targets[head_lr_name]))
            head_ud_target = max(mins[head_ud_name], min(maxs[head_ud_name], head_targets[head_ud_name]))
            body_lr_target = max(mins[body_lr_name], min(maxs[body_lr_name], body_targets[body_lr_name]))
            body_ud_target = max(mins[body_ud_name], min(maxs[body_ud_name], body_targets[body_ud_name]))

            result = {
                "frame_number": frame["frame_number"],
                "session_time": frame["session_time"],
                "raw_world_lr": round(world_lr, 4),
                "raw_world_ud": round(world_ud, 4),
                "smoothed_world_lr": round(smoothed_lr, 4),
                "smoothed_world_ud": round(smoothed_ud, 4),
                "head_lr_target": round(head_lr_target, 2),
                "head_ud_target": round(head_ud_target, 2),
                "body_lr_target": round(body_lr_target, 2),
                "body_ud_target": round(body_ud_target, 2),
            }
            results.append(result)

        return results

    @staticmethod
    def compare(original_file: str, replay_results: List[dict],
                tolerance: float = 1.0) -> dict:
        """Compare replay results against the original recording.

        Args:
            original_file: Path to the original JSONL recording.
            replay_results: Output from replay().
            tolerance: Maximum acceptable angle difference in degrees.

        Returns:
            Dict with comparison metrics and per-frame diffs.
        """
        original_frames = MotionReplay.load_session(original_file)
        diffs = []
        max_diff = 0.0
        total_diff = 0.0
        regression_count = 0

        servo_keys = ["head_lr_target", "head_ud_target", "body_lr_target", "body_ud_target"]

        for orig, replay in zip(original_frames, replay_results):
            frame_diff = {"frame_number": orig["frame_number"]}
            frame_max = 0.0

            orig_targets = orig.get("output_targets", {})
            # Map original output format to flat keys
            orig_flat = {}
            if orig_targets:
                from glados_modules.GladosEnums import ServoEnum
                angle_key = ServoEnum.MSG_ANGLE.value
                loc_map = {
                    ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: "head_lr_target",
                    ServoEnum.LOCATION_HEAD_UP_DOWN.value: "head_ud_target",
                    ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: "body_lr_target",
                    ServoEnum.LOCATION_BODY_UP_DOWN.value: "body_ud_target",
                }
                for loc, data in orig_targets.items():
                    flat_key = loc_map.get(loc)
                    if flat_key:
                        orig_flat[flat_key] = data.get(angle_key, 0)

            for key in servo_keys:
                orig_val = orig_flat.get(key, replay.get(key, 0))
                replay_val = replay.get(key, 0)
                diff = abs(orig_val - replay_val)
                frame_diff[f"{key}_diff"] = round(diff, 2)
                frame_max = max(frame_max, diff)
                total_diff += diff

            frame_diff["max_diff"] = round(frame_max, 2)
            frame_diff["pass"] = frame_max <= tolerance
            if not frame_diff["pass"]:
                regression_count += 1
            diffs.append(frame_diff)
            max_diff = max(max_diff, frame_max)

        n = len(diffs)
        avg_diff = total_diff / (n * len(servo_keys)) if n > 0 else 0

        # Compute smoothness (sum of angle changes between consecutive frames)
        smoothness = _compute_smoothness(replay_results, servo_keys)

        return {
            "total_frames": n,
            "max_diff_degrees": round(max_diff, 2),
            "avg_diff_degrees": round(avg_diff, 4),
            "regressions": regression_count,
            "regression_rate": round(regression_count / n, 4) if n > 0 else 0,
            "tolerance": tolerance,
            "smoothness": smoothness,
            "frame_diffs": diffs,
        }

    @staticmethod
    def compare_two_replays(results_a: List[dict], results_b: List[dict],
                            label_a: str = "A", label_b: str = "B") -> dict:
        """Compare two replay runs against each other (e.g., before/after a code change).

        Returns per-servo smoothness metrics and angle differences.
        """
        servo_keys = ["head_lr_target", "head_ud_target", "body_lr_target", "body_ud_target"]
        smoothness_a = _compute_smoothness(results_a, servo_keys)
        smoothness_b = _compute_smoothness(results_b, servo_keys)

        # Per-frame angle diffs
        diffs = []
        for fa, fb in zip(results_a, results_b):
            frame = {"frame": fa["frame_number"]}
            for key in servo_keys:
                frame[f"{key}_diff"] = round(abs(fa.get(key, 0) - fb.get(key, 0)), 2)
            diffs.append(frame)

        return {
            f"smoothness_{label_a}": smoothness_a,
            f"smoothness_{label_b}": smoothness_b,
            "smoothness_improvement": {
                k: round(smoothness_a[k] - smoothness_b.get(k, 0), 4)
                for k in smoothness_a
            },
            "frame_diffs": diffs,
        }


def _calc_world_angle(detection: dict, camera: str, axis: str, use_point: bool,
                       cam_x: int, cam_y: int,
                       est_state: dict, middles: dict,
                       mins: dict = None, maxs: dict = None) -> float:
    """Pure-function version of MotionTrack._pixel_to_world_angle for replay.

    Uses forward kinematics for the head camera to properly account for
    non-orthogonal joint axes.
    """
    from glados_modules.GladosEnums import CameraEnum, ServoEnum, MotionProfile
    from glados_modules.RobotKinematics import RobotKinematics

    axis_size = float(cam_x) if axis == ServoEnum.X_AXIS.value else float(cam_y)

    if not use_point:
        if axis == ServoEnum.X_AXIS.value:
            pixel_center = (detection.get('x1', 0) + detection.get('x2', 0)) / 2
        else:
            pixel_center = (detection.get('y1', 0) + detection.get('y2', 0)) / 2
    else:
        pixel_center = detection.get('x', 0) if axis == ServoEnum.X_AXIS.value else detection.get('y', 0)

    offset = (axis_size / 2) - pixel_center

    if camera == CameraEnum.CAMERA_HEAD.value:
        fov = CameraEnum.CAMERA_HEAD_FOV_X.value if axis == ServoEnum.X_AXIS.value else CameraEnum.CAMERA_HEAD_FOV_Y.value
    elif camera == CameraEnum.CAMERA_RIGHT.value:
        fov = CameraEnum.CAMERA_RIGHT_FOV.value
    elif camera == CameraEnum.CAMERA_LEFT.value:
        fov = CameraEnum.CAMERA_LEFT_FOV.value
    else:
        fov = 54.0

    focal = (axis_size / 2) / tan(radians(fov) / 2)
    angle_offset = degrees(atan(offset / focal))

    head_lr = ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
    head_ud = ServoEnum.LOCATION_HEAD_UP_DOWN.value
    body_lr = ServoEnum.LOCATION_BODY_LEFT_RIGHT.value
    body_ud = ServoEnum.LOCATION_BODY_UP_DOWN.value

    if camera == CameraEnum.CAMERA_HEAD.value:
        # Use FK for proper non-orthogonal axis handling
        servo_angles = {
            body_lr: est_state[body_lr]["position"],
            body_ud: est_state[body_ud]["position"],
            head_lr: est_state[head_lr]["position"],
            head_ud: est_state[head_ud]["position"],
        }
        # Build kinematics from recorded servo config
        replay_mins = mins if mins else middles
        replay_maxs = maxs if maxs else {k: 180.0 for k in middles}
        kin = RobotKinematics(middles, replay_mins, replay_maxs)
        yaw, pitch = kin.forward_kinematics(servo_angles)
        camera_world = yaw if axis == ServoEnum.X_AXIS.value else pitch
    elif camera == CameraEnum.CAMERA_LEFT.value:
        camera_world = est_state[body_lr]["position"] + MotionProfile.CAMERA_LEFT_MOUNTING_OFFSET.value
    elif camera == CameraEnum.CAMERA_RIGHT.value:
        camera_world = est_state[body_lr]["position"] + MotionProfile.CAMERA_RIGHT_MOUNTING_OFFSET.value
    else:
        camera_world = 90.0

    return camera_world + angle_offset


def _compute_smoothness(results: List[dict], servo_keys: List[str]) -> dict:
    """Compute smoothness metrics: total jerk (sum of angle changes between frames).
    Lower = smoother. Also computes max single-frame jump.
    """
    metrics = {}
    for key in servo_keys:
        total_jerk = 0.0
        max_jump = 0.0
        for i in range(1, len(results)):
            prev = results[i - 1].get(key, 0)
            curr = results[i].get(key, 0)
            jump = abs(curr - prev)
            total_jerk += jump
            max_jump = max(max_jump, jump)
        metrics[f"{key}_total_jerk"] = round(total_jerk, 2)
        metrics[f"{key}_max_jump"] = round(max_jump, 2)
        metrics[f"{key}_avg_jerk"] = round(total_jerk / max(1, len(results) - 1), 4)
    return metrics
