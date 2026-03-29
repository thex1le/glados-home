"""Tests for motion recording and replay system.

Validates that recordings capture the right data, replay produces
consistent results, and comparison metrics are correct.
"""

import pytest
import json
import os
import tempfile

from glados_modules.MotionRecorder import (
    MotionRecorder, build_frame_record, MotionReplay, _compute_smoothness
)
from glados_modules.GladosEnums import ServoEnum


class TestMotionRecorder:
    """Test the JSONL recording writer."""

    def test_creates_file_with_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = MotionRecorder(session_name="test", output_dir=tmpdir)
            rec.close()
            filepath = os.path.join(tmpdir, "test.jsonl")
            assert os.path.exists(filepath)
            with open(filepath) as f:
                lines = f.readlines()
            header = json.loads(lines[0])
            assert header["type"] == "header"
            footer = json.loads(lines[1])
            assert footer["type"] == "footer"
            assert footer["total_frames"] == 0

    def test_log_frame_writes_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = MotionRecorder(session_name="test", output_dir=tmpdir)
            rec.log_frame({"detection": {"x1": 100, "y1": 100}})
            rec.log_frame({"detection": {"x1": 200, "y1": 200}})
            rec.close()
            filepath = os.path.join(tmpdir, "test.jsonl")
            with open(filepath) as f:
                lines = f.readlines()
            # header + 2 frames + footer = 4 lines
            assert len(lines) == 4
            frame0 = json.loads(lines[1])
            assert frame0["type"] == "frame"
            assert frame0["frame_number"] == 0
            frame1 = json.loads(lines[2])
            assert frame1["frame_number"] == 1


class TestBuildFrameRecord:
    """Test frame record structure."""

    def test_has_all_required_fields(self):
        record = build_frame_record(
            camera="camera_head",
            detection={"x1": 100, "y1": 100, "x2": 200, "y2": 200},
            use_point=False,
            estimator_state={"head_left_right": {"position": 90, "velocity": 0, "target": 90}},
            servo_middles={"head_left_right": 92},
            servo_mins={"head_left_right": 6},
            servo_maxs={"head_left_right": 125},
            cam_resolution=(640, 480),
            raw_world_lr=105.2,
            raw_world_ud=88.4,
            smoothed_world_lr=103.1,
            smoothed_world_ud=87.2,
            output_targets={"head_left_right": {"angle": 95}},
        )
        assert record["camera"] == "camera_head"
        assert "detection" in record
        assert "estimator_state" in record
        assert "raw_world_lr" in record
        assert "smoothed_world_lr" in record
        assert "output_targets" in record
        assert record["cam_resolution"] == [640, 480]


class TestSmoothness:
    """Test smoothness metric computation."""

    def test_constant_angle_has_zero_jerk(self):
        results = [{"head_lr_target": 90} for _ in range(10)]
        metrics = _compute_smoothness(results, ["head_lr_target"])
        assert metrics["head_lr_target_total_jerk"] == 0
        assert metrics["head_lr_target_max_jump"] == 0

    def test_linear_ramp_has_constant_jerk(self):
        results = [{"head_lr_target": float(i)} for i in range(10)]
        metrics = _compute_smoothness(results, ["head_lr_target"])
        assert metrics["head_lr_target_avg_jerk"] == 1.0  # 1 degree per step
        assert metrics["head_lr_target_max_jump"] == 1.0

    def test_spike_detected(self):
        results = [{"head_lr_target": 90.0}] * 5 + [{"head_lr_target": 150.0}] + [{"head_lr_target": 90.0}] * 5
        metrics = _compute_smoothness(results, ["head_lr_target"])
        assert metrics["head_lr_target_max_jump"] == 60.0


class TestMotionReplayCompare:
    """Test the comparison logic."""

    def test_identical_results_no_regressions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake recording with output_targets
            filepath = os.path.join(tmpdir, "test.jsonl")
            with open(filepath, 'w') as f:
                f.write(json.dumps({"type": "header"}) + "\n")
                for i in range(5):
                    frame = {
                        "type": "frame",
                        "frame_number": i,
                        "output_targets": {
                            ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: {
                                ServoEnum.MSG_ANGLE.value: 90 + i
                            },
                            ServoEnum.LOCATION_HEAD_UP_DOWN.value: {
                                ServoEnum.MSG_ANGLE.value: 100
                            },
                            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: {
                                ServoEnum.MSG_ANGLE.value: 90
                            },
                            ServoEnum.LOCATION_BODY_UP_DOWN.value: {
                                ServoEnum.MSG_ANGLE.value: 100
                            },
                        }
                    }
                    f.write(json.dumps(frame) + "\n")
                f.write(json.dumps({"type": "footer"}) + "\n")

            # Create matching replay results
            replay_results = [
                {"frame_number": i, "head_lr_target": 90 + i, "head_ud_target": 100,
                 "body_lr_target": 90, "body_ud_target": 100}
                for i in range(5)
            ]

            report = MotionReplay.compare(filepath, replay_results, tolerance=1.0)
            assert report["regressions"] == 0
            assert report["max_diff_degrees"] == 0
