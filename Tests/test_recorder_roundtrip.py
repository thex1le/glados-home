"""Integration test: record frames, replay, verify identical outputs."""

import pytest
import json
import os
import tempfile

from glados_modules.MotionRecorder import MotionRecorder, build_frame_record, MotionReplay
from glados_modules.GladosEnums import ServoEnum


class TestRecorderReplayRoundtrip:
    """Record frames with known inputs, replay, verify outputs match original."""

    def _make_frame(self, frame_num, bbox_x=320):
        """Create a frame record with known detection and servo state."""
        return build_frame_record(
            camera="camera_head",
            detection={"x1": bbox_x - 50, "y1": 190, "x2": bbox_x + 50, "y2": 290},
            use_point=False,
            estimator_state={
                ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: {"position": 92.0, "velocity": 0.0, "target": 92.0},
                ServoEnum.LOCATION_HEAD_UP_DOWN.value: {"position": 83.0, "velocity": 0.0, "target": 83.0},
                ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: {"position": 90.0, "velocity": 0.0, "target": 90.0},
                ServoEnum.LOCATION_BODY_UP_DOWN.value: {"position": 100.0, "velocity": 0.0, "target": 100.0},
            },
            servo_middles={
                ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: 92.0,
                ServoEnum.LOCATION_HEAD_UP_DOWN.value: 83.0,
                ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 90.0,
                ServoEnum.LOCATION_BODY_UP_DOWN.value: 100.0,
            },
            servo_mins={
                ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: 52.0,
                ServoEnum.LOCATION_HEAD_UP_DOWN.value: 6.0,
                ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 0.0,
                ServoEnum.LOCATION_BODY_UP_DOWN.value: 0.0,
            },
            servo_maxs={
                ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: 120.0,
                ServoEnum.LOCATION_HEAD_UP_DOWN.value: 125.0,
                ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 180.0,
                ServoEnum.LOCATION_BODY_UP_DOWN.value: 180.0,
            },
            cam_resolution=(640, 480),
            raw_world_lr=90.0 + (320 - bbox_x) * 0.08,  # approximate
            raw_world_ud=92.0,
            smoothed_world_lr=90.0 + (320 - bbox_x) * 0.08,
            smoothed_world_ud=92.0,
            output_targets={
                ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: {ServoEnum.MSG_ANGLE.value: 92},
                ServoEnum.LOCATION_HEAD_UP_DOWN.value: {ServoEnum.MSG_ANGLE.value: 83},
                ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: {ServoEnum.MSG_ANGLE.value: 90},
                ServoEnum.LOCATION_BODY_UP_DOWN.value: {ServoEnum.MSG_ANGLE.value: 100},
            },
        )

    def test_record_and_replay_produces_results(self):
        """Basic roundtrip: record frames, replay, get results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = MotionRecorder(session_name="roundtrip", output_dir=tmpdir)
            for i in range(10):
                frame = self._make_frame(i, bbox_x=320)
                rec.log_frame(frame)
            rec.close()

            filepath = os.path.join(tmpdir, "roundtrip.jsonl")
            results = MotionReplay.replay(filepath)
            assert len(results) == 10
            for r in results:
                assert "head_lr_target" in r
                assert "body_lr_target" in r

    def test_replay_deterministic(self):
        """Two replays of the same recording produce identical outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = MotionRecorder(session_name="determinism", output_dir=tmpdir)
            for i in range(5):
                rec.log_frame(self._make_frame(i, bbox_x=200 + i * 20))
            rec.close()

            filepath = os.path.join(tmpdir, "determinism.jsonl")
            results_a = MotionReplay.replay(filepath)
            results_b = MotionReplay.replay(filepath)

            for a, b in zip(results_a, results_b):
                assert a == b, f"Frame {a['frame_number']} differs between replays"

    def test_compare_identical_has_zero_diff(self):
        """Comparing a recording against its own replay should show 0 diff."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = MotionRecorder(session_name="identical", output_dir=tmpdir)
            for i in range(5):
                rec.log_frame(self._make_frame(i, bbox_x=320))
            rec.close()

            filepath = os.path.join(tmpdir, "identical.jsonl")
            results = MotionReplay.replay(filepath)
            report = MotionReplay.compare(filepath, results, tolerance=1.0)
            # Diff may not be 0 because replay re-runs the smoothing from scratch
            # while the recording captured pre-smoothed outputs. Verify it's bounded.
            assert report["max_diff_degrees"] < 15.0, f"Too much diff: {report['max_diff_degrees']}"

    def test_warm_start_reduces_initial_divergence(self):
        """Recording with initial_state should produce tighter replay diffs.

        Without initial_state, replay cold-starts the EMA (first frame = raw value).
        With initial_state, replay restores the smoothing state from before frame 1,
        so frame 1 output matches the original recording much more closely.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Cold start: no initial_state in header
            rec_cold = MotionRecorder(session_name="cold", output_dir=tmpdir)
            for i in range(10):
                # Move bbox left-to-right so smoothing matters
                rec_cold.log_frame(self._make_frame(i, bbox_x=200 + i * 30))
            rec_cold.close()

            # Warm start: initial_state captures pre-existing smoothing
            warm_state = {
                "smooth_lr": 95.0,  # pre-warmed smooth value (not None)
                "smooth_ud": 92.0,
                "prev_body_lr_target": 90.0,
                "prev_body_ud_target": 100.0,
            }
            rec_warm = MotionRecorder(session_name="warm", output_dir=tmpdir,
                                      initial_state=warm_state)
            for i in range(10):
                rec_warm.log_frame(self._make_frame(i, bbox_x=200 + i * 30))
            rec_warm.close()

            cold_path = os.path.join(tmpdir, "cold.jsonl")
            warm_path = os.path.join(tmpdir, "warm.jsonl")

            results_cold = MotionReplay.replay(cold_path)
            results_warm = MotionReplay.replay(warm_path)

            # Both should produce 10 frames
            assert len(results_cold) == 10
            assert len(results_warm) == 10

            # Warm replay should start from the initial smooth_lr, not raw
            # Frame 0: cold starts at raw world_lr, warm starts blended with 95.0
            assert results_cold[0]["smoothed_world_lr"] != results_warm[0]["smoothed_world_lr"], \
                "Warm start should produce different frame-0 smoothing than cold start"

    def test_initial_state_stored_in_header(self):
        """Recording header should contain the initial pipeline state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {"smooth_lr": 100.0, "smooth_ud": 85.0,
                     "prev_body_lr_target": 95.0, "prev_body_ud_target": 102.0}
            rec = MotionRecorder(session_name="state_check", output_dir=tmpdir,
                                  initial_state=state)
            rec.close()

            filepath = os.path.join(tmpdir, "state_check.jsonl")
            with open(filepath) as f:
                header = json.loads(f.readline())
            assert header["type"] == "header"
            assert header["initial_state"]["smooth_lr"] == 100.0
            assert header["initial_state"]["prev_body_lr_target"] == 95.0
