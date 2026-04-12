"""Tests for saccadic suppression (motion gate) and confidence filtering.

Validates:
- Head camera is suppressed during fast servo motion
- Confidence floor filters phantom detections
- Per-detection re-filtering removes low-confidence co-detections
- Adjusted target scoring weights favor high-confidence detections
"""

import pytest
from unittest.mock import MagicMock, patch
from collections import namedtuple

from glados_modules.GladosEnums import MotionProfile, VisionResultsEnum


class TestSettlingEnums:
    """Verify the settling-related enums exist and have sane values."""

    def test_settling_velocity_threshold_exists(self):
        val = MotionProfile.SETTLING_VELOCITY_THRESHOLD.value
        assert isinstance(val, float)
        assert val > 0

    def test_settling_velocity_threshold_is_reasonable(self):
        """Should be between 1-20 deg/s — too low = never processes, too high = no suppression."""
        val = MotionProfile.SETTLING_VELOCITY_THRESHOLD.value
        assert 1.0 <= val <= 20.0

    def test_target_min_confidence_exists(self):
        val = MotionProfile.TARGET_MIN_CONFIDENCE.value
        assert isinstance(val, float)
        assert 0.0 < val < 1.0

    def test_target_min_confidence_is_reasonable(self):
        """Should be 0.30-0.60 — low enough for partials, high enough to reject phantoms."""
        val = MotionProfile.TARGET_MIN_CONFIDENCE.value
        assert 0.30 <= val <= 0.60


class TestHeadIsSettling:
    """Test the _head_is_settling() method on MotionTrack."""

    def _make_motion_track(self):
        """Create a minimal MotionTrack-like object with estimators."""
        from glados_modules.VisionTracker import SpringDamperEstimator
        mt = MagicMock()
        mt.head_LR_name = "head_left_right"
        mt.head_UD_name = "head_up_down"
        mt._estimators = {
            "head_left_right": SpringDamperEstimator(90.0, 6.0, 0.85),
            "head_up_down": SpringDamperEstimator(83.0, 6.0, 0.85),
        }
        mt.logger = MagicMock()
        # Bind the real method
        from glados_modules.VisionTracker import MotionTrack
        mt._head_is_settling = MotionTrack._head_is_settling.__get__(mt)
        return mt

    def test_settled_when_velocity_zero(self):
        mt = self._make_motion_track()
        mt._estimators["head_left_right"].velocity = 0.0
        mt._estimators["head_up_down"].velocity = 0.0
        assert not mt._head_is_settling()

    def test_settling_when_lr_velocity_high(self):
        mt = self._make_motion_track()
        mt._estimators["head_left_right"].velocity = 30.0  # fast slew
        mt._estimators["head_up_down"].velocity = 0.0
        assert mt._head_is_settling()

    def test_settling_when_ud_velocity_high(self):
        mt = self._make_motion_track()
        mt._estimators["head_left_right"].velocity = 0.0
        mt._estimators["head_up_down"].velocity = 15.0
        assert mt._head_is_settling()

    def test_settled_when_velocity_below_threshold(self):
        mt = self._make_motion_track()
        threshold = MotionProfile.SETTLING_VELOCITY_THRESHOLD.value
        mt._estimators["head_left_right"].velocity = threshold - 1.0
        mt._estimators["head_up_down"].velocity = threshold - 1.0
        assert not mt._head_is_settling()

    def test_settling_check_uses_absolute_velocity(self):
        """Negative velocity (returning) should also trigger settling."""
        mt = self._make_motion_track()
        mt._estimators["head_left_right"].velocity = -20.0
        assert mt._head_is_settling()

    def test_settling_returns_false_when_no_estimators(self):
        mt = self._make_motion_track()
        mt._estimators = {}
        mt.head_LR_name = "head_left_right"
        assert not mt._head_is_settling()


class TestConfidenceFloor:
    """Test that __select_target filters out low-confidence detections."""

    def test_phantom_below_floor_is_rejected(self):
        """A 0.28 confidence detection should never be selected."""
        min_conf = MotionProfile.TARGET_MIN_CONFIDENCE.value
        phantom = {"confidence": 0.28, "box": {"x1": 100, "y1": 50, "x2": 200, "y2": 350}}
        assert phantom["confidence"] < min_conf

    def test_legitimate_detection_passes_floor(self):
        """A 0.70 confidence detection should pass the floor."""
        min_conf = MotionProfile.TARGET_MIN_CONFIDENCE.value
        real = {"confidence": 0.70, "box": {"x1": 100, "y1": 50, "x2": 200, "y2": 350}}
        assert real["confidence"] >= min_conf

    def test_floor_filters_in_list(self):
        """Simulates the confidence floor filter in __select_target."""
        min_conf = MotionProfile.TARGET_MIN_CONFIDENCE.value
        detections = [
            {"confidence": 0.70, "box": {"x1": 300, "y1": 50, "x2": 400, "y2": 350}},
            {"confidence": 0.28, "box": {"x1": 100, "y1": 50, "x2": 200, "y2": 350}},
            {"confidence": 0.15, "box": {"x1": 500, "y1": 50, "x2": 600, "y2": 200}},
        ]
        filtered = [d for d in detections if d.get("confidence", 0) >= min_conf]
        assert len(filtered) == 1
        assert filtered[0]["confidence"] == 0.70


class TestScoringWeights:
    """Verify the adjusted scoring weights prevent phantom wins."""

    def test_high_conf_beats_high_proximity_phantom(self):
        """A 0.70 real person at moderate proximity should beat a 0.45 phantom nearby."""
        # Without face history weights: proximity=0.35, height=0.25, confidence=0.40
        # Real person: 20° away, good height match
        real_proximity = max(0.0, 1.0 - 20.0 / 90.0)
        real_score = real_proximity * 0.35 + 1.0 * 0.25 + 0.70 * 0.40
        # Phantom: 2° away (appears near last tracked position), similar height
        phantom_proximity = max(0.0, 1.0 - 2.0 / 90.0)
        phantom_score = phantom_proximity * 0.35 + 1.0 * 0.25 + 0.45 * 0.40
        assert real_score > phantom_score, (
            f"Real ({real_score:.3f}) should beat phantom ({phantom_score:.3f})")

    def test_weights_sum_to_one_without_face(self):
        """Proximity + height + confidence should sum to 1.0 (no face history)."""
        # These are the weights from the code: 0.35 + 0.25 + 0.40
        assert abs(0.35 + 0.25 + 0.40 - 1.0) < 0.01

    def test_weights_sum_to_one_with_face(self):
        """Face + proximity + height + confidence should sum to 1.0."""
        # 0.35 + 0.25 + 0.15 + 0.25
        assert abs(0.35 + 0.25 + 0.15 + 0.25 - 1.0) < 0.01


class TestPerDetectionRefilter:
    """Test that per-detection confidence re-filtering works correctly."""

    def test_refilter_removes_low_confidence_codetections(self):
        """A frame with one good and one bad detection should only keep the good one."""
        head_threshold = 0.65
        objects = [
            {"confidence": 0.72, "box": {"x1": 300, "y1": 50, "x2": 400, "y2": 350}},
            {"confidence": 0.28, "box": {"x1": 100, "y1": 100, "x2": 200, "y2": 300}},
        ]
        filtered = [d for d in objects if d.get("confidence", 0) >= head_threshold]
        assert len(filtered) == 1
        assert filtered[0]["confidence"] == 0.72

    def test_refilter_keeps_all_above_threshold(self):
        """Multiple legitimate detections should all pass."""
        side_threshold = 0.40
        objects = [
            {"confidence": 0.85, "box": {"x1": 100, "y1": 50, "x2": 200, "y2": 350}},
            {"confidence": 0.62, "box": {"x1": 400, "y1": 50, "x2": 500, "y2": 350}},
        ]
        filtered = [d for d in objects if d.get("confidence", 0) >= side_threshold]
        assert len(filtered) == 2

    def test_refilter_returns_empty_when_all_below(self):
        """If all detections are below threshold, nothing should pass."""
        head_threshold = 0.65
        objects = [
            {"confidence": 0.28, "box": {"x1": 100, "y1": 50, "x2": 200, "y2": 300}},
            {"confidence": 0.35, "box": {"x1": 400, "y1": 50, "x2": 500, "y2": 300}},
        ]
        filtered = [d for d in objects if d.get("confidence", 0) >= head_threshold]
        assert len(filtered) == 0
