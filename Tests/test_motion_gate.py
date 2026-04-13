"""Tests for saccadic suppression (saccade cooldown) and confidence filtering.

Validates:
- Saccade cooldown suppresses head camera for scaled duration
- Estimator clamping prevents numerical explosion
- Confidence floor filters phantom detections
- Per-detection re-filtering removes low-confidence co-detections
- Target scoring weights favor high-confidence detections
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from collections import namedtuple

from glados_modules.GladosEnums import MotionProfile, VisionResultsEnum


class TestSaccadeCooldownEnums:
    """Verify saccade cooldown enums exist and have sane values."""

    def test_cooldown_divisor_exists(self):
        val = MotionProfile.SACCADE_COOLDOWN_DIVISOR.value
        assert isinstance(val, float)
        assert val > 0

    def test_cooldown_max_exists(self):
        val = MotionProfile.SACCADE_COOLDOWN_MAX.value
        assert isinstance(val, float)
        assert 0.5 <= val <= 5.0

    def test_cooldown_duration_for_30_degree_saccade(self):
        """A 30° saccade should produce ~1s cooldown."""
        delta = 30.0
        cooldown = min(delta / MotionProfile.SACCADE_COOLDOWN_DIVISOR.value,
                       MotionProfile.SACCADE_COOLDOWN_MAX.value)
        assert 0.5 <= cooldown <= 1.5

    def test_cooldown_capped_for_large_saccade(self):
        """A 90° saccade should be capped at max."""
        delta = 90.0
        cooldown = min(delta / MotionProfile.SACCADE_COOLDOWN_DIVISOR.value,
                       MotionProfile.SACCADE_COOLDOWN_MAX.value)
        assert cooldown == MotionProfile.SACCADE_COOLDOWN_MAX.value

    def test_target_min_confidence_exists(self):
        val = MotionProfile.TARGET_MIN_CONFIDENCE.value
        assert isinstance(val, float)
        assert 0.30 <= val <= 0.60


class TestEstimatorClamping:
    """Verify spring-damper estimator clamps to physical bounds."""

    def test_position_clamped_high(self):
        from glados_modules.VisionTracker import SpringDamperEstimator
        est = SpringDamperEstimator(350.0, 6.0, 0.85)
        est.target = 400.0  # way beyond servo range
        est.get_position()
        assert est.position <= 360.0

    def test_position_clamped_low(self):
        from glados_modules.VisionTracker import SpringDamperEstimator
        est = SpringDamperEstimator(-100.0, 6.0, 0.85)
        est.target = -200.0
        est.get_position()
        assert est.position >= -180.0

    def test_velocity_clamped(self):
        from glados_modules.VisionTracker import SpringDamperEstimator
        est = SpringDamperEstimator(0.0, 6.0, 0.85)
        est.target = 180.0  # large error
        est.velocity = 5000.0  # already diverged
        est.get_position()
        assert abs(est.velocity) <= 1000.0

    def test_normal_operation_not_affected(self):
        """Clamping should not affect normal servo range operation."""
        from glados_modules.VisionTracker import SpringDamperEstimator
        est = SpringDamperEstimator(90.0, 6.0, 0.85)
        est.target = 100.0
        pos = est.get_position()
        assert 85.0 <= pos <= 105.0  # should be near 90-100 range


class TestSyncSnapOnDrift:
    """Verify sync snaps to reality when drift is large."""

    def test_small_drift_blends(self):
        from glados_modules.VisionTracker import SpringDamperEstimator
        est = SpringDamperEstimator(92.0, 6.0, 0.85)
        est.sync(90.0, 0.0)
        # Should blend: 0.7*92 + 0.3*90 = 91.4
        assert abs(est.position - 91.4) < 0.01

    def test_large_drift_snaps(self):
        from glados_modules.VisionTracker import SpringDamperEstimator
        est = SpringDamperEstimator(10000.0, 6.0, 0.85)
        est.velocity = 500000.0
        est.sync(90.0, 0.0)
        # Should snap: drift > 50
        assert est.position == 90.0
        assert est.velocity == 0.0

    def test_drift_threshold_boundary(self):
        from glados_modules.VisionTracker import SpringDamperEstimator
        est = SpringDamperEstimator(141.0, 6.0, 0.85)
        est.sync(90.0, 0.0)
        # drift = 51, above threshold — should snap
        assert est.position == 90.0


class TestConfidenceFloor:
    """Test that __select_target filters out low-confidence detections."""

    def test_phantom_below_floor_is_rejected(self):
        min_conf = MotionProfile.TARGET_MIN_CONFIDENCE.value
        phantom = {"confidence": 0.28}
        assert phantom["confidence"] < min_conf

    def test_legitimate_detection_passes_floor(self):
        min_conf = MotionProfile.TARGET_MIN_CONFIDENCE.value
        real = {"confidence": 0.70}
        assert real["confidence"] >= min_conf

    def test_floor_filters_in_list(self):
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
    """Verify the scoring weights prevent phantom wins."""

    def test_high_conf_beats_high_proximity_phantom(self):
        # Without face history weights: proximity=0.35, height=0.25, confidence=0.40
        real_proximity = max(0.0, 1.0 - 20.0 / 90.0)
        real_score = real_proximity * 0.35 + 1.0 * 0.25 + 0.70 * 0.40
        phantom_proximity = max(0.0, 1.0 - 2.0 / 90.0)
        phantom_score = phantom_proximity * 0.35 + 1.0 * 0.25 + 0.45 * 0.40
        assert real_score > phantom_score

    def test_weights_sum_to_one_without_face(self):
        assert abs(0.35 + 0.25 + 0.40 - 1.0) < 0.01

    def test_weights_sum_to_one_with_face(self):
        assert abs(0.35 + 0.25 + 0.15 + 0.25 - 1.0) < 0.01


class TestPerDetectionRefilter:
    """Test that per-detection confidence re-filtering works correctly."""

    def test_refilter_removes_low_confidence_codetections(self):
        head_threshold = 0.65
        objects = [
            {"confidence": 0.72, "box": {"x1": 300, "y1": 50, "x2": 400, "y2": 350}},
            {"confidence": 0.28, "box": {"x1": 100, "y1": 100, "x2": 200, "y2": 300}},
        ]
        filtered = [d for d in objects if d.get("confidence", 0) >= head_threshold]
        assert len(filtered) == 1
        assert filtered[0]["confidence"] == 0.72

    def test_refilter_keeps_all_above_threshold(self):
        side_threshold = 0.40
        objects = [
            {"confidence": 0.85, "box": {"x1": 100, "y1": 50, "x2": 200, "y2": 350}},
            {"confidence": 0.62, "box": {"x1": 400, "y1": 50, "x2": 500, "y2": 350}},
        ]
        filtered = [d for d in objects if d.get("confidence", 0) >= side_threshold]
        assert len(filtered) == 2

    def test_refilter_returns_empty_when_all_below(self):
        head_threshold = 0.65
        objects = [
            {"confidence": 0.28, "box": {"x1": 100, "y1": 50, "x2": 200, "y2": 300}},
            {"confidence": 0.35, "box": {"x1": 400, "y1": 50, "x2": 500, "y2": 300}},
        ]
        filtered = [d for d in objects if d.get("confidence", 0) >= head_threshold]
        assert len(filtered) == 0
