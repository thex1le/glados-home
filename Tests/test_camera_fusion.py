"""Tests for CameraFusionState: state machine transitions and handoff blending."""

import time
import pytest
from unittest.mock import patch

from glados_modules.VisionTracker import CameraFusionState
from glados_modules.GladosEnums import CameraEnum, FusionEnums, ServoEnum


class TestCameraFusionState:
    """Unit tests for the camera fusion state machine."""

    def test_initial_state_is_side_only(self):
        fusion = CameraFusionState()
        assert fusion.state == FusionEnums.STATE_SIDE_ONLY.value

    def test_side_detection_updates_world_angle(self):
        fusion = CameraFusionState()
        fusion.update_side_detection(CameraEnum.CAMERA_LEFT.value, 35.0, 1)
        assert fusion._left_world_lr == 35.0
        assert fusion._left_count == 1

    def test_head_detection_without_side_goes_straight_to_head_tracking(self):
        """If no side camera data exists, skip handoff blend entirely."""
        fusion = CameraFusionState()
        fusion.update_head_detection()
        assert fusion.state == FusionEnums.STATE_HEAD_TRACKING.value

    def test_head_detection_goes_straight_to_head_tracking(self):
        """Head detection should go directly to HEAD_TRACKING to stop side drives."""
        fusion = CameraFusionState()
        fusion.update_side_detection(CameraEnum.CAMERA_LEFT.value, 40.0, 1)
        fusion.update_head_detection()
        assert fusion.state == FusionEnums.STATE_HEAD_TRACKING.value

    def test_head_detection_blocks_side_drives(self):
        """After head detection, side cameras should not drive servos."""
        fusion = CameraFusionState()
        fusion.update_side_detection(CameraEnum.CAMERA_LEFT.value, 40.0, 1)
        fusion.update_head_detection()
        assert not fusion.side_can_drive_servos()

    def test_handoff_blend_completes_to_head(self):
        """After blend duration, should return head's angle and transition to HEAD_TRACKING."""
        fusion = CameraFusionState()
        fusion.update_side_detection(CameraEnum.CAMERA_LEFT.value, 40.0, 1)
        fusion.update_head_detection()
        # Fast-forward past blend duration
        fusion._handoff_start_time = time.time() - FusionEnums.HANDOFF_BLEND_DURATION.value - 0.1
        blended = fusion.get_blended_world_lr(90.0)
        assert blended == 90.0
        assert fusion.state == FusionEnums.STATE_HEAD_TRACKING.value

    def test_head_lost_with_side_detection_triggers_handoff_to_side(self):
        fusion = CameraFusionState()
        fusion.update_side_detection(CameraEnum.CAMERA_LEFT.value, 35.0, 1)
        fusion.update_head_detection()
        # Fast-forward to HEAD_TRACKING
        fusion.state = FusionEnums.STATE_HEAD_TRACKING.value
        # Requires 10 consecutive misses before transitioning
        for i in range(9):
            fusion.head_lost()
            assert fusion.state == FusionEnums.STATE_HEAD_TRACKING.value  # still holding
        fusion.head_lost()  # 10th miss
        assert fusion.state == FusionEnums.STATE_HANDOFF_TO_SIDE.value  # now transitions

    def test_head_lost_without_side_goes_to_side_only(self):
        fusion = CameraFusionState()
        fusion.state = FusionEnums.STATE_HEAD_TRACKING.value
        # Requires 10 consecutive misses
        for _ in range(10):
            fusion.head_lost()
        assert fusion.state == FusionEnums.STATE_SIDE_ONLY.value

    def test_head_lost_resets_on_detection(self):
        """A successful detection resets the miss counter."""
        fusion = CameraFusionState()
        fusion.state = FusionEnums.STATE_HEAD_TRACKING.value
        for _ in range(8):
            fusion.head_lost()  # 8 misses
        fusion.update_head_detection()  # resets counter
        fusion.head_lost()  # miss 1 again
        assert fusion.state == FusionEnums.STATE_HEAD_TRACKING.value  # still holding

    def test_stale_side_detection_ignored(self):
        """Side detection older than staleness threshold returns None."""
        fusion = CameraFusionState()
        fusion.update_side_detection(CameraEnum.CAMERA_LEFT.value, 35.0, 1)
        # Make it stale
        fusion._left_last_seen = time.time() - FusionEnums.SIDE_CAMERA_STALENESS.value - 1.0
        assert fusion.get_best_side_world_lr() is None

    def test_side_can_drive_servos_in_side_only(self):
        fusion = CameraFusionState()
        assert fusion.side_can_drive_servos()

    def test_side_cannot_drive_servos_during_head_tracking(self):
        fusion = CameraFusionState()
        fusion.state = FusionEnums.STATE_HEAD_TRACKING.value
        assert not fusion.side_can_drive_servos()

    def test_best_side_uses_most_recent(self):
        """When both side cameras have fresh data, use the more recent one."""
        fusion = CameraFusionState()
        now = time.time()
        fusion.update_side_detection(CameraEnum.CAMERA_LEFT.value, 35.0, 1)
        fusion._left_last_seen = now - 0.2
        fusion.update_side_detection(CameraEnum.CAMERA_RIGHT.value, 145.0, 1)
        fusion._right_last_seen = now
        assert fusion.get_best_side_world_lr() == 145.0


class TestPredictiveRotation:
    """Tests for side camera angular velocity prediction."""

    def test_prediction_no_history_returns_current(self):
        fusion = CameraFusionState()
        fusion.update_side_detection(CameraEnum.CAMERA_LEFT.value, 35.0, 1)
        result = fusion.get_predicted_world_lr(CameraEnum.CAMERA_LEFT.value)
        assert result == 35.0

    def test_prediction_stationary_returns_current(self):
        """Multiple detections at the same angle should not predict movement."""
        fusion = CameraFusionState()
        for _ in range(5):
            fusion.update_side_detection(CameraEnum.CAMERA_LEFT.value, 35.0, 1)
        result = fusion.get_predicted_world_lr(CameraEnum.CAMERA_LEFT.value)
        assert result == pytest.approx(35.0, abs=1.0)

    def test_prediction_moving_target_leads_ahead(self):
        """Moving target should return an angle ahead of current position."""
        fusion = CameraFusionState()
        now = time.time()
        # Simulate target moving from 30 to 50 degrees over 1 second
        fusion._left_angle_history = [
            (now - 0.8, 30.0),
            (now - 0.6, 35.0),
            (now - 0.4, 40.0),
            (now - 0.2, 45.0),
            (now, 50.0),
        ]
        fusion._left_world_lr = 50.0
        fusion._left_last_seen = now
        result = fusion.get_predicted_world_lr(CameraEnum.CAMERA_LEFT.value)
        # Velocity is ~25 deg/s, lead time 0.75s → predict ~18.75° ahead
        # But clamped to max offset of 15°
        assert result > 50.0
        assert result <= 50.0 + FusionEnums.PREDICTION_MAX_OFFSET.value

    def test_prediction_clamped_to_max_offset(self):
        """Very fast target should not predict beyond max offset."""
        fusion = CameraFusionState()
        now = time.time()
        # Very fast: 0 to 100 degrees in 1 second
        fusion._left_angle_history = [(now - 1.0, 0.0), (now, 100.0)]
        fusion._left_world_lr = 100.0
        fusion._left_last_seen = now
        result = fusion.get_predicted_world_lr(CameraEnum.CAMERA_LEFT.value)
        max_predicted = 100.0 + FusionEnums.PREDICTION_MAX_OFFSET.value
        assert result <= max_predicted

    def test_prediction_history_trimmed(self):
        """Old entries should be removed from history."""
        fusion = CameraFusionState()
        now = time.time()
        # Add old entries that should be trimmed
        fusion._left_angle_history = [
            (now - 5.0, 10.0),  # very old — should be trimmed
            (now - 4.0, 20.0),  # old — should be trimmed
        ]
        # Add new entry — triggers trim
        fusion.update_side_detection(CameraEnum.CAMERA_LEFT.value, 50.0, 1)
        window = FusionEnums.PREDICTION_HISTORY_WINDOW.value
        for t, _ in fusion._left_angle_history:
            assert now - t <= window + 0.1  # allow small timing tolerance


class TestPeripheralConfirmation:
    """Tests for head + side camera agreement detection."""

    def test_confirmed_when_cameras_agree(self):
        fusion = CameraFusionState()
        fusion.update_side_detection(CameraEnum.CAMERA_LEFT.value, 88.0, 1)
        assert fusion.is_confirmed_by_side(90.0)

    def test_not_confirmed_when_cameras_disagree(self):
        fusion = CameraFusionState()
        fusion.update_side_detection(CameraEnum.CAMERA_LEFT.value, 120.0, 1)
        assert not fusion.is_confirmed_by_side(90.0)

    def test_not_confirmed_when_side_stale(self):
        fusion = CameraFusionState()
        fusion.update_side_detection(CameraEnum.CAMERA_LEFT.value, 90.0, 1)
        fusion._left_last_seen = time.time() - FusionEnums.SIDE_CAMERA_STALENESS.value - 1.0
        assert not fusion.is_confirmed_by_side(90.0)


class TestMultiPersonSelection:
    """Tests for human-like target selection scoring."""

    def test_select_prefers_nearby_angle(self):
        """With prior target near center, prefer nearby person over far-off one with same confidence."""
        from Tests.test_tracking_pipeline import _make_motion_track
        mt = _make_motion_track()
        # Set last tracked angle to match what FK returns for center of frame
        center_box = {"x1": 290, "y1": 200, "x2": 350, "y2": 300}
        center_angle = mt._pixel_to_world_angle(
            center_box, CameraEnum.CAMERA_HEAD.value, ServoEnum.X_AXIS.value)
        mt._last_tracked_world_lr = center_angle
        mt._last_tracked_bbox_height = 200.0

        # Same confidence, same height — only angle differs
        nearby = {"confidence": 0.8, "box": {"x1": 290, "y1": 100, "x2": 350, "y2": 300}}   # near center
        far = {"confidence": 0.8, "box": {"x1": 10, "y1": 100, "x2": 70, "y2": 300}}         # far left edge
        result = mt._MotionTrack__select_target([nearby, far], CameraEnum.CAMERA_HEAD.value)
        assert result == nearby

    def test_select_prefers_similar_height(self):
        """Prefer person with similar bbox height (distance proxy) over shorter one far away."""
        from Tests.test_tracking_pipeline import _make_motion_track
        mt = _make_motion_track()
        mt._last_tracked_world_lr = 90.0
        mt._last_tracked_bbox_height = 200.0

        # Both at similar angles, but different heights
        close_dist = {"confidence": 0.7, "box": {"x1": 290, "y1": 100, "x2": 350, "y2": 300}}  # height=200
        far_dist = {"confidence": 0.7, "box": {"x1": 300, "y1": 200, "x2": 360, "y2": 280}}  # height=80
        result = mt._MotionTrack__select_target([close_dist, far_dist], CameraEnum.CAMERA_HEAD.value)
        assert result == close_dist

    def test_select_confidence_breaks_ties(self):
        """When proximity and height are equal, higher confidence wins."""
        from Tests.test_tracking_pipeline import _make_motion_track
        mt = _make_motion_track()
        mt._last_tracked_world_lr = 90.0
        mt._last_tracked_bbox_height = 200.0

        low_conf = {"confidence": 0.6, "box": {"x1": 290, "y1": 100, "x2": 350, "y2": 300}}
        high_conf = {"confidence": 0.95, "box": {"x1": 290, "y1": 100, "x2": 350, "y2": 300}}
        result = mt._MotionTrack__select_target([low_conf, high_conf], CameraEnum.CAMERA_HEAD.value)
        assert result == high_conf

    def test_select_no_history_uses_confidence(self):
        """First detection ever — fall back to highest confidence."""
        from Tests.test_tracking_pipeline import _make_motion_track
        mt = _make_motion_track()
        mt._last_tracked_world_lr = None

        low = {"confidence": 0.5, "box": {"x1": 290, "y1": 100, "x2": 350, "y2": 300}}
        high = {"confidence": 0.9, "box": {"x1": 50, "y1": 100, "x2": 110, "y2": 300}}
        result = mt._MotionTrack__select_target([low, high], CameraEnum.CAMERA_HEAD.value)
        assert result == high

    def test_room_person_count(self):
        fusion = CameraFusionState()
        fusion.update_head_count(1)
        fusion._left_count = 1
        fusion._right_count = 1
        # max(1, 1+1) = 2
        assert fusion.get_room_person_count() == 2
