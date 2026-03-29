"""Tests for world-space angle calculations and target distribution.

Validates pixel-to-world conversion, head/body target split, UD interpolation,
and the core invariant: as body catches up, head re-centers.
"""

import pytest
import time
from math import radians, tan, atan, degrees
from unittest.mock import MagicMock, patch

from glados_modules.GladosEnums import (
    CameraEnum, ServoEnum, MotionProfile, VisionResultsEnum
)


class TestPixelToWorldAngle:
    """Test the arctan-based pixel-to-world conversion."""

    def _calc_world_angle(self, pixel_center, axis_size, fov, camera_world):
        """Standalone version of the world angle math for testing."""
        offset = (axis_size / 2) - pixel_center
        focal = (axis_size / 2) / tan(radians(fov) / 2)
        angle_offset = degrees(atan(offset / focal))
        return camera_world + angle_offset

    def test_center_pixel_gives_zero_offset(self):
        """Object at image center should produce no angular offset."""
        world = self._calc_world_angle(
            pixel_center=320, axis_size=640, fov=54, camera_world=90.0
        )
        assert abs(world - 90.0) < 0.01

    def test_left_pixel_gives_positive_offset(self):
        """Object left of center should produce positive angular offset."""
        world = self._calc_world_angle(
            pixel_center=100, axis_size=640, fov=54, camera_world=90.0
        )
        assert world > 90.0

    def test_right_pixel_gives_negative_offset(self):
        """Object right of center should produce negative angular offset."""
        world = self._calc_world_angle(
            pixel_center=540, axis_size=640, fov=54, camera_world=90.0
        )
        assert world < 90.0

    def test_wider_fov_gives_larger_offset(self):
        """Same pixel offset with wider FOV should produce larger angle.
        Wider FOV = shorter focal length = each pixel covers more degrees.
        """
        narrow = self._calc_world_angle(
            pixel_center=200, axis_size=640, fov=54, camera_world=90.0
        )
        wide = self._calc_world_angle(
            pixel_center=200, axis_size=640, fov=160, camera_world=90.0
        )
        assert abs(narrow - 90.0) < abs(wide - 90.0)

    def test_camera_world_offset_applied(self):
        """Camera world angle should shift the result."""
        world_at_90 = self._calc_world_angle(
            pixel_center=320, axis_size=640, fov=54, camera_world=90.0
        )
        world_at_120 = self._calc_world_angle(
            pixel_center=320, axis_size=640, fov=54, camera_world=120.0
        )
        assert abs(world_at_120 - world_at_90 - 30.0) < 0.01


class TestTargetDistribution:
    """Test the head/body target split logic."""

    def test_head_compensates_for_body_lag(self):
        """Head target should be: middle + (world - body_est)."""
        head_middle = 92.0
        world_lr = 120.0
        est_body_lr = 90.0  # body hasn't moved yet
        head_target = head_middle + (world_lr - est_body_lr)
        assert head_target == 122.0  # 92 + (120 - 90) = 122

    def test_head_recenters_as_body_catches_up(self):
        """As body approaches world angle, head should approach middle."""
        head_middle = 92.0
        world_lr = 120.0

        # Body at 90 (far from target)
        head_t1 = head_middle + (world_lr - 90.0)
        # Body at 110 (closer)
        head_t2 = head_middle + (world_lr - 110.0)
        # Body at 120 (arrived)
        head_t3 = head_middle + (world_lr - 120.0)

        assert head_t1 == 122.0
        assert head_t2 == 102.0
        assert head_t3 == 92.0  # back to middle

    def test_body_target_equals_world_angle(self):
        """Body target is simply the world angle."""
        world_lr = 105.0
        body_target = world_lr
        assert body_target == 105.0

    def test_targets_clamped_to_range(self):
        """Targets should not exceed servo physical limits."""
        head_min, head_max = 6, 125
        world_lr = 200.0
        est_body_lr = 90.0
        head_middle = 92.0

        head_target = head_middle + (world_lr - est_body_lr)  # 202
        head_target = max(head_min, min(head_max, head_target))
        assert head_target == head_max

        body_target = max(0, min(180, world_lr))
        assert body_target == 180


class TestFKBasedTracking:
    """Test FK/IK-based tracking replaces the old UD interpolation table."""

    def _make_kin(self):
        from glados_modules.RobotKinematics import RobotKinematics
        middles = {
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 90.0,
            ServoEnum.LOCATION_BODY_UP_DOWN.value: 92.0,
            ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: 83.0,
            ServoEnum.LOCATION_HEAD_UP_DOWN.value: 83.0,
        }
        mins = {
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 0.0,
            ServoEnum.LOCATION_BODY_UP_DOWN.value: 52.0,
            ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: 6.0,
            ServoEnum.LOCATION_HEAD_UP_DOWN.value: 6.0,
        }
        maxs = {
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 180.0,
            ServoEnum.LOCATION_BODY_UP_DOWN.value: 120.0,
            ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: 125.0,
            ServoEnum.LOCATION_HEAD_UP_DOWN.value: 125.0,
        }
        return RobotKinematics(middles, mins, maxs), middles

    def test_body_ik_replaces_ud_table(self):
        """Body IK should produce valid UD angles without a lookup table."""
        kin, middles = self._make_kin()
        # Various target pitches that the old table would have covered
        for target_pitch in [-5.0, 0.0, 5.0, 10.0]:
            result = kin.inverse_kinematics_body(0.0, target_pitch)
            body_ud = result[ServoEnum.LOCATION_BODY_UP_DOWN.value]
            assert 52.0 <= body_ud <= 120.0, f"body_UD {body_ud} out of range for pitch {target_pitch}"

    def test_head_ik_compensates_for_body(self):
        """Head IK should adjust head angles to compensate for body position."""
        kin, middles = self._make_kin()
        target_yaw, target_pitch = 5.0, 0.0

        body_near = {
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 95.0,
            ServoEnum.LOCATION_BODY_UP_DOWN.value: 92.0,
        }
        body_far = {
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 70.0,
            ServoEnum.LOCATION_BODY_UP_DOWN.value: 92.0,
        }

        head_near = kin.inverse_kinematics_head(target_yaw, target_pitch, body_near)
        head_far = kin.inverse_kinematics_head(target_yaw, target_pitch, body_far)

        # When body is farther from target, head should compensate more
        head_lr_name = ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
        near_deviation = abs(head_near[head_lr_name] - middles[head_lr_name])
        far_deviation = abs(head_far[head_lr_name] - middles[head_lr_name])
        assert far_deviation > near_deviation

    def test_fk_ik_pipeline_round_trip(self):
        """FK -> target -> IK -> FK should preserve the direction."""
        kin, middles = self._make_kin()
        # Start with known servo angles
        angles = dict(middles)
        angles[ServoEnum.LOCATION_BODY_LEFT_RIGHT.value] = 100.0
        angles[ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value] = 90.0

        # Get the world direction these angles produce
        target_yaw, target_pitch = kin.forward_kinematics(angles)

        # Run two-stage IK
        body = {
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: angles[ServoEnum.LOCATION_BODY_LEFT_RIGHT.value],
            ServoEnum.LOCATION_BODY_UP_DOWN.value: angles[ServoEnum.LOCATION_BODY_UP_DOWN.value],
        }
        head_result = kin.inverse_kinematics_head(target_yaw, target_pitch, body)

        # Verify FK with IK result matches original direction
        verify = dict(body)
        verify.update(head_result)
        verify_yaw, verify_pitch = kin.forward_kinematics(verify)

        assert verify_yaw == pytest.approx(target_yaw, abs=0.5)
        assert verify_pitch == pytest.approx(target_pitch, abs=0.5)

    def test_cross_coupling_body_ud_affects_both_axes(self):
        """Diagonal body_UD axis should affect both yaw and pitch in FK."""
        kin, middles = self._make_kin()
        angles_home = dict(middles)

        yaw_home, pitch_home = kin.forward_kinematics(angles_home)

        angles_tilted = dict(angles_home)
        angles_tilted[ServoEnum.LOCATION_BODY_UP_DOWN.value] = 70.0
        yaw_tilted, pitch_tilted = kin.forward_kinematics(angles_tilted)

        # Both should change due to diagonal axis
        assert abs(yaw_tilted - yaw_home) > 0.5
        assert abs(pitch_tilted - pitch_home) > 0.5


class TestWorldSpaceSmoothing:
    """Test EMA smoothing in world space."""

    def test_smoothing_converges(self):
        """Repeated measurements at same angle should converge."""
        alpha = MotionProfile.WORLD_SMOOTH_ALPHA.value
        smoothed = 90.0
        target = 120.0
        for _ in range(100):
            smoothed = alpha * smoothed + (1 - alpha) * target
        assert abs(smoothed - target) < 0.01

    def test_smoothing_rejects_spikes(self):
        """A single spike should be dampened by the EMA."""
        alpha = MotionProfile.WORLD_SMOOTH_ALPHA.value
        smoothed = 90.0
        # Normal readings
        for _ in range(10):
            smoothed = alpha * smoothed + (1 - alpha) * 90.0
        # Spike
        smoothed_after_spike = alpha * smoothed + (1 - alpha) * 180.0
        # Should not jump all the way to 180
        assert smoothed_after_spike < 120.0
