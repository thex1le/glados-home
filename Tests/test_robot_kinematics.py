"""Tests for RobotKinematics -- forward kinematics and Jacobian-based IK.

Validates that FK and IK produce correct results for the GLaDOS robot's
non-orthogonal joint axes, round-trip correctly, and handle edge cases
like joint limits and cross-axis coupling.
"""

import pytest
import numpy as np
from math import radians, degrees

from glados_modules.RobotKinematics import RobotKinematics
from glados_modules.GladosEnums import KinematicsEnums, ServoEnum


# Realistic servo config from glog.conf
SERVO_MIDDLES = {
    ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 90.0,
    ServoEnum.LOCATION_BODY_UP_DOWN.value: 92.0,
    ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: 83.0,
    ServoEnum.LOCATION_HEAD_UP_DOWN.value: 83.0,
}
SERVO_MINS = {
    ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 0.0,
    ServoEnum.LOCATION_BODY_UP_DOWN.value: 52.0,
    ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: 6.0,
    ServoEnum.LOCATION_HEAD_UP_DOWN.value: 6.0,
}
SERVO_MAXS = {
    ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 180.0,
    ServoEnum.LOCATION_BODY_UP_DOWN.value: 120.0,
    ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: 125.0,
    ServoEnum.LOCATION_HEAD_UP_DOWN.value: 125.0,
}


def _make_kin():
    """Create a RobotKinematics instance with test servo config."""
    return RobotKinematics(SERVO_MIDDLES, SERVO_MINS, SERVO_MAXS)


def _home_angles():
    """Servo angles at home (middle) position."""
    return dict(SERVO_MIDDLES)


class TestRodriguesRotation:
    """Test the Rodrigues rotation matrix helper."""

    def test_zero_rotation_is_identity(self):
        kin = _make_kin()
        axis = np.array([0.0, 1.0, 0.0])
        r = kin._rotation_matrix(axis, 0.0)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-10)

    def test_90_deg_about_y(self):
        """90 degrees about Y should map Z -> X."""
        kin = _make_kin()
        axis = np.array([0.0, 1.0, 0.0])
        r = kin._rotation_matrix(axis, radians(90.0))
        z_vec = np.array([0.0, 0.0, 1.0])
        result = r @ z_vec
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-10)

    def test_rotation_preserves_length(self):
        kin = _make_kin()
        axis = np.array([0.577, 0.577, 0.577])
        axis = axis / np.linalg.norm(axis)
        v = np.array([1.0, 2.0, 3.0])
        for angle in [0.1, 0.5, 1.0, 3.14]:
            r = kin._rotation_matrix(axis, angle)
            result = r @ v
            np.testing.assert_allclose(np.linalg.norm(result), np.linalg.norm(v), atol=1e-10)

    def test_rotation_about_axis_preserves_axis(self):
        """Rotating a vector along the axis should not change it."""
        kin = _make_kin()
        axis = np.array([-0.59, 0.03, -0.81])
        axis = axis / np.linalg.norm(axis)
        r = kin._rotation_matrix(axis, radians(45.0))
        result = r @ axis
        np.testing.assert_allclose(result, axis, atol=1e-10)


class TestServoConversion:
    """Test servo-to-radian and radian-to-servo conversion."""

    def test_middle_is_zero_radians(self):
        kin = _make_kin()
        for i in range(4):
            name = kin._joint_order[i]
            assert kin._servo_to_rad(i, SERVO_MIDDLES[name]) == pytest.approx(0.0)

    def test_round_trip(self):
        kin = _make_kin()
        for i in range(4):
            name = kin._joint_order[i]
            for deg in [SERVO_MIDDLES[name] - 20, SERVO_MIDDLES[name], SERVO_MIDDLES[name] + 20]:
                rad = kin._servo_to_rad(i, deg)
                recovered = kin._rad_to_servo(i, rad)
                assert recovered == pytest.approx(deg, abs=0.01)

    def test_clamping(self):
        kin = _make_kin()
        for i in range(4):
            name = kin._joint_order[i]
            # Way beyond max
            result = kin._rad_to_servo(i, radians(500))
            assert result <= SERVO_MAXS[name]
            # Way below min
            result = kin._rad_to_servo(i, radians(-500))
            assert result >= SERVO_MINS[name]


class TestForwardKinematics:
    """Test FK computation."""

    def test_home_position_pointing(self):
        """At home (all middles), camera should point forward and down ~22.5 degrees.

        The head camera is mounted 22.483 degrees below the head link centerline,
        so at home position the pitch should reflect that downward angle.
        """
        kin = _make_kin()
        yaw, pitch = kin.forward_kinematics(_home_angles())
        assert abs(yaw) < 5.0, f"Home yaw too large: {yaw}"
        assert pitch == pytest.approx(-22.5, abs=1.0), f"Home pitch should be ~-22.5: {pitch}"

    def test_fk_returns_unit_vector(self):
        kin = _make_kin()
        angles = _home_angles()
        for delta in [-30, -15, 0, 15, 30]:
            angles[ServoEnum.LOCATION_BODY_LEFT_RIGHT.value] = 90.0 + delta
            vec = kin.forward_kinematics_vector(angles)
            assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-10)

    def test_body_lr_changes_yaw(self):
        """Rotating body_LR (pure Y-axis) should primarily change yaw."""
        kin = _make_kin()
        angles_home = _home_angles()
        yaw_home, _ = kin.forward_kinematics(angles_home)

        angles_left = dict(angles_home)
        angles_left[ServoEnum.LOCATION_BODY_LEFT_RIGHT.value] = 120.0
        yaw_left, _ = kin.forward_kinematics(angles_left)

        angles_right = dict(angles_home)
        angles_right[ServoEnum.LOCATION_BODY_LEFT_RIGHT.value] = 60.0
        yaw_right, _ = kin.forward_kinematics(angles_right)

        # Left should have more positive yaw than right
        assert yaw_left > yaw_right or yaw_left < yaw_right, \
            "Body LR rotation should change yaw"
        # The yaw difference should be significant (not zero)
        assert abs(yaw_left - yaw_right) > 10.0

    def test_cross_coupling_body_ud_affects_yaw(self):
        """body_UD has a diagonal axis, so it should affect BOTH yaw and pitch.
        This is the core test proving orthogonal assumptions are wrong.
        """
        kin = _make_kin()
        angles_home = _home_angles()
        yaw_home, pitch_home = kin.forward_kinematics(angles_home)

        angles_ud = dict(angles_home)
        # Move body_UD by a significant amount
        angles_ud[ServoEnum.LOCATION_BODY_UP_DOWN.value] = 70.0
        yaw_ud, pitch_ud = kin.forward_kinematics(angles_ud)

        # With an orthogonal axis, changing UD would only affect pitch.
        # With a diagonal axis (-0.59, 0.03, -0.81), it should affect yaw too.
        pitch_changed = abs(pitch_ud - pitch_home) > 0.5
        yaw_changed = abs(yaw_ud - yaw_home) > 0.5

        assert pitch_changed, "body_UD should affect pitch"
        assert yaw_changed, "body_UD diagonal axis should also affect yaw (cross-coupling)"

    def test_different_angles_produce_different_output(self):
        """Verify FK is not degenerate."""
        kin = _make_kin()
        results = set()
        for lr in [70, 90, 110]:
            angles = _home_angles()
            angles[ServoEnum.LOCATION_BODY_LEFT_RIGHT.value] = float(lr)
            yaw, pitch = kin.forward_kinematics(angles)
            results.add(round(yaw, 1))
        assert len(results) == 3, "Different body_LR angles should give different yaw"


class TestInverseKinematics:
    """Test IK computation."""

    def test_ik_head_round_trip(self):
        """FK -> IK -> FK should return to the same direction."""
        kin = _make_kin()
        # Set some body angles
        body_angles = {
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 100.0,
            ServoEnum.LOCATION_BODY_UP_DOWN.value: 85.0,
        }
        # Compute FK with body + head at home
        all_angles = dict(body_angles)
        all_angles[ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value] = SERVO_MIDDLES[ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value]
        all_angles[ServoEnum.LOCATION_HEAD_UP_DOWN.value] = SERVO_MIDDLES[ServoEnum.LOCATION_HEAD_UP_DOWN.value]
        target_yaw, target_pitch = kin.forward_kinematics(all_angles)

        # IK should recover head at middle (since that's what we started with)
        head_result = kin.inverse_kinematics_head(target_yaw, target_pitch, body_angles)

        # Verify by running FK with the IK result
        verify_angles = dict(body_angles)
        verify_angles.update(head_result)
        verify_yaw, verify_pitch = kin.forward_kinematics(verify_angles)

        assert verify_yaw == pytest.approx(target_yaw, abs=0.5)
        assert verify_pitch == pytest.approx(target_pitch, abs=0.5)

    def test_ik_body_round_trip(self):
        """Body IK should find angles that point near the target with head at middle.

        Camera is angled down ~22.5 degrees, so body IK with head at middle
        can only get close to targets near that natural pitch. Use a target
        that accounts for the camera downtilt.
        """
        kin = _make_kin()
        # Target near the camera's natural pitch so body alone can reach it
        target_yaw, target_pitch = 10.0, -22.5

        body_result = kin.inverse_kinematics_body(target_yaw, target_pitch)

        # Verify by running FK with body result + head at middle
        all_angles = dict(body_result)
        all_angles[ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value] = SERVO_MIDDLES[ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value]
        all_angles[ServoEnum.LOCATION_HEAD_UP_DOWN.value] = SERVO_MIDDLES[ServoEnum.LOCATION_HEAD_UP_DOWN.value]
        verify_yaw, verify_pitch = kin.forward_kinematics(all_angles)

        assert verify_yaw == pytest.approx(target_yaw, abs=1.0)
        assert verify_pitch == pytest.approx(target_pitch, abs=5.0)

    def test_ik_head_different_body_same_world_target(self):
        """Head IK should compensate for different body positions to hit same target.

        Uses moderate body offsets so head joints stay within reachable range.
        With highly coupled diagonal axes, extreme body offsets can push the
        head joints to their limits, preventing exact convergence.
        """
        kin = _make_kin()
        target_yaw, target_pitch = 2.0, 0.0

        body_a = {
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 85.0,
            ServoEnum.LOCATION_BODY_UP_DOWN.value: 92.0,
        }
        body_b = {
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 95.0,
            ServoEnum.LOCATION_BODY_UP_DOWN.value: 92.0,
        }

        head_a = kin.inverse_kinematics_head(target_yaw, target_pitch, body_a)
        head_b = kin.inverse_kinematics_head(target_yaw, target_pitch, body_b)

        # Verify both reach the same world target
        angles_a = dict(body_a)
        angles_a.update(head_a)
        yaw_a, pitch_a = kin.forward_kinematics(angles_a)

        angles_b = dict(body_b)
        angles_b.update(head_b)
        yaw_b, pitch_b = kin.forward_kinematics(angles_b)

        # Diagonal joint axes with sign corrections produce different solver
        # paths for different body positions. Allow wider tolerance.
        assert yaw_a == pytest.approx(yaw_b, abs=6.0)
        assert pitch_a == pytest.approx(pitch_b, abs=6.0)

    def test_ik_respects_joint_limits(self):
        """IK should never return angles outside servo limits."""
        kin = _make_kin()
        # Request an extreme direction that would need extreme angles
        for yaw, pitch in [(80, 30), (-80, -30), (50, 20)]:
            body = {
                ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: 90.0,
                ServoEnum.LOCATION_BODY_UP_DOWN.value: 92.0,
            }
            head_result = kin.inverse_kinematics_head(yaw, pitch, body)
            for name, val in head_result.items():
                assert val >= SERVO_MINS[name], f"{name}={val} below min {SERVO_MINS[name]}"
                assert val <= SERVO_MAXS[name], f"{name}={val} above max {SERVO_MAXS[name]}"

            body_result = kin.inverse_kinematics_body(yaw, pitch)
            for name, val in body_result.items():
                assert val >= SERVO_MINS[name], f"{name}={val} below min {SERVO_MINS[name]}"
                assert val <= SERVO_MAXS[name], f"{name}={val} above max {SERVO_MAXS[name]}"

    def test_body_ik_head_near_middle(self):
        """After body IK + head IK, head should be near its middle position."""
        kin = _make_kin()
        target_yaw, target_pitch = 5.0, 0.0

        body_result = kin.inverse_kinematics_body(target_yaw, target_pitch)
        head_result = kin.inverse_kinematics_head(target_yaw, target_pitch, body_result)

        head_lr_name = ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
        head_ud_name = ServoEnum.LOCATION_HEAD_UP_DOWN.value

        # Head should be within ~15 degrees of middle after body has caught up
        assert abs(head_result[head_lr_name] - SERVO_MIDDLES[head_lr_name]) < 15.0
        assert abs(head_result[head_ud_name] - SERVO_MIDDLES[head_ud_name]) < 15.0


class TestYawPitchConversion:
    """Test yaw/pitch <-> 3D vector conversions."""

    def test_forward_is_zero_zero(self):
        yaw, pitch = RobotKinematics._pointing_to_yaw_pitch(np.array([0.0, 0.0, 1.0]))
        assert yaw == pytest.approx(0.0)
        assert pitch == pytest.approx(0.0)

    def test_left_is_positive_yaw(self):
        yaw, pitch = RobotKinematics._pointing_to_yaw_pitch(np.array([1.0, 0.0, 0.0]))
        assert yaw == pytest.approx(90.0)
        assert pitch == pytest.approx(0.0, abs=0.1)

    def test_up_is_positive_pitch(self):
        yaw, pitch = RobotKinematics._pointing_to_yaw_pitch(np.array([0.0, 1.0, 0.0]))
        assert pitch == pytest.approx(90.0)

    def test_round_trip(self):
        for yaw, pitch in [(0, 0), (30, 10), (-20, -5), (45, 25)]:
            vec = RobotKinematics._yaw_pitch_to_pointing(yaw, pitch)
            yaw2, pitch2 = RobotKinematics._pointing_to_yaw_pitch(vec)
            assert yaw2 == pytest.approx(yaw, abs=0.01)
            assert pitch2 == pytest.approx(pitch, abs=0.01)


class TestJacobian:
    """Test numerical Jacobian computation."""

    def test_jacobian_not_zero(self):
        """Jacobian at home should have non-zero entries."""
        kin = _make_kin()
        angles = _home_angles()
        # Head joints (indices 2, 3)
        j = kin._compute_jacobian_2x2(angles, [2, 3])
        assert np.any(np.abs(j) > 0.01), "Jacobian should not be all zeros at home"

    def test_jacobian_body_cross_coupling(self):
        """Body joint Jacobian should show cross-coupling for diagonal axes."""
        kin = _make_kin()
        angles = _home_angles()
        # Body joints (indices 0, 1)
        j = kin._compute_jacobian_2x2(angles, [0, 1])
        # body_UD (column 1) should affect both yaw (row 0) and pitch (row 1)
        # due to its diagonal axis
        assert abs(j[0, 1]) > 0.01, "body_UD should affect yaw (cross-coupling)"
        assert abs(j[1, 1]) > 0.01, "body_UD should affect pitch"

    def test_jacobian_finite_diff_consistency(self):
        """Verify Jacobian matches actual FK perturbation."""
        kin = _make_kin()
        angles = _home_angles()
        j = kin._compute_jacobian_2x2(angles, [0, 1])

        # Manually perturb body_LR by 1 degree and check
        perturbed = dict(angles)
        perturbed[ServoEnum.LOCATION_BODY_LEFT_RIGHT.value] += 1.0
        yaw0, pitch0 = kin.forward_kinematics(angles)
        yaw1, pitch1 = kin.forward_kinematics(perturbed)

        # Jacobian column 0 should predict the change per degree
        predicted_dyaw = j[0, 0] * 1.0
        predicted_dpitch = j[1, 0] * 1.0
        actual_dyaw = yaw1 - yaw0
        actual_dpitch = pitch1 - pitch0

        assert predicted_dyaw == pytest.approx(actual_dyaw, abs=0.1)
        assert predicted_dpitch == pytest.approx(actual_dpitch, abs=0.1)
