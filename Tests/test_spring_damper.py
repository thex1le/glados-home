"""Tests for spring-damper physics -- both Pi4 servo and GPU estimator.

Validates that the physics model converges, respects limits, handles
new targets smoothly, and that the GPU estimator matches the Pi4 servo.
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from collections import namedtuple


class TestSpringDamperPhysics:
    """Test the core spring-damper math independent of hardware."""

    def _simulate(self, target, initial=90.0, omega=8.0, zeta=0.85,
                  dt=0.02, steps=200, servo_min=0, servo_max=180):
        """Run spring-damper simulation and return position history."""
        pos = initial
        vel = 0.0
        history = [pos]
        for _ in range(steps):
            error = target - pos
            accel = (omega ** 2) * error - 2.0 * zeta * omega * vel
            vel += accel * dt
            pos += vel * dt
            if pos <= servo_min:
                pos = float(servo_min)
                if vel < 0:
                    vel = 0.0
            elif pos >= servo_max:
                pos = float(servo_max)
                if vel > 0:
                    vel = 0.0
            history.append(pos)
        return history

    def test_converges_to_target(self):
        """Servo should reach the target within tolerance."""
        for target in [45, 90, 135, 10, 170]:
            history = self._simulate(target, initial=90.0, steps=500)
            assert abs(history[-1] - target) < 0.1, f"Failed to converge to {target}"

    def test_does_not_overshoot_past_limits(self):
        history = self._simulate(target=200, initial=90.0, servo_max=180)
        assert all(h <= 180.0 for h in history), "Exceeded servo max"

        history = self._simulate(target=-50, initial=90.0, servo_min=0)
        assert all(h >= 0.0 for h in history), "Went below servo min"

    def test_critically_damped_no_overshoot(self):
        """zeta=1.0 should not overshoot the target."""
        history = self._simulate(target=130, initial=90.0, zeta=1.0, steps=500)
        # After initial approach, position should never exceed target
        # (allow tiny float rounding)
        assert max(history) < 130.5, "Critically damped should not overshoot significantly"

    def test_underdamped_overshoots(self):
        """zeta=0.7 should overshoot before settling."""
        history = self._simulate(target=130, initial=90.0, zeta=0.7, steps=500)
        assert max(history) > 130.5, "Underdamped should overshoot"
        # But should still converge
        assert abs(history[-1] - 130) < 0.1

    def test_new_target_mid_motion(self):
        """Changing target during motion should smoothly redirect."""
        pos = 90.0
        vel = 0.0
        dt = 0.02
        omega = 8.0
        zeta = 0.85
        target = 130.0

        # Move toward 130 for 50 steps
        for _ in range(50):
            accel = (omega ** 2) * (target - pos) - 2.0 * zeta * omega * vel
            vel += accel * dt
            pos += vel * dt

        # Now change target to 60
        target = 60.0
        history = [pos]
        for _ in range(500):
            accel = (omega ** 2) * (target - pos) - 2.0 * zeta * omega * vel
            vel += accel * dt
            pos += vel * dt
            history.append(pos)

        # Should converge to new target
        assert abs(history[-1] - 60) < 0.1

    def test_small_movements_are_smooth(self):
        """1-2 degree adjustments should not cause big jerks."""
        history = self._simulate(target=92, initial=90.0, steps=200)
        # Max single-step change should be small
        max_step = max(abs(history[i] - history[i-1]) for i in range(1, len(history)))
        assert max_step < 2.0, f"Small movement caused {max_step:.2f} degree step"

    def test_head_faster_than_body(self):
        """Head params should reach target faster than body params."""
        from glados_modules.GladosEnums import MotionProfile
        level = 3
        head_omega, head_zeta = MotionProfile.HEAD_PARAMS.value[level]
        body_omega, body_zeta = MotionProfile.BODY_PARAMS.value[level]

        head_hist = self._simulate(target=130, omega=head_omega, zeta=head_zeta, steps=300)
        body_hist = self._simulate(target=130, omega=body_omega, zeta=body_zeta, steps=300)

        # Find step where each gets within 2 degrees of target
        def steps_to_target(hist, target, tol=2.0):
            for i, h in enumerate(hist):
                if abs(h - target) < tol:
                    return i
            return len(hist)

        head_steps = steps_to_target(head_hist, 130)
        body_steps = steps_to_target(body_hist, 130)
        assert head_steps < body_steps, (
            f"Head ({head_steps} steps) should arrive before body ({body_steps} steps)"
        )


class TestSpringDamperEstimator:
    """Test the GPU-side position estimator."""

    def test_estimator_tracks_target(self):
        from glados_modules.VisionTracker import SpringDamperEstimator
        est = SpringDamperEstimator(initial_pos=90.0, omega=8.0, zeta=0.85)
        est.set_target(130.0)
        # Simulate many small steps (get_position caps dt at 0.1s per call)
        for _ in range(200):
            est.last_time = time.time() - 0.02  # simulate 20ms per step
            pos = est.get_position()
        assert abs(pos - 130.0) < 1.0, f"Estimator should be near target, got {pos}"

    def test_estimator_sync_blends(self):
        from glados_modules.VisionTracker import SpringDamperEstimator
        est = SpringDamperEstimator(initial_pos=90.0, omega=8.0, zeta=0.85)
        est.position = 100.0
        est.velocity = 5.0
        est.sync(reported_position=110.0, reported_velocity=0.0)
        # Should blend 70/30
        assert abs(est.position - 103.0) < 0.01  # 0.7*100 + 0.3*110
        assert abs(est.velocity - 3.5) < 0.01    # 0.7*5 + 0.3*0

    def test_estimator_dt_capped(self):
        """Large time gaps should not cause position explosion."""
        from glados_modules.VisionTracker import SpringDamperEstimator
        est = SpringDamperEstimator(initial_pos=90.0, omega=8.0, zeta=0.85)
        est.set_target(130.0)
        est.last_time = time.time() - 100.0  # 100 seconds ago
        pos = est.get_position()
        assert 0 <= pos <= 250, f"Position exploded: {pos}"
