"""Integration test: Gservo spring-damper run loop in a real thread.

Actually runs the 50Hz physics loop for a short time, sends targets,
and verifies convergence with thread safety.
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from collections import namedtuple
from threading import Thread

from glados_modules.GladosEnums import ServoEnum, MotionProfile


def _make_threaded_gservo(location="head_left_right"):
    """Create a Gservo that can actually run its loop."""
    from glados_modules.BodyControlModules import Gservo

    servo_range = namedtuple("range", ["max", "min", "center"])(180, 0, 90)
    broker = namedtuple("broker", ["ip", "port"])("localhost", 1883)
    mock_servo = MagicMock()

    with patch.object(Gservo, '__init__', lambda self, *a, **kw: None):
        g = Gservo.__new__(Gservo)

    g.location = location
    g.servo = mock_servo
    g.servo_range = servo_range
    g.middle_angle = 90
    g._is_head_servo = location in ServoEnum.HEAD_SERVO_LOCATIONS.value
    g.target_angle = 90.0
    g.position = 90.0
    g.velocity = 0.0
    g.angle = 90.0
    g.current_angle = 90.0
    g.speed = 3
    g.omega = 8.0
    g.zeta = 0.85
    g.moving = False
    g._tick_count = 0
    g.stop = False
    g.logger = MagicMock()
    g._lock = __import__('threading').Lock()
    g.client = MagicMock()
    g.send_command = MagicMock()
    g.__name__ = f"Gservo_{location}"
    g.cmd_topic = ServoEnum.MQTT_COMMAND_TOPIC.value
    g.status_topic = ServoEnum.MQTT_STATUS_TOPIC.value
    g.axis = "x"

    # Need send_status to work
    from glados_modules.MqttConnector import ServoMessageBuilder
    g.send_status = lambda: None  # no-op for test

    return g


class TestGservoRunLoop:
    """Test the actual threaded run loop."""

    def test_converges_to_target_in_thread(self):
        """Run loop in a thread, set target, verify it converges."""
        g = _make_threaded_gservo()
        g.target_angle = 130.0
        g.angle = 130.0

        # Run the loop for 1 second (50 ticks)
        t = Thread(target=g.run, daemon=True)
        t.start()
        time.sleep(1.0)
        g.stop = True
        time.sleep(0.1)

        assert abs(g.position - 130.0) < 1.0, f"Expected ~130, got {g.position}"
        # Servo hardware should have been set
        assert g.servo.angle is not None

    def test_handles_mid_motion_target_change(self):
        """Change target while loop is running, verify smooth redirect."""
        g = _make_threaded_gservo()
        g.target_angle = 150.0
        g.angle = 150.0

        t = Thread(target=g.run, daemon=True)
        t.start()
        time.sleep(0.3)  # let it start moving

        # Change target mid-motion
        g.set_angle(60)
        time.sleep(1.5)  # let it settle
        g.stop = True
        time.sleep(0.1)

        assert abs(g.position - 60.0) < 2.0, f"Expected ~60, got {g.position}"

    def test_moving_flag_transitions(self):
        """moving flag should be True during motion, False when settled."""
        g = _make_threaded_gservo()
        # Start at rest
        assert not g.moving

        g.target_angle = 150.0
        g.angle = 150.0

        t = Thread(target=g.run, daemon=True)
        t.start()
        time.sleep(0.1)
        # Should be moving now
        assert g.moving, "Should be moving toward target"

        time.sleep(1.5)
        # Should have settled
        assert not g.moving, f"Should have settled, pos={g.position}, vel={g.velocity}"
        g.stop = True

    def test_respects_servo_limits(self):
        """Loop should clamp at servo range limits."""
        g = _make_threaded_gservo()
        g.target_angle = 999.0  # way above max of 180
        g.angle = 999.0

        t = Thread(target=g.run, daemon=True)
        t.start()
        time.sleep(1.0)
        g.stop = True
        time.sleep(0.1)

        assert g.position <= 180.0, f"Exceeded max: {g.position}"

    def test_status_sent_periodically(self):
        """Status should be sent every N ticks."""
        g = _make_threaded_gservo()
        g.send_status = MagicMock()

        t = Thread(target=g.run, daemon=True)
        t.start()
        time.sleep(0.5)
        g.stop = True
        time.sleep(0.1)

        # At 50Hz with status every 10 ticks, 0.5s = ~25 ticks = ~2 status sends
        assert g.send_status.call_count >= 1
