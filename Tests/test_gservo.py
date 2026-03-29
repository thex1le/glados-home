"""Tests for Gservo command parsing, speed/angle clamping, and spring param selection."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from collections import namedtuple
from json import dumps

from glados_modules.GladosEnums import ServoEnum, MotionProfile, TraceEnums


def _make_gservo(location="head_left_right"):
    """Create a Gservo with mocked hardware."""
    from glados_modules.BodyControlModules import Gservo
    servo_range = namedtuple("range", ["max", "min", "center"])(180, 0, 90)
    broker = namedtuple("broker", ["ip", "port"])("localhost", 1883)
    mock_servo = MagicMock()
    with patch.object(Gservo, '__init__', lambda self, *a, **kw: None):
        g = Gservo.__new__(Gservo)
    # Manually init the fields we need for testing
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
    g.logger = MagicMock()
    g._lock = __import__('threading').Lock()
    g.cmd_topic = ServoEnum.MQTT_COMMAND_TOPIC.value
    g.status_topic = ServoEnum.MQTT_STATUS_TOPIC.value
    g.client = MagicMock()
    g.send_command = MagicMock()
    g.send_status = MagicMock()
    g.__name__ = f"Gservo_{location}"
    return g


class TestGservoSetSpeed:
    def test_valid_speed(self):
        g = _make_gservo()
        g._update_spring_params = MagicMock()
        g.set_speed(3)
        assert g.speed == 3

    def test_clamp_low(self):
        g = _make_gservo()
        g._update_spring_params = MagicMock()
        g.set_speed(-5)
        assert g.speed == 1

    def test_clamp_high(self):
        g = _make_gservo()
        g._update_spring_params = MagicMock()
        g.set_speed(99)
        assert g.speed == 5

    def test_float_rounds(self):
        g = _make_gservo()
        g._update_spring_params = MagicMock()
        g.set_speed(3.7)
        assert g.speed == 4


class TestGservoSetAngle:
    def test_valid_angle(self):
        g = _make_gservo()
        g.set_angle(45)
        assert g.target_angle == 45.0
        assert g.angle == 45.0

    def test_clamp_below_min(self):
        g = _make_gservo()
        g.set_angle(-10)
        assert g.target_angle == 0.0

    def test_clamp_above_max(self):
        g = _make_gservo()
        g.set_angle(200)
        assert g.target_angle == 180.0

    def test_exact_boundaries(self):
        g = _make_gservo()
        g.set_angle(0)
        assert g.target_angle == 0.0
        g.set_angle(180)
        assert g.target_angle == 180.0


class TestGservoSpringParams:
    def test_head_servo_gets_head_params(self):
        g = _make_gservo("head_left_right")
        g._update_spring_params(3)
        expected_omega, expected_zeta = MotionProfile.HEAD_PARAMS.value[3]
        assert g.omega == expected_omega
        assert g.zeta == expected_zeta

    def test_body_servo_gets_body_params(self):
        g = _make_gservo("body_left_right")
        g._update_spring_params(3)
        expected_omega, expected_zeta = MotionProfile.BODY_PARAMS.value[3]
        assert g.omega == expected_omega
        assert g.zeta == expected_zeta

    def test_speed_clamped_in_params(self):
        g = _make_gservo()
        g._update_spring_params(0)  # below 1
        assert g.omega == MotionProfile.HEAD_PARAMS.value[1][0]
        g._update_spring_params(10)  # above 5
        assert g.omega == MotionProfile.HEAD_PARAMS.value[5][0]


class TestGservoHandleCmd:
    def _make_mqtt_msg(self, payload_dict):
        msg = MagicMock()
        msg.payload.decode.return_value = dumps(payload_dict)
        return msg

    def test_move_command(self):
        g = _make_gservo("head_left_right")
        g._update_spring_params = MagicMock()
        msg = self._make_mqtt_msg({
            ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_MOVE.value,
            ServoEnum.MSG_LOCATION_KEY.value: "head_left_right",
            ServoEnum.MSG_ANGLE.value: 45,
            ServoEnum.MSG_SPEED.value: 2,
        })
        g.handle_cmd(msg)
        assert g.target_angle == 45.0
        assert g.speed == 2

    def test_move_all_command(self):
        g = _make_gservo("head_left_right")
        g._update_spring_params = MagicMock()
        msg = self._make_mqtt_msg({
            ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_MOVE_ALL.value,
            ServoEnum.MSG_TARGETS.value: {
                "head_left_right": {ServoEnum.MSG_ANGLE.value: 60, ServoEnum.MSG_SPEED.value: 4},
                "body_left_right": {ServoEnum.MSG_ANGLE.value: 100, ServoEnum.MSG_SPEED.value: 2},
            }
        })
        g.handle_cmd(msg)
        assert g.target_angle == 60.0
        assert g.speed == 4

    def test_move_all_ignores_other_servos(self):
        g = _make_gservo("head_left_right")
        g._update_spring_params = MagicMock()
        msg = self._make_mqtt_msg({
            ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_MOVE_ALL.value,
            ServoEnum.MSG_TARGETS.value: {
                "body_left_right": {ServoEnum.MSG_ANGLE.value: 100, ServoEnum.MSG_SPEED.value: 2},
            }
        })
        g.handle_cmd(msg)
        assert g.target_angle == 90.0  # unchanged

    def test_move_all_with_trace_logs_latency(self):
        g = _make_gservo("head_left_right")
        g._update_spring_params = MagicMock()
        msg = self._make_mqtt_msg({
            ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_MOVE_ALL.value,
            ServoEnum.MSG_TARGETS.value: {
                "head_left_right": {ServoEnum.MSG_ANGLE.value: 60, ServoEnum.MSG_SPEED.value: 3},
            },
            TraceEnums.TRACE_ID.value: "abc123",
            TraceEnums.TS_VISION.value: 1000.0,
        })
        g.handle_cmd(msg)
        # Should have logged with trace info in one of the debug calls
        all_debug_msgs = [call[0][0] for call in g.logger.debug.call_args_list]
        trace_msgs = [m for m in all_debug_msgs if "trace=" in m]
        assert len(trace_msgs) > 0, f"No trace log found in: {all_debug_msgs}"
        assert "latency=" in trace_msgs[0]

    def test_status_command(self):
        g = _make_gservo("head_left_right")
        msg = self._make_mqtt_msg({
            ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_STATUS.value,
            ServoEnum.MSG_LOCATION_KEY.value: "head_left_right",
        })
        g.handle_cmd(msg)
        assert g.send_status.called

    def test_wrong_location_ignored(self):
        g = _make_gservo("head_left_right")
        g._update_spring_params = MagicMock()
        msg = self._make_mqtt_msg({
            ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_MOVE.value,
            ServoEnum.MSG_LOCATION_KEY.value: "body_up_down",
            ServoEnum.MSG_ANGLE.value: 45,
        })
        g.handle_cmd(msg)
        assert g.target_angle == 90.0  # unchanged

    def test_malformed_json(self):
        g = _make_gservo()
        msg = MagicMock()
        msg.payload.decode.return_value = "not json"
        g.handle_cmd(msg)
        assert g.logger.error.called


class TestGservoGetAngles:
    def test_returns_all_required_keys(self):
        g = _make_gservo()
        g.position = 95.5
        g.velocity = 1.2
        g.moving = True
        g.target_angle = 100.0
        g.axis = "x"
        result = g.get_angles()
        assert ServoEnum.MSG_CURRENT_ANGLE.value in result
        assert ServoEnum.MSG_VELOCITY.value in result
        assert ServoEnum.MSG_MAX.value in result
        assert ServoEnum.MSG_MIN.value in result
        assert ServoEnum.MSG_MIDDLE.value in result
        assert ServoEnum.MSG_AXIS.value in result
        assert ServoEnum.MSG_MOVING.value in result
        assert ServoEnum.MSG_LAST_ANGLE.value in result
        assert result[ServoEnum.MSG_MOVING.value] is True
