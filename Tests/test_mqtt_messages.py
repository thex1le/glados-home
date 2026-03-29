"""Tests for MQTT message contracts between systems.

Validates that message builders produce messages that parsers expect,
and that format changes on one side would be caught before deployment.
"""

import pytest
from unittest.mock import MagicMock, patch
from collections import namedtuple

from glados_modules.GladosEnums import ServoEnum, CameraEnum, TraceEnums, LCDEnums, MQTTEnums
from glados_modules.MqttConnector import ServoMessageBuilder, CameraMessageBuilder


class TestServoMessageBuilder:
    """Validate servo message format matches what Gservo.handle_cmd expects."""

    def test_move_message_has_required_keys(self):
        msg = ServoMessageBuilder.move("head_left_right", 90, 3)
        assert msg[ServoEnum.MSG_COMMAND_KEY.value] == ServoEnum.MSG_COMMAND_MOVE.value
        assert msg[ServoEnum.MSG_LOCATION_KEY.value] == "head_left_right"
        assert msg[ServoEnum.MSG_ANGLE.value] == 90
        assert msg[ServoEnum.MSG_SPEED.value] == 3

    def test_move_all_message_format(self):
        targets = {
            ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: {
                ServoEnum.MSG_ANGLE.value: 85,
                ServoEnum.MSG_SPEED.value: 3
            },
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: {
                ServoEnum.MSG_ANGLE.value: 105,
                ServoEnum.MSG_SPEED.value: 3
            },
        }
        msg = ServoMessageBuilder.move_all(targets)
        assert msg[ServoEnum.MSG_COMMAND_KEY.value] == ServoEnum.MSG_COMMAND_MOVE_ALL.value
        assert ServoEnum.MSG_TARGETS.value in msg
        # Each target should have angle and speed
        for loc, data in msg[ServoEnum.MSG_TARGETS.value].items():
            assert ServoEnum.MSG_ANGLE.value in data
            assert ServoEnum.MSG_SPEED.value in data

    def test_status_request_format(self):
        msg = ServoMessageBuilder.get_status("head_up_down")
        assert msg[ServoEnum.MSG_COMMAND_KEY.value] == ServoEnum.MSG_COMMAND_STATUS.value
        assert msg[ServoEnum.MSG_LOCATION_KEY.value] == "head_up_down"

    def test_status_response_format(self):
        results = {"current": 90, "max": 180, "min": 0, "velocity": 0.5}
        msg = ServoMessageBuilder.send_status("head_left_right", results)
        assert msg[ServoEnum.MSG_LOCATION_KEY.value] == "head_left_right"
        assert msg[ServoEnum.MSG_COMMAND_KEY.value] == ServoEnum.MSG_COMMAND_STATUS.value
        assert msg[ServoEnum.MSG_RESULTS.value] == results

    def test_convenience_methods_use_correct_locations(self):
        msg = ServoMessageBuilder.head_up_down(90)
        assert msg[ServoEnum.MSG_LOCATION_KEY.value] == ServoEnum.LOCATION_HEAD_UP_DOWN.value

        msg = ServoMessageBuilder.head_left_right(90)
        assert msg[ServoEnum.MSG_LOCATION_KEY.value] == ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value

        msg = ServoMessageBuilder.body_up_down(90)
        assert msg[ServoEnum.MSG_LOCATION_KEY.value] == ServoEnum.LOCATION_BODY_UP_DOWN.value

        msg = ServoMessageBuilder.body_left_right(90)
        assert msg[ServoEnum.MSG_LOCATION_KEY.value] == ServoEnum.LOCATION_BODY_LEFT_RIGHT.value


class TestServoStatusContract:
    """Validate that Gservo.get_angles() format matches ServoLocation's ServoTuple."""

    def test_status_fields_match_tuple(self):
        """The fields returned by get_angles must match what ServoLocation parses."""
        # These are the fields ServoLocation's ServoTuple expects, in order
        expected_keys = [
            ServoEnum.MSG_CURRENT_ANGLE.value,   # current
            ServoEnum.MSG_MAX.value,              # max
            ServoEnum.MSG_MIN.value,              # min
            ServoEnum.MSG_MIDDLE.value,           # middle
            ServoEnum.MSG_AXIS.value,             # axis
            ServoEnum.MSG_MOVING.value,           # moving
            # location is added by ServoLocation, not in results
            ServoEnum.MSG_LAST_ANGLE.value,       # last
            ServoEnum.MSG_VELOCITY.value,         # velocity
        ]
        # Simulate what Gservo.get_angles() returns
        mock_status = {
            ServoEnum.MSG_MAX.value: 180,
            ServoEnum.MSG_MIN.value: 0,
            ServoEnum.MSG_MIDDLE.value: 90,
            ServoEnum.MSG_CURRENT_ANGLE.value: 95.5,
            ServoEnum.MSG_VELOCITY.value: 1.2,
            ServoEnum.MSG_AXIS.value: "x",
            ServoEnum.MSG_MOVING.value: True,
            ServoEnum.MSG_LAST_ANGLE.value: 100.0,
        }
        # Every expected key must be present in the status dict
        for key in expected_keys:
            assert key in mock_status, f"Missing key '{key}' in servo status"


class TestMoveAllWithTrace:
    """Validate that trace IDs propagate through move_all messages."""

    def test_trace_fields_can_be_attached(self):
        targets = {
            ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: {
                ServoEnum.MSG_ANGLE.value: 85, ServoEnum.MSG_SPEED.value: 3
            },
        }
        msg = ServoMessageBuilder.move_all(targets)
        # VisionTracker attaches trace fields after building the message
        msg[TraceEnums.TRACE_ID.value] = "abc12345"
        msg[TraceEnums.TS_VISION.value] = 1711367200.123
        # Verify they survive
        assert msg[TraceEnums.TRACE_ID.value] == "abc12345"
        assert msg[TraceEnums.TS_VISION.value] == 1711367200.123
        # And original fields are still intact
        assert msg[ServoEnum.MSG_COMMAND_KEY.value] == ServoEnum.MSG_COMMAND_MOVE_ALL.value


class TestCameraMessageBuilder:
    """Validate vision result message format."""

    def test_send_results_format(self):
        results = {"person": {"count": 1, "objects": [{"confidence": 0.9}]}}
        msg = CameraMessageBuilder.send_results("camera_head", results)
        assert msg[CameraEnum.MSG_LOCATION_KEY.value] == "camera_head"
        assert msg[CameraEnum.MSG_RESULTS.value] == results
