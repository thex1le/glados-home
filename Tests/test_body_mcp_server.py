"""Tests for the body MCP server.

Each tool publishes to a known MQTT topic with a known payload shape that
BodyServer.py / BodyControlModules already understand. We mock the global
_mqtt client and assert (topic, payload) on each call.

The conftest mocks mcp.server.fastmcp so FastMCP() and @mcp.tool() are
no-ops at import time, which lets us import and call the tool functions
directly without starting a stdio server.
"""

import pytest
from configparser import ConfigParser
from unittest.mock import MagicMock

from glados_modules.GladosEnums import (
    LCDEnums, LEDHead, MQTTEnums, ServoEnum, SystemEnums,
)


@pytest.fixture(autouse=True)
def reset_mcp_state():
    """Each test gets a fresh module-level _mqtt and _config."""
    import glados_modules.mcp.BodyMCPServer as srv
    saved_mqtt, saved_config = srv._mqtt, srv._config
    srv._mqtt = MagicMock()
    srv._config = MagicMock()
    yield srv
    srv._mqtt, srv._config = saved_mqtt, saved_config


def _last_call(srv):
    """Helper: return (msg, topic) of the most recent send_command call."""
    args, _ = srv._mqtt.send_command.call_args
    return args


class TestLookAt:
    """look_at publishes a single move_all command for both head servos."""

    def test_publishes_to_servo_topic(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.look_at(45, 90)
        srv._mqtt.send_command.assert_called_once()
        _, topic = _last_call(srv)
        assert topic == ServoEnum.MQTT_COMMAND_TOPIC.value

    def test_uses_move_all_command(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.look_at(45, 90)
        msg, _ = _last_call(srv)
        assert msg[ServoEnum.MSG_COMMAND_KEY.value] == \
               ServoEnum.MSG_COMMAND_MOVE_ALL.value

    def test_targets_include_both_head_axes(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.look_at(45, 90)
        msg, _ = _last_call(srv)
        targets = msg[ServoEnum.MSG_TARGETS.value]
        assert ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value in targets
        assert ServoEnum.LOCATION_HEAD_UP_DOWN.value in targets
        assert targets[ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value][
            ServoEnum.MSG_ANGLE.value] == 45
        assert targets[ServoEnum.LOCATION_HEAD_UP_DOWN.value][
            ServoEnum.MSG_ANGLE.value] == 90

    def test_returns_descriptive_string(self, reset_mcp_state):
        srv = reset_mcp_state
        result = srv.look_at(45, 90)
        assert "yaw=45" in result and "pitch=90" in result


class TestTurnBody:
    """turn_body publishes a body_left_right move command."""

    def test_publishes_correct_servo_location(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.turn_body(120)
        msg, topic = _last_call(srv)
        assert topic == ServoEnum.MQTT_COMMAND_TOPIC.value
        assert msg[ServoEnum.MSG_LOCATION_KEY.value] == \
               ServoEnum.LOCATION_BODY_LEFT_RIGHT.value
        assert msg[ServoEnum.MSG_ANGLE.value] == 120


class TestLean:
    """lean publishes a body_up_down move command."""

    def test_publishes_correct_servo_location(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.lean(75)
        msg, topic = _last_call(srv)
        assert topic == ServoEnum.MQTT_COMMAND_TOPIC.value
        assert msg[ServoEnum.MSG_LOCATION_KEY.value] == \
               ServoEnum.LOCATION_BODY_UP_DOWN.value
        assert msg[ServoEnum.MSG_ANGLE.value] == 75


class TestSetEyeAnimation:
    """set_eye_animation publishes a head LED command in LedHead's expected shape."""

    def test_normal_animation_publishes_to_led_topic(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.set_eye_animation("normal")
        msg, topic = _last_call(srv)
        assert topic == MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value

    def test_payload_shape_matches_ledhead_handler(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.set_eye_animation("angry")
        msg, _ = _last_call(srv)
        # LedHead.handle_cmd reads body = j_msg[MSG_COMMAND_KEY], then
        # body[MSG_COMMAND_LOCATION_KEY] == EYE_LED_LOCATION and body[MSG_COMMAND_KEY]
        body = msg[LEDHead.MSG_COMMAND_KEY.value]
        assert body[LEDHead.MSG_COMMAND_LOCATION_KEY.value] == \
               LEDHead.EYE_LED_LOCATION.value
        assert body[LEDHead.MSG_COMMAND_KEY.value] == \
               LEDHead.ANIMATION_ANGRY_EYE_KEY.value

    def test_all_named_animations_publish(self, reset_mcp_state):
        srv = reset_mcp_state
        for name in ("normal", "angry", "disco", "startup"):
            srv._mqtt.send_command.reset_mock()
            result = srv.set_eye_animation(name)
            srv._mqtt.send_command.assert_called_once()
            assert "error" not in result.lower()

    def test_animation_name_is_case_insensitive(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.set_eye_animation("NORMAL")
        srv._mqtt.send_command.assert_called_once()
        msg, _ = _last_call(srv)
        body = msg[LEDHead.MSG_COMMAND_KEY.value]
        assert body[LEDHead.MSG_COMMAND_KEY.value] == \
               LEDHead.ANIMATION_NORMAL_EYE_KEY.value

    def test_invalid_animation_returns_error_and_no_publish(self, reset_mcp_state):
        srv = reset_mcp_state
        result = srv.set_eye_animation("rainbow_unicorn")
        assert "error" in result.lower()
        srv._mqtt.send_command.assert_not_called()


class TestWakeEyes:
    """wake_eyes publishes the LCD startup command for one or both eyes."""

    def test_default_publishes_both_eyes(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.wake_eyes()
        assert srv._mqtt.send_command.call_count == 2
        locations = [call.args[0][LCDEnums.MSG_LOCATION_KEY.value]
                     for call in srv._mqtt.send_command.call_args_list]
        assert SystemEnums.LEFT_LCD.value in locations
        assert SystemEnums.RIGHT_LCD.value in locations

    def test_left_eye_publishes_once(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.wake_eyes(eye="left")
        assert srv._mqtt.send_command.call_count == 1
        msg, _ = _last_call(srv)
        assert msg[LCDEnums.MSG_LOCATION_KEY.value] == SystemEnums.LEFT_LCD.value

    def test_right_eye_publishes_once(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.wake_eyes(eye="right")
        assert srv._mqtt.send_command.call_count == 1
        msg, _ = _last_call(srv)
        assert msg[LCDEnums.MSG_LOCATION_KEY.value] == SystemEnums.RIGHT_LCD.value

    def test_publishes_to_lcd_topic(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.wake_eyes(eye="left")
        _, topic = _last_call(srv)
        assert topic == MQTTEnums.LCD_CONTROL_MQTT_TOPIC.value

    def test_uses_startup_command(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.wake_eyes(eye="both")
        for call in srv._mqtt.send_command.call_args_list:
            msg = call.args[0]
            assert msg[LCDEnums.MSG_COMMAND_KEY.value] == \
                   LCDEnums.COMMAND_STARTUP.value

    def test_invalid_eye_returns_error_and_no_publish(self, reset_mcp_state):
        srv = reset_mcp_state
        result = srv.wake_eyes(eye="middle")
        assert "error" in result.lower()
        srv._mqtt.send_command.assert_not_called()


class TestSetEyeBreath:
    """set_eye_breath publishes a set_breath command per eye with the fast option."""

    def test_default_both_eyes_normal_speed(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.set_eye_breath()
        assert srv._mqtt.send_command.call_count == 2
        for call in srv._mqtt.send_command.call_args_list:
            msg = call.args[0]
            assert msg[LCDEnums.MSG_COMMAND_KEY.value] == \
                   LCDEnums.COMMAND_SET_BREATH.value
            assert msg[LCDEnums.OPTIONS_KEY.value]["fast"] is False

    def test_fast_option_propagates(self, reset_mcp_state):
        srv = reset_mcp_state
        srv.set_eye_breath(eye="left", fast=True)
        msg, _ = _last_call(srv)
        assert msg[LCDEnums.OPTIONS_KEY.value]["fast"] is True

    def test_invalid_eye_returns_error(self, reset_mcp_state):
        srv = reset_mcp_state
        result = srv.set_eye_breath(eye="middle")
        assert "error" in result.lower()
        srv._mqtt.send_command.assert_not_called()


class TestEnsureInit:
    """_ensure_init loads glog.conf and connects to MQTT exactly once."""

    def test_raises_when_config_missing(self, tmp_path, reset_mcp_state):
        srv = reset_mcp_state
        srv._mqtt = None
        with pytest.raises(srv.BodyMCPException):
            srv._ensure_init(str(tmp_path / "missing.conf"))

    def test_idempotent_when_already_initialized(self, reset_mcp_state, tmp_path):
        srv = reset_mcp_state
        # _mqtt is already set by the fixture; calling _ensure_init must be a no-op
        existing = srv._mqtt
        srv._ensure_init(str(tmp_path / "ignored.conf"))
        assert srv._mqtt is existing


class TestPublishGuard:
    """_publish refuses to send when the MQTT client hasn't been initialized."""

    def test_raises_when_mqtt_is_none(self, reset_mcp_state):
        srv = reset_mcp_state
        srv._mqtt = None
        with pytest.raises(srv.BodyMCPException):
            srv._publish({"x": 1}, "topic/x")
