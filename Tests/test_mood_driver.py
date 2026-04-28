"""Tests for MoodDriver — translates PAD into hardware commands with hysteresis.

Bypass __init__ to skip MQTT setup; mock send_command + the consumer.
Drive _tick() directly; verify the published topic + payload + hysteresis.
"""

from configparser import ConfigParser
from unittest.mock import MagicMock, patch

import pytest

from glados_modules.GladosEnums import (LCDEnums, LEDHead, MQTTEnums, SystemEnums)
from glados_modules.MqttConnector import MQTTClient


@pytest.fixture
def mood_config():
    cp = ConfigParser()
    cp.read_string("""
[MQTT]
mqtt_server_ip = 127.0.0.1
mqtt_port = 1883
""")
    return cp


def _make_pad(p, a, d):
    from glados_modules.MoodConsumer import PADState
    return PADState(pleasure=p, arousal=a, dominance=d)


@pytest.fixture
def fake_driver(mood_config):
    from glados_modules.MoodDriver import MoodDriver
    with patch.object(MQTTClient, '__init__', lambda self, *a, **kw: None):
        d = MoodDriver.__new__(MoodDriver)
    d.__name__ = "MoodDriver"
    d.logger = MagicMock()
    d.configFile = mood_config
    d.consumer = MagicMock()
    d.consumer.is_stale.return_value = False
    d.stop = False
    d._current_eye_animation = None
    d._current_breath_fast = None
    d.send_command = MagicMock()
    return d


def _last_call(driver, predicate):
    """Return (msg, topic) of the most recent send_command call matching predicate."""
    for call in reversed(driver.send_command.call_args_list):
        msg, topic = call.args
        if predicate(msg, topic):
            return msg, topic
    raise AssertionError("No matching send_command call")


class TestEyeAnimationMapping:
    """Pure mapping function: PAD → animation key, with hysteresis."""

    def test_neutral_pad_returns_normal(self, fake_driver):
        target = fake_driver._target_eye_animation(_make_pad(0.0, 0.0, 0.0))
        assert target == LEDHead.ANIMATION_NORMAL_EYE_KEY.value

    def test_high_arousal_low_pleasure_triggers_angry(self, fake_driver):
        target = fake_driver._target_eye_animation(_make_pad(-0.5, 0.8, 0.0))
        assert target == LEDHead.ANIMATION_ANGRY_EYE_KEY.value

    def test_high_arousal_high_pleasure_stays_normal(self, fake_driver):
        # Excited but happy — not angry
        target = fake_driver._target_eye_animation(_make_pad(0.6, 0.8, 0.0))
        assert target == LEDHead.ANIMATION_NORMAL_EYE_KEY.value

    def test_high_arousal_neutral_pleasure_stays_normal(self, fake_driver):
        target = fake_driver._target_eye_animation(_make_pad(0.0, 0.8, 0.0))
        assert target == LEDHead.ANIMATION_NORMAL_EYE_KEY.value

    def test_hysteresis_holds_angry_at_boundary(self, fake_driver):
        # Already angry, PAD drops slightly below trigger but above release
        fake_driver._current_eye_animation = LEDHead.ANIMATION_ANGRY_EYE_KEY.value
        target = fake_driver._target_eye_animation(_make_pad(-0.2, 0.6, 0.0))
        assert target == LEDHead.ANIMATION_ANGRY_EYE_KEY.value

    def test_hysteresis_releases_angry_when_clearly_calm(self, fake_driver):
        fake_driver._current_eye_animation = LEDHead.ANIMATION_ANGRY_EYE_KEY.value
        # Both axes well into release band
        target = fake_driver._target_eye_animation(_make_pad(0.2, 0.2, 0.0))
        assert target == LEDHead.ANIMATION_NORMAL_EYE_KEY.value


class TestBreathFastMapping:
    """Pure mapping function: PAD arousal → breath fast bool, with hysteresis."""

    def test_low_arousal_returns_slow(self, fake_driver):
        assert fake_driver._target_breath_fast(_make_pad(0, 0.0, 0)) is False

    def test_high_arousal_triggers_fast(self, fake_driver):
        assert fake_driver._target_breath_fast(_make_pad(0, 0.7, 0)) is True

    def test_hysteresis_holds_fast_in_release_band(self, fake_driver):
        fake_driver._current_breath_fast = True
        # 0.4 is below the 0.5 trigger but above the 0.3 release
        assert fake_driver._target_breath_fast(_make_pad(0, 0.4, 0)) is True

    def test_hysteresis_releases_fast_below_release_threshold(self, fake_driver):
        fake_driver._current_breath_fast = True
        assert fake_driver._target_breath_fast(_make_pad(0, 0.2, 0)) is False


class TestPublishOnChange:
    """_tick publishes on transitions; no-op when target unchanged."""

    def test_first_tick_publishes_both(self, fake_driver):
        fake_driver.consumer.get_pad.return_value = _make_pad(0.0, 0.0, 0.0)
        fake_driver._tick()
        # First tick: animation None → "normal_eye", fast None → False
        # Both transitions fire publishes
        assert fake_driver._current_eye_animation == LEDHead.ANIMATION_NORMAL_EYE_KEY.value
        assert fake_driver._current_breath_fast is False
        # 1 LED publish + 2 LCD publishes (left + right) = 3 sends
        assert fake_driver.send_command.call_count == 3

    def test_eye_animation_publishes_to_correct_topic(self, fake_driver):
        fake_driver.consumer.get_pad.return_value = _make_pad(-0.5, 0.8, 0.0)
        fake_driver._tick()
        msg, topic = _last_call(
            fake_driver,
            lambda m, t: t == MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value)
        body = msg[LEDHead.MSG_COMMAND_KEY.value]
        assert body[LEDHead.MSG_COMMAND_LOCATION_KEY.value] == \
               LEDHead.EYE_LED_LOCATION.value
        assert body[LEDHead.MSG_COMMAND_KEY.value] == \
               LEDHead.ANIMATION_ANGRY_EYE_KEY.value

    def test_breath_publishes_both_eyes_with_full_options(self, fake_driver):
        fake_driver.consumer.get_pad.return_value = _make_pad(0.0, 0.7, 0.0)
        fake_driver._tick()
        lcd_calls = [c.args for c in fake_driver.send_command.call_args_list
                     if c.args[1] == MQTTEnums.LCD_CONTROL_MQTT_TOPIC.value]
        assert len(lcd_calls) == 2
        locations = [m[LCDEnums.MSG_LOCATION_KEY.value] for m, _ in lcd_calls]
        assert SystemEnums.LEFT_LCD.value in locations
        assert SystemEnums.RIGHT_LCD.value in locations
        # GladosLCD.set_breath_options requires fast + animation + rainbow
        for msg, _ in lcd_calls:
            opts = msg[LCDEnums.OPTIONS_KEY.value]
            assert opts["fast"] is True
            assert "animation" in opts
            assert "rainbow" in opts

    def test_no_change_no_publish(self, fake_driver):
        fake_driver._current_eye_animation = LEDHead.ANIMATION_NORMAL_EYE_KEY.value
        fake_driver._current_breath_fast = False
        fake_driver.consumer.get_pad.return_value = _make_pad(0.0, 0.0, 0.0)
        fake_driver._tick()
        fake_driver.send_command.assert_not_called()

    def test_only_changing_axis_publishes(self, fake_driver):
        # Eye stays normal, only breath transitions to fast
        fake_driver._current_eye_animation = LEDHead.ANIMATION_NORMAL_EYE_KEY.value
        fake_driver._current_breath_fast = False
        fake_driver.consumer.get_pad.return_value = _make_pad(0.0, 0.7, 0.0)
        fake_driver._tick()
        # 2 LCD publishes only — no LED publish because eye target unchanged
        assert fake_driver.send_command.call_count == 2
        for call in fake_driver.send_command.call_args_list:
            assert call.args[1] == MQTTEnums.LCD_CONTROL_MQTT_TOPIC.value

    def test_stale_pad_holds_state(self, fake_driver):
        fake_driver.consumer.is_stale.return_value = True
        fake_driver._tick()
        fake_driver.send_command.assert_not_called()


class TestHysteresisInTickLoop:
    """End-to-end transitions across multiple ticks behave as expected."""

    def test_arousal_oscillation_does_not_flap(self, fake_driver):
        # Start neutral
        fake_driver._current_eye_animation = LEDHead.ANIMATION_NORMAL_EYE_KEY.value
        fake_driver._current_breath_fast = False

        # Spike into angry territory
        fake_driver.consumer.get_pad.return_value = _make_pad(-0.5, 0.8, 0)
        fake_driver._tick()
        assert fake_driver._current_eye_animation == \
               LEDHead.ANIMATION_ANGRY_EYE_KEY.value

        # Drop slightly below trigger but above release — should stay angry
        fake_driver.send_command.reset_mock()
        fake_driver.consumer.get_pad.return_value = _make_pad(-0.2, 0.6, 0)
        fake_driver._tick()
        assert fake_driver._current_eye_animation == \
               LEDHead.ANIMATION_ANGRY_EYE_KEY.value
        # No LED publish on this tick — target didn't change
        led_publishes = [c for c in fake_driver.send_command.call_args_list
                         if c.args[1] == MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value]
        assert len(led_publishes) == 0


class TestPublishErrorHandling:
    """Broker exceptions are logged and swallowed."""

    def test_eye_publish_failure_swallowed(self, fake_driver):
        fake_driver.send_command.side_effect = RuntimeError("broker down")
        fake_driver.consumer.get_pad.return_value = _make_pad(-0.5, 0.8, 0)
        fake_driver._tick()  # must not raise
        assert fake_driver.logger.error.called

    def test_tick_exception_swallowed_in_run_loop(self, fake_driver):
        # Simulate an exception during _tick inside run() and confirm the
        # outer try/except keeps the thread alive.
        fake_driver.consumer.get_pad.side_effect = RuntimeError("boom")
        # Make stop happen after one tick so run() returns
        fake_driver.stop = False

        def trip_stop():
            fake_driver.stop = True
            raise RuntimeError("boom")

        fake_driver.consumer.get_pad.side_effect = trip_stop
        fake_driver.consumer.is_stale.return_value = False
        # Patch sleep so the test doesn't actually wait
        with patch("glados_modules.MoodDriver.sleep", lambda _: None):
            fake_driver.run()  # must not raise
        assert fake_driver.logger.error.called
