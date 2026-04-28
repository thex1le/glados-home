"""Tests for GladosLocal mood-event publishing (Step 7c-4).

Verifies that the input detectors (register_interaction, detect_compliment,
room arrivals/departures) publish on system/mood/event without breaking the
existing mood.escalate / mood.calm legacy behavior.

GladosLocal has heavy hardware dependencies (alsaaudio Mixer, pydub, etc.),
so we use the same __new__()-bypass + manual attribute pattern as
test_glados_brain to construct an instance without running __init__.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from glados_modules.GladosEnums import MQTTEnums, PersonalityEnums
from glados_modules.MqttConnector import MQTTClient


@pytest.fixture
def fake_local():
    """GladosLocal instance with __init__ bypassed; only the mood + send_command paths set up."""
    from glados_modules.GLaDOSLocal import GladosLocal
    with patch.object(MQTTClient, '__init__', lambda self, *a, **kw: None):
        gl = GladosLocal.__new__(GladosLocal)
    gl.__name__ = "GladosLocal"
    gl.logger = MagicMock()
    gl.mood = MagicMock()
    gl.mood.register_question.return_value = False
    gl.send_command = MagicMock()
    gl._commentary_enabled = False  # suppress optional speak() in departure path
    return gl


def _last_call(gl, predicate):
    for call in reversed(gl.send_command.call_args_list):
        msg, topic = call.args
        if predicate(msg, topic):
            return msg, topic
    raise AssertionError("No matching send_command call")


class TestPublishMoodEventHelper:
    """The _publish_mood_event helper is the chokepoint for all detector events."""

    def test_publishes_to_mood_event_topic(self, fake_local):
        fake_local._publish_mood_event("vision", "test")
        msg, topic = fake_local.send_command.call_args[0]
        assert topic == MQTTEnums.MOOD_EVENT_TOPIC.value
        assert msg["source"] == "vision"
        assert msg["description"] == "test"

    def test_broker_failure_is_swallowed_and_logged(self, fake_local):
        fake_local.send_command.side_effect = RuntimeError("broker down")
        # Must not raise — detectors fire on every interaction; can't crash
        fake_local._publish_mood_event("user", "x")
        fake_local.logger.error.assert_called_once()


class TestRegisterInteraction:
    """register_interaction publishes pestering vs normal-question events."""

    def test_normal_question_publishes_normal_event(self, fake_local):
        fake_local.mood.register_question.return_value = False
        fake_local.register_interaction()
        msg, _ = _last_call(
            fake_local,
            lambda m, t: t == MQTTEnums.MOOD_EVENT_TOPIC.value)
        assert msg["source"] == "user"
        assert "asked a question" in msg["description"]

    def test_pestering_publishes_pestering_event(self, fake_local):
        fake_local.mood.register_question.return_value = True
        fake_local.register_interaction()
        msg, _ = _last_call(
            fake_local,
            lambda m, t: t == MQTTEnums.MOOD_EVENT_TOPIC.value)
        assert msg["source"] == "user"
        assert "pestering" in msg["description"].lower()

    def test_pestering_also_calls_legacy_mood_escalate(self, fake_local):
        # Legacy behavior preserved — GladosMood still tracked alongside
        fake_local.mood.register_question.return_value = True
        fake_local.register_interaction()
        fake_local.mood.escalate.assert_called_once()
        assert fake_local.mood.escalate.call_args[0][1] == "pestering"


class TestDetectCompliment:
    """detect_compliment publishes a compliment event when matched."""

    def test_match_publishes_event(self, fake_local):
        result = fake_local.detect_compliment("you're amazing")
        assert result is True
        msg, _ = _last_call(
            fake_local,
            lambda m, t: t == MQTTEnums.MOOD_EVENT_TOPIC.value)
        assert msg["source"] == "user"
        assert "compliment" in msg["description"].lower()

    def test_match_also_calls_legacy_mood_calm(self, fake_local):
        fake_local.detect_compliment("thank you")
        fake_local.mood.calm.assert_called_once()

    def test_no_match_does_not_publish(self, fake_local):
        result = fake_local.detect_compliment("what's the weather")
        assert result is False
        fake_local.send_command.assert_not_called()


class TestRoomArrivalDeparture:
    """_on_person_arrived / _on_person_departed publish vision events."""

    def _setup_room_state(self, fake_local):
        fake_local._greeting_timestamps = {}
        fake_local.speak = MagicMock()

    def test_known_person_arrival_publishes_named_event(self, fake_local):
        self._setup_room_state(fake_local)
        fake_local._on_person_arrived("Ben", now=1000.0)
        msg, _ = _last_call(
            fake_local,
            lambda m, t: t == MQTTEnums.MOOD_EVENT_TOPIC.value)
        assert msg["source"] == "vision"
        assert "Ben entered" in msg["description"]

    def test_unknown_person_arrival_publishes_unknown_event(self, fake_local):
        self._setup_room_state(fake_local)
        fake_local._on_person_arrived("unknown_3", now=1000.0)
        msg, _ = _last_call(
            fake_local,
            lambda m, t: t == MQTTEnums.MOOD_EVENT_TOPIC.value)
        assert msg["source"] == "vision"
        assert "unknown" in msg["description"].lower()
        assert "unknown_3" in msg["description"]

    def test_unknown_arrival_also_escalates_legacy_mood(self, fake_local):
        self._setup_room_state(fake_local)
        fake_local._on_person_arrived("unknown_3", now=1000.0)
        fake_local.mood.escalate.assert_called_once()

    def test_known_arrival_does_not_escalate_mood(self, fake_local):
        self._setup_room_state(fake_local)
        fake_local._on_person_arrived("Ben", now=1000.0)
        fake_local.mood.escalate.assert_not_called()

    def test_arrival_within_cooldown_skips_publish(self, fake_local):
        self._setup_room_state(fake_local)
        fake_local._greeting_timestamps["Ben"] = 1000.0  # just greeted
        # Same person 1s later — within cooldown
        fake_local._on_person_arrived("Ben", now=1001.0)
        # No event fired this time (cooldown skip happens before publish)
        fake_local.send_command.assert_not_called()

    def test_departure_publishes_event(self, fake_local):
        fake_local._on_person_departed("Ben", now=1100.0)
        msg, _ = _last_call(
            fake_local,
            lambda m, t: t == MQTTEnums.MOOD_EVENT_TOPIC.value)
        assert msg["source"] == "vision"
        assert "Ben left" in msg["description"]

    def test_departure_also_calms_legacy_mood(self, fake_local):
        fake_local._on_person_departed("Ben", now=1100.0)
        fake_local.mood.calm.assert_called_once()
