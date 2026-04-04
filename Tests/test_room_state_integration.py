"""Tests for GLaDOSLocal room state integration.

Validates arrival greetings, departure mood calming, greeting cooldown,
party commentary, and process_sight room roster enrichment.
"""

import time
import json
import threading
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from glados_modules.GladosEnums import PersonalityEnums, VisionResultsEnum


def _make_gl():
    """Create a minimal GladosLocal with room state attributes for testing."""
    from glados_modules.GLaDOSLocal import GladosLocal
    with patch.object(GladosLocal, '__init__', lambda self, *a, **kw: None):
        gl = GladosLocal.__new__(GladosLocal)
    gl.logger = MagicMock()
    gl.mood = MagicMock()
    gl.speak = MagicMock()
    gl.send_command = MagicMock()
    gl.vision_confidence = 0.65
    gl._commentary_enabled = True
    gl._room_roster = []
    gl._room_count = 0
    gl._known_people = set()
    gl._greeting_timestamps = {}
    gl._last_party_comment = 0.0
    gl._room_lock = threading.Lock()
    return gl


def _make_mqtt_msg(data: dict):
    """Create a fake MQTT message with JSON payload."""
    msg = MagicMock()
    msg.payload = json.dumps(data).encode("utf-8")
    return msg


class TestArrivalEvents:
    """Test person arrival handling."""

    def test_arrival_triggers_greeting_for_known_person(self):
        """Named person arriving should trigger a spoken greeting."""
        gl = _make_gl()
        msg = _make_mqtt_msg({
            "count": 1,
            "roster": [{"person_id": "ben", "face_id": "ben", "emotion": "neutral",
                         "first_seen": time.time(), "world_lr": 0}],
            "arrivals": ["ben"],
            "departures": [],
        })
        gl._handle_room_state(msg)

        gl.speak.assert_called_once()
        call_text = gl.speak.call_args[0][0]
        assert "ben" in call_text.lower() or "Ben" in call_text

    def test_arrival_unknown_escalates_mood(self):
        """Unknown person arriving should escalate mood slightly."""
        gl = _make_gl()
        msg = _make_mqtt_msg({
            "count": 1,
            "roster": [{"person_id": "unknown_1", "face_id": "unknown",
                         "emotion": "neutral", "first_seen": time.time(), "world_lr": 0}],
            "arrivals": ["unknown_1"],
            "departures": [],
        })
        gl._handle_room_state(msg)

        gl.mood.escalate.assert_called_once()
        gl.speak.assert_not_called()  # no greeting for unknowns

    def test_greeting_cooldown_prevents_duplicate(self):
        """Same person arriving twice within cooldown should only get one greeting."""
        gl = _make_gl()
        now = time.time()

        # First arrival
        msg1 = _make_mqtt_msg({
            "count": 1, "roster": [], "arrivals": ["alice"], "departures": [],
        })
        gl._handle_room_state(msg1)
        assert gl.speak.call_count == 1

        # Second arrival within cooldown
        msg2 = _make_mqtt_msg({
            "count": 1, "roster": [], "arrivals": ["alice"], "departures": [],
        })
        gl._handle_room_state(msg2)
        assert gl.speak.call_count == 1  # still 1, not 2


class TestDepartureEvents:
    """Test person departure handling."""

    def test_departure_calms_mood(self):
        """Person leaving should calm mood."""
        gl = _make_gl()
        gl._known_people = {"bob"}
        msg = _make_mqtt_msg({
            "count": 0, "roster": [], "arrivals": [], "departures": ["bob"],
        })
        gl._handle_room_state(msg)

        gl.mood.calm.assert_called_once()
        args = gl.mood.calm.call_args[0]
        assert args[0] == PersonalityEnums.CALM_PERSON_LEFT.value

    def test_departure_removes_from_known(self):
        """Departed person should be removed from known_people set."""
        gl = _make_gl()
        gl._known_people = {"carol"}
        msg = _make_mqtt_msg({
            "count": 0, "roster": [], "arrivals": [], "departures": ["carol"],
        })
        gl._handle_room_state(msg)

        assert "carol" not in gl._known_people


class TestPartyCommentary:
    """Test room count commentary."""

    def test_party_comment_at_threshold(self):
        """3+ people should trigger party commentary."""
        gl = _make_gl()
        msg = _make_mqtt_msg({
            "count": 3,
            "roster": [
                {"person_id": "a", "face_id": "a", "emotion": "neutral",
                 "first_seen": time.time() - 60, "world_lr": 0},
                {"person_id": "b", "face_id": "b", "emotion": "neutral",
                 "first_seen": time.time() - 60, "world_lr": 10},
                {"person_id": "c", "face_id": "c", "emotion": "neutral",
                 "first_seen": time.time() - 60, "world_lr": -10},
            ],
            "arrivals": [],
            "departures": [],
        })
        gl._handle_room_state(msg)

        gl.speak.assert_called_once()

    def test_party_comment_not_repeated(self):
        """Party comment should not repeat within 60 seconds."""
        gl = _make_gl()
        gl._last_party_comment = time.time()  # just commented

        msg = _make_mqtt_msg({
            "count": 3, "roster": [{}, {}, {}], "arrivals": [], "departures": [],
        })
        gl._handle_room_state(msg)

        gl.speak.assert_not_called()

    def test_no_party_comment_below_threshold(self):
        """2 people should not trigger party commentary."""
        gl = _make_gl()
        msg = _make_mqtt_msg({
            "count": 2, "roster": [{}, {}], "arrivals": [], "departures": [],
        })
        gl._handle_room_state(msg)

        gl.speak.assert_not_called()


class TestProcessSightWithRoomRoster:
    """Test process_sight using room roster when available."""

    def test_room_roster_used_when_available(self):
        """When room roster has data, process_sight should use it."""
        gl = _make_gl()
        gl._room_roster = [
            {"person_id": "dave", "emotion": "happy",
             "first_seen": time.time() - 120, "world_lr": 5.0},
        ]
        seen = {"person": {"objects": [{"confidence": 0.9}]}}

        result = gl.process_sight(seen)
        assert "dave" in result
        assert "room" in result.lower()

    def test_fallback_to_raw_sight_when_no_roster(self):
        """When room roster is empty, should use raw sight_results."""
        gl = _make_gl()
        gl._room_roster = []
        seen = {
            "person": {
                "objects": [
                    {"confidence": 0.9,
                     "face": {"face_id": "eve", "emotion": "neutral"}}
                ]
            }
        }

        result = gl.process_sight(seen)
        assert "eve" in result

    def test_non_person_objects_still_included(self):
        """Non-person objects should still appear even with room roster."""
        gl = _make_gl()
        gl._room_roster = [
            {"person_id": "frank", "emotion": "neutral",
             "first_seen": time.time() - 30, "world_lr": 0},
        ]
        seen = {
            "person": {"objects": [{"confidence": 0.9}]},
            "chair": {"objects": [{"confidence": 0.95}]},
        }

        result = gl.process_sight(seen)
        assert "chair" in result


class TestRoomContext:
    """Test the get_room_context method for GPT prompts."""

    def test_room_context_with_people(self):
        """Room context should describe who is present."""
        gl = _make_gl()
        gl._room_roster = [
            {"person_id": "grace", "emotion": "happy",
             "first_seen": time.time() - 180, "world_lr": 0},
        ]

        context = gl.get_room_context()
        assert "grace" in context
        assert "minutes" in context or "seconds" in context

    def test_room_context_empty_room(self):
        """Empty room should return empty string."""
        gl = _make_gl()
        gl._room_roster = []

        context = gl.get_room_context()
        assert context == ""
