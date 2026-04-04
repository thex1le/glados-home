"""Tests for AttentionModel — priority-based target selection.

Validates priority ordering, budget duration system, fatigue glances,
and external input handling (speaker, conversation partner).
"""

import time
import pytest

from glados_modules.AttentionModel import AttentionModel, AttentionPriority
from glados_modules.RoomStateManager import RoomPerson


def _make_person(person_id: str, world_lr: float = 0.0, confidence: float = 0.9,
                 first_seen: float = None, emotion: str = "neutral",
                 face_id: str = None) -> RoomPerson:
    """Build a RoomPerson for testing."""
    now = time.time()
    return RoomPerson(
        person_id=person_id,
        world_lr=world_lr,
        confidence=confidence,
        cameras={"camera_head"},
        first_seen=first_seen or (now - 60),  # default: been here 60s
        last_seen=now,
        emotion=emotion,
        bbox_height=300.0,
        face_id=face_id or person_id,
    )


class TestPriorityOrdering:
    """Test that higher priorities win over lower ones."""

    def test_new_arrival_beats_proximity(self):
        """A person who just arrived should win over an existing person."""
        model = AttentionModel()
        now = time.time()
        roster = {
            "old_person": _make_person("old_person", world_lr=10.0, first_seen=now - 60),
            "new_person": _make_person("new_person", world_lr=-20.0, first_seen=now - 1),
        }
        pid, reason = model.select_target(roster, 0.04)
        assert pid == "new_person"
        assert reason == "new_arrival"

    def test_conversation_beats_proximity(self):
        """Conversation partner should win over highest confidence."""
        model = AttentionModel()
        model.set_conversation_partner("partner")
        roster = {
            "stranger": _make_person("stranger", confidence=0.99),
            "partner": _make_person("partner", confidence=0.7),
        }
        pid, reason = model.select_target(roster, 0.04)
        assert pid == "partner"
        assert reason == "conversation"

    def test_new_arrival_beats_conversation(self):
        """New arrival should interrupt conversation partner."""
        model = AttentionModel()
        model.set_conversation_partner("partner")
        now = time.time()
        roster = {
            "partner": _make_person("partner", first_seen=now - 60),
            "newcomer": _make_person("newcomer", first_seen=now - 0.5),
        }
        pid, reason = model.select_target(roster, 0.04)
        assert pid == "newcomer"
        assert reason == "new_arrival"

    def test_single_person_always_selected(self):
        """With only one person, they're always the target."""
        model = AttentionModel()
        roster = {"solo": _make_person("solo")}
        pid, reason = model.select_target(roster, 0.04)
        assert pid == "solo"

    def test_empty_roster_returns_none(self):
        """Empty roster should return None."""
        model = AttentionModel()
        pid, reason = model.select_target({}, 0.04)
        assert pid is None
        assert reason == ""


class TestBudgetSystem:
    """Test duration budget holding and expiration."""

    def test_budget_holds_target(self):
        """After selecting a target, it should hold for the budget duration."""
        model = AttentionModel()
        now = time.time()
        roster = {
            "alice": _make_person("alice", confidence=0.95, first_seen=now - 0.5),
            "bob": _make_person("bob", confidence=0.99, first_seen=now - 60),
        }
        # First call: alice wins as new arrival
        pid1, reason1 = model.select_target(roster, 0.04)
        assert pid1 == "alice"
        assert reason1 == "new_arrival"

        # Alice is no longer new arrival (simulate time passing in first_seen)
        roster["alice"].first_seen = now - 10

        # Second call: budget not expired yet, should still be alice
        pid2, _ = model.select_target(roster, 0.5)
        assert pid2 == "alice"

    def test_budget_expires_allows_switch(self):
        """After budget expires, model should re-evaluate and possibly switch."""
        model = AttentionModel()
        now = time.time()
        roster = {
            "alice": _make_person("alice", confidence=0.7, first_seen=now - 60),
            "bob": _make_person("bob", confidence=0.99, first_seen=now - 60),
        }

        # First call: picks bob (highest confidence, proximity fallback)
        pid1, _ = model.select_target(roster, 0.04)

        # Exhaust budget by passing large dt (proximity budget = 0, immediate re-eval)
        pid2, _ = model.select_target(roster, 10.0)
        # Should still be bob since he has highest confidence
        assert pid2 == "bob"

    def test_higher_priority_interrupts_budget(self):
        """A higher priority should interrupt even with remaining budget."""
        model = AttentionModel()
        now = time.time()
        roster = {
            "alice": _make_person("alice", confidence=0.99, first_seen=now - 60),
        }
        # Start tracking alice
        model.select_target(roster, 0.04)

        # Bob arrives (new_arrival priority)
        roster["bob"] = _make_person("bob", first_seen=now - 0.5)
        pid, reason = model.select_target(roster, 0.04)
        assert pid == "bob"
        assert reason == "new_arrival"


class TestFatigueGlance:
    """Test attention fatigue — glancing at neglected people."""

    def test_fatigue_triggers_after_threshold(self):
        """After 30s of not looking at someone, fatigue glance should trigger."""
        model = AttentionModel()
        model._fatigue_threshold = 0.1  # 100ms for fast test
        now = time.time()

        roster = {
            "tracked": _make_person("tracked", confidence=0.99, first_seen=now - 60),
            "neglected": _make_person("neglected", confidence=0.5, first_seen=now - 60),
        }

        # Look at tracked first
        pid1, _ = model.select_target(roster, 0.04)
        assert pid1 == "tracked"

        # Simulate time passing — neglected hasn't been looked at
        model._last_looked_at["neglected"] = now - 1.0  # 1 second ago
        # Expire the current budget
        model._budget_remaining = 0

        pid2, reason2 = model.select_target(roster, 0.04)
        assert pid2 == "neglected"
        assert reason2 == "fatigue_glance"

    def test_no_fatigue_with_single_person(self):
        """Fatigue glance should not trigger with only one person."""
        model = AttentionModel()
        model._fatigue_threshold = 0.0  # instant fatigue
        roster = {"solo": _make_person("solo")}

        pid, reason = model.select_target(roster, 0.04)
        assert pid == "solo"
        assert reason != "fatigue_glance"


class TestExternalInputs:
    """Test speaker direction and conversation partner."""

    def test_speaker_direction_triggers_attention(self):
        """Setting speaker direction should target the closest person."""
        model = AttentionModel()
        now = time.time()
        roster = {
            "left": _make_person("left", world_lr=30.0, first_seen=now - 60),
            "right": _make_person("right", world_lr=-30.0, first_seen=now - 60),
        }

        model.set_speaker_direction(28.0)  # close to "left"
        pid, reason = model.select_target(roster, 0.04)
        assert pid == "left"
        assert reason == "speaker"

    def test_conversation_partner_persists(self):
        """Conversation partner should be tracked across calls."""
        model = AttentionModel()
        model.set_conversation_partner("bob")
        roster = {
            "alice": _make_person("alice", confidence=0.99),
            "bob": _make_person("bob", confidence=0.5),
        }

        pid, _ = model.select_target(roster, 0.04)
        assert pid == "bob"

        # Clear conversation
        model.clear_conversation_partner()
        model._budget_remaining = 0  # expire budget
        pid2, _ = model.select_target(roster, 0.04)
        # Should fall back to alice (highest confidence)
        assert pid2 == "alice"


class TestAttentionState:
    """Test the serializable state output."""

    def test_get_state_has_fields(self):
        """get_state should return attention_target, reason, budget."""
        model = AttentionModel()
        roster = {"alice": _make_person("alice")}
        model.select_target(roster, 0.04)

        state = model.get_state()
        assert "attention_target" in state
        assert "attention_reason" in state
        assert "attention_budget" in state
        assert state["attention_target"] == "alice"

    def test_get_state_empty_when_no_roster(self):
        """get_state with no target should have None target."""
        model = AttentionModel()
        model.select_target({}, 0.04)

        state = model.get_state()
        assert state["attention_target"] is None

    def test_personality_modifier_stored(self):
        """Personality modifiers should be stored for later use."""
        model = AttentionModel()
        model.add_personality_modifier("enemy", 15.0)
        assert model._personality_modifiers["enemy"] == 15.0
