"""Tests for RoomStateManager — persistent room roster tracking.

Validates person matching (by face_id and proximity), arrival/departure
detection, multi-camera merging, and room summary format.
"""

import time
import pytest
from unittest.mock import patch

from glados_modules.RoomStateManager import RoomStateManager, RoomPerson


def _make_detection(confidence=0.9, x1=100, y1=50, x2=200, y2=350,
                    face_id="unknown", emotion="neutral"):
    """Build a detection dict matching MachineVision output format."""
    det = {
        "confidence": confidence,
        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }
    if face_id or emotion:
        det["face"] = {"face_id": face_id, "emotion": emotion}
    return det


def _world_lr_fn(bbox, camera):
    """Fake pixel-to-world: returns bbox center x scaled to degrees."""
    cx = (bbox["x1"] + bbox["x2"]) / 2
    return (cx - 320) / 10.0  # center=0, 100px offset = 10 degrees


class TestRoomRoster:
    """Test basic roster operations."""

    def test_new_person_creates_entry(self):
        """A single detection should create one roster entry."""
        rm = RoomStateManager()
        det = _make_detection(face_id="alice")
        rm.update_from_vision("camera_head", [det], _world_lr_fn)

        roster = rm.get_roster()
        assert "alice" in roster
        assert roster["alice"].face_id == "alice"
        assert roster["alice"].confidence == 0.9
        assert "camera_head" in roster["alice"].cameras

    def test_face_id_match_updates_existing(self):
        """Same face_id on subsequent frame should update, not duplicate."""
        rm = RoomStateManager()
        det1 = _make_detection(face_id="bob", x1=100, x2=200)
        det2 = _make_detection(face_id="bob", x1=120, x2=220, confidence=0.95)

        rm.update_from_vision("camera_head", [det1], _world_lr_fn)
        rm.update_from_vision("camera_head", [det2], _world_lr_fn)

        roster = rm.get_roster()
        assert len(roster) == 1
        assert roster["bob"].confidence == 0.95

    def test_proximity_match_updates_existing(self):
        """Unknown person at similar position should update existing entry."""
        rm = RoomStateManager()
        det1 = _make_detection(face_id="unknown", x1=100, x2=200)
        rm.update_from_vision("camera_head", [det1], _world_lr_fn)

        # Same approximate position, still unknown
        det2 = _make_detection(face_id="unknown", x1=105, x2=205, confidence=0.85)
        rm.update_from_vision("camera_head", [det2], _world_lr_fn)

        roster = rm.get_roster()
        assert len(roster) == 1  # matched by proximity, not duplicated

    def test_different_positions_create_separate_entries(self):
        """Two unknowns at very different positions should be separate."""
        rm = RoomStateManager()
        det_left = _make_detection(face_id="unknown", x1=10, x2=60)
        det_right = _make_detection(face_id="unknown", x1=500, x2=600)

        rm.update_from_vision("camera_head", [det_left, det_right], _world_lr_fn)

        roster = rm.get_roster()
        assert len(roster) == 2

    def test_unique_unknown_ids(self):
        """Each new unknown person gets a unique ID."""
        rm = RoomStateManager()
        det_left = _make_detection(face_id="unknown", x1=10, x2=60)
        det_right = _make_detection(face_id="unknown", x1=500, x2=600)

        rm.update_from_vision("camera_head", [det_left, det_right], _world_lr_fn)

        roster = rm.get_roster()
        ids = list(roster.keys())
        assert ids[0] != ids[1]
        assert all(pid.startswith("unknown_") for pid in ids)


class TestArrivalDeparture:
    """Test arrival and departure event detection."""

    def test_arrival_event_on_new_person(self):
        """New person should appear in arrivals from tick()."""
        rm = RoomStateManager()
        det = _make_detection(face_id="carol")
        rm.update_from_vision("camera_head", [det], _world_lr_fn)

        arrivals, departures = rm.tick()
        assert "carol" in arrivals
        assert len(departures) == 0

    def test_departure_after_timeout(self):
        """Person not seen for departure_timeout should appear in departures."""
        rm = RoomStateManager()
        rm._departure_timeout = 0.1  # 100ms for fast test

        det = _make_detection(face_id="dave")
        rm.update_from_vision("camera_head", [det], _world_lr_fn)
        rm.tick()  # consume arrival

        time.sleep(0.15)  # wait for staleness

        arrivals, departures = rm.tick()
        assert "dave" in departures
        assert rm.get_person_count() == 0

    def test_no_departure_while_being_seen(self):
        """Person continuously seen should not depart."""
        rm = RoomStateManager()
        rm._departure_timeout = 0.1

        det = _make_detection(face_id="eve")
        rm.update_from_vision("camera_head", [det], _world_lr_fn)
        rm.tick()

        time.sleep(0.05)
        rm.update_from_vision("camera_head", [det], _world_lr_fn)
        time.sleep(0.05)
        rm.update_from_vision("camera_head", [det], _world_lr_fn)

        arrivals, departures = rm.tick()
        assert len(departures) == 0
        assert rm.get_person_count() == 1

    def test_arrivals_only_reported_once(self):
        """Arrival should only appear in one tick, not repeatedly."""
        rm = RoomStateManager()
        det = _make_detection(face_id="frank")
        rm.update_from_vision("camera_head", [det], _world_lr_fn)

        arrivals1, _ = rm.tick()
        assert "frank" in arrivals1

        arrivals2, _ = rm.tick()
        assert "frank" not in arrivals2


class TestMultiCamera:
    """Test cross-camera detection merging."""

    def test_same_face_different_cameras_merges(self):
        """Same face_id from two cameras should be one entry with both cameras."""
        rm = RoomStateManager()
        det_head = _make_detection(face_id="grace", x1=300, x2=400)
        det_left = _make_detection(face_id="grace", x1=500, x2=600)

        rm.update_from_vision("camera_head", [det_head], _world_lr_fn)
        rm.update_from_vision("camera_left", [det_left], _world_lr_fn)

        roster = rm.get_roster()
        assert len(roster) == 1
        assert "camera_head" in roster["grace"].cameras
        assert "camera_left" in roster["grace"].cameras

    def test_unknown_upgrade_to_face_id(self):
        """Unknown person matched by proximity, then recognized, should upgrade ID."""
        rm = RoomStateManager()
        # Side camera sees unknown person
        det_side = _make_detection(face_id="unknown", x1=300, x2=400)
        rm.update_from_vision("camera_left", [det_side], _world_lr_fn)

        roster = rm.get_roster()
        assert len(roster) == 1
        old_id = list(roster.keys())[0]
        assert old_id.startswith("unknown_")

        # Head camera recognizes the same person at similar position
        det_head = _make_detection(face_id="hannah", x1=295, x2=395)
        rm.update_from_vision("camera_head", [det_head], _world_lr_fn)

        roster = rm.get_roster()
        assert len(roster) == 1
        assert "hannah" in roster


class TestRoomSummary:
    """Test the MQTT-publishable room summary format."""

    def test_summary_has_required_fields(self):
        """Room summary should have count and roster list."""
        rm = RoomStateManager()
        det = _make_detection(face_id="iris", emotion="happy")
        rm.update_from_vision("camera_head", [det], _world_lr_fn)

        summary = rm.get_room_summary()
        assert "count" in summary
        assert "roster" in summary
        assert summary["count"] == 1
        assert len(summary["roster"]) == 1

    def test_summary_person_fields(self):
        """Each person in summary should have the expected fields."""
        rm = RoomStateManager()
        det = _make_detection(face_id="jake", emotion="happy")
        rm.update_from_vision("camera_head", [det], _world_lr_fn)

        person = rm.get_room_summary()["roster"][0]
        required = {"person_id", "world_lr", "confidence", "cameras",
                    "first_seen", "last_seen", "emotion", "bbox_height",
                    "face_id", "attention_time"}
        assert required.issubset(set(person.keys()))
        assert person["person_id"] == "jake"
        assert person["emotion"] == "happy"

    def test_empty_room_summary(self):
        """Empty room should return count=0 and empty roster."""
        rm = RoomStateManager()
        summary = rm.get_room_summary()
        assert summary["count"] == 0
        assert summary["roster"] == []


class TestAttentionTracking:
    """Test attention time accumulation."""

    def test_update_attention_increments(self):
        """Attention time should accumulate for tracked person."""
        rm = RoomStateManager()
        det = _make_detection(face_id="kate")
        rm.update_from_vision("camera_head", [det], _world_lr_fn)

        rm.update_attention("kate", 0.5)
        rm.update_attention("kate", 0.3)

        roster = rm.get_roster()
        assert abs(roster["kate"].attention_time - 0.8) < 0.01

    def test_update_attention_unknown_person_ignored(self):
        """Updating attention for nonexistent person should not crash."""
        rm = RoomStateManager()
        rm.update_attention("nobody", 1.0)  # should not raise
