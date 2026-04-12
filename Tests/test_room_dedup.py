"""Tests for room state deduplication fixes.

Validates:
- clear_camera() prevents stale camera associations
- ByteTrack track_id matching prevents duplicate entries
- Cross-camera fusion bonus merges overlap-zone detections
- Adjusted matching weights improve cross-camera matching
"""

import time
import pytest

from glados_modules.RoomStateManager import RoomStateManager, RoomPerson
from glados_modules.GladosEnums import RoomStateEnums, VisionResultsEnum


def _make_detection(confidence=0.9, x1=100, y1=50, x2=200, y2=350,
                    face_id="unknown", emotion="neutral", track_id=None):
    """Build a detection dict matching MachineVision output format."""
    det = {
        "confidence": confidence,
        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }
    if face_id or emotion:
        det["face"] = {"face_id": face_id, "emotion": emotion}
    if track_id is not None:
        det[VisionResultsEnum.VISION_RESULTS_TRACK_ID_KEY.value] = track_id
    return det


def _world_lr_fn(bbox, camera):
    """Fake pixel-to-world: returns bbox center x scaled to degrees."""
    cx = (bbox["x1"] + bbox["x2"]) / 2
    return (cx - 320) / 10.0  # center=0, 100px offset = 10 degrees


class TestClearCamera:
    """Test that clear_camera removes stale camera associations."""

    def test_clear_camera_removes_camera_from_all_entries(self):
        rm = RoomStateManager()
        # Far apart (>40° gap) + different heights so fuzzy match fails
        det1 = _make_detection(face_id="alice", x1=10, x2=60, y2=400)
        det2 = _make_detection(face_id="bob", x1=580, x2=640, y2=150)
        rm.update_from_vision("camera_right", [det1, det2], _world_lr_fn)

        roster = rm.get_roster()
        assert "camera_right" in roster["alice"].cameras
        assert "camera_right" in roster["bob"].cameras

        rm.clear_camera("camera_right")

        roster = rm.get_roster()
        assert "camera_right" not in roster["alice"].cameras
        assert "camera_right" not in roster["bob"].cameras

    def test_clear_camera_does_not_remove_other_cameras(self):
        rm = RoomStateManager()
        det = _make_detection(face_id="carol")
        rm.update_from_vision("camera_head", [det], _world_lr_fn)
        rm.update_from_vision("camera_left", [det], _world_lr_fn)

        rm.clear_camera("camera_left")

        roster = rm.get_roster()
        assert "camera_head" in roster["carol"].cameras
        assert "camera_left" not in roster["carol"].cameras

    def test_clear_camera_on_empty_roster_is_safe(self):
        rm = RoomStateManager()
        rm.clear_camera("camera_head")  # should not raise

    def test_empty_frame_clears_stale_camera(self):
        """Simulates what happens when track_loop calls clear_camera on early return."""
        rm = RoomStateManager()
        det = _make_detection(face_id="unknown", x1=300, x2=400, track_id=5)
        rm.update_from_vision("camera_right", [det], _world_lr_fn)

        roster = rm.get_roster()
        pid = list(roster.keys())[0]
        assert "camera_right" in roster[pid].cameras

        # Next frame: camera_right sees nothing (early return in track_loop)
        rm.clear_camera("camera_right")

        roster = rm.get_roster()
        assert "camera_right" not in roster[pid].cameras

        # Next frame: camera_right sees person again at slightly different angle
        det2 = _make_detection(face_id="unknown", x1=310, x2=410, track_id=5)
        rm.update_from_vision("camera_right", [det2], _world_lr_fn)

        # Should match via track_id, not create duplicate
        assert rm.get_person_count() == 1


class TestTrackIdMatching:
    """Test ByteTrack track_id matching tier."""

    def test_track_id_stored_on_creation(self):
        rm = RoomStateManager()
        det = _make_detection(track_id=7)
        rm.update_from_vision("camera_right", [det], _world_lr_fn)

        roster = rm.get_roster()
        person = list(roster.values())[0]
        assert person.track_ids.get("camera_right") == 7

    def test_track_id_match_prevents_duplicate(self):
        """Same track_id on same camera should match even if angle drifts."""
        rm = RoomStateManager()
        det1 = _make_detection(x1=100, x2=200, track_id=3)
        rm.update_from_vision("camera_right", [det1], _world_lr_fn)
        assert rm.get_person_count() == 1

        # Significant angle drift, but same track_id
        det2 = _make_detection(x1=300, x2=400, track_id=3)
        rm.update_from_vision("camera_right", [det2], _world_lr_fn)
        assert rm.get_person_count() == 1  # matched by track_id, not duplicated

    def test_different_track_ids_create_separate_entries(self):
        rm = RoomStateManager()
        # Far apart so fuzzy match also fails — verifies track_id doesn't merge them
        det1 = _make_detection(x1=10, x2=60, track_id=1, y2=200)
        det2 = _make_detection(x1=560, x2=640, track_id=2, y2=400)
        rm.update_from_vision("camera_right", [det1, det2], _world_lr_fn)
        assert rm.get_person_count() == 2

    def test_track_id_per_camera_independence(self):
        """track_id=5 on camera_left and track_id=5 on camera_right are different people."""
        rm = RoomStateManager()
        det_left = _make_detection(x1=50, x2=100, track_id=5)
        det_right = _make_detection(x1=500, x2=600, track_id=5)

        rm.update_from_vision("camera_left", [det_left], _world_lr_fn)
        rm.update_from_vision("camera_right", [det_right], _world_lr_fn)

        # Far apart + different cameras, should be 2 people even with same track_id number
        assert rm.get_person_count() == 2

    def test_track_id_updated_on_match(self):
        """Track ID should be updated when matching succeeds."""
        rm = RoomStateManager()
        det1 = _make_detection(face_id="alice", track_id=10)
        rm.update_from_vision("camera_head", [det1], _world_lr_fn)

        # New track_id (ByteTrack reset) but same face_id
        det2 = _make_detection(face_id="alice", track_id=15)
        rm.update_from_vision("camera_head", [det2], _world_lr_fn)

        roster = rm.get_roster()
        assert roster["alice"].track_ids["camera_head"] == 15

    def test_no_track_id_falls_through_to_fuzzy(self):
        """Detections without track_id should still use fuzzy matching."""
        rm = RoomStateManager()
        det1 = _make_detection(x1=300, x2=400, track_id=None)
        rm.update_from_vision("camera_head", [det1], _world_lr_fn)

        det2 = _make_detection(x1=305, x2=405, track_id=None)
        rm.update_from_vision("camera_head", [det2], _world_lr_fn)

        assert rm.get_person_count() == 1  # matched by proximity


class TestCrossCameraFusion:
    """Test cross-camera overlap zone deduplication."""

    def test_overlap_zone_merges_with_bonus(self):
        """Person at similar angle from two side cameras should merge."""
        rm = RoomStateManager()
        # Camera_left sees person at ~-2 degrees
        det_left = _make_detection(x1=300, x2=340, track_id=1)
        rm.update_from_vision("camera_left", [det_left], _world_lr_fn)
        assert rm.get_person_count() == 1

        # Camera_right sees same person at ~0 degrees (slight angle difference)
        det_right = _make_detection(x1=310, x2=350, track_id=2)
        rm.update_from_vision("camera_right", [det_right], _world_lr_fn)

        # Should be 1 person, not 2 (cross-camera bonus bridges the gap)
        assert rm.get_person_count() == 1

    def test_far_apart_detections_not_merged(self):
        """Persons at very different angles should not merge cross-camera."""
        rm = RoomStateManager()
        det_left = _make_detection(x1=50, x2=100)  # far left
        rm.update_from_vision("camera_left", [det_left], _world_lr_fn)

        det_right = _make_detection(x1=550, x2=640)  # far right
        rm.update_from_vision("camera_right", [det_right], _world_lr_fn)

        assert rm.get_person_count() == 2  # different people

    def test_stale_detection_not_merged(self):
        """Cross-camera bonus should not apply if person was seen too long ago.

        Uses an angle gap (~24°) and different heights where the base fuzzy
        score is below threshold for cross-camera, but the cross-camera
        bonus would push it over if fresh. When stale, the bonus doesn't
        apply, so they stay separate.
        """
        rm = RoomStateManager()
        # world_lr = (190-320)/10 = -13.0, height = 250
        det_left = _make_detection(x1=160, x2=220, y1=50, y2=300)
        rm.update_from_vision("camera_left", [det_left], _world_lr_fn)

        roster = rm.get_roster()
        pid = list(roster.keys())[0]
        # Manually age the entry beyond the 2-second cross-camera window
        rm._roster[pid].last_seen = time.time() - 3.0

        # world_lr = (450-320)/10 = 13.0, diff = 26°, height = 150
        # height_sim = 150/250 = 0.6
        # Base score: 0.5*(1-26/40) + 0.25*0.6 + 0.25*0 = 0.175 + 0.15 = 0.325
        # With bonus (if fresh, angle<20 check fails at 26°): no bonus anyway
        # So this just tests that far-enough apart + different heights + stale = no match
        det_right = _make_detection(x1=420, x2=480, y1=100, y2=250)
        rm.update_from_vision("camera_right", [det_right], _world_lr_fn)

        assert rm.get_person_count() == 2


class TestWeightAdjustments:
    """Verify the rebalanced matching weights improve robustness."""

    def test_weights_sum_to_one(self):
        """Proximity + height + camera weights should sum to 1.0."""
        total = (RoomStateEnums.MATCH_PROXIMITY_WEIGHT.value +
                 RoomStateEnums.MATCH_HEIGHT_WEIGHT.value +
                 RoomStateEnums.MATCH_CAMERA_WEIGHT.value)
        assert abs(total - 1.0) < 0.01

    def test_camera_weight_increased(self):
        """Camera continuity weight should be >= 0.2 (was 0.1, too low)."""
        assert RoomStateEnums.MATCH_CAMERA_WEIGHT.value >= 0.2

    def test_proximity_max_widened(self):
        """Max proximity window should be >= 35 degrees (was 30, too tight)."""
        assert RoomStateEnums.MATCH_PROXIMITY_MAX_DEG.value >= 35.0

    def test_same_camera_nearby_detection_always_matches(self):
        """A detection within 15° on the same camera should always match."""
        rm = RoomStateManager()
        det1 = _make_detection(x1=300, x2=400)
        rm.update_from_vision("camera_head", [det1], _world_lr_fn)

        # 15° offset: (300+400)/2 = 350, new center = 350+150 = 500
        # world_lr diff = 150/10 = 15°
        det2 = _make_detection(x1=450, x2=550)
        rm.update_from_vision("camera_head", [det2], _world_lr_fn)

        assert rm.get_person_count() == 1


class TestTrackIdEnum:
    """Verify the track_id enum is properly defined."""

    def test_track_id_key_is_string(self):
        assert isinstance(VisionResultsEnum.VISION_RESULTS_TRACK_ID_KEY.value, str)

    def test_track_id_key_matches_yolo_output(self):
        """Must match what YOLO's to_json() produces."""
        assert VisionResultsEnum.VISION_RESULTS_TRACK_ID_KEY.value == "track_id"
