"""Tests for face recognition database operations and target selection with face ID."""

import json
import os
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from glados_modules.GladosEnums import (
    VisionResultsEnum, FaceEnums, CameraEnum, ServoEnum
)


class TestFaceDatabase:
    """Test face database CRUD operations."""

    def _make_face_rec(self, tmp_path):
        """Create a GladosFaceRecognition with mocked models."""
        from glados_modules.FaceRecognition import GladosFaceRecognition
        with patch.object(GladosFaceRecognition, '__init__', lambda self, *a, **kw: None):
            fr = GladosFaceRecognition.__new__(GladosFaceRecognition)
        fr.__name__ = "GladosFaceRecognition"
        fr.logger = MagicMock()
        fr._db_path = str(tmp_path / "test_faces.json")
        fr._threshold = FaceEnums.SIMILARITY_THRESHOLD.value
        fr._max_embeddings = FaceEnums.MAX_EMBEDDINGS_PER_PERSON.value
        fr._database = {}
        fr._enrollment_pending = None
        fr._face_app = None
        fr._emotion_model = None
        return fr

    def test_enroll_new_face(self, tmp_path):
        fr = self._make_face_rec(tmp_path)
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        fr.enroll("ben", embedding)
        assert "ben" in fr._database
        assert fr._database["ben"]["enrollment_count"] == 1

    def test_enroll_updates_existing(self, tmp_path):
        fr = self._make_face_rec(tmp_path)
        for i in range(3):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            fr.enroll("ben", emb)
        assert fr._database["ben"]["enrollment_count"] == 3
        assert len(fr._database["ben"]["embeddings"]) == 3

    def test_enroll_caps_at_max(self, tmp_path):
        fr = self._make_face_rec(tmp_path)
        for i in range(FaceEnums.MAX_EMBEDDINGS_PER_PERSON.value + 5):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            fr.enroll("ben", emb)
        assert len(fr._database["ben"]["embeddings"]) == FaceEnums.MAX_EMBEDDINGS_PER_PERSON.value

    def test_identify_known_face(self, tmp_path):
        fr = self._make_face_rec(tmp_path)
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        fr.enroll("ben", embedding)
        # Same embedding should match
        face_id, similarity = fr.identify(embedding)
        assert face_id == "ben"
        assert similarity > FaceEnums.SIMILARITY_THRESHOLD.value

    def test_identify_unknown_face(self, tmp_path):
        fr = self._make_face_rec(tmp_path)
        # Enroll one person
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        fr.enroll("ben", emb1)
        # Different random embedding should not match
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        face_id, similarity = fr.identify(emb2)
        assert face_id == "unknown"

    def test_forget_face(self, tmp_path):
        fr = self._make_face_rec(tmp_path)
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        fr.enroll("ben", emb)
        assert fr.forget("ben")
        assert "ben" not in fr._database

    def test_forget_unknown_returns_false(self, tmp_path):
        fr = self._make_face_rec(tmp_path)
        assert not fr.forget("nobody")

    def test_save_and_load(self, tmp_path):
        fr = self._make_face_rec(tmp_path)
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        fr.enroll("ben", emb)
        # Load into new instance
        fr2 = self._make_face_rec(tmp_path)
        fr2._db_path = fr._db_path
        fr2.load_database()
        assert "ben" in fr2._database
        assert fr2._database["ben"]["enrollment_count"] == 1

    def test_name_normalized(self, tmp_path):
        fr = self._make_face_rec(tmp_path)
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        fr.enroll("  Ben  ", emb)
        assert "ben" in fr._database

    def test_match_face_to_person(self, tmp_path):
        fr = self._make_face_rec(tmp_path)
        face_key = VisionResultsEnum.VISION_RESULTS_FACE_KEY.value
        faces = [{
            VisionResultsEnum.VISION_RESULTS_FACE_BOX_KEY.value: {
                "x1": 150, "y1": 60, "x2": 250, "y2": 200
            },
            "embedding": np.zeros(512),
            VisionResultsEnum.VISION_RESULTS_FACE_ID_KEY.value: "ben",
            VisionResultsEnum.VISION_RESULTS_FACE_CONFIDENCE_KEY.value: 0.9,
            VisionResultsEnum.VISION_RESULTS_FACE_DETECTED_KEY.value: True,
            VisionResultsEnum.VISION_RESULTS_EMOTION_KEY.value: "happy",
            VisionResultsEnum.VISION_RESULTS_EMOTION_SCORES_KEY.value: {},
        }]
        persons = [{
            "box": {"x1": 100, "y1": 50, "x2": 300, "y2": 400},
            "confidence": 0.9,
        }]
        fr.match_faces_to_persons(faces, persons)
        assert face_key in persons[0]
        assert persons[0][face_key][VisionResultsEnum.VISION_RESULTS_FACE_ID_KEY.value] == "ben"

    def test_no_match_when_face_outside_person(self, tmp_path):
        fr = self._make_face_rec(tmp_path)
        face_key = VisionResultsEnum.VISION_RESULTS_FACE_KEY.value
        faces = [{
            VisionResultsEnum.VISION_RESULTS_FACE_BOX_KEY.value: {
                "x1": 400, "y1": 60, "x2": 500, "y2": 200
            },
            "embedding": np.zeros(512),
            VisionResultsEnum.VISION_RESULTS_FACE_ID_KEY.value: "ben",
            VisionResultsEnum.VISION_RESULTS_FACE_CONFIDENCE_KEY.value: 0.9,
            VisionResultsEnum.VISION_RESULTS_FACE_DETECTED_KEY.value: True,
            VisionResultsEnum.VISION_RESULTS_EMOTION_KEY.value: "neutral",
            VisionResultsEnum.VISION_RESULTS_EMOTION_SCORES_KEY.value: {},
        }]
        persons = [{
            "box": {"x1": 100, "y1": 50, "x2": 300, "y2": 400},
            "confidence": 0.9,
        }]
        fr.match_faces_to_persons(faces, persons)
        assert face_key not in persons[0]


class TestTargetSelectionWithFaceId:
    """Test that face ID influences target selection scoring."""

    def test_face_id_match_wins_over_confidence(self):
        from Tests.test_tracking_pipeline import _make_motion_track
        mt = _make_motion_track()
        center_box = {"x1": 290, "y1": 200, "x2": 350, "y2": 300}
        center_angle = mt._pixel_to_world_angle(
            center_box, CameraEnum.CAMERA_HEAD.value, ServoEnum.X_AXIS.value)
        mt._last_tracked_world_lr = center_angle
        mt._last_tracked_bbox_height = 100.0
        mt._last_tracked_face_id = "ben"

        face_key = VisionResultsEnum.VISION_RESULTS_FACE_KEY.value
        face_id_key = VisionResultsEnum.VISION_RESULTS_FACE_ID_KEY.value

        # Ben has lower confidence but is the tracked person
        ben = {"confidence": 0.6, "box": {"x1": 290, "y1": 200, "x2": 350, "y2": 300},
               face_key: {face_id_key: "ben"}}
        stranger = {"confidence": 0.95, "box": {"x1": 290, "y1": 200, "x2": 350, "y2": 300},
                     face_key: {face_id_key: "unknown"}}
        result = mt._MotionTrack__select_target([ben, stranger], CameraEnum.CAMERA_HEAD.value)
        assert result[face_key][face_id_key] == "ben"

    def test_no_face_history_falls_back_to_position(self):
        from Tests.test_tracking_pipeline import _make_motion_track
        mt = _make_motion_track()
        center_box = {"x1": 290, "y1": 200, "x2": 350, "y2": 300}
        center_angle = mt._pixel_to_world_angle(
            center_box, CameraEnum.CAMERA_HEAD.value, ServoEnum.X_AXIS.value)
        mt._last_tracked_world_lr = center_angle
        mt._last_tracked_bbox_height = 100.0
        mt._last_tracked_face_id = None  # no face history

        # Same confidence, nearby wins
        nearby = {"confidence": 0.8, "box": {"x1": 290, "y1": 200, "x2": 350, "y2": 300}}
        far = {"confidence": 0.8, "box": {"x1": 10, "y1": 200, "x2": 70, "y2": 300}}
        result = mt._MotionTrack__select_target([nearby, far], CameraEnum.CAMERA_HEAD.value)
        assert result == nearby
