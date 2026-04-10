"""Tests for RTMPose-YOLO person detection fusion.

Validates that unmatched pose detections create synthetic person entries,
matched poses are assigned to YOLO boxes, low-confidence poses are filtered,
and bounding boxes are computed correctly from keypoints.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from glados_modules.GladosEnums import VisionResultsEnum


def _make_ml_detect():
    """Create a minimal MLDetect instance for testing assign_key_points_to_response."""
    from glados_modules.MachineVision import MLDetect
    ml = MLDetect.__new__(MLDetect)
    ml.logger = MagicMock()
    return ml


def _make_keypoints(people_coords, people_scores):
    """Build numpy arrays matching RTMPose output format.

    Args:
        people_coords: List of (N, 2) coordinate arrays per person.
        people_scores: List of (N,) score arrays per person.

    Returns:
        Tuple of (cords_array, scores_array) as numpy arrays.
    """
    return np.array(people_coords), np.array(people_scores)


def _make_response_with_yolo_person(x1=100, y1=100, x2=300, y2=400, conf=0.9):
    """Build a response dict with one YOLO person detection."""
    return {
        "person": {
            "count": 1,
            "objects": [
                {
                    "name": "person",
                    "confidence": conf,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            ]
        }
    }


def _make_pose_inside_box(x_center=200, y_center=250, spread=30, n_points=133, score=0.8):
    """Create a pose with all keypoints clustered inside a region."""
    coords = []
    scores = []
    for i in range(n_points):
        # Distribute points around center with small spread
        x = x_center + (i % 10 - 5) * (spread / 5)
        y = y_center + (i // 10 - 5) * (spread / 5)
        coords.append([x, y])
        scores.append(score)
    return np.array(coords), np.array(scores)


def _make_pose_outside_box(x_center=500, y_center=300, spread=20, n_points=133, score=0.7):
    """Create a pose with all keypoints outside the typical YOLO box (100-300, 100-400)."""
    coords = []
    scores = []
    for i in range(n_points):
        x = x_center + (i % 10 - 5) * (spread / 5)
        y = y_center + (i // 10 - 5) * (spread / 5)
        coords.append([x, y])
        scores.append(score)
    return np.array(coords), np.array(scores)


class TestPoseMatchedToYolo:
    """Test that poses matching YOLO boxes are assigned normally."""

    def test_matching_pose_assigned_to_yolo_person(self):
        """Pose with 80%+ keypoints inside YOLO box should be assigned."""
        ml = _make_ml_detect()
        response = _make_response_with_yolo_person()

        inside_coords, inside_scores = _make_pose_inside_box()
        cords = np.array([inside_coords])
        scores = np.array([inside_scores])

        result = ml.assign_key_points_to_response(response, cords, scores)

        person = result["person"]["objects"][0]
        assert "pose" in person
        assert len(person["pose"]) > 0

    def test_matched_pose_not_duplicated_as_synthetic(self):
        """Pose matched to YOLO should NOT also create a synthetic detection."""
        ml = _make_ml_detect()
        response = _make_response_with_yolo_person()

        inside_coords, inside_scores = _make_pose_inside_box()
        cords = np.array([inside_coords])
        scores = np.array([inside_scores])

        result = ml.assign_key_points_to_response(response, cords, scores)

        # Should still be exactly 1 person (the YOLO one with pose attached)
        assert result["person"]["count"] == 1
        assert len(result["person"]["objects"]) == 1


class TestSyntheticFromUnmatchedPose:
    """Test that unmatched poses create synthetic person detections."""

    def test_unmatched_pose_creates_synthetic_person(self):
        """Pose not matching any YOLO box should create a synthetic detection."""
        ml = _make_ml_detect()
        response = _make_response_with_yolo_person()

        # Two poses: one inside YOLO box, one outside
        inside_coords, inside_scores = _make_pose_inside_box()
        outside_coords, outside_scores = _make_pose_outside_box(score=0.7)
        cords = np.array([inside_coords, outside_coords])
        scores = np.array([inside_scores, outside_scores])

        result = ml.assign_key_points_to_response(response, cords, scores)

        # Should have 2 people: original YOLO + synthetic from unmatched pose
        assert result["person"]["count"] == 2
        assert len(result["person"]["objects"]) == 2

        # Second person should be synthetic
        synthetic = result["person"]["objects"][1]
        assert synthetic.get("source") == "pose"
        assert "pose" in synthetic
        assert "box" in synthetic
        assert synthetic["confidence"] > 0

    def test_synthetic_has_correct_box_from_keypoints(self):
        """Synthetic detection box should be derived from keypoint extremes + padding."""
        ml = _make_ml_detect()
        response = {"person": {"count": 0, "objects": []}}

        # Create a pose with known keypoint range
        coords = np.array([[[400, 200], [500, 300], [450, 350]] +
                           [[450, 275]] * 130])  # pad to 133
        scores_arr = np.array([[0.8, 0.9, 0.7] + [0.6] * 130])

        result = ml.assign_key_points_to_response(response, coords, scores_arr)

        assert result["person"]["count"] == 1
        synthetic = result["person"]["objects"][0]
        box = synthetic["box"]
        # Box should contain the keypoint range with padding
        assert box["x1"] < 400  # padded left of min x
        assert box["y1"] < 200  # padded above min y
        assert box["x2"] > 500  # padded right of max x
        assert box["y2"] > 350  # padded below max y

    def test_no_yolo_persons_only_pose(self):
        """When YOLO detects nothing but pose finds a person, synthetic should be created."""
        ml = _make_ml_detect()
        response = {}  # no person key at all

        outside_coords, outside_scores = _make_pose_outside_box(score=0.6)
        cords = np.array([outside_coords])
        scores = np.array([outside_scores])

        result = ml.assign_key_points_to_response(response, cords, scores)

        assert "person" in result
        assert result["person"]["count"] == 1
        assert result["person"]["objects"][0].get("source") == "pose"


class TestConfidenceFiltering:
    """Test that low-confidence poses are filtered out."""

    def test_low_confidence_pose_not_added(self):
        """Pose with average confidence below threshold should be skipped."""
        ml = _make_ml_detect()
        response = {"person": {"count": 0, "objects": []}}

        # All keypoints very low confidence
        coords = np.array([[[200, 200]] * 133])
        scores_arr = np.array([[0.05] * 133])  # below 0.1 threshold per-keypoint

        result = ml.assign_key_points_to_response(response, coords, scores_arr)

        # No synthetic detection should be created (all keypoints below kp_min_score)
        if "person" in result:
            assert result["person"]["count"] == 0

    def test_high_confidence_pose_added(self):
        """Pose with good average confidence should create synthetic detection."""
        ml = _make_ml_detect()
        response = {"person": {"count": 0, "objects": []}}

        coords = np.array([[[300, 300]] * 133])
        scores_arr = np.array([[0.8] * 133])

        result = ml.assign_key_points_to_response(response, coords, scores_arr)

        assert "person" in result
        assert result["person"]["count"] == 1


class TestGestureOnSynthetic:
    """Test that gesture recognition works on synthetic detections."""

    def test_synthetic_person_has_pose_key(self):
        """Synthetic detection should have pose data for gesture classification."""
        ml = _make_ml_detect()
        response = {"person": {"count": 0, "objects": []}}

        outside_coords, outside_scores = _make_pose_outside_box(score=0.7)
        cords = np.array([outside_coords])
        scores = np.array([outside_scores])

        result = ml.assign_key_points_to_response(response, cords, scores)

        synthetic = result["person"]["objects"][0]
        assert "pose" in synthetic
        # Pose should have named keypoints (not just a list)
        pose = synthetic["pose"]
        assert isinstance(pose, dict)
        assert "Nose" in pose or len(pose) > 0
