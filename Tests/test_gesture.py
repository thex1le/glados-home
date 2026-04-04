"""Tests for hand gesture recognition from COCO WholeBody keypoints."""

import pytest
from glados_modules.GestureRecognition import (
    GestureClassifier, _distance, _get_kp, EXTEND_THRESHOLD
)
from glados_modules.GladosEnums import VisionResultsEnum


def _make_kp(x: float, y: float, confidence: float = 0.9, location: str = "") -> dict:
    """Create a keypoint dict."""
    return {"x": x, "y": y, "confidence": confidence, "location": location}


def _make_hand(prefix: str, wrist_xy: tuple, finger_tips: dict,
               base_offset: float = 20.0) -> dict:
    """Create a hand pose dict with wrist + finger bases and tips.

    Args:
        prefix: "Left_Hand" or "Right_Hand"
        wrist_xy: (x, y) of wrist
        finger_tips: dict of finger_name -> (x, y) for tip (_4)
        base_offset: how far finger bases (_1) are from wrist

    Returns:
        Pose dict with keypoints for this hand.
    """
    wx, wy = wrist_xy
    pose = {f"{prefix}_Wrist": _make_kp(wx, wy)}

    for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
        # Base is close to wrist
        pose[f"{prefix}_{finger}_1"] = _make_kp(wx + base_offset, wy - base_offset)
        pose[f"{prefix}_{finger}_2"] = _make_kp(wx + base_offset, wy - base_offset)
        pose[f"{prefix}_{finger}_3"] = _make_kp(wx + base_offset, wy - base_offset)

        if finger in finger_tips:
            tx, ty = finger_tips[finger]
            pose[f"{prefix}_{finger}_4"] = _make_kp(tx, ty)
        else:
            # Curled: tip close to wrist
            pose[f"{prefix}_{finger}_4"] = _make_kp(wx + 5, wy + 5)

    return pose


class TestDistance:
    def test_same_point(self):
        p = _make_kp(100, 200)
        assert _distance(p, p) == 0.0

    def test_known_distance(self):
        p1 = _make_kp(0, 0)
        p2 = _make_kp(3, 4)
        assert _distance(p1, p2) == pytest.approx(5.0)


class TestGetKp:
    def test_returns_kp_with_confidence(self):
        pose = {"Nose": _make_kp(100, 200, 0.9)}
        assert _get_kp(pose, "Nose") is not None

    def test_returns_none_low_confidence(self):
        pose = {"Nose": _make_kp(100, 200, 0.1)}
        assert _get_kp(pose, "Nose") is None

    def test_returns_none_missing(self):
        assert _get_kp({}, "Nose") is None


class TestMiddleFinger:
    def test_detects_middle_finger(self):
        gc = GestureClassifier()
        # Middle finger extended upward, others curled
        pose = _make_hand("Right_Hand", (300, 400),
                          {"Middle": (300, 200)})  # tip far above wrist
        result = gc.classify(pose)
        assert result[VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value] == "middle_finger"

    def test_not_middle_when_other_fingers_extended(self):
        gc = GestureClassifier()
        # Middle AND index extended = not middle finger
        pose = _make_hand("Right_Hand", (300, 400),
                          {"Middle": (300, 200), "Index": (280, 200)})
        result = gc.classify(pose)
        assert result[VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value] != "middle_finger"

    def test_not_middle_when_hand_below(self):
        gc = GestureClassifier()
        # Middle extended but tip BELOW wrist
        pose = _make_hand("Right_Hand", (300, 400),
                          {"Middle": (300, 600)})  # below wrist
        result = gc.classify(pose)
        assert result[VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value] != "middle_finger"


class TestThumbsUp:
    def test_detects_thumbs_up(self):
        gc = GestureClassifier()
        # Only thumb extended upward
        pose = _make_hand("Right_Hand", (300, 400),
                          {"Thumb": (300, 200)})  # thumb tip above wrist
        result = gc.classify(pose)
        assert result[VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value] == "thumbs_up"

    def test_detects_thumbs_down(self):
        gc = GestureClassifier()
        # Only thumb extended downward
        pose = _make_hand("Right_Hand", (300, 400),
                          {"Thumb": (300, 600)})  # thumb tip below wrist
        result = gc.classify(pose)
        assert result[VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value] == "thumbs_down"


class TestPoint:
    def test_detects_point(self):
        gc = GestureClassifier()
        # Only index extended
        pose = _make_hand("Right_Hand", (300, 400),
                          {"Index": (500, 300)})  # index pointing right-up
        result = gc.classify(pose)
        assert result[VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value] == "point"

    def test_pointing_direction(self):
        gc = GestureClassifier()
        pose = _make_hand("Right_Hand", (300, 400),
                          {"Index": (500, 400)})  # pointing right
        direction = gc.get_pointing_direction(pose, "Right_Hand")
        assert direction is not None
        dx, dy = direction
        assert dx > 0.9  # pointing mostly right


class TestFist:
    def test_detects_fist(self):
        gc = GestureClassifier()
        # All fingers curled (no extended tips)
        pose = _make_hand("Right_Hand", (300, 400), {})
        result = gc.classify(pose)
        assert result[VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value] == "fist"


class TestOpenPalm:
    def test_detects_open_palm(self):
        gc = GestureClassifier()
        # All fingers extended
        pose = _make_hand("Right_Hand", (300, 400), {
            "Thumb": (200, 200),
            "Index": (250, 150),
            "Middle": (300, 140),
            "Ring": (350, 150),
            "Pinky": (400, 200),
        })
        result = gc.classify(pose)
        assert result[VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value] in ("open_palm", "wave")


class TestBothHands:
    def test_different_gestures_per_hand(self):
        gc = GestureClassifier()
        # Left hand fist, right hand middle finger
        pose = {}
        pose.update(_make_hand("Left_Hand", (100, 400), {}))
        pose.update(_make_hand("Right_Hand", (500, 400), {"Middle": (500, 200)}))
        result = gc.classify(pose)
        assert result[VisionResultsEnum.VISION_RESULTS_GESTURE_LEFT.value] == "fist"
        assert result[VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value] == "middle_finger"


class TestNoGesture:
    def test_no_hand_keypoints(self):
        gc = GestureClassifier()
        result = gc.classify({})
        assert result[VisionResultsEnum.VISION_RESULTS_GESTURE_LEFT.value] == "none"
        assert result[VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value] == "none"

    def test_low_confidence_ignored(self):
        gc = GestureClassifier()
        pose = {"Right_Hand_Wrist": _make_kp(300, 400, confidence=0.1)}
        result = gc.classify(pose)
        assert result[VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value] == "none"
