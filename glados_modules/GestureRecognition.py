"""Hand gesture classification from COCO WholeBody hand keypoints.

Detects gestures using pure geometry — distances and angles between finger
keypoints. No ML models required. Uses the 21 keypoints per hand already
produced by RTM Wholebody pose estimation.

Gestures detected:
    - middle_finger: middle extended, others curled, hand above wrist
    - thumbs_up: thumb extended upward, fingers curled
    - thumbs_down: thumb extended downward, fingers curled
    - wave: all fingers extended, hand oscillating (temporal)
    - point: index extended, others curled
    - open_palm: all fingers extended, hand still
    - fist: all fingers curled
    - none: no gesture detected
"""

# builtin
import time
from math import sqrt
from typing import Dict, List, Optional, Tuple

# glados imports
from glados_modules.GladosEnums import VisionResultsEnum


# Finger names in COCO WholeBody order
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Minimum confidence for a keypoint to be used in gesture detection
MIN_CONFIDENCE = 0.3

# Distance ratio threshold: tip-to-wrist / base-to-wrist > this = extended
EXTEND_THRESHOLD = 1.4


def _distance(p1: dict, p2: dict) -> float:
    """Euclidean distance between two keypoints."""
    return sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)


def _get_kp(pose: dict, name: str) -> Optional[dict]:
    """Get a keypoint by name if it exists and has sufficient confidence."""
    kp = pose.get(name)
    if kp and kp.get("confidence", 0) >= MIN_CONFIDENCE:
        return kp
    return None


class GestureClassifier:
    """Classify hand gestures from COCO WholeBody hand keypoints.

    Pure geometry — no ML models. Uses distances and angles between
    finger keypoints to detect common gestures.

    Hand keypoint naming:
        {prefix}_Wrist
        {prefix}_Thumb_1 through _4 (base to tip)
        {prefix}_Index_1 through _4
        {prefix}_Middle_1 through _4
        {prefix}_Ring_1 through _4
        {prefix}_Pinky_1 through _4

    Where prefix is "Left_Hand" or "Right_Hand".
    """

    GESTURES = ["middle_finger", "thumbs_up", "thumbs_down",
                "wave", "point", "open_palm", "fist", "none"]

    def __init__(self) -> None:
        # Wave detection history: {prefix: [(timestamp, wrist_x), ...]}
        self._wave_history: Dict[str, List[Tuple[float, float]]] = {
            "Left_Hand": [],
            "Right_Hand": [],
        }
        self._wave_window: float = 1.5  # seconds of history for wave detection
        self._wave_min_reversals: int = 2  # minimum direction changes to count as wave

    def classify(self, pose_data: dict) -> dict:
        """Classify gestures for both hands.

        Args:
            pose_data: Dict of keypoint_name -> {x, y, confidence, location}

        Returns:
            Dict with left_hand, right_hand gesture names and confidence.
        """
        result = {
            VisionResultsEnum.VISION_RESULTS_GESTURE_LEFT.value: "none",
            VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value: "none",
            VisionResultsEnum.VISION_RESULTS_GESTURE_CONFIDENCE.value: 0.0,
        }

        best_confidence = 0.0
        for prefix, key in [("Left_Hand", VisionResultsEnum.VISION_RESULTS_GESTURE_LEFT.value),
                             ("Right_Hand", VisionResultsEnum.VISION_RESULTS_GESTURE_RIGHT.value)]:
            gesture, confidence = self._classify_hand(pose_data, prefix)
            result[key] = gesture
            if confidence > best_confidence:
                best_confidence = confidence

        result[VisionResultsEnum.VISION_RESULTS_GESTURE_CONFIDENCE.value] = round(best_confidence, 2)
        return result

    def _classify_hand(self, pose: dict, prefix: str) -> Tuple[str, float]:
        """Classify gesture for a single hand.

        Returns:
            Tuple of (gesture_name, confidence).
        """
        wrist = _get_kp(pose, f"{prefix}_Wrist")
        if not wrist:
            return "none", 0.0

        # Get finger extension states
        finger_states = self._get_finger_states(pose, prefix, wrist)
        if not finger_states:
            return "none", 0.0

        # Count extended/curled fingers
        extended = [f for f, ext in finger_states.items() if ext]
        curled = [f for f, ext in finger_states.items() if not ext]

        # Average confidence of hand keypoints for overall gesture confidence
        avg_conf = self._hand_confidence(pose, prefix)

        # Check gestures in priority order
        # Middle finger: only middle extended, others curled, hand above wrist
        if (len(extended) == 1 and "Middle" in extended):
            middle_tip = _get_kp(pose, f"{prefix}_Middle_4")
            if middle_tip and middle_tip["y"] < wrist["y"]:
                return "middle_finger", avg_conf

        # Thumbs up: only thumb extended, tip above wrist
        if len(extended) == 1 and "Thumb" in extended:
            thumb_tip = _get_kp(pose, f"{prefix}_Thumb_4")
            if thumb_tip and thumb_tip["y"] < wrist["y"]:
                return "thumbs_up", avg_conf

        # Thumbs down: only thumb extended, tip below wrist
        if len(extended) == 1 and "Thumb" in extended:
            thumb_tip = _get_kp(pose, f"{prefix}_Thumb_4")
            if thumb_tip and thumb_tip["y"] > wrist["y"]:
                return "thumbs_down", avg_conf

        # Point: only index extended
        if len(extended) == 1 and "Index" in extended:
            return "point", avg_conf

        # Fist: all fingers curled
        if len(curled) == 5:
            return "fist", avg_conf

        # Open palm or wave: all fingers extended
        if len(extended) >= 4:
            # Check for wave (temporal oscillation)
            if self._detect_wave(pose, prefix, wrist):
                return "wave", avg_conf
            return "open_palm", avg_conf

        return "none", 0.0

    def _get_finger_states(self, pose: dict, prefix: str,
                            wrist: dict) -> Dict[str, bool]:
        """Determine which fingers are extended vs curled.

        Args:
            pose: Full pose keypoint dict.
            prefix: "Left_Hand" or "Right_Hand".
            wrist: Wrist keypoint dict.

        Returns:
            Dict of finger_name -> True (extended) / False (curled).
            Returns empty dict if insufficient keypoints.
        """
        states = {}
        for finger in FINGERS:
            base = _get_kp(pose, f"{prefix}_{finger}_1")
            tip = _get_kp(pose, f"{prefix}_{finger}_4")
            if not base or not tip:
                continue

            base_dist = _distance(wrist, base)
            tip_dist = _distance(wrist, tip)

            if base_dist < 1:
                states[finger] = False
                continue

            states[finger] = (tip_dist / base_dist) > EXTEND_THRESHOLD

        # Need at least 4 fingers detected to classify
        if len(states) < 4:
            return {}
        return states

    def _hand_confidence(self, pose: dict, prefix: str) -> float:
        """Average confidence of detected hand keypoints."""
        confidences = []
        for finger in FINGERS:
            for joint in range(1, 5):
                kp = pose.get(f"{prefix}_{finger}_{joint}")
                if kp:
                    confidences.append(kp.get("confidence", 0))
        wrist = pose.get(f"{prefix}_Wrist")
        if wrist:
            confidences.append(wrist.get("confidence", 0))
        return sum(confidences) / len(confidences) if confidences else 0.0

    def _detect_wave(self, pose: dict, prefix: str, wrist: dict) -> bool:
        """Detect waving motion by tracking wrist x-position oscillation.

        A wave is defined as the wrist moving back and forth (2+ direction
        reversals) within a 1.5-second window.
        """
        now = time.time()
        history = self._wave_history[prefix]

        # Add current position
        history.append((now, wrist["x"]))

        # Trim old entries
        self._wave_history[prefix] = [(t, x) for t, x in history
                                       if now - t <= self._wave_window]
        history = self._wave_history[prefix]

        if len(history) < 4:
            return False

        # Count direction reversals in x
        reversals = 0
        for i in range(2, len(history)):
            prev_dx = history[i - 1][1] - history[i - 2][1]
            curr_dx = history[i][1] - history[i - 1][1]
            # Direction reversal: sign change with sufficient magnitude
            if prev_dx * curr_dx < 0 and abs(curr_dx) > 5:
                reversals += 1

        return reversals >= self._wave_min_reversals

    def get_pointing_direction(self, pose: dict, prefix: str) -> Optional[Tuple[float, float]]:
        """Get the direction vector of a pointing gesture.

        Args:
            pose: Full pose keypoint dict.
            prefix: "Left_Hand" or "Right_Hand".

        Returns:
            Tuple of (dx, dy) direction vector from wrist to index tip,
            or None if keypoints are insufficient.
        """
        wrist = _get_kp(pose, f"{prefix}_Wrist")
        index_tip = _get_kp(pose, f"{prefix}_Index_4")
        if not wrist or not index_tip:
            return None
        dx = index_tip["x"] - wrist["x"]
        dy = index_tip["y"] - wrist["y"]
        mag = sqrt(dx * dx + dy * dy)
        if mag < 1:
            return None
        return (dx / mag, dy / mag)
