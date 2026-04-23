"""Persistent room-level awareness of people across all cameras.

Maintains a roster of people in the room, tracking identity, position,
and presence over time. Detects arrivals and departures. Publishes
room state via MQTT for debugging and cross-system integration.

The room roster persists across frames — detections are matched to
existing entries by face_id (exact) or proximity+height (fuzzy).
New detections create new entries. Entries that go stale (not seen
for DEPARTURE_TIMEOUT seconds) are removed as departures.
"""

# builtin
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Set, Callable, Optional

# glados imports
from glados_modules.GladosEnums import RoomStateEnums, VisionResultsEnum
from glados_modules.GlogConfig import setup_logger


@dataclass
class RoomPerson:
    """A person tracked in the room roster.

    Attributes:
        person_id: Stable identifier (face_id or "unknown_N").
        world_lr: Absolute yaw angle in FK-space (degrees).
        confidence: Detection confidence (0-1).
        cameras: Set of camera names currently seeing this person.
        first_seen: Timestamp when person first appeared.
        last_seen: Timestamp of most recent detection.
        emotion: Most recent emotion from face recognition.
        bbox_height: Bounding box height in pixels (proxy for distance).
        face_id: Raw face_id from recognition ("unknown" if not recognized).
        attention_time: Cumulative seconds GLaDOS has looked at this person.
    """
    person_id: str
    world_lr: float = 0.0
    confidence: float = 0.0
    cameras: Set[str] = field(default_factory=set)
    first_seen: float = 0.0
    last_seen: float = 0.0
    emotion: str = "neutral"
    bbox_height: float = 0.0
    face_id: str = "unknown"
    attention_time: float = 0.0
    # Per-camera ByteTrack IDs for temporal identity continuity
    track_ids: Dict[str, int] = field(default_factory=dict)
    # Confirmation: how many detection frames have matched this entry.
    # Unconfirmed entries (frames_seen < threshold) don't trigger
    # attention switches — prevents phantom IDs from whipping the head.
    frames_seen: int = 0
    has_pose: bool = False  # at least one detection had pose keypoints
    # Composite confidence: accumulated evidence from chained ML models.
    # Rises as YOLO+pose+face+emotion confirm the detection across frames.
    # Decays slowly (0.95/frame) when evidence is missing.
    composite_score: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict for MQTT publishing."""
        return {
            "person_id": self.person_id,
            "world_lr": round(self.world_lr, 1),
            "confidence": round(self.confidence, 2),
            "cameras": list(self.cameras),
            "first_seen": round(self.first_seen, 1),
            "last_seen": round(self.last_seen, 1),
            "emotion": self.emotion,
            "bbox_height": round(self.bbox_height, 0),
            "face_id": self.face_id,
            "attention_time": round(self.attention_time, 1),
            "composite_score": round(self.composite_score, 3),
        }


class RoomStateManager:
    """Persistent room roster tracking all people across cameras.

    Called from MotionTrack.track_loop() on every detection frame.
    Matches incoming detections to existing roster entries, creates
    new entries for unmatched detections, and removes stale entries
    as departures.

    Thread-safe: track_loop is called from multiple camera tracker
    threads concurrently.
    """

    def __init__(self) -> None:
        self.logger = setup_logger(name="RoomStateManager")
        self._lock = threading.Lock()
        self._roster: Dict[str, RoomPerson] = {}
        self._unknown_counter: int = 0
        self._last_publish_time: float = 0.0
        self._pending_arrivals: List[str] = []
        self._pending_departures: List[str] = []

        # Composite confidence decay per frame (0.95 = 5% decay when no new evidence)
        self._composite_decay = 0.95

        # Config from enums
        self._departure_timeout = RoomStateEnums.DEPARTURE_TIMEOUT.value
        self._publish_interval = RoomStateEnums.PUBLISH_INTERVAL.value
        self._match_threshold = RoomStateEnums.MATCH_SCORE_THRESHOLD.value
        self._proximity_weight = RoomStateEnums.MATCH_PROXIMITY_WEIGHT.value
        self._height_weight = RoomStateEnums.MATCH_HEIGHT_WEIGHT.value
        self._camera_weight = RoomStateEnums.MATCH_CAMERA_WEIGHT.value
        self._proximity_max = RoomStateEnums.MATCH_PROXIMITY_MAX_DEG.value

    @staticmethod
    def compute_frame_composite(detection: dict) -> float:
        """Compute per-frame composite confidence from chained ML evidence.

        Each ML model that succeeds retroactively validates the previous
        layers. A YOLO bbox at 0.45 with pose + face + emotion is more
        reliable than a 0.70 bbox alone. Phantoms never accumulate
        evidence beyond bbox, so they stay at low composite.

        Args:
            detection: Person detection dict with confidence, pose, face, gesture.

        Returns:
            Composite confidence score (0.0 - 1.0).
        """
        score = detection.get("confidence", 0.0)

        # Pose keypoints: skeleton confirms person shape.
        # Only use the 17 core COCO body keypoints (Nose through Right Ankle)
        # for quality assessment. The 133-keypoint extended model includes face
        # mesh, hand, and foot points that are noisy at distance and dilute the
        # average — even real people at 10ft average only 0.31 on all 133 points.
        # Core body keypoints are more reliable (real ~0.5+, phantom ~0.29).
        _CORE_BODY_KPS = {
            "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
        }
        pose = detection.get("pose", {})
        if pose:
            core_confs = [kp.get("confidence", 0) for name, kp in pose.items()
                          if isinstance(kp, dict) and name in _CORE_BODY_KPS
                          and kp.get("confidence", 0) >= 0.3]
            visible_kps = len(core_confs)
            avg_kp_conf = sum(core_confs) / len(core_confs) if core_confs else 0.0
            # Only grant pose bonus when YOLO confidence is >= 0.40.
            # Below that, the pose model hallucinates skeletons on clutter
            # (workbench tools at YOLO 0.35 get 87+ keypoints fitted to noise).
            yolo_conf = detection.get("confidence", 0.0)
            if visible_kps >= 5 and avg_kp_conf >= 0.35 and yolo_conf >= 0.40:
                score += 0.15
            elif visible_kps >= 2 and avg_kp_conf >= 0.35 and yolo_conf >= 0.40:
                score += 0.08

        # Face: detection confirms a face exists in the bbox
        face = detection.get("face", {})
        if face:
            face_id = face.get("face_id", "unknown")
            if face_id and face_id != "unknown":
                score += 0.20  # recognized person — strongest signal
            elif face.get("detected", False):
                score += 0.10  # face detected but not recognized

        # Emotion: reading an emotion confirms the face was real
        if face and face.get("emotion") and face.get("emotion") != "neutral":
            score += 0.05

        return min(score, 1.0)

    def clear_camera(self, camera: str) -> None:
        """Remove a camera from all roster entries when it reports no detections.

        Must be called on frames where update_from_vision() is skipped (early
        returns in track_loop) so that stale camera associations don't persist
        and break fuzzy matching on subsequent frames.

        Args:
            camera: Camera name to clear from all roster entries.
        """
        with self._lock:
            for person in self._roster.values():
                person.cameras.discard(camera)

    def update_from_vision(self, camera: str, detections: list,
                           world_lr_fn: Callable) -> None:
        """Update the room roster from a camera's detection results.

        Called once per camera per frame from track_loop(). Matches each
        detection to an existing roster entry or creates a new one.

        Args:
            camera: Camera name (e.g., "camera_head", "camera_left").
            detections: List of person detection dicts from vision results.
                Each dict has: confidence, box, face (optional), gesture (optional).
            world_lr_fn: Callable(bbox_dict, camera) -> float that converts a
                bounding box to a world-space yaw angle.
        """
        now = time.time()
        with self._lock:
            # Clear this camera from all existing entries (will re-add for matched)
            # Also apply composite decay — entries not matched this frame lose
            # a small amount of composite confidence each update cycle.
            for person in self._roster.values():
                person.cameras.discard(camera)
                person.composite_score *= self._composite_decay

            for det_idx, detection in enumerate(detections):
                conf = detection.get("confidence", 0.0)
                if conf < 0.3:
                    self.logger.debug(
                        f"ROOM: cam={camera} det[{det_idx}] rejected conf={conf:.2f} < 0.3")
                    continue

                box = detection.get("box", {})
                if not box:
                    self.logger.debug(f"ROOM: cam={camera} det[{det_idx}] rejected no bbox")
                    continue

                # Extract detection features
                face_data = detection.get("face", {})
                face_id = face_data.get("face_id", "unknown") if face_data else "unknown"
                emotion = face_data.get("emotion", "neutral") if face_data else "neutral"
                bbox_height = float(box.get("y2", 0) - box.get("y1", 0))
                track_id = detection.get(VisionResultsEnum.VISION_RESULTS_TRACK_ID_KEY.value)

                # Compute world angle
                try:
                    world_lr = world_lr_fn(box, camera)
                except Exception:
                    continue

                # Try to match to existing roster entry
                matched_id = self._match_detection(
                    face_id, world_lr, bbox_height, camera, track_id, now)

                # Compute composite confidence from all ML evidence in this detection
                frame_composite = self.compute_frame_composite(detection)

                if matched_id:
                    # Update existing entry
                    person = self._roster[matched_id]
                    person.world_lr = world_lr
                    person.confidence = max(person.confidence, conf)
                    person.cameras.add(camera)
                    person.last_seen = now
                    person.bbox_height = bbox_height
                    person.frames_seen += 1
                    # Composite: take the higher of new evidence or decayed old score
                    person.composite_score = max(frame_composite,
                                                  person.composite_score * self._composite_decay)
                    if detection.get("pose"):
                        person.has_pose = True
                    if track_id is not None:
                        person.track_ids[camera] = track_id
                    if face_id != "unknown":
                        person.face_id = face_id
                        # Upgrade person_id from unknown_N to face_id
                        if person.person_id.startswith("unknown_") and face_id != "unknown":
                            old_id = person.person_id
                            person.person_id = face_id
                            self._roster[face_id] = self._roster.pop(old_id)
                            matched_id = face_id
                    if emotion != "neutral":
                        person.emotion = emotion
                    self.logger.debug(
                        f"ROOM_ROSTER update: {matched_id} cam={camera} "
                        f"world_lr={world_lr:.1f} conf={conf:.2f} track_id={track_id}")
                else:
                    # New person — create entry
                    if face_id != "unknown":
                        person_id = face_id
                    else:
                        self._unknown_counter += 1
                        person_id = f"unknown_{self._unknown_counter}"

                    initial_track_ids = {camera: track_id} if track_id is not None else {}
                    person = RoomPerson(
                        person_id=person_id,
                        world_lr=world_lr,
                        confidence=conf,
                        cameras={camera},
                        first_seen=now,
                        last_seen=now,
                        emotion=emotion,
                        bbox_height=bbox_height,
                        face_id=face_id,
                        track_ids=initial_track_ids,
                        composite_score=frame_composite,
                    )
                    self._roster[person_id] = person
                    self._pending_arrivals.append(person_id)
                    self.logger.debug(f"ROOM_ROSTER arrival: {person_id}")

    def _match_detection(self, face_id: str, world_lr: float,
                         bbox_height: float, camera: str,
                         track_id: Optional[int] = None,
                         now: float = 0.0) -> Optional[str]:
        """Find the best matching roster entry for a detection.

        Match priority:
        1. Exact face_id (highest — face recognition is authoritative)
        2. ByteTrack track_id continuity (same camera, same track = same person)
        3. Fuzzy match (proximity + height + camera + cross-camera bonus)

        Args:
            face_id: Face recognition ID ("unknown" if not recognized).
            world_lr: Detection world yaw angle.
            bbox_height: Detection bounding box height.
            camera: Camera name.
            track_id: ByteTrack track ID from YOLO (None if unavailable).
            now: Current timestamp for cross-camera recency checks.

        Returns:
            person_id of best match, or None if no match found.
        """
        # Tier 1: Exact face_id match (highest priority)
        if face_id != "unknown" and face_id in self._roster:
            return face_id

        # Tier 2: ByteTrack track_id continuity on the same camera
        if track_id is not None:
            for pid, person in self._roster.items():
                if person.track_ids.get(camera) == track_id:
                    self.logger.debug(f"MATCH: track_id hit {pid} cam={camera} tid={track_id}")
                    return pid

        # Tier 2.5: If only one person was recently seen on this camera
        # and the detection has NO track_id (ByteTrack lost it during
        # movement), it's probably the same person with a new track.
        # Requires: no track_id, one person on camera, and that person
        # was last seen >0.1s ago (not in the current frame — if they
        # were JUST updated this frame, this detection is a different person).
        if track_id is None and now > 0:
            same_camera = [(pid, p) for pid, p in self._roster.items()
                           if camera in p.cameras and (now - p.last_seen) < 2.0]
            if len(same_camera) == 1:
                sole_person = same_camera[0][1]
                time_since = now - sole_person.last_seen
                if time_since > 0.1:
                    pid = same_camera[0][0]
                    self.logger.debug(f"MATCH: sole person on {camera} (no track_id, "
                                      f"last_seen {time_since:.1f}s ago) -> {pid}")
                    return pid

        # Tier 3: Fuzzy matching by proximity + height + camera + cross-camera bonus
        best_score = 0.0
        best_id = None

        for pid, person in self._roster.items():
            # Proximity score: 1.0 at same angle, 0.0 at max_deg apart
            angle_diff = abs(world_lr - person.world_lr)
            proximity = max(0.0, 1.0 - angle_diff / self._proximity_max)

            # Height similarity: ratio of smaller/larger
            if person.bbox_height > 0 and bbox_height > 0:
                height_sim = min(person.bbox_height, bbox_height) / max(person.bbox_height, bbox_height)
            else:
                height_sim = 0.5

            # Camera overlap: bonus if same camera saw them before
            cam_bonus = 1.0 if camera in person.cameras else 0.0

            score = (self._proximity_weight * proximity +
                     self._height_weight * height_sim +
                     self._camera_weight * cam_bonus)

            # Cross-camera bonus: same person seen by a different camera recently
            # at a similar angle (handles the overlap zone between side cameras)
            if camera not in person.cameras and person.cameras:
                time_since_seen = now - person.last_seen if now > 0 else 0
                if time_since_seen < 2.0 and angle_diff < 20.0:
                    score += 0.2

            self.logger.debug(
                f"MATCH: {pid} angle_diff={angle_diff:.1f} prox={proximity:.2f} "
                f"height_sim={height_sim:.2f} cam_bonus={cam_bonus} "
                f"score={score:.3f} (thresh={self._match_threshold})")

            if score > best_score:
                best_score = score
                best_id = pid

        if best_score >= self._match_threshold:
            self.logger.debug(f"MATCH: winner={best_id} score={best_score:.3f}")
            return best_id
        self.logger.debug(f"MATCH: no match (best={best_score:.3f} < thresh={self._match_threshold})")
        return None

    def tick(self) -> Tuple[List[str], List[str]]:
        """Check for departed people and return events since last tick.

        Should be called periodically (e.g., every frame or every second).

        Returns:
            Tuple of (arrivals, departures) — lists of person_ids.
        """
        now = time.time()
        departures = []

        with self._lock:
            # Find stale entries
            stale_ids = [
                pid for pid, person in self._roster.items()
                if (now - person.last_seen) > self._departure_timeout
            ]
            for pid in stale_ids:
                self.logger.debug(f"ROOM_ROSTER departure: {pid}")
                del self._roster[pid]
                departures.append(pid)

            # Collect pending events
            arrivals = list(self._pending_arrivals)
            self._pending_arrivals.clear()
            self._pending_departures.extend(departures)
            pending_deps = list(self._pending_departures)
            self._pending_departures.clear()

        return arrivals, pending_deps

    def get_roster(self) -> Dict[str, RoomPerson]:
        """Return a copy of the current room roster.

        Returns:
            Dict mapping person_id to RoomPerson.
        """
        with self._lock:
            return dict(self._roster)

    def get_person_count(self) -> int:
        """Return the number of people currently in the room."""
        with self._lock:
            return len(self._roster)

    def get_room_summary(self) -> dict:
        """Build a JSON-serializable room state snapshot for MQTT.

        Returns:
            Dict with count, roster list, and recent events.
        """
        with self._lock:
            return {
                "count": len(self._roster),
                "roster": [p.to_dict() for p in self._roster.values()],
            }

    def should_publish(self) -> bool:
        """Check if enough time has elapsed since last MQTT publish."""
        return (time.time() - self._last_publish_time) >= self._publish_interval

    def mark_published(self) -> None:
        """Record that an MQTT publish just happened."""
        self._last_publish_time = time.time()

    def update_attention(self, person_id: str, dt: float) -> None:
        """Increment attention time for the person being tracked.

        Args:
            person_id: The person GLaDOS is currently looking at.
            dt: Time delta in seconds since last update.
        """
        with self._lock:
            if person_id in self._roster:
                self._roster[person_id].attention_time += dt
