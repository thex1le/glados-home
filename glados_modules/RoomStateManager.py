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
from glados_modules.GladosEnums import RoomStateEnums
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

        # Config from enums
        self._departure_timeout = RoomStateEnums.DEPARTURE_TIMEOUT.value
        self._publish_interval = RoomStateEnums.PUBLISH_INTERVAL.value
        self._match_threshold = RoomStateEnums.MATCH_SCORE_THRESHOLD.value
        self._proximity_weight = RoomStateEnums.MATCH_PROXIMITY_WEIGHT.value
        self._height_weight = RoomStateEnums.MATCH_HEIGHT_WEIGHT.value
        self._camera_weight = RoomStateEnums.MATCH_CAMERA_WEIGHT.value
        self._proximity_max = RoomStateEnums.MATCH_PROXIMITY_MAX_DEG.value

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
            for person in self._roster.values():
                person.cameras.discard(camera)

            for detection in detections:
                conf = detection.get("confidence", 0.0)
                if conf < 0.3:
                    continue

                box = detection.get("box", {})
                if not box:
                    continue

                # Extract detection features
                face_data = detection.get("face", {})
                face_id = face_data.get("face_id", "unknown") if face_data else "unknown"
                emotion = face_data.get("emotion", "neutral") if face_data else "neutral"
                bbox_height = float(box.get("y2", 0) - box.get("y1", 0))

                # Compute world angle
                try:
                    world_lr = world_lr_fn(box, camera)
                except Exception:
                    continue

                # Try to match to existing roster entry
                matched_id = self._match_detection(face_id, world_lr, bbox_height, camera)

                if matched_id:
                    # Update existing entry
                    person = self._roster[matched_id]
                    person.world_lr = world_lr
                    person.confidence = max(person.confidence, conf)
                    person.cameras.add(camera)
                    person.last_seen = now
                    person.bbox_height = bbox_height
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
                        f"world_lr={world_lr:.1f} conf={conf:.2f}")
                else:
                    # New person — create entry
                    if face_id != "unknown":
                        person_id = face_id
                    else:
                        self._unknown_counter += 1
                        person_id = f"unknown_{self._unknown_counter}"

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
                    )
                    self._roster[person_id] = person
                    self._pending_arrivals.append(person_id)
                    self.logger.debug(f"ROOM_ROSTER arrival: {person_id}")

    def _match_detection(self, face_id: str, world_lr: float,
                         bbox_height: float, camera: str) -> Optional[str]:
        """Find the best matching roster entry for a detection.

        Args:
            face_id: Face recognition ID ("unknown" if not recognized).
            world_lr: Detection world yaw angle.
            bbox_height: Detection bounding box height.
            camera: Camera name.

        Returns:
            person_id of best match, or None if no match found.
        """
        # Exact face_id match (highest priority)
        if face_id != "unknown" and face_id in self._roster:
            return face_id

        # Fuzzy matching by proximity + height + camera
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

            if score > best_score:
                best_score = score
                best_id = pid

        if best_score >= self._match_threshold:
            return best_id
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
