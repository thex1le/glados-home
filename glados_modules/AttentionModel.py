"""Priority-based attention model for multi-person tracking.

Decides WHO GLaDOS should look at based on social priorities rather than
simple tracking inertia. Replaces the single-target __select_target()
scoring with a priority stack that considers arrivals, gestures, speech,
conversation context, and attention fatigue.

Each priority level has a duration budget — once a priority wins, that
target holds for its budget duration. Higher priorities can interrupt
lower ones mid-budget.
"""

# builtin
import time
from enum import Enum
from typing import Dict, Tuple, Optional

# glados imports
from glados_modules.GladosEnums import AttentionEnums
from glados_modules.RoomStateManager import RoomPerson
from glados_modules.GlogConfig import setup_logger


class AttentionPriority(Enum):
    """Priority levels for attention targeting. Higher value = higher priority."""
    NEW_ARRIVAL = 100
    GESTURE = 80
    SPEAKER = 70
    CONVERSATION = 50
    FATIGUE_GLANCE = 30
    PROXIMITY = 20


class AttentionModel:
    """Priority-based attention model for multi-person rooms.

    Evaluates the room roster against a priority stack and selects
    the highest-priority person to track. Duration budgets prevent
    rapid target switching — once selected, a target holds for its
    budget duration unless a higher priority interrupts.

    Usage:
        model = AttentionModel()
        person_id, reason = model.select_target(roster, dt)
    """

    def __init__(self) -> None:
        self.logger = setup_logger(name="AttentionModel")

        # Current attention state
        self._current_target: Optional[str] = None
        self._current_priority: AttentionPriority = AttentionPriority.PROXIMITY
        self._current_reason: str = ""
        self._budget_remaining: float = 0.0
        self._target_start_time: float = 0.0

        # Fatigue tracking: person_id -> last time GLaDOS looked at them
        self._last_looked_at: Dict[str, float] = {}

        # External inputs (set by other systems, used in priority evaluation)
        self._speaker_world_lr: Optional[float] = None
        self._speaker_time: float = 0.0
        self._conversation_partner: Optional[str] = None

        # Personality modifiers: person_id -> bonus priority points
        self._personality_modifiers: Dict[str, float] = {}

        # Config
        self._fatigue_threshold = AttentionEnums.FATIGUE_NEGLECT_THRESHOLD.value
        self._arrival_window = AttentionEnums.NEW_ARRIVAL_WINDOW.value
        self._gesture_recency = AttentionEnums.GESTURE_RECENCY.value

    def select_target(self, roster: Dict[str, RoomPerson],
                      dt: float) -> Tuple[Optional[str], str]:
        """Select the best person to look at based on priority stack.

        Args:
            roster: Current room roster from RoomStateManager.
            dt: Time delta since last call (seconds), for budget countdown.

        Returns:
            Tuple of (person_id, reason) or (None, "") if no one to look at.
        """
        if not roster:
            self._current_target = None
            self._current_reason = ""
            return None, ""

        now = time.time()

        # Count down current budget
        if self._budget_remaining > 0:
            self._budget_remaining -= dt

        # Stale target migration: if the current target hasn't been seen
        # in >1 second, look for a fresh person nearby. A moving person
        # often gets a new roster ID when ByteTrack loses the track.
        # Migrating to the nearest fresh entry preserves tracking continuity.
        if self._current_target and self._current_target in roster:
            target = roster[self._current_target]
            if (now - target.last_seen) > 1.0:
                best_fresh = None
                best_dist = 30.0  # max degrees to consider "nearby"
                for pid, person in roster.items():
                    if pid == self._current_target:
                        continue
                    if (now - person.last_seen) > 0.5:
                        continue
                    dist = abs(person.world_lr - target.world_lr)
                    if dist < best_dist:
                        best_dist = dist
                        best_fresh = pid
                if best_fresh:
                    self.logger.debug(
                        f"ATTENTION migrate: {self._current_target} stale -> {best_fresh} "
                        f"(dist={best_dist:.1f})")
                    self._current_target = best_fresh
                    self._budget_remaining = self._get_budget(self._current_priority)

        self.logger.debug(
            f"ATTENTION eval: current={self._current_target} budget={self._budget_remaining:.1f}s "
            f"priority={self._current_priority.name if self._current_priority else 'None'} "
            f"roster={list(roster.keys())}")

        # Evaluate all priorities — find the highest
        best_pid, best_priority, best_reason = self._evaluate_priorities(roster, now)

        if best_pid is None:
            # Fallback: pick the person with highest confidence
            best_pid = max(roster, key=lambda p: roster[p].confidence)
            best_priority = AttentionPriority.PROXIMITY
            best_reason = "proximity"

        # Should we switch targets?
        should_switch = False
        if self._current_target is None:
            should_switch = True
        elif self._current_target not in roster:
            # Current target left the room
            should_switch = True
        elif best_priority.value > self._current_priority.value:
            # Higher priority interrupts
            should_switch = True
        elif self._budget_remaining <= 0:
            # Budget expired, re-evaluate
            should_switch = True

        if should_switch and best_pid != self._current_target:
            old_target = self._current_target
            self._current_target = best_pid
            self._current_priority = best_priority
            self._current_reason = best_reason
            self._budget_remaining = self._get_budget(best_priority)
            self._target_start_time = now

            # Record when we stopped looking at old target
            if old_target and old_target in roster:
                self._last_looked_at[old_target] = now

            self.logger.debug(
                f"ATTENTION switch: {old_target} -> {best_pid} "
                f"reason={best_reason} budget={self._budget_remaining:.1f}s")
        elif should_switch and best_pid == self._current_target:
            # Same target but re-evaluated — refresh budget if priority changed
            if best_priority != self._current_priority:
                self._current_priority = best_priority
                self._current_reason = best_reason
                self._budget_remaining = self._get_budget(best_priority)

        # Update last_looked_at for current target
        if self._current_target:
            self._last_looked_at[self._current_target] = now

        return self._current_target, self._current_reason

    def _evaluate_priorities(self, roster: Dict[str, RoomPerson],
                             now: float) -> Tuple[Optional[str], AttentionPriority, str]:
        """Evaluate all priority levels and return the highest-priority target.

        Args:
            roster: Current room roster.
            now: Current timestamp.

        Returns:
            Tuple of (person_id, priority, reason) for the best candidate.
        """
        best_pid = None
        best_priority = AttentionPriority.PROXIMITY
        best_reason = ""

        for pid, person in roster.items():
            person_priorities = []

            # --- NEW ARRIVAL (highest) ---
            # Only treat as new arrival if composite confidence confirms
            # the detection is real — multiple ML models have validated it
            # (pose + face > bbox alone). Phantoms never accumulate composite
            # evidence, so they stay below the threshold.
            # Also require minimum frames as a secondary check.
            min_composite = 0.55  # requires at least bbox + pose or bbox + partial face
            min_frames = 5  # ~0.3s at 15fps
            time_in_room = now - person.first_seen
            confirmed = (person.composite_score >= min_composite and
                         person.frames_seen >= min_frames)
            if time_in_room < self._arrival_window and confirmed:
                person_priorities.append(
                    f"NEW_ARRIVAL(t={time_in_room:.1f}s composite={person.composite_score:.2f})")
                if AttentionPriority.NEW_ARRIVAL.value > best_priority.value:
                    best_pid = pid
                    best_priority = AttentionPriority.NEW_ARRIVAL
                    best_reason = "new_arrival"
                    self.logger.debug(
                        f"PRIORITY: {pid} -> new_arrival (in room {time_in_room:.1f}s "
                        f"composite={person.composite_score:.2f} frames={person.frames_seen})")
                    continue
            elif time_in_room < self._arrival_window and not confirmed:
                person_priorities.append(
                    f"UNCONFIRMED(t={time_in_room:.1f}s composite={person.composite_score:.2f})")
                self.logger.debug(
                    f"PRIORITY: {pid} -> unconfirmed arrival "
                    f"(composite={person.composite_score:.2f} < {min_composite})")
                continue  # skip other priority checks for unconfirmed entries

            # --- GESTURE ---
            # Gesture detection happens per-frame, so we check if the person
            # has an active gesture. This is populated by update_from_vision
            # when gesture data is present. For now, gesture field on RoomPerson
            # is not yet implemented — this will be wired in Increment 4.

            # --- SPEAKER (set externally from STT) ---
            if (self._speaker_world_lr is not None and
                    (now - self._speaker_time) < self._gesture_recency):
                # Find person closest to speaker direction
                angle_diff = abs(person.world_lr - self._speaker_world_lr)
                person_priorities.append(f"SPEAKER(diff={angle_diff:.1f})")
                if angle_diff < 15.0:  # within 15 degrees of speaker direction
                    if AttentionPriority.SPEAKER.value > best_priority.value:
                        best_pid = pid
                        best_priority = AttentionPriority.SPEAKER
                        best_reason = "speaker"

            # --- CONVERSATION PARTNER ---
            if (self._conversation_partner and
                    pid == self._conversation_partner):
                person_priorities.append("CONVERSATION")
                if AttentionPriority.CONVERSATION.value > best_priority.value:
                    best_pid = pid
                    best_priority = AttentionPriority.CONVERSATION
                    best_reason = "conversation"

            # --- FATIGUE GLANCE ---
            if len(roster) >= 2:
                last_look = self._last_looked_at.get(pid, person.first_seen)
                neglect_time = now - last_look
                if neglect_time > self._fatigue_threshold:
                    person_priorities.append(f"FATIGUE(neglect={neglect_time:.0f}s)")
                    if AttentionPriority.FATIGUE_GLANCE.value > best_priority.value:
                        best_pid = pid
                        best_priority = AttentionPriority.FATIGUE_GLANCE
                        best_reason = "fatigue_glance"
                        self.logger.debug(
                            f"ATTENTION fatigue: glancing at {pid} "
                            f"neglected={neglect_time:.0f}s")

            self.logger.debug(
                f"PRIORITY: {pid} conf={person.confidence:.2f} world_lr={person.world_lr:.1f} "
                f"checks=[{', '.join(person_priorities) if person_priorities else 'none'}]")

        # --- PROXIMITY fallback (with personality modifier) ---
        if best_pid is None:
            # Score: composite confidence + personality grudge bonus
            # Composite is better than raw confidence — it reflects accumulated
            # evidence from all ML models, not just YOLO's single-frame guess.
            def _score(pid: str) -> float:
                composite = roster[pid].composite_score
                grudge = self._personality_modifiers.get(pid, 0.0) / 100.0
                return composite + grudge

            best_pid = max(roster, key=_score)
            best_priority = AttentionPriority.PROXIMITY
            if self._personality_modifiers.get(best_pid, 0.0) > 0:
                best_reason = "grudge"
            else:
                best_reason = "proximity"
            self.logger.debug(
                f"PRIORITY: fallback to proximity -> {best_pid} "
                f"score={_score(best_pid):.3f} grudge={self._personality_modifiers.get(best_pid, 0.0)}")

        self.logger.debug(
            f"PRIORITY result: {best_pid} priority={best_priority.name}({best_priority.value}) "
            f"reason={best_reason} budget={self._budget_remaining:.1f}s")
        return best_pid, best_priority, best_reason

    def _get_budget(self, priority: AttentionPriority) -> float:
        """Return the duration budget for a priority level.

        Args:
            priority: The attention priority that won.

        Returns:
            Duration in seconds to hold this target.
        """
        budgets = {
            AttentionPriority.NEW_ARRIVAL: AttentionEnums.BUDGET_NEW_ARRIVAL.value,
            AttentionPriority.GESTURE: AttentionEnums.BUDGET_GESTURE.value,
            AttentionPriority.SPEAKER: AttentionEnums.BUDGET_SPEAKER.value,
            AttentionPriority.CONVERSATION: AttentionEnums.BUDGET_CONVERSATION.value,
            AttentionPriority.FATIGUE_GLANCE: AttentionEnums.BUDGET_FATIGUE_GLANCE.value,
            AttentionPriority.PROXIMITY: AttentionEnums.BUDGET_PROXIMITY.value,
        }
        return budgets.get(priority, 0.0)

    def set_speaker_direction(self, world_lr: float) -> None:
        """Signal that speech was detected from a direction.

        Called by the STT pipeline when voice activity is detected.
        The attention model will try to match this direction to a
        person in the roster.

        Args:
            world_lr: Estimated world yaw angle of the speaker.
        """
        self._speaker_world_lr = world_lr
        self._speaker_time = time.time()

    def set_conversation_partner(self, person_id: str) -> None:
        """Set the current conversation partner.

        Called when GLaDOS finishes speaking to someone. The attention
        model will prefer looking at this person while awaiting a response.

        Args:
            person_id: The person GLaDOS is in dialogue with.
        """
        self._conversation_partner = person_id
        self.logger.debug(f"ATTENTION conversation: partner={person_id}")

    def on_person_departed(self, person_id: str) -> None:
        """Clean up tracking state when a person leaves the room.

        Args:
            person_id: The person who departed.
        """
        self._last_looked_at.pop(person_id, None)
        self._personality_modifiers.pop(person_id, None)
        if self._current_target == person_id:
            self._current_target = None
            self._budget_remaining = 0
        if self._conversation_partner == person_id:
            self._conversation_partner = None

    def clear_conversation_partner(self) -> None:
        """Clear the conversation partner (e.g., when conversation ends)."""
        self._conversation_partner = None

    def add_personality_modifier(self, person_id: str, modifier: float) -> None:
        """Add a personality-based attention modifier for a person.

        Used for grudges: after someone flips GLaDOS off, they get extra
        attention (she watches her enemies). Positive modifier = more attention.

        Args:
            person_id: The person to modify attention for.
            modifier: Priority bonus points (e.g., 15.0 for grudge).
        """
        self._personality_modifiers[person_id] = modifier
        self.logger.debug(f"ATTENTION grudge: {person_id} bonus={modifier}")

    def get_state(self) -> dict:
        """Return serializable attention state for MQTT publishing.

        Returns:
            Dict with current target, reason, and budget remaining.
        """
        return {
            "attention_target": self._current_target,
            "attention_reason": self._current_reason,
            "attention_budget": round(self._budget_remaining, 1),
        }
