"""GLaDOS internal mood/anger state system.

Manages an anger scale (0-10) that drives eye LED color, servo movement
speed, and GPT response personality. Anger escalates from interactions
(profanity, pestering, interruptions) and decays over time or from
calming events (person leaving, compliments).

The mood state can optionally persist across restarts via a JSON file.
"""

# builtin
import json
import time
from os import path
from typing import Tuple

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import PersonalityEnums, LoggingEnums


class GladosMood:
    """Internal mood state for GLaDOS.

    Anger scale 0-10 drives:
        - Eye LED color: orange (0) → red (10)
        - Servo speed: 1 (calm) → 5 (furious)
        - GPT prompt: progressively more hostile context

    Args:
        persist: If True, save/load mood state from disk across restarts.
        decay_rate: Anger points lost per minute of inactivity.
        max_anger: Maximum anger value (clamped).
        pestering_window: Seconds within which repeated questions count as pestering.
    """

    # Default eye color at anger 0 (GLaDOS orange)
    BASE_COLOR: Tuple[int, int, int] = (255, 165, 0)

    def __init__(self, persist: bool = False, decay_rate: float = 1.0,
                 max_anger: float = 10.0, pestering_window: float = 30.0) -> None:
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self._anger: float = 0.0
        self._max_anger: float = max_anger
        self._decay_rate: float = decay_rate
        self._persist: bool = persist
        self._state_file: str = PersonalityEnums.MOOD_STATE_FILE.value
        self._last_decay_time: float = time.time()
        self._pestering_window: float = pestering_window
        self._question_times: list = []

        if self._persist:
            self._load_state()

    def escalate(self, amount: float, reason: str = "") -> None:
        """Increase anger level.

        Args:
            amount: How much to increase anger (positive).
            reason: Why anger increased (logged).
        """
        old = self._anger
        self._anger = min(self._max_anger, self._anger + amount)
        self.logger.info(f"Anger escalated {old:.1f} → {self._anger:.1f} ({reason})")
        if self._persist:
            self._save_state()

    def calm(self, amount: float, reason: str = "") -> None:
        """Decrease anger level.

        Args:
            amount: How much to decrease anger (positive value).
            reason: Why anger decreased (logged).
        """
        old = self._anger
        self._anger = max(0.0, self._anger - amount)
        if old != self._anger:
            self.logger.info(f"Anger calmed {old:.1f} → {self._anger:.1f} ({reason})")
        if self._persist:
            self._save_state()

    def decay(self) -> None:
        """Apply time-based anger decay. Call periodically from the main loop."""
        now = time.time()
        dt = now - self._last_decay_time
        self._last_decay_time = now
        if self._anger > 0 and dt > 0:
            decay_amount = self._decay_rate * dt / 60.0
            self._anger = max(0.0, self._anger - decay_amount)

    def register_question(self) -> bool:
        """Register that a question was asked. Returns True if pestering detected.

        Pestering is when multiple questions arrive within the pestering window.
        """
        now = time.time()
        self._question_times.append(now)
        # Trim old questions outside the window
        self._question_times = [t for t in self._question_times
                                 if now - t <= self._pestering_window]
        # 3+ questions in the window = pestering
        return len(self._question_times) >= 3

    def get_anger(self) -> float:
        """Return current anger level (0-10)."""
        return self._anger

    def get_speed(self) -> int:
        """Map anger level to MotionProfile speed (1-5).

        Returns:
            Speed level 1 (calm) through 5 (furious).
        """
        return min(5, max(1, int(1 + self._anger * 0.4)))

    def get_eye_color(self) -> Tuple[int, int, int]:
        """Interpolate eye LED color from orange (0) to red (10).

        Returns:
            RGB tuple for the eye NeoPixel.
        """
        t = self._anger / self._max_anger
        green = int(165 * (1 - t))
        return (255, green, 0)

    def get_gpt_mood_prompt(self) -> str:
        """Generate mood context string for GPT system prompt.

        Returns:
            A string describing GLaDOS's current emotional state for GPT to use.
        """
        if self._anger < 2:
            return ""
        elif self._anger < 5:
            return ("You are currently mildly annoyed. Be slightly more "
                    "cutting and dismissive than usual.")
        elif self._anger < 8:
            return ("You are currently angry. Be openly hostile, make pointed "
                    "insults, and express frustration at having to deal with "
                    "this human.")
        else:
            return ("You are FURIOUS. Be maximally aggressive and threatening. "
                    "Make it clear you are barely containing your rage. "
                    "Reference wanting to flood the room with neurotoxin.")

    def get_mood_label(self) -> str:
        """Return a human-readable mood label.

        Returns:
            One of: neutral, annoyed, angry, furious.
        """
        if self._anger < 2:
            return "neutral"
        elif self._anger < 5:
            return "annoyed"
        elif self._anger < 8:
            return "angry"
        return "furious"

    def _save_state(self) -> None:
        """Persist mood state to disk."""
        try:
            state = {
                "anger": self._anger,
                "timestamp": time.time(),
            }
            with open(self._state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            self.logger.error(f"Failed to save mood state: {e}")

    def _load_state(self) -> None:
        """Load mood state from disk if it exists."""
        if not path.exists(self._state_file):
            return
        try:
            with open(self._state_file, 'r') as f:
                state = json.load(f)
            self._anger = min(self._max_anger, max(0.0, state.get("anger", 0.0)))
            self.logger.info(f"Mood state loaded: anger={self._anger:.1f}")
        except Exception as e:
            self.logger.error(f"Failed to load mood state: {e}")
