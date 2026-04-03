"""Tests for GladosMood anger system and behavior state machine."""

import time
import pytest
from unittest.mock import MagicMock, patch

from glados_modules.GladosMood import GladosMood
from glados_modules.GladosEnums import BehaviorEnums, PersonalityEnums


class TestMoodEscalation:
    """Test anger escalation and de-escalation."""

    def test_initial_anger_is_zero(self):
        mood = GladosMood()
        assert mood.get_anger() == 0.0

    def test_escalate_increases_anger(self):
        mood = GladosMood()
        mood.escalate(3.0, "profanity")
        assert mood.get_anger() == 3.0

    def test_escalate_capped_at_max(self):
        mood = GladosMood(max_anger=10.0)
        mood.escalate(15.0, "extreme")
        assert mood.get_anger() == 10.0

    def test_calm_decreases_anger(self):
        mood = GladosMood()
        mood.escalate(5.0, "test")
        mood.calm(2.0, "person_left")
        assert mood.get_anger() == 3.0

    def test_calm_floors_at_zero(self):
        mood = GladosMood()
        mood.escalate(2.0, "test")
        mood.calm(5.0, "excessive_calm")
        assert mood.get_anger() == 0.0

    def test_decay_reduces_anger_over_time(self):
        mood = GladosMood(decay_rate=60.0)  # 60 points/min = 1 point/sec
        mood.escalate(5.0, "test")
        mood._last_decay_time = time.time() - 2.0  # pretend 2 seconds passed
        mood.decay()
        assert mood.get_anger() < 5.0
        assert mood.get_anger() > 2.0  # ~3.0 expected (5 - 2*1.0)

    def test_decay_does_not_go_below_zero(self):
        mood = GladosMood(decay_rate=600.0)  # very fast decay
        mood.escalate(1.0, "test")
        mood._last_decay_time = time.time() - 60.0  # 1 minute ago
        mood.decay()
        assert mood.get_anger() == 0.0


class TestMoodSpeedMapping:
    """Test anger-to-speed conversion."""

    def test_zero_anger_speed_1(self):
        mood = GladosMood()
        assert mood.get_speed() == 1

    def test_mid_anger_speed_3(self):
        mood = GladosMood()
        mood.escalate(5.0, "test")
        assert mood.get_speed() == 3

    def test_max_anger_speed_5(self):
        mood = GladosMood()
        mood.escalate(10.0, "test")
        assert mood.get_speed() == 5


class TestMoodEyeColor:
    """Test anger-to-color interpolation."""

    def test_zero_anger_is_orange(self):
        mood = GladosMood()
        assert mood.get_eye_color() == (255, 165, 0)

    def test_max_anger_is_red(self):
        mood = GladosMood()
        mood.escalate(10.0, "test")
        assert mood.get_eye_color() == (255, 0, 0)

    def test_mid_anger_is_between(self):
        mood = GladosMood()
        mood.escalate(5.0, "test")
        r, g, b = mood.get_eye_color()
        assert r == 255
        assert 0 < g < 165
        assert b == 0


class TestMoodGptPrompt:
    """Test GPT mood context generation."""

    def test_neutral_returns_empty(self):
        mood = GladosMood()
        assert mood.get_gpt_mood_prompt() == ""

    def test_annoyed_returns_context(self):
        mood = GladosMood()
        mood.escalate(3.0, "test")
        prompt = mood.get_gpt_mood_prompt()
        assert "annoyed" in prompt.lower() or "cutting" in prompt.lower()

    def test_furious_mentions_neurotoxin(self):
        mood = GladosMood()
        mood.escalate(10.0, "test")
        prompt = mood.get_gpt_mood_prompt()
        assert "neurotoxin" in prompt.lower()


class TestMoodLabel:
    """Test human-readable mood labels."""

    def test_neutral(self):
        mood = GladosMood()
        assert mood.get_mood_label() == "neutral"

    def test_annoyed(self):
        mood = GladosMood()
        mood.escalate(4.0, "test")
        assert mood.get_mood_label() == "annoyed"

    def test_angry(self):
        mood = GladosMood()
        mood.escalate(7.0, "test")
        assert mood.get_mood_label() == "angry"

    def test_furious(self):
        mood = GladosMood()
        mood.escalate(10.0, "test")
        assert mood.get_mood_label() == "furious"


class TestPestering:
    """Test rapid question detection."""

    def test_single_question_not_pestering(self):
        mood = GladosMood()
        assert not mood.register_question()

    def test_three_questions_is_pestering(self):
        mood = GladosMood(pestering_window=60.0)
        mood.register_question()
        mood.register_question()
        assert mood.register_question()  # 3rd = pestering

    def test_old_questions_expire(self):
        mood = GladosMood(pestering_window=1.0)
        mood.register_question()
        mood.register_question()
        # Fake the timestamps to be old
        mood._question_times = [time.time() - 5.0, time.time() - 5.0]
        assert not mood.register_question()  # old ones expired


class TestMoodPersistence:
    """Test save/load of mood state."""

    def test_save_and_load(self, tmp_path):
        mood = GladosMood(persist=True)
        mood._state_file = str(tmp_path / "test_mood.json")
        mood.escalate(7.5, "test")
        mood._save_state()

        mood2 = GladosMood(persist=True)
        mood2._state_file = mood._state_file
        mood2._load_state()
        assert mood2.get_anger() == pytest.approx(7.5)

    def test_no_file_starts_at_zero(self, tmp_path):
        mood = GladosMood(persist=True)
        mood._state_file = str(tmp_path / "nonexistent.json")
        mood._load_state()
        assert mood.get_anger() == 0.0


class TestBreathingOffset:
    """Test breathing sine wave generation."""

    def test_breathing_returns_float(self):
        from Tests.test_tracking_pipeline import _make_motion_track
        mt = _make_motion_track()
        offset = mt._get_breathing_offset()
        assert isinstance(offset, float)

    def test_breathing_within_amplitude(self):
        from Tests.test_tracking_pipeline import _make_motion_track
        from glados_modules.GladosEnums import MotionProfile
        mt = _make_motion_track()
        amp = MotionProfile.BREATHING_AMPLITUDE.value
        for _ in range(100):
            offset = mt._get_breathing_offset()
            assert -amp <= offset <= amp

    def test_breathing_zero_when_asleep(self):
        from Tests.test_tracking_pipeline import _make_motion_track
        mt = _make_motion_track()
        mt._behavior_state = BehaviorEnums.STATE_ASLEEP.value
        assert mt._get_breathing_offset() == 0.0

    def test_breathing_reduced_when_drowsy(self):
        from Tests.test_tracking_pipeline import _make_motion_track
        from glados_modules.GladosEnums import MotionProfile
        mt = _make_motion_track()
        mt._behavior_state = BehaviorEnums.STATE_DROWSY.value
        mt._drowsy_start_time = time.time() - 15.0  # halfway through drowsy
        amp = MotionProfile.BREATHING_AMPLITUDE.value
        offset = mt._get_breathing_offset()
        assert abs(offset) <= amp * 0.6  # should be reduced
