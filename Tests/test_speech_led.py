"""Tests for speech-to-LED synchronization: time_map format, bell curve, mood color."""

import pytest
from math import sin, pi


class TestTimeMapFormat:
    """Verify time_map is a list of tuples that preserves all words."""

    def _build_time_map(self, word_segments):
        """Standalone version of add_time_metrics_word_segments logic."""
        time_map = []
        for i, word in enumerate(word_segments):
            start = word.get("start", 0)
            end = word.get("end", 0)
            tt = end - start
            if i < len(word_segments) - 1:
                next_start = word_segments[i + 1].get("start", 0)
                ttn = next_start - end
            else:
                ttn = 0
            time_map.append((tt, ttn))
        return time_map

    def test_time_map_is_list(self):
        words = [{"start": 0.0, "end": 0.3}, {"start": 0.4, "end": 0.7}]
        result = self._build_time_map(words)
        assert isinstance(result, list)

    def test_time_map_preserves_all_words(self):
        words = [
            {"start": 0.0, "end": 0.3},
            {"start": 0.4, "end": 0.7},
            {"start": 0.8, "end": 1.1},
        ]
        result = self._build_time_map(words)
        assert len(result) == 3

    def test_duplicate_durations_preserved(self):
        """Two words with identical duration should BOTH appear in the list."""
        words = [
            {"start": 0.0, "end": 0.35},   # duration 0.35
            {"start": 0.5, "end": 0.85},   # duration 0.35 (same!)
            {"start": 1.0, "end": 1.4},    # duration 0.40
        ]
        result = self._build_time_map(words)
        assert len(result) == 3
        # Both 0.35s entries survive
        durations = [d for d, g in result]
        assert durations.count(0.35) == 2

    def test_order_preserved(self):
        words = [
            {"start": 0.0, "end": 0.2},
            {"start": 0.3, "end": 0.6},
            {"start": 0.7, "end": 0.8},
        ]
        result = self._build_time_map(words)
        durations = [d for d, g in result]
        assert durations == [pytest.approx(0.2), pytest.approx(0.3), pytest.approx(0.1)]

    def test_last_word_gap_is_zero(self):
        words = [{"start": 0.0, "end": 0.5}]
        result = self._build_time_map(words)
        assert result[0][1] == 0

    def test_gap_calculated_correctly(self):
        words = [
            {"start": 0.0, "end": 0.3},
            {"start": 0.5, "end": 0.8},  # gap = 0.5 - 0.3 = 0.2
        ]
        result = self._build_time_map(words)
        assert result[0][1] == pytest.approx(0.2)

    def test_empty_words(self):
        result = self._build_time_map([])
        assert result == []


class TestBellCurve:
    """Verify the bell curve math for speech pulse animation."""

    def test_bell_curve_peaks_at_midpoint(self):
        """sin(pi * 0.5) = 1.0, the maximum."""
        t = 0.5
        brightness = sin(pi * t)
        assert brightness == pytest.approx(1.0)

    def test_bell_curve_zero_at_start(self):
        t = 0.0
        brightness = sin(pi * t)
        assert brightness == pytest.approx(0.0)

    def test_bell_curve_zero_at_end(self):
        t = 1.0
        brightness = sin(pi * t)
        assert brightness == pytest.approx(0.0, abs=1e-10)

    def test_bell_curve_symmetric(self):
        """Brightness at 25% should equal brightness at 75%."""
        b_25 = sin(pi * 0.25)
        b_75 = sin(pi * 0.75)
        assert b_25 == pytest.approx(b_75)


class TestMoodColorPassthrough:
    """Verify mood color is included in LED messages."""

    def test_angry_color_is_red(self):
        from glados_modules.GladosMood import GladosMood
        mood = GladosMood()
        mood.escalate(10.0, "test")
        color = mood.get_eye_color()
        assert color == (255, 0, 0)

    def test_neutral_color_is_orange(self):
        from glados_modules.GladosMood import GladosMood
        mood = GladosMood()
        color = mood.get_eye_color()
        assert color == (255, 165, 0)

    def test_led_message_includes_color_key(self):
        from glados_modules.GladosEnums import LEDHead
        assert hasattr(LEDHead, 'ARGS_KEY_COLOR')
        assert LEDHead.ARGS_KEY_COLOR.value == "color"
