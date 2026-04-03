"""Tests for GladosLocal command detection, time parsing, and sight processing."""

import pytest
from unittest.mock import MagicMock, patch
import regex as re


class TestCheckLocalCommand:
    """Test regex matching for user prompt commands."""

    def _check(self, prompt, command):
        """Standalone version of check_local_command logic."""
        if not isinstance(command, re.Pattern):
            command = re.escape(command)
        match = re.search(command, prompt)
        return bool(match)

    def test_string_match(self):
        assert self._check("hello world", "hello")

    def test_string_no_match(self):
        assert not self._check("foo bar", "baz")

    def test_regex_pattern(self):
        assert self._check("test123", re.compile(r'\d+'))

    def test_partial_match(self):
        assert self._check("say hello there", "hello")

    def test_case_sensitive(self):
        # The actual code lowercases the prompt before calling
        assert not self._check("HELLO", "hello")

    def test_fuck_you_detection(self):
        assert self._check("fuck you", "fuck you")
        assert self._check("oh fuck you then", "fuck you")
        assert not self._check("forget about you", "fuck you")


class TestTranslateTime:
    """Test time string parsing."""

    def _translate(self, prompt):
        """Standalone version of translate_time."""
        pattern = r'(\d+)\s*(hour|minute|second)s?'
        matches = re.findall(pattern, prompt)
        time_dict = {f'{unit}s': int(value) for value, unit in matches}
        total = (time_dict.get('seconds', 0) +
                 time_dict.get('minutes', 0) * 60 +
                 time_dict.get('hours', 0) * 3600)
        time_dict['total_seconds'] = total
        return time_dict

    def test_all_units(self):
        result = self._translate("2 hours 30 minutes 15 seconds")
        assert result['hours'] == 2
        assert result['minutes'] == 30
        assert result['seconds'] == 15
        assert result['total_seconds'] == 9015

    def test_single_unit(self):
        result = self._translate("3 minutes")
        assert result['minutes'] == 3
        assert result['total_seconds'] == 180

    def test_singular_form(self):
        result = self._translate("1 hour 1 minute 1 second")
        assert result['hours'] == 1
        assert result['total_seconds'] == 3661

    def test_no_time_found(self):
        result = self._translate("set a timer")
        assert result['total_seconds'] == 0

    def test_seconds_only(self):
        result = self._translate("45 seconds")
        assert result['total_seconds'] == 45


class TestGetTempRegex:
    """Test temperature query regex patterns."""

    def _matches(self, prompt):
        c_str = r"what(?:'?s| is) the (current )?(outside )?(temp(erature)?)( outside)?\??"
        return bool(re.search(re.compile(c_str), prompt.lower()))

    def test_whats_the_temperature(self):
        assert self._matches("what's the temperature?")

    def test_what_is_the_temperature(self):
        assert self._matches("what is the temperature?")

    def test_current_temperature(self):
        assert self._matches("what's the current temperature?")

    def test_outside_temperature(self):
        assert self._matches("what's the outside temperature?")

    def test_temp_abbreviation(self):
        assert self._matches("what's the temp?")

    def test_temperature_outside(self):
        assert self._matches("what is the temperature outside?")

    def test_no_match(self):
        assert not self._matches("is it cold?")
        assert not self._matches("how hot is it?")

    def test_case_insensitive(self):
        assert self._matches("WHAT'S THE TEMPERATURE?")


class TestTimerRegex:
    """Test timer command regex patterns."""

    def _match_set(self, prompt):
        return bool(re.search(re.compile(r'set\s+(a\s+|the\s+)?timer'), prompt.lower()))

    def _match_stop(self, prompt):
        return bool(re.search(re.compile(r'(stop|cancel)\s+(the\s+|a\s+)?timer'), prompt.lower()))

    def test_set_timer(self):
        assert self._match_set("set a timer for 5 minutes")
        assert self._match_set("set the timer for 10 seconds")
        assert self._match_set("set timer for 1 hour")

    def test_stop_timer(self):
        assert self._match_stop("stop the timer")
        assert self._match_stop("cancel timer")
        assert self._match_stop("cancel the timer")

    def test_no_match(self):
        assert not self._match_set("start a timer")
        assert not self._match_stop("pause the timer")


class TestProcessSight:
    """Test vision result string building."""

    def _process(self, seen, confidence=0.65):
        """Standalone version of process_sight + __adjust_count."""
        context = ["You can see the following things in the room"]
        for item in seen.keys():
            count = sum(1 for o in seen[item]["objects"] if o['confidence'] >= confidence)
            if count == 0:
                continue
            context.append(f"{count} {item}")
        return ", ".join(context)

    def test_single_person(self):
        seen = {"person": {"objects": [{"confidence": 0.9}]}}
        result = self._process(seen)
        assert "1 person" in result

    def test_multiple_objects(self):
        seen = {
            "person": {"objects": [{"confidence": 0.9}]},
            "dog": {"objects": [{"confidence": 0.8}]},
        }
        result = self._process(seen)
        assert "1 person" in result
        assert "1 dog" in result

    def test_below_confidence_filtered(self):
        seen = {"person": {"objects": [{"confidence": 0.3}]}}
        result = self._process(seen, confidence=0.65)
        assert "person" not in result

    def test_multiple_same_object(self):
        seen = {"person": {"objects": [
            {"confidence": 0.9}, {"confidence": 0.8}
        ]}}
        result = self._process(seen)
        assert "2 person" in result

    def test_empty(self):
        result = self._process({})
        assert result == "You can see the following things in the room"

    def test_mixed_confidence(self):
        seen = {"person": {"objects": [
            {"confidence": 0.9}, {"confidence": 0.3}, {"confidence": 0.7}
        ]}}
        result = self._process(seen, confidence=0.65)
        assert "2 person" in result  # only 0.9 and 0.7 pass


class TestProcessSightWithFaces:
    """Test vision result string building with face identity and emotion."""

    def _make_gl(self):
        """Create a minimal GladosLocal for testing process_sight."""
        from glados_modules.GLaDOSLocal import GladosLocal
        with patch.object(GladosLocal, '__init__', lambda self, *a, **kw: None):
            gl = GladosLocal.__new__(GladosLocal)
        gl.vision_confidence = 0.65
        gl.logger = MagicMock()
        return gl

    def test_person_with_name(self):
        gl = self._make_gl()
        seen = {"person": {"objects": [
            {"confidence": 0.9, "face": {"face_id": "ben", "emotion": "neutral"}}
        ]}}
        result = gl.process_sight(seen)
        assert "ben" in result

    def test_person_with_emotion(self):
        gl = self._make_gl()
        seen = {"person": {"objects": [
            {"confidence": 0.9, "face": {"face_id": "ben", "emotion": "happy"}}
        ]}}
        result = gl.process_sight(seen)
        assert "happy" in result

    def test_unknown_face_not_named(self):
        gl = self._make_gl()
        seen = {"person": {"objects": [
            {"confidence": 0.9, "face": {"face_id": "unknown", "emotion": "neutral"}}
        ]}}
        result = gl.process_sight(seen)
        assert "unknown" not in result
        assert "person" in result

    def test_neutral_emotion_not_mentioned(self):
        gl = self._make_gl()
        seen = {"person": {"objects": [
            {"confidence": 0.9, "face": {"face_id": "ben", "emotion": "neutral"}}
        ]}}
        result = gl.process_sight(seen)
        assert "neutral" not in result

    def test_no_face_data_still_counts(self):
        gl = self._make_gl()
        seen = {"person": {"objects": [{"confidence": 0.9}]}}
        result = gl.process_sight(seen)
        assert "1 person" in result

    def test_non_person_objects_unchanged(self):
        gl = self._make_gl()
        seen = {"chair": {"objects": [{"confidence": 0.9}]}}
        result = gl.process_sight(seen)
        assert "1 chair" in result


class TestEnrollFaceCommand:
    """Test face enrollment voice command parsing."""

    def _make_gl(self):
        from glados_modules.GLaDOSLocal import GladosLocal
        with patch.object(GladosLocal, '__init__', lambda self, *a, **kw: None):
            gl = GladosLocal.__new__(GladosLocal)
        gl.logger = MagicMock()
        gl.send_command = MagicMock()
        gl.speak = MagicMock()
        return gl

    def test_my_name_is(self):
        gl = self._make_gl()
        assert gl.enroll_face("my name is Ben")
        gl.send_command.assert_called_once()
        args = gl.send_command.call_args
        assert args[1]["command"]["name"] == "ben"
        assert args[1]["command"]["cmd"] == "enroll"

    def test_im(self):
        gl = self._make_gl()
        assert gl.enroll_face("I'm Sarah")
        args = gl.send_command.call_args
        assert args[1]["command"]["name"] == "sarah"

    def test_call_me(self):
        gl = self._make_gl()
        assert gl.enroll_face("call me Dave")
        args = gl.send_command.call_args
        assert args[1]["command"]["name"] == "dave"

    def test_forget_person(self):
        gl = self._make_gl()
        assert gl.enroll_face("forget Ben")
        args = gl.send_command.call_args
        assert args[1]["command"]["cmd"] == "forget"
        assert args[1]["command"]["name"] == "ben"

    def test_no_match(self):
        gl = self._make_gl()
        assert not gl.enroll_face("what is the weather?")


class TestDedupe:
    """Test random choice deduplication."""

    def _dedupe(self, current, last, options):
        import random
        while current == last:
            current = random.choice(options)
        return current

    def test_different_from_last(self):
        result = self._dedupe("a", "a", ["a", "b", "c"])
        assert result != "a"

    def test_last_is_none(self):
        result = self._dedupe("hello", None, ["hello", "world"])
        assert result == "hello"

    def test_returns_string(self):
        result = self._dedupe("x", "y", ["x", "y", "z"])
        assert isinstance(result, str)
