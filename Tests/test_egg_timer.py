"""Tests for EggTimer start, stop, and completion logic."""

import pytest
import time
from unittest.mock import MagicMock, patch


class TestEggTimer:
    def _make_timer(self, duration=10):
        from glados_modules.EggTimer import EggTimer
        speak_mock = MagicMock()
        timer = EggTimer(duration, speak_mock)
        return timer, speak_mock

    def test_timer_start(self):
        timer, _ = self._make_timer(10)
        timer.timer_start()
        assert timer.is_running
        assert timer.start_time is not None

    def test_double_start_idempotent(self):
        timer, _ = self._make_timer(10)
        timer.timer_start()
        first_start = timer.start_time
        timer.timer_start()
        assert timer.start_time == first_start  # unchanged

    def test_stop_speaks_remaining(self):
        timer, speak = self._make_timer(10)
        timer.timer_start()
        time.sleep(0.05)
        timer.stop()
        assert not timer.is_running
        assert speak.called
        msg = speak.call_args[0][0]
        assert "Remaining time" in msg

    def test_stop_when_not_running(self):
        timer, speak = self._make_timer(10)
        timer.stop()  # not started
        assert not speak.called

    def test_check_remaining_not_complete(self):
        timer, _ = self._make_timer(100)  # long timer
        timer.timer_start()
        result = timer.check_remaining_time()
        assert result["complete"] is False
        assert result["remain"] > 0

    def test_check_remaining_complete(self):
        timer, speak = self._make_timer(0)  # zero duration
        timer.timer_start()
        time.sleep(0.01)
        result = timer.check_remaining_time()
        assert result["complete"] is True
        assert result["remain"] == 0
        assert speak.called
        assert "complete" in speak.call_args[0][0]

    def test_check_when_not_running(self):
        timer, _ = self._make_timer(10)
        result = timer.check_remaining_time()
        assert result["complete"] is True
        assert result["remain"] == 0
