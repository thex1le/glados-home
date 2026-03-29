"""Tests for HealthMonitor thread tracking, error counting, and peer detection."""

import pytest
import time
from unittest.mock import MagicMock, patch
from threading import Thread
from json import dumps


class TestHealthMonitorRegistration:
    def _make_monitor(self):
        from glados_modules.HealthMonitor import HealthMonitor
        with patch.object(HealthMonitor, '__init__', lambda self, *a, **kw: None):
            hm = HealthMonitor.__new__(HealthMonitor)
        hm._threads = {}
        hm._error_counts = {}
        hm._health_lock = __import__('threading').Lock()
        hm._peer_last_seen = {}
        hm._peer_status = {}
        hm._peer_warned = {}
        hm._start_time = time.time()
        hm.system_name = "test"
        hm.hostname = "test-host"
        hm.logger = MagicMock()
        hm.client = MagicMock()
        hm.client.is_connected.return_value = True
        return hm

    def test_register_thread(self):
        hm = self._make_monitor()
        t = MagicMock(spec=Thread)
        hm.register("test_thread", t)
        assert "test_thread" in hm._threads
        assert hm._error_counts["test_thread"] == 0

    def test_register_multiple(self):
        hm = self._make_monitor()
        for name in ["a", "b", "c"]:
            hm.register(name, MagicMock(spec=Thread))
        assert len(hm._threads) == 3

    def test_report_error_increments(self):
        hm = self._make_monitor()
        hm.register("cam", MagicMock(spec=Thread))
        hm.report_error("cam")
        hm.report_error("cam")
        assert hm._error_counts["cam"] == 2

    def test_report_error_unknown_ignored(self):
        hm = self._make_monitor()
        hm.report_error("nonexistent")  # should not crash


class TestHealthMonitorStatus:
    def _make_monitor(self):
        from glados_modules.HealthMonitor import HealthMonitor
        with patch.object(HealthMonitor, '__init__', lambda self, *a, **kw: None):
            hm = HealthMonitor.__new__(HealthMonitor)
        hm._threads = {}
        hm._error_counts = {}
        hm._health_lock = __import__('threading').Lock()
        hm._start_time = time.time() - 100  # 100s uptime
        hm.system_name = "body_server"
        hm.hostname = "pi4"
        hm.logger = MagicMock()
        hm.client = MagicMock()
        hm.client.is_connected.return_value = True
        return hm

    def test_status_format(self):
        hm = self._make_monitor()
        alive_thread = MagicMock(spec=Thread)
        alive_thread.is_alive.return_value = True
        hm.register("servo", alive_thread)
        hm.report_error("servo")

        status = hm.get_status()
        assert status["hostname"] == "pi4"
        assert status["system"] == "body_server"
        assert status["uptime_s"] >= 99
        assert status["mqtt_connected"] is True
        assert "ts" in status
        assert status["threads"]["servo"]["alive"] is True
        assert status["threads"]["servo"]["errors"] == 1

    def test_dead_thread_reported(self):
        hm = self._make_monitor()
        dead_thread = MagicMock(spec=Thread)
        dead_thread.is_alive.return_value = False
        hm.register("dead_one", dead_thread)

        status = hm.get_status()
        assert status["threads"]["dead_one"]["alive"] is False


class TestHealthMonitorPeers:
    def _make_monitor(self):
        from glados_modules.HealthMonitor import HealthMonitor
        with patch.object(HealthMonitor, '__init__', lambda self, *a, **kw: None):
            hm = HealthMonitor.__new__(HealthMonitor)
        hm._threads = {}
        hm._error_counts = {}
        hm._health_lock = __import__('threading').Lock()
        hm._peer_last_seen = {}
        hm._peer_status = {}
        hm._peer_warned = {}
        hm._start_time = time.time()
        hm.system_name = "test"
        hm.hostname = "pi4"
        hm.logger = MagicMock()
        hm.PEER_TIMEOUT = 15.0
        return hm

    def test_peer_heartbeat_tracked(self):
        hm = self._make_monitor()
        msg = MagicMock()
        msg.payload.decode.return_value = dumps({
            "hostname": "gpu_server", "system": "ai_server", "uptime_s": 500
        })
        hm._handle_peer_heartbeat(msg)
        assert "gpu_server" in hm._peer_last_seen
        assert hm._peer_status["gpu_server"]["system"] == "ai_server"

    def test_own_heartbeat_ignored(self):
        hm = self._make_monitor()
        msg = MagicMock()
        msg.payload.decode.return_value = dumps({"hostname": "pi4"})
        hm._handle_peer_heartbeat(msg)
        assert "pi4" not in hm._peer_last_seen

    def test_peer_timeout_warns(self):
        hm = self._make_monitor()
        hm._peer_last_seen["gpu"] = time.time() - 20  # 20s ago
        hm._peer_warned["gpu"] = False
        hm._check_peers()
        assert hm.logger.warning.called
        assert hm._peer_warned["gpu"] is True

    def test_peer_within_timeout_no_warning(self):
        hm = self._make_monitor()
        hm._peer_last_seen["gpu"] = time.time() - 5  # 5s ago
        hm._peer_warned["gpu"] = False
        hm._check_peers()
        assert not hm.logger.warning.called

    def test_already_warned_no_repeat(self):
        hm = self._make_monitor()
        hm._peer_last_seen["gpu"] = time.time() - 30
        hm._peer_warned["gpu"] = True
        hm._check_peers()
        assert not hm.logger.warning.called

    def test_peer_comes_back_online(self):
        hm = self._make_monitor()
        hm._peer_warned["gpu"] = True
        msg = MagicMock()
        msg.payload.decode.return_value = dumps({"hostname": "gpu", "system": "ai"})
        hm._handle_peer_heartbeat(msg)
        assert hm._peer_warned["gpu"] is False
        assert hm.logger.info.called

    def test_check_threads_logs_dead(self):
        hm = self._make_monitor()
        dead = MagicMock(spec=Thread)
        dead.is_alive.return_value = False
        hm.register("doomed", dead)
        hm._check_threads()
        assert hm.logger.error.called
        assert "doomed" in hm.logger.error.call_args[0][0]
