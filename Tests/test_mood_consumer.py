"""Tests for MoodConsumer — last-write-wins PAD state cache.

Mirrors the test_glados_brain pattern: bypass MQTTClient.__init__ via __new__()
so the real broker never gets contacted. Drive _handle_pad directly.
"""

import json
import threading
import time
from configparser import ConfigParser
from unittest.mock import MagicMock, patch

import pytest

from glados_modules.MqttConnector import MQTTClient


@pytest.fixture
def mood_config():
    cp = ConfigParser()
    cp.read_string("""
[MQTT]
mqtt_server_ip = 127.0.0.1
mqtt_port = 1883

[BRAIN]
mood_staleness_max_age_s = 60.0

[EMOTION]
baseline_pleasure = 0.1
baseline_arousal = -0.1
baseline_dominance = 0.6
""")
    return cp


@pytest.fixture
def fake_consumer(mood_config):
    from glados_modules.MoodConsumer import MoodConsumer, PADState
    with patch.object(MQTTClient, '__init__', lambda self, *a, **kw: None):
        c = MoodConsumer.__new__(MoodConsumer)
    c.__name__ = "MoodConsumer"
    c.logger = MagicMock()
    c.configFile = mood_config
    c.stop = False
    c._staleness_max_age = 60.0
    c._pad = PADState(pleasure=0.1, arousal=-0.1, dominance=0.6)
    c._pad_ts = 0.0
    c._pad_lock = threading.Lock()
    return c


def _make_msg(payload):
    msg = MagicMock()
    msg.payload.decode.return_value = json.dumps(payload)
    return msg


class TestInitialState:
    """Before the first MQTT update arrives, get_pad returns the baseline."""

    def test_baseline_returned_before_first_update(self, fake_consumer):
        from glados_modules.MoodConsumer import PADState
        pad = fake_consumer.get_pad()
        assert isinstance(pad, PADState)
        assert pad.pleasure == pytest.approx(0.1)
        assert pad.arousal == pytest.approx(-0.1)
        assert pad.dominance == pytest.approx(0.6)

    def test_is_stale_before_first_update(self, fake_consumer):
        # No updates yet → always stale
        assert fake_consumer.is_stale() is True

    def test_age_is_none_before_first_update(self, fake_consumer):
        assert fake_consumer.get_age_s() is None


class TestPADHandling:
    """_handle_pad parses, validates, stores."""

    def test_valid_pad_updates_state(self, fake_consumer):
        msg = _make_msg({"pleasure": 0.5, "arousal": 0.3, "dominance": 0.8,
                         "ts": 12345.0})
        fake_consumer._handle_pad(msg)
        pad = fake_consumer.get_pad()
        assert pad.pleasure == pytest.approx(0.5)
        assert pad.arousal == pytest.approx(0.3)
        assert pad.dominance == pytest.approx(0.8)

    def test_ts_field_recorded(self, fake_consumer):
        msg = _make_msg({"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0,
                         "ts": 99999.0})
        fake_consumer._handle_pad(msg)
        with fake_consumer._pad_lock:
            assert fake_consumer._pad_ts == 99999.0

    def test_missing_ts_falls_back_to_now(self, fake_consumer):
        before = time.time()
        msg = _make_msg({"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0})
        fake_consumer._handle_pad(msg)
        with fake_consumer._pad_lock:
            assert fake_consumer._pad_ts >= before

    def test_malformed_json_logged_not_raised(self, fake_consumer):
        msg = MagicMock()
        msg.payload.decode.return_value = "not json"
        fake_consumer._handle_pad(msg)
        fake_consumer.logger.error.assert_called_once()
        # Baseline state preserved
        assert fake_consumer.get_pad().pleasure == pytest.approx(0.1)

    def test_missing_pad_field_logged_not_raised(self, fake_consumer):
        msg = _make_msg({"pleasure": 0.5, "arousal": 0.3})  # missing dominance
        fake_consumer._handle_pad(msg)
        fake_consumer.logger.error.assert_called_once()
        # Baseline state preserved
        assert fake_consumer.get_pad().pleasure == pytest.approx(0.1)

    def test_non_numeric_pad_value_logged(self, fake_consumer):
        msg = _make_msg({"pleasure": "not a float", "arousal": 0,
                         "dominance": 0})
        fake_consumer._handle_pad(msg)
        fake_consumer.logger.error.assert_called_once()


class TestStaleness:
    """is_stale + get_age_s react to update timestamps."""

    def test_fresh_update_not_stale(self, fake_consumer):
        msg = _make_msg({"pleasure": 0, "arousal": 0, "dominance": 0,
                         "ts": time.time()})
        fake_consumer._handle_pad(msg)
        assert fake_consumer.is_stale() is False

    def test_old_update_is_stale(self, fake_consumer):
        msg = _make_msg({"pleasure": 0, "arousal": 0, "dominance": 0,
                         "ts": time.time() - 120})
        fake_consumer._handle_pad(msg)
        assert fake_consumer.is_stale() is True

    def test_custom_max_age_overrides(self, fake_consumer):
        msg = _make_msg({"pleasure": 0, "arousal": 0, "dominance": 0,
                         "ts": time.time() - 5})
        fake_consumer._handle_pad(msg)
        assert fake_consumer.is_stale(max_age_s=10.0) is False
        assert fake_consumer.is_stale(max_age_s=1.0) is True

    def test_get_age_s_returns_seconds_since_update(self, fake_consumer):
        msg = _make_msg({"pleasure": 0, "arousal": 0, "dominance": 0,
                         "ts": time.time() - 7})
        fake_consumer._handle_pad(msg)
        age = fake_consumer.get_age_s()
        assert age is not None
        assert 6.5 < age < 7.5


class TestThreadSafety:
    """Concurrent reads + writes don't corrupt state."""

    def test_concurrent_handle_and_get_no_crash(self, fake_consumer):
        # Simple smoke: hammer both for a moment, ensure no exception
        from glados_modules.MoodConsumer import PADState
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                msg = _make_msg({"pleasure": (i % 10) / 10,
                                 "arousal": 0.0, "dominance": 0.0,
                                 "ts": time.time()})
                fake_consumer._handle_pad(msg)
                i += 1

        def reader():
            while not stop.is_set():
                pad = fake_consumer.get_pad()
                assert isinstance(pad, PADState)

        threads = [threading.Thread(target=writer, daemon=True),
                   threading.Thread(target=reader, daemon=True),
                   threading.Thread(target=reader, daemon=True)]
        for t in threads:
            t.start()
        time.sleep(0.05)
        stop.set()
        for t in threads:
            t.join(timeout=1.0)
            assert not t.is_alive()
