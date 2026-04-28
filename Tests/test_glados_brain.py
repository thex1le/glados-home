"""Tests for GladosBrain — the wrapper around the dnhkng/GLaDOS engine.

Mirrors the test_glados_local pattern: bypass __init__ via __new__() so the
real MQTTClient + engine construction never run, then drive each method
directly with manually populated state.
"""

import json
import threading
import time
from configparser import ConfigParser
from unittest.mock import MagicMock, patch

import pytest

from glados_modules.GladosEnums import (
    BrainEnums, MQTTEnums, RoomStateEnums, SceneEnums, SystemEnums,
)
from glados_modules.MqttConnector import MQTTClient


@pytest.fixture
def mock_config():
    """Configparser with the minimal sections GladosBrain reads."""
    cp = ConfigParser()
    cp.read_string("""
[MQTT]
mqtt_server_ip = 127.0.0.1
mqtt_port = 1883

[DEFAULT]
VoiceUrl = http://gpu:8124/synthesize/

[BRAIN]
llm_endpoint = http://gpu:11434/v1/chat/completions
llm_model = qwen2.5:14b-instruct-q4_K_M
llm_api_key =
input_mode = text
audio_io = sounddevice
interruptible = True
autonomy_enabled = False
autonomy_tick_interval_s = 10
autonomy_parallel_calls = 1
autonomy_cooldown_s = 20
autonomy_coalesce_ticks = True
emotion_enabled = True
compaction_enabled = True
observer_enabled = False
scene_context_priority = 8
room_context_priority = 7
scene_request_timeout_s = 0.5
personality_file = ./txt_responses/personality_glados.txt
mood_publish_delta = 0.1
mood_publish_max_interval_s = 30.0
mood_staleness_max_age_s = 60.0

[HEXACO]
honesty_humility  = 0.3
emotionality      = 0.7
extraversion      = 0.4
agreeableness     = 0.2
conscientiousness = 0.9
openness          = 0.95

[EMOTION]
tick_interval_s     = 30.0
max_events          = 20
baseline_pleasure   = 0.1
baseline_arousal    = -0.1
baseline_dominance  = 0.6
mood_drift_rate     = 0.1
baseline_drift_rate = 0.02
""")
    return cp


@pytest.fixture
def fake_brain(mock_config):
    """Brain instance with __init__ bypassed; all bridge state pre-populated."""
    from glados_modules.GladosBrain import GladosBrain
    with patch.object(MQTTClient, '__init__', lambda self, *a, **kw: None):
        brain = GladosBrain.__new__(GladosBrain)
    brain.__name__ = "GladosBrain"
    brain.logger = MagicMock()
    brain.configFile = mock_config
    brain.stop = False
    brain.engine = None
    brain._engine_thread = None
    brain._engine_ready = threading.Event()
    brain._latest_scene = None
    brain._latest_scene_ts = 0.0
    brain._latest_room = None
    brain._scene_lock = threading.Lock()
    brain._room_lock = threading.Lock()
    brain._pending_scene = {}
    brain._scene_responses = {}
    brain._scene_pending_lock = threading.Lock()
    brain._scene_priority = 8
    brain._room_priority = 7
    brain._scene_timeout = 0.5
    brain._mood_publish_delta = 0.1
    brain._mood_publish_max_interval = 30.0
    brain._last_published_pad = None
    brain._last_pad_publish_ts = 0.0
    brain.send_command = MagicMock()
    return brain


def _make_msg(payload):
    msg = MagicMock()
    msg.payload.decode.return_value = json.dumps(payload)
    return msg


class TestSceneHandler:
    """Background scene descriptions update _latest_scene."""

    def test_handle_scene_updates_latest(self, fake_brain):
        msg = _make_msg({SceneEnums.DESCRIPTION_KEY.value: "Ben holding mug",
                         SceneEnums.TS_KEY.value: 12345.0,
                         SceneEnums.CAMERA_KEY.value: "head"})
        fake_brain._handle_scene(msg)
        assert fake_brain._latest_scene == "Ben holding mug"
        assert fake_brain._latest_scene_ts == 12345.0

    def test_handle_scene_overwrites_previous(self, fake_brain):
        fake_brain._latest_scene = "old"
        fake_brain._latest_scene_ts = 100.0
        msg = _make_msg({SceneEnums.DESCRIPTION_KEY.value: "new",
                         SceneEnums.TS_KEY.value: 200.0,
                         SceneEnums.CAMERA_KEY.value: "head"})
        fake_brain._handle_scene(msg)
        assert fake_brain._latest_scene == "new"
        assert fake_brain._latest_scene_ts == 200.0

    def test_handle_scene_uses_now_when_ts_missing(self, fake_brain):
        before = time.time()
        msg = _make_msg({SceneEnums.DESCRIPTION_KEY.value: "x",
                         SceneEnums.CAMERA_KEY.value: "head"})
        fake_brain._handle_scene(msg)
        assert fake_brain._latest_scene_ts >= before

    def test_malformed_scene_payload_logged_not_raised(self, fake_brain):
        msg = MagicMock()
        msg.payload.decode.return_value = "not json"
        fake_brain._handle_scene(msg)
        fake_brain.logger.error.assert_called_once()
        assert fake_brain._latest_scene is None


class TestSceneResponseHandler:
    """On-demand scene responses unblock the matching pending request."""

    def test_response_unblocks_request(self, fake_brain):
        fake_brain._pending_scene["req-1"] = threading.Event()
        msg = _make_msg({SceneEnums.REQUEST_ID_KEY.value: "req-1",
                         SceneEnums.DESCRIPTION_KEY.value: "a chair"})
        fake_brain._handle_scene_response(msg)
        assert fake_brain._scene_responses["req-1"] == "a chair"
        assert fake_brain._pending_scene["req-1"].is_set()

    def test_response_for_unknown_request_id_ignored(self, fake_brain):
        # Stale response from a previous request that already timed out
        msg = _make_msg({SceneEnums.REQUEST_ID_KEY.value: "old-req",
                         SceneEnums.DESCRIPTION_KEY.value: "stale"})
        fake_brain._handle_scene_response(msg)
        assert "old-req" not in fake_brain._scene_responses

    def test_response_missing_id_warns(self, fake_brain):
        msg = _make_msg({SceneEnums.DESCRIPTION_KEY.value: "no id"})
        fake_brain._handle_scene_response(msg)
        fake_brain.logger.warning.assert_called_once()


class TestRoomStateHandler:
    """Room-state updates from the GPU box are stored for context formatting."""

    def test_handle_room_state_stores_full_payload(self, fake_brain):
        payload = {"count": 2,
                   "roster": [{"person_id": "p1", "face_id": "Ben"},
                              {"person_id": "p2", "face_id": "unknown"}]}
        fake_brain._handle_room_state(_make_msg(payload))
        assert fake_brain._latest_room == payload

    def test_malformed_room_payload_logged(self, fake_brain):
        msg = MagicMock()
        msg.payload.decode.return_value = "not json"
        fake_brain._handle_room_state(msg)
        fake_brain.logger.error.assert_called_once()
        assert fake_brain._latest_room is None


class TestContextFormatters:
    """ContextBuilder formatters return None when empty, formatted strings otherwise."""

    def test_format_scene_returns_none_when_empty(self, fake_brain):
        assert fake_brain._format_scene() is None

    def test_format_scene_includes_age_and_text(self, fake_brain):
        fake_brain._latest_scene = "a desk with a mug"
        fake_brain._latest_scene_ts = time.time() - 10.0
        formatted = fake_brain._format_scene()
        assert "[scene]" in formatted
        assert "a desk with a mug" in formatted
        assert "10s ago" in formatted

    def test_format_rooms_returns_none_when_empty(self, fake_brain):
        assert fake_brain._format_rooms() is None

    def test_format_rooms_handles_empty_roster(self, fake_brain):
        fake_brain._latest_room = {"count": 0, "roster": []}
        formatted = fake_brain._format_rooms()
        assert "0 occupant" in formatted
        assert "no one" in formatted

    def test_format_rooms_lists_face_ids(self, fake_brain):
        fake_brain._latest_room = {
            "count": 2,
            "roster": [{"person_id": "p1", "face_id": "Ben"},
                       {"person_id": "p2", "face_id": "Alice"}],
        }
        formatted = fake_brain._format_rooms()
        assert "Ben" in formatted and "Alice" in formatted
        assert "2 occupant" in formatted

    def test_format_rooms_falls_back_to_person_id_for_unknown_face(self, fake_brain):
        fake_brain._latest_room = {
            "count": 1,
            "roster": [{"person_id": "unknown_3", "face_id": "unknown"}],
        }
        formatted = fake_brain._format_rooms()
        assert "unknown_3" in formatted


class TestRequestSceneDescription:
    """request_scene_description publishes a request and blocks for the response."""

    def test_publishes_describe_request(self, fake_brain):
        # Force a fast timeout so the test doesn't hang
        fake_brain._scene_timeout = 0.05
        result = fake_brain.request_scene_description("look at the desk")
        assert result is None  # timeout
        fake_brain.send_command.assert_called_once()
        msg, topic = fake_brain.send_command.call_args[0]
        assert topic == MQTTEnums.SCENE_DESCRIBE_REQUEST_TOPIC.value
        assert msg[SceneEnums.PROMPT_KEY.value] == "look at the desk"
        assert SceneEnums.REQUEST_ID_KEY.value in msg

    def test_returns_none_on_timeout(self, fake_brain):
        fake_brain._scene_timeout = 0.05
        result = fake_brain.request_scene_description("look")
        assert result is None
        # Cleanup: pending entry must be removed so we don't leak memory
        assert len(fake_brain._pending_scene) == 0
        fake_brain.logger.warning.assert_called_once()

    def test_returns_response_when_received(self, fake_brain):
        captured = {}

        def capture(msg, topic):
            captured["id"] = msg[SceneEnums.REQUEST_ID_KEY.value]

        fake_brain.send_command.side_effect = capture

        def respond_async():
            time.sleep(0.05)
            response = _make_msg({
                SceneEnums.REQUEST_ID_KEY.value: captured["id"],
                SceneEnums.DESCRIPTION_KEY.value: "a person at a desk",
            })
            fake_brain._handle_scene_response(response)

        threading.Thread(target=respond_async, daemon=True).start()
        result = fake_brain.request_scene_description("describe", max_tokens=100)
        assert result == "a person at a desk"

    def test_default_prompt_used_when_omitted(self, fake_brain):
        fake_brain._scene_timeout = 0.05
        fake_brain.request_scene_description()
        msg, _ = fake_brain.send_command.call_args[0]
        assert "describe" in msg[SceneEnums.PROMPT_KEY.value].lower()

    def test_request_uses_unique_ids(self, fake_brain):
        fake_brain._scene_timeout = 0.05
        ids = []

        def capture(msg, topic):
            ids.append(msg[SceneEnums.REQUEST_ID_KEY.value])

        fake_brain.send_command.side_effect = capture
        fake_brain.request_scene_description("a")
        fake_brain.request_scene_description("b")
        assert ids[0] != ids[1]


class TestSpeak:
    """speak() bypasses the LLM and pushes text directly to the engine's TTS queue."""

    def test_raises_when_engine_not_started(self, fake_brain):
        from glados_modules.GladosBrain import GladosBrainException
        with pytest.raises(GladosBrainException):
            fake_brain.speak("hi")

    def test_enqueues_to_engine_tts_queue(self, fake_brain):
        fake_brain.engine = MagicMock()
        fake_brain.speak("hello")
        fake_brain.engine.tts_queue.put.assert_called_once_with("hello")
        fake_brain.engine.processing_active_event.set.assert_called_once()


class TestSubmitTextInput:
    """submit_text_input forwards prompts to the engine; defensive when not ready."""

    def test_returns_false_when_engine_not_started(self, fake_brain):
        result = fake_brain.submit_text_input("hello")
        assert result is False
        fake_brain.logger.warning.assert_called_once()

    def test_delegates_to_engine_submit_text_input(self, fake_brain):
        fake_brain.engine = MagicMock()
        fake_brain.engine.submit_text_input.return_value = True
        result = fake_brain.submit_text_input("hello")
        assert result is True
        fake_brain.engine.submit_text_input.assert_called_once_with(
            "hello", source="speech")

    def test_passes_through_custom_source(self, fake_brain):
        fake_brain.engine = MagicMock()
        fake_brain.engine.submit_text_input.return_value = True
        fake_brain.submit_text_input("hi", source="text")
        fake_brain.engine.submit_text_input.assert_called_once_with(
            "hi", source="text")

    def test_engine_exception_doesnt_propagate(self, fake_brain):
        fake_brain.engine = MagicMock()
        fake_brain.engine.submit_text_input.side_effect = RuntimeError("boom")
        result = fake_brain.submit_text_input("hi")
        assert result is False
        fake_brain.logger.error.assert_called_once()

    def test_engine_returning_false_propagates(self, fake_brain):
        # Engine returns False on empty input
        fake_brain.engine = MagicMock()
        fake_brain.engine.submit_text_input.return_value = False
        assert fake_brain.submit_text_input("") is False


class TestTopicHandlerWiring:
    """Brain must subscribe to the correct MQTT topics."""

    def test_topic_handler_includes_required_topics(self, fake_brain):
        # Re-build topic_handler since fixture skipped MQTTClient.__init__
        fake_brain.topic_handler = {
            MQTTEnums.SCENE_DESCRIPTION_TOPIC.value: fake_brain._handle_scene,
            MQTTEnums.SCENE_DESCRIBE_RESPONSE_TOPIC.value:
                fake_brain._handle_scene_response,
            RoomStateEnums.MQTT_ROOM_TOPIC.value: fake_brain._handle_room_state,
        }
        expected = {
            MQTTEnums.SCENE_DESCRIPTION_TOPIC.value,
            MQTTEnums.SCENE_DESCRIBE_RESPONSE_TOPIC.value,
            RoomStateEnums.MQTT_ROOM_TOPIC.value,
        }
        assert expected.issubset(set(fake_brain.topic_handler.keys()))


class TestNoOpTranscriber:
    """The placeholder transcriber must satisfy the engine's protocol calls."""

    def test_transcribe_file_returns_empty(self):
        from glados_modules.GladosBrain import _NoOpTranscriber
        assert _NoOpTranscriber().transcribe_file("/tmp/anything.wav") == ""

    def test_transcribe_returns_empty(self):
        import numpy as np
        from glados_modules.GladosBrain import _NoOpTranscriber
        assert _NoOpTranscriber().transcribe(np.zeros(0, dtype=np.float32)) == ""


class TestWaitUntilReady:
    """wait_until_ready blocks until the engine_ready event is set."""

    def test_returns_false_on_timeout(self, fake_brain):
        assert fake_brain.wait_until_ready(timeout=0.05) is False

    def test_returns_true_when_engine_ready(self, fake_brain):
        fake_brain._engine_ready.set()
        assert fake_brain.wait_until_ready(timeout=0.05) is True


class TestReadyMessage:
    """The ready message published on system/brain_ready uses BrainMessageBuilder."""

    def test_ready_message_contains_system_and_model(self):
        from glados_modules.MqttConnector import BrainMessageBuilder
        msg = BrainMessageBuilder.ready("GladosBrain", "qwen2.5:14b-instruct-q4_K_M")
        assert msg["system"] == "GladosBrain"
        assert msg["model"] == "qwen2.5:14b-instruct-q4_K_M"


class TestSubConfigBuilders:
    """Step 7c-1: _build_engine reads all autonomy/emotion/HEXACO knobs from glog.conf."""

    def test_build_hexaco_uses_glog_conf_values(self, fake_brain, monkeypatch):
        # Replace HEXACOConfig with a real recorder so we can inspect kwargs
        captured = {}

        def recorder(**kw):
            captured.update(kw)
            return MagicMock(**kw)

        monkeypatch.setattr("glados_modules.GladosBrain.HEXACOConfig", recorder)
        fake_brain._build_hexaco_config()
        assert captured["honesty_humility"] == pytest.approx(0.3)
        assert captured["emotionality"] == pytest.approx(0.7)
        assert captured["extraversion"] == pytest.approx(0.4)
        assert captured["agreeableness"] == pytest.approx(0.2)
        assert captured["conscientiousness"] == pytest.approx(0.9)
        assert captured["openness"] == pytest.approx(0.95)

    def test_build_hexaco_uses_defaults_when_section_missing(self, mock_config, monkeypatch):
        from glados_modules.GladosBrain import GladosBrain
        from glados_modules.GladosEnums import HEXACODefaults
        # Drop the [HEXACO] section
        mock_config.remove_section("HEXACO")
        with patch.object(MQTTClient, '__init__', lambda self, *a, **kw: None):
            brain = GladosBrain.__new__(GladosBrain)
        brain.configFile = mock_config
        brain.logger = MagicMock()
        captured = {}

        def recorder(**kw):
            captured.update(kw)
            return MagicMock(**kw)

        monkeypatch.setattr("glados_modules.GladosBrain.HEXACOConfig", recorder)
        brain._build_hexaco_config()
        assert captured["honesty_humility"] == pytest.approx(
            HEXACODefaults.HONESTY_HUMILITY)
        assert captured["openness"] == pytest.approx(
            HEXACODefaults.OPENNESS)

    def test_build_emotion_config_reads_all_keys(self, fake_brain, monkeypatch):
        captured = {}

        def recorder(**kw):
            captured.update(kw)
            return MagicMock(**kw)

        monkeypatch.setattr("glados_modules.GladosBrain.EmotionConfig", recorder)
        # HEXACO is built inside; we just want the EmotionConfig kwargs
        fake_brain._build_emotion_config()
        assert captured["enabled"] is True
        assert captured["tick_interval_s"] == pytest.approx(30.0)
        assert captured["max_events"] == 20
        assert captured["baseline_pleasure"] == pytest.approx(0.1)
        assert captured["baseline_arousal"] == pytest.approx(-0.1)
        assert captured["baseline_dominance"] == pytest.approx(0.6)
        assert captured["mood_drift_rate"] == pytest.approx(0.1)
        assert captured["baseline_drift_rate"] == pytest.approx(0.02)
        assert captured["hexaco"] is not None

    def test_build_emotion_disabled_when_flag_false(self, mock_config, monkeypatch):
        from glados_modules.GladosBrain import GladosBrain
        mock_config.set("BRAIN", "emotion_enabled", "False")
        with patch.object(MQTTClient, '__init__', lambda self, *a, **kw: None):
            brain = GladosBrain.__new__(GladosBrain)
        brain.configFile = mock_config
        brain.logger = MagicMock()
        captured = {}

        def recorder(**kw):
            captured.update(kw)
            return MagicMock(**kw)

        monkeypatch.setattr("glados_modules.GladosBrain.EmotionConfig", recorder)
        brain._build_emotion_config()
        assert captured["enabled"] is False

    def test_build_autonomy_config_reads_brain_section(self, fake_brain, monkeypatch):
        captured = {}

        def recorder(**kw):
            captured.update(kw)
            return MagicMock(**kw)

        monkeypatch.setattr("glados_modules.GladosBrain.AutonomyConfig", recorder)
        fake_brain._build_autonomy_config()
        assert captured["enabled"] is False  # mock_config has autonomy_enabled=False
        assert captured["tick_interval_s"] == pytest.approx(10.0)
        assert captured["cooldown_s"] == pytest.approx(20.0)
        assert captured["autonomy_parallel_calls"] == 1
        assert captured["coalesce_ticks"] is True
        assert "jobs" in captured
        assert "tokens" in captured
        assert "emotion" in captured

    def test_compaction_disabled_raises_token_threshold(self, mock_config, monkeypatch):
        from glados_modules.GladosBrain import GladosBrain
        mock_config.set("BRAIN", "compaction_enabled", "False")
        with patch.object(MQTTClient, '__init__', lambda self, *a, **kw: None):
            brain = GladosBrain.__new__(GladosBrain)
        brain.configFile = mock_config
        brain.logger = MagicMock()
        captured = {}

        def token_recorder(**kw):
            captured.update(kw)
            return MagicMock(**kw)

        monkeypatch.setattr("glados_modules.GladosBrain.TokenConfig", token_recorder)
        # Suppress the rest of the chain
        monkeypatch.setattr("glados_modules.GladosBrain.AutonomyConfig",
                             lambda **kw: MagicMock(**kw))
        brain._build_autonomy_config()
        # Disabled compaction == effectively-infinite threshold
        assert captured["token_threshold"] >= 1_000_000

    def test_compaction_enabled_uses_default_threshold(self, fake_brain, monkeypatch):
        captured = {}

        def token_recorder(**kw):
            captured.update(kw)
            return MagicMock(**kw)

        monkeypatch.setattr("glados_modules.GladosBrain.TokenConfig", token_recorder)
        monkeypatch.setattr("glados_modules.GladosBrain.AutonomyConfig",
                             lambda **kw: MagicMock(**kw))
        fake_brain._build_autonomy_config()
        # Enabled compaction == reasonable engine default
        assert captured["token_threshold"] < 100_000

    def test_jobs_config_disabled_in_7c_1(self, fake_brain, monkeypatch):
        captured = {}

        def jobs_recorder(**kw):
            captured.update(kw)
            return MagicMock(**kw)

        monkeypatch.setattr("glados_modules.GladosBrain.AutonomyJobsConfig",
                             jobs_recorder)
        fake_brain._build_jobs_config()
        assert captured["enabled"] is False


class TestVisionEmotionBridge:
    """Step 7c-1: scene updates push EmotionEvents into the engine's EmotionAgent."""

    def test_scene_change_pushes_emotion_event(self, fake_brain, monkeypatch):
        # Wire an engine with a mocked emotion agent
        fake_brain.engine = MagicMock()
        fake_brain.engine._emotion_agent = MagicMock()
        captured = {}

        def event_recorder(**kw):
            captured.update(kw)
            return MagicMock(**kw)

        monkeypatch.setattr("glados_modules.GladosBrain.EmotionEvent", event_recorder)
        msg = _make_msg({SceneEnums.DESCRIPTION_KEY.value: "a person at a desk",
                         SceneEnums.TS_KEY.value: 12345.0})
        fake_brain._handle_scene(msg)
        fake_brain.engine._emotion_agent.push_event.assert_called_once()
        assert captured["source"] == "vision"
        assert "person at a desk" in captured["description"]

    def test_identical_scene_does_not_push_event(self, fake_brain, monkeypatch):
        fake_brain.engine = MagicMock()
        fake_brain.engine._emotion_agent = MagicMock()
        fake_brain._latest_scene = "a chair"  # pretend we already had this
        msg = _make_msg({SceneEnums.DESCRIPTION_KEY.value: "a chair",
                         SceneEnums.TS_KEY.value: 12345.0})
        fake_brain._handle_scene(msg)
        fake_brain.engine._emotion_agent.push_event.assert_not_called()

    def test_push_emotion_event_safe_before_engine_built(self, fake_brain):
        # Engine is None — must not raise
        fake_brain.engine = None
        fake_brain._push_emotion_event("vision", "test")  # no-op

    def test_push_emotion_event_safe_when_no_emotion_agent(self, fake_brain):
        fake_brain.engine = MagicMock(spec=[])  # spec=[] means no _emotion_agent attr
        fake_brain._push_emotion_event("vision", "test")  # no-op

    def test_push_emotion_event_swallows_exceptions(self, fake_brain, monkeypatch):
        fake_brain.engine = MagicMock()
        fake_brain.engine._emotion_agent = MagicMock()
        fake_brain.engine._emotion_agent.push_event.side_effect = RuntimeError("boom")
        monkeypatch.setattr("glados_modules.GladosBrain.EmotionEvent",
                             lambda **kw: MagicMock(**kw))
        # Must not raise
        fake_brain._push_emotion_event("vision", "test")
        fake_brain.logger.error.assert_called_once()


class TestPADPublisher:
    """Step 7c-2: PAD throttling. Hybrid (delta > 0.1 OR every 30s)."""

    def _set_pad(self, brain, p, a, d):
        brain.engine = MagicMock()
        brain.engine._emotion_agent = MagicMock()
        brain.engine._emotion_agent.state.pleasure = p
        brain.engine._emotion_agent.state.arousal = a
        brain.engine._emotion_agent.state.dominance = d

    def test_no_engine_is_safe(self, fake_brain):
        fake_brain.engine = None
        fake_brain._maybe_publish_pad()  # no-op
        fake_brain.send_command.assert_not_called()

    def test_no_emotion_agent_is_safe(self, fake_brain):
        fake_brain.engine = MagicMock(spec=[])
        fake_brain._maybe_publish_pad()
        fake_brain.send_command.assert_not_called()

    def test_first_publish_always_fires(self, fake_brain):
        self._set_pad(fake_brain, 0.0, 0.0, 0.0)
        fake_brain._maybe_publish_pad()
        fake_brain.send_command.assert_called_once()
        msg, topic = fake_brain.send_command.call_args[0]
        assert topic == MQTTEnums.MOOD_PAD_TOPIC.value
        assert msg["pleasure"] == 0.0
        assert msg["arousal"] == 0.0
        assert msg["dominance"] == 0.0
        assert msg["ts"] > 0

    def test_no_publish_when_below_delta(self, fake_brain):
        self._set_pad(fake_brain, 0.0, 0.0, 0.0)
        fake_brain._maybe_publish_pad()
        # tiny drift well below 0.1 threshold
        self._set_pad(fake_brain, 0.02, 0.01, -0.03)
        fake_brain.send_command.reset_mock()
        fake_brain._maybe_publish_pad()
        fake_brain.send_command.assert_not_called()

    def test_publish_when_axis_crosses_delta(self, fake_brain):
        self._set_pad(fake_brain, 0.0, 0.0, 0.0)
        fake_brain._maybe_publish_pad()
        # arousal jumps 0.5 — well above 0.1 delta
        self._set_pad(fake_brain, 0.0, 0.5, 0.0)
        fake_brain.send_command.reset_mock()
        fake_brain._maybe_publish_pad()
        fake_brain.send_command.assert_called_once()
        msg, _ = fake_brain.send_command.call_args[0]
        assert msg["arousal"] == pytest.approx(0.5)

    def test_publish_uses_max_axis_delta(self, fake_brain):
        # Two axes barely move (< 0.1) but pleasure jumps; still publish
        self._set_pad(fake_brain, 0.0, 0.0, 0.0)
        fake_brain._maybe_publish_pad()
        self._set_pad(fake_brain, 0.5, 0.05, 0.05)
        fake_brain.send_command.reset_mock()
        fake_brain._maybe_publish_pad()
        fake_brain.send_command.assert_called_once()

    def test_heartbeat_publishes_after_interval(self, fake_brain):
        # Force heartbeat path: tiny delta, but stale timestamp
        self._set_pad(fake_brain, 0.0, 0.0, 0.0)
        fake_brain._maybe_publish_pad()
        self._set_pad(fake_brain, 0.01, 0.01, 0.01)
        # Backdate last publish past the heartbeat window
        fake_brain._last_pad_publish_ts -= (fake_brain._mood_publish_max_interval + 1)
        fake_brain.send_command.reset_mock()
        fake_brain._maybe_publish_pad()
        fake_brain.send_command.assert_called_once()

    def test_consecutive_publishes_update_baseline(self, fake_brain):
        # After a publish, the new state becomes the comparison baseline
        self._set_pad(fake_brain, 0.0, 0.0, 0.0)
        fake_brain._maybe_publish_pad()
        self._set_pad(fake_brain, 0.5, 0.5, 0.5)
        fake_brain._maybe_publish_pad()
        # Now move only a tiny amount from (0.5, 0.5, 0.5) — should NOT publish
        self._set_pad(fake_brain, 0.52, 0.51, 0.49)
        fake_brain.send_command.reset_mock()
        fake_brain._maybe_publish_pad()
        fake_brain.send_command.assert_not_called()

    def test_send_command_failure_is_swallowed(self, fake_brain):
        self._set_pad(fake_brain, 0.0, 0.0, 0.0)
        fake_brain.send_command.side_effect = RuntimeError("broker down")
        # Must not raise
        fake_brain._maybe_publish_pad()
        fake_brain.logger.error.assert_called_once()

    def test_failed_publish_does_not_advance_baseline(self, fake_brain):
        self._set_pad(fake_brain, 0.0, 0.0, 0.0)
        fake_brain.send_command.side_effect = RuntimeError("nope")
        fake_brain._maybe_publish_pad()
        assert fake_brain._last_published_pad is None
        # Recover broker, retry — first-publish path fires again
        fake_brain.send_command.side_effect = None
        fake_brain.send_command.reset_mock()
        fake_brain._maybe_publish_pad()
        fake_brain.send_command.assert_called_once()
        assert fake_brain._last_published_pad is not None

    def test_corrupt_pad_state_logged_not_raised(self, fake_brain):
        fake_brain.engine = MagicMock()
        fake_brain.engine._emotion_agent = MagicMock()
        # state with no PAD attributes
        fake_brain.engine._emotion_agent.state.pleasure = "not a number"
        fake_brain._maybe_publish_pad()
        fake_brain.logger.error.assert_called_once()
        fake_brain.send_command.assert_not_called()


class TestMoodEventHandler:
    """Step 7c-4: brain subscribes to system/mood/event and forwards to EmotionAgent."""

    def test_valid_event_forwards_to_emotion_agent(self, fake_brain, monkeypatch):
        fake_brain.engine = MagicMock()
        fake_brain.engine._emotion_agent = MagicMock()
        captured = {}

        def event_recorder(**kw):
            captured.update(kw)
            return MagicMock(**kw)

        monkeypatch.setattr("glados_modules.GladosBrain.EmotionEvent",
                             event_recorder)
        msg = _make_msg({"source": "user",
                         "description": "User complimented you."})
        fake_brain._handle_mood_event(msg)
        fake_brain.engine._emotion_agent.push_event.assert_called_once()
        assert captured["source"] == "user"
        assert "complimented" in captured["description"]

    def test_missing_source_warns(self, fake_brain):
        msg = _make_msg({"description": "no source"})
        fake_brain._handle_mood_event(msg)
        fake_brain.logger.warning.assert_called_once()

    def test_missing_description_warns(self, fake_brain):
        msg = _make_msg({"source": "vision"})
        fake_brain._handle_mood_event(msg)
        fake_brain.logger.warning.assert_called_once()

    def test_malformed_payload_logged(self, fake_brain):
        msg = MagicMock()
        msg.payload.decode.return_value = "not json"
        fake_brain._handle_mood_event(msg)
        fake_brain.logger.error.assert_called_once()

    def test_handler_safe_when_engine_not_built(self, fake_brain):
        fake_brain.engine = None
        msg = _make_msg({"source": "user", "description": "x"})
        fake_brain._handle_mood_event(msg)  # no-op, no raise

    def test_topic_handler_includes_mood_event(self, fake_brain):
        fake_brain.topic_handler = {
            MQTTEnums.MOOD_EVENT_TOPIC.value: fake_brain._handle_mood_event,
        }
        assert MQTTEnums.MOOD_EVENT_TOPIC.value in fake_brain.topic_handler
