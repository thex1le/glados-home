"""Tests for SceneDescriber.

Mirrors the test_glados_brain pattern: bypass __init__ via __new__() so we
never open RtspConsumer, FastVLM, or MQTT. Drive each method directly with
manually populated state.
"""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from glados_modules.GladosEnums import (MQTTEnums, SceneEnums)
from glados_modules.MqttConnector import MQTTClient


@pytest.fixture
def fake_describer():
    """SceneDescriber with __init__ bypassed; all bridge state pre-populated."""
    from glados_modules.SceneDescriber import SceneDescriber
    with patch.object(MQTTClient, '__init__', lambda self, *a, **kw: None):
        d = SceneDescriber.__new__(SceneDescriber)
    d.__name__ = "SceneDescriber"
    d.logger = MagicMock()
    d.configFile = MagicMock()
    d.camera_name = "head"
    d.stop = False
    d._poll_interval = 5.0
    d._scene_change_threshold = 0.05
    d._rtsp_uri = "rtsp://127.0.0.1:8554/camera_head"
    d._consumer = MagicMock()
    d._model = MagicMock()
    d._inference_lock = threading.Lock()
    d._last_frame = None
    d.send_command = MagicMock()
    return d


def _make_msg(payload):
    msg = MagicMock()
    msg.payload.decode.return_value = json.dumps(payload)
    return msg


def _frame(value: int = 0, shape=(64, 64, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


class TestSceneChangeScore:
    """Pixel diff used to gate background inference."""

    def test_first_frame_returns_one(self, fake_describer):
        assert fake_describer._scene_change_score(_frame(0)) == 1.0

    def test_identical_frames_return_zero(self, fake_describer):
        frame = _frame(127)
        fake_describer._last_frame = frame.copy()
        assert fake_describer._scene_change_score(frame) == 0.0

    def test_different_shapes_return_one(self, fake_describer):
        fake_describer._last_frame = _frame(0, shape=(32, 32, 3))
        assert fake_describer._scene_change_score(_frame(0)) == 1.0

    def test_completely_different_returns_one(self, fake_describer):
        fake_describer._last_frame = _frame(0)
        assert fake_describer._scene_change_score(_frame(255)) == 1.0

    def test_partial_change_in_range(self, fake_describer):
        fake_describer._last_frame = _frame(100)
        score = fake_describer._scene_change_score(_frame(110))
        assert 0.0 < score < 1.0
        # 10/255 ≈ 0.039
        assert score == pytest.approx(10 / 255, rel=1e-3)


class TestDescribe:
    """_describe runs FastVLM under a lock and swallows errors."""

    def test_returns_model_output(self, fake_describer):
        fake_describer._model.encode_image.return_value = "features"
        fake_describer._model.describe_from_features.return_value = "a chair"
        result = fake_describer._describe(_frame(0), "what?", 64)
        assert result == "a chair"
        fake_describer._model.encode_image.assert_called_once()
        fake_describer._model.describe_from_features.assert_called_once_with(
            "features", prompt="what?", max_tokens=64)

    def test_inference_error_returns_none(self, fake_describer):
        fake_describer._model.encode_image.side_effect = RuntimeError("CUDA OOM")
        assert fake_describer._describe(_frame(0), "what?", 64) is None
        fake_describer.logger.error.assert_called_once()

    def test_describe_holds_inference_lock_during_call(self, fake_describer):
        # Confirm the lock is acquired during inference (call from inside the
        # mocked encode_image must observe a locked lock).
        observed = {}

        def encode_side_effect(frame):
            observed["locked_during"] = fake_describer._inference_lock.locked()
            return "features"

        fake_describer._model.encode_image.side_effect = encode_side_effect
        fake_describer._model.describe_from_features.return_value = "x"
        fake_describer._describe(_frame(0), "p", 32)
        assert observed["locked_during"] is True


class TestPublishDescription:
    """Background publishes use the correct topic and message shape."""

    def test_publishes_to_scene_description_topic(self, fake_describer):
        fake_describer._publish_description("a desk")
        msg, topic = fake_describer.send_command.call_args[0]
        assert topic == MQTTEnums.SCENE_DESCRIPTION_TOPIC.value
        assert msg[SceneEnums.DESCRIPTION_KEY.value] == "a desk"
        assert msg[SceneEnums.CAMERA_KEY.value] == "head"

    def test_publish_uses_current_timestamp(self, fake_describer):
        before = time.time()
        fake_describer._publish_description("now")
        msg, _ = fake_describer.send_command.call_args[0]
        assert msg[SceneEnums.TS_KEY.value] >= before


class TestHandleRequest:
    """_handle_request validates the payload and dispatches to a worker thread."""

    def test_missing_request_id_logs_warning(self, fake_describer):
        msg = _make_msg({SceneEnums.PROMPT_KEY.value: "look"})
        fake_describer._handle_request(msg)
        fake_describer.logger.warning.assert_called_once()

    def test_malformed_payload_logs_error(self, fake_describer):
        msg = MagicMock()
        msg.payload.decode.return_value = "not json"
        fake_describer._handle_request(msg)
        fake_describer.logger.error.assert_called_once()

    def test_dispatches_to_worker_thread(self, fake_describer):
        # Patch Thread inside the SceneDescriber module so we can capture the call
        with patch("glados_modules.SceneDescriber.Thread") as ThreadMock:
            msg = _make_msg({SceneEnums.REQUEST_ID_KEY.value: "req-1",
                             SceneEnums.PROMPT_KEY.value: "look",
                             SceneEnums.MAX_TOKENS_KEY.value: 100})
            fake_describer._handle_request(msg)
            ThreadMock.assert_called_once()
            kwargs = ThreadMock.call_args.kwargs
            assert kwargs["target"] == fake_describer._handle_request_worker
            assert kwargs["args"] == ("req-1", "look", 100)
            ThreadMock.return_value.start.assert_called_once()


class TestHandleRequestWorker:
    """The worker captures a frame, runs inference, publishes describe_response."""

    def test_publishes_response_with_request_id(self, fake_describer):
        fake_describer._consumer.get_frame.return_value = _frame(50)
        fake_describer._model.encode_image.return_value = "f"
        fake_describer._model.describe_from_features.return_value = "a chair"
        fake_describer._handle_request_worker("req-1", "look", 64)
        msg, topic = fake_describer.send_command.call_args[0]
        assert topic == MQTTEnums.SCENE_DESCRIBE_RESPONSE_TOPIC.value
        assert msg[SceneEnums.REQUEST_ID_KEY.value] == "req-1"
        assert msg[SceneEnums.DESCRIPTION_KEY.value] == "a chair"

    def test_no_frame_publishes_error(self, fake_describer):
        fake_describer._consumer.get_frame.return_value = None
        fake_describer._handle_request_worker("req-1", "look", 64)
        msg, _ = fake_describer.send_command.call_args[0]
        assert "error" in msg[SceneEnums.DESCRIPTION_KEY.value].lower()
        assert msg[SceneEnums.REQUEST_ID_KEY.value] == "req-1"

    def test_inference_failure_publishes_error(self, fake_describer):
        fake_describer._consumer.get_frame.return_value = _frame(50)
        fake_describer._model.encode_image.side_effect = RuntimeError("boom")
        fake_describer._handle_request_worker("req-1", "look", 64)
        msg, _ = fake_describer.send_command.call_args[0]
        assert "error" in msg[SceneEnums.DESCRIPTION_KEY.value].lower()


class TestTopicHandlerWiring:
    """SceneDescriber must subscribe to the request topic so the brain can reach it."""

    def test_topic_handler_includes_request_topic(self, fake_describer):
        fake_describer.topic_handler = {
            MQTTEnums.SCENE_DESCRIBE_REQUEST_TOPIC.value:
                fake_describer._handle_request,
        }
        assert MQTTEnums.SCENE_DESCRIBE_REQUEST_TOPIC.value in \
               fake_describer.topic_handler


class TestRunLoop:
    """The main loop publishes only when scene change exceeds threshold."""

    def test_first_iteration_publishes(self, fake_describer):
        fake_describer._consumer.get_frame.return_value = _frame(100)
        fake_describer._model.encode_image.return_value = "f"
        fake_describer._model.describe_from_features.return_value = "first"
        fake_describer._poll_interval = 0  # don't sleep

        # Stop after one iteration
        call_count = {"n": 0}
        original_get_frame = fake_describer._consumer.get_frame

        def stop_after_one():
            call_count["n"] += 1
            if call_count["n"] > 1:
                fake_describer.stop = True
            return original_get_frame.return_value

        fake_describer._consumer.get_frame.side_effect = stop_after_one
        fake_describer.run()
        assert fake_describer.send_command.called
        msg, topic = fake_describer.send_command.call_args[0]
        assert topic == MQTTEnums.SCENE_DESCRIPTION_TOPIC.value
        assert msg[SceneEnums.DESCRIPTION_KEY.value] == "first"

    def test_unchanged_scene_skips_inference(self, fake_describer):
        fake_describer._consumer.get_frame.return_value = _frame(100)
        fake_describer._last_frame = _frame(100)  # identical -> score 0.0
        fake_describer._poll_interval = 0

        # Stop after one iteration
        call_count = {"n": 0}
        original = fake_describer._consumer.get_frame

        def stop_after_one():
            call_count["n"] += 1
            if call_count["n"] > 1:
                fake_describer.stop = True
            return original.return_value

        fake_describer._consumer.get_frame.side_effect = stop_after_one
        fake_describer.run()
        # No FastVLM call, no publish
        fake_describer._model.encode_image.assert_not_called()
        fake_describer.send_command.assert_not_called()

    def test_no_frame_skips_iteration(self, fake_describer):
        fake_describer._consumer.get_frame.return_value = None
        fake_describer._poll_interval = 0

        call_count = {"n": 0}

        def stop_after_one():
            call_count["n"] += 1
            if call_count["n"] > 1:
                fake_describer.stop = True
            return None

        fake_describer._consumer.get_frame.side_effect = stop_after_one
        fake_describer.run()
        fake_describer._model.encode_image.assert_not_called()
        fake_describer.send_command.assert_not_called()
