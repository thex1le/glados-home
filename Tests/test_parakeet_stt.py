"""Tests for ParakeetSTT (Step 8a-2).

Bypass MQTTClient.__init__ via __new__() so the real broker never gets
contacted. Drive process_audio directly with synthetic byte streams; mock
the transcriber via the conftest glados.ASR mock.
"""

import threading
from collections import namedtuple
from configparser import ConfigParser
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from glados_modules.GladosEnums import MQTTEnums, STTEnums
from glados_modules.MqttConnector import MQTTClient


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def asr_config():
    cp = ConfigParser()
    cp.read_string("""
[ASR]
engine_type = tdt
model_dir =
""")
    return cp


@pytest.fixture
def fake_parakeet(asr_config):
    """ParakeetSTT instance with __init__ bypassed; transcriber + broker mocked."""
    from glados_modules.ParakeetSTT import ParakeetSTT
    with patch.object(MQTTClient, '__init__', lambda self, *a, **kw: None):
        p = ParakeetSTT.__new__(ParakeetSTT)
    p.__name__ = "ParakeetSTT"
    p.logger = MagicMock()
    p.configFile = asr_config
    p.transcriber = MagicMock()
    p.transcriber.transcribe.return_value = "hello world"
    p.ffmpeg = MagicMock()
    p._inference_lock = threading.Lock()
    p.send_command = MagicMock()
    return p


def _stub_ffmpeg_process(fake_parakeet, return_int16: np.ndarray):
    """Configure fake ffmpeg to return the given int16 audio bytes."""
    process = MagicMock()
    process.communicate.return_value = (return_int16.tobytes(), b"")
    pipeline = MagicMock()
    pipeline.run_async.return_value = process
    fake_parakeet.ffmpeg.input.return_value.output.return_value = pipeline
    return process


# ----------------------------------------------------------------------
# load_audio_from_bytes — ffmpeg pipeline
# ----------------------------------------------------------------------

class TestLoadAudio:
    """ffmpeg piping decodes bytes into 16kHz float32."""

    def test_decodes_to_float32_normalized(self, fake_parakeet):
        # Synthetic int16 audio: peaks at full-scale, should normalize to ~1.0
        int16_audio = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        _stub_ffmpeg_process(fake_parakeet, int16_audio)
        out = fake_parakeet.load_audio_from_bytes(b"any input bytes")
        assert out.dtype == np.float32
        assert out.shape == (5,)
        # 32767 / 32768 ≈ 0.99997
        assert out.max() == pytest.approx(0.99997, abs=1e-4)
        # -32768 / 32768 = -1.0
        assert out.min() == pytest.approx(-1.0, abs=1e-4)

    def test_passes_correct_ffmpeg_args(self, fake_parakeet):
        _stub_ffmpeg_process(fake_parakeet, np.zeros(0, dtype=np.int16))
        fake_parakeet.load_audio_from_bytes(b"in")
        # Verify ffmpeg.input("pipe:0").output(... ar="16000" ...)
        fake_parakeet.ffmpeg.input.assert_called_with("pipe:0")
        output_kwargs = fake_parakeet.ffmpeg.input.return_value.output.call_args.kwargs
        assert output_kwargs["format"] == "wav"
        assert output_kwargs["ac"] == 1
        assert output_kwargs["ar"] == "16000"


# ----------------------------------------------------------------------
# process_audio — happy path + error handling
# ----------------------------------------------------------------------

class TestProcessAudio:
    """End-to-end: ffmpeg → transcriber → MQTT publish."""

    def test_publishes_to_stt_results_topic(self, fake_parakeet):
        _stub_ffmpeg_process(
            fake_parakeet,
            (np.random.uniform(-1, 1, 16000) * 16000).astype(np.int16))
        fake_parakeet.process_audio(b"audio bytes")
        msg, topic = fake_parakeet.send_command.call_args[0]
        assert topic == MQTTEnums.STT_RESULTS_MQTT_TOPIC.value

    def test_payload_shape_matches_legacy_LocalSTTtx(self, fake_parakeet):
        _stub_ffmpeg_process(
            fake_parakeet,
            (np.random.uniform(-1, 1, 16000) * 16000).astype(np.int16))
        fake_parakeet.transcriber.transcribe.return_value = "hello world"
        fake_parakeet.process_audio(b"audio bytes")
        msg, _ = fake_parakeet.send_command.call_args[0]
        # SttMessageBuilder wraps the dict under STT_RESULTS_KEY
        body = msg[STTEnums.STT_RESULTS_KEY.value]
        assert STTEnums.STT_TEXT_KEY.value in body
        assert STTEnums.STT_RAW_RESULTS_KEY.value in body
        assert STTEnums.STT_LANGUAGE_KEY.value in body
        assert STTEnums.STT_TIME_MAP_KEY.value in body
        assert body[STTEnums.STT_TEXT_KEY.value] == "hello world"
        assert body[STTEnums.STT_LANGUAGE_KEY.value] == STTEnums.STT_EN_LANG_KEY.value

    def test_text_stripped_of_whitespace(self, fake_parakeet):
        _stub_ffmpeg_process(
            fake_parakeet,
            (np.random.uniform(-1, 1, 16000) * 16000).astype(np.int16))
        fake_parakeet.transcriber.transcribe.return_value = "  hello world  \n"
        fake_parakeet.process_audio(b"audio bytes")
        msg, _ = fake_parakeet.send_command.call_args[0]
        body = msg[STTEnums.STT_RESULTS_KEY.value]
        assert body[STTEnums.STT_TEXT_KEY.value] == "hello world"

    def test_none_text_treated_as_empty(self, fake_parakeet):
        _stub_ffmpeg_process(
            fake_parakeet,
            (np.random.uniform(-1, 1, 16000) * 16000).astype(np.int16))
        fake_parakeet.transcriber.transcribe.return_value = None
        fake_parakeet.process_audio(b"audio bytes")
        msg, _ = fake_parakeet.send_command.call_args[0]
        body = msg[STTEnums.STT_RESULTS_KEY.value]
        assert body[STTEnums.STT_TEXT_KEY.value] == ""

    def test_time_map_is_empty_for_parakeet(self, fake_parakeet):
        # Parakeet doesn't expose word-level alignment — verify we publish []
        _stub_ffmpeg_process(
            fake_parakeet,
            (np.random.uniform(-1, 1, 16000) * 16000).astype(np.int16))
        fake_parakeet.process_audio(b"audio bytes")
        msg, _ = fake_parakeet.send_command.call_args[0]
        body = msg[STTEnums.STT_RESULTS_KEY.value]
        assert body[STTEnums.STT_TIME_MAP_KEY.value] == []

    def test_ffmpeg_decode_error_dropped(self, fake_parakeet):
        fake_parakeet.ffmpeg.input.side_effect = RuntimeError("ffmpeg died")
        fake_parakeet.process_audio(b"any")
        fake_parakeet.send_command.assert_not_called()
        fake_parakeet.logger.error.assert_called_once()

    def test_empty_audio_dropped(self, fake_parakeet):
        _stub_ffmpeg_process(fake_parakeet, np.zeros(0, dtype=np.int16))
        fake_parakeet.process_audio(b"empty")
        fake_parakeet.send_command.assert_not_called()
        fake_parakeet.logger.warning.assert_called_once()

    def test_transcriber_failure_dropped(self, fake_parakeet):
        _stub_ffmpeg_process(
            fake_parakeet,
            (np.random.uniform(-1, 1, 16000) * 16000).astype(np.int16))
        fake_parakeet.transcriber.transcribe.side_effect = RuntimeError("CUDA OOM")
        fake_parakeet.process_audio(b"audio bytes")
        fake_parakeet.send_command.assert_not_called()
        fake_parakeet.logger.error.assert_called_once()

    def test_publish_failure_swallowed(self, fake_parakeet):
        _stub_ffmpeg_process(
            fake_parakeet,
            (np.random.uniform(-1, 1, 16000) * 16000).astype(np.int16))
        fake_parakeet.send_command.side_effect = RuntimeError("broker down")
        # Must not raise — ASR backend can't crash AudioServerRX
        fake_parakeet.process_audio(b"audio bytes")
        fake_parakeet.logger.error.assert_called_once()


# ----------------------------------------------------------------------
# Inference locking
# ----------------------------------------------------------------------

class TestInferenceLock:
    """ONNX sessions aren't thread-safe — verify we serialize."""

    def test_transcribe_called_under_lock(self, fake_parakeet):
        observed = {}

        def transcribe_side_effect(audio):
            observed["locked_during"] = fake_parakeet._inference_lock.locked()
            return "x"

        fake_parakeet.transcriber.transcribe.side_effect = transcribe_side_effect
        _stub_ffmpeg_process(
            fake_parakeet,
            (np.random.uniform(-1, 1, 16000) * 16000).astype(np.int16))
        fake_parakeet.process_audio(b"audio bytes")
        assert observed["locked_during"] is True


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

class TestLoadModel:
    """_load_model wraps engine factory + handles failures."""

    def test_factory_failure_raises_descriptive_exception(self, fake_parakeet):
        from glados_modules.ParakeetSTT import ParakeetSTTException
        # Patch the engine factory to raise
        import sys
        sys.modules["glados.ASR"] = MagicMock(
            get_audio_transcriber=lambda engine_type: (_ for _ in ()).throw(
                RuntimeError("engine package missing")))
        try:
            with pytest.raises(ParakeetSTTException) as exc:
                fake_parakeet._load_model("tdt")
            assert "Failed to load Parakeet" in str(exc.value)
            assert "tdt" in str(exc.value)
        finally:
            sys.modules.pop("glados.ASR", None)

    def test_factory_success_stores_transcriber(self, fake_parakeet):
        import sys
        synth = MagicMock()
        sys.modules["glados.ASR"] = MagicMock(
            get_audio_transcriber=lambda engine_type: synth)
        try:
            fake_parakeet._load_model("tdt")
            assert fake_parakeet.transcriber is synth
        finally:
            sys.modules.pop("glados.ASR", None)
