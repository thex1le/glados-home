"""Tests for the RemoteGladosTTS adapter.

The adapter proxies the Pi 5 brain's TTS calls to the Flask :8124/synthesize/
server on the GPU box. We mock requests.get and inject a real soundfile (for
WAV encode/decode) via the wav_codec fixture to verify the full path without
hitting the network.
"""

import base64
import io

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def wav_codec():
    """Provide the real soundfile module.

    conftest.py mocks soundfile at import time so glados_modules can be loaded
    without the C library on dev hosts. For tests that exercise the real WAV
    encode/decode path, we evict the mock and import the real module fresh.
    Skip the test if soundfile isn't installed.
    """
    import importlib
    import sys
    saved = sys.modules.pop("soundfile", None)
    try:
        real = importlib.import_module("soundfile")
    except ImportError:
        if saved is not None:
            sys.modules["soundfile"] = saved
        pytest.skip("soundfile not installed")
    sys.modules["soundfile"] = real
    # Reload RemoteGladosTTS so its `import soundfile` binds to the real module
    if "glados_modules.RemoteGladosTTS" in sys.modules:
        importlib.reload(sys.modules["glados_modules.RemoteGladosTTS"])
    yield real
    if saved is not None:
        sys.modules["soundfile"] = saved
        if "glados_modules.RemoteGladosTTS" in sys.modules:
            importlib.reload(sys.modules["glados_modules.RemoteGladosTTS"])


@pytest.fixture
def silent_wav_bytes(wav_codec):
    """Return bytes of a 1-second silent WAV at 22050Hz mono float32."""
    buf = io.BytesIO()
    wav_codec.write(buf, np.zeros(22050, dtype=np.float32),
                    22050, format="WAV", subtype="FLOAT")
    return buf.getvalue()


@pytest.fixture
def stereo_wav_bytes(wav_codec):
    """Return bytes of a 1-second stereo WAV: left=0.5, right=-0.5."""
    samples = np.zeros((22050, 2), dtype=np.float32)
    samples[:, 0] = 0.5
    samples[:, 1] = -0.5
    buf = io.BytesIO()
    wav_codec.write(buf, samples, 22050, format="WAV", subtype="FLOAT")
    return buf.getvalue()


@pytest.fixture
def wav_44100_bytes(wav_codec):
    """Return bytes of a WAV at 44100Hz to provoke a sample-rate mismatch warning."""
    buf = io.BytesIO()
    wav_codec.write(buf, np.zeros(44100, dtype=np.float32),
                    44100, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _make_tts():
    from glados_modules.RemoteGladosTTS import RemoteGladosTTS
    return RemoteGladosTTS(voice_url="http://gpu:8124/synthesize/")


class TestSpeechSynthesizerProtocol:
    """RemoteGladosTTS must satisfy the engine's SpeechSynthesizerProtocol shape."""

    def test_has_sample_rate_attribute(self):
        tts = _make_tts()
        assert isinstance(tts.sample_rate, int)
        assert tts.sample_rate > 0

    def test_default_sample_rate_is_22050(self):
        tts = _make_tts()
        assert tts.sample_rate == 22050

    def test_overridden_sample_rate_persists(self):
        from glados_modules.RemoteGladosTTS import RemoteGladosTTS
        tts = RemoteGladosTTS(voice_url="http://gpu:8124/synthesize/",
                               sample_rate=24000)
        assert tts.sample_rate == 24000


class TestGenerateSpeechAudio:
    """Happy path + defensive error handling for generate_speech_audio."""

    def test_decodes_wav(self, silent_wav_bytes):
        fake_resp = MagicMock(status_code=200, content=silent_wav_bytes)
        tts = _make_tts()
        with patch("glados_modules.RemoteGladosTTS.requests.get",
                   return_value=fake_resp):
            audio = tts.generate_speech_audio("hello")
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert audio.shape == (22050,)

    def test_empty_text_returns_zero_array(self):
        tts = _make_tts()
        # No HTTP call should be made for empty text
        with patch("glados_modules.RemoteGladosTTS.requests.get") as mock_get:
            audio = tts.generate_speech_audio("")
        mock_get.assert_not_called()
        assert audio.dtype == np.float32
        assert audio.shape == (0,)

    def test_http_500_returns_zero_array_and_logs(self):
        fake_resp = MagicMock(status_code=500, content=b"server boom")
        tts = _make_tts()
        tts.logger = MagicMock()
        with patch("glados_modules.RemoteGladosTTS.requests.get",
                   return_value=fake_resp):
            audio = tts.generate_speech_audio("hi")
        assert audio.shape == (0,)
        tts.logger.error.assert_called_once()
        assert "500" in tts.logger.error.call_args[0][0]

    def test_request_exception_doesnt_propagate(self):
        import requests as real_requests
        tts = _make_tts()
        tts.logger = MagicMock()
        def raises(*a, **kw):
            raise real_requests.exceptions.ConnectionError("offline")
        with patch("glados_modules.RemoteGladosTTS.requests.get",
                   side_effect=raises):
            audio = tts.generate_speech_audio("hi")
        # Engine treats empty audio as a no-op TTS attempt; it must not raise
        assert audio.shape == (0,)
        tts.logger.error.assert_called_once()

    def test_timeout_doesnt_propagate(self):
        import requests as real_requests
        tts = _make_tts()
        tts.logger = MagicMock()
        def raises(*a, **kw):
            raise real_requests.exceptions.Timeout("slow")
        with patch("glados_modules.RemoteGladosTTS.requests.get",
                   side_effect=raises):
            audio = tts.generate_speech_audio("hi")
        assert audio.shape == (0,)

    def test_corrupt_wav_returns_zero_array(self):
        # 200 OK but the body isn't a valid WAV — soundfile.read raises
        fake_resp = MagicMock(status_code=200, content=b"not a wav")
        tts = _make_tts()
        tts.logger = MagicMock()
        with patch("glados_modules.RemoteGladosTTS.requests.get",
                   return_value=fake_resp):
            audio = tts.generate_speech_audio("hi")
        assert audio.shape == (0,)
        tts.logger.error.assert_called_once()


class TestUrlEncoding:
    """The URL passed to the TTS server is base64-encoded text appended to voice_url."""

    def _capture_url(self, text, silent_wav_bytes):
        captured = {}
        def capture(url, *a, **kw):
            captured["url"] = url
            return MagicMock(status_code=200, content=silent_wav_bytes)
        tts = _make_tts()
        with patch("glados_modules.RemoteGladosTTS.requests.get",
                   side_effect=capture):
            tts.generate_speech_audio(text)
        return captured["url"]

    def test_simple_text_roundtrips(self, silent_wav_bytes):
        url = self._capture_url("hello", silent_wav_bytes)
        assert url.startswith("http://gpu:8124/synthesize/")
        encoded = url.split("/synthesize/", 1)[1]
        assert base64.b64decode(encoded) == b"hello"

    def test_punctuation_roundtrips(self, silent_wav_bytes):
        url = self._capture_url("hi! testing, are we?", silent_wav_bytes)
        encoded = url.split("/synthesize/", 1)[1]
        assert base64.b64decode(encoded).decode("utf8") == "hi! testing, are we?"

    def test_unicode_roundtrips(self, silent_wav_bytes):
        url = self._capture_url("café — über naïve", silent_wav_bytes)
        encoded = url.split("/synthesize/", 1)[1]
        assert base64.b64decode(encoded).decode("utf8") == "café — über naïve"

    def test_timeout_passed_to_requests(self, silent_wav_bytes):
        from glados_modules.RemoteGladosTTS import RemoteGladosTTS
        tts = RemoteGladosTTS(voice_url="http://gpu:8124/synthesize/", timeout_s=5.0)
        captured = {}
        def capture(url, *a, **kw):
            captured["timeout"] = kw.get("timeout")
            return MagicMock(status_code=200, content=silent_wav_bytes)
        with patch("glados_modules.RemoteGladosTTS.requests.get",
                   side_effect=capture):
            tts.generate_speech_audio("hi")
        assert captured["timeout"] == 5.0


class TestAudioFormatNormalization:
    """The engine's SpeechPlayer assumes mono at self.sample_rate; we normalize to that."""

    def test_stereo_downmixed_to_mono(self, stereo_wav_bytes):
        fake_resp = MagicMock(status_code=200, content=stereo_wav_bytes)
        tts = _make_tts()
        with patch("glados_modules.RemoteGladosTTS.requests.get",
                   return_value=fake_resp):
            audio = tts.generate_speech_audio("hi")
        assert audio.ndim == 1
        assert audio.shape == (22050,)
        # Mean of (0.5, -0.5) is 0.0
        assert np.allclose(audio, 0.0)

    def test_sample_rate_mismatch_warns_but_returns_audio(self, wav_44100_bytes):
        fake_resp = MagicMock(status_code=200, content=wav_44100_bytes)
        tts = _make_tts()
        tts.logger = MagicMock()
        with patch("glados_modules.RemoteGladosTTS.requests.get",
                   return_value=fake_resp):
            audio = tts.generate_speech_audio("hi")
        # Audio is still returned (engine plays it, just at the wrong pitch)
        assert audio.shape == (44100,)
        tts.logger.warning.assert_called_once()
        warning_text = tts.logger.warning.call_args[0][0]
        assert "44100" in warning_text and "22050" in warning_text

    def test_returned_dtype_is_float32(self, silent_wav_bytes):
        fake_resp = MagicMock(status_code=200, content=silent_wav_bytes)
        tts = _make_tts()
        with patch("glados_modules.RemoteGladosTTS.requests.get",
                   return_value=fake_resp):
            audio = tts.generate_speech_audio("hi")
        assert audio.dtype == np.float32
