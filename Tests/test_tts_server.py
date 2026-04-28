"""Tests for the pluggable TTSServer (Step 8a-1).

Avoids actually starting the Flask listener — uses Flask's built-in test_client
to exercise routes in-process. Backend selection paths are exercised by patching
the engine TTS factory and the legacy Tacotron module.
"""

import base64
import io
import os
import urllib.parse
from configparser import ConfigParser
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _config(engine="glados", port=18124, cache_root=None):
    """Build a minimal configparser sized for TTSServer."""
    cache_root = cache_root or "./_test_audio_cache"
    cp = ConfigParser()
    cp.read_string(f"""
[FEATURES]
tts_engine = {engine}

[TTS]
port = {port}
cache_dir_root = {cache_root}
kokoro_voice = af_bella
""")
    return cp


@pytest.fixture
def tmp_cache_root(tmp_path):
    return str(tmp_path / "audio")


@pytest.fixture(autouse=True)
def real_soundfile():
    """Swap conftest's soundfile mock for the real library.

    TTSServer writes WAVs via soundfile.write — the conftest-installed mock
    silently produces empty bytes, which makes every WAV decode fail. We
    install the real module for the duration of each test.
    """
    import importlib
    import sys
    import glados_modules.TTSServer as tts_mod
    saved_sf = sys.modules.pop("soundfile", None)
    try:
        real = importlib.import_module("soundfile")
    except ImportError:
        if saved_sf is not None:
            sys.modules["soundfile"] = saved_sf
        pytest.skip("soundfile not installed")
    sys.modules["soundfile"] = real
    # Rebind the import inside TTSServer so it picks up the real module.
    tts_mod.soundfile = real
    yield real
    if saved_sf is not None:
        sys.modules["soundfile"] = saved_sf
        tts_mod.soundfile = saved_sf


@pytest.fixture
def make_glados_synth():
    """Factory: returns a fake SpeechSynthesizerProtocol with controllable audio."""
    def _make(sample_rate=22050, audio=None):
        synth = MagicMock()
        synth.sample_rate = sample_rate
        if audio is None:
            audio = np.zeros(sample_rate, dtype=np.float32)
        synth.generate_speech_audio.return_value = audio
        return synth
    return _make


# ----------------------------------------------------------------------
# Backend selection
# ----------------------------------------------------------------------

class TestBackendSelection:
    """TTSServer routes [FEATURES] tts_engine to the right loader."""

    def test_glados_engine_loads_synth(self, tmp_cache_root, make_glados_synth, monkeypatch):
        from glados_modules import TTSServer as tts_mod
        synth = make_glados_synth(sample_rate=22050)
        captured = {}

        def fake_get(voice):
            captured["voice"] = voice
            return synth

        monkeypatch.setattr(tts_mod, "get_speech_synthesizer", fake_get,
                             raising=False)
        # The module imports get_speech_synthesizer inside _load_engine_voice;
        # monkeypatch the source module instead.
        import sys
        fake_glados_tts = MagicMock()
        fake_glados_tts.get_speech_synthesizer = fake_get
        sys.modules["glados.TTS"] = fake_glados_tts
        try:
            srv = tts_mod.TTSServer(_config(engine="glados",
                                             cache_root=tmp_cache_root))
            assert srv.engine_kind == "glados"
            assert srv._engine_synth is synth
            assert srv.sample_rate == 22050
            assert captured["voice"] == "glados"
            assert srv.cache_dir.endswith("_glados")
            assert os.path.isdir(srv.cache_dir)
        finally:
            sys.modules.pop("glados.TTS", None)

    def test_kokoro_engine_uses_configured_voice(self, tmp_cache_root,
                                                  make_glados_synth):
        from glados_modules import TTSServer as tts_mod
        synth = make_glados_synth(sample_rate=24000)
        captured = {}

        def fake_get(voice):
            captured["voice"] = voice
            return synth

        import sys
        fake_glados_tts = MagicMock()
        fake_glados_tts.get_speech_synthesizer = fake_get
        sys.modules["glados.TTS"] = fake_glados_tts
        try:
            srv = tts_mod.TTSServer(_config(engine="kokoro",
                                             cache_root=tmp_cache_root))
            assert srv.engine_kind == "kokoro"
            assert captured["voice"] == "af_bella"
            assert srv.sample_rate == 24000
            assert srv.cache_dir.endswith("_kokoro")
        finally:
            sys.modules.pop("glados.TTS", None)

    def test_unknown_engine_raises(self, tmp_cache_root):
        from glados_modules.TTSServer import TTSServer, TTSServerException
        with pytest.raises(TTSServerException):
            TTSServer(_config(engine="not_a_real_engine",
                              cache_root=tmp_cache_root))

    def test_per_engine_cache_dirs_isolated(self, tmp_cache_root,
                                             make_glados_synth):
        from glados_modules import TTSServer as tts_mod
        import sys
        sys.modules["glados.TTS"] = MagicMock(
            get_speech_synthesizer=lambda voice: make_glados_synth())
        try:
            glados_srv = tts_mod.TTSServer(_config(engine="glados",
                                                    cache_root=tmp_cache_root))
            kokoro_srv = tts_mod.TTSServer(_config(engine="kokoro",
                                                    cache_root=tmp_cache_root))
            assert glados_srv.cache_dir != kokoro_srv.cache_dir
            assert "glados" in glados_srv.cache_dir
            assert "kokoro" in kokoro_srv.cache_dir
        finally:
            sys.modules.pop("glados.TTS", None)


# ----------------------------------------------------------------------
# synthesize() — cache + backend dispatch
# ----------------------------------------------------------------------

class TestSynthesize:
    """synthesize() returns WAV bytes, populates cache, reads cache on hit."""

    def _make_glados_server(self, tmp_cache_root, sample_rate=22050,
                             audio=None):
        from glados_modules import TTSServer as tts_mod
        import sys
        synth = MagicMock()
        synth.sample_rate = sample_rate
        if audio is None:
            audio = (0.1 * np.sin(2 * np.pi * 440 *
                                   np.arange(sample_rate) / sample_rate)
                     ).astype(np.float32)
        synth.generate_speech_audio.return_value = audio
        sys.modules["glados.TTS"] = MagicMock(
            get_speech_synthesizer=lambda voice: synth)
        try:
            srv = tts_mod.TTSServer(_config(engine="glados",
                                             cache_root=tmp_cache_root))
        finally:
            sys.modules.pop("glados.TTS", None)
        return srv, synth, audio

    def test_empty_text_returns_none(self, tmp_cache_root):
        srv, _, _ = self._make_glados_server(tmp_cache_root)
        assert srv.synthesize("") is None
        assert srv.synthesize("   ") is None

    def test_first_call_synthesizes_and_caches(self, tmp_cache_root):
        srv, synth, _ = self._make_glados_server(tmp_cache_root)
        result = srv.synthesize("hello world")
        assert result is not None
        assert len(result) > 100  # WAV header + samples
        assert synth.generate_speech_audio.call_count == 1
        # WAV cached on disk under the engine-specific dir
        files = os.listdir(srv.cache_dir)
        assert any(f.startswith("GLaDOS-tts-") for f in files)

    def test_cache_hit_skips_synthesis(self, tmp_cache_root):
        srv, synth, _ = self._make_glados_server(tmp_cache_root)
        srv.synthesize("hello world")
        synth.generate_speech_audio.reset_mock()
        result2 = srv.synthesize("hello world")
        assert result2 is not None
        synth.generate_speech_audio.assert_not_called()

    def test_returned_wav_decodes_at_correct_rate(self, tmp_cache_root,
                                                    real_soundfile):
        # Use the real_soundfile from the autouse fixture (the module-level
        # import was bound to conftest's mock at collection time).
        srv, _, _ = self._make_glados_server(tmp_cache_root, sample_rate=24000)
        wav_bytes = srv.synthesize("test rate")
        with io.BytesIO(wav_bytes) as buf:
            data, sr = real_soundfile.read(buf)
        assert sr == 24000

    def test_synthesis_failure_returns_none(self, tmp_cache_root):
        srv, synth, _ = self._make_glados_server(tmp_cache_root)
        synth.generate_speech_audio.side_effect = RuntimeError("OOM")
        assert srv.synthesize("anything") is None

    def test_empty_audio_returned_by_engine_treated_as_failure(self, tmp_cache_root):
        srv, synth, _ = self._make_glados_server(tmp_cache_root)
        synth.generate_speech_audio.return_value = np.zeros(0, dtype=np.float32)
        assert srv.synthesize("any text") is None


# ----------------------------------------------------------------------
# Flask route — uses test_client, no actual listener
# ----------------------------------------------------------------------

class TestFlaskRoute:
    """The /synthesize/<base64> route accepts/rejects + returns WAV mimetype."""

    def _make(self, tmp_cache_root):
        from glados_modules import TTSServer as tts_mod
        import sys
        synth = MagicMock()
        synth.sample_rate = 22050
        synth.generate_speech_audio.return_value = (
            0.1 * np.sin(2 * np.pi * 440 *
                          np.arange(22050) / 22050)
        ).astype(np.float32)
        sys.modules["glados.TTS"] = MagicMock(
            get_speech_synthesizer=lambda voice: synth)
        try:
            srv = tts_mod.TTSServer(_config(engine="glados",
                                             cache_root=tmp_cache_root))
        finally:
            sys.modules.pop("glados.TTS", None)
        return srv

    def test_empty_path_returns_400(self, tmp_cache_root):
        srv = self._make(tmp_cache_root)
        client = srv.app.test_client()
        rv = client.get("/synthesize/")
        assert rv.status_code == 400

    def test_bad_base64_returns_400(self, tmp_cache_root):
        srv = self._make(tmp_cache_root)
        client = srv.app.test_client()
        rv = client.get("/synthesize/not-real-base64!!!")
        assert rv.status_code == 400

    def test_valid_request_returns_wav(self, tmp_cache_root):
        srv = self._make(tmp_cache_root)
        client = srv.app.test_client()
        encoded = base64.b64encode(b"hello").decode("ascii")
        rv = client.get(f"/synthesize/{encoded}")
        assert rv.status_code == 200
        assert rv.mimetype == "audio/wav"
        assert len(rv.data) > 100

    def test_synthesis_failure_returns_500(self, tmp_cache_root):
        srv = self._make(tmp_cache_root)
        # Force inner synth to fail
        srv._engine_synth.generate_speech_audio.side_effect = RuntimeError("boom")
        client = srv.app.test_client()
        encoded = base64.b64encode(b"will fail").decode("ascii")
        rv = client.get(f"/synthesize/{encoded}")
        assert rv.status_code == 500

    def test_unicode_text_roundtrips(self, tmp_cache_root):
        srv = self._make(tmp_cache_root)
        client = srv.app.test_client()
        encoded = base64.b64encode("café — naïve".encode("utf8")).decode("ascii")
        rv = client.get(f"/synthesize/{encoded}")
        assert rv.status_code == 200

    def test_url_encoded_plus_in_base64_decodes(self, tmp_cache_root):
        # Standard base64 alphabet includes '+'; clients MUST URL-encode it
        # as %2B. Server has to urllib.parse.unquote before base64.b64decode.
        # Without the unquote (R-1.1 regression), this test fails.
        srv = self._make(tmp_cache_root)
        client = srv.app.test_client()
        # Find a string whose base64 contains '+' (any 3-byte input where the
        # middle byte has high bits set — ">>>" produces "Pj4+").
        text = "test >>>"
        encoded = base64.b64encode(text.encode("utf8")).decode("ascii")
        assert "+" in encoded, "test setup needs base64 with a '+' character"
        # Simulate what a real HTTP client does: URL-encode the path segment
        url_safe = urllib.parse.quote(encoded, safe="")
        rv = client.get(f"/synthesize/{url_safe}")
        assert rv.status_code == 200
        assert rv.mimetype == "audio/wav"

    def test_url_encoded_slash_in_base64_decodes(self, tmp_cache_root):
        # '/' is also in the base64 alphabet; clients send it as %2F.
        srv = self._make(tmp_cache_root)
        client = srv.app.test_client()
        # "???" → "Pz8/"
        text = "test ???"
        encoded = base64.b64encode(text.encode("utf8")).decode("ascii")
        assert "/" in encoded, "test setup needs base64 with a '/' character"
        url_safe = urllib.parse.quote(encoded, safe="")
        rv = client.get(f"/synthesize/{url_safe}")
        assert rv.status_code == 200


# ----------------------------------------------------------------------
# Filename safety
# ----------------------------------------------------------------------

class TestSafeFilename:
    """Cache filenames must be filesystem-safe + deterministic."""

    def test_deterministic(self):
        from glados_modules.TTSServer import _safe_filename
        assert _safe_filename("hello world") == _safe_filename("hello world")

    def test_strips_unsafe_chars(self):
        from glados_modules.TTSServer import _safe_filename
        name = _safe_filename("hi! you're #1, /tmp/etc")
        assert "/" not in name
        assert "!" not in name
        assert "#" not in name

    def test_handles_empty_after_strip(self):
        from glados_modules.TTSServer import _safe_filename
        assert _safe_filename("!!!").endswith("empty.wav")

    def test_caps_long_text(self):
        from glados_modules.TTSServer import _safe_filename
        name = _safe_filename("abc" * 200)
        assert len(name) < 200

    def test_starts_with_known_prefix(self):
        from glados_modules.TTSServer import _safe_filename
        assert _safe_filename("anything").startswith("GLaDOS-tts-")
        assert _safe_filename("anything").endswith(".wav")
