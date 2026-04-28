"""TTSServer — pluggable Flask TTS at :8124/synthesize/<base64>.

Step 8a-1. Replaces the ~50-line inline Flask block in AiServer.py with a
configurable backend selectable via [FEATURES] tts_engine. Same wire
interface so RemoteGladosTTS on the Pi 5 doesn't have to change.

Backends:
    tacotron — legacy Mike Klingbeil Tacotron2 (vendored gladosTTS repo)
    glados   — dnhkng/GLaDOS-ONNX trained voice (the actual Portal voice)
    kokoro   — multi-voice Kokoro ONNX engine (voice selected via [TTS] kokoro_voice)

Each engine writes to its own cache dir (./audio_<engine>) so switching
backends doesn't pollute or invalidate previously synthesized WAVs.
"""

# native imports
import base64
import io
import logging
import os
import re
import shutil
from threading import Thread
from time import sleep, time
from typing import Any, Optional
import urllib.parse

# 3rd party imports
from flask import Flask, request, send_file
import numpy as np
from numpy.typing import NDArray
import soundfile

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import (LoggingEnums, FeatureToggles, TTSEnums)


class TTSServerException(Exception):
    pass


# Filenames must survive a shell + filesystem; strip everything weird.
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(text: str) -> str:
    """Sanitize text into a cache filename.

    Matches the legacy AiServer.py scheme conceptually: collapse whitespace
    + drop punctuation. Keep it deterministic so the same text always maps
    to the same file across processes.
    """
    cleaned = _FILENAME_SAFE.sub("-", text.strip()).strip("-")
    if not cleaned:
        cleaned = "empty"
    # Cap at 120 chars so we don't blow filesystem limits on long phrases
    return f"GLaDOS-tts-{cleaned[:120]}.wav"


class TTSServer(Thread):
    """Flask-based pluggable TTS server.

    Subclass of Thread so HealthMonitor can register and is_alive() it like
    every other long-running service. Flask's blocking app.run() executes
    inside our run() method.

    Attributes:
        logger: Logger instance.
        configFile: Configuration parser instance loaded from glog.conf.
        engine_kind: One of 'tacotron', 'glados', 'kokoro'.
        port: Listener port (default 8124).
        cache_dir: Per-engine WAV cache (./audio_<engine>).
        sample_rate: Loaded engine's output sample rate (advisory; the WAV
            file itself carries authoritative rate metadata).
    """

    def __init__(self, config_file: Any) -> None:
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.configFile = config_file
        self.stop: bool = False

        self.engine_kind: str = config_file.get(
            FeatureToggles.CONFIG_HEAD.value,
            FeatureToggles.TTS_ENGINE.value,
            fallback=FeatureToggles.DEFAULT_TTS_ENGINE.value).strip().lower()

        t_section = TTSEnums.CONFIG_HEAD.value
        t_cfg = (config_file[t_section]
                 if config_file.has_section(t_section) else {})

        def _tcfg(key: TTSEnums, default: Any) -> Any:
            return (t_cfg.get(key.value, str(default))
                    if hasattr(t_cfg, "get") else default)

        self.port: int = int(_tcfg(TTSEnums.PORT, TTSEnums.DEFAULT_PORT.value))
        cache_root: str = _tcfg(TTSEnums.CACHE_DIR_ROOT,
                                 TTSEnums.DEFAULT_CACHE_DIR_ROOT.value)
        # Per-engine cache so switching engines is reversible without churn.
        self.cache_dir: str = f"{cache_root}_{self.engine_kind}"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Backend handles populated by _load_*; exactly one is set per instance.
        self._tacotron_engine: Any = None  # legacy gladosTTS module
        self._engine_synth: Any = None     # glados.TTS SpeechSynthesizerProtocol
        self.sample_rate: int = 22050      # overwritten per backend

        self._load_backend()
        self.app = self._build_flask_app()
        # Silence werkzeug's per-request logs — we have our own.
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    def _load_backend(self) -> None:
        kind = self.engine_kind
        if kind == FeatureToggles.TTS_ENGINE_TACOTRON.value:
            self._load_tacotron()
        elif kind == FeatureToggles.TTS_ENGINE_GLADOS.value:
            self._load_engine_voice("glados")
        elif kind == FeatureToggles.TTS_ENGINE_KOKORO.value:
            t_cfg = self.configFile[TTSEnums.CONFIG_HEAD.value]
            voice = t_cfg.get(TTSEnums.KOKORO_VOICE.value,
                               TTSEnums.DEFAULT_KOKORO_VOICE.value)
            self._load_engine_voice(voice)
        else:
            raise TTSServerException(
                f"Unknown TTS engine: {kind!r}. Expected one of "
                f"tacotron|glados|kokoro.")

    def _load_tacotron(self) -> None:
        """Load the legacy Mike Klingbeil Tacotron2 path (vendored gladosTTS)."""
        import sys
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(repo_root, "gladosTTS"))
        from glados_tts import engine as glados_voice
        self._tacotron_engine = glados_voice
        self.sample_rate = 22050
        self.logger.info("TTS backend: Tacotron2 (legacy gladosTTS)")

    def _load_engine_voice(self, voice: str) -> None:
        """Load a dnhkng/GLaDOS engine voice (glados-onnx or kokoro)."""
        from glados.TTS import get_speech_synthesizer
        self._engine_synth = get_speech_synthesizer(voice)
        self.sample_rate = int(self._engine_synth.sample_rate)
        self.logger.info(
            f"TTS backend: engine voice='{voice}' sr={self.sample_rate}")

    # ------------------------------------------------------------------
    # Synthesis (cache-aware)
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> Optional[bytes]:
        """Return WAV bytes for `text`. Uses cache when available.

        Returns None on synthesis failure so the Flask route can surface
        a 500 error to the caller (RemoteGladosTTS treats that as a no-op).
        """
        text = (text or "").strip()
        if not text:
            return None
        cached_path = os.path.join(self.cache_dir, _safe_filename(text))
        if os.path.isfile(cached_path):
            os.utime(cached_path, None)  # touch for LRU pruning later
            with open(cached_path, "rb") as f:
                return f.read()
        try:
            wav_bytes = self._synthesize_to_wav_bytes(text, cached_path)
        except Exception as e:
            self.logger.error(f"Synthesis failed for {text[:60]!r}: {e}")
            return None
        return wav_bytes

    def _synthesize_to_wav_bytes(self, text: str, cache_path: str) -> bytes:
        """Backend-specific synthesis + cache write. Returns the WAV bytes."""
        if self._tacotron_engine is not None:
            # Tacotron path writes a temp WAV then we move it into the cache.
            key = str(time())[7:]
            ok = self._tacotron_engine.glados_tts(text, key)
            if not ok:
                raise TTSServerException("Tacotron engine returned False")
            tempfile = os.path.join(
                os.getcwd(), "audio", f"GLaDOS-tts-temp-output-{key}.wav")
            if not os.path.isfile(tempfile):
                raise TTSServerException(
                    f"Tacotron output missing: {tempfile}")
            shutil.move(tempfile, cache_path)
            with open(cache_path, "rb") as f:
                return f.read()

        if self._engine_synth is not None:
            audio: NDArray[np.float32] = self._engine_synth.generate_speech_audio(text)
            if audio is None or audio.size == 0:
                raise TTSServerException("Engine synthesizer returned empty audio")
            buf = io.BytesIO()
            soundfile.write(buf, audio, self.sample_rate,
                            format="WAV", subtype="PCM_16")
            data = buf.getvalue()
            with open(cache_path, "wb") as f:
                f.write(data)
            return data

        raise TTSServerException("No TTS backend loaded")

    # ------------------------------------------------------------------
    # Flask app
    # ------------------------------------------------------------------

    def _build_flask_app(self) -> Flask:
        app = Flask(self.__name__)

        @app.route("/synthesize/", defaults={"text": ""})
        @app.route("/synthesize/<path:text>")
        def synthesize_route(text: str):
            # Reconstruct the raw base64 chunk from the URL — matches the
            # legacy quirk where base64 may contain '+' that Flask routes
            # don't tolerate as a path parameter. Standard base64 alphabet
            # includes '+' and '/', which clients must URL-encode (%2B, %2F);
            # we have to URL-decode before handing to base64.b64decode.
            url = request.url
            marker = "synthesize/"
            idx = url.find(marker)
            if idx < 0:
                return "Bad URL", 400
            encoded = urllib.parse.unquote(url[idx + len(marker):])
            if not encoded:
                return "No input", 400
            try:
                line = base64.b64decode(encoded).decode("utf8")
            except Exception:
                return "Bad base64", 400
            wav_bytes = self.synthesize(line)
            if wav_bytes is None:
                return "TTS failed", 500
            return send_file(io.BytesIO(wav_bytes),
                             mimetype="audio/wav",
                             as_attachment=False,
                             download_name=_safe_filename(line))

        return app

    # ------------------------------------------------------------------
    # Thread loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.logger.info(
            f"TTS server listening on 0.0.0.0:{self.port} "
            f"(engine={self.engine_kind}, cache={self.cache_dir})")
        try:
            self.app.run(host="0.0.0.0", port=self.port, threaded=True,
                         use_reloader=False)
        except Exception as e:
            self.logger.error(f"TTS server stopped: {e}")
