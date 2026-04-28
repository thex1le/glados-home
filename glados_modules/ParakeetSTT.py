"""ParakeetSTT — drop-in replacement for LocalSTTtx using NVIDIA Parakeet.

Step 8a-2. Mirrors the LocalSTTtx interface so AudioServerRX's callback
signature and the sst/results MQTT contract stay unchanged. The Pi 5
brain's STT consumer (GladosSTT.get_text()) sees no difference.

Differences from WhisperX:
    * No built-in word-level alignment — time_map field will be empty.
    * Single-language (English by default); language detection is not
      provided, so we publish 'en' unconditionally.
    * Faster (~150-300ms vs WhisperX's 500-1500ms on a 4090).
    * Better punctuation, less silence-hallucination.

Audio bytes are decoded the same way (ffmpeg → 16kHz mono float32) so
AudioServerTx on the Pi 5 doesn't need to change.
"""

# native imports
from typing import Any, Dict, NamedTuple, Optional
from threading import Lock

# 3rd party imports
import numpy as np

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import (LoggingEnums, MQTTEnums, STTEnums,
                                         ASREnums)
from glados_modules.MqttConnector import MQTTClient, SttMessageBuilder


class ParakeetSTTException(Exception):
    pass


class ParakeetSTT(MQTTClient):
    """Wraps the dnhkng/GLaDOS Parakeet ASR for the AiServer pipeline.

    Subclasses MQTTClient (not Thread) — the AudioServerRX thread already
    owns the network listener and calls process_audio() as its callback.
    Same composition pattern as the legacy LocalSTTtx.

    Attributes:
        logger: Logger instance.
        configFile: Configuration parser instance loaded from glog.conf.
        transcriber: A glados.ASR.TranscriberProtocol implementation.
    """

    SAMPLE_RATE: int = 16000

    def __init__(self, broker: NamedTuple, config_file: Any) -> None:
        """Initialize Parakeet ASR.

        Args:
            broker: Broker namedtuple with ip + port (matches LocalSTTtx).
            config_file: configparser instance loaded from glog.conf.
        """
        # ffmpeg held as instance attr so tests can swap it out
        import ffmpeg
        self.ffmpeg = ffmpeg
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_DEBUG.value)
        self.configFile = config_file

        a_section = ASREnums.CONFIG_HEAD.value
        a_cfg = (config_file[a_section]
                 if config_file.has_section(a_section) else {})

        def _acfg(key: ASREnums, default: Any) -> Any:
            return (a_cfg.get(key.value, str(default))
                    if hasattr(a_cfg, "get") else default)

        engine_type = _acfg(ASREnums.ENGINE_TYPE,
                             ASREnums.DEFAULT_ENGINE_TYPE.value)
        # ONNX sessions aren't thread-safe; serialize inference calls in case
        # multiple AudioServerRX clients connect concurrently.
        self._inference_lock: Lock = Lock()
        self._load_model(engine_type)
        super().__init__(ip=broker.ip, port=broker.port)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, engine_type: str) -> None:
        """Load the Parakeet ASR via the engine's get_audio_transcriber factory."""
        from glados.ASR import get_audio_transcriber
        try:
            self.transcriber = get_audio_transcriber(engine_type=engine_type)
        except Exception as e:
            raise ParakeetSTTException(
                f"Failed to load Parakeet engine_type={engine_type}: {e}")
        self.logger.info(f"Parakeet ASR loaded (engine_type={engine_type})")

    # ------------------------------------------------------------------
    # AudioServerRX callback
    # ------------------------------------------------------------------

    def process_audio(self, byte_stream: bytes) -> None:
        """Process an audio byte stream, transcribe, publish results.

        Same callback signature as LocalSTTtx.process_audio so AudioServerRX
        works with either backend without code changes.
        """
        try:
            audio = self.load_audio_from_bytes(byte_stream)
        except Exception as e:
            self.logger.error(f"ffmpeg decode failed, dropping audio: {e}")
            return
        if audio.size == 0:
            self.logger.warning("Decoded audio is empty, dropping")
            return
        try:
            with self._inference_lock:
                text = self.transcriber.transcribe(audio)
        except Exception as e:
            self.logger.error(f"Parakeet transcription failed: {e}")
            return
        if text is None:
            text = ""
        text = text.strip()
        self.logger.debug(f"Detected text: {text!r}")
        # Publish in the same shape as LocalSTTtx so downstream consumers
        # (GladosSTT on the Pi 5, dashboard) don't need to change. Parakeet
        # doesn't provide word-level alignment, so time_map is empty.
        rsp: Dict[str, Any] = {
            STTEnums.STT_TEXT_KEY.value: text,
            STTEnums.STT_RAW_RESULTS_KEY.value: {},
            STTEnums.STT_LANGUAGE_KEY.value: STTEnums.STT_EN_LANG_KEY.value,
            STTEnums.STT_TIME_MAP_KEY.value: [],
        }
        try:
            self.send_command(
                SttMessageBuilder.send_speech_to_text_message(rsp),
                MQTTEnums.STT_RESULTS_MQTT_TOPIC.value)
        except Exception as e:
            self.logger.error(f"Failed to publish STT results: {e}")

    # ------------------------------------------------------------------
    # Audio decode (matches LocalSTTtx for protocol parity)
    # ------------------------------------------------------------------

    def load_audio_from_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """Decode arbitrary audio bytes to 16kHz mono float32 via ffmpeg."""
        process = (
            self.ffmpeg.input("pipe:0")
            .output("pipe:1", format="wav", ac=1, ar=str(self.SAMPLE_RATE))
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        out, _ = process.communicate(input=audio_bytes)
        return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
