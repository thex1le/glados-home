# native imports
import base64
import io
from typing import Optional

# 3rd party imports
import numpy as np
from numpy.typing import NDArray
import requests
import soundfile

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import LoggingEnums


class RemoteGladosTTS:
    """SpeechSynthesizerProtocol adapter that proxies to the Flask GLaDOS TTS server.

    Lets the dnhkng/GLaDOS engine on the Pi 5 reuse the existing Mike Klingbeil
    Tacotron TTS (or its Kokoro replacement) running on the GPU box, without
    loading any TTS model locally. Conforms to glados.TTS.SpeechSynthesizerProtocol:
    exposes a sample_rate attribute and a generate_speech_audio() method.

    Attributes:
        logger: Logger instance for logging.
        voice_url: Base URL of the synthesize endpoint.
        sample_rate: Expected sample rate of the WAVs returned by the server.
        timeout_s: HTTP request timeout in seconds.
    """

    DEFAULT_SAMPLE_RATE: int = 22050

    def __init__(self, voice_url: str, sample_rate: Optional[int] = None,
                 timeout_s: float = 30.0) -> None:
        """Initialize the remote TTS adapter.

        Args:
            voice_url: Base URL ending with '/synthesize/' (the engine appends
                a base64-encoded text segment to this).
            sample_rate: Expected sample rate of returned WAVs. Defaults to
                22050 (Mike Klingbeil Tacotron output).
            timeout_s: HTTP request timeout in seconds.
        """
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.voice_url: str = voice_url
        self.sample_rate: int = sample_rate if sample_rate is not None else self.DEFAULT_SAMPLE_RATE
        self.timeout_s: float = timeout_s

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """Synthesize text to a numpy waveform via the remote TTS server.

        Args:
            text: The text to synthesize.

        Returns:
            A 1-D float32 numpy array of audio samples at self.sample_rate.
            On any failure (empty input, HTTP error, network exception), returns
            a zero-length array so the engine's TTS thread treats it as a no-op
            instead of crashing.
        """
        if not text:
            return np.zeros(0, dtype=np.float32)
        encoded = base64.b64encode(text.encode("utf8")).decode("ascii")
        url = f"{self.voice_url}{encoded}"
        try:
            response = requests.get(url, timeout=self.timeout_s)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Remote TTS request failed: {e}")
            return np.zeros(0, dtype=np.float32)
        if response.status_code != 200:
            self.logger.error(f"Remote TTS returned HTTP {response.status_code}")
            return np.zeros(0, dtype=np.float32)
        try:
            with io.BytesIO(response.content) as wav_io:
                data, sr = soundfile.read(wav_io, dtype="float32")
        except Exception as e:
            self.logger.error(f"Remote TTS WAV decode failed: {e}")
            return np.zeros(0, dtype=np.float32)
        if sr != self.sample_rate:
            self.logger.warning(
                f"Remote TTS sample rate {sr} != expected {self.sample_rate}; "
                "playback pitch may shift")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32)
