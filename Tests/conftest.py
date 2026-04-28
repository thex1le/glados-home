"""Pytest configuration -- mocks all hardware dependencies so tests run on any machine.

This file is loaded by pytest automatically before any test. It patches sys.modules
with mock versions of every hardware library (adafruit, GPIO, picamera2, GStreamer, torch)
so that glados_modules can be imported without real hardware.
"""

import sys
from unittest.mock import MagicMock, PropertyMock


def _mock_module(name, attrs=None):
    """Create a mock module and register it in sys.modules."""
    mock = MagicMock()
    if attrs:
        for attr, value in attrs.items():
            setattr(mock, attr, value)
    sys.modules[name] = mock
    return mock


# ---- Hardware mocks (must happen before any glados_modules import) ----

# Raspberry Pi GPIO
_mock_module("RPi")
_mock_module("RPi.GPIO")

# Adafruit Blinka (board, busio, digitalio)
board_mock = _mock_module("board")
board_mock.SCL = 3
board_mock.SDA = 2
board_mock.D12 = 12
board_mock.D18 = 18
board_mock.D21 = 21
board_mock.D25 = 25
board_mock.D24 = 24
board_mock.CE0 = 8
board_mock.SCK = 11
board_mock.MOSI = 10
board_mock.I2C = MagicMock

_mock_module("busio")
_mock_module("digitalio")
_mock_module("digitalio.DigitalInOut")

# Adafruit hardware libraries
_mock_module("adafruit_pca9685")
_mock_module("adafruit_servokit")
_mock_module("adafruit_rgb_display")
_mock_module("adafruit_rgb_display.st7789")
_mock_module("adafruit_bno055")
_mock_module("adafruit_vl53l4cd")
_mock_module("adafruit_sht4x")
_mock_module("adafruit_ens160")

# NeoPixel
neopixel_mock = _mock_module("neopixel")
neopixel_mock.NeoPixel = MagicMock
neopixel_mock.RGB = "RGB"
neopixel_mock.RGBW = "RGBW"

# PiCamera2
_mock_module("picamera2")
_mock_module("picamera2.Picamera2")
_mock_module("picamera2.MappedArray")
_mock_module("picamera2.Preview")

# GStreamer (gi/GObject)
gi_mock = _mock_module("gi")
gi_mock.require_version = MagicMock()
_mock_module("gi.repository")
_mock_module("gi.repository.Gst")
_mock_module("gi.repository.GstRtspServer")
_mock_module("gi.repository.GLib")

# ALSA audio
_mock_module("alsaaudio")

# PIL/Pillow (partial mock -- keep Image working if available)
try:
    import PIL
except ImportError:
    _mock_module("PIL")
    _mock_module("PIL.Image")
    _mock_module("PIL.ImageDraw")

# OpenCV (cv2) -- prefer the real library so SceneDescriber tests can exercise
# the actual cv2.absdiff scene-change logic. Mock if not installed.
try:
    import cv2
except ImportError:
    _mock_module("cv2")

# Speech recognition
_mock_module("speech_recognition")

# pydub
_mock_module("pydub")
_mock_module("pydub.AudioSegment")
_mock_module("pydub.playback")

# Home Assistant
_mock_module("homeassistant_api")

# Torch and ML (for GPU server code)
torch_mock = _mock_module("torch")
torch_mock.load = MagicMock()
torch_mock.cuda = MagicMock()
torch_mock.cuda.is_available = MagicMock(return_value=False)
_mock_module("torch.serialization")
_mock_module("torch.serialization.add_safe_globals", MagicMock())
_mock_module("torch.serialization.safe_globals", MagicMock())
_mock_module("torch.nn")
_mock_module("torch.nn.modules")
_mock_module("torch.nn.modules.container")
_mock_module("ultralytics")
_mock_module("ultralytics.YOLO")
_mock_module("ultralytics.utils")
_mock_module("ultralytics.utils.plotting")
_mock_module("ultralytics.nn")
_mock_module("ultralytics.nn.tasks")
_mock_module("ultralytics.nn.modules")
_mock_module("rtmlib")
_mock_module("whisperx")
_mock_module("ffmpeg")

# Face recognition / emotion
_mock_module("insightface")
_mock_module("insightface.app")
_mock_module("hsemotion")
_mock_module("hsemotion.facial_emotions")

# gladosTTS (external, not in repo)
_mock_module("gladosTTS")
_mock_module("gladosTTS.engine")

# soundfile (used by RemoteGladosTTS to decode WAV bytes from the GPU TTS server)
# Tests that need real WAV decoding inject the real soundfile via fixture.
_mock_module("soundfile")

# dnhkng/GLaDOS engine package (vendored on the Pi 5 via pip install -e ../GLaDOS).
# Tests for GladosBrain construct it via __new__() to bypass _build_engine, so
# these mocks only need to satisfy import-time references.
_mock_module("glados")
_mock_module("glados.audio_io",
             attrs={"get_audio_system": lambda backend_type: MagicMock()})
_mock_module("glados.autonomy",
             attrs={"AutonomyConfig": lambda **kw: MagicMock(**kw)})
_mock_module("glados.autonomy.config",
             attrs={"EmotionConfig": lambda **kw: MagicMock(**kw),
                    "HEXACOConfig": lambda **kw: MagicMock(**kw),
                    "TokenConfig": lambda **kw: MagicMock(**kw),
                    "AutonomyJobsConfig": lambda **kw: MagicMock(**kw),
                    "HackerNewsJobConfig": lambda **kw: MagicMock(**kw),
                    "WeatherJobConfig": lambda **kw: MagicMock(**kw)})
_mock_module("glados.autonomy.emotion_state",
             attrs={"EmotionEvent": lambda **kw: MagicMock(**kw)})
_mock_module("glados.core")
_mock_module("glados.core.engine", attrs={"Glados": MagicMock})
_mock_module("glados.mcp")
_mock_module("glados.mcp.config",
             attrs={"MCPServerConfig": lambda **kw: MagicMock(**kw)})
_mock_module("glados.vision")
_mock_module("glados.vision.fastvlm", attrs={"FastVLM": MagicMock})
_mock_module("glados.vision.constants",
             attrs={"VISION_DEFAULT_PROMPT": "Describe the scene briefly.",
                    "VISION_DETAIL_PROMPT": "Describe the scene in detail."})

# dnhkng/GLaDOS engine TTS (used by TTSServer when tts_engine = glados|kokoro).
# Returns a stub SpeechSynthesizerProtocol — sample_rate + generate_speech_audio.
def _mock_synth_factory(voice="glados"):
    synth = MagicMock()
    synth.sample_rate = 22050
    # Default to a one-second silent buffer; tests override per-call.
    import numpy as _np
    synth.generate_speech_audio.return_value = _np.zeros(22050, dtype=_np.float32)
    return synth

_mock_module("glados.TTS",
             attrs={"get_speech_synthesizer": _mock_synth_factory})

# dnhkng/GLaDOS engine ASR (used by ParakeetSTT when asr_engine = parakeet).
# Returns a stub TranscriberProtocol — transcribe(audio) → str.
def _mock_transcriber_factory(engine_type="tdt"):
    t = MagicMock()
    t.transcribe.return_value = ""
    t.transcribe_file.return_value = ""
    return t

_mock_module("glados.ASR",
             attrs={"get_audio_transcriber": _mock_transcriber_factory})

# MCP package (used by BodyMCPServer for FastMCP).
_mock_module("mcp")
_mock_module("mcp.server")
# FastMCP() is called at module import time to create the `mcp = FastMCP(...)`
# instance; the @mcp.tool() decorator must be a passthrough so test code can
# import the tool functions and call them directly.
_fake_fastmcp = MagicMock()
_fake_fastmcp.tool = MagicMock(return_value=lambda fn: fn)
_mock_module("mcp.server.fastmcp",
             attrs={"FastMCP": MagicMock(return_value=_fake_fastmcp)})
