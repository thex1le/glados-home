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
