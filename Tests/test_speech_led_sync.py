"""Tests for SpeechLedSync + pad_to_eye_color (Step 8a-0).

Bypass MQTTClient.__init__ via __new__() so the real broker never gets
contacted. Drive generate_speech_audio() directly with synthetic WAVs.
"""

from configparser import ConfigParser
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from glados_modules.GladosEnums import LEDHead, MQTTEnums
from glados_modules.MqttConnector import MQTTClient


@pytest.fixture
def sync_config():
    cp = ConfigParser()
    cp.read_string("""
[MQTT]
mqtt_server_ip = 127.0.0.1
mqtt_port = 1883

[BRAIN]
speech_led_delay_s = 0.25
""")
    return cp


@pytest.fixture
def fake_inner_tts():
    """Minimal SpeechSynthesizerProtocol stand-in."""
    inner = MagicMock()
    inner.sample_rate = 22050
    return inner


@pytest.fixture
def fake_sync(sync_config, fake_inner_tts):
    from glados_modules.SpeechLedSync import SpeechLedSync
    with patch.object(MQTTClient, '__init__', lambda self, *a, **kw: None):
        s = SpeechLedSync.__new__(SpeechLedSync)
    s.__name__ = "SpeechLedSync"
    s.logger = MagicMock()
    s.configFile = sync_config
    s.inner_tts = fake_inner_tts
    s.mood_consumer = None
    s.stop = False
    s._delay_s = 0.25
    s.send_command = MagicMock()
    return s


def _last_call(sync):
    msg, topic = sync.send_command.call_args[0]
    return msg, topic


def _silent_wav(seconds: float, sr: int = 22050) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def _ramp_wav(seconds: float, sr: int = 22050) -> np.ndarray:
    """Linear amplitude ramp from 0 to 1.0 — useful for envelope tests."""
    n = int(seconds * sr)
    return np.linspace(0, 1.0, n, dtype=np.float32)


def _tone_wav(seconds: float, freq: float = 440.0, sr: int = 22050) -> np.ndarray:
    n = int(seconds * sr)
    t = np.arange(n) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ----------------------------------------------------------------------
# pad_to_eye_color — pure mapping function
# ----------------------------------------------------------------------

class TestPadToEyeColor:
    """Four PAD quadrants + the default each map to a known color."""

    def test_furious_quadrant_returns_red(self):
        from glados_modules.SpeechLedSync import pad_to_eye_color
        from glados_modules.MoodConsumer import PADState
        c = pad_to_eye_color(PADState(pleasure=-0.5, arousal=0.8, dominance=0.0))
        assert c == (255, 0, 0)

    def test_excited_quadrant_returns_cyan(self):
        from glados_modules.SpeechLedSync import pad_to_eye_color
        from glados_modules.MoodConsumer import PADState
        c = pad_to_eye_color(PADState(pleasure=0.5, arousal=0.8, dominance=0.0))
        assert c == (0, 200, 255)

    def test_drowsy_quadrant_returns_dim_white(self):
        from glados_modules.SpeechLedSync import pad_to_eye_color
        from glados_modules.MoodConsumer import PADState
        c = pad_to_eye_color(PADState(pleasure=0.0, arousal=-0.7, dominance=0.0))
        assert c == (80, 80, 80)

    def test_content_quadrant_returns_green_amber(self):
        from glados_modules.SpeechLedSync import pad_to_eye_color
        from glados_modules.MoodConsumer import PADState
        c = pad_to_eye_color(PADState(pleasure=0.7, arousal=0.0, dominance=0.0))
        assert c == (150, 200, 50)

    def test_neutral_pad_returns_amber_default(self):
        from glados_modules.SpeechLedSync import (pad_to_eye_color, GLADOS_AMBER)
        from glados_modules.MoodConsumer import PADState
        c = pad_to_eye_color(PADState(pleasure=0.0, arousal=0.0, dominance=0.6))
        assert c == GLADOS_AMBER

    def test_furious_takes_precedence_over_excited(self):
        # Both arousal > 0.7 quadrants check arousal first; pleasure sign decides
        from glados_modules.SpeechLedSync import pad_to_eye_color
        from glados_modules.MoodConsumer import PADState
        # Arousal high, pleasure -0.4 → furious (not excited)
        c = pad_to_eye_color(PADState(pleasure=-0.4, arousal=0.8, dominance=0.0))
        assert c == (255, 0, 0)


# ----------------------------------------------------------------------
# Envelope generation
# ----------------------------------------------------------------------

class TestEnvelope:
    """RMS envelope produces a sane time_dict."""

    def test_silent_wav_produces_zero_intensity(self, fake_sync):
        wav = _silent_wav(0.5)
        env = fake_sync._build_envelope(wav)
        assert all(intensity == 0.0 for intensity in env.values())
        # 0.5s at 50ms windows ≈ 10 windows (±1 for sample-count rounding)
        assert 9 <= len(env) <= 11

    def test_constant_amplitude_normalizes_to_one(self, fake_sync):
        wav = np.full(22050, 0.3, dtype=np.float32)  # 1s at constant amplitude
        env = fake_sync._build_envelope(wav)
        # All windows have identical RMS, so all normalize to 1.0
        for intensity in env.values():
            assert intensity == pytest.approx(1.0, abs=0.01)

    def test_ramp_envelope_increases_monotonically(self, fake_sync):
        wav = _ramp_wav(0.5)
        env = fake_sync._build_envelope(wav)
        # Convert to ordered list of intensities
        values = [env[ms] for ms in sorted(env.keys())]
        # Each window's intensity should be >= the previous (ramp is monotonic)
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1] - 0.01

    def test_envelope_keys_are_50ms_aligned(self, fake_sync):
        wav = _tone_wav(0.5)
        env = fake_sync._build_envelope(wav)
        for ms in env.keys():
            assert ms % 50 == 0

    def test_intensities_in_unit_range(self, fake_sync):
        wav = _tone_wav(1.0, freq=1000)
        env = fake_sync._build_envelope(wav)
        for intensity in env.values():
            assert 0.0 <= intensity <= 1.0

    def test_short_wav_still_produces_envelope(self, fake_sync):
        # 30ms wav — less than one window
        wav = _tone_wav(0.03)
        env = fake_sync._build_envelope(wav)
        assert len(env) >= 1


# ----------------------------------------------------------------------
# generate_speech_audio orchestration
# ----------------------------------------------------------------------

class TestGenerateSpeechAudio:
    """End-to-end: forwards to inner TTS + publishes speech_eye + returns WAV."""

    def test_empty_wav_skips_publish(self, fake_sync):
        fake_sync.inner_tts.generate_speech_audio.return_value = np.zeros(0, dtype=np.float32)
        result = fake_sync.generate_speech_audio("anything")
        assert result.shape == (0,)
        fake_sync.send_command.assert_not_called()

    def test_normal_wav_publishes_to_led_topic(self, fake_sync):
        fake_sync.inner_tts.generate_speech_audio.return_value = _tone_wav(0.5)
        fake_sync.generate_speech_audio("hello")
        msg, topic = _last_call(fake_sync)
        assert topic == MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value

    def test_payload_uses_speech_eye_animation(self, fake_sync):
        fake_sync.inner_tts.generate_speech_audio.return_value = _tone_wav(0.5)
        fake_sync.generate_speech_audio("hi")
        msg, _ = _last_call(fake_sync)
        body = msg[LEDHead.MSG_COMMAND_KEY.value]
        assert body[LEDHead.MSG_COMMAND_LOCATION_KEY.value] == \
               LEDHead.EYE_LED_LOCATION.value
        assert body[LEDHead.MSG_COMMAND_KEY.value] == \
               LEDHead.ANIMATION_SPEECH_EYE_KEY.value

    def test_payload_includes_time_dict_delay_color(self, fake_sync):
        fake_sync.inner_tts.generate_speech_audio.return_value = _tone_wav(0.3)
        fake_sync.generate_speech_audio("hi")
        msg, _ = _last_call(fake_sync)
        args = msg[LEDHead.MSG_COMMAND_KEY.value][
            LEDHead.MSG_COMMAND_ARGUMENTS_KEY.value]
        assert LEDHead.ARGS_KEY_TIME_DICT.value in args
        assert LEDHead.ARGS_KEY_DELAY.value in args
        assert LEDHead.ARGS_KEY_COLOR.value in args
        assert args[LEDHead.ARGS_KEY_DELAY.value] == pytest.approx(0.25)

    def test_time_dict_keys_are_strings(self, fake_sync):
        # JSON dict keys must be strings — verify the wire format is correct
        fake_sync.inner_tts.generate_speech_audio.return_value = _tone_wav(0.2)
        fake_sync.generate_speech_audio("hi")
        msg, _ = _last_call(fake_sync)
        time_dict = msg[LEDHead.MSG_COMMAND_KEY.value][
            LEDHead.MSG_COMMAND_ARGUMENTS_KEY.value][
            LEDHead.ARGS_KEY_TIME_DICT.value]
        assert all(isinstance(k, str) for k in time_dict.keys())
        # Values should be floats
        assert all(isinstance(v, float) for v in time_dict.values())

    def test_color_default_is_amber_when_no_mood_consumer(self, fake_sync):
        fake_sync.inner_tts.generate_speech_audio.return_value = _tone_wav(0.2)
        fake_sync.generate_speech_audio("hi")
        msg, _ = _last_call(fake_sync)
        color = msg[LEDHead.MSG_COMMAND_KEY.value][
            LEDHead.MSG_COMMAND_ARGUMENTS_KEY.value][
            LEDHead.ARGS_KEY_COLOR.value]
        assert color == [255, 165, 0]

    def test_returns_wav_unchanged(self, fake_sync):
        wav = _tone_wav(0.3)
        fake_sync.inner_tts.generate_speech_audio.return_value = wav
        result = fake_sync.generate_speech_audio("hi")
        np.testing.assert_array_equal(result, wav)

    def test_mqtt_failure_swallowed_audio_still_returned(self, fake_sync):
        wav = _tone_wav(0.3)
        fake_sync.inner_tts.generate_speech_audio.return_value = wav
        fake_sync.send_command.side_effect = RuntimeError("broker down")
        # Must not raise; engine plays audio regardless of LED sync failure
        result = fake_sync.generate_speech_audio("hi")
        np.testing.assert_array_equal(result, wav)
        fake_sync.logger.error.assert_called()

    def test_inner_tts_failure_propagates_naturally(self, fake_sync):
        # If inner TTS returns empty (its own failure mode), we skip publish
        # AND don't crash. Inner exceptions are not caught here — that's the
        # engine's domain.
        fake_sync.inner_tts.generate_speech_audio.return_value = np.zeros(0, dtype=np.float32)
        result = fake_sync.generate_speech_audio("hi")
        assert result.shape == (0,)
        fake_sync.send_command.assert_not_called()


# ----------------------------------------------------------------------
# PAD-driven color via MoodConsumer
# ----------------------------------------------------------------------

class TestColorFromMoodConsumer:
    """When MoodConsumer is wired, color follows current PAD."""

    def _attach_consumer(self, fake_sync, p, a, d, stale=False):
        from glados_modules.MoodConsumer import PADState
        consumer = MagicMock()
        consumer.is_stale.return_value = stale
        consumer.get_pad.return_value = PADState(p, a, d)
        fake_sync.mood_consumer = consumer
        return consumer

    def test_furious_pad_yields_red_color(self, fake_sync):
        self._attach_consumer(fake_sync, p=-0.5, a=0.8, d=0.0)
        fake_sync.inner_tts.generate_speech_audio.return_value = _tone_wav(0.2)
        fake_sync.generate_speech_audio("hi")
        msg, _ = _last_call(fake_sync)
        color = msg[LEDHead.MSG_COMMAND_KEY.value][
            LEDHead.MSG_COMMAND_ARGUMENTS_KEY.value][
            LEDHead.ARGS_KEY_COLOR.value]
        assert color == [255, 0, 0]

    def test_stale_pad_falls_back_to_amber(self, fake_sync):
        self._attach_consumer(fake_sync, p=-0.5, a=0.8, d=0.0, stale=True)
        fake_sync.inner_tts.generate_speech_audio.return_value = _tone_wav(0.2)
        fake_sync.generate_speech_audio("hi")
        msg, _ = _last_call(fake_sync)
        color = msg[LEDHead.MSG_COMMAND_KEY.value][
            LEDHead.MSG_COMMAND_ARGUMENTS_KEY.value][
            LEDHead.ARGS_KEY_COLOR.value]
        assert color == [255, 165, 0]

    def test_consumer_exception_falls_back_to_amber(self, fake_sync):
        consumer = MagicMock()
        consumer.is_stale.side_effect = RuntimeError("boom")
        fake_sync.mood_consumer = consumer
        fake_sync.inner_tts.generate_speech_audio.return_value = _tone_wav(0.2)
        fake_sync.generate_speech_audio("hi")
        msg, _ = _last_call(fake_sync)
        color = msg[LEDHead.MSG_COMMAND_KEY.value][
            LEDHead.MSG_COMMAND_ARGUMENTS_KEY.value][
            LEDHead.ARGS_KEY_COLOR.value]
        assert color == [255, 165, 0]
        fake_sync.logger.error.assert_called()


# ----------------------------------------------------------------------
# SpeechSynthesizerProtocol surface
# ----------------------------------------------------------------------

class TestProtocolSurface:
    """The wrapper must satisfy what the engine checks at attach-time."""

    def test_sample_rate_proxies_inner_tts(self, fake_sync):
        fake_sync.inner_tts.sample_rate = 24000
        assert fake_sync.sample_rate == 24000

    def test_sample_rate_changes_track_inner(self, fake_sync):
        fake_sync.inner_tts.sample_rate = 22050
        assert fake_sync.sample_rate == 22050
        fake_sync.inner_tts.sample_rate = 16000
        assert fake_sync.sample_rate == 16000
