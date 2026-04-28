"""SpeechLedSync — wraps a TTS synthesizer to publish LED speech sync.

Step 8a-0. Restores the speech-eye sync feature that broke when the brain
took over TTS in step 4. Drop-in replacement for any
SpeechSynthesizerProtocol implementation (e.g. RemoteGladosTTS): on each
generate_speech_audio() call we:

    1. Forward to the inner TTS to get the WAV.
    2. Compute an RMS envelope in 50ms windows, normalized to peak.
    3. Read current PAD from MoodConsumer (fallback to GLaDOS amber).
    4. Map PAD to an eye color.
    5. Publish a body/led 'speech_eye' command with time_dict + delay + color.
    6. Return the WAV unchanged so the engine plays it normally.

The LED pulse fires AFTER the configured speech_led_delay_s, calibrated via
Tools/measure_latency.py to compensate for audio-pipeline vs. MQTT latency.
"""

# native imports
from threading import Thread
from time import sleep
from typing import Any, Dict, Optional, Tuple

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import (LoggingEnums, MQTTEnums, LEDHead,
                                        BrainEnums, BrainDefaults, SystemEnums)
from glados_modules.MoodConsumer import MoodConsumer, PADState
from glados_modules.MqttConnector import MQTTClient, LEDMessageBuilder


# Default eye color — matches LedHead.glados_eye for visual continuity.
GLADOS_AMBER: Tuple[int, int, int] = (255, 165, 0)


def pad_to_eye_color(pad: PADState) -> Tuple[int, int, int]:
    """Map a PAD vector to an RGB eye color.

    Mapping (in order of precedence):
        arousal > 0.7 AND pleasure < -0.3 → red (255, 0, 0)         furious
        arousal > 0.7 AND pleasure >  0.3 → cyan (0, 200, 255)      excited
        arousal < -0.5                    → dim white (80, 80, 80)  drowsy
        pleasure > 0.5                    → green-amber (150, 200, 50)
        default                           → GLaDOS amber

    Tunable on hardware — these are starting values.
    """
    if pad.arousal > 0.7 and pad.pleasure < -0.3:
        return (255, 0, 0)
    if pad.arousal > 0.7 and pad.pleasure > 0.3:
        return (0, 200, 255)
    if pad.arousal < -0.5:
        return (80, 80, 80)
    if pad.pleasure > 0.5:
        return (150, 200, 50)
    return GLADOS_AMBER


class SpeechLedSync(Thread, MQTTClient):
    """SpeechSynthesizerProtocol wrapper that publishes speech_eye on each utterance.

    Subclasses Thread + MQTTClient so it can publish on body/led using the
    repo's standard pattern. Thread.run() is an idle loop — all real work
    happens in generate_speech_audio() called by the engine's TTS thread.

    Attributes:
        logger: Logger instance.
        inner_tts: The wrapped SpeechSynthesizerProtocol implementation.
        mood_consumer: Optional MoodConsumer for PAD-driven color. None or
            stale → GLaDOS amber fallback.
        configFile: Configuration parser instance loaded from glog.conf.
    """

    HEARTBEAT_INTERVAL: float = 5.0
    ENVELOPE_WINDOW_MS: int = 50

    def __init__(self, inner_tts: Any, config_file: Any,
                 mood_consumer: Optional[MoodConsumer] = None) -> None:
        """Initialize the wrapper.

        Args:
            inner_tts: A SpeechSynthesizerProtocol implementation. We forward
                generate_speech_audio() and proxy sample_rate.
            config_file: configparser instance loaded from glog.conf.
            mood_consumer: Optional MoodConsumer for PAD color. When None or
                stale, falls back to GLaDOS amber.
        """
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.inner_tts = inner_tts
        self.mood_consumer = mood_consumer
        self.configFile = config_file
        self.stop: bool = False

        b_cfg = config_file[BrainEnums.CONFIG_HEAD.value]
        self._delay_s: float = float(b_cfg.get(
            BrainEnums.SPEECH_LED_DELAY.value,
            str(BrainDefaults.SPEECH_LED_DELAY)))

        # MQTTClient needs a topic_handler attribute even when we don't subscribe
        self.topic_handler: Dict = {}
        mqtt_c = config_file[SystemEnums.CONFIG_HEAD_MQTT.value]
        MQTTClient.__init__(self,
                            ip=mqtt_c[SystemEnums.MQTT_SERVER_IP.value],
                            port=int(mqtt_c[SystemEnums.MQTT_PORT.value]))

    # ------------------------------------------------------------------
    # SpeechSynthesizerProtocol surface
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Forward the inner TTS sample rate (protocol requirement)."""
        return self.inner_tts.sample_rate

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """Synthesize via inner TTS, publish speech_eye animation, return WAV.

        Errors in envelope computation, color lookup, or MQTT publish are
        logged but never propagated — TTS playback always proceeds even when
        the LED sync fails.
        """
        wav = self.inner_tts.generate_speech_audio(text)
        if wav.size == 0:
            return wav  # TTS failed or empty input; nothing to sync
        try:
            time_dict = self._build_envelope(wav)
            color = self._current_color()
            self._publish_speech_eye(time_dict, color)
        except Exception as e:
            self.logger.error(
                f"speech_eye publish failed (audio still plays): {e}")
        return wav

    # ------------------------------------------------------------------
    # Envelope computation
    # ------------------------------------------------------------------

    def _build_envelope(self, wav: NDArray[np.float32]) -> Dict[int, float]:
        """RMS envelope in 50ms windows, peak-normalized to [0, 1].

        Returns {ms_offset: intensity} suitable for the LedHead speech_eye
        animation's time_dict argument.
        """
        sr = self.sample_rate
        window_samples = max(1, int(sr * self.ENVELOPE_WINDOW_MS / 1000))
        windows = []
        peak_rms = 0.0
        for window_idx, start in enumerate(range(0, len(wav), window_samples)):
            chunk = wav[start:start + window_samples]
            if chunk.size == 0:
                continue
            # float64 squaring to avoid float32 saturation on loud chunks
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            ms_offset = window_idx * self.ENVELOPE_WINDOW_MS
            windows.append((ms_offset, rms))
            peak_rms = max(peak_rms, rms)
        if peak_rms <= 0:
            return {ms: 0.0 for ms, _ in windows}
        return {int(ms): float(rms / peak_rms) for ms, rms in windows}

    # ------------------------------------------------------------------
    # Color lookup
    # ------------------------------------------------------------------

    def _current_color(self) -> Tuple[int, int, int]:
        """Read latest PAD via MoodConsumer; fall back to amber if unavailable."""
        if self.mood_consumer is None:
            return GLADOS_AMBER
        try:
            if self.mood_consumer.is_stale():
                self.logger.debug(
                    "PAD stale; using amber fallback for speech_eye")
                return GLADOS_AMBER
            return pad_to_eye_color(self.mood_consumer.get_pad())
        except Exception as e:
            self.logger.error(f"Failed to read PAD; using amber: {e}")
            return GLADOS_AMBER

    # ------------------------------------------------------------------
    # MQTT publish
    # ------------------------------------------------------------------

    def _publish_speech_eye(self, time_dict: Dict[int, float],
                             color: Tuple[int, int, int]) -> None:
        """Publish a body/led speech_eye command in LedHead's expected shape.

        time_dict keys are serialized as strings by JSON; LedHead's speech_eye
        animation receives them as such and converts back to int for sorting.
        """
        payload = {
            LEDHead.MSG_COMMAND_LOCATION_KEY.value: LEDHead.EYE_LED_LOCATION.value,
            LEDHead.MSG_COMMAND_KEY.value: LEDHead.ANIMATION_SPEECH_EYE_KEY.value,
            LEDHead.MSG_COMMAND_ARGUMENTS_KEY.value: {
                # JSON dict keys must be strings — convert explicitly so the
                # wire format is unambiguous and not language-dependent.
                LEDHead.ARGS_KEY_TIME_DICT.value: {
                    str(ms): float(intensity)
                    for ms, intensity in time_dict.items()},
                LEDHead.ARGS_KEY_DELAY.value: self._delay_s,
                LEDHead.ARGS_KEY_COLOR.value: list(color),
            },
        }
        msg = LEDMessageBuilder.send_led_animation(payload)
        self.send_command(msg, MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value)

    # ------------------------------------------------------------------
    # Thread loop — purely a HealthMonitor.is_alive() perch.
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.logger.info(
            f"SpeechLedSync wired (delay={self._delay_s:.3f}s, "
            f"mood_consumer={'yes' if self.mood_consumer else 'no'}).")
        while self.stop is False:
            sleep(self.HEARTBEAT_INTERVAL)
