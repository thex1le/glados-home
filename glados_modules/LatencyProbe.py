"""LatencyProbe — measures MQTT + audio pipelines for speech-eye calibration.

Used by Tools/measure_latency.py to compute the right value of
[BRAIN] speech_led_delay_s. The math:

    speech_led_delay_s = audio_output_latency - mqtt_one_way_latency

So when the brain publishes the speech_eye command and submits TTS audio
simultaneously, the LED pulse and the audible speech start in sync.

MQTT measurement is exact (probe/echo over the bus). Audio measurement
requires a microphone and the existing AudioServerTx pipeline on the Pi 5,
or sounddevice loopback on a dev box.
"""

# native imports
from json import loads
from statistics import mean, median
import threading
from threading import Event, Thread
from time import sleep, time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import (LoggingEnums, LatencyEnums, MQTTEnums,
                                        SystemEnums)
from glados_modules.MqttConnector import MQTTClient, LatencyMessageBuilder


class LatencyProbeException(Exception):
    pass


def _percentile(samples: List[float], p: float) -> float:
    """Linear-interpolation percentile to keep us off scipy/numpy.percentile."""
    if not samples:
        return 0.0
    sorted_s = sorted(samples)
    if len(sorted_s) == 1:
        return sorted_s[0]
    k = (len(sorted_s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_s) - 1)
    frac = k - lo
    return sorted_s[lo] + frac * (sorted_s[hi] - sorted_s[lo])


def stats_summary(samples_ms: List[float]) -> Dict[str, float]:
    """Reduce a list of latency samples to mean/median/p95/p99/count."""
    if not samples_ms:
        return {LatencyEnums.SAMPLE_COUNT_KEY.value: 0,
                LatencyEnums.MEAN_MS_KEY.value: 0.0,
                LatencyEnums.MEDIAN_MS_KEY.value: 0.0,
                LatencyEnums.P95_MS_KEY.value: 0.0,
                LatencyEnums.P99_MS_KEY.value: 0.0}
    return {LatencyEnums.SAMPLE_COUNT_KEY.value: len(samples_ms),
            LatencyEnums.MEAN_MS_KEY.value: float(mean(samples_ms)),
            LatencyEnums.MEDIAN_MS_KEY.value: float(median(samples_ms)),
            LatencyEnums.P95_MS_KEY.value: _percentile(samples_ms, 95),
            LatencyEnums.P99_MS_KEY.value: _percentile(samples_ms, 99)}


class LatencyProbe(Thread, MQTTClient):
    """Measures MQTT one-way + audio output latency for speech-eye calibration.

    Subclass of Thread+MQTTClient like every other long-running service. The
    thread itself only exists for HealthMonitor; measurement methods are
    invoked synchronously by the CLI.

    Attributes:
        logger: Logger instance.
        configFile: Configuration parser instance loaded from glog.conf.
    """

    HEARTBEAT_INTERVAL: float = 5.0

    def __init__(self, config_file: Any) -> None:
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.configFile = config_file
        self.stop: bool = False

        # Pending probes — {ping_id: (Event, origin_ts, echo_payload)}
        self._pending: Dict[str, Tuple[Event, float, Optional[dict]]] = {}
        self._pending_lock: threading.Lock = threading.Lock()

        mqtt_c = config_file[SystemEnums.CONFIG_HEAD_MQTT.value]
        self.topic_handler: Dict[str, Callable] = {
            MQTTEnums.LATENCY_ECHO_TOPIC.value: self._handle_echo,
        }
        MQTTClient.__init__(self,
                            ip=mqtt_c[SystemEnums.MQTT_SERVER_IP.value],
                            port=int(mqtt_c[SystemEnums.MQTT_PORT.value]))

    # ------------------------------------------------------------------
    # MQTT echo handler
    # ------------------------------------------------------------------

    def _handle_echo(self, msg: Any) -> None:
        """Match an echo to its outstanding ping."""
        try:
            j_msg = loads(msg.payload.decode())
        except Exception as e:
            self.logger.error(f"Failed to parse latency echo: {e}")
            return
        ping_id = j_msg.get(LatencyEnums.PING_ID_KEY.value)
        if ping_id is None:
            return
        with self._pending_lock:
            entry = self._pending.get(ping_id)
            if entry is None:
                return  # echo arrived after timeout — discard
            event, origin_ts, _ = entry
            self._pending[ping_id] = (event, origin_ts, j_msg)
            event.set()

    # ------------------------------------------------------------------
    # MQTT pipeline measurement
    # ------------------------------------------------------------------

    def measure_mqtt(
        self,
        n_pings: int = LatencyEnums.DEFAULT_PING_COUNT.value,
        interval_s: float = LatencyEnums.DEFAULT_PING_INTERVAL_S.value,
        timeout_s: float = LatencyEnums.DEFAULT_PING_TIMEOUT_S.value,
        assume_ntp_synced: bool = True,
    ) -> Dict[str, Any]:
        """Send N probes, await echoes, return latency stats in milliseconds.

        Args:
            n_pings: How many probes to send.
            interval_s: Pause between probes (rate-limits the broker).
            timeout_s: Per-probe wait before declaring it lost.
            assume_ntp_synced: If True, compute one-way using the responder's
                receive timestamp. If False, use round_trip / 2.

        Returns:
            dict with separate stats for round_trip_ms and one_way_ms,
            plus the count of dropped probes.
        """
        round_trip_samples: List[float] = []
        one_way_samples: List[float] = []
        lost = 0

        for i in range(n_pings):
            ping_id = str(uuid.uuid4())
            origin_ts = time()
            event = Event()
            with self._pending_lock:
                self._pending[ping_id] = (event, origin_ts, None)

            payload = LatencyMessageBuilder.probe(ping_id, origin_ts)
            try:
                self.send_command(payload, MQTTEnums.LATENCY_PROBE_TOPIC.value)
            except Exception as e:
                self.logger.error(f"Failed to publish probe: {e}")
                with self._pending_lock:
                    self._pending.pop(ping_id, None)
                lost += 1
                sleep(interval_s)
                continue

            received = event.wait(timeout=timeout_s)
            with self._pending_lock:
                _, _, echo = self._pending.pop(ping_id, (None, 0.0, None))

            if not received or echo is None:
                lost += 1
                sleep(interval_s)
                continue

            recv_ts = time()
            rtt_ms = (recv_ts - origin_ts) * 1000.0
            round_trip_samples.append(rtt_ms)

            responder_recv = echo.get(LatencyEnums.RESPONDER_RECV_TS_KEY.value)
            if assume_ntp_synced and responder_recv is not None:
                one_way = (float(responder_recv) - origin_ts) * 1000.0
                # Guard against clock skew producing negative values
                if one_way >= 0:
                    one_way_samples.append(one_way)
                else:
                    one_way_samples.append(rtt_ms / 2.0)
            else:
                one_way_samples.append(rtt_ms / 2.0)

            sleep(interval_s)

        return {
            "round_trip": stats_summary(round_trip_samples),
            "one_way": stats_summary(one_way_samples),
            "lost": lost,
            "sent": n_pings,
        }

    # ------------------------------------------------------------------
    # Audio pipeline measurement
    # ------------------------------------------------------------------

    def measure_audio(
        self,
        tone_freq_hz: float = LatencyEnums.DEFAULT_TONE_FREQ_HZ.value,
        duration_s: float = LatencyEnums.DEFAULT_TONE_DURATION_S.value,
        samplerate: int = 22050,
        repetitions: int = 5,
        mic_capture_delay_ms: float = LatencyEnums.DEFAULT_MIC_CAPTURE_DELAY_MS.value,
    ) -> Dict[str, Any]:
        """Measure end-to-end audio output latency via tone playback + capture.

        Plays a short sine tone via sounddevice, captures via the default mic,
        cross-correlates the captured signal with the played tone to find
        the time offset. Subtracts a configured mic capture delay to isolate
        the output side.

        Requires sounddevice and a working mic on the same machine. Raises
        LatencyProbeException if sounddevice can't be imported (use the
        CLI's --skip-audio flag in that case).

        Args:
            tone_freq_hz: Sine wave frequency.
            duration_s: How long the tone plays.
            samplerate: Audio rate for both playback and capture.
            repetitions: Number of measurements to average.
            mic_capture_delay_ms: Estimated mic preamp + ADC delay to subtract.

        Returns:
            dict with output_latency_ms stats and the raw measured latencies.
        """
        try:
            import sounddevice as sd
        except ImportError as e:
            raise LatencyProbeException(
                f"sounddevice not installed; cannot measure audio: {e}")

        # Pre-build the tone we'll play and the reference for correlation
        n_tone = int(duration_s * samplerate)
        t = np.arange(n_tone) / samplerate
        tone = (0.5 * np.sin(2 * np.pi * tone_freq_hz * t)).astype(np.float32)
        # Capture window: tone duration + 1s slack for output+capture latency
        capture_s = duration_s + 1.0
        n_capture = int(capture_s * samplerate)

        samples_ms: List[float] = []
        for _ in range(repetitions):
            try:
                # Start recording, then play. Record for capture_s; play tone.
                recording = sd.rec(n_capture, samplerate=samplerate,
                                    channels=1, dtype="float32",
                                    blocking=False)
                sd.play(tone, samplerate=samplerate, blocking=False)
                sd.wait()  # waits for whichever finishes last
                captured = recording.flatten()
            except Exception as e:
                self.logger.error(f"Audio capture failed: {e}")
                continue

            offset_samples = self._cross_correlate(captured, tone)
            if offset_samples is None:
                self.logger.warning("Failed to detect tone in capture")
                continue
            latency_ms = (offset_samples / samplerate) * 1000.0
            # Subtract estimated mic capture delay to isolate output latency
            output_latency_ms = max(0.0, latency_ms - mic_capture_delay_ms)
            samples_ms.append(output_latency_ms)
            sleep(0.5)  # let the room settle

        return {
            "output_latency_ms": stats_summary(samples_ms),
            "raw_samples_ms": samples_ms,
            "mic_capture_delay_ms_assumed": mic_capture_delay_ms,
            "repetitions": repetitions,
            "tone_freq_hz": tone_freq_hz,
        }

    @staticmethod
    def _cross_correlate(captured: NDArray[np.float32],
                          reference: NDArray[np.float32]) -> Optional[int]:
        """Return the offset (samples) of reference in captured, or None.

        Uses simple FFT-based cross-correlation. Threshold-checks the peak so
        a silent capture doesn't produce a spurious result.
        """
        if len(captured) < len(reference):
            return None
        # Mean-subtract for cleaner correlation
        cap = captured - np.mean(captured)
        ref = reference - np.mean(reference)
        corr = np.correlate(cap, ref, mode="valid")
        peak = int(np.argmax(np.abs(corr)))
        peak_val = float(np.abs(corr[peak]))
        # Reject if the peak isn't well above background noise
        baseline = float(np.median(np.abs(corr)))
        if baseline > 0 and peak_val < 5.0 * baseline:
            return None
        return peak

    # ------------------------------------------------------------------
    # Calibration recommendation
    # ------------------------------------------------------------------

    @staticmethod
    def recommend_speech_led_delay_s(mqtt_one_way_ms: float,
                                      audio_output_latency_ms: float) -> float:
        """Compute the LED-leads-audio delay that should keep them in sync.

        The LedHead 'speech_eye' animation's `delay` argument is how long the
        Pi 4 waits AFTER receiving the command before starting the pulse.
        We want the pulse to begin when audio is audible. So:

            T_pulse = T_recv_command + delay
            T_audio = T_engine_submits + audio_output_latency_ms
            T_recv_command = T_publish_command + mqtt_one_way_ms
            assume T_publish_command == T_engine_submits

        Therefore: delay = audio_output_latency_ms - mqtt_one_way_ms

        Returns delay in seconds (clamped to >= 0).
        """
        delay_ms = audio_output_latency_ms - mqtt_one_way_ms
        return max(0.0, delay_ms / 1000.0)

    # ------------------------------------------------------------------
    # Thread loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.logger.info("LatencyProbe ready (call measure_mqtt / measure_audio).")
        while self.stop is False:
            sleep(self.HEARTBEAT_INTERVAL)
