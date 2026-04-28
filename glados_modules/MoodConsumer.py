"""MoodConsumer — subscribes to the brain's PAD bus and holds latest state.

Step 7c-3 hardware consumer. Last-write-wins state cache for the EmotionAgent's
PAD vector (published on system/mood/pad by GladosBrain in 7c-2). Other
subsystems (MoodDriver, future dashboard panels, etc.) call get_pad() to read
the current mood without doing their own MQTT subscription.

Stale-state check: if no update arrives in mood_staleness_max_age_s (default
60s), is_stale() returns True so consumers can fall back to baseline visuals.
"""

# native imports
from collections import namedtuple
from json import loads
import threading
from threading import Thread
from time import sleep, time
from typing import Any, Callable, Dict, Optional

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import (LoggingEnums, MQTTEnums, BrainEnums,
                                        BrainDefaults, EmotionEnums,
                                        EmotionDefaults, SystemEnums)
from glados_modules.MqttConnector import MQTTClient


# Plain tuple for the held state — matches the PAD axis order used everywhere.
PADState = namedtuple("PADState", ["pleasure", "arousal", "dominance"])


class MoodConsumer(Thread, MQTTClient):
    """Subscribes to system/mood/pad and exposes the latest PAD state.

    Thread+MQTTClient like every other long-running service. The thread itself
    only exists to satisfy HealthMonitor's is_alive() check; all real work
    happens in the MQTT callback.

    Attributes:
        logger: Logger instance for logging.
        configFile: Configuration parser instance loaded from glog.conf.
        topic_handler: Maps MQTT topics to handler methods.
    """

    HEARTBEAT_INTERVAL: float = 5.0  # seconds between liveness sleeps

    def __init__(self, config_file: Any) -> None:
        """Initialize the consumer.

        Args:
            config_file: configparser instance loaded from glog.conf.
        """
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.configFile = config_file
        self.stop: bool = False

        b_cfg = config_file[BrainEnums.CONFIG_HEAD.value]
        self._staleness_max_age: float = float(b_cfg.get(
            BrainEnums.MOOD_STALENESS_MAX_AGE.value,
            str(BrainDefaults.MOOD_STALENESS_MAX_AGE)))

        # Baseline PAD — used as the initial "neutral" state until the first
        # MQTT update arrives. Pulled from [EMOTION] so it matches what the
        # brain will eventually publish.
        e_cfg = (config_file[EmotionEnums.CONFIG_HEAD.value]
                 if config_file.has_section(EmotionEnums.CONFIG_HEAD.value)
                 else {})

        def _ef(key: EmotionEnums, default: float) -> float:
            return float(e_cfg.get(key.value, str(default))
                         if hasattr(e_cfg, "get") else default)

        baseline = PADState(
            pleasure=_ef(EmotionEnums.BASELINE_PLEASURE,
                          EmotionDefaults.BASELINE_PLEASURE),
            arousal=_ef(EmotionEnums.BASELINE_AROUSAL,
                         EmotionDefaults.BASELINE_AROUSAL),
            dominance=_ef(EmotionEnums.BASELINE_DOMINANCE,
                           EmotionDefaults.BASELINE_DOMINANCE),
        )

        self._pad: PADState = baseline
        self._pad_ts: float = 0.0  # 0 means "never updated"
        self._pad_lock: threading.Lock = threading.Lock()

        mqtt_c = config_file[SystemEnums.CONFIG_HEAD_MQTT.value]
        self.topic_handler: Dict[str, Callable] = {
            MQTTEnums.MOOD_PAD_TOPIC.value: self._handle_pad,
        }
        MQTTClient.__init__(self,
                            ip=mqtt_c[SystemEnums.MQTT_SERVER_IP.value],
                            port=int(mqtt_c[SystemEnums.MQTT_PORT.value]))

    # ------------------------------------------------------------------
    # MQTT handler
    # ------------------------------------------------------------------

    def _handle_pad(self, msg: Any) -> None:
        """Store the latest PAD payload."""
        try:
            j_msg = loads(msg.payload.decode())
        except Exception as e:
            self.logger.error(f"Failed to parse PAD message: {e}")
            return
        try:
            new_pad = PADState(
                pleasure=float(j_msg["pleasure"]),
                arousal=float(j_msg["arousal"]),
                dominance=float(j_msg["dominance"]),
            )
        except (KeyError, TypeError, ValueError) as e:
            self.logger.error(f"Malformed PAD payload: {e}")
            return
        ts = float(j_msg.get("ts", time()))
        with self._pad_lock:
            self._pad = new_pad
            self._pad_ts = ts
        self.logger.debug(
            f"PAD received: P={new_pad.pleasure:+.2f} "
            f"A={new_pad.arousal:+.2f} D={new_pad.dominance:+.2f}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pad(self) -> PADState:
        """Return the latest PAD vector. Falls back to baseline before any update."""
        with self._pad_lock:
            return self._pad

    def is_stale(self, max_age_s: Optional[float] = None) -> bool:
        """Whether the latest PAD is older than max_age_s.

        Returns True before any update has been received. Consumers can use
        this to fall back to baseline visuals if the brain or broker is down.

        Args:
            max_age_s: Override the configured staleness window.
        """
        if max_age_s is None:
            max_age_s = self._staleness_max_age
        with self._pad_lock:
            ts = self._pad_ts
        if ts == 0.0:
            return True
        return (time() - ts) > max_age_s

    def get_age_s(self) -> Optional[float]:
        """Seconds since the last PAD update, or None if no update has arrived."""
        with self._pad_lock:
            ts = self._pad_ts
        if ts == 0.0:
            return None
        return time() - ts

    # ------------------------------------------------------------------
    # Thread loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Idle loop — all real work happens in the MQTT callback thread."""
        self.logger.info("MoodConsumer subscribed to system/mood/pad.")
        while self.stop is False:
            sleep(self.HEARTBEAT_INTERVAL)
