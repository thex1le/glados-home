"""LatencyResponder — echoes latency probes from the brain.

Runs on the body machine (Pi 4 BodyServer) so probes from the brain (Pi 5)
traverse the same MQTT path that real LED commands take. Cheap: ~50 lines,
zero state, only fires when a probe arrives. Always-on by design — there's
no benefit to gating it behind a feature toggle.

The echo includes the responder's receive timestamp so the brain can compute
one-way latency directly (assuming NTP-synced clocks). If clocks aren't
synced, the brain falls back to round-trip / 2 which is correct on
symmetric paths.
"""

# native imports
from json import loads
from threading import Thread
from time import sleep, time
from typing import Any, Callable, Dict

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import (LoggingEnums, LatencyEnums, MQTTEnums,
                                        SystemEnums)
from glados_modules.MqttConnector import MQTTClient, LatencyMessageBuilder


class LatencyResponder(Thread, MQTTClient):
    """Echoes latency probes for the brain's speech-eye calibration probe.

    Attributes:
        logger: Logger instance for logging.
        configFile: Configuration parser instance loaded from glog.conf.
        topic_handler: Subscribes to system/latency/probe.
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

        mqtt_c = config_file[SystemEnums.CONFIG_HEAD_MQTT.value]
        self.topic_handler: Dict[str, Callable] = {
            MQTTEnums.LATENCY_PROBE_TOPIC.value: self._handle_probe,
        }
        MQTTClient.__init__(self,
                            ip=mqtt_c[SystemEnums.MQTT_SERVER_IP.value],
                            port=int(mqtt_c[SystemEnums.MQTT_PORT.value]))

    # ------------------------------------------------------------------
    # MQTT handler
    # ------------------------------------------------------------------

    def _handle_probe(self, msg: Any) -> None:
        """Echo a probe with our local receive timestamp for one-way calc."""
        recv_ts = time()  # captured BEFORE any parsing work
        try:
            j_msg = loads(msg.payload.decode())
        except Exception as e:
            self.logger.error(f"Failed to parse latency probe: {e}")
            return
        ping_id = j_msg.get(LatencyEnums.PING_ID_KEY.value)
        origin_ts = j_msg.get(LatencyEnums.ORIGIN_TS_KEY.value)
        if ping_id is None or origin_ts is None:
            self.logger.warning(f"latency probe missing required keys: {j_msg}")
            return
        echo = LatencyMessageBuilder.echo(ping_id=ping_id,
                                           origin_ts=float(origin_ts),
                                           responder_recv_ts=recv_ts)
        try:
            self.send_command(echo, MQTTEnums.LATENCY_ECHO_TOPIC.value)
        except Exception as e:
            self.logger.error(f"Failed to publish latency echo: {e}")

    # ------------------------------------------------------------------
    # Thread loop — purely for HealthMonitor.is_alive(); no real work here.
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.logger.info("LatencyResponder ready on system/latency/probe.")
        while self.stop is False:
            sleep(self.HEARTBEAT_INTERVAL)
