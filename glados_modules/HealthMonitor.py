"""System health monitoring via MQTT heartbeat.

Each GLaDOS system (Pi4, Pi5, GPU server) runs a HealthMonitor that:
- Publishes thread health status every 5 seconds
- Tracks error counts per registered component
- Monitors MQTT connectivity
- Warns if peer systems go silent

Usage:
    health = HealthMonitor(broker=mqtt_connect, system_name="body_server")
    health.register("body_LR", body_LR_thread)
    health.register("IMU", imu_thread)
    health.start()

    # From any component, report errors:
    health.report_error("IMU")
"""

import socket
import time
from json import dumps, loads
from threading import Thread, Lock
from typing import Dict, Any
from collections import namedtuple

from glados_modules.MqttConnector import MQTTClient
from glados_modules.GladosEnums import MQTTEnums, LoggingEnums
from glados_modules.GlogConfig import setup_logger


class HealthMonitor(MQTTClient, Thread):
    """Publishes system health heartbeats and monitors peer systems."""

    HEARTBEAT_INTERVAL = 5.0    # seconds between heartbeats
    PEER_TIMEOUT = 15.0         # seconds before warning about silent peer

    def __init__(self, broker, system_name: str) -> None:
        self.__name__ = f"HealthMonitor_{system_name}"
        Thread.__init__(self)
        self.daemon = True
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)

        self.system_name: str = system_name
        self.hostname: str = socket.gethostname()
        self._start_time: float = time.time()

        # Registered threads to monitor
        self._threads: Dict[str, Thread] = {}
        self._error_counts: Dict[str, int] = {}
        self._health_lock = Lock()

        # Peer tracking
        self._peer_last_seen: Dict[str, float] = {}
        self._peer_status: Dict[str, Dict[str, Any]] = {}  # full heartbeat data per peer
        self._peer_warned: Dict[str, bool] = {}

        # MQTT setup
        self.health_topic = f"{MQTTEnums.SYSTEM_HEALTH_TOPIC.value}/{self.hostname}"
        self.health_sub_topic = f"{MQTTEnums.SYSTEM_HEALTH_TOPIC.value}/+"
        self.topic_handler = {
            self.health_sub_topic: self._handle_peer_heartbeat
        }

        MQTTClient.__init__(self, ip=broker.ip, port=broker.port)
        self.logger.info(f"HealthMonitor initialized for {system_name} on {self.hostname}")

    def register(self, name: str, thread: Thread) -> None:
        """Register a thread for health monitoring."""
        with self._health_lock:
            self._threads[name] = thread
            self._error_counts[name] = 0
        self.logger.debug(f"Registered thread: {name}")

    def report_error(self, name: str) -> None:
        """Increment the error count for a registered component."""
        with self._health_lock:
            if name in self._error_counts:
                self._error_counts[name] += 1

    def get_status(self) -> Dict[str, Any]:
        """Build the current health status dict."""
        with self._health_lock:
            threads = {}
            for name, thread in self._threads.items():
                threads[name] = {
                    "alive": thread.is_alive(),
                    "errors": self._error_counts.get(name, 0),
                }

        return {
            "hostname": self.hostname,
            "system": self.system_name,
            "uptime_s": round(time.time() - self._start_time, 1),
            "threads": threads,
            "mqtt_connected": self.client.is_connected(),
            "ts": time.time(),
        }

    def _handle_peer_heartbeat(self, msg) -> None:
        """Handle heartbeat from another system."""
        try:
            j_msg = loads(msg.payload.decode())
        except Exception:
            return

        peer_host = j_msg.get("hostname", "unknown")
        if peer_host == self.hostname:
            return  # ignore our own heartbeat

        self._peer_last_seen[peer_host] = time.time()
        self._peer_status[peer_host] = j_msg
        if self._peer_warned.get(peer_host, False):
            self.logger.info(f"Peer {peer_host} ({j_msg.get('system', '?')}) is back online")
            self._peer_warned[peer_host] = False

    def _check_peers(self) -> None:
        """Warn if any known peers have gone silent."""
        now = time.time()
        for peer, last_seen in self._peer_last_seen.items():
            if now - last_seen > self.PEER_TIMEOUT:
                if not self._peer_warned.get(peer, False):
                    self.logger.warning(f"Peer {peer} has not sent a heartbeat in "
                                        f"{now - last_seen:.0f}s")
                    self._peer_warned[peer] = True

    def _check_threads(self) -> None:
        """Log warnings for dead threads."""
        with self._health_lock:
            for name, thread in self._threads.items():
                if not thread.is_alive():
                    self.logger.error(f"Thread {name} is DEAD")

    def run(self) -> None:
        """Main heartbeat loop."""
        self.logger.info(f"HealthMonitor started for {self.system_name}")
        while True:
            try:
                status = self.get_status()
                self.send_command(status, self.health_topic)
                self._check_threads()
                self._check_peers()
            except Exception as e:
                self.logger.error(f"HealthMonitor error: {e}")
            time.sleep(self.HEARTBEAT_INTERVAL)
