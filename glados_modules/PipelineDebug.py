#!/usr/bin/env python3
"""Centralized pipeline debug publisher.

Publishes structured debug records to a single MQTT topic so the entire
tracking pipeline (vision -> IK -> servo -> feedback) can be observed
from one place, across all three systems.

Enable/disable at runtime via MQTT:
    topic: system/debug/control
    payload: {"enabled": true}   or   {"enabled": false}

Or enable in config:
    [FEATURES]
    pipeline_debug_enabled = true

Watch with:
    python mqtt_watch.py <broker_ip> system/debug
"""

# builtin
import time
from json import loads, dumps
from typing import Any, Dict, Optional
from uuid import uuid4

# glados imports
from glados_modules.GladosEnums import MQTTEnums, FeatureToggles
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import LoggingEnums


class PipelineDebug:
    """Lightweight MQTT debug publisher for pipeline instrumentation.

    Each system creates one instance with its system_name. Modules call
    log() at key decision points. Messages are published to system/debug
    with consistent structure for easy filtering and correlation.

    All publishing is gated behind self.enabled — when disabled (default),
    log() is a no-op with negligible overhead.

    Args:
        mqtt_client: An MQTTClient instance (has .client.publish()).
        system_name: Identifies the source system ("ai_server", "body_server", "glados").
        enabled: Initial state. Can be toggled via MQTT at runtime.
    """

    def __init__(self, mqtt_client, system_name: str, enabled: bool = False) -> None:
        self._mqtt = mqtt_client
        self._system = system_name
        self.enabled: bool = enabled
        self._topic = MQTTEnums.PIPELINE_DEBUG_TOPIC.value
        self._control_topic = MQTTEnums.PIPELINE_DEBUG_CONTROL_TOPIC.value
        self.logger = setup_logger(name=f"PipelineDebug_{system_name}",
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        # Subscribe to control topic for enable/disable
        if hasattr(mqtt_client, 'client') and mqtt_client.client is not None:
            mqtt_client.client.subscribe(self._control_topic)
            mqtt_client.client.message_callback_add(self._control_topic, self._on_control)

    def _on_control(self, client, userdata, msg) -> None:
        """Handle enable/disable commands via MQTT."""
        try:
            data = loads(msg.payload.decode())
            if "enabled" in data:
                was = self.enabled
                self.enabled = bool(data["enabled"])
                if was != self.enabled:
                    self.logger.info(f"Pipeline debug {'ENABLED' if self.enabled else 'DISABLED'}")
        except Exception:
            pass

    def log(self, module: str, stage: str, data: Dict[str, Any],
            trace_id: Optional[str] = None) -> None:
        """Publish a debug record to the MQTT debug topic.

        No-op when disabled. All values are rounded to keep messages compact.

        Args:
            module: Source module name (e.g., "MotionTrack", "Gservo_head_lr").
            stage: Pipeline stage tag (e.g., "PIX2WORLD", "IK_OUTPUT", "PHYSICS").
            data: Dict of debug values for this stage.
            trace_id: Optional trace ID for correlating across stages.
        """
        if not self.enabled:
            return

        record = {
            "system": self._system,
            "module": module,
            "stage": stage,
            "ts": round(time.time(), 4),
            "data": data,
        }
        if trace_id:
            record["trace_id"] = trace_id

        try:
            self._mqtt.client.publish(
                self._topic, dumps(record, separators=(",", ":")), qos=0)
        except Exception:
            pass  # debug publishing must never crash the pipeline
