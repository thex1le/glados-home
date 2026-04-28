# native imports
import logging
from typing import Dict, Callable, NamedTuple
from json import dumps, loads
from uuid import uuid4
from time import time
from threading import Lock
from collections import namedtuple

# 3rd party imports
import paho.mqtt.client as mqtt
from cachetools import TTLCache

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import (ServoEnum, CameraEnum, TrackingEnums, LoggingEnums,
                                        STTEnums, IMUEnums, TOFEnums, THEnums, MOXEnums, LEDHead,
                                        MQTTEnums, SocTempEnums, SceneEnums, LatencyEnums)


class MQTTClient:
    broker_tuple = namedtuple('broker', ["ip", "port"])

    #TODO convert this to named tuple
    def __init__(self, ip: str = 'localhost', port: int = 1883) -> None:
        self.ip = ip
        self.port = int(port)
        if not hasattr(self, 'topic_handler'):
            self.topic_handler: Dict[str, Callable] = {}
        self.uuid_cache = TTLCache(maxsize=100, ttl=60)
        try:
            self.logger = setup_logger(name=f"{self.__name__}", console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        except AttributeError:
            self.logger = setup_logger(name=f"{self.__class__.__name__}",
                                       console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
            self.__name__ = self.__class__.__name__
        self._lock = Lock()
        # Add log level control topic (all MQTT clients can receive this)
        self._log_level_topic = MQTTEnums.SYSTEM_LOG_LEVEL_TOPIC.value
        if self._log_level_topic not in self.topic_handler:
            self.topic_handler[self._log_level_topic] = self._handle_log_level
        self.client: mqtt.Client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.ip, self.port, 60)
        self.client.loop_start()

    def on_connect(self, client: mqtt.Client, userdata: object, flags: dict, rc: int) -> None:
        self.logger.debug(f"Connecting to mqtt server at {self.ip}:{self.port}")
        with self._lock:
            for topic in self.topic_handler:
                self.client.subscribe(topic, qos=1)

    def on_message(self, client: mqtt.Client, userdata: object, msg: mqtt.MQTTMessage) -> None:
        j_msg = loads(msg.payload.decode())
        uuid = j_msg.get("uuid", None)
        if uuid is None:
            self.logger.error("NO UUID IN MESSAGE")
            return
        handler = None
        with self._lock:
            if uuid not in self.uuid_cache.keys():
                self.uuid_cache[uuid] = time()
                handler = self.topic_handler.get(msg.topic)
        if handler:
            handler(msg)

    def send_command(self, command: dict | list | tuple, topic, qos: int = 1) -> None:
        """
        Generic mqtt sending function for single or multiple messages
        """
        if not isinstance(command, (tuple, list)):
            # make it an object we can iterate on
            command = (command, )
        for m in command:
            # add in uuid for message tracking and debugging
            m["uuid"] = str(uuid4())
            self.logger.debug(f"{self.__name__} sending {m} command")
            self.client.publish(topic, dumps(m), qos=qos)

    def _handle_log_level(self, msg) -> None:
        """Handle remote log level change via MQTT.

        Message format: {"module": "MotionTrack", "level": "DEBUG"}
        Use module "*" to change all loggers on this system.
        """
        try:
            j_msg = loads(msg.payload.decode())
        except Exception:
            return
        target_module = j_msg.get("module", "")
        level_str = j_msg.get("level", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
        if target_module == self.__name__ or target_module == "*":
            self.logger.setLevel(level)
            for handler in self.logger.handlers:
                handler.setLevel(level)
            self.logger.info(f"Log level changed to {level_str} (via MQTT)")


# Message builder classes: each returns a dict ready for MQTT publish.
# Add new builders here when adding new MQTT message types.
# Pattern: static methods using enum values for all keys.
class TargetMessageBuilder:
    @staticmethod
    def send_track_command_start(camera):
        return {TrackingEnums.MSG_COMMAND_KEY.value: TrackingEnums.MSG_COMMAND_START.value,
                TrackingEnums.MSG_CAMERA_KEY.value: camera}


class IMUMessageBuilder:
    @staticmethod
    def send_imu_status_message(message):
        return {IMUEnums.IMU_STATUS_KEY.value: message}


class THMessageBuilder:
    @staticmethod
    def send_th_status_message(message):
        return {THEnums.TH_STATUS_KEY.value: message}


class MoxGasMessageBuilder:
    @staticmethod
    def send_mox_status_message(message):
        return {MOXEnums.MOX_STATUS_KEY.value: message}


class TOFMessageBuilder:
    @staticmethod
    def send_tof_status_message(message):
        return {TOFEnums.TOF_STATUS_KEY.value: message}


class SocTempMessageBuilder:
    @staticmethod
    def send_soc_temp_message(message):
        return {SocTempEnums.SOC_TEMP_KEY.value: message}


class LEDMessageBuilder:
    @staticmethod
    def send_led_animation(message):
        return {LEDHead.MSG_COMMAND_KEY.value: message}


class SttMessageBuilder:
    @staticmethod
    def send_speech_to_text_message(message):
        return {STTEnums.STT_RESULTS_KEY.value: message}


class CameraMessageBuilder:
    @staticmethod
    def send_status(location, status):
        return {CameraEnum.MSG_LOCATION_KEY.value: location,
                CameraEnum.MSG_COMMAND_KEY.value: CameraEnum.MSG_COMMAND_STATUS.value,
                CameraEnum.MSG_RESULTS.value: status}

    @staticmethod
    def send_results(location, results):
        return {CameraEnum.MSG_LOCATION_KEY.value: location,
                CameraEnum.MSG_RESULTS.value: results}


class ServoMessageBuilder:
    """
    Build and return servo messages based on enums
    """
    @staticmethod
    def move(location, angle, speed):
        return {ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_MOVE.value,
                ServoEnum.MSG_LOCATION_KEY.value: location,
                ServoEnum.MSG_ANGLE.value: angle, ServoEnum.MSG_SPEED.value: speed}

    @staticmethod
    def head_up_down(angle: int, speed: int = ServoEnum.SERVO_DEFAULT_SPEED.value) -> dict:
        return ServoMessageBuilder.move(ServoEnum.LOCATION_HEAD_UP_DOWN.value, angle, speed)

    @staticmethod
    def body_left_right(angle: int, speed=ServoEnum.SERVO_DEFAULT_SPEED.value) -> dict:
        return ServoMessageBuilder.move(ServoEnum.LOCATION_BODY_LEFT_RIGHT.value, angle, speed)

    @staticmethod
    def body_up_down(angle: int, speed=ServoEnum.SERVO_DEFAULT_SPEED.value) -> dict:
        return ServoMessageBuilder.move(ServoEnum.LOCATION_BODY_UP_DOWN.value, angle, speed)

    @staticmethod
    def head_left_right(angle: int, speed=ServoEnum.SERVO_DEFAULT_SPEED.value) -> dict:
        return ServoMessageBuilder.move(ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value, angle, speed)

    @staticmethod
    def move_all(targets: dict) -> dict:
        """Build a consolidated move command for all servos in one message.

        Args:
            targets: dict of {location: {"angle": int, "speed": int}, ...}
        """
        return {ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_MOVE_ALL.value,
                ServoEnum.MSG_TARGETS.value: targets}

    @staticmethod
    def get_status(location):
        return {ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_STATUS.value,
                ServoEnum.MSG_LOCATION_KEY.value: location}

    @staticmethod
    def send_status(location, results):
        return {ServoEnum.MSG_LOCATION_KEY.value: location,
                ServoEnum.MSG_COMMAND_KEY.value: ServoEnum.MSG_COMMAND_STATUS.value,
                ServoEnum.MSG_RESULTS.value: results}


class BrainMessageBuilder:
    """Build messages emitted by the GLaDOS brain (utterances, tool-call telemetry, ready)."""

    @staticmethod
    def utterance(text: str, source: str = "llm") -> dict:
        return {"text": text, "source": source}

    @staticmethod
    def tool_call(tool: str, args: dict, lane: str = "priority") -> dict:
        return {"tool": tool, "args": args, "lane": lane}

    @staticmethod
    def ready(system_name: str, llm_model: str) -> dict:
        return {"system": system_name, "model": llm_model}


class MoodMessageBuilder:
    """Build messages for the PAD mood bus (system/mood/pad + system/mood/event).

    Step 7c-2 publishes PAD vectors from the engine's EmotionAgent.
    Step 7c-4 publishes hardware-detected events back into the agent.
    """

    @staticmethod
    def pad(pleasure: float, arousal: float, dominance: float,
            ts: float) -> dict:
        return {"pleasure": float(pleasure),
                "arousal": float(arousal),
                "dominance": float(dominance),
                "ts": float(ts)}

    @staticmethod
    def event(source: str, description: str) -> dict:
        return {"source": source, "description": description}


class LatencyMessageBuilder:
    """Build messages for the speech-eye sync latency probe."""

    @staticmethod
    def probe(ping_id: str, origin_ts: float) -> dict:
        return {LatencyEnums.PING_ID_KEY.value: ping_id,
                LatencyEnums.ORIGIN_TS_KEY.value: float(origin_ts)}

    @staticmethod
    def echo(ping_id: str, origin_ts: float, responder_recv_ts: float) -> dict:
        return {LatencyEnums.PING_ID_KEY.value: ping_id,
                LatencyEnums.ORIGIN_TS_KEY.value: float(origin_ts),
                LatencyEnums.RESPONDER_RECV_TS_KEY.value: float(responder_recv_ts)}

    @staticmethod
    def stats(pipeline: str, samples: int, mean_ms: float, median_ms: float,
              p95_ms: float, p99_ms: float) -> dict:
        return {LatencyEnums.PIPELINE_KEY.value: pipeline,
                LatencyEnums.SAMPLE_COUNT_KEY.value: int(samples),
                LatencyEnums.MEAN_MS_KEY.value: float(mean_ms),
                LatencyEnums.MEDIAN_MS_KEY.value: float(median_ms),
                LatencyEnums.P95_MS_KEY.value: float(p95_ms),
                LatencyEnums.P99_MS_KEY.value: float(p99_ms)}


class SceneMessageBuilder:
    """Build messages for the SceneDescriber (background descriptions + on-demand requests)."""

    @staticmethod
    def description(camera: str, description: str, ts: float) -> dict:
        return {SceneEnums.CAMERA_KEY.value: camera,
                SceneEnums.DESCRIPTION_KEY.value: description,
                SceneEnums.TS_KEY.value: ts}

    @staticmethod
    def describe_request(request_id: str, prompt: str, max_tokens: int = 256) -> dict:
        return {SceneEnums.REQUEST_ID_KEY.value: request_id,
                SceneEnums.PROMPT_KEY.value: prompt,
                SceneEnums.MAX_TOKENS_KEY.value: max_tokens}

    @staticmethod
    def describe_response(request_id: str, description: str) -> dict:
        return {SceneEnums.REQUEST_ID_KEY.value: request_id,
                SceneEnums.DESCRIPTION_KEY.value: description}
