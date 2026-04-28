"""SceneDescriber — periodic FastVLM scene description over RTSP.

Runs on the GPU box alongside MachineVision. Pulls frames from a camera RTSP
stream via the existing RtspConsumer, runs FastVLM inference, and publishes
scene descriptions on MQTT for the Pi 5 brain to consume via its
ContextBuilder. Also handles on-demand describe_request messages from the
brain's vision_look MCP tool.

Two output paths:
    * Background: when the scene changes meaningfully, publish a fresh
      description on vision/scene_description.
    * On-demand: receive a describe_request on vision/describe_request,
      capture a frame, run inference with the requested prompt, publish
      the result on vision/describe_response with the same request_id.
"""

# native imports
from json import loads
import threading
from threading import Thread, Lock
from time import sleep, time
from typing import Any, Callable, Dict, Optional

# 3rd party imports
import numpy as np
from numpy.typing import NDArray

# 3rd party imports (engine package on the GPU box)
from glados.vision.constants import VISION_DEFAULT_PROMPT, VISION_DETAIL_PROMPT
from glados.vision.fastvlm import FastVLM

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import (LoggingEnums, MQTTEnums, SceneEnums,
                                        SystemEnums)
from glados_modules.MqttConnector import MQTTClient, SceneMessageBuilder
from glados_modules.RTSPClient import RtspConsumer


class SceneDescriberException(Exception):
    pass


class SceneDescriber(Thread, MQTTClient):
    """Periodically describes a camera scene; answers on-demand requests.

    Subclasses Thread + MQTTClient like every other long-running service in
    the repo. The reader thread is owned by the embedded RtspConsumer; this
    thread only runs FastVLM inference and publishes results.

    Attributes:
        logger: Logger instance for logging.
        configFile: Configuration parser instance loaded from glog.conf.
        camera_name: Identifier published in description messages
            ("head", "left", "right").
        topic_handler: Maps MQTT topics to handler methods.
    """

    BACKGROUND_MAX_TOKENS: int = 64

    def __init__(self, config_file: Any, camera_name: str = "head") -> None:
        """Initialize the scene describer.

        Args:
            config_file: configparser instance loaded from glog.conf.
            camera_name: Camera identifier (default "head").
        """
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.configFile = config_file
        self.camera_name: str = camera_name
        self.stop: bool = False

        s_cfg = config_file[SceneEnums.CONFIG_HEAD.value]
        mqtt_c = config_file[SystemEnums.CONFIG_HEAD_MQTT.value]
        self._poll_interval: float = float(s_cfg.get(
            SceneEnums.POLL_INTERVAL_KEY.value,
            str(SceneEnums.DEFAULT_POLL_INTERVAL.value)))
        self._scene_change_threshold: float = float(s_cfg.get(
            SceneEnums.SCENE_CHANGE_THRESHOLD_KEY.value,
            str(SceneEnums.DEFAULT_SCENE_CHANGE_THRESHOLD.value)))
        self._rtsp_uri: str = s_cfg[SceneEnums.CAMERA_URI_KEY.value]
        model_dir = s_cfg.get(SceneEnums.MODEL_DIR_KEY.value, "") or None

        self.topic_handler: Dict[str, Callable] = {
            MQTTEnums.SCENE_DESCRIBE_REQUEST_TOPIC.value: self._handle_request,
        }
        MQTTClient.__init__(self,
                            ip=mqtt_c[SystemEnums.MQTT_SERVER_IP.value],
                            port=int(mqtt_c[SystemEnums.MQTT_PORT.value]))

        self._consumer: RtspConsumer = RtspConsumer(
            uri=self._rtsp_uri, location=f"scene_{camera_name}")
        self._model: FastVLM = FastVLM(model_dir)
        self._inference_lock: Lock = Lock()
        self._last_frame: Optional[NDArray[np.uint8]] = None

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _scene_change_score(self, current: NDArray[np.uint8]) -> float:
        """Return a normalized [0..1] difference between current and last frame.

        First frame and shape mismatches always score 1.0 to force inference.
        """
        if self._last_frame is None or current.shape != self._last_frame.shape:
            return 1.0
        diff = np.abs(current.astype(np.int16)
                       - self._last_frame.astype(np.int16))
        return float(np.mean(diff)) / 255.0

    def _describe(self, frame: NDArray[np.uint8], prompt: str,
                  max_tokens: int) -> Optional[str]:
        """Run FastVLM inference under a lock; swallow errors and log them."""
        with self._inference_lock:
            try:
                features = self._model.encode_image(frame)
                description = self._model.describe_from_features(
                    features, prompt=prompt, max_tokens=max_tokens)
                return description
            except Exception as e:
                self.logger.error(f"FastVLM inference failed: {e}")
                return None

    def _publish_description(self, description: str) -> None:
        """Publish a background description on vision/scene_description."""
        msg = SceneMessageBuilder.description(self.camera_name, description, time())
        self.send_command(msg, MQTTEnums.SCENE_DESCRIPTION_TOPIC.value)

    # ------------------------------------------------------------------
    # MQTT handlers
    # ------------------------------------------------------------------

    def _handle_request(self, msg: Any) -> None:
        """Handle an on-demand describe_request from the brain's vision_look tool."""
        try:
            j_msg = loads(msg.payload.decode())
        except Exception as e:
            self.logger.error(f"Failed to parse describe_request: {e}")
            return
        request_id = j_msg.get(SceneEnums.REQUEST_ID_KEY.value)
        if request_id is None:
            self.logger.warning("describe_request missing request_id")
            return
        prompt = j_msg.get(SceneEnums.PROMPT_KEY.value, VISION_DETAIL_PROMPT)
        max_tokens = int(j_msg.get(SceneEnums.MAX_TOKENS_KEY.value, 256))
        # Inference is slow; run off the MQTT callback thread so we don't
        # block the broker's read loop.
        Thread(target=self._handle_request_worker,
               args=(request_id, prompt, max_tokens), daemon=True).start()

    def _handle_request_worker(self, request_id: str, prompt: str,
                                max_tokens: int) -> None:
        """Capture a frame, run inference, publish describe_response."""
        frame = self._consumer.get_frame()
        if frame is None:
            response = "error: failed to capture frame"
        else:
            response = (self._describe(frame, prompt, max_tokens)
                        or "error: inference failed")
        msg = SceneMessageBuilder.describe_response(request_id, response)
        self.send_command(msg, MQTTEnums.SCENE_DESCRIBE_RESPONSE_TOPIC.value)

    # ------------------------------------------------------------------
    # Thread loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Poll the camera, run inference when the scene changes, publish."""
        self.logger.info(f"SceneDescriber started for camera={self.camera_name}, "
                          f"poll={self._poll_interval}s, "
                          f"threshold={self._scene_change_threshold}")
        while self.stop is False:
            frame = self._consumer.get_frame()
            if frame is None:
                sleep(self._poll_interval)
                continue
            change_score = self._scene_change_score(frame)
            if (self._last_frame is not None
                    and change_score <= self._scene_change_threshold):
                self.logger.debug(
                    f"Scene unchanged (score={change_score:.4f}), skipping inference.")
                sleep(self._poll_interval)
                continue
            self._last_frame = frame.copy()
            description = self._describe(frame,
                                          prompt=VISION_DEFAULT_PROMPT,
                                          max_tokens=self.BACKGROUND_MAX_TOKENS)
            if description:
                self._publish_description(description)
                self.logger.debug(f"Scene: {description}")
            sleep(self._poll_interval)
