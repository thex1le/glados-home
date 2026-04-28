"""GladosBrain — wraps the dnhkng/GLaDOS engine for use inside glados-home.

The brain runs on the Pi 5 and replaces the LLMConnector classes used by
GLaDOS.py. It owns:
    * Building and starting the underlying Glados engine instance.
    * MQTT-driven ContextBuilder sources for scene description (from the
      GPU box's SceneDescriber) and room state (from RoomStateManager).
    * On-demand scene-description requests for an MCP vision_look tool.
    * The brain-ready handshake on system/brain_ready (mirrors the
      camera-ready handshake other systems publish).

The brain is passive on STT input — GLaDOS.py keeps the STT polling loop and
calls brain.submit_text_input() so the existing local_commands prefilter and
mood/interaction hooks (register_interaction, detect_compliment) keep firing.

No ML model is loaded locally; LLM/TTS/ASR/FastVLM all live on the GPU box
and are reached over HTTP/MQTT.
"""

# native imports
from json import loads
import os
import threading
from threading import Thread
from time import sleep, time
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

# 3rd party imports
import numpy as np
from numpy.typing import NDArray
from pydantic import HttpUrl

# 3rd party imports (engine package, vendored via pip install -e ../GLaDOS)
from glados.audio_io import get_audio_system
from glados.autonomy import AutonomyConfig
from glados.autonomy.config import (EmotionConfig, HEXACOConfig, TokenConfig,
                                     AutonomyJobsConfig, HackerNewsJobConfig,
                                     WeatherJobConfig)
from glados.autonomy.emotion_state import EmotionEvent
from glados.core.engine import Glados
from glados.mcp.config import MCPServerConfig

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import (LoggingEnums, MQTTEnums, BrainEnums,
                                        BrainDefaults, EmotionEnums,
                                        EmotionDefaults, HEXACOEnums,
                                        HEXACODefaults, SceneEnums,
                                        SystemEnums, RoomStateEnums)
from glados_modules.MqttConnector import (MQTTClient, BrainMessageBuilder,
                                          MoodMessageBuilder, SceneMessageBuilder)
from glados_modules.RemoteGladosTTS import RemoteGladosTTS
from glados_modules.SpeechLedSync import SpeechLedSync


class GladosBrainException(Exception):
    pass


class _NoOpTranscriber:
    """Satisfies the engine's TranscriberProtocol without doing any ASR.

    The brain runs with input_mode="text" so SpeechListener never starts,
    but the engine still calls transcribe_file() once at startup to warm up
    the ASR model. We hand back an empty string for that warm-up call.
    """

    def transcribe_file(self, path: str) -> str:
        return ""

    def transcribe(self, audio: NDArray[np.float32]) -> str:
        return ""


class GladosBrain(Thread, MQTTClient):
    """Wraps the GLaDOS engine and bridges it to the existing MQTT pipeline.

    Subclasses Thread + MQTTClient like GladosLocal. The thread builds the
    engine, publishes the brain-ready handshake, and idles until shutdown.
    The engine runs on its own daughter thread started inside our run().

    The caller (GLaDOS.py) is responsible for STT polling and for forwarding
    each prompt via submit_text_input() — that keeps the existing
    local_commands prefilter and mood hooks intact.

    Attributes:
        logger: Logger instance for logging.
        configFile: Configuration parser instance loaded from glog.conf.
        engine: The underlying Glados engine instance (built in run()).
        topic_handler: Dictionary mapping MQTT topics to handler methods.
    """

    POLL_INTERVAL: float = 0.5
    READY_WAIT_TIMEOUT: float = 30.0

    def __init__(self, config_file: Any, mood_consumer: Any = None) -> None:
        """Initialize the brain.

        Args:
            config_file: configparser instance loaded from glog.conf.
            mood_consumer: Optional MoodConsumer instance. When provided, the
                speech-LED sync wrapper (Step 8a-0) reads PAD-derived eye
                color from it. Pass None in legacy mood mode.
        """
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__,
                                    console_logging=LoggingEnums.LOG_LEVEL_DEBUG.value)
        self.configFile = config_file
        self.mood_consumer = mood_consumer
        self.stop: bool = False
        self.engine: Optional[Glados] = None
        self._engine_thread: Optional[Thread] = None
        self._engine_ready: threading.Event = threading.Event()

        # MQTT bridge state
        self._latest_scene: Optional[str] = None
        self._latest_scene_ts: float = 0.0
        self._latest_room: Optional[Dict[str, Any]] = None
        self._scene_lock: threading.Lock = threading.Lock()
        self._room_lock: threading.Lock = threading.Lock()

        # On-demand scene requests (request_id -> Event + result slot)
        self._pending_scene: Dict[str, threading.Event] = {}
        self._scene_responses: Dict[str, str] = {}
        self._scene_pending_lock: threading.Lock = threading.Lock()

        mqtt_c = config_file[SystemEnums.CONFIG_HEAD_MQTT.value]
        ip: str = mqtt_c[SystemEnums.MQTT_SERVER_IP.value]
        port: int = int(mqtt_c[SystemEnums.MQTT_PORT.value])

        self.topic_handler: Dict[str, Callable] = {
            MQTTEnums.SCENE_DESCRIPTION_TOPIC.value: self._handle_scene,
            MQTTEnums.SCENE_DESCRIBE_RESPONSE_TOPIC.value: self._handle_scene_response,
            MQTTEnums.MOOD_EVENT_TOPIC.value: self._handle_mood_event,
            RoomStateEnums.MQTT_ROOM_TOPIC.value: self._handle_room_state,
        }
        MQTTClient.__init__(self, ip=ip, port=port)

        # ContextBuilder priorities + scene timeout — read once from glog.conf
        b_cfg = config_file[BrainEnums.CONFIG_HEAD.value]
        self._scene_priority: int = int(b_cfg.get(
            BrainEnums.SCENE_PRIORITY.value,
            str(BrainDefaults.SCENE_PRIORITY)))
        self._room_priority: int = int(b_cfg.get(
            BrainEnums.ROOM_PRIORITY.value,
            str(BrainDefaults.ROOM_PRIORITY)))
        self._scene_timeout: float = float(b_cfg.get(
            BrainEnums.SCENE_TIMEOUT.value,
            str(BrainDefaults.SCENE_TIMEOUT)))

        # Step 7c-2 PAD publisher state. Hybrid throttle: publish if any axis
        # moved > _mood_publish_delta since last publish OR if more than
        # _mood_publish_max_interval seconds have elapsed (heartbeat).
        self._mood_publish_delta: float = float(b_cfg.get(
            BrainEnums.MOOD_PUBLISH_DELTA.value,
            str(BrainDefaults.MOOD_PUBLISH_DELTA)))
        self._mood_publish_max_interval: float = float(b_cfg.get(
            BrainEnums.MOOD_PUBLISH_MAX_INTERVAL.value,
            str(BrainDefaults.MOOD_PUBLISH_MAX_INTERVAL)))
        self._last_published_pad: Optional[Tuple[float, float, float]] = None
        self._last_pad_publish_ts: float = 0.0

    # ------------------------------------------------------------------
    # Engine wiring
    # ------------------------------------------------------------------

    def _build_engine(self) -> Glados:
        """Construct the underlying Glados engine, wiring in our adapters.

        Returns:
            A configured Glados instance ready to run.

        Raises:
            GladosBrainException: If a required config value is missing or
                the personality file cannot be read.
        """
        b_cfg = self.configFile[BrainEnums.CONFIG_HEAD.value]
        d_cfg = self.configFile[SystemEnums.CONFIG_HEAD_DEFAULT.value]

        try:
            voice_url = d_cfg[SystemEnums.VOICE_URL.value]
            endpoint = b_cfg[BrainEnums.LLM_ENDPOINT.value]
            model = b_cfg[BrainEnums.LLM_MODEL.value]
        except KeyError as e:
            raise GladosBrainException(f"Missing required config key: {e}")

        api_key = b_cfg.get(BrainEnums.LLM_API_KEY.value, "") or None
        audio_io_kind = b_cfg.get(BrainEnums.AUDIO_IO.value, "sounddevice")
        interruptible = b_cfg.get(BrainEnums.INTERRUPTIBLE.value,
                                   "True").strip().lower() == "true"

        personality_path = b_cfg.get(
            BrainEnums.PERSONALITY_FILE.value,
            "./txt_responses/personality_glados.txt")
        try:
            with open(personality_path, "r", encoding="utf-8") as f:
                personality_text = f.read().strip()
        except OSError as e:
            raise GladosBrainException(
                f"Failed to read personality file {personality_path}: {e}")
        if not personality_text:
            raise GladosBrainException(
                f"Personality file {personality_path} is empty")
        personality_preprompt = ({"role": "system", "content": personality_text}, )

        autonomy_cfg = self._build_autonomy_config()

        # Step 8a-0: wrap the TTS in SpeechLedSync so each utterance fires
        # a body/led 'speech_eye' animation in sync with the audio. PAD-driven
        # color when MoodConsumer is wired; falls back to amber otherwise.
        # Wrapping is unconditional — when mood_consumer is None the wrapper
        # still works, just with the static amber color.
        # Step 8a-1: tts_sample_rate must match whatever the GPU TTS server
        # actually emits, otherwise SpeechPlayer plays at the wrong pitch.
        tts_sample_rate = int(b_cfg.get(
            BrainEnums.TTS_SAMPLE_RATE.value,
            str(BrainDefaults.TTS_SAMPLE_RATE)))
        inner_tts = RemoteGladosTTS(voice_url=voice_url,
                                     sample_rate=tts_sample_rate)
        tts_model = SpeechLedSync(
            inner_tts=inner_tts,
            config_file=self.configFile,
            mood_consumer=self.mood_consumer,
        )
        audio_io = get_audio_system(backend_type=audio_io_kind)
        engine = Glados(
            asr_model=_NoOpTranscriber(),
            tts_model=tts_model,
            audio_io=audio_io,
            completion_url=HttpUrl(endpoint),
            llm_model=model,
            api_key=api_key,
            interruptible=interruptible,
            wake_word=None,
            announcement=None,
            personality_preprompt=personality_preprompt,
            tool_config={},
            autonomy_config=autonomy_cfg,
            mcp_servers=self._mcp_server_configs(),
            input_mode="text",
            tts_enabled=True,
            asr_muted=True,
            llm_headers=None,
        )
        engine.context_builder.register("scene", self._format_scene,
                                         priority=self._scene_priority)
        engine.context_builder.register("rooms", self._format_rooms,
                                         priority=self._room_priority)
        return engine

    # ------------------------------------------------------------------
    # Engine sub-config builders (one per [SECTION] in glog.conf)
    # ------------------------------------------------------------------

    def _build_autonomy_config(self) -> AutonomyConfig:
        """Assemble the engine's AutonomyConfig from glog.conf.

        Reads [BRAIN] for the autonomy loop knobs, [EMOTION] + [HEXACO] for
        the emotion sub-agent, and uses sensible defaults for anything
        missing. Compaction is driven by the engine's TokenConfig — we expose
        compaction_enabled as a coarse on/off toggle that swaps the default
        token threshold for a very high one (effectively disabling it).

        Returns:
            A fully-populated AutonomyConfig.
        """
        b_cfg = self.configFile[BrainEnums.CONFIG_HEAD.value]
        autonomy_enabled = b_cfg.get(BrainEnums.AUTONOMY_ENABLED.value,
                                      "True").strip().lower() == "true"
        compaction_enabled = b_cfg.get(BrainEnums.COMPACTION_ENABLED.value,
                                        "True").strip().lower() == "true"
        coalesce_ticks = b_cfg.get(BrainEnums.AUTONOMY_COALESCE_TICKS.value,
                                    "True").strip().lower() == "true"

        # TokenConfig: compaction toggle is implemented by setting an absurdly
        # high threshold when disabled, so the engine's CompactionAgent never
        # fires. Keeps the engine's own contract intact.
        token_cfg = TokenConfig(
            token_threshold=8000 if compaction_enabled else 10_000_000,
        )

        return AutonomyConfig(
            enabled=autonomy_enabled,
            tick_interval_s=float(b_cfg.get(
                BrainEnums.AUTONOMY_TICK_INTERVAL.value,
                str(BrainDefaults.TICK_INTERVAL))),
            cooldown_s=float(b_cfg.get(
                BrainEnums.AUTONOMY_COOLDOWN.value,
                str(BrainDefaults.COOLDOWN))),
            autonomy_parallel_calls=int(b_cfg.get(
                BrainEnums.AUTONOMY_PARALLEL_CALLS.value,
                str(BrainDefaults.PARALLEL_CALLS))),
            coalesce_ticks=coalesce_ticks,
            jobs=self._build_jobs_config(),
            tokens=token_cfg,
            emotion=self._build_emotion_config(),
        )

    def _build_emotion_config(self) -> EmotionConfig:
        """Assemble EmotionConfig from [EMOTION] + [HEXACO] in glog.conf."""
        b_cfg = self.configFile[BrainEnums.CONFIG_HEAD.value]
        emotion_enabled = b_cfg.get(BrainEnums.EMOTION_ENABLED.value,
                                     "True").strip().lower() == "true"

        e_section = EmotionEnums.CONFIG_HEAD.value
        e_cfg = (self.configFile[e_section]
                 if self.configFile.has_section(e_section)
                 else {})

        def _ef(key: EmotionEnums, default: float) -> float:
            return float(e_cfg.get(key.value, str(default))
                         if hasattr(e_cfg, "get") else default)

        return EmotionConfig(
            enabled=emotion_enabled,
            tick_interval_s=_ef(EmotionEnums.TICK_INTERVAL,
                                EmotionDefaults.TICK_INTERVAL),
            max_events=int(e_cfg.get(EmotionEnums.MAX_EVENTS.value,
                                      str(EmotionDefaults.MAX_EVENTS))
                           if hasattr(e_cfg, "get")
                           else EmotionDefaults.MAX_EVENTS),
            baseline_pleasure=_ef(EmotionEnums.BASELINE_PLEASURE,
                                   EmotionDefaults.BASELINE_PLEASURE),
            baseline_arousal=_ef(EmotionEnums.BASELINE_AROUSAL,
                                  EmotionDefaults.BASELINE_AROUSAL),
            baseline_dominance=_ef(EmotionEnums.BASELINE_DOMINANCE,
                                    EmotionDefaults.BASELINE_DOMINANCE),
            mood_drift_rate=_ef(EmotionEnums.MOOD_DRIFT_RATE,
                                 EmotionDefaults.MOOD_DRIFT_RATE),
            baseline_drift_rate=_ef(EmotionEnums.BASELINE_DRIFT_RATE,
                                     EmotionDefaults.BASELINE_DRIFT_RATE),
            hexaco=self._build_hexaco_config(),
        )

    def _build_hexaco_config(self) -> HEXACOConfig:
        """Assemble HEXACOConfig from [HEXACO] in glog.conf."""
        h_section = HEXACOEnums.CONFIG_HEAD.value
        h_cfg = (self.configFile[h_section]
                 if self.configFile.has_section(h_section)
                 else {})

        def _hf(key: HEXACOEnums, default: float) -> float:
            return float(h_cfg.get(key.value, str(default))
                         if hasattr(h_cfg, "get") else default)

        return HEXACOConfig(
            honesty_humility=_hf(HEXACOEnums.HONESTY_HUMILITY,
                                  HEXACODefaults.HONESTY_HUMILITY),
            emotionality=_hf(HEXACOEnums.EMOTIONALITY,
                              HEXACODefaults.EMOTIONALITY),
            extraversion=_hf(HEXACOEnums.EXTRAVERSION,
                              HEXACODefaults.EXTRAVERSION),
            agreeableness=_hf(HEXACOEnums.AGREEABLENESS,
                               HEXACODefaults.AGREEABLENESS),
            conscientiousness=_hf(HEXACOEnums.CONSCIENTIOUSNESS,
                                   HEXACODefaults.CONSCIENTIOUSNESS),
            openness=_hf(HEXACOEnums.OPENNESS,
                          HEXACODefaults.OPENNESS),
        )

    def _build_jobs_config(self) -> AutonomyJobsConfig:
        """AutonomyJobsConfig — HN + weather subagents.

        Both off by default in step 7c-1; turning them on is a 7c future tweak.
        """
        return AutonomyJobsConfig(
            enabled=False,
            hacker_news=HackerNewsJobConfig(enabled=False),
            weather=WeatherJobConfig(enabled=False),
        )

    def _mcp_server_configs(self) -> List[MCPServerConfig]:
        """Build the MCP server config list.

        BodyMCPServer exposes the robot's actuators as tools the LLM can call
        (look_at, turn_body, lean, set_eye_animation, wake_eyes, set_eye_breath).
        It runs as a stdio subprocess and publishes the existing body MQTT
        topics, so BodyServer.py on the Pi 4 needs no changes.

        memory + system_info come from the engine's built-in MCP servers.
        """
        config_path = os.environ.get("GLADOS_CONFIG", "./glog.conf")
        return [
            MCPServerConfig(name="body", transport="stdio", command="python",
                            args=["-m", "glados_modules.mcp.BodyMCPServer",
                                  "-config", config_path]),
            MCPServerConfig(name="memory", transport="stdio",
                            command="python", args=["-m", "glados.mcp.memory_server"]),
            MCPServerConfig(name="system_info", transport="stdio",
                            command="python", args=["-m", "glados.mcp.system_info_server"]),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until the engine is built and accepting input.

        Args:
            timeout: Max seconds to wait. Defaults to READY_WAIT_TIMEOUT.

        Returns:
            True if the engine became ready, False on timeout.
        """
        if timeout is None:
            timeout = self.READY_WAIT_TIMEOUT
        return self._engine_ready.wait(timeout=timeout)

    def speak(self, text: str) -> None:
        """Synchronously enqueue text for the engine to speak.

        Bypasses the LLM — the text goes straight to the TTS queue. Used
        by the existing _startup_greeting in GLaDOS.py.

        Args:
            text: Text for the engine to synthesize and play.

        Raises:
            GladosBrainException: If the engine has not been started yet.
        """
        if self.engine is None:
            raise GladosBrainException("engine not started; call wait_until_ready first")
        self.engine.tts_queue.put(text)
        self.engine.processing_active_event.set()

    def submit_text_input(self, text: str, source: str = "speech") -> bool:
        """Forward a user prompt to the engine for LLM processing.

        Called by GLaDOS.py's STT loop after local_commands have been
        checked and mood/interaction hooks have fired.

        Args:
            text: User prompt to submit.
            source: Source label for engine observability (default "speech").

        Returns:
            True if the prompt was queued, False if the engine isn't ready
            or the engine itself rejected the input (e.g. empty text).
        """
        if self.engine is None:
            self.logger.warning("submit_text_input called before engine was ready")
            return False
        try:
            return bool(self.engine.submit_text_input(text, source=source))
        except Exception as e:
            self.logger.error(f"submit_text_input failed: {e}")
            return False

    def request_scene_description(self, prompt: str = "Describe what you see.",
                                   max_tokens: int = 256) -> Optional[str]:
        """Block until SceneDescriber returns a description, or timeout.

        Used by the (future) MCP vision_look tool to ask the GPU box for
        a fresh, on-demand scene description.

        Args:
            prompt: Instruction for the VLM.
            max_tokens: Max tokens to generate.

        Returns:
            The description string, or None on timeout.
        """
        request_id = str(uuid.uuid4())
        event = threading.Event()
        with self._scene_pending_lock:
            self._pending_scene[request_id] = event
        msg = SceneMessageBuilder.describe_request(request_id, prompt, max_tokens)
        self.send_command(msg, MQTTEnums.SCENE_DESCRIBE_REQUEST_TOPIC.value)
        if event.wait(timeout=self._scene_timeout):
            with self._scene_pending_lock:
                self._pending_scene.pop(request_id, None)
                return self._scene_responses.pop(request_id, None)
        with self._scene_pending_lock:
            self._pending_scene.pop(request_id, None)
            self._scene_responses.pop(request_id, None)
        self.logger.warning(f"Scene request {request_id} timed out after "
                             f"{self._scene_timeout:.1f}s")
        return None

    # ------------------------------------------------------------------
    # MQTT handlers
    # ------------------------------------------------------------------

    def _handle_scene(self, msg: Any) -> None:
        """Handle background scene descriptions from SceneDescriber on the GPU box.

        Forwards each new description into the EmotionAgent's event queue so
        mood reacts to what GLaDOS sees changing (someone walking in, the room
        going dark, etc.). The agent decides what the change means; we just
        deliver it as an event with source='vision'.
        """
        try:
            j_msg = loads(msg.payload.decode())
        except Exception as e:
            self.logger.error(f"Failed to parse scene message: {e}")
            return
        description = j_msg.get(SceneEnums.DESCRIPTION_KEY.value)
        with self._scene_lock:
            previous = self._latest_scene
            self._latest_scene = description
            self._latest_scene_ts = float(j_msg.get(SceneEnums.TS_KEY.value, time()))
        self.logger.debug(f"Scene updated: {description}")
        # Only push to emotion queue when the description actually changed —
        # SceneDescriber already gates on visual change, but identical text
        # back-to-back (cached prompt result) shouldn't move mood.
        if description and description != previous:
            self._push_emotion_event("vision", f"You see: {description}")

    # ------------------------------------------------------------------
    # PAD publisher (Step 7c-2)
    # ------------------------------------------------------------------

    def _maybe_publish_pad(self) -> None:
        """Publish current EmotionAgent PAD state if throttle rules say so.

        Hybrid throttle:
            * delta path — any of pleasure/arousal/dominance moved more
              than _mood_publish_delta since last publish.
            * heartbeat path — at least _mood_publish_max_interval seconds
              have elapsed since last publish (covers consumer crash recovery).

        Safe to call before the engine is built or when the EmotionAgent
        isn't enabled — silently no-ops in those cases.
        """
        if self.engine is None:
            return
        agent = getattr(self.engine, "_emotion_agent", None)
        if agent is None:
            return
        state = getattr(agent, "state", None)
        if state is None:
            return

        try:
            p, a, d = float(state.pleasure), float(state.arousal), float(state.dominance)
        except (AttributeError, TypeError, ValueError) as e:
            self.logger.error(f"Unable to read PAD state: {e}")
            return

        now = time()
        if self._last_published_pad is None:
            should_publish = True
            delta = 1.0
        else:
            lp, la, ld = self._last_published_pad
            delta = max(abs(p - lp), abs(a - la), abs(d - ld))
            age = now - self._last_pad_publish_ts
            should_publish = (delta > self._mood_publish_delta
                              or age > self._mood_publish_max_interval)

        if not should_publish:
            return

        msg = MoodMessageBuilder.pad(p, a, d, now)
        try:
            self.send_command(msg, MQTTEnums.MOOD_PAD_TOPIC.value)
        except Exception as e:
            self.logger.error(f"Failed to publish PAD: {e}")
            return
        self._last_published_pad = (p, a, d)
        self._last_pad_publish_ts = now
        self.logger.debug(
            f"PAD published: P={p:+.2f} A={a:+.2f} D={d:+.2f} (delta={delta:.3f})")

    def _push_emotion_event(self, source: str, description: str) -> None:
        """Push an EmotionEvent onto the engine's EmotionAgent queue.

        Safe to call before the engine is built — silently drops if the agent
        hasn't been wired yet.

        Args:
            source: Event source label (e.g., 'vision', 'user', 'system').
            description: Human-readable event description for the LLM.
        """
        if self.engine is None:
            return
        emotion_agent = getattr(self.engine, "_emotion_agent", None)
        if emotion_agent is None:
            return
        try:
            emotion_agent.push_event(
                EmotionEvent(source=source, description=description))
        except Exception as e:
            self.logger.error(f"Failed to push emotion event: {e}")

    def _handle_scene_response(self, msg: Any) -> None:
        """Handle on-demand describe_response messages, unblocking the requester."""
        try:
            j_msg = loads(msg.payload.decode())
        except Exception as e:
            self.logger.error(f"Failed to parse scene response: {e}")
            return
        request_id = j_msg.get(SceneEnums.REQUEST_ID_KEY.value)
        description = j_msg.get(SceneEnums.DESCRIPTION_KEY.value)
        if request_id is None:
            self.logger.warning("describe_response missing request_id")
            return
        with self._scene_pending_lock:
            event = self._pending_scene.get(request_id)
            if event is not None:
                self._scene_responses[request_id] = description
                event.set()

    def _handle_room_state(self, msg: Any) -> None:
        """Handle room state updates from the GPU box's RoomStateManager."""
        try:
            j_msg = loads(msg.payload.decode())
        except Exception as e:
            self.logger.error(f"Failed to parse room state: {e}")
            return
        with self._room_lock:
            self._latest_room = j_msg

    def _handle_mood_event(self, msg: Any) -> None:
        """Handle hardware-detected mood events on system/mood/event.

        Step 7c-4: closes the loop on Option C. Hardware-side detectors
        (pestering, compliments, room arrivals, gesture recognition) publish
        here; we forward each event into the EmotionAgent's queue so the
        LLM mood reacts to physical world events.
        """
        try:
            j_msg = loads(msg.payload.decode())
        except Exception as e:
            self.logger.error(f"Failed to parse mood event: {e}")
            return
        source = j_msg.get("source")
        description = j_msg.get("description")
        if not source or not description:
            self.logger.warning(
                f"mood event missing source or description: {j_msg}")
            return
        self._push_emotion_event(source, description)

    # ------------------------------------------------------------------
    # ContextBuilder formatters
    # ------------------------------------------------------------------

    def _format_scene(self) -> Optional[str]:
        """Format the latest scene description for injection into LLM context."""
        with self._scene_lock:
            if not self._latest_scene:
                return None
            age = time() - self._latest_scene_ts
            return f"[scene] (updated {age:.0f}s ago) {self._latest_scene}"

    def _format_rooms(self) -> Optional[str]:
        """Format the latest room roster for injection into LLM context."""
        with self._room_lock:
            if not self._latest_room:
                return None
            count = self._latest_room.get("count", 0)
            roster = self._latest_room.get("roster", [])
            if not roster:
                return f"[room] {count} occupant(s): no one"
            names = []
            for person in roster:
                name = person.get("face_id", "")
                if not name or name == "unknown":
                    name = person.get("person_id", "unknown")
                names.append(name)
            return f"[room] {count} occupant(s): {', '.join(names)}"

    # ------------------------------------------------------------------
    # Thread loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Build the engine, publish ready, then idle until shutdown.

        STT polling lives in GLaDOS.py so the existing local_commands
        prefilter and mood hooks keep firing. The engine runs on its own
        daughter thread started here.
        """
        try:
            self.engine = self._build_engine()
        except Exception as e:
            self.logger.error(f"Failed to build engine: {e}")
            return

        self._engine_thread = Thread(target=self.engine.run,
                                      name="GladosEngine", daemon=True)
        self._engine_thread.start()

        ready_msg = BrainMessageBuilder.ready(
            self.__name__,
            self.configFile[BrainEnums.CONFIG_HEAD.value][BrainEnums.LLM_MODEL.value])
        self.send_command(ready_msg, MQTTEnums.BRAIN_READY_TOPIC.value)
        self._engine_ready.set()
        self.logger.info("GladosBrain ready, awaiting submit_text_input from GLaDOS.py.")

        while self.stop is False:
            self._maybe_publish_pad()
            sleep(self.POLL_INTERVAL)
