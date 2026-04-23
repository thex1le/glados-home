import time
from typing import Dict, Callable, Tuple, NamedTuple, Any, Optional
from json import loads
from collections import namedtuple
from math import sin, radians, tan, atan, degrees, pi

# 3rd party imports
from paho.mqtt.client import MQTTMessage

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.MqttConnector import MQTTClient, ServoMessageBuilder
from glados_modules.MqttConsumerModules import ServoLocation, VisionTracker
from glados_modules.GladosEnums import (CameraEnum, ServoEnum, SystemEnums,
                                        TrackingEnums, VisionResultsEnum, LoggingEnums,
                                        MotionProfile, TraceEnums, KinematicsEnums,
                                        FusionEnums, BehaviorEnums, FeatureToggles,
                                        PersonalityEnums, RoomStateEnums,
                                        AttentionEnums, MQTTEnums)
from glados_modules.RobotKinematics import RobotKinematics
from glados_modules.MotionRecorder import MotionRecorder, build_frame_record
from glados_modules.TraceLog import TraceLog
from glados_modules.PipelineDebug import PipelineDebug
from glados_modules.RoomStateManager import RoomStateManager
from glados_modules.AttentionModel import AttentionModel


class SpringDamperEstimator:
    """Mirrors the Pi4 spring-damper physics to predict servo position on the GPU side.

    This runs the same math as Gservo.run() so we can estimate where each servo
    currently is without waiting for stale MQTT status messages. Periodic sync()
    calls gently correct any drift when status messages do arrive.
    """

    def __init__(self, initial_pos: float, omega: float, zeta: float) -> None:
        self.position: float = initial_pos
        self.velocity: float = 0.0
        self.target: float = initial_pos
        self.omega: float = omega
        self.zeta: float = zeta
        self.last_time: float = time.time()

    def set_target(self, target: float, omega: float = None, zeta: float = None) -> None:
        """Update the target angle (called when we send a command to Pi4)."""
        self.target = target
        if omega is not None:
            self.omega = omega
        if zeta is not None:
            self.zeta = zeta

    def get_position(self) -> float:
        """Advance the simulation to current time and return estimated position."""
        now = time.time()
        dt = min(now - self.last_time, 0.1)  # cap dt to prevent explosion after pauses
        self.last_time = now
        if dt > 0:
            accel = (self.omega ** 2) * (self.target - self.position) - 2.0 * self.zeta * self.omega * self.velocity
            self.velocity += accel * dt
            self.position += self.velocity * dt
        # Clamp to physical bounds always — catches divergence from sync or
        # external mutation, not just from the physics step above.
        self.position = max(-180.0, min(360.0, self.position))
        self.velocity = max(-1000.0, min(1000.0, self.velocity))
        return self.position

    def sync(self, reported_position: float, reported_velocity: float = 0.0) -> None:
        """Correct from MQTT status. Snap if drift is large, blend if small.

        The 70/30 blend works well for small corrections, but if the estimator
        has diverged significantly (e.g., after the spring-damper goes numerically
        unstable), the blend can't recover — 0.7 * 1,000,000 + 0.3 * 90 is still
        700,000. Snap to reported values when drift exceeds a safe threshold.
        """
        drift = abs(self.position - reported_position)
        if drift > 50.0:
            # Estimator has diverged badly — snap to reality
            self.position = reported_position
            self.velocity = reported_velocity
        else:
            self.position = 0.7 * self.position + 0.3 * reported_position
            self.velocity = 0.7 * self.velocity + 0.3 * reported_velocity


class CameraFusionState:
    """Tracks side camera detections and provides base position for tracking.

    Side cameras ALWAYS drive base position (LR + UD). The head camera
    computes a small clamped refinement offset. No state machine —
    side cameras are always authoritative about person position.
    """

    def __init__(self) -> None:
        self.logger = setup_logger(name="CameraFusionState",
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.head_contributing: bool = False
        self._head_last_seen: float = 0.0
        self._head_count: int = 0
        self._left_last_seen: float = 0.0
        self._right_last_seen: float = 0.0
        self._left_world_lr: float = 0.0
        self._right_world_lr: float = 0.0
        self._left_world_ud: float = None
        self._right_world_ud: float = None
        self._left_count: int = 0
        self._right_count: int = 0
        # Angle history for predictive rotation
        self._left_angle_history: list = []
        self._right_angle_history: list = []

    @property
    def state(self) -> str:
        """Backward-compatible state string for diagnostics and recordings."""
        if self.head_contributing:
            return FusionEnums.STATE_REFINEMENT.value
        return FusionEnums.STATE_SIDE_ONLY.value

    def update_side_detection(self, camera: str, world_lr: float, count: int,
                              world_ud: float = None) -> None:
        """Record a side camera detection and angle history for prediction."""
        now = time.time()
        window = FusionEnums.PREDICTION_HISTORY_WINDOW.value
        if camera == CameraEnum.CAMERA_LEFT.value:
            self._left_last_seen = now
            self._left_world_lr = world_lr
            self._left_count = count
            if world_ud is not None:
                self._left_world_ud = world_ud
            self._left_angle_history.append((now, world_lr))
            self._left_angle_history = [(t, a) for t, a in self._left_angle_history
                                         if now - t <= window]
        elif camera == CameraEnum.CAMERA_RIGHT.value:
            self._right_last_seen = now
            self._right_world_lr = world_lr
            self._right_count = count
            if world_ud is not None:
                self._right_world_ud = world_ud
            self._right_angle_history.append((now, world_lr))
            self._right_angle_history = [(t, a) for t, a in self._right_angle_history
                                          if now - t <= window]

    def update_head_count(self, count: int) -> None:
        """Record head camera person count for room-level awareness."""
        self._head_count = count

    def get_best_side_world_lr(self) -> float:
        """Return the most recent non-stale side camera world angle, or None."""
        now = time.time()
        staleness = FusionEnums.SIDE_CAMERA_STALENESS.value
        left_fresh = (now - self._left_last_seen) < staleness if self._left_last_seen > 0 else False
        right_fresh = (now - self._right_last_seen) < staleness if self._right_last_seen > 0 else False

        if left_fresh and right_fresh:
            if self._left_last_seen >= self._right_last_seen:
                return self._left_world_lr
            return self._right_world_lr
        elif left_fresh:
            return self._left_world_lr
        elif right_fresh:
            return self._right_world_lr
        return None

    def get_best_side_world_ud(self) -> float:
        """Return the most recent non-stale side camera world UD estimate, or None."""
        now = time.time()
        staleness = FusionEnums.SIDE_CAMERA_STALENESS.value
        left_fresh = (now - self._left_last_seen) < staleness if self._left_last_seen > 0 else False
        right_fresh = (now - self._right_last_seen) < staleness if self._right_last_seen > 0 else False

        if left_fresh and right_fresh:
            if self._left_last_seen >= self._right_last_seen:
                return self._left_world_ud
            return self._right_world_ud
        elif left_fresh:
            return self._left_world_ud
        elif right_fresh:
            return self._right_world_ud
        return None

    def get_best_side_camera(self) -> str:
        """Return the camera name that produced the best side detection, or None."""
        now = time.time()
        staleness = FusionEnums.SIDE_CAMERA_STALENESS.value
        left_fresh = (now - self._left_last_seen) < staleness if self._left_last_seen > 0 else False
        right_fresh = (now - self._right_last_seen) < staleness if self._right_last_seen > 0 else False

        if left_fresh and right_fresh:
            if self._left_last_seen >= self._right_last_seen:
                return CameraEnum.CAMERA_LEFT.value
            return CameraEnum.CAMERA_RIGHT.value
        elif left_fresh:
            return CameraEnum.CAMERA_LEFT.value
        elif right_fresh:
            return CameraEnum.CAMERA_RIGHT.value
        return None

    def get_predicted_world_lr(self, camera: str) -> float:
        """Predict where the side camera target will be based on angular velocity."""
        if camera == CameraEnum.CAMERA_LEFT.value:
            history = self._left_angle_history
            current = self._left_world_lr
        elif camera == CameraEnum.CAMERA_RIGHT.value:
            history = self._right_angle_history
            current = self._right_world_lr
        else:
            return 0.0

        if len(history) < 2:
            return current

        oldest_t, oldest_a = history[0]
        latest_t, latest_a = history[-1]
        dt = latest_t - oldest_t
        if dt <= 0.05:
            return current

        velocity = (latest_a - oldest_a) / dt
        if abs(velocity) < FusionEnums.PREDICTION_MIN_VELOCITY.value:
            return current

        offset = velocity * FusionEnums.PREDICTION_LEAD_TIME.value
        max_offset = FusionEnums.PREDICTION_MAX_OFFSET.value
        offset = max(-max_offset, min(max_offset, offset))
        return latest_a + offset

    def is_confirmed_by_side(self, head_world_lr: float) -> bool:
        """Check if any non-stale side camera agrees with head's world angle."""
        threshold = FusionEnums.HANDOFF_AGREEMENT_THRESHOLD.value
        best_side = self.get_best_side_world_lr()
        if best_side is None:
            return False
        return abs(head_world_lr - best_side) <= threshold

    def get_room_person_count(self) -> int:
        """Rough estimate of total people visible across all cameras."""
        return max(self._head_count, self._left_count + self._right_count)


class MotionTrack(MQTTClient):
    """World-space motion tracking with spring-damper servo coordination.

    Converts pixel detections to absolute world-space angles, then computes
    head and body servo targets such that:
    - Head locks onto target immediately (fast spring on Pi4)
    - Body swings in behind simultaneously (slow spring on Pi4)
    - Head naturally re-centers as body catches up
    - All targets sent in one MQTT message for simultaneous motion
    """
    broker_tuple = MQTTClient.broker_tuple
    camera_tuple = namedtuple("cam_resolution", ['x', 'y'])

    def __init__(
        self,
        broker: NamedTuple,
        camera_resolution: NamedTuple,
        target: str = "person",
        pose_target: str = VisionResultsEnum.VISION_POSE_KEY_POINTS_COCO_WHOLE_BODY.value[0],
        confidence: float = 0.65,
        config=None,
    ) -> None:
        self.__name__ = self.__class__.__name__
        self.location = self.__name__
        self.logger = setup_logger(self.__name__, console_logging=LoggingEnums.LOG_LEVEL_DEBUG.value)

        # MQTT topics
        self.cmd_topic: str = TrackingEnums.MQTT_COMMAND_TOPIC.value
        self.cmd_trigger: str = TrackingEnums.MSG_COMMAND_KEY.value
        self.intensity_topic: str = SystemEnums.MQTT_INTENSITY_TOPIC.value
        self.count = VisionResultsEnum.VISION_RESULTS_COUNT_KEY.value
        self.intensity: Tuple[float, float] = (.1, .1)

        self.topic_handler: Dict[str, Callable] = {
            self.cmd_topic: self.handle_cmd,
            self.intensity_topic: self.handle_intensity,
            MQTTEnums.PERSONALITY_MODIFIER_TOPIC.value: self._handle_personality_modifier,
            MQTTEnums.ATTENTION_CONVERSATION_TOPIC.value: self._handle_conversation_partner,
        }

        # Head camera resolution
        self.cam_x: int = int(camera_resolution.x)
        self.cam_y: int = int(camera_resolution.y)

        # Camera references
        self.main_camera: str = CameraEnum.CAMERA_HEAD.value
        self.left_camera: str = CameraEnum.CAMERA_LEFT.value
        self.right_camera: str = CameraEnum.CAMERA_RIGHT.value

        # Servo name references
        self.head_LR_name: str = ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
        self.head_UD_name: str = ServoEnum.LOCATION_HEAD_UP_DOWN.value
        self.body_LR_name: str = ServoEnum.LOCATION_BODY_LEFT_RIGHT.value
        self.body_UD_name: str = ServoEnum.LOCATION_BODY_UP_DOWN.value

        # Tracking config
        self.target = target
        self.pose_target = pose_target
        self.confidence = confidence
        self.objects = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
        self.dms: int = MotionProfile.DEFAULT_TRACKING_SPEED.value

        super().__init__(ip=broker.ip, port=broker.port)

        # Servo status tracker (for initial calibration and IMU)
        self.servo_status = ServoLocation(broker)

        # Vision tracker (triggers track_loop on each detection)
        self.vision_tracker = VisionTracker(broker=broker, target=self.target,
                                            confidence=self.confidence, tracker_callback=self.track_loop)

        # Camera readiness gate: hold all movement until every camera has reported
        # at least one detection. Prevents chasing phantoms during the multi-minute
        # camera startup sequence. Times out after 30s from FIRST detection (not
        # from init, since model loading and camera boot can take minutes).
        self._cameras_ready: set = set()
        self._all_cameras = {self.main_camera, self.left_camera, self.right_camera}
        self._cameras_online = False
        self._cameras_gate_start: float = 0.0  # set on first detection, not init
        self._cameras_gate_timeout: float = 30.0
        self._diagnostic_cache: Dict[str, Any] = {}
        self._eye_ud_offset: float = MotionProfile.EYE_UD_OFFSET.value

        # Idle state tracking
        self._last_target_time: float = time.time()
        self._idle_active: bool = False
        self._idle_interval: float = MotionProfile.IDLE_DRIFT_INTERVAL.value
        self._last_idle_send: float = 0.0
        self._idle_timeout: float = MotionProfile.IDLE_TIMEOUT.value

        # Spring-damper estimators (created after first servo status arrives)
        self._estimators: Dict[str, SpringDamperEstimator] = {}
        self._estimators_initialized: bool = False
        self._servo_middles: Dict[str, float] = {}
        self._servo_mins: Dict[str, float] = {}
        self._servo_maxs: Dict[str, float] = {}

        # Camera fusion state machine
        self._fusion = CameraFusionState()
        self._side_world_lr_smooth: float = None
        self._side_drive_last_send: float = 0.0  # throttle side drive to Pi4's 50Hz physics rate

        # IK rate limiting (prevents body target oscillation between solver minima)
        self._prev_body_lr_target: float = None
        self._prev_body_ud_target: float = None

        # Last tracked target state for multi-person selection
        self._last_tracked_world_lr: float = None
        self._last_tracked_bbox_height: float = None
        self._last_tracked_face_id: str = None
        # Nose/bbox hysteresis — prevent rapid switching between nose keypoint and
        # bbox center which causes 10-20° angle jumps per switch.
        self._nose_miss_count: int = 0
        self._nose_min_confidence: float = 0.4
        self._nose_miss_threshold: int = 3
        # Last known positions for memory glances (face_id -> world_lr)
        self._last_known_positions: Dict[str, float] = {}
        # Smoothed head camera refinement offset — decays toward zero when
        # no head detection, preventing jitter from sporadic detections
        self._refinement_lr: float = 0.0
        self._refinement_ud: float = 0.0

        # Feature toggles (read from config, default to True)
        def _toggle(section: str, key: str, fallback: bool = True) -> bool:
            if config is None:
                return fallback
            return config.get(section, key, fallback=str(fallback)).strip().lower() == "true"

        self._enable_predictive = _toggle(FeatureToggles.CONFIG_HEAD.value,
                                           FeatureToggles.PREDICTIVE_ROTATION.value)
        self._enable_confirmation = _toggle(FeatureToggles.CONFIG_HEAD.value,
                                             FeatureToggles.PERIPHERAL_CONFIRMATION.value)
        self._enable_glances = _toggle(FeatureToggles.CONFIG_HEAD.value,
                                        FeatureToggles.MEMORY_GLANCES.value)
        self._enable_breathing = _toggle(PersonalityEnums.CONFIG_HEAD.value,
                                          PersonalityEnums.BREATHING_ENABLED.value)
        self._enable_sleep = _toggle(PersonalityEnums.CONFIG_HEAD.value,
                                      PersonalityEnums.SLEEP_ENABLED.value)
        self._enable_movement = _toggle(FeatureToggles.CONFIG_HEAD.value,
                                         FeatureToggles.MOVEMENT_ENABLED.value)
        self._enable_idle_drift = _toggle(FeatureToggles.CONFIG_HEAD.value,
                                           FeatureToggles.IDLE_DRIFT_ENABLED.value)
        self._enable_room_state = _toggle(FeatureToggles.CONFIG_HEAD.value,
                                           FeatureToggles.ROOM_STATE_ENABLED.value)

        # Room state manager (persistent room roster across frames/cameras)
        self._room_state = RoomStateManager() if self._enable_room_state else None

        self._enable_attention = _toggle(FeatureToggles.CONFIG_HEAD.value,
                                          FeatureToggles.ATTENTION_MODEL_ENABLED.value)
        # Attention model (priority-based target selection from room roster)
        self._attention = AttentionModel() if (self._enable_attention and self._enable_room_state) else None
        self._last_attention_time: float = time.time()

        # Behavior state machine (active → idle → drowsy → asleep)
        self._behavior_state: str = BehaviorEnums.STATE_ACTIVE.value
        self._idle_start_time: float = 0.0
        self._drowsy_start_time: float = 0.0

        # Breathing parameters
        self._breathing_freq: float = MotionProfile.BREATHING_FREQ.value * 2 * pi
        self._breathing_amplitude: float = MotionProfile.BREATHING_AMPLITUDE.value
        self._sway_lr_freq: float = MotionProfile.SWAY_LR_FREQ.value * 2 * pi
        self._sway_lr_amplitude: float = MotionProfile.SWAY_LR_AMPLITUDE.value
        self._sway_head_lr_amplitude: float = MotionProfile.SWAY_HEAD_LR_AMPLITUDE.value

        # Motion recording (set to None to disable, or call enable_recording())
        self._recorder: MotionRecorder = None

        # Pipeline tracing
        self._tracer = TraceLog()

        # Pipeline debug (structured MQTT debug topic)
        _debug_enabled = _toggle(FeatureToggles.CONFIG_HEAD.value,
                                  FeatureToggles.PIPELINE_DEBUG_ENABLED.value, fallback=False)
        self._pdebug = PipelineDebug(self, "ai_server", enabled=_debug_enabled)

        # Debug overlay state (read by MachineVision for RTSP stream annotation)
        self._debug_overlay: Dict[str, Any] = {
            "state": "INIT",
            "world_lr": 0.0,
            "world_ud": 0.0,
            "head_lr": 0.0,
            "head_ud": 0.0,
            "body_lr": 0.0,
            "body_ud": 0.0,
            "est_head_lr": 0.0,
            "est_head_ud": 0.0,
            "est_body_lr": 0.0,
            "est_body_ud": 0.0,
        }
        self.debug_overlay_enabled: bool = True

    def enable_recording(self, session_name: str = None, output_dir: str = "./recordings") -> str:
        """Enable motion frame recording. Returns the recording file path.

        Captures the current smoothing and IK rate-limit state so replay
        can restore it for deterministic frame-1 output.
        """
        initial_state = {
            "smooth_lr": self._fusion.get_best_side_world_lr(),
            "smooth_ud": self._fusion.get_best_side_world_ud(),
            "prev_body_lr_target": self._prev_body_lr_target,
            "prev_body_ud_target": self._prev_body_ud_target,
        }
        self._recorder = MotionRecorder(session_name=session_name, output_dir=output_dir,
                                         initial_state=initial_state)
        self.logger.info(f"Motion recording enabled: {self._recorder.filepath}")
        return self._recorder.filepath

    def disable_recording(self) -> None:
        """Stop recording and close the file."""
        if self._recorder:
            self._recorder.close()
            self.logger.info(f"Motion recording stopped: {self._recorder.frame_count} frames")
            self._recorder = None

    def _get_estimator_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Snapshot all estimator states for recording."""
        snapshot = {}
        for name, est in self._estimators.items():
            snapshot[name] = {
                "position": round(est.position, 4),
                "velocity": round(est.velocity, 4),
                "target": round(est.target, 4),
            }
        return snapshot

    def _update_diagnostic_cache(self) -> None:
        """Build and cache a diagnostic snapshot from the MQTT callback thread.

        Called from track_loop() so all estimator reads happen in the same
        thread that mutates them. MachineVision's tracker threads read the
        cached snapshot via get_diagnostic_snapshot() without touching
        estimator state directly.

        Must be robust to partially-initialized state — track_loop can fire
        before all attributes are set up.
        """
        try:
            diag: Dict[str, Any] = {}

            if hasattr(self, '_estimators_initialized') and self._estimators_initialized:
                snapshot = {}
                for name, est in self._estimators.items():
                    snapshot[name] = {
                        "position": round(est.position, 4),
                        "velocity": round(est.velocity, 4),
                        "target": round(est.target, 4),
                    }
                diag["estimators"] = snapshot
            else:
                diag["estimators"] = {}

            diag["head_cooldown_remaining"] = 0

            diag["cameras_online"] = getattr(self, '_cameras_online', False)
            diag["cameras_ready"] = list(getattr(self, '_cameras_ready', set()))
            diag["fusion_state"] = self._fusion.state if hasattr(self, '_fusion') and self._fusion else ""
            side_lr = self._fusion.get_best_side_world_lr() if hasattr(self, '_fusion') and self._fusion else None
            side_ud = self._fusion.get_best_side_world_ud() if hasattr(self, '_fusion') and self._fusion else None
            diag["world_lr"] = round(side_lr, 2) if side_lr is not None else None
            diag["world_ud"] = round(side_ud, 2) if side_ud is not None else None

            if hasattr(self, '_attention') and self._attention:
                diag["attention"] = self._attention.get_state()
            else:
                diag["attention"] = {}

            # IMU data — SensorTracker is written by its own MQTT thread, but
            # dict reads are atomic in CPython (GIL) so this is safe
            if hasattr(self, 'servo_status') and hasattr(self.servo_status, 'st'):
                imu_status = self.servo_status.st.get_sensor_status("imu_status")
                if imu_status:
                    diag["imu"] = {
                        "euler": list(imu_status["euler"]) if imu_status.get("euler") else None,
                        "gyroscope": list(imu_status["gyroscope"]) if imu_status.get("gyroscope") else None,
                        "linear_accel": list(imu_status["linear"]) if imu_status.get("linear") else None,
                        "quaternion": list(imu_status["quaternion"]) if imu_status.get("quaternion") else None,
                        "calibration": list(imu_status["calibration_status"]) if imu_status.get("calibration_status") else None,
                        "temperature": imu_status.get("temperature"),
                        "ts": imu_status.get("time"),
                    }
                else:
                    diag["imu"] = {}
            else:
                diag["imu"] = {}

            self._diagnostic_cache = diag
        except Exception as e:
            self.logger.error(f"Diagnostic cache update failed: {e}")

    def get_diagnostic_snapshot(self) -> Dict[str, Any]:
        """Return the latest diagnostic snapshot for recording.

        Thread-safe: returns a cached dict built by _update_diagnostic_cache()
        in the MQTT callback thread. Never reads estimator state directly.

        Returns:
            Dict with estimator state, IMU data, motion gate status, etc.
        """
        return getattr(self, '_diagnostic_cache', {})

    def _init_estimators(self) -> bool:
        """Initialize spring-damper estimators from first servo status.
        Returns True if already initialized or successfully initialized now.
        """
        if self._estimators_initialized:
            return True

        angle_map = self.servo_status.get_angle_map()
        if not angle_map or len(angle_map) < 4:
            return False

        default_speed = self.dms
        head_omega, head_zeta = MotionProfile.HEAD_PARAMS.value[default_speed]
        body_omega, body_zeta = MotionProfile.BODY_PARAMS.value[default_speed]

        for name in (self.head_LR_name, self.head_UD_name, self.body_LR_name, self.body_UD_name):
            servo_data = angle_map[name]
            pos = float(servo_data.current)
            self._servo_middles[name] = float(servo_data.middle)
            self._servo_mins[name] = float(servo_data.min)
            self._servo_maxs[name] = float(servo_data.max)
            is_head = name in (self.head_LR_name, self.head_UD_name)
            omega = head_omega if is_head else body_omega
            zeta = head_zeta if is_head else body_zeta
            self._estimators[name] = SpringDamperEstimator(pos, omega, zeta)

        self._estimators_initialized = True

        # Initialize forward/inverse kinematics with servo config
        self._kinematics = RobotKinematics(self._servo_middles, self._servo_mins, self._servo_maxs)

        self.logger.info("Spring-damper estimators and kinematics initialized from servo status")
        return True

    def _sync_estimators_from_status(self) -> None:
        """Sync estimator positions from latest MQTT servo status (soft correction)."""
        if not self._estimators_initialized:
            return
        angle_map = self.servo_status.body_map
        for name, estimator in self._estimators.items():
            if name in angle_map:
                servo_data = angle_map[name]
                pre_pos = estimator.position
                pre_vel = estimator.velocity
                reported_pos = float(servo_data.current)
                reported_vel = float(servo_data.velocity)
                estimator.sync(reported_pos, reported_vel)
                drift = abs(pre_pos - reported_pos)
                if drift > 1.0:
                    self.logger.debug(
                        f"SYNC: {name} est={pre_pos:.1f} reported={reported_pos:.1f} "
                        f"drift={drift:.1f} vel_est={pre_vel:.2f} vel_rep={reported_vel:.2f} "
                        f"-> corrected={estimator.position:.1f}")
                    self._pdebug.log("MotionTrack", "SYNC", {
                        "servo": name, "est": round(pre_pos, 1),
                        "reported": round(reported_pos, 1), "drift": round(drift, 1),
                        "vel_est": round(pre_vel, 2), "vel_rep": round(reported_vel, 2),
                        "corrected": round(estimator.position, 1),
                    })

    def _get_estimated_position(self, servo_name: str) -> float:
        """Get the current estimated position from the spring-damper estimator."""
        if servo_name in self._estimators:
            return self._estimators[servo_name].get_position()
        return self._servo_middles.get(servo_name, MotionProfile.DEFAULT_SERVO_CENTER.value)

    def _compute_pose_correction(self, pose_data: dict) -> Optional[tuple]:
        """Compute world-angle correction from partial body keypoints.

        When the head camera sees part of a person but not the face, the
        visible keypoints tell us which direction the face is. If we see
        legs but no face, the face is above → look up. If we see the left
        side but not the right, the person extends right → look right.

        Args:
            pose_data: Dict of keypoint name → {x, y, confidence, location}.

        Returns:
            Tuple of (lr_correction, ud_correction) in degrees, or None
            if not enough keypoints are visible for a reliable correction.
        """
        min_conf = 0.3
        upper_kps = {"Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear"}
        lower_kps = {"Left Hip", "Right Hip", "Left Knee", "Right Knee",
                     "Left Ankle", "Right Ankle"}

        visible = []
        upper_visible = 0
        lower_visible = 0
        for name, kp in pose_data.items():
            if not isinstance(kp, dict):
                continue
            if kp.get("confidence", 0) >= min_conf:
                visible.append((kp.get("x", 0), kp.get("y", 0), name))
                if name in upper_kps:
                    upper_visible += 1
                if name in lower_kps:
                    lower_visible += 1

        if len(visible) < 3:
            return None

        avg_x = sum(v[0] for v in visible) / len(visible)
        avg_y = sum(v[1] for v in visible) / len(visible)

        # Vertical correction
        ud_correction = 0.0
        if lower_visible >= 2 and upper_visible == 0:
            # See legs, no face → face is above → look UP (negative world_ud)
            offset = (avg_y - self.cam_y / 2) / (self.cam_y / 2)
            ud_correction = -abs(offset) * MotionProfile.POSE_CORRECTION_SCALE_UD.value
        elif upper_visible >= 2 and lower_visible == 0:
            # See face near edge, no legs → might need to look DOWN
            offset = (self.cam_y / 2 - avg_y) / (self.cam_y / 2)
            ud_correction = abs(offset) * MotionProfile.POSE_CORRECTION_SCALE_UD.value

        # Horizontal correction — only if significantly off-center
        lr_correction = 0.0
        x_offset = (self.cam_x / 2 - avg_x) / (self.cam_x / 2)
        if abs(x_offset) > 0.2:
            lr_correction = x_offset * MotionProfile.POSE_CORRECTION_SCALE_LR.value

        if abs(ud_correction) > 0.5 or abs(lr_correction) > 0.5:
            self.logger.debug(
                f"POSE_CORRECTION: upper={upper_visible} lower={lower_visible} "
                f"visible={len(visible)} avg=({avg_x:.0f},{avg_y:.0f}) "
                f"lr_corr={lr_correction:.1f} ud_corr={ud_correction:.1f}")
            return (lr_correction, ud_correction)

        return None

    def _compute_bbox_edge_correction(self, bbox: dict) -> Optional[tuple]:
        """Compute world-angle correction from bounding box edge clipping.

        When a person's bbox touches a frame edge, they extend beyond the
        frame in that direction. This is a less precise fallback when pose
        keypoints aren't available.

        Args:
            bbox: Bounding box dict with x1, y1, x2, y2.

        Returns:
            Tuple of (lr_correction, ud_correction) in degrees, or None
            if the bbox isn't clipped on any edge.
        """
        edge_margin = 10.0  # pixels — consider "clipped" if within this margin of edge
        x1 = bbox.get("x1", 0)
        y1 = bbox.get("y1", 0)
        x2 = bbox.get("x2", 0)
        y2 = bbox.get("y2", 0)

        clipped_top = y1 < edge_margin
        clipped_bottom = y2 > (self.cam_y - edge_margin)
        clipped_left = x1 < edge_margin
        clipped_right = x2 > (self.cam_x - edge_margin)

        # Only correct if ONE side is clipped but not the opposite
        # (both sides clipped = person fills frame, no directional info)
        ud_correction = 0.0
        if clipped_top and not clipped_bottom:
            # Person extends above frame → look UP
            ud_correction = -MotionProfile.POSE_CORRECTION_SCALE_UD.value * 0.5
        elif clipped_bottom and not clipped_top:
            # Person extends below frame → look DOWN
            ud_correction = MotionProfile.POSE_CORRECTION_SCALE_UD.value * 0.5

        lr_correction = 0.0
        if clipped_left and not clipped_right:
            # Person extends left → look LEFT
            lr_correction = MotionProfile.POSE_CORRECTION_SCALE_LR.value * 0.5
        elif clipped_right and not clipped_left:
            # Person extends right → look RIGHT
            lr_correction = -MotionProfile.POSE_CORRECTION_SCALE_LR.value * 0.5

        if abs(ud_correction) > 0.1 or abs(lr_correction) > 0.1:
            self.logger.debug(
                f"BBOX_EDGE_CORRECTION: top={clipped_top} bot={clipped_bottom} "
                f"left={clipped_left} right={clipped_right} "
                f"lr_corr={lr_correction:.1f} ud_corr={ud_correction:.1f}")
            return (lr_correction, ud_correction)

        return None

    def _pixel_to_world_angle(self, bbox: dict, camera: str, axis: str, point: bool = False) -> float:
        """Convert a pixel detection to an absolute world-space angle.

        World angle = estimated camera world pointing direction + pixel offset in degrees.
        Uses forward kinematics for the head camera to properly account for
        non-orthogonal joint axes (eliminates cross-axis coupling errors).
        """
        # Determine pixel axis size
        if axis == ServoEnum.X_AXIS.value:
            axis_size = float(self.cam_x)
        else:
            axis_size = float(self.cam_y)

        # Get target pixel position
        if not point:
            if axis == ServoEnum.X_AXIS.value:
                pixel_center = (bbox['x1'] + bbox['x2']) / 2
            else:
                pixel_center = (bbox['y1'] + bbox['y2']) / 2
        else:
            pixel_center = bbox['x'] if axis == ServoEnum.X_AXIS.value else bbox['y']

        # Pixel offset from image center
        offset_from_center = (axis_size / 2) - pixel_center

        # Determine FOV for this camera/axis
        if camera == CameraEnum.CAMERA_HEAD.value:
            if axis == ServoEnum.X_AXIS.value:
                fov = CameraEnum.CAMERA_HEAD_FOV_X.value
            else:
                fov = CameraEnum.CAMERA_HEAD_FOV_Y.value
        elif camera == CameraEnum.CAMERA_RIGHT.value:
            if axis == ServoEnum.X_AXIS.value:
                fov = CameraEnum.CAMERA_RIGHT_FOV.value
            else:
                fov = CameraEnum.CAMERA_RIGHT_FOV_Y.value
        elif camera == CameraEnum.CAMERA_LEFT.value:
            if axis == ServoEnum.X_AXIS.value:
                fov = CameraEnum.CAMERA_LEFT_FOV.value
            else:
                fov = CameraEnum.CAMERA_LEFT_FOV_Y.value
        else:
            fov = 54.0

        # Compute angular offset from pixel displacement.
        # Wide-angle fisheye lenses (>120° FOV) use equidistant projection (r = f*θ),
        # which is a linear pixel-to-angle mapping. Narrow lenses use rectilinear (r = f*tan(θ)).
        # Using rectilinear on fisheye over-estimates angles at the edges by ~20°,
        # causing left and right cameras to disagree on the same person's position.
        if fov > 120:
            # Equidistant fisheye: angle is proportional to pixel offset
            angle_offset_deg = offset_from_center * (fov / axis_size)
        else:
            # Rectilinear (standard lens): use arctan projection
            focal_length = (axis_size / 2) / tan(radians(fov) / 2)
            angle_offset_deg = degrees(atan(offset_from_center / focal_length))

        # Compute camera's current world-space pointing direction
        if camera == CameraEnum.CAMERA_HEAD.value:
            # Use FK with all 4 servo positions to get true pointing direction
            current_angles = {
                self.body_LR_name: self._get_estimated_position(self.body_LR_name),
                self.body_UD_name: self._get_estimated_position(self.body_UD_name),
                self.head_LR_name: self._get_estimated_position(self.head_LR_name),
                self.head_UD_name: self._get_estimated_position(self.head_UD_name),
            }
            yaw, pitch = self._kinematics.forward_kinematics(current_angles)
            self.logger.debug(
                f"FK state: body_lr={current_angles[self.body_LR_name]:.1f} "
                f"body_ud={current_angles[self.body_UD_name]:.1f} "
                f"head_lr={current_angles[self.head_LR_name]:.1f} "
                f"head_ud={current_angles[self.head_UD_name]:.1f} -> yaw={yaw:.1f} pitch={pitch:.1f}")
            self._pdebug.log("MotionTrack", "FK_STATE", {
                "body_lr": round(current_angles[self.body_LR_name], 1),
                "body_ud": round(current_angles[self.body_UD_name], 1),
                "head_lr": round(current_angles[self.head_LR_name], 1),
                "head_ud": round(current_angles[self.head_UD_name], 1),
                "yaw": round(yaw, 1), "pitch": round(pitch, 1),
            })
            if axis == ServoEnum.X_AXIS.value:
                camera_world = yaw
            else:
                camera_world = pitch
        elif camera == CameraEnum.CAMERA_LEFT.value:
            # Side cameras are fixed — they don't rotate with the body.
            if axis == ServoEnum.X_AXIS.value:
                camera_world = MotionProfile.CAMERA_LEFT_MOUNTING_OFFSET.value
            else:
                camera_world = MotionProfile.CAMERA_LEFT_MOUNTING_PITCH.value
        elif camera == CameraEnum.CAMERA_RIGHT.value:
            if axis == ServoEnum.X_AXIS.value:
                camera_world = MotionProfile.CAMERA_RIGHT_MOUNTING_OFFSET.value
            else:
                camera_world = MotionProfile.CAMERA_RIGHT_MOUNTING_PITCH.value
        else:
            camera_world = 0.0

        # World angle = where camera is pointing + offset from frame center
        world_angle = camera_world + angle_offset_deg
        self.logger.debug(
            f"PIX2WORLD: cam={camera} axis={axis} pixel={pixel_center:.0f}/{axis_size:.0f} "
            f"offset_px={offset_from_center:.0f} fov={fov} "
            f"angle_offset={angle_offset_deg:.2f} cam_world={camera_world:.1f} -> world={world_angle:.1f}")
        self._pdebug.log("MotionTrack", "PIX2WORLD", {
            "cam": camera, "axis": axis,
            "pixel": round(pixel_center, 1), "axis_size": axis_size,
            "offset_px": round(offset_from_center, 1),
            "fov": fov, "angle_offset": round(angle_offset_deg, 2),
            "cam_world": round(camera_world, 1), "world": round(world_angle, 1),
        })
        return world_angle

    def _clamp(self, value: float, servo_name: str) -> float:
        """Clamp a value to the servo's physical range."""
        return max(self._servo_mins[servo_name], min(self._servo_maxs[servo_name], value))

    def _get_breathing_offset(self) -> float:
        """Compute the current breathing oscillation offset for body_UD.

        Returns a small sine wave offset (~0.3 degrees at 12 breaths/min).
        Amplitude reduces during drowsy state and stops during sleep.
        Returns 0.0 if breathing is disabled via config.
        """
        if not self._enable_breathing:
            return 0.0
        if self._behavior_state == BehaviorEnums.STATE_ASLEEP.value:
            return 0.0

        amplitude = self._breathing_amplitude
        if self._behavior_state == BehaviorEnums.STATE_DROWSY.value:
            # Reduce breathing amplitude as drowsy progresses
            elapsed = time.time() - self._drowsy_start_time
            duration = BehaviorEnums.DROWSY_TO_SLEEP_DURATION.value
            factor = max(0.0, 1.0 - elapsed / duration)
            amplitude *= factor

        return amplitude * sin(self._breathing_freq * time.time())

    def _get_sway_offsets(self) -> tuple:
        """Compute LR idle sway offsets for body and head.

        Uses a frequency incommensurate with breathing so the combined
        motion traces a Lissajous curve that never exactly repeats.
        Head counter-sways at half amplitude (opposite phase) so the
        camera stays roughly stable while the body drifts.

        Returns:
            Tuple of (body_lr_offset, head_lr_offset) in degrees.
        """
        if not self._enable_breathing:
            return 0.0, 0.0
        if self._behavior_state == BehaviorEnums.STATE_ASLEEP.value:
            return 0.0, 0.0

        body_amp = self._sway_lr_amplitude
        head_amp = self._sway_head_lr_amplitude
        if self._behavior_state == BehaviorEnums.STATE_DROWSY.value:
            elapsed = time.time() - self._drowsy_start_time
            duration = BehaviorEnums.DROWSY_TO_SLEEP_DURATION.value
            factor = max(0.0, 1.0 - elapsed / duration)
            body_amp *= factor
            head_amp *= factor

        t = time.time()
        sway = sin(self._sway_lr_freq * t)
        return body_amp * sway, -head_amp * sway

    def _update_targets(self, target_world_lr: float, target_world_ud: float,
                         trace_id: str = None, source: str = "head") -> None:
        """Compute head and body servo targets from world-space angles and send one MQTT message.

        Uses two-stage Jacobian IK to account for non-orthogonal joint axes:
        1. Head IK: given current body position, find head angles to point at target
        2. Body IK: find where body should go so head can re-center at home

        This preserves head-locks-first behavior: head snaps to target (fast spring),
        body drifts toward ideal (slow spring), head naturally re-centers as body catches up.
        """
        # Current body positions from spring-damper estimators
        current_body = {
            self.body_LR_name: self._get_estimated_position(self.body_LR_name),
            self.body_UD_name: self._get_estimated_position(self.body_UD_name),
        }

        # Stage 1: Head IK -- point camera at target given current body
        head_targets = self._kinematics.inverse_kinematics_head(
            target_world_lr, target_world_ud, current_body)

        # Stage 2: Body IK -- where body should eventually go
        body_targets = self._kinematics.inverse_kinematics_body(
            target_world_lr, target_world_ud)

        head_lr_target = head_targets[self.head_LR_name]
        head_ud_target = head_targets[self.head_UD_name]
        body_lr_target = body_targets[self.body_LR_name]
        body_ud_target = body_targets[self.body_UD_name]

        self.logger.debug(
            f"IK input: world_lr={target_world_lr:.1f} world_ud={target_world_ud:.1f} "
            f"cur_body_lr={current_body[self.body_LR_name]:.1f} "
            f"cur_body_ud={current_body[self.body_UD_name]:.1f}")
        self.logger.debug(
            f"IK output: head_lr={head_lr_target:.1f} head_ud={head_ud_target:.1f} "
            f"body_lr={body_lr_target:.1f} body_ud={body_ud_target:.1f}")
        self._pdebug.log("MotionTrack", "IK_INPUT", {
            "world_lr": round(target_world_lr, 1), "world_ud": round(target_world_ud, 1),
            "cur_body_lr": round(current_body[self.body_LR_name], 1),
            "cur_body_ud": round(current_body[self.body_UD_name], 1),
        }, trace_id=trace_id)
        self._pdebug.log("MotionTrack", "IK_OUTPUT", {
            "head_lr": round(head_lr_target, 1), "head_ud": round(head_ud_target, 1),
            "body_lr": round(body_lr_target, 1), "body_ud": round(body_ud_target, 1),
        }, trace_id=trace_id)

        # Saccadic movement: detect large head repositions and use fast speed.
        # Compare against last COMMANDED target, not estimated position — the
        # estimator lags during convergence, which would falsely trigger saccades
        # on every frame while the spring-damper is still settling.
        prev_head_lr = self._estimators[self.head_LR_name].target
        prev_head_ud = self._estimators[self.head_UD_name].target
        head_delta = max(abs(head_lr_target - prev_head_lr), abs(head_ud_target - prev_head_ud))
        saccade = head_delta > MotionProfile.SACCADE_THRESHOLD.value
        head_speed = MotionProfile.SACCADE_SPEED.value if saccade else self.dms
        if saccade:
            self.logger.debug(f"SACCADE: head_delta={head_delta:.1f} -> speed {head_speed} source={source}")
            # Only suppress head camera when the HEAD camera itself drives a saccade.
            # Side-camera-driven saccades happen during UD sweep (large IK deltas from
            # changing world_ud) and should NOT suppress the head camera — it needs to
            # process frames to lock on and stop the sweep.
        # Body UD urgency: when head UD is near its physical limit, speed up body UD
        # and widen the rate limit so the body can catch up
        head_ud_margin = min(
            abs(head_ud_target - self._servo_mins[self.head_UD_name]),
            abs(head_ud_target - self._servo_maxs[self.head_UD_name]))
        body_ud_urgent = head_ud_margin < MotionProfile.BODY_UD_URGENCY_MARGIN.value
        body_ud_speed = min(self.dms + 1, 5) if body_ud_urgent else self.dms

        # Rate-limit body IK output to prevent frame-to-frame oscillation
        # between solver local minima (especially at steep pitch angles)
        max_lr = MotionProfile.BODY_LR_MAX_STEP_DEG.value
        max_ud = MotionProfile.BODY_UD_MAX_STEP_DEG_URGENT.value if body_ud_urgent else MotionProfile.BODY_UD_MAX_STEP_DEG.value
        body_lr_pre_clamp = body_lr_target
        body_ud_pre_clamp = body_ud_target
        if self._prev_body_lr_target is not None:
            delta = body_lr_target - self._prev_body_lr_target
            body_lr_target = self._prev_body_lr_target + max(-max_lr, min(max_lr, delta))
        if self._prev_body_ud_target is not None:
            delta = body_ud_target - self._prev_body_ud_target
            body_ud_target = self._prev_body_ud_target + max(-max_ud, min(max_ud, delta))
        if body_lr_target != body_lr_pre_clamp or body_ud_target != body_ud_pre_clamp:
            self.logger.debug(
                f"IK rate-limit: body_lr {body_lr_pre_clamp:.1f}->{body_lr_target:.1f} "
                f"body_ud {body_ud_pre_clamp:.1f}->{body_ud_target:.1f} "
                f"(max_step lr={max_lr} ud={max_ud})")
            self._pdebug.log("MotionTrack", "IK_RATE_LIMIT", {
                "body_lr_pre": round(body_lr_pre_clamp, 1), "body_lr_post": round(body_lr_target, 1),
                "body_ud_pre": round(body_ud_pre_clamp, 1), "body_ud_post": round(body_ud_target, 1),
                "max_lr": max_lr, "max_ud": max_ud,
            }, trace_id=trace_id)
        self._prev_body_lr_target = body_lr_target
        self._prev_body_ud_target = body_ud_target

        # Add organic micro-motion: breathing (UD) + sway (LR)
        # Applied after IK rate limiting but before clamping so they
        # don't interfere with the IK solver or accumulate in prev targets.
        breathing = self._get_breathing_offset()
        sway_body_lr, sway_head_lr = self._get_sway_offsets()
        body_ud_target += breathing
        body_lr_target += sway_body_lr
        head_lr_target += sway_head_lr

        pre_clamp = (head_lr_target, head_ud_target, body_lr_target, body_ud_target)
        head_lr_target = self._clamp(head_lr_target, self.head_LR_name)
        head_ud_target = self._clamp(head_ud_target, self.head_UD_name)
        body_lr_target = self._clamp(body_lr_target, self.body_LR_name)
        body_ud_target = self._clamp(body_ud_target, self.body_UD_name)
        post_clamp = (head_lr_target, head_ud_target, body_lr_target, body_ud_target)
        if pre_clamp != post_clamp:
            self.logger.debug(
                f"IK clamp: head_lr {pre_clamp[0]:.1f}->{post_clamp[0]:.1f} "
                f"head_ud {pre_clamp[1]:.1f}->{post_clamp[1]:.1f} "
                f"body_lr {pre_clamp[2]:.1f}->{post_clamp[2]:.1f} "
                f"body_ud {pre_clamp[3]:.1f}->{post_clamp[3]:.1f}")
            self._pdebug.log("MotionTrack", "IK_CLAMP", {
                "head_lr": f"{pre_clamp[0]:.1f}->{post_clamp[0]:.1f}",
                "head_ud": f"{pre_clamp[1]:.1f}->{post_clamp[1]:.1f}",
                "body_lr": f"{pre_clamp[2]:.1f}->{post_clamp[2]:.1f}",
                "body_ud": f"{pre_clamp[3]:.1f}->{post_clamp[3]:.1f}",
            }, trace_id=trace_id)

        # Build consolidated move_all message with per-servo speeds
        targets = {
            self.head_LR_name: {ServoEnum.MSG_ANGLE.value: round(head_lr_target),
                                ServoEnum.MSG_SPEED.value: head_speed},
            self.head_UD_name: {ServoEnum.MSG_ANGLE.value: round(head_ud_target),
                                ServoEnum.MSG_SPEED.value: head_speed},
            self.body_LR_name: {ServoEnum.MSG_ANGLE.value: round(body_lr_target),
                                ServoEnum.MSG_SPEED.value: self.dms},
            self.body_UD_name: {ServoEnum.MSG_ANGLE.value: round(body_ud_target),
                                ServoEnum.MSG_SPEED.value: body_ud_speed},
        }

        msg = ServoMessageBuilder.move_all(targets)
        if trace_id:
            msg[TraceEnums.TRACE_ID.value] = trace_id
            msg[TraceEnums.TS_VISION.value] = self._tracer._active.get(trace_id, {}).get("ts_vision")
        if self._enable_movement:
            self.send_command(msg, ServoEnum.MQTT_COMMAND_TOPIC.value)
        else:
            self.logger.debug(f"Movement disabled — would send: {targets}")

        # Update local estimators with matching per-servo spring-damper params
        head_omega, head_zeta = MotionProfile.HEAD_PARAMS.value[head_speed]
        body_omega, body_zeta = MotionProfile.BODY_PARAMS.value[self.dms]
        body_ud_omega, body_ud_zeta = MotionProfile.BODY_PARAMS.value[body_ud_speed]
        self._estimators[self.head_LR_name].set_target(head_lr_target, head_omega, head_zeta)
        self._estimators[self.head_UD_name].set_target(head_ud_target, head_omega, head_zeta)
        self._estimators[self.body_LR_name].set_target(body_lr_target, body_omega, body_zeta)
        self._estimators[self.body_UD_name].set_target(body_ud_target, body_ud_omega, body_ud_zeta)

        self.logger.debug(
            f"Targets sent: head_LR={head_lr_target:.1f} head_UD={head_ud_target:.1f} "
            f"body_LR={body_lr_target:.1f} body_UD={body_ud_target:.1f}"
        )
        self._pdebug.log("MotionTrack", "TARGETS_SENT", {
            "head_lr": round(head_lr_target, 1), "head_ud": round(head_ud_target, 1),
            "body_lr": round(body_lr_target, 1), "body_ud": round(body_ud_target, 1),
            "speed": self.dms, "movement": self._enable_movement,
        }, trace_id=trace_id)

    def _generate_idle_drift(self) -> None:
        """Generate behavior based on the current state when no target is actively tracked.

        Behavior hierarchy:
            1. DRIFT TOWARD: side camera sees someone → slowly rotate toward them
            2. IDLE: random layered sine waves
            3. DROWSY: decreasing amplitude, head drooping (after 2 min idle)
            4. ASLEEP: still, no movement (after 30s drowsy)
        """
        now = time.time()

        # Check for sleep/drowsy transitions (skip if sleep disabled)
        if not self._enable_sleep:
            self._behavior_state = BehaviorEnums.STATE_IDLE.value
        if self._behavior_state == BehaviorEnums.STATE_ASLEEP.value:
            # Asleep — no movement. Side camera wake-up is handled in track_loop.
            if self.debug_overlay_enabled:
                self._debug_overlay["state"] = "ASLEEP"
            return

        if self._behavior_state == BehaviorEnums.STATE_DROWSY.value:
            elapsed = now - self._drowsy_start_time
            if elapsed >= BehaviorEnums.DROWSY_TO_SLEEP_DURATION.value:
                self._behavior_state = BehaviorEnums.STATE_ASLEEP.value
                self.logger.info("Falling asleep")
                if self.debug_overlay_enabled:
                    self._debug_overlay["state"] = "ASLEEP"
                return

        # Transition idle → drowsy after 2 minutes
        if self._behavior_state == BehaviorEnums.STATE_IDLE.value:
            if self._idle_start_time > 0:
                idle_duration = now - self._idle_start_time
                if idle_duration >= BehaviorEnums.IDLE_TO_DROWSY_TIMEOUT.value:
                    if self._behavior_state != BehaviorEnums.STATE_DROWSY.value:
                        self._behavior_state = BehaviorEnums.STATE_DROWSY.value
                        self._drowsy_start_time = now
                        self.logger.info("Getting drowsy")

        # Rate limit drift updates
        if now - self._last_idle_send < self._idle_interval:
            return
        self._last_idle_send = now

        # Check if side camera sees someone — drift toward them
        best_side = self._fusion.get_best_side_world_lr()
        if best_side is not None:
            # Someone is visible on side camera — drift toward them
            current_angles = {
                self.body_LR_name: self._get_estimated_position(self.body_LR_name),
                self.body_UD_name: self._get_estimated_position(self.body_UD_name),
                self.head_LR_name: self._get_estimated_position(self.head_LR_name),
                self.head_UD_name: self._get_estimated_position(self.head_UD_name),
            }
            _, current_pitch = self._kinematics.forward_kinematics(current_angles)
            self._update_targets(best_side, current_pitch, source="side")
            if self.debug_overlay_enabled:
                self._debug_overlay["state"] = "DRIFT_TOWARD"
            return

        # No one detected — check for memory glance opportunity
        if self._enable_glances and self._last_known_positions and not self._idle_active:
            # Occasionally glance at where someone was last seen
            import random
            if random.random() < 0.02:  # ~2% chance per update cycle
                face_id = random.choice(list(self._last_known_positions.keys()))
                glance_lr = self._last_known_positions[face_id]
                head_ud_mid = self._servo_middles[self.head_UD_name]
                self._update_targets(glance_lr, head_ud_mid, source="side")
                self.logger.debug(f"Memory glance toward {face_id}")
                if self.debug_overlay_enabled:
                    self._debug_overlay["state"] = "GLANCE"
                return

        # Random idle drift — layered sine waves
        t = now
        # Scale amplitude based on behavior state (drowsy reduces amplitude)
        amplitude_scale = 1.0
        if self._behavior_state == BehaviorEnums.STATE_DROWSY.value:
            elapsed = now - self._drowsy_start_time
            duration = BehaviorEnums.DROWSY_TO_SLEEP_DURATION.value
            amplitude_scale = max(0.0, 1.0 - elapsed / duration)

        lr = amplitude_scale * (2.5 * sin(0.3 * t) + 1.0 * sin(0.7 * t + 1.2) + 0.4 * sin(1.4 * t + 0.5))
        ud = amplitude_scale * (1.8 * sin(0.25 * t + 0.8) + 0.6 * sin(0.55 * t + 2.1))

        # Drowsy head droop — gradually lower the head
        if self._behavior_state == BehaviorEnums.STATE_DROWSY.value:
            droop = BehaviorEnums.DROOP_ANGLE.value * (1.0 - amplitude_scale)
            ud -= droop

        head_lr_mid = self._servo_middles[self.head_LR_name]
        head_ud_mid = self._servo_middles[self.head_UD_name]

        self._update_targets(head_lr_mid + lr, head_ud_mid + ud, source="side")
        if not self._idle_active:
            self._idle_active = True
            self._idle_start_time = now
            self._behavior_state = BehaviorEnums.STATE_IDLE.value
            self.logger.debug("Entering idle drift mode")
        if self.debug_overlay_enabled:
            self._debug_overlay["state"] = self._behavior_state

    def _find_detection_for_person(self, detections: list, person_id: str,
                                    roster: Dict[str, Any]) -> Optional[dict]:
        """Find the detection in the current frame that matches a room roster person.

        Matches by face_id first, then by proximity to the roster person's
        last known world_lr position.

        Args:
            detections: List of person detection dicts from this frame.
            person_id: The room roster person_id to find.
            roster: Current room roster for position lookup.

        Returns:
            The best matching detection dict, or None.
        """
        if person_id not in roster:
            return None

        person = roster[person_id]

        # Try face_id match first
        for det in detections:
            face_data = det.get("face", {})
            if face_data and face_data.get("face_id") == person.face_id and person.face_id != "unknown":
                return det

        # Fall back to closest bbox center to person's world_lr
        best_det = None
        best_dist = float("inf")
        for det in detections:
            box = det.get("box", {})
            if not box:
                continue
            cx = (box.get("x1", 0) + box.get("x2", 0)) / 2
            # Approximate: bbox center offset from frame center as proxy for angle
            # Not exact but good enough for matching within a single frame
            det_height = box.get("y2", 0) - box.get("y1", 0)
            height_diff = abs(det_height - person.bbox_height)
            height_ratio = height_diff / max(person.bbox_height, 1.0)
            dist = height_ratio  # Lower = better match
            if dist < best_dist:
                best_dist = dist
                best_det = det

        return best_det

    def _drive_from_room_state(self) -> None:
        """Single decision point for side-camera-driven servo movement.

        Uses the attention model's current target (if available) to pick
        a specific person's world_lr from the room roster. Falls back to
        the best side camera angle if attention model is disabled or the
        target isn't in the roster. This prevents oscillation when left
        and right cameras see different people.
        """
        best_lr = None

        # Prefer attention model target — prevents oscillation between people.
        # But if the target's position is stale (>1s since last update), the
        # person likely moved and got a new roster ID. Fall through to the
        # raw side camera angle so the robot still tracks SOMEBODY.
        if self._attention and self._room_state:
            roster = self._room_state.get_roster()
            target_id = self._attention._current_target
            if target_id and target_id in roster:
                person = roster[target_id]
                if (time.time() - person.last_seen) < 1.0:
                    best_lr = person.world_lr
                else:
                    self.logger.debug(
                        f"SIDE_DRIVE: target {target_id} stale "
                        f"({time.time() - person.last_seen:.1f}s) — using raw side angle")

        # Fall back to raw side camera angle
        if best_lr is None:
            best_lr = self._fusion.get_best_side_world_lr()
        if best_lr is None:
            self.logger.info("SIDE_DRIVE: no best_lr — skipping")
            return

        # Predictive rotation if enabled
        if self._enable_predictive:
            best_camera = self._fusion.get_best_side_camera()
            if best_camera:
                best_lr = self._fusion.get_predicted_world_lr(best_camera)

        # EMA smoothing — dampens noise from fisheye distortion and bbox jitter
        alpha = FusionEnums.SIDE_WORLD_SMOOTH_ALPHA.value
        if self._side_world_lr_smooth is None:
            self._side_world_lr_smooth = best_lr
        else:
            old_smooth = self._side_world_lr_smooth
            self._side_world_lr_smooth = alpha * self._side_world_lr_smooth + (1 - alpha) * best_lr
            self.logger.debug(
                f"SIDE_DRIVE: raw_lr={best_lr:.1f} smooth_lr={old_smooth:.1f}->{self._side_world_lr_smooth:.1f} "
                f"alpha={alpha:.2f}")

        # Throttle servo commands to match Pi4's 50Hz physics loop.
        now = time.time()
        if now - self._side_drive_last_send < 0.05:
            return
        self._side_drive_last_send = now

        # UD pitch from side camera estimate or default.
        # Side cameras aim at the body center (chest/lap). The face is
        # above that by ~8 degrees for a seated person at typical distance.
        # Apply an upward bias so the head camera sees the face, not the chest.
        side_ud = self._fusion.get_best_side_world_ud()
        face_bias = 12.0  # degrees upward from body center to face
        if side_ud is not None:
            world_ud = max(-50.0, min(10.0, side_ud + face_bias))
        else:
            world_ud = MotionProfile.SIDE_DRIVE_DEFAULT_PITCH.value

        self.logger.info(
            f"SIDE_DRIVE: best_lr={best_lr:.1f} smooth_lr={self._side_world_lr_smooth:.1f} "
            f"world_ud={world_ud:.1f} source=side")
        self._pdebug.log("MotionTrack", "SIDE_DRIVE", {
            "target_lr": round(self._side_world_lr_smooth, 1),
            "world_ud": round(world_ud, 1),
            "fusion_state": self._fusion.state,
        })
        self._update_targets(self._side_world_lr_smooth, world_ud, source="side")

    def __select_target(self, seen_data: list, camera: str) -> dict:
        """Select the best target using human-like priority.

        When face ID history is available:
            1. Face ID match (40%) — same person as last tracked
            2. Proximity to last tracked world angle (30%)
            3. Similar bbox height (15%) — distance proxy
            4. Confidence (15%) — tiebreaker

        Without face ID history:
            1. Proximity to last tracked world angle (50%)
            2. Similar bbox height (30%)
            3. Confidence (20%)

        Falls back to highest confidence when no prior tracking state exists.

        Args:
            seen_data: List of detected person dicts with box and confidence.
            camera: Camera name for world angle calculation.

        Returns:
            The best-scoring detection dict, or empty dict if none.
        """
        confidence_key = VisionResultsEnum.VISION_RESULTS_CONFIDENCE_KEY.value
        box_key = TrackingEnums.KEY_BOX.value
        face_key = VisionResultsEnum.VISION_RESULTS_FACE_KEY.value
        face_id_key = VisionResultsEnum.VISION_RESULTS_FACE_ID_KEY.value

        # Confidence floor: filter out low-confidence phantoms before scoring.
        # Filter using composite confidence (multi-model evidence) rather than
        # raw YOLO confidence alone. A detection at YOLO 0.45 with pose keypoints
        # (composite ~0.60) is more reliable than YOLO 0.50 with no pose.
        from glados_modules.RoomStateManager import RoomStateManager
        min_conf = MotionProfile.TARGET_MIN_CONFIDENCE.value
        seen_data = [d for d in seen_data
                     if RoomStateManager.compute_frame_composite(d) >= min_conf]
        if not seen_data:
            return {}

        # No prior target — fall back to highest confidence
        if self._last_tracked_world_lr is None:
            best = {}
            highest = 0
            for p in seen_data:
                if p.get(confidence_key, 0) > highest:
                    highest = p[confidence_key]
                    best = p
            self.logger.debug(f"SELECT: no prior target, picked highest conf={highest:.2f} "
                              f"from {len(seen_data)} detections")
            return best

        has_face_history = self._last_tracked_face_id is not None

        best = {}
        best_score = -1.0
        for p in seen_data:
            score = 0.0
            bbox = p.get(box_key, {})
            confidence = p.get(confidence_key, 0)
            face_data = p.get(face_key, {})
            face_id = face_data.get(face_id_key) if face_data else None

            if has_face_history:
                # Face ID match: strong signal (35%)
                if face_id and face_id == self._last_tracked_face_id:
                    score += 0.35
                # Proximity (25%)
                if bbox:
                    world_lr = self._pixel_to_world_angle(bbox, camera, ServoEnum.X_AXIS.value)
                    angle_diff = abs(world_lr - self._last_tracked_world_lr)
                    proximity = max(0.0, 1.0 - angle_diff / 90.0)
                    score += proximity * 0.25
                # Height similarity (15%)
                if bbox and self._last_tracked_bbox_height and self._last_tracked_bbox_height > 0:
                    height = bbox.get('y2', 0) - bbox.get('y1', 0)
                    if height > 0:
                        height_ratio = min(height, self._last_tracked_bbox_height) / max(height, self._last_tracked_bbox_height)
                        score += height_ratio * 0.15
                # Confidence (25%) — high weight prevents phantoms from winning
                score += confidence * 0.25
            else:
                # No face history — use position-based scoring
                if bbox:
                    world_lr = self._pixel_to_world_angle(bbox, camera, ServoEnum.X_AXIS.value)
                    angle_diff = abs(world_lr - self._last_tracked_world_lr)
                    proximity = max(0.0, 1.0 - angle_diff / 90.0)
                    score += proximity * 0.35
                if bbox and self._last_tracked_bbox_height and self._last_tracked_bbox_height > 0:
                    height = bbox.get('y2', 0) - bbox.get('y1', 0)
                    if height > 0:
                        height_ratio = min(height, self._last_tracked_bbox_height) / max(height, self._last_tracked_bbox_height)
                        score += height_ratio * 0.25
                # Confidence (40%) — dominant factor when no face history
                score += confidence * 0.40

            self.logger.debug(
                f"SELECT: candidate conf={confidence:.2f} face={face_id} score={score:.3f} "
                f"(face_match={'Y' if face_id and face_id == self._last_tracked_face_id else 'N'})")

            if score > best_score:
                best_score = score
                best = p

        self.logger.debug(f"SELECT: winner score={best_score:.3f} from {len(seen_data)} candidates "
                          f"(has_face_history={has_face_history})")
        return best

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """Handle incoming tracking command messages."""
        j_msg = loads(msg.payload.decode())
        if j_msg.get(self.cmd_trigger, "") == TrackingEnums.MSG_COMMAND_START.value:
            self.logger.debug(f"Tracking command received: {j_msg}")
            camera = j_msg.get(TrackingEnums.MSG_CAMERA_KEY.value, "")
            self.track_loop(camera)

    def track_loop(self, camera: str) -> None:
        """Main tracking loop: convert detection to world angle, compute targets, send.

        Called by VisionTracker on each detection frame. Non-blocking.
        """
        # Ensure estimators are initialized
        if not self._init_estimators():
            self.logger.debug("Waiting for servo status to initialize estimators")
            return

        # Camera readiness gate: track which cameras have come online and hold
        # all movement until all three are reporting. This prevents the robot
        # from chasing single-camera phantoms during the startup sequence.
        # Head camera override: if the head camera has a strong detection,
        # start tracking immediately — don't wait for side cameras. The head
        # camera is the primary tracking sensor and shouldn't be blocked by
        # slow or offline side cameras.
        # Times out after 30s so a missing camera doesn't block forever.
        if not self._cameras_online:
            self._cameras_ready.add(camera)
            if self._cameras_gate_start == 0.0:
                self._cameras_gate_start = time.time()
            elapsed = time.time() - self._cameras_gate_start
            if self._cameras_ready >= self._all_cameras:
                self._cameras_online = True
                self.logger.info(f"All cameras online: {self._cameras_ready}")
            elif camera == self.main_camera:
                # Head camera override: start immediately if head has a detection.
                # The head camera passing VisionTracker's 0.65 gate means it has
                # a high-confidence person — no need to wait for side cameras.
                self._cameras_online = True
                missing = self._all_cameras - self._cameras_ready
                self.logger.info(
                    f"Head camera override — starting tracking (ready: "
                    f"{self._cameras_ready}, pending: {missing})")
            elif elapsed > self._cameras_gate_timeout:
                self._cameras_online = True
                missing = self._all_cameras - self._cameras_ready
                self.logger.warning(
                    f"Camera gate timeout after {elapsed:.0f}s — starting with "
                    f"{self._cameras_ready}, missing: {missing}")
            else:
                missing = self._all_cameras - self._cameras_ready
                self.logger.debug(f"Waiting for cameras ({elapsed:.0f}s): {missing}")
                return

        # Periodically sync estimators from MQTT status
        self._sync_estimators_from_status()

        # Cache diagnostic snapshot for recording BEFORE any early returns.
        # Built here in the MQTT thread where estimator state is safe to read.
        # MachineVision's tracker threads read the cached copy.
        self._update_diagnostic_cache()

        vision_map = self.vision_tracker.get_vision_map()
        if camera not in vision_map:
            if camera == self.main_camera:
                self._fusion.head_contributing = False
            # Clear stale camera from roster so fuzzy matching isn't poisoned
            if self._room_state:
                self._room_state.clear_camera(camera)
            if time.time() - self._last_target_time > self._idle_timeout:
                if self._enable_idle_drift:
                    self._generate_idle_drift()
            return

        target_data = vision_map[camera].get(self.target, {})
        if target_data.get(self.count, 0) == 0:
            if camera == self.main_camera:
                self._fusion.head_contributing = False
            # Clear stale camera from roster so fuzzy matching isn't poisoned
            if self._room_state:
                self._room_state.clear_camera(camera)
            if time.time() - self._last_target_time > self._idle_timeout:
                if self._enable_idle_drift:
                    self._generate_idle_drift()
            return

        # Found a target — wake up if sleeping/drowsy
        was_asleep = self._behavior_state in (BehaviorEnums.STATE_ASLEEP.value,
                                               BehaviorEnums.STATE_DROWSY.value)
        self._last_target_time = time.time()
        self._idle_active = False
        self._behavior_state = BehaviorEnums.STATE_ACTIVE.value
        if was_asleep:
            self.logger.info("Waking up — target detected")

        # Extract trace_id from vision results (stamped by MachineVision)
        trace_id = vision_map[camera].get(TraceEnums.TRACE_ID.value)
        ts_vision = vision_map[camera].get(TraceEnums.TS_VISION.value)

        # Per-detection re-filter using composite confidence (YOLO + pose + face).
        # Pose-confirmed detections at low YOLO confidence are real people, not
        # phantoms — phantoms never accumulate pose/face evidence. All cameras
        # run pose estimation so all benefit from composite scoring.
        from glados_modules.RoomStateManager import RoomStateManager
        all_objects = target_data.get(self.objects, [])
        composite_threshold = MotionProfile.TARGET_MIN_CONFIDENCE.value
        filtered_objects = [d for d in all_objects
                           if RoomStateManager.compute_frame_composite(d) >= composite_threshold]
        if not filtered_objects:
            if camera == self.main_camera:
                self._fusion.head_contributing = False
            if self._room_state:
                self._room_state.clear_camera(camera)
            return

        # Target selection: attention model only evaluates on head camera.
        # Side cameras must not switch targets — they follow whatever the head last chose.
        if self._attention and self._room_state and camera == self.main_camera:
            now = time.time()
            dt = now - self._last_attention_time
            self._last_attention_time = now
            roster = self._room_state.get_roster()
            attention_id, attention_reason = self._attention.select_target(roster, dt)
            self.logger.debug(
                f"ATTENTION: target={attention_id} reason={attention_reason} "
                f"roster_size={len(roster)} detections={len(filtered_objects)}")
            self._pdebug.log("MotionTrack", "ATTENTION", {
                "target": attention_id, "reason": attention_reason,
                "roster_size": len(roster),
                "detections": len(filtered_objects),
            }, trace_id=trace_id)
            if attention_id:
                # Find the detection matching the attention target
                best_target = self._find_detection_for_person(
                    filtered_objects, attention_id, roster)
                if not best_target:
                    # Attention target not in this frame — fall back to legacy
                    self.logger.debug(f"ATTENTION: {attention_id} not found in detections, falling back to legacy")
                    best_target = self.__select_target(filtered_objects, camera)
                # Update attention time on room roster
                self._room_state.update_attention(attention_id, dt)
            else:
                best_target = self.__select_target(filtered_objects, camera)
        else:
            best_target = self.__select_target(filtered_objects, camera)
        if not best_target:
            if camera in (self.left_camera, self.right_camera):
                self.logger.info(
                    f"TRACK_LOOP: {camera} no best_target, filtered={len(filtered_objects)}")
            # Still update room roster and FUSION STATE even without a tracking
            # target. Side cameras must update their world angles or the robot
            # freezes when __select_target can't find a match.
            if camera in (self.left_camera, self.right_camera):
                # Update fusion state directly from filtered_objects
                if filtered_objects:
                    top = max(filtered_objects,
                              key=lambda d: d.get("confidence", 0))
                    bbox = top.get(TrackingEnums.KEY_BOX.value, {})
                    if bbox:
                        side_lr = self._pixel_to_world_angle(
                            bbox, camera, ServoEnum.X_AXIS.value)
                        side_ud = self._pixel_to_world_angle(
                            bbox, camera, ServoEnum.Y_AXIS.value)
                        self._fusion.update_side_detection(
                            camera, side_lr,
                            len(filtered_objects), world_ud=side_ud)
                if self._room_state:
                    self._room_state.update_from_vision(
                        camera, filtered_objects,
                        lambda bbox, cam: self._pixel_to_world_angle(
                            bbox, cam, ServoEnum.X_AXIS.value))
            return

        # Update room roster with filtered detections from this camera.
        # Only side cameras provide roster world_lr — head camera FK is
        # unreliable (cross-coupling errors, frozen when servos don't move).
        if self._room_state and camera != self.main_camera:
            self._room_state.update_from_vision(
                camera, filtered_objects,
                lambda bbox, cam: self._pixel_to_world_angle(bbox, cam, ServoEnum.X_AXIS.value))

            # Periodically publish room state and process arrivals/departures
            if self._room_state.should_publish():
                arrivals, departures = self._room_state.tick()
                # Clean up attention model state for departed people
                if self._attention and departures:
                    for pid in departures:
                        self._attention.on_person_departed(pid)
                summary = self._room_state.get_room_summary()
                summary["arrivals"] = arrivals
                summary["departures"] = departures
                if self._attention:
                    summary.update(self._attention.get_state())
                self.send_command(summary, RoomStateEnums.MQTT_ROOM_TOPIC.value)
                self._room_state.mark_published()
                if arrivals or departures:
                    self.logger.debug(
                        f"ROOM_ROSTER publish: count={summary['count']} "
                        f"arrivals={arrivals} departures={departures}")

        # Side cameras: record both LR and UD world angles in fusion state.
        # For UD, aim at the FACE (top ~30% of bbox), not the bbox center
        # (which is chest/lap for a seated person). Use face keypoints from
        # pose if available, otherwise bias toward top of bbox.
        if camera in (self.left_camera, self.right_camera):
            bbox = best_target.get(TrackingEnums.KEY_BOX.value, {})
            if bbox:
                side_world_lr = self._pixel_to_world_angle(bbox, camera, ServoEnum.X_AXIS.value)

                # UD: prefer face keypoints from pose, fall back to top-biased bbox
                face_y = None
                pose = best_target.get(TrackingEnums.KEY_POSE.value, {})
                if pose:
                    _FACE_KPS = ("Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear")
                    face_ys = [pose[k]["y"] for k in _FACE_KPS
                               if k in pose and isinstance(pose[k], dict)
                               and pose[k].get("confidence", 0) >= 0.3]
                    if len(face_ys) >= 2:
                        face_y = sum(face_ys) / len(face_ys)

                if face_y is not None:
                    # Use face keypoint centroid for precise UD
                    face_point = {"x": (bbox.get('x1', 0) + bbox.get('x2', 0)) / 2,
                                  "y": face_y, "confidence": 1.0}
                    side_world_ud = self._pixel_to_world_angle(
                        face_point, camera, ServoEnum.Y_AXIS.value, point=True)
                else:
                    # Bias toward top 30% of bbox (face region)
                    y1 = bbox.get('y1', 0)
                    y2 = bbox.get('y2', 0)
                    face_biased_bbox = dict(bbox)
                    face_biased_bbox['y2'] = y1 + (y2 - y1) * 0.6  # shrink bbox to upper 60%
                    side_world_ud = self._pixel_to_world_angle(
                        face_biased_bbox, camera, ServoEnum.Y_AXIS.value)

                bbox_cx = (bbox.get('x1', 0) + bbox.get('x2', 0)) / 2
                bbox_cy = (bbox.get('y1', 0) + bbox.get('y2', 0)) / 2
                face_y_str = f"{face_y:.0f}" if face_y is not None else "N/A"
                self.logger.debug(f"Side raw: {camera} bbox_cx={bbox_cx:.0f} bbox_cy={bbox_cy:.0f} "
                                  f"face_y={face_y_str} "
                                  f"raw_world_lr={side_world_lr:.1f} raw_world_ud={side_world_ud:.1f}")
                self._fusion.update_side_detection(camera, side_world_lr,
                                                    target_data.get(self.count, 0),
                                                    world_ud=side_world_ud)

        # Head camera: compute refinement offset on side camera base
        if camera == self.main_camera:
            side_lr = self._fusion.get_best_side_world_lr()
            side_ud = self._fusion.get_best_side_world_ud()
            if side_lr is None:
                self._fusion.head_contributing = False
                return

            # Apply same face bias as _drive_from_room_state
            face_bias = 12.0
            if side_ud is not None:
                side_ud = max(-50.0, min(10.0, side_ud + face_bias))
            else:
                side_ud = MotionProfile.SIDE_DRIVE_DEFAULT_PITCH.value

            from glados_modules.RoomStateManager import RoomStateManager
            composite = RoomStateManager.compute_frame_composite(best_target)

            # Smoothed refinement: blend new offset when detection is good,
            # decay toward zero when detection is absent or weak. Prevents
            # jitter from sporadic detections toggling the offset on/off.
            decay = 0.85  # 85% retain per frame without detection (~1s to halve at 15fps)
            blend = 0.4   # 40% new, 60% old when detection is present

            if composite >= MotionProfile.HEAD_REFINEMENT_MIN_COMPOSITE.value:
                # Good detection — compute raw refinement and blend.
                # Use centroid of visible face keypoints (more stable than
                # single nose point). Falls back to bbox center if no face.
                _FACE_KPS = ("Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear")
                use_point = False
                target_data_for_calc = best_target.get(TrackingEnums.KEY_BOX.value, {})
                if TrackingEnums.KEY_POSE.value in best_target:
                    pose_data = best_target[TrackingEnums.KEY_POSE.value]
                    face_xs = []
                    face_ys = []
                    for kp_name in _FACE_KPS:
                        if kp_name in pose_data:
                            kp = pose_data[kp_name]
                            if isinstance(kp, dict) and kp.get("confidence", 0) >= 0.3:
                                face_xs.append(kp["x"])
                                face_ys.append(kp["y"])
                    if len(face_xs) >= 2:
                        # Centroid of visible face keypoints
                        target_data_for_calc = {
                            "x": sum(face_xs) / len(face_xs),
                            "y": sum(face_ys) / len(face_ys),
                            "confidence": composite,
                        }
                        use_point = True

                # Compute refinement directly from pixel offset — no FK.
                # "Person is N degrees from frame center" IS the refinement.
                # FK has 10-25x cross-coupling errors (IMU sweep confirmed),
                # so bypassing it eliminates the biggest error source.
                if use_point:
                    px = target_data_for_calc["x"]
                    py = target_data_for_calc["y"]
                else:
                    px = (target_data_for_calc.get("x1", 0) + target_data_for_calc.get("x2", 0)) / 2
                    py = (target_data_for_calc.get("y1", 0) + target_data_for_calc.get("y2", 0)) / 2

                # Pixel offset from frame center (positive = person is left/above center)
                lr_offset_px = (self.cam_x / 2) - px
                ud_offset_px = (self.cam_y / 2) - py

                # Convert to degrees using rectilinear projection (head camera is not fisheye)
                fov_x = CameraEnum.CAMERA_HEAD_FOV_X.value
                fov_y = CameraEnum.CAMERA_HEAD_FOV_Y.value
                from math import atan, degrees as _deg
                focal_x = (self.cam_x / 2) / atan(radians(fov_x) / 2)
                focal_y = (self.cam_y / 2) / atan(radians(fov_y) / 2)
                raw_lr = _deg(atan(lr_offset_px / focal_x))
                raw_ud = _deg(atan(ud_offset_px / focal_y))
                max_lr = MotionProfile.HEAD_REFINEMENT_MAX_LR.value
                max_ud = MotionProfile.HEAD_REFINEMENT_MAX_UD.value
                raw_lr = max(-max_lr, min(max_lr, raw_lr))
                raw_ud = max(-max_ud, min(max_ud, raw_ud))

                # Blend into smoothed refinement
                self._refinement_lr = (1 - blend) * self._refinement_lr + blend * raw_lr
                self._refinement_ud = (1 - blend) * self._refinement_ud + blend * raw_ud

                self._fusion.head_contributing = True
                self._fusion._head_last_seen = time.time()
            else:
                # No good detection — decay refinement toward zero
                self._refinement_lr *= decay
                self._refinement_ud *= decay
                if abs(self._refinement_lr) < 0.1:
                    self._refinement_lr = 0.0
                if abs(self._refinement_ud) < 0.1:
                    self._refinement_ud = 0.0
                self._fusion.head_contributing = abs(self._refinement_lr) > 0.1 or abs(self._refinement_ud) > 0.1

            self._fusion.update_head_count(target_data.get(self.count, 0))

            final_lr = side_lr + self._refinement_lr
            final_ud = side_ud + self._refinement_ud

            self.logger.debug(
                f"HEAD_REFINE: side=({side_lr:.1f},{side_ud:.1f}) "
                f"refine=({self._refinement_lr:.1f},{self._refinement_ud:.1f}) "
                f"final=({final_lr:.1f},{final_ud:.1f}) comp={composite:.2f}")

            self._update_targets(final_lr, final_ud, trace_id=trace_id, source="head")

            # Save tracking state
            self._last_tracked_world_lr = final_lr
            bbox_for_height = best_target.get(TrackingEnums.KEY_BOX.value, {})
            if bbox_for_height:
                self._last_tracked_bbox_height = float(
                    bbox_for_height.get('y2', 0) - bbox_for_height.get('y1', 0))
            face_data = best_target.get(
                VisionResultsEnum.VISION_RESULTS_FACE_KEY.value, {})
            if face_data:
                fid = face_data.get(VisionResultsEnum.VISION_RESULTS_FACE_ID_KEY.value)
                if fid and fid != "unknown":
                    self._last_tracked_face_id = fid
                    self._last_known_positions[fid] = final_lr

            # Recording
            if self._recorder:
                estimator_snapshot = self._get_estimator_snapshot()
                if estimator_snapshot is not None:
                    output_targets = {
                        self.head_LR_name: {"angle": round(self._estimators[self.head_LR_name].target, 2)},
                        self.head_UD_name: {"angle": round(self._estimators[self.head_UD_name].target, 2)},
                        self.body_LR_name: {"angle": round(self._estimators[self.body_LR_name].target, 2)},
                        self.body_UD_name: {"angle": round(self._estimators[self.body_UD_name].target, 2)},
                    }
                    record = build_frame_record(
                        camera=camera, detection=target_data_for_calc,
                        use_point=use_point, estimator_state=estimator_snapshot,
                        servo_middles=self._servo_middles.copy(),
                        servo_mins=self._servo_mins.copy(),
                        servo_maxs=self._servo_maxs.copy(),
                        cam_resolution=(self.cam_x, self.cam_y),
                        raw_world_lr=head_world_lr, raw_world_ud=head_world_ud,
                        smoothed_world_lr=final_lr, smoothed_world_ud=final_ud,
                        output_targets=output_targets,
                        fusion_state=self._fusion.state,
                        composite_score=composite,
                    )
                    self._recorder.log_frame(record)

        # Side cameras: drive servos from fused room state (single decision point)
        elif camera in (self.left_camera, self.right_camera):
            # Log side camera frames to recorder for debugging fusion
            if self._recorder:
                side_bbox = best_target.get(TrackingEnums.KEY_BOX.value, {})
                if side_bbox:
                    side_snap = self._get_estimator_snapshot()
                    side_lr = self._pixel_to_world_angle(side_bbox, camera, ServoEnum.X_AXIS.value)
                    side_record = build_frame_record(
                        camera=camera,
                        detection=side_bbox,
                        use_point=False,
                        estimator_state=side_snap,
                        servo_middles=self._servo_middles.copy(),
                        servo_mins=self._servo_mins.copy(),
                        servo_maxs=self._servo_maxs.copy(),
                        cam_resolution=(self.cam_x, self.cam_y),
                        raw_world_lr=side_lr,
                        raw_world_ud=0.0,
                        smoothed_world_lr=self._side_world_lr_smooth or side_lr,
                        smoothed_world_ud=0.0,
                        output_targets={},
                        fusion_state=self._fusion.state,
                        composite_score=RoomStateManager.compute_frame_composite(best_target) if self._room_state else 0.0,
                    )
                    self._recorder.log_frame(side_record)
            self._drive_from_room_state()

    def _handle_personality_modifier(self, msg: MQTTMessage) -> None:
        """Handle personality modifier from GLaDOSLocal (grudge + mood speed).

        When GLaDOS gets angry at someone (e.g., middle finger), this message
        arrives so the attention model watches that person more closely.
        Also updates tracking speed based on mood for physically aggressive movement.
        """
        j_msg = loads(msg.payload.decode())
        # Grudge tracking (attention model)
        if self._attention:
            person_id = j_msg.get("person_id", "")
            modifier = j_msg.get("modifier", 0.0)
            if person_id and modifier:
                self._attention.add_personality_modifier(person_id, modifier)
                self.logger.debug(
                    f"ATTENTION grudge: {person_id} bonus={modifier}")
        # Mood-driven tracking speed
        mood = j_msg.get("mood", "")
        if mood:
            speed_map = MotionProfile.MOOD_SPEED_MAP.value
            new_speed = speed_map.get(mood, speed_map.get("default", 3))
            if new_speed != self.dms:
                self.logger.info(f"Mood speed: {mood} -> speed {new_speed} (was {self.dms})")
                self.dms = new_speed

    def _handle_conversation_partner(self, msg: MQTTMessage) -> None:
        """Handle conversation partner from GLaDOSLocal.

        After GLaDOS speaks to someone, this message arrives so the attention
        model maintains gaze on the conversation partner while awaiting response.
        """
        if not self._attention:
            return
        j_msg = loads(msg.payload.decode())
        person_id = j_msg.get("person_id", "")
        if person_id:
            self._attention.set_conversation_partner(person_id)
            self.logger.debug(
                f"ATTENTION conversation: partner={person_id}")

    def handle_intensity(self, msg: MQTTMessage) -> None:
        """Handle intensity messages."""
        j_msg = loads(msg.payload.decode())
        if j_msg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic}, {j_msg}")
            self.intensity = j_msg.get("intensity", self.intensity)
