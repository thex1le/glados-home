import time
from typing import Dict, Callable, Tuple, NamedTuple, Any
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
                                        PersonalityEnums)
from glados_modules.RobotKinematics import RobotKinematics
from glados_modules.MotionRecorder import MotionRecorder, build_frame_record
from glados_modules.TraceLog import TraceLog


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
        if dt <= 0:
            return self.position
        accel = (self.omega ** 2) * (self.target - self.position) - 2.0 * self.zeta * self.omega * self.velocity
        self.velocity += accel * dt
        self.position += self.velocity * dt
        return self.position

    def sync(self, reported_position: float, reported_velocity: float = 0.0) -> None:
        """Gently correct from MQTT status (blend, don't snap)."""
        self.position = 0.7 * self.position + 0.3 * reported_position
        self.velocity = 0.7 * self.velocity + 0.3 * reported_velocity


class CameraFusionState:
    """Tracks which cameras see the target and manages handoff blending.

    State machine:
        SIDE_ONLY -> HANDOFF_TO_HEAD -> HEAD_TRACKING -> HANDOFF_TO_SIDE -> SIDE_ONLY

    Side cameras are fixed to the ceiling mount and provide absolute world-space
    yaw angles. The head camera provides precise yaw + pitch via FK/IK. During
    handoff between cameras, the world_lr target is linearly blended to prevent
    jerky transitions.
    """

    def __init__(self) -> None:
        self.state: str = FusionEnums.STATE_SIDE_ONLY.value
        self._head_last_seen: float = 0.0
        self._head_count: int = 0
        self._left_last_seen: float = 0.0
        self._right_last_seen: float = 0.0
        self._left_world_lr: float = 0.0
        self._right_world_lr: float = 0.0
        self._left_count: int = 0
        self._right_count: int = 0
        self._handoff_start_time: float = 0.0
        self._handoff_start_lr: float = 0.0
        # Angle history for predictive rotation
        self._left_angle_history: list = []
        self._right_angle_history: list = []

    def update_side_detection(self, camera: str, world_lr: float, count: int) -> None:
        """Record a side camera detection and angle history for prediction."""
        now = time.time()
        window = FusionEnums.PREDICTION_HISTORY_WINDOW.value
        if camera == CameraEnum.CAMERA_LEFT.value:
            self._left_last_seen = now
            self._left_world_lr = world_lr
            self._left_count = count
            self._left_angle_history.append((now, world_lr))
            self._left_angle_history = [(t, a) for t, a in self._left_angle_history
                                         if now - t <= window]
        elif camera == CameraEnum.CAMERA_RIGHT.value:
            self._right_last_seen = now
            self._right_world_lr = world_lr
            self._right_count = count
            self._right_angle_history.append((now, world_lr))
            self._right_angle_history = [(t, a) for t, a in self._right_angle_history
                                          if now - t <= window]

    def update_head_count(self, count: int) -> None:
        """Record head camera person count for room-level awareness."""
        self._head_count = count

    def update_head_detection(self) -> None:
        """Signal that the head camera has a detection this frame."""
        now = time.time()
        was_side_only = self.state in (FusionEnums.STATE_SIDE_ONLY.value,
                                       FusionEnums.STATE_HANDOFF_TO_SIDE.value)
        self._head_last_seen = now
        if was_side_only:
            best_side = self.get_best_side_world_lr()
            if best_side is not None:
                # Side camera had a recent detection — blend from its angle to head's
                self.state = FusionEnums.STATE_HANDOFF_TO_HEAD.value
                self._handoff_start_time = now
                self._handoff_start_lr = best_side
            else:
                # No side camera data — go straight to head tracking (no blend needed)
                self.state = FusionEnums.STATE_HEAD_TRACKING.value

    def head_lost(self) -> None:
        """Signal that the head camera lost the target."""
        if self.state in (FusionEnums.STATE_HEAD_TRACKING.value,
                          FusionEnums.STATE_HANDOFF_TO_HEAD.value):
            best_side = self.get_best_side_world_lr()
            if best_side is not None:
                self.state = FusionEnums.STATE_HANDOFF_TO_SIDE.value
                self._handoff_start_time = time.time()
            else:
                self.state = FusionEnums.STATE_SIDE_ONLY.value

    def get_best_side_world_lr(self) -> float:
        """Return the most recent non-stale side camera world angle, or None."""
        now = time.time()
        staleness = FusionEnums.SIDE_CAMERA_STALENESS.value
        left_fresh = (now - self._left_last_seen) < staleness if self._left_last_seen > 0 else False
        right_fresh = (now - self._right_last_seen) < staleness if self._right_last_seen > 0 else False

        if left_fresh and right_fresh:
            # Both fresh — use the more recent one
            if self._left_last_seen >= self._right_last_seen:
                return self._left_world_lr
            return self._right_world_lr
        elif left_fresh:
            return self._left_world_lr
        elif right_fresh:
            return self._right_world_lr
        return None

    def get_blended_world_lr(self, head_world_lr: float) -> float:
        """During handoff to head, blend from side angle to head angle.

        Args:
            head_world_lr: The head camera's computed world yaw angle.

        Returns:
            Blended world_lr (lerp from side to head over blend duration).
        """
        if self.state != FusionEnums.STATE_HANDOFF_TO_HEAD.value:
            return head_world_lr

        elapsed = time.time() - self._handoff_start_time
        duration = FusionEnums.HANDOFF_BLEND_DURATION.value
        if elapsed >= duration:
            self.state = FusionEnums.STATE_HEAD_TRACKING.value
            return head_world_lr

        # Linear interpolation: t=0 -> side angle, t=1 -> head angle
        t = elapsed / duration
        return self._handoff_start_lr + t * (head_world_lr - self._handoff_start_lr)

    def get_predicted_world_lr(self, camera: str) -> float:
        """Predict where the side camera target will be based on angular velocity.

        Uses angle history to estimate velocity, then leads the target
        by PREDICTION_LEAD_TIME seconds. Returns current angle if the
        target is stationary or history is insufficient.

        Args:
            camera: Which side camera to predict for.

        Returns:
            Predicted world_lr angle.
        """
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
        """Check if any non-stale side camera agrees with head's world angle.

        Args:
            head_world_lr: The head camera's computed world yaw angle.

        Returns:
            True if a side camera sees a target within the agreement threshold.
        """
        threshold = FusionEnums.HANDOFF_AGREEMENT_THRESHOLD.value
        best_side = self.get_best_side_world_lr()
        if best_side is None:
            return False
        return abs(head_world_lr - best_side) <= threshold

    def get_room_person_count(self) -> int:
        """Rough estimate of total people visible across all cameras.

        Uses max(head_count, left_count + right_count) to avoid
        double-counting people in overlapping FOVs.
        """
        return max(self._head_count, self._left_count + self._right_count)

    def side_can_drive_servos(self) -> bool:
        """Return True if side cameras should command servo movement.

        Also transitions back to SIDE_ONLY if the head camera has gone stale
        (e.g., RTSP stream dropped without sending a 'no target' frame).
        """
        if self.state in (FusionEnums.STATE_SIDE_ONLY.value,
                          FusionEnums.STATE_HANDOFF_TO_SIDE.value):
            return True

        # Head camera stream may have dropped — force transition after staleness timeout
        if self._head_last_seen > 0:
            head_age = time.time() - self._head_last_seen
            if head_age > FusionEnums.SIDE_CAMERA_STALENESS.value:
                self.state = FusionEnums.STATE_SIDE_ONLY.value
                return True

        return False


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
            self.intensity_topic: self.handle_intensity
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

        # World-space angle estimates (smoothed)
        self._world_lr: float = None
        self._world_ud: float = None
        self._world_smooth_alpha: float = MotionProfile.WORLD_SMOOTH_ALPHA.value

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

        # Last tracked target state for multi-person selection
        self._last_tracked_world_lr: float = None
        self._last_tracked_bbox_height: float = None
        self._last_tracked_face_id: str = None
        # Last known positions for memory glances (face_id -> world_lr)
        self._last_known_positions: Dict[str, float] = {}

        # Feature toggles (read from config, default to True)
        def _toggle(section: str, key: str, fallback: bool = True) -> bool:
            if config is None:
                return fallback
            return config.get(section, key, fallback=str(fallback)).strip().lower() == "true"

        self._enable_predictive = _toggle(FeatureToggles.CONFIG_HEAD.value,
                                           FeatureToggles.PREDICTIVE_ROTATION.value)
        self._enable_confirmation = _toggle(FeatureToggles.CONFIG_HEAD.value,
                                             FeatureToggles.PERIPHERAL_CONFIRMATION.value)
        self._enable_blending = _toggle(FeatureToggles.CONFIG_HEAD.value,
                                         FeatureToggles.HANDOFF_BLENDING.value)
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

        # Behavior state machine (active → idle → drowsy → asleep)
        self._behavior_state: str = BehaviorEnums.STATE_ACTIVE.value
        self._idle_start_time: float = 0.0
        self._drowsy_start_time: float = 0.0

        # Breathing parameters
        self._breathing_freq: float = MotionProfile.BREATHING_FREQ.value * 2 * pi
        self._breathing_amplitude: float = MotionProfile.BREATHING_AMPLITUDE.value

        # Motion recording (set to None to disable, or call enable_recording())
        self._recorder: MotionRecorder = None

        # Pipeline tracing
        self._tracer = TraceLog()

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
        """Enable motion frame recording. Returns the recording file path."""
        self._recorder = MotionRecorder(session_name=session_name, output_dir=output_dir)
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
                estimator.sync(float(servo_data.current), float(servo_data.velocity))

    def _get_estimated_position(self, servo_name: str) -> float:
        """Get the current estimated position from the spring-damper estimator."""
        if servo_name in self._estimators:
            return self._estimators[servo_name].get_position()
        return self._servo_middles.get(servo_name, MotionProfile.DEFAULT_SERVO_CENTER.value)

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
            fov = CameraEnum.CAMERA_RIGHT_FOV.value
        elif camera == CameraEnum.CAMERA_LEFT.value:
            fov = CameraEnum.CAMERA_LEFT_FOV.value
        else:
            fov = 54.0

        # Compute angular offset using arctan and focal length
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
            if axis == ServoEnum.X_AXIS.value:
                camera_world = yaw
            else:
                camera_world = pitch
        elif camera == CameraEnum.CAMERA_LEFT.value:
            # Side cameras are fixed to the ceiling mount — they don't rotate with the body.
            # Use FK-space yaw (0 = forward), not servo-space, so the IK interprets it correctly.
            camera_world = MotionProfile.CAMERA_LEFT_MOUNTING_OFFSET.value
        elif camera == CameraEnum.CAMERA_RIGHT.value:
            camera_world = MotionProfile.CAMERA_RIGHT_MOUNTING_OFFSET.value
        else:
            camera_world = 0.0

        # World angle = where camera is pointing + offset from frame center
        world_angle = camera_world + angle_offset_deg
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

    def _update_targets(self, target_world_lr: float, target_world_ud: float,
                         trace_id: str = None) -> None:
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

        self.logger.debug(f"IK debug: world_lr={target_world_lr:.1f} world_ud={target_world_ud:.1f} "
                          f"-> body_lr={body_lr_target:.1f} head_lr={head_lr_target:.1f}")

        # Clamp all to physical ranges
        # Add breathing oscillation to body_UD
        body_ud_target += self._get_breathing_offset()

        head_lr_target = self._clamp(head_lr_target, self.head_LR_name)
        head_ud_target = self._clamp(head_ud_target, self.head_UD_name)
        body_lr_target = self._clamp(body_lr_target, self.body_LR_name)
        body_ud_target = self._clamp(body_ud_target, self.body_UD_name)

        # Build consolidated move_all message
        targets = {
            self.head_LR_name: {ServoEnum.MSG_ANGLE.value: round(head_lr_target),
                                ServoEnum.MSG_SPEED.value: self.dms},
            self.head_UD_name: {ServoEnum.MSG_ANGLE.value: round(head_ud_target),
                                ServoEnum.MSG_SPEED.value: self.dms},
            self.body_LR_name: {ServoEnum.MSG_ANGLE.value: round(body_lr_target),
                                ServoEnum.MSG_SPEED.value: self.dms},
            self.body_UD_name: {ServoEnum.MSG_ANGLE.value: round(body_ud_target),
                                ServoEnum.MSG_SPEED.value: self.dms},
        }

        msg = ServoMessageBuilder.move_all(targets)
        if trace_id:
            msg[TraceEnums.TRACE_ID.value] = trace_id
            msg[TraceEnums.TS_VISION.value] = self._tracer._active.get(trace_id, {}).get("ts_vision")
        if self._enable_movement:
            self.send_command(msg, ServoEnum.MQTT_COMMAND_TOPIC.value)
        else:
            self.logger.debug(f"Movement disabled — would send: {targets}")

        # Update local estimators to match what we just commanded
        speed = self.dms
        head_omega, head_zeta = MotionProfile.HEAD_PARAMS.value[speed]
        body_omega, body_zeta = MotionProfile.BODY_PARAMS.value[speed]
        self._estimators[self.head_LR_name].set_target(head_lr_target, head_omega, head_zeta)
        self._estimators[self.head_UD_name].set_target(head_ud_target, head_omega, head_zeta)
        self._estimators[self.body_LR_name].set_target(body_lr_target, body_omega, body_zeta)
        self._estimators[self.body_UD_name].set_target(body_ud_target, body_omega, body_zeta)

        self.logger.debug(
            f"Targets sent: head_LR={head_lr_target:.1f} head_UD={head_ud_target:.1f} "
            f"body_LR={body_lr_target:.1f} body_UD={body_ud_target:.1f}"
        )

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
            self._update_targets(best_side, current_pitch)
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
                self._update_targets(glance_lr, head_ud_mid)
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

        self._update_targets(head_lr_mid + lr, head_ud_mid + ud)
        if not self._idle_active:
            self._idle_active = True
            self._idle_start_time = now
            self._behavior_state = BehaviorEnums.STATE_IDLE.value
            self.logger.debug("Entering idle drift mode")
        if self.debug_overlay_enabled:
            self._debug_overlay["state"] = self._behavior_state

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
                # Face ID match: strong signal (40%)
                if face_id and face_id == self._last_tracked_face_id:
                    score += 0.4
                # Proximity (30%)
                if bbox:
                    world_lr = self._pixel_to_world_angle(bbox, camera, ServoEnum.X_AXIS.value)
                    angle_diff = abs(world_lr - self._last_tracked_world_lr)
                    proximity = max(0.0, 1.0 - angle_diff / 90.0)
                    score += proximity * 0.3
                # Height similarity (15%)
                if bbox and self._last_tracked_bbox_height and self._last_tracked_bbox_height > 0:
                    height = bbox.get('y2', 0) - bbox.get('y1', 0)
                    if height > 0:
                        height_ratio = min(height, self._last_tracked_bbox_height) / max(height, self._last_tracked_bbox_height)
                        score += height_ratio * 0.15
                # Confidence (15%)
                score += confidence * 0.15
            else:
                # No face history — use position-based scoring
                if bbox:
                    world_lr = self._pixel_to_world_angle(bbox, camera, ServoEnum.X_AXIS.value)
                    angle_diff = abs(world_lr - self._last_tracked_world_lr)
                    proximity = max(0.0, 1.0 - angle_diff / 90.0)
                    score += proximity * 0.5
                if bbox and self._last_tracked_bbox_height and self._last_tracked_bbox_height > 0:
                    height = bbox.get('y2', 0) - bbox.get('y1', 0)
                    if height > 0:
                        height_ratio = min(height, self._last_tracked_bbox_height) / max(height, self._last_tracked_bbox_height)
                        score += height_ratio * 0.3
                score += confidence * 0.2

            if score > best_score:
                best_score = score
                best = p

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

        # Periodically sync estimators from MQTT status
        self._sync_estimators_from_status()

        vision_map = self.vision_tracker.get_vision_map()
        if camera not in vision_map:
            if camera == self.main_camera:
                self._fusion.head_lost()
            if time.time() - self._last_target_time > self._idle_timeout:
                if self._enable_idle_drift:
                    self._generate_idle_drift()
            return

        target_data = vision_map[camera].get(self.target, {})
        if target_data.get(self.count, 0) == 0:
            if camera == self.main_camera:
                self._fusion.head_lost()
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

        best_target = self.__select_target(target_data[self.objects], camera)
        if not best_target:
            return

        # Side cameras: always record world angle in fusion state (even during head tracking)
        if camera in (self.left_camera, self.right_camera):
            bbox = best_target.get(TrackingEnums.KEY_BOX.value, {})
            if bbox:
                side_world_lr = self._pixel_to_world_angle(bbox, camera, ServoEnum.X_AXIS.value)
                bbox_cx = (bbox.get('x1', 0) + bbox.get('x2', 0)) / 2
                self.logger.info(f"Side raw: {camera} bbox_cx={bbox_cx:.0f} raw_world_lr={side_world_lr:.1f}")
                self._fusion.update_side_detection(camera, side_world_lr,
                                                    target_data.get(self.count, 0))

        # Head camera: full tracking with world-space angles
        if camera == self.main_camera:
            # Signal head detection to fusion state machine
            self._fusion.update_head_detection()

            # Check for pose data (prefer nose point over bounding box)
            use_point = False
            target_data_for_calc = best_target.get(TrackingEnums.KEY_BOX.value, {})
            if TrackingEnums.KEY_POSE.value in best_target:
                pose_data = best_target[TrackingEnums.KEY_POSE.value]
                if self.pose_target in pose_data:
                    target_data_for_calc = pose_data[self.pose_target]
                    use_point = True

            # Convert to world-space angles
            world_lr = self._pixel_to_world_angle(target_data_for_calc, camera,
                                                   ServoEnum.X_AXIS.value, point=use_point)
            world_ud = self._pixel_to_world_angle(target_data_for_calc, camera,
                                                   ServoEnum.Y_AXIS.value, point=use_point)

            # Apply handoff blending if transitioning from side camera
            if self._enable_blending:
                world_lr = self._fusion.get_blended_world_lr(world_lr)

            # Update head person count for room-level awareness
            self._fusion.update_head_count(target_data.get(self.count, 0))

            # Smooth in world space — tighter alpha when side camera confirms
            if self._world_lr is None:
                self._world_lr = world_lr
                self._world_ud = world_ud
            else:
                alpha = self._world_smooth_alpha
                if self._enable_confirmation and self._fusion.is_confirmed_by_side(world_lr):
                    alpha = FusionEnums.CONFIRMED_SMOOTH_ALPHA.value
                self._world_lr = alpha * self._world_lr + (1 - alpha) * world_lr
                self._world_ud = alpha * self._world_ud + (1 - alpha) * world_ud

            # Record frame BEFORE sending (captures inputs + will capture outputs)
            estimator_snapshot = self._get_estimator_snapshot() if self._recorder else None

            # Trace: mark tracking start
            ts_track_start = time.time()

            self._update_targets(self._world_lr, self._world_ud, trace_id=trace_id)

            # Save tracking state for multi-person target selection scoring
            self._last_tracked_world_lr = self._world_lr
            bbox_for_height = best_target.get(TrackingEnums.KEY_BOX.value, {})
            if bbox_for_height:
                self._last_tracked_bbox_height = float(
                    bbox_for_height.get('y2', 0) - bbox_for_height.get('y1', 0))
            # Store face ID for identity-based target selection
            face_data = best_target.get(VisionResultsEnum.VISION_RESULTS_FACE_KEY.value, {})
            if face_data.get(VisionResultsEnum.VISION_RESULTS_FACE_ID_KEY.value):
                self._last_tracked_face_id = face_data[VisionResultsEnum.VISION_RESULTS_FACE_ID_KEY.value]
                # Remember where this person was for memory glances
                if self._last_tracked_face_id != "unknown":
                    self._last_known_positions[self._last_tracked_face_id] = self._world_lr

            # Update debug overlay for RTSP stream visualization
            if self.debug_overlay_enabled:
                self._debug_overlay = {
                    "state": self._fusion.state,
                    "world_lr": round(self._world_lr, 1),
                    "world_ud": round(self._world_ud, 1),
                    "head_lr": round(self._estimators[self.head_LR_name].target, 1),
                    "head_ud": round(self._estimators[self.head_UD_name].target, 1),
                    "body_lr": round(self._estimators[self.body_LR_name].target, 1),
                    "body_ud": round(self._estimators[self.body_UD_name].target, 1),
                    "est_head_lr": round(self._get_estimated_position(self.head_LR_name), 1),
                    "est_head_ud": round(self._get_estimated_position(self.head_UD_name), 1),
                    "est_body_lr": round(self._get_estimated_position(self.body_LR_name), 1),
                    "est_body_ud": round(self._get_estimated_position(self.body_UD_name), 1),
                }

            # Trace: log the full pipeline record
            if trace_id:
                self._tracer.end_trace(trace_id,
                    ts_track_start=ts_track_start,
                    ts_track_end=time.time(),
                    world_lr=round(self._world_lr, 2),
                    world_ud=round(self._world_ud, 2),
                    targets={
                        self.head_LR_name: round(self._estimators[self.head_LR_name].target, 1),
                        self.head_UD_name: round(self._estimators[self.head_UD_name].target, 1),
                        self.body_LR_name: round(self._estimators[self.body_LR_name].target, 1),
                        self.body_UD_name: round(self._estimators[self.body_UD_name].target, 1),
                    }
                )

            # Log frame to recorder if enabled
            if self._recorder and estimator_snapshot is not None:
                output_targets = {
                    self.head_LR_name: {"angle": round(self._estimators[self.head_LR_name].target, 2)},
                    self.head_UD_name: {"angle": round(self._estimators[self.head_UD_name].target, 2)},
                    self.body_LR_name: {"angle": round(self._estimators[self.body_LR_name].target, 2)},
                    self.body_UD_name: {"angle": round(self._estimators[self.body_UD_name].target, 2)},
                }
                record = build_frame_record(
                    camera=camera,
                    detection=target_data_for_calc,
                    use_point=use_point,
                    estimator_state=estimator_snapshot,
                    servo_middles=self._servo_middles.copy(),
                    servo_mins=self._servo_mins.copy(),
                    servo_maxs=self._servo_maxs.copy(),
                    cam_resolution=(self.cam_x, self.cam_y),
                    raw_world_lr=world_lr,
                    raw_world_ud=world_ud,
                    smoothed_world_lr=self._world_lr,
                    smoothed_world_ud=self._world_ud,
                    output_targets=output_targets,
                    fusion_state=self._fusion.state,
                )
                self._recorder.log_frame(record)

        # Side cameras: drive servos through full IK when head camera is not tracking
        elif camera in (self.left_camera, self.right_camera):
            if self._fusion.side_can_drive_servos():
                bbox = best_target.get(TrackingEnums.KEY_BOX.value, {})
                if bbox:
                    # Use predicted angle if enabled, otherwise raw angle
                    if self._enable_predictive:
                        world_lr = self._fusion.get_predicted_world_lr(camera)
                    else:
                        world_lr = self._pixel_to_world_angle(bbox, camera, ServoEnum.X_AXIS.value)
                    # Use current body_UD as the UD target (side cameras don't provide vertical info)
                    current_ud = self._get_estimated_position(self.body_UD_name)
                    # Use the head camera's FK pitch if available, otherwise keep body UD stable
                    current_angles = {
                        self.body_LR_name: self._get_estimated_position(self.body_LR_name),
                        self.body_UD_name: current_ud,
                        self.head_LR_name: self._get_estimated_position(self.head_LR_name),
                        self.head_UD_name: self._get_estimated_position(self.head_UD_name),
                    }
                    _, world_ud = self._kinematics.forward_kinematics(current_angles)
                    self.logger.info(f"Side camera {camera}: world_lr={world_lr:.1f} world_ud={world_ud:.1f} "
                                     f"bbox_cx={(bbox.get('x1',0)+bbox.get('x2',0))/2:.0f} "
                                     f"predictive={self._enable_predictive} fusion={self._fusion.state}")
                    self._update_targets(world_lr, world_ud)

    def handle_intensity(self, msg: MQTTMessage) -> None:
        """Handle intensity messages."""
        j_msg = loads(msg.payload.decode())
        if j_msg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic}, {j_msg}")
            self.intensity = j_msg.get("intensity", self.intensity)
