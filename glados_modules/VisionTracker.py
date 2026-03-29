import time
from typing import Dict, Callable, Tuple, NamedTuple, Any
from json import loads
from collections import namedtuple
from math import sin, radians, tan, atan, degrees

# 3rd party imports
from paho.mqtt.client import MQTTMessage

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.MqttConnector import MQTTClient, ServoMessageBuilder
from glados_modules.MqttConsumerModules import ServoLocation, VisionTracker
from glados_modules.GladosEnums import (CameraEnum, ServoEnum, SystemEnums,
                                        TrackingEnums, VisionResultsEnum, LoggingEnums,
                                        MotionProfile, TraceEnums, KinematicsEnums)
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

        # Side camera tracking
        self.side_camera_count: int = 0

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
            body_lr = self._get_estimated_position(self.body_LR_name)
            camera_world = body_lr + MotionProfile.CAMERA_LEFT_MOUNTING_OFFSET.value
        elif camera == CameraEnum.CAMERA_RIGHT.value:
            body_lr = self._get_estimated_position(self.body_LR_name)
            camera_world = body_lr + MotionProfile.CAMERA_RIGHT_MOUNTING_OFFSET.value
        else:
            camera_world = 90.0

        # World angle = where camera is pointing + offset from frame center
        world_angle = camera_world + angle_offset_deg
        return world_angle

    def _clamp(self, value: float, servo_name: str) -> float:
        """Clamp a value to the servo's physical range."""
        return max(self._servo_mins[servo_name], min(self._servo_maxs[servo_name], value))

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

        # Clamp all to physical ranges
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
        self.send_command(msg, ServoEnum.MQTT_COMMAND_TOPIC.value)

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
        """Generate subtle organic drift when no target is detected.
        Layered sine waves at irrational-ratio frequencies produce smooth, non-repeating sway.
        """
        now = time.time()
        if now - self._last_idle_send < self._idle_interval:
            return
        self._last_idle_send = now

        t = now
        lr = 2.5 * sin(0.3 * t) + 1.0 * sin(0.7 * t + 1.2) + 0.4 * sin(1.4 * t + 0.5)
        ud = 1.8 * sin(0.25 * t + 0.8) + 0.6 * sin(0.55 * t + 2.1)

        head_lr_mid = self._servo_middles[self.head_LR_name]
        head_ud_mid = self._servo_middles[self.head_UD_name]

        # Idle drift uses the same _update_targets path so body follows naturally
        self._update_targets(head_lr_mid + lr, head_ud_mid + ud)
        if not self._idle_active:
            self._idle_active = True
            self.logger.debug("Entering idle drift mode")
        if self.debug_overlay_enabled:
            self._debug_overlay["state"] = "IDLE"

    def __find_target(self, seen_data: list) -> dict:
        """Find the detection with the highest confidence."""
        confidence_key = VisionResultsEnum.VISION_RESULTS_CONFIDENCE_KEY.value
        best = {}
        highest = 0
        for p in seen_data:
            if p[confidence_key] > highest:
                highest = p[confidence_key]
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
            # No data for this camera -- check if we should idle
            if time.time() - self._last_target_time > self._idle_timeout:
                self._generate_idle_drift()
            return

        target_data = vision_map[camera].get(self.target, {})
        if target_data.get(self.count, 0) == 0:
            if time.time() - self._last_target_time > self._idle_timeout:
                self._generate_idle_drift()
            return

        # Found a target
        self._last_target_time = time.time()
        self._idle_active = False

        # Extract trace_id from vision results (stamped by MachineVision)
        trace_id = vision_map[camera].get(TraceEnums.TRACE_ID.value)
        ts_vision = vision_map[camera].get(TraceEnums.TS_VISION.value)

        best_target = self.__find_target(target_data[self.objects])
        if not best_target:
            return

        # Head camera: full tracking with world-space angles
        if camera == self.main_camera:
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

            # Smooth in world space
            if self._world_lr is None:
                self._world_lr = world_lr
                self._world_ud = world_ud
            else:
                alpha = self._world_smooth_alpha
                self._world_lr = alpha * self._world_lr + (1 - alpha) * world_lr
                self._world_ud = alpha * self._world_ud + (1 - alpha) * world_ud

            # Record frame BEFORE sending (captures inputs + will capture outputs)
            estimator_snapshot = self._get_estimator_snapshot() if self._recorder else None

            # Trace: mark tracking start
            ts_track_start = time.time()

            self._update_targets(self._world_lr, self._world_ud, trace_id=trace_id)
            self.side_camera_count = 0

            # Update debug overlay for RTSP stream visualization
            if self.debug_overlay_enabled:
                self._debug_overlay = {
                    "state": "TRACKING",
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
                # Reconstruct the output targets from what _update_targets computed
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
                )
                self._recorder.log_frame(record)

        # Side cameras: rotate body toward detection so head camera can pick it up
        elif camera in (self.left_camera, self.right_camera):
            if self.side_camera_count <= 5:
                bbox = best_target.get(TrackingEnums.KEY_BOX.value, {})
                if bbox:
                    world_lr = self._pixel_to_world_angle(bbox, camera, ServoEnum.X_AXIS.value)
                    # Only send body LR target for side camera detections
                    body_lr_target = self._clamp(world_lr, self.body_LR_name)
                    targets = {
                        self.body_LR_name: {ServoEnum.MSG_ANGLE.value: round(body_lr_target),
                                            ServoEnum.MSG_SPEED.value: self.dms},
                    }
                    msg = ServoMessageBuilder.move_all(targets)
                    self.send_command(msg, ServoEnum.MQTT_COMMAND_TOPIC.value)
                    self._estimators[self.body_LR_name].set_target(body_lr_target)
                    self.side_camera_count += 1

    def handle_intensity(self, msg: MQTTMessage) -> None:
        """Handle intensity messages."""
        j_msg = loads(msg.payload.decode())
        if j_msg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic}, {j_msg}")
            self.intensity = j_msg.get("intensity", self.intensity)
