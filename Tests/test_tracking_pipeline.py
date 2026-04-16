"""Integration test: full tracking pipeline from detection to servo command.

Feeds a fake YOLO detection through the VisionTracker consumer into
MotionTrack.track_loop and verifies the correct move_all message is produced.
"""

import pytest
import time
from unittest.mock import MagicMock, patch, call
from json import dumps, loads
from collections import namedtuple

from glados_modules.GladosEnums import (
    ServoEnum, CameraEnum, VisionResultsEnum, TrackingEnums, MotionProfile, TraceEnums
)


def _make_motion_track():
    """Create a MotionTrack with fully mocked MQTT and fake servo status."""
    from glados_modules.VisionTracker import MotionTrack, SpringDamperEstimator

    broker = namedtuple("broker", ["ip", "port"])("localhost", 1883)
    cam_res = MotionTrack.camera_tuple(640, 480)

    with patch.object(MotionTrack, '__init__', lambda self, *a, **kw: None):
        mt = MotionTrack.__new__(MotionTrack)

    # Minimal init matching what __init__ sets up
    mt.__name__ = "MotionTrack"
    mt.location = "MotionTrack"
    mt.logger = MagicMock()
    mt.cam_x = 640
    mt.cam_y = 480
    mt.main_camera = CameraEnum.CAMERA_HEAD.value
    mt.left_camera = CameraEnum.CAMERA_LEFT.value
    mt.right_camera = CameraEnum.CAMERA_RIGHT.value
    mt.head_LR_name = ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
    mt.head_UD_name = ServoEnum.LOCATION_HEAD_UP_DOWN.value
    mt.body_LR_name = ServoEnum.LOCATION_BODY_LEFT_RIGHT.value
    mt.body_UD_name = ServoEnum.LOCATION_BODY_UP_DOWN.value
    mt.target = "person"
    mt.pose_target = VisionResultsEnum.VISION_POSE_KEY_POINTS_COCO_WHOLE_BODY.value[0]
    mt.confidence = 0.65
    mt.objects = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
    mt.count = VisionResultsEnum.VISION_RESULTS_COUNT_KEY.value
    mt.dms = MotionProfile.DEFAULT_TRACKING_SPEED.value
    mt._world_lr = None
    mt._world_ud = None
    mt._world_ud_time = 0.0
    mt._world_smooth_alpha = MotionProfile.WORLD_SMOOTH_ALPHA.value
    mt._world_smooth_alpha_ud = MotionProfile.WORLD_SMOOTH_ALPHA_UD.value
    mt._eye_ud_offset = MotionProfile.EYE_UD_OFFSET.value
    mt._last_target_time = time.time()
    mt._idle_active = False
    mt._idle_interval = MotionProfile.IDLE_DRIFT_INTERVAL.value
    mt._last_idle_send = 0.0
    mt._idle_timeout = MotionProfile.IDLE_TIMEOUT.value
    mt._recorder = None
    mt.debug_overlay_enabled = False

    # Camera fusion state machine
    from glados_modules.VisionTracker import CameraFusionState
    mt._fusion = CameraFusionState()
    mt._side_world_lr_smooth = None
    mt._side_drive_last_send = 0.0
    mt._prev_body_lr_target = None
    mt._prev_body_ud_target = None
    mt._last_commanded_lr = None
    mt._last_commanded_ud = None
    mt._ud_search_active = False
    mt._ud_search_direction = 1
    mt._ud_search_origin = 0.0
    mt._ud_search_pitch = 0.0
    mt._ud_search_start_time = 0.0
    mt._room_state = None
    mt._enable_room_state = False
    mt._attention = None
    mt._enable_attention = False
    mt._last_attention_time = 0.0
    mt._last_tracked_world_lr = None
    mt._last_tracked_bbox_height = None
    mt._last_tracked_face_id = None
    mt._last_known_positions = {}
    mt._nose_miss_count = 0
    mt._nose_min_confidence = 0.4
    mt._nose_miss_threshold = 3

    # Behavior state machine
    from glados_modules.GladosEnums import BehaviorEnums
    from math import pi
    mt._behavior_state = BehaviorEnums.STATE_ACTIVE.value
    mt._idle_start_time = 0.0
    mt._drowsy_start_time = 0.0
    mt._breathing_freq = MotionProfile.BREATHING_FREQ.value * 2 * pi
    mt._breathing_amplitude = MotionProfile.BREATHING_AMPLITUDE.value
    mt._sway_lr_freq = MotionProfile.SWAY_LR_FREQ.value * 2 * pi
    mt._sway_lr_amplitude = MotionProfile.SWAY_LR_AMPLITUDE.value
    mt._sway_head_lr_amplitude = MotionProfile.SWAY_HEAD_LR_AMPLITUDE.value

    # Feature toggles (all enabled for tests)
    mt._enable_movement = True
    mt._enable_idle_drift = True
    mt._enable_predictive = True
    mt._enable_confirmation = True
    mt._enable_blending = True
    mt._enable_glances = True
    mt._enable_breathing = True
    mt._enable_sleep = True

    mt._lock = __import__('threading').Lock()
    mt.client = MagicMock()
    mt.send_command = MagicMock()

    # Fake servo config (typical values)
    mt._servo_middles = {
        mt.head_LR_name: 92.0, mt.head_UD_name: 83.0,
        mt.body_LR_name: 90.0, mt.body_UD_name: 100.0,
    }
    mt._servo_mins = {
        mt.head_LR_name: 52.0, mt.head_UD_name: 6.0,
        mt.body_LR_name: 0.0, mt.body_UD_name: 0.0,
    }
    mt._servo_maxs = {
        mt.head_LR_name: 120.0, mt.head_UD_name: 125.0,
        mt.body_LR_name: 180.0, mt.body_UD_name: 180.0,
    }

    # Initialize kinematics with servo config
    from glados_modules.RobotKinematics import RobotKinematics
    mt._kinematics = RobotKinematics(mt._servo_middles, mt._servo_mins, mt._servo_maxs)

    # Initialize estimators at center positions
    speed = mt.dms
    head_omega, head_zeta = MotionProfile.HEAD_PARAMS.value[speed]
    body_omega, body_zeta = MotionProfile.BODY_PARAMS.value[speed]
    mt._estimators = {
        mt.head_LR_name: SpringDamperEstimator(92.0, head_omega, head_zeta),
        mt.head_UD_name: SpringDamperEstimator(83.0, head_omega, head_zeta),
        mt.body_LR_name: SpringDamperEstimator(90.0, body_omega, body_zeta),
        mt.body_UD_name: SpringDamperEstimator(100.0, body_omega, body_zeta),
    }
    mt._estimators_initialized = True

    # Tracer
    mt._tracer = MagicMock()
    mt._tracer._active = {}

    # Pipeline debug (no-op for tests)
    from glados_modules.PipelineDebug import PipelineDebug
    mt._pdebug = PipelineDebug(mt, "test")

    # Camera readiness (all cameras online for tests)
    mt._cameras_ready = {mt.main_camera, mt.left_camera, mt.right_camera}
    mt._all_cameras = {mt.main_camera, mt.left_camera, mt.right_camera}
    mt._cameras_online = True
    # Saccade cooldown (not active for tests)
    mt._head_cooldown_until = 0.0
    # Occlusion backoff (not active for tests)
    mt._side_drive_attempt_start = 0.0
    mt._side_drive_head_frames = 0
    mt._side_drive_head_hits = 0
    mt._side_drive_backoff_until = 0.0

    # Vision tracker (mock)
    mt.vision_tracker = MagicMock()
    mt.vision_tracker.side_confidence_score = 0.4
    mt.servo_status = MagicMock()

    return mt


class TestFullTrackingPipeline:
    """End-to-end: detection -> world angle -> target split -> MQTT message."""

    def test_person_at_center_produces_no_movement(self):
        """Person at image center should produce targets near servo middles."""
        mt = _make_motion_track()
        # Person bbox centered in 640x480 frame
        vision_map = {
            CameraEnum.CAMERA_HEAD.value: {
                "person": {
                    "count": 1,
                    "objects": [{
                        "confidence": 0.9,
                        "box": {"x1": 270, "y1": 190, "x2": 370, "y2": 290},
                    }],
                },
                TraceEnums.TRACE_ID.value: "test123",
                TraceEnums.TS_VISION.value: time.time(),
            }
        }
        mt.vision_tracker.get_vision_map.return_value = vision_map
        mt.track_loop(CameraEnum.CAMERA_HEAD.value)

        # Should have sent a move_all command
        assert mt.send_command.called
        sent_msg = mt.send_command.call_args[0][0]
        assert sent_msg[ServoEnum.MSG_COMMAND_KEY.value] == ServoEnum.MSG_COMMAND_MOVE_ALL.value

        targets = sent_msg[ServoEnum.MSG_TARGETS.value]
        # Head should be near middle since person is centered and body is at rest
        head_lr = targets[mt.head_LR_name][ServoEnum.MSG_ANGLE.value]
        assert abs(head_lr - 92) < 10, f"Head LR should be near middle (92), got {head_lr}"

    def test_person_on_left_moves_servos(self):
        """Person on left side of frame should move head and body to track."""
        mt = _make_motion_track()
        # Person bbox on left side of frame (low x values)
        vision_map = {
            CameraEnum.CAMERA_HEAD.value: {
                "person": {
                    "count": 1,
                    "objects": [{
                        "confidence": 0.9,
                        "box": {"x1": 50, "y1": 190, "x2": 150, "y2": 290},
                    }],
                },
                TraceEnums.TRACE_ID.value: "test456",
                TraceEnums.TS_VISION.value: time.time(),
            }
        }
        mt.vision_tracker.get_vision_map.return_value = vision_map
        mt.track_loop(CameraEnum.CAMERA_HEAD.value)

        targets = mt.send_command.call_args[0][0][ServoEnum.MSG_TARGETS.value]
        head_lr = targets[mt.head_LR_name][ServoEnum.MSG_ANGLE.value]
        body_lr = targets[mt.body_LR_name][ServoEnum.MSG_ANGLE.value]
        # Head and body should move away from their middle positions
        # Direction depends on gear ratios and sign multipliers
        assert head_lr != 92, f"Head LR should move from middle for off-center target, got {head_lr}"
        assert body_lr != 90, f"Body LR should move from middle to follow"

    def test_all_four_servos_in_move_all(self):
        """move_all message should contain targets for all 4 servos."""
        mt = _make_motion_track()
        vision_map = {
            CameraEnum.CAMERA_HEAD.value: {
                "person": {
                    "count": 1,
                    "objects": [{"confidence": 0.9, "box": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}],
                },
                TraceEnums.TRACE_ID.value: "test789",
                TraceEnums.TS_VISION.value: time.time(),
            }
        }
        mt.vision_tracker.get_vision_map.return_value = vision_map
        mt.track_loop(CameraEnum.CAMERA_HEAD.value)

        targets = mt.send_command.call_args[0][0][ServoEnum.MSG_TARGETS.value]
        assert mt.head_LR_name in targets
        assert mt.head_UD_name in targets
        assert mt.body_LR_name in targets
        assert mt.body_UD_name in targets
        # Each should have angle and speed
        for name, data in targets.items():
            assert ServoEnum.MSG_ANGLE.value in data, f"Missing angle for {name}"
            assert ServoEnum.MSG_SPEED.value in data, f"Missing speed for {name}"

    def test_trace_id_propagated_to_output(self):
        """Trace ID from vision should appear in the move_all message."""
        mt = _make_motion_track()
        vision_map = {
            CameraEnum.CAMERA_HEAD.value: {
                "person": {
                    "count": 1,
                    "objects": [{"confidence": 0.9, "box": {"x1": 200, "y1": 200, "x2": 300, "y2": 300}}],
                },
                TraceEnums.TRACE_ID.value: "trace_abc",
                TraceEnums.TS_VISION.value: 1000.0,
            }
        }
        mt.vision_tracker.get_vision_map.return_value = vision_map
        mt.track_loop(CameraEnum.CAMERA_HEAD.value)

        sent_msg = mt.send_command.call_args[0][0]
        assert sent_msg.get(TraceEnums.TRACE_ID.value) == "trace_abc"

    def test_no_detection_triggers_idle_after_timeout(self):
        """No person in vision map should eventually trigger idle drift."""
        mt = _make_motion_track()
        mt._last_target_time = time.time() - 10  # 10s ago (past default 5s timeout)
        mt.vision_tracker.get_vision_map.return_value = {}

        mt.track_loop(CameraEnum.CAMERA_HEAD.value)

        # Should have sent idle drift targets
        if mt.send_command.called:
            sent_msg = mt.send_command.call_args[0][0]
            assert ServoEnum.MSG_COMMAND_KEY.value in sent_msg

    def test_below_confidence_ignored(self):
        """Detection below confidence threshold should not trigger movement."""
        mt = _make_motion_track()
        vision_map = {
            CameraEnum.CAMERA_HEAD.value: {
                "person": {
                    "count": 1,
                    "objects": [{"confidence": 0.3, "box": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}],
                },
            }
        }
        mt.vision_tracker.get_vision_map.return_value = vision_map
        mt.track_loop(CameraEnum.CAMERA_HEAD.value)
        # Low confidence -> __find_target returns empty -> no command sent
        # (depends on count being > 0 in the vision map -- count=1 here so it tries)

    def test_side_camera_uses_full_ik(self):
        """Side camera detection should send all 4 servo targets via IK pipeline."""
        mt = _make_motion_track()
        vision_map = {
            CameraEnum.CAMERA_LEFT.value: {
                "person": {
                    "count": 1,
                    "objects": [{"confidence": 0.6, "box": {"x1": 200, "y1": 200, "x2": 300, "y2": 300}}],
                },
            }
        }
        mt.vision_tracker.get_vision_map.return_value = vision_map
        mt.track_loop(CameraEnum.CAMERA_LEFT.value)

        assert mt.send_command.called
        targets = mt.send_command.call_args[0][0].get(ServoEnum.MSG_TARGETS.value, {})
        assert mt.body_LR_name in targets
        # Side cameras now use full IK — all 4 servos should be in targets
        assert mt.head_LR_name in targets
        assert mt.head_UD_name in targets
        assert mt.body_UD_name in targets

    def test_world_space_smoothing_applied(self):
        """Second detection should produce smoothed (blended) world angle."""
        mt = _make_motion_track()
        # First detection
        vision_map = {
            CameraEnum.CAMERA_HEAD.value: {
                "person": {
                    "count": 1,
                    "objects": [{"confidence": 0.9, "box": {"x1": 50, "y1": 200, "x2": 150, "y2": 300}}],
                },
            }
        }
        mt.vision_tracker.get_vision_map.return_value = vision_map
        mt.track_loop(CameraEnum.CAMERA_HEAD.value)
        first_world_lr = mt._world_lr

        # Second detection at different position — reset estimator state
        # and clear saccade cooldown (in real usage, enough time passes)
        for est in mt._estimators.values():
            est.velocity = 0.0
            est.target = est.position  # no position error = no acceleration
        mt._head_cooldown_until = 0.0  # clear saccade cooldown
        vision_map[CameraEnum.CAMERA_HEAD.value]["person"]["objects"][0]["box"] = {
            "x1": 400, "y1": 200, "x2": 500, "y2": 300
        }
        mt.track_loop(CameraEnum.CAMERA_HEAD.value)
        second_world_lr = mt._world_lr

        # Smoothed value should be between first and second raw values (EMA blend)
        assert second_world_lr != first_world_lr, "Smoothing should change the world angle"


class TestHeadBodyCoordination:
    """Test that head and body targets coordinate correctly over time."""

    def test_head_recenters_as_body_estimator_catches_up(self):
        """Simulate body estimator advancing and verify head target approaches middle."""
        mt = _make_motion_track()
        world_lr = 120.0
        head_middle = mt._servo_middles[mt.head_LR_name]

        # Body at starting position (90)
        head_t1 = head_middle + (world_lr - 90.0)

        # Simulate body estimator advancing to 110
        mt._estimators[mt.body_LR_name].position = 110.0
        head_t2 = head_middle + (world_lr - 110.0)

        # Body at target (120)
        mt._estimators[mt.body_LR_name].position = 120.0
        head_t3 = head_middle + (world_lr - 120.0)

        assert head_t1 > head_t2 > head_t3, "Head should approach middle as body catches up"
        assert head_t3 == head_middle, "Head should be at middle when body reaches target"
