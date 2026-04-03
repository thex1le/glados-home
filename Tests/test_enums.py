"""Tests for enum consistency and completeness.

Validates that all enums referenced in code actually exist, that MQTT topics
are defined as enums, and that cross-system message contracts are consistent.
"""

import pytest
from glados_modules.GladosEnums import (
    ServoEnum, CameraEnum, MQTTEnums, TrackingEnums, VisionResultsEnum,
    MotionProfile, LEDHead, LEDShoulders, LEDLampStrip8, LCDEnums,
    TraceEnums, DashboardEnums, LoggingEnums, SystemEnums,
    IMUEnums, TOFEnums, THEnums, MOXEnums, FusionEnums,
)


class TestServoEnums:
    """Validate servo location and message enums."""

    def test_all_servo_locations_defined(self):
        locations = [
            ServoEnum.LOCATION_HEAD_UP_DOWN,
            ServoEnum.LOCATION_HEAD_LEFT_RIGHT,
            ServoEnum.LOCATION_BODY_UP_DOWN,
            ServoEnum.LOCATION_BODY_LEFT_RIGHT,
        ]
        for loc in locations:
            assert isinstance(loc.value, str)
            assert len(loc.value) > 0

    def test_head_body_location_sets_complete(self):
        head_locs = ServoEnum.HEAD_SERVO_LOCATIONS.value
        body_locs = ServoEnum.BODY_SERVO_LOCATIONS.value
        assert ServoEnum.LOCATION_HEAD_UP_DOWN.value in head_locs
        assert ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value in head_locs
        assert ServoEnum.LOCATION_BODY_UP_DOWN.value in body_locs
        assert ServoEnum.LOCATION_BODY_LEFT_RIGHT.value in body_locs
        # No overlap
        assert not set(head_locs) & set(body_locs)

    def test_message_keys_exist(self):
        required_keys = [
            ServoEnum.MSG_COMMAND_KEY, ServoEnum.MSG_COMMAND_MOVE,
            ServoEnum.MSG_COMMAND_MOVE_ALL, ServoEnum.MSG_COMMAND_STATUS,
            ServoEnum.MSG_LOCATION_KEY, ServoEnum.MSG_ANGLE,
            ServoEnum.MSG_SPEED, ServoEnum.MSG_RESULTS,
            ServoEnum.MSG_TARGETS, ServoEnum.MSG_VELOCITY,
            ServoEnum.MSG_CURRENT_ANGLE, ServoEnum.MSG_MOVING,
            ServoEnum.MSG_AXIS, ServoEnum.MSG_MAX, ServoEnum.MSG_MIN,
            ServoEnum.MSG_MIDDLE, ServoEnum.MSG_LAST_ANGLE,
        ]
        for key in required_keys:
            assert isinstance(key.value, str)

    def test_mqtt_topics_are_strings(self):
        assert isinstance(ServoEnum.MQTT_COMMAND_TOPIC.value, str)
        assert isinstance(ServoEnum.MQTT_STATUS_TOPIC.value, str)
        assert "/" in ServoEnum.MQTT_COMMAND_TOPIC.value
        assert "/" in ServoEnum.MQTT_STATUS_TOPIC.value


class TestMotionProfile:
    """Validate spring-damper parameter tables."""

    def test_head_params_all_5_levels(self):
        params = MotionProfile.HEAD_PARAMS.value
        for level in range(1, 6):
            assert level in params
            omega, zeta = params[level]
            assert omega > 0
            assert 0 < zeta <= 1.5

    def test_body_params_all_5_levels(self):
        params = MotionProfile.BODY_PARAMS.value
        for level in range(1, 6):
            assert level in params
            omega, zeta = params[level]
            assert omega > 0
            assert 0 < zeta <= 1.5

    def test_head_faster_than_body_at_every_level(self):
        head = MotionProfile.HEAD_PARAMS.value
        body = MotionProfile.BODY_PARAMS.value
        for level in range(1, 6):
            assert head[level][0] > body[level][0], (
                f"Head omega ({head[level][0]}) must be > body omega ({body[level][0]}) at level {level}"
            )

    def test_ud_interpolation_table_sorted(self):
        table = MotionProfile.HEAD_UD_TO_BODY_UD_TABLE.value
        assert len(table) > 2
        for i in range(1, len(table)):
            assert table[i][0] > table[i-1][0], "Table head angles must be ascending"
            assert table[i][1] > table[i-1][1], "Table body angles must be ascending"

    def test_physics_dt_positive(self):
        assert MotionProfile.PHYSICS_DT.value > 0
        assert MotionProfile.PHYSICS_DT.value < 0.1  # sanity: faster than 10Hz

    def test_camera_mounting_offsets_opposite(self):
        left = MotionProfile.CAMERA_LEFT_MOUNTING_OFFSET.value
        right = MotionProfile.CAMERA_RIGHT_MOUNTING_OFFSET.value
        assert left < 0
        assert right > 0
        assert abs(left) == abs(right)  # symmetric


class TestVisionEnums:
    """Validate vision result key completeness."""

    def test_box_coordinate_keys(self):
        for key in ['x1', 'y1', 'x2', 'y2']:
            assert hasattr(VisionResultsEnum, f'BOX_{key.upper()}')

    def test_keypoint_keys(self):
        assert VisionResultsEnum.KEYPOINT_X.value == "x"
        assert VisionResultsEnum.KEYPOINT_Y.value == "y"
        assert VisionResultsEnum.KEYPOINT_CONFIDENCE.value == "confidence"
        assert VisionResultsEnum.KEYPOINT_LOCATION.value == "location"

    def test_coco_wholebody_has_133_keypoints(self):
        kp = VisionResultsEnum.VISION_POSE_KEY_POINTS_COCO_WHOLE_BODY.value
        assert len(kp) == 133
        assert kp[0] == "Nose"
        assert kp[132] == "Right_Foot_Heel"

    def test_draw_threshold_in_range(self):
        thresh = VisionResultsEnum.KEYPOINT_DRAW_THRESHOLD.value
        assert 0.0 < thresh < 1.0


class TestMQTTTopics:
    """Validate all MQTT topics are defined and formatted correctly."""

    def test_all_topics_are_strings(self):
        for member in MQTTEnums:
            assert isinstance(member.value, str)

    def test_no_leading_slash(self):
        for member in MQTTEnums:
            assert not member.value.startswith("/"), f"{member.name} has leading slash"

    def test_required_topics_exist(self):
        required = [
            'STT_RESULTS_MQTT_TOPIC', 'VISION_RESULTS_MQTT_TOPIC',
            'BODY_LED_CONTROL_MQTT_TOPIC', 'LCD_CONTROL_MQTT_TOPIC',
            'IMU_STATUS_TOPIC', 'TOF_STATUS_TOPIC', 'TH_STATUS_TOPIC',
            'MOX_STATUS_TOPIC', 'SYSTEM_HEALTH_TOPIC', 'SYSTEM_LOG_LEVEL_TOPIC',
        ]
        for name in required:
            assert hasattr(MQTTEnums, name), f"Missing MQTT topic: {name}"


class TestFusionEnums:
    """Validate camera fusion state machine enums."""

    def test_all_states_defined(self):
        required = ['STATE_HEAD_TRACKING', 'STATE_SIDE_ONLY',
                     'STATE_HANDOFF_TO_HEAD', 'STATE_HANDOFF_TO_SIDE']
        for name in required:
            assert hasattr(FusionEnums, name)

    def test_blend_duration_positive(self):
        assert FusionEnums.HANDOFF_BLEND_DURATION.value > 0

    def test_staleness_positive(self):
        assert FusionEnums.SIDE_CAMERA_STALENESS.value > 0

    def test_agreement_threshold_positive(self):
        assert FusionEnums.HANDOFF_AGREEMENT_THRESHOLD.value > 0


class TestTraceEnums:
    """Validate trace pipeline keys."""

    def test_all_trace_keys_exist(self):
        required = ['TRACE_ID', 'TS_VISION', 'TS_TRACK_START', 'TS_TRACK_END', 'TS_SERVO_RX']
        for name in required:
            assert hasattr(TraceEnums, name)
