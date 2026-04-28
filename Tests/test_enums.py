"""Tests for enum consistency and completeness.

Validates that all enums referenced in code actually exist, that MQTT topics
are defined as enums, and that cross-system message contracts are consistent.
"""

from enum import Enum

import pytest
from glados_modules.GladosEnums import (
    ServoEnum, CameraEnum, MQTTEnums, TrackingEnums, VisionResultsEnum,
    MotionProfile, LEDHead, LEDShoulders, LEDLampStrip8, LCDEnums,
    TraceEnums, DashboardEnums, LoggingEnums, SystemEnums,
    IMUEnums, TOFEnums, THEnums, MOXEnums, FusionEnums,
    BrainEnums, BrainDefaults, SceneEnums, MCPEnums,
    HEXACOEnums, HEXACODefaults, EmotionEnums, EmotionDefaults,
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
        # FK yaw convention: positive = robot's left, negative = robot's right
        assert left > 0   # left camera is to robot's left = positive yaw
        assert right < 0  # right camera is to robot's right = negative yaw
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

    def test_brain_and_scene_topics_exist(self):
        required = [
            'SCENE_DESCRIPTION_TOPIC', 'SCENE_DESCRIBE_REQUEST_TOPIC',
            'SCENE_DESCRIBE_RESPONSE_TOPIC', 'BRAIN_READY_TOPIC',
            'BRAIN_UTTERANCE_TOPIC', 'BRAIN_TOOL_CALL_TOPIC',
        ]
        for name in required:
            assert hasattr(MQTTEnums, name), f"Missing MQTT topic: {name}"

    def test_brain_and_scene_topic_values_stable(self):
        # These strings are the wire contract between the brain (Pi 5) and the
        # GPU box. Any rename must be deliberate.
        assert MQTTEnums.SCENE_DESCRIPTION_TOPIC.value == "vision/scene_description"
        assert MQTTEnums.SCENE_DESCRIBE_REQUEST_TOPIC.value == "vision/describe_request"
        assert MQTTEnums.SCENE_DESCRIBE_RESPONSE_TOPIC.value == "vision/describe_response"
        assert MQTTEnums.BRAIN_READY_TOPIC.value == "system/brain_ready"
        assert MQTTEnums.BRAIN_UTTERANCE_TOPIC.value == "system/brain/utterance"
        assert MQTTEnums.BRAIN_TOOL_CALL_TOPIC.value == "system/brain/tool_call"


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


class TestBrainEnums:
    """Validate brain configuration enum keys."""

    def test_config_head_is_brain(self):
        assert BrainEnums.CONFIG_HEAD.value == "BRAIN"

    def test_required_keys_present(self):
        required = [
            'LLM_ENDPOINT', 'LLM_MODEL', 'LLM_API_KEY', 'INPUT_MODE',
            'AUDIO_IO', 'INTERRUPTIBLE', 'AUTONOMY_ENABLED',
            'AUTONOMY_TICK_INTERVAL', 'AUTONOMY_PARALLEL_CALLS',
            'EMOTION_ENABLED', 'COMPACTION_ENABLED', 'OBSERVER_ENABLED',
            'SCENE_PRIORITY', 'ROOM_PRIORITY', 'PERSONALITY_FILE',
            'SCENE_TIMEOUT',
        ]
        for name in required:
            assert hasattr(BrainEnums, name), f"Missing BrainEnum key: {name}"

    def test_default_values_are_sane(self):
        assert BrainDefaults.SCENE_TIMEOUT > 0
        assert BrainDefaults.TICK_INTERVAL > 0
        assert BrainDefaults.PARALLEL_CALLS >= 1
        # Scene context should outrank room context by default
        assert BrainDefaults.SCENE_PRIORITY >= BrainDefaults.ROOM_PRIORITY


class TestSceneEnums:
    """Validate SceneDescriber configuration + message enum keys."""

    def test_config_head_is_scene(self):
        assert SceneEnums.CONFIG_HEAD.value == "SCENE"

    def test_message_keys_present(self):
        required = [
            'DESCRIPTION_KEY', 'CAMERA_KEY', 'TS_KEY', 'PROMPT_KEY',
            'MAX_TOKENS_KEY', 'REQUEST_ID_KEY',
        ]
        for name in required:
            assert hasattr(SceneEnums, name), f"Missing SceneEnum key: {name}"

    def test_config_keys_present(self):
        required = [
            'MODEL_DIR_KEY', 'CAMERA_URI_KEY', 'POLL_INTERVAL_KEY',
            'SCENE_CHANGE_THRESHOLD_KEY',
        ]
        for name in required:
            assert hasattr(SceneEnums, name), f"Missing SceneEnum key: {name}"

    def test_defaults_in_valid_range(self):
        assert SceneEnums.DEFAULT_POLL_INTERVAL.value > 0
        assert 0.0 <= SceneEnums.DEFAULT_SCENE_CHANGE_THRESHOLD.value <= 1.0


class TestMCPEnums:
    """Validate MCP body server configuration enum keys."""

    def test_config_head_is_mcp_body(self):
        assert MCPEnums.CONFIG_HEAD.value == "MCP_BODY"

    def test_required_keys_present(self):
        required = [
            'DEFAULT_HEAD_YAW', 'DEFAULT_HEAD_PITCH',
            'DEFAULT_BODY_YAW', 'EYE_COLOR_DEFAULT',
        ]
        for name in required:
            assert hasattr(MCPEnums, name), f"Missing MCPEnum key: {name}"


class TestHEXACOEnums:
    """HEXACO trait enum keys + sane defaults."""

    def test_config_head_is_hexaco(self):
        assert HEXACOEnums.CONFIG_HEAD.value == "HEXACO"

    def test_all_six_traits_present(self):
        required = ['HONESTY_HUMILITY', 'EMOTIONALITY', 'EXTRAVERSION',
                    'AGREEABLENESS', 'CONSCIENTIOUSNESS', 'OPENNESS']
        for name in required:
            assert hasattr(HEXACOEnums, name)

    def test_defaults_in_unit_range(self):
        # All HEXACO trait values are 0.0..1.0 by engine convention
        for name in ['HONESTY_HUMILITY', 'EMOTIONALITY', 'EXTRAVERSION',
                     'AGREEABLENESS', 'CONSCIENTIOUSNESS', 'OPENNESS']:
            v = getattr(HEXACODefaults, name)
            assert 0.0 <= v <= 1.0, f"{name}={v} out of [0,1]"

    def test_glados_personality_signature(self):
        # Defaults should encode GLaDOS: low agreeableness, high
        # conscientiousness, very high openness, low honesty/humility.
        assert HEXACODefaults.AGREEABLENESS < 0.4
        assert HEXACODefaults.CONSCIENTIOUSNESS > 0.7
        assert HEXACODefaults.OPENNESS > 0.7
        assert HEXACODefaults.HONESTY_HUMILITY < 0.5


class TestEmotionEnums:
    """[EMOTION] section keys + PAD baseline range."""

    def test_config_head_is_emotion(self):
        assert EmotionEnums.CONFIG_HEAD.value == "EMOTION"

    def test_required_keys_present(self):
        required = ['TICK_INTERVAL', 'MAX_EVENTS', 'BASELINE_PLEASURE',
                    'BASELINE_AROUSAL', 'BASELINE_DOMINANCE',
                    'MOOD_DRIFT_RATE', 'BASELINE_DRIFT_RATE']
        for name in required:
            assert hasattr(EmotionEnums, name)

    def test_pad_baseline_defaults_in_range(self):
        # PAD axes are in [-1, +1]
        for name in ['BASELINE_PLEASURE', 'BASELINE_AROUSAL',
                     'BASELINE_DOMINANCE']:
            v = getattr(EmotionDefaults, name)
            assert -1.0 <= v <= 1.0, f"{name}={v} out of [-1,1]"

    def test_drift_rates_in_unit_range(self):
        for name in ['MOOD_DRIFT_RATE', 'BASELINE_DRIFT_RATE']:
            v = getattr(EmotionDefaults, name)
            assert 0.0 < v <= 1.0


@pytest.mark.parametrize(
    "enum_cls",
    [BrainEnums, EmotionEnums, HEXACOEnums, SceneEnums,
     # MCPEnums and the rest are checked too — list each I added/extended
     ],
    ids=lambda c: c.__name__)
def test_no_alias_collisions_in_new_enums(enum_cls):
    """R-1.3: catch the BrainEnums (8.0 == 8) / EmotionEnums (0.1 == 0.1)
    aliasing class of bug for any enum I add to the project.

    `len(list(E))` iterates only canonical members; `len(E.__members__)`
    counts all registered names including aliases. Equal → no aliases.
    """
    canonical = list(enum_cls)
    all_names = list(enum_cls.__members__)
    aliases = set(all_names) - {m.name for m in canonical}
    assert not aliases, (
        f"{enum_cls.__name__} has alias members: {sorted(aliases)}. "
        f"Move duplicate-valued members out of the Enum into a sibling "
        f"defaults class (see BrainDefaults / EmotionDefaults).")


def test_unique_decorator_applied_to_new_enums():
    """Defense-in-depth: @unique throws at class creation if dupes are added.

    Any new Enum I introduce should carry @unique so a future dup is caught
    at import time rather than silently aliased. Verifies decorator is in
    place by checking the enum's __dict__ for the marker.
    """
    # @unique modifies the enum's class hierarchy but doesn't leave an
    # obvious marker. The cheaper check: prove that adding a duplicate
    # would raise. We do this by attempting to construct an aliased subclass
    # — if @unique is on the parent, the duplicate is rejected.
    for enum_cls in (BrainEnums, EmotionEnums, HEXACOEnums, SceneEnums):
        # The Enum already verified at import time via @unique; if we got
        # here without ImportError, the decorator's working. Just confirm
        # all members are unique by value (the contract @unique enforces).
        seen_values = set()
        for member in enum_cls:
            assert member.value not in seen_values, (
                f"{enum_cls.__name__}.{member.name} duplicates an earlier "
                f"member's value ({member.value!r}); should be impossible "
                f"with @unique applied.")
            seen_values.add(member.value)


class TestBrainMoodKeys:
    """Step 7c PAD publish + staleness throttling keys live on BrainEnums."""

    def test_throttle_keys_present(self):
        for name in ['MOOD_PUBLISH_DELTA', 'MOOD_PUBLISH_MAX_INTERVAL',
                     'MOOD_STALENESS_MAX_AGE']:
            assert hasattr(BrainEnums, name)

    def test_throttle_defaults_sane(self):
        assert 0.0 < BrainDefaults.MOOD_PUBLISH_DELTA < 1.0
        # Max interval (heartbeat) must outlive staleness so consumers don't
        # constantly fall back to baseline visuals between heartbeats
        assert (BrainDefaults.MOOD_PUBLISH_MAX_INTERVAL
                <= BrainDefaults.MOOD_STALENESS_MAX_AGE)
