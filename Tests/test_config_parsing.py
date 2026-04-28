"""Tests for config file parsing -- verifies all systems can read their required sections."""

import pytest
import os
from configparser import ConfigParser

from glados_modules.GladosEnums import (
    SystemEnums, CameraEnum, ServoEnum, STTEnums, DashboardEnums, MQTTEnums,
    BrainEnums, BrainDefaults, SceneEnums, MCPEnums, HEXACOEnums, EmotionEnums,
)


@pytest.fixture
def config():
    """Load the actual glog.conf from the repo."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'glog.conf')
    if not os.path.exists(config_path):
        pytest.skip("glog.conf not found")
    cp = ConfigParser()
    cp.read(config_path)
    return cp


class TestConfigSections:
    """Verify all required sections exist."""

    def test_default_section(self, config):
        assert config.has_section(SystemEnums.CONFIG_HEAD_DEFAULT.value) or \
               SystemEnums.CONFIG_HEAD_DEFAULT.value == "DEFAULT"
        # DEFAULT is always implicit in ConfigParser
        assert config.get("DEFAULT", SystemEnums.VOICE_URL.value, fallback=None) is not None

    def test_mqtt_section(self, config):
        section = SystemEnums.CONFIG_HEAD_MQTT.value
        assert config.has_section(section)
        assert config.get(section, SystemEnums.MQTT_SERVER_IP.value)
        assert config.get(section, SystemEnums.MQTT_PORT.value)

    def test_cameras_section(self, config):
        section = CameraEnum.CONFIG_HEAD.value
        assert config.has_section(section)
        assert config.get(section, CameraEnum.CAMERA_HEAD_FACTORY.value)

    def test_servos_section(self, config):
        section = ServoEnum.CONFIG_HEAD.value
        assert config.has_section(section)
        assert config.get(section, ServoEnum.DEFAULT_MAX_MIN_CENTER.value)

    def test_rtsp_section(self, config):
        section = SystemEnums.CONFIG_HEAD_RTSP.value
        assert config.has_section(section)
        assert config.get(section, SystemEnums.RTSP_PORT.value)

    def test_stt_section(self, config):
        section = STTEnums.CONFIG_HEAD_STT.value
        assert config.has_section(section)
        assert config.get(section, STTEnums.STT_SERVER_IP.value)

    def test_yolo_section(self, config):
        assert config.has_section("YOLO")
        assert config.get("YOLO", "model")

    def test_dashboard_section_optional(self, config):
        # Dashboard is optional with fallback
        port = config.get(DashboardEnums.CONFIG_HEAD.value,
                          DashboardEnums.DASHBOARD_PORT.value,
                          fallback=DashboardEnums.DEFAULT_PORT.value)
        assert int(port) > 0


class TestConfigServoParsing:
    """Verify servo config values parse correctly."""

    def test_pulse_widths_parse_as_two_ints(self, config):
        section = ServoEnum.CONFIG_HEAD.value
        for key in [ServoEnum.SERVO_MG90D_PULSE.value, ServoEnum.SERVO_MG92B_PULSE.value,
                    ServoEnum.SERVO_GS3508MG_PULSE.value]:
            raw = config.get(section, key)
            parts = raw.split(',')
            assert len(parts) == 2, f"{key} should have 2 values, got {parts}"
            int(parts[0].strip())  # should not raise
            int(parts[1].strip())

    def test_min_max_center_parse_as_three_ints(self, config):
        section = ServoEnum.CONFIG_HEAD.value
        for key in [ServoEnum.DEFAULT_MAX_MIN_CENTER.value,
                    ServoEnum.HEAD_MIN_MAX_CENTER.value,
                    ServoEnum.NECK_MIN_MAX_CENTER.value]:
            raw = config.get(section, key)
            parts = raw.split(',')
            assert len(parts) == 3, f"{key} should have 3 values, got {parts}"
            for p in parts:
                int(p.strip())

    def test_speeds_parse_as_float(self, config):
        section = ServoEnum.CONFIG_HEAD.value
        for key in [ServoEnum.SERVO_MG90D_SPEED.value, ServoEnum.SERVO_MG92B_SPEED.value,
                    ServoEnum.SERVO_GS3508MG_SPEED.value]:
            float(config.get(section, key))  # should not raise


class TestConfigCameraParsing:
    """Verify camera config values parse correctly."""

    def test_resolution_parse(self, config):
        section = CameraEnum.CONFIG_HEAD.value
        for key in [CameraEnum.CAMERA_HEAD_RESOLUTION.value,
                    CameraEnum.CAMERA_LEFT_RESOLUTION.value,
                    CameraEnum.CAMERA_RIGHT_RESOLUTION.value]:
            raw = config.get(section, key)
            parts = raw.split(',')
            assert len(parts) == 2
            x, y = int(parts[0]), int(parts[1])
            assert x > 0 and y > 0

    def test_fps_parse(self, config):
        section = CameraEnum.CONFIG_HEAD.value
        for key in [CameraEnum.CAMERA_HEAD_FPS.value,
                    CameraEnum.CAMERA_LEFT_FPS.value,
                    CameraEnum.CAMERA_RIGHT_FPS.value]:
            fps = int(config.get(section, key))
            assert fps > 0

    def test_ports_are_valid(self, config):
        section = CameraEnum.CONFIG_HEAD.value
        for key in [CameraEnum.CAMERA_HEAD_PORT.value,
                    CameraEnum.CAMERA_LEFT_PORT.value,
                    CameraEnum.CAMERA_RIGHT_PORT.value]:
            port = int(config.get(section, key))
            assert 1 <= port <= 65535


class TestBrainConfigSection:
    """Verify the [BRAIN] section parses cleanly and types as expected."""

    def test_brain_section_exists(self, config):
        assert config.has_section(BrainEnums.CONFIG_HEAD.value)

    def test_required_keys_present(self, config):
        section = BrainEnums.CONFIG_HEAD.value
        for key in [BrainEnums.LLM_ENDPOINT.value,
                    BrainEnums.LLM_MODEL.value,
                    BrainEnums.INPUT_MODE.value,
                    BrainEnums.AUDIO_IO.value,
                    BrainEnums.PERSONALITY_FILE.value]:
            assert config.get(section, key), f"{key} missing or empty"

    def test_llm_endpoint_is_url(self, config):
        endpoint = config.get(BrainEnums.CONFIG_HEAD.value,
                               BrainEnums.LLM_ENDPOINT.value)
        assert endpoint.startswith("http://") or endpoint.startswith("https://")

    def test_input_mode_valid(self, config):
        mode = config.get(BrainEnums.CONFIG_HEAD.value,
                           BrainEnums.INPUT_MODE.value)
        assert mode in {"audio", "text", "both"}

    def test_boolean_flags_parse(self, config):
        section = BrainEnums.CONFIG_HEAD.value
        for key in [BrainEnums.INTERRUPTIBLE.value,
                    BrainEnums.AUTONOMY_ENABLED.value,
                    BrainEnums.EMOTION_ENABLED.value,
                    BrainEnums.COMPACTION_ENABLED.value,
                    BrainEnums.OBSERVER_ENABLED.value]:
            raw = config.get(section, key)
            assert raw.strip().lower() in {"true", "false"}

    def test_numeric_values_parse(self, config):
        section = BrainEnums.CONFIG_HEAD.value
        float(config.get(section, BrainEnums.AUTONOMY_TICK_INTERVAL.value))
        float(config.get(section, BrainEnums.SCENE_TIMEOUT.value))
        int(config.get(section, BrainEnums.AUTONOMY_PARALLEL_CALLS.value))
        int(config.get(section, BrainEnums.SCENE_PRIORITY.value))
        int(config.get(section, BrainEnums.ROOM_PRIORITY.value))

    def test_personality_file_exists(self, config):
        rel_path = config.get(BrainEnums.CONFIG_HEAD.value,
                               BrainEnums.PERSONALITY_FILE.value)
        # Path is relative to the repo root; tests run from any cwd
        repo_root = os.path.join(os.path.dirname(__file__), '..')
        full = os.path.normpath(os.path.join(repo_root, rel_path))
        assert os.path.isfile(full), f"personality file missing: {full}"

    def test_fallback_when_key_missing(self):
        cp = ConfigParser()
        cp.read_string("[BRAIN]\nllm_model = test\n")
        val = cp.get(BrainEnums.CONFIG_HEAD.value,
                      BrainEnums.SCENE_TIMEOUT.value,
                      fallback=str(BrainDefaults.SCENE_TIMEOUT))
        assert float(val) == BrainDefaults.SCENE_TIMEOUT


class TestSceneConfigSection:
    """Verify the [SCENE] section parses cleanly."""

    def test_scene_section_exists(self, config):
        assert config.has_section(SceneEnums.CONFIG_HEAD.value)

    def test_rtsp_uri_present(self, config):
        uri = config.get(SceneEnums.CONFIG_HEAD.value,
                          SceneEnums.CAMERA_URI_KEY.value)
        assert uri.startswith("rtsp://")

    def test_poll_interval_positive(self, config):
        interval = float(config.get(SceneEnums.CONFIG_HEAD.value,
                                      SceneEnums.POLL_INTERVAL_KEY.value))
        assert interval > 0

    def test_scene_change_threshold_in_range(self, config):
        threshold = float(config.get(SceneEnums.CONFIG_HEAD.value,
                                       SceneEnums.SCENE_CHANGE_THRESHOLD_KEY.value))
        assert 0.0 <= threshold <= 1.0

    def test_model_dir_optional(self, config):
        # Empty model_dir means use the package default; must not error
        model_dir = config.get(SceneEnums.CONFIG_HEAD.value,
                                SceneEnums.MODEL_DIR_KEY.value, fallback="")
        # Either empty (use default) or a path-like string
        assert model_dir is not None


class TestMCPBodyConfigSection:
    """Verify the [MCP_BODY] section parses cleanly."""

    def test_mcp_body_section_exists(self, config):
        assert config.has_section(MCPEnums.CONFIG_HEAD.value)

    def test_default_angles_parse_as_int(self, config):
        section = MCPEnums.CONFIG_HEAD.value
        for key in [MCPEnums.DEFAULT_HEAD_YAW.value,
                    MCPEnums.DEFAULT_HEAD_PITCH.value,
                    MCPEnums.DEFAULT_BODY_YAW.value]:
            angle = int(config.get(section, key))
            assert 0 <= angle <= 180

    def test_eye_color_parses_as_three_ints(self, config):
        raw = config.get(MCPEnums.CONFIG_HEAD.value,
                          MCPEnums.EYE_COLOR_DEFAULT.value)
        parts = raw.split(',')
        assert len(parts) == 3
        for p in parts:
            channel = int(p.strip())
            assert 0 <= channel <= 255


class TestHEXACOConfigSection:
    """Step 7c-1: [HEXACO] section parses + values fall in 0..1."""

    def test_hexaco_section_exists(self, config):
        assert config.has_section(HEXACOEnums.CONFIG_HEAD.value)

    def test_all_traits_parse_as_floats_in_range(self, config):
        section = HEXACOEnums.CONFIG_HEAD.value
        for key in [HEXACOEnums.HONESTY_HUMILITY.value,
                    HEXACOEnums.EMOTIONALITY.value,
                    HEXACOEnums.EXTRAVERSION.value,
                    HEXACOEnums.AGREEABLENESS.value,
                    HEXACOEnums.CONSCIENTIOUSNESS.value,
                    HEXACOEnums.OPENNESS.value]:
            v = float(config.get(section, key))
            assert 0.0 <= v <= 1.0, f"{key}={v} out of [0,1]"


class TestEmotionConfigSection:
    """Step 7c-1: [EMOTION] section parses + PAD baseline in [-1, +1]."""

    def test_emotion_section_exists(self, config):
        assert config.has_section(EmotionEnums.CONFIG_HEAD.value)

    def test_pad_baseline_in_signed_unit_range(self, config):
        section = EmotionEnums.CONFIG_HEAD.value
        for key in [EmotionEnums.BASELINE_PLEASURE.value,
                    EmotionEnums.BASELINE_AROUSAL.value,
                    EmotionEnums.BASELINE_DOMINANCE.value]:
            v = float(config.get(section, key))
            assert -1.0 <= v <= 1.0, f"{key}={v} out of [-1,1]"

    def test_tick_interval_positive(self, config):
        v = float(config.get(EmotionEnums.CONFIG_HEAD.value,
                              EmotionEnums.TICK_INTERVAL.value))
        assert v > 0

    def test_max_events_positive(self, config):
        v = int(config.get(EmotionEnums.CONFIG_HEAD.value,
                            EmotionEnums.MAX_EVENTS.value))
        assert v > 0

    def test_drift_rates_in_unit_range(self, config):
        for key in [EmotionEnums.MOOD_DRIFT_RATE.value,
                    EmotionEnums.BASELINE_DRIFT_RATE.value]:
            v = float(config.get(EmotionEnums.CONFIG_HEAD.value, key))
            assert 0.0 < v <= 1.0


class TestBrainMoodThrottleKeys:
    """Step 7c-1: [BRAIN] gains PAD throttling keys (used by 7c-2)."""

    def test_throttle_keys_present_and_typed(self, config):
        section = BrainEnums.CONFIG_HEAD.value
        delta = float(config.get(section, BrainEnums.MOOD_PUBLISH_DELTA.value))
        max_interval = float(config.get(section,
                                         BrainEnums.MOOD_PUBLISH_MAX_INTERVAL.value))
        staleness = float(config.get(section,
                                      BrainEnums.MOOD_STALENESS_MAX_AGE.value))
        assert 0.0 < delta < 1.0
        assert max_interval > 0
        assert staleness >= max_interval, (
            "staleness window must outlive heartbeat interval")

    def test_autonomy_cooldown_present(self, config):
        v = float(config.get(BrainEnums.CONFIG_HEAD.value,
                              BrainEnums.AUTONOMY_COOLDOWN.value))
        assert v > 0

    def test_autonomy_coalesce_ticks_parses_as_bool(self, config):
        raw = config.get(BrainEnums.CONFIG_HEAD.value,
                          BrainEnums.AUTONOMY_COALESCE_TICKS.value)
        assert raw.strip().lower() in {"true", "false"}
