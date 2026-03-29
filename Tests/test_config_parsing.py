"""Tests for config file parsing -- verifies all systems can read their required sections."""

import pytest
import os
from configparser import ConfigParser

from glados_modules.GladosEnums import (
    SystemEnums, CameraEnum, ServoEnum, STTEnums, DashboardEnums, MQTTEnums
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
