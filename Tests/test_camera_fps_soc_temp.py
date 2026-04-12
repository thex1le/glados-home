"""Tests for camera FPS telemetry and SoC temperature monitoring.

Validates MQTT message formats, SocTemp sensor behavior, SensorTracker
handlers for camera FPS and SoC temperature, and dashboard API output.
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from collections import namedtuple
from json import dumps

from glados_modules.GladosEnums import CameraEnum, SocTempEnums
from glados_modules.MqttConnector import SocTempMessageBuilder


class TestCameraFpsMessage:
    """Validate camera FPS telemetry message format."""

    def test_fps_message_has_required_keys(self):
        """FPS message must contain location, actual/target FPS, and timing."""
        msg = {
            CameraEnum.MSG_LOCATION_KEY.value: "camera_head",
            CameraEnum.MSG_ACTUAL_FPS.value: 14.8,
            CameraEnum.MSG_TARGET_FPS.value: 15,
            CameraEnum.MSG_CAPTURE_MS.value: 2.1,
            CameraEnum.MSG_SEND_MS.value: 3.4,
        }
        assert msg[CameraEnum.MSG_LOCATION_KEY.value] == "camera_head"
        assert msg[CameraEnum.MSG_ACTUAL_FPS.value] == 14.8
        assert msg[CameraEnum.MSG_TARGET_FPS.value] == 15
        assert msg[CameraEnum.MSG_CAPTURE_MS.value] == 2.1
        assert msg[CameraEnum.MSG_SEND_MS.value] == 3.4

    def test_fps_topic_follows_camera_namespace(self):
        """FPS topic should be under vision/camera/ namespace."""
        topic = CameraEnum.MQTT_FPS_TOPIC.value
        assert topic.startswith("vision/camera/")
        assert "fps" in topic

    def test_fps_enum_values_are_strings(self):
        """All FPS-related enum values must be strings for JSON keys."""
        assert isinstance(CameraEnum.MSG_ACTUAL_FPS.value, str)
        assert isinstance(CameraEnum.MSG_TARGET_FPS.value, str)
        assert isinstance(CameraEnum.MSG_CAPTURE_MS.value, str)
        assert isinstance(CameraEnum.MSG_SEND_MS.value, str)


class TestSocTempEnums:
    """Validate SoC temperature enum completeness."""

    def test_soc_temp_topic_under_system_namespace(self):
        assert SocTempEnums.SOC_TEMP_TOPIC.value.startswith("system/")

    def test_soc_temp_message_keys_are_strings(self):
        assert isinstance(SocTempEnums.SOC_TEMP_KEY.value, str)
        assert isinstance(SocTempEnums.CELSIUS_KEY.value, str)
        assert isinstance(SocTempEnums.HOSTNAME_KEY.value, str)
        assert isinstance(SocTempEnums.TIMESTAMP_KEY.value, str)

    def test_default_poll_interval_is_reasonable(self):
        interval = SocTempEnums.DEFAULT_POLL_INTERVAL.value
        assert 0.5 <= interval <= 30.0


class TestSocTempMessageBuilder:
    """Validate SoC temp message builder wraps correctly."""

    def test_message_wraps_with_soc_temp_key(self):
        inner = {
            SocTempEnums.HOSTNAME_KEY.value: "pi4",
            SocTempEnums.CELSIUS_KEY.value: 52.3,
            SocTempEnums.TIMESTAMP_KEY.value: 1234567890.0,
        }
        msg = SocTempMessageBuilder.send_soc_temp_message(inner)
        assert SocTempEnums.SOC_TEMP_KEY.value in msg
        assert msg[SocTempEnums.SOC_TEMP_KEY.value] == inner

    def test_message_preserves_hostname(self):
        inner = {SocTempEnums.HOSTNAME_KEY.value: "pi5", SocTempEnums.CELSIUS_KEY.value: 61.0}
        msg = SocTempMessageBuilder.send_soc_temp_message(inner)
        assert msg[SocTempEnums.SOC_TEMP_KEY.value][SocTempEnums.HOSTNAME_KEY.value] == "pi5"


class TestSocTempSensor:
    """Validate SocTemp sensor class behavior."""

    def _make_soc_temp(self, thermal_exists=True):
        """Create a SocTemp instance with mocked MQTT."""
        from glados_modules.BodyControlModules import SocTemp
        broker = namedtuple("broker", ["ip", "port"])("localhost", 1883)
        with patch.object(SocTemp, '__init__', lambda self, **kw: None):
            sensor = SocTemp.__new__(SocTemp)
            sensor.__name__ = "SocTemp"
            sensor.hostname = "test-host"
            sensor.poll_interval = 2.0
            sensor.logger = MagicMock()
            sensor._available = thermal_exists
        return sensor

    def test_get_celsius_reads_thermal_zone(self):
        sensor = self._make_soc_temp(thermal_exists=True)
        mock_data = "52300\n"
        with patch("builtins.open", mock_open(read_data=mock_data)):
            temp = sensor.get_celsius()
        assert temp == 52.3

    def test_get_celsius_returns_negative_when_unavailable(self):
        sensor = self._make_soc_temp(thermal_exists=False)
        temp = sensor.get_celsius()
        assert temp == -1.0

    def test_get_celsius_handles_read_error(self):
        sensor = self._make_soc_temp(thermal_exists=True)
        with patch("builtins.open", side_effect=OSError("Permission denied")):
            temp = sensor.get_celsius()
        assert temp == -1.0

    def test_get_celsius_rounds_to_one_decimal(self):
        sensor = self._make_soc_temp(thermal_exists=True)
        mock_data = "67891\n"
        with patch("builtins.open", mock_open(read_data=mock_data)):
            temp = sensor.get_celsius()
        assert temp == 67.9


class TestSensorTrackerCameraFps:
    """Validate SensorTracker handles camera FPS messages."""

    def _make_tracker(self):
        from glados_modules.MqttConsumerModules import SensorTracker
        broker = namedtuple("broker", ["ip", "port"])("localhost", 1883)
        with patch.object(SensorTracker, '__init__', lambda self, b: None):
            st = SensorTracker.__new__(SensorTracker)
            st.__name__ = "SensorTracker"
            st.logger = MagicMock()
            st.camera_fps = {}
            st.soc_temp = {}
            st._last_update = {}
        return st

    def test_camera_fps_stored_by_location(self):
        st = self._make_tracker()
        msg = MagicMock()
        msg.payload.decode.return_value = dumps({
            CameraEnum.MSG_LOCATION_KEY.value: "camera_head",
            CameraEnum.MSG_ACTUAL_FPS.value: 14.2,
            CameraEnum.MSG_TARGET_FPS.value: 15,
        })
        st.camera_fps_handle_cmd(msg)
        assert "camera_head" in st.camera_fps
        assert st.camera_fps["camera_head"][CameraEnum.MSG_ACTUAL_FPS.value] == 14.2

    def test_camera_fps_updates_timestamp(self):
        st = self._make_tracker()
        msg = MagicMock()
        msg.payload.decode.return_value = dumps({
            CameraEnum.MSG_LOCATION_KEY.value: "camera_left",
            CameraEnum.MSG_ACTUAL_FPS.value: 9.8,
            CameraEnum.MSG_TARGET_FPS.value: 10,
        })
        st.camera_fps_handle_cmd(msg)
        assert "camera_fps_camera_left" in st._last_update
        assert st._last_update["camera_fps_camera_left"] > 0

    def test_soc_temp_stored_by_hostname(self):
        st = self._make_tracker()
        msg = MagicMock()
        msg.payload.decode.return_value = dumps({
            SocTempEnums.SOC_TEMP_KEY.value: {
                SocTempEnums.HOSTNAME_KEY.value: "pi4",
                SocTempEnums.CELSIUS_KEY.value: 55.2,
                SocTempEnums.TIMESTAMP_KEY.value: 1234567890.0,
            }
        })
        st.soc_temp_handle_cmd(msg)
        assert "pi4" in st.soc_temp
        assert st.soc_temp["pi4"][SocTempEnums.CELSIUS_KEY.value] == 55.2

    def test_soc_temp_updates_timestamp(self):
        st = self._make_tracker()
        msg = MagicMock()
        msg.payload.decode.return_value = dumps({
            SocTempEnums.SOC_TEMP_KEY.value: {
                SocTempEnums.HOSTNAME_KEY.value: "pi5",
                SocTempEnums.CELSIUS_KEY.value: 61.0,
            }
        })
        st.soc_temp_handle_cmd(msg)
        assert "soc_temp_pi5" in st._last_update

    def test_multiple_cameras_tracked_independently(self):
        st = self._make_tracker()
        for loc, fps in [("camera_head", 14.5), ("camera_left", 9.2), ("camera_right", 14.8)]:
            msg = MagicMock()
            msg.payload.decode.return_value = dumps({
                CameraEnum.MSG_LOCATION_KEY.value: loc,
                CameraEnum.MSG_ACTUAL_FPS.value: fps,
                CameraEnum.MSG_TARGET_FPS.value: 15,
            })
            st.camera_fps_handle_cmd(msg)
        assert len(st.camera_fps) == 3
        assert st.camera_fps["camera_head"][CameraEnum.MSG_ACTUAL_FPS.value] == 14.5
        assert st.camera_fps["camera_left"][CameraEnum.MSG_ACTUAL_FPS.value] == 9.2


class TestConfigParsing:
    """Validate glog.conf has the new SOC_TEMP section."""

    def test_soc_temp_section_exists(self):
        from configparser import ConfigParser
        config = ConfigParser()
        config.read("glog.conf")
        assert config.has_section(SocTempEnums.CONFIG_HEAD.value)

    def test_soc_temp_poll_interval_parseable(self):
        from configparser import ConfigParser
        config = ConfigParser()
        config.read("glog.conf")
        val = float(config.get(SocTempEnums.CONFIG_HEAD.value,
                               SocTempEnums.POLL_INTERVAL.value))
        assert val > 0
