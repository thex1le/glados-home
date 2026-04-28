"""Integration tests: message serialization roundtrips.

Tests that messages built by one system can be correctly parsed by another.
"""

import pytest
from unittest.mock import MagicMock, patch
from json import dumps, loads
from collections import namedtuple

from glados_modules.GladosEnums import (
    ServoEnum, CameraEnum, VisionResultsEnum, IMUEnums, TOFEnums, THEnums, MOXEnums,
    SceneEnums,
)
from glados_modules.MqttConnector import (
    ServoMessageBuilder, CameraMessageBuilder, IMUMessageBuilder,
    TOFMessageBuilder, THMessageBuilder, MoxGasMessageBuilder,
    BrainMessageBuilder, SceneMessageBuilder, MoodMessageBuilder,
)


class TestServoCommandRoundtrip:
    """Build a servo command, serialize it, deserialize it, verify parsed correctly."""

    def test_move_roundtrip(self):
        # Build
        msg = ServoMessageBuilder.move(ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value, 95, 3)
        msg["uuid"] = "test-uuid"
        # Serialize (what send_command does)
        wire = dumps(msg)
        # Deserialize (what on_message does)
        parsed = loads(wire)
        # Verify (what handle_cmd checks)
        assert parsed[ServoEnum.MSG_COMMAND_KEY.value] == ServoEnum.MSG_COMMAND_MOVE.value
        assert parsed[ServoEnum.MSG_LOCATION_KEY.value] == ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
        assert parsed[ServoEnum.MSG_ANGLE.value] == 95
        assert parsed[ServoEnum.MSG_SPEED.value] == 3

    def test_move_all_roundtrip(self):
        targets = {
            ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: {
                ServoEnum.MSG_ANGLE.value: 85, ServoEnum.MSG_SPEED.value: 3
            },
            ServoEnum.LOCATION_HEAD_UP_DOWN.value: {
                ServoEnum.MSG_ANGLE.value: 110, ServoEnum.MSG_SPEED.value: 3
            },
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value: {
                ServoEnum.MSG_ANGLE.value: 100, ServoEnum.MSG_SPEED.value: 3
            },
            ServoEnum.LOCATION_BODY_UP_DOWN.value: {
                ServoEnum.MSG_ANGLE.value: 95, ServoEnum.MSG_SPEED.value: 3
            },
        }
        msg = ServoMessageBuilder.move_all(targets)
        msg["uuid"] = "test-uuid"
        wire = dumps(msg)
        parsed = loads(wire)

        assert parsed[ServoEnum.MSG_COMMAND_KEY.value] == ServoEnum.MSG_COMMAND_MOVE_ALL.value
        ptargets = parsed[ServoEnum.MSG_TARGETS.value]
        for loc in [ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value,
                    ServoEnum.LOCATION_HEAD_UP_DOWN.value,
                    ServoEnum.LOCATION_BODY_LEFT_RIGHT.value,
                    ServoEnum.LOCATION_BODY_UP_DOWN.value]:
            assert loc in ptargets
            assert ServoEnum.MSG_ANGLE.value in ptargets[loc]
            assert ServoEnum.MSG_SPEED.value in ptargets[loc]

    def test_status_roundtrip(self):
        # What Gservo.get_angles() returns
        results = {
            ServoEnum.MSG_MAX.value: 180,
            ServoEnum.MSG_MIN.value: 0,
            ServoEnum.MSG_MIDDLE.value: 90,
            ServoEnum.MSG_CURRENT_ANGLE.value: 95.5,
            ServoEnum.MSG_VELOCITY.value: 1.2,
            ServoEnum.MSG_AXIS.value: "x",
            ServoEnum.MSG_MOVING.value: True,
            ServoEnum.MSG_LAST_ANGLE.value: 100.0,
        }
        msg = ServoMessageBuilder.send_status(ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value, results)
        msg["uuid"] = "test-uuid"
        wire = dumps(msg)
        parsed = loads(wire)

        assert parsed[ServoEnum.MSG_LOCATION_KEY.value] == ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value
        r = parsed[ServoEnum.MSG_RESULTS.value]
        assert r[ServoEnum.MSG_CURRENT_ANGLE.value] == 95.5
        assert r[ServoEnum.MSG_VELOCITY.value] == 1.2
        assert r[ServoEnum.MSG_MOVING.value] is True


class TestSensorMessageRoundtrip:
    """Build sensor messages, serialize, verify structure."""

    def test_imu_roundtrip(self):
        sensor_data = {
            IMUEnums.TEMP_KEY.value: 25,
            IMUEnums.ACCEL_KEY.value: (0.1, 0.2, 9.8),
            IMUEnums.GYRO_KEY.value: (0.01, 0.02, 0.03),
            IMUEnums.EULER_KEY.value: (10, 20, 30),
            IMUEnums.LINEAR_KEY.value: (0.05, 0.1, 0.02),
            IMUEnums.GRAVITY_KEY.value: (0, 0, 9.8),
            IMUEnums.IMU_TIME_STAMP_KEY.value: 1234567.0,
        }
        msg = IMUMessageBuilder.send_imu_status_message(sensor_data)
        msg["uuid"] = "test"
        wire = dumps(msg)
        parsed = loads(wire)
        assert IMUEnums.IMU_STATUS_KEY.value in parsed
        imu = parsed[IMUEnums.IMU_STATUS_KEY.value]
        assert imu[IMUEnums.GYRO_KEY.value] == [0.01, 0.02, 0.03]

    def test_tof_roundtrip(self):
        sensor_data = {TOFEnums.TOF_STATUS_KEY.value: 1.5, TOFEnums.TOF_TIME_STAMP_KEY.value: 123.0}
        msg = TOFMessageBuilder.send_tof_status_message(sensor_data)
        msg["uuid"] = "test"
        wire = dumps(msg)
        parsed = loads(wire)
        assert TOFEnums.TOF_STATUS_KEY.value in parsed

    def test_th_roundtrip(self):
        sensor_data = {
            THEnums.TH_HUMIDITY_KEY.value: 45.0,
            THEnums.TH_CELSIUS_KEY.value: 22.5,
            THEnums.TH_TIME_STAMP_KEY.value: 123.0,
        }
        msg = THMessageBuilder.send_th_status_message(sensor_data)
        msg["uuid"] = "test"
        wire = dumps(msg)
        parsed = loads(wire)
        assert THEnums.TH_STATUS_KEY.value in parsed

    def test_mox_roundtrip(self):
        sensor_data = {
            MOXEnums.MOX_AQI.value: 2,
            MOXEnums.MOX_TVOC.value: 150,
            MOXEnums.MOX_ECO2.value: 600,
            MOXEnums.MOX_TIME_STAMP_KEY.value: 123.0,
        }
        msg = MoxGasMessageBuilder.send_mox_status_message(sensor_data)
        msg["uuid"] = "test"
        wire = dumps(msg)
        parsed = loads(wire)
        assert MOXEnums.MOX_STATUS_KEY.value in parsed


class TestSensorTrackerParsing:
    """Verify SensorTracker correctly stores sensor data from MQTT messages."""

    def _make_tracker(self):
        from glados_modules.MqttConsumerModules import SensorTracker
        broker = namedtuple("broker", ["ip", "port"])("localhost", 1883)
        with patch.object(SensorTracker, '__init__', lambda self, *a, **kw: None):
            st = SensorTracker.__new__(SensorTracker)
        st.__name__ = "SensorTracker"
        st.logger = MagicMock()
        st._lock = __import__('threading').Lock()
        st.imu_status = {}
        st.tof_status = {}
        st.th_status = {}
        st.mox_status = {}
        st._last_update = {}
        st.topic_handler = {}
        return st

    def _make_msg(self, payload):
        msg = MagicMock()
        msg.payload.decode.return_value = dumps(payload)
        return msg

    def test_imu_stored(self):
        st = self._make_tracker()
        data = {IMUEnums.IMU_STATUS_KEY.value: {"gyroscope": [0, 0, 0], "linear": [0, 0, 0]}}
        st._handle_status(self._make_msg(data), IMUEnums.IMU_STATUS_KEY.value, "IMU")
        assert st.imu_status == data[IMUEnums.IMU_STATUS_KEY.value]

    def test_tof_stored(self):
        st = self._make_tracker()
        data = {TOFEnums.TOF_STATUS_KEY.value: {"distance": 1.5}}
        st._handle_status(self._make_msg(data), TOFEnums.TOF_STATUS_KEY.value, "TOF")
        assert st.tof_status == data[TOFEnums.TOF_STATUS_KEY.value]

    def test_get_sensor_status(self):
        st = self._make_tracker()
        st.imu_status = {"gyroscope": [1, 2, 3]}
        result = st.get_sensor_status(IMUEnums.IMU_STATUS_KEY.value)
        assert result == {"gyroscope": [1, 2, 3]}

    def test_get_nonexistent_sensor(self):
        st = self._make_tracker()
        result = st.get_sensor_status("nonexistent")
        assert result is None


class TestBrainMessageRoundtrip:
    """Brain emits utterance / tool_call / ready messages; verify wire format."""

    def test_utterance_roundtrip(self):
        msg = BrainMessageBuilder.utterance("test the cake", source="llm")
        msg["uuid"] = "test-uuid"
        wire = dumps(msg)
        parsed = loads(wire)
        assert parsed["text"] == "test the cake"
        assert parsed["source"] == "llm"

    def test_utterance_default_source(self):
        msg = BrainMessageBuilder.utterance("hello")
        assert msg["source"] == "llm"

    def test_tool_call_roundtrip(self):
        msg = BrainMessageBuilder.tool_call(
            "look_at", {"yaw": 45, "pitch": 90}, lane="priority")
        msg["uuid"] = "test-uuid"
        wire = dumps(msg)
        parsed = loads(wire)
        assert parsed["tool"] == "look_at"
        assert parsed["args"]["yaw"] == 45
        assert parsed["args"]["pitch"] == 90
        assert parsed["lane"] == "priority"

    def test_tool_call_default_lane(self):
        msg = BrainMessageBuilder.tool_call("speak", {"text": "hi"})
        assert msg["lane"] == "priority"

    def test_ready_roundtrip(self):
        msg = BrainMessageBuilder.ready("GladosBrain", "qwen2.5:14b-instruct-q4_K_M")
        msg["uuid"] = "test-uuid"
        wire = dumps(msg)
        parsed = loads(wire)
        assert parsed["system"] == "GladosBrain"
        assert parsed["model"] == "qwen2.5:14b-instruct-q4_K_M"


class TestSceneMessageRoundtrip:
    """SceneDescriber publishes background descriptions and answers tool requests."""

    def test_description_roundtrip(self):
        msg = SceneMessageBuilder.description("head", "a person at a desk", 1234.5)
        msg["uuid"] = "test-uuid"
        wire = dumps(msg)
        parsed = loads(wire)
        assert parsed[SceneEnums.CAMERA_KEY.value] == "head"
        assert parsed[SceneEnums.DESCRIPTION_KEY.value] == "a person at a desk"
        assert parsed[SceneEnums.TS_KEY.value] == 1234.5

    def test_describe_request_roundtrip(self):
        msg = SceneMessageBuilder.describe_request(
            "req-123", "what do you see?", max_tokens=200)
        msg["uuid"] = "test-uuid"
        wire = dumps(msg)
        parsed = loads(wire)
        assert parsed[SceneEnums.REQUEST_ID_KEY.value] == "req-123"
        assert parsed[SceneEnums.PROMPT_KEY.value] == "what do you see?"
        assert parsed[SceneEnums.MAX_TOKENS_KEY.value] == 200

    def test_describe_request_default_max_tokens(self):
        msg = SceneMessageBuilder.describe_request("r1", "look")
        assert msg[SceneEnums.MAX_TOKENS_KEY.value] == 256

    def test_describe_response_roundtrip(self):
        msg = SceneMessageBuilder.describe_response("req-123", "a chair")
        msg["uuid"] = "test-uuid"
        wire = dumps(msg)
        parsed = loads(wire)
        assert parsed[SceneEnums.REQUEST_ID_KEY.value] == "req-123"
        assert parsed[SceneEnums.DESCRIPTION_KEY.value] == "a chair"

    def test_request_response_id_pairs(self):
        # The request_id field name must match between request and response so
        # GladosBrain can correlate replies.
        request = SceneMessageBuilder.describe_request("abc", "look")
        response = SceneMessageBuilder.describe_response("abc", "result")
        assert request[SceneEnums.REQUEST_ID_KEY.value] == \
               response[SceneEnums.REQUEST_ID_KEY.value]


class TestMoodMessageRoundtrip:
    """Brain publishes PAD vectors and receives hardware mood events."""

    def test_pad_roundtrip(self):
        msg = MoodMessageBuilder.pad(0.3, -0.5, 0.7, 1234.5)
        msg["uuid"] = "test-uuid"
        wire = dumps(msg)
        parsed = loads(wire)
        assert parsed["pleasure"] == pytest.approx(0.3)
        assert parsed["arousal"] == pytest.approx(-0.5)
        assert parsed["dominance"] == pytest.approx(0.7)
        assert parsed["ts"] == pytest.approx(1234.5)

    def test_pad_coerces_to_floats(self):
        # Even if the EmotionAgent hands us numpy floats or ints, the wire
        # format must be plain float-compatible JSON
        msg = MoodMessageBuilder.pad(1, 0, -1, 100)
        wire = dumps(msg)  # would raise if numpy types leaked through
        parsed = loads(wire)
        assert isinstance(parsed["pleasure"], float)

    def test_event_roundtrip(self):
        msg = MoodMessageBuilder.event(
            source="user",
            description="User asked another question (#3 in 30s)")
        wire = dumps(msg)
        parsed = loads(wire)
        assert parsed["source"] == "user"
        assert "pestering" not in parsed["description"]  # sanity
        assert "#3" in parsed["description"]
