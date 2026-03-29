"""Tests for MQTTClient base class: UUID dedup, send_command, log level handler."""

import pytest
import logging
from unittest.mock import MagicMock, patch
from json import dumps, loads
from uuid import uuid4


class TestMQTTClientOnMessage:
    """Test UUID deduplication in on_message."""

    def _make_client(self):
        from glados_modules.MqttConnector import MQTTClient
        with patch.object(MQTTClient, '__init__', lambda self, *a, **kw: None):
            c = MQTTClient.__new__(MQTTClient)
        from cachetools import TTLCache
        from threading import Lock
        c.uuid_cache = TTLCache(maxsize=100, ttl=60)
        c._lock = Lock()
        c.logger = MagicMock()
        c.topic_handler = {}
        c.__name__ = "test_client"
        return c

    def _make_msg(self, payload, topic="test/topic"):
        msg = MagicMock()
        msg.payload.decode.return_value = dumps(payload)
        msg.topic = topic
        return msg

    def test_unique_uuid_calls_handler(self):
        c = self._make_client()
        handler = MagicMock()
        c.topic_handler["test/topic"] = handler
        msg = self._make_msg({"uuid": "unique-1", "data": "hello"})
        c.on_message(None, None, msg)
        assert handler.called

    def test_duplicate_uuid_skipped(self):
        c = self._make_client()
        handler = MagicMock()
        c.topic_handler["test/topic"] = handler
        msg1 = self._make_msg({"uuid": "dup-1"})
        msg2 = self._make_msg({"uuid": "dup-1"})
        c.on_message(None, None, msg1)
        c.on_message(None, None, msg2)
        assert handler.call_count == 1

    def test_no_uuid_logs_error(self):
        c = self._make_client()
        msg = self._make_msg({"data": "no uuid"})
        c.on_message(None, None, msg)
        assert c.logger.error.called

    def test_unknown_topic_no_crash(self):
        c = self._make_client()
        # No handler registered for this topic
        msg = self._make_msg({"uuid": "x"}, topic="unknown/topic")
        c.on_message(None, None, msg)  # should not crash

    def test_handler_none_when_uuid_duplicate(self):
        """Regression test: handler must be initialized to None before lock block."""
        c = self._make_client()
        handler = MagicMock()
        c.topic_handler["test/topic"] = handler
        msg = self._make_msg({"uuid": "dup-2"})
        c.on_message(None, None, msg)  # first call
        c.on_message(None, None, msg)  # duplicate - handler should be None, not UnboundLocalError
        assert handler.call_count == 1


class TestMQTTClientSendCommand:
    """Test UUID injection and publishing."""

    def _make_client(self):
        from glados_modules.MqttConnector import MQTTClient
        with patch.object(MQTTClient, '__init__', lambda self, *a, **kw: None):
            c = MQTTClient.__new__(MQTTClient)
        c.client = MagicMock()
        c.logger = MagicMock()
        c.__name__ = "test_client"
        return c

    def test_single_dict_gets_uuid(self):
        c = self._make_client()
        cmd = {"angle": 90}
        c.send_command(cmd, "test/topic")
        assert "uuid" in cmd
        assert c.client.publish.called

    def test_list_of_commands(self):
        c = self._make_client()
        cmds = [{"a": 1}, {"b": 2}]
        c.send_command(cmds, "test/topic")
        assert c.client.publish.call_count == 2
        assert "uuid" in cmds[0]
        assert "uuid" in cmds[1]
        assert cmds[0]["uuid"] != cmds[1]["uuid"]

    def test_tuple_of_commands(self):
        c = self._make_client()
        cmds = ({"x": 1},)
        c.send_command(cmds, "test/topic")
        assert c.client.publish.call_count == 1


class TestLogLevelHandler:
    """Test remote log level control via MQTT."""

    def _make_client(self):
        from glados_modules.MqttConnector import MQTTClient
        with patch.object(MQTTClient, '__init__', lambda self, *a, **kw: None):
            c = MQTTClient.__new__(MQTTClient)
        c.logger = MagicMock()
        c.logger.handlers = [MagicMock()]
        c.__name__ = "TestModule"
        return c

    def _make_msg(self, payload):
        msg = MagicMock()
        msg.payload.decode.return_value = dumps(payload)
        return msg

    def test_matching_module_changes_level(self):
        c = self._make_client()
        msg = self._make_msg({"module": "TestModule", "level": "DEBUG"})
        c._handle_log_level(msg)
        c.logger.setLevel.assert_called_with(logging.DEBUG)

    def test_wildcard_changes_level(self):
        c = self._make_client()
        msg = self._make_msg({"module": "*", "level": "ERROR"})
        c._handle_log_level(msg)
        c.logger.setLevel.assert_called_with(logging.ERROR)

    def test_non_matching_module_ignored(self):
        c = self._make_client()
        msg = self._make_msg({"module": "OtherModule", "level": "DEBUG"})
        c._handle_log_level(msg)
        assert not c.logger.setLevel.called

    def test_invalid_level_defaults_to_info(self):
        c = self._make_client()
        msg = self._make_msg({"module": "TestModule", "level": "NONEXISTENT"})
        c._handle_log_level(msg)
        c.logger.setLevel.assert_called_with(logging.INFO)

    def test_malformed_json_no_crash(self):
        c = self._make_client()
        msg = MagicMock()
        msg.payload.decode.return_value = "not json"
        c._handle_log_level(msg)  # should not crash
