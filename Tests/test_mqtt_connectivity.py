"""Live MQTT broker connectivity test.

Reads the broker IP/port from glog.conf and verifies:
1. TCP connection to the broker succeeds
2. Can publish a message
3. Can subscribe and receive the published message

This test requires a running MQTT broker. Skip with: pytest -m "not live"

Usage:
    python -m pytest Tests/test_mqtt_connectivity.py -v
"""

import pytest
import time
import os
from configparser import ConfigParser
from json import dumps, loads
from threading import Event
from uuid import uuid4

from glados_modules.GladosEnums import SystemEnums


def _get_broker():
    """Read broker IP and port from glog.conf."""
    config = ConfigParser()
    # Try local glog.conf first, fall back to repo root
    for conf_path in ["glog.conf", os.path.join(os.path.dirname(__file__), "..", "glog.conf")]:
        if os.path.exists(conf_path):
            config.read(conf_path)
            break
    try:
        ip = config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_SERVER_IP.value)
        port = int(config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_PORT.value))
        return ip, port
    except Exception:
        return None, None


def _can_reach_broker():
    """Quick TCP check to see if broker is reachable."""
    import socket
    ip, port = _get_broker()
    if not ip:
        return False
    try:
        sock = socket.create_connection((ip, port), timeout=2)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


# Skip all tests in this file if broker is unreachable
pytestmark = pytest.mark.skipif(
    not _can_reach_broker(),
    reason="MQTT broker not reachable (is it running?)"
)


class TestMQTTBrokerConnectivity:
    """Live tests against the MQTT broker configured in glog.conf."""

    def _make_client(self):
        import paho.mqtt.client as mqtt
        ip, port = _get_broker()
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect(ip, port)
        return client

    def test_broker_accepts_connection(self):
        """Verify TCP connection to the broker succeeds."""
        client = self._make_client()
        client.disconnect()

    def test_publish_succeeds(self):
        """Verify a message can be published without error."""
        client = self._make_client()
        client.loop_start()
        result = client.publish("glados/test/ping", dumps({"test": True}))
        result.wait_for_publish(timeout=5)
        assert result.is_published()
        client.loop_stop()
        client.disconnect()

    def test_subscribe_and_receive(self):
        """Verify pub/sub round-trip: publish a message and receive it back."""
        import paho.mqtt.client as mqtt
        ip, port = _get_broker()
        topic = f"glados/test/{uuid4().hex[:8]}"
        test_payload = {"uuid": uuid4().hex, "ts": time.time()}
        received = Event()
        received_msg = {}

        def on_message(client, userdata, msg):
            data = loads(msg.payload.decode())
            received_msg.update(data)
            received.set()

        # Subscriber
        sub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        sub.on_message = on_message
        sub.connect(ip, port)
        sub.subscribe(topic)
        sub.loop_start()

        # Give subscription time to register
        time.sleep(0.5)

        # Publisher
        pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        pub.connect(ip, port)
        pub.loop_start()
        pub.publish(topic, dumps(test_payload))

        # Wait for message
        assert received.wait(timeout=5), "Did not receive message within 5 seconds"
        assert received_msg["uuid"] == test_payload["uuid"]

        pub.loop_stop()
        pub.disconnect()
        sub.loop_stop()
        sub.disconnect()

    def test_broker_ip_matches_config(self):
        """Verify glog.conf has a valid MQTT section."""
        ip, port = _get_broker()
        assert ip is not None, "Could not read mqtt_server_ip from glog.conf"
        assert port > 0, "Invalid MQTT port"
        assert port == 1883, f"Unexpected MQTT port: {port}"
