"""Tests for WebDashboard Flask routes."""

import pytest
from unittest.mock import MagicMock, patch
from threading import Lock


def _make_dashboard():
    """Create a WebDashboard with mocked dependencies."""
    from glados_modules.WebDashboard import WebDashboard
    with patch.object(WebDashboard, '__init__', lambda self, *a, **kw: None):
        dash = WebDashboard.__new__(WebDashboard)
    # Must call Thread.__init__ so .daemon setter works
    from threading import Thread
    Thread.__init__(dash)
    dash.__name__ = "WebDashboard_test"
    dash.system_name = "test_server"
    dash.health_monitor = MagicMock()
    dash.health_monitor.get_status.return_value = {
        "hostname": "test", "system": "test_server",
        "uptime_s": 100, "threads": {}, "mqtt_connected": True, "ts": 1000.0,
    }
    dash.health_monitor._peer_status = {"pi4": {"hostname": "pi4", "system": "body"}}
    dash.rtsp_server = None
    dash.feeds = {}
    dash.feed_uris = {}
    dash._all_feeds = {}
    dash._grabbers = {}
    dash.port = 8080
    dash.logger = MagicMock()
    dash.daemon = True
    # Build the Flask app
    dash._app = dash._create_app()
    return dash


class TestDashboardRoutes:
    def test_index_returns_html(self):
        dash = _make_dashboard()
        with dash._app.test_client() as client:
            resp = client.get('/')
            assert resp.status_code == 200
            assert b'GLaDOS' in resp.data
            assert b'test_server' in resp.data

    def test_health_api_returns_json(self):
        dash = _make_dashboard()
        with dash._app.test_client() as client:
            resp = client.get('/api/health')
            assert resp.status_code == 200
            data = resp.get_json()
            assert "local" in data
            assert "peers" in data
            assert data["local"]["hostname"] == "test"
            assert data["local"]["mqtt_connected"] is True

    def test_health_api_includes_peers(self):
        dash = _make_dashboard()
        with dash._app.test_client() as client:
            resp = client.get('/api/health')
            data = resp.get_json()
            assert "pi4" in data["peers"]
            assert data["peers"]["pi4"]["system"] == "body"

    def test_feed_returns_mjpeg_content_type(self):
        dash = _make_dashboard()
        with dash._app.test_client() as client:
            # No actual frame source -- should return "Connecting..." frame
            resp = client.get('/feed/buffer/camera_head')
            assert resp.status_code == 200
            assert 'multipart/x-mixed-replace' in resp.content_type

    def test_health_api_no_monitor(self):
        dash = _make_dashboard()
        dash.health_monitor = None
        with dash._app.test_client() as client:
            resp = client.get('/api/health')
            data = resp.get_json()
            assert data["local"] == {}
            assert data["peers"] == {}
