"""Web dashboard for GLaDOS system monitoring and live video feeds.

Runs a lightweight Flask server in a daemon thread alongside the main
GLaDOS process. Serves MJPEG video streams and system health data.

Two modes for video:
- **Direct buffer** (GPU server): reads frames from RTSPServer.factories[path].data
- **RTSP consumer** (Pi4/Pi5): connects to RTSP stream via OpenCV since
  the camera runs in a separate process and frames aren't in shared memory

Usage:
    # GPU server (direct access to annotated frames):
    dashboard = WebDashboard(
        system_name="ai_server",
        health_monitor=health,
        rtsp_server=mv.rtsp,  # MLDetect's RTSPServer
        feeds={"Head (Annotated)": "/camera_head", ...},
        port=8080
    )

    # Pi4/Pi5 (consume RTSP streams):
    dashboard = WebDashboard(
        system_name="body_server",
        health_monitor=health,
        feed_uris={"Head Camera": "rtsp://localhost:8554/camera_head"},
        port=8080
    )
"""

import time
import cv2
import numpy as np
from threading import Thread, Lock
from typing import Dict, Any, Optional

from flask import Flask, Response, jsonify, render_template_string

from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import LoggingEnums


# Suppress Flask/Werkzeug request logging to keep console clean
import logging as _logging
_logging.getLogger('werkzeug').setLevel(_logging.WARNING)


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GLaDOS - {{ system_name }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0a0a; color: #ff9a00; font-family: 'Courier New', monospace;
            padding: 16px;
        }
        h1 { font-size: 1.4em; margin-bottom: 12px; color: #ff6600; }
        h2 { font-size: 1em; margin-bottom: 8px; color: #ff9a00; border-bottom: 1px solid #333; padding-bottom: 4px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 16px; }
        .card {
            background: #111; border: 1px solid #333; border-radius: 6px;
            padding: 12px; overflow: hidden;
        }
        .card img { width: 100%; border-radius: 4px; background: #000; }
        .status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; }
        .status-item { background: #1a1a1a; padding: 8px; border-radius: 4px; font-size: 0.85em; }
        .status-label { color: #888; font-size: 0.75em; text-transform: uppercase; }
        .status-value { color: #ff9a00; font-size: 1.1em; margin-top: 2px; }
        .alive { color: #00ff44; }
        .dead { color: #ff0044; animation: blink 1s infinite; }
        @keyframes blink { 50% { opacity: 0.3; } }
        .thread-row { display: flex; justify-content: space-between; padding: 4px 8px;
                      background: #1a1a1a; margin: 2px 0; border-radius: 3px; font-size: 0.85em; }
        .thread-name { color: #ccc; }
        .error-count { color: #ff6600; font-size: 0.8em; margin-left: 8px; }
        .peer-section { margin-top: 12px; }
        .peer-card { background: #1a1a1a; padding: 8px; border-radius: 4px; margin: 4px 0; }
        .peer-name { color: #ff9a00; font-weight: bold; }
        .peer-info { color: #888; font-size: 0.8em; }
        .no-feed { width: 100%; height: 300px; background: #000; display: flex;
                   align-items: center; justify-content: center; color: #333;
                   font-size: 1.2em; border-radius: 4px; }
        .header-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
        .uptime { color: #888; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header-bar">
        <h1>GLaDOS // {{ system_name }}</h1>
        <span class="uptime" id="uptime"></span>
    </div>

    <div class="grid">
        {% for label, path in feeds.items() %}
        <div class="card">
            <h2>{{ label }}</h2>
            <img src="/feed/{{ path }}" alt="{{ label }}" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"/>
            <div class="no-feed" style="display:none">No Feed</div>
        </div>
        {% endfor %}

        <div class="card">
            <h2>System Health</h2>
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-label">MQTT</div>
                    <div class="status-value" id="mqtt-status">--</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Uptime</div>
                    <div class="status-value" id="sys-uptime">--</div>
                </div>
            </div>
            <h2 style="margin-top:12px">Threads</h2>
            <div id="threads"></div>
        </div>

        <div class="card">
            <h2>Connected Systems</h2>
            <div id="peers" class="peer-section">
                <div style="color:#555">Waiting for heartbeats...</div>
            </div>
        </div>
    </div>

    <script>
        function formatUptime(s) {
            const h = Math.floor(s / 3600);
            const m = Math.floor((s % 3600) / 60);
            const sec = Math.floor(s % 60);
            return h > 0 ? h+'h '+m+'m' : m > 0 ? m+'m '+sec+'s' : sec+'s';
        }

        async function updateHealth() {
            try {
                const r = await fetch('/api/health');
                const d = await r.json();

                // Local system
                document.getElementById('mqtt-status').textContent = d.local.mqtt_connected ? 'Connected' : 'Disconnected';
                document.getElementById('mqtt-status').className = 'status-value ' + (d.local.mqtt_connected ? 'alive' : 'dead');
                document.getElementById('sys-uptime').textContent = formatUptime(d.local.uptime_s || 0);
                document.getElementById('uptime').textContent = new Date().toLocaleTimeString();

                // Threads
                const tDiv = document.getElementById('threads');
                let html = '';
                const threads = d.local.threads || {};
                for (const [name, info] of Object.entries(threads)) {
                    const cls = info.alive ? 'alive' : 'dead';
                    const err = info.errors > 0 ? '<span class="error-count">(' + info.errors + ' errors)</span>' : '';
                    html += '<div class="thread-row"><span class="thread-name">' + name + '</span>'
                          + '<span class="' + cls + '">' + (info.alive ? 'ALIVE' : 'DEAD') + '</span>' + err + '</div>';
                }
                tDiv.innerHTML = html || '<div style="color:#555">No threads registered</div>';

                // Peers
                const pDiv = document.getElementById('peers');
                let phtml = '';
                const peers = d.peers || {};
                for (const [host, info] of Object.entries(peers)) {
                    const age = Math.round(Date.now()/1000 - (info.ts || 0));
                    const cls = age < 15 ? 'alive' : 'dead';
                    const threadCount = Object.keys(info.threads || {}).length;
                    const deadCount = Object.values(info.threads || {}).filter(t => !t.alive).length;
                    const threadStr = deadCount > 0 ? threadCount-deadCount+'/'+threadCount+' alive' : threadCount+' threads';
                    phtml += '<div class="peer-card"><span class="peer-name">' + host + '</span>'
                           + ' <span class="peer-info">(' + (info.system||'?') + ')</span>'
                           + '<div class="peer-info">' + threadStr
                           + ' &bull; uptime ' + formatUptime(info.uptime_s || 0)
                           + ' &bull; <span class="'+cls+'">' + age + 's ago</span></div></div>';
                }
                pDiv.innerHTML = phtml || '<div style="color:#555">No peers detected</div>';
            } catch(e) {
                document.getElementById('uptime').textContent = 'API Error';
            }
        }

        updateHealth();
        setInterval(updateHealth, 2000);
    </script>
</body>
</html>
"""


class RTSPFrameGrabber:
    """Background thread that consumes an RTSP stream and caches the latest frame.
    Used on Pi4/Pi5 where camera frames live in a separate process.
    """

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.frame = None
        self.lock = Lock()
        self._running = True
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        cap = None
        while self._running:
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(self.uri)
                    if not cap.isOpened():
                        time.sleep(2)
                        continue
                ret, frame = cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    cap.release()
                    cap = None
                    time.sleep(1)
            except Exception:
                time.sleep(2)

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self) -> None:
        self._running = False


class WebDashboard(Thread):
    """Flask-based web dashboard running in a daemon thread.

    Serves:
    - / : HTML dashboard with video feeds and health status
    - /feed/<path> : MJPEG stream (direct buffer or RTSP consumer)
    - /api/health : JSON health data for local system + peers

    Args:
        rtsp_server: Direct frame buffer access (GPU server where MLDetect runs in-process)
        feed_uris: Dict of {label: rtsp_uri} for consuming external RTSP streams (Pi4/Pi5)
        feeds: Dict of {label: factory_path} for direct buffer feeds (GPU server)
    """

    def __init__(self, system_name: str, health_monitor=None,
                 rtsp_server=None, feeds: Dict[str, str] = None,
                 feed_uris: Dict[str, str] = None,
                 port: int = 8080) -> None:
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = f"WebDashboard_{system_name}"
        self.logger = setup_logger(self.__name__, LoggingEnums.LOG_LEVEL_INFO.value)
        self.system_name = system_name
        self.health_monitor = health_monitor
        self.rtsp_server = rtsp_server  # direct buffer (GPU server)
        self.feeds = feeds or {}        # {label: factory_path} for direct buffer
        self.feed_uris = feed_uris or {}  # {label: rtsp_uri} for RTSP consumer
        self.port = port

        # Start RTSP frame grabbers for URI-based feeds
        self._grabbers: Dict[str, RTSPFrameGrabber] = {}
        for label, uri in self.feed_uris.items():
            self._grabbers[label] = RTSPFrameGrabber(uri)
            self.logger.info(f"RTSP grabber started for {label}: {uri}")

        # Merge both feed types into one display dict {label: key}
        self._all_feeds: Dict[str, str] = {}
        for label, path in self.feeds.items():
            self._all_feeds[label] = f"buffer{path}"  # prefix to distinguish
        for label, uri in self.feed_uris.items():
            self._all_feeds[label] = f"grabber:{label}"

        self._app = self._create_app()

    def _create_app(self) -> Flask:
        app = Flask(__name__)
        dashboard = self

        @app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML,
                                          system_name=dashboard.system_name,
                                          feeds=dashboard._all_feeds)

        @app.route('/feed/<path:feed_key>')
        def video_feed(feed_key):
            """MJPEG stream from either direct buffer or RTSP grabber."""
            return Response(
                dashboard._generate_mjpeg(feed_key),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @app.route('/api/health')
        def health_api():
            result = {"local": {}, "peers": {}}
            if dashboard.health_monitor:
                result["local"] = dashboard.health_monitor.get_status()
                result["peers"] = dict(dashboard.health_monitor._peer_status)
            return jsonify(result)

        return app

    def _get_frame(self, feed_key: str):
        """Get the latest frame for a feed, from either buffer or grabber."""
        if feed_key.startswith("buffer"):
            # Direct buffer: strip prefix, read from RTSPServer
            path = feed_key[len("buffer"):]
            if self.rtsp_server and path in self.rtsp_server.factories:
                rtsp_sys = self.rtsp_server.factories[path]
                with rtsp_sys.data_lock:
                    return rtsp_sys.data
        elif feed_key.startswith("grabber:"):
            # RTSP consumer: look up grabber by label
            label = feed_key[len("grabber:"):]
            if label in self._grabbers:
                return self._grabbers[label].get_frame()
        return None

    def _generate_mjpeg(self, feed_key: str):
        """Generator that yields JPEG frames as MJPEG stream."""
        # Send a "No Feed" frame first in case connection is slow
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Connecting...", (190, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (80, 80, 80), 2)
        _, jpeg = cv2.imencode('.jpg', blank)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        while True:
            frame = self._get_frame(feed_key)
            if frame is not None:
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            # ~15 FPS for web stream
            time.sleep(0.066)

    def run(self) -> None:
        """Start the Flask server."""
        self.logger.info(f"Web dashboard starting at http://0.0.0.0:{self.port}")
        self._app.run(host='0.0.0.0', port=self.port, threaded=True, use_reloader=False)
