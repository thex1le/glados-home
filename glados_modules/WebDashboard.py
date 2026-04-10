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

import os
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
        .layout { display: grid; grid-template-columns: 1fr 360px; gap: 16px; }
        .video-columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 16px; }
        .sidebar { display: flex; flex-direction: column; gap: 16px; }
        .card {
            background: #111; border: 1px solid #333; border-radius: 6px;
            padding: 12px; overflow: hidden;
        }
        .card img { width: 100%; border-radius: 4px; background: #000; }
        .video-pair { display: flex; flex-direction: column; gap: 8px; }
        .video-pair img { width: 100%; border-radius: 4px; background: #000; }
        .video-label { font-size: 0.75em; color: #888; text-transform: uppercase; margin-bottom: 2px; }
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
        .peer-section { margin-top: 8px; }
        .peer-card { background: #1a1a1a; padding: 8px; border-radius: 4px; margin: 6px 0; border-left: 3px solid #333; }
        .peer-card.healthy { border-left-color: #00ff44; }
        .peer-card.unhealthy { border-left-color: #ff0044; }
        .peer-card.stale { border-left-color: #ff6600; }
        .peer-name { color: #ff9a00; font-weight: bold; }
        .peer-info { color: #888; font-size: 0.8em; }
        .peer-threads { margin-top: 4px; }
        .no-feed { width: 100%; height: 200px; background: #000; display: flex;
                   align-items: center; justify-content: center; color: #333;
                   font-size: 1em; border-radius: 4px; }
        .header-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
        .uptime { color: #888; font-size: 0.9em; }
        @media (max-width: 900px) {
            .layout { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header-bar">
        <h1>GLaDOS // {{ system_name }}</h1>
        <span class="uptime" id="uptime"></span>
    </div>

    <div class="layout">
        <div class="video-columns">
            {% for pair in feed_pairs %}
            <div class="card">
                <h2>{{ pair.name }}</h2>
                <div class="video-pair">
                    {% for feed in pair.feeds %}
                    <div>
                        <div class="video-label">{{ feed.label }}</div>
                        <img src="/feed/{{ feed.key }}" alt="{{ feed.label }}"
                             onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"/>
                        <div class="no-feed" style="display:none">No Feed</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}

            {% for label, path in unpaired_feeds.items() %}
            <div class="card">
                <h2>{{ label }}</h2>
                <img src="/feed/{{ path }}" alt="{{ label }}"
                     onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"/>
                <div class="no-feed" style="display:none">No Feed</div>
            </div>
            {% endfor %}
        </div>

        <div class="sidebar">
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
                <h2>Tracking State</h2>
                <div id="tracking-state">
                    <div style="color:#555">Waiting for data...</div>
                </div>
            </div>

            <div class="card">
                <h2>Connected Systems</h2>
                <div id="peers" class="peer-section">
                    <div style="color:#555">Waiting for heartbeats...</div>
                </div>
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

                // Peers -- full thread detail view
                const pDiv = document.getElementById('peers');
                let phtml = '';
                const peers = d.peers || {};
                for (const [host, info] of Object.entries(peers)) {
                    const age = Math.round(Date.now()/1000 - (info.ts || 0));
                    const peerThreads = info.threads || {};
                    const deadCount = Object.values(peerThreads).filter(t => !t.alive).length;
                    const cardCls = age > 15 ? 'stale' : deadCount > 0 ? 'unhealthy' : 'healthy';
                    const mqttStr = info.mqtt_connected ? '<span class="alive">MQTT OK</span>' : '<span class="dead">MQTT DOWN</span>';
                    phtml += '<div class="peer-card ' + cardCls + '">'
                           + '<span class="peer-name">' + host + '</span>'
                           + ' <span class="peer-info">(' + (info.system||'?') + ')</span>'
                           + '<div class="peer-info">' + mqttStr
                           + ' &bull; uptime ' + formatUptime(info.uptime_s || 0)
                           + ' &bull; <span class="' + (age < 15 ? 'alive' : 'dead') + '">' + age + 's ago</span></div>'
                           + '<div class="peer-threads">';
                    for (const [tname, tinfo] of Object.entries(peerThreads)) {
                        const tcls = tinfo.alive ? 'alive' : 'dead';
                        const terr = tinfo.errors > 0 ? '<span class="error-count">(' + tinfo.errors + ' errors)</span>' : '';
                        phtml += '<div class="thread-row"><span class="thread-name">' + tname + '</span>'
                               + '<span class="' + tcls + '">' + (tinfo.alive ? 'ALIVE' : 'DEAD') + '</span>' + terr + '</div>';
                    }
                    phtml += '</div></div>';
                }
                pDiv.innerHTML = phtml || '<div style="color:#555">No peers detected</div>';
            } catch(e) {
                document.getElementById('uptime').textContent = 'API Error';
            }
        }

        async function updateTracking() {
            try {
                const r = await fetch('/api/tracking');
                const d = await r.json();
                const div = document.getElementById('tracking-state');
                let html = '';

                if (d.overlay) {
                    const o = d.overlay;
                    html += '<div class="status-grid">';
                    html += '<div class="status-item"><div class="status-label">State</div>'
                          + '<div class="status-value">' + (o.state || '?') + '</div></div>';
                    html += '<div class="status-item"><div class="status-label">World</div>'
                          + '<div class="status-value">LR ' + (o.world_lr||0).toFixed(1) + '° UD ' + (o.world_ud||0).toFixed(1) + '°</div></div>';
                    html += '<div class="status-item"><div class="status-label">Head Target</div>'
                          + '<div class="status-value">LR ' + Math.round(o.head_lr||0) + '° UD ' + Math.round(o.head_ud||0) + '°</div></div>';
                    html += '<div class="status-item"><div class="status-label">Body Target</div>'
                          + '<div class="status-value">LR ' + Math.round(o.body_lr||0) + '° UD ' + Math.round(o.body_ud||0) + '°</div></div>';
                    html += '</div>';
                }

                if (d.room && d.room.count > 0) {
                    html += '<h2 style="margin-top:8px">Room (' + d.room.count + ')</h2>';
                    for (const p of d.room.roster || []) {
                        const emo = p.emotion && p.emotion !== 'neutral' ? ' · ' + p.emotion : '';
                        const cams = (p.cameras||[]).map(c => c.replace('camera_','')).join(',');
                        html += '<div class="thread-row"><span class="thread-name">' + p.person_id + emo + '</span>'
                              + '<span style="color:#888">[' + cams + '] ' + (p.world_lr||0).toFixed(0) + '°</span></div>';
                    }
                }

                if (d.attention && d.attention.attention_target) {
                    html += '<div style="margin-top:6px;padding:6px;background:#1a1a1a;border-radius:4px;border-left:3px solid #ff6600">'
                          + '<span style="color:#ff6600">▸ </span><span class="thread-name">' + d.attention.attention_target + '</span>'
                          + ' <span style="color:#888">(' + d.attention.attention_reason + ')</span></div>';
                }

                div.innerHTML = html || '<div style="color:#555">No tracking data</div>';
            } catch(e) {}
        }

        updateHealth();
        updateTracking();
        setInterval(updateHealth, 2000);
        setInterval(updateTracking, 500);
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
        self.frame_time: float = 0.0
        self.lock = Lock()
        self._running = True
        self.logger = setup_logger(name=f"RTSPGrabber_{uri.split('/')[-1]}",
                                    console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _safe_release(self, cap) -> None:
        """Release VideoCapture and force GStreamer pipeline to NULL state."""
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass  # Expected during teardown — GStreamer may already be NULL
            # Give GStreamer time to tear down before creating a new pipeline
            time.sleep(0.5)

    def _capture_loop(self) -> None:
        cap = None
        backoff = 2.0
        max_backoff = 30.0
        while self._running:
            try:
                if cap is None or not cap.isOpened():
                    # Try GStreamer pipeline first (GPU server has GStreamer-enabled OpenCV)
                    # latency=0 and drop=true minimize decode buffering
                    gst_uri = (f"rtspsrc location={self.uri} latency=0 "
                               f"tcp-timeout=5000000 timeout=5000000 ! "
                               f"rtph264depay ! h264parse ! avdec_h264 ! "
                               f"videoconvert ! appsink drop=true max-buffers=1 sync=false")
                    cap = cv2.VideoCapture(gst_uri, cv2.CAP_GSTREAMER)
                    if not cap.isOpened():
                        # Fall back to FFmpeg with plain RTSP URL (Pi4/Pi5 without GStreamer OpenCV)
                        # Set low-latency flags, then restore env to avoid affecting other captures
                        old_opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                            "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|"
                            "max_delay;0|analyzeduration;0|probesize;32768")
                        cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG)
                        # Restore immediately — env var only needed during open()
                        if old_opts:
                            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = old_opts
                        else:
                            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                    if not cap.isOpened():
                        self._safe_release(cap)
                        cap = None
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, max_backoff)
                        continue
                    backoff = 2.0  # reset on successful connect
                    last_frame_time = time.time()
                ret, frame = cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                        self.frame_time = time.time()
                    last_frame_time = time.time()
                else:
                    # Check if stream went stale (connected but no frames)
                    if time.time() - last_frame_time > 5.0:
                        self.logger.debug(f"RTSP grabber: no frames for 5s, reconnecting {self.uri}")
                        self._safe_release(cap)
                        cap = None
                        time.sleep(1)
                        continue
                    time.sleep(0.1)  # brief pause before retry, don't spin
            except Exception:
                self._safe_release(cap)
                cap = None
                time.sleep(backoff)
                backoff = min(backoff * 1.5, max_backoff)

    def get_frame(self):
        with self.lock:
            return self.frame, self.frame_time

    def stop(self) -> None:
        self._running = False


class WebDashboard(Thread):
    """Flask-based web dashboard running in a daemon thread.

    Serves:
    - / : HTML dashboard with video feeds, health status, and tracking state
    - /feed/<path> : MJPEG stream (direct buffer or RTSP consumer)
    - /api/health : JSON health data for local system + peers
    - /api/tracking : JSON tracking state (overlay, room roster, attention)

    Args:
        rtsp_server: Direct frame buffer access (GPU server where MLDetect runs in-process)
        feed_uris: Dict of {label: rtsp_uri} for consuming external RTSP streams (Pi4/Pi5)
        feeds: Dict of {label: factory_path} for direct buffer feeds (GPU server)
    """

    def __init__(self, system_name: str, health_monitor=None,
                 rtsp_server=None, feeds: Dict[str, str] = None,
                 feed_uris: Dict[str, str] = None,
                 motion_tracking=None,
                 port: int = 8080) -> None:
        Thread.__init__(self)
        self.daemon = True
        self.__name__ = f"WebDashboard_{system_name}"
        self.logger = setup_logger(self.__name__, LoggingEnums.LOG_LEVEL_INFO.value)
        self.system_name = system_name
        self.health_monitor = health_monitor
        self.rtsp_server = rtsp_server  # direct buffer (GPU server)
        self.motion_tracking = motion_tracking  # MotionTrack for tracking state API
        self.feeds = feeds or {}        # {label: factory_path} for direct buffer
        self.feed_uris = feed_uris or {}  # {label: rtsp_uri} for RTSP consumer
        self.port = port

        # Start RTSP frame grabbers for URI-based feeds
        self._grabbers: Dict[str, RTSPFrameGrabber] = {}
        for label, uri in self.feed_uris.items():
            self._grabbers[label] = RTSPFrameGrabber(uri)
            self.logger.info(f"RTSP grabber started for {label}: {uri}")

        # Build internal feed key map {label: key}
        self._all_feeds: Dict[str, str] = {}
        for label, path in self.feeds.items():
            self._all_feeds[label] = f"buffer{path}"
        for label, uri in self.feed_uris.items():
            self._all_feeds[label] = f"grabber:{label}"

        # Build paired feeds for stacked display (annotated above raw)
        # Match by camera name: "Head (Annotated)" pairs with "Head (Raw)"
        self._feed_pairs = []
        self._unpaired_feeds: Dict[str, str] = {}
        paired_labels = set()

        for buf_label, buf_key in [(l, f"buffer{p}") for l, p in self.feeds.items()]:
            # Extract camera name from label (e.g. "Head" from "Head (Annotated)")
            cam_name = buf_label.split(" (")[0] if " (" in buf_label else buf_label
            # Find matching raw feed
            raw_label = None
            raw_key = None
            for uri_label in self.feed_uris:
                uri_cam = uri_label.split(" (")[0] if " (" in uri_label else uri_label
                if uri_cam == cam_name:
                    raw_label = uri_label
                    raw_key = f"grabber:{uri_label}"
                    break
            if raw_label:
                self._feed_pairs.append({
                    "name": cam_name,
                    "feeds": [
                        {"label": "Annotated", "key": buf_key},
                        {"label": "Raw", "key": raw_key},
                    ]
                })
                paired_labels.add(buf_label)
                paired_labels.add(raw_label)
            else:
                self._unpaired_feeds[buf_label] = buf_key

        # Add any unpaired URI feeds
        for label, uri in self.feed_uris.items():
            if label not in paired_labels:
                self._unpaired_feeds[label] = f"grabber:{label}"

        self._app = self._create_app()

    def _create_app(self) -> Flask:
        app = Flask(__name__)
        dashboard = self

        @app.route('/')
        def index():
            return render_template_string(DASHBOARD_HTML,
                                          system_name=dashboard.system_name,
                                          feeds=dashboard._all_feeds,
                                          feed_pairs=dashboard._feed_pairs,
                                          unpaired_feeds=dashboard._unpaired_feeds)

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

        @app.route('/api/tracking')
        def tracking_api():
            """Return current tracking state for dashboard sidebar."""
            result = {}
            mt = dashboard.motion_tracking
            if mt:
                if hasattr(mt, '_debug_overlay'):
                    result["overlay"] = mt._debug_overlay
                if hasattr(mt, '_room_state') and mt._room_state:
                    result["room"] = mt._room_state.get_room_summary()
                if hasattr(mt, '_attention') and mt._attention:
                    result["attention"] = mt._attention.get_state()
            return jsonify(result)

        return app

    STALE_FRAME_TIMEOUT = 5.0  # seconds before showing "No Signal" overlay

    def _get_frame(self, feed_key: str):
        """Get the latest frame and timestamp for a feed.

        Returns:
            Tuple of (frame, timestamp) or (None, 0).
        """
        if feed_key.startswith("buffer"):
            path = feed_key[len("buffer"):]
            if self.rtsp_server and path in self.rtsp_server.factories:
                rtsp_sys = self.rtsp_server.factories[path]
                with rtsp_sys.data_lock:
                    frame = rtsp_sys.data
                    ts = getattr(rtsp_sys, 'data_time', time.time()) if frame is not None else 0
                    return frame, ts
        elif feed_key.startswith("grabber:"):
            label = feed_key[len("grabber:"):]
            if label in self._grabbers:
                return self._grabbers[label].get_frame()
        return None, 0

    def _make_no_signal_frame(self, width: int = 640, height: int = 480,
                               message: str = "No Signal") -> np.ndarray:
        """Generate a dark frame with a centered message."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        x = (width - text_size[0]) // 2
        y = (height + text_size[1]) // 2
        cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 200), 2)
        return frame

    def _generate_mjpeg(self, feed_key: str):
        """Generator that yields JPEG frames as MJPEG stream.

        Shows "Connecting..." initially, overlays "No Signal" if the frame
        hasn't updated in STALE_FRAME_TIMEOUT seconds.
        """
        blank = self._make_no_signal_frame(message="Connecting...")
        _, jpeg = cv2.imencode('.jpg', blank)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        while True:
            frame, frame_time = self._get_frame(feed_key)
            now = time.time()

            if frame is not None:
                stale = (now - frame_time) > self.STALE_FRAME_TIMEOUT if frame_time > 0 else False
                if stale:
                    # Overlay "No Signal" on the last frame so you can still see it
                    frame = frame.copy()
                    h, w = frame.shape[:2]
                    # Semi-transparent dark bar
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, h // 2 - 30), (w, h // 2 + 30), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                    age = int(now - frame_time)
                    msg = f"No Signal ({age}s)"
                    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    x = (w - text_size[0]) // 2
                    y = h // 2 + text_size[1] // 2
                    cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2)
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                no_sig = self._make_no_signal_frame()
                _, jpeg = cv2.imencode('.jpg', no_sig)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            # ~15 FPS for web stream
            time.sleep(0.066)

    def run(self) -> None:
        """Start the Flask server."""
        self.logger.info(f"Web dashboard starting at http://0.0.0.0:{self.port}")
        self._app.run(host='0.0.0.0', port=self.port, threaded=True, use_reloader=False)
