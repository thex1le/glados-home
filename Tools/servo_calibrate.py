#!/usr/bin/env python3
"""Servo angle calibration tool — web-based.

Shows the head camera feed and live servo angles in a browser.
Click "Snapshot" when the camera is aimed at your face.
Computes FK world pitch/yaw for calibration.

Usage:
    python Tools/servo_calibrate.py -config glog.conf
    Then open http://<gpu-server-ip>:8090 in a browser.
"""
import argparse
import json
import sys
import threading
import time
from configparser import ConfigParser

import cv2
from flask import Flask, Response, render_template_string, jsonify

import paho.mqtt.client as mqtt

app = Flask(__name__)

# Global state
servo_state = {
    "head_left_right": {"current": 0.0, "target": 0.0, "velocity": 0.0},
    "head_up_down": {"current": 0.0, "target": 0.0, "velocity": 0.0},
    "body_left_right": {"current": 0.0, "target": 0.0, "velocity": 0.0},
    "body_up_down": {"current": 0.0, "target": 0.0, "velocity": 0.0},
}
lock = threading.Lock()
snapshots = []
cap = None
kin = None

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>GLaDOS Servo Calibration</title>
    <style>
        body { font-family: monospace; background: #1a1a1a; color: #0f0; margin: 20px; }
        .container { display: flex; gap: 20px; }
        .video { position: relative; }
        .video img { border: 2px solid #0f0; }
        .crosshair { position: absolute; top: 50%%; left: 50%%;
                      width: 40px; height: 40px; margin: -20px 0 0 -20px;
                      border: 1px solid rgba(0,255,0,0.5); border-radius: 50%%; }
        .crosshair::before, .crosshair::after {
            content: ''; position: absolute; background: rgba(0,255,0,0.5); }
        .crosshair::before { width: 1px; height: 40px; left: 50%%; }
        .crosshair::after { width: 40px; height: 1px; top: 50%%; }
        .panel { min-width: 350px; }
        .angles { font-size: 18px; line-height: 2; }
        .fk { font-size: 22px; color: #ff0; margin: 15px 0; }
        button { font-size: 20px; padding: 15px 40px; background: #0a0;
                 color: #000; border: none; cursor: pointer; font-family: monospace;
                 font-weight: bold; margin: 10px 0; }
        button:hover { background: #0f0; }
        .snapshots { margin-top: 20px; }
        .snap { background: #222; padding: 10px; margin: 5px 0; border-left: 3px solid #0f0; }
        .snap img { max-width: 200px; margin-top: 5px; }
        h2 { color: #0f0; border-bottom: 1px solid #333; padding-bottom: 5px; }
    </style>
</head>
<body>
    <h1>GLaDOS Servo Calibration</h1>
    <p>Position the head so your face is centered on the crosshair. Click Snapshot.</p>
    <div class="container">
        <div class="video">
            <img id="feed" src="/feed" width="640" height="480">
            <div class="crosshair"></div>
        </div>
        <div class="panel">
            <div class="angles" id="angles">Loading...</div>
            <div class="fk" id="fk"></div>
            <button onclick="snapshot()">SNAPSHOT</button>
            <div class="snapshots" id="snapshots"></div>
        </div>
    </div>
    <script>
        function update() {
            fetch('/status').then(r => r.json()).then(data => {
                document.getElementById('angles').innerHTML =
                    'Head LR: ' + data.head_lr.toFixed(1) + '<br>' +
                    'Head UD: ' + data.head_ud.toFixed(1) + '<br>' +
                    'Body LR: ' + data.body_lr.toFixed(1) + '<br>' +
                    'Body UD: ' + data.body_ud.toFixed(1);
                document.getElementById('fk').innerHTML =
                    'FK Yaw: ' + data.fk_yaw.toFixed(1) +
                    '&deg; &nbsp; Pitch: ' + data.fk_pitch.toFixed(1) + '&deg;';
            });
        }
        function snapshot() {
            fetch('/snapshot', {method: 'POST'}).then(r => r.json()).then(data => {
                let div = document.getElementById('snapshots');
                let snap = document.createElement('div');
                snap.className = 'snap';
                snap.innerHTML = '<b>#' + data.id + '</b> ' + data.time + '<br>' +
                    'HLR=' + data.head_lr.toFixed(1) + ' HUD=' + data.head_ud.toFixed(1) +
                    ' BLR=' + data.body_lr.toFixed(1) + ' BUD=' + data.body_ud.toFixed(1) + '<br>' +
                    '<b>FK Yaw=' + data.fk_yaw.toFixed(1) + ' Pitch=' + data.fk_pitch.toFixed(1) + '</b>';
                div.insertBefore(snap, div.firstChild);
            });
        }
        setInterval(update, 200);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/feed')
def feed():
    def gen():
        while True:
            if cap is None or not cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    with lock:
        hlr = servo_state["head_left_right"]["current"]
        hud = servo_state["head_up_down"]["current"]
        blr = servo_state["body_left_right"]["current"]
        bud = servo_state["body_up_down"]["current"]

    fk_yaw, fk_pitch = 0.0, 0.0
    if kin:
        angles = {"head_left_right": hlr, "head_up_down": hud,
                  "body_left_right": blr, "body_up_down": bud}
        fk_yaw, fk_pitch = kin.forward_kinematics(angles)

    return jsonify(head_lr=hlr, head_ud=hud, body_lr=blr, body_ud=bud,
                   fk_yaw=round(fk_yaw, 2), fk_pitch=round(fk_pitch, 2))


@app.route('/snapshot', methods=['POST'])
def snapshot():
    with lock:
        hlr = servo_state["head_left_right"]["current"]
        hud = servo_state["head_up_down"]["current"]
        blr = servo_state["body_left_right"]["current"]
        bud = servo_state["body_up_down"]["current"]

    fk_yaw, fk_pitch = 0.0, 0.0
    if kin:
        angles = {"head_left_right": hlr, "head_up_down": hud,
                  "body_left_right": blr, "body_up_down": bud}
        fk_yaw, fk_pitch = kin.forward_kinematics(angles)

    snap = {
        "id": len(snapshots) + 1,
        "time": time.strftime("%H:%M:%S"),
        "head_lr": round(hlr, 1), "head_ud": round(hud, 1),
        "body_lr": round(blr, 1), "body_ud": round(bud, 1),
        "fk_yaw": round(fk_yaw, 2), "fk_pitch": round(fk_pitch, 2),
    }
    snapshots.append(snap)

    print(f"\n  SNAPSHOT #{snap['id']} at {snap['time']}")
    print(f"    Head LR: {hlr:.1f}   Head UD: {hud:.1f}")
    print(f"    Body LR: {blr:.1f}   Body UD: {bud:.1f}")
    print(f"    FK Yaw: {fk_yaw:+.1f}   FK Pitch: {fk_pitch:+.1f}")

    if len(snapshots) >= 2:
        pitches = [s["fk_pitch"] for s in snapshots]
        mean_pitch = sum(pitches) / len(pitches)
        print(f"    Mean FK Pitch across {len(snapshots)} snaps: {mean_pitch:+.1f}")
        print(f"    Required face_bias from -41 raw: {mean_pitch - (-41):+.1f}")

    return jsonify(snap)


def main():
    global cap, kin

    parser = argparse.ArgumentParser(description="Servo calibration web tool")
    parser.add_argument("-config", required=True, help="Path to glog.conf")
    parser.add_argument("-port", type=int, default=8090, help="Web server port")
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    broker_ip = config.get("MQTT", "mqtt_server_ip")
    broker_port = config.getint("MQTT", "mqtt_port")
    rtsp_ip = config.get("CAMERAS", "camera_head_rtsp_ip")
    rtsp_port = config.get("RTSP", "rtsp_port")

    # MQTT for servo status
    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            location = data.get("location")
            results = data.get("results", {})
            if location in servo_state:
                with lock:
                    servo_state[location]["current"] = float(results.get("current", 0))
                    servo_state[location]["target"] = float(results.get("last", 0))
                    servo_state[location]["velocity"] = float(results.get("velocity", 0))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    client = mqtt.Client()
    client.on_message = on_message
    client.connect(broker_ip, broker_port)
    client.subscribe("body/servo/status")
    client.loop_start()

    # RTSP feed
    rtsp_uri = f"rtsp://{rtsp_ip}:{rtsp_port}/camera_head"
    print(f"Connecting to head camera: {rtsp_uri}")
    cap = cv2.VideoCapture(
        f"rtspsrc location={rtsp_uri} latency=200 ! "
        f"rtph264depay ! h264parse ! avdec_h264 ! "
        f"videoconvert ! appsink drop=true max-buffers=1 sync=false",
        cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture(rtsp_uri, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"WARNING: Could not open {rtsp_uri} — feed will be unavailable")
        print("Servo angles still work via MQTT. Open the page and use snapshots.")

    # FK
    try:
        from glados_modules.RobotKinematics import RobotKinematics
        head_min_max = config.get("SERVOS", "head_min_max_center").split(",")
        neck_min_max = config.get("SERVOS", "neck_min_max_center").split(",")
        middles = {"body_left_right": float(neck_min_max[2]),
                   "body_up_down": float(neck_min_max[2]),
                   "head_left_right": float(head_min_max[2]),
                   "head_up_down": float(head_min_max[2])}
        mins = {"body_left_right": float(neck_min_max[1]),
                "body_up_down": float(neck_min_max[1]),
                "head_left_right": float(head_min_max[1]),
                "head_up_down": float(head_min_max[1])}
        maxs = {"body_left_right": float(neck_min_max[0]),
                "body_up_down": float(neck_min_max[0]),
                "head_left_right": float(head_min_max[0]),
                "head_up_down": float(head_min_max[0])}
        kin = RobotKinematics(middles, mins, maxs)
        print("FK computation ready")
    except Exception as e:
        print(f"WARNING: FK unavailable: {e}")

    print(f"\nOpen http://<this-server>:{args.port} in a browser")
    print("Position head manually, click Snapshot when face is centered.\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
