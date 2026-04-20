#!/usr/bin/env python3
"""Servo angle calibration tool — web-based with servo controls.

Shows the head camera feed, live servo angles, and arrow buttons to
move each servo. Click "Snapshot" when the camera is aimed at your face.
Computes FK world pitch/yaw for calibration.

Usage:
    python Tools/servo_calibrate.py -config glog.conf
    Then open http://<gpu-server-ip>:8090 in a browser.
"""
import argparse
import json
import threading
import time
from configparser import ConfigParser

import cv2
from flask import Flask, Response, render_template_string, jsonify, request

import paho.mqtt.client as mqtt

app = Flask(__name__)

# Global state
servo_state = {
    "head_left_right": {"current": 83.0},
    "head_up_down": {"current": 83.0},
    "body_left_right": {"current": 92.0},
    "body_up_down": {"current": 92.0},
}
lock = threading.Lock()
snapshots = []
cap = None
kin = None
mqtt_client = None

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>GLaDOS Servo Calibration</title>
    <style>
        body { font-family: monospace; background: #1a1a1a; color: #0f0; margin: 20px; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .video { position: relative; }
        .video img { border: 2px solid #0f0; max-width: 100%%; }
        .crosshair { position: absolute; top: 50%%; left: 50%%;
                      width: 40px; height: 40px; margin: -20px 0 0 -20px;
                      border: 1px solid rgba(0,255,0,0.5); border-radius: 50%%; pointer-events: none; }
        .crosshair::before, .crosshair::after {
            content: ''; position: absolute; background: rgba(0,255,0,0.5); }
        .crosshair::before { width: 1px; height: 40px; left: 50%%; }
        .crosshair::after { width: 40px; height: 1px; top: 50%%; }
        .panel { min-width: 380px; }
        .fk { font-size: 22px; color: #ff0; margin: 15px 0; padding: 10px; background: #222; }
        .servo-group { background: #222; padding: 10px; margin: 8px 0; border-left: 3px solid #0f0; }
        .servo-group h3 { margin: 0 0 8px 0; color: #0f0; }
        .servo-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
        .servo-val { font-size: 20px; min-width: 60px; text-align: center; color: #ff0; }
        .btn { font-size: 16px; padding: 8px 12px; background: #333; color: #0f0;
               border: 1px solid #0f0; cursor: pointer; font-family: monospace; min-width: 36px; }
        .btn:hover { background: #0a0; color: #000; }
        .btn:active { background: #0f0; }
        .btn-snap { font-size: 20px; padding: 15px 40px; background: #0a0; color: #000;
                    border: none; cursor: pointer; font-family: monospace; font-weight: bold;
                    margin: 15px 0; display: block; width: 100%%; }
        .btn-snap:hover { background: #0f0; }
        .btn-center { font-size: 14px; padding: 6px 12px; background: #550; color: #ff0;
                      border: 1px solid #ff0; cursor: pointer; font-family: monospace; }
        .btn-center:hover { background: #880; }
        .step-select { background: #222; color: #0f0; border: 1px solid #0f0;
                       font-family: monospace; padding: 4px; font-size: 14px; }
        .snapshots { margin-top: 15px; }
        .snap { background: #222; padding: 8px; margin: 5px 0; border-left: 3px solid #ff0;
                font-size: 13px; }
        h1 { margin-bottom: 5px; }
        .help { color: #888; font-size: 12px; margin-bottom: 15px; }
        .kbd { display: inline-block; background: #333; border: 1px solid #666;
               padding: 2px 6px; border-radius: 3px; font-size: 11px; }
    </style>
</head>
<body>
    <h1>GLaDOS Servo Calibration</h1>
    <p class="help"><b>IMPORTANT:</b> Stop AiServer before using this tool, otherwise tracking overrides servo commands.<br>
    Keyboard: <span class="kbd">W</span>/<span class="kbd">S</span> = Head UD,
    <span class="kbd">A</span>/<span class="kbd">D</span> = Head LR,
    <span class="kbd">I</span>/<span class="kbd">K</span> = Body UD,
    <span class="kbd">J</span>/<span class="kbd">L</span> = Body LR,
    <span class="kbd">Space</span> = Snapshot</p>
    <div class="container">
        <div class="video">
            <img id="feed" src="/feed" width="640" height="480">
            <div class="crosshair"></div>
        </div>
        <div class="panel">
            <div class="fk" id="fk">FK: loading...</div>

            Step size: <select id="step" class="step-select">
                <option value="1">1&deg;</option>
                <option value="2">2&deg;</option>
                <option value="5" selected>5&deg;</option>
                <option value="10">10&deg;</option>
            </select>
            Speed: <select id="speed" class="step-select">
                <option value="1">1 (slow)</option>
                <option value="2">2</option>
                <option value="3" selected>3 (normal)</option>
                <option value="4">4</option>
                <option value="5">5 (fast)</option>
            </select>

            <div class="servo-group">
                <h3>Head Up/Down</h3>
                <div class="servo-row">
                    <button class="btn" onclick="move('head_up_down', -1)">&uarr; Up</button>
                    <span class="servo-val" id="hud">--</span>
                    <button class="btn" onclick="move('head_up_down', 1)">&darr; Down</button>
                    <button class="btn-center" onclick="center('head_up_down', 83)">Center</button>
                </div>
            </div>
            <div class="servo-group">
                <h3>Head Left/Right</h3>
                <div class="servo-row">
                    <button class="btn" onclick="move('head_left_right', -1)">&larr;</button>
                    <span class="servo-val" id="hlr">--</span>
                    <button class="btn" onclick="move('head_left_right', 1)">&rarr;</button>
                    <button class="btn-center" onclick="center('head_left_right', 83)">Center</button>
                </div>
            </div>
            <div class="servo-group">
                <h3>Body Up/Down</h3>
                <div class="servo-row">
                    <button class="btn" onclick="move('body_up_down', -1)">&uarr; Up</button>
                    <span class="servo-val" id="bud">--</span>
                    <button class="btn" onclick="move('body_up_down', 1)">&darr; Down</button>
                    <button class="btn-center" onclick="center('body_up_down', 92)">Center</button>
                </div>
            </div>
            <div class="servo-group">
                <h3>Body Left/Right</h3>
                <div class="servo-row">
                    <button class="btn" onclick="move('body_left_right', -1)">&larr;</button>
                    <span class="servo-val" id="blr">--</span>
                    <button class="btn" onclick="move('body_left_right', 1)">&rarr;</button>
                    <button class="btn-center" onclick="center('body_left_right', 92)">Center</button>
                </div>
            </div>

            <button class="btn-snap" onclick="snapshot()">SNAPSHOT</button>

            <div class="snapshots" id="snapshots"></div>
        </div>
    </div>
    <script>
        function getStep() { return parseInt(document.getElementById('step').value); }
        function getSpeed() { return parseInt(document.getElementById('speed').value); }

        function move(servo, direction) {
            let step = getStep() * direction;
            let speed = getSpeed();
            fetch('/move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({servo: servo, step: step, speed: speed})
            });
        }

        function center(servo, angle) {
            fetch('/move_to', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({servo: servo, angle: angle, speed: getSpeed()})
            });
        }

        function snapshot() {
            fetch('/snapshot', {method: 'POST'}).then(r => r.json()).then(data => {
                let div = document.getElementById('snapshots');
                let snap = document.createElement('div');
                snap.className = 'snap';
                snap.innerHTML = '<b>#' + data.id + '</b> ' + data.time +
                    ' | HLR=' + data.head_lr.toFixed(1) + ' HUD=' + data.head_ud.toFixed(1) +
                    ' BLR=' + data.body_lr.toFixed(1) + ' BUD=' + data.body_ud.toFixed(1) +
                    ' | <b>Yaw=' + data.fk_yaw.toFixed(1) + ' Pitch=' + data.fk_pitch.toFixed(1) + '</b>';
                if (data.mean_pitch !== null) {
                    snap.innerHTML += '<br>Mean pitch across ' + data.count + ' snaps: <b>' +
                        data.mean_pitch.toFixed(1) + '&deg;</b>  bias needed: <b>' +
                        data.bias_needed.toFixed(1) + '&deg;</b>';
                }
                div.insertBefore(snap, div.firstChild);
            });
        }

        function update() {
            fetch('/status').then(r => r.json()).then(data => {
                document.getElementById('hlr').textContent = data.head_lr.toFixed(1);
                document.getElementById('hud').textContent = data.head_ud.toFixed(1);
                document.getElementById('blr').textContent = data.body_lr.toFixed(1);
                document.getElementById('bud').textContent = data.body_ud.toFixed(1);
                document.getElementById('fk').innerHTML =
                    'FK Yaw: <b>' + data.fk_yaw.toFixed(1) + '&deg;</b> &nbsp; ' +
                    'Pitch: <b>' + data.fk_pitch.toFixed(1) + '&deg;</b>';
            });
        }

        // Keyboard controls
        document.addEventListener('keydown', function(e) {
            switch(e.key.toLowerCase()) {
                case 'w': move('head_up_down', -1); break;
                case 's': move('head_up_down', 1); break;
                case 'a': move('head_left_right', -1); break;
                case 'd': move('head_left_right', 1); break;
                case 'i': move('body_up_down', -1); break;
                case 'k': move('body_up_down', 1); break;
                case 'j': move('body_left_right', -1); break;
                case 'l': move('body_left_right', 1); break;
                case ' ': e.preventDefault(); snapshot(); break;
            }
        });

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


@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    servo = data.get("servo")
    step = data.get("step", 5)
    speed = data.get("speed", 3)

    with lock:
        current = servo_state.get(servo, {}).get("current", 90)
    target = current + step

    _send_servo_command(servo, target, speed)
    return jsonify(ok=True, servo=servo, target=round(target, 1))


@app.route('/move_to', methods=['POST'])
def move_to():
    data = request.get_json()
    servo = data.get("servo")
    angle = data.get("angle", 90)
    speed = data.get("speed", 3)

    _send_servo_command(servo, angle, speed)
    return jsonify(ok=True, servo=servo, target=angle)


def _send_servo_command(servo, angle, speed):
    if mqtt_client:
        from uuid import uuid4
        msg = {
            "cmd": "move",
            "location": servo,
            "angle": round(angle, 1),
            "speed": speed,
            "uuid": str(uuid4()),
        }
        mqtt_client.publish("body/servo", json.dumps(msg))


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
        "mean_pitch": None, "bias_needed": None, "count": 0,
    }
    snapshots.append(snap)

    if len(snapshots) >= 2:
        pitches = [s["fk_pitch"] for s in snapshots]
        mean_pitch = sum(pitches) / len(pitches)
        snap["mean_pitch"] = round(mean_pitch, 1)
        snap["bias_needed"] = round(mean_pitch - (-41), 1)
        snap["count"] = len(snapshots)

    print(f"\n  SNAPSHOT #{snap['id']} at {snap['time']}")
    print(f"    Head LR: {hlr:.1f}   Head UD: {hud:.1f}")
    print(f"    Body LR: {blr:.1f}   Body UD: {bud:.1f}")
    print(f"    FK Yaw: {fk_yaw:+.1f}   FK Pitch: {fk_pitch:+.1f}")
    if snap["mean_pitch"] is not None:
        print(f"    Mean pitch ({len(snapshots)} snaps): {snap['mean_pitch']:+.1f}")
        print(f"    Bias needed from -41 raw: {snap['bias_needed']:+.1f}")

    return jsonify(snap)


def main():
    global cap, kin, mqtt_client

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

    # MQTT
    def on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            location = data.get("location")
            results = data.get("results", {})
            if location in servo_state:
                with lock:
                    servo_state[location]["current"] = float(results.get("current", 0))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    mqtt_client = mqtt.Client()
    mqtt_client.on_message = on_message
    mqtt_client.connect(broker_ip, broker_port)
    mqtt_client.subscribe("body/servo/status")
    mqtt_client.loop_start()

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
    if cap.isOpened():
        print("Connected to head camera feed")
    else:
        print("WARNING: Could not open head camera feed — servo controls still work")

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
    print("Use arrows to position head, click Snapshot when face is centered.\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
