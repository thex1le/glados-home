# builtin
import signal
import os
import time
from argparse import ArgumentParser
from configparser import ConfigParser
import sys
from os import path

# glados imports
from glados_modules.MachineVision import MLDetect, GLaDOSServerException
from glados_modules.WhisperXSpeech2Text import AudioServerRX, LocalSTTtx
from glados_modules.GladosEnums import STTEnums, SystemEnums, DashboardEnums, CameraEnum
from glados_modules.HealthMonitor import HealthMonitor
from glados_modules.MqttConsumerModules import SensorTracker
from glados_modules.WebDashboard import WebDashboard

# gladosTTS is an external repo cloned into the project -- its internal imports
# use bare "from utils.tools import ..." so it needs its own dir on sys.path
sys.path.insert(0, path.join(path.dirname(path.abspath(__file__)), 'gladosTTS'))
from glados_tts import engine as glados_voice

if __name__ == "__main__":
    # Handle Ctrl+C cleanly -- set up early so it works during model loading
    def shutdown_handler(signum, frame):
        print("\nShutting down AiServer...")
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    parser = ArgumentParser(description='Evil Home AI Senses Server')
    parser.add_argument('-confif g', type=str, default=1, dest='conf', nargs=1, help='Config File')
    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    config_p = ConfigParser()
    if path.isfile(args.conf[0]) is True:
        config_p.read(args.conf[0], encoding=('utf-8'))
    else:
        raise GLaDOSServerException("Unable to load file {}".format(args.conf[0]))
    # Start the TTS HTTP server early — it doesn't depend on cameras.
    from threading import Thread as _TtsThread
    from flask import Flask as TtsFlask, request as tts_request, send_file as tts_send_file
    import urllib.parse
    import base64
    import shutil
    import logging as _tts_logging

    tts_app = TtsFlask("glados_tts_server")
    TTS_PORT = 8124

    @tts_app.route('/synthesize/', defaults={'text': ''})
    @tts_app.route('/synthesize/<path:text>')
    def tts_synthesize(text):
        if not text:
            return 'No input'
        line = urllib.parse.unquote(tts_request.url[tts_request.url.find('synthesize/') + 11:])
        line = base64.b64decode(line).decode('utf8')
        filename = f"GLaDOS-tts-{line.replace(' ', '-').replace('!', '').replace(',', '')}.wav"
        filename = filename.replace("°c", "degrees celcius")
        cached = os.path.join(os.getcwd(), 'audio', filename)
        if os.path.isfile(cached):
            os.utime(cached, None)
            return tts_send_file(cached)
        key = str(time.time())[7:]
        if glados_voice.glados_tts(line, key):
            tempfile = os.path.join(os.getcwd(), 'audio', f'GLaDOS-tts-temp-output-{key}.wav')
            if len(line) < 200:
                shutil.move(tempfile, cached)
                return tts_send_file(cached)
            else:
                return tts_send_file(tempfile)
        return 'TTS Engine Failed'

    _tts_logging.getLogger('werkzeug').setLevel(_tts_logging.WARNING)
    _tts_thread = _TtsThread(target=lambda: tts_app.run(host="0.0.0.0", port=TTS_PORT), daemon=True)
    _tts_thread.start()
    print(f"\033[92m  TTS server started on 0.0.0.0:{TTS_PORT}\033[0m")

    # start up machine vision
    mv = MLDetect(config_p)

    # Wait for Pi4/Pi5 cameras to signal ready over MQTT before connecting
    # RTSP streams. Without this, MLDetect and the dashboard retry against
    # offline cameras, flooding logs and wasting GPU resources.
    from threading import Event
    from json import loads as _json_loads
    from glados_modules.MqttConnector import MQTTClient as _WaitClient
    from glados_modules.GladosEnums import MQTTEnums as _MQTTEnums

    mqtt_conf = config_p[SystemEnums.CONFIG_HEAD_MQTT.value]
    _wait_ip = mqtt_conf[SystemEnums.MQTT_SERVER_IP.value]
    _wait_port = int(mqtt_conf[SystemEnums.MQTT_PORT.value])

    _ready_systems = set()
    _expected_systems = {"body_server", "glados_main"}
    _all_ready = Event()

    class _CameraWaiter(_WaitClient):
        def __init__(self):
            self.topic_handler = {
                _MQTTEnums.CAMERA_READY_TOPIC.value: self._on_ready
            }
            super().__init__(ip=_wait_ip, port=_wait_port)

        def _on_ready(self, msg):
            try:
                data = _json_loads(msg.payload.decode())
                system = data.get("system", "unknown")
                cameras = data.get("cameras", [])
                _ready_systems.add(system)
                print(f"\033[92m  ✓ {system} cameras ready: {cameras}\033[0m")
                if _ready_systems >= _expected_systems:
                    _all_ready.set()
            except Exception:
                pass

    _waiter = _CameraWaiter()
    _timeout = 120  # seconds — generous fallback if a Pi is down

    print("\033[94m" + "=" * 60)
    print("  Waiting for Pi4 and Pi5 to signal cameras ready...")
    print("  Start BodyServer.py on Pi4 and GLaDOS.py on Pi5 now.")
    print(f"  (timeout: {_timeout}s)")
    print("=" * 60 + "\033[0m")

    if _all_ready.wait(timeout=_timeout):
        print("\033[92m  All camera systems ready — starting vision pipeline.\033[0m")
    else:
        missing = _expected_systems - _ready_systems
        print(f"\033[93m  Timeout after {_timeout}s — starting without: {missing}\033[0m")

    mv.start()
    # start the audio receive server
    broker = AudioServerRX.broker_tuple
    stt_conf = config_p[STTEnums.CONFIG_HEAD_STT.value]
    audio_b = broker(stt_conf[STTEnums.STT_SERVER_IP.value], int(stt_conf[STTEnums.STT_SERVER_PORT.value]))
    # reuse ip port broker tuple
    mqtt_b = broker(_wait_ip, _wait_port)
    lstt_tx = LocalSTTtx(mqtt_b)
    stt_audio_rx = AudioServerRX(audio_b, callback=lstt_tx.process_audio)
    stt_audio_rx.start()
    # Start health monitoring
    health = HealthMonitor(broker=mqtt_b, system_name="ai_server")
    health.register("MLDetect", mv)
    health.register("AudioRX", stt_audio_rx)
    health.start()
    # Start web dashboard (GPU: direct buffer for annotated + RTSP consumer for raw)
    # Build annotated feeds from direct frame buffer
    ai_feeds = {}
    for cam_name in mv.cam_configs.keys():
        label = cam_name.replace("camera_", "").replace("_", " ").title()
        ai_feeds[f"{label} (Annotated)"] = f"/{cam_name}"

    # Build raw camera feeds from RTSP URIs in config (Pi4/Pi5 cameras)
    cam_conf = config_p[CameraEnum.CONFIG_HEAD.value]
    raw_feeds = {}
    for cam_key, factory_key, ip_key, port_key in [
        (CameraEnum.CAMERA_HEAD.value, CameraEnum.CAMERA_HEAD_FACTORY.value,
         CameraEnum.CAMERA_HEAD_RTSP_IP.value, CameraEnum.CAMERA_HEAD_PORT.value),
        (CameraEnum.CAMERA_LEFT.value, CameraEnum.CAMERA_LEFT_FACTORY.value,
         CameraEnum.CAMERA_LEFT_RTSP_IP.value, CameraEnum.CAMERA_LEFT_PORT.value),
        (CameraEnum.CAMERA_RIGHT.value, CameraEnum.CAMERA_RIGHT_FACTORY.value,
         CameraEnum.CAMERA_RIGHT_RTSP_IP.value, CameraEnum.CAMERA_RIGHT_PORT.value),
    ]:
        cam_ip = cam_conf.get(ip_key, "")
        cam_port = cam_conf.get(port_key, "8554")
        cam_factory = cam_conf.get(factory_key, cam_key)
        if cam_ip:
            label = cam_key.replace("camera_", "").replace("_", " ").title()
            raw_feeds[f"{label} (Raw)"] = f"rtsp://{cam_ip}:{cam_port}/{cam_factory}"

    dash_port = int(config_p.get(DashboardEnums.CONFIG_HEAD.value,
                                  DashboardEnums.DASHBOARD_PORT.value,
                                  fallback=DashboardEnums.DEFAULT_PORT.value))
    # Subscribe to sensor MQTT topics so dashboard can display Pi sensors + camera FPS
    sensor_tracker = SensorTracker(mqtt_b)
    dashboard = WebDashboard(
        system_name="ai_server",
        health_monitor=health,
        sensor_tracker=sensor_tracker,
        rtsp_server=mv.rtsp,
        feeds=ai_feeds,
        feed_uris=raw_feeds,
        motion_tracking=mv.motion_tracking,
        port=dash_port
    )
    dashboard.start()

    # Keep main thread alive — TTS runs in a daemon thread, everything else
    # is in its own thread/process.  Block until interrupted.
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
