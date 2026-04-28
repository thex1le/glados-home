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
from glados_modules.GladosEnums import (STTEnums, SystemEnums, DashboardEnums,
                                        CameraEnum, FeatureToggles)
from glados_modules.HealthMonitor import HealthMonitor
from glados_modules.MqttConsumerModules import SensorTracker
from glados_modules.SceneDescriber import SceneDescriber
from glados_modules.TTSServer import TTSServer
from glados_modules.WebDashboard import WebDashboard

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
    # Step 8a-1: TTSServer is the pluggable Flask TTS at :8124. Backend
    # selection lives in [FEATURES] tts_engine; legacy 'tacotron' default
    # preserves existing behavior, 'glados' / 'kokoro' opt into the engine
    # package's better voices.
    tts_server = TTSServer(config_p)
    tts_server.start()
    print(f"\033[92m  TTS server started on 0.0.0.0:{tts_server.port} "
           f"(engine={tts_server.engine_kind})\033[0m")

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
    # SceneDescriber (FastVLM) — periodic + on-demand scene descriptions over
    # MQTT for the Pi 5 brain's ContextBuilder. Fail-soft: if FastVLM or RTSP
    # init blows up, the rest of the AI server keeps running (matches the IMU
    # pattern in GLaDOS.py).
    scene_describer = None
    try:
        scene_describer = SceneDescriber(config_p, camera_name="head")
        scene_describer.start()
    except Exception as e:
        print(f"\033[93m  SceneDescriber init failed: {e} — continuing without it\033[0m")
    # start the audio receive server
    broker = AudioServerRX.broker_tuple
    stt_conf = config_p[STTEnums.CONFIG_HEAD_STT.value]
    audio_b = broker(stt_conf[STTEnums.STT_SERVER_IP.value], int(stt_conf[STTEnums.STT_SERVER_PORT.value]))
    # reuse ip port broker tuple
    mqtt_b = broker(_wait_ip, _wait_port)
    # Step 8a-2: pluggable ASR backend. Both implementations expose a
    # process_audio(bytes) callback and publish on STT_RESULTS_MQTT_TOPIC,
    # so downstream consumers (Pi 5 brain, dashboard) see no difference.
    asr_engine = config_p.get(FeatureToggles.CONFIG_HEAD.value,
                               FeatureToggles.ASR_ENGINE.value,
                               fallback=FeatureToggles.DEFAULT_ASR_ENGINE.value
                               ).strip().lower()
    if asr_engine == FeatureToggles.ASR_ENGINE_PARAKEET.value:
        from glados_modules.ParakeetSTT import ParakeetSTT
        lstt_tx = ParakeetSTT(mqtt_b, config_p)
        print(f"\033[92m  ASR engine: Parakeet\033[0m")
    else:
        lstt_tx = LocalSTTtx(mqtt_b)
        print(f"\033[92m  ASR engine: WhisperX\033[0m")
    stt_audio_rx = AudioServerRX(audio_b, callback=lstt_tx.process_audio)
    stt_audio_rx.start()
    # Start health monitoring
    health = HealthMonitor(broker=mqtt_b, system_name="ai_server")
    health.register("MLDetect", mv)
    health.register("AudioRX", stt_audio_rx)
    health.register("TTSServer", tts_server)
    if scene_describer is not None:
        health.register("SceneDescriber", scene_describer)
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
