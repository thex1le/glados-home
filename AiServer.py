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
    # start up machine vision
    mv = MLDetect(config_p)
    mv.start()
    # start the audio receive server
    broker = AudioServerRX.broker_tuple
    stt_conf = config_p[STTEnums.CONFIG_HEAD_STT.value]
    mqtt_conf = config_p[SystemEnums.CONFIG_HEAD_MQTT.value]
    audio_b = broker(stt_conf[STTEnums.STT_SERVER_IP.value], int(stt_conf[STTEnums.STT_SERVER_PORT.value]))
    # reuse ip port broker tuple
    mqtt_b = broker(mqtt_conf[SystemEnums.MQTT_SERVER_IP.value], int(mqtt_conf[SystemEnums.MQTT_PORT.value]))
    lstt_tx = LocalSTTtx(mqtt_b)
    stt_audio_rx = AudioServerRX(audio_b, callback=lstt_tx.process_audio)
    stt_audio_rx.start()
    # Start health monitoring
    health = HealthMonitor(broker=mqtt_b, system_name="ai_server")
    health.register("MLDetect", mv)
    health.register("AudioRX", stt_audio_rx)
    health.start()
    # Start web dashboard (GPU: direct buffer for both annotated + raw feeds)
    # MachineVision stores both annotated and raw frames in RTSPServer buffer,
    # so we use direct buffer access for all feeds — no RTSP round-trip needed.
    ai_feeds = {}
    for cam_name in mv.cam_configs.keys():
        label = cam_name.replace("camera_", "").replace("_", " ").title()
        ai_feeds[f"{label} (Annotated)"] = f"/{cam_name}"
        ai_feeds[f"{label} (Raw)"] = f"/{cam_name}_raw"

    dash_port = int(config_p.get(DashboardEnums.CONFIG_HEAD.value,
                                  DashboardEnums.DASHBOARD_PORT.value,
                                  fallback=DashboardEnums.DEFAULT_PORT.value))
    dashboard = WebDashboard(
        system_name="ai_server",
        health_monitor=health,
        rtsp_server=mv.rtsp,
        feeds=ai_feeds,
        motion_tracking=mv.motion_tracking,
        port=dash_port
    )
    dashboard.start()
    # Start the TTS HTTP server. The glados_tts engine loads models at import
    # time and exposes glados_tts() but its Flask server is inside __main__.
    # We recreate the server here so it runs in-process.
    from flask import Flask as TtsFlask, request as tts_request, send_file as tts_send_file
    import urllib.parse
    import base64
    import shutil

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

    import logging as _tts_logging
    _tts_logging.getLogger('werkzeug').setLevel(_tts_logging.WARNING)
    tts_app.run(host="0.0.0.0", port=TTS_PORT)
