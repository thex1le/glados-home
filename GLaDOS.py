# builtin
import random
import time
import signal
import os
from os import path
import argparse
import sys
import configparser

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.Speech2Text import GladosSTT
from glados_modules.GLaDOSLocal import GladosLocal
from glados_modules.CameraModule import Camera, CameraWatchdog
from glados_modules.GladosEnums import CameraEnum, SystemEnums, DashboardEnums, SocTempEnums
from glados_modules.BodyControlModules import IMU, SocTemp, GladosLCD
from glados_modules.HealthMonitor import HealthMonitor
from glados_modules.MqttConsumerModules import SensorTracker
from glados_modules.WebDashboard import WebDashboard


class GladosException(Exception):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evil Home AI')
    parser.add_argument('-config', type=str, default=1, dest='conf', nargs=1, help='Config File')
    try: 
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    configp = configparser.ConfigParser()
    if path.isfile(args.conf[0]) is True:
        configp.read(args.conf[0])
    else:
        raise GladosException("Unable to load file {}".format(args.conf[0]))

    logger = setup_logger(name="GLaDOS_main")
    # Select LLM provider from config (default: openai for backward compatibility)
    llm_provider = configp.get(SystemEnums.CONFIG_HEAD_DEFAULT.value,
                                SystemEnums.LLM_PROVIDER.value, fallback="openai").strip().lower()
    if llm_provider == "claude":
        from glados_modules.ClaudeConnector import GladosClaude as LLMConnector
    else:
        from glados_modules.ChatGPTConnector import GladosGPT as LLMConnector

    # start the IMU to track movement
    mqtt_c = configp[SystemEnums.CONFIG_HEAD_MQTT.value]
    # pull the MQTT server connection info from the config
    mqtt_broker = IMU.broker_tuple(mqtt_c[SystemEnums.MQTT_SERVER_IP.value], mqtt_c[SystemEnums.MQTT_PORT.value])
    # pass it to the imu body module and start the IMU polling server
    imu = None
    try:
        imu = IMU(broker=mqtt_broker)
        imu.start()
    except (ValueError, OSError, TimeoutError) as e:
        logger.error(f"IMU initialization failed: {e} — continuing without IMU")
    # Start left LCD display
    animation_path = configp[SystemEnums.CONFIG_HEAD_DEFAULT.value][SystemEnums.APERTURE_ANIMATION.value]
    left_lcd = GladosLCD(broker=mqtt_broker, location=SystemEnums.LEFT_LCD.value,
                         animation_path=animation_path)
    left_lcd.start()

    gl = GladosLocal(configp, LLMConnector)
    gl.start()
    # Greeting in background thread — retries if TTS isn't ready yet
    def _startup_greeting():
        for attempt in range(3):
            try:
                gl.speak("Oh Its you! , , Its been a long time...")
                return
            except Exception:
                if attempt < 2:
                    time.sleep(15)
        print("TTS unavailable — startup greeting skipped")
    from threading import Thread as _Thread
    _Thread(target=_startup_greeting, daemon=True).start()
    gstt = GladosSTT(configp, gl)
    gstt.start()
    local_commands = gl.get_local_commands()
    left_camera_location = configp[CameraEnum.CONFIG_HEAD.value][CameraEnum.CAMERA_LEFT_FACTORY.value]
    right_camera_location = configp[CameraEnum.CONFIG_HEAD.value][CameraEnum.CAMERA_RIGHT_FACTORY.value]
    port = int(configp[CameraEnum.CONFIG_HEAD.value][CameraEnum.CAMERA_LEFT_PORT.value])
    left_camera = Camera(configfile=configp, location=left_camera_location, rtspport=port)
    port = int(configp[CameraEnum.CONFIG_HEAD.value][CameraEnum.CAMERA_RIGHT_PORT.value])
    right_camera = Camera(configfile=configp, location=right_camera_location, rtspport=port)
    left_camera.start()
    # give time for first camera to start before we spin up the second
    time.sleep(5)
    right_camera.start()
    # Signal AI server that cameras are ready for RTSP connection
    from glados_modules.MqttConnector import MQTTClient as _ReadyClient
    from glados_modules.GladosEnums import MQTTEnums as _MQTTEnums
    _ready = _ReadyClient(ip=mqtt_broker.ip, port=mqtt_broker.port)
    _ready.send_command(
        {"system": "glados_main", "cameras": [left_camera_location, right_camera_location]},
        _MQTTEnums.CAMERA_READY_TOPIC.value)
    print(f"Published camera ready signal for left + right cameras")
    # Start health monitoring
    health = HealthMonitor(broker=mqtt_broker, system_name="glados_main")
    if imu:
        health.register("IMU", imu)
    health.register("left_lcd", left_lcd)
    health.register("GladosLocal", gl)
    health.register("STT", gstt)
    health.register("left_camera", left_camera)
    health.register("right_camera", right_camera)
    soc_poll = float(configp.get(SocTempEnums.CONFIG_HEAD.value,
                                  SocTempEnums.POLL_INTERVAL.value,
                                  fallback=SocTempEnums.DEFAULT_POLL_INTERVAL.value))
    soc_temp = SocTemp(broker=mqtt_broker, poll_interval=soc_poll)
    soc_temp.start()
    health.register("soc_temp", soc_temp)
    health.start()
    # Start web dashboard (Pi5: consumes RTSP since cameras are separate processes)
    left_port = configp[CameraEnum.CONFIG_HEAD.value][CameraEnum.CAMERA_LEFT_PORT.value]
    right_port = configp[CameraEnum.CONFIG_HEAD.value][CameraEnum.CAMERA_RIGHT_PORT.value]
    dash_port = int(configp.get(DashboardEnums.CONFIG_HEAD.value,
                                 DashboardEnums.DASHBOARD_PORT.value,
                                 fallback=DashboardEnums.DEFAULT_PORT.value))
    sensor_tracker = SensorTracker(mqtt_broker)
    dashboard = WebDashboard(
        system_name="glados_main",
        health_monitor=health,
        sensor_tracker=sensor_tracker,
        feed_uris={
            "Left Eye (Raw)": f"rtsp://localhost:{left_port}/{left_camera_location}",
            "Right Eye (Raw)": f"rtsp://localhost:{right_port}/{right_camera_location}",
        },
        port=dash_port
    )
    dashboard.start()

    # Camera watchdog: respawn camera processes if they die
    cam_watchdog = CameraWatchdog(health_monitor=health)
    cam_watchdog.add_camera("left_camera", left_camera)
    cam_watchdog.add_camera("right_camera", right_camera)
    cam_watchdog.start()

    def shutdown_handler(signum, frame):
        print("\nShutting down GLaDOS...")
        cam_watchdog.shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    while True:
        prompt = gstt.get_text()
        if prompt is not None:
            cmd_bool = False
            # check for local commands (registered in GLaDOSLocal.get_local_commands)
            for cmd in local_commands:
                cmd_bool = cmd(user_prompt=prompt)
                if cmd_bool is True:
                    # break the for loop
                    break
            if cmd_bool is True:
                # skip the rest on the while loop
                continue
            # Track interaction frequency for pestering detection
            gl.register_interaction()
            # Check for compliments (calms mood)
            gl.detect_compliment(prompt)
            gladosgpt = LLMConnector(configp, prompt)
            gladosgpt.add_prompt(gl.get_seen_prompt())
            gladosgpt.start()
            time.sleep(0.2)
            while gladosgpt.real_response is None:
                gl.random_processing()
                time.sleep(0.3)
                rfunc = random.choice((gl.random_processing,
                                       gl.random_insult))
                rfunc()
            time.sleep(0.2)
            gl.speak(gladosgpt.real_response)
