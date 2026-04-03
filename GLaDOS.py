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
from glados_modules.ChatGPTConnector import GladosGPT
from glados_modules.Speech2Text import GladosSTT
from glados_modules.GLaDOSLocal import GladosLocal
from glados_modules.CameraModule import Camera, CameraWatchdog
from glados_modules.GladosEnums import CameraEnum, SystemEnums, DashboardEnums
from glados_modules.BodyControlModules import IMU
from glados_modules.HealthMonitor import HealthMonitor
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
    # start the IMU to track movement
    mqtt_c = configp[SystemEnums.CONFIG_HEAD_MQTT.value]
    # pull the MQTT server connection info from the config
    mqtt_broker = IMU.broker_tuple(mqtt_c[SystemEnums.MQTT_SERVER_IP.value], mqtt_c[SystemEnums.MQTT_PORT.value])
    # pass it to the imu body module and start the IMU polling server
    imu = IMU(broker=mqtt_broker)
    imu.start()
    gl = GladosLocal(configp, GladosGPT)
    gl.start()
    try:
        gl.speak("Oh Its you! , , Its been a long time...")
    except Exception:
        print("TTS unavailable at startup — continuing without greeting")
    gstt = GladosSTT(configp, gl)
    gstt.start()
    local_commands = (gl.get_temp, gl.fuck_you, gl.timer, gl.set_volume)
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
    # Start health monitoring
    health = HealthMonitor(broker=mqtt_broker, system_name="glados_main")
    health.register("IMU", imu)
    health.register("GladosLocal", gl)
    health.register("STT", gstt)
    health.register("left_camera", left_camera)
    health.register("right_camera", right_camera)
    health.start()
    # Start web dashboard (Pi5: consumes RTSP since cameras are separate processes)
    left_port = configp[CameraEnum.CONFIG_HEAD.value][CameraEnum.CAMERA_LEFT_PORT.value]
    right_port = configp[CameraEnum.CONFIG_HEAD.value][CameraEnum.CAMERA_RIGHT_PORT.value]
    dash_port = int(configp.get(DashboardEnums.CONFIG_HEAD.value,
                                 DashboardEnums.DASHBOARD_PORT.value,
                                 fallback=DashboardEnums.DEFAULT_PORT.value))
    dashboard = WebDashboard(
        system_name="glados_main",
        health_monitor=health,
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
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    while True:
        prompt = gstt.get_text()
        if prompt is not None:
            cmd_bool = False
            # check for local commands
            # TODO load commands from config?
            for cmd in local_commands:
                cmd_bool = cmd(user_prompt=prompt)
                if cmd_bool is True:
                    # break the for loop
                    break
            if cmd_bool is True:
                # skip the rest on the while loop
                continue
            gladosgpt = GladosGPT(configp, prompt)
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
