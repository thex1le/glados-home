# builtin
import signal
import os
from time import sleep
from argparse import ArgumentParser
from configparser import ConfigParser
import sys
from os import path
from collections import namedtuple

# 3rd party
from adafruit_servokit import ServoKit

# glados imports
from glados_modules.BodyControlModules import Gservo, LedHead, LedShoulders, GladosLCD, SocTemp
from glados_modules.CameraModule import GLaDOSServerException, Camera, CameraWatchdog
from glados_modules.GladosEnums import CameraEnum, ServoEnum, SystemEnums, DashboardEnums, SocTempEnums, MQTTEnums
from glados_modules.HealthMonitor import HealthMonitor
from glados_modules.LatencyResponder import LatencyResponder
from glados_modules.MqttConsumerModules import SensorTracker
from glados_modules.WebDashboard import WebDashboard


if __name__ == "__main__":
    parser = ArgumentParser(description='Evil Home AI Senses Server')
    parser.add_argument('-config', type=str, default=1, dest='conf', nargs=1, help='Config File')
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
        config_p.read(args.conf[0])
    else:
        raise GLaDOSServerException("Unable to load file {}".format(args.conf[0]))

    animation_path = path.abspath(
        config_p[SystemEnums.CONFIG_HEAD_DEFAULT.value][SystemEnums.APERTURE_ANIMATION.value])
    max_min_tuple = namedtuple("max_min", ['max', 'min'])
    max_min_center_tuple = namedtuple("max_min_center", ['max', 'min', 'center'])
    Mqtt_tuple = namedtuple("service_address", ["ip", "port"])
    mqtt_connect = Mqtt_tuple(config_p[SystemEnums.CONFIG_HEAD_MQTT.value][SystemEnums.MQTT_SERVER_IP.value],
                              int(config_p[SystemEnums.CONFIG_HEAD_MQTT.value][SystemEnums.MQTT_PORT.value]))
    sech = config_p[ServoEnum.CONFIG_HEAD.value]
    pulse_90 = sech[ServoEnum.SERVO_MG90D_PULSE.value].split(',')
    pulse_92 = sech[ServoEnum.SERVO_MG92B_PULSE.value].split(',')
    pulse_3508 = sech[ServoEnum.SERVO_GS3508MG_PULSE.value].split(',')
    default = sech[ServoEnum.DEFAULT_MAX_MIN_CENTER.value].split(',')
    head_min_max = sech[ServoEnum.HEAD_MIN_MAX_CENTER.value].split(',')
    neck_min_max = sech[ServoEnum.NECK_MIN_MAX_CENTER.value].split(',')
    mg92d_speed = float(sech[ServoEnum.SERVO_MG92B_SPEED.value])
    mg90d_speed = float(sech[ServoEnum.SERVO_MG90D_SPEED.value])
    gs3508mg_speed = float(sech[ServoEnum.SERVO_GS3508MG_SPEED.value])
    mg90d_pulse = max_min_tuple(int(pulse_90[0]), int(pulse_90[1]))
    mg92b_pulse = max_min_tuple(int(pulse_92[0]), int(pulse_92[1]))
    gs3508_pulse = max_min_tuple(int(pulse_3508[0]), int(pulse_3508[1]))
    default_angle = max_min_center_tuple(int(default[0]), int(default[1]), int(default[2]))
    head_angle = max_min_center_tuple(int(head_min_max[0]), int(head_min_max[1]), int(head_min_max[2]))
    neck_angle = max_min_center_tuple(int(neck_min_max[0]), int(neck_min_max[1]), int(neck_min_max[2]))
    # Initialize PCA9685 boards (supports multiple boards at different I2C addresses)
    pca_addrs = [int(a.strip(), 0) for a in sech.get(
        ServoEnum.PCA9685_ADDRESSES.value, ServoEnum.PCA9685_DEFAULT_ADDRESSES.value).split(',')]
    kits = [ServoKit(channels=16, address=addr) for addr in pca_addrs]

    def get_servo(board_channel_key, fallback_board=0, fallback_channel=0):
        """Get a servo object from board,channel config string."""
        raw = sech.get(board_channel_key, f"{fallback_board},{fallback_channel}")
        parts = raw.split(',')
        board_idx, channel = int(parts[0]), int(parts[1])
        return kits[board_idx].servo[channel], pca_addrs[board_idx]

    body_lr_servo, _ = get_servo(ServoEnum.BODY_LR_BOARD_CHANNEL.value, 0, 0)
    body_ud_servo, _ = get_servo(ServoEnum.BODY_UD_BOARD_CHANNEL.value, 0, 1)
    head_lr_servo, _ = get_servo(ServoEnum.HEAD_LR_BOARD_CHANNEL.value, 0, 2)
    head_ud_servo, _ = get_servo(ServoEnum.HEAD_UD_BOARD_CHANNEL.value, 0, 3)
    led_raw = sech.get(ServoEnum.LED_HEAD_BOARD_CHANNEL.value, "0,4").split(',')
    led_pca_addr = pca_addrs[int(led_raw[0])]
    led_pwm_channel = int(led_raw[1])

    led_head = LedHead(broker=mqtt_connect, pca_address=led_pca_addr, pwm_channel=led_pwm_channel)
    body_LR = Gservo(location=ServoEnum.LOCATION_BODY_LEFT_RIGHT.value,
                     servo=body_lr_servo, axis='x', servo_range=default_angle,
                     broker=mqtt_connect, servo_speed=gs3508mg_speed, pulse_max_min=gs3508_pulse)
    body_UD = Gservo(location=ServoEnum.LOCATION_BODY_UP_DOWN.value, servo=body_ud_servo,
                     axis='y', servo_range=default_angle,
                     broker=mqtt_connect, pulse_max_min=mg92b_pulse, servo_speed=mg92d_speed)
    head_LR = Gservo(location=ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value, servo=head_lr_servo,
                     axis='x', servo_range=neck_angle,
                     broker=mqtt_connect, pulse_max_min=mg92b_pulse, servo_speed=mg92d_speed)
    head_UD = Gservo(location=ServoEnum.LOCATION_HEAD_UP_DOWN.value, servo=head_ud_servo,
                     axis='y', servo_range=head_angle,
                     broker=mqtt_connect, pulse_max_min=mg90d_pulse, servo_speed=mg90d_speed)
    body_LR.start()
    body_UD.start()
    head_UD.start()
    head_LR.start()
    led_shoulders = LedShoulders(broker=mqtt_connect)
    glados_right_lcd = GladosLCD(broker=mqtt_connect, location=SystemEnums.RIGHT_LCD.value,
                                 animation_path=animation_path, sync_leader=True)
    glados_right_lcd.start()
    led_head.startup()
    cefh = config_p[CameraEnum.CONFIG_HEAD.value]
    head_camera = Camera(configfile=config_p, location=cefh[CameraEnum.CAMERA_HEAD_FACTORY.value],
                         rtspport=int(cefh[CameraEnum.CAMERA_HEAD_PORT.value]))
    head_camera.start()
    # Signal AI server that cameras are ready for RTSP connection
    from glados_modules.MqttConnector import MQTTClient
    _ready_client = MQTTClient(ip=mqtt_connect.ip, port=mqtt_connect.port)
    _ready_client.send_command(
        {"system": "body_server", "cameras": [cefh[CameraEnum.CAMERA_HEAD_FACTORY.value]]},
        MQTTEnums.CAMERA_READY_TOPIC.value)
    print(f"Published camera ready signal for head camera")
    # Latency responder — echoes brain probes for speech-eye sync calibration.
    # Always-on: ~50 lines, zero state, only fires when a probe arrives.
    latency_responder = LatencyResponder(config_p)
    latency_responder.start()
    # Start health monitoring
    health = HealthMonitor(broker=mqtt_connect, system_name="body_server")
    health.register("body_LR", body_LR)
    health.register("body_UD", body_UD)
    health.register("head_LR", head_LR)
    health.register("head_UD", head_UD)
    health.register("lcd", glados_right_lcd)
    health.register("camera", head_camera)
    health.register("latency_responder", latency_responder)
    soc_poll = float(config_p.get(SocTempEnums.CONFIG_HEAD.value,
                                   SocTempEnums.POLL_INTERVAL.value,
                                   fallback=SocTempEnums.DEFAULT_POLL_INTERVAL.value))
    soc_temp = SocTemp(broker=mqtt_connect, poll_interval=soc_poll)
    soc_temp.start()
    health.register("soc_temp", soc_temp)
    health.start()
    # Start web dashboard (Pi4: consumes RTSP since Camera is a separate process)
    head_cam_factory = cefh[CameraEnum.CAMERA_HEAD_FACTORY.value]
    head_cam_rtsp_port = cefh[CameraEnum.CAMERA_HEAD_PORT.value]
    dash_port = int(config_p.get(DashboardEnums.CONFIG_HEAD.value,
                                  DashboardEnums.DASHBOARD_PORT.value,
                                  fallback=DashboardEnums.DEFAULT_PORT.value))
    # Subscribe to sensor MQTT topics for dashboard display
    sensor_tracker = SensorTracker(mqtt_connect)
    dashboard = WebDashboard(
        system_name="body_server",
        health_monitor=health,
        sensor_tracker=sensor_tracker,
        feed_uris={"Head Camera (Raw)": f"rtsp://localhost:{head_cam_rtsp_port}/{head_cam_factory}"},
        port=dash_port
    )
    dashboard.start()

    # Camera watchdog: respawn camera processes if they die
    cam_watchdog = CameraWatchdog(health_monitor=health)
    cam_watchdog.add_camera("camera", head_camera)
    cam_watchdog.start()

    def shutdown_handler(signum, frame):
        print("\nShutting down BodyServer...")
        # Restore default handlers so a second Ctrl+C kills us instead
        # of re-entering this function while it's still running.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        try:
            cam_watchdog.shutdown()
        except Exception as e:
            print(f"cam_watchdog shutdown error: {e}")
        # SIGKILL the whole process group. Catches GStreamer/x264enc
        # helpers spawned by the camera's RTSP pipeline, the
        # fuser/modprobe shells from the kernel-module reset path,
        # and any other descendants os._exit alone can't reach.
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except Exception:
            os._exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    while True:
        sleep(1)
