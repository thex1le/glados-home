# builtin
from time import sleep
from argparse import ArgumentParser
from configparser import ConfigParser
import sys
from os import path
from collections import namedtuple

# 3rd party
from adafruit_servokit import ServoKit

# glados imports
from glados_modules.BodyControlModules import Gservo, LedHead, LedShoulders, GladosLCD, IMU
from glados_modules.CameraModule import GLaDOSServerException, Camera
from glados_modules.GladosEnums import CameraEnum, ServoEnum, SystemEnums


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
    kit = ServoKit(channels=16)
    led_head = LedHead(broker=mqtt_connect)
    body_LR = Gservo(location=ServoEnum.LOCATION_BODY_LEFT_RIGHT.value,
                     servo=kit.servo[0], axis='x', servo_range=default_angle,
                     broker=mqtt_connect, servo_speed=gs3508mg_speed, pulse_max_min=gs3508_pulse)
    body_UD = Gservo(location=ServoEnum.LOCATION_BODY_UP_DOWN.value, servo=kit.servo[1],
                     axis='y', servo_range=default_angle,
                     broker=mqtt_connect, pulse_max_min=mg92b_pulse, servo_speed=mg92d_speed)
    head_LR = Gservo(location=ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value, servo=kit.servo[2],
                     axis='x', servo_range=neck_angle,
                     broker=mqtt_connect, pulse_max_min=mg92b_pulse, servo_speed=mg92d_speed)
    head_UD = Gservo(location=ServoEnum.LOCATION_HEAD_UP_DOWN.value, servo=kit.servo[3],
                     axis='y', servo_range=head_angle,
                     broker=mqtt_connect, pulse_max_min=mg90d_pulse, servo_speed=mg90d_speed)
    body_LR.start()
    body_UD.start()
    head_UD.start()
    head_LR.start()
    led_shoulders = LedShoulders(broker=mqtt_connect)
    glados_right_lcd = GladosLCD(broker=mqtt_connect, location=SystemEnums.RIGHT_LCD.value,
                                 animation_path=animation_path)
    glados_right_lcd.start()
    led_head.startup()
    # startup the IMU sensor in the head
    imu = IMU(broker=mqtt_connect)
    imu.start()

    cefh = config_p[CameraEnum.CONFIG_HEAD.value]
    head_camera = Camera(configfile=config_p, location=cefh[CameraEnum.CAMERA_HEAD_FACTORY.value],
                         rtspport=int(cefh[CameraEnum.CAMERA_HEAD_PORT.value]))
    head_camera.start()
    while True:
        sleep(1)
