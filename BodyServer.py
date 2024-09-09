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
from glados_modules.BodyControl import Gservo, LedHead, LedShoulders, GladosLCD
from glados_modules.Camera import GLaDOSServerException, Camera


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

    animation_path = path.abspath(config_p["DEFAULT"]["aperture_animation"])
    head_camera_location = config_p["CAMERAS"]["Camera_Head_Factory"]
    max_min_tuple = namedtuple("max_min", ['max', 'min'])
    max_min_center_tuple = namedtuple("max_min_center", ['max', 'min', 'center'])
    Mqtt_tuple = namedtuple("service_address", ["ip", "port"])
    mqtt_connect = Mqtt_tuple(config_p["MQTT"]["mqtt_server_ip"], int(config_p["MQTT"]["mqtt_port"]))
    pulse_90 = config_p["SERVOS"]["mg90d_pulse"].split(',')
    pulse_92 = config_p["SERVOS"]["mg92b_pulse"].split(',')
    default = config_p["SERVOS"]["default_max_min_center"].split(',')
    head_min_max = config_p["SERVOS"]["head_min_max_center"].split(',')
    neck_min_max = config_p["SERVOS"]["head_min_max_center"].split(',')
    mg92d_speed = float(config_p["SERVOS"]["mg92b_speed"])
    mg90d_speed = float(config_p["SERVOS"]["mg90d_speed"])
    mg90d_pulse = max_min_tuple(int(pulse_90[0]), int(pulse_90[1]))
    mg92b_pulse = max_min_tuple(int(pulse_92[0]), int(pulse_92[1]))
    default_angle = max_min_center_tuple(int(default[0]), int(default[1]), int(default[2]))
    head_angle = max_min_center_tuple(int(head_min_max[0]), int(head_min_max[1]), int(head_min_max[2]))
    neck_angle = max_min_center_tuple(int(neck_min_max[0]), int(neck_min_max[1]), int(neck_min_max[2]))
    kit = ServoKit(channels=16)
    led_head = LedHead(broker=mqtt_connect)
    body_LR = Gservo(location='body_left_right', servo=kit.servo[0], axis='x', servo_range=default_angle,
                     broker=mqtt_connect)
    body_UD = Gservo(location='body_up_down', servo=kit.servo[1], axis='y', servo_range=default_angle,
                     broker=mqtt_connect, pulse_max_min=mg92b_pulse, servo_speed=mg92d_speed)
    head_LR = Gservo(location='head_left_right', servo=kit.servo[2], axis='y', servo_range=neck_angle,
                     broker=mqtt_connect, pulse_max_min=mg92b_pulse, servo_speed=mg92d_speed)
    head_UD = Gservo(location='head_up_down', servo=kit.servo[3], axis='x', servo_range=head_angle,
                     broker=mqtt_connect, pulse_max_min=mg90d_pulse, servo_speed=mg90d_speed)
    led_shoulders = LedShoulders(broker=mqtt_connect)
    glados_right_lcd = GladosLCD(broker=mqtt_connect, location="right_lcd", animation_path=animation_path)
    glados_right_lcd.start()
    led_head.startup()
    head_camera = Camera(configfile=config_p, location=head_camera_location)
    body_LR.start()
    body_UD.start()
    head_LR.start()
    head_UD.start()
    head_camera.start()
    while True:
        sleep(1)
