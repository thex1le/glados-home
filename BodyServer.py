# builtin
from time import sleep
from argparse import ArgumentParser
from configparser import ConfigParser
import sys
from os import path
from threading import Thread

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

    mqtt_ip = config_p["MQTT"]["mqtt_server_ip"]
    mqtt_port = int(config_p["MQTT"]["mqtt_port"])
    animation_path = config_p["DEFAULT"]["animation_root"]
    kit = ServoKit(channels=16)
    led_head = LedHead(broker=mqtt_ip, port=mqtt_port)
    body_LR = Gservo(location='body_left_right', skit=kit.servo[0], axis='x', max_angle=180,
                     broker=mqtt_ip, port=mqtt_port)
    body_UD = Gservo(location='body_up_down', skit=kit.servo[1], axis='y', max_angle=180,
                     broker=mqtt_ip, port=mqtt_port)
    head_UD = Gservo(location='head_left_right', skit=kit.servo[2], axis='y', max_angle=180,
                     broker=mqtt_ip, port=mqtt_port)
    head_LR = Gservo(location='head_up_down', skit=kit.servo[3], axis='x', max_angle=180,
                     broker=mqtt_ip, port=mqtt_port)
    led_shoulders = LedShoulders(broker=mqtt_ip, port=mqtt_port)
    glados_right_lcd = GladosLCD(broker=mqtt_ip, port=mqtt_port, location="right_lcd", animation_path=animation_path)
    head_camera = Camera(configfile=config_p, location="Camera_Head")
    body_LR.start()
    body_UD.start()
    head_LR.start()
    head_UD.start()
    head_camera.start()

    # todo figure out how to pass images_path for the animation to pay
    glados_right_lcd.start()
    while True:
        sleep(1)
