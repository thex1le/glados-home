import random
import time
from os import path
import argparse
import sys
import configparser

# glados imports
from glados_modules.ChatGPTConnector import GladosGPT
from glados_modules.Speech2Text import GladosSTT
from glados_modules.GLaDOSLocal import GladosLocal
from glados_modules.CameraModule import Camera
from glados_modules.GladosEnums import CameraEnum


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
    gl = GladosLocal(configp, GladosGPT)
    gl.start()
    gl.speak("Oh Its you! , , Its been a long time...")
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
