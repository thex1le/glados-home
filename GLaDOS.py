import random
import time
from os import path
import argparse
import sys
import configparser

# glados imports
from glados_modules.GLaDOSGpt import GladosGPT
from glados_modules.GlogConfig import setup_logger
from glados_modules.HomeAssistantConnector import HomeAssistantLink
from glados_modules.ChatGPTConnector import GladosGPT
from glados_modules.EggTimer import EggTimer
from glados_modules.Speech2Text import GladosSTT
from glados_modules.CameraRTSP import Camera
from glados_modules.GLaDosEnums import CameraEnum
from glados_modules.GLaDOSLocal import GladosLocal
from glados_modules.MqttConnector import MQTTClient
from glados_modules.CameraModule import Camera
from glados_modules.GladosEnums import CameraEnum, SystemEnums, MQTTEnums, LoggingEnums


# silence some errors on the terminal
def py_error_handler(filename, line, function, err, fmt):
    pass


# silence some errors on the terminal
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)


@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary(SystemEnums.LIB_ASOUND.value)
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)


with noalsaerr():
    p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=1)
stream = p.open(format=pyaudio.paFloat32, channels=2, rate=44100, output=1)


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
