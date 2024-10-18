#built in
from argparse import ArgumentParser
from configparser import ConfigParser
import sys
from os import path

# glados imports
from glados_modules.MachineVision import MLDetect, GLaDOSServerException
from gladosTTS import engine as glados_voice

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
    # start up machine vision
    mv = MLDetect(config_p)
    mv.start()
    # start the text to speech engine
    glados_voice.main()
