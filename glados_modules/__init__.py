from argparse import ArgumentParser
import sys
from configparser import ConfigParser
from os import path

# glados imports
from gladosTTS import engine as glados_voice
from glados_modules.MachineVison import YoloDetect


class GLaDOSSAIServerException(Exception):
    pass


if __name__ == "__main__":
    parser = ArgumentParser(description='Evil Home GLaDOS AI Senses Server')
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
        raise GLaDOSSAIServerException("Unable to load file {}".format(args.conf[0]))
    # import and init the 3rd part glados text to speach engine,
    # this prevents init of the engine when you just want to print the help
    ImageProcessing = YoloDetect(config_p)
    ImageProcessing.start()
    # start the text to speech engine
    glados_voice.main()
    # ttsengine = mp.Process(target=engine.main, args=())
    # t tsengine.start()
    # loop
    # do we keep looping hear? what blocks on main?
    # while True:
    #    time.sleep(1)
