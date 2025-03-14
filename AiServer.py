#built in
from argparse import ArgumentParser
from configparser import ConfigParser
import sys
from os import path

# glados imports
from glados_modules.MachineVision import MLDetect, GLaDOSServerException
from glados_modules.WhisperXSpeech2Text import AudioServerRX, LocalSTTtx, SystemEnums
from glados_modules.GladosEnums import STTEnums
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
        config_p.read(args.conf[0], encoding=('utf-8'))
    else:
        raise GLaDOSServerException("Unable to load file {}".format(args.conf[0]))
    # start up machine vision
    mv = MLDetect(config_p)
    mv.start()
    # start the audio receive server
    broker = AudioServerRX.broker_tuple
    stt_conf = config_p[STTEnums.CONFIG_HEAD_STT.value]
    mqtt_conf = config_p[SystemEnums.CONFIG_HEAD_MQTT.value]
    audio_b = broker(stt_conf[STTEnums.STT_SERVER_IP.value], int(stt_conf[STTEnums.STT_SERVER_PORT.value]))
    # reuse ip port broker tuple
    mqtt_b = broker(mqtt_conf[SystemEnums.MQTT_SERVER_IP.value], int(mqtt_conf[SystemEnums.MQTT_PORT.value]))
    lstt_tx = LocalSTTtx(mqtt_b)
    stt_audio_rx = AudioServerRX(audio_b, callback=lstt_tx.process_audio)
    stt_audio_rx.start()
    # start the text to speech engine
    glados_voice.main()
