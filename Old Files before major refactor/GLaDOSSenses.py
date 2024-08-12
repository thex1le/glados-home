import sys
sys.path.insert(0, '/home/textile/glados-home/gladosTTS')
import json
from threading import Thread
import argparse
import configparser
from os import path
from pickle import loads

#3rd party
import cv2

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.RxTx import DataRecv, DataSend


class GLaDOSServerException(Exception):
    pass


# TODO YOU LEFT OFF CONNECTING IN MQQT CLIENT, it needs to send data to a debug channel
# DATASEND can be replaced by mqtt to send text back to main running program
# CONSIDER setting up a listener to take commands if needed? maybe not this portion kind of runs alone by its self

class YoloDetect(Thread):
    def __init__(self, configfile):
        # internal libs, import here so its deps are not needed on other devices
        Thread.__init__(self)
        Thread.daemon = True
        self.logger = setup_logger(name=self.__name__)
        self.configfile = configfile
        cam_res = self.configfile['DEFAULT']['camera_resolution'].split(',')
        self.cam_res_x = int(cam_res[0])
        self.cam_res_y = int(cam_res[1])
        self.factory = self.configfile['DEFAULT']['rtsp_factory']
        self.rtsp_port = int(self.configfile['DEFAULT']['rtsp_port'])
        self.rtsp_server_ip = self.configfile['DEFAULT']['rtsp_server_ip']
        self.confyolo = configfile["YOLO"]
        self.logger.debug(f"YOLO model started with {self.confyolo["model"]}")
        self.model = YOLO(self.confyolo["model"])
        self.sight = None
        self.imageget = DataRecv(self.configfile)
        self.imageget.start()
        self.imagesend = DataSend(self.configfile)
        self.imagesend.start()
        # TODO add this to config file
        msg = f"Starting the RTSP server on rtsp://{self.rtsp_server_ip}:{self.rtsp_port}/{self.factory}"
        self.logger.info(msg)
        print(msg)
        #TODO fix this to support all 3 cameras
        self.rtsp = GLaDOSRTSP.RTSPServer(cam_x=self.cam_res_x, cam_y=self.cam_res_y,
                                          port=self.rtsp_port, factory=self.factory)

    def get_sight(self):
        return self.sight

    def __translate_results(self, results):
        results_dict = {}
        for yclass in results:
            if yclass is None:
                continue
            self.logger.debug(f"Translating {yclass} with type {type(yclass)}")
            jclass = json.loads(yclass.tojson())
            for cname in jclass:
                name = cname["name"]
                if name in list(results_dict.keys()):
                    results_dict[name]["count"] += 1
                    results_dict[name]["objects"].append(cname)
                else:
                    results_dict[name] = {"count": 1, "objects": [cname], "class_name": name}
        self.logger.debug(results_dict)
        print(results_dict)

    def __yolo_process_image(self, image):
        # pass image to rtsp...
        results = self.model(image)
        annotator = Annotator(image)
        for r in results:
            annotator = Annotator(image)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])
                self.logger.debug(f"Labeled image with, {self.model[int(c)]}")
        a_image = annotator.result()
        self.rtsp.send_data(a_image)
        return results

    def process_image(self, image, debug_file_name='raw_rx.jpg'):
        final_image = loads(image)
        cv2.imwrite(debug_file_name, final_image)
        self.logger.debug(f"Wrote out sample debug image to {debug_file_name}")
        self.sight = self.__yolo_process_image(final_image)
        self.logger.debug("Sending back process dict of seen data")
        self.imagesend.send_data(self.__translate_results(self.sight))

    def run(self):
        while True:
            image = self.imageget.get_data(True) 
            msg = "Got image from sender"
            print(msg)
            self.logger.debug(msg)
            try:
                self.process_image(image)
            except Exception:
                msg = "Image Error"
                print(msg)
                self.logger.error(msg)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evil Home AI Senses Server')
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
        raise GLaDOSServerException("Unable to load file {}".format(args.conf[0]))
    # import and init the 3rd part glados text to speach engine,
    # this prevents init of the engine when you just want to print the help
    from gladosTTS import engine
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator
    eyes = YoloDetect(configp)
    eyes.start()
    # start the text to speech engine
    engine.main()
    # ttsengine = mp.Process(target=engine.main, args=())
    # t tsengine.start()
    # loop
    # do we keep looping hear? what blocks on main?
    # while True:
    #    time.sleep(1)
