import io
import base64
import random
from threading import Thread, Lock
import time
from os import path
import argparse
import sys
import configparser
from ctypes import *
from contextlib import contextmanager
import multiprocessing as mp
from queue import Queue
from typing import Dict, Callable, Tuple, NamedTuple
from json import loads
from collections import namedtuple

# 3rd party imports
import requests
import pyaudio
from pydub import AudioSegment
from pydub.playback import play
from alsaaudio import Mixer
import regex as re
from paho.mqtt.client import MQTTMessage

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosHomeAssistant import HomeAssistantLink
from glados_modules.GLaDOSGpt import GladosGPT
from glados_modules.EggTimer import EggTimer
from glados_modules.Speech2Text import GladosSTT
from glados_modules.MqttClient import MQTTClient, ServoMessageBuilder
from glados_modules.Camera import Camera
from glados_modules.GladosData import ServoLocation, VisionTracker
from glados_modules.GLaDosEnums import CameraEnum, ServoEnum, SystemEnums, TrackingEnums, VisionResultsEnum


# silence some errors on the terminal
def py_error_handler(filename, line, function, err, fmt):
    pass


# silence some errors on the terminal
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)


@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)


with noalsaerr():
    p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=1)
stream = p.open(format=pyaudio.paFloat32, channels=2, rate=44100, output=1)


class GladosException(Exception):
    pass


class MotionTrack(MQTTClient):
    # class for motion tracking on a target
    # TODO figure out if we want this here, or in teh Gbody class in the body server?
    def __init__(self, broker: NamedTuple,  camera_resolution: NamedTuple, target: str = "person",
                 confidence: float = 0.65, move_fudge_factor: int = 3):
        self.__name__ = self.__class__.__name__
        self.location = self.__name__
        self.logger = setup_logger(self.__name__)
        self.cmd_topic: str = TrackingEnums.MQTT_COMMAND_TOPIC.value
        self.cmd_trigger: str = TrackingEnums.MSG_COMMAND_KEY.value
        self.intensity_topic: str = SystemEnums.MQTT_INTENSITY_TOPIC.value
        self.count = VisionResultsEnum.VISION_RESULTS_COUNT_KEY.value
        self.intensity: Tuple[float, float] = (.1, .1)
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd,
                                                   self.intensity_topic: self.handle_intensity}
        # head camera resolution
        # TODO this will work for now but need to get all camera resolution to account for side cameras
        self.cam_x = int(camera_resolution.x)
        self.cam_y = int(camera_resolution.y)
        self.main_camera = CameraEnum.CAMERA_HEAD.value
        self.left_camera = CameraEnum.CAMERA_LEFT.value
        self.right_camera = CameraEnum.CAMERA_RIGHT.value
        self.move_fudge_factor = move_fudge_factor
        servo = namedtuple("servo", ["name", "move"])
        # servo names
        self.head_LR = servo(ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value, ServoMessageBuilder.head_left_right)
        self.head_UD = servo(ServoEnum.LOCATION_HEAD_UP_DOWN.value, ServoMessageBuilder.head_up_down)
        self.body_LR = servo(ServoEnum.LOCATION_BODY_LEFT_RIGHT.value, ServoMessageBuilder.body_left_right)
        self.body_UD = servo(ServoEnum.LOCATION_BODY_UP_DOWN.value, ServoMessageBuilder.body_up_down)
        self.target = target
        self.confidence = confidence
        self.servos = dict()
        # default movement speed
        self.dms: int = 3
        # bool if movement on left or right cameras, true we move, false we dont
        self.peripheral_hunt = True
        # bool if the head is currently tracking something
        self.head_tracking = False
        super().__init__(ip=broker.ip, port=broker.port)
        # Create Servo Location Tracker
        self.servo_status = ServoLocation(broker)
        # Vision seen Tracker
        self.objects = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
        self.vision_tracker = VisionTracker(broker, self.target, self.confidence, self.track_loop)
        # TODO do we need these there? are we sending signals? maybe trigger LED events? Maybe pulse eye down?
        # TODO figure out how we are going to track anger intensity over various body parts
        # TODO likely remove this next line
        self.scan_success = False

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """
        Trigger the loop that hunts and locks onto target...
        """
        j_msg = loads(msg.payload.decode())
        print(f"*** TRACKING FIRED", {self.cmd_trigger}, {TrackingEnums.MSG_COMMAND_START.value})
        print(j_msg, j_msg.get(self.cmd_trigger))
        if j_msg.get(self.cmd_trigger, "") == TrackingEnums.MSG_COMMAND_START.value:
            self.logger.debug(f"Tracking Command Received, {msg.topic}, {j_msg}")
            trigger_camera = j_msg.get(TrackingEnums.MSG_CAMERA_KEY.value, "")
            self.logger.debug(f"Running Track loop for {trigger_camera}")
            self.track_loop(trigger_camera)

    def track_loop(self, camera):
        # main tracking loop
        # find target
        # don't double call if head_tracking is True, just skip this detection
        if self.__check_tracking() is True:
            self.logger.debug("Getting Vision Map")
            vision_map = self.vision_tracker.get_vision_map()
            self.logger.debug("Looping though vision map")
            if camera in vision_map.keys():
                if vision_map[camera][self.target].get(self.count, 0) != 0:
                    target_bounding = self.__find_target(vision_map[camera][self.target][self.objects])
                    if camera == TrackingEnums.BODY_HEAD_CAMERA.value:
                        self.logger.debug("Ready to move all servos")
                        self.move_all_servos(target_bounding)
                    elif camera in (TrackingEnums.BODY_LEFT_CAMERA.value, TrackingEnums.BODY_RIGHT_CAMERA.value):
                        self.logger.debug("Rotating Body to face target")
                        self.rotate_body(target_bounding)
            with self._lock:
                self.head_tracking = False

    def __check_tracking(self) -> bool:
        """
        Check if we are tracking and return a bool, if we are not set the master bool to true
        """
        with self._lock:
            if self.head_tracking is False:
                self.logger.debug(f"Tracking is allowed, Moving To track {self.target}")
                self.head_tracking = True
                rtn = True
            else:
                self.logger.debug("Tracking requested but already currently moving to track target")
                rtn = False
        return rtn

    def rotate_body(self, target: dict) -> None:
        # Get current servo position
        self.logger.debug("Moving servos getting angle map")
        self.servos = self.servo_status.get_angle_map()
        self.logger.debug("Calculating movement for servos")
        mv_list = list()
        if target != {}:
            # account for left right swap
            body_lr = self.__mirror_calc(self.__calc_servo(self.servos[self.body_LR.name], target))
            mv_list.append(self.body_LR.move(body_lr))
            if mv_list:
                self.logger.debug("Sending Move commands for Head and Neck")
                self.servo_status.send_command(mv_list, ServoEnum.MQTT_COMMAND_TOPIC.value)
                body_movement = {self.body_LR.name: body_lr}
                self.__block_for_update(body_movement)

    def move_all_servos(self, target: dict) -> None:
        # Get current servo position
        self.logger.debug("Moving servos getting angle map")
        self.servos = self.servo_status.get_angle_map()
        mv_list = list()
        self.logger.debug("Calculating movement for servos")
        if target != {}:
            # Move head left-right and up-down first
            head_lr = self.__calc_servo(self.servos[self.head_LR.name], target)
            head_ud = self.__calc_servo(self.servos[self.head_UD.name], target)
            if self.__distance_check(self.servos[self.head_LR.name], head_lr, self.move_fudge_factor):
                mv_list.append(self.head_LR.move(head_lr))
            else:
                # don't try small movements just set it to current
                head_lr = self.servos[self.head_LR.name].current
            if self.__distance_check(self.servos[self.head_UD.name], head_ud, self.move_fudge_factor):
                mv_list.append(self.head_UD.move(head_ud))
            else:
                head_ud = self.servos[self.head_UD.name].current
            if mv_list:
                self.logger.debug("Sending Move commands for Head and Neck")
                self.servo_status.send_command(mv_list, ServoEnum.MQTT_COMMAND_TOPIC.value)
                head_movement = {self.head_LR.name: head_lr, self.head_UD.name: head_ud}
                self.__block_for_update(head_movement)
            # Check if head movement reached its limit and compensate with body movement
            if self.__reached_limit(self.servos[self.head_LR.name]):
                self.logger.debug("Head reached left/right limit, rotating body to extend range")
                self.__rotate_body_to_extend_range()
            if self.__reached_limit(self.servos[self.head_UD.name]):
                self.logger.debug("Head reached up/down limit, bending body to extend range")
                self.__bend_body_to_extend_range()
            # Level the head with the body after movement
            servo_1, servo_2 = self.__level_servos(self.head_LR, self.body_LR)
            servo_3, servo_4 = self.__level_servos(self.head_UD, self.body_UD)
            body_level = {self.head_LR.name: servo_1, self.body_LR.name: servo_2,
                          self.head_UD.name: servo_3, self.body_UD.name: servo_4}
            self.__block_for_update(body_level)
            # Add a small delay to make the movement seem more deliberate
            time.sleep(2.5)

    def __block_for_update(self, target_positions: Dict[str, int]) -> None:
        # Loop until all servos reach their target positions
        count = 0
        self.logger.debug(f"Waiting for updates on {len(target_positions.keys())}")
        while True:
            self.servos = self.servo_status.get_angle_map()
            all_reached = True
            for name, target in target_positions.items():
                if self.servos[name].current != target:
                    all_reached = False
                    self.logger.debug(f"{name} servo is currently blocking attempting to get to {target}")
                    break
                else:
                    self.logger.debug(f"{name} servo has updated and reached {target}")
            if all_reached:
                break
            time.sleep(0.2)
            count += 1
            if count >= 15:
                # hard block for some reason, trigger servo updates
                count = 0
                self.servo_status.update_servo_status()
            self.logger.debug(f"Blocking Updates Complete")

    def __reached_limit(self, servo) -> bool:
        """
        Check if the servo has reached its movement limit.
        """
        return servo.current == servo.min or servo.current == servo.max

    def __rotate_body_to_extend_range(self):
        self.logger.debug("Rotating body to extend range of neck")
        # Calculate the difference between head's current position and middle
        diff = self.servos[self.head_LR.name].current - self.servos[self.head_LR.name].middle
        # Adjust body servo in the same direction
        new_body_angle = self.servos[self.body_LR.name].current + diff
        # Clamp the new angle within body's allowed range
        new_body_angle = max(min(new_body_angle, self.servos[self.body_LR.name].max),
                             self.servos[self.body_LR.name].min)
        # Send movement command
        self.servo_status.send_command(
            [self.body_LR.move(new_body_angle)],
            ServoEnum.MQTT_COMMAND_TOPIC.value)
        # Block until the movement is completed
        self.__block_for_update({self.body_LR.name: new_body_angle})

    def __bend_body_to_extend_range(self):
        self.logger.debug("Bending body to extend range of head")
        # Calculate the difference between head's current position and middle
        diff = self.servos[self.head_UD.name].current - self.servos[self.head_UD.name].middle
        # Adjust body servo in the same direction
        new_body_angle = self.servos[self.body_UD.name].current + diff
        # Clamp the new angle within body's allowed range
        new_body_angle = max(min(new_body_angle, self.servos[self.body_UD.name].max),
                             self.servos[self.body_UD.name].min)
        # Send movement command
        self.servo_status.send_command(
            [self.body_UD.move(new_body_angle)],
            ServoEnum.MQTT_COMMAND_TOPIC.value)
        # Block until the movement is completed
        self.__block_for_update({self.body_UD.name: new_body_angle})

    def __find_target(self, seen_data) -> dict:
        """
        Find the highest confidence target and return their bounding box from current data set
        """
        confidence = VisionResultsEnum.VISION_RESULTS_CONFIDENCE_KEY.value
        bbox = VisionResultsEnum.VISION_RESULTS_BOX_KEY.value
        rtn = dict()
        highest_confidence = 0
        for p in seen_data:
            if p[confidence] > highest_confidence:
                highest_confidence = p[confidence]
                rtn = p[bbox]
        self.logger.debug(f"Confidence box found {rtn} with confidence score of {highest_confidence}")
        return rtn

    def __calc_servo(self, servo, bbox: dict) -> int:
        # Determine axis and image dimensions
        if servo.axis == 'x':
            bbox_edge_1 = bbox['x1']
            bbox_edge_2 = bbox['x2']
            axis_size = self.cam_x
            # Determine direction factor based on servo location
            if servo.location == ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value:
                direction_factor = 1  # Head LR servo moves with image shift
            else:
                direction_factor = -1  # Body LR servo compensates
        else:
            bbox_edge_1 = bbox['y1']
            bbox_edge_2 = bbox['y2']
            axis_size = self.cam_y
            # Determine direction factor based on servo location
            if servo.location == ServoEnum.LOCATION_HEAD_UP_DOWN.value:
                direction_factor = 1  # Head UD servo moves with image shift
            else:
                direction_factor = -1  # Body UD servo compensates
        # Calculate the center of the bounding box on the axis
        center_of_bbox = (bbox_edge_1 + bbox_edge_2) / 2
        # Calculate the offset from the image center (in pixels)
        offset_from_center = (axis_size / 2) - center_of_bbox  # Reverse due to camera movement
        # Calculate the proportion of the offset relative to the image size
        offset_proportion = offset_from_center / (axis_size / 2)  # Normalize between -1 and 1
        # Calculate the angle adjustment based on the proportion
        angle_range = servo.max - servo.min
        angle_adjustment = direction_factor * offset_proportion * (angle_range / 2)
        # Determine the new servo angle based on the current position
        new_servo_angle = servo.current + angle_adjustment
        # Clamp the new angle within servo's min and max
        new_servo_angle = max(min(new_servo_angle, servo.max), servo.min)
        # Round to the nearest whole number
        return round(new_servo_angle)

    def __distance_check(self, servo, new_angle, degree_diff=2):
        move = False
        current_angle = servo.current
        difference = 0
        angle_gl = "not greater or less"
        movement = "not moving"
        move_factor = angle_gl
        if new_angle > current_angle:
            angle_gl = "greater"
            difference = new_angle - current_angle
            if abs(difference) > degree_diff:
                move_factor = "greater"
                movement = "moving"
                move = True
            else:
                move_factor = "greater"
                movement = "not moving"
        elif new_angle < current_angle:
            difference = new_angle - current_angle
            angle_gl = "less"
            if abs(difference) > degree_diff:
                move = True
                move_factor = "greater"
                movement = "moving"
            else:
                movement = "not moving"
                move_factor = "less"
        self.logger.debug(f"{servo.location} {new_angle} is {angle_gl} than {current_angle} and "
                          f"with a difference of {abs(difference)} which is {move_factor} than small movement factor"
                          f" of {degree_diff}, {movement}")
        return move

    def __level_servos(self, servo1, servo2) -> tuple:
        # bring servo1 to midpoint by moving servo2
        # ensure servos are on the same axis
        self.logger.debug(f"Leveling Servos {self.servos[servo1.name].location} & {self.servos[servo2.name].location}")
        if self.servos[servo1.name].axis != self.servos[servo2.name].axis:
            msg = "Servos are not on same axis"
            self.logger.error(msg)
            raise Exception(msg)
        current = self.servos[servo1.name].current
        if servo1.name in (ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value, ServoEnum.LOCATION_HEAD_UP_DOWN.value):
            angle = self.__mirror_calc(current)
        else:
            angle = current
        mv_list = [servo2.move(angle),
                   servo1.move(self.servos[servo1.name].middle)]
        self.send_command(mv_list, ServoEnum.MQTT_COMMAND_TOPIC.value)
        # servo 1, servo 2
        return self.servos[servo1.name].middle, self.servos[servo1.name].current

    def __mirror_calc(self, servo_angle) -> int:
        """
        Figure out degree on other side when we have servos that need to align and their left and right's are flipped
        """
        return 180 - servo_angle

    def handle_intensity(self, msg: MQTTMessage) -> None:
        # TODO figure out update commands
        j_msg = loads(msg.payload.decode())
        if j_msg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {j_msg}")
            self.intensity = j_msg["intensity"]


class GladosLocal(Thread, MQTTClient):
    def __init__(self, config_file, remote_llm):
        Thread.__init__(self)
        Thread.daemon = True
        ip = configp["MQTT"]["mqtt_server_ip"]
        port = int(configp["MQTT"]["mqtt_port"])
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__)
        MQTTClient.__init__(self, ip=ip, port=port)
        self.cmd_topic: str = "vision/camera_response"
        self.intensity_topic: str = "intensity"
        # TODO consider how we want to handle all 3 cameras...
        self.main_camera = "Camera_Head"
        self.topic_handler: Dict[str, Callable] = {self.intensity_topic: self.handle_intensity}
        self.llm = remote_llm
        self.last_greeting = None
        self.last_insult = None
        self.last_process = None
        self.last_question = None
        self.last_qresponse = None
        self.last_fresponse = None
        self.last_cresponse = None
        self.timers = Queue()
        self.configFile = config_file
        # TODO need to fix config file
        self.voiceurl = config_file["DEFAULT"]["VoiceUrl"]
        self.configp = config_file["LOCALSPEAK"]
        root_path = self.configp.get("localpath", "./txt_responses")
        self.greetings = self.llp(self.configp.get("greetings", list()), root_path)
        self.processing = self.llp(self.configp.get("processing", list()), root_path)
        self.insults = self.llp(self.configp.get("insults", list()), root_path)
        self.questions = self.llp(self.configp.get("questions", list()), root_path)
        self.qresponse = self.llp(self.configp.get("qresponses", list()), root_path)
        self.cancel = self.llp(self.configp.get('cancel', list()), root_path)
        self.vision_confidence = float(self.configp.get("VisionConfidence", 0.0))
        self.fuck = self.llp(self.configp.get("fuck", list()), root_path)
        self.mixer = Mixer("Speaker")
        self.__change_volume(int(config_file["DEFAULT"]["VolumeLevel"]))
        self.current_vol = int(self.mixer.getvolume()[0])
        self.sight_results = mp.Manager().dict()
        self.stop = False
        self.homeass = HomeAssistantLink(config_file)
        #self.homeass.get_temp()
        # TODO figure out how to implement the songs
        #self.portal1song()
        self.mp_lock = mp.Lock()
        self.seen = None
        self.last_seen_human = time.time()
        # TODO setup LEFT LCD

    def handle_intensity(self):
        # TODO FIGURE OUT WHAT TO DO HERE
        pass

    def __random_audio(self, choice, last, options_list, last_attr_name, just_text=False):
        proc = self.__dedupe(choice, last, options_list)
        if just_text is False:
            self.speak(proc)
        if hasattr(self, last_attr_name):
            setattr(self, last_attr_name, proc)
        return proc

    def random_response(self, category: str, last_response: str, responses: list, last_response_attr: str,
                        just_text: bool = False) -> str:
        response = self.__random_audio(random.choice(responses), last_response,
                                       responses, last_response_attr, just_text)
        self.logger.debug(f"Random {category}: {response}")
        return response

    def random_cancel_response(self, just_text: bool = False) -> str:
        return self.random_response('Cancel Command response', self.last_cresponse,
                                    self.cancel, 'last_cresponse', just_text)

    def random_question_response(self, just_text: bool = False) -> str:
        return self.random_response('Question Response', self.last_qresponse,
                                    self.qresponse, 'last_qresponse', just_text)

    def random_question(self, just_text: bool = False) -> str:
        return self.random_response('Question', self.last_question,
                                    self.questions, 'last_question', just_text)

    def random_insult(self, just_text: bool = False) -> str:
        return self.random_response('Insult', self.last_insult,
                                    self.insults, 'last_insult', just_text)

    def random_processing(self, just_text: bool = False) -> str:
        return self.random_response('Processing', self.last_process,
                                    self.processing, 'last_process', just_text)

    def random_fuck_response(self, just_text: bool = False) -> str:
        return self.random_response('Fuck Off Response', self.last_fresponse,
                                    self.fuck, 'last_fresponse', just_text)

    def random_greeting(self, just_text: bool = False) -> str:
        return self.random_response('Greeting', self.last_greeting,
                                    self.greetings, 'last_greeting', just_text)

    def __dedupe(self, current, last, options):
        while current == last:
            current = random.choice(options)
        return current

    def llp(self, file, root_path):
        file = path.abspath(path.join(root_path, file))
        if path.isfile(file) is True:
            with open(file, 'r') as f:
                lines = f.readlines()
            # clean the strings
            clines = list()
            for i in lines:
                clines.append(i.strip())
            return clines
        else:
            msg = f"Unable to load file {file}"
            self.logger.error(msg)
            raise GladosException(msg)
        # load local phrases

    def __check_local_command(self, user_prompt, command):
        if type(command) is not re.Pattern:
            command = re.escape(command)
        match = re.search(command, user_prompt)
        return bool(match)
    
    def get_seen_prompt(self):
        return self.seen
    
    def portal1song(self):
        # TODO fix filepath here
        with open('./wav/portal_still_alive.wav', 'rb') as wav:
            self.__play_audio(wav.read())
    
    def portal2song(self):
        # TODO fix filepath here
        with open('./wav/portal2_want_you_gone.wav', 'rb') as wav:
            self.__play_audio(wav.read())
    
    def get_temp(self, user_prompt):
        c_str = r"what(?:'?s| is) the (current )?(outside )?(temp(erature)?)( outside)?\??"
        check = self.__check_local_command(user_prompt.lower(), re.compile(c_str))
        if check is True:
            self.speak(self.homeass.get_temp())
        return check

    def fuck_you(self, user_prompt):
        check = self.__check_local_command(user_prompt.lower(), "fuck you")
        if check is True:
            self.random_fuck_response()
        return check
    
    def translate_time(self, user_prompt: str) -> dict:
        pattern = r'(\d+)\s*(hour|minute|second)s?'
        matches = re.findall(pattern, user_prompt)
        time_dict = {f'{time_unit}s': int(value) for value, time_unit in matches}
        total_seconds = time_dict.get('seconds', 0) \
                    + time_dict.get('minutes', 0) * 60 \
                    + time_dict.get('hours', 0) * 3600
        time_dict['total_seconds'] = total_seconds
        self.logger.debug(f"User requested time: {time_dict}")
        return time_dict

    def timer(self, user_prompt):
        user_prompt = user_prompt.lower()
        check = self.__check_local_command(user_prompt, re.compile(r'set\s+(a\s+|the\s+)?timer'))
        if check is True:
            time_dict = self.translate_time(user_prompt)
            egg = EggTimer(time_dict['total_seconds'], self.speak)
            egg.start()
            self.timers.put(egg)
            time_units = list()
            if 'hours' in time_dict:
                time_units.append(f"{time_dict['hours']} hours")
            if 'minutes' in time_dict:
                time_units.append(f"{time_dict['minutes']} minutes")
            if 'seconds' in time_dict:
                time_units.append(f"{time_dict['seconds']} seconds")
            time_string = ', '.join(time_units)
            if ',' in time_string:
                time_string = time_string.rsplit(', ', 1)
                time_string = ' and '.join(time_string)
            self.speak(time_string)
        else: 
            check = self.__check_local_command(user_prompt, re.compile(r'(stop|cancel)\s+(the\s+|a\s+)?timer'))
            if check is True:
                if self.timers.empty() is True:
                    msg = "You have no running Timers"
                    self.logger.debug(msg)
                    self.speak(msg)
                else:
                    # TODO when stopping timers track which one we stop...
                    t = self.timers.get()
                    t.stop()
                    t.join()
        return check

    def run(self):
        self.last_seen_human = time.time()
        scan_room = 0
        while self.stop is False:
            self.seen = self.process_sight(self.sight_results)
            if self.sight_results.get("person", None) is None:
                # TODO this where you will do human detector millimeter wave
                # TODO set scan config time and number of times to look in conf file
                # TODO consider scanning for other things?
                if (time.time() - self.last_seen_human) < 120 and scan_room <= 2:
                    # PUT SCANNING FUNCTION HERE...
                    scan_room += 1
                else:
                    time.sleep(5)
            else:
                self.last_seen_human = time.time()
                time.sleep(1)
    
    def __adjust_count(self, obj):
        count = 0
        for o in obj:
            if o['confidence'] >= self.vision_confidence:
                count += 1
        return count

    def process_sight(self, seen):
        context = ["You can see the following things in the room"]
        for item in seen.keys():
            count = self.__adjust_count(seen[item]["objects"])
            if count == 0:
                continue
            context.append(f"{count} {item}")
        return ", ".join(context)
    
    def __get_audio(self, response):
        response = ", , " + response
        rsp = base64.b64encode(response.encode("utf8"))
        url = '{}{}'.format(self.voiceurl, str(rsp, 'utf8'))
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            msg = "Failed to translate text"
            self.logger.debug(msg)
            return -1

    def __play_audio(self, data):
        play(AudioSegment.from_file(io.BytesIO(data)))

    def speak(self, text):
        self.__play_audio(self.__get_audio(text))

    def __change_volume(self, level):
        # Set the volume
        self.mixer.setvolume(int(level)) 
    
    def set_volume(self, user_prompt):
        check = False
        pc_command = ["set volume", "change volume"]
        user_prompt = user_prompt.lower()
        for pc in pc_command:
            check = self.__check_local_command(user_prompt, pc)
            if check is True:
                break
        scheck = self.__check_local_command(user_prompt, re.compile(r'%'))
        if scheck is True:
            level = re.findall(r'\b\d+\b', user_prompt)
            # TODO figure out why vol level doesn't set correctly
            self.__change_volume(int(level[0]))
            self.current_vol = level[0]
            msg = f"I have set the volume to {level[0]} percent"
            self.logger.debug(msg)
            self.speak(msg)
        return check


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
    gstt = GladosSTT(gl)
    gstt.start()
    local_commands = (gl.get_temp, gl.fuck_you, gl.timer, gl.set_volume)
    left_camera_location = configp[CameraEnum.CONFIG_HEAD.value][CameraEnum.CAMERA_LEFT_FACTORY.value]
    right_camera_location = configp[CameraEnum.CONFIG_HEAD.value][CameraEnum.CAMERA_RIGHT_FACTORY.value]
    mqtt_broker = namedtuple("mqtt_broker", ["ip", "port"])
    cam_resolution = namedtuple("cam_resolution", ['x', 'y'])
    r_x, r_y = configp[CameraEnum.CONFIG_HEAD.value][
                       f"{CameraEnum.CAMERA_HEAD.value}_{CameraEnum.MSG_RESOLUTION.value}"].split(',')
    head_cam_resolution = cam_resolution(int(r_x), int(r_y))
    broker = mqtt_broker(configp["MQTT"]["mqtt_server_ip"], configp["MQTT"]["mqtt_port"])
    confidence = float(configp["REACTIONS"]["VisionConfidence"])
    mt = MotionTrack(broker=broker, camera_resolution=head_cam_resolution, target="person", confidence=confidence)
    left_camera = Camera(configfile=configp, location=left_camera_location)
    right_camera = Camera(configfile=configp, location=right_camera_location)
    left_camera.start()
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
