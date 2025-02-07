from time import sleep, time
from threading import Thread
import socketserver
from json import loads, dumps
import argparse
from os import path
import sys
import configparser
from multiprocessing import Manager, Lock

#3rd party imports
from adafruit_servokit import ServoKit
import GLaDOSDisplay
import board
import ledhelper
import adafruit_pca9685
import neopixel
import busio

# glados imports
from GLaDOSSenses import Camera as gleyes
from glados_modules.GlogConfig import setup_logger


class Gservo(Thread):
    def __init__(self, location, skit, axis, servo_range: tuple = (), max_angle=90):
        Thread.__init__(self)
        Thread.daemon = True
        self.logger = setup_logger(name=f"{self.__name__}_{location}")
        self.location = location
        # lock skit to the channel for this class
        self.min_angle = 0
        self.skit = skit
        # start at middle speed
        self.speed = 5
        # default to 45 for the 90's 
        self.max_angle = max_angle
        # set it in the middle_angle
        self.middle_angle = int(self.max_angle / 2)
        self.angle = self.middle_angle
        self.current_angle = self.angle
        self.first_boot = True
        self.move()
        self.exec_command = False
        self.moving = False
        self.axis = axis.lower()
        if servo_range == ():
            self.allowed_servo_range = {"min_travel": 0, "max_travel": max_angle}
        else:
            self.allowed_servo_range = {"min_travel": servo_range[0], "max_travel": servo_range[1]}

    def get_max_angle(self):
        return self.max_angle

    def get_middle_angle(self):
        return self.middle_angle

    def set_speed(self, speed):
        if speed >= 10:
            # top speed of 10
            speed = 10
        if speed <= 1:
            # go as slow as 1
            speed = 1
        self.speed = round(speed)
        self.logger.debug(f"Speed set to {self.speed}")

    def set_angle(self, angle):
        max = self.allowed_servo_range["max_travel"]
        min = self.allowed_servo_range["min_travel"]
        if angle >= max:
            self.angle = max
            self.logger.debug(f"{angle} is above {max}, setting to {max}")
        elif angle <= min:
            self.angle = min
            self.logger.debug(f"{angle} is below {min}, setting to {min}")
        else:
            self.angle = angle
            self.logger.debug(f"Angle set to {self.angle}")

    def set_speed_angle(self, speed_angle: tuple, execute=False):
        self.set_speed(speed_angle[0])
        self.set_angle(speed_angle[1])
        if execute is True:
            self.exec_command = True

    def get_angle(self):
        return self.current_angle

    def execute(self):
        self.exec_command = True

    def __get_direction_speed(self):
        # determine current angle and if were going up or down return a range object
        # moving higher
        rtn = range(0, 0)
        if self.angle > self.current_angle:
            rtn = range(self.current_angle, (self.angle + 1), self.speed)
        # moving lower()
        if self.angle < self.current_angle:
            rtn = range(self.current_angle, (self.angle + 1), (self.speed * -1))
        return rtn

    def __increment(self):
        # print you left off here trying to handle positive and negative values
        for s in self.__get_direction_speed():
            self.skit.angle = s
            sleep(.1)
        self.current_angle = self.angle

    def get_moving_status(self):
        # return if motor is moving or not
        return self.moving

    def move(self):
        if self.speed == 10 or self.first_boot is True:
            self.logger.debug(f"moving to {self.angle}")
            self.skit.angle = self.angle
            sleep(.3)
            self.moving = True
            self.current_angle = self.angle
            self.moving = False
            self.first_boot = False
        else:
            if self.angle != self.current_angle:
                self.logger.debug(f"moving to {self.angle}")
                self.moving = True
                self.__increment()
                self.moving = False

    def run(self):
        while True:
            if self.exec_command is True:
                self.move()
                self.exec_command = False
            else:
                sleep(.1)


class GBody(Thread):
    # class for managing all the servo and body movement in relation to the camera
    def __init__(self, config_file, cam_x_width: int, cam_y_width: int, lock):
        Thread.__init__(self)
        Thread.daemon = True
        self.logger = setup_logger(self.__name__)
        # access the servos
        kit = ServoKit(channels=16)
        # build a servo control for each joint
        self.body_LR = Gservo(location='body_left_right', skit=kit.servo[0], axis='x', max_angle=180)
        self.body_UD = Gservo(location='body_up_down', skit=kit.servo[1], axis='y', max_angle=60)
        self.head_UD = Gservo(location='head_up_down', skit=kit.servo[2], axis='y', max_angle=60)
        self.head_LR = Gservo(location='head_lef_right', servo_range=(15, 45), skit=kit.servo[3], axis='x', max_angle=60)
        self.seen_data = Manager().dict()
        self.cam_x_width = cam_x_width
        self.cam_y_width = cam_y_width
        self.stop = False
        self.lock = lock
        self.eyes = gleyes(self.set_scan_success, config_file)
        self.eyes.start()
        # find the x1 x2, y1, y2 of the target,
        # figure out if the head can look at it...
        # if we can then head / neck moves to it...
        # then recalculate so the head and neck can move back to center
        # and the body will rotate and middle_angle will move up or down
        # order of off center is self.body_LR > self.body_UD,> self.head.UP> self, head left right
        # TODO figure out how we are going to track anger intensity over various body parts
        self.led_head = LedHead()
        # thread the startup of the led head
        led_head_start = Thread(target=self.led_head.startup, args=())
        self.big_lcd_left = GLaDOSDisplay.GladosLCD()
        self.little_lcd_right = GLaDOSDisplay.GladosLCD(cs=board.D23, rst=board.D5, dc=board.D6,
                                                        sck=board.SCK_1, mosi=board.MOSI_1, flip=True)
        self.big_lcd_left.start()
        self.little_lcd_right.start()
        led_head_start.start()
        led_head_start.join()
        self.scan_success = False

    def callback_handler(self, command: dict):
        for i in command:
            print(i)

    def set_scan_success(self):
        # callback for the camera thread to signal the servos to stop moving
        self.logger.debug("Body Callback triggered")
        self.scan_success = True

    def scan_room(self, scan_speed=3, search_time=90, confidence=.70):
        #TODO consider how this will change with left and right cameras...,
        self.logger.debug("Scanning Room for Target")
        self.eyes.target_scan(search_time=search_time, confidence=confidence)
        t = time()
        while (time() - t) < search_time and self.scan_success is False:
            if self.scan_success is False:
                self.head_LR.set_speed_angle((scan_speed, self.head_LR.min_angle), execute=True)
                self.body_LR.set_speed_angle((scan_speed, self.body_LR.min_angle), execute=True)
                # TODO change when threading is enabled
                self.head_LR.move()
                self.body_LR.move()
            else:
                break
            # block till head and body are at min
            while (self.body_LR.get_angle() != self.body_LR.min_angle and
                   self.head_LR.get_angle() != self.head_LR.min_angle or self.scan_success is True):
                sleep(.2)
            if self.scan_success is False:
                self.head_LR.set_speed_angle((scan_speed, self.head_LR.max_angle), execute=True)
                self.body_LR.set_speed_angle((scan_speed, self.body_LR.max_angle), execute=True)
                # TODO change when threading is enabled
                self.head_LR.move()
                self.body_LR.move()
            else:
                break
            # block till head and body are at max
            while (self.body_LR.get_angle() != self.body_LR.max_angle and
                   self.head_LR.get_angle() != self.head_LR.max_angle or self.scan_success is True):
                sleep(.2)
        if self.scan_success is True:
            with self.lock:
                self.seen_data = self.eyes.get_results()
            self.scan_success = False
            self.move_servos()
        self.logger.debug("Scanning For Target Complete")

    def stop_body(self):
        """
        Stop body movement
        """
        self.stop = True

    def __find_person(self, target='person', confidence=.7) -> dict:
        """
        Find the highest confidence person and return their bounding box from current data set
        self.seen_data expected to be YOLO8 data response object
        """
        with self.lock:
            rtn = dict()
            if target in self.seen_data and self.seen_data[target]['count'] > 0:
                highest_confidence = 0
                highest_confidence_person = None
                for p in self.seen_data[target]['objects']:
                    if p['confidence'] > highest_confidence:
                        highest_confidence = p['confidence']
                        highest_confidence_person = p
                if highest_confidence_person is not None:
                    if highest_confidence >= confidence:
                        # take the highest confidence and return the bounding box
                        rtn = highest_confidence_person['box']
        self.logger.debug(f"Confidence box found {rtn} with confidence score of {confidence}")
        return rtn

    def __calc_servo(self, servo: Gservo, bbox: dict) -> int:
        """
        Calculate servo angle correction to target
        """
        # TODO determine if we need current_angle? does it matter?
        if servo.axis == 'x':
            bbox_edge_1 = bbox['x1']
            bbox_edge_2 = bbox['x2']
            axis_size = self.cam_x_width
        else:
            bbox_edge_1 = bbox['y1']
            bbox_edge_2 = bbox['y2']
            axis_size = self.cam_y_width
        # Calculate the center of the new person's bounding box on the x-axis
        center_updated = (bbox_edge_1 + bbox_edge_2) / 2
        # Calculate the offset of the person's center from the image center with the updated data
        offset_from_center = center_updated - (axis_size / 2)
        # Calculate the new servo angle to center on the person with the updated data
        new_servo_angle_updated = servo.middle_angle - (offset_from_center / axis_size * servo.max_angle)
        # Round to nearest whole
        return round(new_servo_angle_updated)

    def __level_servos(self, servo1: Gservo, servo2: Gservo) -> None:
        # bring servo1 to midpoint by moving servo2
        # ensure servos are on the same axis
        self.logger.debug(f"Leveling Servos {servo1.location} & {servo2.location}")
        if servo1.axis != servo2.axis:
            msg = "Servers are not on same axsis"
            self.logger.error(msg)
            raise Exception(msg)
        servo2.set_angle(servo1.get_angle())
        servo1.set_angle(servo1.get_middle_angle())
        servo1.move()
        servo2.move()

    def __distance_check(self, servo, new_angle, degree_diff=2):
        # TODO get degrees of difference from config file
        move = False
        current_angle = servo.get_angle()
        if new_angle > current_angle:
            if (new_angle - current_angle) > degree_diff:
                self.logger.debug(f"Going up, {new_angle} is greater than current {current_angle}, moving")
                move = True
            else:
                self.logger.debug(f"Going up, {new_angle} is less than current {current_angle}, not moving")
        elif new_angle < current_angle:
            if (current_angle - new_angle) > degree_diff:
                self.logger.debug(f"Going Down, {new_angle} is less than current {current_angle}, moving")
                move = True
            else:
                self.logger.debug(f"Going Down, {new_angle} is more than current {current_angle}, not moving")
        return move

    def move_servos(self):
        target = self.__find_person()
        if target != {}:
            # move "shoulders" first
            head_lr = self.__calc_servo(self.head_LR, target)
            head_ud = self.__calc_servo(self.head_UD, target)
            if self.__distance_check(self.head_LR, head_lr ) is True:
                self.head_LR.set_angle(head_lr)
                # don't use threading for now
                self.head_LR.move()
            if self.__distance_check(self.head_UD, head_ud) is True:
                self.head_UD.set_angle(head_ud)
                # dont use threading for now
                self.head_UD.move()
            # head should now be centered on the target
            # level the head and arm with body and rotation
            # x-axis
            self.__level_servos(self.head_LR, self.body_LR)
            self.__level_servos(self.head_UD, self.body_UD)

    def run(self):
        while self.stop is False:
            with self.lock:
                self.seen_data = self.eyes.get_results()
            self.move_servos()
            sleep(.2)


class LedHead:
    def __init__(self):
        # note the LED in the eye is GRB not RGB make sure to convert
        self.logger = setup_logger(self.__name__)
        self.pixels = neopixel.NeoPixel(board.D21, 1, brightness=1, auto_write=True)
        self.lh = ledhelper.LedHelper
        self.ani = ledhelper.NeoPixelAnimations(self.pixels, 1)
        self.swap = self.lh.rgb2grb_swap
        # power led
        self.hat = adafruit_pca9685.PCA9685(busio.I2C(board.SCL, board.SDA))
        self.pwm_led = self.hat.channels[4]
        self.hat.frequency = 60
        self.pwm_led.duty_cycle = 250
        # self.anger is a tuple which represents the major and minor anger, first being major, second being minor
        self.intensity = (.1, .1)

    def startup(self):
        # Do a startup sequence plusing the eye and head power LED from low to high...
        self.logger.debug("Startup Sequence")
        eye_led_thread = Thread(target=self.ani.intensity, args=(10, self.swap((255, 255, 0))))
        pwm_led_thread = Thread(target=self.ani.pwmintensity, args=(10, self.pwm_led))
        eye_led_thread.start()
        pwm_led_thread.start()
        eye_led_thread.join()
        pwm_led_thread.join()
        self.pwm_led.duty_cycle = 150
        self.pixels.brightness = self.intensity[0]
        self.pixels.autowrite = True
        self.pixels[0] = self.lh.adjust_brightness(self.swap((255, 255, 0)), self.intensity[1])
        self.pixels.show()

    def disco(self):
        # set intensity to half
        self.logger.debug("Triggered Disco Mode")
        self.intensity = (.8, .8)
        self.pixels.brightness = self.intensity[0]
        eye_led_thread = Thread(target=self.ani.rainbow_cycle, args=(.05, "GRB"))
        pwm_led_thread = Thread(target=self.ani.pwmintensity, args=(10, self.pwm_led))
        eye_led_thread.start()
        pwm_led_thread.start()
        eye_led_thread.join()
        pwm_led_thread.join()

    # TODO you left off considering how to handle intensity across the entire robot
    def angry_eye(self, steps=20, very_angry=True):
        self.logger.debug("Triggered Angry Eye")
        self.intensity = (.1, .1)
        self.pixels.brightness = self.intensity[0]
        self.pixels[0] = (255, 255, 0)
        self.pixels.show()
        sleep(1.4)
        anger = (255, 69, 0)
        if very_angry is True:
            anger = (139, 0, 0)
            self.pwm_led.duty_cycle = 65535
            self.intensity = (0.9, 0.9)
        self.pixels.brightness = self.intensity[0]
        eye_led_thread = Thread(target=self.ani.fade_color, args=((255, 255, 0), anger, steps, "GRB", self.intensity))
        eye_led_thread.start()
        eye_led_thread.join()


class DumbLEDController:
    def __init__(self, pwd_hat: classmethod, channel: int, duty_cycle: int = 100) -> None:
        self.led = pwd_hat.hat.channels[channel]
        self.led.duty_cycle = duty_cycle
        self.current_brightness = duty_cycle

    def set_brightness(self, brightness: int) -> bool:
        """
        Returns a Bool if it was able to set the brightness
        """
        ret = False
        if 0 < brightness <= 1000:
            self.led.duty_cycle = brightness
            self.current_brightness = brightness
            ret = True
        return ret

    def get_brightness(self) -> int:
        return self.current_brightness


class BodyTCPHandler(socketserver.BaseRequestHandler):
    @staticmethod
    def receive_all(socket_obj):
        """
        Receive all data from a socket until the connection is closed.
        :param socket_obj: socket.socket
        :return: bytes
        """
        buffer_size = 4096  # Define the size of the buffer
        data = b''  # This will store all the data received
        while True:
            part = socket_obj.recv(buffer_size)
            data += part
            if len(part) < buffer_size:
                # No more data to read or connection closed
                break
        return data

    def handle(self):
        callbacks = self.callbacks
        data = BodyTCPHandler.receive_all(self.request)
        commands = loads(data.decode('ascii'))
        for i in commands:
            #TODO Figure out how commands will work
            pass


class BodyTCPServer(socketserver.TCPServer):
    def __init__(self, server_address, request_handler_class, callbacks: dict):
        super().__init__(server_address, request_handler_class)
        self.callbacks = callbacks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evil Home Body Control Server')
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
        raise Exception("Unable to load file {}".format(args.conf[0]))
    handler_dicts = dict()
    ip = configp['DEFAULT']['body_server_ip']
    port = configp["DEFAULT"]["body_server_port"]
    #TODO YOU LEFT OFF TRYING TO FIGURE OUT HOW TO PASS SSERVER COMMANDS IN what they look like and how it calls shit... OG idea was multiple call backs based on commands
    # however since its all run by the GBODY anyways is that needed?
    body_server = BodyTCPServer((ip, port), BodyTCPHandler, handler_dicts)
    cam_x, cam_y = configp['DEFAULT']['camera_resolution'].split(',')
    gl = GBody(configp, cam_x, cam_y, Manager.dict(), Lock())
    gl.start()
