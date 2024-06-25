import random
from time import sleep, time
from threading import Thread
from json import loads, dumps

# 3rd party
import paho.mqtt.client as mqtt
from adafruit_servokit import ServoKit
import neopixel
import ledhelper
import adafruit_pca9685
import busio
import board

# glados imports
from glog_conifig import setup_logger


class Gservo(Thread):
    def __init__(self, location, skit, axis, servo_range: tuple = (), max_angle=90,
                 broker='localhost', port=1883, topic="body/servo"):
        Thread.__init__(self)
        Thread.daemon = True
        self.logger = setup_logger(name=f"{self.__class__.__name__}_{location}")
        self.location = location
        self.broker = broker
        self.port = port
        self.cmd_topic = "body/servo"
        self.intensity_topic = "intensity"
        self.topic_handler = {self.cmd_topic: self.handle_cmd, self.intensity_topic: self.handle_intensity}
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
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
        self.stop_bool = False
        if servo_range == ():
            self.allowed_servo_range = {"min_travel": 0, "max_travel": max_angle}
        else:
            self.allowed_servo_range = {"min_travel": servo_range[0], "max_travel": servo_range[1]}
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()
    
    def on_connect(self, client, userdata, flags, rc):
        self.logger.debug(f"Connecting to {self.broker}:{self.port} on channel {self.cmd_topic}")
        self.client.subscribe(self.cmd_topic)
        self.client.subscribe(self.intensity_topic)

    def on_message(self, client, userdata, msg):
        if msg.topic in self.topic_handler:
            self.topic_handler[msg.topic](msg)

    def handle_cmd(self, msg):
        jmsg = loads(msg.payload.decode())
        if jmsg.get("servo", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {jmsg}")
            # message is the correct servo
            angle = int(jmsg.get("angle", self.middle_angle))
            speed = int(jmsg.get("speed", self.speed))
            self.set_speed_angle((speed, angle), execute=True)

    def handle_intensity(self, msg):
        #TODO figure out update commands
        pass

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
            self.skit.angle = self.angle
            sleep(.3)
            self.moving = True
            self.logger.debug(f"moving to {self.angle}")
            self.current_angle = self.angle
            self.moving = False
            self.first_boot = False
        else:
            if self.angle != self.current_angle:
                self.moving = True
                self.logger.debug(f"moving to {self.angle}")
                self.__increment()
                self.moving = False

    def run(self):
        while self.stop_bool is False:
            if self.exec_command is True:
                self.move()
                self.exec_command = False
            else:
                sleep(.1)
        self.client.loop_stop()

    def stop(self):
        self.stop_bool = True


class LedShoulders:
    def __init__(self, broker='localhost', port=1883):
        # GPIO 12 hookup
        self.logger = setup_logger(self.__name__)
        led_num = 64
        self.pixels = neopixel.NeoPixel(board.D12, led_num, brightness=1, auto_write=True, pixel_order=neopixel.RGB)
        self.lh = ledhelper.LedHelper
        self.ani = ledhelper.NeoPixelAnimations(self.pixels, led_num)
        self.swap = self.lh.rgb2grb_swap
        self.intensity = (0.5, 0.5)
        self.stripes = list()
        self.stripes.extend(range(8,24))
        self.stripes.extend(range(40,56))
        self.broker = broker
        self.port = port
        self.client = mqtt.Client()
        self.cmd_topic = "body/led"
        self.intensity_topic = "intensity"
        self.topic_handler = {self.cmd_topic: self.handle_cmd, self.intensity_topic: self.handle_intensity}
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()
        self.location = "shoulder_led"
        self.animations = {"startup": self.startup, "disco": self.disco, "twinkle": self.twinkle}
        self.twinkle_loop = False

    def on_connect(self, client, userdata, flags, rc):
        self.logger.debug(f"Connecting to {self.broker}:{self.port} on channel {self.cmd_topic}")
        self.client.subscribe(self.cmd_topic)
        self.client.subscribe(self.intensity_topic)

    def on_message(self, client, userdata, msg):
        if msg.topic in self.topic_handler:
            self.topic_handler[msg.topic](msg)

    def handle_cmd(self, msg):
        jmsg = loads(msg.payload.decode())
        if jmsg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {jmsg}")
            if jmsg[self.location]['command'] in self.animations.keys():
                jmsg[self.location]['command']()

    def handle_intensity(self, msg):
        # TODO figure out update commands
        jmsg = loads(msg.payload.decode())
        if jmsg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {jmsg}")
            self.intensity = jmsg["intensity"]

    def startup(self):
        # do a knight rider style startup
        self.twinkle_loop = False
        for p in range(0, 63):
            self.pixels[p] = self.lh.adjust_brightness((255,0, 0), self.intensity[0])
            sleep(.2)

    def disco(self):
        self.twinkle_loop = False
        self.logger.debug("Triggered Disco Mode")
        self.pixels.brightness = self.intensity[0]
        eye_led_thread = Thread(target=self.ani.rainbow_cycle, args=(.05, "RGB"))
        eye_led_thread.start()

    def twinkle(self):
        self.twinkle_loop = True
        self.logger.debug("Triggered Disco Mode")
        self.pixels.brightness = self.intensity[0]
        while self.twinkle_loop:
            for p in self.stripes:
                cd = self.lh.adjust_brightness((255, 0, 0), random.choice([x / 10.0 for x in range(1, 9)]))
                self.pixels[p] = cd
            self.pixels.show()


class LedHead:
    def __init__(self, broker='localhost', port=1883):
        self.logger = setup_logger(self.__name__)
        self.pixels = neopixel.NeoPixel(board.D18, 1, brightness=1, auto_write=True, pixel_order=neopixel.RGB)
        self.lh = ledhelper.LedHelper
        self.ani = ledhelper.NeoPixelAnimations(self.pixels, 1)
        self.swap = self.lh.rgb2grb_swap
        # power led
        self.hat = adafruit_pca9685.PCA9685(busio.I2C(board.SCL, board.SDA))
        self.pwm_led = self.hat.channels[4]
        self.hat.frequency = 60
        self.pwm_led.duty_cycle = 250
        # self.intensity is a tuple which represents the major and minor anger, first being major, second being minor
        self.intensity = (.1, .1)
        self.broker = broker
        self.port = port
        self.client = mqtt.Client()
        self.cmd_topic = "body/led"
        self.intensity_topic = "intensity"
        self.topic_handler = {self.cmd_topic: self.handle_cmd, self.intensity_topic: self.handle_intensity}
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()
        self.location = "eye_led"
        self.animations = {"startup": self.startup, "disco": self.disco, "angry_eye": self.angry_eye,
                           "normal_eye": self.normal_eye}
        self.yellow_eye = (246, 216, 121)

    def on_connect(self, client, userdata, flags, rc):
        self.logger.debug(f"Connecting to {self.broker}:{self.port} on channel {self.cmd_topic}")
        self.client.subscribe(self.cmd_topic)
        self.client.subscribe(self.intensity_topic)

    def on_message(self, client, userdata, msg):
        if msg.topic in self.topic_handler:
            self.topic_handler[msg.topic](msg)

    def handle_cmd(self, msg):
        jmsg = loads(msg.payload.decode())
        if jmsg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {jmsg}")
            if jmsg[self.location]['command'] in self.animations.keys():
                jmsg[self.location]['command']()

    def handle_intensity(self, msg):
        # TODO figure out update commands
        jmsg = loads(msg.payload.decode())
        if jmsg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {jmsg}")
            self.intensity = jmsg["intensity"]

    def startup(self):
        # Do a startup sequence plusing the eye and head power LED from low to high...
        self.logger.debug("Startup Sequence")
        eye_led_thread = Thread(target=self.ani.intensity, args=(10, self.yellow_eye))
        pwm_led_thread = Thread(target=self.ani.pwmintensity, args=(10, self.pwm_led))
        eye_led_thread.start()
        pwm_led_thread.start()
        eye_led_thread.join()
        pwm_led_thread.join()
        self.normal_eye()

    def disco(self):
        # set intensity to half
        self.logger.debug("Triggered Disco Mode")
        self.pixels.brightness = self.intensity[0]
        eye_led_thread = Thread(target=self.ani.rainbow_cycle, args=(.05, "RGB"))
        pwm_led_thread = Thread(target=self.ani.pwmintensity, args=(10, self.pwm_led))
        eye_led_thread.start()
        pwm_led_thread.start()
        eye_led_thread.join()
        pwm_led_thread.join()

    def angry_eye(self, steps=20, very_angry=True):
        self.logger.debug("Triggered Angry Eye")
        # TODO review this later
        # ignore global intensity because we want to ramp up to it...
        self.intensity = (.1, .1)
        self.pixels.brightness = self.intensity[0]
        self.pixels[0] = (255, 255, 0)
        self.pixels.show()
        sleep(1.4)
        anger = (255, 69, 0)
        # TODO figure out how to handle this via msg system
        if very_angry is True:
            anger = (139, 0, 0)
            self.pwm_led.duty_cycle = 65535
            self.intensity = (0.9, 0.9)
        self.pixels.brightness = self.intensity[0]
        eye_led_thread = Thread(target=self.ani.fade_color, args=((255, 255, 0), anger, steps, "RGB", self.intensity))
        eye_led_thread.start()
        eye_led_thread.join()

    def normal_eye(self):
        self.pwm_led.duty_cycle = 150
        self.pixels.brightness = self.intensity[0]
        self.pixels.autowrite = True
        self.pixels[0] = self.lh.adjust_brightness(self.yellow_eye, self.intensity[1])
        self.pixels.show()

# NOTE you need to code up a class for the "shoulders"
# NOTE you also need to code up a class for the Lamp portion its self...

# on the pi5 code, need to have classes to read from LIDAR sensor to channel..
# also need class to read temp senders and have them take action
# bird detection to kill external power? how will that work...

if __name__ == "__main__":
    ip = '192.168.86.52'
    kit = ServoKit(channels=16)
    led_head = LedHead(broker=ip)
    body_LR = Gservo(location='body_left_right', skit=kit.servo[0], axis='x', max_angle=180, broker=ip)
    body_UD = Gservo(location='body_up_down', skit=kit.servo[1], axis='y', max_angle=180, broker=ip)
    head_UD = Gservo(location='head_left_right', skit=kit.servo[2], axis='y', max_angle=180, broker=ip)
    head_LR = Gservo(location='head_up_down', skit=kit.servo[3], axis='x', max_angle=180, broker=ip)
    body_LR.start()
    body_UD.start()
    head_LR.start()
    head_UD.start()
    while True:
        sleep(1)
