import random
from time import sleep
from threading import Thread
from json import loads, dumps
from typing import Dict, Callable, Tuple
from os import path
from glob import glob

# 3rd party
import paho.mqtt.client as mqtt
from adafruit_servokit import ServoKit
import neopixel
import adafruit_pca9685
import busio
import board
from digitalio import DigitalInOut, Direction
from PIL import Image, ImageDraw
from adafruit_rgb_display import st7789

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.MqttClient import MQTTClient
from glados_modules.LedHelper import LedHelper, NeoPixelAnimations


class GladosLCD(Thread, MQTTClient):
    def __init__(self, broker, location, port, animation_path, cs=board.CE0, dc=board.D25, rst=board.D24,
                 sck=board.SCK, mosi=board.MOSI, flip=False):
        # Configuration for CS and DC pins (these are PiTFT defaults):
        Thread.__init__(self)
        Thread.daemon = True
        self.__name__ =  f"{self.__class__.__name__}_{location}"
        self.logger = setup_logger(name=self.__name__)
        MQTTClient.__init__(self, broker, port)
        self.location: str = location
        self.animation_path: str = path.join(path.abspath(animation_path), "aperture_logo")
        self.cmd_topic: str = "body/lcd"
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd}
        cs_pin = DigitalInOut(cs)
        dc_pin = DigitalInOut(dc)
        reset_pin = DigitalInOut(rst)
        baud_rate = 24000000
        spi = busio.SPI(clock=sck, MOSI=mosi)
        self.disp = st7789.ST7789(spi, rotation=0, width=240, height=198, x_offset=0,
                                  y_offset=122, cs=cs_pin, dc=dc_pin, rst=reset_pin, baudrate=baud_rate)
        self.dot_on_positions: tuple = ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4),
                                        (3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2), (4, 3), (4, 4),
                                        (5, 1), (5, 2), (5, 3), (5, 4), (6, 1), (6, 2), (6, 3), (6, 4),
                                        (7, 1), (7, 2), (7, 3), (7, 4), (8, 2), (8, 4), (9, 1), (9, 3),
                                        (9, 4), (10, 3), (11, 1), (11, 2), (11, 4), (12, 2))
        self.rainbow = False
        self.g_color = 0
        self.counter = 1
        self.disp.spi_device.cs_active_value = False
        self.flip = flip
        self.breath_fast = False
        self.breathe_animation = True
        self.breathe_loop = True

    def handle_cmd(self, msg) -> None:
        j_msg = loads(msg.payload.decode())
        if j_msg.get("lcd", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic}, {j_msg}")
            if j_msg.get("cmd", "") == "set_breath":
                # command looks like nested {"cmd", "set_breath", options: { COMMAND DICT}}
                self.set_breath_options(j_msg["options"])
            elif j_msg.get("cmd", "") == "get_breath":
                # mark the location of response
                rsp = self.get_breath_options()
                rsp["location"] = self.location
                self.client.publish(rsp, dumps(self.get_breath_options()))

    def set_breath_options(self, breath_dict: dict) -> None:
        self.breath_fast = breath_dict['fast']
        self.breathe_animation = breath_dict['animation']
        self.rainbow = breath_dict['rainbow']

    def get_breath_options(self) -> dict:
        return {'fast': self.breath_fast, 'rainbow': self.rainbow,
                'animation': self.breathe_animation}

    def draw_image(self, times, color: tuple = (255, 0, 0)):
        c = 0
        while c <= times:
            image = self.create_custom_circles_image(circle_color=color)
            if self.flip is True:
                image = image.rotate(180)
            self.disp.image(image)
            c += 1

    def create_custom_circles_image(self, circle_color: tuple = (255, 0, 0)):
        """
        Creates an image with a 4x12 grid of circles where specific circles can be turned on (red) or off (black)
        based on a list of positions provided.

        Parameters:
        - on_positions: A list of tuples, where each tuple represents the (x, y) position of a red circle in the grid.
        """
        # Adjusted image dimensions for the LCD screen
        width, height = 240, 198
        background_color = (0, 0, 0)  # Black
        black_color = (0, 0, 0)  # Black
        # Number of circles and circle radius
        num_circles_x = 12
        num_circles_y = 4
        radius = 5
        # Create a new black image
        image = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(image)
        # Calculate spacing between circles
        spacing_x = (width - 2 * radius) // (num_circles_x - 1) - 5
        spacing_y = (height - 2 * radius) // (num_circles_y - 1) - 48
        # Adjust vertical starting point to center the grid
        start_x = (width - ((num_circles_x - 1) * spacing_x + 2 * radius)) // 2
        start_y = (height - ((num_circles_y - 1) * spacing_y + 2 * radius)) // 2 - 15

        if self.rainbow is True:
            circle_color = LedHelper.color_wheel(self.g_color)
            if self.g_color == 255:
                self.g_color = 0
            else:
                self.g_color += 1

        # Draw the circles, turning specific circles red based on on_positions
        for x in range(num_circles_x):
            for y in range(num_circles_y):
                cd = LedHelper.adjust_brightness(circle_color, random.choice([x / 10.0 for x in range(6, 9)]))
                center_x = start_x + radius + x * spacing_x
                center_y = start_y + radius + y * spacing_y
                # Determine the color of the circle based on its position in on_positions list
                if (x + 1, y + 1) in self.dot_on_positions:
                    current_color = cd
                else:
                    current_color = black_color
                # Draw the circle with the determined color
                draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                             fill=current_color)
        return image

    def __display_frame(self, filename):
        image = Image.open(filename)
        # Scale the image to the smaller screen dimension
        image_ratio = image.width / image.height
        image = image.resize((160, int((160 / image_ratio))), Image.BICUBIC)
        if image.height < 80:
            # create new canvas (color format, size, background color) default is aperture orange
            new_canvas = Image.new("RGB", (160, 80), "#ff9a00")
            vertical_offset = (80 - image.height) // 2
            new_canvas.paste(image, (0, vertical_offset))
            if self.flip is True:
                new_canvas = new_canvas.rotate(180)
            self.disp.image(new_canvas)

    def aperture_animation(self, ftype='.bmp'):
        # play an animation of the aperture science logo
        frame_filenames = sorted(glob(path.join(self.animation_path, "*{}".format(ftype))))
        for filename in frame_filenames:
            self.__display_frame(filename)
            sleep(1/29.97)

    def breathe(self):
        self.breathe_loop = True
        up = True
        self.counter = 1
        slpm = 0.12
        slptb = 0.18
        tb = 12
        mid = 5
        if self.breath_fast is True:
            slpm = 0.
            slptb = 0
            tb = 0
            mid = 0
        while self.breathe_loop is True:
            self.draw_image(times=mid)
            if self.breathe_animation is True:
                if self.breath_fast is False:
                    sleep(slpm)
                if self.counter > 12:
                    up = False
                    self.draw_image(times=tb)
                    if self.breath_fast is False:
                        sleep(slptb)
                if self.counter <= 1:
                    up = True
                    self.draw_image(times=tb)
                    if self.breath_fast is False:
                        sleep(slptb)
                if up is True:
                    self.counter += 1
                else:
                    self.counter -= 1
            else:
                self.counter = 12
                sleep(.2)

    def run(self):
        self.aperture_animation()
        self.breathe()

    def stop(self):
        # end all loops so you can join thread
        self.breathe_loop = False


class Gservo(Thread, MQTTClient):
    def __init__(self, location: str, skit: ServoKit, axis: str, servo_range: Tuple[int, int] = (),
                 max_angle: int = 90, broker: str = 'localhost', port: int = 1883) -> None:
        Thread.__init__(self)
        Thread.daemon = True
        self.__name__ = f"{self.__class__.__name__}_{location}"
        self.logger = setup_logger(name=self.__name__)
        MQTTClient.__init__(self, broker, port)
        self.location: str = location
        self.cmd_topic: str = "body/servo"
        self.intensity_topic: str = "intensity"
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd,
                                                   self.intensity_topic: self.handle_intensity}
        self.min_angle: int = 0
        self.skit: ServoKit = skit
        self.speed: int = 5
        self.max_angle: int = max_angle
        self.middle_angle: int = int(self.max_angle / 2)
        self.angle: int = self.middle_angle
        self.current_angle: int = self.angle
        self.first_boot: bool = True
        self.move()
        self.exec_command: bool = False
        self.moving: bool = False
        self.axis: str = axis.lower()
        self.stop_bool: bool = False
        if servo_range == ():
            self.allowed_servo_range: Dict[str, int] = {"min_travel": 0, "max_travel": max_angle}
        else:
            self.allowed_servo_range: Dict[str, int] = {"min_travel": servo_range[0], "max_travel": servo_range[1]}

    def handle_cmd(self, msg: mqtt.MQTTMessage) -> None:
        j_msg = loads(msg.payload.decode())
        if j_msg.get("servo", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic}, {j_msg}")
            angle: int = int(j_msg.get("angle", self.middle_angle))
            speed: int = int(j_msg.get("speed", self.speed))
            self.set_speed_angle((speed, angle), execute=True)

    def handle_intensity(self, msg: mqtt.MQTTMessage) -> None:
        # TODO figure out update commands
        pass

    def get_max_angle(self) -> int:
        return self.max_angle

    def get_middle_angle(self) -> int:
        return self.middle_angle

    def set_speed(self, speed: int) -> None:
        if speed >= 10:
            speed = 10
        if speed <= 1:
            speed = 1
        self.speed = round(speed)
        self.logger.debug(f"Speed set to {self.speed}")

    def set_angle(self, angle: int) -> None:
        max_angle = self.allowed_servo_range["max_travel"]
        min_angle = self.allowed_servo_range["min_travel"]
        if angle >= max_angle:
            self.angle = max_angle
            self.logger.debug(f"{angle} is above {max_angle}, setting to {max_angle}")
        elif angle <= min_angle:
            self.angle = min_angle
            self.logger.debug(f"{angle} is below {min_angle}, setting to {min_angle}")
        else:
            self.angle = angle
            self.logger.debug(f"Angle set to {self.angle}")

    def set_speed_angle(self, speed_angle: Tuple[int, int], execute: bool = False) -> None:
        self.set_speed(speed_angle[0])
        self.set_angle(speed_angle[1])
        if execute:
            self.exec_command = True

    def get_angle(self) -> int:
        return self.current_angle

    def execute(self) -> None:
        self.exec_command = True

    def __get_direction_speed(self) -> range:
        rtn: range = range(0, 0)
        if self.angle > self.current_angle:
            rtn = range(self.current_angle, (self.angle + 1), self.speed)
        if self.angle < self.current_angle:
            rtn = range(self.current_angle, (self.angle + 1), (self.speed * -1))
        return rtn

    def __increment(self) -> None:
        for s in self.__get_direction_speed():
            self.skit.angle = s
            sleep(.1)
        self.current_angle = self.angle

    def get_moving_status(self) -> bool:
        return self.moving

    def move(self) -> None:
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

    def run(self) -> None:
        while self.stop_bool is False:
            if self.exec_command is True:
                self.move()
                self.exec_command = False
            else:
                sleep(.1)
        self.client.loop_stop()

    def stop(self) -> None:
        self.stop_bool = True


class LedShoulders(MQTTClient):
    def __init__(self, broker: str = 'localhost', port: int = 1883) -> None:
        self.__name__ = "LED_Shoulder_Controller"
        MQTTClient.__init__(self, broker, port)
        self.logger = setup_logger(self.__name__)
        led_num: int = 64
        self.pixels = neopixel.NeoPixel(board.D12, led_num, brightness=1, auto_write=True, pixel_order=neopixel.RGB)
        self.lh = LedHelper
        self.ani = NeoPixelAnimations(self.pixels, led_num)
        self.swap = self.lh.rgb2grb_swap
        self.intensity: Tuple[float, float] = (0.5, 0.5)
        self.stripes: list = list()
        self.stripes.extend(range(8, 24))
        self.stripes.extend(range(40, 56))
        self.cmd_topic: str = "body/led"
        self.intensity_topic: str = "intensity"
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd,
                                                   self.intensity_topic: self.handle_intensity}
        self.location: str = "shoulder_led"
        self.animations: Dict[str, Callable] = {"startup": self.startup, "disco": self.disco, "twinkle": self.twinkle}
        self.twinkle_loop: bool = False

    def handle_cmd(self, msg: mqtt.MQTTMessage) -> None:
        j_msg = loads(msg.payload.decode())
        if j_msg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {j_msg}")
            if j_msg[self.location]['command'] in self.animations.keys():
                self.animations[j_msg[self.location]['command']]()

    def handle_intensity(self, msg: mqtt.MQTTMessage) -> None:
        # TODO figure out update commands
        j_msg = loads(msg.payload.decode())
        if j_msg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {j_msg}")
            self.intensity = j_msg["intensity"]

    def startup(self) -> None:
        self.twinkle_loop = False
        for p in range(0, 63):
            self.pixels[p] = self.lh.adjust_brightness((255, 0, 0), self.intensity[0])
            sleep(.2)

    def disco(self) -> None:
        self.twinkle_loop = False
        self.logger.debug("Triggered Disco Mode")
        self.pixels.brightness = self.intensity[0]
        eye_led_thread = Thread(target=self.ani.rainbow_cycle, args=(.05, "RGB"))
        eye_led_thread.start()

    def twinkle(self) -> None:
        self.twinkle_loop = True
        self.logger.debug("Triggered Disco Mode")
        self.pixels.brightness = self.intensity[0]
        while self.twinkle_loop:
            for p in self.stripes:
                cd = self.lh.adjust_brightness((255, 0, 0), random.choice([x / 10.0 for x in range(1, 9)]))
                self.pixels[p] = cd
            self.pixels.show()


class DumbLEDController(Thread):
    """
    LED Controller for pca9685
    """
    def __init__(self, channel: int, duty_cycle: int = 100) -> None:
        Thread.__init__(self)
        self.daemon = True
        self.hat = adafruit_pca9685.PCA9685(busio.I2C(board.SCL, board.SDA))
        self.led = self.hat.channels[channel]
        self.led.duty_cycle = duty_cycle
        self.current_brightness = duty_cycle
        self.stop = False
        self.animation = self.null_animation

    def null_animation(self):
        """
        NUll animation the thread can chew on when we want to do nothing
        """
        sleep(.1)

    def set_stop(self):
        self.stop = True

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

    def set_twinkle(self):
        """
        Set twinkle animation
        """
        self.animation = self.twinkle_animation

    def get_brightness(self) -> int:
        return self.current_brightness

    def twinkle_animation(self):
        """
        pick a random number between the range and set the value
        then sleep for a random amount of time before return
        """
        self.set_brightness(random.randrange(50, 500))
        sleep(random.uniform(.1, 1))

    def pulse_animation(self):
        """
        Pulse the led from low to high and back to low
        """
        # each iteration should take
        for i in range(1, 800):
            self.set_brightness(i)
            sleep(0.0025)

    def run(self):
        while self.stop is False:
            self.animation()


class LedHead(MQTTClient):
    def __init__(self, broker: str = 'localhost', port: int = 1883) -> None:
        self.__name__ = "Head_LED_Controller"
        MQTTClient.__init__(self, broker, port)
        self.logger = setup_logger()
        self.pixels = neopixel.NeoPixel(board.D18, 1, brightness=1, auto_write=True, pixel_order=neopixel.RGB)
        self.ani = NeoPixelAnimations(self.pixels, 1)
        self.swap = LedHelper.rgb2grb_swap
        self.hat = adafruit_pca9685.PCA9685(busio.I2C(board.SCL, board.SDA))
        self.pwm_led = self.hat.channels[4]
        self.hat.frequency = 60
        self.pwm_led.duty_cycle = 250
        self.intensity: Tuple[float, float] = (.1, .1)
        self.cmd_topic: str = "body/led"
        self.intensity_topic: str = "intensity"
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd,
                                                   self.intensity_topic: self.handle_intensity}
        self.location: str = "eye_led"
        self.animations: Dict[str, Callable] = {"startup": self.startup, "disco": self.disco,
                                                "angry_eye": self.angry_eye, "normal_eye": self.normal_eye}
        self.yellow_eye: Tuple[int, int, int] = (246, 216, 121)

    def handle_cmd(self, msg: mqtt.MQTTMessage) -> None:
        j_msg = loads(msg.payload.decode())
        if j_msg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {j_msg}")
            if j_msg[self.location]['command'] in self.animations.keys():
                self.animations[j_msg[self.location]['command']]()

    def handle_intensity(self, msg: mqtt.MQTTMessage) -> None:
        # TODO figure out update commands
        j_msg = loads(msg.payload.decode())
        if j_msg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {j_msg}")
            self.intensity = j_msg["intensity"]

    def startup(self) -> None:
        self.logger.debug("Startup Sequence")
        eye_led_thread = Thread(target=self.ani.intensity, args=(10, self.yellow_eye))
        pwm_led_thread = Thread(target=self.ani.pwmintensity, args=(10, self.pwm_led))
        eye_led_thread.start()
        pwm_led_thread.start()
        eye_led_thread.join()
        pwm_led_thread.join()
        self.normal_eye()

    def disco(self) -> None:
        self.logger.debug("Triggered Disco Mode")
        self.pixels.brightness = self.intensity[0]
        eye_led_thread = Thread(target=self.ani.rainbow_cycle, args=(.05, "RGB"))
        pwm_led_thread = Thread(target=self.ani.pwmintensity, args=(10, self.pwm_led))
        eye_led_thread.start()
        pwm_led_thread.start()
        eye_led_thread.join()
        pwm_led_thread.join()

    def angry_eye(self, steps: int = 20, very_angry: bool = True) -> None:
        self.logger.debug("Triggered Angry Eye")
        self.intensity = (.1, .1)
        self.pixels.brightness = self.intensity[0]
        self.pixels[0] = (255, 255, 0)
        self.pixels.show()
        sleep(1.4)
        anger: Tuple[int, int, int] = (255, 69, 0)
        if very_angry:
            anger = (139, 0, 0)
            self.pwm_led.duty_cycle = 65535
            self.intensity = (0.9, 0.9)
        self.pixels.brightness = self.intensity[0]
        eye_led_thread = Thread(target=self.ani.fade_color, args=((255, 255, 0), anger, steps, "RGB", self.intensity))
        eye_led_thread.start()
        eye_led_thread.join()

    def normal_eye(self) -> None:
        self.pwm_led.duty_cycle = 150
        self.pixels.brightness = self.intensity[0]
        self.pixels.autowrite = True
        self.pixels[0] = LedHelper.adjust_brightness(self.yellow_eye, self.intensity[1])
        self.pixels.show()

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
    led_head.startup()
    while True:
        sleep(1)
