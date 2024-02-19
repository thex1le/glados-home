# standard imports
import time
from threading import Thread
# 3rd party imports
import board
import adafruit_dotstar as dotstar
from adafruit_led_animation.animation.sparklepulse import SparklePulse
from adafruit_led_animation.animation.sparkle import Sparkle
from adafruit_led_animation.animation.comet import Comet
from adafruit_led_animation.animation.rainbowcomet import RainbowComet
from adafruit_led_animation.animation.rainbowchase import RainbowChase
from adafruit_led_animation.animation.chase import Chase
from adafruit_led_animation.animation.rainbow import Rainbow
from adafruit_led_animation.sequence import AnimationSequence
from adafruit_led_animation import helper
from adafruit_led_animation.color import AMBER, AQUA, BLACK, BLUE, CYAN, GOLD, GREEN, JADE, MAGENTA, OLD_LACE, ORANGE, PINK, PURPLE, RAINBOW, RED, RGBW_WHITE_RGB, RGBW_WHITE_RGBW, RGBW_WHITE_W, TEAL, WHITE, YELLOW


class EggTimer(Thread):
    def __init__(self, duration_in_seconds, callback):
        Thread.__init__(self)
        Thread.daemon = True
        self.duration = duration_in_seconds
        self.start_time = None
        self.is_running = False
        self.callback = callback

    def tstart(self):
        if not self.is_running:
            self.start_time = time.time()
            self.is_running = True
            print("Body timer started for {} seconds.".format(self.duration))

    def stop(self):
        if self.is_running:
            elapsed_time = time.time() - self.start_time
            remaining_time = max(0, self.duration - elapsed_time)
            self.is_running = False

    def check_remaining_time(self):
        rtn = {"remain": 0, "complete":False}
        if self.is_running:
            elapsed_time = time.time() - self.start_time
            remaining_time = max(0, self.duration - elapsed_time)
            rtn["remain"] = remaining_time
            if remaining_time == 0:
                rtn["remian"] = 0
                rtn["complete"] = True
                self.callback()
        else:
            rtn["remain"] = 0
            rtn["complete"] = True
        return rtn
    
    def run(self):
        self.tstart()
        while True:    
            r = self.check_remaining_time()
            print(r)
            if r["complete"] is True:
                break
            time.sleep(.1)
            

class rgbwLEDs:
    # control adafruit HD107 LED's
    def __init__(self, led_num: int, brightness: float ): -> None
        self.led_num = led_num
        self.bright = brightness
        self.leds = dotstar.DotStar(board.SCK, board.MOSI, led_num)
        self.leds.brightness_red = self.bright
        self.leds.brightness_green = self.bright
        self.leds.brightness_blue = self.bright

    def adjust_brightness(self, color: tuple, brightness_factor: float): -> tuple
        """Adjust the brightness of a color.

        Args:
            color (tuple): The original color as a tuple of (R, G, B).
            brightness_factor (float): The brightness factor, between 0.0 (off) and 1.0 (full brightness).

        Returns:
            tuple: The adjusted color as a tuple of (R, G, B).
        """
        return tuple(int(c * brightness_factor) for c in color)
    
    def set_led(self, led_num: int, color: tuple, brightness: float): -> None
        """Set and led color and brightness"""
        self.leds[led_num] = adjust_brightness(color)


class GLaDOSBody(Thread):
    """
    All the controls for GLaDOS Body movment
    """
    def __init__(self): -> None
        # set the head leds to max brightness as we will dim them as needed
        Thread.__init__(self)
        Thread.daemon = True
        self.annoyance = 0
        self.headleds = rgbwLEDs(2, 1)
        self.annoance_timer = False

    def 
    # you left off here writing the annoance timer and how it incriments and decriments 

    def eye(self, color: tuple, brightness: float = 0.5): -> None
        self.headleds.set_led(0, YELLOW, brightness)
    
    def head_power(self, color: tuple, brightness: float = 0.2: -> None
        self.headleds.set_led(1, RED, brightness)
    
    def moreannoyed(self, level: int=1):
        if self.annoance_timer = False:
                # start timer....
                pass
        self.annoyance += level

    def annoance_callback(self, level: int= -1):
        self.annoyance -= level
        if self.annoance <= 0:
                   self.annoance_timer = False

# Original colors
yellow = (255, 255, 0)
red = (255, 0, 0)

# Adjust brightness for individual LEDs
yellow_dimmed = adjust_brightness(yellow, 0.5)  # Dim yellow to 50% brightness
red_dimmed = adjust_brightness(red, 0.75)  # Dim red to 75% brightness

# Set the colors to the LEDs
pixels[0] = yellow_dimmed
pixels[1] = red_dimmed
pixels[35] = RED
pixels[70] = RED
