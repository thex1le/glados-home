import random
from time import sleep
from glob import glob
from os import path
from threading import Thread
# 3rd party imports
from digitalio import DigitalInOut, Direction
import busio
import board
from PIL import Image, ImageDraw
from adafruit_rgb_display import st7735  # pylint: disable=unused-import


class GladosLCD(Thread):
    def __init__(self, cs=board.CE0, dc=board.D25, rst=board.D24, sck=board.SCK, mosi=board.MOSI, flip=False):
        # Configuration for CS and DC pins (these are PiTFT defaults):
        Thread.__init__(self)
        Thread.daemon = True
        cs_pin = DigitalInOut(cs)
        dc_pin = DigitalInOut(dc)
        reset_pin = DigitalInOut(rst)
        baud_rate = 24000000
        spi = busio.SPI(clock=sck, MOSI=mosi)
        self.breath_stop = False
        self.disp = st7735.ST7735R(spi, rotation=90, invert=True, width=80, x_offset=26, 
                                   y_offset=1, cs=cs_pin, dc=dc_pin, rst=reset_pin, baudrate=baud_rate)
        self.on_positions = [(1, 1), (1,2), (1,3), (1,4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3,2), (3,3), (3,4), 
                             (4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (5, 3), (5, 4), 
                             (6, 1), (6, 2), (6, 3), (6, 4), (7, 1), (7, 2), (7, 3), (7, 4), (8, 2), (8, 4), 
                             (9, 1), (9, 3), (9, 4), (10, 3), (11, 1), (11, 2), (11, 4), (12, 2)]
        self.rainbow = False
        self.gcolor = 0
        self.counter = 1
        self.disp.spi_device.cs_active_value = False
        self.flip = flip
        self.breath_fast = False
        self.breathe_animation = True

    def set_breath_options(self, breath_dict:dict):
        self.breath_fast = breath_dict['fast']
        self.breathe_animation = breath_dict['animation']
        self.rainbow = breath_dict['rainbow']

    def get_breath_options(self) -> dict:
        return {'fast': self.breath_fast, 'rainbow': self.rainbow,
                'animation': self.breathe_animation}

    def adjust_brightness(self, color, brightness_factor):
        return tuple(int(c * brightness_factor) for c in color)

    def draw_image(self, times, color=(255, 0 , 0)):
        c = 0
        while c <= times:
            image = self.create_custom_circles_image(circle_color=color)
            if self.flip is True:
                image = image.rotate(180)
            self.disp.image(image)
            c += 1

    @staticmethod
    def color_wheel(pos):
        # return a position on the color wheel based on input 0 to 255
        if pos < 0 or pos > 255:
            r = g = b = 0
        elif pos < 85:
            r = int(pos * 3)
            g = int(255 - pos * 3)
            b = 0
        elif pos < 170:
            pos -= 85
            r = int(255 - pos * 3)
            g = 0
            b = int(pos * 3)
        else:
            pos -= 170
            r = 0
            g = int(pos * 3)
            b = int(255 - pos * 3)
        return r, g, b

    # Define a new function to selectively turn a red dot on or off in the 4x12 grid
    def create_custom_circles_image(self, circle_color=(255, 0, 0)):
        # Adjusted image dimensions for the LCD screen
        width, height = 160, 80
        background_color = (0, 0, 0)  # Black
        black_color = (0, 0, 0)  # Black
        # Number of circles and circle radius
        num_circles_x = 12
        num_circles_y = 4
        radius = 4.5
        # Create a new black image
        image = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(image)
        # Calculate spacing between circles
        spacing_x = width // (num_circles_x + 1)
        spacing_y = height // (num_circles_y + 1)
        if self.rainbow is True:
            circle_color = GladosLCD.color_wheel(self.gcolor)
            if self.gcolor == 255:
                self.gcolor = 0
            else:
                self.gcolor += 1
        # Draw the circles, turning specific circles red based on on_positions
        for x in range(1, num_circles_x + 1):
            for y in range(1, num_circles_y + 1):
                cd = self.adjust_brightness(circle_color, random.choice([x / 10.0 for x in range(6, 9)]))
                center_x = x * spacing_x
                center_y = y * spacing_y
                # Determine the color of the circle based on its position in on_positions list
                if (x , y) in self.on_positions and x <= self.counter:
                    current_color = cd 
                else:
                    current_color = black_color
                # Draw the circle with the determined color
                draw.ellipse([center_x - radius, center_y - radius, center_x + radius,
                              center_y + radius], fill=current_color)
        return image

    def __display_frame(self, filename):
        image = Image.open(filename)
        # Scale the image to the smaller screen dimension
        image_ratio = image.width / image.height
        image = image.resize((160, int((160/ image_ratio))), Image.BICUBIC)
        if image.height < 80:
            # create new canvas (color format, size, background color) default is aperture orange
            new_canvas = Image.new("RGB", (160, 80), "#ff9a00")
            vertical_offset = (80 - image.height) // 2
            new_canvas.paste(image, (0, vertical_offset))
            if self.flip is True:
                new_canvas = new_canvas.rotate(180)
            self.disp.image(new_canvas)

    def aperture_animation(self, imagespath='aperture_logo', ftype='.bmp'):
        # play an animation of the aperture science logo
        frame_filenames = sorted(glob(path.join(imagespath, "*{}".format(ftype))))
        for filename in frame_filenames:
            self.__display_frame(filename)
            sleep(1/29.97)

    def breathe(self, fast=False, breathe=True):
        self.breath_stop = False
        up = True
        self.counter = 1
        slpm = 0.1
        slptb = 0.15
        tb = 8
        mid = 3
        if fast is True:
            slpm = 0.
            slptb = 0
            tb = 0
            mid = 0
        while self.breath_stop is False:
            self.draw_image(times=mid)
            if breathe is True:
                if fast is False:
                    sleep(slpm)
                if self.counter > 12:
                    up = False
                    self.draw_image(times=tb)
                    if fast is False:
                        sleep(slptb)
                if self.counter <= 1:
                    up = True
                    self.draw_image(times=tb)
                    if fast is False:
                        sleep(slptb)
                if up is True:
                    self.counter += 1
                else:
                    self.counter -= 1
            else:
                self.counter = 12
                sleep(.2)

if __name__ == "__main__":
    # glcd0 is the big right side
    glcd0 = GladosLCD()
    # glcd1 is the little left side
    glcd1 = GladosLCD(cs=board.D18, rst=board.D5, dc=board.D6, sck=board.SCK_1, mosi=board.MOSI_1, flip=True)
    gl1t = Thread(target = glcd1.aperture_animation, args=())
    gl1t.start()
    glcd0.rainbow=True
    gl0t = Thread(target = glcd0.aperture_animation, args=())
    gl0t.start()
    gl1t.join()
    gl0t.join()
    glcd1.rainbow = True
    g1breath = Thread(target=glcd1.breathe, kwargs={"fast":False, "breathe":True})
    g1breath.start()
    glcd0.breathe(fast=False, breathe=False)
