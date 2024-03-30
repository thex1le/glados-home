from time import sleep, time


class LedHelper:
    @staticmethod
    def adjust_brightness(color: tuple, brightness_factor: float) -> tuple:
        """
        Adjust the brightness of a color.
        color (tuple): The original color as a tuple of (R, G, B).
        brightness_factor (float): The brightness factor, between 0.0 (off) and 1.0 (full brightness).

        Returns: tuple: The adjusted color as a tuple of (R, G, B).
        """
        return tuple(int(c * brightness_factor) for c in color)
    
    @staticmethod
    def rgb2grbswap(color: tuple) -> tuple:
        """
        Convert swap postion of R and G to convert from RGB to GRB or GRB to RGB
        color (tuple)
        Returns: tuple of color
        """
        return (color[1], color[0], color[2])

    @staticmethod
    def color_wheel(pos: int, order="RGB") -> tuple:
        """
        Input a value 0 to 255 to get a color value.
        ags: pos on color wheel as int
        order: String of either RGB, GRB, RGBW or GRBW
        returns tuple of color in RGB
        """
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
        color = (r, g, b)
        if order in ("GRB", "GRBW"):
            color = (g, r, b)
            if order == "GRBW":
                color = (g, r, b, 0)
        elif order == "RGBW":
            color = (r, g, b, w)
        return color


class NeoPixelAnimations:
    """
    Pass in a neo pixel object and you can trigger animations
    Also supports loading pixel grids
    """
    def __init__(self, pixel, pixel_number: int, pixel_grid: tuple = ()) -> None:
        """
        init
        pixel is neopixel class object
        """
        self.pixels = pixel
        # set the auto pixel write false
        self.pixels.auto_write = False
        self.pixel_number = pixel_number
        self.pixel_grid = pixel_grid
         
        if self.pixel_grid == ():
            # generate a grid for all leds if none provided
            self.pixel_grid= tuple(range(0, self.pixel_number))
        self.wheel = LedHelper.color_wheel

    def rainbow_cycle(self, wait, order="RGB"):
        """
        Cycle through all the colors of rainbow spectrum
        wait is how quickly it cycles as a int or float
        """
        self.pixels.auto_write = False
        wait = float(wait)
        for j in range(255):
            for i in self.pixel_grid:
                pixel_index = (i * 256 // self.pixel_number) + j
                self.pixels[i] = self.wheel(pixel_index & 255, order=order)
            self.pixels.show()
            sleep(wait)

    def intensity(self, wait, color):
        """
        Start out dim dark and get brighter and more intense
        """
        self.pixels.brightness = .1
        self.pixels.auto_write = False
        s = time()
        intense = .1
        st = wait / 10
        while (time() - s) <= wait:
            if intense > 10:
                intense = 10
            for i in self.pixel_grid:
                self.pixels[i] = LedHelper.adjust_brightness(color, intense)
            self.pixels.show()
            intense += .1
            self.pixels.brightness = intense
            sleep(st)
    
    
    def fade_color(self, start_color, end_color, steps, order="RGB", intensity = (.1, .1)):
        # calculate intensity steps
        ic = steps / (intensity[0] * 10)
        ic_count = 1
        # order sets how we write it out, either RGB or GRB
        for i in range(steps + 1):
            r = start_color[0] + int((end_color[0] - start_color[0]) * i / steps)
            g = start_color[1] + int((end_color[1] - start_color[1]) * i / steps)
            b = start_color[2] + int((end_color[2] - start_color[2]) * i / steps)
            self.pixels.auto_write = False
            for i in self.pixel_grid:
                if order in ["RGB", "RGBW"]:
                    self.pixels[i] = (r, g, b)
                if order in ["GRB", "GRBW"]:
                    self.pixels[i] = (g, r, b)
                self.pixels.show()
            ic_count += 1
            if ic_count >= ic:
                pb = self.pixels.brightness
                if pb < intensity[0]:
                    pb += .1
                    self.pixels.brightness = pb
                    ic_count = 1
            # Delay for smooth transition
            sleep(0.1)
    
    @staticmethod
    def pwmintensity(wait, pwmled):
        """
        Start out dim dark and get brighter and more intense
        """
        dc = 100
        pwmled.duty_cycle = dc
        s = time()
        increase = 65535 / wait
        st = wait / 65535
        while (time() - s) <= wait:
            dc += increase
            if dc > 65535:
                dc = 65535
            pwmled.duty_cycle = int(dc)
            sleep(1)


