from time import sleep, time
from typing import Any, Tuple, Union


class LedHelper:
    """Helper class for LED operations."""

    @staticmethod
    def adjust_brightness(color: Tuple[int, int, int], brightness_factor: float) -> Tuple[int, int, int]:
        """Adjust the brightness of a color.

        Args:
            color (Tuple[int, int, int]): The original color as a tuple of (R, G, B).
            brightness_factor (float): The brightness factor, between 0.0 (off) and 1.0
                (full brightness).

        Returns:
            Tuple[int, int, int]: The adjusted color as a tuple of (R, G, B).
        """
        return tuple(int(c * brightness_factor) for c in color)

    @staticmethod
    def rgb2grb_swap(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Swap the red and green channels in a color tuple.

        Args:
            color (Tuple[int, int, int]): The original color as a tuple of (R, G, B).

        Returns:
            Tuple[int, int, int]: The color with red and green swapped (G, R, B).
        """
        return color[1], color[0], color[2]

    @staticmethod
    def color_wheel(pos: int, order: str = "RGB") -> Tuple[int, int, int] | Tuple[int, int, int, int]:
        """Generate a color from a color wheel position.

        Args:
            pos (int): A value from 0 to 255 representing a position on the color wheel.
            order (str, optional): A string indicating the color channel order. Options are
                "RGB", "GRB", "RGBW", or "GRBW". Defaults to "RGB".

        Returns:
            Union[Tuple[int, int, int], Tuple[int, int, int, int]]: The color tuple in the
                specified order.
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
        return color


class NeoPixelAnimations:
    """Class to trigger animations on NeoPixel objects and support loading pixel grids."""

    def __init__(self, pixel: Any, pixel_number: int, pixel_grid: Tuple[int, ...] = ()) -> None:
        """Initialize NeoPixelAnimations.

        Args:
            pixel (Any): The NeoPixel class object.
            pixel_number (int): The number of pixels.
            pixel_grid (Tuple[int, ...], optional): A tuple representing the pixel grid.
                If not provided, a grid for all LEDs will be generated.
        """
        self.pixels = pixel
        self.pixels.auto_write = False
        self.pixel_number = pixel_number
        self.pixel_grid = pixel_grid

        if self.pixel_grid == ():
            # Generate a grid for all LEDs if none provided.
            self.pixel_grid = tuple(range(self.pixel_number))
        self.wheel = LedHelper.color_wheel

    def rainbow_cycle(self, wait: Union[int, float], order: str = "RGB") -> None:
        """Cycle through all the colors of the rainbow spectrum.

        Args:
            wait (Union[int, float]): Time delay between color updates.
            order (str, optional): Color channel order. Defaults to "RGB".
        """
        self.pixels.auto_write = False
        wait = float(wait)
        for j in range(255):
            for i in self.pixel_grid:
                pixel_index = (i * 256 // self.pixel_number) + j
                self.pixels[i] = self.wheel(pixel_index & 255, order=order)
            self.pixels.show()
            sleep(wait)

    def intensity(self, wait: Union[int, float],
                  color: Tuple[int, int, int], intensity_change: float = 0.1) -> None:
        """Increase the intensity of the given color gradually.

        The method starts out dim and gets brighter over the specified wait time.

        Args:
            wait (Union[int, float]): Duration for the intensity change.
            color (Tuple[int, int, int]): The base color as a tuple of (R, G, B).
            intensity_change (float): default .1, how much more intense the changes are at each step
        """
        self.pixels.brightness = 0.1
        self.pixels.auto_write = False
        start_time = time()
        intense = 0.1
        st = wait / 10
        while (time() - start_time) <= wait:
            if intense > 10:
                intense = 10
            for i in self.pixel_grid:
                self.pixels[i] = LedHelper.adjust_brightness(color, intense)
            self.pixels.show()
            intense += intensity_change
            self.pixels.brightness = intense
            sleep(st)

    def fade_color(self,
                   start_color: Tuple[int, int, int],
                   end_color: Tuple[int, int, int],
                   steps: int,
                   order: str = "RGB",
                   intensity: Tuple[float, float] = (0.1, 0.1)) -> None:
        """Fade between two colors over a number of steps.

        This method gradually changes the LED color from start_color to end_color.
        It also adjusts brightness intensity based on the provided intensity tuple.

        Args:
            start_color (Tuple[int, int, int]): The starting color as a tuple of (R, G, B).
            end_color (Tuple[int, int, int]): The ending color as a tuple of (R, G, B).
            steps (int): The number of steps in the fade transition.
            order (str, optional): The color order, either "RGB", "GRB", "RGBW", or "GRBW".
                Defaults to "RGB".
            intensity (Tuple[float, float], optional): A tuple representing brightness intensity.
                The first value is the maximum brightness and the second value is currently unused.
                Defaults to (0.1, 0.1).
        """
        intensity_cycle = steps / (intensity[0] * 10)
        ic_count = 1
        for step in range(steps + 1):
            r = start_color[0] + int((end_color[0] - start_color[0]) * step / steps)
            g = start_color[1] + int((end_color[1] - start_color[1]) * step / steps)
            b = start_color[2] + int((end_color[2] - start_color[2]) * step / steps)
            self.pixels.auto_write = False
            for i in self.pixel_grid:
                if order in ["RGB", "RGBW"]:
                    self.pixels[i] = (r, g, b)
                elif order in ["GRB", "GRBW"]:
                    self.pixels[i] = (g, r, b)
                self.pixels.show()
            ic_count += 1
            if ic_count >= intensity_cycle:
                pb = self.pixels.brightness
                if pb < intensity[0]:
                    pb += 0.1
                    self.pixels.brightness = pb
                    ic_count = 1
            sleep(0.1)

    @staticmethod
    def pwmintensity(wait: Union[int, float], pwmled: Any) -> None:
        """Increase the duty cycle of a PWM LED gradually.

        Args:
            wait (Union[int, float]): Duration for the PWM intensity change.
            pwmled (Any): An object representing a PWM LED that has a duty_cycle attribute.
        """
        dc = 100
        pwmled.duty_cycle = dc
        start_time = time()
        increase = 65535 / wait
        st = wait / 65535
        while (time() - start_time) <= wait:
            dc += increase
            if dc > 65535:
                dc = 65535
            pwmled.duty_cycle = int(dc)
            sleep(1)
