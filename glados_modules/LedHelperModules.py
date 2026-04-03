import random
from math import sin, pi
from time import sleep, time
from typing import Any, List, Optional, Tuple, Union


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
        self._twinkle_loop: bool = False
        self._disco_loop: bool = False
        self._pulse_loop: bool = False

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
        self.pixels.auto_write = False
        start_time = time()
        intense = 0.1
        st = wait / 10
        while (time() - start_time) <= wait:
            if intense >= 1.0:
                intense = 1.0
            for i in self.pixel_grid:
                self.pixels[i] = LedHelper.adjust_brightness(color, intense)
            self.pixels.show()
            intense += intensity_change
            sleep(st)

    def speech_pulse(self, wait: float, color: Tuple[int, int, int],
                     peak: float = 1.0) -> None:
        """Pulse LED with a smooth bell curve over the word duration.

        Brightness follows sin(pi * t / wait) — rises to peak at midpoint,
        falls back to dim at the end. Produces natural-looking speech pulses
        where short words flash and long words breathe.

        Args:
            wait: Duration of the pulse in seconds (word duration).
            color: Base RGB color for the pulse.
            peak: Maximum brightness (0.0-1.0). Defaults to 1.0.
        """
        self.pixels.auto_write = False
        start_time = time()
        min_bright = 0.05
        steps = max(10, int(wait * 30))
        step_dt = wait / steps

        for i in range(steps):
            elapsed = time() - start_time
            if elapsed > wait:
                break
            t = elapsed / wait
            brightness = min_bright + (peak - min_bright) * sin(pi * t)
            for px in self.pixel_grid:
                self.pixels[px] = LedHelper.adjust_brightness(color, brightness)
            self.pixels.show()
            sleep(step_dt)

        # End dim
        for px in self.pixel_grid:
            self.pixels[px] = LedHelper.adjust_brightness(color, min_bright)
        self.pixels.show()

    def disco(self, wait: Union[int, float] = 0.05, order: str = "RGB") -> None:
        """Run a continuous rainbow cycle animation until stop_disco is called.

        Args:
            wait (Union[int, float]): Time delay between color updates. Defaults to 0.05.
            order (str): Color channel order. Defaults to "RGB".
        """
        self.stop_twinkle()
        self.pixels.auto_write = False
        wait = float(wait)
        self._disco_loop = True
        j = 0
        while self._disco_loop:
            for i in self.pixel_grid:
                pixel_index = (i * 256 // self.pixel_number) + j
                self.pixels[i] = self.wheel(pixel_index & 255, order=order)
            self.pixels.show()
            sleep(wait)
            j = (j + 1) % 256

    def stop_disco(self) -> None:
        """Stop a running disco animation."""
        self._disco_loop = False
        self.stop_pulse()

    def twinkle(self, color: Tuple, pixel_grid: Optional[List[int]] = None,
                wait: Union[int, float] = 0.1) -> None:
        """Randomly vary brightness across pixels in a loop until stop_twinkle is called.

        Args:
            color (Tuple): The base color to twinkle.
            pixel_grid (Optional[List[int]]): Pixel indices to animate. Defaults to self.pixel_grid.
            wait (Union[int, float]): Sleep duration between frames. Defaults to 0.1.
        """
        self.stop_disco()
        self.stop_pulse()
        grid = pixel_grid if pixel_grid is not None else self.pixel_grid
        self._twinkle_loop = True
        while self._twinkle_loop:
            for i in grid:
                self.pixels[i] = LedHelper.adjust_brightness(
                    color, random.choice([x / 10.0 for x in range(1, 9)])
                )
            self.pixels.show()
            sleep(wait)

    def stop_twinkle(self) -> None:
        """Stop a running twinkle animation."""
        self._twinkle_loop = False

    def pulse(self, color: Tuple, mode: str = "all", pixel_index: int = 0,
              steps: int = 20, wait: Union[int, float] = 0.05) -> None:
        """Continuously fade a color in and out using adjust_brightness until stop_pulse is called.

        Args:
            color (Tuple): The color to pulse.
            mode (str): "all" pulses every pixel together, "single" pulses one pixel,
                        "random" picks a new random subset of pixels each cycle. Defaults to "all".
            pixel_index (int): Pixel to pulse when mode is "single". Defaults to 0.
            steps (int): Number of brightness steps per ramp direction. Defaults to 20.
            wait (Union[int, float]): Sleep between each step in seconds. Defaults to 0.05.
        """
        self.stop_disco()
        self.stop_twinkle()
        factors = [i / steps for i in list(range(1, steps + 1)) + list(range(steps - 1, 0, -1))]
        self._pulse_loop = True
        while self._pulse_loop:
            if mode == "random":
                targets = random.sample(list(self.pixel_grid), k=random.randint(1, len(self.pixel_grid)))
            elif mode == "single":
                targets = [pixel_index]
            else:
                targets = self.pixel_grid
            for factor in factors:
                if not self._pulse_loop:
                    break
                for i in targets:
                    self.pixels[i] = LedHelper.adjust_brightness(color, factor)
                self.pixels.show()
                sleep(wait)

    def stop_pulse(self) -> None:
        """Stop a running pulse animation."""
        self._pulse_loop = False

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
                The first value is the maximum brightness, and the second value is currently unused.
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


class PWMLedAnimations:
    """Animations for PWM-controlled LEDs (e.g. adafruit PCA9685 channels)."""

    def __init__(self, pwmled: Any) -> None:
        """Initialize PWMLedAnimations.

        Args:
            pwmled (Any): An object representing a PWM LED with a duty_cycle attribute.
        """
        self.pwmled = pwmled
        self._pulse_loop: bool = False

    def intensity(self, wait: Union[int, float], steps: int = 100) -> None:
        """Increase the duty cycle of the PWM LED gradually from low to full.

        Args:
            wait (Union[int, float]): Duration for the ramp in seconds.
            steps (int): Number of increments across the ramp. Defaults to 100.
        """
        st = float(wait) / steps
        increase = 65535 / steps
        dc = 100.0
        self.pwmled.duty_cycle = int(dc)
        for _ in range(steps):
            dc = min(dc + increase, 65535)
            self.pwmled.duty_cycle = int(dc)
            sleep(st)

    def pulse(self, period: Union[int, float] = 2.0, steps: int = 100) -> None:
        """Continuously ramp the PWM LED up and down until stop_pulse is called.

        Args:
            period (Union[int, float]): Duration of one full up-and-down cycle in seconds. Defaults to 2.0.
            steps (int): Number of increments per ramp direction. Defaults to 100.
        """
        st = float(period) / (steps * 2)
        increase = 65535 / steps
        self._pulse_loop = True
        while self._pulse_loop:
            dc = 100.0
            for _ in range(steps):
                if not self._pulse_loop:
                    break
                dc = min(dc + increase, 65535)
                self.pwmled.duty_cycle = int(dc)
                sleep(st)
            for _ in range(steps):
                if not self._pulse_loop:
                    break
                dc = max(dc - increase, 100)
                self.pwmled.duty_cycle = int(dc)
                sleep(st)

    def stop_pulse(self) -> None:
        """Stop a running pulse animation."""
        self._pulse_loop = False
