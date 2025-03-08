import random
from time import sleep, time
from threading import Thread
from json import loads, JSONDecodeError
from typing import Dict, Callable, Tuple, NamedTuple, Any
from os import path
from glob import glob
from collections import namedtuple
from copy import copy

# 3rd party
from paho.mqtt.client import MQTTMessage
from adafruit_servokit import ServoKit
import neopixel
import adafruit_pca9685
import busio
import board
from digitalio import DigitalInOut, Direction
from PIL import Image, ImageDraw
from adafruit_rgb_display import st7789
import adafruit_bno055
import adafruit_vl53l4cd


# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.MqttClient import MQTTClient, ServoMessageBuilder, IMUMessageBuilder, TOFMessageBuilder
from glados_modules.LedHelper import LedHelper, NeoPixelAnimations
from glados_modules.GLaDosEnums import ServoEnum, SystemEnums, LoggingEnums, MQTTEnums, IMUEnums, TOFEnums


class GladosLCD(Thread, MQTTClient):
    def __init__(self, broker, location, animation_path="./aperture_logo", cs=board.CE0, dc=board.D25, rst=board.D24,
                 sck=board.SCK, mosi=board.MOSI, flip=False):
        # Configuration for CS and DC pins (these are PiTFT defaults):
        Thread.__init__(self)
        Thread.daemon = True
        self.location = location
        self.__name__ = f"{self.__class__.__name__}_{location}"
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.location: str = location
        self.animation_path: str = animation_path
        self.cmd_topic: str = "body/lcd"
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd}
        self.disp = st7789.ST7789(spi=busio.SPI(clock=sck, MOSI=mosi), rotation=0, width=240, height=198, x_offset=0,
                                  y_offset=122, cs=DigitalInOut(cs), dc=DigitalInOut(dc),
                                  rst=DigitalInOut(rst), baudrate=25000000)
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
        self.breath_animation = True
        self.breathe_loop = True
        self.stop_loop = False
        MQTTClient.__init__(self, broker.ip, broker.port)

    def handle_cmd(self, msg) -> None:
        j_msg = loads(msg.payload.decode())
        if j_msg.get("lcd", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic}, {j_msg}")
            if j_msg.get("cmd", "") == "set_breath":
                # command looks like nested {"cmd", "set_breath", options: { COMMAND DICT}}
                self.set_breath_options(j_msg["options"])
                # TODO make this enum
                self.send_command({self.location: self.get_breath_options()}, "status")
            elif j_msg.get("cmd", "") == "get_breath":
                # mark the location of response
                # TODO make this enum
                self.send_command({self.location: self.get_breath_options()}, "body/lcd")
            # comment this command out right now till we know if we need syncing between LCD's
            # calling startup via mqtt causes a dead_lock that doesn't return and stops all other commands
            """
            elif j_msg.get("cmd", "") == "startup":
                # trigger startup animation
                self.stop()
                self.__startup()
                self.client.publish("status", dumps({self.location: {"cmd": "startup",  "status": "complete"}}))
            """
    def set_breath_options(self, breath_dict: dict) -> None:
        self.breath_fast = breath_dict['fast']
        self.breath_animation = breath_dict['animation']
        self.rainbow = breath_dict['rainbow']

    def get_breath_options(self) -> dict:
        return {"response": {'fast': self.breath_fast, 'rainbow': self.rainbow,
                'animation': self.breath_animation, "location": self.location}}

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
        start_x = (width - ((num_circles_x - 1) * spacing_x + 2 * radius)) // 2 - 10
        start_y = (height - ((num_circles_y - 1) * spacing_y + 2 * radius)) // 2 - 25

        if self.rainbow is True:
            circle_color = LedHelper.color_wheel(self.g_color)
            if self.g_color == 255:
                self.g_color = 0
            else:
                self.g_color += 1

        # Draw the circles, turning specific circles red based on on_positions
        for x in range(1, num_circles_x + 1):
            for y in range(1, num_circles_y + 1):
                cd = LedHelper.adjust_brightness(circle_color, random.choice([x / 10.0 for x in range(6, 9)]))
                center_x = start_x + radius + x * spacing_x
                center_y = start_y + radius + y * spacing_y
                # Determine the color of the circle based on its position in on_positions list
                if (x, y) in self.dot_on_positions and x <= self.counter:
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
        image = image.resize((240, int((240 / image_ratio))), Image.BICUBIC)
        if image.height < 198:
            # create new canvas (color format, size, background color) default is aperture orange
            new_canvas = Image.new("RGB", (240, 198), "#ff9a00")
            vertical_offset = (198 - image.height) // 2
            new_canvas.paste(image, (0, vertical_offset))
            if self.flip is True:
                new_canvas = new_canvas.rotate(180)
            self.disp.image(new_canvas)

    def aperture_animation(self, f_type: str = '.bmp') -> None:
        """
        Play a 30-second animation of the aperture logo on an orange background
        """
        apath = path.join(self.animation_path, "*{}".format(f_type))
        self.logger.debug(f"Loading animations from {apath}")
        frame_filenames = sorted(glob(apath))
        for filename in frame_filenames:
            self.__display_frame(filename)
            sleep(1/29.97)

    def breathe(self) -> None:
        """
        A looping animation for the LCD screens where the circle grid pulses up and down like breathing
        Can be set to fast or slow. Circle colors are changed else ware, blocking call
        """
        self.breathe_loop = True
        up = True
        self.counter = 1
        slpm = 0.
        slptb = 0
        tb = 0
        mid = 0
        while self.breathe_loop is True:
            if self.breath_fast is False:
                slpm = 0.12
                slptb = 0.18
                tb = 12
                mid = 5
            elif self.breath_fast is True:
                slpm = 0.
                slptb = 0
                tb = 0
                mid = 0
            self.draw_image(times=mid)
            if self.breath_animation is True:
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

    def __startup(self):
        # startup animation
        self.stop_breath()
        self.aperture_animation()
        self.breathe()

    def run(self):
        self.__startup()
        # note does this need to be a running thread?
        while self.stop_loop is False:
            sleep(1)

    def stop_breath(self):
        # end all loops so you can join thread
        self.breathe_loop = False

    def stop(self):
        # end all loops so you can join thread
        self.breathe_loop = False
        self.stop_loop = True


class TOF(MQTTClient, Thread):
    """TOF control class to poll the TOF and push updates to MQTT.

    This class is a multithreaded TOF controller that polls sensor data
    from an Adafruit VL53L4CD sensor via I2C and publishes updates using MQTT.

    Attributes:
        TOF_broker (tuple): The broker tuple obtained from MQTTClient.
        vl53 (adafruit_vl53l4cd): The sensor object used for readings.
    """

    TOF_broker = MQTTClient.broker_tuple

    def __init__(self, broker: Any) -> None:
        """Initializes the TOF object.

        This method sets up the thread as a daemon, initializes the I2C
        interface, and creates the sensor object. It also initializes the
        MQTT client using the provided broker details.

        Args:
            broker (Any): An object with 'ip' and 'port' attributes required
                to establish the MQTT connection.
        """
        self.__name__ = self.__class__.__name__
        Thread.__init__(self)
        self.daemon = True
        i2c = board.I2C()  # uses board.SCL and board.SDA
        self.vl53 = adafruit_vl53l4cd.VL53L4CD(i2c)
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        MQTTClient.__init__(self, ip=broker.ip, port=broker.port)
        # adjust values as needed
        self.vl53.inter_measurement = 0
        self.vl53.timing_budget = 200
        model_id, module_type = self.vl53.model_info
        self.logger.debug("Model ID: 0x{:0x}".format(model_id))
        self.logger.debug("Module Type: 0x{:0x}".format(module_type))
        self.logger.debug(f"Timing Budget: {self.vl53.timing_budget}")
        self.logger.debug(f"Inter-Measurement: {self.vl53.inter_measurement}")

    def get_sensor(self) -> Dict[str, Any]:
        """Retrieve sensor data.

        This method gathers sensor TOF data, It also attaches a timestamp to the reading.

        Returns:
            Dict[str, Any]: A dictionary containing the sensor data, where the keys
            are defined by the TOFEnums and the values are the corresponding sensor
            readings.
        """
        while not self.vl53.data_ready:
            pass
        self.vl53.clear_interrupt()
        sdata: Dict[str, Any] = {
            TOFEnums.TOF_STATUS_KEY: self.vl53.distance,
            TOFEnums.TOF_TIME_STAMP_KEY.value: time(),
        }
        self.logger.debug(f"TOF Data: {sdata}")
        return sdata

    def run(self) -> None:
        """Thread loop to send a sensor command 10x a second.

        This method continuously retrieves sensor data, builds a status
        message using the TOFMessageBuilder, and sends the command to the
        designated MQTT topic. The loop runs approximately 10 times per second.
        """
        self.vl53.start_ranging()
        self.logger.info("TOF Sensor polling started")
        while True:
            status = TOFMessageBuilder.send_tof_status_message(self.get_sensor())
            self.send_command(topic=MQTTEnums.TOF_STATUS_TOPIC.value, command=status)
            sleep(0.1)


class IMU(MQTTClient, Thread):
    """IMU control class to poll the IMU and push updates to MQTT.

    This class is a multithreaded IMU controller that polls sensor data
    from an Adafruit BNO055 sensor via I2C and publishes updates using MQTT.

    Attributes:
        imu_broker (tuple): The broker tuple obtained from MQTTClient.
        sensor (adafruit_bno055.BNO055_I2C): The sensor object used for readings.
        last_val (int): The last recorded temperature value for correction.
    """

    imu_broker = MQTTClient.broker_tuple

    def __init__(self, broker: Any) -> None:
        """Initializes the IMU object.

        This method sets up the thread as a daemon, initializes the I2C
        interface, and creates the sensor object. It also initializes the
        MQTT client using the provided broker details.

        Args:
            broker (Any): An object with 'ip' and 'port' attributes required
                to establish the MQTT connection.
        """
        self.__name__ = self.__class__.__name__
        Thread.__init__(self)
        self.daemon = True
        i2c = board.I2C()  # uses board.SCL and board.SDA
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        MQTTClient.__init__(self, ip=broker.ip, port=broker.port)
        self.last_val = 0xFFFF

    def temperature(self) -> int:
        """Get the corrected temperature value from the sensor.

        This method reads the temperature from the sensor and applies a
        correction specific to running off a Raspberry Pi. If a temperature
        jump of 128 units is detected twice consecutively, the temperature is
        corrected by masking the result.

        Returns:
            int: The corrected temperature reading.
        """
        result = self.sensor.temperature
        if abs(result - self.last_val) == 128:
            result = self.sensor.temperature
            if abs(result - self.last_val) == 128:
                return 0b00111111 & result
        self.last_val = result
        return result

    def get_sensor(self) -> Dict[str, Any]:
        """Retrieve sensor data.

        This method gathers sensor data including temperature, acceleration,
        magnetic field, gyro, Euler angles, quaternion, linear acceleration, and
        gravity. It also attaches a timestamp to the reading.

        Returns:
            Dict[str, Any]: A dictionary containing the sensor data, where the keys
            are defined by the IMUEnums and the values are the corresponding sensor
            readings.
        """
        sdata: Dict[str, Any] = {
            IMUEnums.TEMP_KEY.value: self.temperature(),
            IMUEnums.ACCEL_KEY.value: self.sensor.acceleration,
            IMUEnums.MAGNETO_KEY.value: self.sensor.magnetic,
            IMUEnums.GYRO_KEY.value: self.sensor.gyro,
            IMUEnums.EULER_KEY.value: self.sensor.euler,
            IMUEnums.QUAT_KEY.value: self.sensor.quaternion,
            IMUEnums.LINEAR_KEY.value: self.sensor.linear_acceleration,
            IMUEnums.GRAVITY_KEY.value: self.sensor.gravity,
            IMUEnums.IMU_TIME_STAMP_KEY.value: time(),
        }
        self.logger.debug(f"IMU Data: {sdata}")
        return sdata

    def run(self) -> None:
        """Thread loop to send a sensor command 10x a second.

        This method continuously retrieves sensor data, builds a status
        message using the IMUMessageBuilder, and sends the command to the
        designated MQTT topic. The loop runs approximately 10 times per second.
        """
        self.logger.info("IMU Sensor polling started")
        while True:
            status = IMUMessageBuilder.send_imu_status_message(self.get_sensor())
            self.send_command(topic=MQTTEnums.IMU_STATUS_TOPIC.value, command=status)
            sleep(0.1)


class Gservo(MQTTClient, Thread):
    """
    Generic Servo Class to take movement commands from MQTT for a servo and send status to MQTT
    """
    def __init__(self, location: str, servo: ServoKit.servo, axis: str, broker: NamedTuple,
                 servo_range: NamedTuple, pulse_max_min=None, servo_speed: float = 0.1) -> None:
        self.__name__ = f"{self.__class__.__name__}_{location}"
        Thread.__init__(self)
        Thread.daemon = True
        self.stop = False
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        # 1 degree movement speed
        degree_per_second = servo_speed / 60
        self.speed_settings = {
            1: degree_per_second * 5,  # Calm movement
            2: degree_per_second * 4,  # Neutral
            3: degree_per_second * 3,  # Slightly agitated
            4: degree_per_second * 2,  # Angry
            5: degree_per_second * 1   # Frustrated/fastest
        }
        self.location: str = location
        self.cmd_topic = ServoEnum.MQTT_COMMAND_TOPIC.value
        self.status_topic = ServoEnum.MQTT_STATUS_TOPIC.value
        self.intensity_topic = SystemEnums.MQTT_INTENSITY_TOPIC.value
        self.topic_handler: Dict[str, Callable] = {
            self.cmd_topic: self.handle_cmd,
            self.intensity_topic: self.handle_intensity}
        self.min_angle: int = 0
        self.servo = servo
        if pulse_max_min is not None:
            self.servo.set_pulse_width_range(min_pulse=pulse_max_min.min, max_pulse=pulse_max_min.max)
        self.speed: int = 5
        self.servo_range = servo_range
        self.middle_angle = int(self.servo_range.center)
        self.angle: int = self.middle_angle
        self.current_angle: int = self.angle
        self.first_boot: bool = True
        self.axis: str = axis.lower()
        self.moving: bool = False
        # Call the superclass constructor to initialize MQTTClient and the lock
        MQTTClient.__init__(self, ip=broker.ip, port=broker.port)
        #super().__init__(broker.ip, broker.port)
        # Move to the initial position and send status
        self.move()
        self.send_status()

    def calculate_move_time(self, target_angle: int) -> float:
        """
        Calculate the time it will take the servo to move to the target angle based on its current speed setting.
        :param target_angle: The desired angle to which the servo should move.
        :return: The time (in seconds) it will take to complete the movement.
        """
        with self._lock:
            # Determine the distance (degrees) the servo will move
            distance_to_travel = abs(self.current_angle - target_angle)
            # Get the speed in degrees per second from the speed settings
            speed_in_degrees_per_second = self.speed_settings[self.speed]

        # Calculate the time required to move the specified distance at the current speed
        move_time = distance_to_travel / speed_in_degrees_per_second
        self.logger.debug(f"Calculated move time: {move_time} seconds to move {distance_to_travel} degrees "
                          f"at speed setting {self.speed} ({speed_in_degrees_per_second} degrees/second)")
        return move_time

    def send_status(self) -> None:
        """
        Send current angle status to mqtt
        :return:
        """
        # Send current status
        status = ServoMessageBuilder.send_status(self.location, self.get_angles())
        self.send_command(topic=self.status_topic, command=status)

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """
        Handle incoming MQTT commands for the servo
        :return:
        """
        try:
            j_msg = loads(msg.payload.decode())
        except JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON message: {e}")
            return

        if j_msg.get(ServoEnum.MSG_LOCATION_KEY.value, "") == self.location:
            # Move command
            cmd = j_msg.get(ServoEnum.MSG_COMMAND_KEY.value, "")
            if cmd == ServoEnum.MSG_COMMAND_MOVE.value:
                self.logger.debug(f"{self.location}, {msg.topic}, {j_msg}")
                angle: int = int(j_msg.get(ServoEnum.MSG_ANGLE.value, self.middle_angle))
                speed: int = int(j_msg.get(ServoEnum.MSG_SPEED.value, self.speed))
                self.set_speed_angle((speed, angle))
            elif cmd == ServoEnum.MSG_COMMAND_STATUS.value:
                self.send_status()

    def run(self):
        while self.stop is False:
            # note there is a sleep in the move
            self.move()

    def handle_intensity(self, msg: MQTTMessage) -> None:
        # TODO: Implement intensity handling
        pass

    def get_angles(self) -> dict:
        """
        Return a dict object with severo values
        :return: dict object of max, min, middle, current_angle and axis location
        """
        with self._lock:
            return {
                ServoEnum.MSG_MAX.value: self.servo_range.max,
                ServoEnum.MSG_MIN.value: self.servo_range.min,
                ServoEnum.MSG_MIDDLE.value: self.middle_angle,
                ServoEnum.MSG_CURRENT_ANGLE.value: self.current_angle,
                ServoEnum.MSG_AXIS.value: self.axis,
                ServoEnum.MSG_MOVING.value: self.moving,
                ServoEnum.MSG_LAST_ANGLE.value: self.angle
            }

    def set_speed(self, speed: int) -> None:
        """
        Scrub any input and make sure it fits the speed between 1-5
        :return:
        """
        if speed >= 5:
            speed = 5
        elif speed <= 1:
            speed = 1
        with self._lock:
            self.speed = round(speed)
            self.logger.debug(f"Speed set to {self.speed}")

    def set_angle(self, angle: int) -> None:
        """
        Set angle for servo and make sure it fits with in max and min range for servo
        :return:
        """
        max_angle = self.servo_range.max
        min_angle = self.servo_range.min
        with self._lock:
            if angle >= max_angle:
                self.angle = max_angle
                self.logger.debug(f"{angle} is above {max_angle}, setting to {max_angle}")
            elif angle <= min_angle:
                self.angle = min_angle
                self.logger.debug(f"{angle} is below {min_angle}, setting to {min_angle}")
            else:
                self.angle = angle
                self.logger.debug(f"Angle set to {self.angle}")

    def set_speed_angle(self, speed_angle: Tuple[int, int]) -> None:
        """
        Set speed an angle as one call
        :return:
        """
        speed, angle = speed_angle
        self.set_speed(speed)
        self.set_angle(angle)

    def get_angle(self) -> int:
        """
        Return value of current angle
        :return: int value of current angle
        """
        with self._lock:
            return self.current_angle

    def s_curve_move(self) -> None:
        """
        Calculate S curves and send small steps to the servo to move
        :return:
        """
        with self._lock:
            total_distance = abs(self.angle - self.current_angle)
            target_angle = copy(self.angle)
            current_angle = copy(self.current_angle)
            speed_setting = copy(self.speed_settings[self.speed])

        if total_distance != 0:
            # Time for full move
            full_time = total_distance * speed_setting
            # Divide the movement into small steps
            steps = 100
            time_per_step = full_time / steps
            for i in range(steps + 1):
                # Calculate S-curve (using a simple cosine-based ease-in and ease-out)
                t = i / steps
                if t < 0.5:
                    t = 2 * t ** 2
                else:
                    t = -1 + (4 - 2 * t) * t
                new_angle = current_angle + (target_angle - current_angle) * t
                with self._lock:
                    self.servo.angle = new_angle
                    self.current_angle = target_angle
                    # allow for dynamic updates and break the loop if there is an update
                    if target_angle != self.angle:
                        # we have a new angle break the loop
                        self.logger.debug(f"New angle Request breaking movement for {self.location}")
                        break
                    else:
                        sleep(time_per_step)
            self.logger.debug(f"{self.location}, sleeping for {full_time} seconds while we move")
            self.logger.debug(f"Set {self.location} angle to {self.current_angle}")
            return

    def get_moving_status(self) -> bool:
        """
        Return boolean if we are moving
        :return: boolean
        """
        with self._lock:
            return self.moving

    def move(self) -> None:
        """
        Move the robot
        :return:
        """
        with self._lock:
            angle = self.angle
            current_angle = self.current_angle
            first_boot = self.first_boot
        self.send_status()

        if first_boot:
            with self._lock:
                self.servo.angle = angle
                self.moving = True
            sleep(self.calculate_move_time(angle))
            with self._lock:
                self.current_angle = angle
                self.moving = False
                self.first_boot = False
                self.logger.debug(f"Set {self.location} angle to {self.current_angle}")
        else:
            if angle != current_angle:
                self.logger.debug(f"New angle {angle} does not equal current angle {current_angle}")
                with self._lock:
                    self.moving = True
                self.s_curve_move()
                self.logger.debug(f"Moved to {angle}")
                with self._lock:
                    self.moving = False
        # settle after the move
        self.send_status()
        sleep(0.2)


class LedShoulders(MQTTClient):
    def __init__(self, broker: NamedTuple) -> None:
        self.__name__ = "LED_Shoulder_Controller"
        self.logger = setup_logger(self.__name__, LoggingEnums.LOG_LEVEL_INFO.value)
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
        MQTTClient.__init__(self, broker.ip, broker.port)

    def handle_cmd(self, msg: MQTTMessage) -> None:
        j_msg = loads(msg.payload.decode())
        if j_msg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {j_msg}")
            if j_msg[self.location]['command'] in self.animations.keys():
                self.animations[j_msg[self.location]['command']]()

    def handle_intensity(self, msg: MQTTMessage) -> None:
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
        # TODO add a logger and mqtt here
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
        Pulse the LED from low to high and back to low
        """
        # each iteration should take
        for i in range(1, 800):
            self.set_brightness(i)
            sleep(0.0025)

    def run(self):
        while self.stop is False:
            self.animation()


class LedHead(MQTTClient):
    def __init__(self, broker: NamedTuple) -> None:
        self.__name__ = "Head_LED_Controller"
        # TODO do we need to remove the logger here or in mqtt object?
        # TODO split out LED control into its own module so i can reduce code to control the dot stars on the pi5?
        self.logger = setup_logger(self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.pixels = neopixel.NeoPixel(board.D18, 1, brightness=1, auto_write=True, pixel_order=neopixel.RGB)
        self.ani = NeoPixelAnimations(self.pixels, 1)
        self.swap = LedHelper.rgb2grb_swap
        self.hat = adafruit_pca9685.PCA9685(busio.I2C(board.SCL, board.SDA))
        self.pwm_led = self.hat.channels[4]
        self.hat.frequency = 60
        self.pwm_led.duty_cycle = 250
        self.intensity: Tuple[float, float] = (.1, .5)
        self.cmd_topic: str = MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value
        self.intensity_topic: str = MQTTEnums.SYSTEM_INTENSITY_TOPIC.value
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd,
                                                   self.intensity_topic: self.handle_intensity}
        self.location: str = "eye_led"
        self.animations: Dict[str, Callable] = {"startup": self.startup, "disco": self.disco,
                                                "angry_eye": self.angry_eye, "normal_eye": self.normal_eye}
        self.glados_eye: Tuple[int, int, int] = (255, 165, 0)
        MQTTClient.__init__(self, broker.ip, broker.port)

    def handle_cmd(self, msg: MQTTMessage) -> None:
        j_msg = loads(msg.payload.decode())
        if j_msg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {j_msg}")
            if j_msg[self.location]['command'] in self.animations.keys():
                self.animations[j_msg[self.location]['command']]()

    def handle_intensity(self, msg: MQTTMessage) -> None:
        # TODO figure out update commands
        j_msg = loads(msg.payload.decode())
        if j_msg.get("led", "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {j_msg}")
            self.intensity = j_msg["intensity"]

    def startup(self) -> None:
        self.logger.debug("Startup Sequence")
        eye_led_thread = Thread(target=self.ani.intensity, args=(10, self.glados_eye))
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
        eye_led_thread = Thread(target=self.ani.fade_color, args=((255, 255, 0), anger, steps, "RB", self.intensity))
        eye_led_thread.start()
        eye_led_thread.join()

    def normal_eye(self) -> None:
        self.pwm_led.duty_cycle = 150
        self.pixels.brightness = self.intensity[0]
        self.pixels.autowrite = True
        self.pixels[0] = LedHelper.adjust_brightness(self.glados_eye, self.intensity[1])
        self.pixels.show()

# NOTE you also need to code up a class for the Lamp portion its self...

# on the pi5 code, need to have classes to read from LIDAR sensor to channel..
# also need class to read temp senders and have them take action
# bird detection to kill external power? how will that work...


if __name__ == "__main__":
    ip = '192.168.1.29'
    Angle_tuple = namedtuple("angle", ['max', 'min', 'center'])
    Pulse_tuple = namedtuple("pulse", ['max', 'min'])
    Mqtt_tuple = namedtuple("mqtt", ["ip", "port"])
    mqtt_connect = Mqtt_tuple(ip, 1883)
    mg90d_pulse = Pulse_tuple(2665, 610)
    mg92b_pulse = Pulse_tuple(2550, 605)
    head_angle = Angle_tuple(173, 6, 83)
    neck_angle = Angle_tuple(120, 52, 92)
    default_angle = Angle_tuple(180, 0, 90)
    kit = ServoKit(channels=16)
    led_head = LedHead(broker=mqtt_connect)
    right_lcd = GladosLCD(broker=mqtt_connect, location="right_lcd")
    body_LR = Gservo(location='body_left_right', servo=kit.servo[0], axis='x', servo_range=default_angle,
                     broker=mqtt_connect)
    body_UD = Gservo(location='body_up_down', servo=kit.servo[1], axis='y',broker=mqtt_connect,
                     pulse_max_min=mg92b_pulse, servo_range=default_angle)
    head_UD = Gservo(location='head_left_right', servo=kit.servo[2], axis='y', servo_range=head_angle,
                     broker=mqtt_connect)
    head_LR = Gservo(location='head_up_down', servo=kit.servo[3], axis='x', servo_range=neck_angle,
                     broker=mqtt_connect)
    right_lcd.start()
    led_head.startup()
    imu = IMU(broker=mqtt_connect)
    imu.start()
    tof = TOF(broker=mqtt_connect)
    tof.start()
    while True:
        sleep(1)
