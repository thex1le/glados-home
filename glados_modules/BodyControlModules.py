import random
from time import sleep, time
from threading import Thread, Event
from json import loads, JSONDecodeError
from typing import Dict, Callable, Tuple, NamedTuple, Any, Optional
from os import path
from glob import glob
from collections import namedtuple
from copy import copy

# 3rd party
from paho.mqtt.client import MQTTMessage
import adafruit_pca9685
import busio
import board
from digitalio import DigitalInOut, Direction
from PIL import Image, ImageDraw
import neopixel
import RPi.GPIO as GPIO

# glados imports
from glados_modules.MqttConsumerModules import SensorTracker
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import (ServoEnum, SystemEnums, LoggingEnums, MQTTEnums,
                                        IMUEnums, TOFEnums, THEnums, MOXEnums, LEDHead,
                                        LEDLampStrip8, LEDShoulders, MotionProfile, TraceEnums,
                                        LCDEnums, SocTempEnums)
from glados_modules.MqttConnector import (MQTTClient, ServoMessageBuilder, IMUMessageBuilder,
                                          TOFMessageBuilder, THMessageBuilder, MoxGasMessageBuilder,
                                          SocTempMessageBuilder)
from glados_modules.LedHelperModules import LedHelper, NeoPixelAnimations, PWMLedAnimations
from glados_modules.PipelineDebug import PipelineDebug


class _ST7789SpidevWrapper:
    """Thin wrapper around the spidev-based st7789 driver so it exposes the
    same ``.image(pil_image)`` API that the Adafruit driver uses."""

    def __init__(self, port=0, cs=0, dc=25, rst=24, width=240, height=198,
                 spi_speed_hz=25000000, offset_top=122):
        import st7789 as _st7789
        self._disp = _st7789.ST7789(
            port=port, cs=cs, dc=dc, rst=rst,
            width=width, height=height,
            rotation=0,
            spi_speed_hz=spi_speed_hz,
            offset_left=0, offset_top=offset_top,
        )

    def image(self, pil_image):
        self._disp.display(pil_image)


class GladosLCD(Thread, MQTTClient):
    # BCM pin numbers used for DC and RST (same on Pi 4 and Pi 5 hardware)
    _DEFAULT_DC_BCM = 25
    _DEFAULT_RST_BCM = 24

    def __init__(self, broker, location, animation_path="./aperture_logo", cs=board.CE0, dc=board.D25, rst=board.D24,
                 sck=board.SCK, mosi=board.MOSI, flip=False, sync_leader=False):
        Thread.__init__(self)
        self.daemon = True
        self.location = location
        self.__name__ = f"{self.__class__.__name__}_{location}"
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.location: str = location
        self.animation_path: str = animation_path
        self.cmd_topic: str = MQTTEnums.LCD_CONTROL_MQTT_TOPIC.value
        self.sync_topic: str = MQTTEnums.LCD_SYNC_MQTT_TOPIC.value
        self.sync_leader: bool = sync_leader
        self._sync_counter: Optional[int] = None
        self._startup_go: Event = Event()
        self._peer_ready: bool = False
        self.topic_handler: Dict[str, Callable] = {
            self.cmd_topic: self.handle_cmd,
            self.sync_topic: self._handle_sync,
        }
        # Try the Adafruit blinka driver (works on Pi 4).
        # If it fails (Pi 5 RP1 lgpio 'GPIO busy'), fall back to the
        # spidev-based st7789 driver which uses the kernel SPI + gpiod.
        try:
            from adafruit_rgb_display import st7789
            try:
                cs_pin = DigitalInOut(cs)
            except Exception:
                cs_pin = None
            self.disp = st7789.ST7789(
                spi=busio.SPI(clock=sck, MOSI=mosi), rotation=0,
                width=240, height=198, x_offset=0, y_offset=122,
                cs=cs_pin, dc=DigitalInOut(dc), rst=DigitalInOut(rst),
                baudrate=25000000)
            self.disp.spi_device.cs_active_value = False
            self.logger.info("LCD using Adafruit blinka driver")
        except Exception as e:
            self.logger.warning(f"Adafruit LCD driver failed ({e}) — using spidev/gpiod driver (Pi 5)")
            self.disp = _ST7789SpidevWrapper(
                port=0, cs=0,
                dc=self._DEFAULT_DC_BCM, rst=self._DEFAULT_RST_BCM,
                width=240, height=198,
                spi_speed_hz=25000000, offset_top=122,
            )
            self.logger.info("LCD using spidev st7789 driver")
        self.dot_on_positions: tuple = ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4),
                                        (3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2), (4, 3), (4, 4),
                                        (5, 1), (5, 2), (5, 3), (5, 4), (6, 1), (6, 2), (6, 3), (6, 4),
                                        (7, 1), (7, 2), (7, 3), (7, 4), (8, 2), (8, 4), (9, 1), (9, 3),
                                        (9, 4), (10, 3), (11, 1), (11, 2), (11, 4), (12, 2))
        self.rainbow = False
        self.g_color = 0
        self.counter = 1
        self.flip = flip
        self.breath_fast = False
        self.breath_animation = True
        self.breathe_loop = True
        self.stop_loop = False
        MQTTClient.__init__(self, broker.ip, broker.port)

    def handle_cmd(self, msg) -> None:
        j_msg = loads(msg.payload.decode())
        if j_msg.get(LCDEnums.MSG_LOCATION_KEY.value, "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic}, {j_msg}")
            cmd = j_msg.get(LCDEnums.MSG_COMMAND_KEY.value, "")
            if cmd == LCDEnums.COMMAND_SET_BREATH.value:
                self.set_breath_options(j_msg[LCDEnums.OPTIONS_KEY.value])
                self.send_command({self.location: self.get_breath_options()},
                                  MQTTEnums.LCD_STATUS_MQTT_TOPIC.value)
            elif cmd == LCDEnums.COMMAND_GET_BREATH.value:
                self.send_command({self.location: self.get_breath_options()},
                                  MQTTEnums.LCD_CONTROL_MQTT_TOPIC.value)
            elif cmd == LCDEnums.COMMAND_STARTUP.value:
                # Run in thread — __startup() calls breathe() which blocks,
                # so it can't run in the MQTT callback thread
                self.stop_breath()
                Thread(target=self.__startup, daemon=True).start()
    def _handle_sync(self, msg) -> None:
        """Handle sync messages: ready/go handshake and breathing counter."""
        try:
            data = loads(msg.payload.decode())
        except Exception:
            return

        msg_type = data.get("type")

        if msg_type == "ready":
            # A peer announced it's ready — leader notes this
            if self.sync_leader:
                self._peer_ready = True
        elif msg_type == "go":
            # Leader says go — follower starts
            if not self.sync_leader:
                self._startup_go.set()
        elif msg_type == "counter":
            # Breathing sync — followers only
            if not self.sync_leader:
                self._sync_counter = data.get("value")

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
        Can be set to fast or slow. Circle colors are changed else ware, blocking call.
        If sync_leader is True, publishes counter via MQTT so followers stay in sync.
        If sync_leader is False, follows the counter from the leader when available.
        """
        self.breathe_loop = True
        up = True
        self.counter = 1
        slpm = 0.
        slptb = 0
        tb = 0
        mid = 0
        while self.breathe_loop is True:
            # Follower: snap to leader's counter when available
            if not self.sync_leader and self._sync_counter is not None:
                self.counter = self._sync_counter
                up = self.counter < 7  # infer direction from position
                self._sync_counter = None

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
                # Leader: publish counter for followers
                if self.sync_leader:
                    self.send_command({"type": "counter", "value": self.counter},
                                      MQTTEnums.LCD_SYNC_MQTT_TOPIC.value)
            else:
                self.counter = 12
                sleep(.2)

    def _wait_for_sync_start(self, timeout: float = 30.0) -> None:
        """Coordinate startup so both LCDs begin the aperture animation together.

        Leader: waits for a 'ready' message from the follower (or timeout),
                then sends 'go'.
        Follower: sends 'ready', then waits for 'go' from the leader (or timeout).
        """
        if self.sync_leader:
            self.logger.info("LCD sync: leader waiting for follower ready...")
            deadline = time() + timeout
            while not self._peer_ready and time() < deadline:
                sleep(0.2)
            if self._peer_ready:
                self.logger.info("LCD sync: follower ready — sending go")
            else:
                self.logger.warning("LCD sync: timeout waiting for follower — starting alone")
            self.send_command({"type": "go"}, MQTTEnums.LCD_SYNC_MQTT_TOPIC.value)
        else:
            self.logger.info("LCD sync: follower sending ready, waiting for go...")
            self.send_command({"type": "ready"}, MQTTEnums.LCD_SYNC_MQTT_TOPIC.value)
            if not self._startup_go.wait(timeout=timeout):
                self.logger.warning("LCD sync: timeout waiting for leader go — starting alone")
            else:
                self.logger.info("LCD sync: received go from leader")

    def __startup(self):
        # wait for both LCDs to be ready, then start together
        self._wait_for_sync_start()
        self.stop_breath()
        self.aperture_animation()
        self.breathe()

    def run(self):
        self.__startup()
        while self.stop_loop is False:
            sleep(1)

    def stop_breath(self):
        # end all loops so you can join thread
        self.breathe_loop = False

    def stop(self):
        # end all loops so you can join thread
        self.breathe_loop = False
        self.stop_loop = True


class MoxGas(MQTTClient, Thread):
    """Manage MOX gas sensor polling and MQTT communication.

    This class inherits from both MQTTClient and Thread to allow continuous sensor
    data polling in a separate thread and to facilitate MQTT messaging.

    Attributes:
        MOX_GAS_broker: Broker tuple provided by MQTTClient.
        logger: Logger instance for recording events.
        st: SensorTracker instance for temperature and humidity data.
        mox: MOX sensor instance from adafruit_ens160.ENS160.
    """

    MOX_GAS_broker = MQTTClient.broker_tuple

    def __init__(self, broker: Any) -> None:
        """Initialize the MOX gas sensor, MQTT client, and sensor tracking.

        Args:
            broker: An object containing broker information with attributes 'ip' and 'port'.
        """
        import adafruit_ens160
        self.__name__ = self.__class__.__name__
        Thread.__init__(self)
        self.daemon = True
        self.logger = setup_logger(
            name=self.__name__,
            console_logging=LoggingEnums.LOG_LEVEL_INFO.value
        )
        MQTTClient.__init__(self, ip=broker.ip, port=broker.port)
        i2c = board.I2C()  # Uses board.SCL and board.SDA for I2C communication
        # Attach sensor tracker to enable access to temperature and humidity readings.
        self.st = SensorTracker(broker=broker)
        self.mox = adafruit_ens160.ENS160(i2c)
        self.logger.debug(f"Firmware Version: {self.mox.firmware_version}")

    def get_sensor(self) -> Dict[str, Any]:
        """Retrieve and process sensor data from the MOX gas sensor.

        This method retrieves temperature and humidity data via a SensorTracker,
        applies compensation to the sensor readings, and gathers the sensor's AQI,
        TVOC, and eCO2 measurements along with a timestamp.

        Returns:
            A dictionary containing:
                - MOX AQI value.
                - MOX TVOC value.
                - MOX eCO2 value.
                - A timestamp indicating when the data was retrieved.
        """
        # Get temperature and humidity data
        th_data = self.st.get_sensor_status(THEnums.TH_STATUS_KEY.value)
        # Retrieve temperature and humidity compensation values, using defaults if necessary.
        tc = th_data.get(THEnums.TH_CELSIUS_KEY.value, 25)
        hc = th_data.get(THEnums.TH_HUMIDITY_KEY.value, 50)
        self.logger.debug(f"Setting Temperature Compensation to: {tc}")
        self.mox.temperature_compensation = tc
        self.logger.debug(f"Setting Humidity Compensation to: {hc}")
        self.mox.humidity_compensation = hc

        sensor_data: Dict[str, Any] = {
            MOXEnums.MOX_AQI.value: self.mox.AQI,
            MOXEnums.MOX_TVOC.value: self.mox.TVOC,
            MOXEnums.MOX_ECO2.value: self.mox.eCO2,
            MOXEnums.MOX_TIME_STAMP_KEY.value: time(),
        }
        self.logger.debug(f"MOX Data: {sensor_data}")
        return sensor_data

    def run(self) -> None:
        """Continuously poll the sensor and publish its status via MQTT.

        This method runs in a separate thread, continuously retrieving sensor data,
        building a status message, and sending it through the MQTT client.
        """
        self.logger.info("MOX Gas Sensor polling started")
        while True:
            status = MoxGasMessageBuilder.send_mox_status_message(self.get_sensor())
            self.send_command(
                topic=MQTTEnums.MOX_STATUS_TOPIC.value,
                command=status
            )
            sleep(0.1)


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
        import adafruit_vl53l4cd
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
            sleep(0.001)  # 1ms poll — prevents 100% CPU spin
        self.vl53.clear_interrupt()
        sdata: Dict[str, Any] = {
            TOFEnums.TOF_STATUS_KEY.value: self.vl53.distance,
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


class TempHumSensor(MQTTClient, Thread):
    """SHT40 Temp and Humidity control class to poll and push updates to MQTT.

    This class is a multithreaded SHT40 controller that polls sensor data
    from an Adafruit SHT40 sensor via I2C and publishes updates using MQTT.

    Attributes:
        SHT40_broker (tuple): The broker tuple obtained from MQTTClient.
        sht40 (adafruit_sht4x): The sensor object used for readings.
    """

    SHT40_broker = MQTTClient.broker_tuple

    def __init__(self, broker: Any) -> None:
        """Initializes the TOF object.

        This method sets up the thread as a daemon, initializes the I2C
        interface, and creates the sensor object. It also initializes the
        MQTT client using the provided broker details.

        Args:
            broker (Any): An object with 'ip' and 'port' attributes required
                to establish the MQTT connection.
        """
        import adafruit_sht4x
        self.__name__ = self.__class__.__name__
        Thread.__init__(self)
        self.daemon = True
        i2c = board.I2C()  # uses board.SCL and board.SDA
        self.sht40 = adafruit_sht4x.SHT4x(i2c)
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        MQTTClient.__init__(self, ip=broker.ip, port=broker.port)
        self.logger.debug(adafruit_sht4x.Mode.string[self.sht40.mode])

    def convert_c2f(self, temp: float) -> float:
        """Convert Celsius to Fahrenheit

        This method converts Celsius Temperature data to Fahrenheit

        Args:
            temp (float): Celsius temperature to convert

        Returns:
            float: Returns a float of the temperature in Fahrenheit from Celsius
        """
        f_temp = (temp * 9 / 5) + 32
        self.logger.debug(f"Converted {temp}C to {f_temp}F")
        return round(f_temp, 1)

    def get_sensor(self) -> Dict[str, Any]:
        """Retrieve sensor data.

        This method gathers sensor Temp Humidity data, It also attaches a timestamp to the reading.

        Returns:
            Dict[str, Any]: A dictionary containing the sensor data, where the keys
            are defined by the Enums and the values are the corresponding sensor
            readings.
        """
        temp, relative_humid = self.sht40.measurements
        temp = round(temp, 1)
        relative_humid = round(relative_humid, 1)
        sdata: Dict[str, Any] = {
                THEnums.TH_HUMIDITY_KEY.value: relative_humid,
                THEnums.TH_CELSIUS_KEY.value: temp,
                THEnums.TH_FAHRENHEIT_KEY.value: self.convert_c2f(temp),
                THEnums.TH_TIME_STAMP_KEY.value: time()
        }
        self.logger.debug(f"Time and Humidity Data: {sdata}")
        return sdata

    def run(self) -> None:
        """Thread loop to send a sensor command 10x a second.

        This method continuously retrieves sensor data, builds a status
        message using the THMessageBuilder, and sends the command to the
        designated MQTT topic. The loop runs approximately 10 times per second.
        """
        self.logger.info("Temp & Humidity Sensor polling started")
        while True:
            status = THMessageBuilder.send_th_status_message(self.get_sensor())
            self.send_command(topic=MQTTEnums.TH_STATUS_TOPIC.value, command=status)
            sleep(0.1)


class SocTemp(MQTTClient, Thread):
    """SoC chip temperature sensor for Raspberry Pi thermal monitoring.

    Reads the CPU/SoC temperature from the Linux thermal zone sysfs interface
    and publishes it over MQTT. Intended for fan speed control and thermal
    monitoring on the dashboard.

    On non-Linux systems (dev machines), publishes -1 so the class is
    importable and testable everywhere without crashing.

    Attributes:
        hostname (str): System hostname, included in each MQTT message.
        poll_interval (float): Seconds between temperature reads (default 2.0).
    """

    THERMAL_ZONE_PATH = "/sys/class/thermal/thermal_zone0/temp"

    def __init__(self, broker: Any, poll_interval: float = 2.0) -> None:
        """Initialize the SoC temperature sensor.

        Args:
            broker: MQTT broker namedtuple with ip and port.
            poll_interval: Seconds between temperature reads.
        """
        import socket
        self.__name__ = self.__class__.__name__
        Thread.__init__(self)
        self.daemon = True
        self.hostname = socket.gethostname()
        self.poll_interval = poll_interval
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        MQTTClient.__init__(self, ip=broker.ip, port=broker.port)
        self._available = path.exists(self.THERMAL_ZONE_PATH)
        if not self._available:
            self.logger.warning(f"Thermal zone not found at {self.THERMAL_ZONE_PATH} "
                                f"— SoC temp will publish -1")

    def get_celsius(self) -> float:
        """Read the SoC temperature in degrees Celsius.

        Returns:
            Temperature in Celsius, or -1.0 if the thermal zone is unavailable.
        """
        if not self._available:
            return -1.0
        try:
            with open(self.THERMAL_ZONE_PATH, "r") as f:
                millidegrees = int(f.read().strip())
            return round(millidegrees / 1000.0, 1)
        except (OSError, ValueError) as e:
            self.logger.error(f"Failed to read SoC temp: {e}")
            return -1.0

    def run(self) -> None:
        """Thread loop to publish SoC temperature at the configured interval."""
        self.logger.info(f"SoC temperature polling started (interval={self.poll_interval}s)")
        while True:
            celsius = self.get_celsius()
            message = {
                SocTempEnums.HOSTNAME_KEY.value: self.hostname,
                SocTempEnums.CELSIUS_KEY.value: celsius,
                SocTempEnums.TIMESTAMP_KEY.value: time(),
            }
            status = SocTempMessageBuilder.send_soc_temp_message(message)
            self.send_command(topic=SocTempEnums.SOC_TEMP_TOPIC.value, command=status)
            self.logger.debug(f"SoC temp: {celsius}C")
            sleep(self.poll_interval)


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

    CALIBRATION_FILE = "imu_calibration.json"

    def __init__(self, broker: Any) -> None:
        """Initializes the IMU object.

        This method sets up the thread as a daemon, initializes the I2C
        interface, and creates the sensor object. It also initializes the
        MQTT client using the provided broker details. If a saved calibration
        file exists, it is loaded to skip the manual calibration dance.

        Args:
            broker (Any): An object with 'ip' and 'port' attributes required
                to establish the MQTT connection.
        """
        import adafruit_bno055
        self.__name__ = self.__class__.__name__
        Thread.__init__(self)
        self.daemon = True
        i2c = board.I2C()  # uses board.SCL and board.SDA
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        MQTTClient.__init__(self, ip=broker.ip, port=broker.port)
        self.last_val = 0xFFFF
        self._load_calibration()

    def _load_calibration(self) -> None:
        """Load saved calibration offsets from disk if available."""
        import json
        cal_path = self.CALIBRATION_FILE
        if not path.exists(cal_path):
            self.logger.info("No saved IMU calibration found, manual calibration needed")
            return
        try:
            with open(cal_path, 'r') as f:
                cal = json.load(f)
            # Must switch to config mode to write offsets
            import adafruit_bno055
            self.sensor.mode = adafruit_bno055.CONFIG_MODE
            sleep(0.025)
            self.sensor.offsets_accelerometer = tuple(cal["accelerometer"])
            self.sensor.offsets_magnetometer = tuple(cal["magnetometer"])
            self.sensor.offsets_gyroscope = tuple(cal["gyroscope"])
            self.sensor.mode = adafruit_bno055.NDOF_MODE
            sleep(0.025)
            self.logger.info(f"IMU calibration loaded from {cal_path}")
        except Exception as e:
            self.logger.error(f"Failed to load IMU calibration: {e}")

    def save_calibration(self) -> bool:
        """Save current calibration offsets to disk for next boot.

        Only saves if the sensor is fully calibrated (all status values == 3).

        Returns:
            True if calibration was saved, False if not fully calibrated.
        """
        import json
        sys_cal, gyro_cal, accel_cal, mag_cal = self.sensor.calibration_status
        if not all(v == 3 for v in (sys_cal, gyro_cal, accel_cal, mag_cal)):
            self.logger.warning(
                f"IMU not fully calibrated (sys={sys_cal} gyro={gyro_cal} "
                f"accel={accel_cal} mag={mag_cal}), skipping save"
            )
            return False
        cal = {
            "accelerometer": list(self.sensor.offsets_accelerometer),
            "magnetometer": list(self.sensor.offsets_magnetometer),
            "gyroscope": list(self.sensor.offsets_gyroscope),
        }
        cal_path = self.CALIBRATION_FILE
        with open(cal_path, 'w') as f:
            json.dump(cal, f, indent=2)
        self.logger.info(f"IMU calibration saved to {cal_path}")
        return True

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

        Reads each I2C register individually with error handling. The BNO055
        returns (None, None, None) during calibration or bus glitches — those
        are passed through and handled downstream. Only essential registers
        (euler, gyro, quaternion, linear_accel) are read every cycle.
        Temperature and calibration are read less frequently to reduce bus load.

        Returns:
            Dict[str, Any]: A dictionary containing the sensor data.
        """
        sdata: Dict[str, Any] = {IMUEnums.IMU_TIME_STAMP_KEY.value: time()}

        # Essential readings (every cycle)
        for key, attr in [
            (IMUEnums.EULER_KEY.value, "euler"),
            (IMUEnums.GYRO_KEY.value, "gyro"),
            (IMUEnums.QUAT_KEY.value, "quaternion"),
            (IMUEnums.LINEAR_KEY.value, "linear_acceleration"),
        ]:
            try:
                sdata[key] = getattr(self.sensor, attr)
            except OSError:
                sdata[key] = None

        # Less critical readings (still included for completeness)
        for key, attr in [
            (IMUEnums.ACCEL_KEY.value, "acceleration"),
            (IMUEnums.MAGNETO_KEY.value, "magnetic"),
            (IMUEnums.GRAVITY_KEY.value, "gravity"),
            (IMUEnums.CALIBRATION_STATUS_KEY.value, "calibration_status"),
        ]:
            try:
                sdata[key] = getattr(self.sensor, attr)
            except OSError:
                sdata[key] = None

        try:
            sdata[IMUEnums.TEMP_KEY.value] = self.temperature()
        except OSError:
            sdata[IMUEnums.TEMP_KEY.value] = None

        self.logger.debug(f"IMU Data: {sdata}")
        return sdata

    def run(self) -> None:
        """Thread loop to send a sensor command 10x a second.

        This method continuously retrieves sensor data, builds a status
        message using the IMUMessageBuilder, and sends the command to the
        designated MQTT topic. The loop runs approximately 10 times per second.
        Auto-saves calibration once fully calibrated.
        """
        self.logger.info("IMU Sensor polling started")
        calibration_saved = path.exists(self.CALIBRATION_FILE)
        backoff = 0.1
        max_backoff = 5.0
        while True:
            try:
                status = IMUMessageBuilder.send_imu_status_message(self.get_sensor())
                self.send_command(topic=MQTTEnums.IMU_STATUS_TOPIC.value, command=status)
                # Auto-save calibration once all sensors reach status 3
                if not calibration_saved:
                    if self.save_calibration():
                        calibration_saved = True
                backoff = 0.1  # reset on success
            except OSError as e:
                self.logger.error(f"IMU I2C error: {e} — retrying in {backoff:.1f}s")
                sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue
            sleep(0.1)


class Gservo(MQTTClient, Thread):
    """
    Spring-damper servo controller. Runs a continuous 50Hz physics loop that smoothly
    chases target angles with organic acceleration, overshoot, and deceleration.
    New targets blend in seamlessly -- no abort/restart jitter.

    Head servos use fast response (high omega) with slight overshoot (zeta < 1).
    Body servos use slow response (low omega) with critical damping (zeta = 1).
    Both move simultaneously; head locks on first, body swings in behind.
    """
    def __init__(self, location: str, servo, axis: str, broker: NamedTuple,
                 servo_range: NamedTuple, pulse_max_min=None, servo_speed: float = 0.1) -> None:
        self.__name__ = f"{self.__class__.__name__}_{location}"
        Thread.__init__(self)
        self.daemon = True
        self.stop = False
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.location: str = location
        self.cmd_topic = ServoEnum.MQTT_COMMAND_TOPIC.value
        self.status_topic = ServoEnum.MQTT_STATUS_TOPIC.value
        self.intensity_topic = SystemEnums.MQTT_INTENSITY_TOPIC.value
        self.topic_handler: Dict[str, Callable] = {
            self.cmd_topic: self.handle_cmd,
            self.intensity_topic: self.handle_intensity}
        self.servo = servo
        if pulse_max_min is not None:
            self.servo.set_pulse_width_range(min_pulse=pulse_max_min.min, max_pulse=pulse_max_min.max)
        self.servo_range = servo_range
        self.middle_angle: int = int(self.servo_range.center)
        self.axis: str = axis.lower()

        # Determine servo type (head vs body) for spring-damper parameter selection
        self._is_head_servo: bool = location in ServoEnum.HEAD_SERVO_LOCATIONS.value

        # Spring-damper state
        self.target_angle: float = float(self.middle_angle)
        self.position: float = float(self.middle_angle)
        self.velocity: float = 0.0
        self.speed: int = MotionProfile.DEFAULT_TRACKING_SPEED.value
        self._update_spring_params(self.speed)

        # Backward-compatible aliases (used by external code reading servo state)
        self.angle: float = self.target_angle
        self.current_angle: float = self.position
        self.moving: bool = False
        self._tick_count: int = 0

        # Call the superclass constructor to initialize MQTTClient and the lock
        MQTTClient.__init__(self, ip=broker.ip, port=broker.port)

        # Pipeline debug (structured MQTT debug topic)
        self._pdebug = PipelineDebug(self, "body_server")

        # Set hardware to initial position
        self.servo.angle = self.position
        self.send_status()

    def _update_spring_params(self, speed_level: int) -> None:
        """Update omega and zeta from the MotionProfile based on speed level and servo type."""
        speed_level = max(1, min(5, speed_level))
        if self._is_head_servo:
            params = MotionProfile.HEAD_PARAMS.value
        else:
            params = MotionProfile.BODY_PARAMS.value
        self.omega, self.zeta = params[speed_level]
        self.logger.debug(f"{self.location}: spring params speed={speed_level} omega={self.omega} zeta={self.zeta}")

    def send_status(self) -> None:
        """Send current position, velocity, and servo state to MQTT."""
        status = ServoMessageBuilder.send_status(self.location, self.get_angles())
        self.send_command(topic=self.status_topic, command=status)

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """Handle incoming MQTT commands: move, move_all, and status requests."""
        try:
            j_msg = loads(msg.payload.decode())
        except JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON message: {e}")
            return

        cmd = j_msg.get(ServoEnum.MSG_COMMAND_KEY.value, "")

        # Single servo move command (backward compatible)
        if j_msg.get(ServoEnum.MSG_LOCATION_KEY.value, "") == self.location:
            if cmd == ServoEnum.MSG_COMMAND_MOVE.value:
                self.logger.debug(f"{self.location}, {msg.topic}, {j_msg}")
                angle = int(j_msg.get(ServoEnum.MSG_ANGLE.value, self.middle_angle))
                speed = int(j_msg.get(ServoEnum.MSG_SPEED.value, self.speed))
                self.set_speed_angle((speed, angle))
            elif cmd == ServoEnum.MSG_COMMAND_STATUS.value:
                self.send_status()

        # Consolidated move_all command: all servos targeted in one message
        elif cmd == ServoEnum.MSG_COMMAND_MOVE_ALL.value:
            targets = j_msg.get(ServoEnum.MSG_TARGETS.value, {})
            my_target = targets.get(self.location, None)
            if my_target is not None:
                angle = int(my_target.get(ServoEnum.MSG_ANGLE.value, self.middle_angle))
                speed = int(my_target.get(ServoEnum.MSG_SPEED.value, self.speed))
                # Log trace latency if present
                trace_id = j_msg.get(TraceEnums.TRACE_ID.value)
                ts_vision = j_msg.get(TraceEnums.TS_VISION.value)
                if trace_id and ts_vision:
                    from time import time as _time
                    latency_ms = round((_time() - ts_vision) * 1000, 1)
                    self.logger.debug(f"{self.location}: trace={trace_id} latency={latency_ms}ms "
                                      f"angle={angle} speed={speed}")
                else:
                    self.logger.debug(f"{self.location}: move_all target angle={angle} speed={speed}")
                self.set_speed_angle((speed, angle))

    def run(self) -> None:
        """Continuous 50Hz spring-damper physics loop.

        Every tick:
        - Read target under lock
        - Compute spring-damper acceleration and integrate
        - Write to servo hardware
        - Periodically send MQTT status
        """
        dt = MotionProfile.PHYSICS_DT.value
        status_interval = MotionProfile.STATUS_SEND_INTERVAL.value
        vel_thresh = MotionProfile.MOVING_VELOCITY_THRESHOLD.value
        pos_thresh = MotionProfile.MOVING_POSITION_THRESHOLD.value

        # Gentle startup: skip hardware writes for 2 seconds so the
        # servo doesn't snap from its physical position. The spring-damper
        # physics still run (position/velocity update), and the first
        # external command (from tracking or calibration) sets the real
        # target. After 2 seconds, writes begin and the servo eases to
        # wherever the spring-damper has reached.
        startup_grace = time() + 2.0
        self.logger.info(f"{self.location}: startup grace period (2s)")

        _prev_moving = False
        _prev_clamped = False
        _physics_log_counter = 0

        while not self.stop:
            # Read state under lock
            with self._lock:
                target = self.target_angle
                pos = self.position
                vel = self.velocity
                omega = self.omega
                zeta = self.zeta

            # Spring-damper physics
            error = target - pos
            accel = (omega ** 2) * error - 2.0 * zeta * omega * vel
            vel += accel * dt
            pos += vel * dt

            # Clamp to physical servo range
            clamped = False
            if pos <= self.servo_range.min:
                pos = float(self.servo_range.min)
                if vel < 0:
                    vel = 0.0
                clamped = True
            elif pos >= self.servo_range.max:
                pos = float(self.servo_range.max)
                if vel > 0:
                    vel = 0.0
                clamped = True

            # Write to hardware (skip during startup grace to avoid snap)
            if time() > startup_grace:
                self.servo.angle = pos
            elif _physics_log_counter == 0:
                self.logger.debug(f"{self.location}: startup grace — skipping hardware write")

            # Update state under lock
            with self._lock:
                self.position = pos
                self.velocity = vel
                now_moving = abs(vel) > vel_thresh or abs(error) > pos_thresh
                self.moving = now_moving
                # Keep backward-compatible aliases in sync
                self.current_angle = pos

            # Log moving state transitions
            if now_moving != _prev_moving:
                self.logger.debug(
                    f"SERVO {self.location}: moving={'START' if now_moving else 'STOP'} "
                    f"pos={pos:.2f} target={target:.2f} vel={vel:.2f} error={error:.2f}")
                self._pdebug.log(self.__name__, "MOVING", {
                    "state": "START" if now_moving else "STOP",
                    "pos": round(pos, 2), "target": round(target, 2),
                    "vel": round(vel, 2), "error": round(error, 2),
                })
                _prev_moving = now_moving

            # Log physics state every ~1s (every 50 ticks at 50Hz) when moving
            _physics_log_counter += 1
            if now_moving and _physics_log_counter % 50 == 0:
                self.logger.debug(
                    f"SERVO {self.location}: pos={pos:.2f} target={target:.2f} "
                    f"vel={vel:.2f} accel={accel:.2f} error={error:.2f} "
                    f"omega={omega:.1f} zeta={zeta:.2f} clamped={clamped}")
                self._pdebug.log(self.__name__, "PHYSICS", {
                    "pos": round(pos, 2), "target": round(target, 2),
                    "vel": round(vel, 2), "accel": round(accel, 2),
                    "error": round(error, 2), "omega": round(omega, 1),
                    "zeta": round(zeta, 2), "clamped": clamped,
                })

            if clamped and not _prev_clamped:
                self.logger.debug(
                    f"SERVO {self.location}: CLAMPED at "
                    f"{'min=' + str(self.servo_range.min) if pos <= self.servo_range.min else 'max=' + str(self.servo_range.max)} "
                    f"target={target:.2f}")
                self._pdebug.log(self.__name__, "CLAMPED", {
                    "pos": round(pos, 2), "target": round(target, 2),
                    "limit": self.servo_range.min if pos <= self.servo_range.min else self.servo_range.max,
                })
            _prev_clamped = clamped

            # Periodic MQTT status broadcast (~5Hz)
            self._tick_count += 1
            if self._tick_count % status_interval == 0:
                self.send_status()

            sleep(dt)

    def handle_intensity(self, msg: MQTTMessage) -> None:
        # TODO: Implement intensity handling
        pass

    def get_angles(self) -> dict:
        """Return servo state including position, velocity, and range."""
        with self._lock:
            return {
                ServoEnum.MSG_MAX.value: self.servo_range.max,
                ServoEnum.MSG_MIN.value: self.servo_range.min,
                ServoEnum.MSG_MIDDLE.value: self.middle_angle,
                ServoEnum.MSG_CURRENT_ANGLE.value: round(self.position, 2),
                ServoEnum.MSG_VELOCITY.value: round(self.velocity, 2),
                ServoEnum.MSG_AXIS.value: self.axis,
                ServoEnum.MSG_MOVING.value: self.moving,
                ServoEnum.MSG_LAST_ANGLE.value: round(self.target_angle, 2)
            }

    def set_speed(self, speed: int) -> None:
        """Set speed level (1-5) and update spring-damper parameters."""
        speed = max(1, min(5, round(speed)))
        with self._lock:
            self.speed = speed
        self._update_spring_params(speed)

    def set_angle(self, angle: float) -> None:
        """Set target angle, clamped to servo range."""
        max_angle = self.servo_range.max
        min_angle = self.servo_range.min
        with self._lock:
            self.target_angle = float(max(min_angle, min(max_angle, angle)))
            # Keep backward-compatible alias in sync
            self.angle = self.target_angle
            self.logger.debug(f"Target angle set to {self.target_angle}")

    def set_speed_angle(self, speed_angle: Tuple[int, int]) -> None:
        """Set speed and target angle as one call."""
        speed, angle = speed_angle
        self.set_speed(speed)
        self.set_angle(angle)

    def get_angle(self) -> float:
        """Return estimated current position."""
        with self._lock:
            return self.position

    def get_moving_status(self) -> bool:
        """Return whether servo is currently in motion."""
        with self._lock:
            return self.moving


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
        self.cmd_topic: str = MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value
        self.intensity_topic: str = MQTTEnums.SYSTEM_INTENSITY_TOPIC.value
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd,
                                                   self.intensity_topic: self.handle_intensity}
        self.location: str = LEDShoulders.LOCATION.value
        self.animations: Dict[str, Callable] = {
            LEDShoulders.ANIMATION_STARTUP.value: self.startup,
            LEDShoulders.ANIMATION_DISCO.value: self.disco,
            LEDShoulders.ANIMATION_TWINKLE.value: self.twinkle,
        }
        MQTTClient.__init__(self, broker.ip, broker.port)

    def handle_cmd(self, msg: MQTTMessage) -> None:
        j_msg = loads(msg.payload.decode())
        if j_msg.get(LEDShoulders.MSG_TYPE_KEY.value, "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {j_msg}")
            cmd = j_msg[self.location].get(LEDShoulders.MSG_COMMAND_KEY.value, "")
            if cmd in self.animations.keys():
                self.animations[cmd]()

    def handle_intensity(self, msg: MQTTMessage) -> None:
        # TODO figure out update commands
        j_msg = loads(msg.payload.decode())
        if j_msg.get(LEDShoulders.MSG_TYPE_KEY.value, "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {j_msg}")
            self.intensity = j_msg[LEDShoulders.MSG_INTENSITY_KEY.value]

    def startup(self) -> None:
        self.ani.stop_twinkle()
        self.ani.stop_disco()
        for p in range(0, 63):
            self.pixels[p] = self.lh.adjust_brightness((255, 0, 0), self.intensity[0])
            sleep(.2)

    def disco(self) -> None:
        self.ani.stop_twinkle()
        self.logger.debug("Triggered Disco Mode")
        self.pixels.brightness = self.intensity[0]
        Thread(target=self.ani.disco, args=(0.05, "RGB")).start()

    def twinkle(self) -> None:
        self.logger.debug("Triggered Twinkle Mode")
        self.pixels.brightness = self.intensity[0]
        Thread(target=self.ani.twinkle, args=((255, 0, 0),), kwargs={"pixel_grid": self.stripes}).start()


class DumbLEDController(Thread):
    """
    LED Controller for pca9685
    """
    def __init__(self, channel: int, duty_cycle: int = 100, pca_address: int = 0x40) -> None:
        Thread.__init__(self)
        self.daemon = True
        self.logger = setup_logger(name="DumbLEDController", console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        self.hat = adafruit_pca9685.PCA9685(busio.I2C(board.SCL, board.SDA), address=pca_address)
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
    def __init__(self, broker: NamedTuple, pca_address: int = 0x40, pwm_channel: int = 4) -> None:
        self.__name__ = "Head_LED_Controller"
        # Logger is owned by this class — MQTTClient does not create one
        # TODO: Extract NeoPixelAnimations + LedHelper into a standalone LedController
        # module so Pi5 DotStar LEDs can reuse the same animation code without
        # importing BodyControlModules (which pulls in hardware deps)
        self.logger = setup_logger(self.__name__, console_logging=LoggingEnums.LOG_LEVEL_DEBUG.value)
        self.pixels = neopixel.NeoPixel(board.D18, 1, brightness=1, auto_write=True, pixel_order=neopixel.RGB)
        self.ani = NeoPixelAnimations(self.pixels, 1)
        self.swap = LedHelper.rgb2grb_swap
        self.hat = adafruit_pca9685.PCA9685(busio.I2C(board.SCL, board.SDA), address=pca_address)
        self.pwm_led = self.hat.channels[pwm_channel]
        self.hat.frequency = 60
        self.pwm_led.duty_cycle = 250
        self.pwm_ani = PWMLedAnimations(self.pwm_led)
        self.intensity: Tuple[float, float] = (.1, .5)
        self.cmd_topic: str = MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value
        self.intensity_topic: str = MQTTEnums.SYSTEM_INTENSITY_TOPIC.value
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd,
                                                   self.intensity_topic: self.handle_intensity}
        self.location: str = LEDHead.EYE_LED_LOCATION.value
        self.animations: Dict[str, Callable] = {LEDHead.ANIMATION_STARTUP_KEY.value: self.startup,
                                                LEDHead.ANIMATION_DISCO_KEY.value: self.disco,
                                                LEDHead.ANIMATION_ANGRY_EYE_KEY.value: self.angry_eye,
                                                LEDHead.ANIMATION_NORMAL_EYE_KEY.value: self.normal_eye,
                                                LEDHead.ANIMATION_SPEECH_EYE_KEY.value: self.speach_eye}
        self.glados_eye: Tuple[int, int, int] = (255, 165, 0)
        MQTTClient.__init__(self, broker.ip, broker.port)

    def handle_cmd(self, msg: MQTTMessage) -> None:
        j_msg = loads(msg.payload.decode())
        body = j_msg.get(LEDHead.MSG_COMMAND_KEY.value, {})
        if body.get(LEDHead.MSG_COMMAND_LOCATION_KEY.value, "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic}, {j_msg}")
            animation_key = body.get(LEDHead.MSG_COMMAND_KEY.value, "")
            if animation_key in self.animations.keys():
                if animation_key == LEDHead.ANIMATION_SPEECH_EYE_KEY.value:
                    args = body.get(LEDHead.MSG_COMMAND_ARGUMENTS_KEY.value, '')
                    if args != '':
                        self.logger.debug("LED Head Animation Triggered with arguments")
                        # Run in separate thread so MQTT callback isn't blocked
                        Thread(target=self.animations[animation_key],
                               kwargs=args, daemon=True).start()
                else:
                    self.logger.debug("LED Head Animation Triggered with no arguments")
                    self.animations[animation_key]()

    def handle_intensity(self, msg: MQTTMessage) -> None:
        # TODO figure out update commands
        j_msg = loads(msg.payload.decode())
        if j_msg.get(LEDHead.MSG_COMMAND_TYPE_KEY.value, "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic},  {j_msg}")
            self.intensity = j_msg[LEDHead.MSG_INTENSITY_KEY.value]

    def startup(self) -> None:
        self.logger.debug("Startup Sequence")
        eye_led_thread = Thread(target=self.ani.intensity, args=(10, self.glados_eye))
        pwm_led_thread = Thread(target=self.pwm_ani.intensity, args=(10,))
        eye_led_thread.start()
        pwm_led_thread.start()
        eye_led_thread.join()
        pwm_led_thread.join()
        self.normal_eye()

    def disco(self) -> None:
        self.logger.debug("Triggered Disco Mode")
        self.pixels.brightness = self.intensity[0]
        eye_led_thread = Thread(target=self.ani.rainbow_cycle, args=(.05, "RGB"))
        pwm_led_thread = Thread(target=self.pwm_ani.intensity, args=(10,))
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

    def speach_eye(self, **kwargs) -> None:
        """Pulse eye LED in sync with speech word timing.

        Accepts time_dict as a list of (word_duration, gap_to_next) tuples.
        Also accepts an optional color tuple to pulse with mood-driven colors.
        """
        self.logger.debug("Speech Eye Pulse Triggered")
        td = LEDHead.ARGS_KEY_TIME_DICT.value
        delay = LEDHead.ARGS_KEY_DELAY.value
        color_key = LEDHead.ARGS_KEY_COLOR.value
        if {td, delay}.issubset(kwargs):
            ds = float(kwargs[delay])
            pulse_color = tuple(kwargs.get(color_key, self.glados_eye))
            self.logger.debug(f"Sleeping for {ds} seconds for audio sync")
            sleep(ds)
            time_entries = kwargs[td]
            for entry in time_entries:
                # entry is (word_duration, gap_to_next) tuple/list
                word_duration = float(entry[0])
                gap = float(entry[1])
                self.ani.speech_pulse(wait=word_duration, color=pulse_color)
                sleep(gap)
            self.normal_eye()
        else:
            self.logger.error(f"Missing either {td} or {delay} in {kwargs} arguments, speech eye animation failed")


class Relay:
    """Controls an active-high relay connected to a Raspberry Pi GPIO pin.

    This class assumes an *active-high* relay:

    - GPIO HIGH -> Relay ON
    - GPIO LOW -> Relay OFF

    Attributes:
        pin: BCM GPIO pin number used to control the relay.
    """

    def __init__(self, pin: int) -> None:
        """Initializes the relay GPIO pin as an output and turns it OFF.

        Args:
            pin: BCM GPIO pin number (not physical board pin number).
        """
        self.pin: int = pin

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)  # Start OFF

    def on(self) -> None:
        """Turns the relay ON (sets GPIO HIGH)."""
        GPIO.output(self.pin, GPIO.HIGH)

    def off(self) -> None:
        """Turns the relay OFF (sets GPIO LOW)."""
        GPIO.output(self.pin, GPIO.LOW)

    def toggle(self) -> bool:
        """Toggles the relay state.

        Returns:
            The new relay state (True for ON, False for OFF).
        """
        current_state: bool = bool(GPIO.input(self.pin))
        new_state: bool = not current_state
        GPIO.output(self.pin, GPIO.HIGH if new_state else GPIO.LOW)
        return new_state

    def pulse(self, duration_s: float) -> None:
        """Turns the relay ON for a fixed duration, then turns it OFF.

        Args:
            duration_s: Pulse duration in seconds.

        Raises:
            ValueError: If `duration_s` is negative.
        """
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative.")

        self.on()
        sleep(duration_s)
        self.off()

    def cleanup(self) -> None:
        """Turns the relay OFF and releases the GPIO pin."""
        self.off()  # Fail-safe OFF
        GPIO.cleanup(self.pin)


class LampStrip8(MQTTClient):
    """Controls a strip of 8 NeoPixel LEDs via MQTT."""

    NUM_PIXELS = 8
    MAX_INTENSITY = 0.9

    def __init__(self, broker: NamedTuple, pin=board.D21, pixel_order=neopixel.RGBW) -> None:
        self.__name__ = "Lamp_Strip8_Controller"
        self.logger = setup_logger(self.__name__, LoggingEnums.LOG_LEVEL_INFO.value)
        self.pixels = neopixel.NeoPixel(pin, self.NUM_PIXELS, brightness=0.5, auto_write=False,
                                        pixel_order=pixel_order)
        self.lh = LedHelper
        self.ani = NeoPixelAnimations(self.pixels, self.NUM_PIXELS)
        self.intensity: Tuple[float, float] = (0.5, 0.5)
        self.cmd_topic: str = MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value
        self.intensity_topic: str = MQTTEnums.SYSTEM_INTENSITY_TOPIC.value
        self.topic_handler: Dict[str, Callable] = {self.cmd_topic: self.handle_cmd,
                                                   self.intensity_topic: self.handle_intensity}
        self.location: str = LEDLampStrip8.LOCATION.value
        self.animations: Dict[str, Callable] = {
            LEDLampStrip8.ANIMATION_STARTUP.value: self.startup,
            LEDLampStrip8.ANIMATION_DISCO.value: self.disco,
            LEDLampStrip8.ANIMATION_TWINKLE.value: self.twinkle,
            LEDLampStrip8.ANIMATION_PULSE.value: self.pulse,
            LEDLampStrip8.ANIMATION_CLEAR.value: self.clear,
            LEDLampStrip8.ANIMATION_WHITE_LIGHT.value: self.white_light,
        }
        self._twinkle_loop: bool = False
        MQTTClient.__init__(self, broker.ip, broker.port)

    def handle_cmd(self, msg: MQTTMessage) -> None:
        j_msg = loads(msg.payload.decode())
        if j_msg.get(LEDLampStrip8.MSG_TYPE_KEY.value, "") == self.location:
            self.logger.debug(f"{self.location}, {msg.topic}, {j_msg}")
            body = j_msg.get(self.location, {})
            cmd = body.get(LEDLampStrip8.MSG_COMMAND_KEY.value, "")
            if cmd in self.animations:
                args = body.get(LEDLampStrip8.MSG_ARGUMENTS_KEY.value, {})
                if args:
                    self.animations[cmd](**args)
                else:
                    self.animations[cmd]()

    def handle_intensity(self, msg: MQTTMessage) -> None:
        j_msg = loads(msg.payload.decode())
        if j_msg.get(LEDLampStrip8.MSG_TYPE_KEY.value, "") == self.location:
            raw = j_msg[LEDLampStrip8.MSG_INTENSITY_KEY.value]
            self.intensity = (self._validate_intensity(raw[0]),
                              self._validate_intensity(raw[1]))
            self.pixels.brightness = self.intensity[0]

    def _validate_intensity(self, value: float) -> float:
        if value >= 1.0:
            msg = f"Intensity {value} is >= 1.0, which is not allowed for LampStrip8 (max {self.MAX_INTENSITY})"
            self.logger.error(msg)
            raise ValueError(msg)
        return min(value, self.MAX_INTENSITY)

    def set_all(self, color: Tuple[int, int, int, int]) -> None:
        """Set all 8 pixels to the same RGBW color."""
        self.pixels.fill(color)
        self.pixels.show()

    def set_pixel(self, index: int, color: Tuple[int, int, int, int]) -> None:
        """Set a single pixel by index (0-7)."""
        if 0 <= index < self.NUM_PIXELS:
            self.pixels[index] = color
            self.pixels.show()

    def white_light(self, intensity: float = 0.5) -> None:
        """Set all pixels to pure white using the W channel at the given intensity."""
        self._twinkle_loop = False
        safe_intensity = self._validate_intensity(intensity)
        self.pixels.brightness = safe_intensity
        self.pixels.fill((0, 0, 0, 255))
        self.pixels.show()

    def clear(self) -> None:
        """Turn off all pixels."""
        self._twinkle_loop = False
        self.pixels.fill((0, 0, 0, 0))
        self.pixels.show()

    def startup(self) -> None:
        """Sequential fill from first to last pixel."""
        self._twinkle_loop = False
        self.pixels.brightness = self.intensity[0]
        color = self.lh.adjust_brightness((255, 165, 0, 0), self.intensity[0])
        for i in range(self.NUM_PIXELS):
            self.pixels[i] = color
            self.pixels.show()
            sleep(0.1)

    def disco(self) -> None:
        """Rainbow cycle across all pixels."""
        self._twinkle_loop = False
        self.pixels.brightness = self.intensity[0]
        thread = Thread(target=self.ani.rainbow_cycle, args=(0.05, "RGBW"))
        thread.start()

    def twinkle(self, color: Tuple[int, int, int, int] = (255, 165, 0, 0)) -> None:
        """Randomly vary brightness across pixels."""
        self._twinkle_loop = True
        self.pixels.brightness = self.intensity[0]
        while self._twinkle_loop:
            for i in range(self.NUM_PIXELS):
                self.pixels[i] = self.lh.adjust_brightness(color, random.choice([x / 10.0 for x in range(1, 9)]))
            self.pixels.show()
            sleep(0.1)

    def pulse(self, color: Tuple[int, int, int, int] = (255, 165, 0, 0)) -> None:
        """Pulse all pixels from dim to bright and back."""
        self._twinkle_loop = False
        for level in list(range(1, 11)) + list(range(10, 0, -1)):
            self.pixels.fill(self.lh.adjust_brightness(color, level / 10.0))
            self.pixels.show()
            sleep(0.05)


if __name__ == "__main__":
    from adafruit_servokit import ServoKit
    ip = '192.168.1.39'
    Angle_tuple = namedtuple("angle", ['max', 'min', 'center'])
    Pulse_tuple = namedtuple("pulse", ['max', 'min'])
    Mqtt_tuple = namedtuple("mqtt", ["ip", "port"])
    mqtt_connect = Mqtt_tuple(ip, 1883)
    mg90d_pulse = Pulse_tuple(2665, 610)
    mg92b_pulse = Pulse_tuple(2550, 605)
    head_angle = Angle_tuple(173, 6, 83)
    neck_angle = Angle_tuple(120, 52, 92)
    default_angle = Angle_tuple(180, 0, 90)
    kit = ServoKit(channels=16, address=0x40)
    led_head = LedHead(broker=mqtt_connect, pca_address=0x40)
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
    th = TempHumSensor(broker=mqtt_connect)
    th.start()
    mg = MoxGas(broker=mqtt_connect)
    mg.start()
    while True:
        sleep(1)
