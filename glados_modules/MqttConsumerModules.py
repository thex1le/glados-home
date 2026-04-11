from typing import Dict, Callable, NamedTuple, Any
from json import loads, JSONDecodeError
from time import time, sleep
from collections import namedtuple

# 3rd party imports
from paho.mqtt.client import MQTTMessage
from cachetools import TTLCache

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.MqttConnector import MQTTClient, TargetMessageBuilder, ServoMessageBuilder
from glados_modules.PipelineDebug import PipelineDebug
from glados_modules.GladosEnums import (
    ServoEnum, CameraEnum, VisionResultsEnum, TrackingEnums,
    LoggingEnums, IMUEnums, MQTTEnums, TOFEnums, THEnums, MOXEnums
)


class ServoLocation(MQTTClient):
    """Keep track of all the angles based on MQTT status updates.

    This class monitors servo status messages received via MQTT and updates
    an internal mapping of servo locations and their current states.
    """

    def __init__(self, broker: NamedTuple) -> None:
        """Initialize a ServoLocation instance.

        This method sets up the logger, topic handlers, servo configurations,
        and initializes the MQTT client with broker information.

        Args:
            broker (NamedTuple): A named tuple containing broker details with
                attributes 'ip' and 'port'.
        """
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(
            name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value
        )
        # Initialize topic handlers for servo and IMU messages
        self.topic_handler: Dict[str, Callable[[MQTTMessage], None]] = {
            ServoEnum.MQTT_STATUS_TOPIC.value: self.servo_handle_cmd,
        }
        self.body_map: Dict[Any, Any] = {}
        self.min: str = ServoEnum.MSG_MIN.value
        self.max: str = ServoEnum.MSG_MAX.value
        self.current_angle: str = ServoEnum.MSG_CURRENT_ANGLE.value
        self.middle: str = ServoEnum.MSG_MIDDLE.value
        self.moving: str = ServoEnum.MSG_MOVING.value
        self.axis: str = ServoEnum.MSG_AXIS.value
        self.last_angle: str = ServoEnum.MSG_LAST_ANGLE.value
        self.velocity: str = ServoEnum.MSG_VELOCITY.value
        self.gyro_thresh: float = float(ServoEnum.IMU_GYRO_THRESH.value)
        self.gyro_spike_thresh: float = float(ServoEnum.IMU_GYRO_SPIKE_THRESH.value)
        self.accel_thresh: float = float(ServoEnum.IMU_ACCEL_THRESH.value)
        self.accel_spike_thresh: float = float(ServoEnum.IMU_ACCEL_SPIKE_THRESH.value)
        self.jerk_threshold: float = float(ServoEnum.IMU_JERK_THRESHOLD.value)
        self.stable_frames: int = int(ServoEnum.IMU_HOLD_FRAME_COUNT.value)
        self.gyro_stable_count: int = 0
        self.accel_stable_count: int = 0
        self._last_time: float = time()
        self._last_accel: int = 0
        self.ServoTuple = namedtuple(
            ServoEnum.MSG_LOCATION_KEY.value,
            [
                self.current_angle,
                self.max,
                self.min,
                self.middle,
                self.axis,
                self.moving,
                ServoEnum.MSG_LOCATION.value,
                self.last_angle,
                self.velocity,
            ],
        )
        self.servo_list = (
            ServoEnum.LOCATION_BODY_UP_DOWN.value,
            ServoEnum.LOCATION_HEAD_UP_DOWN.value,
            ServoEnum.LOCATION_BODY_LEFT_RIGHT.value,
            ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value,
        )
        # Initialize IMU status mapping
        self.imu_status: Dict[Any, Any] = {}
        self.st = SensorTracker(broker=broker)
        # Call the superclass constructor for MQTT client initialization
        super().__init__(ip=broker.ip, port=broker.port)
        # Pipeline debug (receives servo status from Pi4, runs on GPU)
        self._pdebug = PipelineDebug(self, "ai_server")

    def update_servo_status(self) -> None:
        """Trigger servo message status update.

        This method sends status request commands to all servos and waits
        until responses are received or a timeout is reached.
        """
        self.logger.debug("Updating Servo angle status")
        msg = []
        for servo_location in self.servo_list:
            msg.append(ServoMessageBuilder.get_status(servo_location))
        # Send the status request commands
        self.send_command(msg, ServoEnum.MQTT_COMMAND_TOPIC.value)
        fail_count = 0
        while True:
            with self._lock:
                current_servo_count = len(self.body_map.keys())
            # Timeout after approximately 2 minutes (600 * 0.1 seconds)
            if current_servo_count >= len(self.servo_list) or fail_count >= 600:
                break
            else:
                # Keep sending requests until all servo statuses are received
                for servo in self.servo_list:
                    with self._lock:
                        if servo not in self.body_map:
                            self.send_command(
                                ServoMessageBuilder.get_status(servo),
                                ServoEnum.MQTT_COMMAND_TOPIC.value,
                            )
                sleep(0.1)
                fail_count += 1
                self.logger.debug("Waiting for servo statuses to update...")

    def __load_message(self, json_message: MQTTMessage) -> Dict[str, Any]:
        """Decode a JSON message from MQTT.

        This private method attempts to decode the payload of a JSON MQTT
        message. If decoding fails, an error is logged.

        Args:
            json_message (MQTTMessage): The incoming MQTT message containing JSON.

        Returns:
            Dict[str, Any]: The decoded JSON as a dictionary. Returns an empty
                dictionary if decoding fails.
        """
        rtn: Dict[str, Any] = {}
        try:
            rtn = loads(json_message.payload.decode())
            self.logger.debug(f"JSON message decoded as: {rtn}")
        except JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON message: {e}")
        return rtn

    def servo_handle_cmd(self, msg: MQTTMessage) -> None:
        """Handle incoming servo status messages.

        This method processes an incoming servo status MQTT message, extracts
        the servo location and its results, and updates the internal mapping.

        Args:
            msg (MQTTMessage): The MQTT message containing servo status.
        """
        j_msg: Dict[str, Any] = self.__load_message(msg)
        if not j_msg:
            self.logger.error("Failed to decode servo status message")
        elif ServoEnum.MSG_LOCATION_KEY.value in j_msg:
            location = j_msg.get(ServoEnum.MSG_LOCATION_KEY.value)
            results = j_msg.get(ServoEnum.MSG_RESULTS.value, {})
            self.logger.debug(
                f"SERVO_STATUS: {location} pos={results.get(self.current_angle)} "
                f"target={results.get(self.last_angle)} vel={results.get(self.velocity, 0.0)} "
                f"moving={results.get(self.moving)} "
                f"range=[{results.get(self.min)},{results.get(self.max)}] "
                f"center={results.get(self.middle)}")
            self._pdebug.log("ServoLocation", "SERVO_STATUS", {
                "servo": location,
                "pos": results.get(self.current_angle),
                "target": results.get(self.last_angle),
                "vel": results.get(self.velocity, 0.0),
                "moving": results.get(self.moving),
            })
            with self._lock:
                self.body_map[location] = self.ServoTuple(
                    results.get(self.current_angle),
                    results.get(self.max),
                    results.get(self.min),
                    results.get(self.middle),
                    results.get(self.axis),
                    results.get(self.moving),
                    location,
                    results.get(self.last_angle),
                    results.get(self.velocity, 0.0),
                )

    def get_imu_status(self, block: bool = True) -> Dict[Any, Any]:
        """Get the current IMU status.

        Returns:
            Dict[Any, Any]: A dictionary representing the latest raw IMU status.
        """
        imu_status = self.st.get_sensor_status(IMUEnums.IMU_STATUS_KEY.value)
        while imu_status == {} and block is True:
            # block until we get a status
            imu_status = self.st.get_sensor_status(IMUEnums.IMU_STATUS_KEY.value)
        self.imu_status = imu_status
        return imu_status

    def get_imu_moving_status(self) -> bool:
        """Return if the IMU is showing movement

        Returns:
            Bool[True, False]: A bool if the IMU in robot head is stable or not
        """
        ret = False
        if self.imu_status != {}:
            # 1) rotational rate
            gx, gy, gz = self.imu_status['gyroscope']
            gyro_mag = max(abs(gx), abs(gy), abs(gz))
            # 2) linear acceleration (gravity‐subtracted)
            lx, ly, lz = self.imu_status['linear']
            accel_mag = max(abs(lx), abs(ly), abs(lz))
            # if we're below threshold, increment stability counter
            if gyro_mag < self.gyro_thresh:
                self.gyro_stable_count += 1
            else:
                self.gyro_stable_count = 0
            if accel_mag < self.accel_thresh:
                self.accel_stable_count += 1
            else:
                self.accel_stable_count = 0
            ret = self.gyro_stable_count >= self.stable_frames and self.accel_stable_count >= self.stable_frames
        return ret

    def get_imu_movement_type(self) -> str:
        """
        Returns one of:
          - IMU_MOVEMENT_SERVO  ("normal_movement")
          - IMU_MOVEMENT_SHOCK  ("shock")
          - None                 (i.e. still/idle, no movement at all)
        """
        imu = self.get_imu_status()
        # 1) commanded?
        commanded = self.check_movement()

        # 2) compute gyro & accel magnitudes
        gx, gy, gz = imu['gyroscope']
        lx, ly, lz = imu['linear']  # already gravity‑subtracted
        gyro_mag = max(abs(gx), abs(gy), abs(gz))
        accel_mag = max(abs(lx), abs(ly), abs(lz))

        # 3) compute jerk
        now = imu['time']
        # assume self._last_accel and self._last_time stored from previous call
        if hasattr(self, '_last_time') and now > self._last_time:
            dt = now - self._last_time
            jerk = abs(accel_mag - self._last_accel) / dt
        else:
            jerk = 0.0
        self._last_accel = accel_mag
        self._last_time = now

        # 4) spike test
        is_spike = (
                gyro_mag > self.gyro_spike_thresh or
                accel_mag > self.accel_spike_thresh or
                jerk > self.jerk_threshold
        )

        # 5) classify
        if commanded:
            # during a servo‐driven motion
            return (
                ServoEnum.IMU_MOVEMENT_SHOCK.value
                if is_spike
                else ServoEnum.IMU_MOVEMENT_SERVO.value
            )
        else:
            # idle robot
            if is_spike:
                return ServoEnum.IMU_MOVEMENT_SHOCK.value
            else:
                return None

    def get_angle_map(self) -> Dict[Any, Any]:
        """Retrieve a copy of the servo angle map.

        If the internal angle map is empty or incomplete, a status update is
        triggered before returning the map.

        Returns:
            Dict[Any, Any]: A copy of the current servo angle mapping.
        """
        if not self.body_map or len(self.body_map) != len(self.servo_list):
            self.logger.debug("Servo map incomplete, updating servo statuses.")
            self.update_servo_status()
        return self.body_map.copy()

    def check_movement(self) -> bool:
        """Check if any servos are currently moving.

        Returns:
            bool: True if any servo is moving, False otherwise.
        """
        ret = False
        # prefer an imu check over a much longer servo status check
        if self.get_imu_moving_status() is True:
            ret = True
        else:
            if not self.body_map or len(self.body_map) != len(self.servo_list):
                self.logger.debug("Servo map incomplete, updating servo statuses.")
                self.update_servo_status()
            for servo in self.servo_list:
                if servo not in self.body_map:
                    continue
                if self.body_map[servo].moving is True:
                    ret = True
        return ret


class VisionTracker(MQTTClient):
    """Keep track of vision results based on MQTT status updates.

    This class monitors vision-related MQTT messages, caches results,
    and triggers tracking commands based on confidence thresholds.
    """

    def __init__(
        self,
        broker: NamedTuple,
        target: str,
        confidence: float,
        tracker_callback: Callable[[Dict[str, Any]], None],
        side_confidence: float = 0.4,
    ) -> None:
        """Initialize a VisionTracker instance.

        This method sets up the logger, target, confidence thresholds, and
        initializes the MQTT client. It also prepares caches and tracking
        variables for vision processing.

        Args:
            broker: A named tuple containing broker details with
                attributes 'ip' and 'port'.
            target: The target object to track.
            confidence: The minimum confidence threshold for head camera tracking.
            tracker_callback: A callback function to handle tracking updates.
            side_confidence: Confidence threshold for side cameras (lower due to fisheye).
        """
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(
            name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value
        )
        self.target: str = target
        self.tracker_callback: Callable[[Dict[str, Any]], None] = tracker_callback
        self.confidence_score: float = confidence
        self.side_confidence_score: float = side_confidence
        self.cmd_topic: str = CameraEnum.MQTT_RESPONSE_TOPIC.value
        self.main_camera: str = CameraEnum.CONFIG_HEAD.value
        self.left_camera: str = CameraEnum.CAMERA_LEFT.value
        self.right_camera: str = CameraEnum.CAMERA_RIGHT.value
        self.cam_key: str = CameraEnum.MSG_LOCATION_KEY.value
        self.results_key: str = CameraEnum.MSG_RESULTS.value
        self.ts_key: str = VisionResultsEnum.VISION_RESULTS_TS_KEY.value
        self.count: str = VisionResultsEnum.VISION_RESULTS_COUNT_KEY.value
        self.objects_key: str = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
        self.confidence_key: str = VisionResultsEnum.VISION_RESULTS_CONFIDENCE_KEY.value
        # Initialize topic handlers for vision commands
        self.topic_handler: Dict[str, Callable[[MQTTMessage], None]] = {
            self.cmd_topic: self.handle_cmd,
        }
        # Call the superclass constructor for MQTT client initialization
        super().__init__(ip=broker.ip, port=broker.port)
        # Initialize a TTL cache for vision responses (expires after 10 seconds)
        self.response_cache: TTLCache = TTLCache(maxsize=200, ttl=10)
        self.response_map: Dict[Any, Any] = {}
        # Tracking variables
        self.head_target: bool = False
        self.left_target: bool = False
        self.right_target: bool = False
        self.last_message: Any = None

    def handle_cmd(self, msg: MQTTMessage) -> None:
        """Handle incoming vision command messages.

        This method processes an MQTT message for vision commands. It attempts
        to decode the JSON payload and, if successful, passes the data to the
        parse_camera method.

        Args:
            msg (MQTTMessage): The incoming MQTT message.
        """
        try:
            j_msg: Dict[str, Any] = loads(msg.payload.decode())
        except JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON message: {e}")
            return

        if self.cam_key in j_msg:
            self.logger.debug(f"Camera message received, {msg.topic}, {j_msg}")
            self.parse_camera(msg=j_msg)

    def parse_camera(self, msg: Dict[str, Any]) -> None:
        """Parse a camera message and update vision tracking data.

        This method extracts camera information from the message, checks for
        the target with sufficient confidence, updates the response cache and
        map, and sends a tracking command if necessary.

        Args:
            msg (Dict[str, Any]): The dictionary containing the camera message.
        """
        camera: str = msg.get(self.cam_key, "")
        sight_results: Dict[str, Any] = msg.get(self.results_key, {})
        if self.target in sight_results:
            with self._lock:
                for p in sight_results[self.target][self.objects_key]:
                    c = p.get(self.confidence_key, 0.0)
                    if camera == CameraEnum.CAMERA_HEAD.value:
                        cf_score = self.confidence_score
                    elif camera in (CameraEnum.CAMERA_RIGHT.value, CameraEnum.CAMERA_LEFT.value):
                        cf_score = self.side_confidence_score
                    else:
                        cf_score = 0.0
                    if float(c) >= cf_score:
                        self.logger.debug(f"Confidence of {c} found for {self.target}")
                        # Update response_map with current sight results
                        self.response_map[camera] = sight_results
                        current_time: float = time()
                        last_ts: float = self.response_map[camera].get(self.ts_key, 0)
                        if current_time - last_ts <= 0.5:
                            self.response_map[camera][self.count] = (
                                self.response_map[camera].get(self.count, 0) + 1
                            )
                        else:
                            self.response_map[camera][self.count] = max(
                                0, self.response_map[camera].get(self.count, 1) - 1
                            )
                        self.response_map[camera][self.ts_key] = current_time
                        # Update response_cache for the camera
                        if camera in self.response_cache:
                            self.response_cache[camera][current_time] = sight_results
                        else:
                            self.response_cache[camera] = {current_time: sight_results}
                        self.logger.debug(
                            f"Sending Start command to track object {self.target} with a score of {c}"
                        )
                        if self.last_message is None:
                            self.last_message = time()

                        if time() - self.last_message >= 0.5:
                            if camera == TrackingEnums.BODY_HEAD_CAMERA.value:
                                self.send_command(
                                    TargetMessageBuilder.send_track_command_start(camera),
                                    TrackingEnums.MQTT_COMMAND_TOPIC.value,
                                )
                            elif camera in (TrackingEnums.BODY_LEFT_CAMERA.value, TrackingEnums.BODY_RIGHT_CAMERA.value):
                                # Side cameras always send track commands so room state stays fresh.
                                # Servo driving is gated by side_can_drive_servos() in MotionTrack.
                                self.send_command(
                                    TargetMessageBuilder.send_track_command_start(camera),
                                    TrackingEnums.MQTT_COMMAND_TOPIC.value,
                                )
                        else:
                            self.logger.debug("Skipping update as last message was recently sent")

    def get_vision_map(self) -> Dict[Any, Any]:
        """Retrieve the latest vision response messages.

        Returns:
            Dict[Any, Any]: A copy of the current vision response mapping.
        """
        with self._lock:
            return self.response_map.copy()

    def get_vision_cache(self) -> Dict[Any, Any]:
        """Retrieve the vision results cache.

        Returns:
            Dict[Any, Any]: A copy of the current vision results cache.
        """
        with self._lock:
            return dict(self.response_cache)


class SensorTracker(MQTTClient):
    """Keep track of all the various sensors based on MQTT status updates.

    This class monitors sensor status messages received via MQTT and updates
    internal state for different sensors.
    """
    sensor_broker = MQTTClient.broker_tuple

    def __init__(self, broker: NamedTuple) -> None:
        """Initialize the SensorTracker instance.

        Args:
            broker (NamedTuple): A named tuple containing broker details with
                attributes 'ip' and 'port'.
        """
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(
            name=self.__name__,
            console_logging=LoggingEnums.LOG_LEVEL_INFO.value
        )
        self.topic_handler: Dict[str, Callable[[MQTTMessage], None]] = {
            MQTTEnums.IMU_STATUS_TOPIC.value: self.imu_handle_cmd,
            MQTTEnums.TOF_STATUS_TOPIC.value: self.tof_handle_cmd,
            MQTTEnums.TH_STATUS_TOPIC.value: self.th_handle_cmd,
            MQTTEnums.MOX_STATUS_TOPIC.value: self.mox_handle_cmd
        }
        super().__init__(ip=broker.ip, port=broker.port)
        self.imu_status: dict = {}
        self.tof_status: dict = {}
        self.mox_status: dict = {}
        self.th_status: dict = {}

    def _handle_status(
        self,
        msg: MQTTMessage,
        sensor_key: str,
        sensor_name: str
    ) -> None:
        """Generic handler for processing sensor status messages.

        Args:
            msg (MQTTMessage): The incoming MQTT message.
            sensor_key (str): The expected key in the decoded JSON message.
            sensor_name (str): The name of the sensor (for logging purposes).
        """
        data: Dict[str, Any] = self.__load_message(msg)
        if not data:
            self.logger.error(f"Failed to decode {sensor_name} status message")
        elif sensor_key in data:
            self.logger.debug(f"Received {sensor_name} status message")
            setattr(self, sensor_key, data[sensor_key])

    def tof_handle_cmd(self, msg: MQTTMessage) -> None:
        """Handle incoming TOF status messages.

        Args:
            msg (MQTTMessage): The MQTT message containing TOF status.
        """
        self.logger.debug(f"TOF MQTT Handler Triggered")
        self._handle_status(
            msg,
            sensor_key=TOFEnums.TOF_STATUS_KEY.value,
            sensor_name=TOFEnums.SENSOR_NAME.value
        )

    def imu_handle_cmd(self, msg: MQTTMessage) -> None:
        """Handle incoming IMU status messages.

        Args:
            msg (MQTTMessage): The MQTT message containing IMU status.
        """
        self.logger.debug(f"IMU MQTT Handler Triggered")
        self._handle_status(
            msg,
            sensor_key=IMUEnums.IMU_STATUS_KEY.value,
            sensor_name=IMUEnums.SENSOR_NAME.value
        )

    def mox_handle_cmd(self, msg: MQTTMessage) -> None:
        """Handle incoming MOX GAS status messages.

        Args:
            msg (MQTTMessage): The MQTT message containing IMU status.
        """
        self.logger.debug(f"MOX MQTT Handler Triggered")
        self._handle_status(
            msg,
            sensor_key=MOXEnums.MOX_STATUS_KEY.value,
            sensor_name=MOXEnums.SENSOR_NAME.value
        )

    def th_handle_cmd(self, msg: MQTTMessage) -> None:
        """Handle incoming Temp * Humidity status messages.

        Args:
            msg (MQTTMessage): The MQTT message containing Temp & Humidity status.
        """
        self.logger.debug(f"Temp & Humidity MQTT Handler Triggered")
        self._handle_status(
            msg,
            sensor_key=THEnums.TH_STATUS_KEY.value,
            sensor_name=THEnums.SENSOR_NAME.value
        )

    def __load_message(self, json_message: MQTTMessage) -> Dict[str, Any]:
        """Decode a JSON message from MQTT.

        Args:
            json_message (MQTTMessage): The incoming MQTT message with a JSON payload.

        Returns:
            Dict[str, Any]: The decoded JSON as a dictionary. Returns an empty
            dictionary if decoding fails.
        """
        try:
            data = loads(json_message.payload.decode())
            self.logger.debug(f"JSON message decoded as: {data}")
            return data
        except JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON message: {e}")
            return {}

    def get_sensor_status(self, sensor_attr: str) -> dict:
        """Retrieve the status of a sensor using its attribute name.

        Args:
            sensor_attr (str): The attribute name for the sensor (e.g., 'imu_status' or 'tof_status').

        Returns:
            Optional[Any]: The current status of the sensor, or None if it is not set.
        """
        return getattr(self, sensor_attr, None)


if __name__ == "__main__":
    b = namedtuple("broker", ["ip", "port"])
    broker = b('192.168.1.39', 1883)
    # Assuming broker is a NamedTuple with 'ip' and 'port' attributes
    servo_location_tracker = ServoLocation(broker)
    # Retrieve the current servo angles
    angle_map = servo_location_tracker.get_angle_map()

    # Access servo data
    for servo_name, servo_data in angle_map.items():
        print(f"Servo {servo_name}:")
        print(f"  Current Angle: {servo_data.current}")
        print(f"  Min Angle: {servo_data.min}")
        print(f"  Max Angle: {servo_data.max}")
        print(f"  Middle Angle: {servo_data.middle}")
        print(f"  Axis: {servo_data.axis}")
        print(servo_location_tracker.get_imu_status())
    st = SensorTracker(broker=broker)
    sleep(2)
    print(st.get_sensor_status(TOFEnums.TOF_STATUS_KEY.value))
    print(st.get_sensor_status(IMUEnums.IMU_STATUS_KEY.value))
    print(st.get_sensor_status(THEnums.TH_STATUS_KEY.value))
    print(st.get_sensor_status(MOXEnums.MOX_STATUS_KEY.value))
    while True:
        print(st.get_sensor_status(IMUEnums.IMU_STATUS_KEY.value))
        sleep(.2)
