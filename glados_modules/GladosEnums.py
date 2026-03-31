from enum import Enum
from logging import DEBUG, INFO


class LoggingEnums(Enum):
    """Enum class to store logging-related constants."""
    LOG_LEVEL_DEBUG = DEBUG
    LOG_LEVEL_INFO = INFO
    LOG_FOLDER_DEFAULT_NAME = "logs"
    LOG_FILE_TYPE = ".log"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class SystemEnums(Enum):
    """Enum class to store GLaDOS system-related constants."""
    MQTT_INTENSITY_TOPIC = "intensity"
    CONFIG_HEAD_MQTT = "MQTT"
    CONFIG_HEAD_DEFAULT = "DEFAULT"
    CONFIG_HEAD_LOCALSPEAK = "LOCALSPEAK"
    MQTT_PORT = "mqtt_port"
    MQTT_SERVER_IP = "mqtt_server_ip"
    CONFIG_HEAD_RTSP = "RTSP"
    RTSP_PORT = "rtsp_port"
    RTSP_SERVER_IP = "rtsp_server_ip"
    APERTURE_ANIMATION = "aperture_animation"
    WORKING_ROOT = "working_root"
    VOLUME_LEVEL = "VolumeLevel"
    VOICE_URL = "VoiceUrl"
    RIGHT_LCD = "right_lcd"
    LEFT_LCD = "left_lcd"
    LIB_ASOUND = "libasound.so"
    BROKER_IP = 'ip'
    BROKER_PORT = 'port'
    BROKER = 'broker'
    # SPEAKER_DELAY: int = 1.6686527729034424
    SPEAKER_DELAY = 1.12


class LEDShoulders(Enum):
    LOCATION = "shoulder_led"
    MSG_TYPE_KEY = "led"
    MSG_COMMAND_KEY = "command"
    MSG_INTENSITY_KEY = "intensity"
    ANIMATION_STARTUP = "startup"
    ANIMATION_DISCO = "disco"
    ANIMATION_TWINKLE = "twinkle"


class LEDLampStrip8(Enum):
    LOCATION = "strip8_led"
    MSG_TYPE_KEY = "led"
    MSG_COMMAND_KEY = "command"
    MSG_ARGUMENTS_KEY = "arguments"
    MSG_INTENSITY_KEY = "intensity"
    ANIMATION_STARTUP = "startup"
    ANIMATION_DISCO = "disco"
    ANIMATION_TWINKLE = "twinkle"
    ANIMATION_PULSE = "pulse"
    ANIMATION_CLEAR = "clear"
    ANIMATION_WHITE_LIGHT = "white_light"


class LEDHead(Enum):
    EYE_LED_LOCATION = "eye_led"
    ANIMATION_STARTUP_KEY = "startup"
    ANIMATION_DISCO_KEY = "disco"
    ANIMATION_ANGRY_EYE_KEY = "angry_eye"
    ANIMATION_NORMAL_EYE_KEY = "normal_eye"
    ANIMATION_SPEECH_EYE_KEY = "speech_eye"
    MSG_COMMAND_TYPE_KEY = "led"
    MSG_COMMAND_KEY = "command"
    MSG_INTENSITY_KEY = "intensity"
    MSG_COMMAND_ARGUMENTS_KEY = "arguments"
    MSG_COMMAND_LOCATION_KEY = "location"
    MSG_COMMAND_START_DELAY = "start_delay"
    ARGS_KEY_TIME_DICT = "time_dict"
    ARGS_KEY_DELAY = "delay"


class LCDEnums(Enum):
    """Enums for LCD display commands."""
    MSG_LOCATION_KEY = "lcd"
    MSG_COMMAND_KEY = "cmd"
    COMMAND_SET_BREATH = "set_breath"
    COMMAND_GET_BREATH = "get_breath"
    COMMAND_STARTUP = "startup"
    OPTIONS_KEY = "options"
    RESPONSE_KEY = "response"


class STTEnums(Enum):
    """
    Enums for the Speech to text engine
    """
    CONFIG_HEAD_STT = "STT"
    STT_SERVER_IP = "stt_server_ip"
    STT_SERVER_PORT = "stt_server_port"
    STT_RESULTS_KEY = "STT_RESULTS"
    STT_TEXT_KEY = "text"
    STT_LANGUAGE_KEY = "language"
    STT_SEGMENTS_KEY = "segments"
    STT_WORD_SEGMENTS_KEY = "word_segments"
    STT_TIME_MAP_KEY = "time_map"
    STT_RAW_RESULTS_KEY = "raw"
    STT_EN_LANG_KEY = "en"


class IMUEnums(Enum):
    """
    Enums for the IMU Sensor Data
    """
    TEMP_KEY = "temperature"
    ACCEL_KEY = "accelerometer"
    MAGNETO_KEY = "magnetometer"
    GYRO_KEY = "gyroscope"
    EULER_KEY = "euler"
    QUAT_KEY = "quaternion"
    LINEAR_KEY = "linear"
    GRAVITY_KEY = "gravity"
    IMU_STATUS_KEY = "imu_status"
    IMU_TIME_STAMP_KEY = "time"
    SENSOR_NAME = "IMU"


class TOFEnums(Enum):
    """
    Enums for the TOF Sensor Data
    """
    TOF_TIME_STAMP_KEY = "time"
    TOF_STATUS_KEY = "tof_status"
    SENSOR_NAME = "TOF"


class THEnums(Enum):
    """
    Enums for the Temp and Humidity sensor
    """
    TH_TIME_STAMP_KEY = "time"
    TH_STATUS_KEY = "th_status"
    TH_HUMIDITY_KEY = "humidity"
    TH_CELSIUS_KEY = "celsius"
    TH_FAHRENHEIT_KEY = "fahrenheit"
    SENSOR_NAME = "TH"


class MOXEnums(Enum):
    """
    Enums for the MOX Gas Sensor Data
    """
    MOX_TIME_STAMP_KEY = "time"
    MOX_STATUS_KEY = "mox_status"
    SENSOR_NAME = "MOX"
    MOX_AQI = "AQI (1-5):"
    MOX_TVOC = "TVOC (ppb):"
    MOX_ECO2 = "eCO2 (ppm):"


class MQTTEnums(Enum):
    """
    Enum for MQTT Topics
    """
    STT_RESULTS_MQTT_TOPIC = "sst/results"
    VISION_RESULTS_MQTT_TOPIC = "vision/camera_response"
    BODY_LED_CONTROL_MQTT_TOPIC = "body/led"
    LCD_CONTROL_MQTT_TOPIC = "body/lcd"
    LCD_STATUS_MQTT_TOPIC = "status"
    IMU_STATUS_TOPIC = "body/imu/status"
    TOF_STATUS_TOPIC = "body/tof/status"
    MOX_STATUS_TOPIC = "body/mox/status"
    SYSTEM_INTENSITY_TOPIC = "intensity"
    TH_STATUS_TOPIC = "body/th/status"
    SYSTEM_HEALTH_TOPIC = "system/health"
    SYSTEM_LOG_LEVEL_TOPIC = "system/log_level"


class DashboardEnums(Enum):
    CONFIG_HEAD = "DASHBOARD"
    DASHBOARD_PORT = "dashboard_port"
    DEFAULT_PORT = 8080


class TraceEnums(Enum):
    """Keys used in pipeline trace messages."""
    TRACE_ID = "trace_id"
    TS_VISION = "ts_vision"
    TS_TRACK_START = "ts_track_start"
    TS_TRACK_END = "ts_track_end"
    TS_SERVO_RX = "ts_servo_rx"


class VisionResultsEnum(Enum):
    """
    Enum for dictionary keys that track objects detected by YOLO.

    This enum defines string keys used in vision result dictionaries to
    standardize the format of object detection outputs, including count,
    class names, confidence levels, timestamps, and bounding boxes.

    Additionally, it includes COCO WholeBody keypoint mappings for
    133 key points (body, face, hands, and feet).
    """

    # Keys for object detection results
    VISION_RESULTS_COUNT_KEY = "count"
    VISION_RESULTS_RESULTS_KEY = "results"
    VISION_RESULTS_OBJECTS_KEY = "objects"
    VISION_RESULTS_CLASS_KEY = "class_name"
    VISION_RESULTS_CONFIDENCE_KEY = "confidence"
    VISION_RESULTS_TS_KEY = "ts"
    VISION_RESULTS_BOX_KEY = "box"
    VISION_RESULTS_PERSON_KEY = "person"
    VISION_RESULTS_POSE_KEY = "pose"
    # Bounding box coordinate keys
    BOX_X1 = "x1"
    BOX_Y1 = "y1"
    BOX_X2 = "x2"
    BOX_Y2 = "y2"
    # Keypoint coordinate keys
    KEYPOINT_X = "x"
    KEYPOINT_Y = "y"
    KEYPOINT_CONFIDENCE = "confidence"
    KEYPOINT_LOCATION = "location"
    # Keypoint confidence threshold for drawing
    KEYPOINT_DRAW_THRESHOLD = 0.5
    # Key used for class name in YOLO results
    YOLO_CLASS_NAME_KEY = "name"

    VISION_POSE_KEY_POINTS_COCO_WHOLE_BODY = {
                # COCO WholeBody keypoint mapping (133 Key points)
                # Body Key points (0-16)
                0: "Nose",
                1: "Left Eye",
                2: "Right Eye",
                3: "Left Ear",
                4: "Right Ear",
                5: "Left Shoulder",
                6: "Right Shoulder",
                7: "Left Elbow",
                8: "Right Elbow",
                9: "Left Wrist",
                10: "Right Wrist",
                11: "Left Hip",
                12: "Right Hip",
                13: "Left Knee",
                14: "Right Knee",
                15: "Left Ankle",
                16: "Right Ankle",
                # Face Key points (17-84)
                # Jawline (17 points)
                17: "Face_Jawline_0",
                18: "Face_Jawline_1",
                19: "Face_Jawline_2",
                20: "Face_Jawline_3",
                21: "Face_Jawline_4",
                22: "Face_Jawline_5",
                23: "Face_Jawline_6",
                24: "Face_Jawline_7",
                25: "Face_Jawline_8",
                26: "Face_Jawline_9",
                27: "Face_Jawline_10",
                28: "Face_Jawline_11",
                29: "Face_Jawline_12",
                30: "Face_Jawline_13",
                31: "Face_Jawline_14",
                32: "Face_Jawline_15",
                33: "Face_Jawline_16",
                # Left Eyebrow (5 points)
                34: "Face_Left_Eyebrow_0",
                35: "Face_Left_Eyebrow_1",
                36: "Face_Left_Eyebrow_2",
                37: "Face_Left_Eyebrow_3",
                38: "Face_Left_Eyebrow_4",
                # Right Eyebrow (5 points)
                39: "Face_Right_Eyebrow_0",
                40: "Face_Right_Eyebrow_1",
                41: "Face_Right_Eyebrow_2",
                42: "Face_Right_Eyebrow_3",
                43: "Face_Right_Eyebrow_4",
                # Nose (9 points)
                44: "Face_Nose_0",
                45: "Face_Nose_1",
                46: "Face_Nose_2",
                47: "Face_Nose_3",
                48: "Face_Nose_4",
                49: "Face_Nose_5",
                50: "Face_Nose_6",
                51: "Face_Nose_7",
                52: "Face_Nose_8",
                # Left Eye (6 points)
                53: "Face_Left_Eye_0",
                54: "Face_Left_Eye_1",
                55: "Face_Left_Eye_2",
                56: "Face_Left_Eye_3",
                57: "Face_Left_Eye_4",
                58: "Face_Left_Eye_5",
                # Right Eye (6 points)
                59: "Face_Right_Eye_0",
                60: "Face_Right_Eye_1",
                61: "Face_Right_Eye_2",
                62: "Face_Right_Eye_3",
                63: "Face_Right_Eye_4",
                64: "Face_Right_Eye_5",
                # Mouth Outer (12 points)
                65: "Face_Mouth_Outer_0",
                66: "Face_Mouth_Outer_1",
                67: "Face_Mouth_Outer_2",
                68: "Face_Mouth_Outer_3",
                69: "Face_Mouth_Outer_4",
                70: "Face_Mouth_Outer_5",
                71: "Face_Mouth_Outer_6",
                72: "Face_Mouth_Outer_7",
                73: "Face_Mouth_Outer_8",
                74: "Face_Mouth_Outer_9",
                75: "Face_Mouth_Outer_10",
                76: "Face_Mouth_Outer_11",
                # Mouth Inner (8 points)
                77: "Face_Mouth_Inner_0",
                78: "Face_Mouth_Inner_1",
                79: "Face_Mouth_Inner_2",
                80: "Face_Mouth_Inner_3",
                81: "Face_Mouth_Inner_4",
                82: "Face_Mouth_Inner_5",
                83: "Face_Mouth_Inner_6",
                84: "Face_Mouth_Inner_7",
                # Left Hand Key points (85-105)
                85: "Left_Hand_Wrist",
                86: "Left_Hand_Thumb_1",
                87: "Left_Hand_Thumb_2",
                88: "Left_Hand_Thumb_3",
                89: "Left_Hand_Thumb_4",
                90: "Left_Hand_Index_1",
                91: "Left_Hand_Index_2",
                92: "Left_Hand_Index_3",
                93: "Left_Hand_Index_4",
                94: "Left_Hand_Middle_1",
                95: "Left_Hand_Middle_2",
                96: "Left_Hand_Middle_3",
                97: "Left_Hand_Middle_4",
                98: "Left_Hand_Ring_1",
                99: "Left_Hand_Ring_2",
                100: "Left_Hand_Ring_3",
                101: "Left_Hand_Ring_4",
                102: "Left_Hand_Pinky_1",
                103: "Left_Hand_Pinky_2",
                104: "Left_Hand_Pinky_3",
                105: "Left_Hand_Pinky_4",
                # Right Hand Key points (106-126)
                106: "Right_Hand_Wrist",
                107: "Right_Hand_Thumb_1",
                108: "Right_Hand_Thumb_2",
                109: "Right_Hand_Thumb_3",
                110: "Right_Hand_Thumb_4",
                111: "Right_Hand_Index_1",
                112: "Right_Hand_Index_2",
                113: "Right_Hand_Index_3",
                114: "Right_Hand_Index_4",
                115: "Right_Hand_Middle_1",
                116: "Right_Hand_Middle_2",
                117: "Right_Hand_Middle_3",
                118: "Right_Hand_Middle_4",
                119: "Right_Hand_Ring_1",
                120: "Right_Hand_Ring_2",
                121: "Right_Hand_Ring_3",
                122: "Right_Hand_Ring_4",
                123: "Right_Hand_Pinky_1",
                124: "Right_Hand_Pinky_2",
                125: "Right_Hand_Pinky_3",
                126: "Right_Hand_Pinky_4",
                # Foot Key points (127-132)
                127: "Left_Foot_BigToe",
                128: "Left_Foot_SmallToe",
                129: "Left_Foot_Heel",
                130: "Right_Foot_BigToe",
                131: "Right_Foot_SmallToe",
                132: "Right_Foot_Heel"}


class ServoEnum(Enum):
    """
    Enum of servo location names for use in mqtt topics and other interactions with them
    """
    CONFIG_HEAD = "SERVOS"
    DEFAULT_MAX_MIN_CENTER = "default_max_min_center"
    HEAD_MIN_MAX_CENTER = "head_min_max_center"
    NECK_MIN_MAX_CENTER = "neck_min_max_center"
    SERVO_MG90D_SPEED = "mg90d_speed"
    SERVO_MG92B_SPEED = "mg92b_speed"
    SERVO_M995R_SPEED = "mg995r_speed"
    SERVO_MG90D_PULSE = "mg90d_pulse"
    SERVO_MG92B_PULSE = "mg92b_pulse"
    SERVO_M995R_PULSE = "mg995r_pulse"
    SERVO_GS3508MG_PULSE = "gs3508mg_pulse"
    SERVO_GS3508MG_SPEED = "gs3508mg_speed"
    # PCA9685 board addresses (comma-separated, e.g. "0x40,0x41" for two boards)
    PCA9685_ADDRESSES = "pca9685_addresses"
    PCA9685_DEFAULT_ADDRESSES = "0x40"
    # Per-servo board assignment: "board_index,channel" (e.g. "0,0" = board 0 channel 0)
    BODY_LR_BOARD_CHANNEL = "body_left_right_board_channel"
    BODY_UD_BOARD_CHANNEL = "body_up_down_board_channel"
    HEAD_LR_BOARD_CHANNEL = "head_left_right_board_channel"
    HEAD_UD_BOARD_CHANNEL = "head_up_down_board_channel"
    LED_HEAD_BOARD_CHANNEL = "led_head_board_channel"
    LOCATION_HEAD_UP_DOWN = "head_up_down"
    LOCATION_HEAD_LEFT_RIGHT = "head_left_right"
    LOCATION_BODY_UP_DOWN = "body_up_down"
    LOCATION_BODY_LEFT_RIGHT = "body_left_right"
    LOCATION_CORE = "body"
    SERVO_DEFAULT_SPEED = 1
    # rads/s
    IMU_GYRO_THRESH = 0.03
    IMU_GYRO_SPIKE_THRESH = 0.5
    # m/s squared
    IMU_ACCEL_THRESH = 0.2
    IMU_ACCEL_SPIKE_THRESH = 5.0
    IMU_JERK_THRESHOLD = 20
    # frame count to wait
    IMU_HOLD_FRAME_COUNT = 1
    IMU_MOVEMENT_SHOCK = "shock"
    IMU_MOVEMENT_SERVO = "normal_movement"
    MSG_LOCATION_KEY = "servo"
    MSG_COMMAND_KEY = "cmd"
    MSG_COMMAND_MOVE = "move"
    MSG_COMMAND_MOVE_ALL = "move_all"
    MSG_COMMAND_STATUS = "status"
    MSG_RESULTS = "results"
    MSG_TARGETS = "targets"
    MSG_ANGLE = "angle"
    MSG_SPEED = "speed"
    MSG_MAX = "max"
    MSG_MIN = "min"
    MSG_MIDDLE = "middle"
    MSG_CURRENT_ANGLE = "current"
    MSG_LAST_ANGLE = "last"
    MSG_VELOCITY = "velocity"
    MSG_LOCATION = "location"
    MSG_MOVING = "moving"
    MSG_AXIS = "axis"
    X_AXIS = "x"
    Y_AXIS = "y"
    MQTT_COMMAND_TOPIC = f"{LOCATION_CORE}/{MSG_LOCATION_KEY}"
    MQTT_STATUS_TOPIC = f"{LOCATION_CORE}/{MSG_LOCATION_KEY}/{MSG_COMMAND_STATUS}"
    # Head servo locations for spring-damper type selection
    HEAD_SERVO_LOCATIONS = (LOCATION_HEAD_UP_DOWN, LOCATION_HEAD_LEFT_RIGHT)
    BODY_SERVO_LOCATIONS = (LOCATION_BODY_UP_DOWN, LOCATION_BODY_LEFT_RIGHT)


class MotionProfile(Enum):
    """Spring-damper motion parameters for organic servo movement.

    Each speed level (1-5) maps to (omega, zeta) where:
    - omega: natural frequency (rad/s) - controls response speed
    - zeta: damping ratio - <1.0 gives organic overshoot, 1.0 is critically damped
    """
    # Head servos: fast response, slight overshoot for organic feel
    HEAD_PARAMS = {
        1: (4.0, 0.95),    # Calm: slow, nearly critically damped
        2: (6.0, 0.90),    # Neutral: moderate, minimal overshoot
        3: (8.0, 0.85),    # Attentive: quick, slight overshoot
        4: (11.0, 0.78),   # Angry: snappy, visible overshoot
        5: (14.0, 0.70),   # Frustrated: very fast, pronounced overshoot
    }
    # Body servos: slow follow, minimal overshoot
    BODY_PARAMS = {
        1: (1.5, 1.0),     # Calm: very slow, critically damped
        2: (2.0, 1.0),     # Neutral: slow, critically damped
        3: (2.5, 1.0),     # Attentive: moderate, critically damped
        4: (3.5, 0.95),    # Angry: faster, nearly critically damped
        5: (4.5, 0.90),    # Frustrated: fast follow, tiny overshoot
    }
    # Physics loop timing
    PHYSICS_DT = 0.02              # 50Hz update rate (seconds)
    STATUS_SEND_INTERVAL = 10      # send MQTT status every N ticks (~5Hz)
    # Movement detection thresholds (for self.moving flag)
    MOVING_VELOCITY_THRESHOLD = 0.1     # deg/s
    MOVING_POSITION_THRESHOLD = 0.5     # degrees
    # Default speed level for tracking
    DEFAULT_TRACKING_SPEED = 3
    # Idle drift speed level
    IDLE_DRIFT_SPEED = 1
    # Piecewise interpolation: head_UD angle -> body_UD angle
    # Empirically calibrated from physical robot
    HEAD_UD_TO_BODY_UD_TABLE = [
        (64, 30), (66, 40), (68, 50), (70, 60), (72, 70), (79, 80), (83, 90),
        (92, 100), (97, 110), (104, 120), (114, 130), (121, 140), (126, 150)
    ]
    # World-space tracking parameters
    WORLD_SMOOTH_ALPHA = 0.7           # EMA factor for world-space angle smoothing
    IDLE_DRIFT_INTERVAL = 0.5          # seconds between idle drift target sends
    IDLE_TIMEOUT = 5.0                 # seconds without target before idle drift starts
    DEFAULT_SERVO_CENTER = 90.0        # fallback center angle
    # Camera mounting offsets (degrees from body center)
    CAMERA_LEFT_MOUNTING_OFFSET = -55.0
    CAMERA_RIGHT_MOUNTING_OFFSET = 55.0


class KinematicsEnums(Enum):
    """URDF-derived kinematic parameters for the GLaDOS robot arm.

    Joint axes are unit vectors from the URDF, defined in the parent link frame.
    Offsets are translation vectors in mm from parent joint to child joint.
    Chain order: base -> body_LR -> body_UD -> head_LR -> head_UD -> camera

    Sign multipliers convert servo-degree direction to URDF rotation direction.
    A value of 1.0 means increasing servo degrees = positive rotation about the
    URDF axis. A value of -1.0 means they are opposite. These must be calibrated
    on the physical robot.
    """
    # Joint rotation axes (unit vectors in parent link frame)
    # Source: UDRF_description/urdf/UDRF.xacro
    AXIS_BODY_LR = (0.0, 1.0, 0.0)
    AXIS_BODY_UD = (-0.590388, 0.028881, -0.806602)
    AXIS_HEAD_LR = (-0.670409, 0.538936, 0.509999)
    AXIS_HEAD_UD = (0.513853, -0.158613, 0.843088)

    # Joint-to-joint translation offsets in mm (parent -> child)
    # Source: UDRF_description/urdf/UDRF.xacro joint origin xyz (meters * 1000)
    OFFSET_BODY_LR_TO_BODY_UD = (8.552, -192.946, -9.436)
    OFFSET_BODY_UD_TO_HEAD_LR = (-84.274, 5.831, 56.934)
    OFFSET_HEAD_LR_TO_HEAD_UD = (-64.709, -34.328, 27.644)

    # Camera optical axis in the head_UD link frame (unit vector pointing "forward")
    # Initial assumption -- must be calibrated on real hardware
    CAMERA_OPTICAL_AXIS_LOCAL = (0.0, 0.0, 1.0)

    # Per-joint sign multipliers (servo degrees -> URDF rotation direction)
    # Must be calibrated on real hardware: command +10 deg, observe direction
    SIGN_BODY_LR = 1.0
    SIGN_BODY_UD = 1.0
    SIGN_HEAD_LR = 1.0
    SIGN_HEAD_UD = 1.0

    # Joint ordering (matches ServoEnum.LOCATION_* values)
    JOINT_ORDER = ("body_left_right", "body_up_down",
                   "head_left_right", "head_up_down")

    # IK solver parameters
    IK_MAX_ITERATIONS = 10
    IK_TOLERANCE_DEG = 0.1
    IK_DAMPING_LAMBDA = 0.01
    IK_FINITE_DIFF_EPS = 1e-4


class CameraEnum(Enum):
    """
    Enum of camera location names for use in mqtt topics and other interactions with them
    """
    CONFIG_HEAD = "CAMERAS"
    LOCATION_CORE = "vision"
    MSG_LOCATION_KEY = "camera"
    MSG_CAMERA_NUMBER = "number"
    MSG_COMMAND_STATUS = "status"
    MSG_COMMAND_KEY = "cmd"
    MSG_RESULTS = "results"
    MSG_RAW_IMAGE = "raw"
    MSG_RESOLUTION = "resolution"
    MSG_FPS = "fps"
    MSG_FACTORY = "factory"
    MSG_RTSP_URI = "rtsp_uri"
    CAMERA_AI_SERVER_RX = "camera_ai_server_rx"
    CAMERA_AI_SERVER_RX_TIMEOUT = "camera_ai_server_rx_timeout"
    CAMERA_HEAD = f"{MSG_LOCATION_KEY}_head"
    CAMERA_LEFT = f"{MSG_LOCATION_KEY}_left"
    CAMERA_RIGHT = f"{MSG_LOCATION_KEY}_right"
    MQTT_RESPONSE_TOPIC = f"{LOCATION_CORE}/{MSG_LOCATION_KEY}/{MSG_RESULTS}"
    MQTT_STATUS_TOPIC = f"{LOCATION_CORE}/{MSG_LOCATION_KEY}/{MSG_COMMAND_STATUS}"
    CAMERA_HEAD_RESOLUTION = f"{CAMERA_HEAD}_{MSG_RESOLUTION}"
    CAMERA_LEFT_RESOLUTION = f"{CAMERA_LEFT}_{MSG_RESOLUTION}"
    CAMERA_RIGHT_RESOLUTION = f"{CAMERA_RIGHT}_{MSG_RESOLUTION}"
    CAMERA_HEAD_FPS = f"{CAMERA_HEAD}_{MSG_FPS}"
    CAMERA_LEFT_FPS = f"{CAMERA_LEFT}_{MSG_FPS}"
    CAMERA_RIGHT_FPS = f"{CAMERA_RIGHT}_{MSG_FPS}"
    CAMERA_HEAD_FACTORY = f"{CAMERA_HEAD}_{MSG_FACTORY}"
    CAMERA_LEFT_FACTORY = f"{CAMERA_LEFT}_{MSG_FACTORY}"
    CAMERA_RIGHT_FACTORY = f"{CAMERA_RIGHT}_{MSG_FACTORY}"
    CAMERA_HEAD_PORT = f"{CAMERA_HEAD}_rtsp_port"
    CAMERA_RIGHT_PORT = f"{CAMERA_RIGHT}_rtsp_port"
    CAMERA_LEFT_PORT = f"{CAMERA_LEFT}_rtsp_port"
    CAMERA_HEAD_RTSP_IP = f"{CAMERA_HEAD}_rtsp_ip"
    CAMERA_RIGHT_RTSP_IP = f"{CAMERA_RIGHT}_rtsp_ip"
    CAMERA_LEFT_RTSP_IP = f"{CAMERA_LEFT}_rtsp_ip"
    CAMERA_HEAD_FOV_X = 54
    CAMERA_HEAD_FOV_Y = 41
    CAMERA_RIGHT_FOV = 160
    CAMERA_LEFT_FOV = 160
    X_RESOLUTION = 'x'
    Y_RESOLUTION = 'y'


class TrackingEnums(Enum):
    MSG_LOCATION_KEY = "system"
    MSG_COMMAND_KEY = "track"
    MSG_COMMAND_START = "start"
    MSG_CAMERA_KEY = "camera"
    MQTT_COMMAND_TOPIC = f"{MSG_LOCATION_KEY}/{MSG_COMMAND_KEY}"
    BODY_LEFT_CAMERA_ANGLE = 134
    BODY_RIGHT_CAMERA_ANGLE = 44
    BODY_HEAD_CAMERA = f"{MSG_CAMERA_KEY}_head"
    BODY_LEFT_CAMERA = f"{MSG_CAMERA_KEY}_left"
    BODY_RIGHT_CAMERA = f"{MSG_CAMERA_KEY}_right"
    KEY_CONFIDENCE = "confidence"
    KEY_POSE = "pose"
    KEY_BOX = "box"
