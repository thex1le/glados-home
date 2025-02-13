from enum import Enum
from logging import DEBUG, INFO
from typing import Dict


class LoggingEnums(Enum):
    """Enum class to store logging-related constants."""
    LOG_LEVEL_DEBUG: int = DEBUG
    LOG_LEVEL_INFO: int = INFO
    LOG_FOLDER_DEFAULT_NAME: str = "logs"
    LOG_FILE_TYPE: str = ".log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class SystemEnums(Enum):
    """Enum class to store GLaDOS system-related constants."""
    MQTT_INTENSITY_TOPIC: str = "intensity"
    CONFIG_HEAD_MQTT: str = "MQTT"
    CONFIG_HEAD_DEFAULT: str = "DEFAULT"
    CONFIG_HEAD_LOCALSPEAK: str = "LOCALSPEAK"
    MQTT_PORT: str = "mqtt_port"
    MQTT_SERVER_IP: str = "mqtt_server_ip"
    CONFIG_HEAD_RTSP: str = "RTSP"
    RTSP_PORT: str = "rtsp_port"
    RTSP_SERVER_IP: str = "rtsp_server_ip"
    APERTURE_ANIMATION: str = "aperture_animation"
    WORKING_ROOT: str = "working_root"
    VOLUME_LEVEL: str = "VolumeLevel"
    VOICE_URL: str = "VoiceUrl"
    RIGHT_LCD: str = "right_lcd"
    LEFT_LCD: str = "left_lcd"
    LIB_ASOUND: str = "libasound.so"


class MQTTEnums(Enum):
    """
    Enum for MQTT Topics
    """
    VISION_RESULTS_MQTT_TOPIC: str = "vision/camera_response"
    BODY_LED_CONTROL_MQTT_TOPIC: str = "body/led"
    SYSTEM_INTENSITY_TOPIC: str = "intensity"


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
    VISION_RESULTS_COUNT_KEY: str = "count"
    VISION_RESULTS_RESULTS_KEY: str = "results"
    VISION_RESULTS_OBJECTS_KEY: str = "objects"
    VISION_RESULTS_CLASS_KEY: str = "class_name"
    VISION_RESULTS_CONFIDENCE_KEY: str = "confidence"
    VISION_RESULTS_TS_KEY: str = "ts"
    VISION_RESULTS_BOX_KEY: str = "box"
    # Key used for class name in YOLO results
    YOLO_CLASS_NAME_KEY: str = "name"

    VISION_POSE_KEY_POINTS_COCO_WHOLE_BODY: Dict[int, str] = {
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
    CONFIG_HEAD: str = "SERVOS"
    DEFAULT_MAX_MIN_CENTER = "default_max_min_center"
    HEAD_MIN_MAX_CENTER = "head_min_max_center"
    NECK_MIN_MAX_CENTER = "neck_min_max_center"
    SERVO_MG90D_SPEED: str = "mg90d_speed"
    SERVO_MG92B_SPEED: str = "mg92b_speed"
    SERVO_M995R_SPEED: str = "mg995r_speed"
    SERVO_MG90D_PULSE: str = "mg90d_pulse"
    SERVO_MG92B_PULSE: str = "mg92b_pulse"
    SERVO_M995R_PULSE: str = "mg995r_pulse"
    SERVO_GS3508MG_PULSE: str = "gs3508mg_pulse"
    SERVO_GS3508MG_SPEED: str = "gs3508mg_speed"
    LOCATION_HEAD_UP_DOWN: str = "head_up_down"
    LOCATION_HEAD_LEFT_RIGHT: str = "head_left_right"
    LOCATION_BODY_UP_DOWN: str = "body_up_down"
    LOCATION_BODY_LEFT_RIGHT: str = "body_left_right"
    LOCATION_CORE: str = "body"
    SERVO_DEFAULT_SPEED: int = 1
    MSG_LOCATION_KEY: str = "servo"
    MSG_COMMAND_KEY: str = "cmd"
    MSG_COMMAND_MOVE: str = "move"
    MSG_COMMAND_STATUS: str = "status"
    MSG_RESULTS: str = "results"
    MSG_ANGLE: str = "angle"
    MSG_SPEED: str = "speed"
    MSG_MAX: str = "max"
    MSG_MIN: str = "min"
    MSG_MIDDLE: str = "middle"
    MSG_CURRENT_ANGLE: str = "current"
    MSG_LOCATION: str = "location"
    MSG_MOVING: str = "moving"
    MSG_AXIS: str = "axis"
    X_AXIS: str = "x"
    Y_AXIS: str = "y"
    MQTT_COMMAND_TOPIC: str = f"{LOCATION_CORE}/{MSG_LOCATION_KEY}"
    MQTT_STATUS_TOPIC: str = f"{LOCATION_CORE}/{MSG_LOCATION_KEY}/{MSG_COMMAND_STATUS}"


class CameraEnum(Enum):
    """
    Enum of camera location names for use in mqtt topics and other interactions with them
    """
    CONFIG_HEAD: str = "CAMERAS"
    LOCATION_CORE: str = "vision"
    MSG_LOCATION_KEY: str = "camera"
    MSG_CAMERA_NUMBER: str = "number"
    MSG_COMMAND_STATUS: str = "status"
    MSG_COMMAND_KEY: str = "cmd"
    MSG_RESULTS: str = "results"
    MSG_RAW_IMAGE: str = "raw"
    MSG_RESOLUTION: str = "resolution"
    MSG_FPS: str = "fps"
    MSG_FACTORY: str = "factory"
    MSG_RTSP_URI: str = "rtsp_uri"
    CAMERA_AI_SERVER_RX = "camera_ai_server_rx"
    CAMERA_AI_SERVER_RX_TIMEOUT = "camera_ai_server_rx_timeout"
    CAMERA_HEAD: str = f"{MSG_LOCATION_KEY}_head"
    CAMERA_LEFT: str = f"{MSG_LOCATION_KEY}_left"
    CAMERA_RIGHT: str = f"{MSG_LOCATION_KEY}_right"
    MQTT_RESPONSE_TOPIC: str = f"{LOCATION_CORE}/{MSG_LOCATION_KEY}/{MSG_RESULTS}"
    MQTT_STATUS_TOPIC: str = f"{LOCATION_CORE}/{MSG_LOCATION_KEY}/{MSG_COMMAND_STATUS}"
    CAMERA_HEAD_RESOLUTION: str = f"{CAMERA_HEAD}_{MSG_RESOLUTION}"
    CAMERA_LEFT_RESOLUTION: str = f"{CAMERA_LEFT}_{MSG_RESOLUTION}"
    CAMERA_RIGHT_RESOLUTION: str = f"{CAMERA_RIGHT}_{MSG_RESOLUTION}"
    CAMERA_HEAD_FPS: str = f"{CAMERA_HEAD}_{MSG_FPS}"
    CAMERA_LEFT_FPS: str = f"{CAMERA_LEFT}_{MSG_FPS}"
    CAMERA_RIGHT_FPS: str = f"{CAMERA_RIGHT}_{MSG_FPS}"
    CAMERA_HEAD_FACTORY: str = f"{CAMERA_HEAD}_{MSG_FACTORY}"
    CAMERA_LEFT_FACTORY: str = f"{CAMERA_LEFT}_{MSG_FACTORY}"
    CAMERA_RIGHT_FACTORY: str = f"{CAMERA_RIGHT}_{MSG_FACTORY}"
    CAMERA_HEAD_PORT: str = f"{CAMERA_HEAD}_rtsp_port"
    CAMERA_RIGHT_PORT: str = f"{CAMERA_RIGHT}_rtsp_port"
    CAMERA_LEFT_PORT: str = f"{CAMERA_LEFT}_rtsp_port"
    CAMERA_HEAD_RTSP_IP: str = f"{CAMERA_HEAD}_rtsp_ip"
    CAMERA_RIGHT_RTSP_IP: str = f"{CAMERA_RIGHT}_rtsp_ip"
    CAMERA_LEFT_RTSP_IP: str = f"{CAMERA_LEFT}_rtsp_ip"
    CAMERA_HEAD_FOV_X: int = 54
    CAMERA_HEAD_FOV_Y: int = 41
    CAMERA_RIGHT_FOV: int = 160
    CAMERA_LEFT_FOV: int = 160


class TrackingEnums(Enum):
    MSG_LOCATION_KEY: str = "system"
    MSG_COMMAND_KEY: str = "track"
    MSG_COMMAND_START: str = "start"
    MSG_CAMERA_KEY: str = "camera"
    MQTT_COMMAND_TOPIC: str = f"{MSG_LOCATION_KEY}/{MSG_COMMAND_KEY}"
    BODY_LEFT_CAMERA_ANGLE: int = 134
    BODY_RIGHT_CAMERA_ANGLE: int = 44
    BODY_HEAD_CAMERA: str = f"{MSG_CAMERA_KEY}_head"
    BODY_LEFT_CAMERA: str = f"{MSG_CAMERA_KEY}_left"
    BODY_RIGHT_CAMERA: str = f"{MSG_CAMERA_KEY}_right"
    KEY_CONFIDENCE: str = "confidence"
    KEY_POSE: str = "pose"
    KEY_BOX: str = "box"
