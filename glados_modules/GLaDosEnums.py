from enum import Enum


class SystemEnums(Enum):
    MQTT_INTENSITY_TOPIC: str = "intensity"


class VisionResultsEnum(Enum):
    """
    Enum for dict that tracks objects found by yolo
    """
    VISION_RESULTS_COUNT_KEY = "count"
    VISION_RESULTS_RESULTS_KEY = "results"
    VISION_RESULTS_OBJECTS_KEY = "objects"
    VISION_RESULTS_CLASS_KEY = "class_name"
    VISION_RESULTS_CONFIDENCE_KEY = "confidence"
    VISION_RESULTS_TS_KEY = "ts"
    VISION_RESULTS_BOX_KEY = "box"
    YOLO_CLASS_NAME_KEY = "name"


class ServoEnum(Enum):
    """
    Enum of servo location names for use in mqtt topics and other interactions with them
    """
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
    MSG_AXIS: str = "axis"
    MQTT_COMMAND_TOPIC: str = f"{LOCATION_CORE}/{MSG_LOCATION_KEY}"
    MQTT_STATUS_TOPIC: str = f"{LOCATION_CORE}/{MSG_LOCATION_KEY}/{MSG_COMMAND_STATUS}"


class CameraEnum(Enum):
    """
    Enum of camera location names for use in mqtt topics and other interactions with them
    """
    CONFIG_HEAD: str = "CAMERAS"
    LOCATION_CORE: str = "vision"
    MSG_LOCATION_KEY: str = "camera"
    MSG_COMMAND_STATUS: str = "status"
    MSG_COMMAND_KEY: str = "cmd"
    MSG_RESULTS: str = "results"
    MSG_RAW_IMAGE: str = "raw"
    MSG_RESOLUTION: str = "resolution"
    MSG_FPS: str = "fps"
    MSG_FACTORY: str = "factory"
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
    CAMERA_HEAD_FOCAL: int = 54
    CAMERA_RIGHT_FOCAL: int = 160
    CAMERA_LEFT_FOCAL: int = 160


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


