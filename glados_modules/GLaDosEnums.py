from enum import Enum


class GLaDOSEnums(Enum):
    MQTT_INTENSITY_TOPIC = "intensity"


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
    LOCATION_CORE: str = "vision"
    MSG_LOCATION_KEY: str = "camera"
    MSG_COMMAND_STATUS: str = "status"
    MSG_COMMAND_KEY: str = "cmd"
    MSG_RESULTS: str = "results"
    MSG_RAW_IMAGE: str = "raw"
    MSG_RESOLUTION: str = "resolution"
    MSG_FPS: str = "fps"
    MSG_FACTORY: str = "factory"
    CAMERA_HEAD: str = "camera_head"
    CAMERA_LEFT: str = "camera_left"
    CAMERA_RIGHT: str = "camera_right"
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
