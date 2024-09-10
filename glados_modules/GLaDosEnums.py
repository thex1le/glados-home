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
    MSG_LOCATION_KEY: str = "servo"
    MSG_COMMAND_KEY: str = "cmd"
    MSG_COMMAND_STATUS: str = "status"
    MSG_RESULTS: str = "results"
    MSG_ANGLE: str = "angle"
    MSG_SPEED: str = "speed"
    MSG_MAX: str = "max"
    MSG_MIN: str = "min"
    MSG_MIDDLE: str= "middle"
    MSG_CURRENT_ANGLE: str = "current"
    MSG_AXIS: str = "axis"
    MQTT_COMMAND_TOPIC: str = "body/servo"
    MQTT_STATUS_TOPIC: str = "body/servo/status"
