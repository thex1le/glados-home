"""MCP server exposing GLaDOS body controls as LLM tools.

Spawned as a stdio subprocess by the dnhkng/GLaDOS engine (configured in
GladosBrain._mcp_server_configs). Each tool publishes to the existing body
MQTT topics that BodyServer.py (Pi 4) and BodyControlModules already subscribe
to. The MCP server itself never touches GPIO — all hardware contact happens on
the Pi 4 in response to MQTT messages.

Run directly:
    python -m glados_modules.mcp.BodyMCPServer -config glog.conf
"""

# native imports
from argparse import ArgumentParser
from configparser import ConfigParser
from os import path
import sys
from typing import Optional

# 3rd party imports
from mcp.server.fastmcp import FastMCP

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import (LoggingEnums, MQTTEnums, ServoEnum,
                                        SystemEnums, LCDEnums, LEDHead)
from glados_modules.MqttConnector import (MQTTClient, ServoMessageBuilder,
                                          LEDMessageBuilder)


class BodyMCPException(Exception):
    pass


_logger = setup_logger(name="BodyMCPServer",
                        console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
_mqtt: Optional[MQTTClient] = None
_config: Optional[ConfigParser] = None

mcp = FastMCP("body")

# Animation keys understood by LedHead.handle_cmd (see BodyControlModules.py).
_EYE_ANIMATIONS = {
    "normal": LEDHead.ANIMATION_NORMAL_EYE_KEY.value,
    "angry":  LEDHead.ANIMATION_ANGRY_EYE_KEY.value,
    "disco":  LEDHead.ANIMATION_DISCO_KEY.value,
    "startup": LEDHead.ANIMATION_STARTUP_KEY.value,
}


def _ensure_init(config_path: str) -> None:
    """Load glog.conf and connect to MQTT once per subprocess.

    Args:
        config_path: Path to glog.conf.

    Raises:
        BodyMCPException: If the config file is missing.
    """
    global _mqtt, _config
    if _mqtt is not None:
        return
    if not path.isfile(config_path):
        raise BodyMCPException(f"Unable to load config file {config_path}")
    _config = ConfigParser()
    _config.read(config_path)
    mqtt_c = _config[SystemEnums.CONFIG_HEAD_MQTT.value]
    _mqtt = MQTTClient(ip=mqtt_c[SystemEnums.MQTT_SERVER_IP.value],
                        port=int(mqtt_c[SystemEnums.MQTT_PORT.value]))
    _logger.info("BodyMCPServer initialized; MQTT broker connected.")


def _publish(message: dict, topic: str) -> None:
    """Wrap _mqtt.send_command with a not-initialized guard."""
    if _mqtt is None:
        raise BodyMCPException("BodyMCPServer not initialized; call _ensure_init first")
    _mqtt.send_command(message, topic)


# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------

@mcp.tool()
def look_at(yaw: int, pitch: int) -> str:
    """Aim GLaDOS's head at a target.

    Args:
        yaw: Head left/right angle in degrees. Range comes from
            [SERVOS] neck_min_max_center in glog.conf (default 52..120,
            center 92).
        pitch: Head up/down angle in degrees. Range comes from
            [SERVOS] head_min_max_center in glog.conf (default 6..125,
            center 83). Lower values look down, higher values look up.

    Out-of-range values are clamped by the Gservo class on the Pi 4.
    """
    targets = {
        ServoEnum.LOCATION_HEAD_LEFT_RIGHT.value: {
            ServoEnum.MSG_ANGLE.value: int(yaw),
            ServoEnum.MSG_SPEED.value: ServoEnum.SERVO_DEFAULT_SPEED.value},
        ServoEnum.LOCATION_HEAD_UP_DOWN.value: {
            ServoEnum.MSG_ANGLE.value: int(pitch),
            ServoEnum.MSG_SPEED.value: ServoEnum.SERVO_DEFAULT_SPEED.value},
    }
    msg = ServoMessageBuilder.move_all(targets)
    _publish(msg, ServoEnum.MQTT_COMMAND_TOPIC.value)
    return f"head moved to yaw={yaw} pitch={pitch}"


@mcp.tool()
def turn_body(yaw: int) -> str:
    """Rotate GLaDOS's body around its vertical axis.

    Args:
        yaw: Body left/right angle in degrees (default range 0..180,
            center 90).
    """
    msg = ServoMessageBuilder.body_left_right(angle=int(yaw))
    _publish(msg, ServoEnum.MQTT_COMMAND_TOPIC.value)
    return f"body rotated to yaw={yaw}"


@mcp.tool()
def lean(pitch: int) -> str:
    """Tilt GLaDOS's body forward or backward.

    Args:
        pitch: Body up/down angle in degrees (default range 0..180,
            center 90).
    """
    msg = ServoMessageBuilder.body_up_down(angle=int(pitch))
    _publish(msg, ServoEnum.MQTT_COMMAND_TOPIC.value)
    return f"body leaned to pitch={pitch}"


@mcp.tool()
def set_eye_animation(animation: str) -> str:
    """Trigger a head NeoPixel animation.

    Args:
        animation: One of 'normal', 'angry', 'disco', 'startup'.
    """
    key = animation.strip().lower()
    if key not in _EYE_ANIMATIONS:
        valid = ", ".join(sorted(_EYE_ANIMATIONS.keys()))
        return f"error: animation must be one of {valid} (got {animation!r})"
    payload = {LEDHead.MSG_COMMAND_LOCATION_KEY.value: LEDHead.EYE_LED_LOCATION.value,
               LEDHead.MSG_COMMAND_KEY.value: _EYE_ANIMATIONS[key]}
    msg = LEDMessageBuilder.send_led_animation(payload)
    _publish(msg, MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value)
    return f"eye animation '{key}' triggered"


@mcp.tool()
def wake_eyes(eye: str = "both") -> str:
    """Trigger the LCD eye 'startup' animation.

    Args:
        eye: 'left', 'right', or 'both'. Defaults to both eyes.
    """
    locations = _resolve_eye_locations(eye)
    if locations is None:
        return f"error: eye must be left|right|both (got {eye!r})"
    for loc in locations:
        msg = {LCDEnums.MSG_LOCATION_KEY.value: loc,
               LCDEnums.MSG_COMMAND_KEY.value: LCDEnums.COMMAND_STARTUP.value}
        _publish(msg, MQTTEnums.LCD_CONTROL_MQTT_TOPIC.value)
    return f"wake_eyes triggered on {eye}"


@mcp.tool()
def set_eye_breath(eye: str = "both", fast: bool = False) -> str:
    """Set the LCD eye breathing speed.

    Args:
        eye: 'left', 'right', or 'both'.
        fast: True for the faster breath animation, False for normal speed.
    """
    locations = _resolve_eye_locations(eye)
    if locations is None:
        return f"error: eye must be left|right|both (got {eye!r})"
    for loc in locations:
        msg = {LCDEnums.MSG_LOCATION_KEY.value: loc,
               LCDEnums.MSG_COMMAND_KEY.value: LCDEnums.COMMAND_SET_BREATH.value,
               LCDEnums.OPTIONS_KEY.value: {"fast": bool(fast)}}
        _publish(msg, MQTTEnums.LCD_CONTROL_MQTT_TOPIC.value)
    return f"eye breath set on {eye} (fast={fast})"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _resolve_eye_locations(eye: str):
    """Map an eye argument to one or both LCD location strings."""
    eye = eye.strip().lower()
    if eye == "both":
        return [SystemEnums.LEFT_LCD.value, SystemEnums.RIGHT_LCD.value]
    if eye == "left":
        return [SystemEnums.LEFT_LCD.value]
    if eye == "right":
        return [SystemEnums.RIGHT_LCD.value]
    return None


if __name__ == "__main__":
    parser = ArgumentParser(description="GLaDOS body MCP server")
    parser.add_argument('-config', type=str, default=1, dest='conf', nargs=1,
                        help='Config File')
    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    _ensure_init(args.conf[0])
    mcp.run()
