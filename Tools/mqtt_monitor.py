"""Live MQTT message monitor for debugging the GLaDOS pipeline.

Subscribes to all GLaDOS MQTT topics and displays messages in a readable format.
Supports filtering by topic, compact vs verbose output, and logging to file.

Usage:
    # Watch everything
    python3 Tools/mqtt_monitor.py -config glog.conf

    # Watch only servo commands and status
    python3 Tools/mqtt_monitor.py -config glog.conf --filter servo

    # Watch vision results
    python3 Tools/mqtt_monitor.py -config glog.conf --filter vision

    # Watch IMU data (compact, updates in place)
    python3 Tools/mqtt_monitor.py -config glog.conf --filter imu --compact

    # Watch multiple topics
    python3 Tools/mqtt_monitor.py -config glog.conf --filter servo vision

    # Log to file
    python3 Tools/mqtt_monitor.py -config glog.conf --log mqtt_debug.log

    # Raw mode (show full JSON)
    python3 Tools/mqtt_monitor.py -config glog.conf --raw

Available filters:
    servo    - Servo commands and status (body/servo, body/servo/status)
    vision   - Vision/YOLO results (vision/camera/results)
    imu      - IMU sensor data (body/imu/status)
    tof      - Time-of-flight distance (body/tof/status)
    th       - Temperature/humidity (body/th/status)
    mox      - Air quality (body/mox/status)
    led      - LED commands (body/led)
    lcd      - LCD commands (body/lcd)
    track    - Tracking commands (system/track)
    health   - Health heartbeats (system/health/#)
    log      - Log level changes (system/log_level)
    all      - Everything (default)
"""

# builtin
import argparse
import sys
import os
import time
from configparser import ConfigParser
from json import loads, dumps
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from glados_modules.GladosEnums import (
    ServoEnum, MQTTEnums, IMUEnums, TOFEnums, THEnums, MOXEnums,
    SystemEnums, TrackingEnums, CameraEnum, VisionResultsEnum, TraceEnums
)


# Topic groups for filtering
TOPIC_FILTERS = {
    "servo": [
        ServoEnum.MQTT_COMMAND_TOPIC.value,
        ServoEnum.MQTT_STATUS_TOPIC.value,
    ],
    "vision": [
        MQTTEnums.VISION_RESULTS_MQTT_TOPIC.value,
    ],
    "imu": [
        MQTTEnums.IMU_STATUS_TOPIC.value,
    ],
    "tof": [
        MQTTEnums.TOF_STATUS_TOPIC.value,
    ],
    "th": [
        MQTTEnums.TH_STATUS_TOPIC.value,
    ],
    "mox": [
        MQTTEnums.MOX_STATUS_TOPIC.value,
    ],
    "led": [
        MQTTEnums.BODY_LED_CONTROL_MQTT_TOPIC.value,
    ],
    "lcd": [
        MQTTEnums.LCD_CONTROL_MQTT_TOPIC.value,
    ],
    "track": [
        TrackingEnums.MQTT_COMMAND_TOPIC.value,
    ],
    "health": [
        f"{MQTTEnums.SYSTEM_HEALTH_TOPIC.value}/#",
    ],
    "log": [
        MQTTEnums.SYSTEM_LOG_LEVEL_TOPIC.value,
    ],
    "intensity": [
        MQTTEnums.SYSTEM_INTENSITY_TOPIC.value,
    ],
}


def format_servo_cmd(data: dict) -> str:
    """Format a servo command message."""
    cmd = data.get(ServoEnum.MSG_COMMAND_KEY.value, "?")
    if cmd == ServoEnum.MSG_COMMAND_MOVE.value:
        loc = data.get(ServoEnum.MSG_LOCATION_KEY.value, "?")
        angle = data.get(ServoEnum.MSG_ANGLE.value, "?")
        speed = data.get(ServoEnum.MSG_SPEED.value, "?")
        trace = data.get(TraceEnums.TRACE_ID.value, "")
        trace_str = f"  trace={trace}" if trace else ""
        return f"MOVE {loc} -> {angle}° speed={speed}{trace_str}"
    elif cmd == ServoEnum.MSG_COMMAND_MOVE_ALL.value:
        targets = data.get(ServoEnum.MSG_TARGETS.value, {})
        parts = []
        for loc, t in targets.items():
            short_loc = loc.replace("_left_right", "_LR").replace("_up_down", "_UD")
            parts.append(f"{short_loc}={t.get(ServoEnum.MSG_ANGLE.value, '?')}°")
        trace = data.get(TraceEnums.TRACE_ID.value, "")
        trace_str = f"  trace={trace}" if trace else ""
        return f"MOVE_ALL {' '.join(parts)}{trace_str}"
    elif cmd == ServoEnum.MSG_COMMAND_STATUS.value:
        loc = data.get(ServoEnum.MSG_LOCATION_KEY.value, "?")
        return f"STATUS request for {loc}"
    return f"cmd={cmd}"


def format_servo_status(data: dict) -> str:
    """Format a servo status message."""
    results = data.get(ServoEnum.MSG_RESULTS.value, {})
    loc = results.get(ServoEnum.MSG_LOCATION.value, "?")
    current = results.get(ServoEnum.MSG_CURRENT_ANGLE.value, "?")
    velocity = results.get(ServoEnum.MSG_VELOCITY.value, 0)
    moving = results.get(ServoEnum.MSG_MOVING.value, False)
    short_loc = loc.replace("_left_right", "_LR").replace("_up_down", "_UD")
    status = "MOVING" if moving else "still"
    return f"{short_loc}: {current}° vel={velocity:.1f} [{status}]"


def format_imu(data: dict) -> str:
    """Format IMU status message."""
    imu = data.get(IMUEnums.IMU_STATUS_KEY.value, {})
    euler = imu.get(IMUEnums.EULER_KEY.value, (0, 0, 0))
    gyro = imu.get(IMUEnums.GYRO_KEY.value, (0, 0, 0))
    linear = imu.get(IMUEnums.LINEAR_KEY.value, (0, 0, 0))
    if euler:
        return (f"euler=({euler[0]:6.1f}, {euler[1]:5.1f}, {euler[2]:5.1f})  "
                f"gyro=({gyro[0]:5.2f}, {gyro[1]:5.2f}, {gyro[2]:5.2f})  "
                f"linear=({linear[0]:5.2f}, {linear[1]:5.2f}, {linear[2]:5.2f})")
    return str(imu)[:100]


def format_vision(data: dict) -> str:
    """Format vision results message."""
    results = data.get(CameraEnum.MSG_RESULTS.value, {})
    parts = []
    for camera, detections in results.items():
        if isinstance(detections, dict):
            for obj_type, obj_data in detections.items():
                if isinstance(obj_data, dict):
                    count = obj_data.get(VisionResultsEnum.VISION_RESULTS_COUNT_KEY.value, 0)
                    if count > 0:
                        parts.append(f"{camera}: {count}x {obj_type}")
    trace = data.get(TraceEnums.TRACE_ID.value, "")
    trace_str = f"  trace={trace}" if trace else ""
    return " | ".join(parts) + trace_str if parts else "no detections"


def format_sensor(data: dict, status_key: str) -> str:
    """Format a generic sensor status message."""
    sensor = data.get(status_key, {})
    if isinstance(sensor, dict):
        # Show key-value pairs compactly
        parts = [f"{k}={v}" for k, v in sensor.items() if k != "time"]
        return "  ".join(parts[:5])
    return str(sensor)[:100]


def format_health(data: dict) -> str:
    """Format health heartbeat message."""
    threads = data.get("threads", {})
    alive = sum(1 for t in threads.values() if t.get("alive", False))
    dead = sum(1 for t in threads.values() if not t.get("alive", True))
    host = data.get("hostname", "?")
    return f"{host}: {alive} alive, {dead} dead"


def format_message(topic: str, data: dict) -> str:
    """Format a message based on its topic."""
    if topic == ServoEnum.MQTT_COMMAND_TOPIC.value:
        return format_servo_cmd(data)
    elif topic == ServoEnum.MQTT_STATUS_TOPIC.value:
        return format_servo_status(data)
    elif topic == MQTTEnums.IMU_STATUS_TOPIC.value:
        return format_imu(data)
    elif topic == MQTTEnums.VISION_RESULTS_MQTT_TOPIC.value:
        return format_vision(data)
    elif topic == MQTTEnums.TOF_STATUS_TOPIC.value:
        return format_sensor(data, TOFEnums.TOF_STATUS_KEY.value)
    elif topic == MQTTEnums.TH_STATUS_TOPIC.value:
        return format_sensor(data, THEnums.TH_STATUS_KEY.value)
    elif topic == MQTTEnums.MOX_STATUS_TOPIC.value:
        return format_sensor(data, MOXEnums.MOX_STATUS_KEY.value)
    elif topic.startswith(MQTTEnums.SYSTEM_HEALTH_TOPIC.value):
        return format_health(data)
    return ""


class MQTTMonitor:
    """MQTT subscriber that monitors and displays GLaDOS messages."""

    def __init__(self, broker_ip: str, broker_port: int, topics: list,
                 raw: bool = False, compact: bool = False, log_file: str = None) -> None:
        import paho.mqtt.client as mqtt
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._broker_ip = broker_ip
        self._broker_port = broker_port
        self._topics = topics
        self._raw = raw
        self._compact = compact
        self._log_file = None
        self._msg_count = 0
        self._start_time = time.time()
        self._compact_state = {}

        if log_file:
            self._log_file = open(log_file, 'a')
            self._log_file.write(f"\n--- Session started {datetime.now().isoformat()} ---\n")

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def connect(self) -> None:
        self._client.connect(self._broker_ip, self._broker_port)
        print(f"Connected to MQTT broker at {self._broker_ip}:{self._broker_port}")
        print(f"Subscribed topics: {', '.join(self._topics)}")
        print(f"Mode: {'raw JSON' if self._raw else 'compact' if self._compact else 'formatted'}")
        print("-" * 80)

    def run(self) -> None:
        try:
            self._client.loop_forever()
        except KeyboardInterrupt:
            elapsed = time.time() - self._start_time
            print(f"\n\nStopped. {self._msg_count} messages in {elapsed:.1f}s "
                  f"({self._msg_count / max(1, elapsed):.1f} msg/s)")
            if self._log_file:
                self._log_file.close()
                print(f"Log saved.")
            self._client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        for topic in self._topics:
            client.subscribe(topic)

    def _on_message(self, client, userdata, msg) -> None:
        self._msg_count += 1
        try:
            data = loads(msg.payload.decode())
        except Exception:
            return

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        topic_short = msg.topic

        if self._raw:
            line = f"[{timestamp}] {topic_short}: {dumps(data, indent=2)}"
            print(line)
        elif self._compact:
            formatted = format_message(msg.topic, data)
            if formatted:
                # Overwrite line for high-frequency topics (IMU, sensors)
                if msg.topic in (MQTTEnums.IMU_STATUS_TOPIC.value,
                                 MQTTEnums.TOF_STATUS_TOPIC.value,
                                 MQTTEnums.TH_STATUS_TOPIC.value,
                                 MQTTEnums.MOX_STATUS_TOPIC.value):
                    self._compact_state[msg.topic] = formatted
                    # Redraw all compact lines
                    lines = [f"  {t.split('/')[-1]:>10}: {v}" for t, v in sorted(self._compact_state.items())]
                    sys.stdout.write(f"\r\033[{len(lines)}A" if len(lines) > 1 else "\r")
                    for l in lines:
                        sys.stdout.write(f"\033[K{l}\n")
                    sys.stdout.flush()
                else:
                    print(f"[{timestamp}] {topic_short}: {formatted}")
            else:
                print(f"[{timestamp}] {topic_short}: {str(data)[:120]}")
        else:
            formatted = format_message(msg.topic, data)
            if formatted:
                print(f"[{timestamp}] {topic_short}: {formatted}")
            else:
                print(f"[{timestamp}] {topic_short}: {str(data)[:120]}")

        if self._log_file:
            self._log_file.write(f"[{timestamp}] {msg.topic}: {dumps(data)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor GLaDOS MQTT messages for debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Filters: " + ", ".join(sorted(TOPIC_FILTERS.keys())),
    )
    parser.add_argument("-config", required=True, help="Path to glog.conf")
    parser.add_argument("--filter", nargs="+", default=["all"],
                        help="Topic filter(s): servo, vision, imu, tof, th, mox, led, lcd, "
                             "track, health, log, all (default: all)")
    parser.add_argument("--raw", action="store_true",
                        help="Show full JSON payloads")
    parser.add_argument("--compact", action="store_true",
                        help="Compact display (high-frequency topics update in place)")
    parser.add_argument("--log", type=str, default=None,
                        help="Log all messages to file (JSON lines)")
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)
    broker_ip = config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_SERVER_IP.value)
    broker_port = int(config.get(SystemEnums.CONFIG_HEAD_MQTT.value, SystemEnums.MQTT_PORT.value))

    # Build topic list from filters
    if "all" in args.filter:
        topics = ["#"]
    else:
        topics = []
        for f in args.filter:
            if f in TOPIC_FILTERS:
                topics.extend(TOPIC_FILTERS[f])
            else:
                print(f"WARNING: Unknown filter '{f}', subscribing as raw topic")
                topics.append(f)

    monitor = MQTTMonitor(
        broker_ip=broker_ip,
        broker_port=broker_port,
        topics=topics,
        raw=args.raw,
        compact=args.compact,
        log_file=args.log,
    )
    monitor.connect()
    monitor.run()


if __name__ == "__main__":
    main()
