#!/usr/bin/env python3
"""MQTT topic watcher for debugging GLaDOS tracking.

Usage:
    python mqtt_watch.py <broker_ip> [topics...]

Examples:
    python mqtt_watch.py 192.168.1.2                    # watch all tracking-related topics
    python mqtt_watch.py 192.168.1.2 body/servo/#       # watch servo commands only
    python mqtt_watch.py 192.168.1.2 '#'                # watch everything (noisy)
"""

# builtin
import sys
import json
import time
from datetime import datetime

# 3rd party
import paho.mqtt.client as mqtt

# Default topics relevant to tracking debugging
DEFAULT_TOPICS = [
    "body/servo",           # servo commands (move, move_all)
    "body/servo/status",    # servo position feedback
    "system/track",         # track start command
    "vision/camera_response",  # vision detection results
    "system/mood",          # mood changes
    "system/room",          # room state (roster, arrivals, departures)
]

# ANSI colors for readability
COLORS = {
    "body/servo": "\033[93m",       # yellow
    "body/servo/status": "\033[96m", # cyan
    "system/track": "\033[92m",     # green
    "vision/camera_response": "\033[95m",  # magenta
    "system/mood": "\033[91m",      # red
    "system/room": "\033[94m",      # blue
}
RESET = "\033[0m"
DIM = "\033[2m"


def color_for_topic(topic: str) -> str:
    """Get ANSI color for a topic."""
    for prefix, color in COLORS.items():
        if topic.startswith(prefix):
            return color
    return ""


def format_servo_cmd(data: dict) -> str:
    """Compact format for servo commands."""
    cmd = data.get("cmd", "?")
    if cmd == "move_all":
        targets = data.get("targets", {})
        parts = []
        for loc, vals in targets.items():
            short = loc.replace("head_", "h_").replace("body_", "b_").replace("left_right", "lr").replace("up_down", "ud")
            angle = vals.get("angle", "?")
            speed = vals.get("speed", "?")
            parts.append(f"{short}={angle:.1f}@{speed}")
        trace = data.get("trace_id", "")[:8]
        return f"move_all [{', '.join(parts)}] trace={trace}"
    elif cmd == "move":
        loc = data.get("location", "?")
        angle = data.get("angle", "?")
        speed = data.get("speed", "?")
        return f"move {loc}={angle}@{speed}"
    elif cmd == "status":
        return "status request"
    return json.dumps(data, separators=(",", ":"))


def format_servo_status(data: dict) -> str:
    """Compact format for servo status."""
    results = data.get("results", data)
    if isinstance(results, dict):
        loc = results.get("location", "?")
        current = results.get("current_angle", "?")
        moving = results.get("moving", "?")
        vel = results.get("velocity", "?")
        short = loc.replace("head_", "h_").replace("body_", "b_").replace("left_right", "lr").replace("up_down", "ud")
        return f"{short}: angle={current} moving={moving} vel={vel}"
    return json.dumps(data, separators=(",", ":"))


def format_vision(data: dict) -> str:
    """Compact format for vision results."""
    camera = data.get("camera", "?")
    parts = [f"cam={camera}"]
    for cls_name, cls_data in data.items():
        if isinstance(cls_data, dict) and "objects" in cls_data:
            objs = cls_data["objects"]
            parts.append(f"{cls_name}:{len(objs)}")
            for i, obj in enumerate(objs[:3]):
                box = obj.get("box", {})
                x1 = box.get("x1", "?")
                x2 = box.get("x2", "?")
                cx = (x1 + x2) / 2 if isinstance(x1, (int, float)) and isinstance(x2, (int, float)) else "?"
                conf = obj.get("confidence", "?")
                if isinstance(conf, float):
                    conf = f"{conf:.2f}"
                parts.append(f"  obj{i}: cx={cx} conf={conf}")
    return " | ".join(parts)


def format_room_state(data: dict) -> str:
    """Compact format for room state messages."""
    count = data.get("count", 0)
    arrivals = data.get("arrivals", [])
    departures = data.get("departures", [])
    roster = data.get("roster", [])
    parts = [f"room={count}"]
    if arrivals:
        parts.append(f"+[{','.join(arrivals)}]")
    if departures:
        parts.append(f"-[{','.join(departures)}]")
    for p in roster[:5]:
        pid = p.get("person_id", "?")
        lr = p.get("world_lr", "?")
        cams = ",".join(c.replace("camera_", "") for c in p.get("cameras", []))
        emo = p.get("emotion", "")
        emo_str = f" {emo}" if emo and emo != "neutral" else ""
        parts.append(f"  {pid}: lr={lr}° [{cams}]{emo_str}")
    attn = data.get("attention_target")
    if attn:
        reason = data.get("attention_reason", "?")
        parts.append(f"  >> {attn} ({reason})")
    return " | ".join(parts[:3]) + ("\n" + "\n".join(parts[3:]) if len(parts) > 3 else "")


def format_message(topic: str, data: dict) -> str:
    """Format a message based on its topic."""
    if topic == "body/servo":
        return format_servo_cmd(data)
    elif topic == "body/servo/status":
        return format_servo_status(data)
    elif topic.startswith("vision/"):
        return format_vision(data)
    elif topic == "system/room":
        return format_room_state(data)
    else:
        # Generic compact JSON
        compact = json.dumps(data, separators=(",", ":"))
        if len(compact) > 200:
            return compact[:200] + "..."
        return compact


def on_connect(client, userdata, flags, rc, properties=None):
    """Subscribe to topics on connect."""
    topics = userdata["topics"]
    print(f"{DIM}Connected to broker (rc={rc}), subscribing to {len(topics)} topics...{RESET}")
    for topic in topics:
        client.subscribe(topic)
        print(f"  + {topic}")
    print(f"{DIM}--- Watching (Ctrl+C to stop) ---{RESET}")
    print()


def on_message(client, userdata, msg):
    """Print received messages with formatting."""
    topic = msg.topic
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    color = color_for_topic(topic)

    try:
        data = json.loads(msg.payload.decode("utf-8"))
        # Strip uuid for readability
        data.pop("uuid", None)
        formatted = format_message(topic, data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        formatted = msg.payload.decode("utf-8", errors="replace")[:200]

    print(f"{DIM}{ts}{RESET} {color}{topic}{RESET}  {formatted}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    broker_ip = sys.argv[1]
    port = 1883

    if len(sys.argv) > 2:
        topics = sys.argv[2:]
    else:
        topics = DEFAULT_TOPICS

    # Add wildcard suffixes for topics that don't already have them
    subscribe_topics = []
    for t in topics:
        subscribe_topics.append(t)
        # Also subscribe to subtopics if not already a wildcard
        if not t.endswith("#") and not t.endswith("+"):
            subscribe_topics.append(f"{t}/#")

    # Deduplicate
    subscribe_topics = list(dict.fromkeys(subscribe_topics))

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"mqtt_watch_{int(time.time())}",
        userdata={"topics": subscribe_topics},
    )
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {broker_ip}:{port}...")
    client.connect(broker_ip, port)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print(f"\n{DIM}Disconnected.{RESET}")
        client.disconnect()


if __name__ == "__main__":
    main()
