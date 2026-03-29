#!/usr/bin/env python3
"""Collect debug files into a zip for offline analysis.

Run this on any of the 3 GLaDOS systems (Pi4, Pi5, GPU server).
It auto-detects which system it's on, bundles all relevant logs,
recordings, and config into a single zip file you can move to
your dev machine.

Usage:
    python collect_debug.py
    python collect_debug.py --name "tracking_jitter_test"
    python collect_debug.py --hours 2  # only files from last 2 hours
"""

import os
import sys
import zipfile
import time
import socket
import platform
import argparse
from datetime import datetime
from glob import glob
from pathlib import Path


# Files/dirs to collect from any system
COMMON_GLOBS = [
    "glados_modules/logs/*.log",
    "logs/traces.jsonl",
    "recordings/*.jsonl",
    "txt_responses/*.conf",
    "*.yaml",
]

# System-specific globs (collected if they exist)
SYSTEM_GLOBS = [
    "Tests/logs/*.log",
    "glados_modules/logs/*.log",
]

# Always include these specific files
ALWAYS_INCLUDE = [
    "GLaDOS.py",
    "AiServer.py",
    "BodyServer.py",
    "glados_modules/GladosEnums.py",
    "glados_modules/VisionTracker.py",
    "glados_modules/BodyControlModules.py",
    "glados_modules/MachineVision.py",
    "glados_modules/MqttConnector.py",
    "glados_modules/MqttConsumerModules.py",
    "glados_modules/MotionRecorder.py",
]


def detect_system() -> str:
    """Guess which GLaDOS system we're running on."""
    hostname = socket.gethostname().lower()
    # Check for CUDA (GPU server)
    try:
        import torch
        if torch.cuda.is_available():
            return f"gpu_server ({hostname})"
    except ImportError:
        pass
    # Check for Pi hardware
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip()
            if "Pi 5" in model:
                return f"pi5 ({hostname})"
            elif "Pi 4" in model:
                return f"pi4 ({hostname})"
            return f"pi_unknown ({hostname})"
    except FileNotFoundError:
        pass
    return f"unknown ({hostname})"


def find_project_root() -> str:
    """Find the glados-home project root."""
    # Try common locations
    candidates = [
        os.path.dirname(os.path.abspath(__file__)),
        os.getcwd(),
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "GLaDOS.py")) or \
           os.path.isfile(os.path.join(path, "BodyServer.py")) or \
           os.path.isfile(os.path.join(path, "AiServer.py")):
            return path
    return candidates[0]


def collect_files(root: str, max_age_hours: float = None) -> list:
    """Collect all relevant files, optionally filtered by age."""
    files = set()
    now = time.time()
    cutoff = now - (max_age_hours * 3600) if max_age_hours else 0

    # Glob patterns
    for pattern in COMMON_GLOBS + SYSTEM_GLOBS:
        for match in glob(os.path.join(root, pattern), recursive=True):
            if max_age_hours and os.path.getmtime(match) < cutoff:
                continue
            files.add(match)

    # Always-include files (no age filter -- these are source code)
    for rel_path in ALWAYS_INCLUDE:
        full_path = os.path.join(root, rel_path)
        if os.path.isfile(full_path):
            files.add(full_path)

    return sorted(files)


def create_system_info(root: str) -> str:
    """Generate a system info summary."""
    lines = [
        f"System: {detect_system()}",
        f"Hostname: {socket.gethostname()}",
        f"Platform: {platform.platform()}",
        f"Python: {platform.python_version()}",
        f"Time: {datetime.now().isoformat()}",
        f"Project root: {root}",
        "",
        "--- Installed packages (glados-relevant) ---",
    ]
    # Check for key packages
    for pkg in ["paho-mqtt", "torch", "ultralytics", "rtmlib", "opencv-python",
                "adafruit-circuitpython-servokit", "adafruit-circuitpython-bno055",
                "pydub", "numpy", "cachetools", "picamera2"]:
        try:
            mod = __import__(pkg.replace("-", "_").split("-")[0])
            ver = getattr(mod, "__version__", "installed")
            lines.append(f"  {pkg}: {ver}")
        except ImportError:
            pass

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect GLaDOS debug files into a zip")
    parser.add_argument("--name", default=None, help="Session name (default: auto-generated)")
    parser.add_argument("--hours", type=float, default=None,
                        help="Only include log/recording files from the last N hours")
    parser.add_argument("--output-dir", default=".", help="Where to write the zip (default: current dir)")
    args = parser.parse_args()

    root = find_project_root()
    system = detect_system().split(" ")[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or "debug"
    zip_name = f"glados_{system}_{name}_{ts}.zip"
    zip_path = os.path.join(args.output_dir, zip_name)

    print(f"System:       {detect_system()}")
    print(f"Project root: {root}")
    if args.hours:
        print(f"Age filter:   last {args.hours} hours")

    files = collect_files(root, max_age_hours=args.hours)
    print(f"Files found:  {len(files)}")

    if not files:
        print("No files to collect!")
        return

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add system info
        zf.writestr("system_info.txt", create_system_info(root))

        # Add collected files
        for filepath in files:
            rel = os.path.relpath(filepath, root)
            zf.write(filepath, rel)
            print(f"  + {rel}")

    size_kb = os.path.getsize(zip_path) / 1024
    print(f"\nCreated: {zip_path} ({size_kb:.1f} KB)")
    print(f"Transfer this file to your dev machine for analysis.")


if __name__ == "__main__":
    main()
