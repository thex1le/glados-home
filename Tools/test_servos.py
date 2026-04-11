#!/usr/bin/env python3
"""Servo test script — moves each servo through its full range of motion.

Reads servo config from glog.conf (min/max/center, pulse widths, board/channel).
Moves each servo: center → min → center → max → center with a pause at each position.

Usage:
    sudo $(which python3) Tools/test_servos.py -c glog.conf
    sudo $(which python3) Tools/test_servos.py -c dont_commit_glados.conf
    sudo $(which python3) Tools/test_servos.py -c glog.conf --servo head_lr
    sudo $(which python3) Tools/test_servos.py -c glog.conf --speed 0.5
"""

import argparse
import configparser
import sys
import time
from collections import namedtuple
from os import path

from adafruit_servokit import ServoKit

# Reuse the enum for config keys
sys.path.insert(0, path.join(path.dirname(__file__), '..'))
from glados_modules.GladosEnums import ServoEnum


def main():
    parser = argparse.ArgumentParser(description="GLaDOS Servo Test — full range of motion")
    parser.add_argument('-c', '--config', required=True, help='Config file path')
    parser.add_argument('--servo', choices=['body_lr', 'body_ud', 'head_lr', 'head_ud', 'all'],
                        default='all', help='Which servo to test (default: all)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Pause duration at each position in seconds (default: 1.0)')
    parser.add_argument('--steps', type=int, default=5,
                        help='Number of intermediate steps between positions (default: 5)')
    args = parser.parse_args()

    if not path.isfile(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(args.config)
    sech = config[ServoEnum.CONFIG_HEAD.value]

    # Parse pulse widths
    max_min = namedtuple("max_min", ['max', 'min'])
    pulse_90 = sech[ServoEnum.SERVO_MG90D_PULSE.value].split(',')
    pulse_92 = sech[ServoEnum.SERVO_MG92B_PULSE.value].split(',')
    pulse_3508 = sech[ServoEnum.SERVO_GS3508MG_PULSE.value].split(',')
    mg90d_pulse = max_min(int(pulse_90[0]), int(pulse_90[1]))
    mg92b_pulse = max_min(int(pulse_92[0]), int(pulse_92[1]))
    gs3508_pulse = max_min(int(pulse_3508[0]), int(pulse_3508[1]))

    # Parse angle ranges
    default = sech[ServoEnum.DEFAULT_MAX_MIN_CENTER.value].split(',')
    head_min_max = sech[ServoEnum.HEAD_MIN_MAX_CENTER.value].split(',')
    neck_min_max = sech[ServoEnum.NECK_MIN_MAX_CENTER.value].split(',')

    default_range = (int(default[0]), int(default[1]), int(default[2]))  # max, min, center
    head_range = (int(head_min_max[0]), int(head_min_max[1]), int(head_min_max[2]))
    neck_range = (int(neck_min_max[0]), int(neck_min_max[1]), int(neck_min_max[2]))

    # Initialize PCA9685
    pca_addrs = [int(a.strip(), 0) for a in sech.get(
        ServoEnum.PCA9685_ADDRESSES.value, ServoEnum.PCA9685_DEFAULT_ADDRESSES.value).split(',')]
    kits = [ServoKit(channels=16, address=addr) for addr in pca_addrs]
    print(f"PCA9685 boards: {[hex(a) for a in pca_addrs]}")

    def get_servo(board_channel_key, fallback_board=0, fallback_channel=0):
        raw = sech.get(board_channel_key, f"{fallback_board},{fallback_channel}")
        parts = raw.split(',')
        board_idx, channel = int(parts[0]), int(parts[1])
        return kits[board_idx].servo[channel]

    # Servo definitions: (name, servo_obj, pulse_range, angle_range(max, min, center))
    servos = {
        'body_lr': {
            'name': 'Body Left/Right',
            'servo': get_servo(ServoEnum.BODY_LR_BOARD_CHANNEL.value, 0, 0),
            'pulse': gs3508_pulse,
            'range': default_range,
        },
        'body_ud': {
            'name': 'Body Up/Down',
            'servo': get_servo(ServoEnum.BODY_UD_BOARD_CHANNEL.value, 0, 1),
            'pulse': mg92b_pulse,
            'range': default_range,
        },
        'head_lr': {
            'name': 'Head Left/Right',
            'servo': get_servo(ServoEnum.HEAD_LR_BOARD_CHANNEL.value, 0, 2),
            'pulse': mg92b_pulse,
            'range': neck_range,
        },
        'head_ud': {
            'name': 'Head Up/Down',
            'servo': get_servo(ServoEnum.HEAD_UD_BOARD_CHANNEL.value, 0, 3),
            'pulse': mg90d_pulse,
            'range': head_range,
        },
    }

    # Configure pulse ranges
    for key, info in servos.items():
        info['servo'].actuation_range = 180
        info['servo'].set_pulse_width_range(info['pulse'].min, info['pulse'].max)

    def move_smooth(servo, from_angle, to_angle, steps, step_delay):
        """Move servo smoothly from one angle to another."""
        for i in range(steps + 1):
            t = i / steps
            angle = from_angle + (to_angle - from_angle) * t
            angle = max(0, min(180, angle))
            servo.angle = angle
            time.sleep(step_delay)

    def test_servo(key, info):
        """Run full range test on a single servo."""
        s = info['servo']
        ang_max, ang_min, ang_center = info['range']
        name = info['name']
        step_delay = args.speed / args.steps

        print(f"\n{'='*50}")
        print(f"Testing: {name} ({key})")
        print(f"  Range: min={ang_min}° center={ang_center}° max={ang_max}°")
        print(f"  Pulse: {info['pulse'].min}-{info['pulse'].max} μs")
        print(f"{'='*50}")

        # Center
        print(f"  → Center ({ang_center}°)")
        s.angle = ang_center
        time.sleep(args.speed)

        # Center → Min
        print(f"  → Min ({ang_min}°)")
        move_smooth(s, ang_center, ang_min, args.steps, step_delay)
        time.sleep(args.speed)

        # Min → Center
        print(f"  → Center ({ang_center}°)")
        move_smooth(s, ang_min, ang_center, args.steps, step_delay)
        time.sleep(args.speed)

        # Center → Max
        print(f"  → Max ({ang_max}°)")
        move_smooth(s, ang_center, ang_max, args.steps, step_delay)
        time.sleep(args.speed)

        # Max → Center
        print(f"  → Center ({ang_center}°)")
        move_smooth(s, ang_max, ang_center, args.steps, step_delay)
        time.sleep(args.speed)

        print(f"  ✓ {name} complete")

    # Run tests
    print(f"\nGLaDOS Servo Test")
    print(f"Speed: {args.speed}s pause, {args.steps} steps between positions")
    print(f"Testing: {args.servo}")

    try:
        if args.servo == 'all':
            for key in ['body_lr', 'body_ud', 'head_lr', 'head_ud']:
                test_servo(key, servos[key])
        else:
            test_servo(args.servo, servos[args.servo])

        print(f"\n{'='*50}")
        print("All tests complete. Servos at center positions.")
        print(f"{'='*50}")

    except KeyboardInterrupt:
        print("\n\nInterrupted — centering all servos...")
        for key, info in servos.items():
            try:
                info['servo'].angle = info['range'][2]  # center
            except Exception:
                pass
        print("Done.")


if __name__ == "__main__":
    main()
