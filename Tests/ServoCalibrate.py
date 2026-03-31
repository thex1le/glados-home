from adafruit_servokit import ServoKit
import argparse
import time

parser = argparse.ArgumentParser(description="Servo pulse width calibration")
parser.add_argument("--address", type=lambda x: int(x, 0), default=0x40,
                    help="PCA9685 I2C address (default: 0x40, e.g. 0xA0 for second board)")
parser.add_argument("--channel", type=int, default=3, help="Servo channel (default: 3)")
args = parser.parse_args()

kit = ServoKit(channels=16, address=args.address)
servo_channel = args.channel
print(f"Board address: {hex(args.address)}  Channel: {servo_channel}")

# Set the pulse width range
# These values might need to be adjusted based on your servo's specifications
# mg90
kit.servo[servo_channel].set_pulse_width_range(min_pulse=610, max_pulse=2665)
# BIG SERVO 92B
#kit.servo[servo_channel].set_pulse_width_range(min_pulse=605, max_pulse=2550)
s = 1
# Test the servo by setting it to various angles
kit.servo[servo_channel].angle = 0  # Move to 0 degrees
time.sleep(s)  # Wait for 1 second

kit.servo[servo_channel].angle = 45  # Move to 90 degrees
time.sleep(s)

kit.servo[servo_channel].angle = 90  # Move to 90 degrees
time.sleep(s)

kit.servo[servo_channel].angle = 135  # Move to 180 degrees
time.sleep(s)

kit.servo[servo_channel].angle = 180  # Move to 180 degrees
time.sleep(s)

kit.servo[servo_channel].angle = 1
time.sleep(s)
for i in range(0,181, 1):
    kit.servo[servo_channel].angle = i
    time.sleep(.01)

for i in range(180,-1, -1):
    kit.servo[servo_channel].angle = i
    time.sleep(.01)
# Move to 180 degrees
kit.servo[servo_channel].angle = 180
