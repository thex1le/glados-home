from adafruit_servokit import ServoKit
import time
kit = ServoKit(channels=16)
# Specify the servo channel you are calibrating
#servo_channel = 15  # Change this to the correct channel
servo_channel = 3 # Change this to the correct channel

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