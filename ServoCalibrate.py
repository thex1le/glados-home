from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)
skit = kit[15]
skit.angle(90)
#import adafruit_motor.servo
#servo = adafruit_motor.servo.Servo(servo_channel)