from time import sleep, time
from threading import Thread
from json import loads, dumps

# 3rd party
import paho.mqtt.client as mqtt
from adafruit_servokit import ServoKit

# glados imports
from glog_conifig import setup_logger


class Gservo(Thread):
    def __init__(self, location, skit, axis, servo_range: tuple = (), max_angle=90,
                 broker='localhost', port=1883, topic="body/servo"):
        Thread.__init__(self)
        Thread.daemon = True
        self.logger = setup_logger(name=f"{self.__name__}_{location}")
        self.location = location
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        # lock skit to the channel for this class
        self.min_angle = 0
        self.skit = skit
        # start at middle speed
        self.speed = 5
        # default to 45 for the 90's
        self.max_angle = max_angle
        # set it in the middle_angle
        self.middle_angle = int(self.max_angle / 2)
        self.angle = self.middle_angle
        self.current_angle = self.angle
        self.first_boot = True
        self.move()
        self.exec_command = False
        self.moving = False
        self.axis = axis.lower()
        if servo_range == ():
            self.allowed_servo_range = {"min_travel": 0, "max_travel": max_angle}
        else:
            self.allowed_servo_range = {"min_travel": servo_range[0], "max_travel": servo_range[1]}

    def on_connect(self, client, userdata, flags, rc):
        self.logger.debug(f"Connecting to {self.broker}:{self.port} on channel {self.topic}")
        self.client.subscribe(self.topic)

    def on_message(self, client, userdata, msg):
        cmd = msg.payload.decode()
        self.logger.debug(f"{self.location}, {msg.topic},  {cmd}")
        jmsg = loads(cmd)
        if jmsg.get("servo", "") == self.location:
            # message is the correct servo
            angle = jmsg.get("angle", self.middle_angle)
            speed = jmsg.get("speed", self.speed)
            self.set_speed_angle((angle, speed), execute=True)

    def get_max_angle(self):
        return self.max_angle

    def get_middle_angle(self):
        return self.middle_angle

    def set_speed(self, speed):
        if speed >= 10:
            # top speed of 10
            speed = 10
        if speed <= 1:
            # go as slow as 1
            speed = 1
        self.speed = round(speed)
        self.logger.debug(f"Speed set to {self.speed}")

    def set_angle(self, angle):
        max = self.allowed_servo_range["max_travel"]
        min = self.allowed_servo_range["min_travel"]
        if angle >= max:
            self.angle = max
            self.logger.debug(f"{angle} is above {max}, setting to {max}")
        elif angle <= min:
            self.angle = min
            self.logger.debug(f"{angle} is below {min}, setting to {min}")
        else:
            self.angle = angle
            self.logger.debug(f"Angle set to {self.angle}")

    def set_speed_angle(self, speed_angle: tuple, execute=False):
        self.set_speed(speed_angle[0])
        self.set_angle(speed_angle[1])
        if execute is True:
            self.exec_command = True

    def get_angle(self):
        return self.current_angle

    def execute(self):
        self.exec_command = True

    def __get_direction_speed(self):
        # determine current angle and if were going up or down return a range object
        # moving higher
        rtn = range(0, 0)
        if self.angle > self.current_angle:
            rtn = range(self.current_angle, (self.angle + 1), self.speed)
        # moving lower()
        if self.angle < self.current_angle:
            rtn = range(self.current_angle, (self.angle + 1), (self.speed * -1))
        return rtn

    def __increment(self):
        # print you left off here trying to handle positive and negative values
        for s in self.__get_direction_speed():
            self.skit.angle = s
            sleep(.1)
        self.current_angle = self.angle

    def get_moving_status(self):
        # return if motor is moving or not
        return self.moving

    def move(self):
        if self.speed == 10 or self.first_boot is True:
            self.logger.debug(f"moving to {self.angle}")
            self.skit.angle = self.angle
            sleep(.3)
            self.moving = True
            self.current_angle = self.angle
            self.moving = False
            self.first_boot = False
        else:
            if self.angle != self.current_angle:
                self.logger.debug(f"moving to {self.angle}")
                self.moving = True
                self.__increment()
                self.moving = False

    def run(self):
        while True:
            if self.exec_command is True:
                self.move()
                self.exec_command = False
            else:
                sleep(.1)


if __name__ == "__main__":
    ip = '192.168.86.52'
    kit = ServoKit(channels=16)
    body_LR = Gservo(location='body_left_right', skit=kit.servo[0], axis='x', max_angle=180, broker=ip)
    body_UD = Gservo(location='body_up_down', skit=kit.servo[1], axis='y', max_angle=60, broker=ip)
    head_UD = Gservo(location='head_up_down', skit=kit.servo[2], axis='y', max_angle=60, broker=ip)
    head_LR = Gservo(location='head_left_right', servo_range=(15, 45), skit=kit.servo[3], axis='x', max_angle=60,
                     broker=ip)
    while True:
        sleep(1)
