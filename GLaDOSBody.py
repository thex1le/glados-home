from adafruit_servokit import ServoKit
from time import sleep
from threading import Thread


class gservo(Thread):
    def __init__(self, skit, max_angle=90):
        Thread.__init__(self)
        Thread.daemon = True
        # lock skit to the channel for this class
        self.skit = skit
        # start at slowest speed
        self.speed = 1
        # default to 45 for the 90's 
        self.max_angle = max_angle
        # set it in the middle
        self.middle = int(self.max_angle / 2)
        self.angle = self.middle
        self.current_angle = self.angle
        self.first_boot = True
        self.move()
        self.exec_command = False
        self.moveing = False

    def map_pixel(self):
        # calculate the offset for a given image side
        half = self.cfov / 2
        fov_l = self.middle - half
        fov_r = self.middle + half
        degree_per_pixel = (fov_r - fov_l) / self.image_axis
        return degree_per_pixel
    
    def get_max_angle(self):
        return self.max_angle

    def set_speed(self, speed):
        if speed >= 10:
            # top speed of 10
            speed = 10
        if speed <= 1:
            # go as slow as 1
            speed = 1
        self.speed = round(speed)
    
    def set_angle(self, angle):
        self.angle = angle
    
    def set_speed_angle(self, speed_angle:tuple, execute=False):
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
        if self.angle > self.current_angle:
            rtn = range(self.current_angle, (self.angle + 1), self.speed)
        # moving lower()
        if self.angle < self.current_angle:
            rtn = range(self.current_angle, (self.angle +1), (self.speed * -1))
        return rtn

    def __increment(self):
        # print you left off here trying to handle positive and negative values
        for s in self.__get_direction_speed():
            self.skit.angle = s
            sleep(.1)
        self.current_angle = self.angle

    def get_moving_status(self):
        # return if motor is moving or not
        return self.moveing

    def move(self):
        if self.speed == 10 or self.first_boot is True:
            self.skit.angle = self.angle
            sleep(.3)
            self.moveing = True
            self.current_angle = self.angle
            self.moveing = False
            self.first_boot = False
        else:
            if self.angle != self.current_angle:
                self.moving = True
                self.__increment()
                self.moving = False
    
    def run(self):
        while True:
            if self.exec_command is True:
                self.move()
                self.exec_comand = False
            else:
                sleep(.1)


if __name__ == "__main__":
    kit = ServoKit(channels=16)
    body_rotate = gservo(skit=kit.servo[0], max_angle=180)
    #body_rotate.start()
    while True:
        print("enter degrees, current angle is {}".format(body_rotate.get_angle()))
        x = int(input())
        body_rotate.set_speed_angle((5, x), execute=True)
        body_rotate.move()
        # TO DO THE SYSTEM DOESN"T YET KNOWF IT MOVES AND MAY RETURN WRONG LOCATION IF ITS MOVING>>> SO CONFUSING AND WRONG>> CANT TRUST LIB>>> PROB NEED CALLBACK>>>>
        #    #TODO consider if we should block on movment..
        sleep(2)
    #    sleep(2)
    """
    body_up_down = gservo(kit, 1, 60)
    body_up_down.start()
    body_rotate.set_speed_angle((1, 1))
    body_up_down.set_speed_angle((2, 10))
    body_rotate.execute()
    body_up_down.
    sleep(15) 
    body_rotate.set_angle(180)
    body_rotate.set_speed(5)
    body_rotate.execute()
    sleep(30)
    """
