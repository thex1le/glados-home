from adafruit_servokit import ServoKit
from time import sleep
from threading import Thread


class Gservo(Thread):
    def __init__(self, skit, axis, max_angle=90):
        Thread.__init__(self)
        Thread.daemon = True
        # lock skit to the channel for this class
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

    def set_angle(self, angle):
        self.angle = angle

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
            self.skit.angle = self.angle
            sleep(.3)
            self.moving = True
            self.current_angle = self.angle
            self.moving = False
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
                self.exec_command = False
            else:
                sleep(.1)


class GBody(Thread):
    # class for managing all the servo and body movement in relation to the camera
    def __init__(self, cam_x_width: int, cam_y_width: int, seen_data, lock):
        Thread.__init__(self)
        Thread.daemon = True
        # access the servos
        kit = ServoKit(channels=16)
        # build a servo control for each joint
        self.body_LR = Gservo(skit=kit.servo[0], axis='x', max_angle=180)
        self.body_UD = Gservo(skit=kit.servo[1], axis='y', max_angle=60)
        self.head_UD = Gservo(skit=kit.servo[2], axis='x', max_angle=60)
        self.head_LR = Gservo(skit=kit.servo[3], axis='y', max_angle=60)
        self.seen_data = seen_data
        self.cam_x_width = cam_x_width
        self.cam_y_width = cam_y_width
        self.stop = False
        self.lock = lock
        # find the x1 x2, y1, y2 of the target,
        # figure out if the head can look at it...
        # if we can then head / neck moves to it...
        # then recalculate so the head and neck can move back to center
        # and the body will rotate and middle_angle will move up or down
        # order of off center is self.body_LR > self.body_UD,> self.head.UP> self, head left right

    def stop_body(self):
        """
        Stop body movement
        """
        self.stop = True

    def __find_person(self, target='person') -> dict:
        """
        Find the highest confidence person and return their bounding box from current data set
        self.seen_data expected to be YOLO8 data response object
        """
        with self.lock:
            rtn = dict()
            if target in self.seen_data and self.seen_data[target]['count'] > 0:
                highest_confidence = 0
                highest_confidence_person = None
                for p in self.seen_data[target]['objects']:
                    if p['confidence'] > highest_confidence:
                        highest_confidence = p['confidence']
                        highest_confidence_person = p
                if highest_confidence_person is not None:
                    if highest_confidence >= .70:
                        # take the highest confidence and return the bounding box
                        rtn = highest_confidence_person['box']
        return rtn

    def __calc_servo(self, servo: Gservo, bbox: dict) -> int:
        """
        Calculate servo angle correction to target
        """
        # TODO determine if we need current_angle? does it matter?
        if servo.axis == 'x':
            bbox_edge_1 = bbox['x1']
            bbox_edge_2 = bbox['x2']
            axis_size = self.cam_x_width
        else:
            bbox_edge_1 = bbox['y1']
            bbox_edge_2 = bbox['y2']
            axis_size = self.cam_y_width
        # Calculate the center of the new person's bounding box on the x-axis
        center_updated = (bbox_edge_1 + bbox_edge_2) / 2
        # Calculate the offset of the person's center from the image center with the updated data
        offset_from_center = center_updated - (axis_size / 2)
        # Calculate the new servo angle to center on the person with the updated data
        new_servo_angle_updated = servo.middle_angle - (offset_from_center / axis_size * servo.max_angle)
        # Round to nearest whole
        return round(new_servo_angle_updated)

    def __level_servos(self, servo1: Gservo, servo2: Gservo) -> None:
        # bring servo1 to midpoint by moving servo2
        # ensure servos are on the same axis
        if servo1.axis != servo2.axis:
            raise Exception("Servos are not on same axis")
        servo2.set_angle(servo1.get_angle())
        servo1.set_angle(servo1.get_middle_angle())
        servo1.move()
        servo2.move()

    def move_servos(self):
        target = self.__find_person()
        # move "shoulders" first
        head_lr = self.__calc_servo(self.head_LR, target)
        head_ud = self.__calc_servo(self.head_UD, target)
        # don't use threading for now
        self.head_LR.set_angle(head_lr)
        self.head_UD.set_angle(head_ud)
        self.head_LR.move()
        self.head_UD.move()
        # head should now be centered on the target
        # level the head and arm with body and rotation
        # x-axis
        self.__level_servos(self.head_LR, self.body_LR)
        self.__level_servos(self.head_UD, self.body_UD)

    def run(self):
        while self.stop is False:
            self.move_servos()
            sleep(.2)


if __name__ == "__main__":
    from multiprocessing import Manager, Lock
    gl = GBody(640, 640, Manager.dict(), Lock())
    gl.start()
