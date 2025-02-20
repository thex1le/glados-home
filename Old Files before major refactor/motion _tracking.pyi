"""
class GBody(Thread):
    # class for managing all the servo and body movement in relation to the camera
    def __init__(self, config_file, cam_x_width: int, cam_y_width: int, lock):
        Thread.__init__(self)
        Thread.daemon = True
        self.logger = setup_logger(self.__name__)
        # access the servos
        kit = ServoKit(channels=16)
        # build a servo control for each joint
        self.body_LR = Gservo(location='body_left_right', skit=kit.servo[0], axis='x', max_angle=180)
        self.body_UD = Gservo(location='body_up_down', skit=kit.servo[1], axis='y', max_angle=60)
        self.head_UD = Gservo(location='head_up_down', skit=kit.servo[2], axis='y', max_angle=60)
        self.head_LR = Gservo(location='head_lef_right', servo_range=(15, 45), skit=kit.servo[3], axis='x', max_angle=60)
        self.seen_data = Manager().dict()
        self.cam_x_width = cam_x_width
        self.cam_y_width = cam_y_width
        self.stop = False
        self.lock = lock
        self.eyes = gleyes(self.set_scan_success, config_file)
        self.eyes.start()
        # find the x1 x2, y1, y2 of the target,
        # figure out if the head can look at it...
        # if we can then head / neck moves to it...
        # then recalculate so the head and neck can move back to center
        # and the body will rotate and middle_angle will move up or down
        # order of off center is self.body_LR > self.body_UD,> self.head.UP> self, head left right
        # TODO figure out how we are going to track anger intensity over various body parts
        self.led_head = LedHead()
        # thread the startup of the led head
        led_head_start = Thread(target=self.led_head.startup, args=())
        self.big_lcd_left = GLaDOSDisplay.GladosLCD()
        self.little_lcd_right = GLaDOSDisplay.GladosLCD(cs=board.D23, rst=board.D5, dc=board.D6,
                                                        sck=board.SCK_1, mosi=board.MOSI_1, flip=True)
        self.big_lcd_left.start()
        self.little_lcd_right.start()
        led_head_start.start()
        led_head_start.join()
        self.scan_success = False

    def set_scan_success(self):
        # callback for the camera thread to signal the servos to stop moving
        self.logger.debug("Body Callback triggered")
        self.scan_success = True

    def scan_room(self, scan_speed=3, search_time=90, confidence=.70):
        #TODO consider how this will change with left and right cameras...,
        self.logger.debug("Scanning Room for Target")
        self.eyes.target_scan(search_time=search_time, confidence=confidence)
        t = time()
        while (time() - t) < search_time and self.scan_success is False:
            if self.scan_success is False:
                self.head_LR.set_speed_angle((scan_speed, self.head_LR.min_angle), execute=True)
                self.body_LR.set_speed_angle((scan_speed, self.body_LR.min_angle), execute=True)
                # TODO change when threading is enabled
                self.head_LR.move()
                self.body_LR.move()
            else:
                break
            # block till head and body are at min
            while (self.body_LR.get_angle() != self.body_LR.min_angle and
                   self.head_LR.get_angle() != self.head_LR.min_angle or self.scan_success is True):
                sleep(.2)
            if self.scan_success is False:
                self.head_LR.set_speed_angle((scan_speed, self.head_LR.max_angle), execute=True)
                self.body_LR.set_speed_angle((scan_speed, self.body_LR.max_angle), execute=True)
                # TODO change when threading is enabled
                self.head_LR.move()
                self.body_LR.move()
            else:
                break
            # block till head and body are at max
            while (self.body_LR.get_angle() != self.body_LR.max_angle and
                   self.head_LR.get_angle() != self.head_LR.max_angle or self.scan_success is True):
                sleep(.2)
        if self.scan_success is True:
            with self.lock:
                self.seen_data = self.eyes.get_results()
            self.scan_success = False
            self.move_servos()
        self.logger.debug("Scanning For Target Complete")

    def stop_body(self):
        """
        # Stop body movement
        """
        self.stop = True

    def __find_person(self, target='person', confidence=.7) -> dict:
        """
        # Find the highest confidence person and return their bounding box from current data set
        # self.seen_data expected to be YOLO8 data response object
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
                    if highest_confidence >= confidence:
                        # take the highest confidence and return the bounding box
                        rtn = highest_confidence_person['box']
        self.logger.debug(f"Confidence box found {rtn} with confidence score of {confidence}")
        return rtn

    def __calc_servo(self, servo: Gservo, bbox: dict) -> int:
        """
        # Calculate servo angle correction to target
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
        self.logger.debug(f"Leveling Servos {servo1.location} & {servo2.location}")
        if servo1.axis != servo2.axis:
            msg = "Servers are not on same axsis"
            self.logger.error(msg)
            raise Exception(msg)
        servo2.set_angle(servo1.get_angle())
        servo1.set_angle(servo1.get_middle_angle())
        servo1.move()
        servo2.move()

    def __distance_check(self, servo, new_angle, degree_diff=2):
        # TODO get degrees of difference from config file
        move = False
        current_angle = servo.get_angle()
        if new_angle > current_angle:
            if (new_angle - current_angle) > degree_diff:
                self.logger.debug(f"Going up, {new_angle} is greater than current {current_angle}, moving")
                move = True
            else:
                self.logger.debug(f"Going up, {new_angle} is less than current {current_angle}, not moving")
        elif new_angle < current_angle:
            if (current_angle - new_angle) > degree_diff:
                self.logger.debug(f"Going Down, {new_angle} is less than current {current_angle}, moving")
                move = True
            else:
                self.logger.debug(f"Going Down, {new_angle} is more than current {current_angle}, not moving")
        return move

    def move_servos(self):
        target = self.__find_person()
        if target != {}:
            # move "shoulders" first
            head_lr = self.__calc_servo(self.head_LR, target)
            head_ud = self.__calc_servo(self.head_UD, target)
            if self.__distance_check(self.head_LR, head_lr ) is True:
                self.head_LR.set_angle(head_lr)
                # don't use threading for now
                self.head_LR.move()
            if self.__distance_check(self.head_UD, head_ud) is True:
                self.head_UD.set_angle(head_ud)
                # dont use threading for now
                self.head_UD.move()
            # head should now be centered on the target
            # level the head and arm with body and rotation
            # x-axis
            self.__level_servos(self.head_LR, self.body_LR)
            self.__level_servos(self.head_UD, self.body_UD)

    def run(self):
        while self.stop is False:
            with self.lock:
                self.seen_data = self.eyes.get_results()
            self.move_servos()
            sleep(.2)

"""
