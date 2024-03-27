from adafruit_servokit import ServoKit
from time import sleep
from threading import Thread
from GLaDOSBody import gservo

class GLaDOSTracker(Thread):
    def __init__(self, servos: dict, cam_x_width: int = 640, cam_y_width: int = 640):
        Thread.__init__(self)
        Thread.daemon = True
        self.cam_x_width = cam_x_width
        self.cam_y_width = cam_y_width
        self.servos = servos

    def find_person(self, data):
        """
        Find the highest confidence person and return their bounding box
        """
        rtn = dict()
        if 'person' in data and data['person']['count'] > 0:
            highest_confidence = 0
            highest_confidence_person = None

            for person in data['person']['objects']:
                if person['confidence'] > highest_confidence:
                    highest_confidence = person['confidence']
                    print(highest_confidence)
                    highest_confidence_person = person
            
            if highest_confidence_person is not None:
                if highest_confidence >= .60:
                # take the highest confidence and return the bounding box
                    rtn = highest_confidence_person['box']
        return rtn

    def run(self):
        pass

    def calc_servo(self, width, servo_max, bbox_edge_1, bbox_edge_2, current_angle):
        """
        Calculate servo angle correction to target
        """
        servo_mid = servo_max / 2
        # Calculate the center of the new person's bounding box on the x-axis
        center_x_updated = (bbox_edge_1 + bbox_edge_2) / 2
        # Calculate the offset of the person's center from the image center with the updated data
        offset_from_center = center_x_updated - (width / 2)
        # Calculate the new servo angle to center on the person with the updated data
        new_servo_angle_updated = servo_mid - (offset_from_center / width * servo_max)
        # Round to nearest whole
        return round(new_servo_angle_updated)

if __name__ == "__main__":
    kit = ServoKit(channels=16)
    gservos = dict()
    gservos["body_rotate"] = gservo(skit=kit.servo[0], max_angle=180)
    gdtrack = GLaDOSTracker(640, 640, gservos)
    while True:
        print("Paste data")
        x = input()
        bbox = gdtrack.find_person(x)
        new_angle = gdtrack.calc_servo(640, 180, bbox['x1'], bbox['x2'], gservos["body_rotate"].get_angle())
        # move this into the lib... but do here for debug now
        gservos["body_rotate"].set_speed_angle((5, new_angle))
        gservos["body_rotate"].move()
