from math import radians, tan, atan, degrees


# Dummy Enums for our simulation
class ServoEnum:
    X_AXIS = "x"
    LOCATION_HEAD_LEFT_RIGHT = "head_lr"
    LOCATION_HEAD_UP_DOWN = "head_ud"
    LOCATION_BODY_LEFT_RIGHT = "body_lr"
    LOCATION_BODY_UP_DOWN = "body_ud"


class CameraEnum:
    CAMERA_HEAD = "HEAD"
    CAMERA_LEFT = "LEFT"
    CAMERA_RIGHT = "RIGHT"
    CAMERA_HEAD_FOV_X = 54.0
    CAMERA_HEAD_FOV_Y = 54.0
    CAMERA_LEFT_FOV = 60.0
    CAMERA_RIGHT_FOV = 60.0


# A simple dummy servo class
class DummyServo:
    def __init__(self, name, current, min_angle, max_angle, middle, axis, location):
        self.name = name
        self.current = current
        self.min = min_angle
        self.max = max_angle
        self.middle = middle
        self.axis = axis
        self.location = location

    def move(self, angle):
        # Instead of sending a servo command, just return the new angle.
        return angle


# A stripped-down version of the MotionTrack class
class MotionTrackSimulator:
    def __init__(self, cam_x: int, cam_y: int, dead_zone_factor: int = 3):
        # Set the camera resolution
        self.cam_x = cam_x
        self.cam_y = cam_y
        self.dead_zone_factor = dead_zone_factor

        # Create dummy servo objects with typical ranges
        self.head_LR = DummyServo("head_lr", current=90, min_angle=0, max_angle=180, middle=90,
                                  axis=ServoEnum.X_AXIS, location=ServoEnum.LOCATION_HEAD_LEFT_RIGHT)
        self.head_UD = DummyServo("head_ud", current=90, min_angle=0, max_angle=180, middle=90,
                                  axis="y", location=ServoEnum.LOCATION_HEAD_UP_DOWN)
        self.body_LR = DummyServo("body_lr", current=90, min_angle=0, max_angle=180, middle=90,
                                  axis=ServoEnum.X_AXIS, location=ServoEnum.LOCATION_BODY_LEFT_RIGHT)
        self.body_UD = DummyServo("body_ud", current=90, min_angle=0, max_angle=180, middle=90,
                                  axis="y", location=ServoEnum.LOCATION_BODY_UP_DOWN)

        # Dictionary for quick lookup
        self.servos = {
            "head_lr": self.head_LR,
            "head_ud": self.head_UD,
            "body_lr": self.body_LR,
            "body_ud": self.body_UD,
        }

    def __calc_servo(self, servo: DummyServo, bbox: dict, camera: str, point: bool = False) -> int:
        # Determine axis size (using cam resolution)
        axis_size = float(self.cam_x) if servo.axis == ServoEnum.X_AXIS else float(self.cam_y)

        # Calculate the target’s center on this axis.
        if not point:
            if servo.axis == ServoEnum.X_AXIS:
                center_of_bbox = (bbox['x1'] + bbox['x2']) / 2
            else:
                center_of_bbox = (bbox['y1'] + bbox['y2']) / 2
        else:
            center_of_bbox = bbox['x'] if servo.axis == ServoEnum.X_AXIS else bbox['y']

        # Compute the pixel offset from the image center.
        offset_from_center = (axis_size / 2) - center_of_bbox

        # Default field of view (FOV) and mounting angle.
        fov = 54.0
        mounting_angle = 0.0
        current = servo.current

        # Adjust FOV and mounting angle based on the camera.
        if camera == CameraEnum.CAMERA_HEAD:
            if servo.axis == ServoEnum.X_AXIS:
                fov = CameraEnum.CAMERA_HEAD_FOV_X
            else:
                fov = CameraEnum.CAMERA_HEAD_FOV_Y
        elif camera == CameraEnum.CAMERA_RIGHT:
            fov = CameraEnum.CAMERA_RIGHT_FOV
            if servo.axis == ServoEnum.X_AXIS and servo.location == ServoEnum.LOCATION_BODY_LEFT_RIGHT:
                mounting_angle = 55.0
                current = 90.0
        elif camera == CameraEnum.CAMERA_LEFT:
            fov = CameraEnum.CAMERA_LEFT_FOV
            if servo.axis == ServoEnum.X_AXIS and servo.location == ServoEnum.LOCATION_BODY_LEFT_RIGHT:
                mounting_angle = -55.0
                current = 90.0

        # Compute focal length from the FOV.
        fov_rad = radians(fov)
        focal_length = (axis_size / 2) / tan(fov_rad / 2)

        # Calculate the angular offset (in degrees) using arctan.
        angle_offset_rad = atan(offset_from_center / focal_length)
        angle_offset_deg = degrees(angle_offset_rad)

        # Choose the direction factor.
        direction_factor = 1 if servo.location in (ServoEnum.LOCATION_HEAD_LEFT_RIGHT,
                                                   ServoEnum.LOCATION_HEAD_UP_DOWN) else -1

        # Calculate and clamp the new servo angle.
        new_servo_angle = current + direction_factor * angle_offset_deg + mounting_angle
        new_servo_angle = max(min(new_servo_angle, servo.max), servo.min)
        return round(new_servo_angle)

    def __dead_zone_check(
            self,
            servo: DummyServo,
            new_angle: int,
            degree_diff: int = 2,
            confidence: float = 0.8,
            depth: float = 1.0
    ) -> bool:
        # For head up/down, increase the base threshold.
        if servo.location == ServoEnum.LOCATION_HEAD_UP_DOWN:
            degree_diff = 5
        dynamic_diff = (degree_diff * (1 - confidence) + 1)
        if 0.6 <= confidence <= 0.7:
            dynamic_diff += 1.5
        distance_factor = max(0.5, min(2.0, depth))
        dynamic_diff *= distance_factor
        dynamic_diff = max(dynamic_diff, 3)  # enforce a minimum threshold

        move = abs(new_angle - servo.current) > dynamic_diff
        return move

    def __level_servos(self, servo1: DummyServo, servo2: DummyServo) -> tuple:
        # Ensure servos are on the same axis.
        if servo1.axis != servo2.axis:
            raise Exception("Servos are not on the same axis")
        current = servo1.current
        # For head servos, interpolate a new body angle if within a valid range.
        if servo1.name in (ServoEnum.LOCATION_HEAD_LEFT_RIGHT, ServoEnum.LOCATION_HEAD_UP_DOWN):
            head_angle = current
            if 64 <= head_angle <= 126:
                # Piecewise linear interpolation data points: (head_angle, body_angle)
                data = [
                    (64, 30), (66, 40), (68, 50), (70, 60), (72, 70),
                    (79, 80), (83, 90), (92, 100), (97, 110), (104, 120),
                    (114, 130), (121, 140), (126, 150)
                ]
                if head_angle <= data[0][0]:
                    angle = data[0][1]
                elif head_angle >= data[-1][0]:
                    angle = data[-1][1]
                else:
                    for i in range(len(data) - 1):
                        H1, B1 = data[i]
                        H2, B2 = data[i + 1]
                        if H1 <= head_angle <= H2:
                            ratio = (head_angle - H1) / (H2 - H1)
                            angle = round(B1 + ratio * (B2 - B1))
                            break
            else:
                angle = servo2.current
        else:
            angle = current

        # Clamp the computed angle to servo2's limits.
        angle = max(min(angle, servo2.max), servo2.min)

        # Use a dead-zone check to decide on a final move.
        if abs(servo1.middle - angle) > self.dead_zone_factor:
            servo1_move = servo1.middle
            servo2_move = angle
        else:
            servo1_move = servo1.current
            servo2_move = servo2.current

        return servo1_move, servo2_move

    def move_all_servos(self, target: dict, camera: str = CameraEnum.CAMERA_HEAD, pose: bool = False) -> None:
        """
        Given a target bounding box (with keys 'x1', 'y1', 'x2', 'y2', and 'confidence'),
        calculate new servo angles and print them instead of sending move commands.
        """
        # Calculate new angles for head servos.
        head_lr = self.__calc_servo(self.servos["head_lr"], target, camera, point=pose)
        head_ud = self.__calc_servo(self.servos["head_ud"], target, camera, point=pose)

        print(f"Calculated head left/right angle: {head_lr}")
        print(f"Calculated head up/down angle: {head_ud}")

        # Determine if movement is needed based on dead zone.
        if self.__dead_zone_check(self.servos["head_ud"], head_ud, degree_diff=self.dead_zone_factor,
                                  confidence=target.get("confidence", 0.8)):
            print(f"Head up/down movement required: from {self.servos['head_ud'].current} to {head_ud}")
        else:
            print("Head up/down within dead zone; no movement needed.")

        if self.__dead_zone_check(self.servos["head_lr"], head_lr, self.dead_zone_factor):
            print(f"Head left/right movement required: from {self.servos['head_lr'].current} to {head_lr}")
        else:
            print("Head left/right within dead zone; no movement needed.")

        # Simulate body rotation (for left/right).
        body_lr = self.__calc_servo(self.servos["body_lr"], target, camera, point=pose)
        if self.__dead_zone_check(self.servos["body_lr"], body_lr, self.dead_zone_factor,
                                  confidence=target.get("confidence", 0.8)):
            print(f"Body left/right movement required: from {self.servos['body_lr'].current} to {body_lr}")
        else:
            print("Body left/right within dead zone; no movement needed.")

        # Simulate leveling for head up/down and body up/down.
        servo_head_ud_target, servo_body_ud_target = self.__level_servos(self.head_UD, self.body_UD)
        print(f"Leveling: Head up/down should be set to {servo_head_ud_target}, "
              f"and Body up/down should be set to {servo_body_ud_target}")


# Example usage:
if __name__ == '__main__':
    # Example bounding box dictionary with dummy coordinates and a confidence value.
    # (Assume a 640x480 camera resolution)
    target_bbox = {'x1': 162.36327, 'y1': 22.84308, 'x2': 638.2605, 'y2': 473.11267, 'confidence': 0.66521}
    tracker = MotionTrackSimulator(cam_x=640, cam_y=480)
    tracker.move_all_servos(target_bbox, camera=CameraEnum.CAMERA_HEAD)
