# native
# native
from threading import Thread
from json import loads
from time import sleep

# 3rd party
import numpy as np
from ikpy.chain import Chain
from typing import List

# glados
from glados_modules.MqttClient import ServoMessageBuilder, MQTTClient
from glados_modules.GlogConfig import setup_logger
from glados_modules.GLaDosEnums import LoggingEnums, TrackingEnums
from glados_modules.GladosData import VisionTracker, SensorTracker, VisionResultsEnum

# debug
import man_tracker

class IKTracker(MQTTClient, Thread):

    def __init__(self):
        self.__name__ = self.__class__.__name__
        Thread.__init__(self)
        self.daemon = True
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        mqtt_broker = self.broker_tuple("192.168.1.39", 1883)
        MQTTClient.__init__(self, ip=mqtt_broker.ip, port=mqtt_broker.port)
        self.topic = "body/servo"
        self.target = "person"
        cmd_topic = TrackingEnums.MQTT_COMMAND_TOPIC.value
        self.confidence = 0.65
        self.debug_tracker = man_tracker.MotionTrackSimulator(cam_x=640, cam_y=480)
        self.SMB = {1: ServoMessageBuilder.body_left_right,
                    2: ServoMessageBuilder.body_up_down,
                    3: ServoMessageBuilder.head_left_right,
                    4: ServoMessageBuilder.head_up_down}
        # Default initial guess: note that index 0 is the fixed base, so we set it to 0.0.
        self.last_guess = [0.0, 1.5708, 1.5708, 1.6057, 1.44862]
        self.cmd_trigger: str = TrackingEnums.MSG_COMMAND_KEY.value
        self.topic_handler = {cmd_topic: self.handle_cmd}
        self.vision_tracker = VisionTracker(broker=mqtt_broker, target=self.target,
                                            confidence=self.confidence, tracker_callback=self.track_loop)

    def handle_cmd(self, msg) -> None:
        """
        Handle incoming command messages from MQTT. If a START command is received,
        trigger the tracking loop for the specified camera.
        """
        j_msg = loads(msg.payload.decode())
        print(f"*** TRACKING FIRED", {self.cmd_trigger}, {TrackingEnums.MSG_COMMAND_START.value})
        print(j_msg, j_msg.get(self.cmd_trigger))
        if j_msg.get(self.cmd_trigger, "") == TrackingEnums.MSG_COMMAND_START.value:
            self.logger.debug(f"Tracking Command Received, {msg.topic}, {j_msg}")
            trigger_camera = j_msg.get(TrackingEnums.MSG_CAMERA_KEY.value, "")
            self.logger.debug(f"Running Track loop for {trigger_camera}")
            self.track_loop(trigger_camera)
            self.logger.debug(f"Tracking complete for {trigger_camera}")

    @staticmethod
    def mirror_calc(servo_angle: int) -> int:
        """
        Mirror the given servo angle around the midpoint (usually 90 or 180).
        Useful for servos that need to align in opposite directions.
        """
        return 180 - servo_angle

    @staticmethod
    def get_bbox_center(bbox: dict) -> (float, float):
        """
        Calculate the center of a bounding box.
        """
        x_center = (bbox["x1"] + bbox["x2"]) / 2.0
        y_center = (bbox["y1"] + bbox["y2"]) / 2.0
        return x_center, y_center

    @staticmethod
    def map_to_world(x_center: float, y_center: float, fixed_depth: float = 0.5) -> np.ndarray:
        """
        Convert the 2D bounding box center (in pixel coordinates) into a 3D target point
        using a pinhole camera model.

        Assumptions:
          - Resolution: 640 x 480
          - Horizontal FOV: 54°
          - Optical center: (320, 240)
          - fixed_depth: provided depth (in meters)
        """
        width, height = 640, 480
        horizontal_fov_deg = 54.0

        cx, cy = width / 2.0, height / 2.0
        # Compute horizontal focal length in pixels.
        fx = cx / np.tan(np.radians(horizontal_fov_deg / 2.0))
        # Compute vertical FOV from fx and image height.
        vertical_fov_rad = 2 * np.arctan(cy / fx)
        fy = cy / np.tan(vertical_fov_rad / 2.0)

        # Convert pixel offsets to world offsets at the given depth.
        x_world = (x_center - cx) * (fixed_depth / fx)
        y_world = (y_center - cy) * (fixed_depth / fy)
        z_world = fixed_depth
        return np.array([x_world, y_world, z_world])

    @staticmethod
    def compute_joint_angles(chain: Chain, target_point: np.ndarray, initial_position: List[float] = None) -> np.ndarray:
        """
        Compute the joint angles needed to position the robot's end-effector at the target point.
        Provide an initial guess that is within the joint limits to avoid errors.
        """
        if initial_position is None:
            # For the fixed base, use 0.0. For the active joints, choose values within their bounds.
            initial_position = [0.0, 0.0, 0.0, 1.5, 1.14]
        # You can also experiment with passing a full homogeneous transformation instead:
        # target_transform = np.eye(4)
        # target_transform[:3, 3] = target_point
        # joint_angles = chain.inverse_kinematics(target_transform, initial_position=initial_position)
        joint_angles = chain.inverse_kinematics(target_point, initial_position=initial_position)
        return joint_angles

    def send_servo_commands(self, joint_angles: np.ndarray) -> None:
        """
        Convert joint angles (in radians) to degrees and send servo commands.
        """
        next_guess = []
        mqttlist = []
        for i, angle in enumerate(joint_angles):
            next_guess.append(angle)
            angle_degrees = np.degrees(angle)
            if i == 5:
                angle_degrees = self.mirror_calc(angle_degrees)
            elif i == 0:
                # Skip the fixed base
                continue
            mqttlist.append(self.SMB[i](angle=angle_degrees, speed=1))
            print(f"Servo {i}: {angle_degrees:.2f} degrees, Radian {angle}")
        self.last_guess = next_guess
        self.send_command(mqttlist, self.topic)

    def __find_target(self, seen_data: dict) -> dict:
        """
        From the detection data, return the bounding box of the target with the highest confidence.
        """
        confidence = VisionResultsEnum.VISION_RESULTS_CONFIDENCE_KEY.value
        rtn = {}
        highest_confidence = 0
        for p in seen_data:
            if p[confidence] > highest_confidence:
                highest_confidence = p[confidence]
                rtn = p
        self.logger.debug(f"Confidence box found {rtn} with confidence score of {highest_confidence}")
        return rtn

    def find_target(self, vision_map: dict, camera: str):
        """
        For now, only supports head camera detection.
        """
        target_bounding = {}
        count = VisionResultsEnum.VISION_RESULTS_COUNT_KEY.value
        objects = VisionResultsEnum.VISION_RESULTS_OBJECTS_KEY.value
        confidence = VisionResultsEnum.VISION_RESULTS_CONFIDENCE_KEY.value
        if camera in vision_map.keys() and camera == TrackingEnums.BODY_HEAD_CAMERA.value:
            if self.target in vision_map[camera]:
                if vision_map[camera][self.target].get(count) != 0:
                    best_target = self.__find_target((vision_map[camera][self.target][objects]))
                    target_ts = vision_map[camera][self.target][objects]
                    target_bounding = best_target[TrackingEnums.KEY_BOX.value]
                    target_bounding[TrackingEnums.KEY_CONFIDENCE.value] = best_target[confidence]
                    self.logger.debug(
                        f"Ready to move all servos for target {self.target} "
                        f"message time stamp {target_ts} for {camera}"
                    )
        return target_bounding

    def track_loop(self, camera) -> None:
        """
        Main tracking loop:
          1. Load the robot's chain from the URDF.
          2. Find the target bounding box from vision data.
          3. Compute the 2D center of the bounding box.
          4. Map the 2D center to a 3D target point.
          5. Apply an extrinsic transformation to account for the camera offset (38.36mm ahead of the head_up_down joint).
          6. Compute joint angles with IK and send servo commands.
        """
        print("track loop fired")
        urdf_path = "GLaDOS.urdf"  # Replace with your URDF file path
        robot_chain: Chain = Chain.from_urdf_file(urdf_path, base_elements=["ceiling_link"])
        # Get the target bounding box from vision data.
        person_bbox = self.find_target(vision_map=self.vision_tracker.get_vision_map(), camera=camera)
        print("Detected bounding box:", person_bbox)
        # Use a simulator to test servo movements (if needed)
        self.debug_tracker.move_all_servos(person_bbox)
        # Compute the center of the bounding box
        x_center, y_center = self.get_bbox_center(person_bbox)
        # Map the 2D center from the camera image to a 3D target point (in camera coordinates)
        target_point = self.map_to_world(x_center, y_center, fixed_depth=0.5)
        print("Target point in camera coordinates:", target_point)
        # Apply extrinsic transformation: the camera is 38.36mm (0.03836 m) in front of the head_up_down joint.
        offset = np.array([0, 0, 0.03836])
        target_point_head = target_point + offset
        print("Target point in head joint coordinates:", target_point_head)
        # Set the active links mask; ensure the fixed base is inactive.
        robot_chain.active_links_mask = [False, True, True, True, True]
        # Compute joint angles using the updated target point and the last guess.
        joint_angles = self.compute_joint_angles(robot_chain, target_point_head, initial_position=self.last_guess)
        print("Calculated joint angles:", joint_angles)
        # Send servo commands based on the computed joint angles.
        self.send_servo_commands(joint_angles)

    def run(self):
        print("starting")
        while True:
            print("in loop")
            sleep(10)
            print("loop done")

if __name__ == "__main__":
    ik = IKTracker()
    ik.start()
    while True:
        sleep(5)
