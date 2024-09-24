import numpy
import numpy as np
from collections import namedtuple
from typing import NamedTuple


class KalmanFilter:
    def __init__(self, bbox: dict) -> None:
        dt = 0.25  # Time interval (seconds) between frames per camera
        self.center = namedtuple("center", ['x', 'y'])
        self.init_center = self.__calc_center(bbox)
        # State Transition Matrix (F)
        self.F = np.array([[1, 0, dt,  0],
                           [0, 1,  0, dt],
                           [0, 0,  1,  0],
                           [0, 0,  0,  1]])
        # Observation Matrix (H)
        self.H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        # Process Noise Covariance (Q)
        q = 1.0
        self.Q = q * np.array([[dt**4/4, 0, dt**3/2, 0],
                               [0, dt**4/4, 0, dt**3/2],
                               [dt**3/2, 0, dt**2,0],
                               [0, dt**3/2, 0, dt**2]])
        # Measurement Noise Covariance (R)
        r = 10.0
        self.R = r * np.eye(2)
        # Initial State Estimate (x)
        self.x = np.array([[self.init_center.x],
                          [self.init_center.y],
                          [0],  # Initial velocity_x
                          [0]])  # Initial velocity_y
        # Initial Estimate Covariance (P)
        self.P = np.eye(4)
        # Identity Matrix (I)
        self.I = np.eye(4)

    def __kalman_filter(self, z_measured: numpy.ndarray) -> numpy.ndarray:
        # Prediction Step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        # Update Step
        y = z_measured - (self.H @ x_pred)  # Innovation
        S = self.H @ P_pred @ self.H.T + self.R       # Innovation Covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman Gain
        self.x = x_pred + K @ y              # Updated State Estimate
        self.P = (self.I - K @ self.H) @ P_pred        # Updated Estimate Covariance
        # Extract position estimates
        estimated_position = self.x[:2]
        return estimated_position.flatten()

    def __calc_center(self, box: dict) -> tuple:
        """
        Calculate the center of a bounding box
        """
        # box['x1'], box['x2'], box['y1'], box['y2']
        x1, x2, y1, y2 = box['x1'], box['x2'], box['y1'], box['y2']
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        return self.center(bbox_center_x, bbox_center_y)

    def get_estimated_position(self, bbox: dict) -> numpy.ndarray:
        center = self.__calc_center(box=bbox)
        z_measured = np.array([[center.x, center.y]])
        return self.__kalman_filter(z_measured=z_measured)


if __name__ == "__main__":
    box = {'x1': 413.74338, 'y1': 69.39278, 'x2': 638.90991, 'y2': 479.07965}
    print("creating filter")
    kf = KalmanFilter(bbox=box)
    print("getting estimated position")
    ep = kf.get_estimated_position(bbox={'x1': 396.42682, 'y1': 1.88287, 'x2': 638.69537, 'y2': 478.90295})
    print(f"actual center X {kf.init_center.x}, Y {kf.init_center.y}")
    print(f"predicted center, X {ep[0]}, Y {ep[1]}")
    ep = kf.get_estimated_position(bbox={'x1': 396.42682, 'y1': 1.88287, 'x2': 638.69537, 'y2': 478.90295})
    print(f"actual center X {kf.init_center.x}, Y {kf.init_center.y}")
    print(f"predicted center, X {ep[0]}, Y {ep[1]}")




