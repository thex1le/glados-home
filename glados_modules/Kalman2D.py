import numpy as np

import numpy as np


class KalmanFilter2D:
    """
    A simple 2D Kalman filter for tracking an object's center position and velocity.
    The state vector is [x, y, vx, vy]^T.
    """

    def __init__(self, dt=0.1):
        self.dt = dt  # time step

        # State vector: [x, y, vx, vy]^T (initialized to zeros)
        self.x = np.zeros((4, 1))

        # State transition matrix (F)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Measurement matrix (H) - we only measure [x, y]
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Process noise covariance (Q)
        # Increased slightly to allow the filter to adapt faster when the target reverses direction.
        self.Q = np.eye(4) * 0.07

        # Measurement noise covariance (R)
        # Increased to dampen the influence of noisy measurements.
        self.R = np.eye(2) * 20.5

        # Error covariance matrix (P)
        self.P = np.eye(4) * 500.0

    def predict(self):
        """ Predict the next state. """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        """
        Update the state with a new measurement z (shape: 2x1).
        Also checks for velocity reversals and resets velocity components if a sign change is detected.
        """
        # Save previous velocity for reversal detection
        prev_vx = self.x[2, 0]
        prev_vy = self.x[3, 0]

        # Innovation or measurement residual
        y = z - np.dot(self.H, self.x)
        # Innovation covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Kalman gain
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        # Updated state estimate
        self.x = self.x + np.dot(K, y)

        # Handle reversal in velocity: if velocity sign flips, reset to zero.
        if np.sign(self.x[2, 0]) != np.sign(prev_vx):
            self.x[2, 0] = 0
        if np.sign(self.x[3, 0]) != np.sign(prev_vy):
            self.x[3, 0] = 0

        # Updated error covariance
        I = np.eye(self.F.shape[0])
        self.P = (I - np.dot(K, self.H)) @ self.P
        return self.x
