import numpy as np


class KalmanFilter2D:
    """
    A simple 2D Kalman filter for tracking an object's center position and velocity.
    The state vector is [x, y, vx, vy]^T.
    """
    def __init__(self, dt=0.1):
        self.dt = dt  # time step

        # State vector: [x, y, vx, vy]^T
        self.x = np.zeros((4, 1))

        # State transition matrix (F)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])

        # Measurement matrix (H) - we only measure [x, y]
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Process noise covariance (Q)
        self.Q = np.eye(4) * 0.01

        # Measurement noise covariance (R)
        self.R = np.eye(2) * 5.0

        # Error covariance matrix (P)
        self.P = np.eye(4) * 500.0

    def predict(self):
        """ Predict the next state. """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        """ Update the state with a new measurement z (shape: 2x1). """
        y = z - np.dot(self.H, self.x)                   # Innovation or measurement residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Innovation covariance
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))  # Kalman gain
        self.x = self.x + np.dot(K, y)                     # Updated state estimate
        I = np.eye(self.F.shape[0])
        self.P = (I - np.dot(K, self.H)) @ self.P          # Updated error covariance
        return self.x
