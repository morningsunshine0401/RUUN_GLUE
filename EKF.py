# kalman_filter.py

import numpy as np
import cv2
from utils import rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix

class KalmanFilterPose:
    def __init__(self, dt, n_states=18, n_measurements=6):
        self.dt = dt
        self.n_states = n_states
        self.n_measurements = n_measurements
        self.kf = cv2.KalmanFilter(n_states, n_measurements, 0, cv2.CV_64F)
        self._init_kalman_filter()

    def _init_kalman_filter(self):
        kf = self.kf
        dt = self.dt
        kf.transitionMatrix = np.eye(self.n_states, dtype=np.float64)
        for i in range(3):
            kf.transitionMatrix[i, i+3] = dt
            kf.transitionMatrix[i, i+6] = 0.5 * dt**2
            kf.transitionMatrix[i+3, i+6] = dt
        for i in range(9, 12):
            kf.transitionMatrix[i, i+3] = dt
            kf.transitionMatrix[i, i+6] = 0.5 * dt**2
            kf.transitionMatrix[i+3, i+6] = dt
        kf.measurementMatrix = np.zeros((self.n_measurements, self.n_states), dtype=np.float64)
        kf.measurementMatrix[0, 0] = 1.0  # x
        kf.measurementMatrix[1, 1] = 1.0  # y
        kf.measurementMatrix[2, 2] = 1.0  # z
        kf.measurementMatrix[3, 9] = 1.0  # roll
        kf.measurementMatrix[4, 10] = 1.0  # pitch
        kf.measurementMatrix[5, 11] = 1.0  # yaw
        kf.processNoiseCov = np.eye(self.n_states, dtype=np.float64) * 1e-5
        kf.measurementNoiseCov = np.eye(self.n_measurements, dtype=np.float64) * 1e-4
        kf.errorCovPost = np.eye(self.n_states, dtype=np.float64)
        kf.statePost = np.zeros((self.n_states, 1), dtype=np.float64)
        self.measurement = np.zeros((self.n_measurements, 1), dtype=np.float64)

    def state_transition(self, state):
        x, y, z = state[:3, 0]
        vx, vy, vz = state[3:6, 0]
        ax, ay, az = state[6:9, 0]
        dt = self.dt

        new_x = x + vx * dt + 0.5 * ax * dt**2
        new_y = y + vy * dt + 0.5 * ay * dt**2
        new_z = z + vz * dt + 0.5 * az * dt**2

        new_state = state.copy()
        new_state[0, 0] = new_x
        new_state[1, 0] = new_y
        new_state[2, 0] = new_z
        return new_state

    def measurement_model(self, state):
        x, y, z = state[:3, 0]
        roll, pitch, yaw = state[9:12, 0]
        return np.array([x, y, z, roll, pitch, yaw])

    def compute_jacobian_f(self, state):
        dt = self.dt
        jacobian = np.eye(self.n_states)
        for i in range(3):  # Position-velocity-acceleration coupling
            jacobian[i, i + 3] = dt
            jacobian[i, i + 6] = 0.5 * dt**2
            jacobian[i + 3, i + 6] = dt
        return jacobian

    def compute_jacobian_h(self, state):
        jacobian = np.zeros((self.n_measurements, self.n_states))
        jacobian[0, 0] = 1.0  # x
        jacobian[1, 1] = 1.0  # y
        jacobian[2, 2] = 1.0  # z
        jacobian[3, 9] = 1.0  # roll
        jacobian[4, 10] = 1.0  # pitch
        jacobian[5, 11] = 1.0  # yaw
        return jacobian

    def predict(self):
        # Nonlinear state transition
        self.kf.statePost = self.state_transition(self.kf.statePost)

        # Compute the Jacobian of f (partial derivatives)
        self.kf.transitionMatrix = self.compute_jacobian_f(self.kf.statePost)

        # Standard Kalman filter predict step
        predicted = self.kf.predict()
        translation_estimated = predicted[:3].flatten()
        eulers_estimated = predicted[9:12].flatten()
        return translation_estimated, eulers_estimated

    def correct(self, tvec, R):
        # Nonlinear measurement
        measured = np.zeros((self.n_measurements, 1))
        measured[:3, 0] = tvec.flatten()
        measured[3:6, 0] = rotation_matrix_to_euler_angles(R)

        # Compute the Jacobian of h
        self.kf.measurementMatrix = self.compute_jacobian_h(self.kf.statePost)

        # Correct with the standard Kalman filter correction
        self.kf.correct(measured)