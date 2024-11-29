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

    def predict(self):
        predicted = self.kf.predict()
        translation_estimated = predicted[0:3].flatten()
        eulers_estimated = predicted[9:12].flatten()
        return translation_estimated, eulers_estimated

    def correct(self, tvec, R):
        eulers_measured = rotation_matrix_to_euler_angles(R)
        self.measurement[0:3, 0] = tvec.flatten()
        self.measurement[3:6, 0] = eulers_measured
        self.kf.correct(self.measurement)
