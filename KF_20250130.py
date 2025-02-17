# kalman_filter.py

import numpy as np
import cv2
from utils import rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix
import logging

# Configure logging
logger = logging.getLogger(__name__)

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
        # 3) Set Process Noise & Measurement Noise
        #    (You can fine-tune these values.)

        #Original KF
        # kf.processNoiseCov = np.eye(self.n_states, dtype=np.float64) * 1e-5
        #kf.measurementNoiseCov = np.eye(self.n_measurements, dtype=np.float64) * 1e-4


        kf.processNoiseCov = np.eye(self.n_states, dtype=np.float64)
        # Typically, orientation changes can be a bit noisier or uncertain,
        # so you might allow more process noise for orientation states:
        kf.processNoiseCov[0:9, 0:9]   *= 1e-5   # translation (pos, vel, acc)
        kf.processNoiseCov[9:18, 9:18] *= 1e-4   # orientation (roll, pitch, yaw, etc.)

        # If your measurements are relatively accurate, keep measurement noise
        # modest. If your orientation or position estimates are quite noisy,
        # increase this.
        kf.measurementNoiseCov = np.eye(self.n_measurements, dtype=np.float64) * 1e-3

        # 4) Initialize the state vector
        #    State order: (x, y, z, vx, vy, vz, ax, ay, az, roll, pitch, yaw, ...)
        kf.statePost = np.zeros((self.n_states, 1), dtype=np.float64)

        # If you know approximate position, set it (e.g., x=0, y=0, z=5):
        # kf.statePost[0, 0] = 0.0   # x
        # kf.statePost[1, 0] = 0.0   # y
        # kf.statePost[2, 0] = 5.0   # z

        ## Insert the known initial Euler angles (in radians)
        #roll_init  = np.radians(18.0)
        #pitch_init = np.radians(0.0)
        #yaw_init   = np.radians(-30.0)

        #kf.statePost[9,  0] = roll_init   # roll
        #kf.statePost[10, 0] = pitch_init  # pitch
        #kf.statePost[11, 0] = yaw_init    # yaw

        # 5) Initialize error covariance (P matrix)
        # Start with large uncertainty in everything but orientation:
        kf.errorCovPost = np.eye(self.n_states, dtype=np.float64)

        # For position & velocity & acceleration, we’re uncertain => bigger values
        kf.errorCovPost[0:9, 0:9] *= 1e3

        # For orientation (roll, pitch, yaw), we’re more certain => smaller values
        # If you want to be absolutely sure, use e.g. 1e-2 or 1e-1:
        kf.errorCovPost[9:12, 9:12] *= 1e-2

        # For orientation rates & acceleration, if you don't know them, keep bigger
        kf.errorCovPost[12:18, 12:18] *= 1e2

        # 6) Prepare a 6D measurement placeholder
        self.measurement = np.zeros((self.n_measurements, 1), dtype=np.float64)
    
    def predict(self):
        predicted = self.kf.predict()
        translation_estimated = predicted[0:3].flatten()
        eulers_estimated = predicted[9:12].flatten()
        logger.debug(f"KF Predicted State: {self.kf.statePre.flatten()}")
        return translation_estimated, eulers_estimated
    
    def correct(self, tvec, R):
        eulers_measured = rotation_matrix_to_euler_angles(R)
        self.measurement[0:3, 0] = tvec.flatten()
        self.measurement[3:6, 0] = eulers_measured
        self.kf.correct(self.measurement)
        logger.debug(f"KF Corrected State: {self.kf.statePost.flatten()}")

