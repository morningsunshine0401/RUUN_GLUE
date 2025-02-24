import numpy as np
import cv2
from utils import rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix
import logging

import json

logger = logging.getLogger(__name__)

class KalmanFilterPose:
    def __init__(self, dt, n_states=18, n_measurements=6):
        """
        n_states=18, n_measurements=6:
          States (x,y,z, vx,vy,vz, ax,ay,az,
                  roll,pitch,yaw, vroll,vpitch,vyaw, aroll,apitch,ayaw).

          Measurements are (x,y,z, roll,pitch,yaw).
        """
        self.dt = dt
        self.n_states = n_states
        self.n_measurements = n_measurements
        self.kf = cv2.KalmanFilter(n_states, n_measurements, 0, cv2.CV_64F)
        # List to collect debug information for JSON output
        self.debug_info = []
        self._init_kalman_filter()
        

    def _init_kalman_filter(self):
        kf = self.kf
        dt = self.dt
        
        # 1) Transition Matrix (18x18)
        kf.transitionMatrix = np.eye(self.n_states, dtype=np.float64)

        # Indices:
        #  pos: 0,1,2 ; vel: 3,4,5 ; accel: 6,7,8
        #  roll=9, pitch=10, yaw=11
        #  vroll=12, vpitch=13, vyaw=14
        #  aroll=15, apitch=16, ayaw=17
        #
        # Position <-> velocity <-> accel
        for i in range(3):
            kf.transitionMatrix[i, i+3] = dt              # pos -> vel
            kf.transitionMatrix[i, i+6] = 0.5 * (dt**2)   # pos -> accel
            kf.transitionMatrix[i+3, i+6] = dt            # vel -> accel

        # Orientation <-> angular velocity <-> angular accel
        for i in range(9, 12):
            kf.transitionMatrix[i, i+3] = dt
            kf.transitionMatrix[i, i+6] = 0.5 * (dt**2)
            kf.transitionMatrix[i+3, i+6] = dt

        # 2) Measurement Matrix (6x18)
        # Measuring x,y,z and roll,pitch,yaw directly
        kf.measurementMatrix = np.zeros((self.n_measurements, self.n_states), dtype=np.float64)
        # x, y, z
        kf.measurementMatrix[0, 0] = 1.0
        kf.measurementMatrix[1, 1] = 1.0
        kf.measurementMatrix[2, 2] = 1.0
        # roll, pitch, yaw
        kf.measurementMatrix[3, 9]  = 1.0
        kf.measurementMatrix[4, 10] = 1.0
        kf.measurementMatrix[5, 11] = 1.0

        # 3) Covariance
        kf.processNoiseCov = np.eye(self.n_states, dtype=np.float64) * 1e-7 #* 1e-5
        kf.measurementNoiseCov = np.eye(self.n_measurements, dtype=np.float64) * 1e-2 #* 1e-4
        kf.errorCovPost = np.eye(self.n_states, dtype=np.float64) * 0.1

        # 4) Initial State
        kf.statePost = np.zeros((self.n_states, 1), dtype=np.float64)
        self.measurement = np.zeros((self.n_measurements, 1), dtype=np.float64)

        # Save initialization debug info
        init_debug = {
            "step": "init",
            "transitionMatrix": kf.transitionMatrix.tolist(),
            "measurementMatrix": kf.measurementMatrix.tolist(),
            "processNoiseCov": kf.processNoiseCov.tolist(),
            "measurementNoiseCov": kf.measurementNoiseCov.tolist(),
            "errorCovPost": kf.errorCovPost.tolist(),
            "statePost": kf.statePost.flatten().tolist()
        }
        self.debug_info.append(init_debug)
        logger.debug("Initialized Kalman Filter")

    def predict(self,frame=None):
        logger.debug("Predict step: Starting statePost: %s", self.kf.statePost.flatten())
        logger.debug("Predict step: Using Transition Matrix:\n%s", self.kf.transitionMatrix)
        predicted = self.kf.predict()
        # Log predicted state and error covariance
        logger.debug("Predict step: Predicted state (statePre): %s", self.kf.statePre.flatten())
        logger.debug("Predict step: Predicted error covariance (errorCovPre):\n%s", self.kf.errorCovPre)
        # translation = x[0:3], orientation = x[9:12]
        translation_estimated = predicted[0:3].flatten()      # x, y, z
        velocity_estimated    = predicted[3:6].flatten()      # linear velocity
        lin_acc              = predicted[6:9].flatten()       # linear acceleration
        eulers_estimated     = predicted[9:12].flatten()      # roll, pitch, yaw
        angular_velocity     = predicted[12:15].flatten()     # angular velocity
        ang_acc             = predicted[15:18].flatten()      # angular acceleration
        
        logger.debug("Predict step: Velocity estimated: %s", velocity_estimated)
        logger.debug("Predict step: Linear acceleration: %s", lin_acc)
        logger.debug("Predict step: Angular velocity: %s", angular_velocity)
        logger.debug("Predict step: Angular acceleration: %s", ang_acc)
        
        print("Predicted velocity:\n", velocity_estimated)
        print("Predicted angular velocity:\n", angular_velocity)
        print("Predicted linear acceleration:\n", lin_acc)
        print("Predicted angular acceleration:\n", ang_acc)
        
        # Save prediction debug info
        predict_debug = {
            "step": "predict",
            "did_correct": False,  # Indicate no correction in this step
            "statePre": self.kf.statePre.flatten().tolist(),
            "errorCovPre": self.kf.errorCovPre.tolist(),
            "translation_estimated": translation_estimated.tolist(),
            "velocity_estimated": velocity_estimated.tolist(),
            "linear_acceleration": lin_acc.tolist(),
            "eulers_estimated": eulers_estimated.tolist(),
            "angular_velocity": angular_velocity.tolist(),
            "angular_acceleration": ang_acc.tolist()
        }
        if frame is not None:
            predict_debug["frame"] = frame
        self.debug_info.append(predict_debug)

        return translation_estimated, eulers_estimated

    def correct(self, tvec, R, frame=None):
        """
        Standard Kalman correct with measurement = (x, y, z, roll, pitch, yaw).
        """
        eulers_measured = rotation_matrix_to_euler_angles(R)
        self.measurement[0:3, 0] = tvec.flatten()
        self.measurement[3:6, 0] = eulers_measured

        logger.debug("Correct step: Measurement vector: %s", self.measurement.flatten())
        logger.debug("Correct step: Using Measurement Matrix:\n%s", self.kf.measurementMatrix)
        logger.debug("Correct step: Error covariance before correction (errorCovPre):\n%s",
                     self.kf.errorCovPre)
        
        # Calculate the Kalman Gain manually for logging
        P = self.kf.errorCovPre
        H = self.kf.measurementMatrix
        R_meas = self.kf.measurementNoiseCov
        S = H @ P @ H.T + R_meas  # Innovation covariance
        K = P @ H.T @ np.linalg.inv(S)
        
        # Compute the innovation (residual): measurement - predicted_measurement
        zPred = H @ self.kf.statePre
        residual = self.measurement - zPred
        residual_norm = float(np.linalg.norm(residual))  # convert to float for JSON
        
        logger.debug("Computed Kalman Gain:\n%s", K)
        logger.debug("Innovation Covariance S:\n%s", S)
        logger.debug("Error Covariance Pre (P):\n%s", P)
        logger.debug("Innovation (residual): %s", residual.flatten())
        logger.debug("Innovation norm: %f", residual_norm)
        
        correct_debug_before = {
            "step": "correct_before",
            "did_correct": True,  # Indicate that we do a correction here
            "measurement_vector": self.measurement.flatten().tolist(),
            "innovation": residual.flatten().tolist(),
            "innovation_norm": residual_norm,
            "innovation_covariance": S.tolist(),
            "kalman_gain": K.tolist(),
            "errorCovPre": P.tolist()
        }
        if frame is not None:
            correct_debug_before["frame"] = frame
        self.debug_info.append(correct_debug_before)
        

        self.kf.correct(self.measurement)
        logger.debug("Correct step: Updated statePost: %s", self.kf.statePost.flatten())
        logger.debug("Correct step: Updated error covariance (errorCovPost):\n%s",
                     self.kf.errorCovPost)
        print("Correct step: Updated state:\n", self.kf.statePost.flatten())
        
        correct_debug_after = {
            "step": "correct_after",
            "did_correct": True,  # still part of correction
            "statePost": self.kf.statePost.flatten().tolist(),
            "errorCovPost": self.kf.errorCovPost.tolist()
        }
        if frame is not None:
            correct_debug_after["frame"] = frame
        self.debug_info.append(correct_debug_after)
    

    def get_covariance(self):
        """Return the full 18x18 error covariance matrix."""
        return self.kf.errorCovPost.copy()

    def get_state(self):
        """
        Return the full 18x1 state vector (for debugging or advanced gating).
        Indices:
         0: x, 1: y, 2: z,
         3: vx, 4: vy, 5: vz,
         6: ax, 7: ay, 8: az,
         9: roll,10:pitch,11:yaw,
         12:vroll,13:vpitch,14:vyaw,
         15:aroll,16:apitch,17:ayaw
        """
        return self.kf.statePost.copy()

    def correct_partial(self, tvec, R, alpha):
        """
        Basic partial blend of position + orientation with factor alpha in [0..1].
        Leaves velocity/acc as-is, then inflates covariance.
        """
        current_state = self.kf.statePost.copy()  # shape (18,1)
        eulers_measured = rotation_matrix_to_euler_angles(R)

        # Position
        current_state[0, 0] = (1 - alpha)*current_state[0, 0] + alpha*tvec[0]
        current_state[1, 0] = (1 - alpha)*current_state[1, 0] + alpha*tvec[1]
        current_state[2, 0] = (1 - alpha)*current_state[2, 0] + alpha*tvec[2]

        # Orientation
        current_state[9, 0]  = (1 - alpha)*current_state[9, 0]  + alpha*eulers_measured[0]
        current_state[10, 0] = (1 - alpha)*current_state[10, 0] + alpha*eulers_measured[1]
        current_state[11, 0] = (1 - alpha)*current_state[11, 0] + alpha*eulers_measured[2]

        self.kf.statePost = current_state

        # Inflate covariance by 1/alpha
        inflation_factor = 1.0 / alpha
        self.kf.errorCovPost *= inflation_factor

        logger.debug("Partial blend update done.")
        logger.debug(f"New statePost: {self.kf.statePost.flatten()}")

    def correct_partial_separate(self, tvec, R, alpha_pos=0.8, alpha_rot=0.3):
        """
        Partial-blend position vs. orientation with different alpha factors.
        Leaves velocity/acc as-is, then inflates covariance by 1/min(alpha_pos, alpha_rot).
        """
        current_state = self.kf.statePost.copy()  # shape (18,1)
        eulers_measured = rotation_matrix_to_euler_angles(R)

        # Blend position
        current_state[0, 0] = (1 - alpha_pos)*current_state[0, 0] + alpha_pos*tvec[0]
        current_state[1, 0] = (1 - alpha_pos)*current_state[1, 0] + alpha_pos*tvec[1]
        current_state[2, 0] = (1 - alpha_pos)*current_state[2, 0] + alpha_pos*tvec[2]

        # Blend orientation
        current_state[9, 0]  = (1 - alpha_rot)*current_state[9, 0]  + alpha_rot*eulers_measured[0]
        current_state[10, 0] = (1 - alpha_rot)*current_state[10, 0] + alpha_rot*eulers_measured[1]
        current_state[11, 0] = (1 - alpha_rot)*current_state[11, 0] + alpha_rot*eulers_measured[2]

        self.kf.statePost = current_state

        # Inflate covariance by 1/min(alpha_pos, alpha_rot)
        inflation_factor = 1.0 / min(alpha_pos, alpha_rot)
        self.kf.errorCovPost *= inflation_factor

        logger.debug("Partial separate blend update done.")
        logger.debug(f"New statePost: {self.kf.statePost.flatten()}")


    def save_debug_info(self, filename="kalman_debug.json"):
        """Save the collected debug info to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.debug_info, f, indent=4)
