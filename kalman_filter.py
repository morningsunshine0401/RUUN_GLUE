# # kalman_filter.py

# import numpy as np
# import cv2
# from utils import rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix
# import logging

# # Configure logging
# logger = logging.getLogger(__name__)

####################################################################################################

# #Default

# class KalmanFilterPose:
#     def __init__(self, dt, n_states=18, n_measurements=6):
#         self.dt = dt
#         self.n_states = n_states
#         self.n_measurements = n_measurements
#         self.kf = cv2.KalmanFilter(n_states, n_measurements, 0, cv2.CV_64F)
#         self._init_kalman_filter()
    
#     def _init_kalman_filter(self):
#         kf = self.kf
#         dt = self.dt
#         kf.transitionMatrix = np.eye(self.n_states, dtype=np.float64)
#         for i in range(3):
#             kf.transitionMatrix[i, i+3] = dt
#             kf.transitionMatrix[i, i+6] = 0.5 * dt**2
#             kf.transitionMatrix[i+3, i+6] = dt
#         for i in range(9, 12):
#             kf.transitionMatrix[i, i+3] = dt
#             kf.transitionMatrix[i, i+6] = 0.5 * dt**2
#             kf.transitionMatrix[i+3, i+6] = dt
#         kf.measurementMatrix = np.zeros((self.n_measurements, self.n_states), dtype=np.float64)
#         kf.measurementMatrix[0, 0] = 1.0  # x
#         kf.measurementMatrix[1, 1] = 1.0  # y
#         kf.measurementMatrix[2, 2] = 1.0  # z
#         kf.measurementMatrix[3, 9] = 1.0  # roll
#         kf.measurementMatrix[4, 10] = 1.0  # pitch
#         kf.measurementMatrix[5, 11] = 1.0  # yaw
#         kf.processNoiseCov = np.eye(self.n_states, dtype=np.float64) * 1e-7   #* 1e-5
#         kf.measurementNoiseCov = np.eye(self.n_measurements, dtype=np.float64) * 1e-2 #* 1e-4
#         kf.errorCovPost = np.eye(self.n_states, dtype=np.float64) * 0.1 # * 1.0
#         kf.statePost = np.zeros((self.n_states, 1), dtype=np.float64)
#         self.measurement = np.zeros((self.n_measurements, 1), dtype=np.float64)
    
#     # def predict(self):
#     #     predicted = self.kf.predict()
#     #     translation_estimated = predicted[0:3].flatten()
#     #     eulers_estimated = predicted[9:12].flatten()
#     #     logger.debug(f"KF Predicted State: {self.kf.statePre.flatten()}")
#     #     return translation_estimated, eulers_estimated
    
#     def predict(self):
#         predicted = self.kf.predict()
#         translation_estimated = predicted[0:3].flatten()      # x, y, z
#         velocity_estimated    = predicted[3:6].flatten()        # linear velocity (if needed)
#         lin_acc = predicted[6:9].flatten()                    # linear acceleration
#         eulers_estimated = predicted[9:12].flatten()            # roll, pitch, yaw
#         angular_velocity = predicted[12:15].flatten()         # angular velocity (if needed)
#         ang_acc = predicted[15:18].flatten()                  # angular acceleration

#         logger.debug(f"KF Predicted State: {self.kf.statePre.flatten()}")
#         logger.debug(f"Linear Acceleration: {lin_acc}")
#         logger.debug(f"Angular Acceleration: {ang_acc}")
#         print("velocity_estimated:\n",velocity_estimated)
#         print("angular_velocity:\n",angular_velocity)
#         print("lin_acc:\n",lin_acc)
#         print("ang_acc:\n",ang_acc)
#         # Return whichever values you want to use; here we return all for example.
#         return translation_estimated, eulers_estimated

    
#     def correct(self, tvec, R):
#         eulers_measured = rotation_matrix_to_euler_angles(R)
#         self.measurement[0:3, 0] = tvec.flatten()
#         self.measurement[3:6, 0] = eulers_measured
#         self.kf.correct(self.measurement)
#         logger.debug(f"KF Corrected State: {self.kf.statePost.flatten()}")

###################################################################################3


####################################################################################################

# KalmanFilterPose with detailed debug logging
import cv2
import numpy as np
import json
from utils import rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix
import logging

logger = logging.getLogger(__name__)

class KalmanFilterPose:
    def __init__(self, dt, n_states=18, n_measurements=6):
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
        kf.transitionMatrix = np.eye(self.n_states, dtype=np.float64)
        
        # Position + velocity + accel for x, y, z
        for i in range(3):
            kf.transitionMatrix[i, i+3] = dt
            kf.transitionMatrix[i, i+6] = 0.5 * dt**2
            kf.transitionMatrix[i+3, i+6] = dt
        
        # Orientation (Euler angles) + angular velocity + angular accel
        for i in range(9, 12):
            kf.transitionMatrix[i, i+3] = dt
            kf.transitionMatrix[i, i+6] = 0.5 * dt**2
            kf.transitionMatrix[i+3, i+6] = dt
        
        # Measurement matrix: we measure x, y, z, roll, pitch, yaw
        kf.measurementMatrix = np.zeros((self.n_measurements, self.n_states), dtype=np.float64)
        kf.measurementMatrix[0, 0] = 1.0  # x
        kf.measurementMatrix[1, 1] = 1.0  # y
        kf.measurementMatrix[2, 2] = 1.0  # z
        kf.measurementMatrix[3, 9] = 1.0  # roll
        kf.measurementMatrix[4, 10] = 1.0 # pitch
        kf.measurementMatrix[5, 11] = 1.0 # yaw
        
        # Noise and covariance initialization
        kf.processNoiseCov = np.eye(self.n_states, dtype=np.float64) * 1e-7
        kf.measurementNoiseCov = np.eye(self.n_measurements, dtype=np.float64) * 1e-2
        kf.errorCovPost = np.eye(self.n_states, dtype=np.float64) * 0.1
        
        # Initial state
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
    
    def predict(self, frame=None):
        logger.debug("Predict step: Starting statePost: %s", self.kf.statePost.flatten())
        logger.debug("Predict step: Using Transition Matrix:\n%s", self.kf.transitionMatrix)
        
        # Perform prediction
        predicted = self.kf.predict()
        
        # Log predicted state and error covariance
        logger.debug("Predict step: Predicted state (statePre): %s", self.kf.statePre.flatten())
        logger.debug("Predict step: Predicted error covariance (errorCovPre):\n%s", self.kf.errorCovPre)
        
        # Extract state components
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
        # Convert rotation matrix to Euler angles
        eulers_measured = rotation_matrix_to_euler_angles(R)
        self.measurement[0:3, 0] = tvec.flatten()       # x, y, z
        self.measurement[3:6, 0] = eulers_measured      # roll, pitch, yaw
        
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
        
        # Apply the measurement update.
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
    
    def save_debug_info(self, filename="kalman_debug.json"):
        """Save the collected debug info to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.debug_info, f, indent=4)



##############################################################################




# ##This is custom code


# import numpy as np
# import json
# import logging
# from utils import rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


# class KalmanFilterPose:
#     def __init__(self, dt, n_states=18, n_measurements=6):
#         """
#         dt: time interval between measurements
#         n_states: dimension of state vector (default 18)
#         n_measurements: dimension of measurement vector (default 6)
#         """
#         self.dt = dt
#         self.n_states = n_states
#         self.n_measurements = n_measurements

#         # Initialize state, covariance, etc.
#         self._init_kalman_filter()

#         # List for collecting debug info (for later saving to JSON)
#         self.debug_info = []
#         # Log initialization info
#         self._log_init()

#     def _init_kalman_filter(self):
#         dt = self.dt
#         n = self.n_states
#         m = self.n_measurements

#         # State vector (absolute pose): [x, y, z, vx, vy, vz, ax, ay, az, roll, pitch, yaw, 
#         #                                angular_velocity_x, angular_velocity_y, angular_velocity_z,
#         #                                angular_acceleration_x, angular_acceleration_y, angular_acceleration_z]^T
#         self.x = np.zeros((n, 1), dtype=np.float64)

#         # Error covariance matrix (initial uncertainty)
#         self.P = np.eye(n, dtype=np.float64) * 0.1

#         # Transition matrix F: for the translation block (first 9 states)
#         self.F = np.eye(n, dtype=np.float64)
#         for i in range(3):
#             self.F[i, i+3] = dt            # position updated by velocity
#             self.F[i, i+6] = 0.5 * dt**2     # position updated by acceleration
#             self.F[i+3, i+6] = dt            # velocity updated by acceleration

#         # For orientation block: assume indices 9 to 17 correspond to Euler angles, angular velocity, and angular acceleration.
#         for i in range(9, 12):
#             self.F[i, i+3] = dt            # Euler angles updated by angular velocity
#             self.F[i, i+6] = 0.5 * dt**2     # Euler angles updated by angular acceleration
#             self.F[i+3, i+6] = dt            # angular velocity updated by angular acceleration

#         # Measurement matrix H: we measure only [x, y, z, roll, pitch, yaw]
#         self.H = np.zeros((m, n), dtype=np.float64)
#         self.H[0, 0] = 1.0  # x
#         self.H[1, 1] = 1.0  # y
#         self.H[2, 2] = 1.0  # z
#         self.H[3, 9] = 1.0  # roll
#         self.H[4, 10] = 1.0  # pitch
#         self.H[5, 11] = 1.0  # yaw

#         # Process noise covariance Q
#         self.Q = np.eye(n, dtype=np.float64) * 1e-7

#         # Measurement noise covariance R
#         self.R = np.eye(m, dtype=np.float64) * 1e-2

#     def _log_init(self):
#         init_info = {
#             "step": "init",
#             "F": self.F.tolist(),
#             "H": self.H.tolist(),
#             "Q": self.Q.tolist(),
#             "R": self.R.tolist(),
#             "P": self.P.tolist(),
#             "x": self.x.flatten().tolist()
#         }
#         self.debug_info.append(init_info)
#         logger.debug("Initialized custom Kalman Filter with F, H, Q, R, P, x.")

#     def predict(self, frame=None):
#         """
#         Predict the next state.
#         Optionally, provide a frame number (or identifier) to log with this prediction.
#         Returns the predicted translation (x, y, z) and Euler angles (roll, pitch, yaw).
#         """
#         # Predict state: x_pre = F * x
#         x_pre = self.F @ self.x
#         # Predict error covariance: P_pre = F * P * F^T + Q
#         P_pre = self.F @ self.P @ self.F.T + self.Q

#         # Save prediction debug info
#         pred_debug = {
#             "step": "predict",
#             "frame": frame if frame is not None else "N/A",
#             "x_pre": x_pre.flatten().tolist(),
#             "P_pre": P_pre.tolist()
#         }
#         self.debug_info.append(pred_debug)
#         logger.debug("Predict: x_pre=%s", x_pre.flatten())

#         # Store prediction as current values (for update step)
#         self.x = x_pre
#         self.P = P_pre

#         # Extract the measured parts: translation from indices 0-2, Euler angles from 9-11
#         translation_estimated = x_pre[0:3].flatten()
#         eulers_estimated = x_pre[9:12].flatten()

#         return translation_estimated, eulers_estimated

#     def correct(self, tvec, R_input, frame=None):
#         """
#         Correct the state with measurement.
#         tvec: 3x1 translation measurement (absolute pose, in camera frame)
#         R_input: Rotation matrix measurement; convert to Euler angles.
#         Optionally, provide a frame number to log with this correction.
#         """
#         # Convert the rotation matrix to Euler angles.
#         eulers_measured = rotation_matrix_to_euler_angles(R_input)
#         # Build measurement vector z: [x, y, z, roll, pitch, yaw]^T
#         z = np.zeros((self.n_measurements, 1), dtype=np.float64)
#         z[0:3, 0] = tvec.flatten()
#         z[3:6, 0] = eulers_measured.flatten()

#         # Compute innovation: y = z - H*x_pre
#         y_innov = z - self.H @ self.x

#         # Innovation covariance: S = H*P*H^T + R
#         S = self.H @ self.P @ self.H.T + self.R

#         # Kalman gain: K = P*H^T * inv(S)
#         K = self.P @ self.H.T @ np.linalg.inv(S)

#         # Save pre-update debug info
#         corr_debug_before = {
#             "step": "correct_before",
#             "frame": frame if frame is not None else "N/A",
#             "z": z.flatten().tolist(),
#             "x_pre": self.x.flatten().tolist(),
#             "y_innov": y_innov.flatten().tolist(),
#             "S": S.tolist(),
#             "K": K.tolist()
#         }
#         self.debug_info.append(corr_debug_before)
#         logger.debug("Correct before: innovation=%s", y_innov.flatten())

#         # Update state: x_post = x_pre + K*y
#         x_post = self.x + K @ y_innov

#         # Update error covariance: P_post = (I - K*H) * P
#         I = np.eye(self.n_states)
#         P_post = (I - K @ self.H) @ self.P

#         # Save post-update debug info
#         corr_debug_after = {
#             "step": "correct_after",
#             "frame": frame if frame is not None else "N/A",
#             "x_post": x_post.flatten().tolist(),
#             "P_post": P_post.tolist()
#         }
#         self.debug_info.append(corr_debug_after)
#         logger.debug("Correct after: x_post=%s", x_post.flatten())

#         # Update internal state with corrected values.
#         self.x = x_post
#         self.P = P_post

#         # For external usage, we return the corrected measurement (absolute pose) parts:
#         translation_estimated = x_post[0:3].flatten()
#         eulers_estimated = x_post[9:12].flatten()
#         return translation_estimated, eulers_estimated

#     def save_debug_info(self, filename="kalman_debug.json"):
#         """Save the collected debug info to a JSON file."""
#         with open(filename, "w") as f:
#             json.dump(self.debug_info, f, indent=4)










############## ZERO movements

# class KalmanFilterPose:
#     def __init__(self, dt, n_states=6, n_measurements=6):
#         self.dt = dt
#         self.n_states = n_states
#         self.n_measurements = n_measurements
#         self.kf = cv2.KalmanFilter(n_states, n_measurements, 0, cv2.CV_64F)
#         self._init_kalman_filter()
    
#     def _init_kalman_filter(self):
#         kf = self.kf
#         # For a stationary object, the state doesn't change, so use the identity matrix.
#         kf.transitionMatrix = np.eye(self.n_states, dtype=np.float64)
        
#         # The measurement matrix maps the state directly to the measurements.
#         kf.measurementMatrix = np.eye(self.n_measurements, self.n_states, dtype=np.float64)
        
#         # Set the process and measurement noise covariances.
#         kf.processNoiseCov = np.eye(self.n_states, dtype=np.float64) * 1e-7
#         kf.measurementNoiseCov = np.eye(self.n_measurements, dtype=np.float64) * 1e-2
#         kf.errorCovPost = np.eye(self.n_states, dtype=np.float64) * 0.1
        
#         # Initialize the state (pose) to zero.
#         kf.statePost = np.zeros((self.n_states, 1), dtype=np.float64)
#         self.measurement = np.zeros((self.n_measurements, 1), dtype=np.float64)
    
#     def predict(self):
#         predicted = self.kf.predict()
#         # First three states are translation, last three are Euler angles.
#         translation_estimated = predicted[0:3].flatten()
#         eulers_estimated = predicted[3:6].flatten()
#         logger.debug(f"KF Predicted State: {self.kf.statePre.flatten()}")
#         return translation_estimated, eulers_estimated
    
#     def correct(self, tvec, R):
#         # Convert the rotation matrix to Euler angles.
#         eulers_measured = rotation_matrix_to_euler_angles(R)
#         self.measurement[0:3, 0] = tvec.flatten()
#         self.measurement[3:6, 0] = eulers_measured
#         self.kf.correct(self.measurement)
#         logger.debug(f"KF Corrected State: {self.kf.statePost.flatten()}")


#####################################################################################

# # # # # kalman_filter.py

# # # # import numpy as np
# # # # import cv2
# # # # from utils import rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix

# # # # class KalmanFilterPose:
# # # #     def __init__(self, dt, n_states=18, n_measurements=6):
# # # #         self.dt = dt
# # # #         self.n_states = n_states
# # # #         self.n_measurements = n_measurements
# # # #         self.kf = cv2.KalmanFilter(n_states, n_measurements, 0, cv2.CV_64F)
# # # #         self._init_kalman_filter()

# # # #     def _init_kalman_filter(self):
# # # #         kf = self.kf
# # # #         dt = self.dt
# # # #         kf.transitionMatrix = np.eye(self.n_states, dtype=np.float64)
# # # #         for i in range(3):
# # # #             kf.transitionMatrix[i, i+3] = dt
# # # #             kf.transitionMatrix[i, i+6] = 0.5 * dt**2
# # # #             kf.transitionMatrix[i+3, i+6] = dt
# # # #         for i in range(9, 12):
# # # #             kf.transitionMatrix[i, i+3] = dt
# # # #             kf.transitionMatrix[i, i+6] = 0.5 * dt**2
# # # #             kf.transitionMatrix[i+3, i+6] = dt
# # # #         kf.measurementMatrix = np.zeros((self.n_measurements, self.n_states), dtype=np.float64)
# # # #         kf.measurementMatrix[0, 0] = 1.0  # x
# # # #         kf.measurementMatrix[1, 1] = 1.0  # y
# # # #         kf.measurementMatrix[2, 2] = 1.0  # z
# # # #         kf.measurementMatrix[3, 9] = 1.0  # roll
# # # #         kf.measurementMatrix[4, 10] = 1.0  # pitch
# # # #         kf.measurementMatrix[5, 11] = 1.0  # yaw
# # # #         kf.processNoiseCov = np.eye(self.n_states, dtype=np.float64) * 1e-5
# # # #         kf.measurementNoiseCov = np.eye(self.n_measurements, dtype=np.float64) * 1e-4
# # # #         kf.errorCovPost = np.eye(self.n_states, dtype=np.float64)
# # # #         kf.statePost = np.zeros((self.n_states, 1), dtype=np.float64)
# # # #         self.measurement = np.zeros((self.n_measurements, 1), dtype=np.float64)

# # # #     def predict(self):
# # # #         predicted = self.kf.predict()
# # # #         translation_estimated = predicted[0:3].flatten()
# # # #         eulers_estimated = predicted[9:12].flatten()
# # # #         return translation_estimated, eulers_estimated

# # # #     def correct(self, tvec, R):
# # # #         eulers_measured = rotation_matrix_to_euler_angles(R)
# # # #         self.measurement[0:3, 0] = tvec.flatten()
# # # #         self.measurement[3:6, 0] = eulers_measured
# # # #         self.kf.correct(self.measurement)
