import numpy as np
import cv2
from utils import rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix
import logging

logger = logging.getLogger(__name__)

class KalmanFilterPose:
    def __init__(self, dt, n_states=18, n_measurements=6):
        """
        n_states=18, n_measurements=6:
          States (x,y,z, vx,vy,vz, ax,ay,az, roll,pitch,yaw, vroll,vpitch,vyaw, aroll,apitch,ayaw).
          Measurements are (x,y,z, roll,pitch,yaw).
        """
        self.dt = dt
        self.n_states = n_states
        self.n_measurements = n_measurements
        self.kf = cv2.KalmanFilter(n_states, n_measurements, 0, cv2.CV_64F)
        self._init_kalman_filter()

    def _init_kalman_filter(self):
        kf = self.kf
        dt = self.dt
        
        # 1) Define Transition Matrix (18x18)
        kf.transitionMatrix = np.eye(self.n_states, dtype=np.float64)

        # Position: x, y, z at indices [0,1,2]
        # Velocity: vx, vy, vz at indices [3,4,5]
        # Accel: ax, ay, az at indices [6,7,8]
        #
        # Orientation: roll, pitch, yaw at indices [9,10,11]
        # Angular velocity: vroll, vpitch, vyaw at [12,13,14]
        # Angular accel: aroll, apitch, ayaw at [15,16,17]

        # Position <-> velocity <-> accel coupling
        for i in range(3):
            # pos -> vel
            kf.transitionMatrix[i, i+3] = dt
            # pos -> accel
            kf.transitionMatrix[i, i+6] = 0.5 * dt**2
            # vel -> accel
            kf.transitionMatrix[i+3, i+6] = dt

        # Orientation <-> angular velocity <-> angular accel coupling
        # roll=9, pitch=10, yaw=11
        # vroll=12, vpitch=13, vyaw=14
        # aroll=15, apitch=16, ayaw=17
        for i in range(9, 12):
            kf.transitionMatrix[i, i+3] = dt
            kf.transitionMatrix[i, i+6] = 0.5 * dt**2
            kf.transitionMatrix[i+3, i+6] = dt

        # 2) Define Measurement Matrix (6x18)
        # We measure x, y, z, roll, pitch, yaw directly
        kf.measurementMatrix = np.zeros((self.n_measurements, self.n_states), dtype=np.float64)
        # x, y, z
        kf.measurementMatrix[0, 0] = 1.0  # x
        kf.measurementMatrix[1, 1] = 1.0  # y
        kf.measurementMatrix[2, 2] = 1.0  # z
        # roll, pitch, yaw
        kf.measurementMatrix[3, 9]  = 1.0  # roll
        kf.measurementMatrix[4, 10] = 1.0  # pitch
        kf.measurementMatrix[5, 11] = 1.0  # yaw

        # 3) Covariances
        kf.processNoiseCov = np.eye(self.n_states, dtype=np.float64) * 1e-5
        kf.measurementNoiseCov = np.eye(self.n_measurements, dtype=np.float64) * 1e-4
        kf.errorCovPost = np.eye(self.n_states, dtype=np.float64)

        # 4) Initial State
        kf.statePost = np.zeros((self.n_states, 1), dtype=np.float64)
        # We'll keep a separate measurement buffer
        self.measurement = np.zeros((self.n_measurements, 1), dtype=np.float64)

    def predict(self):
        predicted = self.kf.predict()
        # translation = x[0:3], orientation(eulers) = x[9:12]
        translation_estimated = predicted[0:3].flatten()
        eulers_estimated = predicted[9:12].flatten()
        logger.debug(f"KF Predicted State: {self.kf.statePre.flatten()}")
        return translation_estimated, eulers_estimated

    def correct(self, tvec, R):
        """
        Standard Kalman update with measurement = (x, y, z, roll, pitch, yaw).
        """
        eulers_measured = rotation_matrix_to_euler_angles(R)
        self.measurement[0:3, 0] = tvec.flatten()
        self.measurement[3:6, 0] = eulers_measured
        self.kf.correct(self.measurement)
        logger.debug(f"KF Corrected State: {self.kf.statePost.flatten()}")

    def get_covariance(self):
        """
        Return the full 18x18 error covariance matrix.
        """
        return self.kf.errorCovPost.copy()

    def get_state(self):
        """
        Return the full 18x1 state vector for debugging or advanced gating.
        Indices:
         0: x, 1: y, 2: z,
         3: vx, 4: vy, 5: vz,
         6: ax, 7: ay, 8: az,
         9: roll,10:pitch,11:yaw,
         12:vroll,13:vpitch,14:vyaw,
         15:aroll,16:apitch,17:ayaw
        """
        return self.kf.statePost.copy()

    def correct_partial(self, tvec, R, alpha=0.3):
        """
        A simplistic approach to 'partially blend' the measured pose
        (x, y, z, roll, pitch, yaw) into the current 18D state.
        
        We do *not* do a standard Kalman correct. Instead, we:
          1) read the current state,
          2) compute eulers from measurement,
          3) blend position + eulers by alpha,
          4) leave velocity/acc states as-is, or possibly inflate them,
          5) inflate the covariance to reflect uncertain update.
        """
        current_state = self.kf.statePost.copy()  # shape (18, 1)
        eulers_measured = rotation_matrix_to_euler_angles(R)

        # Blend position
        current_state[0, 0] = (1 - alpha)*current_state[0, 0] + alpha*tvec[0]
        current_state[1, 0] = (1 - alpha)*current_state[1, 0] + alpha*tvec[1]
        current_state[2, 0] = (1 - alpha)*current_state[2, 0] + alpha*tvec[2]

        # Blend orientation
        current_state[9, 0]  = (1 - alpha)*current_state[9, 0]  + alpha*eulers_measured[0]
        current_state[10, 0] = (1 - alpha)*current_state[10, 0] + alpha*eulers_measured[1]
        current_state[11, 0] = (1 - alpha)*current_state[11, 0] + alpha*eulers_measured[2]

        # Optionally, could do something for velocity if desired
        # e.g., set velocity to 0 or partial blend, but often you just let the filter handle it

        # Save back
        self.kf.statePost = current_state

        # Inflate covariance to reflect partial trust
        # This is a simple approach: multiply errorCovPost by e.g. 1/alpha or 2 or 10
        # depending on how uncertain you want to be:
        inflation_factor = 1.0 / alpha  
        self.kf.errorCovPost *= inflation_factor

        logger.debug("Partial blend update done.")
        logger.debug(f"New statePost: {self.kf.statePost.flatten()}")
