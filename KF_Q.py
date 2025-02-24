# import cv2
# import numpy as np
# import json
# import logging

# from utils import (
#     normalize_quaternion,
#     rotation_matrix_to_quaternion,
#     quaternion_to_rotation_matrix,
# )

# logger = logging.getLogger(__name__)

# class KalmanFilterPose:
#     """
#     Kalman Filter with 18 states:

#       * [0..2]:   x,   y,   z          (position)
#       * [3..5]:   vx,  vy,  vz         (linear velocity)
#       * [6..8]:   ax,  ay,  az         (linear acceleration)
#       * [9..12]:  qx,  qy,  qz,  qw    (orientation as quaternion)
#       * [13..15]: wx,  wy,  wz         (angular velocity)
#       * [16..17]: leftover if needed (e.g. angular accel)

#     Measurement example = 10 dims: (x,y,z, qx,qy,qz,qw, wx,wy,wz)
#     If you don't measure angular velocity, adapt the measurement dims below.

#     Because orientation dynamics are nonlinear, we do a *manual* quaternion integration
#     after the linear predict step, storing it back into statePost. This is an approximation.
#     A true approach would be an Extended or Unscented Kalman Filter.
#     """
#     def __init__(self, dt):
#         self.dt = dt
#         self.n_states = 18
#         # EXAMPLE: We'll measure 10 dims => (x,y,z, qx,qy,qz,qw, wx,wy,wz).
#         # If you measure only orientation, do 7 dims, etc. Adjust below.
#         self.n_measurements = 10  

#         self.kf = cv2.KalmanFilter(self.n_states, self.n_measurements, 0, cv2.CV_64F)
#         self.debug_info = []
#         self._init_kalman_filter()

#     def _init_kalman_filter(self):
#         kf = self.kf
#         dt = self.dt

#         #-------------------------
#         # 1) Transition Matrix
#         #-------------------------
#         # Start as identity
#         kf.transitionMatrix = np.eye(self.n_states, dtype=np.float64)

#         # (a) Position integration
#         for i in range(3):
#             kf.transitionMatrix[i,   i+3] = dt       # x += vx * dt
#             kf.transitionMatrix[i,   i+6] = 0.5*dt**2 # + 1/2*ax*dt^2
#             kf.transitionMatrix[i+3, i+6] = dt       # vx += ax*dt

#         # (b) Keep quaternion [9..12] in identity => we do custom integration in `predict()`.
#         # (c) Keep angular velocity [13..15] as constant => identity block
#         # You could add angular acceleration if you want to model w += alpha*dt, etc.

#         #-------------------------
#         # 2) Measurement Matrix
#         #-------------------------
#         # We'll measure x,y,z => states [0..2]
#         # and qx,qy,qz,qw => states [9..12]
#         # and wx,wy,wz => states [13..15]
#         # => total 10 measurements
#         H = np.zeros((self.n_measurements, self.n_states), dtype=np.float64)
#         # Position
#         H[0,0] = 1.0  # x
#         H[1,1] = 1.0  # y
#         H[2,2] = 1.0  # z
#         # Quaternion
#         H[3,9]  = 1.0 # qx
#         H[4,10] = 1.0 # qy
#         H[5,11] = 1.0 # qz
#         H[6,12] = 1.0 # qw
#         # Angular velocity
#         H[7,13] = 1.0 # wx
#         H[8,14] = 1.0 # wy
#         H[9,15] = 1.0 # wz
#         kf.measurementMatrix = H

#         #-------------------------
#         # 3) Noise & Covariances
#         #-------------------------
#         kf.processNoiseCov = np.eye(self.n_states, dtype=np.float64) * 1e-5
#         kf.measurementNoiseCov = np.eye(self.n_measurements, dtype=np.float64) * 1e-2
#         kf.errorCovPost = np.eye(self.n_states, dtype=np.float64) * 0.1

#         #-------------------------
#         # 4) Initial State
#         #-------------------------
#         kf.statePost = np.zeros((self.n_states,1), dtype=np.float64)
#         # Identity quaternion
#         kf.statePost[12,0] = 1.0
#         # measurement vector
#         self.measurement = np.zeros((self.n_measurements,1), dtype=np.float64)

#         # Debug
#         init_debug = {
#             "step": "init",
#             "transitionMatrix": kf.transitionMatrix.tolist(),
#             "measurementMatrix": kf.measurementMatrix.tolist(),
#             "processNoiseCov": kf.processNoiseCov.tolist(),
#             "measurementNoiseCov": kf.measurementNoiseCov.tolist(),
#             "errorCovPost": kf.errorCovPost.tolist(),
#             "statePost": kf.statePost.flatten().tolist()
#         }
#         self.debug_info.append(init_debug)
#         logger.debug("Init KalmanFilterPose with orientation + angular velocity states.")

#     def predict(self, frame=None):
#         """
#         1) Do a standard linear KF predict => statePre
#         2) Then manually integrate quaternion with the predicted angular velocity
#            for a small dt approximation.
#         3) Re-normalize the quaternion and store it in statePost.
#         """
#         # 1) Standard KF predict
#         predicted = self.kf.predict()

#         # 2) Extract states we care about
#         # translation: [0..2]
#         translation_est = predicted[0:3, 0]
#         # quaternion: [9..12]
#         qx, qy, qz, qw = predicted[9,0], predicted[10,0], predicted[11,0], predicted[12,0]
#         q_est = np.array([qx, qy, qz, qw], dtype=np.float64)
#         # angular velocity: [13..15]
#         wx, wy, wz = predicted[13,0], predicted[14,0], predicted[15,0]

#         # 3) Integrate quaternion with small-angle assumption
#         dt = self.dt
#         # derivative of quaternion = 0.5 * Omega(angVel) * q
#         # We'll do: q_k+1 = q_k + 0.5*dt * Omega(w)*q_k
#         # Then normalize
#         dq = self._quaternion_derivative(q_est, np.array([wx,wy,wz]), dt)
#         q_new = q_est + dq
#         q_new = normalize_quaternion(q_new)

#         # 4) Store back to statePost
#         self.kf.statePost[9,0]  = q_new[0]
#         self.kf.statePost[10,0] = q_new[1]
#         self.kf.statePost[11,0] = q_new[2]
#         self.kf.statePost[12,0] = q_new[3]

#         # For consistency, you might also copy any other predicted states if you do partial overrides.

#         # Debug logs
#         logger.debug(f"Predict step: Pos={translation_est}, Q(before)={q_est}, Q(after)={q_new}, w=({wx},{wy},{wz})")
#         predict_debug = {
#             "step": "predict",
#             "did_correct": False,
#             "translation_est": translation_est.tolist(),
#             "quaternion_before_integration": q_est.tolist(),
#             "quaternion_after_integration": q_new.tolist(),
#             "angular_velocity_est": [wx, wy, wz],
#             "errorCovPre": self.kf.errorCovPre.tolist()
#         }
#         if frame is not None:
#             predict_debug["frame"] = frame
#         self.debug_info.append(predict_debug)

#         return translation_est, q_new, np.array([wx,wy,wz])

#     def correct(self, tvec, R, ang_vel=None, frame=None):
#         """
#         Provide a measurement of: (x,y,z, qx,qy,qz,qw, wx,wy,wz).
#           - If you don't have angular velocity measurement, set `ang_vel=None`.
#             Then either skip those 3 lines or set them to zeros with high noise.

#         :param tvec: shape (3,1) or (3,) for translation
#         :param R: 3x3 rotation matrix => convert to quaternion
#         :param ang_vel: shape (3,) for (wx, wy, wz). If unknown, use None or zeros.
#         """
#         q_measured = rotation_matrix_to_quaternion(R)
#         q_measured = normalize_quaternion(q_measured)

#         # Fill measurement
#         self.measurement[0,0] = tvec[0]
#         self.measurement[1,0] = tvec[1]
#         self.measurement[2,0] = tvec[2]
#         self.measurement[3,0] = q_measured[0]
#         self.measurement[4,0] = q_measured[1]
#         self.measurement[5,0] = q_measured[2]
#         self.measurement[6,0] = q_measured[3]
#         if ang_vel is not None:
#             # if you have IMU or some source of w
#             self.measurement[7,0] = ang_vel[0]
#             self.measurement[8,0] = ang_vel[1]
#             self.measurement[9,0] = ang_vel[2]
#         else:
#             # no measurement => set to 0 or old state with high measurement noise
#             self.measurement[7,0] = 0
#             self.measurement[8,0] = 0
#             self.measurement[9,0] = 0

#         # Manual logging of gain, etc. (like your original code)
#         P = self.kf.errorCovPre
#         H = self.kf.measurementMatrix
#         R_meas = self.kf.measurementNoiseCov
#         S = H @ P @ H.T + R_meas
#         K = P @ H.T @ np.linalg.inv(S)
#         zPred = H @ self.kf.statePre
#         residual = self.measurement - zPred
#         residual_norm = float(np.linalg.norm(residual))

#         logger.debug(f"Correct step: measurement={self.measurement.flatten()}, residual={residual.flatten()}")
#         correct_debug_before = {
#             "step": "correct_before",
#             "did_correct": True,
#             "measurement": self.measurement.flatten().tolist(),
#             "innovation": residual.flatten().tolist(),
#             "innovation_norm": residual_norm,
#             "kalman_gain": K.tolist(),
#             "errorCovPre": P.tolist(),
#         }
#         if frame is not None:
#             correct_debug_before["frame"] = frame
#         self.debug_info.append(correct_debug_before)

#         # Official correction
#         self.kf.correct(self.measurement)

#         # Re-normalize quaternion in statePost
#         qx = self.kf.statePost[9,0]
#         qy = self.kf.statePost[10,0]
#         qz = self.kf.statePost[11,0]
#         qw = self.kf.statePost[12,0]
#         q_new = normalize_quaternion([qx,qy,qz,qw])
#         self.kf.statePost[9,0]  = q_new[0]
#         self.kf.statePost[10,0] = q_new[1]
#         self.kf.statePost[11,0] = q_new[2]
#         self.kf.statePost[12,0] = q_new[3]

#         logger.debug("Correct step: updated statePost: %s", self.kf.statePost.flatten())
#         correct_debug_after = {
#             "step": "correct_after",
#             "did_correct": True,
#             "statePost": self.kf.statePost.flatten().tolist(),
#             "errorCovPost": self.kf.errorCovPost.tolist()
#         }
#         if frame is not None:
#             correct_debug_after["frame"] = frame
#         self.debug_info.append(correct_debug_after)

#     def save_debug_info(self, filename="kalman_debug.json"):
#         with open(filename, "w") as f:
#             json.dump(self.debug_info, f, indent=4)

#     #----------------------------------------
#     # Small helper: discrete quaternion integration
#     #----------------------------------------
#     def _quaternion_derivative(self, q, w, dt):
#         """
#         Approx discrete step: dq = 0.5 * dt * Omega(w) * q
#         Where w = (wx,wy,wz) in rad/s, and q = [qx,qy,qz,qw].
#         Returns the delta to be added to q.
#         """
#         # Build the 'Omega' matrix for w
#         # Omega(w) = [[ 0, -wx, -wy, -wz],
#         #             [wx,  0,  wz, -wy],
#         #             [wy, -wz,  0,  wx],
#         #             [wz,  wy, -wx,  0 ]]
#         wx, wy, wz = w
#         Omega = np.array([
#             [0,  -wx, -wy, -wz],
#             [wx,   0,  wz, -wy],
#             [wy, -wz,   0,  wx],
#             [wz,  wy, -wx,   0 ]
#         ], dtype=np.float64)

#         q_col = q.reshape(4,1)
#         dq = 0.5 * dt * (Omega @ q_col)
#         return dq.flatten()


##########################################################################

# EKF

import numpy as np
import logging

from utils import normalize_quaternion, rotation_matrix_to_quaternion

logger = logging.getLogger(__name__)

class KalmanFilterPose:
    """
    EKF with state x = [px, py, pz, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]^T
    For simplicity, we assume constant velocity + constant angular velocity in predict step.

    'f' (process) and 'h' (measurement) are nonlinear, so we compute their Jacobians.
    """
    def __init__(self, dt):
        self.dt = dt

        self.n_states = 13
        # Example: measure position (3) + orientation (3x3 => or 4 if quaternion),
        # but for a simple example we'll do measure pos(3) + orientation as a quaternion(4).
        # That yields 7 measurements. 
        # If you measure angular velocity or anything else, adapt.
        self.n_measurements = 7  

        # State: [p(3), v(3), q(4), w(3)]
        self.x = np.zeros(self.n_states)
        self.x[9] = 1.0  # quaternion w=1, x=y=z=0

        # Covariance
        self.P = np.eye(self.n_states) * 0.1

        # Tuning: process noise Q, measurement noise R
        self.Q = np.eye(self.n_states)*1e-4
        self.R = np.eye(self.n_measurements)*1e-2

    def predict(self):
        """
        EKF predict step: x_k+1^- = f(x_k^+).
        We also compute the Jacobian F = df/dx.
        Then P^- = F P F^T + Q.
        """
        dt = self.dt
        px,py,pz   = self.x[0:3]
        vx,vy,vz   = self.x[3:6]
        qx,qy,qz,qw= self.x[6:10]
        wx,wy,wz   = self.x[10:13]

        # 1) Nonlinear state update
        # a) position
        px_new = px + vx*dt
        py_new = py + vy*dt
        pz_new = pz + vz*dt

        # b) velocity => constant
        vx_new, vy_new, vz_new = vx, vy, vz

        # c) orientation => integrate quaternion by small-angle
        # dq = 0.5 * dt * Omega(w) * q
        q = np.array([qx,qy,qz,qw])
        w = np.array([wx,wy,wz])
        dq = 0.5*dt*self._omega_mat(w)@q
        q_new = q + dq
        q_new = normalize_quaternion(q_new)

        # d) angular velocity => constant
        wx_new, wy_new, wz_new = wx, wy, wz

        x_pred = np.array([
            px_new, py_new, pz_new,
            vx_new, vy_new, vz_new,
            q_new[0], q_new[1], q_new[2], q_new[3],
            wx_new, wy_new, wz_new
        ])

        # 2) Build the Jacobian F
        # We'll approximate partial derivatives. For short dt, this is close.
        F = np.eye(self.n_states)

        # dx/dv = dt
        for i in range(3):
            F[i, 3+i] = dt

        # dquat/dquat => identity, but there's a cross term from w. This is an approximation
        # for short dt. For a proper EKF, you'd compute the partial derivative carefully.
        # We'll do a minimal approach:
        # quaternion block -> there's a derivative w.r.t. w. 
        # We'll keep it simple: F[6..9, 10..12] ~ 0.5 * dt * partial stuff...
        # For short dt, we might skip it or do a small approximation.

        # This is a minimal example, so let's skip that. 
        # A better approach: build partial derivatives for the quaternion integration.

        # 3) Update covariance
        P_pred = F @ self.P @ F.T + self.Q

        # Store
        self.x = x_pred
        self.P = P_pred

        return self.x.copy(), self.P.copy()

    def update(self, z):
        """
        z = [px, py, pz, qx, qy, qz, qw] (7 dims)
        We do standard EKF update. 
        h(x) => returns the predicted measurement from the state.
        H => partial derivative of h wrt x at current x^-.
        """
        # 1) Compute predicted measurement h(x^-)
        z_pred = self._h(self.x)
        # 2) Jacobian H
        H = self._H_jacobian(self.x)
        # 3) Innovation
        y = z - z_pred
        # Might re-normalize the orientation residual. 
        # e.g. if we measure q, we can do something like: y[3..6] = small angle from difference 
        # (Trick: for small orientation error, or convert measured q & predicted q to angle-axis.)
        # For simplicity, we keep direct difference.

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 4) Correct
        self.x = self.x + K @ y
        # Re-normalize quaternion part
        q = self.x[6:10]
        q = normalize_quaternion(q)
        self.x[6:10] = q

        # 5) Cov update
        I = np.eye(self.n_states)
        self.P = (I - K@H) @ self.P

        return self.x.copy(), self.P.copy()

    #----------------------------------------
    # Nonlinear measurement function h(x)
    #----------------------------------------
    def _h(self, x):
        """
        If measuring position + orientation quaternion:
        z = [px, py, pz, qx, qy, qz, qw].
        """
        px,py,pz = x[0:3]
        qx,qy,qz,qw = x[6:10]
        return np.array([px,py,pz, qx,qy,qz,qw])

    #----------------------------------------
    # Jacobian of h wrt x
    #----------------------------------------
    def _H_jacobian(self, x):
        """
        partial of [px,py,pz, qx,qy,qz,qw] wrt x(= [px,py,pz, vx,vy,vz, qx,qy,qz,qw, wx,wy,wz])
        is mostly identity in pos + orientation blocks.
        """
        H = np.zeros((7, self.n_states))
        # position block
        H[0,0] = 1.0
        H[1,1] = 1.0
        H[2,2] = 1.0
        # orientation block
        H[3,6] = 1.0  # d(qx)/d(qx)
        H[4,7] = 1.0
        H[5,8] = 1.0
        H[6,9] = 1.0
        return H

    def _omega_mat(self, w):
        """
        Returns Omega(w) for w=(wx,wy,wz):
            [[0, -wx, -wy, -wz],
             [wx,  0,  wz, -wy],
             [wy, -wz,  0,  wx],
             [wz,  wy, -wx,  0 ]]
        """
        wx,wy,wz = w
        return np.array([
            [0,   -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy,  -wz,  0,   wx],
            [wz,   wy, -wx,  0 ]
        ], dtype=float)

    #-----------------------------------------
    # Helper to retrieve current states
    #-----------------------------------------
    def get_state(self):
        """
        Return the entire state vector for external usage, e.g.:
        p = x[0:3]
        q = x[6:10]
        ...
        """
        return self.x.copy(), self.P.copy()
