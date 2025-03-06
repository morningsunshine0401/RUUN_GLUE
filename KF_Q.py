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
