import numpy as np

def normalize_quaternion(q):
    norm_q = np.linalg.norm(q)
    if norm_q < 1e-15:
        return np.array([0,0,0,1], dtype=float)
    return q / norm_q

def quaternion_multiply(q1, q2):
    """
    Hamiltonian quaternion multiplication.
    q1, q2 are [x, y, z, w].
    Returns q = q1 ⊗ q2.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x, y, z, w], dtype=float)

def small_angle_quat(delta_theta):
    """
    Convert a small rotation vector 'delta_theta' in R^3
    to a quaternion [x, y, z, w].
    For small angles, angle = ||delta_theta||,
    axis = delta_theta / angle,
    q = [axis*sin(angle/2), cos(angle/2)].
    """
    angle = np.linalg.norm(delta_theta)
    if angle < 1e-12:
        # Linear approximation for very small angles
        return np.array([0.5*delta_theta[0],
                         0.5*delta_theta[1],
                         0.5*delta_theta[2],
                         1.0], dtype=float)
    axis = delta_theta / angle
    half = 0.5 * angle
    s = np.sin(half)
    w = np.cos(half)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, w], dtype=float)

class MEKF12D:
    def __init__(self, dt):
        self.dt = dt
        self.n_states = 12
        
        # 12D error-state: [dp(3), dv(3), dphi(3), dw(3)]
        self.x = np.zeros(self.n_states, dtype=float)
        
        # 12x12 covariance
        self.P = np.eye(self.n_states, dtype=float)*0.1
        
        # Nominal states stored outside x
        self.p_nom = np.zeros(3, dtype=float)
        self.v_nom = np.zeros(3, dtype=float)
        self.w_nom = np.zeros(3, dtype=float)
        self.q_nom = np.array([0,0,0,1], dtype=float)  # identity quaternion

        # Noise parameters (tune as needed)
        self.Q_p   = np.eye(3)*5e-1
        self.Q_v   = np.eye(3)*1e-1
        self.Q_phi = np.eye(3)* 4#1.0
        self.Q_w   = np.eye(3)*0.3

    def set_nominal_pose(self, p_init, q_init):
        """Initialize the nominal position & quaternion."""
        self.p_nom = p_init.copy()
        self.q_nom = normalize_quaternion(q_init)
        # By default, v_nom and w_nom can remain zero (or set them if you know them).
        self.x[:] = 0.0  # reset error-state

    def predict(self):
        """
        Propagate the nominal states (p_nom, v_nom, w_nom, q_nom)
        and linearize the 12D error-state x, P for the next time step.
        NO damping: we assume constant velocity and constant angular velocity.
        """
        dt = self.dt
        
        # 1) Integrate nominal states (constant velocity, constant angular velocity)
        self.p_nom += self.v_nom * dt
        # v_nom, w_nom remain unchanged (no damping):
        # self.v_nom = self.v_nom
        # self.w_nom = self.w_nom

        # quaternion derivative
        w = self.w_nom
        q = self.q_nom
        # For small dt, dq ~ 0.5 * Omega * q
        # Build 4x4 Omega
        w_skew = np.array([
            [0,    -w[2], w[1]],
            [w[2],  0,    -w[0]],
            [-w[1], w[0],  0   ]
        ])
        Omega = np.zeros((4,4))
        Omega[0:3, 0:3] = -w_skew
        Omega[0:3, 3]   = w
        Omega[3,   0:3] = -w
        dq = 0.5 * Omega @ q
        q_new = q + dq * dt
        self.q_nom = normalize_quaternion(q_new)

        # 2) Build F (12x12) for the error-state
        F = np.eye(12)
        # dp depends on dv => partial derivative is dt
        F[0:3, 3:6] = np.eye(3) * dt
        # The rest remain identity because we assume:
        #   dv = const,  dphi integrates w, but for small dt we approximate near identity
        #   w = const
        # So no damping factors:
        #   F[3:6, 3:6] = np.eye(3)
        #   F[9:12,9:12] = np.eye(3)
        #
        # If you want a more accurate Jacobian for orientation error, you can do so, but
        # for small dt, identity is typically acceptable.

        # 3) Process noise (12x12)
        Qk = np.zeros((12,12))
        Qk[0:3, 0:3]   = self.Q_p * (dt**2)
        Qk[3:6, 3:6]   = self.Q_v * (dt**2)
        Qk[6:9, 6:9]   = self.Q_phi * (dt**2)
        Qk[9:12, 9:12] = self.Q_w * (dt**2)

        # 4) Propagate covariance
        P_pred = F @ self.P @ F.T + Qk
        P_pred = 0.5 * (P_pred + P_pred.T)  # ensure symmetry

        self.P = P_pred
        self.x = F @ self.x  # error-state update

        return (self.p_nom.copy(),
                self.v_nom.copy(),
                self.q_nom.copy(),
                self.w_nom.copy()), P_pred.copy()

    def update_pose(self, pose_measurement):
        """
        Pose measurement = [pos_meas(3), quat_meas(4)] in [x,y,z,w].
        We'll form a 6D residual: r = [dp, dtheta].
          dp = p_meas - p_nom
          dtheta ≈ 2*(q_meas ⊗ q_nom^*)_{xyz}
        Then do the Kalman update in the error-state, and apply the correction
        to p_nom, v_nom, q_nom, w_nom.
        """
        p_meas = pose_measurement[0:3]
        q_meas = normalize_quaternion(pose_measurement[3:7])
        
        # 1) Build residual (6D)
        r = np.zeros(6)
        # position part
        r[0:3] = p_meas - self.p_nom
        
        # orientation part
        q_nom_conj = np.array([-self.q_nom[0], 
                               -self.q_nom[1],
                               -self.q_nom[2],
                                self.q_nom[3]])
        q_err = quaternion_multiply(q_meas, q_nom_conj)
        # Ensure q_err is in the half of the sphere with positive w
        if q_err[3] < 0:
            q_err = -q_err
        r[3:6] = 2.0 * q_err[0:3]  # small-angle assumption

        # 2) H Jacobian (6x12)
        #   residual r = [dp, dtheta]
        #   dp wrt error-state: dp block => identity on [dp(3)]
        #   dtheta wrt error-state: depends on [dphi(3)]
        H = np.zeros((6,12))
        # dp wrt dp
        H[0:3, 0:3] = np.eye(3)
        # dtheta wrt dphi
        H[3:6, 6:9] = np.eye(3)

        # 3) measurement noise R
        R = np.zeros((6,6))
        R[0:3, 0:3] = np.eye(3)*0.001
        #R[3:6, 3:6] = np.eye(3)*0.005
        R[3:6, 3:6] = np.eye(3)*0.01

        # 4) Kalman Gain
        P_pred = self.P
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        dx = K @ r

        # 5) Apply correction to nominal states
        # p, v
        self.p_nom += dx[0:3]
        self.v_nom += dx[3:6]
        
        # orientation from dphi
        dphi = dx[6:9]
        dq = small_angle_quat(dphi)  # [x,y,z,w]
        q_new = quaternion_multiply(dq, self.q_nom)
        self.q_nom = normalize_quaternion(q_new)

        # w
        self.w_nom += dx[9:12]

        # 6) Covariance update in Joseph form
        I_KH = np.eye(12) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)
        self.P = P_upd
        
        # 7) Reset the error state to zero
        self.x = np.zeros(12)

        return (self.p_nom.copy(),
                self.v_nom.copy(),
                self.q_nom.copy(),
                self.w_nom.copy()), P_upd.copy()






# import numpy as np

# def normalize_quaternion(q):
#     """Normalize a 4D quaternion [x, y, z, w]."""
#     norm_q = np.linalg.norm(q)
#     if norm_q < 1e-15:
#         # If it's really degenerate, fall back to identity
#         return np.array([0, 0, 0, 1], dtype=float)
#     return q / norm_q

# def quaternion_multiply(q1, q2):
#     """
#     Hamiltonian quaternion multiplication.
#     q1, q2 are [x, y, z, w].
#     Returns q = q1 ⊗ q2.
#     """
#     x1, y1, z1, w1 = q1
#     x2, y2, z2, w2 = q2
#     x = w1*x2 + x1*w2 + y1*z2 - z1*y2
#     y = w1*y2 - x1*z2 + y1*w2 + z1*x2
#     z = w1*z2 + x1*y2 - y1*x2 + z1*w2
#     w = w1*w2 - x1*x2 - y1*y2 - z1*z2
#     return np.array([x, y, z, w], dtype=float)

# def small_angle_quat(delta_theta):
#     """
#     Convert small rotation vector 'delta_theta' in R^3
#     to a 4D quaternion [x, y, z, w].
#     For small angles, angle = ||delta_theta||,
#     axis = delta_theta / angle,
#     q = [axis*sin(angle/2), cos(angle/2)].
#     """
#     angle = np.linalg.norm(delta_theta)
#     if angle < 1e-12:
#         return np.array([0.5*delta_theta[0],
#                          0.5*delta_theta[1],
#                          0.5*delta_theta[2],
#                          1.0], dtype=float)
#     axis = delta_theta / angle
#     half = 0.5 * angle
#     s = np.sin(half)
#     w = np.cos(half)
#     return np.array([axis[0]*s, axis[1]*s, axis[2]*s, w], dtype=float)


# class MultExtendedKalmanFilter:
#     """
#     13D state: x = [p(3), v(3), q(4), w(3)].
#       * p: position
#       * v: velocity
#       * q: quaternion (x,y,z,w)
#       * w: angular velocity

#     We keep the quaternion in x[6:10] but re-normalize after updates.
#     """
#     def __init__(self, dt):
#         self.dt = dt
#         self.n_states = 13
        
#         # State vector
#         self.x = np.zeros(self.n_states, dtype=float)
#         # By default, set quaternion = identity
#         self.x[6] = 0.0
#         self.x[7] = 0.0
#         self.x[8] = 0.0
#         self.x[9] = 1.0  # q = [0,0,0,1]

#         # Covariance
#         self.P = np.eye(self.n_states, dtype=float) * 0.1

#         # Example process noise
#         self.Q_p = np.eye(3, dtype=float) * 5e-1   # position noise
#         self.Q_v = np.eye(3, dtype=float) * 1e-1   # velocity noise
#         self.Q_q = np.eye(3, dtype=float) * 1.0    # quaternion process noise (for the vector part)
#         self.Q_w = np.eye(3, dtype=float) * 0.3    # angular velocity noise

#         self.velocity_damping = 0.6
#         self.angular_damping  = 0.6

#     def predict(self):
#         """
#         Predict step using the 13D nominal state:
#           - p_{k+1} = p_k + v_k * dt
#           - v_{k+1} = v_k * velocity_damping
#           - q_{k+1} = q_k + 0.5 * Omega(w_k) * q_k * dt   (then normalize)
#           - w_{k+1} = w_k * angular_damping
#         """
#         dt = self.dt
#         x_pred = self.x.copy()

#         # Extract
#         p = x_pred[0:3]
#         v = x_pred[3:6]
#         q = x_pred[6:10]    # [x,y,z,w]
#         w = x_pred[10:13]

#         # 1) Nominal integration
#         p_new = p + v * dt
#         v_new = v * self.velocity_damping
#         w_new = w * self.angular_damping
        
#         # quaternion derivative
#         w_skew = np.array([
#             [0,       -w[2],  w[1]],
#             [w[2],     0,    -w[0]],
#             [-w[1],  w[0],     0  ]
#         ], dtype=float)
#         Omega = np.zeros((4,4), dtype=float)
#         Omega[0:3, 0:3] = -w_skew
#         Omega[0:3, 3]   = w
#         Omega[3,   0:3] = -w
        
#         dq = 0.5 * Omega @ q
#         q_new = q + dq*dt
#         q_new = normalize_quaternion(q_new)

#         # Store into x_pred
#         x_pred[0:3]   = p_new
#         x_pred[3:6]   = v_new
#         x_pred[6:10]  = q_new
#         x_pred[10:13] = w_new

#         # 2) Build F (13x13) - linearization
#         F = np.eye(13, dtype=float)
#         # Position depends on velocity
#         F[0:3, 3:6] = np.eye(3, dtype=float) * dt
#         # Velocity damping
#         F[3:6, 3:6] = np.eye(3, dtype=float) * self.velocity_damping
#         # Angular velocity damping
#         F[10:13, 10:13] = np.eye(3, dtype=float) * self.angular_damping
        
#         # Optional: a naive identity block for the quaternion. A more accurate approach
#         #   would incorporate partial derivatives wrt. w, q, etc.
#         #
#         # F[6:10, 6:10] = ???

#         # 3) Process noise Q (13x13)
#         Qk = np.zeros((13,13), dtype=float)
#         # p
#         Qk[0:3, 0:3]     = self.Q_p * (dt**2)
#         # v
#         Qk[3:6, 3:6]     = self.Q_v * (dt**2)
#         # q (only 3 DOF, but we put noise in q_x,q_y,q_z block)
#         Qk[6:9, 6:9]     = self.Q_q * (dt**2)
#         # w
#         Qk[10:13, 10:13] = self.Q_w * (dt**2)

#         # 4) Covariance propagation
#         P_pred = F @ self.P @ F.T + Qk
#         P_pred = 0.5 * (P_pred + P_pred.T)  # enforce symmetry

#         # 5) Store results
#         self.x = x_pred
#         self.P = P_pred

#         return x_pred.copy(), P_pred.copy()

#     def proper_mekf_update(self, pose_measurement):
#         """
#         Loosely-coupled update with a pose measurement: [pos(3), quat(4)] (x,y,z,w).
#         We'll do a 6D measurement residual:
#            r_pos = p_meas - p
#            r_ori = 2*(q_meas ⊗ q_pred^*)_{xyz}
#         Then the update is:
#            x_upd = x_pred + K * r
#         But for orientation, we do a *multiplicative* step:
#            q_new = δq ⊗ q_pred
#         where δq is derived from dx[6:10] (i.e., do NOT add them directly).
#         Finally, re-normalize q_new.
#         """
#         x_pred = self.x
#         P_pred = self.P
        
#         # parse measurement
#         p_meas = pose_measurement[0:3]
#         q_meas = normalize_quaternion(pose_measurement[3:7])

#         # predicted states
#         p_pred = x_pred[0:3]
#         q_pred = x_pred[6:10]  # [x,y,z,w]

#         # 1) Build measurement residual
#         r = np.zeros(6, dtype=float)
#         # position residual
#         r[0:3] = p_meas - p_pred

#         # orientation residual:
#         #   q_err = q_meas ⊗ conj(q_pred)
#         q_pred_conj = np.array([-q_pred[0], -q_pred[1], -q_pred[2], q_pred[3]], dtype=float)
#         q_err = quaternion_multiply(q_meas, q_pred_conj)
#         # if q_err.w < 0, flip to ensure smallest rotation
#         if q_err[3] < 0:
#             q_err = -q_err
#         # small angle approximation => r[3:6] = 2 * q_err.xyz
#         r[3:6] = 2.0 * q_err[0:3]

#         # 2) Measurement Jacobian H (6x13)
#         H = np.zeros((6,13), dtype=float)
#         # position block
#         H[0:3, 0:3] = np.eye(3, dtype=float)
#         # orientation block ~ for the first 3 comps of quaternion
#         # we place an identity mapping in [3:6, 6:9]
#         H[3:6, 6:9] = np.eye(3, dtype=float)

#         # 3) Measurement noise R = 6x6
#         R = np.zeros((6,6), dtype=float)
#         R[0:3, 0:3] = np.eye(3, dtype=float) * 0.001
#         R[3:6, 3:6] = np.eye(3, dtype=float) * 0.005

#         # 4) Kalman gain
#         S = H @ P_pred @ H.T + R
#         K = P_pred @ H.T @ np.linalg.inv(S)

#         # 5) dx = K * r
#         dx = K @ r

#         # 6) Apply the correction
#         x_upd = x_pred.copy()

#         # linear parts
#         x_upd[0:3]   += dx[0:3]   # position
#         x_upd[3:6]   += dx[3:6]   # velocity

#         # orientation
#         #
#         # Now we incorporate *all 4* corrections dx[6:10] in a *multiplicative* manner.
#         # Step (a): form a small quaternion increment from dx[6:10].
#         # A simple way: interpret the first 3 as the axis, the 4th as a small shift in w.
#         # Then re-normalize that increment. This ensures we use dx[9] as well.
#         dq_4 = dx[6:10].copy()  # [dq_x, dq_y, dq_z, dq_w]
        
#         # Build a 'small' quaternion from dq_4. One naive approach:
#         #   For small angles, the vector part is half of dx[6:9], and the w part is ~1 + dx[9].
#         #   Then re-normalize. This at least uses dx[9].
#         # If dx[9] is large, you might need a more careful approach.
        
#         dq_vec = 0.5 * dq_4[0:3]  # half of the axis increments
#         dq_w   = 1.0 + dq_4[3]    # add the w increment
#         dq_raw = np.array([dq_vec[0], dq_vec[1], dq_vec[2], dq_w], dtype=float)
#         dq_small = normalize_quaternion(dq_raw)

#         # Step (b): multiply q_pred by dq_small
#         q_new = quaternion_multiply(dq_small, q_pred)
#         q_new = normalize_quaternion(q_new)
#         x_upd[6:10] = q_new

#         # angular velocity
#         x_upd[10:13] += dx[10:13]

#         # 7) Joseph form covariance update
#         I_KH = np.eye(13, dtype=float) - K @ H
#         P_upd = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
#         P_upd = 0.5 * (P_upd + P_upd.T)  # enforce symmetry

#         # 8) Store
#         self.x = x_upd
#         self.P = P_upd

#         return x_upd.copy(), P_upd.copy()