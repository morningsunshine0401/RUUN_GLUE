import numpy as np
import logging

from utils import normalize_quaternion, rotation_matrix_to_quaternion, quaternion_to_rotation_matrix

logger = logging.getLogger(__name__)

class MultExtendedKalmanFilter:
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
        self.Q = np.eye(self.n_states)*1e-4#*1e-3
        self.R = np.eye(self.n_measurements)*1e-2#*1e-4
        #self.R_tight = np.eye(2) * 4.0 

    def predict(self):
        """
        EKF predict step: x_k+1^- = f(x_k^+).
        We also compute the Jacobian F = df/dx.
        Then P^- = F P F^T + Q.
        """
        dt = self.dt
        px, py, pz = self.x[0:3]
        vx, vy, vz = self.x[3:6]
        qx, qy, qz, qw = self.x[6:10]
        wx, wy, wz = self.x[10:13]

        # 1) Nonlinear state update
        # a) position
        px_new = px + vx*dt
        py_new = py + vy*dt
        pz_new = pz + vz*dt

        # b) velocity => constant
        vx_new, vy_new, vz_new = vx, vy, vz

        # c) orientation => proper quaternion integration for finite rotations
        q = np.array([qx, qy, qz, qw])
        w = np.array([wx, wy, wz])
        
        # Proper quaternion integration using angle-axis representation
        w_norm = np.linalg.norm(w)
        
        if w_norm < 1e-10:  # Handle very small angular velocities
            q_new = q.copy()
        else:
            # Compute rotation quaternion using angle-axis representation
            angle = w_norm * dt
            axis = w / w_norm
            
            sin_half_angle = np.sin(angle / 2)
            cos_half_angle = np.cos(angle / 2)
            
            dq = np.array([
                axis[0] * sin_half_angle,
                axis[1] * sin_half_angle,
                axis[2] * sin_half_angle,
                cos_half_angle
            ])
            
            # Apply rotation: q_new = dq * q (quaternion multiplication)
            q_new = self._quaternion_multiply(dq, q)
        
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
        F = np.eye(self.n_states)

        # dx/dv = dt for position update
        for i in range(3):
            F[i, 3+i] = dt

        # Jacobian for quaternion update with respect to both quaternion and angular velocity
        if w_norm > 1e-10:
            # Jacobian block for dq/dq
            F_q_q = self._quaternion_rotation_jacobian(w, dt)
            F[6:10, 6:10] = F_q_q
            
            # Jacobian block for dq/dw
            F_q_w = self._quaternion_angular_velocity_jacobian(q, w, dt)
            F[6:10, 10:13] = F_q_w

        # 3) Update covariance
        P_pred = F @ self.P @ F.T + self.Q

        # Store
        self.x = x_pred
        self.P = P_pred

        return self.x.copy(), self.P.copy()
        
    def _quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions q1 * q2
        q = [x, y, z, w] format
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([x, y, z, w])
    
    def _quaternion_rotation_jacobian(self, w, dt):
        """
        Compute the Jacobian of the quaternion update with respect to the current quaternion
        when applying a rotation with angular velocity w for duration dt.
        """
        w_norm = np.linalg.norm(w)
        
        if w_norm < 1e-10:
            return np.eye(4)  # Identity for very small rotations
            
        angle = w_norm * dt
        axis = w / w_norm
        
        sin_half = np.sin(angle/2)
        cos_half = np.cos(angle/2)
        
        # Create rotation quaternion
        dq = np.array([
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            cos_half
        ])
        
        # Create quaternion left-multiplication matrix
        # This represents how dq * q changes as q changes
        J_q = np.array([
            [dq[3],  dq[2], -dq[1], dq[0]],
            [-dq[2],  dq[3],  dq[0], dq[1]],
            [dq[1], -dq[0],  dq[3], dq[2]],
            [-dq[0], -dq[1], -dq[2], dq[3]]
        ])
        
        return J_q
        
    def _quaternion_angular_velocity_jacobian(self, q, w, dt):
        """
        Compute the Jacobian of the quaternion update with respect to angular velocity.
        """
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-10:
            # For very small angular velocities, use a simple linear approximation
            qx, qy, qz, qw = q
            half_dt = 0.5 * dt
            
            J_w = np.zeros((4, 3))
            J_w[0, 0] = half_dt * qw
            J_w[0, 1] = -half_dt * qz
            J_w[0, 2] = half_dt * qy
            
            J_w[1, 0] = half_dt * qz
            J_w[1, 1] = half_dt * qw
            J_w[1, 2] = -half_dt * qx
            
            J_w[2, 0] = -half_dt * qy
            J_w[2, 1] = half_dt * qx
            J_w[2, 2] = half_dt * qw
            
            J_w[3, 0] = -half_dt * qx
            J_w[3, 1] = -half_dt * qy
            J_w[3, 2] = -half_dt * qz
            
            return J_w
        
        # For non-negligible angular velocity, compute a more accurate Jacobian
        # This is a numerical approximation of the partial derivatives
        epsilon = 1e-6
        J_w = np.zeros((4, 3))
        
        for i in range(3):
            # Perturb angular velocity in each dimension
            w_plus = w.copy()
            w_plus[i] += epsilon
            
            # Compute quaternion update with perturbed angular velocity
            w_plus_norm = np.linalg.norm(w_plus)
            axis_plus = w_plus / w_plus_norm
            angle_plus = w_plus_norm * dt
            
            sin_half_plus = np.sin(angle_plus/2)
            cos_half_plus = np.cos(angle_plus/2)
            
            dq_plus = np.array([
                axis_plus[0] * sin_half_plus,
                axis_plus[1] * sin_half_plus,
                axis_plus[2] * sin_half_plus,
                cos_half_plus
            ])
            
            q_plus = self._quaternion_multiply(dq_plus, q)
            
            # Original quaternion update
            axis = w / w_norm
            angle = w_norm * dt
            
            sin_half = np.sin(angle/2)
            cos_half = np.cos(angle/2)
            
            dq = np.array([
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
                cos_half
            ])
            
            q_orig = self._quaternion_multiply(dq, q)
            
            # Compute partial derivative using finite difference
            J_w[:, i] = (q_plus - q_orig) / epsilon
            
        return J_w

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
        
        # Proper handling of quaternion differences
        # Ensure we're taking the shortest path for orientation
        q_meas = z[3:7]
        q_pred = z_pred[3:7]
        
        # If the dot product is negative, flip the measured quaternion
        # to ensure we're using the shorter rotation path
        if np.dot(q_meas, q_pred) < 0:
            q_meas = -q_meas
            y[3:7] = q_meas - q_pred

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 4) Correct
        self.x = self.x + K @ y
        # Re-normalize quaternion part
        q = self.x[6:10]
        q = normalize_quaternion(q)
        self.x[6:10] = q

        # 5) Cov update - Joseph form for better numerical stability
        I = np.eye(self.n_states)
        self.P = (I - K@H) @ self.P @ (I - K@H).T + K @ self.R @ K.T

        return self.x.copy(), self.P.copy()

    #----------------------------------------
    # Nonlinear measurement function h(x)
    #----------------------------------------
    def _h(self, x):
        """
        If measuring position + orientation quaternion:
        z = [px, py, pz, qx, qy, qz, qw]. 
        """
        px, py, pz = x[0:3]
        qx, qy, qz, qw = x[6:10]
        return np.array([px, py, pz, qx, qy, qz, qw])

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
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        # orientation block
        H[3, 6] = 1.0  # d(qx)/d(qx)
        H[4, 7] = 1.0
        H[5, 8] = 1.0
        H[6, 9] = 1.0
        return H

    def _omega_mat(self, w):
        """
        Returns Omega(w) for w=(wx,wy,wz):
            [[0, -wx, -wy, -wz],
             [wx,  0,  wz, -wy],
             [wy, -wz,  0,  wx],
             [wz,  wy, -wx,  0 ]]
        """
        wx, wy, wz = w
        return np.array([
            [0,   -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy,  -wz,  0,   wx],
            [wz,   wy, -wx,  0 ]
        ], dtype=float)
        
    def get_state(self):
        """
        Return the entire state vector for external usage, e.g.:
        p = x[0:3]
        q = x[6:10]
        ...
        """
        return self.x.copy(), self.P.copy()
    


# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################

# # EKF  Default that works

# import numpy as np
# import logging

# from utils import normalize_quaternion, rotation_matrix_to_quaternion ,quaternion_to_rotation_matrix

# logger = logging.getLogger(__name__)

# class MultExtendedKalmanFilter:

#     """
#     EKF with state x = [px, py, pz, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]^T
#     For simplicity, we assume constant velocity + constant angular velocity in predict step.

#     'f' (process) and 'h' (measurement) are nonlinear, so we compute their Jacobians.
#     """
#     def __init__(self, dt):
#         self.dt = dt

#         self.n_states = 13
#         # Example: measure position (3) + orientation (3x3 => or 4 if quaternion),
#         # but for a simple example we'll do measure pos(3) + orientation as a quaternion(4).
#         # That yields 7 measurements. 
#         # If you measure angular velocity or anything else, adapt.
#         self.n_measurements = 7  

#         # State: [p(3), v(3), q(4), w(3)]
#         self.x = np.zeros(self.n_states)
#         self.x[9] = 1.0  # quaternion w=1, x=y=z=0

#         # Covariance
#         self.P = np.eye(self.n_states) * 0.1

#         # Tuning: process noise Q, measurement noise R
#         self.Q = np.eye(self.n_states)*1e-3
#         self.R = np.eye(self.n_measurements)*1e-3#5


#         self.R_tight = np.eye(2) * 4.0 

#     def predict(self):
#         """
#         EKF predict step: x_k+1^- = f(x_k^+).
#         We also compute the Jacobian F = df/dx.
#         Then P^- = F P F^T + Q.
#         """
#         dt = self.dt
#         px,py,pz   = self.x[0:3]
#         vx,vy,vz   = self.x[3:6]
#         qx,qy,qz,qw= self.x[6:10]
#         wx,wy,wz   = self.x[10:13]

#         # 1) Nonlinear state update
#         # a) position
#         px_new = px + vx*dt
#         py_new = py + vy*dt
#         pz_new = pz + vz*dt

#         # b) velocity => constant
#         vx_new, vy_new, vz_new = vx, vy, vz

#         # c) orientation => integrate quaternion by small-angle
#         # dq = 0.5 * dt * Omega(w) * q
#         q = np.array([qx,qy,qz,qw])
#         w = np.array([wx,wy,wz])
#         dq = 0.5*dt*self._omega_mat(w)@q
#         q_new = q + dq
#         q_new = normalize_quaternion(q_new)

#         # d) angular velocity => constant
#         wx_new, wy_new, wz_new = wx, wy, wz

#         x_pred = np.array([
#             px_new, py_new, pz_new,
#             vx_new, vy_new, vz_new,
#             q_new[0], q_new[1], q_new[2], q_new[3],
#             wx_new, wy_new, wz_new
#         ])

#         # 2) Build the Jacobian F
#         # We'll approximate partial derivatives. For short dt, this is close.
#         F = np.eye(self.n_states)

#         # dx/dv = dt
#         for i in range(3):
#             F[i, 3+i] = dt

#         # dquat/dquat => identity, but there's a cross term from w. This is an approximation
#         # for short dt. For a proper EKF, you'd compute the partial derivative carefully.
#         # We'll do a minimal approach:
#         # quaternion block -> there's a derivative w.r.t. w. 
#         # We'll keep it simple: F[6..9, 10..12] ~ 0.5 * dt * partial stuff...
#         # For short dt, we might skip it or do a small approximation.

#         # This is a minimal example, so let's skip that. 
#         # A better approach: build partial derivatives for the quaternion integration.

#         # 3) Update covariance
#         P_pred = F @ self.P @ F.T + self.Q

#         # Store
#         self.x = x_pred
#         self.P = P_pred

#         return self.x.copy(), self.P.copy()

#     def update(self, z):
#         """
#         z = [px, py, pz, qx, qy, qz, qw] (7 dims)
#         We do standard EKF update. 
#         h(x) => returns the predicted measurement from the state.
#         H => partial derivative of h wrt x at current x^-.
#         """
#         # 1) Compute predicted measurement h(x^-)
#         z_pred = self._h(self.x)
#         # 2) Jacobian H
#         H = self._H_jacobian(self.x)
#         # 3) Innovation
#         y = z - z_pred
#         # Might re-normalize the orientation residual. 
#         # e.g. if we measure q, we can do something like: y[3..6] = small angle from difference 
#         # (Trick: for small orientation error, or convert measured q & predicted q to angle-axis.)
#         # For simplicity, we keep direct difference.

#         S = H @ self.P @ H.T + self.R
#         K = self.P @ H.T @ np.linalg.inv(S)

#         # 4) Correct
#         self.x = self.x + K @ y
#         # Re-normalize quaternion part
#         q = self.x[6:10]
#         q = normalize_quaternion(q)
#         self.x[6:10] = q

#         # 5) Cov update
#         I = np.eye(self.n_states)
#         self.P = (I - K@H) @ self.P

#         return self.x.copy(), self.P.copy()

#     #----------------------------------------
#     # Nonlinear measurement function h(x)
#     #----------------------------------------
#     def _h(self, x):
#         """
#         If measuring position + orientation quaternion:
#         z = [px, py, pz, qx, qy, qz, qw]. 
#         """
#         px,py,pz = x[0:3]
#         qx,qy,qz,qw = x[6:10]
#         return np.array([px,py,pz, qx,qy,qz,qw])

#     #----------------------------------------
#     # Jacobian of h wrt x
#     #----------------------------------------
#     def _H_jacobian(self, x):
#         """
#         partial of [px,py,pz, qx,qy,qz,qw] wrt x(= [px,py,pz, vx,vy,vz, qx,qy,qz,qw, wx,wy,wz])
#         is mostly identity in pos + orientation blocks.
#         """
#         H = np.zeros((7, self.n_states))
#         # position block
#         H[0,0] = 1.0
#         H[1,1] = 1.0
#         H[2,2] = 1.0
#         # orientation block
#         H[3,6] = 1.0  # d(qx)/d(qx)
#         H[4,7] = 1.0
#         H[5,8] = 1.0
#         H[6,9] = 1.0
#         return H

#     def _omega_mat(self, w):
#         """
#         Returns Omega(w) for w=(wx,wy,wz):
#             [[0, -wx, -wy, -wz],
#              [wx,  0,  wz, -wy],
#              [wy, -wz,  0,  wx],
#              [wz,  wy, -wx,  0 ]]
#         """
#         wx,wy,wz = w
#         return np.array([
#             [0,   -wx, -wy, -wz],
#             [wx,   0,   wz, -wy],
#             [wy,  -wz,  0,   wx],
#             [wz,   wy, -wx,  0 ]
#         ], dtype=float)





# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################



    # #-----------------------------------------
    # # Helper to retrieve current states
    # #-----------------------------------------
    # def get_state(self):
    #     """
    #     Return the entire state vector for external usage, e.g.:
    #     p = x[0:3]
    #     q = x[6:10]
    #     ...
    #     """
    #     return self.x.copy(), self.P.copy()
    
    # def update_tightly_coupled(self, feature_points, model_points, K, dist_coeffs=None):
    #     """
    #     Tightly coupled update using 2D feature points and 3D model points.
        
    #     Args:
    #         feature_points: 2D points in image coordinates (Nx2 array)
    #         model_points: 3D points in model coordinates (Nx3 array)
    #         K: Camera intrinsic matrix (3x3)
    #         dist_coeffs: Distortion coefficients (can be None)
            
    #     Returns:
    #         Updated state and covariance
    #     """
    #     if len(feature_points) < 4:
    #         raise ValueError("Need at least 4 point correspondences for tightly coupled update")
            
    #     # Extract current pose estimate from state
    #     position = self.x[0:3]
    #     quaternion = self.x[6:10]
        
    #     # Convert quaternion to rotation matrix
    #     R = quaternion_to_rotation_matrix(quaternion)
        
    #     # Initialize cumulative H, R, innovation
    #     num_points = len(feature_points)
    #     H_all = np.zeros((2 * num_points, self.n_states))
    #     R_all = np.zeros((2 * num_points, 2 * num_points))
    #     y_all = np.zeros(2 * num_points)
        
    #     # For each point correspondence, compute Jacobians and innovations
    #     for i, (feature_pt, model_pt) in enumerate(zip(feature_points, model_points)):
    #         # Project the 3D point to image plane
    #         proj_point = self._project_point(model_pt, position, R, K, dist_coeffs)
            
    #         # Compute innovation (measurement - prediction)
    #         innovation = feature_pt - proj_point
            
    #         # Compute Jacobians
    #         H_i = self._measurement_jacobian(model_pt, position, quaternion, K, dist_coeffs)
            
    #         # Store in the combined matrices
    #         H_all[2*i:2*i+2, :] = H_i
    #         R_all[2*i:2*i+2, 2*i:2*i+2] = self.R_tight
    #         y_all[2*i:2*i+2] = innovation
        
    #     # Standard EKF update equations
    #     S = H_all @ self.P @ H_all.T + R_all
        
    #     try:
    #         # Try to compute Kalman gain
    #         K_gain = self.P @ H_all.T @ np.linalg.inv(S)
            
    #         # Apply correction
    #         delta_x = K_gain @ y_all
            
    #         # Update state
    #         self.x[0:6] += delta_x[0:6]  # Position and velocity update
            
    #         # For quaternion update, we need to convert the delta rotation to quaternion
    #         # and apply it to the current quaternion
    #         dtheta = delta_x[6:9]  # Small rotation vector
    #         theta = np.linalg.norm(dtheta)
            
    #         if theta > 1e-10:  # Only apply if non-zero
    #             axis = dtheta / theta
    #             dq = np.zeros(4)
    #             dq[0:3] = np.sin(theta/2.0) * axis
    #             dq[3] = np.cos(theta/2.0)
                
    #             # Apply rotation to current quaternion
    #             q_new = self._quaternion_multiply(dq, self.x[6:10])
    #             self.x[6:10] = normalize_quaternion(q_new)
            
    #         # Update angular velocity components
    #         self.x[10:13] += delta_x[9:12]
            
    #         # Update covariance
    #         I = np.eye(self.n_states)
    #         self.P = (I - K_gain @ H_all) @ self.P
            
    #         return self.x.copy(), self.P.copy()
            
    #     except np.linalg.LinAlgError:
    #         # If matrix inversion fails, don't update the filter
    #         logger.warning("Tightly coupled update failed: Singular matrix in Kalman gain computation")
    #         raise RuntimeError("Matrix inversion failed in tightly coupled update")

    # # Add these methods to your KF_MK3.py class to complete the implementation

    # def _project_point(self, point_3d, position, rotation_matrix, K, dist_coeffs=None):
    #     """
    #     Project a 3D point to the image plane based on current camera pose.
        
    #     Args:
    #         point_3d: 3D point in model coordinates
    #         position: Camera position in world coordinates
    #         rotation_matrix: Camera rotation matrix
    #         K: Camera intrinsics matrix
    #         dist_coeffs: Distortion coefficients (optional)
            
    #     Returns:
    #         2D point in image coordinates
    #     """
    #     # Transform point to camera coordinates
    #     p_camera = rotation_matrix @ point_3d + position
        
    #     # Handle points behind the camera
    #     if p_camera[2] <= 0:
    #         logger.warning(f"Point behind camera: {p_camera}")
    #         return np.array([0.0, 0.0])  # Return a placeholder
        
    #     # Project to normalized image coordinates
    #     p_normalized = p_camera[:2] / p_camera[2]
        
    #     # Apply distortion if provided
    #     if dist_coeffs is not None and np.any(dist_coeffs):
    #         # Apply distortion model (this is simplified, full model should handle all distortion params)
    #         x, y = p_normalized
    #         r2 = x*x + y*y
            
    #         # Radial distortion
    #         k1, k2, p1, p2, k3 = dist_coeffs[:5]  # Extract first 5 coeffs
            
    #         # Radial component
    #         x_distorted = x * (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2)
    #         y_distorted = y * (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2)
            
    #         # Tangential component
    #         x_distorted += 2*p1*x*y + p2*(r2 + 2*x*x)
    #         y_distorted += p1*(r2 + 2*y*y) + 2*p2*x*y
            
    #         p_normalized = np.array([x_distorted, y_distorted])
        
    #     # Apply camera intrinsics
    #     p_image = np.array([
    #         K[0, 0] * p_normalized[0] + K[0, 2],
    #         K[1, 1] * p_normalized[1] + K[1, 2]
    #     ])
        
    #     return p_image

    # def _measurement_jacobian(self, point_3d, position, quaternion, K, dist_coeffs=None):
    #     """
    #     Compute the Jacobian of the measurement model for a single point.
        
    #     For a 3D point projected to 2D, we need:
    #     - Derivative of pixel coordinates with respect to state variables
        
    #     Args:
    #         point_3d: 3D point in model coordinates
    #         position: Camera position
    #         quaternion: Camera orientation as quaternion
    #         K: Camera intrinsics matrix
    #         dist_coeffs: Distortion coefficients (optional)
            
    #     Returns:
    #         2xN Jacobian matrix (N = number of state variables)
    #     """
    #     # Get camera rotation matrix from quaternion
    #     R = quaternion_to_rotation_matrix(quaternion)
        
    #     # Transform point to camera coordinates
    #     p_camera = R @ point_3d + position
        
    #     # Check if point is in front of camera
    #     if p_camera[2] <= 0:
    #         # Return zeros for Jacobian if point is behind camera
    #         return np.zeros((2, self.n_states))
        
    #     # Numerically compute Jacobian for better stability
    #     # We'll compute derivatives with respect to position and orientation
        
    #     H = np.zeros((2, self.n_states))
        
    #     # Derivatives with respect to position (px, py, pz)
    #     for i in range(3):
    #         # Small perturbation in position
    #         delta = 1e-6
    #         delta_pos = np.zeros(3)
    #         delta_pos[i] = delta
            
    #         # Project point with perturbed position
    #         pos_plus = position + delta_pos
    #         projected_plus = self._project_point(point_3d, pos_plus, R, K, dist_coeffs)
            
    #         # Compute numerical derivative
    #         H[0, i] = (projected_plus[0] - self._project_point(point_3d, position, R, K, dist_coeffs)[0]) / delta
    #         H[1, i] = (projected_plus[1] - self._project_point(point_3d, position, R, K, dist_coeffs)[1]) / delta
        
    #     # Derivatives with respect to quaternion (more complex)
    #     # We'll perturb each quaternion component slightly
    #     for i in range(4):
    #         delta = 1e-6
    #         delta_q = np.zeros(4)
    #         delta_q[i] = delta
            
    #         # Create perturbed quaternion (normalize to ensure valid rotation)
    #         q_plus = normalize_quaternion(quaternion + delta_q)
    #         R_plus = quaternion_to_rotation_matrix(q_plus)
            
    #         # Project point with perturbed orientation
    #         projected_plus = self._project_point(point_3d, position, R_plus, K, dist_coeffs)
            
    #         # Compute numerical derivative
    #         H[0, 6+i] = (projected_plus[0] - self._project_point(point_3d, position, R, K, dist_coeffs)[0]) / delta
    #         H[1, 6+i] = (projected_plus[1] - self._project_point(point_3d, position, R, K, dist_coeffs)[1]) / delta
        
    #     # The Jacobian with respect to velocity and angular velocity is zero
    #     # (these don't directly affect the measurement)
        
    #     return H

    # def _quaternion_multiply(self, q1, q2):
    #     """
    #     Multiply two quaternions: q1 * q2
        
    #     Args:
    #         q1: First quaternion [x, y, z, w]
    #         q2: Second quaternion [x, y, z, w]
            
    #     Returns:
    #         Quaternion product [x, y, z, w]
    #     """
    #     x1, y1, z1, w1 = q1
    #     x2, y2, z2, w2 = q2
        
    #     return np.array([
    #         w1*x2 + x1*w2 + y1*z2 - z1*y2,
    #         w1*y2 - x1*z2 + y1*w2 + z1*x2,
    #         w1*z2 + x1*y2 - y1*x2 + z1*w2,
    #         w1*w2 - x1*x2 - y1*y2 - z1*z2
    #     ])

    # def improved_update(self, z):
    #     """
    #     An improved version of the loosely coupled update that handles quaternion differences
    #     better by working with the error quaternion.
        
    #     z = [px, py, pz, qx, qy, qz, qw] (7 dims)
    #     """
    #     # Extract current state
    #     px, py, pz = self.x[0:3]
    #     q_state = self.x[6:10].copy()  # Current quaternion in state
        
    #     # Extract measured pose
    #     px_meas, py_meas, pz_meas = z[0:3]
    #     q_meas = z[3:7].copy()
        
    #     # Position innovation is straightforward
    #     pos_innovation = np.array([px_meas - px, py_meas - py, pz_meas - pz])
        
    #     # For quaternion, we need to compute error quaternion
    #     # q_e = q_meas * q_state^-1 (quaternion that rotates from state to measurement)
    #     q_state_conj = q_state.copy()
    #     q_state_conj[0:3] = -q_state_conj[0:3]  # Conjugate (inverse of unit quaternion)
        
    #     # Compute error quaternion using quaternion multiplication
    #     q_error = self._quaternion_multiply(q_meas, q_state_conj)
        
    #     # Convert error quaternion to a small rotation vector (if error is small)
    #     # For small rotations, the vector part is approximately half the rotation vector
    #     if q_error[3] < 0:
    #         q_error = -q_error  # Ensure shortest path
            
    #     rot_error = np.zeros(3)
    #     if q_error[3] > 0.99:
    #         # Very small rotation, use linear approximation
    #         rot_error = 2.0 * q_error[0:3]
    #     else:
    #         # Extract axis-angle representation
    #         theta = 2.0 * np.arccos(q_error[3])
    #         axis = q_error[0:3] / np.sin(theta/2.0)
    #         rot_error = theta * axis
            
    #     # Combine position and rotation errors
    #     innovation = np.concatenate([pos_innovation, rot_error])
        
    #     # Create appropriate Jacobian for this representation
    #     H = np.zeros((6, self.n_states))
    #     # Position block is straightforward
    #     H[0:3, 0:3] = np.eye(3)
        
    #     # Orientation block maps quaternion error to rotation vector
    #     # This is a simplified Jacobian that approximates the mapping
    #     # between quaternion and rotation vector near the identity
    #     H[3:6, 6:10] = self._quaternion_to_rotation_jacobian(q_state)
        
    #     # Use 6x6 measurement noise
    #     R_improved = np.eye(6) * 0.01  # Tuned based on expected measurement noise
        
    #     # Compute Kalman gain
    #     S = H @ self.P @ H.T + R_improved
    #     K = self.P @ H.T @ np.linalg.inv(S)
        
    #     # Apply correction to state
    #     delta_x = K @ innovation
        
    #     # Apply correction to position, velocity components directly
    #     self.x[0:6] += delta_x[0:6]
        
    #     # For quaternion, apply small rotation from delta_x[6:9]
    #     rot_update = delta_x[3:6]
    #     theta = np.linalg.norm(rot_update)
    #     if theta > 1e-10:
    #         axis = rot_update / theta
    #         dq = np.zeros(4)
    #         dq[0:3] = np.sin(theta/2.0) * axis
    #         dq[3] = np.cos(theta/2.0)
            
    #         # Apply rotation to current quaternion
    #         self.x[6:10] = self._quaternion_multiply(dq, q_state)
            
    #     # Apply correction to angular velocity
    #     self.x[10:13] += delta_x[6:9]
        
    #     # Re-normalize quaternion part
    #     self.x[6:10] = normalize_quaternion(self.x[6:10])
        
    #     # Update covariance
    #     I = np.eye(self.n_states)
    #     self.P = (I - K @ H) @ self.P
        
    #     return self.x.copy(), self.P.copy()

    # def _quaternion_to_rotation_jacobian(self, q):
    #     """
    #     Compute the Jacobian that maps changes in quaternion to changes in rotation vector.
    #     This is used for the improved update method.
        
    #     Args:
    #         q: Current quaternion state [x, y, z, w]
            
    #     Returns:
    #         3x4 Jacobian matrix
    #     """
    #     # This is an approximation valid for small rotations around the identity
    #     J = np.zeros((3, 4))
        
    #     # For small rotations, the vector part is approximately half the rotation vector
    #     # and the scalar part doesn't contribute much to small changes
    #     J[0, 0] = 2.0  # d(rot_x)/d(q_x)
    #     J[1, 1] = 2.0  # d(rot_y)/d(q_y)
    #     J[2, 2] = 2.0  # d(rot_z)/d(q_z)
        
    #     return J

    # def _quaternion_conjugate(self, q):
    #     """
    #     Return the conjugate of a quaternion [x,y,z,w] -> [-x,-y,-z,w]
    #     """
    #     return np.array([-q[0], -q[1], -q[2], q[3]])
