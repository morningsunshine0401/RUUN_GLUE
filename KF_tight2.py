import numpy as np
from scipy.linalg import block_diag
from utils import normalize_quaternion, quaternion_to_rotation_matrix , rotation_matrix_to_quaternion
import cv2

class MultExtendedKalmanFilter:
    """
    Multiplicative Extended Kalman Filter for tightly-coupled pose tracking
    Using constant velocity motion model with damping for formation flight
    """
    def __init__(self, dt):
        self.dt = dt
        
        # State: [position(3), velocity(3), quaternion(4), angular_velocity(3)]
        self.n_states = 13
        
        # Initialize state
        self.x = np.zeros(self.n_states)
        self.x[9] = 1.0  # quaternion w component = 1 (identity rotation)
        
        # Initialize covariance
        self.P = np.eye(self.n_states) * 0.1
        
        
        ## Process noise - tuned for slow, smooth formation flight
        ## DEFAULT
        # self.Q_p = np.eye(3) * 8e-5  # Position noise
        # self.Q_v = np.eye(3) * 8e-4  # Velocity noise
        # self.Q_q = np.eye(3) * 8e-5  # Quaternion noise
        # self.Q_w = np.eye(3) * 8e-4  # Angular velocity noise
        
        # Increase these values in __init__
        self.Q_p = np.eye(3) * 5e-1#4  # Position noise (increased)
        self.Q_v = np.eye(3) * 1e-1#3   # Velocity noise (increased)
        self.Q_q = np.eye(3) * 1 #5e-1#4  # Quaternion noise (increased)
        self.Q_w = np.eye(3) * 0.3 #1e-1#3   # Angular velocity noise (increased)
        
        # Add slight damping to velocity model for smoother behavior
        self.velocity_damping = 0.6#0.8 #0.98  # Slight damping for formation flight
        
        # Add adaptive measurement noise setting
        self.measurement_base_noise = 2#0.5 #1.0 #2.0
        self.measurement_error_scale = 1#0.2 #0.5  # How quickly to increase noise with error
        
        # Track feature consistency
        self.prev_feature_points = None
        self.feature_consistency_weights = None
    
    # DEFAULT
    def predict(self):
        """
        Predict step using constant velocity model with damping for position/velocity
        and quaternion kinematics for attitude
        """
        dt = self.dt
        
        # Extract state components
        p = self.x[0:3]  # Position
        v = self.x[3:6]  # Velocity
        q = self.x[6:10]  # Quaternion
        w = self.x[10:13]  # Angular velocity
        
        # 1) Position/velocity update using constant velocity model with damping
        p_new = p + v * dt
        v_new = v * self.velocity_damping  # Apply damping to velocity
        
        # 2) Quaternion update using angular velocity
        # Skew-symmetric matrix for angular velocity
        w_skew = np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])
        
        # Omega matrix for quaternion derivative
        Omega = np.zeros((4, 4))
        Omega[0:3, 0:3] = -w_skew
        Omega[0:3, 3] = w
        Omega[3, 0:3] = -w
        
        # Quaternion derivative
        q_dot = 0.5 * Omega @ q
        
        # Integrate quaternion
        q_new = q + q_dot * dt
        q_new = normalize_quaternion(q_new)
        
        # 3) Angular velocity with damping
        w_new = w * self.velocity_damping
        
        # Update state
        x_pred = np.concatenate([p_new, v_new, q_new, w_new])
        
        # Process noise covariance matrix
        Q = np.zeros((self.n_states, self.n_states))
        # Position noise increases with time squared
        Q[0:3, 0:3] = self.Q_p * dt**2 + self.Q_v * dt**4 / 4  
        # Velocity noise
        Q[3:6, 3:6] = self.Q_v * dt**2
        # Cross-correlation between position and velocity
        Q[0:3, 3:6] = self.Q_v * dt**3 / 2
        Q[3:6, 0:3] = self.Q_v * dt**3 / 2
        # Quaternion and angular velocity noise
        Q[6:9, 6:9] = self.Q_q * dt**2  # Only for vector part
        Q[10:13, 10:13] = self.Q_w * dt**2
        
        # State transition matrix (linear part)
        F = np.eye(self.n_states)
        # Position is affected by velocity
        F[0:3, 3:6] = np.eye(3) * dt
        # Add damping to velocity part
        F[3:6, 3:6] = np.eye(3) * self.velocity_damping
        # Add damping to angular velocity part
        F[10:13, 10:13] = np.eye(3) * self.velocity_damping
        
        # Update covariance
        P_pred = F @ self.P @ F.T + Q
        
        # Store prediction
        self.x = x_pred
        self.P = P_pred
        
        return x_pred.copy(), P_pred.copy()


    
    def calculate_feature_consistency(self, feature_points):
        """
        Calculate consistency weights for features based on temporal consistency
        """
        if self.prev_feature_points is None or len(self.prev_feature_points) == 0:
            # First time, initialize weights to 1.0
            self.feature_consistency_weights = np.ones(len(feature_points))
            self.prev_feature_points = feature_points.copy()
            return np.ones(len(feature_points))
        
        # Calculate correspondences and consistency
        weights = np.ones(len(feature_points))
        
        # Find closest previous point for each current point
        for i, curr_pt in enumerate(feature_points):
            min_dist = float('inf')
            for prev_pt in self.prev_feature_points:
                dist = np.linalg.norm(curr_pt - prev_pt)
                min_dist = min(min_dist, dist)
            
            # Weight based on consistency - more consistent (smaller distance) = higher weight
            if min_dist > 30: #20.0:  # Large jump - suspicious
                weights[i] = 0.6 #0.5
            elif min_dist > 15: #10.0:  # Medium jump
                weights[i] = 0.9 #0.8
            else:  # Small, expected movement
                weights[i] = 1.0
        
        # Update for next time
        self.prev_feature_points = feature_points.copy()
        self.feature_consistency_weights = weights
        
        return weights
        
    def update_tightly_coupled(self, feature_points, model_points, camera_matrix, distCoeffs):
        """
        Tightly-coupled update using feature points directly.
        Modified to perform a multiplicative quaternion update for orientation.
        """
        import numpy as np
        import cv2  # if you use solvePnP etc.

        # Current state
        x_pred = self.x
        P_pred = self.P

        # Number of feature points
        n_points = len(feature_points)
        if n_points < 3:
            # Not enough points for update
            return x_pred.copy(), P_pred.copy()

        # Calculate feature consistency weights
        consistency_weights = self.calculate_feature_consistency(feature_points)

        # Camera parameters
        f_x = camera_matrix[0, 0]
        f_y = camera_matrix[1, 1]
        c_x = camera_matrix[0, 2]
        c_y = camera_matrix[1, 2]

        # Extract pose from state
        position = x_pred[0:3]         # e.g. [x, y, z]
        quaternion = x_pred[6:10]      # e.g. [qx, qy, qz, qw]
        R = quaternion_to_rotation_matrix(quaternion)

        # Prepare measurement matrix, predicted measurements, actual measurements
        H = np.zeros((2 * n_points, self.n_states))
        z_pred = np.zeros(2 * n_points)
        z_meas = feature_points.flatten()

        # Measurement noise covariance (diagonal blocks for each feature)
        R_diag = np.eye(2 * n_points) * self.measurement_base_noise

        for i in range(n_points):
            # Transform 3D model point into camera frame
            p_model = model_points[i]
            p_cam = R @ p_model + position

            # Check if point is in front of camera
            if p_cam[2] > 1e-6:
                # Project into image
                u_pred = f_x * p_cam[0] / p_cam[2] + c_x
                v_pred = f_y * p_cam[1] / p_cam[2] + c_y

                # Store predicted measurement
                z_pred[2*i]   = u_pred
                z_pred[2*i+1] = v_pred

                # Jacobian of projection wrt point in camera frame
                J_proj = np.zeros((2, 3))
                J_proj[0, 0] = f_x / p_cam[2]
                J_proj[0, 2] = -f_x * p_cam[0] / (p_cam[2]**2)
                J_proj[1, 1] = f_y / p_cam[2]
                J_proj[1, 2] = -f_y * p_cam[1] / (p_cam[2]**2)

                # Position part of the Jacobian: d(p_cam)/d(position) = I
                # Then multiplied by J_proj
                H[2*i:2*i+2, 0:3] = J_proj

                # If your velocity is in [3:6], you might skip or fill with zeros
                # H[2*i:2*i+2, 3:6] = 0.0  # (assuming no direct velocity effect)

                # Orientation error part: for a small rotation-vector error δθ
                # p_cam = R * p_model + position
                # d(p_cam)/d(δθ) ≈ R * (p_model x)  => the skew
                p_skew = np.array([
                    [0,          -p_model[2],  p_model[1]],
                    [p_model[2], 0,           -p_model[0]],
                    [-p_model[1], p_model[0],  0        ]
                ])
                # Factor of 2 is often used with an "additive quaternion" error parameterization;
                # for a true small-angle MEKF, you might see R @ p_skew (no factor 2).
                # We'll keep the original approach from your code for continuity:
                J_quat = 2.0 * (R @ p_skew)

                # Multiply by the 2D projection Jacobian
                H[2*i:2*i+2, 6:9] = J_proj @ J_quat

                # Reprojection error
                u_actual, v_actual = feature_points[i]
                error = np.sqrt((u_pred - u_actual)**2 + (v_pred - v_actual)**2)

                # Adjust measurement noise based on error + consistency
                consistency_factor = 1.0 / consistency_weights[i]  # Inverse of consistency weight
                k = self.measurement_base_noise + (max(error - 2.0, 0.0))**2 * self.measurement_error_scale
                k *= consistency_factor
                R_diag[2*i:2*i+2, 2*i:2*i+2] = np.eye(2) * k
            else:
                # Point behind camera -> extremely high uncertainty
                R_diag[2*i:2*i+2, 2*i:2*i+2] = np.eye(2) * 1e6

        # Innovation
        y = z_meas - z_pred

        # Compute Kalman gain
        S = H @ P_pred @ H.T + R_diag
        K = P_pred @ H.T @ np.linalg.inv(S)

        # ---- Extract the state correction dx from K @ y ----
        dx = K @ y  # dimension = self.n_states

        # ------------------------------------------------------------
        #  APPLY THE CORRECTION MULTIPLICATIVELY TO ORIENTATION
        # ------------------------------------------------------------
        x_upd = x_pred.copy()

        # 1) Position update (and velocity if you store it in [3:6])
        x_upd[0:3] += dx[0:3]
        # If you have velocity at [3:6], you might do:
        # x_upd[3:6] += dx[3:6]

        # 2) Orientation update (MEKF style)
        #    dx[6:9] is our small rotation vector δθ
        q_pred = x_pred[6:10]
        dtheta = dx[6:9]
        delta_angle = np.linalg.norm(dtheta)

        if delta_angle > 1e-12:
            axis = dtheta / delta_angle
            half_angle = 0.5 * delta_angle
            sin_half  = np.sin(half_angle)
            delta_q = np.array([
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
                np.cos(half_angle)
            ])
        else:
            # Very small rotation -> identity quaternion
            delta_q = np.array([0.0, 0.0, 0.0, 1.0])

        # Quaternion multiplication: q_new = delta_q ⊗ q_pred
        # (Make sure the ordering [qx,qy,qz,qw] is consistent)
        q_new = np.array([
            delta_q[3]*q_pred[0] + delta_q[0]*q_pred[3] + delta_q[1]*q_pred[2] - delta_q[2]*q_pred[1],
            delta_q[3]*q_pred[1] - delta_q[0]*q_pred[2] + delta_q[1]*q_pred[3] + delta_q[2]*q_pred[0],
            delta_q[3]*q_pred[2] + delta_q[0]*q_pred[1] - delta_q[1]*q_pred[0] + delta_q[2]*q_pred[3],
            delta_q[3]*q_pred[3] - delta_q[0]*q_pred[0] - delta_q[1]*q_pred[1] - delta_q[2]*q_pred[2]
        ])

        # Normalize the updated quaternion
        q_new = normalize_quaternion(q_new)
        x_upd[6:10] = q_new

        # 3) (Optional) If you have angular velocity in your state (e.g. [10:13]),
        #    you could incorporate a small correction from dtheta / dt.

        # ------------------------------------------------------------
        #  Finally, update the covariance (Joseph form)
        # ------------------------------------------------------------
        I_KH = np.eye(self.n_states) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ R_diag @ K.T
        # Force symmetry
        P_upd = 0.5 * (P_upd + P_upd.T)

        # Store and return
        self.x = x_upd
        self.P = P_upd

        return x_upd.copy(), P_upd.copy()

    
    def update_loosely_coupled(self, pose_measurement):
        """
        Loosely-coupled update using direct pose measurement
        Modified for formation flight scenario with improved robustness
        
        Args:
            pose_measurement: np.array of shape (7,) containing [x, y, z, qx, qy, qz, qw]
        
        Returns:
            Updated state and covariance
        """
        # Current state estimate
        x_pred = self.x
        P_pred = self.P
        
        # Extract position and quaternion from measurement
        pos_meas = pose_measurement[0:3]
        quat_meas = pose_measurement[3:7]
        
        # Normalize quaternion measurement
        quat_meas = normalize_quaternion(quat_meas)
        
        # Measurement matrix for position (simple direct observation)
        H_pos = np.zeros((3, self.n_states))
        H_pos[0:3, 0:3] = np.eye(3)  # Position elements
        
        # For quaternion, we need to handle it differently due to multiplicative error
        # We'll use a linearized approach around the current quaternion
        
        # Create measurement function and Jacobian for quaternion part
        # Quaternion error is expressed in vector space (3 elements)
        H_quat = np.zeros((3, self.n_states))
        H_quat[0:3, 6:9] = np.eye(3)  # Quaternion vector part
        
        # Compute quaternion error in vector space (using small-angle approximation)
        # q_err = q_meas ⊗ q_pred^(-1) ≈ [rotation vector, 1]
        q_pred = x_pred[6:10]
        # q_pred_conj = np.array([q_pred[0], q_pred[1], q_pred[2], -q_pred[3]])  # Conjugate
        
        # # For small rotations, vector part of error quaternion approximates the rotation vector
        # q_err_vec = 2 * (quat_meas[0:3] * q_pred[3] - q_pred[0:3] * quat_meas[3] - 
        #                 np.cross(quat_meas[0:3], q_pred[0:3]))

        q_pred_conj = np.array([-q_pred[0], -q_pred[1], -q_pred[2], q_pred[3]])  # Conjugate (sign corrected)

        # For small rotations, vector part of error quaternion approximates the rotation vector
        q_err_vec = 2 * (quat_meas[0:3] * q_pred[3] - q_pred[0:3] * quat_meas[3] - 
                        np.cross(quat_meas[0:3], q_pred[0:3])) 
        
        # Combine position and orientation measurements
        z = np.concatenate([pos_meas, q_err_vec])
        
        # Expected measurement (for position part)
        z_pred_pos = x_pred[0:3]
        
        # Expected measurement (for quaternion part)
        z_pred_quat = np.zeros(3)  # Should be zero if prediction matches measurement
        
        # Combined expected measurement
        z_pred = np.concatenate([z_pred_pos, z_pred_quat])
        
        # Combine H matrices
        H = np.vstack([H_pos, H_quat])
        
        # Measurement noise - adjusted for formation flight scenario
        R_pos = np.eye(3) * 0.002 #0.01  # Position uncertainty (in meters)
        R_quat = np.eye(3) * 0.002 #0.01  # Orientation uncertainty (in radians)
        R = block_diag(R_pos, R_quat)
        
        # Innovation
        y = z - z_pred
        
        # Check for large innovations (outlier detection)
        pos_innovation = np.linalg.norm(y[0:3])
        quat_innovation = np.linalg.norm(y[3:6])
        
        # Threshold for rejecting significant outliers
        pos_threshold = 2#0.5  # meters
        quat_threshold = 1.6#0.4  # radians
        
        if pos_innovation > pos_threshold or quat_innovation > quat_threshold:
            # Increase measurement noise for suspicious measurements
            R_scale = max(1.0, (pos_innovation / pos_threshold)**2, (quat_innovation / quat_threshold)**2)
            R *= R_scale
        
        # Innovation covariance
        S = H @ P_pred @ H.T + R
        
        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # Update state
        dx = K @ y
        
        # Apply state correction
        x_upd = x_pred.copy()
        
        # Update position and velocities directly
        x_upd[0:3] += dx[0:3]
        
        # For quaternion, we need to apply multiplicative update
        # Convert vector update to error quaternion
        angle = np.linalg.norm(dx[3:6])
        if angle > 1e-10:
            axis = dx[3:6] / angle
        else:
            axis = np.array([1, 0, 0])
            angle = 0
        
        # Convert axis-angle to quaternion
        q_update = np.zeros(4)
        q_update[0:3] = axis * np.sin(angle/2)
        q_update[3] = np.cos(angle/2)
        
        # Apply quaternion update: q = q_update ⊗ q_pred
        q_new = np.array([
            q_update[3]*q_pred[0] + q_update[0]*q_pred[3] + q_update[1]*q_pred[2] - q_update[2]*q_pred[1],
            q_update[3]*q_pred[1] - q_update[0]*q_pred[2] + q_update[1]*q_pred[3] + q_update[2]*q_pred[0],
            q_update[3]*q_pred[2] + q_update[0]*q_pred[1] - q_update[1]*q_pred[0] + q_update[2]*q_pred[3],
            q_update[3]*q_pred[3] - q_update[0]*q_pred[0] - q_update[1]*q_pred[1] - q_update[2]*q_pred[2]
        ])
        
        # Normalize updated quaternion
        q_new = normalize_quaternion(q_new)
        
        # Update quaternion in state
        x_upd[6:10] = q_new
        
        # Update covariance using Joseph form for better numerical stability
        I_KH = np.eye(self.n_states) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
        
        # Ensure symmetry
        P_upd = (P_upd + P_upd.T) / 2
        
        # Store updates
        self.x = x_upd
        self.P = P_upd
        
        return x_upd.copy(), P_upd.copy()
    
    def get_state_uncertainty(self):
        """
        Get the uncertainty (standard deviation) of the current state
        Useful for monitoring filter performance
        """
        # Extract diagonal elements of covariance matrix (variances)
        variances = np.diag(self.P)
        
        # Convert to standard deviations
        stds = np.sqrt(variances)
        
        # Structure the uncertainties
        uncertainty = {
            'position': stds[0:3],
            'velocity': stds[3:6],
            'quaternion': stds[6:10],
            'angular_velocity': stds[10:13]
        }
        
        return uncertainty
    
    def improved_update(self, pose_measurement):
        """
        Improved update that balances simplicity and proper Kalman filtering
        """
        # Current state estimate
        x_pred = self.x
        P_pred = self.P
        
        # Extract measurements
        pos_meas = pose_measurement[0:3]
        quat_meas = normalize_quaternion(pose_measurement[3:7])
        
        # Extract current state
        pos_pred = x_pred[0:3]
        q_pred = x_pred[6:10]
        
        # 1. Position update - use standard Kalman filter approach
        # Measurement matrix for position
        H_pos = np.zeros((3, self.n_states))
        H_pos[0:3, 0:3] = np.eye(3)
        
        # Measurement noise
        R_pos = np.eye(3) * 0.001  # Small measurement noise for responsive updates
        
        # Kalman gain for position
        S_pos = H_pos @ P_pred @ H_pos.T + R_pos
        K_pos = P_pred @ H_pos.T @ np.linalg.inv(S_pos)
        
        # Position update - fix the dimension mismatch
        pos_innovation = pos_meas - pos_pred
        dx_pos = K_pos @ pos_innovation  # This applies the innovation to all state variables
        
        # Update entire state with position part
        x_upd = x_pred.copy() + dx_pos
        
        # 2. Quaternion update - use simplified approach
        # Calculate quaternion difference (error quaternion)
        q_pred_conj = np.array([-q_pred[0], -q_pred[1], -q_pred[2], q_pred[3]])
        
        # Quaternion multiplication: q_err = q_meas ⊗ q_pred_conj
        q_err = np.array([
            quat_meas[3]*q_pred_conj[0] + quat_meas[0]*q_pred_conj[3] + quat_meas[1]*q_pred_conj[2] - quat_meas[2]*q_pred_conj[1],
            quat_meas[3]*q_pred_conj[1] - quat_meas[0]*q_pred_conj[2] + quat_meas[1]*q_pred_conj[3] + quat_meas[2]*q_pred_conj[0],
            quat_meas[3]*q_pred_conj[2] + quat_meas[0]*q_pred_conj[1] - quat_meas[1]*q_pred_conj[0] + quat_meas[2]*q_pred_conj[3],
            quat_meas[3]*q_pred_conj[3] - quat_meas[0]*q_pred_conj[0] - quat_meas[1]*q_pred_conj[1] - quat_meas[2]*q_pred_conj[2]
        ])
        
        # Convert to axis-angle (only using the vector part for small rotations)
        rot_vec = q_err[0:3] * 2.0  # Simplified approximation for small rotations
        
        # Apply rotation update with damping
        alpha_rot = 0.5  # Rotation update strength (0-1)
        
        # Create rotation update quaternion
        angle = np.linalg.norm(rot_vec * alpha_rot)
        if angle > 1e-6:
            axis = rot_vec / np.linalg.norm(rot_vec)
            q_update = np.zeros(4)
            q_update[0:3] = axis * np.sin(angle/2.0)
            q_update[3] = np.cos(angle/2.0)
        else:
            q_update = np.array([0, 0, 0, 1])  # Identity quaternion
        
        # Apply quaternion update: q_new = q_update ⊗ q_pred
        q_new = np.array([
            q_update[3]*q_pred[0] + q_update[0]*q_pred[3] + q_update[1]*q_pred[2] - q_update[2]*q_pred[1],
            q_update[3]*q_pred[1] - q_update[0]*q_pred[2] + q_update[1]*q_pred[3] + q_update[2]*q_pred[0],
            q_update[3]*q_pred[2] + q_update[0]*q_pred[1] - q_update[1]*q_pred[0] + q_update[2]*q_pred[3],
            q_update[3]*q_pred[3] - q_update[0]*q_pred[0] - q_update[1]*q_pred[1] - q_update[2]*q_pred[2]
        ])
        
        # Normalize the quaternion and update state
        x_upd[6:10] = normalize_quaternion(q_new)
        
        # 4. Update covariance
        # We'll use a simplified approach
        alpha_P = 0.9  # Covariance update factor (0-1)
        P_upd = alpha_P * P_pred  # Simplified covariance update
        
        # Store updates
        self.x = x_upd
        self.P = P_upd
        
        return x_upd.copy(), P_upd.copy()
    

    def enhanced_tightly_coupled(self, feature_points, model_points, camera_matrix, distCoeffs):
        """
        Enhanced tightly-coupled update using feature points directly.
        Incorporates a proper MEKF-like multiplicative orientation update
        rather than directly adding to the quaternion.
        """
        import numpy as np
        import cv2  # If you're using OpenCV's solvePnP

        # Current state estimate
        x_pred = self.x
        P_pred = self.P

        # Number of feature points
        n_points = len(feature_points)
        if n_points < 3:
            # Not enough points for update
            return x_pred.copy(), P_pred.copy()

        # Camera parameters
        f_x = camera_matrix[0, 0]
        f_y = camera_matrix[1, 1]
        c_x = camera_matrix[0, 2]
        c_y = camera_matrix[1, 2]

        # Extract pose
        position = x_pred[0:3]
        quaternion = x_pred[6:10]  # [qx, qy, qz, qw]
        R = quaternion_to_rotation_matrix(quaternion)  # Implement/Use your quaternion->Rotation function

        # Step 1: Filter out outliers & compute weights (as you did before)
        consistency_weights = self.calculate_feature_consistency(feature_points)
        filtered_points = []
        filtered_model_points = []
        filtered_weights = []

        for i in range(n_points):
            p_model = model_points[i]
            p_cam = R @ p_model + position

            if p_cam[2] > 1e-6:
                u_pred = f_x * p_cam[0] / p_cam[2] + c_x
                v_pred = f_y * p_cam[1] / p_cam[2] + c_y

                u_actual, v_actual = feature_points[i]
                error = np.sqrt((u_pred - u_actual)**2 + (v_pred - v_actual)**2)

                # Simple outlier rejection
                if error < 20.0:
                    point_weight = 1.0 / (1.0 + error * 0.1) * consistency_weights[i]
                    filtered_points.append([u_actual, v_actual])
                    filtered_model_points.append(p_model)
                    filtered_weights.append(point_weight)

        # If too few good points, do nothing
        if len(filtered_points) < 3:
            return x_pred.copy(), P_pred.copy()

        # Convert to arrays
        filtered_points = np.array(filtered_points)
        filtered_model_points = np.array(filtered_model_points)
        filtered_weights = np.array(filtered_weights)

        # Step 2: Build the measurement Jacobian (H), predicted measurement (z_pred), and measurement covariance (R_diag)
        n_good_points = len(filtered_points)
        H = np.zeros((2 * n_good_points, self.n_states))  # 2D measurement per point
        z_pred = np.zeros(2 * n_good_points)
        z_meas = filtered_points.flatten()
        R_diag = np.eye(2 * n_good_points)  # We'll populate with per-point noise

        for i in range(n_good_points):
            p_model = filtered_model_points[i]
            p_cam = R @ p_model + position

            # Project to image
            u_pred = f_x * p_cam[0] / p_cam[2] + c_x
            v_pred = f_y * p_cam[1] / p_cam[2] + c_y
            z_pred[2*i]   = u_pred
            z_pred[2*i+1] = v_pred

            # Jacobian of the projection wrt camera-frame coords
            # (A standard pinhole projection linearization)
            J_proj = np.zeros((2, 3))
            J_proj[0, 0] = f_x / p_cam[2]
            J_proj[0, 2] = -f_x * p_cam[0] / (p_cam[2]**2)
            J_proj[1, 1] = f_y / p_cam[2]
            J_proj[1, 2] = -f_y * p_cam[1] / (p_cam[2]**2)

            # Jacobian wrt position
            # position -> p_cam is just p_model rotated by R, so
            # d(p_cam)/d(position) = I
            H[2*i:2*i+2, 0:3] = J_proj

            # Jacobian wrt orientation error (3x1 rotation vector in MEKF style).
            # We do p_cam = R * p_model + position. If dR/d(3D rot vector) is complicated,
            # you can approximate as R @ p_skew * some factor. For the standard "additive
            # quaternion" approach we had: J_quat = 2 * R @ p_skew. For the MEKF small-angle
            # approach, it is also a small rotation around the predicted orientation.
            #
            # A typical approach is:
            p_skew = np.array([
                [0,          -p_model[2],  p_model[1]],
                [p_model[2], 0,           -p_model[0]],
                [-p_model[1], p_model[0],  0        ]
            ])
            # For small rotation dθ near identity, d(R)/dθ ≈ R * p_skew
            # We'll let the local 3D error be mapped by J_proj @ R @ p_skew
            J_orient = J_proj @ (R @ p_skew)

            H[2*i:2*i+2, 6:9] = J_orient

            # Per-point measurement noise scaled by weight
            w_i = filtered_weights[i]
            R_diag[2*i:2*i+2, 2*i:2*i+2] = np.eye(2) * (0.5 / w_i)

        # Step 3: Compute innovation
        y = z_meas - z_pred

        # Step 4: Kalman update
        S = H @ P_pred @ H.T + R_diag
        K = P_pred @ H.T @ np.linalg.inv(S)
        dx = K @ y  # This is the incremental update to [pos, vel, ... , 3D-orient-error, etc.]

        # Step 5: Apply state update. For position, velocity, etc. we can do normal addition:
        x_upd = x_pred.copy()
        x_upd[0:6] += dx[0:6]

        # ---- The CRITICAL PART: multiplicative update for quaternion ----
        q_pred = x_pred[6:10]
        dtheta = dx[6:9]
        delta_angle = np.linalg.norm(dtheta)
        if delta_angle > 1e-10:
            axis = dtheta / delta_angle
            half_angle = 0.5 * delta_angle
            sin_half = np.sin(half_angle)
            delta_q = np.array([axis[0]*sin_half,
                                axis[1]*sin_half,
                                axis[2]*sin_half,
                                np.cos(half_angle)])
        else:
            # No significant rotation
            delta_q = np.array([0., 0., 0., 1.])

        # Quaternion multiplication: q_new = delta_q ⊗ q_pred
        # (Make sure your quaternion ordering matches your internal convention)
        q_new = np.array([
            delta_q[3]*q_pred[0] + delta_q[0]*q_pred[3] + delta_q[1]*q_pred[2] - delta_q[2]*q_pred[1],
            delta_q[3]*q_pred[1] - delta_q[0]*q_pred[2] + delta_q[1]*q_pred[3] + delta_q[2]*q_pred[0],
            delta_q[3]*q_pred[2] + delta_q[0]*q_pred[1] - delta_q[1]*q_pred[0] + delta_q[2]*q_pred[3],
            delta_q[3]*q_pred[3] - delta_q[0]*q_pred[0] - delta_q[1]*q_pred[1] - delta_q[2]*q_pred[2]
        ])

        # Normalize
        q_new = normalize_quaternion(q_new)
        x_upd[6:10] = q_new

        # Optionally, if you have angular rates in your state at [10:13], you can
        # incorporate a small correction from dtheta/self.dt, exactly as you did in
        # the loosely coupled version (adjusted to your actual state indexing).
        # For example:
        # rotation_rate_gain = 0.1
        # implied_angular_velocity = dtheta / self.dt * rotation_rate_gain
        # x_upd[10:13] = 0.8*x_pred[10:13] + 0.2*implied_angular_velocity

        # Step 6: Update covariance (Joseph form to preserve symmetry)
        I_KH = np.eye(self.n_states) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ R_diag @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)  # Force symmetry

        # Store
        self.x = x_upd
        self.P = P_upd

        return x_upd.copy(), P_upd.copy()

    
    #DEFAULT
    def proper_mekf_update(self, pose_measurement):
        """
        Full Multiplicative Extended Kalman Filter update for quaternion-based pose estimation
        """
        # Current state estimate
        x_pred = self.x
        P_pred = self.P
        
        # Extract measurements
        pos_meas = pose_measurement[0:3]
        quat_meas = normalize_quaternion(pose_measurement[3:7])
        
        # Extract current state
        pos_pred = x_pred[0:3]
        q_pred = x_pred[6:10]
        
        # 1. Build complete measurement model Jacobian
        H = np.zeros((6, self.n_states))  # 3 for position, 3 for orientation error
        
        # Position part is linear
        H[0:3, 0:3] = np.eye(3)
        
        # Orientation part - this is critical and was missing
        # For MEKF, this maps the global error quaternion to the local rotation vector
        H[3:6, 6:9] = np.eye(3)  # Direct mapping to quaternion vector part
        
        # 2. Define measurement noise
        R_meas = np.zeros((6, 6))
        R_meas[0:3, 0:3] = np.eye(3) * 0.001  # Position measurement noise
        R_meas[3:6, 3:6] = np.eye(3) * 0.005  # Orientation measurement noise
        
        # 3. Compute quaternion error (convert to 3D rotation vector)
        # First, ensure we're using the right convention [x,y,z,w]
        q_pred_conj = np.array([-q_pred[0], -q_pred[1], -q_pred[2], q_pred[3]])
        
        # Quaternion multiplication: q_err = q_meas ⊗ q_pred_conj
        q_err = np.array([
            quat_meas[3]*q_pred_conj[0] + quat_meas[0]*q_pred_conj[3] + quat_meas[1]*q_pred_conj[2] - quat_meas[2]*q_pred_conj[1],
            quat_meas[3]*q_pred_conj[1] - quat_meas[0]*q_pred_conj[2] + quat_meas[1]*q_pred_conj[3] + quat_meas[2]*q_pred_conj[0],
            quat_meas[3]*q_pred_conj[2] + quat_meas[0]*q_pred_conj[1] - quat_meas[1]*q_pred_conj[0] + quat_meas[2]*q_pred_conj[3],
            quat_meas[3]*q_pred_conj[3] - quat_meas[0]*q_pred_conj[0] - quat_meas[1]*q_pred_conj[1] - quat_meas[2]*q_pred_conj[2]
        ])
        
        # Small angle approximation for error quaternion
        # Check if we need to negate the quaternion (to ensure shortest path)
        if q_err[3] < 0:
            q_err = -q_err
        
        # For small angles: rotation vector ≈ 2 * quaternion vector part
        delta_theta = 2 * q_err[0:3]
        
        # 4. Build complete measurement residual
        residual = np.zeros(6)
        residual[0:3] = pos_meas - pos_pred
        residual[3:6] = delta_theta
        
        # 5. Compute Kalman gain
        S = H @ P_pred @ H.T + R_meas
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # 6. Compute state update
        dx = K @ residual
        
        # 7. Apply state update
        x_upd = x_pred.copy()
        
        # Linear state updates (position, velocity)
        x_upd[0:6] += dx[0:6]
        
        # Quaternion update (multiplicative)
        delta_q = np.zeros(4)
        delta_angle = np.linalg.norm(dx[6:9])
        
        if delta_angle > 1e-10:  # Use smaller threshold
            delta_axis = dx[6:9] / delta_angle
            delta_q[0:3] = delta_axis * np.sin(delta_angle/2)
            delta_q[3] = np.cos(delta_angle/2)
        else:
            delta_q = np.array([0, 0, 0, 1])  # Identity quaternion
        
        # Apply quaternion update: q_new = delta_q ⊗ q_pred
        q_new = np.array([
            delta_q[3]*q_pred[0] + delta_q[0]*q_pred[3] + delta_q[1]*q_pred[2] - delta_q[2]*q_pred[1],
            delta_q[3]*q_pred[1] - delta_q[0]*q_pred[2] + delta_q[1]*q_pred[3] + delta_q[2]*q_pred[0],
            delta_q[3]*q_pred[2] + delta_q[0]*q_pred[1] - delta_q[1]*q_pred[0] + delta_q[2]*q_pred[3],
            delta_q[3]*q_pred[3] - delta_q[0]*q_pred[0] - delta_q[1]*q_pred[1] - delta_q[2]*q_pred[2]
        ])
        
        # Update angular velocity with some influence from the quaternion error
        # This helps the filter track rotational motion better
        rotation_rate_gain = 0.1
        implied_angular_velocity = delta_theta / self.dt * rotation_rate_gain
        x_upd[10:13] = x_upd[10:13] * 0.8 + implied_angular_velocity * 0.2
        
        # Normalize and update quaternion
        x_upd[6:10] = normalize_quaternion(q_new)
        
        # 8. Update covariance using Joseph form
        I_KH = np.eye(self.n_states) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ R_meas @ K.T
        
        # Ensure symmetry
        P_upd = (P_upd + P_upd.T) / 2
        
        # Store updates
        self.x = x_upd
        self.P = P_upd
        
        return x_upd.copy(), P_upd.copy()

    