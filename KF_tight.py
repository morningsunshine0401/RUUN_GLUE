# Modify KF_Q.py to implement MEKF with constant velocity model

import numpy as np
from scipy.linalg import block_diag
from utils import normalize_quaternion, quaternion_to_rotation_matrix

class MultExtendedKalmanFilter:
    """
    Multiplicative Extended Kalman Filter for tightly-coupled pose tracking
    Using constant velocity motion model for aircraft
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
        
        # Process noise
        self.Q_p = np.eye(3) * 1e-4  # Position noise
        self.Q_v = np.eye(3) * 1e-3  # Velocity noise
        self.Q_q = np.eye(3) * 1e-4  # Quaternion noise
        self.Q_w = np.eye(3) * 1e-3  # Angular velocity noise
    
    def predict(self):
        """
        Predict step using constant velocity model for position/velocity
        and quaternion kinematics for attitude
        """
        dt = self.dt
        
        # Extract state components
        p = self.x[0:3]  # Position
        v = self.x[3:6]  # Velocity
        q = self.x[6:10]  # Quaternion
        w = self.x[10:13]  # Angular velocity
        
        # 1) Position/velocity update using constant velocity model
        p_new = p + v * dt
        v_new = v  # Constant velocity assumption
        
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
        
        # 3) Angular velocity (constant)
        w_new = w
        
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
        
        # Update covariance
        P_pred = F @ self.P @ F.T + Q
        
        # Store prediction
        self.x = x_pred
        self.P = P_pred
        
        return x_pred.copy(), P_pred.copy()
    
    def update_tightly_coupled(self, feature_points, model_points, camera_matrix, distCoeffs):
        """
        Tightly-coupled update using feature points directly
        """
        # Current state
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
        quaternion = x_pred[6:10]
        R = quaternion_to_rotation_matrix(quaternion)
        
        # Measurement matrix
        H = np.zeros((2 * n_points, self.n_states))
        
        # Expected measurements
        z_pred = np.zeros(2 * n_points)
        
        # Actual measurements
        z = feature_points.flatten()
        
        # Measurement noise covariance - will be updated based on reprojection errors
        R_diag = np.eye(2 * n_points) * 1e-2
        
        # Process each point
        for i in range(n_points):
            # Transform 3D point to camera frame
            p_model = model_points[i]
            p_cam = R @ p_model + position
            
            # Project point to image plane
            if p_cam[2] > 1e-6:  # Check if point is in front of camera
                u_pred = f_x * p_cam[0] / p_cam[2] + c_x
                v_pred = f_y * p_cam[1] / p_cam[2] + c_y
                
                # Store predicted measurement
                z_pred[2*i] = u_pred
                z_pred[2*i+1] = v_pred
                
                # Calculate Jacobian
                # Jacobian of projection with respect to point in camera frame
                J_proj = np.zeros((2, 3))
                J_proj[0, 0] = f_x / p_cam[2]
                J_proj[0, 2] = -f_x * p_cam[0] / (p_cam[2]**2)
                J_proj[1, 1] = f_y / p_cam[2]
                J_proj[1, 2] = -f_y * p_cam[1] / (p_cam[2]**2)
                
                # Jacobian of point in camera frame with respect to position
                J_pos = np.eye(3)
                
                # Jacobian of point in camera frame with respect to quaternion
                # Using small-angle approximation for quaternion error
                p_skew = np.array([
                    [0, -p_model[2], p_model[1]],
                    [p_model[2], 0, -p_model[0]],
                    [-p_model[1], p_model[0], 0]
                ])
                J_quat = 2 * R @ p_skew
                
                # Fill measurement Jacobian
                H[2*i:2*i+2, 0:3] = J_proj @ J_pos  # Position part
                # Skip velocity part (no direct influence)
                H[2*i:2*i+2, 6:9] = J_proj @ J_quat  # Quaternion part (3 vector components)
                # Skip angular velocity part
                
                # Calculate reprojection error
                u_actual, v_actual = feature_points[i]
                error = np.sqrt((u_pred - u_actual)**2 + (v_pred - v_actual)**2)
                
                # Dynamic measurement covariance based on reprojection error
                k = (max(error - 5.0, 1.0))**2 / 6.0  # From thesis
                R_diag[2*i:2*i+2, 2*i:2*i+2] = np.eye(2) * k
                
            else:
                # Point behind camera, set very high uncertainty
                R_diag[2*i:2*i+2, 2*i:2*i+2] = np.eye(2) * 1e6
        
        # Innovation
        y = z - z_pred
        
        # Kalman gain
        S = H @ P_pred @ H.T + R_diag
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # Update state
        x_upd = x_pred + K @ y
        
        # Normalize quaternion
        x_upd[6:10] = normalize_quaternion(x_upd[6:10])
        
        # Update covariance using Joseph form for better numerical stability
        I_KH = np.eye(self.n_states) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ R_diag @ K.T
        
        # Ensure symmetry
        P_upd = (P_upd + P_upd.T) / 2
        
        # Store updates
        self.x = x_upd
        self.P = P_upd
        
        return x_upd.copy(), P_upd.copy()
    
    def update_loosely_coupled(self, pose_measurement):
        """
        Loosely-coupled update using direct pose measurement
        
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
        q_pred_conj = np.array([q_pred[0], q_pred[1], q_pred[2], -q_pred[3]])  # Conjugate
        
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
        
        # Measurement noise - adjust based on PnP quality
        R_pos = np.eye(3) * 0.01  # Position uncertainty (in meters)
        R_quat = np.eye(3) * 0.01  # Orientation uncertainty (in radians)
        R = block_diag(R_pos, R_quat)
        
        # Innovation
        y = z - z_pred
        
        # Innovation covariance
        S = H @ P_pred @ H.T + R
        
        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # Update state
        dx = K @ y
        
        # Apply state correction
        x_upd = x_pred.copy()
        
        # Update position and velocities directly
        #x_upd[0:6] += dx[0:3]  # Only apply position part of correction
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