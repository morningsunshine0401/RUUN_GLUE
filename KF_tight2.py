# Modify KF_Q.py to implement MEKF with constant velocity model

import numpy as np
from scipy.linalg import block_diag
from utils import quaternion_to_rotation_matrix  #, normalize_quaternion,

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
        q_new = MultExtendedKalmanFilter.normalize_quaternion(q_new)
        
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
    
    # 1. First, ensure quaternion utility functions are correct and consistent
    @staticmethod
    def normalize_quaternion(q):
        """
        Normalize a quaternion to unit norm.
        Input/output quaternion format: [x, y, z, w]
        """
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            # If the quaternion is nearly zero, return identity
            return np.array([0.0, 0.0, 0.0, 1.0])
        return q / norm

    @staticmethod
    def quaternion_multiply(q1, q2):
        """
        Multiply two quaternions.
        Input/output quaternion format: [x, y, z, w]
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

    @staticmethod
    def quaternion_inverse(q):
        """
        Compute the inverse of a quaternion.
        For unit quaternions, this is the conjugate.
        Input/output quaternion format: [x, y, z, w]
        """
        q = MultExtendedKalmanFilter.normalize_quaternion(q)
        return np.array([-q[0], -q[1], -q[2], q[3]])

    @staticmethod
    def quaternion_error(q1, q2):
        """
        Compute the quaternion that rotates from q2 to q1.
        q_err = q1 ⊗ q2^(-1)
        Input/output quaternion format: [x, y, z, w]
        """
        q2_inv = MultExtendedKalmanFilter.quaternion_inverse(q2)
        q_err = MultExtendedKalmanFilter.quaternion_multiply(q1, q2_inv)
        return q_err

    @staticmethod
    def quaternion_to_rotation_vector(q):
        """
        Convert quaternion to rotation vector (axis-angle representation).
        For small rotations, this approximates the vector part of the quaternion.
        Input quaternion format: [x, y, z, w]
        """
        q = MultExtendedKalmanFilter.normalize_quaternion(q)
        if q[3] < 0:
            q = -q  # Ensure w is positive for consistent results
            
        sin_theta_2 = np.linalg.norm(q[0:3])
        if sin_theta_2 < 1e-10:
            return np.zeros(3)
            
        theta = 2 * np.arctan2(sin_theta_2, q[3])
        if abs(theta) < 1e-10:
            return np.zeros(3)
            
        axis = q[0:3] / sin_theta_2
        return axis * theta

    @staticmethod
    def rotation_vector_to_quaternion(rv):
        """
        Convert rotation vector (axis-angle) to quaternion.
        Output quaternion format: [x, y, z, w]
        """
        theta = np.linalg.norm(rv)
        if theta < 1e-10:
            return np.array([0.0, 0.0, 0.0, 1.0])
            
        axis = rv / theta
        sin_theta_2 = np.sin(theta / 2)
        cos_theta_2 = np.cos(theta / 2)
        
        q = np.zeros(4)
        q[0:3] = axis * sin_theta_2
        q[3] = cos_theta_2
        
        return q

    # 2. Improved update_tightly_coupled method

    def update_tightly_coupled(self, feature_points, model_points, camera_matrix, distCoeffs):
        """
        Tightly-coupled update using feature points directly with improved quaternion handling
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
        quaternion = MultExtendedKalmanFilter.normalize_quaternion(quaternion) # Ensure normalized
        R = quaternion_to_rotation_matrix(quaternion)
        
        # Define error state dimensions (12D state: position, velocity, 3D rotation, angular velocity)
        error_state_dim = self.n_states - 1  # One less dimension because we use 3D rotation vector instead of 4D quaternion
        
        # Define indices for error state representation
        error_state_indices = list(range(0, 6)) + list(range(6, 9)) + list(range(10, 13))
        
        # Measurement matrix - maps from error state to measurement
        H = np.zeros((2 * n_points, error_state_dim))
        
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
                J_quat = 2 * R @ p_skew  # Relates rotation vector to point change
                
                # Fill measurement Jacobian - using error state notation
                # Position part
                H[2*i:2*i+2, 0:3] = J_proj @ J_pos
                # Skip velocity part (no direct influence)
                # Use indices for error-state quaternion (3D rotation vector)
                H[2*i:2*i+2, 6:9] = J_proj @ J_quat
                # Skip angular velocity part
                
                # Calculate reprojection error
                u_actual, v_actual = feature_points[i]
                error = np.sqrt((u_pred - u_actual)**2 + (v_pred - v_actual)**2)
                
                # Dynamic measurement covariance based on reprojection error
                k = (max(error - 5.0, 1.0))**2 / 6.0
                R_diag[2*i:2*i+2, 2*i:2*i+2] = np.eye(2) * k
                
            else:
                # Point behind camera, set very high uncertainty
                R_diag[2*i:2*i+2, 2*i:2*i+2] = np.eye(2) * 1e6
        
        # Innovation
        y = z - z_pred
        
        # Extract error state covariance from full state covariance
        P_error = P_pred[np.ix_(error_state_indices, error_state_indices)]
        
        # Innovation covariance
        S = H @ P_error @ H.T + R_diag
        
        # Kalman gain for error state
        K_error = P_error @ H.T @ np.linalg.inv(S)
        
        # Compute error state correction
        dx_error = K_error @ y
        
        # Construct full state correction vector
        dx = np.zeros(self.n_states)
        dx[0:6] = dx_error[0:6]  # Position and velocity directly
        # Skip quaternion part for now (handled separately)
        dx[10:13] = dx_error[9:12]  # Angular velocity
        
        # Convert rotation vector correction to quaternion update
        dq = MultExtendedKalmanFilter.rotation_vector_to_quaternion(dx_error[6:9])
        
        # Update state
        x_upd = x_pred.copy()
        
        # Position, velocity, and angular velocity update
        x_upd[0:6] = x_pred[0:6] + dx[0:6]
        x_upd[10:13] = x_pred[10:13] + dx[10:13]
        
        # Quaternion update using multiplicative approach
        q_pred = x_pred[6:10]
        q_upd = MultExtendedKalmanFilter.quaternion_multiply(dq, q_pred)
        x_upd[6:10] = MultExtendedKalmanFilter.normalize_quaternion(q_upd)
        
        # Update error state covariance
        I_KH = np.eye(error_state_dim) - K_error @ H
        P_error_upd = I_KH @ P_error @ I_KH.T + K_error @ R_diag @ K_error.T
        
        # Ensure symmetry
        P_error_upd = (P_error_upd + P_error_upd.T) / 2
        
        # Put back into full state covariance
        P_upd = P_pred.copy()
        for i, ei in enumerate(error_state_indices):
            for j, ej in enumerate(error_state_indices):
                P_upd[ei, ej] = P_error_upd[i, j]
        
        # Store updates
        self.x = x_upd
        self.P = P_upd
        
        return x_upd.copy(), P_upd.copy()


    def update_loosely_coupled(self, pose_measurement):
        """
        Loosely-coupled update using direct pose measurement with improved quaternion handling
        
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
        quat_meas = MultExtendedKalmanFilter.normalize_quaternion(quat_meas)
        
        # Extract pose from predicted state
        pos_pred = x_pred[0:3]
        quat_pred = MultExtendedKalmanFilter.normalize_quaternion(x_pred[6:10])
        
        # Define error state dimensions (12D state: position, velocity, 3D rotation, angular velocity)
        error_state_dim = self.n_states - 1
        
        # Define indices for error state representation
        error_state_indices = list(range(0, 6)) + list(range(6, 9)) + list(range(10, 13))
        
        # Compute position error (direct subtraction)
        pos_error = pos_meas - pos_pred
        
        # Compute quaternion error (proper quaternion difference)
        q_err = MultExtendedKalmanFilter.quaternion_error(quat_meas, quat_pred)
        
        # Convert quaternion error to rotation vector (3D error state)
        rot_vec_err = MultExtendedKalmanFilter.quaternion_to_rotation_vector(q_err)
        
        # Combine into measurement vector in error state
        y = np.concatenate([pos_error, rot_vec_err])
        
        # Measurement matrix for error state
        H = np.zeros((6, error_state_dim))
        H[0:3, 0:3] = np.eye(3)  # Position part
        H[3:6, 6:9] = np.eye(3)  # Rotation vector part
        
        # Measurement noise - adjust based on PnP quality
        R_pos = np.eye(3) * 0.01  # Position uncertainty (in meters)
        R_rot = np.eye(3) * 0.01  # Orientation uncertainty (in radians)
        R = block_diag(R_pos, R_rot)
        
        # Extract error state covariance from full state covariance
        P_error = P_pred[np.ix_(error_state_indices, error_state_indices)]
        
        # Innovation covariance
        S = H @ P_error @ H.T + R
        
        # Kalman gain for error state
        K_error = P_error @ H.T @ np.linalg.inv(S)
        
        # Compute error state correction
        dx_error = K_error @ y
        
        # Construct full state correction vector
        dx = np.zeros(self.n_states)
        dx[0:6] = dx_error[0:6]  # Position and velocity directly
        # Skip quaternion part for now (handled separately)
        dx[10:13] = dx_error[9:12]  # Angular velocity
        
        # Convert rotation vector correction to quaternion update
        dq = MultExtendedKalmanFilter.rotation_vector_to_quaternion(dx_error[6:9])
        
        # Update state
        x_upd = x_pred.copy()
        
        # Position, velocity, and angular velocity update
        x_upd[0:6] = x_pred[0:6] + dx[0:6]
        x_upd[10:13] = x_pred[10:13] + dx[10:13]
        
        # Quaternion update using multiplicative approach
        q_pred = x_pred[6:10]
        q_upd = MultExtendedKalmanFilter.quaternion_multiply(dq, q_pred)
        x_upd[6:10] = MultExtendedKalmanFilter.normalize_quaternion(q_upd)
        
        # Update error state covariance
        I_KH = np.eye(error_state_dim) - K_error @ H
        P_error_upd = I_KH @ P_error @ I_KH.T + K_error @ R @ K_error.T
        
        # Ensure symmetry
        P_error_upd = (P_error_upd + P_error_upd.T) / 2
        
        # Put back into full state covariance
        P_upd = P_pred.copy()
        for i, ei in enumerate(error_state_indices):
            for j, ej in enumerate(error_state_indices):
                P_upd[ei, ej] = P_error_upd[i, j]
        
        # Store updates
        self.x = x_upd
        self.P = P_upd
        
        return x_upd.copy(), P_upd.copy()