import numpy as np
import logging
from utils import normalize_quaternion, quaternion_to_rotation_matrix
import cv2

logger = logging.getLogger(__name__)


class KalmanFilterFeatureBased:
    """
    EKF with state x = [px, py, pz, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]^T
    
    Measurement model directly uses 2D feature points (ui, vi) from image.
    """
    def __init__(self, dt, camera_matrix, dist_coeffs):
        self.dt = dt
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # State: [p(3), v(3), q(4), w(3)]
        self.n_states = 13
        self.x = np.zeros(self.n_states)
        self.x[9] = 1.0  # quaternion w=1, x=y=z=0

        # Covariance
        self.P = np.eye(self.n_states) * 0.1

        # Tuning: process noise (adjust based on your target's motion characteristics)
        self.Q = np.eye(self.n_states)
        # Position noise
        self.Q[0:3, 0:3] *= 0.01  # Position uncertainty growth
        # Velocity noise
        self.Q[3:6, 3:6] *= 0.1   # Velocity uncertainty growth
        # Quaternion noise
        self.Q[6:10, 6:10] *= 0.001  # Small orientation uncertainty
        # Angular velocity noise
        self.Q[10:13, 10:13] *= 0.01  # Angular velocity uncertainty growth
        
        # Measurement noise - will be set dynamically based on number of features
        self.R = None  # This will be initialized during update() based on feature count
        
        # Store 3D model points (in target frame) for future use
        self.model_points_3d = None
        
        # Debug info
        self.last_innovation = None
        self.last_measurement = None
        self.last_predicted_measurement = None

    def set_model_points(self, points_3d):
        """Set the 3D model points (in target body frame) that will be tracked"""
        self.model_points_3d = points_3d
        logger.info(f"Set model with {len(points_3d)} 3D points")
    
    def predict(self):
        """
        EKF predict step: x_k+1^- = f(x_k^+)
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
        F = np.eye(self.n_states)

        # dx/dv = dt
        for i in range(3):
            F[i, 3+i] = dt

        # For a proper EKF, we would compute the full Jacobian including orientation
        # integration derivatives, but this simplified approach works well for small dt

        # 3) Update covariance
        P_pred = F @ self.P @ F.T + self.Q

        # Store
        self.x = x_pred
        self.P = P_pred

        return self.x.copy(), self.P.copy()

    def update(self, points_2d, feature_indices, feature_confidences=None):
        """
        Update step using direct 2D feature measurements
        
        Args:
            points_2d: Nx2 array of observed 2D feature points [u,v]
            feature_indices: Indices that map each 2D point to the corresponding 3D model point
            feature_confidences: Optional confidence values for each feature (0-1)
        """
        if self.model_points_3d is None:
            logger.error("No 3D model points set. Call set_model_points() first.")
            return self.x.copy(), self.P.copy()
            
        if len(points_2d) < 4:
            logger.warning(f"Not enough features ({len(points_2d)}) for update. Skipping.")
            return self.x.copy(), self.P.copy()
            
        # Extract 3D points corresponding to the observed features
        points_3d = self.model_points_3d[feature_indices]
        
        # 1. Build measurement vector z (all observed 2D points flattened to [u1,v1,u2,v2,...])
        z = points_2d.reshape(-1)
        n_measurements = len(z)
        
        # 2. Compute predicted measurement h(x) by projecting 3D points using current state
        z_pred = self._project_points(points_3d)
        
        # 3. Compute measurement Jacobian H = dh/dx
        H = self._compute_projection_jacobian(points_3d)
        
        # 4. Create measurement noise matrix (can be adjusted based on feature confidence)
        base_pixel_variance = 2.0  # Base variance in pixels^2
        R = np.eye(n_measurements) * base_pixel_variance
        
        # If feature confidences provided, adjust noise accordingly
        if feature_confidences is not None:
            for i, conf in enumerate(feature_confidences):
                # Higher confidence -> lower noise, scale inversely
                R[2*i:2*i+2, 2*i:2*i+2] *= (1.0 / max(0.1, conf))
        
        # 5. Compute innovation and its covariance
        y = z - z_pred
        S = H @ self.P @ H.T + R
        
        # Store for debugging
        self.last_innovation = y
        self.last_measurement = z
        self.last_predicted_measurement = z_pred
        
        # 6. Compute Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 7. Update state
        self.x = self.x + K @ y
        
        # 8. Normalize quaternion part
        q = normalize_quaternion(self.x[6:10])
        self.x[6:10] = q
        
        # 9. Update covariance
        I = np.eye(self.n_states)
        self.P = (I - K @ H) @ self.P
        
        # Log info
        mean_innovation = np.mean(np.abs(y))
        logger.debug(f"Update with {len(points_2d)} features. Mean innovation: {mean_innovation:.2f} px")
        
        return self.x.copy(), self.P.copy()

    # def _project_points(self, points_3d):
    #     """
    #     Project 3D points (in target body frame) to camera image plane 
    #     using current state estimate
        
    #     Returns flattened array of [u1,v1,u2,v2,...]
    #     """
    #     # Extract pose from state
    #     position = self.x[0:3]
    #     q = self.x[6:10]
        
    #     # Create rotation matrix from quaternion
    #     R = quaternion_to_rotation_matrix(q)
        
    #     # Transform points from target body frame to camera frame
    #     points_cam = []
    #     for p in points_3d:
    #         # Rotate and translate
    #         p_cam = R @ p + position
    #         points_cam.append(p_cam)
        
    #     points_cam = np.array(points_cam).reshape(-1, 3)
        
    #     # Project to image plane using camera matrix
    #     if self.dist_coeffs is not None:
    #         points_2d, _ = cv2.projectPoints(
    #             points_cam, np.zeros(3), np.zeros(3), 
    #             self.camera_matrix, self.dist_coeffs
    #         )
    #         points_2d = points_2d.reshape(-1, 2)
    #     else:
    #         # Manual projection without distortion
    #         fx = self.camera_matrix[0, 0]
    #         fy = self.camera_matrix[1, 1]
    #         cx = self.camera_matrix[0, 2]
    #         cy = self.camera_matrix[1, 2]
            
    #         points_2d = []
    #         for p in points_cam:
    #             if p[2] <= 0:  # Point is behind camera
    #                 u, v = 0, 0  # Arbitrary value, will be far from actual
    #             else:
    #                 u = fx * p[0] / p[2] + cx
    #                 v = fy * p[1] / p[2] + cy
    #             points_2d.append([u, v])
    #         points_2d = np.array(points_2d)
        
    #     # Return flattened array [u1,v1,u2,v2,...]
    #     return points_2d.reshape(-1)

    def _project_points(self, points_3d):
        """
        Project 3D points (in target body frame) to camera image plane 
        using current state estimate
        
        Returns flattened array of [u1,v1,u2,v2,...]
        """
        # Extract pose from state - represents target pose in camera frame
        t_target_in_camera = self.x[0:3].copy()  # Copy to avoid modifying state
        q_target_in_camera = self.x[6:10].copy()  # Copy to avoid modifying state
        
        # ADDED: Debug logging
        logger.debug(f"Projecting with t={t_target_in_camera}, q={q_target_in_camera}")
        
        # Convert quaternion to rotation matrix - this rotates from target frame to camera frame
        R_target_to_camera = quaternion_to_rotation_matrix(q_target_in_camera)
        
        # Convert rotation matrix to rvec (axis-angle representation)
        rvec, _ = cv2.Rodrigues(R_target_to_camera)
        tvec = t_target_in_camera.reshape(3, 1)
        
        # Project points using OpenCV's projectPoints
        # This handles both the transformation from body to camera frame AND the projection
        points_2d, _ = cv2.projectPoints(
            points_3d,            # Points in target body frame (objectPoints)
            rvec,                 # Rotation from target to camera frame (rvec)
            tvec,                 # Translation from target to camera frame (tvec)
            self.camera_matrix,   # Camera intrinsics (K)
            self.dist_coeffs      # Distortion coefficients
        )
        
        # Return flattened array [u1,v1,u2,v2,...]
        return points_2d.reshape(-1)
        
    def _compute_projection_jacobian(self, points_3d):
        """
        Compute the Jacobian of the projection function with respect to state
        """
        n_points = len(points_3d)
        n_measurements = 2 * n_points  # Each point has u,v
        
        # Initialize Jacobian matrix
        H = np.zeros((n_measurements, self.n_states))
        
        # Current state
        position = self.x[0:3]
        q = self.x[6:10]
        R = quaternion_to_rotation_matrix(q)
        
        # Camera intrinsics
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # For each 3D point
        for i, p_body in enumerate(points_3d):
            # Transform to camera frame
            p_cam = R @ p_body + position
            X, Y, Z = p_cam
            
            if Z <= 0:  # Point is behind camera
                # Set zero Jacobian for this point
                continue
                
            # Pre-compute common terms
            Z_inv = 1.0 / Z
            Z_inv_2 = Z_inv * Z_inv
            
            # Jacobian of projection with respect to camera point [X, Y, Z]
            # du/dX = fx/Z, du/dY = 0, du/dZ = -fx*X/Z^2
            # dv/dX = 0, dv/dY = fy/Z, dv/dZ = -fy*Y/Z^2
            J_proj = np.array([
                [fx * Z_inv, 0, -fx * X * Z_inv_2],
                [0, fy * Z_inv, -fy * Y * Z_inv_2]
            ])
            
            # Jacobian of camera point w.r.t. position is identity
            J_pos = np.eye(3)
            
            # Jacobian of camera point w.r.t. quaternion
            # This is an approximation - we're using small angle approximation
            # For a full derivation, see quaternion rotation Jacobians in robotics literature
            
            # For position derivatives (much easier)
            H[2*i:2*i+2, 0:3] = J_proj @ J_pos
            
            # For orientation derivatives (approximate)
            # R * p is equivalent to q * p * q^{-1} in quaternion form
            # Computing exact derivatives of this is complex
            # We'll use a numerical approximation for simplicity
            eps = 1e-6
            for j in range(4):  # quaternion components
                q_perturbed = q.copy()
                q_perturbed[j] += eps
                q_perturbed = normalize_quaternion(q_perturbed)
                R_perturbed = quaternion_to_rotation_matrix(q_perturbed)
                p_cam_perturbed = R_perturbed @ p_body + position
                
                # Project perturbed point
                if p_cam_perturbed[2] <= 0:
                    # Skip this component
                    continue
                
                u_perturbed = fx * p_cam_perturbed[0] / p_cam_perturbed[2] + cx
                v_perturbed = fy * p_cam_perturbed[1] / p_cam_perturbed[2] + cy
                
                # Compute numerical derivative
                u_orig = fx * X * Z_inv + cx
                v_orig = fy * Y * Z_inv + cy
                
                du_dq = (u_perturbed - u_orig) / eps
                dv_dq = (v_perturbed - v_orig) / eps
                
                H[2*i, 6+j] = du_dq
                H[2*i+1, 6+j] = dv_dq
        
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

    def get_reprojection_errors(self, points_2d, feature_indices):
        """
        Compute reprojection errors for the given 2D points
        
        Returns:
            errors: List of reprojection errors (in pixels)
            projected_points: 2D points projected using current state
        """
        if self.model_points_3d is None:
            return None, None
            
        # Get 3D model points for these features
        points_3d = self.model_points_3d[feature_indices]
        
        # Project using current state
        projected_points_flat = self._project_points(points_3d)
        projected_points = projected_points_flat.reshape(-1, 2)
        
        # Calculate reprojection errors
        errors = np.linalg.norm(points_2d - projected_points, axis=1)
        
        return errors, projected_points
    
    def debug_projection(self, frame, points_2d_observed, points_3d, save_path=None):
        """
        Draw observed points (green) and projected points (red) with lines between them
        """
        debug_frame = frame.copy()
        if len(debug_frame.shape) == 2:  # Convert grayscale to color for visualization
            debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)
        
        # Ensure correct data types
        points_2d_observed = points_2d_observed.astype(np.float32)
        points_3d = points_3d.astype(np.float32)
        
        # Extract pose from state
        t = self.x[0:3].copy()
        q = self.x[6:10].copy()
        
        # Convert quaternion to rotation matrix
        R = quaternion_to_rotation_matrix(q)
        
        # Convert to rvec/tvec
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)
        
        # Project 3D points
        projected_points, _ = cv2.projectPoints(
            points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        points_2d_projected = projected_points.reshape(-1, 2)
        
        # Draw observed points (green), projected points (red) and lines between them
        for i, (pt_obs, pt_proj) in enumerate(zip(points_2d_observed, points_2d_projected)):
            # Convert to integer coordinates
            pt_obs = tuple(map(int, pt_obs))
            pt_proj = tuple(map(int, pt_proj))
            
            # Draw observed point (green)
            cv2.circle(debug_frame, pt_obs, 4, (0, 255, 0), -1)
            
            # Draw projected point (red)
            cv2.circle(debug_frame, pt_proj, 4, (0, 0, 255), -1)
            
            # Draw line between them (magenta)
            cv2.line(debug_frame, pt_obs, pt_proj, (255, 0, 255), 1)
            
            # Add distance text
            dist = np.linalg.norm(np.array(pt_obs) - np.array(pt_proj))
            cv2.putText(debug_frame, f"{dist:.1f}", 
                    ((pt_obs[0] + pt_proj[0])//2, (pt_obs[1] + pt_proj[1])//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add overall metrics
        errors = np.linalg.norm(points_2d_observed - points_2d_projected, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        cv2.putText(debug_frame, f"Mean error: {mean_error:.1f}px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(debug_frame, f"Max error: {max_error:.1f}px", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save if path provided
        if save_path is not None:
            cv2.imwrite(save_path, debug_frame)
        
        return debug_frame, mean_error
    
    def compare_with_pnp(self, points_2d, points_3d):
        """Compare Kalman filter results with direct PnP solution"""
        # Convert points to the right format
        objectPoints = points_3d.reshape(-1, 1, 3)#.astype(np.float32)
        imagePoints = points_2d.reshape(-1, 1, 2).astype(np.float32)
        
        # Solve PnP directly - note the different parameter order in older OpenCV versions
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=objectPoints,
            imagePoints=imagePoints,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
            reprojectionError=5.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP
         # Note: parameter name difference from newer versions
        )
    
        if not success or inliers is None or len(inliers) < 4:
            logger.warning("Direct PnP failed or insufficient inliers")
            return None, None, None
        
        # Refine with LM
        points_3d_inliers = objectPoints[inliers].reshape(-1, 3)
        points_2d_inliers = imagePoints[inliers].reshape(-1, 2)
        
        try:
            success, rvec_refined, tvec_refined = cv2.solvePnPRansac(
                objectPoints=objectPoints,
                imagePoints=imagePoints,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                reprojectionError=5.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_P3P
            # Note: parameter name difference from newer versions
            )
            if not success:
                rvec_refined, tvec_refined = rvec, tvec
        except Exception as e:
            logger.warning(f"PnP refinement failed: {str(e)}")
            rvec_refined, tvec_refined = rvec, tvec
        
        # Project points using direct PnP solution
        projected_2d, _ = cv2.projectPoints(
            points_3d, rvec_refined, tvec_refined, self.camera_matrix, self.dist_coeffs)
        projected_2d = projected_2d.reshape(-1, 2)
        
        # Calculate reprojection errors
        errors = np.linalg.norm(points_2d - projected_2d, axis=1)
        mean_error = np.mean(errors)
        
        logger.info(f"Direct PnP mean reprojection error: {mean_error:.2f} px")
        
        # Get rotation matrix
        R, _ = cv2.Rodrigues(rvec_refined)
        
        return R, tvec_refined.flatten(), mean_error
    
    def check_3d_model(self):
        """Verify 3D model points are reasonable"""
        if self.model_points_3d is None:
            logger.warning("No 3D model points set")
            return
            
        points_3d = self.model_points_3d
        min_coords = np.min(points_3d, axis=0)
        max_coords = np.max(points_3d, axis=0)
        mean_coords = np.mean(points_3d, axis=0)
        std_coords = np.std(points_3d, axis=0)
        
        logger.info("\n3D Model Statistics:")
        logger.info(f"Min: {min_coords}")
        logger.info(f"Max: {max_coords}")
        logger.info(f"Mean: {mean_coords}")
        logger.info(f"Std Dev: {std_coords}")
        logger.info(f"Number of points: {len(points_3d)}")
        
        # Check for extreme values that might indicate scaling issues
        if np.max(np.abs(points_3d)) > 10:
            logger.warning("3D points have unusually large values (>10)")
        if np.max(np.abs(points_3d)) < 0.01:
            logger.warning("3D points have unusually small values (<0.01)")