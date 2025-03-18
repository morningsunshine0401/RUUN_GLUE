import threading
import cv2
import torch
# Disable gradient computation globally
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)
import time
import numpy as np
from scipy.spatial import cKDTree
from utils import (
    frame2tensor,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    normalize_quaternion
)

from KF_tight2 import MultExtendedKalmanFilter
import matplotlib.cm as cm
from models.utils import make_matching_plot_fast
import logging

# Import LightGlue and SuperPoint
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

# Configure logging
logging.basicConfig(
    #level=logging.DEBUG,
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pose_estimator.log"),  # Logs will be saved in this file
        logging.StreamHandler()  # Logs will also be printed to console
    ]
)
logger = logging.getLogger(__name__)

class PoseEstimator:
    def __init__(self, opt, device):
        self.session_lock = threading.Lock()
        self.opt = opt
        self.device = device
        self.initial_z_set = False  # Flag for first-frame Z override (if desired)
        self.kf_initialized = False  # To track if Kalman filter was ever updated
        self.pred_only =0

        logger.info("Initializing PoseEstimator with separate SuperPoint and LightGlue models")

        # Load anchor (leader) image
        self.anchor_image = cv2.imread(opt.anchor)
        assert self.anchor_image is not None, f'Failed to load anchor image at {opt.anchor}'
        self.anchor_image = self._resize_image(self.anchor_image, opt.resize)
        logger.info(f"Loaded and resized anchor image from {opt.anchor}")

        # Initialize SuperPoint and LightGlue models
        #self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        self.extractor = SuperPoint(max_num_keypoints=512).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)
        logger.info("Initialized SuperPoint and LightGlue models")

        # We will store the anchor's 2D/3D keypoints here.
        # For your anchor image, you can define them directly or load from a file.
        anchor_keypoints_2D = np.array([
            [511, 293], #0
            [591, 284], #
            [587, 330], #
            [413, 249], #
            [602, 348], #
            [715, 384], #
            [598, 298], #
            [656, 171], #
            [805, 213],#
            [703, 392],#10 
            [523, 286],#
            [519, 327],#12
            [387, 289],#13
            [727, 126],# 14
            [425, 243],# 15
            [636, 358],#
            [745, 202],#
            [595, 388],#
            [436, 260],#
            [539, 313], # 20
            [795, 220],# 
            [351, 291],#
            [665, 165],# 
            [611, 353], #
            [650, 377],# 25
            [516, 389],## 
            [727, 143], #
            [496, 378], #
            [575, 312], #
            [617, 368],# 30
            [430, 312], #
            [480, 281], #
            [834, 225], #
            [469, 339], #
            [705, 223], # 35
            [637, 156], 
            [816, 414], 
            [357, 195], 
            [752, 77], 
            [642, 451]
        ], dtype=np.float32)

        anchor_keypoints_3D = np.array([
            [-0.014,  0.000,  0.042],
            [ 0.025, -0.014, -0.011],
            [-0.014,  0.000, -0.042],
            [-0.014,  0.000,  0.156],
            [-0.023,  0.000, -0.065],
            [ 0.000,  0.000, -0.156],
            [ 0.025,  0.000, -0.015],
            [ 0.217,  0.000,  0.070],#
            [ 0.230,  0.000, -0.070],
            [-0.014,  0.000, -0.156],
            [ 0.000,  0.000,  0.042],
            [-0.057, -0.018, -0.010],
            [-0.074, -0.000,  0.128],
            [ 0.206, -0.070, -0.002],
            [-0.000, -0.000,  0.156],
            [-0.017, -0.000, -0.092],
            [ 0.217, -0.000, -0.027],#
            [-0.052, -0.000, -0.097],
            [-0.019, -0.000,  0.128],
            [-0.035, -0.018, -0.010],
            [ 0.217, -0.000, -0.070],#
            [-0.080, -0.000,  0.156],
            [ 0.230, -0.000,  0.070],
            [-0.023, -0.000, -0.075],
            [-0.029, -0.000, -0.127],
            [-0.090, -0.000, -0.042],
            [ 0.206, -0.055, -0.002],
            [-0.090, -0.000, -0.015],
            [ 0.000, -0.000, -0.015],
            [-0.037, -0.000, -0.097],
            [-0.074, -0.000,  0.074],
            [-0.019, -0.000,  0.074],
            [ 0.230, -0.000, -0.113],#
            [-0.100, -0.030,  0.000],#
            [ 0.170, -0.000, -0.015],
            [ 0.230, -0.000,  0.113],
            [-0.000, -0.025, -0.240],
            [-0.000, -0.025,  0.240],
            [ 0.243, -0.104,  0.000],
            [-0.080, -0.000, -0.156]
        ], dtype=np.float32)

        # Set anchor features (run SuperPoint on anchor, match to known 2D->3D)
        self._set_anchor_features(
            anchor_bgr_image=self.anchor_image,
            anchor_keypoints_2D=anchor_keypoints_2D,
            anchor_keypoints_3D=anchor_keypoints_3D
        )

        # Suppose the anchor was taken at ~ yaw=0°, pitch=-20°, roll=0°, in radians:
        self.anchor_viewpoint_eulers = np.array([0.0, -0.35, 0.0], dtype=np.float32)
        # This is just an example – adjust to your actual anchor viewpoint.

        # Initialize Kalman filter
        #self.kf_pose = self._init_kalman_filter()
        #self.kf_pose_first_update = True 
        #logger.info("Kalman filter initialized")

        # Add these lines:
        self.kf_initialized = False
        self.tracking_3D_points = None
        self.tracking_2D_points = None
        
        # Replace your existing Kalman filter init with MEKF
        # We'll initialize it properly when we get the first good pose
        self.mekf = None

    def reinitialize_anchor(self, new_anchor_path, new_2d_points, new_3d_points):
        """
        Re-load a new anchor image and re-compute relevant data (2D->3D correspondences).
        Called on-the-fly (e.g. after 200 frames).
        """
        try:
            logger.info(f"Re-initializing anchor with new image: {new_anchor_path}")

            # 1. Load new anchor image
            new_anchor_image = cv2.imread(new_anchor_path)
            if new_anchor_image is None:
                logger.error(f"Failed to load new anchor image at {new_anchor_path}")
                raise ValueError(f"Cannot read anchor image from {new_anchor_path}")

            logger.info(f"Successfully loaded anchor image: shape={new_anchor_image.shape}")

            # 2. Resize the image
            new_anchor_image = self._resize_image(new_anchor_image, self.opt.resize)
            logger.info(f"Resized anchor image to {new_anchor_image.shape}")

            # 3. Update anchor image (with lock)
            lock_acquired = self.session_lock.acquire(timeout=5.0)
            if not lock_acquired:
                logger.error("Could not acquire session lock to update anchor image (timeout)")
                raise TimeoutError("Lock acquisition timed out during anchor update")
                
            try:
                self.anchor_image = new_anchor_image
                logger.info("Anchor image updated")
            finally:
                self.session_lock.release()

            # 4. Recompute anchor features with the new image and 2D/3D
            logger.info(f"Setting anchor features with {len(new_2d_points)} 2D points and {len(new_3d_points)} 3D points")
            success = self._set_anchor_features(
                anchor_bgr_image=new_anchor_image,
                anchor_keypoints_2D=new_2d_points,
                anchor_keypoints_3D=new_3d_points
            )
            
            if not success:
                logger.error("Failed to set anchor features")
                raise RuntimeError("Failed to set anchor features")

            logger.info("Anchor re-initialization complete.")
            return True
            
        except Exception as e:
            logger.error(f"Error during anchor reinitialization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _convert_cv2_to_tensor(self, image):
        """Convert OpenCV BGR image to RGB tensor"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to float and normalize to [0, 1]
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        # Permute dimensions from (H, W, C) to (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)

    def _set_anchor_features(self, anchor_bgr_image, anchor_keypoints_2D, anchor_keypoints_3D):
        """
        Run SuperPoint on the anchor image to get anchor_keypoints_sp.
        Then match those keypoints to known 2D->3D correspondences via KDTree.
        """
        try:
            # Record the start time
            start_time = time.time()
            logger.info("Starting anchor feature extraction...")
            
            # Try to acquire the lock with a timeout
            lock_acquired = self.session_lock.acquire(timeout=10.0)  # 10 second timeout
            if not lock_acquired:
                logger.error("Could not acquire session lock for anchor feature extraction (timeout)")
                return False
            
            try:
                # Precompute anchor's SuperPoint descriptors with gradients disabled
                logger.info("Processing anchor image...")
                with torch.no_grad():
                    anchor_tensor = self._convert_cv2_to_tensor(anchor_bgr_image)
                    self.extractor.to(self.device)  # Ensure extractor is on the correct device
                    self.anchor_feats = self.extractor.extract(anchor_tensor)
                    logger.info(f"Anchor features extracted in {time.time() - start_time:.3f}s")
                
                # Get anchor keypoints
                self.anchor_keypoints_sp = self.anchor_feats['keypoints'][0].detach().cpu().numpy()
                
                if len(self.anchor_keypoints_sp) == 0:
                    logger.error("No keypoints detected in anchor image!")
                    return False
                
                # Print shape and sample of keypoints for debugging
                logger.info(f"Anchor keypoints shape: {self.anchor_keypoints_sp.shape}")
                if len(self.anchor_keypoints_sp) > 5:
                    logger.info(f"First 5 keypoints: {self.anchor_keypoints_sp[:5]}")
                
                # Build KDTree to match anchor_keypoints_sp -> known anchor_keypoints_2D
                logger.info("Building KDTree for anchor keypoints...")
                sp_tree = cKDTree(self.anchor_keypoints_sp)
                distances, indices = sp_tree.query(anchor_keypoints_2D, k=1)
                valid_matches = distances < 5.0  # Increased threshold for "close enough"
                
                logger.info(f"KDTree query completed in {time.time() - start_time:.3f}s")
                logger.info(f"Valid matches: {sum(valid_matches)} out of {len(anchor_keypoints_2D)}")
                
                # Check if we have any valid matches
                if not np.any(valid_matches):
                    logger.error("No valid matches found between anchor keypoints and 2D points!")
                    return False
                
                self.matched_anchor_indices = indices[valid_matches]
                self.matched_3D_keypoints = anchor_keypoints_3D[valid_matches]
                
                logger.info(f"Matched {len(self.matched_anchor_indices)} keypoints to 3D points")
                logger.info(f"Anchor feature extraction completed in {time.time() - start_time:.3f}s")
                
                return True
                
            finally:
                # Always release the lock in the finally block to ensure it gets released
                # even if an exception occurs
                self.session_lock.release()
                logger.debug("Released session lock after anchor feature extraction")
                
        except Exception as e:
            logger.error(f"Error during anchor feature extraction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Make sure lock is released if we acquired it and an exception occurred
            # outside the 'with' block
            try:
                if hasattr(self.session_lock, '_is_owned') and self.session_lock._is_owned():
                    self.session_lock.release()
                    logger.debug("Released session lock after exception")
            except Exception as release_error:
                logger.error(f"Error releasing lock: {release_error}")
            
            return False

    def _resize_image(self, image, resize):
        logger.debug("Resizing image")
        if len(resize) == 2:
            return cv2.resize(image, tuple(resize))
        elif len(resize) == 1 and resize[0] > 0:
            h, w = image.shape[:2]
            scale = resize[0] / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(image, new_size)
        return image

    
    def process_frame(self, frame, frame_idx):
        """
        Process a frame to estimate the pose
        - Always match against the anchor image, never use frame-to-frame tracking
        - Uses PnP estimation followed by KF update (loosely-coupled)
        """
        logger.info(f"Processing frame {frame_idx}")
        start_time = time.time()

        # Ensure gradients are disabled
        with torch.no_grad():
            # Resize frame to target resolution
            frame = self._resize_image(frame, self.opt.resize)
            
            # Extract features from the frame
            frame_tensor = self._convert_cv2_to_tensor(frame)
            frame_feats = self.extractor.extract(frame_tensor)

        # Get keypoints from current frame
        frame_keypoints = frame_feats["keypoints"][0].detach().cpu().numpy()

        # For every frame, do PnP with the anchor image
        # Perform pure PnP estimation
        pnp_pose_data, visualization, mkpts0, mkpts1, mpts3D = self.perform_pnp_estimation(
            frame, frame_idx, frame_feats, frame_keypoints
        )
        
        # Check if PnP succeeded
        if pnp_pose_data is None or pnp_pose_data.get('pose_estimation_failed', True):
            logger.warning(f"PnP failed for frame {frame_idx}")
            
            # If KF initialized, try using prediction
            if self.kf_initialized:
                # Get the prediction from the filter
                x_pred, P_pred = self.mekf.predict()
                
                # Extract prediction state
                position_pred = x_pred[0:3]
                quaternion_pred = x_pred[6:10]
                R_pred = quaternion_to_rotation_matrix(quaternion_pred)
                
                # Create pose data from prediction
                pose_data = {
                    'frame': frame_idx,
                    'kf_translation_vector': position_pred.tolist(),
                    'kf_quaternion': quaternion_pred.tolist(),
                    'kf_rotation_matrix': R_pred.tolist(),
                    'pose_estimation_failed': True,
                    'tracking_method': 'prediction'
                }
                
                # Create simple visualization
                visualization = frame.copy()
                cv2.putText(visualization, "PnP Failed - Using Prediction", 
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                return pose_data, visualization
            else:
                # No KF initialized, return failure
                return {
                    'frame': frame_idx,
                    'pose_estimation_failed': True,
                    'tracking_method': 'failed'
                }, frame
        
        # PnP succeeded - extract pose data
        reprojection_error = pnp_pose_data['mean_reprojection_error']
        num_inliers = pnp_pose_data['num_inliers']
        
        # Extract raw PnP pose
        tvec = np.array(pnp_pose_data['object_translation_in_cam'])
        R = np.array(pnp_pose_data['object_rotation_in_cam'])
        q = rotation_matrix_to_quaternion(R)
        
        # Check if KF is already initialized
        if not self.kf_initialized:
            # Initialize Kalman filter if PnP is good enough
            if reprojection_error < 3.0 and num_inliers >= 6:
                self.mekf = MultExtendedKalmanFilter(dt=1.0/30.0)
                
                # Set initial state
                x_init = np.zeros(self.mekf.n_states)
                x_init[0:3] = tvec  # Position
                x_init[6:10] = q    # Quaternion
                self.mekf.x = x_init
                
                self.kf_initialized = True
                logger.info(f"MEKF initialized with PnP pose (error: {reprojection_error:.2f}, inliers: {num_inliers})")
                
                # Just return the raw PnP results for the first frame
                return pnp_pose_data, visualization
            else:
                # PnP not good enough for initialization
                logger.warning(f"PnP pose not good enough for KF initialization: " +
                            f"error={reprojection_error:.2f}px, inliers={num_inliers}")
                return pnp_pose_data, visualization
        
        # If we get here, KF is initialized and PnP succeeded
        # Predict next state
        x_pred, P_pred = self.mekf.predict()
        
        # Create pose measurement for loosely-coupled update
        pose_measurement = np.concatenate([tvec.flatten(), q])
        
        # Check if PnP result is reliable enough for KF update
        if reprojection_error < 3.0 and num_inliers >= 5:
            # Update KF with pose measurement (loosely-coupled)
            x_upd, P_upd = self.mekf.update_loosely_coupled(pose_measurement)
            
            # Extract updated pose
            position_upd = x_upd[0:3]
            quaternion_upd = x_upd[6:10]
            R_upd = quaternion_to_rotation_matrix(quaternion_upd)
            
            logger.info(f"Frame {frame_idx}: KF updated with PnP pose " +
                    f"(error: {reprojection_error:.2f}px, inliers: {num_inliers})")
            
            # Create pose data using KF-updated pose
            pose_data = {
                'frame': frame_idx,
                'kf_translation_vector': position_upd.tolist(),
                'kf_quaternion': quaternion_upd.tolist(),
                'kf_rotation_matrix': R_upd.tolist(),
                'raw_pnp_translation': tvec.flatten().tolist(),
                'raw_pnp_rotation': R.tolist(),
                'pose_estimation_failed': False,
                'num_inliers': num_inliers,
                'reprojection_error': reprojection_error,
                'tracking_method': 'anchor_pnp_with_kf'
            }
            
            # Create visualization with KF pose
            K, distCoeffs = self._get_camera_intrinsics()
            inliers = np.array(pnp_pose_data['inliers'])
            
            # Use all keypoints for visualization
            visualization = frame.copy()

            # Add KF-specific information to visualization
            cv2.putText(visualization, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(visualization, f"Reprojection Error: {reprojection_error:.2f}px", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(visualization, f"Inliers: {num_inliers}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(visualization, f"Method: Anchor PnP + KF", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Define axis parameters
            axis_length = 0.1  # 10cm for visualization
            axis_points = np.float32([
                [0, 0, 0],
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, axis_length]
            ])

            # DRAW RAW PNP AXES
            rvec_raw, _ = cv2.Rodrigues(R)  # R is the raw PnP rotation
            axis_proj_raw, _ = cv2.projectPoints(axis_points, rvec_raw, tvec.reshape(3, 1), K, distCoeffs)
            axis_proj_raw = axis_proj_raw.reshape(-1, 2)
            origin_raw = tuple(map(int, axis_proj_raw[0]))

            # Draw raw PnP axes with thinner lines
            visualization = cv2.line(visualization, origin_raw, tuple(map(int, axis_proj_raw[1])), (0, 0, 255), 2)  # X-axis (red)
            visualization = cv2.line(visualization, origin_raw, tuple(map(int, axis_proj_raw[2])), (0, 255, 0), 2)  # Y-axis (green)
            visualization = cv2.line(visualization, origin_raw, tuple(map(int, axis_proj_raw[3])), (255, 0, 0), 2)  # Z-axis (blue)

            # Add label for raw PnP axes
            cv2.putText(visualization, "Raw PnP", (origin_raw[0] + 5, origin_raw[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # DRAW KALMAN FILTER AXES
            rvec_upd, _ = cv2.Rodrigues(R_upd)  # R_upd is the Kalman filter rotation
            axis_proj_kf, _ = cv2.projectPoints(axis_points, rvec_upd, position_upd.reshape(3, 1), K, distCoeffs)
            axis_proj_kf = axis_proj_kf.reshape(-1, 2)
            origin_kf = tuple(map(int, axis_proj_kf[0]))

            # Draw KF axes with thicker lines
            visualization = cv2.line(visualization, origin_kf, tuple(map(int, axis_proj_kf[1])), (0, 0, 200), 3)  # X-axis (darker red)
            visualization = cv2.line(visualization, origin_kf, tuple(map(int, axis_proj_kf[2])), (0, 200, 0), 3)  # Y-axis (darker green)
            visualization = cv2.line(visualization, origin_kf, tuple(map(int, axis_proj_kf[3])), (200, 0, 0), 3)  # Z-axis (darker blue)

            # Add label for KF axes
            cv2.putText(visualization, "Kalman Filter", (origin_kf[0] + 5, origin_kf[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Calculate and display rotation difference
            raw_quat = rotation_matrix_to_quaternion(R)
            kf_quat = quaternion_upd
            dot_product = min(1.0, abs(np.dot(raw_quat, kf_quat)))
            angle_diff = np.arccos(dot_product) * 2 * 180 / np.pi
            cv2.putText(visualization, f"Rot Diff: {angle_diff:.1f}°", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # # Add KF-specific information to visualization
            # cv2.putText(visualization, f"Frame: {frame_idx}", (10, 30), 
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(visualization, f"Reprojection Error: {reprojection_error:.2f}px", 
            #         (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(visualization, f"Inliers: {num_inliers}", 
            #         (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(visualization, f"Method: Anchor PnP + KF", 
            #         (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # # Draw coordinate axes with KF pose
            # axis_length = 0.1  # 10cm for visualization
            # axis_points = np.float32([
            #     [0, 0, 0],
            #     [axis_length, 0, 0],
            #     [0, axis_length, 0],
            #     [0, 0, axis_length]
            # ])
            
            # rvec_upd, _ = cv2.Rodrigues(R_upd)
            # axis_proj, _ = cv2.projectPoints(axis_points, rvec_upd, position_upd.reshape(3, 1), K, distCoeffs)
            # axis_proj = axis_proj.reshape(-1, 2)
            
            # # Draw axes
            # origin = tuple(map(int, axis_proj[0]))
            # visualization = cv2.line(visualization, origin, tuple(map(int, axis_proj[1])), (0, 0, 255), 3)  # X-axis (red)
            # visualization = cv2.line(visualization, origin, tuple(map(int, axis_proj[2])), (0, 255, 0), 3)  # Y-axis (green)
            # visualization = cv2.line(visualization, origin, tuple(map(int, axis_proj[3])), (255, 0, 0), 3)  # Z-axis (blue)
            
            return pose_data, visualization
        else:
            # PnP successful but not reliable enough - use KF prediction only
            logger.warning(f"Frame {frame_idx}: PnP pose not reliable enough for KF update: " +
                        f"error={reprojection_error:.2f}px, inliers={num_inliers}")
            
            # Extract prediction state
            position_pred = x_pred[0:3]
            quaternion_pred = x_pred[6:10]
            R_pred = quaternion_to_rotation_matrix(quaternion_pred)
            
            # Create pose data from prediction
            pose_data = {
                'frame': frame_idx,
                'kf_translation_vector': position_pred.tolist(),
                'kf_quaternion': quaternion_pred.tolist(),
                'kf_rotation_matrix': R_pred.tolist(),
                'raw_pnp_translation': tvec.flatten().tolist(),
                'raw_pnp_rotation': R.tolist(),
                'pose_estimation_failed': False,
                'tracking_method': 'prediction',
                'pnp_result': 'not_reliable_enough'
            }
            
            # Create simple visualization
            visualization = frame.copy()
            cv2.putText(visualization, "PnP Not Reliable - Using Prediction", 
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return pose_data, visualization

        

    def _init_kalman_filter(self):
        frame_rate = 30
        dt = 1 / frame_rate
        kf_pose = MultExtendedKalmanFilter(dt)
        return kf_pose

    
   

    
    def _visualize_tracking(self, frame, feature_points_or_inliers, model_points_or_pose_data, state_or_frame_idx, extra_info=None):
        """
        Unified visualization function for both initialization (PnP) and tracking modes
        
        Args:
            frame: Current frame (image)
            feature_points_or_inliers: Either 2D feature points or inliers indices
            model_points_or_pose_data: Either 3D model points or pose_data dictionary
            state_or_frame_idx: Either MEKF state vector or frame index
            extra_info: Additional information depending on mode
        """
        # Make a copy for visualization
        vis_img = frame.copy()
        
        # Determine which mode we're in
        if isinstance(model_points_or_pose_data, dict):
            # Initialization/PnP mode
            pose_data = model_points_or_pose_data
            frame_idx = state_or_frame_idx
            inliers = feature_points_or_inliers
            
            # Unpack extra info if provided
            if extra_info is not None and len(extra_info) >= 3:
                mkpts0, mkpts1, mconf = extra_info[:3]
                frame_keypoints = extra_info[3] if len(extra_info) > 3 else None
                
                # Get inlier points
                if inliers is not None:
                    inlier_idx = inliers.flatten()
                    inlier_mkpts0 = mkpts0[inlier_idx]
                    inlier_mkpts1 = mkpts1[inlier_idx]
                    feature_points = inlier_mkpts1
                    
                    # Draw matched keypoints (green)
                    for pt in inlier_mkpts1:
                        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
            
            # Extract pose from pose_data
            if 'kf_rotation_matrix' in pose_data and 'kf_translation_vector' in pose_data:
                R = np.array(pose_data['kf_rotation_matrix'])
                tvec = np.array(pose_data['kf_translation_vector']).reshape(3, 1)
            else:
                R = np.array(pose_data['object_rotation_in_cam'])
                tvec = np.array(pose_data['object_translation_in_cam']).reshape(3, 1)
                print("PNP&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
                
            
            # Convert to rotation vector for axis visualization
            rvec, _ = cv2.Rodrigues(R)
            
            # Get reprojection error if available
            reprojection_text = ""
            if 'mean_reprojection_error' in pose_data:
                mean_error = pose_data['mean_reprojection_error']
                reprojection_text = f"Reprojection Error: {mean_error:.2f}px"
            
            # Get method info
            method_text = "PnP Initialization" if frame_idx == 1 else "PnP Fallback"
            if 'tracking_method' in pose_data:
                if pose_data['tracking_method'] == 'tracking':
                    method_text = "Tracking"
                elif pose_data['tracking_method'] == 'prediction':
                    method_text = "Prediction Only"
        
        else:
            # Tracking mode
            feature_points = feature_points_or_inliers
            model_points = model_points_or_pose_data
            state = state_or_frame_idx
            frame_idx = extra_info  # In tracking mode, frame_idx is passed in extra_info
            
            # Extract pose from MEKF state
            position = state[0:3]
            quaternion = state[6:10]
            R = quaternion_to_rotation_matrix(quaternion)
            tvec = position.reshape(3, 1)
            rvec, _ = cv2.Rodrigues(R)
            
            # Get camera parameters
            K, distCoeffs = self._get_camera_intrinsics()
            
            # Project model points to check reprojection error
            proj_points, _ = cv2.projectPoints(model_points, rvec, tvec, K, distCoeffs)
            proj_points = proj_points.reshape(-1, 2)
            
            # Calculate average reprojection error
            reprojection_errors = np.linalg.norm(proj_points - feature_points, axis=1)
            mean_error = np.mean(reprojection_errors)
            reprojection_text = f"Reprojection Error: {mean_error:.2f}px"
            
            # Draw feature points (green) and projections (red)
            for i, (feat_pt, proj_pt) in enumerate(zip(feature_points, proj_points)):
                # Draw actual feature point
                cv2.circle(vis_img, (int(feat_pt[0]), int(feat_pt[1])), 3, (0, 255, 0), -1)
                
                # Draw projected point
                cv2.circle(vis_img, (int(proj_pt[0]), int(proj_pt[1])), 2, (0, 0, 255), -1)
                
                # Draw line between them
                cv2.line(vis_img, 
                        (int(feat_pt[0]), int(feat_pt[1])), 
                        (int(proj_pt[0]), int(proj_pt[1])), 
                        (255, 0, 255), 1)
            
            method_text = "Tracking"
        
        # Draw coordinate axes (works for both modes)
        K, distCoeffs = self._get_camera_intrinsics()
        axis_length = 0.1  # 10cm for visualization
        axis_points = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ])
        
        axis_proj, _ = cv2.projectPoints(axis_points, rvec, tvec, K, distCoeffs)
        axis_proj = axis_proj.reshape(-1, 2)
        
        # Draw axes
        origin = tuple(map(int, axis_proj[0]))
        vis_img = cv2.line(vis_img, origin, tuple(map(int, axis_proj[1])), (0, 0, 255), 3)  # X-axis (red)
        vis_img = cv2.line(vis_img, origin, tuple(map(int, axis_proj[2])), (0, 255, 0), 3)  # Y-axis (green)
        vis_img = cv2.line(vis_img, origin, tuple(map(int, axis_proj[3])), (255, 0, 0), 3)  # Z-axis (blue)
        
        # Add telemetry
        cv2.putText(vis_img, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_img, reprojection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add position info
        if isinstance(model_points_or_pose_data, dict):
            if 'kf_translation_vector' in pose_data:
                pos = pose_data['kf_translation_vector']
            else:
                pos = pose_data['object_translation_in_cam']
        else:
            pos = position
        
        pos_text = f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
        cv2.putText(vis_img, pos_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add method info
        status = "Success"
        if isinstance(model_points_or_pose_data, dict) and pose_data.get('pose_estimation_failed', False):
            status = "Failed"
        
        cv2.putText(vis_img, f"Method: {method_text} ({status})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_img



    def _get_camera_intrinsics(self):
        
        # # Camera calibration parameters - DEFAULT
        # focal_length_x = 1430.10150
        # focal_length_y = 1430.48915
        # cx = 640.85462
        # cy = 480.64800

        # distCoeffs = np.array([0.3393, 2.0351, 0.0295, -0.0029, -10.9093], dtype=np.float32)


        # Calib_webcam ICUAS LAB 20250124
        focal_length_x = 1460.10150  # fx from the calibrated camera matrix
        focal_length_y = 1456.48915  # fy from the calibrated camera matrix
        cx = 604.85462               # cx from the calibrated camera matrix
        cy = 328.64800               # cy from the calibrated camera matrix

        distCoeffs = np.array(
            [3.56447550e-01, -1.09206851e+01, 1.40564820e-03, -1.10856449e-02, 1.20471120e+02],
            dtype=np.float32
        )

        # distCoeffs = None




        K = np.array([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K, distCoeffs
    

    def enhance_pose_initialization(self, initial_pose, mkpts0, mkpts1, mpts3D, frame):
        """
        Enhance initial pose estimate by finding additional correspondences
        
        Args:
            initial_pose: Initial pose estimate (rvec, tvec)
            mkpts0: Matched keypoints from anchor
            mkpts1: Matched keypoints from current frame
            mpts3D: 3D points corresponding to mkpts0
            frame: Current frame
            
        Returns:
            Refined pose, updated correspondences, inliers
        """
        rvec, tvec = initial_pose
        K, distCoeffs = self._get_camera_intrinsics()
        
        # Get all keypoints from the current frame
        frame_tensor = self._convert_cv2_to_tensor(frame)
        frame_feats = self.extractor.extract(frame_tensor)
        frame_keypoints = frame_feats['keypoints'][0].detach().cpu().numpy()
        
        # Project all 3D model points to find additional correspondences
        all_3d_points = self.matched_3D_keypoints  
        
        # Project all 3D model points to the image plane
        projected_points, _ = cv2.projectPoints(
            all_3d_points, rvec, tvec, K, distCoeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Find additional correspondences
        additional_corrs = []
        for i, model_pt in enumerate(all_3d_points):
            # Skip points already in the initial correspondences
            if model_pt in mpts3D:
                continue
                
            proj_pt = projected_points[i]
            
            # Find the closest feature point
            distances = np.linalg.norm(frame_keypoints - proj_pt, axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            # If close enough, consider it a match
            if min_dist < 5.0:  # Threshold in pixels
                additional_corrs.append((i, min_idx))
        
        # If we found additional correspondences, refine the pose
        if additional_corrs:
            # Combine initial correspondences with new ones
            all_3d = list(mpts3D)
            all_2d = list(mkpts1)
            
            for i3d, i2d in additional_corrs:
                all_3d.append(all_3d_points[i3d])
                all_2d.append(frame_keypoints[i2d])
            
            all_3d = np.array(all_3d)
            all_2d = np.array(all_2d)
            
            # Refine pose using all correspondences
            success, refined_rvec, refined_tvec, inliers = cv2.solvePnPRansac(
                objectPoints=all_3d.reshape(-1, 1, 3),
                imagePoints=all_2d.reshape(-1, 1, 2),
                cameraMatrix=K,
                distCoeffs=distCoeffs,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                reprojectionError=4.0,
                iterationsCount=1000,#100,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if success and inliers is not None and len(inliers) >= 6:
                # Further refine with VVS
                refined_rvec, refined_tvec = cv2.solvePnPRefineVVS(
                    objectPoints=all_3d[inliers].reshape(-1, 1, 3),
                    imagePoints=all_2d[inliers].reshape(-1, 1, 2),
                    cameraMatrix=K,
                    distCoeffs=distCoeffs,
                    rvec=refined_rvec,
                    tvec=refined_tvec
                )
                
                return (refined_rvec, refined_tvec), all_3d, all_2d, inliers
        
        # If no additional correspondences or refinement failed, return original pose
        return (rvec, tvec), mpts3D, mkpts1, None
    
   
    def perform_pnp_estimation(self, frame, frame_idx, frame_feats, frame_keypoints):
        """
        Perform pure PnP pose estimation without Kalman filtering.
        Used for both initialization and fallback when tracking fails.
        
        Args:
            frame: Current video frame
            frame_idx: Frame index/number
            frame_feats: SuperPoint features from the current frame
            frame_keypoints: Keypoints from the current frame
        
        Returns:
            tuple: (pose_data, visualization, mkpts0, mkpts1, mpts3D)
            where pose_data contains the raw PnP results without Kalman filtering
        """
        # Match features between anchor and frame
        with torch.no_grad():
            with self.session_lock:
                matches_dict = self.matcher({
                    'image0': self.anchor_feats, 
                    'image1': frame_feats
                })

        # Remove batch dimension and move to CPU
        feats0, feats1, matches01 = [rbd(x) for x in [self.anchor_feats, frame_feats, matches_dict]]
        
        # Get keypoints and matches
        kpts0 = feats0["keypoints"].detach().cpu().numpy()
        matches = matches01["matches"].detach().cpu().numpy()
        confidence = matches01.get("scores", torch.ones(len(matches))).detach().cpu().numpy()
        
        if len(matches) == 0:
            logger.warning(f"No matches found for PnP in frame {frame_idx}")
            return None, None, None, None, None
            
        mkpts0 = kpts0[matches[:, 0]]
        mkpts1 = frame_keypoints[matches[:, 1]]
        mconf = confidence

        # Filter to known anchor indices
        valid_indices = matches[:, 0]
        known_mask = np.isin(valid_indices, self.matched_anchor_indices)
        
        if not np.any(known_mask):
            logger.warning(f"No valid matches to 3D points found for PnP in frame {frame_idx}")
            return None, None, None, None, None
        
        # Filter matches to known 3D points
        mkpts0 = mkpts0[known_mask]
        mkpts1 = mkpts1[known_mask]
        mconf = mconf[known_mask]
        
        # Get corresponding 3D points
        idx_map = {idx: i for i, idx in enumerate(self.matched_anchor_indices)}
        mpts3D = np.array([
            self.matched_3D_keypoints[idx_map[aidx]] 
            for aidx in valid_indices[known_mask] if aidx in idx_map
        ])

        if len(mkpts0) < 4:
            logger.warning(f"Not enough matches for PnP in frame {frame_idx}")
            return None, None, None, None, None

        # Get camera intrinsics
        K, distCoeffs = self._get_camera_intrinsics()
        
        # Prepare data for PnP
        objectPoints = mpts3D.reshape(-1, 1, 3)
        imagePoints = mkpts1.reshape(-1, 1, 2).astype(np.float32)

        # Solve initial PnP
        success, rvec_o, tvec_o, inliers = cv2.solvePnPRansac(
            objectPoints=objectPoints,
            imagePoints=imagePoints,
            cameraMatrix=K,
            distCoeffs=distCoeffs,
            reprojectionError=4,
            confidence=0.99,
            iterationsCount=1500,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success or inliers is None or len(inliers) < 6:
            logger.warning("PnP pose estimation failed or not enough inliers.")
            return None, None, None, None, None

        # Enhance the initial pose by finding additional correspondences
        (rvec, tvec), enhanced_3d, enhanced_2d, enhanced_inliers = self.enhance_pose_initialization(
            (rvec_o, tvec_o), mkpts0, mkpts1, mpts3D, frame
        )

        # If enhancement failed, use the original results
        if enhanced_inliers is None:
            # Use the original results
            objectPoints_inliers = objectPoints[inliers.flatten()]
            imagePoints_inliers = imagePoints[inliers.flatten()]
            final_inliers = inliers
            
            # Refine with VVS
            rvec, tvec = cv2.solvePnPRefineVVS(
                objectPoints=objectPoints_inliers,
                imagePoints=imagePoints_inliers,
                cameraMatrix=K,
                distCoeffs=distCoeffs,
                rvec=rvec_o,
                tvec=tvec_o
            )
        else:
            # Use the enhanced results
            objectPoints_inliers = enhanced_3d[enhanced_inliers.flatten()]
            imagePoints_inliers = enhanced_2d[enhanced_inliers.flatten()]
            final_inliers = enhanced_inliers

        # Convert to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Initialize region counters for coverage score
        regions = {"front-right": 0, "front-left": 0, "back-right": 0, "back-left": 0}

        # Classify points into regions
        for point in objectPoints_inliers[:, 0]:
            if point[0] < 0 and point[2] > 0:  # Front-Right
                regions["front-right"] += 1
            elif point[0] < 0 and point[2] < 0:  # Front-Left
                regions["front-left"] += 1
            elif point[0] > 0 and point[2] > 0:  # Back-Right
                regions["back-right"] += 1
            elif point[0] > 0 and point[2] < 0:  # Back-Left
                regions["back-left"] += 1

        # Calculate coverage score
        total_points = sum(regions.values())
        if total_points > 0:
            used_mconf = mconf[final_inliers.flatten()] if len(final_inliers) > 0 else []
            
            if len(used_mconf) == 0 or np.isnan(used_mconf).any():
                coverage_score = 0
            else:
                # Calculate entropy term
                entropy_sum = 0
                for count in regions.values():
                    if count > 0:
                        proportion = count / total_points
                        entropy_sum += proportion * np.log(proportion)
                
                # Normalize by log(4) as specified in the paper
                normalized_entropy = -entropy_sum / np.log(4)
                
                # Final coverage score (using fixed confidence value for simplicity)
                coverage_score = normalized_entropy
                
                # Ensure score is in valid range [0,1]
                coverage_score = np.clip(coverage_score, 0, 1)
                print(f'Coverage score: {coverage_score:.2f}')
        else:
            coverage_score = 0

        # Compute reprojection errors
        projected_points, _ = cv2.projectPoints(
            objectPoints_inliers, rvec, tvec, K, distCoeffs
        )
        reprojection_errors = np.linalg.norm(imagePoints_inliers - projected_points, axis=2).flatten()
        mean_reprojection_error = np.mean(reprojection_errors)
        std_reprojection_error = np.std(reprojection_errors)

        # Create raw PnP pose_data WITHOUT Kalman filtering
        pose_data = {
            'frame': frame_idx,
            'object_rotation_in_cam': R.tolist(),
            'object_translation_in_cam': tvec.flatten().tolist(),
            'raw_rvec': rvec_o.flatten().tolist(),
            'refined_raw_rvec': rvec.flatten().tolist(),
            'num_inliers': len(final_inliers) if final_inliers is not None else 0,
            'total_matches': len(mkpts0),
            'inlier_ratio': len(final_inliers) / len(mkpts0) if len(mkpts0) > 0 else 0,
            'reprojection_errors': reprojection_errors.tolist(),
            'mean_reprojection_error': float(mean_reprojection_error),
            'std_reprojection_error': float(std_reprojection_error),
            'inliers': final_inliers.flatten().tolist(),
            'mkpts0': mkpts0.tolist(),
            'mkpts1': mkpts1.tolist(),
            'mpts3D': mpts3D.tolist(),
            'mconf': mconf.tolist(),
            'coverage_score': coverage_score,
            'pose_estimation_failed': False,
            'tracking_method': 'pnp'
        }
        
        # Also set KF values to the raw PnP results (will be updated if KF is used)
        #pose_data['kf_translation_vector'] = tvec.flatten().tolist()
        #pose_data['kf_quaternion'] = rotation_matrix_to_quaternion(R).tolist()
        #ose_data['kf_rotation_matrix'] = R.tolist()

        #pose_data['kf_translation_vector'] = None
        #pose_data['kf_quaternion'] = None
        #pose_data['kf_rotation_matrix'] = None
        
        # Create visualization
        visualization = self._visualize_tracking(
            frame, final_inliers, pose_data, frame_idx, (mkpts0, mkpts1, mconf, frame_keypoints)
        )
        
        # Return the PnP results along with matching data for future use
        return pose_data, visualization, mkpts0, mkpts1, mpts3D