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

from KF_tight import MultExtendedKalmanFilter
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
        - First frame or failed tracking: Use SuperPoint+LightGlue+PnP
        - Subsequent frames when tracking good: Use tightly-coupled MEKF
        
        Now properly separates PnP estimation from Kalman filtering
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

        # CASE 1: First frame - always use PnP
        if frame_idx == 1 or not self.kf_initialized:
            logger.info(f"Frame {frame_idx}: INITIALIZATION MODE - Using PnP")
            
            # Perform pure PnP estimation (without Kalman filtering)
            # Pass in all required parameters
            pnp_pose_data, visualization, mkpts0, mkpts1, mpts3D = self.perform_pnp_estimation(
                frame, frame_idx, frame_feats, frame_keypoints
            )
            
            if pnp_pose_data is None:
                # PnP failed
                logger.warning(f"PnP initialization failed for frame {frame_idx}")
                return {
                    'frame': frame_idx,
                    'pose_estimation_failed': True,
                    'tracking_method': 'pnp'
                }, frame
            
            # Check if pose is good enough for future tracking (use reprojection error threshold)
            if pnp_pose_data['mean_reprojection_error'] < 5.0:
                # Good PnP pose - initialize MEKF with this pose
                self.kf_initialized = True
                
                # Extract translation and quaternion from PnP result
                tvec = np.array(pnp_pose_data['object_translation_in_cam'])
                R = np.array(pnp_pose_data['object_rotation_in_cam'])
                q = rotation_matrix_to_quaternion(R)
                
                # Create MEKF instance
                self.mekf = MultExtendedKalmanFilter(dt=1.0/30.0)  # Assuming 30 fps
                
                # Set initial state
                x_init = np.zeros(self.mekf.n_states)
                x_init[0:3] = tvec  # Position
                x_init[6:10] = q    # Quaternion
                self.mekf.x = x_init
                
                # Store tracking information - use inlier points for tracking
                inliers = np.array(pnp_pose_data['inliers'])
                self.tracking_3D_points = np.array(mpts3D)[inliers]
                self.tracking_2D_points = np.array(mkpts1)[inliers]
                
                logger.info(f"MEKF initialized with {len(inliers)} tracking points")
                
                # On first frame, we just return the PnP result
                return pnp_pose_data, visualization
            else:
                logger.warning(f"PnP pose not good enough for tracking: " +
                            f"reprojection error = {pnp_pose_data['mean_reprojection_error']:.2f}")
                self.kf_initialized = False
                
                # Return pure PnP result
                return pnp_pose_data, visualization
        
        # # CASE 2: Tracking mode - try tightly-coupled approach first  DEFAULT
        # if self.kf_initialized:
        #     logger.info(f"Frame {frame_idx}: TRACKING MODE - Attempting tightly-coupled tracking")
            
        #     # Predict next state using MEKF
        #     x_pred, P_pred = self.mekf.predict()
            
        #     # Extract predicted pose
        #     position_pred = x_pred[0:3]
        #     quaternion_pred = x_pred[6:10]
        #     R_pred = quaternion_to_rotation_matrix(quaternion_pred)
            
        #     # Get camera parameters
        #     K, distCoeffs = self._get_camera_intrinsics()
            
        #     # Project 3D model points to image plane using predicted pose
        #     rvec_pred, _ = cv2.Rodrigues(R_pred)
        #     projected_points, _ = cv2.projectPoints(
        #         self.tracking_3D_points, rvec_pred, position_pred.reshape(3, 1), K, distCoeffs
        #     )
        #     projected_points = projected_points.reshape(-1, 2)
            
        #     # Find correspondences between projected points and detected keypoints
        #     correspondences = []
        #     for i, proj_point in enumerate(projected_points):
        #         # Find closest keypoint to this projected point
        #         distances = np.linalg.norm(frame_keypoints - proj_point, axis=1)
        #         min_idx = np.argmin(distances)
        #         min_dist = distances[min_idx]
                
        #         # If close enough, consider it a match
        #         # Increased threshold for better tracking
        #         if min_dist < 5.0:  # Threshold in pixels
        #             correspondences.append((i, min_idx, min_dist))
            
        #     # Sort by distance and remove duplicates
        #     correspondences.sort(key=lambda x: x[2])
            
        #     used_3d = set()
        #     used_2d = set()
        #     final_correspondences = []
            
        #     for i3d, i2d, _ in correspondences:
        #         if i3d not in used_3d and i2d not in used_2d:
        #             final_correspondences.append((i3d, i2d))
        #             used_3d.add(i3d)
        #             used_2d.add(i2d)
            
        #     # Need at least 3 points for tightly-coupled update
        #     if len(final_correspondences) >= 3:
        #         logger.info(f"Frame {frame_idx}: Using tightly-coupled update with {len(final_correspondences)} correspondences")
                
        #         # Extract matched points
        #         model_indices = [idx for idx, _ in final_correspondences]
        #         feature_indices = [idx for _, idx in final_correspondences]
                
        #         model_points = self.tracking_3D_points[model_indices]
        #         image_points = frame_keypoints[feature_indices]
                
        #         # Update MEKF with feature points
        #         x_upd, P_upd = self.mekf.update_tightly_coupled(
        #             image_points, model_points, K, distCoeffs
        #         )
                
        #         # Extract updated pose
        #         position_upd = x_upd[0:3]
        #         quaternion_upd = x_upd[6:10]
        #         R_upd = quaternion_to_rotation_matrix(quaternion_upd)
                
        #         # Create visualization
        #         visualization = self._visualize_tracking(
        #             frame, image_points, model_points, x_upd, frame_idx
        #         )
                
        #         # Create pose data
        #         pose_data = {
        #             'frame': frame_idx,
        #             'kf_translation_vector': position_upd.tolist(),
        #             'kf_quaternion': quaternion_upd.tolist(),
        #             'kf_rotation_matrix': R_upd.tolist(),
        #             'pose_estimation_failed': False,
        #             'num_correspondences': len(final_correspondences),
        #             'tracking_method': 'tracking'
        #         }
                
        #         # Update tracking points for next frame
        #         self.tracking_3D_points = model_points
        #         self.tracking_2D_points = image_points
                
        #         return pose_data, visualization
        
        ###################################################################################################3
         ###################################################################################################3
          ###################################################################################################3
        # CASE 2: Tracking mode - try tightly-coupled approach first
        if self.kf_initialized:
            logger.info(f"Frame {frame_idx}: TRACKING MODE - Attempting tightly-coupled tracking")
            
            # Predict next state using MEKF
            x_pred, P_pred = self.mekf.predict()
            
            # Extract predicted pose
            position_pred = x_pred[0:3]
            quaternion_pred = x_pred[6:10]
            R_pred = quaternion_to_rotation_matrix(quaternion_pred)
            
            # Get camera parameters
            K, distCoeffs = self._get_camera_intrinsics()
            
            # Project 3D model points to image plane using predicted pose
            rvec_pred, _ = cv2.Rodrigues(R_pred)
            projected_points, _ = cv2.projectPoints(
                self.tracking_3D_points, rvec_pred, position_pred.reshape(3, 1), K, distCoeffs
            )
            projected_points = projected_points.reshape(-1, 2)
            
            # Find correspondences between projected points and detected keypoints
            correspondences = []
            for i, proj_point in enumerate(projected_points):
                # Find closest keypoint to this projected point
                distances = np.linalg.norm(frame_keypoints - proj_point, axis=1)
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                
                # If close enough, consider it a match

                ######
                ###### This threshold is tuned for trakcing
                ######

                if min_dist < 4.0:#5.0:#20.0:  # Threshold in pixels
                    correspondences.append((i, min_idx, min_dist))
            
            # Sort by distance and remove duplicates
            correspondences.sort(key=lambda x: x[2])
            used_3d = set()
            used_2d = set()
            final_correspondences = []
            
            for i3d, i2d, dist in correspondences:
                if i3d not in used_3d and i2d not in used_2d:
                    final_correspondences.append((i3d, i2d, dist))
                    used_3d.add(i3d)
                    used_2d.add(i2d)
            
            # Need at least 3 points for tightly-coupled update
            # AND average reprojection error should be reasonable
            if len(final_correspondences) >= 3:
                # Calculate average reprojection error of correspondences
                avg_reproj_error = sum(dist for _, _, dist in final_correspondences) / len(final_correspondences)
                
                # Only proceed if average error is below threshold
                if avg_reproj_error < 10.0:  # More permissive threshold for tracking
                    logger.info(f"Frame {frame_idx}: Using tightly-coupled update with {len(final_correspondences)} " + 
                            f"correspondences (avg error: {avg_reproj_error:.2f}px)")
                    
                    # Extract matched points
                    model_indices = [idx for idx, _, _ in final_correspondences]
                    feature_indices = [idx for _, idx, _ in final_correspondences]
                    
                    model_points = self.tracking_3D_points[model_indices]
                    image_points = frame_keypoints[feature_indices]
                    
                    # Update MEKF with feature points
                    x_upd, P_upd = self.mekf.update_tightly_coupled(
                        image_points, model_points, K, distCoeffs
                    )
                    
                    # Extract updated pose
                    position_upd = x_upd[0:3]
                    quaternion_upd = x_upd[6:10]
                    R_upd = quaternion_to_rotation_matrix(quaternion_upd)
                    
                    print("ㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗ\n")
                    # Create visualization
                    visualization = self._visualize_tracking(
                        frame, image_points, model_points, x_upd, frame_idx
                    )


                    
                    
                    # Create pose data
                    pose_data = {
                        'frame': frame_idx,
                        'kf_translation_vector': position_upd.tolist(),
                        'kf_quaternion': quaternion_upd.tolist(),
                        'kf_rotation_matrix': R_upd.tolist(),
                        'pose_estimation_failed': False,
                        'num_correspondences': len(final_correspondences),
                        'tracking_method': 'tracking',
                        'avg_reprojection_error': avg_reproj_error
                    }
                    
                    # Update tracking points for next frame
                    self.tracking_3D_points = model_points
                    self.tracking_2D_points = image_points
                    
                    return pose_data, visualization
                else:
                    logger.warning(f"Frame {frame_idx}: Tracking correspondences have high error ({avg_reproj_error:.2f}px)")
                    # Fall through to PnP

        ###################################################################################################3
         ###################################################################################################3
          ###################################################################################################3
        
        # # CASE 3: Tracking failed or not initialized - fall back to PnP DEFAULT
        # logger.info(f"Frame {frame_idx}: FALLBACK MODE - Using PnP")
        
        # # Perform pure PnP estimation (without Kalman filtering)
        # # Pass in all required parameters
        # pnp_pose_data, visualization, mkpts0, mkpts1, mpts3D = self.perform_pnp_estimation(
        #     frame, frame_idx, frame_feats, frame_keypoints
        # )
        
        # if pnp_pose_data is None:
        #     # PnP also failed - use prediction if MEKF is initialized
        #     if self.kf_initialized:
        #         logger.warning(f"Frame {frame_idx}: PnP fallback failed - using prediction only")
                
        #         # Extract from predicted state
        #         position_pred = x_pred[0:3]
        #         quaternion_pred = x_pred[6:10]
        #         R_pred = quaternion_to_rotation_matrix(quaternion_pred)
                
        #         pose_data = {
        #             'frame': frame_idx,
        #             'kf_translation_vector': position_pred.tolist(),
        #             'kf_quaternion': quaternion_pred.tolist(),
        #             'kf_rotation_matrix': R_pred.tolist(),
        #             'pose_estimation_failed': True,
        #             'tracking_method': 'prediction'
        #         }
                
        #         # Create simple visualization
        #         visualization = frame.copy()
        #         cv2.putText(visualization, "Tracking Failed - Using Prediction", 
        #                     (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        #         return pose_data, visualization
        #     else:
        #         # Complete failure - no MEKF and PnP failed
        #         logger.error(f"Frame {frame_idx}: Complete pose estimation failure")
        #         return {
        #             'frame': frame_idx,
        #             'pose_estimation_failed': True,
        #             'tracking_method': 'failed'
        #         }, frame
        
        # # PnP succeeded - check if we should use it to update/initialize the Kalman filter
        # if pnp_pose_data['mean_reprojection_error'] < 5.0:
        #     # Good PnP pose - initialize or update MEKF
        #     if not self.kf_initialized:
        #         # Initialize MEKF
        #         self.mekf = MultExtendedKalmanFilter(dt=1.0/30.0)
        #         self.kf_initialized = True
        #         logger.info(f"Frame {frame_idx}: Initializing MEKF from PnP fallback")
        #     else:
        #         logger.info(f"Frame {frame_idx}: Resetting MEKF from PnP fallback")
            
        #     # Extract translation and quaternion from PnP result
        #     tvec = np.array(pnp_pose_data['object_translation_in_cam'])
        #     R = np.array(pnp_pose_data['object_rotation_in_cam'])

        #     # pnp_pose_data['kf_translation_vector'] = tvec.flatten().tolist()
        #     # pnp_pose_data['kf_quaternion'] = rotation_matrix_to_quaternion(R).tolist()
        #     # pnp_pose_data['kf_rotation_matrix'] = R.tolist()


            
        #     q = rotation_matrix_to_quaternion(R)
            
        #     # Update/initialize MEKF state
        #     self.mekf.x[0:3] = tvec  # Position
        #     self.mekf.x[6:10] = q    # Quaternion
            
        #     # Update tracking points
        #     inliers = np.array(pnp_pose_data['inliers'])
        #     self.tracking_3D_points = np.array(mpts3D)[inliers]
        #     self.tracking_2D_points = np.array(mkpts1)[inliers]
            
        #     # Return the pure PnP result (will be used to update KF next frame)
        #     return pnp_pose_data, visualization
        # else:
        #     logger.warning(f"Frame {frame_idx}: PnP fallback not good enough for tracking: " +
        #                 f"reprojection error = {pnp_pose_data['mean_reprojection_error']:.2f}")
            
        #     # Return pure PnP result without updating Kalman filter
        #     return pnp_pose_data, visualization
        
        
        ####################################################################3#########################################3

        # This is the loose+tight hybrid
        # CASE 3: Tracking failed - fall back to PnP
        logger.info(f"Frame {frame_idx}: FALLBACK MODE - Using PnP")


        print("333333333333333333333333333333333333333333333333333333333333333333333\n")

        # Perform pure PnP estimation
        pnp_pose_data, visualization, mkpts0, mkpts1, mpts3D = self.perform_pnp_estimation(
            frame, frame_idx, frame_feats, frame_keypoints
        )
        if self.pred_only > 5:
            self.kf_initialized = False
            self.pred_only = 0

        # If KF is already initialized
        if self.kf_initialized:
            # Get the prediction from the filter
            x_pred, P_pred = self.mekf.predict()
            
            # If PnP succeeded
            if pnp_pose_data is not None and not pnp_pose_data.get('pose_estimation_failed', False):
                # Extract PnP pose
                tvec = np.array(pnp_pose_data['object_translation_in_cam'])
                R = np.array(pnp_pose_data['object_rotation_in_cam'])
                q = rotation_matrix_to_quaternion(R)
                
                # Calculate differences between prediction and PnP result
                position_diff = np.linalg.norm(tvec - x_pred[0:3])
                
                # For orientation, calculate angle between quaternions
                def quaternion_angle_degrees(q1, q2):
                    q1 = normalize_quaternion(q1)
                    q2 = normalize_quaternion(q2)
                    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
                    angle = 2.0 * np.degrees(np.arccos(dot))
                    if angle > 180.0:
                        angle = 360.0 - angle
                    return angle
                
                orientation_diff = quaternion_angle_degrees(q, x_pred[6:10])
                
                # Check PnP quality criteria
                reprojection_error = pnp_pose_data['mean_reprojection_error']
                num_inliers = pnp_pose_data['num_inliers']
                
                # Define thresholds for measurement acceptance
                max_position_jump = 20#0.3  # meters
                max_orientation_jump = 400#20.0  # degrees
                max_reprojection_error = 10#6#8#10#5.0  # pixels
                min_inliers = 6#5  # points
                
                # Validation gate: Check if PnP result is valid for updating the filter
                if (position_diff <= max_position_jump and 
                    orientation_diff <= max_orientation_jump and
                    reprojection_error <= max_reprojection_error and
                    num_inliers >= min_inliers):
                    
                    logger.info(f"Frame {frame_idx}: PnP passed validation, using loosely-coupled update")
                    

                    print("UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\n")
                    # Create measurement vector [x, y, z, qx, qy, qz, qw]
                    z_pose = np.concatenate([tvec.flatten(), q])
                    
                    print("Z_pose before KF:\n",z_pose)
                    
                    # Use loosely-coupled update with direct pose measurement
                    
                    x_upd, P_upd = self.mekf.update_loosely_coupled(z_pose)

                    
                    
                    
                    # Update tracking points for next frame
                    inliers = np.array(pnp_pose_data['inliers'])
                    self.tracking_3D_points = np.array(mpts3D)[inliers]
                    self.tracking_2D_points = np.array(mkpts1)[inliers]
                    
                    # Extract updated state
                    position_upd = x_upd[0:3]
                    quaternion_upd = x_upd[6:10]
                    R_upd = quaternion_to_rotation_matrix(quaternion_upd)

                    print("position_upd after KF:\n",position_upd)
                    print("quaternion_upd after KF:\n",quaternion_upd)
                    
                    # Create pose data
                    pose_data = pnp_pose_data.copy()
                    pose_data['kf_translation_vector'] = position_upd.tolist()
                    pose_data['kf_quaternion'] = quaternion_upd.tolist()
                    pose_data['kf_rotation_matrix'] = R_upd.tolist()
                    pose_data['tracking_method'] = 'loosely_coupled'
                    pose_data['position_diff'] = position_diff
                    pose_data['orientation_diff'] = orientation_diff
                    
                    return pose_data, visualization
                    
                else:
                    # PnP failed validation - use prediction only
                    logger.warning(f"Frame {frame_idx}: PnP failed validation " +
                                f"(pos_diff={position_diff:.2f}m, orient_diff={orientation_diff:.2f}°, " +
                                f"reproj_err={reprojection_error:.2f}, inliers={num_inliers}), " +
                                f"using KF prediction")
                    
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
                        'pose_estimation_failed': False,
                        'tracking_method': 'prediction',
                        'rejected_pnp': True,
                        'rejection_reason': f"pos_diff={position_diff:.2f}, orient_diff={orientation_diff:.2f}, " +
                                        f"reproj_err={reprojection_error:.2f}, inliers={num_inliers}"
                    }
                    
                    # Create simple visualization
                    visualization = frame.copy()
                    cv2.putText(visualization, "PnP Rejected - Using Prediction", 
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    self.pred_only = +1
                    
                    return pose_data, visualization
            
            else:
                # PnP completely failed - use pure prediction
                logger.warning(f"Frame {frame_idx}: Complete PnP failure, using KF prediction")
                
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
                
                self.pred_only = +1
                
                return pose_data, visualization

        else:
            # KF not yet initialized - first initialization only
            if pnp_pose_data is not None and not pnp_pose_data.get('pose_estimation_failed', False):
                # Only initialize KF on first good PnP
                if pnp_pose_data['mean_reprojection_error'] < 5.0:
                    tvec = np.array(pnp_pose_data['object_translation_in_cam'])
                    R = np.array(pnp_pose_data['object_rotation_in_cam'])
                    q = rotation_matrix_to_quaternion(R)
                    
                    # Initialize MEKF
                    self.mekf = MultExtendedKalmanFilter(dt=1.0/30.0)
                    
                    # Set initial state
                    x_init = np.zeros(self.mekf.n_states)
                    x_init[0:3] = tvec.flatten()  # Position
                    x_init[6:10] = q              # Quaternion
                    self.mekf.x = x_init
                    
                    # Initialize tracking points
                    inliers = np.array(pnp_pose_data['inliers'])
                    self.tracking_3D_points = np.array(mpts3D)[inliers]
                    self.tracking_2D_points = np.array(mkpts1)[inliers]
                    
                    self.kf_initialized = True
                    logger.info(f"Frame {frame_idx}: First MEKF initialization")
                
                # Return pure PnP result for first frame
                return pnp_pose_data, visualization
            else:
                # PnP failed and no KF initialized yet
                return {
                    'frame': frame_idx,
                    'pose_estimation_failed': True,
                    'tracking_method': 'failed'
                }, frame

        ###########################################################################
        
        # # CASE 3: Tracking failed or not initialized - fall back to PnP
        # logger.info(f"Frame {frame_idx}: FALLBACK MODE - Using PnP")

        # print("33333333333333333333333333333333333333333333333333333333333333333\n")

        # # Perform pure PnP estimation (without Kalman filtering)
        # pnp_pose_data, visualization, mkpts0, mkpts1, mpts3D = self.perform_pnp_estimation(
        #     frame, frame_idx, frame_feats, frame_keypoints
        # )

        # if pnp_pose_data is None:
        #     # PnP also failed - use prediction if MEKF is initialized
        #     if self.kf_initialized:
        #         logger.warning(f"Frame {frame_idx}: PnP fallback failed - using prediction only")
                
        #         # Extract from predicted state
        #         position_pred = x_pred[0:3]
        #         quaternion_pred = x_pred[6:10]
        #         R_pred = quaternion_to_rotation_matrix(quaternion_pred)
                
        #         pose_data = {
        #             'frame': frame_idx,
        #             'kf_translation_vector': position_pred.tolist(),
        #             'kf_quaternion': quaternion_pred.tolist(),
        #             'kf_rotation_matrix': R_pred.tolist(),
        #             'pose_estimation_failed': True,
        #             'tracking_method': 'prediction'
        #         }
                
        #         # Create simple visualization
        #         visualization = frame.copy()
        #         cv2.putText(visualization, "Tracking Failed - Using Prediction", 
        #                     (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        #         return pose_data, visualization
        #     else:
        #         # Complete failure - no MEKF and PnP failed
        #         logger.error(f"Frame {frame_idx}: Complete pose estimation failure")
        #         return {
        #             'frame': frame_idx,
        #             'pose_estimation_failed': True,
        #             'tracking_method': 'failed'
        #         }, frame

        # # PnP succeeded - check if we should use it to update/initialize the Kalman filter
        # if pnp_pose_data['mean_reprojection_error'] < 5.0:
        #     # Good PnP pose - initialize if not initialized, but never reinitialize
        #     if not self.kf_initialized:
        #         # Initialize MEKF only if it hasn't been initialized before
        #         self.mekf = MultExtendedKalmanFilter(dt=1.0/30.0)
                
        #         # Extract translation and quaternion from PnP result
        #         tvec = np.array(pnp_pose_data['object_translation_in_cam'])
        #         R = np.array(pnp_pose_data['object_rotation_in_cam'])
        #         q = rotation_matrix_to_quaternion(R)
                
        #         # Set initial state
        #         x_init = np.zeros(self.mekf.n_states)
        #         x_init[0:3] = tvec  # Position
        #         x_init[6:10] = q    # Quaternion
        #         self.mekf.x = x_init
                
        #         # Store tracking information
        #         inliers = np.array(pnp_pose_data['inliers'])
        #         self.tracking_3D_points = np.array(mpts3D)[inliers]
        #         self.tracking_2D_points = np.array(mkpts1)[inliers]
                
        #         self.kf_initialized = True
        #         logger.info(f"Frame {frame_idx}: First MEKF initialization")
        #     else:
        #         # Kalman filter already initialized - never reinitialize
        #         # Instead, just update tracking points for next frame's tracking attempt
        #         logger.info(f"Frame {frame_idx}: Updating tracking points from PnP")
        #         inliers = np.array(pnp_pose_data['inliers'])
        #         self.tracking_3D_points = np.array(mpts3D)[inliers]
        #         self.tracking_2D_points = np.array(mkpts1)[inliers]
            
        #     # Return the pure PnP result - we're not updating the KF state here
        #     return pnp_pose_data, visualization
        # else:
        #     logger.warning(f"Frame {frame_idx}: PnP fallback not good enough for tracking: " +
        #                 f"reprojection error = {pnp_pose_data['mean_reprojection_error']:.2f}")
            
        #     # Return pure PnP result without updating Kalman filter
        #     return pnp_pose_data, visualization


    def _init_kalman_filter(self):
        frame_rate = 30
        dt = 1 / frame_rate
        kf_pose = MultExtendedKalmanFilter(dt)
        return kf_pose

    
    # First, let's fix the _visualize_tracking method to handle both initialization and tracking modes



    # def _kalman_filter_update(
    #     self, R, tvec, reprojection_errors, mean_reprojection_error,
    #     std_reprojection_error, inliers, mkpts0, mkpts1, mpts3D,
    #     mconf, frame_idx, rvec_o, rvec, coverage_score
    # ):
    #     """
    #     Update the Kalman filter with new pose measurements from PnP.
    #     Performs quality checks before applying the update.

    #     Args:
    #         R: Rotation matrix from PnP
    #         tvec: Translation vector from PnP
    #         reprojection_errors: Array of reprojection errors for each point
    #         mean_reprojection_error: Mean reprojection error
    #         std_reprojection_error: Standard deviation of reprojection errors
    #         inliers: Indices of inlier points
    #         mkpts0: Matched keypoints from anchor
    #         mkpts1: Matched keypoints from current frame
    #         mpts3D: 3D points corresponding to mkpts0
    #         mconf: Confidence scores for matches
    #         frame_idx: Frame index
    #         rvec_o: Original rotation vector from PnP
    #         rvec: Refined rotation vector
    #         coverage_score: Score for spatial distribution of feature points

    #     Returns:
    #         dict: Pose data including original and filtered poses
    #     """
    #     num_inliers = len(inliers)
    #     inlier_ratio = num_inliers / len(mkpts0) if len(mkpts0) > 0 else 0

    #     # Thresholds for quality checks
    #     reprojection_error_threshold = 4.0
    #     max_translation_jump = 2.0
    #     max_orientation_jump = 20.0  # degrees
    #     min_inlier = 4
    #     coverage_threshold = 0.3

    #     if coverage_score is None:
    #         logger.info("Coverage score not calculated, using default value")
    #         coverage_score = 0.0

    #     # Convert measured rotation R -> quaternion
    #     q_measured = rotation_matrix_to_quaternion(R)

    #     # Check viewpoint if anchor_viewpoint information is available
    #     anchor_q = None
    #     if hasattr(self, "anchor_viewpoint_eulers"):
    #         # Convert Euler angles to quaternion if needed
    #         # This is just a placeholder - actual conversion would depend on your convention
    #         anchor_q = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion placeholder
            
    #     viewpoint_max_diff_deg = 380.0  # Very permissive threshold
    #     viewpoint_diff = 0.0

    #     def quaternion_angle_degrees(q1, q2):
    #         """Calculate angle between two quaternions in degrees"""
    #         q1 = normalize_quaternion(q1)
    #         q2 = normalize_quaternion(q2)
    #         dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    #         angle = 2.0 * np.degrees(np.arccos(dot))
    #         if angle > 180.0:
    #             angle = 360.0 - angle
    #         return angle

    #     if anchor_q is not None:
    #         viewpoint_diff = quaternion_angle_degrees(q_measured, anchor_q)

    #             # Initialize the MEKF if this is the first good frame
    #     if not self.kf_initialized:
    #         logger.info("Initializing MEKF with first pose")
    #         # Initialize the MEKF with the measured pose
    #         self.mekf = MultExtendedKalmanFilter(dt=1.0/30.0)  # Assuming 30 fps
            
    #         # Set initial state
    #         x_init = np.zeros(self.mekf.n_states)
    #         x_init[0:3] = tvec.flatten()  # Position
    #         x_init[6:10] = q_measured     # Quaternion
    #         self.mekf.x = x_init
            
    #         # Store tracking information for future frames - validate indices first
    #         if isinstance(inliers, np.ndarray):
    #             inlier_indices = inliers.flatten()
    #         else:
    #             inlier_indices = np.array(inliers)
                
    #         # Validate indices to prevent out-of-bounds errors
    #         valid_indices = []
    #         for idx in inlier_indices:
    #             if 0 <= idx < len(mpts3D):
    #                 valid_indices.append(idx)
                    
    #         # Convert to numpy array for indexing
    #         if valid_indices:
    #             valid_indices = np.array(valid_indices)
    #             self.tracking_3D_points = mpts3D[valid_indices]
    #             self.tracking_2D_points = mkpts1[valid_indices]
    #         else:
    #             # If no valid indices, use all points but log a warning
    #             logger.warning("No valid inlier indices during initialization, using all points")
    #             self.tracking_3D_points = mpts3D
    #             self.tracking_2D_points = mkpts1
            
    #         self.kf_initialized = True
    #         self.kf_pose_first_update = False
            
    #         # For initialization, use the measured pose directly
    #         R_estimated = R
    #         px, py, pz = tvec.flatten()
    #         qx, qy, qz, qw = q_measured
    #     else:
    #         # MEKF is already initialized, perform normal update process
            
    #         # 1) Get prediction from MEKF
    #         x_pred, P_pred = self.mekf.predict()
            
    #         # Parse predicted state for threshold checks
    #         px_pred, py_pred, pz_pred = x_pred[0:3]
    #         qx_pred, qy_pred, qz_pred, qw_pred = x_pred[6:10]
            
    #         # Calculate changes from prediction to measurement
    #         pred_quat = np.array([qx_pred, qy_pred, qz_pred, qw_pred])
    #         orientation_change = quaternion_angle_degrees(q_measured, pred_quat)
    #         translation_change = np.linalg.norm(tvec.flatten() - x_pred[0:3])
            
    #         # 2) Build measurement vector z = [px, py, pz, qx, qy, qz, qw]
    #         tvec_flat = tvec.flatten()
    #         z_meas = np.array([
    #             tvec_flat[0], tvec_flat[1], tvec_flat[2],
    #             q_measured[0], q_measured[1], q_measured[2], q_measured[3]
    #         ], dtype=np.float64)
            
    #         # 3) Apply quality checks before updating
    #         update_valid = True
            
    #         if mean_reprojection_error >= reprojection_error_threshold:
    #             logger.debug(f"Skipping update: high reprojection error {mean_reprojection_error:.2f} >= {reprojection_error_threshold:.2f}")
    #             update_valid = False
            
    #         if num_inliers <= min_inlier:
    #             logger.debug(f"Skipping update: insufficient inliers {num_inliers} <= {min_inlier}")
    #             update_valid = False
                
    #         if translation_change >= max_translation_jump:
    #             logger.debug(f"Skipping update: large translation jump {translation_change:.2f} >= {max_translation_jump:.2f}")
    #             update_valid = False
                
    #         if orientation_change >= max_orientation_jump:
    #             logger.debug(f"Skipping update: large orientation jump {orientation_change:.2f} >= {max_orientation_jump:.2f}")
    #             update_valid = False
                
    #         if coverage_score < coverage_threshold:
    #             logger.debug(f"Skipping update: poor coverage score {coverage_score:.2f} < {coverage_threshold:.2f}")
    #             update_valid = False
                
    #         if viewpoint_diff > viewpoint_max_diff_deg:
    #             logger.debug(f"Skipping update: large viewpoint diff {viewpoint_diff:.2f} > {viewpoint_max_diff_deg:.2f}")
    #             update_valid = False
            
    #         # 4) Update MEKF if quality checks pass
    #         if update_valid:
    #             # For loosely-coupled update, we'd use:
    #             # x_upd, P_upd = self.mekf.update(z_meas)
                
    #             # But since we're using tightly-coupled update, we need to extract valid correspondences
    #             K, distCoeffs = self._get_camera_intrinsics()
                
    #             try:
    #                 # Validate inliers format and convert to usable indices
    #                 if isinstance(inliers, np.ndarray):
    #                     inlier_indices = inliers.flatten()
    #                 elif isinstance(inliers, list):
    #                     inlier_indices = np.array(inliers)
    #                 else:
    #                     logger.warning(f"Unexpected inliers type: {type(inliers)}")
    #                     inlier_indices = np.array([], dtype=int)
                    
    #                 # Make sure indices are within bounds
    #                 valid_indices = []
    #                 for idx in inlier_indices:
    #                     if 0 <= idx < len(mpts3D) and 0 <= idx < len(mkpts1):
    #                         valid_indices.append(idx)
    #                     else:
    #                         logger.warning(f"Index {idx} is out of bounds (mpts3D size: {len(mpts3D)}, mkpts1 size: {len(mkpts1)})")
                    
    #                 # Use only valid indices
    #                 if not valid_indices:
    #                     logger.warning("No valid indices found for tightly-coupled update")
    #                     # Fall back to using prediction
    #                     px, py, pz = px_pred, py_pred, pz_pred
    #                     qx, qy, qz, qw = qx_pred, qy_pred, qz_pred, qw_pred
    #                     R_estimated = quaternion_to_rotation_matrix([qx, qy, qz, qw])
    #                     update_valid = False
    #                 else:
    #                     # Convert to numpy array for indexing
    #                     valid_indices = np.array(valid_indices)
    #                     model_points = mpts3D[valid_indices]
    #                     image_points = mkpts1[valid_indices]
                        
    #                     # Ensure we have at least 3 points for tightly-coupled update
    #                     if len(valid_indices) < 3:
    #                         logger.warning(f"Not enough valid points for tightly-coupled update: {len(valid_indices)} < 3")
    #                         # Fall back to using prediction
    #                         px, py, pz = px_pred, py_pred, pz_pred
    #                         qx, qy, qz, qw = qx_pred, qy_pred, qz_pred, qw_pred
    #                         R_estimated = quaternion_to_rotation_matrix([qx, qy, qz, qw])
    #                         update_valid = False
    #             except Exception as e:
    #                 logger.error(f"Error preparing data for tightly-coupled update: {e}")
    #                 # Fall back to using prediction
    #                 px, py, pz = px_pred, py_pred, pz_pred
    #                 qx, qy, qz, qw = qx_pred, qy_pred, qz_pred, qw_pred
    #                 R_estimated = quaternion_to_rotation_matrix([qx, qy, qz, qw])
    #                 update_valid = False
                
    #             # Only perform update if we still have valid data
    #             if update_valid and len(model_points) >= 3:
    #                 try:
    #                     # Perform tightly-coupled update
    #                     x_upd, P_upd = self.mekf.update_tightly_coupled(
    #                         image_points, model_points, K, distCoeffs
    #                     )
                        
    #                     # Update tracking points for next frame
    #                     if len(model_points) > 0 and len(image_points) > 0:
    #                         self.tracking_3D_points = model_points
    #                         self.tracking_2D_points = image_points
    #                     else:
    #                         logger.warning("Empty model or image points after update, keeping previous tracking points")
                        
    #                     # Extract updated pose
    #                     px, py, pz = x_upd[0:3]
    #                     qx, qy, qz, qw = x_upd[6:10]
    #                     R_estimated = quaternion_to_rotation_matrix([qx, qy, qz, qw])
                        
    #                     logger.debug("MEKF update applied successfully")
    #                 except Exception as e:
    #                     logger.error(f"Error in tightly-coupled update: {e}")
    #                     # Fall back to using prediction
    #                     px, py, pz = px_pred, py_pred, pz_pred
    #                     qx, qy, qz, qw = qx_pred, qy_pred, qz_pred, qw_pred
    #                     R_estimated = quaternion_to_rotation_matrix([qx, qy, qz, qw])
    #             else:
    #                 logger.debug("Skipping MEKF update due to validation failure")
    #         else:
    #             # Use prediction when update is invalid
    #             logger.debug("Using predicted state due to failed quality checks")
    #             px, py, pz = px_pred, py_pred, pz_pred
    #             qx, qy, qz, qw = qx_pred, qy_pred, qz_pred, qw_pred
    #             R_estimated = quaternion_to_rotation_matrix([qx, qy, qz, qw])

    #     # Build final pose_data dictionary
    #     pose_data = {
    #         'frame': frame_idx,
    #         'object_rotation_in_cam': R.tolist(),
    #         'object_translation_in_cam': tvec.flatten().tolist(),
    #         'raw_rvec': rvec_o.flatten().tolist(),
    #         'refined_raw_rvec': rvec.flatten().tolist(),
    #         'num_inliers': num_inliers,
    #         'total_matches': len(mkpts0),
    #         'inlier_ratio': inlier_ratio,
    #         'reprojection_errors': reprojection_errors.tolist(),
    #         'mean_reprojection_error': float(mean_reprojection_error),
    #         'std_reprojection_error': float(std_reprojection_error),
    #         'inliers': inliers.flatten().tolist(),
    #         'mkpts0': mkpts0.tolist(),
    #         'mkpts1': mkpts1.tolist(),
    #         'mpts3D': mpts3D.tolist(),
    #         'mconf': mconf.tolist(),

    #         # Filtered results from updated state:
    #         'kf_translation_vector': [px, py, pz],
    #         'kf_quaternion': [qx, qy, qz, qw],
    #         'kf_rotation_matrix': R_estimated.tolist(),

    #         # Additional coverage / viewpoint metrics
    #         'coverage_score': coverage_score,
    #         'viewpoint_diff_deg': viewpoint_diff
    #     }
        
    #     return pose_data


    
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
    
    # # Update estimate_pose to use the fixed visualization function
    # def estimate_pose(self, mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, frame_keypoints):
    #     logger.debug(f"Estimating pose for frame {frame_idx}")
    #     K, distCoeffs = self._get_camera_intrinsics()

    #     objectPoints = mpts3D.reshape(-1, 1, 3)
    #     imagePoints = mkpts1.reshape(-1, 1, 2).astype(np.float32)

    #     # Solve initial PnP
    #     success, rvec_o, tvec_o, inliers = cv2.solvePnPRansac(
    #         objectPoints=objectPoints,
    #         imagePoints=imagePoints,
    #         cameraMatrix=K,
    #         distCoeffs=distCoeffs,
    #         reprojectionError=4,
    #         confidence=0.99,
    #         iterationsCount=1500,
    #         flags=cv2.SOLVEPNP_EPNP
    #     )

    #     if not success or inliers is None or len(inliers) < 6:
    #         logger.warning("PnP pose estimation failed or not enough inliers.")
    #         return None, frame

    #     # Enhance the initial pose by finding additional correspondences
    #     (rvec, tvec), enhanced_3d, enhanced_2d, enhanced_inliers = self.enhance_pose_initialization(
    #         (rvec_o, tvec_o), mkpts0, mkpts1, mpts3D, frame
    #     )

    #     # If enhancement failed, use the original results
    #     if enhanced_inliers is None:
    #         # Use the original results
    #         objectPoints_inliers = objectPoints[inliers.flatten()]
    #         imagePoints_inliers = imagePoints[inliers.flatten()]
            
    #         # Refine with VVS
    #         rvec, tvec = cv2.solvePnPRefineVVS(
    #             objectPoints=objectPoints_inliers,
    #             imagePoints=imagePoints_inliers,
    #             cameraMatrix=K,
    #             distCoeffs=distCoeffs,
    #             rvec=rvec_o,
    #             tvec=tvec_o
    #         )
    #     else:
    #         # Use the enhanced results
    #         objectPoints_inliers = enhanced_3d[enhanced_inliers.flatten()]
    #         imagePoints_inliers = enhanced_2d[enhanced_inliers.flatten()]
    #         inliers = enhanced_inliers

    #     # Convert to rotation matrix
    #     R, _ = cv2.Rodrigues(rvec)

    #     # Initialize region counters
    #     regions = {"front-right": 0, "front-left": 0, "back-right": 0, "back-left": 0}

    #     # Classify points into regions
    #     for point in objectPoints_inliers[:, 0]:
    #         if point[0] < 0 and point[2] > 0:  # Front-Right
    #             regions["front-right"] += 1
    #         elif point[0] < 0 and point[2] < 0:  # Front-Left
    #             regions["front-left"] += 1
    #         elif point[0] > 0 and point[2] > 0:  # Back-Right
    #             regions["back-right"] += 1
    #         elif point[0] > 0 and point[2] < 0:  # Back-Left
    #             regions["back-left"] += 1

    #     # Calculate coverage score
    #     total_points = sum(regions.values())
    #     if total_points > 0:
    #         valid_conf = mconf[inliers.flatten()] if len(inliers) > 0 else []
            
    #         if len(valid_conf) == 0 or np.isnan(valid_conf).any():
    #             coverage_score = 0
    #         else:
    #             # Calculate entropy term
    #             entropy_sum = 0
    #             for count in regions.values():
    #                 if count > 0:
    #                     proportion = count / total_points
    #                     entropy_sum += proportion * np.log(proportion)
                
    #             # Normalize by log(4) as specified in the paper
    #             normalized_entropy = -entropy_sum / np.log(4)
                
    #             # Calculate mean confidence
    #             mean_confidence = 1
                
    #             # Final coverage score
    #             coverage_score = normalized_entropy * mean_confidence
                
    #             # Ensure score is in valid range [0,1]
    #             coverage_score = np.clip(coverage_score, 0, 1)
    #             print('Coverage score:', coverage_score)
    #     else:
    #         coverage_score = 0

    #     # Compute reprojection errors
    #     projected_points, _ = cv2.projectPoints(
    #         objectPoints_inliers, rvec, tvec, K, distCoeffs
    #     )
    #     reprojection_errors = np.linalg.norm(imagePoints_inliers - projected_points, axis=2).flatten()
    #     mean_reprojection_error = np.mean(reprojection_errors)
    #     std_reprojection_error = np.std(reprojection_errors)

    #     # Update Kalman filter
    #     pose_data = self._kalman_filter_update(
    #         R, tvec, reprojection_errors, mean_reprojection_error,
    #         std_reprojection_error, inliers, 
    #         enhanced_3d if enhanced_inliers is not None else mkpts0, 
    #         enhanced_2d if enhanced_inliers is not None else mkpts1, 
    #         objectPoints_inliers.reshape(-1, 3),
    #         mconf, frame_idx, rvec_o, rvec, coverage_score=coverage_score
    #     )

    #     # Store the inlier 3D points for tracking
    #     self.tracking_3D_points = objectPoints_inliers.reshape(-1, 3)
    #     self.tracking_2D_points = imagePoints_inliers.reshape(-1, 2)

    #     # Use the updated visualization function with additional info
    #     visualization = self._visualize_tracking(
    #         frame, inliers, pose_data, frame_idx, (mkpts0, mkpts1, mconf, frame_keypoints)
    #     )
        
    #     return pose_data, visualization
    
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