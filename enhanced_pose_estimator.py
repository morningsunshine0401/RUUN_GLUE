import threading
import cv2
import torch
import time
import numpy as np
from scipy.spatial import cKDTree
import logging
from utils import frame2tensor, rotation_matrix_to_quaternion, quaternion_to_rotation_matrix
from KF_Q import KalmanFilterPose
import matplotlib.cm as cm
from models.utils import make_matching_plot_fast

# Import LightGlue and SuperPoint
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

# Import our enhanced point tracker
from enhanced_point_tracker import EnhancedPointTracker

logger = logging.getLogger(__name__)

class EnhancedPoseEstimator:
    def __init__(self, opt, device):
        self.session_lock = threading.Lock()
        self.opt = opt
        self.device = device
        self.initial_z_set = False
        self.kf_initialized = False
        
        self.debug_force_tracking_after = 5  # Force tracking after frame 10 for testing
        self.debug_tracking_enabled = True    # Set to False to disable this debug feature

        
        logger.info("Initializing Enhanced Pose Estimator with tracking capabilities")

        # Load anchor (leader) image
        self.anchor_image = cv2.imread(opt.anchor)
        assert self.anchor_image is not None, f'Failed to load anchor image at {opt.anchor}'
        self.anchor_image = self._resize_image(self.anchor_image, opt.resize)
        logger.info(f"Loaded and resized anchor image from {opt.anchor}")

        # Initialize SuperPoint and LightGlue models
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)
        logger.info("Initialized SuperPoint and LightGlue models")

        # Set up the point tracker
        self.point_tracker = EnhancedPointTracker(
            max_length=opt.max_length if hasattr(opt, 'max_length') else 5,
            nn_thresh=opt.nn_thresh if hasattr(opt, 'nn_thresh') else 0.7,
            sliding_window_size=3
        )
        
        # State variables for tracking vs. matching
        self.tracking_mode = False  # Start in matching mode
        self.frame_count = 0
        self.last_frame = None
        self.last_frame_features = None
        self.last_frame_descriptors = None
        
        # We will store the anchor's 2D/3D keypoints here.
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

        # Set anchor features
        self._set_anchor_features(
            anchor_bgr_image=self.anchor_image,
            anchor_keypoints_2D=anchor_keypoints_2D,
            anchor_keypoints_3D=anchor_keypoints_3D
        )

        # Initialize Kalman filter
        self.kf_pose = self._init_kalman_filter()
        self.kf_pose_first_update = True 
        logger.info("Kalman filter initialized")
        
        # Parameters for determining when to switch between tracking and matching
        self.tracking_quality_threshold = 0.45#0.6  # Coverage score threshold
        self.min_track_points = 4#20  # Minimum number of tracked points
        self.reinit_interval = 10  # Check if reinitialization needed every N frames

    # def _ensure_descriptor_shape(self, descriptors):
    #     """
    #     Ensure descriptors have the shape DxN (descriptor_dim x num_points)
    #     """
    #     if descriptors is None:
    #         return None
            
    #     # If there's only 1 descriptor, reshape it
    #     if len(descriptors.shape) == 1:
    #         return descriptors.reshape(-1, 1)
            
    #     # If the descriptor has more rows than columns, it's likely NxD instead of DxN
    #     if descriptors.shape[0] > descriptors.shape[1]:
    #         return descriptors.T
        
    #     return descriptors
    
    def _ensure_descriptor_shape(self, descriptors):
        if descriptors is None:
            return None
        # If descriptors are already in (N,256) format, return as is.
        if descriptors.shape[1] == 256:
            return descriptors
        # If descriptors are in (256, N) format, transpose them.
        if descriptors.shape[0] == 256:
            return descriptors.T
        # Otherwise, choose based on which dimension is likely to be the number of keypoints.
        if descriptors.shape[0] > descriptors.shape[1]:
            return descriptors.T
        return descriptors



    def _process_descriptor_batch(self, frame_feats):
        """
        Processes a batch of descriptors to ensure consistent format
        Returns descriptors in shape DxN (descriptor_dim x num_points)
        """
        # Get descriptors from frame features
        descriptors = frame_feats["descriptors"][0].detach().cpu().numpy()
        
        # SuperPoint descriptors usually come in shape DxN
        # But sometimes we might need to transpose
        return self._ensure_descriptor_shape(descriptors)
        

    def reinitialize_anchor(self, new_anchor_path, new_2d_points, new_3d_points):
        """
        Re-load a new anchor image and re-compute relevant data.
        """
        try:
            logger.info(f"Re-initializing anchor with new image: {new_anchor_path}")

            # Load new anchor image
            new_anchor_image = cv2.imread(new_anchor_path)
            if new_anchor_image is None:
                logger.error(f"Failed to load new anchor image at {new_anchor_path}")
                raise ValueError(f"Cannot read anchor image from {new_anchor_path}")

            # Resize the image
            new_anchor_image = self._resize_image(new_anchor_image, self.opt.resize)
            
            # Update anchor image (with lock)
            lock_acquired = self.session_lock.acquire(timeout=5.0)
            if not lock_acquired:
                logger.error("Could not acquire session lock to update anchor image (timeout)")
                raise TimeoutError("Lock acquisition timed out during anchor update")
                
            try:
                self.anchor_image = new_anchor_image
                logger.info("Anchor image updated")
            finally:
                self.session_lock.release()

            # Recompute anchor features
            success = self._set_anchor_features(
                anchor_bgr_image=new_anchor_image,
                anchor_keypoints_2D=new_2d_points,
                anchor_keypoints_3D=new_3d_points
            )
            
            if not success:
                logger.error("Failed to set anchor features")
                raise RuntimeError("Failed to set anchor features")

            # Reset tracking state
            self.tracking_mode = False
            self.last_frame = None
            self.last_frame_features = None
            self.last_frame_descriptors = None
            
            # Reset point tracker
            self.point_tracker = EnhancedPointTracker(
                max_length=self.opt.max_length if hasattr(self.opt, 'max_length') else 5,
                nn_thresh=self.opt.nn_thresh if hasattr(self.opt, 'nn_thresh') else 0.7,
                sliding_window_size=3
            )
            
            logger.info("Anchor re-initialization complete.")
            return True
            
        except Exception as e:
            logger.error(f"Error during anchor reinitialization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _convert_cv2_to_tensor(self, image):
        """Convert OpenCV BGR image to RGB tensor"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)

    def _set_anchor_features(self, anchor_bgr_image, anchor_keypoints_2D, anchor_keypoints_3D):
        """
        Run SuperPoint on the anchor image to get anchor_keypoints_sp.
        Then match those keypoints to known 2D->3D correspondences via KDTree.
        """
        try:
            start_time = time.time()
            logger.info("Starting anchor feature extraction...")
            
            lock_acquired = self.session_lock.acquire(timeout=10.0)
            if not lock_acquired:
                logger.error("Could not acquire session lock for anchor feature extraction (timeout)")
                return False
            
            try:
                # Precompute anchor's SuperPoint descriptors
                with torch.no_grad():
                    anchor_tensor = self._convert_cv2_to_tensor(anchor_bgr_image)
                    self.extractor.to(self.device)
                    self.anchor_feats = self.extractor.extract(anchor_tensor)
                    logger.info(f"Anchor features extracted in {time.time() - start_time:.3f}s")
                
                # Get anchor keypoints
                self.anchor_keypoints_sp = self.anchor_feats['keypoints'][0].detach().cpu().numpy()
                
                if len(self.anchor_keypoints_sp) == 0:
                    logger.error("No keypoints detected in anchor image!")
                    return False
                
                # Build KDTree to match anchor_keypoints_sp -> known anchor_keypoints_2D
                logger.info("Building KDTree for anchor keypoints...")
                sp_tree = cKDTree(self.anchor_keypoints_sp)
                distances, indices = sp_tree.query(anchor_keypoints_2D, k=1)
                valid_matches = distances < 5.0  # Threshold for "close enough"
                
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
                self.session_lock.release()
                logger.debug("Released session lock after anchor feature extraction")
                
        except Exception as e:
            logger.error(f"Error during anchor feature extraction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Make sure lock is released
            try:
                if hasattr(self.session_lock, '_is_owned') and self.session_lock._is_owned():
                    self.session_lock.release()
            except Exception as release_error:
                logger.error(f"Error releasing lock: {release_error}")
            
            return False

    def _resize_image(self, image, resize):
        """Resize image based on resize parameter."""
        logger.debug("Resizing image")
        if len(resize) == 2:
            return cv2.resize(image, tuple(resize))
        elif len(resize) == 1 and resize[0] > 0:
            h, w = image.shape[:2]
            scale = resize[0] / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(image, new_size)
        return image

    # def process_frame(self, frame, frame_idx):
    #     """
    #     Process a frame using either matching or tracking approach
    #     based on tracking conditions.
    #     """
    #     logger.info(f"Processing frame {frame_idx}")
    #     start_time = time.time()
    #     self.frame_count += 1

    #     # Resize frame to target resolution
    #     frame = self._resize_image(frame, self.opt.resize)
        
    #     # Determine whether to use tracking or matching
    #     should_reinitialize = (self.frame_count % self.reinit_interval == 0)
        
    #     if self.tracking_mode and not should_reinitialize:
    #         # Use tracking approach
    #         pose_data, visualization = self._process_with_tracking(frame, frame_idx)
            
    #         # If tracking fails, fall back to matching
    #         if pose_data is None or pose_data.get('pose_estimation_failed', False):
    #             logger.info(f"Tracking failed, falling back to matching for frame {frame_idx}")
    #             self.tracking_mode = False
    #             pose_data, visualization = self._process_with_matching(frame, frame_idx)
    #     else:
    #         # Use matching approach
    #         pose_data, visualization = self._process_with_matching(frame, frame_idx)
            
    #         # Check tracking quality and decide if we should switch to tracking mode
    #         quality_ok, metrics = self.point_tracker.get_quality_trend()
            
    #         if (metrics.get('coverage_score', 0) > self.tracking_quality_threshold and 
    #             metrics.get('inlier_ratio', 0) > 0.3 and #0.5 and 
    #             pose_data.get('num_inliers', 0) > self.min_track_points and
    #             not pose_data.get('pose_estimation_failed', False)):
    #             logger.info(f"Switching to tracking mode at frame {frame_idx}")
    #             self.tracking_mode = True
            
    #     # Store timing information
    #     process_time = time.time() - start_time
    #     logger.info(f"Frame {frame_idx} processed in {process_time:.3f}s. Mode: {'tracking' if self.tracking_mode else 'matching'}")
        
    #     return pose_data, visualization
    def process_frame(self, frame, frame_idx):
        """
        Process a frame using either matching or tracking approach
        based on tracking conditions.
        """
        logger.info(f"Processing frame {frame_idx}")
        start_time = time.time()
        self.frame_count += 1

        # Resize frame to target resolution
        frame = self._resize_image(frame, self.opt.resize)
        
        # Debug force tracking (optional)
        if hasattr(self, 'debug_tracking_enabled') and self.debug_tracking_enabled and frame_idx >= self.debug_force_tracking_after and not self.tracking_mode:
            logger.info(f"DEBUG: Forcing tracking mode at frame {frame_idx}")
            # Force tracking mode for debugging
            self.tracking_mode = True
            # Process with tracking first (will fall back to matching if it fails)
            pose_data, visualization = self._process_with_tracking(frame, frame_idx)
            if pose_data is None or pose_data.get('pose_estimation_failed', False):
                logger.info(f"DEBUG: Forced tracking failed, falling back to matching for frame {frame_idx}")
                self.tracking_mode = False
                pose_data, visualization = self._process_with_matching(frame, frame_idx)
            
            # Store timing information
            process_time = time.time() - start_time
            logger.info(f"Frame {frame_idx} processed in {process_time:.3f}s. Mode: {'tracking' if self.tracking_mode else 'matching'}")
            
            return pose_data, visualization
        
        # Determine whether to use tracking or matching
        should_reinitialize = (self.frame_count % self.reinit_interval == 0)
        
        # Process using the appropriate method
        if self.tracking_mode and not should_reinitialize:
            # Use tracking approach
            pose_data, visualization = self._process_with_tracking(frame, frame_idx)
            
            # If tracking fails, fall back to matching
            if pose_data is None or pose_data.get('pose_estimation_failed', False):
                logger.info(f"Tracking failed, falling back to matching for frame {frame_idx}")
                self.tracking_mode = False
                pose_data, visualization = self._process_with_matching(frame, frame_idx)
        else:
            # Use matching approach
            pose_data, visualization = self._process_with_matching(frame, frame_idx)
            
            # Only check tracking decision if pose_data exists and was successful
            if pose_data is not None and not pose_data.get('pose_estimation_failed', False):
                quality_ok, metrics = self.point_tracker.get_quality_trend()
                
                # Add detailed logging about tracking decision
                coverage_score = pose_data.get('coverage_score', 0)
                inlier_ratio = pose_data.get('inlier_ratio', 0)
                num_inliers = pose_data.get('num_inliers', 0)
                
                logger.info(f"Tracking decision metrics - Frame {frame_idx}:")
                logger.info(f"  Coverage score: {coverage_score:.3f} (threshold: {self.tracking_quality_threshold:.3f})")
                logger.info(f"  Inlier ratio: {inlier_ratio:.3f} (threshold: 0.5)")
                logger.info(f"  Num inliers: {num_inliers} (threshold: {self.min_track_points})")
                
                # Explicitly log each condition
                coverage_ok = coverage_score >= self.tracking_quality_threshold
                inlier_ratio_ok = inlier_ratio >= 0.5
                num_inliers_ok = num_inliers >= self.min_track_points
                
                logger.info(f"  Conditions met - Coverage: {coverage_ok}, Inlier ratio: {inlier_ratio_ok}, Num inliers: {num_inliers_ok}")
                
                if coverage_ok and inlier_ratio_ok and num_inliers_ok:
                    logger.info(f">>> Switching to tracking mode at frame {frame_idx}")
                    self.tracking_mode = True
                    # Force the point tracker to be updated with quality metrics
                    self.point_tracker.update_quality_metrics(inlier_ratio, 0, coverage_score)
                else:
                    logger.info(f">>> Staying in matching mode at frame {frame_idx}")
        
        # Store timing information
        process_time = time.time() - start_time
        logger.info(f"Frame {frame_idx} processed in {process_time:.3f}s. Mode: {'tracking' if self.tracking_mode else 'matching'}")
        
        return pose_data, visualization

    # def _process_with_matching(self, frame, frame_idx):
    #     """
    #     Process frame using feature matching against the anchor frame.
    #     This is similar to the original approach in PoseEstimator.
    #     """
    #     logger.info(f"Using feature matching approach for frame {frame_idx}")
    #     frame_tensor = self._convert_cv2_to_tensor(frame)
        
    #     # Extract features from the frame
    #     with torch.no_grad():
    #         frame_feats = self.extractor.extract(frame_tensor)
            
    #         # Match features between anchor and frame
    #         with self.session_lock:
    #             matches_dict = self.matcher({
    #                 'image0': self.anchor_feats, 
    #                 'image1': frame_feats
    #             })
        
    #     # Remove batch dimension and move to CPU
    #     feats0, feats1, matches01 = [rbd(x) for x in [self.anchor_feats, frame_feats, matches_dict]]
        
    #     # Get keypoints and matches
    #     kpts0 = feats0["keypoints"].detach().cpu().numpy()
    #     kpts1 = feats1["keypoints"].detach().cpu().numpy()
    #     matches = matches01["matches"].detach().cpu().numpy()
    #     confidence = matches01.get("scores", torch.ones(len(matches))).detach().cpu().numpy()
        
    #     mkpts0 = kpts0[matches[:, 0]]
    #     mkpts1 = kpts1[matches[:, 1]]
    #     mconf = confidence

    #     # Store for potential use in tracking
    #     # Store for potential use in tracking
    #     self.last_frame = frame
    #     self.last_frame_features = kpts1
    #     descriptors = feats1["descriptors"].detach().cpu().numpy()
    #     self.last_frame_descriptors = self._ensure_descriptor_shape(descriptors)
    #     logger.debug(f"Found {len(mkpts0)} raw matches in frame {frame_idx}")

    #     # Filter to known anchor indices
    #     valid_indices = matches[:, 0]
    #     known_mask = np.isin(valid_indices, self.matched_anchor_indices)
        
    #     if not np.any(known_mask):
    #         logger.warning(f"No valid matches to 3D points found for frame {frame_idx}")
            
    #         # Use Kalman prediction
    #         x_pred, P_pred = self.kf_pose.predict()
    #         translation_estimated = x_pred[0:3]
    #         q_estimated = x_pred[6:10]
    #         R_estimated = quaternion_to_rotation_matrix(q_estimated)
            
    #         pose_data = {
    #             'frame': frame_idx,
    #             'kf_translation_vector': translation_estimated.tolist(),
    #             'kf_quaternion': q_estimated.tolist(),
    #             'kf_rotation_matrix': R_estimated.tolist(),
    #             'pose_estimation_failed': True
    #         }
            
    #         return pose_data, frame
        
    #     # Filter matches to known 3D points
    #     mkpts0 = mkpts0[known_mask]
    #     mkpts1 = mkpts1[known_mask]
    #     mconf = mconf[known_mask]
    #     valid_indices = valid_indices[known_mask]
        
    #     # Get corresponding 3D points
    #     idx_map = {idx: i for i, idx in enumerate(self.matched_anchor_indices)}
    #     mpts3D = np.array([
    #         self.matched_3D_keypoints[idx_map[aidx]] 
    #         for aidx in valid_indices if aidx in idx_map
    #     ])

    #     # Update the point tracker with matched points
    #     # Create a properly shaped matched_pts array [x, y, conf]
    #     matched_pts = np.zeros((3, len(mkpts1)))
    #     matched_pts[0, :] = mkpts1[:, 0]  # x coordinates
    #     matched_pts[1, :] = mkpts1[:, 1]  # y coordinates
    #     matched_pts[2, :] = mconf          # confidence scores

    #     # Get descriptors for matched points with correct shape
    #     # This ensures desc shape is DxN where D is descriptor dimension and N is number of points
    #     matched_desc = self.last_frame_descriptors[matches[:, 1][known_mask]]
    #     if matched_desc.shape[0] < matched_desc.shape[1]:  # If shape is NxD instead of DxN
    #         matched_desc = matched_desc.T

    #     # Update tracker with new points and their associated 3D coordinates
    #     # Pass mpts3D directly as it already matches the valid points
    #     self.point_tracker.update(matched_pts, matched_desc, mpts3D)

    #     if len(mkpts1) < 4:
    #         logger.warning(f"Not enough matches for pose (frame {frame_idx})")
            
    #         # Use Kalman prediction
    #         x_pred, P_pred = self.kf_pose.predict()
    #         translation_estimated = x_pred[0:3]
    #         q_estimated = x_pred[6:10]
    #         R_estimated = quaternion_to_rotation_matrix(q_estimated)
            
    #         pose_data = {
    #             'frame': frame_idx,
    #             'kf_translation_vector': translation_estimated.tolist(),
    #             'kf_quaternion': q_estimated.tolist(),
    #             'kf_rotation_matrix': R_estimated.tolist(),
    #             'pose_estimation_failed': True
    #         }
            
    #         return pose_data, frame

    #     # Estimate pose
    #     pose_data, visualization = self.estimate_pose(
    #         mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, kpts1
    #     )
        
    #     # If pose estimation failed, use the Kalman filter prediction
    #     if pose_data is None:
    #         logger.warning(f"Pose estimation failed for frame {frame_idx}; using Kalman prediction.")
    #         x_pred, P_pred = self.kf_pose.predict()
    #         translation_estimated = x_pred[0:3]
    #         q_estimated = x_pred[6:10]
    #         R_estimated = quaternion_to_rotation_matrix(q_estimated)
            
    #         pose_data = {
    #             'frame': frame_idx,
    #             'kf_translation_vector': translation_estimated.tolist(),
    #             'kf_quaternion': q_estimated.tolist(),
    #             'kf_rotation_matrix': R_estimated.tolist(),
    #             'pose_estimation_failed': True
    #         }
    #    
    #    return pose_data, visualization

    # Modify the _process_with_matching method to properly update the point tracker

    # def _process_with_matching(self, frame, frame_idx):
    #     """
    #     Process frame using feature matching against the anchor frame.
    #     This is similar to the original approach in PoseEstimator.
    #     """
    #     logger.info(f"Using feature matching approach for frame {frame_idx}")
    #     frame_tensor = self._convert_cv2_to_tensor(frame)
        
    #     # Extract features from the frame
    #     with torch.no_grad():
    #         frame_feats = self.extractor.extract(frame_tensor)
            
    #         # Match features between anchor and frame
    #         with self.session_lock:
    #             matches_dict = self.matcher({
    #                 'image0': self.anchor_feats, 
    #                 'image1': frame_feats
    #             })
        
    #     # Remove batch dimension and move to CPU
    #     feats0, feats1, matches01 = [rbd(x) for x in [self.anchor_feats, frame_feats, matches_dict]]
        
    #     # Get keypoints and matches
    #     kpts0 = feats0["keypoints"].detach().cpu().numpy()
    #     kpts1 = feats1["keypoints"].detach().cpu().numpy()
    #     matches = matches01["matches"].detach().cpu().numpy()
    #     confidence = matches01.get("scores", torch.ones(len(matches))).detach().cpu().numpy()
        
    #     mkpts0 = kpts0[matches[:, 0]]
    #     mkpts1 = kpts1[matches[:, 1]]
    #     mconf = confidence

    #     # Store for potential use in tracking - use helper method to ensure consistent shape
    #     self.last_frame = frame
    #     self.last_frame_features = kpts1
    #     descriptors = feats1["descriptors"].detach().cpu().numpy()
    #     self.last_frame_descriptors = self._ensure_descriptor_shape(descriptors)
        
    #     logger.debug(f"Found {len(mkpts0)} raw matches in frame {frame_idx}")

    #     # Filter to known anchor indices
    #     valid_indices = matches[:, 0]
    #     known_mask = np.isin(valid_indices, self.matched_anchor_indices)
        
    #     if not np.any(known_mask):
    #         logger.warning(f"No valid matches to 3D points found for frame {frame_idx}")
            
    #         # Use Kalman prediction
    #         x_pred, P_pred = self.kf_pose.predict()
    #         translation_estimated = x_pred[0:3]
    #         q_estimated = x_pred[6:10]
    #         R_estimated = quaternion_to_rotation_matrix(q_estimated)
            
    #         pose_data = {
    #             'frame': frame_idx,
    #             'kf_translation_vector': translation_estimated.tolist(),
    #             'kf_quaternion': q_estimated.tolist(),
    #             'kf_rotation_matrix': R_estimated.tolist(),
    #             'pose_estimation_failed': True
    #         }
            
    #         return pose_data, frame
        
    #     # Filter matches to known 3D points
    #     mkpts0 = mkpts0[known_mask]
    #     mkpts1 = mkpts1[known_mask]
    #     mconf = mconf[known_mask]
    #     valid_indices = valid_indices[known_mask]
        
    #     # Get corresponding 3D points
    #     idx_map = {idx: i for i, idx in enumerate(self.matched_anchor_indices)}
    #     mpts3D = np.array([
    #         self.matched_3D_keypoints[idx_map[aidx]] 
    #         for aidx in valid_indices if aidx in idx_map
    #     ])

    #     # Create the proper matched_pts array [x, y, conf]
    #     matched_pts = np.zeros((3, len(mkpts1)))
    #     matched_pts[0, :] = mkpts1[:, 0]  # x coordinates
    #     matched_pts[1, :] = mkpts1[:, 1]  # y coordinates
    #     matched_pts[2, :] = mconf          # confidence scores

    #     # Get descriptors for matched points with correct shape
    #     filtered_descriptors = self.last_frame_descriptors[:, matches[:, 1][known_mask]] if self.last_frame_descriptors.shape[0] > self.last_frame_descriptors.shape[1] else self.last_frame_descriptors[matches[:, 1][known_mask]].T
        
    #     # Log shapes for debugging
    #     logger.debug(f"Matched pts shape: {matched_pts.shape}, Filtered desc shape: {filtered_descriptors.shape}, 3D points shape: {mpts3D.shape}")
        
    #     # Update tracker with new points and their associated 3D coordinates
    #     try:
    #         self.point_tracker.update(matched_pts, filtered_descriptors, mpts3D)
    #     except Exception as e:
    #         logger.error(f"Error updating point tracker: {e}")
    #         import traceback
    #         logger.error(traceback.format_exc())

    #     if len(mkpts1) < 4:
    #         logger.warning(f"Not enough matches for pose (frame {frame_idx})")
            
    #         # Use Kalman prediction
    #         x_pred, P_pred = self.kf_pose.predict()
    #         translation_estimated = x_pred[0:3]
    #         q_estimated = x_pred[6:10]
    #         R_estimated = quaternion_to_rotation_matrix(q_estimated)
            
    #         pose_data = {
    #             'frame': frame_idx,
    #             'kf_translation_vector': translation_estimated.tolist(),
    #             'kf_quaternion': q_estimated.tolist(),
    #             'kf_rotation_matrix': R_estimated.tolist(),
    #             'pose_estimation_failed': True
    #         }
            
    #         return pose_data, frame

    #     # Estimate pose
    #     pose_data, visualization = self.estimate_pose(
    #         mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, kpts1
    #     )
        
    #     # If pose estimation failed, use the Kalman filter prediction
    #     if pose_data is None:
    #         logger.warning(f"Pose estimation failed for frame {frame_idx}; using Kalman prediction.")
    #         x_pred, P_pred = self.kf_pose.predict()
    #         translation_estimated = x_pred[0:3]
    #         q_estimated = x_pred[6:10]
    #         R_estimated = quaternion_to_rotation_matrix(q_estimated)
            
    #         pose_data = {
    #             'frame': frame_idx,
    #             'kf_translation_vector': translation_estimated.tolist(),
    #             'kf_quaternion': q_estimated.tolist(),
    #             'kf_rotation_matrix': R_estimated.tolist(),
    #             'pose_estimation_failed': True
    #         }
    #     else:
    #         # If pose estimation succeeded, update tracking quality metrics
    #         self.point_tracker.update_quality_metrics(
    #             pose_data.get('inlier_ratio', 0),
    #             pose_data.get('mean_reprojection_error', 10),
    #             pose_data.get('coverage_score', 0)
    #         )
        
    #     return pose_data, visualization
    def _process_with_matching(self, frame, frame_idx):
        """
        Process frame using feature matching against the anchor frame.
        This is similar to the original approach in PoseEstimator.
        """
        logger.info(f"Using feature matching approach for frame {frame_idx}")
        frame_tensor = self._convert_cv2_to_tensor(frame)
        
        # Extract features from the frame
        with torch.no_grad():
            frame_feats = self.extractor.extract(frame_tensor)
            
            # Match features between anchor and frame
            with self.session_lock:
                matches_dict = self.matcher({
                    'image0': self.anchor_feats, 
                    'image1': frame_feats
                })
        
        # Remove batch dimension and move to CPU
        feats0, feats1, matches01 = [rbd(x) for x in [self.anchor_feats, frame_feats, matches_dict]]
        
        # Get keypoints and matches
        kpts0 = feats0["keypoints"].detach().cpu().numpy()
        kpts1 = feats1["keypoints"].detach().cpu().numpy()
        matches = matches01["matches"].detach().cpu().numpy()
        confidence = matches01.get("scores", torch.ones(len(matches))).detach().cpu().numpy()
        
        mkpts0 = kpts0[matches[:, 0]]
        mkpts1 = kpts1[matches[:, 1]]
        mconf = confidence

        # Store for potential use in tracking - directly store descriptors without transformation
        self.last_frame = frame
        self.last_frame_features = kpts1
        self.last_frame_descriptors = feats1["descriptors"].detach().cpu().numpy()
        
        logger.debug(f"Found {len(mkpts0)} raw matches in frame {frame_idx}")

        # Filter to known anchor indices
        valid_indices = matches[:, 0]
        known_mask = np.isin(valid_indices, self.matched_anchor_indices)
        
        if not np.any(known_mask):
            logger.warning(f"No valid matches to 3D points found for frame {frame_idx}")
            
            # Use Kalman prediction
            x_pred, P_pred = self.kf_pose.predict()
            translation_estimated = x_pred[0:3]
            q_estimated = x_pred[6:10]
            R_estimated = quaternion_to_rotation_matrix(q_estimated)
            
            pose_data = {
                'frame': frame_idx,
                'kf_translation_vector': translation_estimated.tolist(),
                'kf_quaternion': q_estimated.tolist(),
                'kf_rotation_matrix': R_estimated.tolist(),
                'pose_estimation_failed': True
            }
            
            return pose_data, frame
        
        # Filter matches to known 3D points
        mkpts0 = mkpts0[known_mask]
        mkpts1 = mkpts1[known_mask]
        mconf = mconf[known_mask]
        valid_indices = valid_indices[known_mask]
        
        # Get corresponding 3D points
        idx_map = {idx: i for i, idx in enumerate(self.matched_anchor_indices)}
        mpts3D = np.array([
            self.matched_3D_keypoints[idx_map[aidx]] 
            for aidx in valid_indices if aidx in idx_map
        ])

        # Prepare filtered descriptors for the tracker
        filtered_indices = matches[:, 1][known_mask]
        
        # Create the proper matched_pts array [x, y, conf]
        matched_pts = np.zeros((3, len(mkpts1)))
        matched_pts[0, :] = mkpts1[:, 0]  # x coordinates
        matched_pts[1, :] = mkpts1[:, 1]  # y coordinates
        matched_pts[2, :] = mconf          # confidence scores

        # Use indices to get the correct descriptors and let the tracker handle normalization
        try:
            # Get the descriptors using the filtered indices
            filtered_descriptors = self.last_frame_descriptors[:, filtered_indices]
            
            logger.debug(f"Matched pts shape: {matched_pts.shape}, 3D points shape: {mpts3D.shape}")
            
            # Update tracker with new points and their associated 3D coordinates
            self.point_tracker.update(matched_pts, filtered_descriptors, mpts3D)
        except Exception as e:
            logger.error(f"Error updating point tracker: {e}")
            import traceback
            logger.error(traceback.format_exc())

        if len(mkpts1) < 4:
            logger.warning(f"Not enough matches for pose (frame {frame_idx})")
            
            # Use Kalman prediction
            x_pred, P_pred = self.kf_pose.predict()
            translation_estimated = x_pred[0:3]
            q_estimated = x_pred[6:10]
            R_estimated = quaternion_to_rotation_matrix(q_estimated)
            
            pose_data = {
                'frame': frame_idx,
                'kf_translation_vector': translation_estimated.tolist(),
                'kf_quaternion': q_estimated.tolist(),
                'kf_rotation_matrix': R_estimated.tolist(),
                'pose_estimation_failed': True
            }
            
            return pose_data, frame

        # Estimate pose
        pose_data, visualization = self.estimate_pose(
            mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, kpts1
        )
        
        # If pose estimation failed, use the Kalman filter prediction
        if pose_data is None:
            logger.warning(f"Pose estimation failed for frame {frame_idx}; using Kalman prediction.")
            x_pred, P_pred = self.kf_pose.predict()
            translation_estimated = x_pred[0:3]
            q_estimated = x_pred[6:10]
            R_estimated = quaternion_to_rotation_matrix(q_estimated)
            
            pose_data = {
                'frame': frame_idx,
                'kf_translation_vector': translation_estimated.tolist(),
                'kf_quaternion': q_estimated.tolist(),
                'kf_rotation_matrix': R_estimated.tolist(),
                'pose_estimation_failed': True
            }
        else:
            # If pose estimation succeeded, update tracking quality metrics
            self.point_tracker.update_quality_metrics(
                pose_data.get('inlier_ratio', 0),
                pose_data.get('mean_reprojection_error', 10),
                pose_data.get('coverage_score', 0)
            )
            
            # Include tracking decision metrics in log
            coverage_score = pose_data.get('coverage_score', 0)
            inlier_ratio = pose_data.get('inlier_ratio', 0)
            num_inliers = pose_data.get('num_inliers', 0)
            
            logger.debug(f"Frame {frame_idx} metrics - Coverage: {coverage_score:.2f}, IR: {inlier_ratio:.2f}, Inliers: {num_inliers}")
        
        return pose_data, visualization

    # def _process_with_tracking(self, frame, frame_idx):
    #     """
    #     Process frame using tracking between consecutive frames.
    #     This uses the EnhancedPointTracker to maintain consistent tracks.
    #     """
    #     logger.info(f"Using tracking approach for frame {frame_idx}")
        
    #     # Extract features for the current frame
    #     frame_tensor = self._convert_cv2_to_tensor(frame)
        
    #     with torch.no_grad():
    #         frame_feats = self.extractor.extract(frame_tensor)
        
    #     # Get keypoints and descriptors
    #     kpts = frame_feats["keypoints"][0].detach().cpu().numpy()
    #     #desc = frame_feats["descriptors"][0].detach().cpu().numpy()
    #     desc = self._process_descriptor_batch(frame_feats)

    #     # Match descriptors between last frame and current frame
    #     if self.last_frame_descriptors is not None and self.last_frame_features is not None:
    #         try:
    #             # Make sure descriptors have matching dimensions for nn_match_two_way
    #             last_descs = self.last_frame_descriptors
    #             curr_descs = desc
                
    #             # Ensure descriptor matrices have shape DxN (D dimensions, N points)
    #             if last_descs.shape[0] < last_descs.shape[1]:  # If shape is NxD instead of DxN
    #                 last_descs = last_descs.T
    #             if curr_descs.shape[0] < curr_descs.shape[1]:  # If shape is NxD instead of DxN
    #                 curr_descs = curr_descs.T
                
    #             # Log descriptor shapes for debugging
    #             logger.debug(f"Last descriptor shape: {last_descs.shape}, Current descriptor shape: {curr_descs.shape}")
                
    #             # Verify descriptor dimensions match
    #             if last_descs.shape[0] != curr_descs.shape[0]:
    #                 logger.warning(f"Descriptor dimensions don't match: {last_descs.shape[0]} vs {curr_descs.shape[0]}")
    #                 if last_descs.shape[0] == curr_descs.shape[1] and last_descs.shape[1] == curr_descs.shape[0]:
    #                     logger.warning("Transposing current descriptors to match dimensions")
    #                     curr_descs = curr_descs.T
                
    #             # Perform matching with correct dimensions
    #             matches = self.point_tracker.nn_match_two_way(
    #                 last_descs, 
    #                 curr_descs, 
    #                 self.point_tracker.nn_thresh
    #             )
                
    #             if matches.shape[1] > 0:
    #                 # Get matched points from last frame and current frame
    #                 last_kpts_idx = matches[0, :].astype(int)
    #                 curr_kpts_idx = matches[1, :].astype(int)
    #                 match_scores = matches[2, :]
                    
    #                 # Create point observation matrix [x, y, conf]
    #                 pts = np.zeros((3, len(curr_kpts_idx)))
    #                 pts[0, :] = kpts[curr_kpts_idx, 0]  # x coordinates
    #                 pts[1, :] = kpts[curr_kpts_idx, 1]  # y coordinates
    #                 pts[2, :] = match_scores  # confidence scores
                    
    #                 # Get corresponding descriptors
    #                 matched_desc = curr_descs[:, curr_kpts_idx]  # Get descriptors with correct shape (DxN)
                    
    #                 # Update point tracker
    #                 tracked_pts3d = self.point_tracker.update(pts, matched_desc)
                    
    #                 # Get active tracks with 3D points for pose estimation
    #                 tracks_2d, tracks_3d = self.point_tracker.get_active_tracks_with_3d()
                    
    #                 if len(tracks_2d) >= 4 and len(tracks_3d) >= 4:
    #                     # We have enough tracks with 3D points for pose estimation
    #                     # Convert to the format expected by estimate_pose
    #                     mkpts0 = np.zeros((0, 2))  # Not used in tracking mode
    #                     mkpts1 = tracks_2d
    #                     mpts3D = tracks_3d
    #                     mconf = np.ones(len(tracks_2d))  # Placeholder confidences
                        
    #                     # Estimate pose using tracked points
    #                     pose_data, visualization = self.estimate_pose(
    #                         mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, kpts
    #                     )
                        
    #                     # Check if estimation was successful
    #                     if pose_data is not None and not pose_data.get('pose_estimation_failed', False):
    #                         # Update tracking quality metrics
    #                         self.point_tracker.update_quality_metrics(
    #                             pose_data.get('inlier_ratio', 0),
    #                             pose_data.get('mean_reprojection_error', 10),
    #                             pose_data.get('coverage_score', 0)
    #                         )
                            
    #                         # Check if tracking quality is good enough to continue in tracking mode
    #                         quality_ok, metrics = self.point_tracker.get_quality_trend()
                            
    #                         if not quality_ok:
    #                             logger.info(f"Tracking quality degraded at frame {frame_idx}, will switch to matching mode")
    #                             self.tracking_mode = False
                            
    #                         # Store current frame for next iteration
    #                         self.last_frame = frame
    #                         self.last_frame_features = kpts
    #                         self.last_frame_descriptors = curr_descs  # Save descriptors in their current shape
                            
    #                         return pose_data, visualization
    #         except Exception as e:
    #             logger.error(f"Error during tracking: {e}")
    #             import traceback
    #             logger.error(traceback.format_exc())
        
    #     # If tracking failed or we don't have enough points, fall back to matching
    #     logger.info(f"Tracking approach failed, falling back to matching for frame {frame_idx}")
    #     self.tracking_mode = False
    #     return self._process_with_matching(frame, frame_idx)
    
    def _process_with_tracking(self, frame, frame_idx):
        """
        Process frame using tracking between consecutive frames.
        This uses the EnhancedPointTracker to maintain consistent tracks.
        """
        logger.info(f"Using tracking approach for frame {frame_idx}")
        
        # Extract features for the current frame
        frame_tensor = self._convert_cv2_to_tensor(frame)
        
        with torch.no_grad():
            frame_feats = self.extractor.extract(frame_tensor)
        
        # Get keypoints and descriptors
        kpts = frame_feats["keypoints"][0].detach().cpu().numpy()
        desc = frame_feats["descriptors"][0].detach().cpu().numpy()
        
        # Log descriptor shape for debugging
        logger.debug(f"Current frame descriptors shape: {desc.shape}")
        
        # Match descriptors between last frame and current frame
        if self.last_frame_descriptors is not None and self.last_frame_features is not None:
            try:
                # Let the point tracker handle the descriptor matching with its robust handling
                matches = self.point_tracker.nn_match_two_way(
                    self.last_frame_descriptors, 
                    desc, 
                    self.point_tracker.nn_thresh
                )
                
                if matches.shape[1] > 0:
                    # Get matched points from last frame and current frame
                    last_kpts_idx = matches[0, :].astype(int)
                    curr_kpts_idx = matches[1, :].astype(int)
                    match_scores = matches[2, :]
                    
                    # Create point observation matrix [x, y, conf]
                    pts = np.zeros((3, len(curr_kpts_idx)))
                    pts[0, :] = kpts[curr_kpts_idx, 0]  # x coordinates
                    pts[1, :] = kpts[curr_kpts_idx, 1]  # y coordinates
                    pts[2, :] = match_scores  # confidence scores
                    
                    # Log indices and shapes for debugging
                    logger.debug(f"Matched {len(curr_kpts_idx)} points between frames")
                    logger.debug(f"Current keypoints shape: {kpts.shape}")
                    logger.debug(f"Matched indices range: min={np.min(curr_kpts_idx) if len(curr_kpts_idx) > 0 else 'N/A'}, max={np.max(curr_kpts_idx) if len(curr_kpts_idx) > 0 else 'N/A'}")
                    
                    # Get descriptors for matched points - create a proper descriptor matrix
                    # Handling descriptor shapes with special care
                    if desc.shape[0] == 256 or (desc.shape[0] > desc.shape[1] and desc.shape[0] > 200):
                        # Descriptors are in format (256, N) - correct format
                        if np.max(curr_kpts_idx) < desc.shape[1]:
                            matched_desc = desc[:, curr_kpts_idx]
                        else:
                            # Indices out of bounds - reshape descriptor
                            logger.warning(f"Descriptor indices out of bounds: {np.max(curr_kpts_idx)} >= {desc.shape[1]}")
                            # Create a normalized descriptor matrix for all keypoints
                            all_desc = np.zeros((desc.shape[0], len(kpts)))
                            all_desc[:, :desc.shape[1]] = desc
                            # Now select the matched keypoints
                            valid_idx = curr_kpts_idx[curr_kpts_idx < all_desc.shape[1]]
                            matched_desc = all_desc[:, valid_idx]
                            # Update points to match valid descriptors
                            pts = pts[:, curr_kpts_idx < all_desc.shape[1]]
                    else:
                        # Descriptors might be in format (N, 256) - need transpose
                        matched_desc = desc[curr_kpts_idx].T
                    
                    logger.debug(f"Matched descriptors shape: {matched_desc.shape}")
                    
                    # Update point tracker - let it handle descriptor normalization
                    tracked_pts3d = self.point_tracker.update(pts, matched_desc)
                    
                    # Get active tracks with 3D points for pose estimation
                    tracks_2d, tracks_3d = self.point_tracker.get_active_tracks_with_3d()
                    
                    if len(tracks_2d) >= 4 and len(tracks_3d) >= 4:
                        # We have enough tracks with 3D points for pose estimation
                        # Convert to the format expected by estimate_pose
                        mkpts0 = np.zeros((0, 2))  # Not used in tracking mode
                        mkpts1 = tracks_2d
                        mpts3D = tracks_3d
                        mconf = np.ones(len(tracks_2d))  # Placeholder confidences
                        
                        logger.info(f"Tracking with {len(tracks_2d)} tracked points having 3D associations")
                        
                        # Estimate pose using tracked points
                        pose_data, visualization = self.estimate_pose(
                            mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, kpts
                        )
                        
                        # Check if estimation was successful
                        if pose_data is not None and not pose_data.get('pose_estimation_failed', False):
                            # Update tracking quality metrics
                            self.point_tracker.update_quality_metrics(
                                pose_data.get('inlier_ratio', 0),
                                pose_data.get('mean_reprojection_error', 10),
                                pose_data.get('coverage_score', 0)
                            )
                            
                            # Check if tracking quality is good enough to continue in tracking mode
                            quality_ok, metrics = self.point_tracker.get_quality_trend()
                            
                            if not quality_ok:
                                logger.info(f"Tracking quality degraded at frame {frame_idx}, will switch to matching mode")
                                self.tracking_mode = False
                            
                            # Store current frame for next iteration
                            self.last_frame = frame
                            self.last_frame_features = kpts
                            self.last_frame_descriptors = desc
                            
                            # Add tracking mode flag to pose data
                            pose_data['tracking_mode'] = True
                            
                            return pose_data, visualization
                    else:
                        logger.info(f"Not enough tracked points with 3D associations: {len(tracks_2d)}")
                else:
                    logger.info(f"No matches found between frames")
            except Exception as e:
                logger.error(f"Error during tracking: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # If tracking failed or we don't have enough points, fall back to matching
        logger.info(f"Tracking approach failed, falling back to matching for frame {frame_idx}")
        self.tracking_mode = False
        return self._process_with_matching(frame, frame_idx)

    def _init_kalman_filter(self):
        """Initialize Kalman filter for pose tracking."""
        frame_rate = 30
        dt = 1 / frame_rate
        kf_pose = KalmanFilterPose(dt)
        return kf_pose

    def estimate_pose(self, mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, frame_keypoints):
        """
        Estimate the pose using PnP-RANSAC.
        This remains similar to the original implementation.
        """
        logger.debug(f"Estimating pose for frame {frame_idx}")
        K, distCoeffs = self._get_camera_intrinsics()

        objectPoints = mpts3D.reshape(-1, 1, 3)
        imagePoints = mkpts1.reshape(-1, 1, 2).astype(np.float32)

        # Solve PnP
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
            return None, frame

        # Refine with VVS
        objectPoints_inliers = objectPoints[inliers.flatten()]
        imagePoints_inliers = imagePoints[inliers.flatten()]

        rvec, tvec = cv2.solvePnPRefineVVS(
            objectPoints=objectPoints_inliers,
            imagePoints=imagePoints_inliers,
            cameraMatrix=K,
            distCoeffs=distCoeffs,
            rvec=rvec_o,
            tvec=tvec_o
        )

        # R, t => object_in_cam
        R, _ = cv2.Rodrigues(rvec)

        # Calculate region distribution
        regions = {"front-right": 0, "front-left": 0, "back-right": 0, "back-left": 0}
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
            valid_conf = mconf[inliers.flatten()] if len(inliers) > 0 else []
            
            if len(valid_conf) == 0 or np.isnan(valid_conf).any():
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
                
                # Calculate mean confidence
                mean_confidence = np.mean(valid_conf)
                
                # Final coverage score
                coverage_score = normalized_entropy * mean_confidence
                
                # Ensure score is in valid range [0,1]
                coverage_score = np.clip(coverage_score, 0, 1)
        else:
            coverage_score = 0

        # Compute reprojection errors
        projected_points, _ = cv2.projectPoints(
            objectPoints_inliers, rvec, tvec, K, distCoeffs
        )
        reprojection_errors = np.linalg.norm(imagePoints_inliers - projected_points, axis=2).flatten()
        mean_reprojection_error = np.mean(reprojection_errors)
        std_reprojection_error = np.std(reprojection_errors)

        # Update pose with Kalman filter
        pose_data = self._kalman_filter_update(
            R, tvec, reprojection_errors, mean_reprojection_error,
            std_reprojection_error, inliers, mkpts0, mkpts1, mpts3D,
            mconf, frame_idx, rvec_o, rvec, coverage_score
        )
        
        pose_data["region_distribution"] = regions
        pose_data["coverage_score"] = coverage_score

        # Create visualization
        if self.tracking_mode:
            # In tracking mode, visualize tracks
            visualization = self._visualize_tracks(frame, inliers, mpts3D, pose_data, frame_keypoints)
        else:
            # In matching mode, visualize matches
            visualization = self._visualize_matches(
                frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints
            )
            
        return pose_data, visualization

    def _kalman_filter_update(
        self, R, tvec, reprojection_errors, mean_reprojection_error,
        std_reprojection_error, inliers, mkpts0, mkpts1, mpts3D,
        mconf, frame_idx, rvec_o, rvec, coverage_score
    ):
        """
        Update Kalman filter with new pose measurements.
        Remains similar to the original implementation.
        """
        num_inliers = len(inliers)
        inlier_ratio = num_inliers / len(mkpts0) if len(mkpts0) > 0 else 0

        reprojection_error_threshold = 4.0
        max_translation_jump = 2.0
        max_orientation_jump = 20.0  # degrees
        min_inlier = 4
        coverage_threshold = 0.3  # Lower threshold than switch to tracking

        # 1) Convert measured rotation R -> quaternion
        q_measured = rotation_matrix_to_quaternion(R)

        # 2) EKF PREDICT: get prior (x^-)
        x_pred, P_pred = self.kf_pose.predict()  

        # Parse predicted orientation for threshold checks
        px_pred, py_pred, pz_pred = x_pred[0:3]
        qx_pred, qy_pred, qz_pred, qw_pred = x_pred[6:10]

        # Calculate orientation change
        def quaternion_angle_degrees(q1, q2):
            q1 = q1 / np.linalg.norm(q1)
            q2 = q2 / np.linalg.norm(q2)
            dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
            angle = 2.0 * np.degrees(np.arccos(dot))
            if angle > 180.0:
                angle = 360.0 - angle
            return angle

        orientation_change = quaternion_angle_degrees(q_measured, [qx_pred,qy_pred,qz_pred,qw_pred])
        translation_change = np.linalg.norm(tvec.flatten() - x_pred[0:3])

        # 3) Build the measurement vector z = [px,py,pz, qx,qy,qz,qw]
        tvec = tvec.flatten()
        z_meas = np.array([
            tvec[0], tvec[1], tvec[2],
            q_measured[0], q_measured[1], q_measured[2], q_measured[3]
        ], dtype=np.float64)

        # First update or check thresholds
        if not hasattr(self, 'kf_pose_first_update') or self.kf_pose_first_update:
            # First update - skip thresholds
            x_upd, P_upd = self.kf_pose.update(z_meas)
            self.kf_pose_first_update = False
            logger.debug("EKF first update: skipping threshold checks.")
        else:
            # Check quality thresholds
            quality_check = (
                mean_reprojection_error < reprojection_error_threshold and 
                num_inliers > min_inlier and
                translation_change < max_translation_jump and 
                orientation_change < max_orientation_jump and
                coverage_score >= coverage_threshold
            )
            
            if quality_check:
                # All checks passed - do the update
                x_upd, P_upd = self.kf_pose.update(z_meas)
                logger.debug("EKF update (all thresholds passed).")
            else:
                # Some check failed - use prediction
                x_upd, P_upd = x_pred, P_pred
                logger.debug(f"Skipping EKF update: quality check failed (repro_err={mean_reprojection_error:.2f}, inliers={num_inliers}, coverage={coverage_score:.2f})")

        # Get final pose
        x_final = x_upd
        px, py, pz = x_final[0:3]
        qx, qy, qz, qw = x_final[6:10]
        R_estimated = quaternion_to_rotation_matrix([qx, qy, qz, qw])

        # Build result
        pose_data = {
            'frame': frame_idx,
            'object_rotation_in_cam': R.tolist(),
            'object_translation_in_cam': tvec.flatten().tolist(),
            'raw_rvec': rvec_o.flatten().tolist(),
            'refined_raw_rvec': rvec.flatten().tolist(),
            'num_inliers': num_inliers,
            'total_matches': len(mkpts0) if len(mkpts0) > 0 else len(mkpts1),
            'inlier_ratio': inlier_ratio,
            'reprojection_errors': reprojection_errors.tolist(),
            'mean_reprojection_error': float(mean_reprojection_error),
            'std_reprojection_error': float(std_reprojection_error),
            'inliers': inliers.flatten().tolist(),
            'mkpts0': mkpts0.tolist() if len(mkpts0) > 0 else [],
            'mkpts1': mkpts1.tolist(),
            'mpts3D': mpts3D.tolist(),
            'mconf': mconf.tolist(),

            # Filtered results from updated state:
            'kf_translation_vector': [px, py, pz],
            'kf_quaternion': [qx, qy, qz, qw],
            'kf_rotation_matrix': R_estimated.tolist(),

            # Additional metrics
            'coverage_score': coverage_score,
            'tracking_mode': self.tracking_mode,
        }
        return pose_data

    def _visualize_matches(self, frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints):
        """
        Visualize feature matches between anchor and current frame.
        Used when in matching mode.
        """
        anchor_image_gray = cv2.cvtColor(self.anchor_image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        inlier_idx = inliers.flatten()
        
        # Make sure indices are valid
        valid_indices = (inlier_idx < len(mkpts0)) & (inlier_idx < len(mkpts1))
        inlier_idx = inlier_idx[valid_indices]
        
        if len(inlier_idx) == 0:
            # No valid inliers - create a simple visualization
            h, w = frame_gray.shape
            out = np.zeros((h, w*2), dtype=np.uint8)
            out[:, :w] = anchor_image_gray
            out[:, w:] = frame_gray
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            
            # Add text indicating no valid matches
            cv2.putText(out, "No valid inliers", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Show pose information
            t_in_cam = pose_data.get('kf_translation_vector', [0, 0, 0])
            position_text = (f"Leader in Cam (KF): "
                            f"x={t_in_cam[0]:.3f}, y={t_in_cam[1]:.3f}, z={t_in_cam[2]:.3f}")
            cv2.putText(out, position_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2, cv2.LINE_AA)
            
            return out
        
        inlier_mkpts0 = mkpts0[inlier_idx]
        inlier_mkpts1 = mkpts1[inlier_idx]
        inlier_conf = mconf[inlier_idx]
        color = cm.jet(inlier_conf)

        # Create matching visualization
        out = make_matching_plot_fast(
            anchor_image_gray,
            frame_gray,
            self.anchor_keypoints_sp,
            frame_keypoints,
            inlier_mkpts0,
            inlier_mkpts1,
            color,
            text=[],
            path=None,
            show_keypoints=self.opt.show_keypoints,
            small_text=[]
        )

        # Show the object position and info
        t_in_cam = pose_data['object_translation_in_cam']
        position_text = (f"Leader in Cam: "
                         f"x={t_in_cam[0]:.3f}, y={t_in_cam[1]:.3f}, z={t_in_cam[2]:.3f}")
        cv2.putText(out, position_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Add mode and quality info
        quality_text = (f"Mode: Matching | "
                       f"Inliers: {pose_data['num_inliers']}/{pose_data['total_matches']} | "
                       f"Coverage: {pose_data['coverage_score']:.2f}")
        cv2.putText(out, quality_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        return out

    def _visualize_tracks(self, frame, inliers, mpts3D, pose_data, frame_keypoints):
        """
        Visualize feature tracks over time.
        Used when in tracking mode.
        """
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Get tracks
        tracks = self.point_tracker.get_tracks(min_length=2)
        
        # Draw the tracked points
        if tracks.shape[0] > 0:
            # Get the most recent points from tracks
            offsets = self.point_tracker.get_offsets()
            for track in tracks:
                # Get the most recent valid point in the track
                last_valid_idx = -1
                for i in range(track.shape[0] - 1, 1, -1):
                    if track[i] != -1:
                        last_valid_idx = i
                        break
                
                if last_valid_idx != -1:
                    point_id = int(track[last_valid_idx])
                    # Find which frame this point belongs to
                    for i in range(len(offsets)-1, -1, -1):
                        if point_id >= offsets[i]:
                            frame_idx = i
                            pt_idx = point_id - offsets[i]
                            break
                    
                    if frame_idx < len(self.point_tracker.all_pts):
                        pt = self.point_tracker.all_pts[frame_idx][:, pt_idx]
                        p = (int(round(pt[0])), int(round(pt[1])))
                        
                        # Color based on track score (better tracks are more green)
                        track_score = track[1]
                        normalized_score = 1.0 - min(1.0, track_score / self.point_tracker.nn_thresh)
                        color = (0, int(255 * normalized_score), int(255 * (1 - normalized_score)))
                        
                        # Draw the point
                        cv2.circle(vis_frame, p, 2, color, -1, cv2.LINE_AA)
                        
                        # Draw lines for track history
                        prev_point = None
                        for i in range(last_valid_idx, 1, -1):
                            if track[i] != -1:
                                point_id = int(track[i])
                                # Find frame
                                for j in range(len(offsets)-1, -1, -1):
                                    if point_id >= offsets[j]:
                                        frame_idx = j
                                        pt_idx = point_id - offsets[j]
                                        break
                                
                                if frame_idx < len(self.point_tracker.all_pts):
                                    pt = self.point_tracker.all_pts[frame_idx][:, pt_idx]
                                    current_point = (int(round(pt[0])), int(round(pt[1])))
                                    
                                    if prev_point is not None:
                                        cv2.line(vis_frame, current_point, prev_point, color, 1, cv2.LINE_AA)
                                    
                                    prev_point = current_point
        
        # Show pose information
        t_in_cam = pose_data['object_translation_in_cam']
        position_text = (f"Leader in Cam: "
                        f"x={t_in_cam[0]:.3f}, y={t_in_cam[1]:.3f}, z={t_in_cam[2]:.3f}")
        cv2.putText(vis_frame, position_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Add mode and quality info
        quality_text = (f"Mode: Tracking | "
                       f"Tracks: {tracks.shape[0]} | "
                       f"Inliers: {pose_data['num_inliers']} | "
                       f"Coverage: {pose_data['coverage_score']:.2f}")
        cv2.putText(vis_frame, quality_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        return vis_frame

    def _get_camera_intrinsics(self):
        """Get camera intrinsic parameters."""
        # Camera calibration parameters
        focal_length_x = 1430.10150
        focal_length_y = 1430.48915
        cx = 640.85462
        cy = 480.64800

        distCoeffs = np.array([0.3393, 2.0351, 0.0295, -0.0029, -10.9093], dtype=np.float32)

        K = np.array([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K, distCoeffs