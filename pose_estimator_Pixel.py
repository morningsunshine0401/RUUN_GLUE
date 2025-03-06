import cv2
import torch
torch.set_grad_enabled(False)
import time
import numpy as np
from scipy.spatial import cKDTree
import onnxruntime as ort
from utils import (
    frame2tensor,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    normalize_quaternion
)
from KF_Pixel import KalmanFilterFeatureBased
import matplotlib.cm as cm
from models.utils import make_matching_plot_fast
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pose_estimator_feature.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the SuperPointPreprocessor
from superpoint_LG import SuperPointPreprocessor

class PoseEstimator:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.initial_z_set = False  # Flag for first-frame Z override (if desired)
        self.kf_initialized = False  # To track if Kalman filter was ever updated

        logger.info("Initializing PoseEstimator with Feature-Based Kalman Filter")

        # Load anchor (leader) image
        self.anchor_image = cv2.imread(opt.anchor)
        assert self.anchor_image is not None, f'Failed to load anchor image at {opt.anchor}'
        self.anchor_image = self._resize_image(self.anchor_image, opt.resize)
        logger.info(f"Loaded and resized anchor image from {opt.anchor}")

        # Initialize ONNX session for LightGlue
        # (Try CUDA first, then CPU)
        providers = [("CPUExecutionProvider", {})]
        providers.insert(0, ("CUDAExecutionProvider", {}))
        self.session = ort.InferenceSession(
            #"weights/superpoint_lightglue_pipeline_1280x720_multihead.onnx",
            "weights/superpoint_lightglue_pipeline.onnx",
            providers=providers
        )
        logger.info("ONNX session initialized")

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
            [ 0.217,  0.000,  0.070],
            [ 0.230,  0.000, -0.070],
            [-0.014,  0.000, -0.156],
            [ 0.000,  0.000,  0.042],
            [-0.057, -0.018, -0.010],
            [-0.074, -0.000,  0.128],
            [ 0.206, -0.070, -0.002],
            [-0.000, -0.000,  0.156],
            [-0.017, -0.000, -0.092],
            [ 0.217, -0.000, -0.027],
            [-0.052, -0.000, -0.097],
            [-0.019, -0.000,  0.128],
            [-0.035, -0.018, -0.010],
            [ 0.217, -0.000, -0.070],
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
            [ 0.230, -0.000, -0.113],
            [-0.100, -0.030,  0.000],
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

        # Initialize Kalman filter
        K, distCoeffs = self._get_camera_intrinsics()
        frame_rate = 30  # Adjust this to your actual frame rate
        dt = 1.0 / frame_rate
        self.kf_pose = KalmanFilterFeatureBased(dt, K, distCoeffs)
        
        # Set 3D model points in the filter
        self.kf_pose.set_model_points(anchor_keypoints_3D)
        
        # Flag for initialization
        self.kf_pose_first_update = True
        logger.info("Feature-based Kalman filter initialized")

    def reinitialize_anchor(self, new_anchor_path, new_2d_points, new_3d_points):
        """
        Re-load a new anchor image and re-compute relevant data (2D->3D correspondences).
        Called on-the-fly (e.g. after 200 frames).
        """
        logger.info(f"Re-initializing anchor with new image: {new_anchor_path}")

        # 1. Load new anchor image
        new_anchor_image = cv2.imread(new_anchor_path)
        assert new_anchor_image is not None, f"Failed to load new anchor image at {new_anchor_path}"
        new_anchor_image = self._resize_image(new_anchor_image, self.opt.resize)

        # 2. Update anchor image
        self.anchor_image = new_anchor_image

        # 3. Recompute anchor features with the new image and 2D/3D
        self._set_anchor_features(
            anchor_bgr_image=new_anchor_image,
            anchor_keypoints_2D=new_2d_points,
            anchor_keypoints_3D=new_3d_points
        )

        # 4. Update 3D model points in the Kalman filter
        self.kf_pose.set_model_points(new_3d_points)

        logger.info("Anchor re-initialization complete.")

    def _set_anchor_features(self, anchor_bgr_image, anchor_keypoints_2D, anchor_keypoints_3D):
        """
        Run SuperPoint on the anchor image to get anchor_keypoints_sp.
        Then match those keypoints to known 2D->3D correspondences via KDTree.
        """
        # Precompute anchor's SuperPoint descriptors
        self.anchor_proc = SuperPointPreprocessor.preprocess(anchor_bgr_image)
        self.anchor_proc = self.anchor_proc[None].astype(np.float32)

        # Dummy forward pass to get anchor keypoints from LightGlue
        anchor_batch = np.concatenate([self.anchor_proc, self.anchor_proc], axis=0)
        keypoints, matches, mscores = self.session.run(None, {"images": anchor_batch})
        self.anchor_keypoints_sp = keypoints[0]  # (N, 2)

        # Build KDTree to match anchor_keypoints_sp -> known anchor_keypoints_2D
        sp_tree = cKDTree(self.anchor_keypoints_sp)
        distances, indices = sp_tree.query(anchor_keypoints_2D, k=1)
        valid_matches = distances < 1.0  # Threshold for "close enough"

        self.matched_anchor_indices = indices[valid_matches]
        self.matched_3D_keypoints = anchor_keypoints_3D[valid_matches]

        logger.info(f"Matched {len(self.matched_anchor_indices)} keypoints to 3D points")

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
        logger.info(f"Processing frame {frame_idx}")
        start_time = time.time()

        # Resize frame to target size
        frame = self._resize_image(frame, self.opt.resize)
        
        # Get feature matches between anchor and current frame
        feature_data = self._extract_features(frame)
        
        if feature_data is None or len(feature_data['mkpts1']) < 4:
            logger.warning(f"Not enough matches for pose estimation (frame {frame_idx})")
            
            # If we've initialized the filter, we can still predict
            if self.kf_initialized:
                x_pred, P_pred = self.kf_pose.predict()
                
                # Build a basic pose data structure for consistency
                translation = x_pred[0:3]
                q = x_pred[6:10]
                R = quaternion_to_rotation_matrix(q)
                
                pose_data = {
                    'frame': frame_idx,
                    'num_feature_matches': 0,
                    'kf_translation_vector': translation.tolist(),
                    'kf_quaternion': q.tolist(),
                    'kf_rotation_matrix': R.tolist(),
                    'pose_estimation_failed': True
                }
                
                return pose_data, frame
            else:
                # If filter isn't initialized, we have no pose estimate
                pose_data = {
                    'frame': frame_idx,
                    'num_feature_matches': 0,
                    'pose_estimation_failed': True
                }
                return pose_data, frame
        
        # Extract matched 2D points and their corresponding indices
        mkpts1 = feature_data['mkpts1']        # 2D points in current frame
        anchor_indices = feature_data['anchor_indices']  # Indices in anchor feature set
        mconf = feature_data['mconf']          # Match confidence scores
        
        # Get indices in our 3D keypoint set
        # This maps from anchor feature indices to indices in the 3D keypoint array
        idx_map = {idx: i for i, idx in enumerate(self.matched_anchor_indices)}
        
        # Find which anchor indices have corresponding 3D points
        valid_indices = [i for i, aidx in enumerate(anchor_indices) 
                         if aidx in idx_map]
        
        if len(valid_indices) < 4:
            logger.warning(f"Not enough valid 3D-2D correspondences (frame {frame_idx})")
            if self.kf_initialized:
                x_pred, P_pred = self.kf_pose.predict()
                translation = x_pred[0:3]
                q = x_pred[6:10]
                R = quaternion_to_rotation_matrix(q)
                
                pose_data = {
                    'frame': frame_idx,
                    'num_feature_matches': len(valid_indices),
                    'kf_translation_vector': translation.tolist(),
                    'kf_quaternion': q.tolist(),
                    'kf_rotation_matrix': R.tolist(),
                    'pose_estimation_failed': True
                }
                return pose_data, frame
            else:
                pose_data = {
                    'frame': frame_idx,
                    'num_feature_matches': len(valid_indices),
                    'pose_estimation_failed': True
                }
                return pose_data, frame
        
        # Extract valid matches
        valid_mkpts1 = mkpts1[valid_indices]
        valid_anchor_indices = anchor_indices[valid_indices]
        valid_mconf = mconf[valid_indices]
        
        # Convert anchor indices to 3D model indices
        feature_indices = np.array([self.matched_anchor_indices.tolist().index(idx) 
                                    for idx in valid_anchor_indices])
        
        # Apply RANSAC to filter outliers if needed
        if len(valid_mkpts1) > 6:#10:  # Only use RANSAC if we have enough points
            # In a feature-based filter, we could pre-filter outliers with RANSAC
            # but strictly speaking it's not necessary since the filter will handle them
            # Try to get a quick PnP solution to find outliers
            try:
                K, distCoeffs = self._get_camera_intrinsics()
                points_3d = self.matched_3D_keypoints[feature_indices].reshape(-1, 3)
                points_2d = valid_mkpts1.reshape(-1, 2).astype(np.float32)
                
                success, rvec_o, tvec_o, inliers = cv2.solvePnPRansac(
                    objectPoints=points_3d,
                    imagePoints=points_2d,
                    cameraMatrix=K,
                    distCoeffs=distCoeffs,
                    reprojectionError=5.0,
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_EPNP
                )

                objectPoints_inliers = points_3d[inliers.flatten()]
                imagePoints_inliers = points_2d[inliers.flatten()]

                rvec, tvec = cv2.solvePnPRefineVVS(
                objectPoints=objectPoints_inliers,
                imagePoints=imagePoints_inliers,
                cameraMatrix=K,
                distCoeffs=distCoeffs,
                rvec=rvec_o,
                tvec=tvec_o
                )
                
                if success and inliers is not None and len(inliers) >= 4:
                    # Extract inliers only
                    inlier_indices = inliers.flatten()
                    valid_mkpts1 = valid_mkpts1[inlier_indices]
                    feature_indices = feature_indices[inlier_indices]
                    valid_mconf = valid_mconf[inlier_indices] if valid_mconf is not None else None
                    print("PnP is done $##$##$#$#$#$#$$$$$$$$$$$$$$$$$$$$$$$\n")
                    
                    logger.info(f"RANSAC filtered to {len(inlier_indices)} inliers out of {len(valid_mkpts1)}")
            except Exception as e:
                logger.warning(f"RANSAC filtering failed: {str(e)}")
                # Continue with all matches
        
        # Predict state (if filter is already initialized)
        if self.kf_initialized:
            x_pred, P_pred = self.kf_pose.predict()
        
        # Update Kalman filter with feature measurements
        update_success = False
        if len(valid_mkpts1) >= 6:
            try:
                # Check 3D model on first frame
                if frame_idx == 1:
                    self.kf_pose.check_3d_model()
                
                # Get corresponding 3D points for these features
                points_3d = self.matched_3D_keypoints[feature_indices]
                    
                # ADDED: Compare with direct PnP solution
                pnp_R, pnp_t, pnp_error = self.kf_pose.compare_with_pnp(
                    valid_mkpts1, points_3d)
                    
                # Update with feature measurements
                x_upd, P_upd = self.kf_pose.update(
                    points_2d=valid_mkpts1,
                    feature_indices=feature_indices,
                    feature_confidences=valid_mconf if valid_mconf is not None else None
                )
                update_success = True
                self.kf_initialized = True
                
                # ADDED: Create debug visualization
                debug_img, mean_error = self.kf_pose.debug_projection(
                    frame, valid_mkpts1, points_3d,
                    save_path=f"debug_projections/frame_{frame_idx:04d}.png")
                    
                # Get reprojection errors for diagnostics
                errors, projected_points = self.kf_pose.get_reprojection_errors(
                    valid_mkpts1, feature_indices)
                
                mean_error = np.mean(errors) if errors is not None else None
                logger.info(f"Kalman update successful, mean reprojection error: {mean_error:.2f} px")
                
                # ADDED: If direct PnP gave better results, optionally use its values
                if pnp_error is not None and (mean_error is None or pnp_error < mean_error):
                    logger.info(f"Direct PnP has better results ({pnp_error:.2f} px vs {mean_error:.2f} px)")
                    # Optionally override filter state with PnP result
                    # self.kf_pose.x[0:3] = pnp_t
                    # q_pnp = rotation_matrix_to_quaternion(pnp_R)
                    # self.kf_pose.x[6:10] = q_pnp
                    
            except Exception as e:
                logger.error(f"Kalman update failed: {str(e)}")
                update_success = False
        
        # Get final state from filter
        x_final = self.kf_pose.x  # Current state estimate
        
        # Extract pose components
        translation = x_final[0:3]
        q = x_final[6:10]
        R = quaternion_to_rotation_matrix(q)
        
        # Create visualization
        visualization = self._visualize_features(
            frame, valid_mkpts1, feature_indices, valid_mconf, x_final, feature_data["frame_keypoints"])
        
        # Create result data
        pose_data = {
            'frame': frame_idx,
            'num_feature_matches': len(valid_mkpts1),
            'feature_indices': feature_indices.tolist(),
            'kf_translation_vector': translation.tolist(),
            'kf_quaternion': q.tolist(),
            'kf_rotation_matrix': R.tolist(),
            'kf_update_success': update_success,
            'kf_initialized': self.kf_initialized,
            'processing_time': time.time() - start_time
        }
        
        # If we also calculated a PnP solution earlier, add it for comparison
        if 'rvec' in locals() and 'tvec' in locals() and success:
            R_pnp, _ = cv2.Rodrigues(rvec)
            q_pnp = rotation_matrix_to_quaternion(R_pnp)
            pose_data['pnp_translation_vector'] = tvec.flatten().tolist()
            pose_data['pnp_rotation_matrix'] = R_pnp.tolist()
            pose_data['pnp_quaternion'] = q_pnp.tolist()
            pose_data['pnp_num_inliers'] = len(inliers) if inliers is not None else 0
        
        return pose_data, visualization
    

    

    def _extract_features(self, frame):
        """Extract features from current frame and match with anchor"""
        try:
            # Preprocess image
            frame_proc = SuperPointPreprocessor.preprocess(frame)
            frame_proc = frame_proc[None].astype(np.float32)
            
            # Batch processing with anchor
            batch = np.concatenate([self.anchor_proc, frame_proc], axis=0)
            
            # Run feature detection and matching
            keypoints, matches, mscores = self.session.run(None, {"images": batch})
            
            # Filter valid matches (anchor -> frame)
            valid_mask = (matches[:, 0] == 0)
            valid_matches = matches[valid_mask]
            
            mkpts0 = keypoints[0][valid_matches[:, 1]]  # Anchor keypoints
            mkpts1 = keypoints[1][valid_matches[:, 2]]  # Frame keypoints
            mconf = mscores[valid_mask]                # Match confidence
            anchor_indices = valid_matches[:, 1]       # Indices in anchor keypoints
            
            return {
                'mkpts0': mkpts0,
                'mkpts1': mkpts1,
                'mconf': mconf,
                'anchor_indices': anchor_indices,
                'frame_keypoints': keypoints[1]  # All keypoints in frame (for visualization)
            }
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return None

    def _visualize_features(self, frame, valid_mkpts1, feature_indices, valid_mconf, state_vector, frame_keypoints):
        """Visualize feature matches and estimated pose"""
        anchor_image_gray = cv2.cvtColor(self.anchor_image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get corresponding anchor points
        feature_3d_indices = feature_indices
        valid_mkpts0 = self.anchor_keypoints_sp[self.matched_anchor_indices[feature_3d_indices]]
        
        # Use confidence for coloring, or default to ones
        color = cm.jet(valid_mconf) if valid_mconf is not None else cm.jet(np.ones(len(valid_mkpts1)))
        
        # Get pose data for text overlay
        translation = state_vector[0:3]
        q = state_vector[6:10]
        
        # Generate visualization
        text = [
            f"Translation: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]",
            f"Quaternion: [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]",
            f"Points: {len(valid_mkpts1)}"
        ]
        
        out = make_matching_plot_fast(
            anchor_image_gray,
            frame_gray,
            self.anchor_keypoints_sp,
            frame_keypoints,
            valid_mkpts0,
            valid_mkpts1,
            color,
            text=text,
            path=None,
            show_keypoints=self.opt.show_keypoints,
            small_text=[]
        )
        
        # Draw reprojection visualization
        try:
            # Project 3D points using current state
            points_3d = self.matched_3D_keypoints[feature_indices]
            errors, projected_points = self.kf_pose.get_reprojection_errors(valid_mkpts1, feature_indices)
            
            if projected_points is not None:
                # Draw the projections and errors
                for i, (pt_obs, pt_proj) in enumerate(zip(valid_mkpts1, projected_points)):
                    # Draw observed point (green)
                    cv2.circle(out, (int(pt_obs[0]), int(pt_obs[1])), 3, (0, 255, 0), -1)
                    
                    # Draw projected point (red)
                    cv2.circle(out, (int(pt_proj[0]), int(pt_proj[1])), 3, (0, 0, 255), -1)
                    
                    # Draw line between them
                    cv2.line(out, 
                             (int(pt_obs[0]), int(pt_obs[1])), 
                             (int(pt_proj[0]), int(pt_proj[1])), 
                             (255, 0, 255), 1)
            
            # Add average reprojection error text
            if errors is not None:
                mean_error = np.mean(errors)
                cv2.putText(out, f"Mean Reprojection Error: {mean_error:.2f} px", 
                           (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        except Exception as e:
            logger.warning(f"Visualization error: {str(e)}")
        
        return out

    def _get_camera_intrinsics(self):
        # Camera calibration parameters
        focal_length_x = 14301.0150
        focal_length_y = 14304.8915
        #focal_length_x = 15001.0150
        #focal_length_y = 15004.8915
        cx = 640.85462
        cy = 480.64800

        # # Camera intrinsic parameters (replace with your camera's parameters)
        # focal_length_x = 2666.66666666666  # px
        # focal_length_y = 2666.66666666666  # py
        # cx = 639.5  # Principal point u0
        # cy = 479.5  # Principal point v0

        #distCoeffs = np.array([0.3393, 2.0351, 0.0295, -0.0029, -10.9093], dtype=np.float32)
        distCoeffs = None
        
        K = np.array([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K, distCoeffs