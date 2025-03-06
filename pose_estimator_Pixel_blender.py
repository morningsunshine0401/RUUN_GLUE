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
    #compare_with_ground_truth
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
        anchor_keypoints_2D = np.array([[ 860.,  388.],
                            [ 467.,  394.],
                            [ 881.,  414.],
                            [ 466.,  421.],
                            [ 668.,  421.],
                            [ 591.,  423.],
                            [1078.,  481.],
                            [ 195.,  494.],
                            [ 183.,  540.],
                            [ 626.,  592.],
                            [ 723.,  592.]], dtype=np.float32)

        anchor_keypoints_3D = np.array([[-0.60385, -0.3258 ,  5.28664],
                            [-0.60385,  0.33329,  5.28664],
                            [-0.81972, -0.3258 ,  5.28664],
                            [-0.81972,  0.33329,  5.28664],
                            [ 0.26951, -0.07631,  5.0646 ],
                            [ 0.26951,  0.0838 ,  5.0646 ],
                            [-0.29297, -0.83895,  4.96825],
                            [-0.04106,  0.84644,  4.995  ],
                            [-0.29297,  0.84644,  4.96825],
                            [-0.81973,  0.0838 ,  4.99302],
                            [-0.81973, -0.07631,  4.99302]], dtype=np.float32)

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
    
    def compare_with_ground_truth(self, estimated_R, estimated_t, gt_pose):
        """
        Compare the estimated pose with ground truth
        
        Args:
            estimated_R: Estimated rotation matrix (3x3)
            estimated_t: Estimated translation vector (3,)
            gt_pose: Ground truth pose matrix (4x4)
            
        Returns:
            rotation_error (radians), translation_error (same units as translation)
        """
        # Extract ground truth rotation and translation
        gt_R = gt_pose[:3, :3]
        gt_t = gt_pose[:3, 3]
        
        # Compute rotation error
        R_diff = estimated_R @ gt_R.T
        rotation_error = self.rotation_matrix_to_axis_angle(R_diff)
        
        # Compute translation error
        translation_error = np.linalg.norm(estimated_t - gt_t)
        
        return rotation_error, translation_error

    def rotation_matrix_to_axis_angle(self, R):
        """
        Convert a rotation matrix to axis-angle representation
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Angle in radians
        """
        # Compute the angle from the trace
        theta = np.arccos((np.trace(R) - 1) / 2)
        return theta


    def process_frame(self, frame, frame_idx, frame_name=None):
        """
        Process a single frame and estimate pose, comparing with ground truth if available
        
        Args:
            frame: Input frame/image
            frame_idx: Frame index
            frame_name: Name of the frame/image file (for ground truth comparison)
        
        Returns:
            pose_data: Dictionary with pose estimation results
            visualization: Visualization of feature matches and pose
        """
        logger.info(f"Processing frame {frame_idx}")
        start_time = time.time()

        # Resize frame to target size
        frame = self._resize_image(frame, self.opt.resize)
        
        # Get feature matches between anchor and current frame
        feature_data = self._extract_features(frame)
        
        # Initialize pose_data with basic information
        pose_data = {
            'frame': frame_idx,
            'image_file': frame_name,
            'num_feature_matches': 0,
            'pose_estimation_failed': True
        }
        
        if feature_data is None or len(feature_data['mkpts1']) < 4:
            logger.warning(f"Not enough matches for pose estimation (frame {frame_idx})")
            
            # If we've initialized the filter, we can still predict
            if self.kf_initialized:
                x_pred, P_pred = self.kf_pose.predict()
                
                # Extract pose from predicted state
                translation = x_pred[0:3]
                q = x_pred[6:10]
                R = quaternion_to_rotation_matrix(q)
                
                # Update pose_data with predicted pose
                pose_data.update({
                    'kf_translation_vector': translation.tolist(),
                    'kf_quaternion': q.tolist(),
                    'kf_rotation_matrix': R.tolist(),
                })
                
                # Compare with ground truth if available
                if hasattr(self, 'gt_poses') and self.gt_poses and frame_name in self.gt_poses:
                    gt_pose = self.gt_poses[frame_name]
                    rotation_error, translation_error = self.compare_with_ground_truth(R, translation, gt_pose)
                    pose_data.update({
                        'rotation_error_rad': float(rotation_error),
                        'translation_error': float(translation_error)
                    })
                
                return pose_data, frame
            else:
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
                
                pose_data.update({
                    'num_feature_matches': len(valid_indices),
                    'kf_translation_vector': translation.tolist(),
                    'kf_quaternion': q.tolist(),
                    'kf_rotation_matrix': R.tolist(),
                })
                
                # Compare with ground truth if available
                if hasattr(self, 'gt_poses') and self.gt_poses and frame_name in self.gt_poses:
                    gt_pose = self.gt_poses[frame_name]
                    rotation_error, translation_error = self.compare_with_ground_truth(R, translation, gt_pose)
                    pose_data.update({
                        'rotation_error_rad': float(rotation_error),
                        'translation_error': float(translation_error)
                    })
                
                return pose_data, frame
            else:
                pose_data['num_feature_matches'] = len(valid_indices)
                return pose_data, frame
        
        # Extract valid matches
        valid_mkpts1 = mkpts1[valid_indices]
        valid_anchor_indices = anchor_indices[valid_indices]
        valid_mconf = mconf[valid_indices]
        
        # Convert anchor indices to 3D model indices
        feature_indices = np.array([self.matched_anchor_indices.tolist().index(idx) 
                                    for idx in valid_anchor_indices])
        
        # Apply RANSAC to filter outliers
        if len(valid_mkpts1) > 6:#10:  # Only use RANSAC if we have enough points
            # Get corresponding 3D points
            points_3d = self.matched_3D_keypoints[feature_indices]
            
            # Use PnP for initial pose estimation and outlier removal
            try:
                K, distCoeffs = self._get_camera_intrinsics()
                
                # Convert points to the right format
                objectPoints = points_3d.reshape(-1, 1, 3).astype(np.float32)
                imagePoints = valid_mkpts1.reshape(-1, 1, 2).astype(np.float32)
                
                # Run PnP RANSAC
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=objectPoints,
                    imagePoints=imagePoints,
                    cameraMatrix=K,
                    distCoeffs=distCoeffs
                )
                
                if success and inliers is not None and len(inliers) >= 4:
                    logger.info(f"RANSAC filtered to {len(inliers)} inliers out of {len(valid_mkpts1)}")
                    
                    # Extract inliers
                    inlier_indices = inliers.flatten()
                    valid_mkpts1 = valid_mkpts1[inlier_indices]
                    feature_indices = feature_indices[inlier_indices]
                    valid_mconf = valid_mconf[inlier_indices] if valid_mconf is not None else None
                    
                    # Also store the PnP pose for comparison
                    R_pnp, _ = cv2.Rodrigues(rvec)
                    t_pnp = tvec.flatten()
                    
                    # Add PnP data to pose_data
                    pose_data.update({
                        'pnp_rotation_matrix': R_pnp.tolist(),
                        'pnp_translation_vector': t_pnp.tolist(),
                        'num_inliers': len(inliers)
                    })
            except Exception as e:
                logger.warning(f"PnP RANSAC failed: {str(e)}")
        
        # Predict state (if filter is already initialized)
        if self.kf_initialized:
            x_pred, P_pred = self.kf_pose.predict()
        
        # Update Kalman filter with feature measurements
        update_success = False
        if len(valid_mkpts1) >= 6:#4:
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
        
        # Update pose_data with Kalman filter results
        pose_data.update({
            'num_feature_matches': len(valid_mkpts1),
            'feature_indices': feature_indices.tolist(),
            'kf_translation_vector': translation.tolist(),
            'kf_quaternion': q.tolist(),
            'kf_rotation_matrix': R.tolist(),
            'kf_update_success': update_success,
            'kf_initialized': self.kf_initialized,
            'processing_time': time.time() - start_time,
            'pose_estimation_failed': False
        })
        
        # Compare with ground truth if available
        if hasattr(self, 'gt_poses') and self.gt_poses and frame_name in self.gt_poses:
            gt_pose = self.gt_poses[frame_name]
            rotation_error, translation_error = self.compare_with_ground_truth(R, translation, gt_pose)
            pose_data.update({
                'rotation_error_rad': float(rotation_error),
                'translation_error': float(translation_error)
            })
            
            # If we have PnP results, also compare those
            if 'pnp_rotation_matrix' in pose_data:
                R_pnp = np.array(pose_data['pnp_rotation_matrix'])
                t_pnp = np.array(pose_data['pnp_translation_vector'])
                pnp_rotation_error, pnp_translation_error = self.compare_with_ground_truth(R_pnp, t_pnp, gt_pose)
                pose_data.update({
                    'pnp_rotation_error_rad': float(pnp_rotation_error),
                    'pnp_translation_error': float(pnp_translation_error)
                })
        
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
    

    # Add these functions to your pose_estimator_feature.py file

    def load_ground_truth_poses(self, json_path):
        """
        Load ground truth poses from a JSON file.
        
        Args:
            json_path: Path to the JSON file containing ground truth poses
            
        Returns:
            Dictionary mapping frame/image names to pose matrices
        """
        import json
        
        logger.info(f"Loading ground truth poses from {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        gt_poses = {}
        
        # Handle the case where we have a "frames" key (like in your Blender simulation)
        if 'frames' in data:
            # Transformation matrix from Blender to OpenCV coordinate system (if needed)
            T_blender_to_opencv = np.array([
                # [1,  0,  0],
                # [0,  0,  1],
                # [0, -1,  0]
                [1,  1,  1],
                [1,  1,  1],
                [1, 1,  1]

            ], dtype=np.float32)
            
            for frame in data['frames']:
                image_name = frame['image']
                for obj in frame['object_poses']:
                    if obj['name'] == 'Camera':
                        pose = np.array(obj['pose'], dtype=np.float32)
                        # Extract rotation and translation
                        R_blender = pose[:3, :3]
                        t_blender = pose[:3, 3]
                        # Transform rotation and translation
                        R_opencv = T_blender_to_opencv @ R_blender
                        t_opencv = T_blender_to_opencv @ t_blender
                        # Reconstruct pose matrix
                        pose_opencv = np.eye(4, dtype=np.float32)
                        pose_opencv[:3, :3] = R_opencv
                        pose_opencv[:3, 3] = t_opencv
                        gt_poses[image_name] = pose_opencv
                        break  # Assume only one camera pose per frame
        
        # Handle the case where we have a list of pose data (like in your existing code)
        elif isinstance(data, list):
            for entry in data:
                if 'image_file' in entry:
                    image_name = entry['image_file']
                    if 'rotation_matrix' in entry and 'translation_vector' in entry:
                        R = np.array(entry['rotation_matrix'], dtype=np.float32)
                        t = np.array(entry['translation_vector'], dtype=np.float32)
                        pose = np.eye(4, dtype=np.float32)
                        pose[:3, :3] = R
                        pose[:3, 3] = t
                        gt_poses[image_name] = pose
        
        # Handle CSV file (similar to how you read image_index.csv)
        elif json_path.endswith('.csv'):
            import csv
            with open(json_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'Filename' in row and 'Rx' in row and 'Ry' in row and 'Rz' in row and 'Tx' in row and 'Ty' in row and 'Tz' in row:
                        image_name = row['Filename']
                        # Convert rotation (Euler angles) to rotation matrix
                        rx, ry, rz = float(row['Rx']), float(row['Ry']), float(row['Rz'])
                        Rx = np.array([
                            [1, 0, 0],
                            [0, np.cos(rx), -np.sin(rx)],
                            [0, np.sin(rx), np.cos(rx)]
                        ])
                        Ry = np.array([
                            [np.cos(ry), 0, np.sin(ry)],
                            [0, 1, 0],
                            [-np.sin(ry), 0, np.cos(ry)]
                        ])
                        Rz = np.array([
                            [np.cos(rz), -np.sin(rz), 0],
                            [np.sin(rz), np.cos(rz), 0],
                            [0, 0, 1]
                        ])
                        R = Rz @ Ry @ Rx
                        t = np.array([float(row['Tx']), float(row['Ty']), float(row['Tz'])])
                        
                        pose = np.eye(4, dtype=np.float32)
                        pose[:3, :3] = R
                        pose[:3, 3] = t
                        gt_poses[image_name] = pose
        
        logger.info(f"Loaded {len(gt_poses)} ground truth poses")
        return gt_poses

    
    def _get_camera_intrinsics(self):
        # # Camera calibration parameters
        # focal_length_x = 14301.0150
        # focal_length_y = 14304.8915
        # #focal_length_x = 15001.0150
        # #focal_length_y = 15004.8915
        # cx = 640.85462
        # cy = 480.64800

        # Camera intrinsic parameters (replace with your camera's parameters)
        focal_length_x = 266666.66666666666  # px
        focal_length_y = 266666.66666666666  # py
        #focal_length_x = 133300.3333333333*2  # px
        #focal_length_y = 133300.3333333333*2  # py
        cx = 639.5  # Principal point u0
        cy = 479.5  # Principal point v0

        #distCoeffs = np.array([0.3393, 2.0351, 0.0295, -0.0029, -10.9093], dtype=np.float32)
        distCoeffs = None
        
        K = np.array([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K, distCoeffs