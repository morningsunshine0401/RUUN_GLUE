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

# Import LightGlue and SuperPoint
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

from KF_Q import KalmanFilterPose
import matplotlib.cm as cm
from models.utils import make_matching_plot_fast
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pose_estimator.log"),  # Logs will be saved in this file
        logging.StreamHandler()  # Logs will also be printed to console
    ]
)
logger = logging.getLogger(__name__)

class PoseEstimator:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.initial_z_set = False  # Flag for first-frame Z override (if desired)
        self.kf_initialized = False  # To track if Kalman filter was ever updated

        logger.info("Initializing PoseEstimator with separate SuperPoint and LightGlue models")

        # Load anchor (leader) image
        self.anchor_image = cv2.imread(opt.anchor)
        assert self.anchor_image is not None, f'Failed to load anchor image at {opt.anchor}'
        self.anchor_image = self._resize_image(self.anchor_image, opt.resize)
        logger.info(f"Loaded and resized anchor image from {opt.anchor}")

        # Initialize SuperPoint and LightGlue models
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)
        logger.info("Initialized SuperPoint and LightGlue models")

        # Define anchor keypoints (2D -> 3D correspondence)
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
        self.kf_pose = self._init_kalman_filter()
        self.kf_pose_first_update = True 
        logger.info("Kalman filter initialized")

    def _convert_cv2_to_tensor(self, image):
        """Convert OpenCV BGR image to RGB tensor"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to float and normalize to [0, 1]
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        # Permute dimensions from (H, W, C) to (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)

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

            # 3. Update anchor image
            self.anchor_image = new_anchor_image
            logger.info("Anchor image updated")

            # 4. Recompute anchor features with the new image and 2D/3D
            logger.info(f"Setting anchor features with {len(new_2d_points)} 2D points and {len(new_3d_points)} 3D points")
            self._set_anchor_features(
                anchor_bgr_image=new_anchor_image,
                anchor_keypoints_2D=new_2d_points,
                anchor_keypoints_3D=new_3d_points
            )

            logger.info("Anchor re-initialization complete.")
        except Exception as e:
            logger.error(f"Error during anchor reinitialization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _set_anchor_features(self, anchor_bgr_image, anchor_keypoints_2D, anchor_keypoints_3D):
        """
        Run SuperPoint on the anchor image to get anchor_keypoints_sp.
        Then match those keypoints to known 2D->3D correspondences via KDTree.
        """
        try:
            # Record the start time
            start_time = time.time()
            logger.info("Starting anchor feature extraction...")
            
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
                return
            
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
                return
            
            self.matched_anchor_indices = indices[valid_matches]
            self.matched_3D_keypoints = anchor_keypoints_3D[valid_matches]
            
            logger.info(f"Matched {len(self.matched_anchor_indices)} keypoints to 3D points")
            logger.info(f"Anchor feature extraction completed in {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error during anchor feature extraction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

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

        # Ensure gradients are disabled
        with torch.no_grad():
            # Resize frame to target resolution
            frame = self._resize_image(frame, self.opt.resize)
            resize_time = time.time()

            # Extract features from the frame
            frame_tensor = self._convert_cv2_to_tensor(frame)
            frame_feats = self.extractor.extract(frame_tensor)
            extract_time = time.time()

            # Match features between anchor and frame
            matches_dict = self.matcher({
                'image0': self.anchor_feats, 
                'image1': frame_feats
            })
            match_time = time.time()

        logger.info(
            f"Frame {frame_idx} times: Resize={resize_time - start_time:.3f}s, "
            f"Extract={extract_time - resize_time:.3f}s, "
            f"Match={match_time - extract_time:.3f}s, "
            f"Total={time.time() - start_time:.3f}s"
        )

        # Remove batch dimension and move to CPU
        feats0, feats1, matches01 = [rbd(x) for x in [self.anchor_feats, frame_feats, matches_dict]]
        
        # Get keypoints and matches - detach tensors before converting to numpy
        kpts0 = feats0["keypoints"].detach().cpu().numpy()
        kpts1 = feats1["keypoints"].detach().cpu().numpy()
        matches = matches01["matches"].detach().cpu().numpy()
        confidence = matches01.get("scores", torch.ones(len(matches))).detach().cpu().numpy()
        
        mkpts0 = kpts0[matches[:, 0]]
        mkpts1 = kpts1[matches[:, 1]]
        mconf = confidence

        logger.debug(f"Found {len(mkpts0)} raw matches in frame {frame_idx}")

        # Filter to known anchor indices
        valid_indices = matches[:, 0]
        known_mask = np.isin(valid_indices, self.matched_anchor_indices)
        
        if not np.any(known_mask):
            logger.warning(f"No valid matches to 3D points found for frame {frame_idx}")
            
            # If there are not enough matches, use Kalman prediction
            x_pred, P_pred = self.kf_pose.predict()
            translation_estimated = x_pred[0:3]        # px, py, pz
            q_estimated = x_pred[6:10]                # qx, qy, qz, qw
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
        filtered_indices = valid_indices[known_mask]
        
        # Get corresponding 3D points
        idx_map = {idx: i for i, idx in enumerate(self.matched_anchor_indices)}
        mpts3D = np.array([
            self.matched_3D_keypoints[idx_map[aidx]] 
            for aidx in filtered_indices if aidx in idx_map
        ])

        if len(mkpts0) < 4:
            logger.warning(f"Not enough matches for pose (frame {frame_idx})")
            
            # If there are not enough matches, use Kalman prediction
            x_pred, P_pred = self.kf_pose.predict()
            translation_estimated = x_pred[0:3]        # px, py, pz
            q_estimated = x_pred[6:10]                # qx, qy, qz, qw
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
            translation_estimated = x_pred[0:3]        # px, py, pz
            q_estimated = x_pred[6:10]                # qx, qy, qz, qw
            R_estimated = quaternion_to_rotation_matrix(q_estimated)
            
            pose_data = {
                'frame': frame_idx,
                'kf_translation_vector': translation_estimated.tolist(),
                'kf_quaternion': q_estimated.tolist(),
                'kf_rotation_matrix': R_estimated.tolist(),
                'pose_estimation_failed': True
            }
        
        return pose_data, visualization

    def _init_kalman_filter(self):
        frame_rate = 30
        dt = 1 / frame_rate
        kf_pose = KalmanFilterPose(dt)
        return kf_pose

    def estimate_pose(self, mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, frame_keypoints):
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
            confidence=0.999,
            iterationsCount=2000,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success or inliers is None or len(inliers) < 6:
            logger.warning("PnP pose estimation failed or not enough inliers.")
            print("PnP pose estimation failed or not enough inliers IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n.")
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

        # Initialize region counters
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
                mean_confidence = 1
                
                # Final coverage score
                coverage_score = normalized_entropy * mean_confidence
                
                # Ensure score is in valid range [0,1]
                coverage_score = np.clip(coverage_score, 0, 1)
                print('Coverage score:', coverage_score)
        else:
            coverage_score = 0

        # Compute reprojection errors
        projected_points, _ = cv2.projectPoints(
            objectPoints_inliers, rvec, tvec, K, distCoeffs
        )
        reprojection_errors = np.linalg.norm(imagePoints_inliers - projected_points, axis=2).flatten()
        mean_reprojection_error = np.mean(reprojection_errors)
        std_reprojection_error = np.std(reprojection_errors)

        # Update pose_data with additional region and coverage details
        pose_data = self._kalman_filter_update(
            R, tvec, reprojection_errors, mean_reprojection_error,
            std_reprojection_error, inliers, mkpts0, mkpts1, mpts3D,
            mconf, frame_idx, rvec_o, rvec, coverage_score=coverage_score
        )
        pose_data["region_distribution"] = regions
        pose_data["coverage_score"] = coverage_score

        visualization = self._visualize_matches(
            frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints
        )
        return pose_data, visualization

    def _kalman_filter_update(
        self, R, tvec, reprojection_errors, mean_reprojection_error,
        std_reprojection_error, inliers, mkpts0, mkpts1, mpts3D,
        mconf, frame_idx, rvec_o, rvec, coverage_score
    ):
        num_inliers = len(inliers)
        inlier_ratio = num_inliers / len(mkpts0) if len(mkpts0) > 0 else 0

        reprojection_error_threshold = 4.0
        max_translation_jump = 2.0
        max_orientation_jump = 20.0  # degrees
        min_inlier = 4
        coverage_threshold = -1

        if coverage_score is None:
            print("Coverage score not found, using default...")

        # 1) Convert measured rotation R -> quaternion
        q_measured = rotation_matrix_to_quaternion(R)

        # (Optional) viewpoint check if you have anchor_viewpoint_quat
        anchor_q = getattr(self, "anchor_viewpoint_quat", None)
        viewpoint_max_diff_deg = 380.0

        def quaternion_angle_degrees(q1, q2):
            q1 = normalize_quaternion(q1)
            q2 = normalize_quaternion(q2)
            dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
            angle = 2.0 * np.degrees(np.arccos(dot))
            if angle > 180.0:
                angle = 360.0 - angle
            return angle

        if anchor_q is not None:
            viewpoint_diff = quaternion_angle_degrees(q_measured, anchor_q)
        else:
            viewpoint_diff = 0.0

        #---------------------------------------------------------------------
        # 2) EKF PREDICT: get prior (x^-)
        #---------------------------------------------------------------------
        x_pred, P_pred = self.kf_pose.predict()  
        # x_pred is shape (13,), e.g. [px,py,pz, vx,vy,vz, qx,qy,qz,qw, wx,wy,wz]

        # Parse predicted orientation for threshold checks if desired
        px_pred, py_pred, pz_pred = x_pred[0:3]
        qx_pred, qy_pred, qz_pred, qw_pred = x_pred[6:10]

        # Convert measured q, predicted q -> orientation_change (deg)
        orientation_change = quaternion_angle_degrees(q_measured, [qx_pred,qy_pred,qz_pred,qw_pred])
        # Also check translation change
        translation_change = np.linalg.norm(tvec.flatten() - x_pred[0:3])

        # 3) Build the measurement vector z = [px,py,pz, qx,qy,qz,qw]
        tvec = tvec.flatten()
        z_meas = np.array([
            tvec[0], tvec[1], tvec[2],
            q_measured[0], q_measured[1], q_measured[2], q_measured[3]
        ], dtype=np.float64)

        # If first update, skip thresholds
        if not hasattr(self, 'kf_pose_first_update') or self.kf_pose_first_update:
            # We always do the update
            x_upd, P_upd = self.kf_pose.update(z_meas)
            self.kf_pose_first_update = False
            logger.debug("EKF first update: skipping threshold checks.")
        else:
            # Normal frames: check thresholds 
            if mean_reprojection_error < reprojection_error_threshold and num_inliers > min_inlier:
                if translation_change < max_translation_jump and orientation_change < max_orientation_jump:
                    if coverage_score >= coverage_threshold:
                        if viewpoint_diff <= viewpoint_max_diff_deg:
                            # pass all checks => do update
                            x_upd, P_upd = self.kf_pose.update(z_meas)
                            logger.debug("EKF update (all thresholds passed).")
                        else:
                            logger.debug(f"Skipping EKF update: viewpoint diff={viewpoint_diff:.1f}>{viewpoint_max_diff_deg}")
                            x_upd, P_upd = x_pred, P_pred
                    else:
                        logger.debug(f"Skipping EKF update: coverage_score={coverage_score:.2f} < {coverage_threshold}")
                        x_upd, P_upd = x_pred, P_pred
                else:
                    logger.debug("Skipping EKF update: large translation/orientation jump.")
                    x_upd, P_upd = x_pred, P_pred
            else:
                logger.debug("Skipping EKF update: high repro error or insufficient inliers.")
                x_upd, P_upd = x_pred, P_pred

        # For simplicity, let's assume x_upd is final
        x_final = x_upd  
        # parse final
        px, py, pz = x_final[0:3]
        qx, qy, qz, qw = x_final[6:10]
        R_estimated = quaternion_to_rotation_matrix([qx, qy, qz, qw])

        # Build final pose_data
        pose_data = {
            'frame': frame_idx,
            'object_rotation_in_cam': R.tolist(),
            'object_translation_in_cam': tvec.flatten().tolist(),
            'raw_rvec': rvec_o.flatten().tolist(),
            'refined_raw_rvec': rvec.flatten().tolist(),
            'num_inliers': num_inliers,
            'total_matches': len(mkpts0),
            'inlier_ratio': inlier_ratio,
            'reprojection_errors': reprojection_errors.tolist(),
            'mean_reprojection_error': float(mean_reprojection_error),
            'std_reprojection_error': float(std_reprojection_error),
            'inliers': inliers.flatten().tolist(),
            'mkpts0': mkpts0.tolist(),
            'mkpts1': mkpts1.tolist(),
            'mpts3D': mpts3D.tolist(),
            'mconf': mconf.tolist(),

            # Filtered results from updated state:
            'kf_translation_vector': [px, py, pz],
            'kf_quaternion': [qx, qy, qz, qw],
            'kf_rotation_matrix': R_estimated.tolist(),

            # Additional coverage / viewpoint metrics
            'coverage_score': coverage_score,
            'viewpoint_diff_deg': viewpoint_diff
        }
        return pose_data

    def _visualize_matches(self, frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints):
        anchor_image_gray = cv2.cvtColor(self.anchor_image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        inlier_idx = inliers.flatten()
        inlier_mkpts0 = mkpts0[inlier_idx]
        inlier_mkpts1 = mkpts1[inlier_idx]
        inlier_conf = mconf[inlier_idx]
        color = cm.jet(inlier_conf)

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

        # Show the object (leader) position in camera frame from pose_data
        # E.g. we can just show tvec:
        t_in_cam = pose_data['object_translation_in_cam']
        position_text = (f"Leader in Cam: "
                         f"x={t_in_cam[0]:.3f}, y={t_in_cam[1]:.3f}, z={t_in_cam[2]:.3f}")
        cv2.putText(out, position_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2, cv2.LINE_AA)

        return out

    def _get_camera_intrinsics(self):
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