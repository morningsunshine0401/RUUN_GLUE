# pose_estimator.py
import cv2
import torch
torch.set_grad_enabled(False)

import numpy as np
from scipy.spatial import cKDTree
from models.matching import Matching
from utils import frame2tensor
from utils import *
from kalman_filter import KalmanFilterPose
import matplotlib.cm as cm
from models.utils import make_matching_plot_fast

class PoseEstimator:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.matching = self._init_matching()
        self.anchor_image, self.anchor_tensor = self._load_anchor_image()
        self.anchor_keypoints_sp, self.anchor_descriptors_sp, self.anchor_scores_sp = self._extract_anchor_features()
        self.matched_anchor_keypoints, self.matched_descriptors, self.matched_scores, self.matched_3D_keypoints = self._match_anchor_keypoints()
        self.kf_pose = self._init_kalman_filter()

    def _init_matching(self):
        config = {
            'superpoint': {
                'nms_radius': self.opt.nms_radius,
                'keypoint_threshold': self.opt.keypoint_threshold,
                'max_keypoints': self.opt.max_keypoints,
            },
            'superglue': {
                'weights': self.opt.superglue,
                'sinkhorn_iterations': self.opt.sinkhorn_iterations,
                'match_threshold': self.opt.match_threshold,
            }
        }
        matching = Matching(config).eval().to(self.device)
        return matching

    def _load_anchor_image(self):
        anchor_image = cv2.imread(self.opt.anchor)
        assert anchor_image is not None, f'Failed to load anchor image at {self.opt.anchor}'
        if len(self.opt.resize) == 2:
            anchor_image = cv2.resize(anchor_image, tuple(self.opt.resize))
        elif len(self.opt.resize) == 1 and self.opt.resize[0] > 0:
            h, w = anchor_image.shape[:2]
            scale = self.opt.resize[0] / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            anchor_image = cv2.resize(anchor_image, new_size)
        anchor_tensor = frame2tensor(anchor_image, self.device)
        return anchor_image, anchor_tensor

    def _extract_anchor_features(self):
        anchor_data = self.matching.superpoint({'image': self.anchor_tensor})
        keypoints = anchor_data['keypoints'][0].cpu().numpy()
        descriptors = anchor_data['descriptors'][0].cpu().numpy()
        scores = anchor_data['scores'][0].cpu().numpy()
        return keypoints, descriptors, scores

    def _match_anchor_keypoints(self):
        # Load your provided 2D and 3D keypoints here
        anchor_keypoints_2D = np.array([
            [563, 565], 
            [77, 582], 
            [515, 318], 
            [606, 317], 
            [612, 411],
            [515, 414], 
            [420, 434], 
            [420, 465], 
            [618, 455], 
            [500, 123], 
            [418, 153], 
            [417, 204], 
            [417, 243], 
            [502, 279],
            [585, 240],  
            [289, 26],  
            [322, 339], 
            [349, 338], 
            [349, 374], 
            [321, 375],
            [390, 349], 
            [243, 462], 
            [367, 550], 
            [368, 595], 
            [383, 594],
            [386, 549], 
            [779, 518], 
            [783, 570]
            
        ], dtype=np.float32)
        
        anchor_keypoints_3D = np.array([
            [0.03, -0.165, 0.05],
            [-0.190, -0.165, 0.050],
            [0.010, -0.025, 0.0],
            [0.060, -0.025, 0.0],
            [0.06, -0.080, 0.0],
            [0.010, -0.080, 0.0],
            [-0.035, -0.087, 0.0],
            [-0.035, -0.105, 0.0],
            [0.065, -0.105, 0.0],
            [0.0, 0.045, 0.0],
            [-0.045, 0.078, 0.0],
            [-0.045, 0.046, 0.0],
            [-0.045, 0.023, 0.0],
            [0.0, -0.0, 0.0],
            [0.045, 0.022, 0.0],
            [-0.120, 0.160, 0.0],
            [-0.095, -0.035,0.0],
            [-0.080, -0.035, 0.0],
            [-0.080, -0.055, 0.0],
            [-0.095, -0.055, 0.0],
            [-0.050, -0.040, 0.010],
            [-0.135, -0.100, 0.0],
            [-0.060, -0.155, 0.050],
            [-0.060, -0.175, 0.050],
            [-0.052, -0.175, 0.050],
            [-0.052, -0.155, 0.050],
            [0.135, -0.147, 0.050],
            [0.135, -0.172, 0.050]

        ], dtype=np.float32)

        sp_tree = cKDTree(self.anchor_keypoints_sp)
        distances, indices = sp_tree.query(anchor_keypoints_2D, k=1)
        distance_threshold = 1
        valid_matches = distances < distance_threshold

        if not np.any(valid_matches):
            raise ValueError('No matching keypoints found within the distance threshold.')

        matched_anchor_indices = indices[valid_matches]
        matched_3D_keypoints = anchor_keypoints_3D[valid_matches]

        matched_anchor_keypoints = self.anchor_keypoints_sp[matched_anchor_indices]
        matched_descriptors = self.anchor_descriptors_sp[:, matched_anchor_indices]
        matched_scores = self.anchor_scores_sp[matched_anchor_indices]

        return matched_anchor_keypoints, matched_descriptors, matched_scores, matched_3D_keypoints

    def _init_kalman_filter(self):
        frame_rate = 30  # Adjust this to your video frame rate
        dt = 1 / frame_rate
        kf_pose = KalmanFilterPose(dt)
        return kf_pose

    def process_frame(self, frame, frame_idx):
        # Resize frame if needed
        if len(self.opt.resize) == 2:
            frame = cv2.resize(frame, tuple(self.opt.resize))
        elif len(self.opt.resize) == 1 and self.opt.resize[0] > 0:
            h, w = frame.shape[:2]
            scale = self.opt.resize[0] / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            frame = cv2.resize(frame, new_size)

        frame_tensor = frame2tensor(frame, self.device)

        # Extract features from current frame
        with torch.no_grad():
            frame_data = self.matching.superpoint({'image': frame_tensor})
            frame_keypoints = frame_data['keypoints'][0].cpu().numpy()
            frame_descriptors = frame_data['descriptors'][0].cpu().numpy()
            frame_scores = frame_data['scores'][0].cpu().numpy()

        # Prepare data for SuperGlue matching
        input_superglue = {
            'keypoints0': torch.from_numpy(self.matched_anchor_keypoints).unsqueeze(0).to(self.device),
            'keypoints1': torch.from_numpy(frame_keypoints).unsqueeze(0).to(self.device),
            'descriptors0': torch.from_numpy(self.matched_descriptors).unsqueeze(0).to(self.device),
            'descriptors1': torch.from_numpy(frame_descriptors).unsqueeze(0).to(self.device),
            'scores0': torch.from_numpy(self.matched_scores).unsqueeze(0).to(self.device),
            'scores1': torch.from_numpy(frame_scores).unsqueeze(0).to(self.device),
            'image0': self.anchor_tensor,
            'image1': frame_tensor,
        }

        # Perform matching
        pred = self.matching.superglue(input_superglue)
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

        # Process matches
        valid = matches > -1
        mkpts0 = self.matched_anchor_keypoints[valid]
        mkpts1 = frame_keypoints[matches[valid]]
        mpts3D = self.matched_3D_keypoints[valid]
        mconf = confidence[valid]

        total_matches = len(mkpts0)

        if total_matches >= 4:
            pose_data, visualization = self.estimate_pose(
                mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, frame_keypoints)
            return pose_data, visualization
        else:
            print('Not enough matches to compute pose.')
            return None, frame

    def estimate_pose(self, mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, frame_keypoints):
        # Camera intrinsics (adjust with your calibration data)
        K, distCoeffs = self._get_camera_intrinsics()
        objectPoints = mpts3D.reshape(-1, 1, 3)
        imagePoints = mkpts1.reshape(-1, 1, 2)

        # Solve PnP
        success, rvec_o, tvec_o, inliers = cv2.solvePnPRansac(
            objectPoints=objectPoints,
            imagePoints=imagePoints,
            cameraMatrix=K,
            distCoeffs=distCoeffs,
            reprojectionError=5,
            confidence=0.9,
            iterationsCount=1500,
            flags=cv2.SOLVEPNP_P3P
        )

        if success and inliers is not None and len(inliers) >= 3:
            # Refine pose
            objectPoints_inliers = mpts3D[inliers.flatten()].reshape(-1, 1, 3)
            imagePoints_inliers = mkpts1[inliers.flatten()].reshape(-1, 1, 2)
            rvec, tvec = cv2.solvePnPRefineVVS(
                objectPoints=objectPoints_inliers,
                imagePoints=imagePoints_inliers,
                cameraMatrix=K,
                distCoeffs=distCoeffs,
                rvec=rvec_o,
                tvec=tvec_o
            )

            # Convert to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            camera_position = -R.T @ tvec

            # Compute reprojection errors
            projected_points, _ = cv2.projectPoints(
                objectPoints=objectPoints_inliers,
                rvec=rvec,
                tvec=tvec,
                cameraMatrix=K,
                distCoeffs=distCoeffs
            )
            reprojection_errors = np.linalg.norm(imagePoints_inliers - projected_points, axis=2).flatten()
            mean_reprojection_error = np.mean(reprojection_errors)
            std_reprojection_error = np.std(reprojection_errors)

            # Kalman Filter Update
            pose_data = self._kalman_filter_update(
                R, tvec, reprojection_errors, mean_reprojection_error, std_reprojection_error,
                inliers, mkpts0, mkpts1, mpts3D, mconf, frame_idx, camera_position
            )

            # Visualization
            visualization = self._visualize_matches(
                frame, inliers, mkpts0, mkpts1, mconf, pose_data , frame_keypoints
            )

            return pose_data, visualization
        else:
            print('PnP pose estimation failed.')
            return None, frame

    def _get_camera_intrinsics(self):
        # Replace with your camera's intrinsic parameters
        focal_length_x = 1079.83796  # px
        focal_length_y = 1081.11500  # py
        cx = 627.318141
        cy = 332.745740
        distCoeffs = np.array([0.0314, -0.2847, -0.0105, -0.00005, 1.0391], dtype=np.float32)
        K = np.array([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return K, distCoeffs

    def _kalman_filter_update(self, R, tvec, reprojection_errors, mean_reprojection_error,
                              std_reprojection_error, inliers, mkpts0, mkpts1, mpts3D,
                              mconf, frame_idx, camera_position):
        num_inliers = len(inliers)
        inlier_ratio = num_inliers / len(mkpts0)

        # Thresholds
        reprojection_error_threshold = 10
        max_translation_jump = 7.0

        # Kalman Filter Update
        translation_estimated, eulers_estimated = self.kf_pose.predict()
        eulers_measured = rotation_matrix_to_euler_angles(R)

        translation_change = np.linalg.norm(tvec.flatten() - translation_estimated)
        rotation_change = np.linalg.norm(eulers_measured - eulers_estimated)

        if mean_reprojection_error < reprojection_error_threshold:
            if translation_change < max_translation_jump:
                self.kf_pose.correct(tvec, R)
            else:
                print("Skipping Kalman update due to large jump in translation/rotation.")
        else:
            print("Skipping Kalman update due to high reprojection error.")

        # Get updated state
        translation_estimated, eulers_estimated = self.kf_pose.predict()
        R_estimated = euler_angles_to_rotation_matrix(eulers_estimated)
        t_estimated = translation_estimated.reshape(3, 1)

        pose_data = {
            'frame': frame_idx,
            'rotation_matrix': R.tolist(),
            'translation_vector': tvec.flatten().tolist(),
            'camera_position': camera_position.flatten().tolist(),
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
            'kf_translation_vector': translation_estimated.tolist(),
            'kf_rotation_matrix': R_estimated.tolist(),
            'kf_euler_angles': eulers_estimated.tolist()
        }
        return pose_data

    def _visualize_matches(self, frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints):
        # Convert images to grayscale for visualization
        if len(self.anchor_image.shape) == 2 or self.anchor_image.shape[2] == 1:
            anchor_image_gray = self.anchor_image
        else:
            anchor_image_gray = cv2.cvtColor(self.anchor_image, cv2.COLOR_BGR2GRAY)

        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            frame_gray = frame
        else:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        inlier_idx = inliers.flatten()
        inlier_mkpts0 = mkpts0[inlier_idx]
        inlier_mkpts1 = mkpts1[inlier_idx]
        inlier_conf = mconf[inlier_idx]
        color = cm.jet(inlier_conf)

        out = make_matching_plot_fast(
            anchor_image_gray,
            frame_gray,
            self.matched_anchor_keypoints,
            frame_keypoints,
            inlier_mkpts0,
            inlier_mkpts1,
            color,
            text=[],
            path=None,
            show_keypoints=self.opt.show_keypoints,
            small_text=[]
        )

        position_text = f"Position: {pose_data['camera_position']}"
        cv2.putText(out, position_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2, cv2.LINE_AA)
        return out
