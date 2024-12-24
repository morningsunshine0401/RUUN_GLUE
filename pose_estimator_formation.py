import cv2
import torch
torch.set_grad_enabled(False)
import time
import numpy as np
from scipy.spatial import cKDTree
import onnxruntime as ort
from utils import frame2tensor, rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix
from kalman_filter import KalmanFilterPose
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

# Import the SuperPointPreprocessor
from superpoint_LG import SuperPointPreprocessor

class PoseEstimator:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.initial_z_set = False  # Flag to track initial Z application

        logger.info("Initializing PoseEstimator")

        # Load anchor (leader) image
        self.anchor_image = cv2.imread(opt.anchor)
        assert self.anchor_image is not None, f'Failed to load anchor image at {opt.anchor}'
        self.anchor_image = self._resize_image(self.anchor_image, opt.resize)
        logger.info(f"Loaded and resized anchor image from {opt.anchor}")

        # Initialize ONNX Session for LightGlue
        providers = [("CPUExecutionProvider", {})]
        providers.insert(0, ("CUDAExecutionProvider", {}))
        self.session = ort.InferenceSession("weights/superpoint_lightglue_pipeline.onnx",
                                            providers=providers)
        logger.info("ONNX session initialized with CUDAExecutionProvider")

        # Precompute anchor features
        self.anchor_proc = SuperPointPreprocessor.preprocess(self.anchor_image)
        self.anchor_proc = self.anchor_proc[None].astype(np.float32)

        anchor_batch = np.concatenate([self.anchor_proc, self.anchor_proc], axis=0)
        keypoints, matches, mscores = self.session.run(None, {"images": anchor_batch})
        self.anchor_keypoints_sp = keypoints[0]
        logger.info("Extracted anchor keypoints")

        # Known 2D <-> 3D correspondences
        anchor_keypoints_2D = np.array([
            [563, 565], [77, 582], [515, 318], [606, 317], [612, 411],
            [515, 414], [420, 434], [420, 465], [618, 455], [500, 123],
            [418, 153], [417, 204], [417, 243], [502, 279], [585, 240],
            [289, 26], [322, 339], [349, 338], [349, 374], [321, 375],
            [390, 349], [243, 462], [367, 550], [368, 595], [383, 594],
            [386, 549], [779, 518], [783, 570]
        ], dtype=np.float32)

        # Leader's 3D coordinates (anchor frame)
        anchor_keypoints_3D = np.array([
            [0.03, -0.05, -0.165],
            [-0.190, -0.050, -0.165],
            [0.010, -0.0, -0.025],
            [0.060, -0.0, -0.025],
            [0.06, -0.0, -0.080],
            [0.010, -0.0, -0.080],
            [-0.035, -0.0, -0.087],
            [-0.035, -0.0, -0.105],
            [0.065, -0.0, -0.105],
            [0.0, 0.0, 0.045],
            [-0.045, 0.0, 0.078],
            [-0.045, 0.0, 0.046],
            [-0.045, 0.0, 0.023],
            [0.0, 0.0, 0.0],
            [0.045, 0.0, 0.022],
            [-0.120, 0.0, 0.160],
            [-0.095, 0.0, -0.035],
            [-0.080, 0.0, -0.035],
            [-0.080, 0.0, -0.055],
            [-0.095, 0.0, -0.055],
            [-0.050, -0.010, -0.040],
            [-0.135, 0.0, -0.1],
            [-0.060, -0.050, -0.155],
            [-0.060, -0.050, -0.175],
            [-0.052, -0.050, -0.175],
            [-0.052, -0.050, -0.155],
            [0.135, -0.050, -0.147],
            [0.135, -0.050, -0.172]
        ], dtype=np.float32)

        # Build KDTree on anchor keypoints
        sp_tree = cKDTree(self.anchor_keypoints_sp)
        distances, indices = sp_tree.query(anchor_keypoints_2D, k=1)
        valid_matches = distances < 1.0

        self.matched_anchor_indices = indices[valid_matches]
        self.matched_3D_keypoints = anchor_keypoints_3D[valid_matches]
        logger.info(f"Matched {len(self.matched_anchor_indices)} keypoints to 3D points")

        # Initialize Kalman filter
        self.kf_pose = self._init_kalman_filter()
        logger.info("Kalman filter initialized")

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

        frame = self._resize_image(frame, self.opt.resize)
        resize_time = time.time()

        anchor_proc_time = time.time()
        anchor_proc = self.anchor_proc  # Precomputed anchor

        frame_proc = SuperPointPreprocessor.preprocess(frame)
        frame_proc = frame_proc[None].astype(np.float32)
        frame_proc_time = time.time()

        batch = np.concatenate([anchor_proc, frame_proc], axis=0).astype(np.float32)
        batch = torch.tensor(batch, device=self.device).cpu().numpy()
        batch_time = time.time()

        keypoints, matches, mscores = self.session.run(None, {"images": batch})
        keypoint_time = time.time()

        logger.info(
            f"Frame {frame_idx} times: Resize: {resize_time - start_time:.3f}s, "
            f"Anchor_proc: {anchor_proc_time - resize_time:.3f}s, "
            f"Frame_proc: {frame_proc_time - anchor_proc_time:.3f}s, "
            f"Batch: {batch_time - frame_proc_time:.3f}s, "
            f"Keypoint Extraction: {keypoint_time - batch_time:.3f}s"
        )

        valid_mask = (matches[:, 0] == 0)
        valid_matches = matches[valid_mask]

        mkpts0 = keypoints[0][valid_matches[:, 1]]
        mkpts1 = keypoints[1][valid_matches[:, 2]]
        mconf = mscores[valid_mask]
        anchor_indices = valid_matches[:, 1]

        logger.debug(f"Found {len(mkpts0)} raw matches in frame {frame_idx}")

        # Filter to known anchor indices
        known_mask = np.isin(anchor_indices, self.matched_anchor_indices)
        mkpts0 = mkpts0[known_mask]
        mkpts1 = mkpts1[known_mask]
        mconf = mconf[known_mask]

        idx_map = {idx: i for i, idx in enumerate(self.matched_anchor_indices)}
        mpts3D = np.array([self.matched_3D_keypoints[idx_map[aidx]] 
                           for aidx in anchor_indices[known_mask]])

        if len(mkpts0) < 4:
            logger.warning(f"Not enough matches for pose (frame {frame_idx})")
            return None, frame

        pose_data, visualization = self.estimate_pose(
            mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, keypoints[1]
        )
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
            reprojectionError=8,
            confidence=0.99,
            iterationsCount=2000,
            flags=cv2.SOLVEPNP_P3P
        )

        if not success or inliers is None or len(inliers) < 7:
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
        # "t" is the object's origin in the camera frame

        # For debugging, we used to compute camera_position = -R.T @ tvec
        # but you want the object's position in camera frame, which is just tvec!

        # Compute reprojection errors
        projected_points, _ = cv2.projectPoints(
            objectPoints_inliers, rvec, tvec, K, distCoeffs
        )
        reprojection_errors = np.linalg.norm(imagePoints_inliers - projected_points, axis=2).flatten()
        mean_reprojection_error = np.mean(reprojection_errors)
        std_reprojection_error = np.std(reprojection_errors)

        pose_data = self._kalman_filter_update(
            R, tvec, reprojection_errors, mean_reprojection_error,
            std_reprojection_error, inliers, mkpts0, mkpts1, mpts3D,
            mconf, frame_idx, rvec_o, rvec
        )

        visualization = self._visualize_matches(
            frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints
        )
        return pose_data, visualization

    def _kalman_filter_update(
        self, R, tvec, reprojection_errors, mean_reprojection_error,
        std_reprojection_error, inliers, mkpts0, mkpts1, mpts3D,
        mconf, frame_idx, rvec_o, rvec
    ):
        num_inliers = len(inliers)
        inlier_ratio = num_inliers / len(mkpts0) if len(mkpts0) > 0 else 0

        reprojection_error_threshold = 10
        max_translation_jump = 4
        min_inlier = 5

        # Predict
        translation_estimated, eulers_estimated = self.kf_pose.predict()
        eulers_measured = rotation_matrix_to_euler_angles(R)

        translation_change = np.linalg.norm(tvec.flatten() - translation_estimated)

        # Apply ground truth Z for the first frame only (if desired)
        if not self.initial_z_set:
            # tvec[2] = ...
            self.initial_z_set = True

        # Update conditions
        if mean_reprojection_error < reprojection_error_threshold and num_inliers > min_inlier:
            if translation_change < max_translation_jump:
                self.kf_pose.correct(tvec, R)
                logger.debug("Kalman Filter corrected.")
            else:
                logger.debug("Skipping Kalman update: large jump in translation.")
        else:
            logger.debug("Skipping Kalman update: high reprojection error.")

        # Final predicted
        translation_estimated, eulers_estimated = self.kf_pose.predict()
        R_estimated = euler_angles_to_rotation_matrix(eulers_estimated)

        # Just for reference, if you want to see the final predicted object position in camera frame:
        # predicted_tvec = translation_estimated
        # predicted_R     = R_estimated

        pose_data = {
            'frame': frame_idx,
            # The next two store the refined object->camera transform:
            'object_rotation_in_cam': R.tolist(),
            'object_translation_in_cam': tvec.flatten().tolist(),

            'raw_rvec': rvec_o.flatten().tolist(),
            'refined_raw_rvec': rvec.flatten().tolist(),

            # If you still want to store how many inliers, errors, etc.
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

            # Kalman filter states (predicted transform):
            'kf_translation_vector': translation_estimated.tolist(),
            'kf_rotation_matrix': R_estimated.tolist(),
            'kf_euler_angles': eulers_estimated.tolist()
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
        # # Replace with your real camera intrinsics: Lab
        # focal_length_x = 1121.87155
        # focal_length_y = 1125.27185
        # cx = 642.208561
        # cy = 394.971663

        # distCoeffs = np.array(
        #     [-2.28097367e-03, 1.33152199e+00, 1.09716884e-02, 1.68743767e-03, -8.17039260e+00],
        #     dtype=np.float32
        # )
        
        # Calib_Phone_Opti
        focal_length_x = 1078.06451
        focal_length_y = 1081.77221
        cx = 628.078538
        cy = 362.156441

        distCoeffs = np.array(
            [5.63748710e-02, -7.51721332e-01, -6.97952865e-04, -3.84299642e-03,6.18234012e+00],
            dtype=np.float32
        )

        K = np.array([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K, distCoeffs
