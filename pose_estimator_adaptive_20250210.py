import cv2
import torch
torch.set_grad_enabled(False)
import time
import numpy as np
from scipy.spatial import cKDTree
import onnxruntime as ort
from utils import frame2tensor, rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix
from KF_ICUAS import KalmanFilterPose
import matplotlib.cm as cm
from models.utils import make_matching_plot_fast
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pose_estimator.log"),  # Logs will be saved in this file
        logging.StreamHandler()                     # Logs will also be printed to console
    ]
)
logger = logging.getLogger(__name__)

# Import the SuperPointPreprocessor
from superpoint_LG import SuperPointPreprocessor


class PoseEstimator:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.initial_z_set = False
        self.kf_initialized = False

        logger.info("Initializing PoseEstimator")

        # Load anchor (leader) image
        self.anchor_image = cv2.imread(opt.anchor)
        assert self.anchor_image is not None, f"Failed to load anchor image at {opt.anchor}"
        self.anchor_image = self._resize_image(self.anchor_image, opt.resize)
        logger.info(f"Loaded and resized anchor image from {opt.anchor}")

        # Initialize ONNX session for LightGlue
        providers = [("CPUExecutionProvider", {})]
        providers.insert(0, ("CUDAExecutionProvider", {}))
        self.session = ort.InferenceSession(
            "weights/superpoint_lightglue_pipeline_1280x720_multihead.onnx",
            providers=providers
        )
        logger.info("ONNX session initialized with CUDAExecutionProvider")

        # Example anchor keypoints (2D -> 3D). Adapt these to your scenario.
        anchor_keypoints_2D = np.array([
            [511, 293], [591, 284], #[610, 269],
              [587, 330], [413, 249],
            [602, 348], [715, 384], [598, 298], [656, 171], [805, 213],
            [703, 392], [523, 286], [519, 327], [387, 289], [727, 126],
            [425, 243], [636, 358], [745, 202], [595, 388], [436, 260],
            [539, 313], [795, 220], [351, 291], [665, 165], [611, 353],
            [650, 377], [516, 389], [727, 143], [496, 378], [575, 312],
            [617, 368], [430, 312], [480, 281], [834, 225], [469, 339],
            [705, 223], [637, 156], [816, 414], [357, 195], [752, 77],
            [642, 451]
        ], dtype=np.float32)

        anchor_keypoints_3D = np.array([
            [-0.014, 0.000,  0.042], [ 0.025, -0.014, -0.011], #[ 0.049, -0.016, -0.011],
            [-0.014, 0.000, -0.042], [-0.014,  0.000,  0.156], [-0.023,  0.000, -0.065],
            [ 0.000,  0.000, -0.156], [ 0.025,  0.000, -0.015], [ 0.217,  0.000,  0.070],
            [ 0.230,  0.000, -0.070], [-0.014,  0.000, -0.156], [ 0.000,  0.000,  0.042],
            [-0.057, -0.018, -0.010], [-0.074, -0.000,  0.128], [ 0.206, -0.070, -0.002],
            [-0.000, -0.000,  0.156], [-0.017, -0.000, -0.092], [ 0.217, -0.000, -0.027],
            [-0.052, -0.000, -0.097], [-0.019, -0.000,  0.128], [-0.035, -0.018, -0.010],
            [ 0.217, -0.000, -0.070], [-0.080, -0.000,  0.156], [ 0.230, -0.000,  0.070],
            [-0.023, -0.000, -0.075], [-0.029, -0.000, -0.127], [-0.090, -0.000, -0.042],
            [ 0.206, -0.055, -0.002], [-0.090, -0.000, -0.015], [ 0.000, -0.000, -0.015],
            [-0.037, -0.000, -0.097], [-0.074, -0.000,  0.074], [-0.019, -0.000,  0.074],
            [ 0.230, -0.000, -0.113], [-0.100, -0.030,  0.000], [ 0.170, -0.000, -0.015],
            [ 0.230, -0.000,  0.113], [-0.000, -0.025, -0.240], [-0.000, -0.025,  0.240],
            [ 0.243, -0.104,  0.000], [-0.080, -0.000, -0.156]
        ], dtype=np.float32)

        #  Anchor_B
        #  # Example new 2D/3D correspondences for the new anchor
        #     # You must define these for your anchor
        # anchor_keypoints_2D = np.array([
        #         [650, 312],
        #         [630, 306],
        #         [907, 443],
        #         [814, 291],
        #         [599, 349],
        #         [501, 386],
        #         [965, 359],
        #         [649, 355],
        #         [635, 346],
        #         [930, 335],
        #         [843, 467],
        #         [702, 339],
        #         [718, 321],
        #         [930, 322],
        #         [727, 346],
        #         [539, 364],
        #         [786, 297],
        #         [1022, 406],
        #         [1004, 399],
        #         [539, 344],
        #         [536, 309],
        #         [864, 478],
        #         [745, 310],
        #         [1049, 393],
        #         [895, 258],
        #         [674, 347],
        #         [741, 281],
        #         [699, 294],
        #         [817, 494],
        #         [992, 281]
        #     ], dtype=np.float32)

        # anchor_keypoints_3D= np.array([
        #         [-0.035, -0.018, -0.010],
        #         [-0.057, -0.018, -0.010],
        #         [ 0.217, -0.000, -0.027],
        #         [-0.014, -0.000,  0.156],
        #         [-0.023, -0.000, -0.065],
        #         [-0.014, -0.000, -0.156],
        #         [ 0.234, -0.050, -0.002],
        #         [ 0.000, -0.000, -0.042],
        #         [-0.014, -0.000, -0.042],
        #         [ 0.206, -0.055, -0.002],
        #         [ 0.217, -0.000, -0.070],
        #         [ 0.025, -0.014, -0.011],
        #         [-0.014, -0.000,  0.042],
        #         [ 0.206, -0.070, -0.002],
        #         [ 0.049, -0.016, -0.011],
        #         [-0.029, -0.000, -0.127],
        #         [-0.019, -0.000,  0.128],
        #         [ 0.230, -0.000,  0.070],
        #         [ 0.217, -0.000,  0.070],
        #         [-0.052, -0.000, -0.097],
        #         [-0.175, -0.000, -0.015],
        #         [ 0.230, -0.000, -0.070],
        #         [-0.019, -0.000,  0.074],
        #         [ 0.230, -0.000,  0.113],
        #         [-0.000, -0.025,  0.240],
        #         [-0.000, -0.000, -0.015],
        #         [-0.074, -0.000,  0.128],
        #         [-0.074, -0.000,  0.074],
        #         [ 0.230, -0.000, -0.113],
        #         [ 0.243, -0.104,  0.000]
        #     ], dtype=np.float32)

        # Set anchor features (run SuperPoint on anchor, match to known 2D->3D)
        self._set_anchor_features(
            anchor_bgr_image=self.anchor_image,
            anchor_keypoints_2D=anchor_keypoints_2D,
            anchor_keypoints_3D=anchor_keypoints_3D
        )

        # Suppose the anchor is at yaw=0°, pitch=-20°, roll=0° (radians):
        self.anchor_viewpoint_eulers = np.array([0.0, -0.35, 0.0], dtype=np.float32)

        # Initialize Kalman filter
        self.kf_pose = self._init_kalman_filter()
        logger.info("Kalman filter initialized")

        # Adaptive gating / partial blending config
        self.skip_count = 0
        self.max_skip_before_reset = 4#3#5
        self.high_coverage_override = 0.63#0.8  # coverage override example
        self.use_partial_blending = True
        #self.blend_factor = 0.3

        # To track first update
        self.kf_pose_first_update = True


    


    

    def reinitialize_anchor(self, new_anchor_path, new_2d_points, new_3d_points):
        """
        Re-load a new anchor image and re-compute relevant data (2D->3D correspondences).
        """
        logger.info(f"Re-initializing anchor with new image: {new_anchor_path}")

        new_anchor_image = cv2.imread(new_anchor_path)
        assert new_anchor_image is not None, f"Failed to load new anchor image at {new_anchor_path}"
        new_anchor_image = self._resize_image(new_anchor_image, self.opt.resize)
        self.anchor_image = new_anchor_image

        self._set_anchor_features(
            anchor_bgr_image=new_anchor_image,
            anchor_keypoints_2D=new_2d_points,
            anchor_keypoints_3D=new_3d_points
        )

        logger.info("Anchor re-initialization complete.")



    def coverage_to_alpha(self, coverage_score, alpha_min=0.1, alpha_max=0.8):
        alpha_dynamic = alpha_min + (alpha_max - alpha_min) * coverage_score
        return alpha_dynamic

    def _set_anchor_features(self, anchor_bgr_image, anchor_keypoints_2D, anchor_keypoints_3D):
        """
        Run SuperPoint on the anchor image to get anchor_keypoints_sp.
        Then match those keypoints to known 2D->3D correspondences via KDTree.
        """
        self.anchor_proc = SuperPointPreprocessor.preprocess(anchor_bgr_image)
        self.anchor_proc = self.anchor_proc[None].astype(np.float32)

        # Forward pass on itself to get SuperPoint keypoints:
        anchor_batch = np.concatenate([self.anchor_proc, self.anchor_proc], axis=0)
        keypoints, matches, mscores = self.session.run(None, {"images": anchor_batch})
        self.anchor_keypoints_sp = keypoints[0]

        # Build KDTree to match anchor_keypoints_sp -> known anchor_keypoints_2D
        sp_tree = cKDTree(self.anchor_keypoints_sp)
        distances, indices = sp_tree.query(anchor_keypoints_2D, k=1)
        valid_matches = distances < 1.0

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
            f"Frame {frame_idx} times: Resize={resize_time - start_time:.3f}s, "
            f"Anchor_proc={anchor_proc_time - resize_time:.3f}s, "
            f"Frame_proc={frame_proc_time - anchor_proc_time:.3f}s, "
            f"Batch={batch_time - frame_proc_time:.3f}s, "
            f"Keypoint Extraction={keypoint_time - batch_time:.3f}s"
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
        dt = 1.0 / frame_rate
        return KalmanFilterPose(dt)

    def estimate_pose(self, mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, frame_keypoints):
        logger.debug(f"Estimating pose for frame {frame_idx}")
        K, distCoeffs = self._get_camera_intrinsics()

        objectPoints = mpts3D.reshape(-1, 1, 3)
        imagePoints = mkpts1.reshape(-1, 1, 2).astype(np.float32)

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

        R, _ = cv2.Rodrigues(rvec)

        # Classify for coverage score
        regions = {"front-right": 0, "front-left": 0, "back-right": 0, "back-left": 0}
        for pt in objectPoints_inliers[:, 0]:
            if pt[0] < 0 and pt[2] > 0:
                regions["front-right"] += 1
            elif pt[0] < 0 and pt[2] < 0:
                regions["front-left"] += 1
            elif pt[0] > 0 and pt[2] > 0:
                regions["back-right"] += 1
            elif pt[0] > 0 and pt[2] < 0:
                regions["back-left"] += 1

        total_points = sum(regions.values())
        if total_points > 0:
            proportions = {k: v / total_points for k, v in regions.items()}
            entropy = -sum(p * np.log(p) for p in proportions.values() if p > 0)
            max_entropy = np.log(len(regions))
            coverage_score = entropy / max_entropy if max_entropy > 0 else 0
        else:
            coverage_score = 0

        # Reprojection error
        projected_points, _ = cv2.projectPoints(objectPoints_inliers, rvec, tvec, K, distCoeffs)
        reprojection_errors = np.linalg.norm(imagePoints_inliers - projected_points, axis=2).flatten()
        mean_rep_error = float(np.mean(reprojection_errors))
        std_rep_error = float(np.std(reprojection_errors))

        pose_data = self._kalman_filter_update(
            R=R,
            tvec=tvec,
            reprojection_errors=reprojection_errors,
            mean_reprojection_error=mean_rep_error,
            std_reprojection_error=std_rep_error,
            inliers=inliers,
            mkpts0=mkpts0,
            mkpts1=mkpts1,
            mpts3D=mpts3D,
            mconf=mconf,
            frame_idx=frame_idx,
            rvec_o=rvec_o,
            rvec=rvec,
            coverage_score=coverage_score
        )

        pose_data["region_distribution"] = regions
        pose_data["coverage_score"] = coverage_score

        visualization = self._visualize_matches(
            frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints
        )
        return pose_data, visualization

    def _kalman_filter_update(
        self,
        R,
        tvec,
        reprojection_errors,
        mean_reprojection_error,
        std_reprojection_error,
        inliers,
        mkpts0,
        mkpts1,
        mpts3D,
        mconf,
        frame_idx,
        rvec_o,
        rvec,
        coverage_score
    ):
        """
        Adaptive gating / partial blending approach:
        1) Use Mahalanobis gating or a 3-sigma rule from KF covariance
        2) Optionally partial-blend if measurement is outside gate but not insane
        3) Skip-count to forcibly reset or forcibly accept after N consecutive skips
        """
        num_inliers = len(inliers)
        inlier_ratio = num_inliers / len(mkpts0) if len(mkpts0) > 0 else 0

        # 1) Convert R to Euler angles (measured)
        eulers_measured = rotation_matrix_to_euler_angles(R)

        # 2) Get prior (predicted) from KF
        translation_estimated, eulers_estimated = self.kf_pose.predict()

        # Basic differences just for logs or 3-sigma gating
        orientation_change_deg = np.linalg.norm(eulers_measured - eulers_estimated) * (180.0 / np.pi)
        translation_change = np.linalg.norm(tvec.flatten() - translation_estimated)

        # Compare viewpoint to anchor
        anchor_eulers = getattr(self, "anchor_viewpoint_eulers", np.array([0.0, 0.0, 0.0]))
        anchor_eulers_deg = np.degrees(anchor_eulers)
        eulers_measured_deg = np.degrees(eulers_measured)
        viewpoint_diff_deg = np.linalg.norm(eulers_measured_deg - anchor_eulers_deg)
        viewpoint_max_diff_deg = 380#120#380.0

        # Inlier / reprojection threshold
        reprojection_error_threshold = 5.0
        min_inliers = 6

        # ---------- Extract the 6×6 sub-block from the 18×18 covariance ----------
        full_P = self.kf_pose.get_covariance()  # shape (18, 18)
        # Indices for x,y,z,roll,pitch,yaw in your state
        idx_pose = [0, 1, 2, 9, 10, 11]
        P_pose = full_P[np.ix_(idx_pose, idx_pose)]  # shape (6,6)

        # ---------- Construct a 6×1 difference vector for x,y,z,roll,pitch,yaw ----------
        dx_6 = np.zeros((6, 1), dtype=np.float32)
        dx_6[0:3, 0] = tvec.flatten() - translation_estimated
        dx_6[3, 0] = eulers_measured[0] - eulers_estimated[0]
        dx_6[4, 0] = eulers_measured[1] - eulers_estimated[1]
        dx_6[5, 0] = eulers_measured[2] - eulers_estimated[2]

        # ---------- Mahalanobis gating ----------
        try:
            P_pose_inv = np.linalg.inv(P_pose)  # invert 6×6
            mahalanobis_sq = float(dx_6.T @ P_pose_inv @ dx_6)
            mahalanobis_dist = np.sqrt(mahalanobis_sq)
        except np.linalg.LinAlgError:
            logger.warning("Covariance is singular, fallback to large gating.")
            mahalanobis_dist = 9999.0

        # ~95% chi-square threshold for 6 DOF is ~12.59
        chi2_threshold_95_6dof = 12.59
        passed_mahalanobis = (0 < mahalanobis_sq < chi2_threshold_95_6dof)

        # ---------- 3-sigma approach (simple fallback) ----------
        # We'll approximate position/orientation stdev from diagonal
        # Compute the 3-sigma adaptive gating thresholds
        std_pos = np.sqrt(np.diag(P_pose)[0:3])
        std_rot = np.sqrt(np.diag(P_pose)[3:6])
        max_translation_jump_adapt = 3.0 * np.linalg.norm(std_pos)
        max_orientation_jump_adapt_deg = 3.0 * np.linalg.norm(std_rot) * (180.0 / np.pi)

        # Log the gating thresholds
        logger.info(
            f"Frame {frame_idx}: Gating Thresholds -> "
            f"max_translation_jump_adapt={max_translation_jump_adapt:.3f}, "
            f"max_orientation_jump_adapt_deg={max_orientation_jump_adapt_deg:.2f}°"
        )


        # coverage override if coverage is extremely high
        high_coverage_override = (coverage_score >= self.high_coverage_override)

        # ---------- If first KF update, always correct ----------
        if self.kf_pose_first_update:
            self.kf_pose.correct(tvec, R)
            self.skip_count = 0
            self.kf_pose_first_update = False
            logger.debug("First KF update: skipping gating checks.")
        else:
            # Basic checks
            enough_inliers = (num_inliers >= min_inliers)
            low_rep_error = (mean_reprojection_error < reprojection_error_threshold)
            viewpoint_ok = (viewpoint_diff_deg <= viewpoint_max_diff_deg)

            # Evaluate simpler gating
            passed_3sigma_gating = (
                (translation_change < max_translation_jump_adapt)
                and (orientation_change_deg < max_orientation_jump_adapt_deg)
            )

            # ---------- Decide if we accept or skip ----------
            if enough_inliers and low_rep_error and viewpoint_ok:
                # Accept if coverage override OR within gating
                
                #if high_coverage_override or passed_mahalanobis or passed_3sigma_gating:
                
                if high_coverage_override and passed_mahalanobis:
                    # Full correction
                    self.kf_pose.correct(tvec, R)
                    self.skip_count = 0
                    logger.debug("Kalman Filter corrected (passed gating).")

                    # Log why the correction was accepted
                    acceptance_reason = []
                    if high_coverage_override:
                        acceptance_reason.append("High Coverage Override")
                    if passed_mahalanobis:
                        acceptance_reason.append("Passed Mahalanobis Gating")
                    if passed_3sigma_gating:
                        acceptance_reason.append("Passed 3-Sigma Gating")

                    logger.info(
                        f"Frame {frame_idx}: Kalman Filter correction accepted due to {', '.join(acceptance_reason)}. "
                        f"Details: inliers={num_inliers}, inlier_ratio={inlier_ratio:.3f}, "
                        f"mean_reprojection_error={mean_reprojection_error:.3f}, viewpoint_diff={viewpoint_diff_deg:.2f}°, "
                        f"mahalanobis_sq={mahalanobis_sq:.3f}, coverage_score={coverage_score:.3f}, "
                        f"translation_change={translation_change:.3f}, orientation_change_deg={orientation_change_deg:.2f}°"
                    )

                else:
                #elif high_coverage_override or passed_mahalanobis:
                    # Outside gating => partial blend or skip
                    logger.debug(f"Frame {frame_idx}: Measurement out of gating => partial blend or skip.")

                    self.skip_count += 1
                    if self.use_partial_blending:
                        coverage_alpha = self.coverage_to_alpha(coverage_score)
                        self.kf_pose.correct_partial(tvec, R, alpha=coverage_alpha)

                        logger.info(
                            f"Frame {frame_idx}: Partial blend applied with alpha={coverage_alpha:.3f}. "
                            f"Details: inliers={num_inliers}, inlier_ratio={inlier_ratio:.3f}, "
                            f"mean_reprojection_error={mean_reprojection_error:.3f}, viewpoint_diff={viewpoint_diff_deg:.2f}°, "
                            f"mahalanobis_sq={mahalanobis_sq:.3f}, coverage_score={coverage_score:.3f}, "
                            f"translation_change={translation_change:.3f}, orientation_change_deg={orientation_change_deg:.2f}°"
                        )
                    else:
                        logger.debug(f"Frame {frame_idx}: Skipping update (hard skip).")

                    # If the skip count exceeds the max, force a correction
                    if self.skip_count > self.max_skip_before_reset:
                        logger.warning(f"Frame {frame_idx}: Exceeded max skip count ({self.max_skip_before_reset}) => Forcing correction.")
                        self.kf_pose.correct(tvec, R)
                        self.skip_count = 0

            else:
                # Not enough inliers or too large repro error => skip or partial
                logger.debug("Skipping (insufficient inliers or high repro error).")
                self.skip_count += 1

                if self.use_partial_blending:
                    coverage_alpha = self.coverage_to_alpha(coverage_score)
                    self.kf_pose.correct_partial(tvec, R, alpha=coverage_alpha)
                    logger.debug(f"Partial blend used despite low inliers, alpha={self.coverage_to_alpha}")

                    #self.kf_pose.correct_partial(tvec, R, alpha=self.blend_factor)
                    #logger.debug(f"Partial blend used despite low inliers, alpha={self.blend_factor}")

                if self.skip_count > self.max_skip_before_reset:
                    logger.warning("Exceeded max skip => forcing correction.")
                    self.kf_pose.correct(tvec, R)
                    self.skip_count = 0

        # ---------- Final predict ----------
        translation_estimated, eulers_estimated = self.kf_pose.predict()
        R_estimated = euler_angles_to_rotation_matrix(eulers_estimated)

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
            'kf_translation_vector': translation_estimated.tolist(),
            'kf_rotation_matrix': R_estimated.tolist(),
            'kf_euler_angles': eulers_estimated.tolist(),
            'coverage_score': coverage_score,
            'viewpoint_diff_deg': viewpoint_diff_deg,
            'skip_count': self.skip_count,
            'mahalanobis_dist': float(mahalanobis_dist),
            'mahalanobis_sq': float(mahalanobis_sq),
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

        # Show the object translation in text
        t_in_cam = pose_data['object_translation_in_cam']
        pos_text = (f"Leader in Cam: x={t_in_cam[0]:.3f}, y={t_in_cam[1]:.3f}, z={t_in_cam[2]:.3f}")
        cv2.putText(out, pos_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2, cv2.LINE_AA)

        return out

    def _get_camera_intrinsics(self):
        """
        Example function that returns camera matrix K and distortion.
        Customize as needed.
        """
        # Example #1 (comment out if not used)
        # focal_length_x = 1078.06451
        # focal_length_y = 1081.77221
        # cx = 628.078538
        # cy = 362.156441
        # distCoeffs = np.array([5.63748710e-02, -7.51721332e-01, -6.97952865e-04,
        #                        -3.84299642e-03, 6.18234012e+00], dtype=np.float32)


        ## 20250210
        # Example #2
        #focal_length_x = 1460.10150
        #focal_length_y = 1456.48915
        focal_length_x = 1430.10150
        focal_length_y = 1430.48915
        cx = 604.85462
        cy = 328.64800
        #cx = 640.85462
        #cy = 480.64800
        #distCoeffs = np.array([0.3393,2.0351,0.0295,-0.0029,-10.9093], dtype=np.float32)
        distCoeffs = None



         # Example #3 Opti web cali
        # focal_length_x = 1874.766
        # focal_length_y = 1878.368
        # cx = 640
        # cy = 360
        # focal_length_x = 1492.51791
        # focal_length_y = 1496.10186
        # cx = 615.01330
        # cy = 430.52894

        # distCoeffs = np.array([
        #     0.32152755,
        #     -1.29280843,
        #     0.03931456,
        #     -0.00651277,
        #     6.27888484
        # ], dtype=np.float32)


        #distCoeffs = np.array([0.3393,2.0351,0.0295,-0.0029,-10.9093], dtype=np.float32)
        #distCoeffs = None

        K = np.array([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K, distCoeffs
