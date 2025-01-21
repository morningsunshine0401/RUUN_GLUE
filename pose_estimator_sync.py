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
        self.initial_z_set = False  # Flag for first-frame Z override (if desired)
        self.kf_initialized = False  # To track if Kalman filter was ever updated

        logger.info("Initializing PoseEstimator")

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
            #"weights/superpoint_lightglue_pipeline.onnx",
            #"weights/superpoint_lightglue_pipeline_1280x720.onnx",
            "weights/superpoint_lightglue_pipeline_1280x720_multihead.onnx",
            providers=providers
        )
        logger.info("ONNX session initialized with CUDAExecutionProvider")

        # We will store the anchor’s 2D/3D keypoints here.
        # For your anchor image, you can define them directly or load from a file.
        anchor_keypoints_2D = np.array([
            [511, 293], #
            [591, 284], #
            [610, 269], #
            [587, 330], #
            [413, 249], #
            [602, 348], #
            [715, 384], #
            [598, 298], #
            [656, 171], #
            [805, 213],#
            [703, 392],# 
            [523, 286],#
            [519, 327],#
            [387, 289],#
            [727, 126],# 
            [425, 243],# 
            [636, 358],#
            [745, 202],#
            [595, 388],#
            [436, 260],#
            [539, 313], #
            [795, 220],# 
            [351, 291],#
            [665, 165],# 
            [611, 353], #
            [650, 377],#
            [516, 389],## 
            [727, 143], #
            [496, 378], #
            [575, 312], #
            [617, 368],#
            [430, 312], #
            [480, 281], #
            [834, 225], #
            [469, 339], #
            [705, 223], #
            [637, 156], 
            [816, 414], 
            [357, 195], 
            [752, 77], 
            [642, 451]
        ], dtype=np.float32)

        # # 640 x 360
        # anchor_keypoints_2D = np.array([
        #         [255.5, 146.5],
        #     [295.5, 142.0],
        #     [305.0, 134.5],
        #     [293.5, 165.0],
        #     [206.5, 124.5],
        #     [301.0, 174.0],
        #     [357.5, 192.0],
        #     [299.0, 149.0],
        #     [328.0,  85.5],
        #     [402.5, 106.5],
        #     [351.5, 196.0],
        #     [261.5, 143.0],
        #     [259.5, 163.5],
        #     [193.5, 144.5],
        #     [363.5,  63.0],
        #     [212.5, 121.5],
        #     [318.0, 179.0],
        #     [372.5, 101.0],
        #     [297.5, 194.0],
        #     [218.0, 130.0],
        #     [269.5, 156.5],
        #     [397.5, 110.0],
        #     [175.5, 145.5],
        #     [332.5,  82.5],
        #     [305.5, 176.5],
        #     [325.0, 188.5],
        #     [258.0, 194.5],
        #     [363.5,  71.5],
        #     [248.0, 189.0],
        #     [287.5, 156.0],
        #     [308.5, 184.0],
        #     [215.0, 156.0],
        #     [240.0, 140.5],
        #     [417.0, 112.5],
        #     [234.5, 169.5],
        #     [352.5, 111.5],
        #     [318.5,  78.0],
        #     [408.0, 207.0],
        #     [178.5,  97.5],
        #     [376.0,  38.5],
        #     [321.0, 225.5]]

        #     , dtype=np.float32)

        anchor_keypoints_3D = np.array([
            [-0.014,  0.000,  0.042],
            [ 0.025, -0.014, -0.011],
            [ 0.049, -0.016, -0.011],
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
        logger.info("Kalman filter initialized")


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

        # 4. (Optional) Reset the Kalman filter if you want a fresh start
        #    Otherwise, keep it to maintain continuity.
        # self.kf_pose = self._init_kalman_filter()
        # self.kf_initialized = False

        logger.info("Anchor re-initialization complete.")

    def _set_anchor_features(self, anchor_bgr_image, anchor_keypoints_2D, anchor_keypoints_3D):
        """
        Run SuperPoint on the anchor image to get anchor_keypoints_sp.
        Then match those keypoints to known 2D->3D correspondences via KDTree.
        """
        # Precompute anchor’s SuperPoint descriptors
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
        print("$$$$$$$$$$$tvec_o:",tvec_o)

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
            proportions = {region: count / total_points for region, count in regions.items()}
            entropy = -sum(p * np.log(p) for p in proportions.values() if p > 0)
            max_entropy = np.log(len(regions))  # ln(4) for 4 regions
            coverage_score = entropy / max_entropy if max_entropy > 0 else 0
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
            mconf, frame_idx, rvec_o, rvec,coverage_score=coverage_score
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
        mconf, frame_idx, rvec_o, rvec,coverage_score
    ):
        num_inliers = len(inliers)
        inlier_ratio = num_inliers / len(mkpts0) if len(mkpts0) > 0 else 0

        reprojection_error_threshold = 5.0
        max_translation_jump = 0.15
        max_orientation_jump = 15.0  # degrees
        min_inlier = 6

        # ---------------------------
        coverage_threshold = 0.55  # e.g., need at least 0.55 coverage to trust this frame

        
        coverage_score = coverage_score#getattr(self, "last_coverage_score", 1.0)

        # ---------------------------
        # EXTRA CHECKS: viewpoint
        # Suppose we compare eulers_measured to self.anchor_viewpoint_eulers
        anchor_eulers = getattr(self, "anchor_viewpoint_eulers", np.array([0.0, 0.0, 0.0]))
        viewpoint_max_diff_deg = 380.0  # Example: if viewpoint differs >80°, skip

        # Convert to degrees:
        eulers_measured_deg = np.degrees(rotation_matrix_to_euler_angles(R))
        anchor_eulers_deg = np.degrees(anchor_eulers)
        viewpoint_diff = np.linalg.norm(eulers_measured_deg - anchor_eulers_deg)  # simple Euclidian in Euler angles


        # 1) PREDICT
        translation_estimated, eulers_estimated = self.kf_pose.predict()
        eulers_measured = rotation_matrix_to_euler_angles(R)  # (in radians)
        orientation_change = np.linalg.norm(eulers_measured - eulers_estimated) * (180 / np.pi)
        translation_change = np.linalg.norm(tvec.flatten() - translation_estimated)

        # If first KF update
        if not hasattr(self, 'kf_pose_first_update') or self.kf_pose_first_update:
            self.kf_pose.correct(tvec, R)
            self.kf_pose_first_update = False
            logger.debug("Kalman Filter first update: skipping threshold checks.")
        else:
            # Normal frames: check if we pass all thresholds
            # 1) Enough inliers + low reprojection error
            if mean_reprojection_error < reprojection_error_threshold and num_inliers > min_inlier:
                # 2) Check translation/orientation jump
                if translation_change < max_translation_jump and orientation_change < max_orientation_jump:
                    # 3) Check coverage
                    if coverage_score >= coverage_threshold:
                        # 4) Check viewpoint difference
                        if viewpoint_diff <= viewpoint_max_diff_deg:
                            # --> Everything is okay, do correction
                            self.kf_pose.correct(tvec, R)
                            logger.debug("Kalman Filter corrected (all thresholds passed).")
                        else:
                            logger.debug(f"Skipping KF update: viewpoint diff = {viewpoint_diff:.1f} deg > {viewpoint_max_diff_deg}")
                    else:
                        logger.debug(f"Skipping KF update: coverage_score={coverage_score:.2f} < {coverage_threshold}")
                else:
                    # Exceeded motion thresholds
                    if translation_change >= max_translation_jump:
                        logger.debug(f"Skipping KF update: large translation jump={translation_change:.3f}m")
                    if orientation_change >= max_orientation_jump:
                        logger.debug(f"Skipping KF update: large orientation jump={orientation_change:.3f}deg")
            else:
                logger.debug("Skipping KF update: high repro error or insufficient inliers.")

        # FINAL PREDICT (gets final state after optional correction)
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
            # Optionally store coverage/viewpoint metrics
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
        # # Replace with your real camera intrinsics: Lab
        # focal_length_x = 1121.87155
        # focal_length_y = 1125.27185
        # cx = 642.208561
        # cy = 394.971663

        # distCoeffs = np.array(
        #     [-2.28097367e-03, 1.33152199e+00, 1.09716884e-02, 1.68743767e-03, -8.17039260e+00],
        #     dtype=np.float32
        # )
        
        #################################################################################
        # # Calib_Phone_Opti
        # focal_length_x = 1078.06451
        # focal_length_y = 1081.77221
        # cx = 628.078538
        # cy = 362.156441

        # distCoeffs = np.array(
        #     [5.63748710e-02, -7.51721332e-01, -6.97952865e-04, -3.84299642e-03,6.18234012e+00],
        #     dtype=np.float32
        # )
        ################################################################################

        #webcam???? NO...
        focal_length_x = 1526.22  # px
        focal_length_y = 1531.18  # py
        cx = 637.98  # Principal point u0
        cy = 416.04  # Principal point v0
        distCoeffs = None


        ##########################################################

        # # Calib_Phone_Opti: 640 x 360
        # focal_length_x = 539.032255
        # focal_length_y = 540.886105
        # cx = 314.039269
        # cy = 181.0782205

        # distCoeffs = np.array(
        #     [5.6374870e-02, -7.5172132e-01, -6.9795287e-04, -3.8429964e-03, 6.1823401e+00],
        #     dtype=np.float32
        # )

        # distCoeffs = None


        K = np.array([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K, distCoeffs
