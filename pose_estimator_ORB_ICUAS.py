import cv2
import numpy as np
from scipy.spatial import cKDTree
import logging

# If you have a local KalmanFilterPose and utility functions:
from kalman_filter import KalmanFilterPose  # or your KF code
from utils import rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PoseEstimator:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        logger.info("Using ORB + BFMatcher + cKDTree to map anchor keypoints to 3D.")

        # ----------------------------------------------------------------------
        # 1) LOAD ANCHOR & 2D->3D
        # ----------------------------------------------------------------------
        self.anchor_image = cv2.imread(opt.anchor)
        assert self.anchor_image is not None, f"Failed to load anchor image at {opt.anchor}"
        self.anchor_image = self._resize_image(self.anchor_image, opt.resize)
        logger.info(f"Loaded & resized anchor image: {opt.anchor}")

        # We will store the anchor’s 2D/3D keypoints here.
        # For your anchor image, you can define them directly or load from a file.
        self.anchor_keypoints_2D = np.array([
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

        self.anchor_keypoints_3D = np.array([
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

        assert len(self.anchor_keypoints_2D) == len(self.anchor_keypoints_3D), \
            "anchor_keypoints_2D and anchor_keypoints_3D must have same length."

        # ----------------------------------------------------------------------
        # 2) Initialize ORB on anchor
        # ----------------------------------------------------------------------
        self.orb = cv2.ORB_create(nfeatures=2000)  # or tweak nfeatures
        self.anchor_kp, self.anchor_desc, self.idx_map = self._compute_anchor_descriptors()

        # BFMatcher for matching anchor_desc -> frame_desc
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # ----------------------------------------------------------------------
        # 3) Initialize Kalman Filter
        # ----------------------------------------------------------------------
        self.kf_pose = KalmanFilterPose(dt=1/30.0)
        self.kf_pose_first_update = True
        logger.info("Kalman filter initialized.")



    def reinitialize_anchor(self, new_anchor_path, new_2d_points, new_3d_points):
        """
        Re-load a new anchor image and re-compute relevant data (2D->3D correspondences).
        Called on-the-fly (e.g. after 200 frames).
        """
        logger.info(f"Re-initializing anchor with new image: {new_anchor_path}")

        # Load new anchor image
        new_anchor_image = cv2.imread(new_anchor_path)
        assert new_anchor_image is not None, f"Failed to load new anchor image at {new_anchor_path}"
        new_anchor_image = self._resize_image(new_anchor_image, self.opt.resize)
        self.anchor_image = new_anchor_image

        # Validate 2D/3D points
        assert new_2d_points.ndim == 2 and new_2d_points.shape[1] == 2, \
            f"new_2d_points must be a 2D array with shape (N, 2). Got {new_2d_points.shape}."
        assert new_3d_points.ndim == 2 and new_3d_points.shape[1] == 3, \
            f"new_3d_points must be a 2D array with shape (N, 3). Got {new_3d_points.shape}."
        assert len(new_2d_points) == len(new_3d_points), \
            "new_2d_points and new_3d_points must have the same number of elements."

        # Update anchor points
        self.anchor_keypoints_2D = np.array(new_2d_points, dtype=np.float32)
        self.anchor_keypoints_3D = np.array(new_3d_points, dtype=np.float32)

        # Recompute descriptors and index map
        self.anchor_kp, self.anchor_desc, self.idx_map = self._compute_anchor_descriptors()
        logger.info("Anchor re-initialization complete.")


    def _resize_image(self, image, resize):
        """Resize according to user opts (e.g. [640, 480] or single max dimension)."""
        if len(resize) == 2:
            return cv2.resize(image, tuple(resize))
        elif len(resize) == 1 and resize[0] > 0:
            h, w = image.shape[:2]
            scale = resize[0] / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(image, new_size)
        return image

    def _compute_anchor_descriptors(self):
        """
        1) Detect & compute ORB descriptors on anchor.
        2) Build a cKDTree for anchor_keypoints_2D (the physically measured coords).
        3) For each ORB keypoint, find the nearest measured point. If close enough,
           we map ORB keypoint i -> anchor_keypoints_3D[j].
        4) Return (anchor_kp, anchor_desc, idx_map).
        """
        anchor_gray = cv2.cvtColor(self.anchor_image, cv2.COLOR_BGR2GRAY)

        # 1) ORB detect + compute
        anchor_kp = self.orb.detect(anchor_gray, None)
        anchor_kp, anchor_desc = self.orb.compute(anchor_gray, anchor_kp)
        logger.info(f"Anchor ORB keypoints detected: {len(anchor_kp)}")

        # 2) Build cKDTree on physically measured 2D
        tree = cKDTree(self.anchor_keypoints_2D)
        dist_threshold = 5.0  # e.g. 5 px distance threshold; tune as needed
        idx_map = {}

        # 3) For each ORB keypoint, find nearest measured point
        #    If dist < dist_threshold => we store the mapping
        for i, kp in enumerate(anchor_kp):
            x_orb, y_orb = kp.pt
            dist, j = tree.query([x_orb, y_orb], k=1)
            if dist < dist_threshold:
                idx_map[i] = j

        logger.info(f"Matched {len(idx_map)} out of {len(anchor_kp)} ORB keypoints "
                    f"to physically measured 2D points within {dist_threshold}px.")
        return anchor_kp, anchor_desc, idx_map

    def process_frame(self, frame, frame_idx):
        """
        1) ORB detect + compute on frame
        2) BF-match anchor_desc -> frame_desc
        3) Build arrays for PnP (2D->3D)
        4) SolvePnP + optional refine
        5) Kalman Filter update
        6) Return (pose_data, visualization)
        """
        logger.info(f"Processing frame {frame_idx}")

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp = self.orb.detect(frame_gray, None)
        frame_kp, frame_desc = self.orb.compute(frame_gray, frame_kp)

        if frame_desc is None or len(frame_kp) < 4:
            logger.warning(f"No ORB keypoints in frame {frame_idx} or less than 4.")
            return None, frame

        matches = self.bf.match(self.anchor_desc, frame_desc)
        matches = sorted(matches, key=lambda x: x.distance)

        topN = 200
        matches = matches[:topN]
        logger.debug(f"ORB matching => {len(matches)} raw matches (frame {frame_idx})")

        # Build arrays for PnP
        pts2D = []
        pts3D = []
        for m in matches:
            # m.queryIdx => index in anchor_desc
            # m.trainIdx => index in frame_desc
            anchor_idx = m.queryIdx
            frame_idx2 = m.trainIdx

            if anchor_idx not in self.idx_map:
                # This ORB keypoint was not matched to a physically measured anchor point
                continue

            j = self.idx_map[anchor_idx]  # physically measured anchor index

            # 2D in the live frame
            x2, y2 = frame_kp[frame_idx2].pt
            pts2D.append([x2, y2])

            # 3D from anchor
            X3, Y3, Z3 = self.anchor_keypoints_3D[j]
            pts3D.append([X3, Y3, Z3])

        pts2D = np.array(pts2D, dtype=np.float32).reshape(-1,1,2)
        pts3D = np.array(pts3D, dtype=np.float32).reshape(-1,1,3)

        if len(pts2D) < 4:
            logger.warning(f"Not enough matched points for PnP (frame {frame_idx}).")
            return None, frame

        # SolvePnP
        K, distCoeffs = self._get_camera_intrinsics()
        success, rvec_o, tvec_o, inliers = cv2.solvePnPRansac(
            objectPoints=pts3D,
            imagePoints=pts2D,
            cameraMatrix=K,
            distCoeffs=distCoeffs,
            reprojectionError=8,
            confidence=0.99,
            iterationsCount=2000,
            flags=cv2.SOLVEPNP_P3P
        )

        if not success or inliers is None or len(inliers) < 7:
            logger.warning(f"PnP failed or insufficient inliers (frame {frame_idx}).")
            return None, frame

        # Optional refine
        obj_inliers = pts3D[inliers.flatten()]
        img_inliers = pts2D[inliers.flatten()]
        rvec, tvec = cv2.solvePnPRefineVVS(
            objectPoints=obj_inliers,
            imagePoints=img_inliers,
            cameraMatrix=K,
            distCoeffs=distCoeffs,
            rvec=rvec_o,
            tvec=tvec_o
        )

        R, _ = cv2.Rodrigues(rvec)

        # Kalman filter update
        pose_data = self._kalman_filter_update(R, tvec,frame_idx)

        visualization = self._draw_matches(frame, frame_kp, matches, inliers)
        return pose_data, visualization

    def _kalman_filter_update(self, R, tvec,frame_idx):
        """Example minimal KF usage."""
        trans_est, eul_est = self.kf_pose.predict()
        eul_measured = rotation_matrix_to_euler_angles(R)

        orientation_change = np.linalg.norm(eul_measured - eul_est) * (180/np.pi)
        translation_change = np.linalg.norm(tvec.flatten() - trans_est)

        # First update or threshold checks
        if self.kf_pose_first_update:
            self.kf_pose.correct(tvec, R)
            self.kf_pose_first_update = False
        else:
            # Simple thresholds
            if orientation_change < 50 and translation_change < 1.0:
                self.kf_pose.correct(tvec, R)

        # Final predicted
        trans_est, eul_est = self.kf_pose.predict()
        R_est = euler_angles_to_rotation_matrix(eul_est)

        pose_data = {
            'frame': frame_idx,
            'object_translation_in_cam': tvec.flatten().tolist(),
            'object_rotation_in_cam': R.tolist(),
            'kf_translation_vector': trans_est.tolist(),
            'kf_rotation_matrix': R_est.tolist(),
            'kf_euler_angles': eul_est.tolist()
        }
        return pose_data

    def _draw_matches(self, frame, frame_kp, matches, inliers):
        """Visualize some matches on the frame for debugging."""
        out = frame.copy()
        showN = min(50, len(matches))
        for i in range(showN):
            m = matches[i]
            pt = tuple(np.round(frame_kp[m.trainIdx].pt).astype(int))
            cv2.circle(out, pt, 5, (0,255,0), -1)
        cv2.putText(out, f"ORB matches: {len(matches)}", (30,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        return out

    def _get_camera_intrinsics(self):
        """Replace with your actual intrinsics."""
        fx, fy = 1078.0, 1081.0
        cx, cy = 628.0, 362.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)
        distCoeffs = np.array([0,0,0,0], dtype=np.float32)
        return K, distCoeffs
