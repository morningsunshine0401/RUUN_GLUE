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

        # Log initialization
        logger.info("Initializing PoseEstimator")

        # Load anchor image
        self.anchor_image = cv2.imread(opt.anchor)
        assert self.anchor_image is not None, f'Failed to load anchor image at {opt.anchor}'
        self.anchor_image = self._resize_image(self.anchor_image, opt.resize)
        logger.info(f"Loaded and resized anchor image from {opt.anchor}")

        # Initialize ONNX Session for LightGlue
        providers = [("CPUExecutionProvider", {})]
        providers.insert(0, ("CUDAExecutionProvider", {}))
        self.session = ort.InferenceSession("weights/superpoint_lightglue_pipeline.onnx", providers=providers)
        logger.info("ONNX session initialized with CUDAExecutionProvider")

        # Extract anchor keypoints
        # In the PoseEstimator class __init__ method
        self.anchor_proc = SuperPointPreprocessor.preprocess(self.anchor_image)
        self.anchor_proc = self.anchor_proc[None].astype(np.float32)  # Precompute and cache the tensor


        anchor_batch = np.concatenate([self.anchor_proc, self.anchor_proc], axis=0)
        keypoints, matches, mscores = self.session.run(None, {"images": anchor_batch})

        self.anchor_keypoints_sp = keypoints[0]
        logger.info("Extracted anchor keypoints")

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # # Known 2D and 3D correspondences (pre-defined : Opti_BOX
        # anchor_keypoints_2D = np.array([
        #     [780, 216],  
        #     [464, 111],  
        #     [258, 276], 
        #     [611, 538],  
        #     [761, 324],
        #     [644, 168],
        #     [479, 291] ,
        #     [586, 149] ,
        #     [550, 182] ,
        #     [610, 202] ,
        #     [361, 193] ,
        #     [319, 298] ,
        #     [344, 418] ,
        #     [440, 460] ,
        #     [502, 489] ,
        #     [496, 372]
        # ], dtype=np.float32)

        # anchor_keypoints_3D = np.array([
        #     [0.049, 0.045, 0],     
        #     [-0.051, 0.045, 0],    
        #     [-0.051, -0.044, 0],    
        #     [0.049, -0.044, -0.04],     
        #     [0.049, 0.045, 0],
        #     [0.01, 0.045, 0],
        #     [-0.003, -0.023, 0],
        #     [-0.001, 0.045, 0],
        #     [-0.001, 0.025, 0],
        #     [0.001, 0.025, 0],
        #     [-0.051, -0.005, 0],
        #     [-0.035, -0.044, 0],
        #     [-0.035, -0.044, -0.04],
        #     [-0.002, -0.044, -0.04], 
        #     [0.017, -0.044, -0.04],
        #     [0.017, -0.044, 0.0]
        # ], dtype=np.float32)

        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        # Known 2D and 3D correspondences (pre-defined)
        anchor_keypoints_2D = np.array([
            [563, 565], [77, 582], [515, 318], [606, 317], [612, 411],
            [515, 414], [420, 434], [420, 465], [618, 455], [500, 123],
            [418, 153], [417, 204], [417, 243], [502, 279], [585, 240],
            [289, 26], [322, 339], [349, 338], [349, 374], [321, 375],
            [390, 349], [243, 462], [367, 550], [368, 595], [383, 594],
            [386, 549], [779, 518], [783, 570]
        ], dtype=np.float32)

        # anchor_keypoints_3D = np.array([
        #     [0.03, -0.165, 0.05],
        #     [-0.190, -0.165, 0.050],
        #     [0.010, -0.025, 0.0],
        #     [0.060, -0.025, 0.0],
        #     [0.06, -0.080, 0.0],
        #     [0.010, -0.080, 0.0],
        #     [-0.035, -0.087, 0.0],
        #     [-0.035, -0.105, 0.0],
        #     [0.065, -0.105, 0.0],
        #     [0.0, 0.045, 0.0],
        #     [-0.045, 0.078, 0.0],
        #     [-0.045, 0.046, 0.0],
        #     [-0.045, 0.023, 0.0],
        #     [0.0, -0.0, 0.0],
        #     [0.045, 0.022, 0.0],
        #     [-0.120, 0.160, 0.0],
        #     [-0.095, -0.035,0.0],
        #     [-0.080, -0.035, 0.0],
        #     [-0.080, -0.055, 0.0],
        #     [-0.095, -0.055, 0.0],
        #     [-0.050, -0.040, 0.010],
        #     [-0.135, -0.100, 0.0],
        #     [-0.060, -0.155, 0.050],
        #     [-0.060, -0.175, 0.050],
        #     [-0.052, -0.175, 0.050],
        #     [-0.052, -0.155, 0.050],
        #     [0.135, -0.147, 0.050],
        #     [0.135, -0.172, 0.050]
        # ], dtype=np.float32)

        # anchor_keypoints_3D = np.array([
        #     [0.03, -0.165, 0.05],
        #     [-0.190, -0.165, 0.050],
        #     [0.010, -0.025, 0.0],
        #     [0.060, -0.025, 0.0],
        #     [0.06, -0.080, 0.0],
        #     [0.010, -0.080, 0.0],
        #     [-0.035, -0.087, 0.0],
        #     [-0.035, -0.105, 0.0],
        #     [0.065, -0.105, 0.0],
        #     [0.0, 0.045, 0.0],
        #     [-0.045, 0.078, 0.0],
        #     [-0.045, 0.046, 0.0],
        #     [-0.045, 0.023, 0.0],
        #     [0.0, -0.0, 0.0],
        #     [0.045, 0.022, 0.0],
        #     [-0.120, 0.160, 0.0],
        #     [-0.095, -0.035,0.0],
        #     [-0.080, -0.035, 0.0],
        #     [-0.080, -0.055, 0.0],
        #     [-0.095, -0.055, 0.0],
        #     [-0.050, -0.040, 0.010],
        #     [-0.135, -0.100, 0.0],
        #     [-0.060, -0.155, 0.050],
        #     [-0.060, -0.175, 0.050],
        #     [-0.052, -0.175, 0.050],
        #     [-0.052, -0.155, 0.050],
        #     [0.135, -0.147, 0.050],
        #     [0.135, -0.172, 0.050]
        # ], dtype=np.float32)

        # R_anchor_to_cv = np.array([
        #     [1, 0, 0],
        #     [0, -1, 0],
        #     [0, 0, -1]
        # ], dtype=np.float32)
        
        # anchor_keypoints_3D = anchor_keypoints_3D_raw @ R_anchor_to_cv.T


        # This is a test I changed it into GT coordinate frame 20241218 #########################################
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
            [-0.045, 0.0,0.078 ],
            [-0.045, 0.0, 0.046],
            [-0.045, 0.0, 0.023],
            [0.0, -0.0, 0.0],
            [0.045, 0.0, 0.022],
            [-0.120, 0.0, 0.160],
            [-0.095, -0.0,-0.035],
            [-0.080, -0.0, -0.035],
            [-0.080, -0.0, -0.055],
            [-0.095, -0.0, -0.055],
            [-0.050, -0.010, -0.040],
            [-0.135, -0.0, -0.1],
            [-0.060, -0.050,-0.155],
            [-0.060, -0.050,-0.175],
            [-0.052, -0.050, -0.175],
            [-0.052, -0.050, -0.155],
            [0.135, -0.050, -0.147],
            [0.135, -0.050, -0.172]
        ], dtype=np.float32)

        # # Define the transformation matrix
        # T = np.array([
        #     [1, 0, 0],
        #     [0, 0, 1],
        #     [0, -1, 0]
        # ], dtype=np.float32)

        # # Transform the points
        # anchor_keypoints_3D = (T @ anchor_keypoints_3D_raw.T).T
        ########################################################################
        

        # Build KDTree on anchor_keypoints_sp
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

        anchor_proc = self.anchor_proc
        anchor_proc_time = time.time()

        frame_proc = SuperPointPreprocessor.preprocess(frame)
        frame_proc = frame_proc[None].astype(np.float32)
        frame_proc_time = time.time()

        batch = np.concatenate([anchor_proc, frame_proc], axis=0).astype(np.float32)
        batch = torch.tensor(batch, device=self.device).cpu().numpy()
        #print("Batch!!!!!!!!!!@@@@@@@@@@@@@:\n",batch)
        batch_time = time.time()

        keypoints, matches, mscores = self.session.run(None, {"images": batch})
        keypoint_time = time.time()

        logger.info(
        f"Frame {frame_idx} processing times: Resize: {resize_time - start_time:.3f}s, "
        f": Anchor_proc: {anchor_proc_time - resize_time:.3f}s, "
        f": Frame_proc: {frame_proc_time - anchor_proc_time:.3f}s, "
        f": Batch: {batch_time - frame_proc_time:.3f}s, "
        f"Keypoint Extraction: {keypoint_time - batch_time:.3f}s"
        )

        valid_mask = (matches[:, 0] == 0)
        valid_matches = matches[valid_mask]

        mkpts0 = keypoints[0][valid_matches[:, 1]]
        mkpts1 = keypoints[1][valid_matches[:, 2]]
        mconf = mscores[valid_mask]

        logger.debug(f"Found {len(mkpts0)} matches in frame {frame_idx}")

        anchor_indices = valid_matches[:, 1]
        known_mask = np.isin(anchor_indices, self.matched_anchor_indices)
        mkpts0 = mkpts0[known_mask]
        mkpts1 = mkpts1[known_mask]
        mconf = mconf[known_mask]

        idx_map = {idx: i for i, idx in enumerate(self.matched_anchor_indices)}
        mpts3D = np.array([self.matched_3D_keypoints[idx_map[aidx]] for aidx in anchor_indices[known_mask]])

        if len(mkpts0) < 4:
            logger.warning(f"Not enough matches to compute pose for frame {frame_idx}")
            return None, frame

        pose_data, visualization = self.estimate_pose(
            mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, keypoints[1])
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
        imagePoints = mkpts1.reshape(-1, 1, 2)
        imagePoints = imagePoints.astype(np.float32)

        ################################################################################################
        # #@@@@@@@@@@@@@ NEEDs UPGRADE OR FOR LATER USAGE@@@@@@@@@@@@@@@@@@@@@@@@@ 
        # # Use Kalman Filter to adaptively set reprojectionError threshold (Needs further testing and upgrade)
        # translation_estimated, eulers_estimated = self.kf_pose.predict()

        # # Convert Kalman Filter predicted Euler angles to rotation matrix
        # R_predicted = euler_angles_to_rotation_matrix(eulers_estimated)

        # # Calculate rvec from predicted rotation matrix
        # rvec_predicted, _ = cv2.Rodrigues(R_predicted)

        # # Compare predicted rvec with initial or current rvec_o (from solvePnPRansac)
        # if 'rvec_o' in locals():  # Ensure rvec_o is defined
        #     rvec_error = np.linalg.norm(rvec_predicted.flatten() - rvec_o.flatten())
        # else:
        #     rvec_error = float('inf')  # Default to a large error if rvec_o is unavailable

        # translation_change = np.linalg.norm(translation_estimated - objectPoints.mean(axis=0).flatten())

        # if translation_change < 2.0:  # Low translation change
        #     reprojectionError = 1.0  # Strict threshold
        # elif translation_change < 5.0:
        #     reprojectionError = 3.0  # Moderate threshold
        # else:
        #     reprojectionError = 5.0  # Flexible threshold

        # if translation_change < 3 and rvec_error < 0.05:
        #     confidence = 0.99  # High confidence
        # else:
        #     confidence = 0.8   # Lower confidence to allow more matches

        # if translation_change < 3 and rvec_error < 0.05:
        #     iterationsCount = 1000  # Lower iterations for efficiency
        # else:
        #     iterationsCount = 1500  # Higher iterations for robustness
        ################################################################################################

        success, rvec_o, tvec_o, inliers = cv2.solvePnPRansac(
            objectPoints=objectPoints,
            imagePoints=imagePoints,
            cameraMatrix=K,
            distCoeffs=distCoeffs,
            reprojectionError=8,#reprojectionError,
            confidence=0.99,#confidence,
            iterationsCount=2000,#iterationsCount,
            flags=cv2.SOLVEPNP_P3P
        )

        if success and inliers is not None and len(inliers) >= 7:#3:
            objectPoints_inliers = mpts3D[inliers.flatten()].reshape(-1, 1, 3)
            imagePoints_inliers = mkpts1[inliers.flatten()].reshape(-1, 1, 2)
            imagePoints_inliers = imagePoints_inliers.astype(np.float32)

            rvec, tvec = cv2.solvePnPRefineVVS(
                objectPoints=objectPoints_inliers,
                imagePoints=imagePoints_inliers,
                cameraMatrix=K,
                distCoeffs=distCoeffs,
                rvec=rvec_o,
                tvec=tvec_o
            )

            R, _ = cv2.Rodrigues(rvec)
            camera_position = -R.T @ tvec

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

            pose_data = self._kalman_filter_update(
                R, tvec, reprojection_errors, mean_reprojection_error, std_reprojection_error,
                inliers, mkpts0, mkpts1, mpts3D, mconf, frame_idx, camera_position
            )

            visualization = self._visualize_matches(
                frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints
            )
            return pose_data, visualization
        else:
            logger.warning("PnP pose estimation failed.")
            return None, frame


    def _kalman_filter_update(self, R, tvec, reprojection_errors, mean_reprojection_error,
                              std_reprojection_error, inliers, mkpts0, mkpts1, mpts3D,
                              mconf, frame_idx, camera_position):
        num_inliers = len(inliers)
        inlier_ratio = num_inliers / len(mkpts0) if len(mkpts0) > 0 else 0

        reprojection_error_threshold = 10#15
        max_translation_jump = 4
        min_inlier = 5

        translation_estimated, eulers_estimated = self.kf_pose.predict()
        eulers_measured = rotation_matrix_to_euler_angles(R)

        translation_change = np.linalg.norm(tvec.flatten() - translation_estimated)

        ###############################################################################################
        # Apply ground truth Z for the first frame only
        if not self.initial_z_set:
            # tvec[0] = 0.0
            # tvec[1] = -1.0
            # tvec[2] = 0.0
            self.initial_z_set = True
        ###############################################################################################
        if mean_reprojection_error < reprojection_error_threshold and num_inliers > min_inlier:
            print("*****************************************.")
            if translation_change < max_translation_jump:
                self.kf_pose.correct(tvec, R)
                print("Correct ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            else:
                print("Skipping Kalman update due to large jump in translation $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4.")
        else:
            print("Skipping Kalman update due to high reprojection error.^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        translation_estimated, eulers_estimated = self.kf_pose.predict()
        R_estimated = euler_angles_to_rotation_matrix(eulers_estimated)
        
        #translation_estimated = -R_estimated.T @ translation_estimated

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

        position_text = f"Position: {pose_data['camera_position']}"
        cv2.putText(out, position_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2, cv2.LINE_AA)
        return out

    def _get_camera_intrinsics(self):
        # Replace with your camera's intrinsic parameters
        focal_length_x = 1195.08491
        focal_length_y = 1354.35538
        cx = 581.022033
        cy = 571.458522

        # cx = 620.022033
        # cy = 500.458522

        distCoeffs = np.array([0.10058526, 0.4507094, 0.13687279, -0.01839536, 0.13001669], dtype=np.float32)
        

        K = np.array([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return K, distCoeffs
