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

from KF_Q import KalmanFilterPose
import matplotlib.cm as cm
from models.utils import make_matching_plot_fast
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pose_estimator_tracking.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the SuperPointPreprocessor
from superpoint_LG import SuperPointPreprocessor

class PoseEstimatorWithTracking:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.initial_z_set = False
        self.kf_initialized = False
        
        # Tracking-related variables
        self.prev_frame = None
        self.active_keypoints = None
        self.active_3D_points = None
        self.tracking_initialized = False

        self.dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        
        # SuperPoint tracking inspired members
        self.max_track_length = 5  # Maximum track history to keep
        self.tracks = []  # Track history
        self.track_ages = []  # Age of each track
        self.track_ids = []  # Unique IDs for each track
        self.next_track_id = 0  # Counter for assigning track IDs
        
        # Tracking parameters
        self.min_tracked_points = opt.min_tracked_points if hasattr(opt, 'min_tracked_points') else 4#7
        self.max_tracking_error = opt.max_tracking_error if hasattr(opt, 'max_tracking_error') else 8#5.0
        self.min_track_len = 2  # Minimum track length to consider for pose estimation
        
        # LK Parameters for optical flow
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            minEigThreshold=0.001
        )

        logger.info("Initializing PoseEstimatorWithTracking")

        # Load anchor image
        self.anchor_image = cv2.imread(opt.anchor)
        assert self.anchor_image is not None, f'Failed to load anchor image at {opt.anchor}'
        self.anchor_image = self._resize_image(self.anchor_image, opt.resize)
        logger.info(f"Loaded and resized anchor image from {opt.anchor}")

        # Initialize ONNX session for SuperPoint + LightGlue
        providers = [
            ("CUDAExecutionProvider", {}),
            ("CPUExecutionProvider", {})
        ]
        self.session = ort.InferenceSession(
            "weights/superpoint_lightglue_pipeline_1280x720.onnx",
            providers=providers
        )
        logger.info("ONNX session initialized with CUDAExecutionProvider")

        # Define anchor keypoints (2D -> 3D correspondence)
        anchor_keypoints_2D = np.array([
            [511, 293], [591, 284], [587, 330], [413, 249], [602, 348],
            [715, 384], [598, 298], [656, 171], [805, 213], [703, 392],
            [523, 286], [519, 327], [387, 289], [727, 126], [425, 243],
            [636, 358], [745, 202], [595, 388], [436, 260], [539, 313],
            [795, 220], [351, 291], [665, 165], [611, 353], [650, 377],
            [516, 389], [727, 143], [496, 378], [575, 312], [617, 368],
            [430, 312], [480, 281], [834, 225], [469, 339], [705, 223],
            [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]
        ], dtype=np.float32)

        anchor_keypoints_3D = np.array([
            [-0.014,  0.000,  0.042], [ 0.025, -0.014, -0.011], 
            [-0.014,  0.000, -0.042], [-0.014,  0.000,  0.156], 
            [-0.023,  0.000, -0.065], [ 0.000,  0.000, -0.156], 
            [ 0.025,  0.000, -0.015], [ 0.217,  0.000,  0.070], 
            [ 0.230,  0.000, -0.070], [-0.014,  0.000, -0.156], 
            [ 0.000,  0.000,  0.042], [-0.057, -0.018, -0.010], 
            [-0.074, -0.000,  0.128], [ 0.206, -0.070, -0.002], 
            [-0.000, -0.000,  0.156], [-0.017, -0.000, -0.092], 
            [ 0.217, -0.000, -0.027], [-0.052, -0.000, -0.097], 
            [-0.019, -0.000,  0.128], [-0.035, -0.018, -0.010], 
            [ 0.217, -0.000, -0.070], [-0.080, -0.000,  0.156], 
            [ 0.230, -0.000,  0.070], [-0.023, -0.000, -0.075], 
            [-0.029, -0.000, -0.127], [-0.090, -0.000, -0.042], 
            [ 0.206, -0.055, -0.002], [-0.090, -0.000, -0.015], 
            [ 0.000, -0.000, -0.015], [-0.037, -0.000, -0.097], 
            [-0.074, -0.000,  0.074], [-0.019, -0.000,  0.074], 
            [ 0.230, -0.000, -0.113], [-0.100, -0.030,  0.000], 
            [ 0.170, -0.000, -0.015], [ 0.230, -0.000,  0.113], 
            [-0.000, -0.025, -0.240], [-0.000, -0.025,  0.240], 
            [ 0.243, -0.104,  0.000], [-0.080, -0.000, -0.156]
        ], dtype=np.float32)

        # Set anchor features
        self._set_anchor_features(
            anchor_bgr_image=self.anchor_image,
            anchor_keypoints_2D=anchor_keypoints_2D,
            anchor_keypoints_3D=anchor_keypoints_3D
        )

        # Initialize Kalman filter
        self.anchor_viewpoint_eulers = np.array([0.0, -0.35, 0.0], dtype=np.float32)
        self.kf_pose = self._init_kalman_filter()
        self.kf_pose_first_update = True 
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

        # 4. Reset tracking state
        self.tracking_initialized = False
        self.active_keypoints = None
        self.active_3D_points = None
        self.prev_frame = None
        self.tracks = []
        self.track_ages = []
        self.track_ids = []

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

    # def process_frame(self, frame, frame_idx):
    #     """
    #     Process a new frame for pose estimation.
    #     Either initialize by matching to anchor, track previous features,
    #     or fall back to Kalman prediction when both fail.
    #     """
    #     logger.info(f"Processing frame {frame_idx}")
    #     start_time = time.time()

    #     # Resize frame to target size
    #     frame = self._resize_image(frame, self.opt.resize)
        
    #     # Convert to grayscale for tracking
    #     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    #     # Get a Kalman prediction regardless of what happens next (for fallback)
    #     x_pred, P_pred = self.kf_pose.predict()
    #     kf_translation = x_pred[0:3]
    #     kf_quaternion = x_pred[6:10]
    #     kf_rotation = quaternion_to_rotation_matrix(kf_quaternion)
        
    #     # Store the Kalman prediction as a fallback
    #     fallback_pose_data = {
    #         'frame': frame_idx,
    #         'kf_translation_vector': kf_translation.tolist(),
    #         'kf_quaternion': kf_quaternion.tolist(),
    #         'kf_rotation_matrix': kf_rotation.tolist(),
    #         'pose_estimation_failed': True,
    #         'estimation_method': 'kalman_prediction_only'
    #     }
        
    #     # Create a simple visualization for fallback case
    #     fallback_vis = frame.copy()
    #     position_text = (f"Leader in Cam (KF prediction): "
    #                     f"x={kf_translation[0]:.3f}, y={kf_translation[1]:.3f}, z={kf_translation[2]:.3f}")
    #     cv2.putText(fallback_vis, position_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    #     cv2.putText(fallback_vis, "KALMAN PREDICTION ONLY", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
    #                0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
    #     # Check if we need to initialize or re-initialize the tracking
    #     if not self.tracking_initialized or self.active_keypoints is None or len(self.active_keypoints) < self.min_tracked_points:
    #         # Initialize/re-initialize using SuperPoint + LightGlue
    #         logger.info(f"Initializing/re-initializing tracking for frame {frame_idx}")
    #         pose_data, visualization = self._initialize_tracking(frame, frame_idx)
            
    #         # If initialization failed, use Kalman prediction
    #         if pose_data is None:
    #             logger.warning(f"Initialization failed for frame {frame_idx}, using Kalman prediction")
    #             pose_data = fallback_pose_data
    #             visualization = fallback_vis
    #         else:
    #             pose_data['estimation_method'] = 'initialization'
                
    #         # Store the current frame for future tracking
    #         self.prev_frame = frame_gray.copy()
            
    #         return pose_data, visualization
    #     #################################################################################################
    #     else:
    #         if frame_idx % 10 == 0:
    #             logger.info(f"Reinforcing tracking for frame {frame_idx}")
    #             self._reinforce_tracking(frame, frame_idx)
    #     ############################################################################################################
    #     # If we're here, we're in tracking mode - try to track features
    #     pose_data, visualization = self._track_features(frame, frame_gray, frame_idx)
        
    #     # If tracking failed, use Kalman prediction
    #     if pose_data is None:
    #         logger.warning(f"Tracking failed for frame {frame_idx}, using Kalman prediction")
    #         pose_data = fallback_pose_data
    #         visualization = fallback_vis
    #     else:
    #         pose_data['estimation_method'] = 'tracking'
        
    #     # Store the current frame for future tracking
    #     self.prev_frame = frame_gray.copy()
        
    #     return pose_data, visualization

    #### With thresholds
    def process_frame(self, frame, frame_idx):
        logger.info(f"Processing frame {frame_idx}")
        start_time = time.time()

        # Resize frame and convert to grayscale for tracking.
        frame = self._resize_image(frame, self.opt.resize)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get a Kalman prediction for fallback.
        x_pred, P_pred = self.kf_pose.predict()
        kf_translation = x_pred[0:3]
        kf_quaternion = x_pred[6:10]
        kf_rotation = quaternion_to_rotation_matrix(kf_quaternion)
        fallback_pose_data = {
            'frame': frame_idx,
            'kf_translation_vector': kf_translation.tolist(),
            'kf_quaternion': kf_quaternion.tolist(),
            'kf_rotation_matrix': kf_rotation.tolist(),
            'pose_estimation_failed': True,
            'estimation_method': 'kalman_prediction_only'
        }
        fallback_vis = frame.copy()
        cv2.putText(fallback_vis, f"KF: x={kf_translation[0]:.3f}, y={kf_translation[1]:.3f}, z={kf_translation[2]:.3f}",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(fallback_vis, "KALMAN PREDICTION ONLY", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Check if we need to initialize/reinitialize tracking:
        if not self.tracking_initialized or self.active_keypoints is None or len(self.active_keypoints) < 5:
            logger.info(f"Initializing/re-initializing tracking for frame {frame_idx}")
            pose_data, visualization = self._initialize_tracking(frame, frame_idx)
            if pose_data is None:
                logger.warning(f"Initialization failed for frame {frame_idx}, using Kalman prediction")
                pose_data, visualization = fallback_pose_data, fallback_vis
            else:
                pose_data['estimation_method'] = 'initialization'
            self.prev_frame = frame_gray.copy()
            return pose_data, visualization

        # Otherwise, attempt to track features using DIS optical flow.
        pose_data, visualization = self._track_features(frame, frame_gray, frame_idx)

        # First, check if _track_features failed.
        if pose_data is None:
            logger.warning(f"Tracking failed for frame {frame_idx}, reinitializing tracking")
            pose_data, visualization = self._initialize_tracking(frame, frame_idx)
            if pose_data is None:
                logger.warning(f"Reinitialization failed for frame {frame_idx}, using Kalman prediction")
                pose_data, visualization = fallback_pose_data, fallback_vis
            else:
                pose_data['estimation_method'] = 'initialization'
        # If tracking produced data, check quality.
        elif (pose_data.get("coverage_score", 0) < 0.4 or 
            pose_data.get("num_tracked_points", 0) < 5):
            logger.warning(f"Tracking quality insufficient at frame {frame_idx} (coverage: {pose_data.get('coverage_score', 0):.2f}, "
                        f"features: {pose_data.get('num_tracked_points', 0)}). Reinitializing tracking.")
            pose_data, visualization = self._initialize_tracking(frame, frame_idx)
            if pose_data is None:
                logger.warning(f"Reinitialization failed for frame {frame_idx}, using Kalman prediction")
                pose_data, visualization = fallback_pose_data, fallback_vis
            else:
                pose_data['estimation_method'] = 'initialization'

        self.prev_frame = frame_gray.copy()
        return pose_data, visualization



    def _initialize_tracking(self, frame, frame_idx):
        """
        Initialize or re-initialize tracking by matching current frame to anchor
        """
        logger.info(f"Performing full matching for frame {frame_idx}")
        
        # Process with SuperPoint + LightGlue
        anchor_proc = self.anchor_proc  # Precomputed anchor
        frame_proc = SuperPointPreprocessor.preprocess(frame)
        frame_proc = frame_proc[None].astype(np.float32)
        
        batch = np.concatenate([anchor_proc, frame_proc], axis=0).astype(np.float32)
        batch = torch.tensor(batch, device=self.device).cpu().numpy()
        
        keypoints, matches, mscores = self.session.run(None, {"images": batch})
        
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
        
        if len(mkpts0) < 8:  # Need at least 8 points for good initialization
            logger.warning(f"Not enough matches for initialization (frame {frame_idx})")
            return None, frame
        
        # Reset track data
        self.tracks = []
        self.track_ages = []
        self.track_ids = []
        
        # Initialize active keypoints and corresponding 3D points
        self.active_keypoints = mkpts1.copy()
        self.active_3D_points = mpts3D.copy()
        self.tracking_initialized = True
        
        # Initialize tracks
        for i in range(len(mkpts1)):
            self.tracks.append([mkpts1[i]])
            self.track_ages.append(1)
            self.track_ids.append(self.next_track_id)
            self.next_track_id += 1
        
        logger.info(f"Tracking initialized with {len(self.active_keypoints)} keypoints")
        
        # Estimate pose using the current set of matches
        pose_data, visualization = self.estimate_pose(
            mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, keypoints[1]
        )
        
        return pose_data, visualization

    ## This one uused optical flow for tracking
    # def _track_features(self, frame, frame_gray, frame_idx):
    #     """
    #     Track features from previous frame to current frame using optical flow
    #     and maintain track history
    #     """
    #     logger.info(f"Tracking features for frame {frame_idx}")
        
    #     if self.prev_frame is None or self.active_keypoints is None or len(self.active_keypoints) == 0:
    #         logger.warning("Cannot track: No previous frame or active keypoints")
    #         return None, frame
        
    #     # Convert active keypoints to proper format for tracking
    #     prev_pts = self.active_keypoints.reshape(-1, 1, 2).astype(np.float32)
        
    #     # Calculate optical flow
    #     curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
    #         self.prev_frame, frame_gray, prev_pts, None, **self.lk_params
    #     )
        
    #     # Filter out points that couldn't be tracked
    #     status = status.flatten().astype(bool)
    #     err = err.flatten()
        
    #     # Additional filtering: remove points with high error
    #     valid_mask = status & (err < self.max_tracking_error)
        
    #     # Update track histories
    #     tracked_keypoints = []
    #     tracked_3D_points = []
    #     updated_tracks = []
    #     updated_track_ages = []
    #     updated_track_ids = []
        
    #     for i, valid in enumerate(valid_mask):
    #         if valid:
    #             # Add current point to the track
    #             curr_pt = curr_pts[i, 0]
    #             tracked_keypoints.append(curr_pt)
    #             tracked_3D_points.append(self.active_3D_points[i])
                
    #             # Update track history
    #             if len(self.tracks[i]) >= self.max_track_length:
    #                 # Remove oldest point to maintain max length
    #                 track_history = self.tracks[i][1:] + [curr_pt]
    #             else:
    #                 track_history = self.tracks[i] + [curr_pt]
                
    #             updated_tracks.append(track_history)
    #             updated_track_ages.append(self.track_ages[i] + 1)
    #             updated_track_ids.append(self.track_ids[i])
        
    #     # Update tracking state
    #     if len(tracked_keypoints) < self.min_tracked_points:
    #         logger.warning(f"Too few tracked points ({len(tracked_keypoints)}), need re-initialization")
    #         self.tracking_initialized = False
    #         return None, frame
        
    #     # Update tracking state variables
    #     self.active_keypoints = np.array(tracked_keypoints)
    #     self.active_3D_points = np.array(tracked_3D_points)
    #     self.tracks = updated_tracks
    #     self.track_ages = updated_track_ages
    #     self.track_ids = updated_track_ids
        
    #     logger.info(f"Successfully tracked {len(tracked_keypoints)}/{len(self.tracks)} points")
        
    #     # For visualization and PnP, create necessary arrays
    #     dummy_mkpts0 = np.zeros_like(self.active_keypoints)  # We don't have actual anchor correspondences here
    #     dummy_mconf = np.ones(len(self.active_keypoints))  # Set all confidences to 1.0
        
    #     # Estimate pose using the tracked keypoints
    #     pose_data, visualization = self.estimate_pose_from_tracked(
    #         self.active_keypoints, self.active_3D_points, dummy_mconf, frame, frame_idx, 
    #         np.array(self.track_ages)
    #     )
        
    #     return pose_data, visualization

#######################################################################

    # ## This one used SuperPoint

    # def _track_features(self, frame, frame_gray,frame_idx):
    #     """
    #     Track features using descriptor matching instead of optical flow
    #     """
    #     logger.info(f"Tracking features using descriptors for frame {frame_idx}")
        
    #     if self.active_keypoints is None or len(self.active_keypoints) == 0:
    #         logger.warning("Cannot track: No active keypoints")
    #         return None, frame
        
    #     # Extract SuperPoint features for current frame
    #     frame_proc = SuperPointPreprocessor.preprocess(frame)
    #     frame_proc = frame_proc[None].astype(np.float32)
        
    #     # Run SuperPoint on the current frame
    #     batch = np.concatenate([frame_proc, frame_proc], axis=0)
    #     keypoints, _, mscores = self.session.run(None, {"images": batch})
        
    #     frame_keypoints = keypoints[0]  # N x 2
        
    #     # Create batches for descriptor matching
    #     frame_proc_1 = frame_proc.copy()
    #     batch = np.concatenate([frame_proc, frame_proc_1], axis=0)
    #     keypoints, matches, mscores = self.session.run(None, {"images": batch})
        
    #     # Match descriptors between previous points and current frame
    #     tracked_keypoints = []
    #     tracked_3D_points = []
    #     updated_tracks = []
    #     updated_track_ages = []
    #     updated_track_ids = []
        
    #     # Use KDTree for efficient nearest neighbor search
    #     if len(frame_keypoints) > 0:
    #         curr_tree = cKDTree(frame_keypoints)
            
    #         # For each active keypoint, find the closest matching keypoint in current frame
    #         for i, (prev_pt, prev_3d) in enumerate(zip(self.active_keypoints, self.active_3D_points)):
    #             # Find nearest keypoint in current frame
    #             distances, indices = curr_tree.query(prev_pt, k=1)
                
    #             # If close enough, consider it a match
    #             if distances < 5.0:#15.0:  # Distance threshold in pixels
    #                 closest_idx = indices
    #                 curr_pt = frame_keypoints[closest_idx]
                    
    #                 tracked_keypoints.append(curr_pt)
    #                 tracked_3D_points.append(prev_3d)
                    
    #                 # Update track history
    #                 if len(self.tracks[i]) >= self.max_track_length:
    #                     # Remove oldest point to maintain max length
    #                     track_history = self.tracks[i][1:] + [curr_pt]
    #                 else:
    #                     track_history = self.tracks[i] + [curr_pt]
                    
    #                 updated_tracks.append(track_history)
    #                 updated_track_ages.append(self.track_ages[i] + 1)
    #                 updated_track_ids.append(self.track_ids[i])
        
    #     # Update tracking state
    #     if len(tracked_keypoints) < self.min_tracked_points:
    #         logger.warning(f"Too few tracked points ({len(tracked_keypoints)}), need re-initialization")
    #         self.tracking_initialized = False
    #         return None, frame
        
    #     # Update tracking state variables
    #     self.active_keypoints = np.array(tracked_keypoints)
    #     self.active_3D_points = np.array(tracked_3D_points)
    #     self.tracks = updated_tracks
    #     self.track_ages = updated_track_ages
    #     self.track_ids = updated_track_ids
        
    #     logger.info(f"Successfully tracked {len(tracked_keypoints)} points using descriptors")
        
    #     # For PnP, create necessary arrays
    #     dummy_mkpts0 = np.zeros_like(self.active_keypoints)
    #     dummy_mconf = np.ones(len(self.active_keypoints))
        
    #     # Estimate pose using the tracked keypoints
    #     pose_data, visualization = self.estimate_pose_from_tracked(
    #         self.active_keypoints, self.active_3D_points, dummy_mconf, frame, frame_idx,
    #         np.array(self.track_ages)
    #     )
        
    #     return pose_data, visualization


#################################################################

    ## This one uses openCV DIS optical flow

    def _track_features(self, frame, frame_gray,frame_idx):
        """
        Track features using descriptor matching instead of optical flow
        """
        logger.info(f"Tracking features using descriptors for frame {frame_idx}")
        if self.prev_frame is None or self.active_keypoints is None or len(self.active_keypoints) == 0:
            logger.warning("Cannot track: No previous frame or active keypoints")
            return None, frame

        # Calculate dense optical flow using DIS between the previous frame and current frame
        flow = self.dis_flow.calc(self.prev_frame, frame_gray, None)

        tracked_keypoints = []
        tracked_3D_points = []
        updated_tracks = []
        updated_track_ages = []
        updated_track_ids = []

        h, w = flow.shape[:2]
        for i, pt in enumerate(self.active_keypoints):
            x, y = pt
            # Ensure the keypoint is within image bounds
            ix, iy = int(round(x)), int(round(y))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                continue
            # Get the displacement from the dense flow
            displacement = flow[iy, ix]
            new_pt = np.array([pt[0] + displacement[0], pt[1] + displacement[1]])
            # Optional: Filter out large displacements if they exceed the threshold
            if np.linalg.norm(displacement) > self.max_tracking_error:
                continue

            tracked_keypoints.append(new_pt)
            tracked_3D_points.append(self.active_3D_points[i])
            
            # Update track history
            if len(self.tracks[i]) >= self.max_track_length:
                track_history = self.tracks[i][1:] + [new_pt]
            else:
                track_history = self.tracks[i] + [new_pt]
            updated_tracks.append(track_history)
            updated_track_ages.append(self.track_ages[i] + 1)
            updated_track_ids.append(self.track_ids[i])

        if len(tracked_keypoints) < self.min_tracked_points:
            logger.warning(f"Too few tracked points ({len(tracked_keypoints)}), need re-initialization")
            self.tracking_initialized = False
            return None, frame

        # Update tracking state with the new positions
        self.active_keypoints = np.array(tracked_keypoints)
        self.active_3D_points = np.array(tracked_3D_points)
        self.tracks = updated_tracks
        self.track_ages = updated_track_ages
        self.track_ids = updated_track_ids

        logger.info(f"Successfully tracked {len(tracked_keypoints)} points using DIS optical flow")

        # Use the tracked keypoints for pose estimation
        dummy_mkpts0 = np.zeros_like(self.active_keypoints)
        dummy_mconf = np.ones(len(self.active_keypoints))
        pose_data, visualization = self.estimate_pose_from_tracked(
            self.active_keypoints, self.active_3D_points, dummy_mconf, frame, frame_idx,
            np.array(self.track_ages)
        )

        return pose_data, visualization

    def estimate_pose_from_tracked(self, tracked_pts, pts3D, mconf, frame, frame_idx, point_ages):
        """
        Estimate pose using tracked points (instead of matches)
        """
        logger.debug(f"Estimating pose from tracked points for frame {frame_idx}")
        K, distCoeffs = self._get_camera_intrinsics()

        objectPoints = pts3D.reshape(-1, 1, 3)
        imagePoints = tracked_pts.reshape(-1, 1, 2).astype(np.float32)

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

        # Get rotation matrix
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
            # Calculate entropy term
            entropy_sum = 0
            for count in regions.values():
                if count > 0:
                    proportion = count / total_points
                    entropy_sum += proportion * np.log(proportion)
            
            # Normalize by log(4) as specified in the paper
            normalized_entropy = -entropy_sum / np.log(4)
            
            # Final coverage score
            coverage_score = normalized_entropy
            
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

        # Update pose_data with additional region and coverage details
        pose_data = self._kalman_filter_update(
            R, tvec, reprojection_errors, mean_reprojection_error,
            std_reprojection_error, inliers, tracked_pts, tracked_pts, pts3D,
            mconf, frame_idx, rvec_o, rvec, coverage_score=coverage_score
        )
        
        # Add tracking-specific information
        pose_data["region_distribution"] = regions
        pose_data["coverage_score"] = coverage_score
        pose_data["tracking_mode"] = True
        pose_data["num_tracked_points"] = len(tracked_pts)
        pose_data["max_point_age"] = int(np.max(point_ages)) if len(point_ages) > 0 else 0
        pose_data["avg_point_age"] = float(np.mean(point_ages)) if len(point_ages) > 0 else 0
        pose_data["inlier_percentage"] = 100.0 * len(inliers) / len(tracked_pts) if len(tracked_pts) > 0 else 0

        # Create visualization with tracked points
        visualization = self._visualize_tracked_points(
            frame, inliers, tracked_pts, point_ages, pose_data
        )
        return pose_data, visualization
    
    def _init_kalman_filter(self):
         frame_rate = 30
         dt = 1 / frame_rate
         kf_pose = KalmanFilterPose(dt)
         return kf_pose

    def _visualize_tracked_points(self, frame, inliers, tracked_pts, point_ages, pose_data):
        """
        Visualize tracked points with age-based coloring
        """
        # Convert frame to show annotations
        vis_frame = frame.copy()
        
        # Determine color based on point age
        max_age = max(point_ages) if len(point_ages) > 0 else 1
        
        # Draw all tracked points
        inlier_indices = set(inliers.flatten()) if inliers is not None else set()
        
        for i, (pt, age) in enumerate(zip(tracked_pts, point_ages)):
            # Normalize age for color (older points = bluer)
            normalized_age = age / max_age
            
            # Create color based on point age (red->green->blue)
            if normalized_age < 0.5:
                # Red to green
                r = 255 * (1 - 2 * normalized_age)
                g = 255 * (2 * normalized_age)
                b = 0
            else:
                # Green to blue
                r = 0
                g = 255 * (2 - 2 * normalized_age)
                b = 255 * (2 * normalized_age - 1)
            
            color = (int(b), int(g), int(r))  # BGR format for OpenCV
            
            # Draw circle for the point
            cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 2, color, -1)
            
            # If this is an inlier, draw a larger circle around it
            if i in inlier_indices:
                cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 5, color, 1)
                
                # Draw small track history if available
                try:
                    track = self.tracks[i]
                    if len(track) > 1:
                        for j in range(len(track) - 1):
                            pt1 = (int(track[j][0]), int(track[j][1]))
                            pt2 = (int(track[j+1][0]), int(track[j+1][1]))
                            cv2.line(vis_frame, pt1, pt2, color, 1)
                except (IndexError, AttributeError):
                    pass  # Skip if track history isn't available
                    
        
        # Draw pose information
        t_in_cam = pose_data['object_translation_in_cam']
        position_text = (f"Raw: x={t_in_cam[0]:.3f}, y={t_in_cam[1]:.3f}, z={t_in_cam[2]:.3f}")
        cv2.putText(vis_frame, position_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2, cv2.LINE_AA)
                    
        kf_t = pose_data['kf_translation_vector']
        kf_text = (f"KF: x={kf_t[0]:.3f}, y={kf_t[1]:.3f}, z={kf_t[2]:.3f}")
        cv2.putText(vis_frame, kf_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        tracking_text = f"Tracking mode: {len(tracked_pts)} points, max age: {int(max_age)}"
        cv2.putText(vis_frame, tracking_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Show inlier information
        inlier_text = f"Inliers: {len(inlier_indices)}/{len(tracked_pts)} ({pose_data['inlier_percentage']:.1f}%)"
        cv2.putText(vis_frame, inlier_text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
        # Show reprojection error
        error_text = f"Reprojection error: {pose_data['mean_reprojection_error']:.2f} px"
        cv2.putText(vis_frame, error_text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        return vis_frame
    
    def _get_camera_intrinsics(self):
        """
        Returns camera intrinsics matrix K and distortion coefficients.
        Replace with your actual camera parameters.
        """
        # Camera intrinsics from the original code
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

    def _kalman_filter_update(
        self, R, tvec, reprojection_errors, mean_reprojection_error,
        std_reprojection_error, inliers, mkpts0, mkpts1, mpts3D,
        mconf, frame_idx, rvec_o, rvec, coverage_score=None
    ):
        num_inliers = len(inliers)
        inlier_ratio = num_inliers / len(mkpts0) if len(mkpts0) > 0 else 0

        reprojection_error_threshold = 4.0
        max_translation_jump = 2.0
        max_orientation_jump = 20.0  # degrees
        min_inlier = 4
        coverage_threshold = -1#0.4#-1

        if coverage_score is None:
            logger.info("Coverage score not found, using default...")
            coverage_score = 0.5#0#0.5

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

    def estimate_pose(self, mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, frame_keypoints):
        """
        Original pose estimation method (used during initialization)
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
            confidence=0.999,
            iterationsCount=2000,
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
            # Calculate entropy term
            entropy_sum = 0
            for count in regions.values():
                if count > 0:
                    proportion = count / total_points
                    entropy_sum += proportion * np.log(proportion)
            
            # Normalize by log(4) as specified in the paper
            normalized_entropy = -entropy_sum / np.log(4)
            
            # Calculate mean confidence
            mean_confidence = 1  # np.mean(valid_conf)
            
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

        # Update pose_data with additional region and coverage details
        pose_data = self._kalman_filter_update(
            R, tvec, reprojection_errors, mean_reprojection_error,
            std_reprojection_error, inliers, mkpts0, mkpts1, mpts3D,
            mconf, frame_idx, rvec_o, rvec, coverage_score=coverage_score
        )
        pose_data["region_distribution"] = regions
        pose_data["coverage_score"] = coverage_score
        pose_data["tracking_mode"] = False  # This is from initialization, not tracking
        pose_data["inlier_percentage"] = 100.0 * len(inliers) / len(mkpts0) if len(mkpts0) > 0 else 0

        visualization = self._visualize_matches(
            frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints
        )
        return pose_data, visualization

    def _visualize_matches(self, frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints):
        """
        Visualize matches between anchor and current frame (used during initialization)
        """
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

        # Show both raw and filtered pose information
        # Raw position from PnP
        t_in_cam = pose_data['object_translation_in_cam']
        position_text = (f"Raw: x={t_in_cam[0]:.3f}, y={t_in_cam[1]:.3f}, z={t_in_cam[2]:.3f}")
        cv2.putText(out, position_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Filtered position from Kalman filter
        kf_t = pose_data['kf_translation_vector']
        kf_text = (f"KF: x={kf_t[0]:.3f}, y={kf_t[1]:.3f}, z={kf_t[2]:.3f}")
        cv2.putText(out, kf_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Add initialization mode indicator
        cv2.putText(out, "INITIALIZATION MODE", (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Add stats about matches
        stats_text = f"Inliers: {len(inlier_idx)}/{len(mkpts0)}, Reprojection error: {pose_data['mean_reprojection_error']:.2f}"
        cv2.putText(out, stats_text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2, cv2.LINE_AA)

        return out
    

    def _reinforce_tracking(self, frame, frame_idx):
        """
        Periodically reinforce tracking with full feature matching 
        without resetting current tracks
        """
        # Process with SuperPoint + LightGlue (similar to _initialize_tracking)
        anchor_proc = self.anchor_proc
        frame_proc = SuperPointPreprocessor.preprocess(frame)
        frame_proc = frame_proc[None].astype(np.float32)
        
        batch = np.concatenate([anchor_proc, frame_proc], axis=0).astype(np.float32)
        keypoints, matches, mscores = self.session.run(None, {"images": batch})
        
        # Process matches similar to _initialize_tracking
        valid_mask = (matches[:, 0] == 0)
        valid_matches = matches[valid_mask]
        
        mkpts0 = keypoints[0][valid_matches[:, 1]]
        mkpts1 = keypoints[1][valid_matches[:, 2]]
        mconf = mscores[valid_mask]
        anchor_indices = valid_matches[:, 1]
        
        # Filter to known anchor indices
        known_mask = np.isin(anchor_indices, self.matched_anchor_indices)
        mkpts0 = mkpts0[known_mask]
        mkpts1 = mkpts1[known_mask]
        mconf = mconf[known_mask]
        
        # Map to 3D points
        idx_map = {idx: i for i, idx in enumerate(self.matched_anchor_indices)}
        mpts3D = np.array([self.matched_3D_keypoints[idx_map[aidx]] 
                        for aidx in anchor_indices[known_mask]])
        
        # Merge with existing tracks
        # Use a KD tree to find correspondences
        if len(self.active_keypoints) > 0 and len(mkpts1) > 0:
            tree = cKDTree(self.active_keypoints)
            distances, indices = tree.query(mkpts1, k=1)
            
            # For points that match existing tracks, update the tracks
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if dist < 10.0:  # Close enough to be the same point
                    # Update the track
                    self.active_keypoints[idx] = mkpts1[i]
                    # Don't change the 3D point - keep consistency
            
            # For new points that don't match existing tracks, add them
            new_mask = distances > 10.0
            if np.any(new_mask):
                new_mkpts1 = mkpts1[new_mask]
                new_mpts3D = mpts3D[new_mask]
                
                # Add new points to tracking
                self.active_keypoints = np.vstack((self.active_keypoints, new_mkpts1))
                self.active_3D_points = np.vstack((self.active_3D_points, new_mpts3D))
                
                # Add new tracks
                for pt in new_mkpts1:
                    self.tracks.append([pt])
                    self.track_ages.append(1)
                    self.track_ids.append(self.next_track_id)
                    self.next_track_id += 1
        
        return True

##############################################################################################################################################################

# [Input Video/Image Sequence] → [main_track.py]
#   │
#   ↓
# [PoseEstimatorWithTracking]
#   │
#   ├─→ [Initialization Mode] ───┐
#   │   │                        │
#   │   ↓                        │
#   │   [SuperPoint + LightGlue] │
#   │   │                        │
#   │   ↓                        │
#   │   [2D-3D Correspondences]  │
#   │                            │
#   ├─→ [Tracking Mode] ─────────┤
#   │   │                        │
#   │   ↓                        │
#   │   [Optical Flow Tracking]  │
#   │                            │
#   ├─→ [Pose Estimation] ───────┤
#   │   │                        │
#   │   ↓                        │
#   │   [SolvePnP + Refinement]  │
#   │                            │
#   ├─→ [Kalman Filter] ─────────┘
#   │   │
#   │   ↓
#   │   [Smooth Pose]
#   │
#   ↓
# [Visualization & Data Output]

###############################################################################################################################################################

# import cv2
# import torch
# torch.set_grad_enabled(False)
# import time
# import numpy as np
# from scipy.spatial import cKDTree
# import onnxruntime as ort
# from utils import (
#     frame2tensor,
#     rotation_matrix_to_quaternion,
#     quaternion_to_rotation_matrix,
#     normalize_quaternion
# )

# from KF_Q import KalmanFilterPose
# import matplotlib.cm as cm
# from models.utils import make_matching_plot_fast
# import logging

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,  # Change to DEBUG for detailed logs
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("pose_estimator_tracking.log"),  # Logs will be saved in this file
#         logging.StreamHandler()  # Logs will also be printed to console
#     ]
# )
# logger = logging.getLogger(__name__)

# # Import the SuperPointPreprocessor
# from superpoint_LG import SuperPointPreprocessor

# class PoseEstimatorWithTracking:
#     def __init__(self, opt, device):
#         self.opt = opt
#         self.device = device
#         self.initial_z_set = False  # Flag for first-frame Z override (if desired)
#         self.kf_initialized = False  # To track if Kalman filter was ever updated
        
#         # Tracking-related variables
#         self.prev_frame = None  # Store previous frame for optical flow
#         self.active_keypoints = None  # Current tracked keypoints
#         self.active_3D_points = None  # Corresponding 3D points for tracked keypoints
#         self.keypoint_ages = None  # Track how long each keypoint has been tracked
#         self.tracking_initialized = False  # Flag to know if we're in tracking mode
        
#         # Tracking parameters
#         self.min_tracked_points = 20  # Minimum number of tracked points before reinitializing
#         self.max_tracking_error = 10.0  # Maximum allowed optical flow error
#         self.lk_params = dict(
#             winSize=(21, 21),
#             maxLevel=3,
#             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
#             minEigThreshold=0.001
#         )

#         logger.info("Initializing PoseEstimatorWithTracking")

#         # Load anchor (leader) image
#         self.anchor_image = cv2.imread(opt.anchor)
#         assert self.anchor_image is not None, f'Failed to load anchor image at {opt.anchor}'
#         self.anchor_image = self._resize_image(self.anchor_image, opt.resize)
#         logger.info(f"Loaded and resized anchor image from {opt.anchor}")

#         # Initialize ONNX session for SuperPoint + LightGlue
#         providers = [
#             ("CUDAExecutionProvider", {}),
#             ("CPUExecutionProvider", {})
#         ]
#         self.session = ort.InferenceSession(
#             "weights/superpoint_lightglue_pipeline_1280x720.onnx",
#             providers=providers
#         )
#         logger.info("ONNX session initialized with CUDAExecutionProvider")

#         # Define anchor keypoints (2D -> 3D correspondence)
#         anchor_keypoints_2D = np.array([
#             [511, 293], #0
#             [591, 284], #
#             #[610, 269], #
#             [587, 330], #
#             [413, 249], #
#             [602, 348], #
#             [715, 384], #
#             [598, 298], #
#             [656, 171], #
#             [805, 213],#
#             [703, 392],#10 
#             [523, 286],#
#             [519, 327],#12
#             [387, 289],#13
#             [727, 126],# 14
#             [425, 243],# 15
#             [636, 358],#
#             [745, 202],#
#             [595, 388],#
#             [436, 260],#
#             [539, 313], # 20
#             [795, 220],# 
#             [351, 291],#
#             [665, 165],# 
#             [611, 353], #
#             [650, 377],# 25
#             [516, 389],## 
#             [727, 143], #
#             [496, 378], #
#             [575, 312], #
#             [617, 368],# 30
#             [430, 312], #
#             [480, 281], #
#             [834, 225], #
#             [469, 339], #
#             [705, 223], # 35
#             [637, 156], 
#             [816, 414], 
#             [357, 195], 
#             [752, 77], 
#             [642, 451]
#         ], dtype=np.float32)

#         anchor_keypoints_3D = np.array([
#             [-0.014,  0.000,  0.042],
#             [ 0.025, -0.014, -0.011],
#             #[ 0.049, -0.016, -0.011],
#             [-0.014,  0.000, -0.042],
#             [-0.014,  0.000,  0.156],
#             [-0.023,  0.000, -0.065],
#             [ 0.000,  0.000, -0.156],
#             [ 0.025,  0.000, -0.015],
#             [ 0.217,  0.000,  0.070],#
#             [ 0.230,  0.000, -0.070],
#             [-0.014,  0.000, -0.156],
#             [ 0.000,  0.000,  0.042],
#             [-0.057, -0.018, -0.010],
#             [-0.074, -0.000,  0.128],
#             [ 0.206, -0.070, -0.002],
#             [-0.000, -0.000,  0.156],
#             [-0.017, -0.000, -0.092],
#             [ 0.217, -0.000, -0.027],#
#             [-0.052, -0.000, -0.097],
#             [-0.019, -0.000,  0.128],
#             [-0.035, -0.018, -0.010],
#             [ 0.217, -0.000, -0.070],#
#             [-0.080, -0.000,  0.156],
#             [ 0.230, -0.000,  0.070],
#             [-0.023, -0.000, -0.075],
#             [-0.029, -0.000, -0.127],
#             [-0.090, -0.000, -0.042],
#             [ 0.206, -0.055, -0.002],
#             [-0.090, -0.000, -0.015],
#             [ 0.000, -0.000, -0.015],
#             [-0.037, -0.000, -0.097],
#             [-0.074, -0.000,  0.074],
#             [-0.019, -0.000,  0.074],
#             [ 0.230, -0.000, -0.113],#
#             [-0.100, -0.030,  0.000],#
#             [ 0.170, -0.000, -0.015],
#             [ 0.230, -0.000,  0.113],
#             [-0.000, -0.025, -0.240],
#             [-0.000, -0.025,  0.240],
#             [ 0.243, -0.104,  0.000],
#             [-0.080, -0.000, -0.156]
#         ], dtype=np.float32)

#         # Set anchor features
#         self._set_anchor_features(
#             anchor_bgr_image=self.anchor_image,
#             anchor_keypoints_2D=anchor_keypoints_2D,
#             anchor_keypoints_3D=anchor_keypoints_3D
#         )

#         # Initialize Kalman filter
#         self.anchor_viewpoint_eulers = np.array([0.0, -0.35, 0.0], dtype=np.float32)
#         self.kf_pose = self._init_kalman_filter()
#         self.kf_pose_first_update = True 
#         logger.info("Kalman filter initialized")

#     def reinitialize_anchor(self, new_anchor_path, new_2d_points, new_3d_points):
#         """
#         Re-load a new anchor image and re-compute relevant data (2D->3D correspondences).
#         Called on-the-fly (e.g. after 200 frames).
#         """
#         logger.info(f"Re-initializing anchor with new image: {new_anchor_path}")

#         # 1. Load new anchor image
#         new_anchor_image = cv2.imread(new_anchor_path)
#         assert new_anchor_image is not None, f"Failed to load new anchor image at {new_anchor_path}"
#         new_anchor_image = self._resize_image(new_anchor_image, self.opt.resize)

#         # 2. Update anchor image
#         self.anchor_image = new_anchor_image

#         # 3. Recompute anchor features with the new image and 2D/3D
#         self._set_anchor_features(
#             anchor_bgr_image=new_anchor_image,
#             anchor_keypoints_2D=new_2d_points,
#             anchor_keypoints_3D=new_3d_points
#         )

#         # 4. Reset tracking state
#         self.tracking_initialized = False
#         self.active_keypoints = None
#         self.active_3D_points = None
#         self.keypoint_ages = None
#         self.prev_frame = None

#         logger.info("Anchor re-initialization complete.")

#     def _set_anchor_features(self, anchor_bgr_image, anchor_keypoints_2D, anchor_keypoints_3D):
#         """
#         Run SuperPoint on the anchor image to get anchor_keypoints_sp.
#         Then match those keypoints to known 2D->3D correspondences via KDTree.
#         """
#         # Precompute anchor's SuperPoint descriptors
#         self.anchor_proc = SuperPointPreprocessor.preprocess(anchor_bgr_image)
#         self.anchor_proc = self.anchor_proc[None].astype(np.float32)

#         # Dummy forward pass to get anchor keypoints from LightGlue
#         anchor_batch = np.concatenate([self.anchor_proc, self.anchor_proc], axis=0)
#         keypoints, matches, mscores = self.session.run(None, {"images": anchor_batch})
#         self.anchor_keypoints_sp = keypoints[0]  # (N, 2)

#         # Build KDTree to match anchor_keypoints_sp -> known anchor_keypoints_2D
#         sp_tree = cKDTree(self.anchor_keypoints_sp)
#         distances, indices = sp_tree.query(anchor_keypoints_2D, k=1)
#         valid_matches = distances < 1.0  # Threshold for "close enough"

#         self.matched_anchor_indices = indices[valid_matches]
#         self.matched_3D_keypoints = anchor_keypoints_3D[valid_matches]

#         logger.info(f"Matched {len(self.matched_anchor_indices)} keypoints to 3D points")

#     def _resize_image(self, image, resize):
#         logger.debug("Resizing image")
#         if len(resize) == 2:
#             return cv2.resize(image, tuple(resize))
#         elif len(resize) == 1 and resize[0] > 0:
#             h, w = image.shape[:2]
#             scale = resize[0] / max(h, w)
#             new_size = (int(w * scale), int(h * scale))
#             return cv2.resize(image, new_size)
#         return image

#     # def process_frame(self, frame, frame_idx):
#     #     """
#     #     Process a new frame for pose estimation.
#     #     Either initialize by matching to anchor, or track previous features.
#     #     """
#     #     logger.info(f"Processing frame {frame_idx}")
#     #     start_time = time.time()

#     #     # Resize frame to target size
#     #     frame = self._resize_image(frame, self.opt.resize)
        
#     #     # Convert to grayscale for tracking
#     #     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#     #     # Check if we need to initialize or re-initialize the tracking
#     #     if not self.tracking_initialized or self.active_keypoints is None or len(self.active_keypoints) < self.min_tracked_points:
#     #         # Initialize/re-initialize using SuperPoint + LightGlue
#     #         logger.info(f"Initializing/re-initializing tracking for frame {frame_idx}")
#     #         pose_data, visualization = self._initialize_tracking(frame, frame_idx)
            
#     #         # Store the current frame for future tracking
#     #         self.prev_frame = frame_gray.copy()
            
#     #         return pose_data, visualization
        
#     #     # If we're here, we're in tracking mode
#     #     tracked_pose_data, visualization = self._track_features(frame, frame_gray, frame_idx)
        
#     #     # Store the current frame for future tracking
#     #     self.prev_frame = frame_gray.copy()
        
#     #     return tracked_pose_data, visualization

#     def process_frame(self, frame, frame_idx):
#         """
#         Process a new frame for pose estimation.
#         Either initialize by matching to anchor, track previous features,
#         or fall back to Kalman prediction when both fail.
#         """
#         logger.info(f"Processing frame {frame_idx}")
#         start_time = time.time()

#         # Resize frame to target size
#         frame = self._resize_image(frame, self.opt.resize)
        
#         # Convert to grayscale for tracking
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Get a Kalman prediction regardless of what happens next (for fallback)
#         x_pred, P_pred = self.kf_pose.predict()
#         kf_translation = x_pred[0:3]
#         kf_quaternion = x_pred[6:10]
#         kf_rotation = quaternion_to_rotation_matrix(kf_quaternion)
        
#         # Store the Kalman prediction as a fallback
#         fallback_pose_data = {
#             'frame': frame_idx,
#             'kf_translation_vector': kf_translation.tolist(),
#             'kf_quaternion': kf_quaternion.tolist(),
#             'kf_rotation_matrix': kf_rotation.tolist(),
#             'pose_estimation_failed': True,
#             'estimation_method': 'kalman_prediction_only'
#         }
        
#         # Create a simple visualization for fallback case
#         fallback_vis = frame.copy()
#         position_text = (f"Leader in Cam (KF prediction): "
#                         f"x={kf_translation[0]:.3f}, y={kf_translation[1]:.3f}, z={kf_translation[2]:.3f}")
#         cv2.putText(fallback_vis, position_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (0, 0, 255), 2, cv2.LINE_AA)
#         cv2.putText(fallback_vis, "KALMAN PREDICTION ONLY", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
#         # Check if we need to initialize or re-initialize the tracking
#         if not self.tracking_initialized or self.active_keypoints is None or len(self.active_keypoints) < self.min_tracked_points:
#             # Initialize/re-initialize using SuperPoint + LightGlue
#             logger.info(f"Initializing/re-initializing tracking for frame {frame_idx}")
#             pose_data, visualization = self._initialize_tracking(frame, frame_idx)
            
#             # If initialization failed, use Kalman prediction
#             if pose_data is None:
#                 logger.warning(f"Initialization failed for frame {frame_idx}, using Kalman prediction")
#                 pose_data = fallback_pose_data
#                 visualization = fallback_vis
#             else:
#                 pose_data['estimation_method'] = 'initialization'
                
#             # Store the current frame for future tracking
#             self.prev_frame = frame_gray.copy()
            
#             return pose_data, visualization
        
#         # If we're here, we're in tracking mode - try to track features
#         pose_data, visualization = self._track_features(frame, frame_gray, frame_idx)
        
#         # If tracking failed, use Kalman prediction
#         if pose_data is None:
#             logger.warning(f"Tracking failed for frame {frame_idx}, using Kalman prediction")
#             pose_data = fallback_pose_data
#             visualization = fallback_vis
#         else:
#             pose_data['estimation_method'] = 'tracking'
        
#         # Store the current frame for future tracking
#         self.prev_frame = frame_gray.copy()
        
#         return pose_data, visualization

#     def _initialize_tracking(self, frame, frame_idx):
#         """
#         Initialize or re-initialize tracking by matching current frame to anchor
#         """
#         logger.info(f"Performing full matching for frame {frame_idx}")
        
#         # Process with SuperPoint + LightGlue
#         anchor_proc = self.anchor_proc  # Precomputed anchor

#         frame_proc = SuperPointPreprocessor.preprocess(frame)
#         frame_proc = frame_proc[None].astype(np.float32)

#         batch = np.concatenate([anchor_proc, frame_proc], axis=0).astype(np.float32)
#         batch = torch.tensor(batch, device=self.device).cpu().numpy()

#         keypoints, matches, mscores = self.session.run(None, {"images": batch})

#         valid_mask = (matches[:, 0] == 0)
#         valid_matches = matches[valid_mask]

#         mkpts0 = keypoints[0][valid_matches[:, 1]]
#         mkpts1 = keypoints[1][valid_matches[:, 2]]
#         mconf = mscores[valid_mask]
#         anchor_indices = valid_matches[:, 1]

#         logger.debug(f"Found {len(mkpts0)} raw matches in frame {frame_idx}")

#         # Filter to known anchor indices
#         known_mask = np.isin(anchor_indices, self.matched_anchor_indices)
#         mkpts0 = mkpts0[known_mask]
#         mkpts1 = mkpts1[known_mask]
#         mconf = mconf[known_mask]

#         idx_map = {idx: i for i, idx in enumerate(self.matched_anchor_indices)}
#         mpts3D = np.array([self.matched_3D_keypoints[idx_map[aidx]] 
#                            for aidx in anchor_indices[known_mask]])

#         if len(mkpts0) < 8:  # Need at least 8 points for good initialization
#             logger.warning(f"Not enough matches for initialization (frame {frame_idx})")
#             return None, frame

#         # Set up tracking state
#         self.active_keypoints = mkpts1.copy()  # Current 2D positions
#         self.active_3D_points = mpts3D.copy()  # Corresponding 3D points
#         self.keypoint_ages = np.zeros(len(mkpts1), dtype=np.int32)  # Initialize ages
#         self.tracking_initialized = True

#         logger.info(f"Tracking initialized with {len(self.active_keypoints)} keypoints")

#         # Estimate pose using the current set of matches
#         pose_data, visualization = self.estimate_pose(
#             mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, keypoints[1]
#         )
        
#         return pose_data, visualization

#     def _track_features(self, frame, frame_gray, frame_idx):
#         """
#         Track features from previous frame to current frame using optical flow
#         """
#         logger.info(f"Tracking features for frame {frame_idx}")
        
#         if self.prev_frame is None or self.active_keypoints is None or len(self.active_keypoints) == 0:
#             logger.warning("Cannot track: No previous frame or active keypoints")
#             return None, frame
        
#         # Convert active keypoints to proper format for tracking
#         prev_pts = self.active_keypoints.reshape(-1, 1, 2).astype(np.float32)
        
#         # Calculate optical flow
#         curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
#             self.prev_frame, frame_gray, prev_pts, None, **self.lk_params
#         )
        
#         # Filter out points that couldn't be tracked
#         status = status.flatten()
#         err = err.flatten()
        
#         # Additional filtering: remove points with high error
#         valid = (status == 1) & (err < self.max_tracking_error)
        
#         # Update tracked keypoints and their corresponding 3D points
#         good_curr_pts = curr_pts[valid].reshape(-1, 2)
#         good_3D_pts = self.active_3D_points[valid]
#         good_ages = self.keypoint_ages[valid] + 1  # Increment age of tracked points
        
#         # Update tracking state
#         self.active_keypoints = good_curr_pts
#         self.active_3D_points = good_3D_pts
#         self.keypoint_ages = good_ages
        
#         logger.info(f"Successfully tracked {len(good_curr_pts)}/{len(prev_pts)} points")
        
#         # If too few points remain, signal that we need to re-initialize
#         if len(good_curr_pts) < self.min_tracked_points:
#             logger.warning(f"Too few tracked points ({len(good_curr_pts)}), need re-initialization")
#             self.tracking_initialized = False
#             return None, frame
        
#         # Create dummy data for visualization purposes (no actual matching happened)
#         dummy_mkpts0 = np.zeros_like(good_curr_pts)  # We don't have actual anchor correspondences here
#         dummy_mconf = np.ones(len(good_curr_pts))  # Set all confidences to 1.0
        
#         # Estimate pose using the tracked keypoints
#         pose_data, visualization = self.estimate_pose_from_tracked(
#             good_curr_pts, good_3D_pts, dummy_mconf, frame, frame_idx, good_ages
#         )
        
#         return pose_data, visualization

#     def estimate_pose_from_tracked(self, tracked_pts, pts3D, mconf, frame, frame_idx, point_ages):
#         """
#         Estimate pose using tracked points (instead of matches)
#         """
#         logger.debug(f"Estimating pose from tracked points for frame {frame_idx}")
#         K, distCoeffs = self._get_camera_intrinsics()

#         objectPoints = pts3D.reshape(-1, 1, 3)
#         imagePoints = tracked_pts.reshape(-1, 1, 2).astype(np.float32)

#         # Solve PnP
#         success, rvec_o, tvec_o, inliers = cv2.solvePnPRansac(
#             objectPoints=objectPoints,
#             imagePoints=imagePoints,
#             cameraMatrix=K,
#             distCoeffs=distCoeffs,
#             reprojectionError=4,
#             confidence=0.999,
#             iterationsCount=2000,
#             flags=cv2.SOLVEPNP_EPNP
#         )

#         if not success or inliers is None or len(inliers) < 6:
#             logger.warning("PnP pose estimation failed or not enough inliers.")
#             return None, frame

#         # Refine with VVS
#         objectPoints_inliers = objectPoints[inliers.flatten()]
#         imagePoints_inliers = imagePoints[inliers.flatten()]

#         rvec, tvec = cv2.solvePnPRefineVVS(
#             objectPoints=objectPoints_inliers,
#             imagePoints=imagePoints_inliers,
#             cameraMatrix=K,
#             distCoeffs=distCoeffs,
#             rvec=rvec_o,
#             tvec=tvec_o
#         )

#         # Get rotation matrix
#         R, _ = cv2.Rodrigues(rvec)

#         # Initialize region counters
#         regions = {"front-right": 0, "front-left": 0, "back-right": 0, "back-left": 0}

#         # Classify points into regions
#         for point in objectPoints_inliers[:, 0]:
#             if point[0] < 0 and point[2] > 0:  # Front-Right
#                 regions["front-right"] += 1
#             elif point[0] < 0 and point[2] < 0:  # Front-Left
#                 regions["front-left"] += 1
#             elif point[0] > 0 and point[2] > 0:  # Back-Right
#                 regions["back-right"] += 1
#             elif point[0] > 0 and point[2] < 0:  # Back-Left
#                 regions["back-left"] += 1

#         # Calculate coverage score
#         total_points = sum(regions.values())
#         if total_points > 0:
#             # Calculate entropy term
#             entropy_sum = 0
#             for count in regions.values():
#                 if count > 0:
#                     proportion = count / total_points
#                     entropy_sum += proportion * np.log(proportion)
            
#             # Normalize by log(4) as specified in the paper
#             normalized_entropy = -entropy_sum / np.log(4)
            
#             # Final coverage score
#             coverage_score = normalized_entropy
            
#             # Ensure score is in valid range [0,1]
#             coverage_score = np.clip(coverage_score, 0, 1)
#         else:
#             coverage_score = 0

#         # Compute reprojection errors
#         projected_points, _ = cv2.projectPoints(
#             objectPoints_inliers, rvec, tvec, K, distCoeffs
#         )
#         reprojection_errors = np.linalg.norm(imagePoints_inliers - projected_points, axis=2).flatten()
#         mean_reprojection_error = np.mean(reprojection_errors)
#         std_reprojection_error = np.std(reprojection_errors)

#         # Update pose_data with additional region and coverage details
#         pose_data = self._kalman_filter_update(
#             R, tvec, reprojection_errors, mean_reprojection_error,
#             std_reprojection_error, inliers, tracked_pts, tracked_pts, pts3D,
#             mconf, frame_idx, rvec_o, rvec, coverage_score=coverage_score
#         )
#         pose_data["region_distribution"] = regions
#         pose_data["coverage_score"] = coverage_score
#         pose_data["tracking_mode"] = True
#         pose_data["num_tracked_points"] = len(tracked_pts)
#         pose_data["max_point_age"] = int(np.max(point_ages)) if len(point_ages) > 0 else 0
#         pose_data["avg_point_age"] = float(np.mean(point_ages)) if len(point_ages) > 0 else 0

#         # Create visualization with tracked points
#         visualization = self._visualize_tracked_points(
#             frame, inliers, tracked_pts, point_ages, pose_data
#         )
#         return pose_data, visualization

#     def _visualize_tracked_points(self, frame, inliers, tracked_pts, point_ages, pose_data):
#         """
#         Visualize tracked points with age-based coloring
#         """
#         # Convert frame to show annotations
#         vis_frame = frame.copy()
        
#         # Determine color based on point age
#         max_age = max(point_ages) if len(point_ages) > 0 else 1
        
#         # Draw all tracked points
#         for i, (pt, age) in enumerate(zip(tracked_pts, point_ages)):
#             # Normalize age for color (older points = bluer)
#             normalized_age = age / max_age
            
#             # Create color based on point age (red->green->blue)
#             if normalized_age < 0.5:
#                 # Red to green
#                 r = 255 * (1 - 2 * normalized_age)
#                 g = 255 * (2 * normalized_age)
#                 b = 0
#             else:
#                 # Green to blue
#                 r = 0
#                 g = 255 * (2 - 2 * normalized_age)
#                 b = 255 * (2 * normalized_age - 1)
            
#             color = (int(b), int(g), int(r))  # BGR format for OpenCV
            
#             # Draw circle for the point
#             cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 2, color, -1)
            
#             # If this is an inlier, draw a larger circle around it
#             if inliers is not None and i in inliers:
#                 cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 5, color, 1)
        
#         # Draw pose information
#         t_in_cam = pose_data['object_translation_in_cam']
#         position_text = (f"Leader in Cam: "
#                         f"x={t_in_cam[0]:.3f}, y={t_in_cam[1]:.3f}, z={t_in_cam[2]:.3f}")
        
#         tracking_text = f"Tracking mode: {len(tracked_pts)} points, max age: {max_age}"
        
#         cv2.putText(vis_frame, position_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (255, 0, 0), 2, cv2.LINE_AA)
#         cv2.putText(vis_frame, tracking_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
#         return vis_frame

#     def _init_kalman_filter(self):
#         frame_rate = 30
#         dt = 1 / frame_rate
#         kf_pose = KalmanFilterPose(dt)
#         return kf_pose
    
#     def _get_camera_intrinsics(self):
#         """
#         Returns camera intrinsics matrix K and distortion coefficients.
#         Replace with your actual camera parameters.
#         """
#         # Camera intrinsics from the original code
#         focal_length_x = 1430.10150
#         focal_length_y = 1430.48915
#         cx = 640.85462
#         cy = 480.64800

#         distCoeffs = np.array([0.3393, 2.0351, 0.0295, -0.0029, -10.9093], dtype=np.float32)

#         K = np.array([
#             [focal_length_x, 0, cx],
#             [0, focal_length_y, cy],
#             [0, 0, 1]
#         ], dtype=np.float32)

#         return K, distCoeffs

#     def estimate_pose(self, mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, frame_keypoints):
#         """
#         Original pose estimation method (used during initialization)
#         """
#         logger.debug(f"Estimating pose for frame {frame_idx}")
#         K, distCoeffs = self._get_camera_intrinsics()

#         objectPoints = mpts3D.reshape(-1, 1, 3)
#         imagePoints = mkpts1.reshape(-1, 1, 2).astype(np.float32)

#         # Solve PnP
#         success, rvec_o, tvec_o, inliers = cv2.solvePnPRansac(
#             objectPoints=objectPoints,
#             imagePoints=imagePoints,
#             cameraMatrix=K,
#             distCoeffs=distCoeffs,
#             reprojectionError=4,
#             confidence=0.999,
#             iterationsCount=2000,
#             flags=cv2.SOLVEPNP_EPNP
#         )

#         if not success or inliers is None or len(inliers) < 6:
#             logger.warning("PnP pose estimation failed or not enough inliers.")
#             return None, frame

#         # Refine with VVS
#         objectPoints_inliers = objectPoints[inliers.flatten()]
#         imagePoints_inliers = imagePoints[inliers.flatten()]

#         rvec, tvec = cv2.solvePnPRefineVVS(
#             objectPoints=objectPoints_inliers,
#             imagePoints=imagePoints_inliers,
#             cameraMatrix=K,
#             distCoeffs=distCoeffs,
#             rvec=rvec_o,
#             tvec=tvec_o
#         )

#         # R, t => object_in_cam
#         R, _ = cv2.Rodrigues(rvec)

#         # Initialize region counters
#         regions = {"front-right": 0, "front-left": 0, "back-right": 0, "back-left": 0}

#         # Classify points into regions
#         for point in objectPoints_inliers[:, 0]:
#             if point[0] < 0 and point[2] > 0:  # Front-Right
#                 regions["front-right"] += 1
#             elif point[0] < 0 and point[2] < 0:  # Front-Left
#                 regions["front-left"] += 1
#             elif point[0] > 0 and point[2] > 0:  # Back-Right
#                 regions["back-right"] += 1
#             elif point[0] > 0 and point[2] < 0:  # Back-Left
#                 regions["back-left"] += 1

#         # Calculate coverage score
#         total_points = sum(regions.values())
#         if total_points > 0:
#             # Calculate entropy term
#             entropy_sum = 0
#             for count in regions.values():
#                 if count > 0:
#                     proportion = count / total_points
#                     entropy_sum += proportion * np.log(proportion)
            
#             # Normalize by log(4) as specified in the paper
#             normalized_entropy = -entropy_sum / np.log(4)
            
#             # Calculate mean confidence
#             mean_confidence = 1  # np.mean(valid_conf)
            
#             # Final coverage score
#             coverage_score = normalized_entropy * mean_confidence
            
#             # Ensure score is in valid range [0,1]
#             coverage_score = np.clip(coverage_score, 0, 1)
#         else:
#             coverage_score = 0

#         # Compute reprojection errors
#         projected_points, _ = cv2.projectPoints(
#             objectPoints_inliers, rvec, tvec, K, distCoeffs
#         )
#         reprojection_errors = np.linalg.norm(imagePoints_inliers - projected_points, axis=2).flatten()
#         mean_reprojection_error = np.mean(reprojection_errors)
#         std_reprojection_error = np.std(reprojection_errors)

#         # Update pose_data with additional region and coverage details
#         pose_data = self._kalman_filter_update(
#             R, tvec, reprojection_errors, mean_reprojection_error,
#             std_reprojection_error, inliers, mkpts0, mkpts1, mpts3D,
#             mconf, frame_idx, rvec_o, rvec, coverage_score=coverage_score
#         )
#         pose_data["region_distribution"] = regions
#         pose_data["coverage_score"] = coverage_score
#         pose_data["tracking_mode"] = False  # This is from initialization, not tracking

#         visualization = self._visualize_matches(
#             frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints
#         )
#         return pose_data, visualization
    
#     def _kalman_filter_update(
#         self, R, tvec, reprojection_errors, mean_reprojection_error,
#         std_reprojection_error, inliers, mkpts0, mkpts1, mpts3D,
#         mconf, frame_idx, rvec_o, rvec, coverage_score=None
#     ):
#         num_inliers = len(inliers)
#         inlier_ratio = num_inliers / len(mkpts0) if len(mkpts0) > 0 else 0

#         reprojection_error_threshold = 4.0
#         max_translation_jump = 2.0
#         max_orientation_jump = 20.0  # degrees
#         min_inlier = 4
#         coverage_threshold = -1

#         if coverage_score is None:
#             logger.info("Coverage score not found, using default...")
#             coverage_score = 0.5

#         # 1) Convert measured rotation R -> quaternion
#         q_measured = rotation_matrix_to_quaternion(R)

#         # (Optional) viewpoint check if you have anchor_viewpoint_quat
#         anchor_q = getattr(self, "anchor_viewpoint_quat", None)
#         viewpoint_max_diff_deg = 380.0

#         def quaternion_angle_degrees(q1, q2):
#             q1 = normalize_quaternion(q1)
#             q2 = normalize_quaternion(q2)
#             dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
#             angle = 2.0 * np.degrees(np.arccos(dot))
#             if angle > 180.0:
#                 angle = 360.0 - angle
#             return angle

#         if anchor_q is not None:
#             viewpoint_diff = quaternion_angle_degrees(q_measured, anchor_q)
#         else:
#             viewpoint_diff = 0.0

#         #---------------------------------------------------------------------
#         # 2) EKF PREDICT: get prior (x^-)
#         #---------------------------------------------------------------------
#         x_pred, P_pred = self.kf_pose.predict()  
#         # x_pred is shape (13,), e.g. [px,py,pz, vx,vy,vz, qx,qy,qz,qw, wx,wy,wz]

#         # Parse predicted orientation for threshold checks if desired
#         px_pred, py_pred, pz_pred = x_pred[0:3]
#         qx_pred, qy_pred, qz_pred, qw_pred = x_pred[6:10]

#         # Convert measured q, predicted q -> orientation_change (deg)
#         orientation_change = quaternion_angle_degrees(q_measured, [qx_pred,qy_pred,qz_pred,qw_pred])
#         # Also check translation change
#         translation_change = np.linalg.norm(tvec.flatten() - x_pred[0:3])

#         # 3) Build the measurement vector z = [px,py,pz, qx,qy,qz,qw]
#         tvec = tvec.flatten()
#         z_meas = np.array([
#             tvec[0], tvec[1], tvec[2],
#             q_measured[0], q_measured[1], q_measured[2], q_measured[3]
#         ], dtype=np.float64)

#         # If first update, skip thresholds
#         if not hasattr(self, 'kf_pose_first_update') or self.kf_pose_first_update:
#             # We always do the update
#             x_upd, P_upd = self.kf_pose.update(z_meas)
#             self.kf_pose_first_update = False
#             logger.debug("EKF first update: skipping threshold checks.")
#         else:
#             # Normal frames: check thresholds 
#             if mean_reprojection_error < reprojection_error_threshold and num_inliers > min_inlier:
#                 if translation_change < max_translation_jump and orientation_change < max_orientation_jump:
#                     if coverage_score >= coverage_threshold:
#                         if viewpoint_diff <= viewpoint_max_diff_deg:
#                             # pass all checks => do update
#                             x_upd, P_upd = self.kf_pose.update(z_meas)
#                             logger.debug("EKF update (all thresholds passed).")
#                         else:
#                             logger.debug(f"Skipping EKF update: viewpoint diff={viewpoint_diff:.1f}>{viewpoint_max_diff_deg}")
#                             x_upd, P_upd = x_pred, P_pred
#                     else:
#                         logger.debug(f"Skipping EKF update: coverage_score={coverage_score:.2f} < {coverage_threshold}")
#                         x_upd, P_upd = x_pred, P_pred
#                 else:
#                     logger.debug("Skipping EKF update: large translation/orientation jump.")
#                     x_upd, P_upd = x_pred, P_pred
#             else:
#                 logger.debug("Skipping EKF update: high repro error or insufficient inliers.")
#                 x_upd, P_upd = x_pred, P_pred

#         # For simplicity, let's assume x_upd is final
#         x_final = x_upd  
#         # parse final
#         px, py, pz = x_final[0:3]
#         qx, qy, qz, qw = x_final[6:10]
#         R_estimated = quaternion_to_rotation_matrix([qx, qy, qz, qw])

#         # Build final pose_data
#         pose_data = {
#             'frame': frame_idx,
#             'object_rotation_in_cam': R.tolist(),
#             'object_translation_in_cam': tvec.flatten().tolist(),
#             'raw_rvec': rvec_o.flatten().tolist(),
#             'refined_raw_rvec': rvec.flatten().tolist(),
#             'num_inliers': num_inliers,
#             'total_matches': len(mkpts0),
#             'inlier_ratio': inlier_ratio,
#             'reprojection_errors': reprojection_errors.tolist(),
#             'mean_reprojection_error': float(mean_reprojection_error),
#             'std_reprojection_error': float(std_reprojection_error),
#             'inliers': inliers.flatten().tolist(),
#             'mkpts0': mkpts0.tolist(),
#             'mkpts1': mkpts1.tolist(),
#             'mpts3D': mpts3D.tolist(),
#             'mconf': mconf.tolist(),

#             # Filtered results from updated state:
#             'kf_translation_vector': [px, py, pz],
#             'kf_quaternion': [qx, qy, qz, qw],
#             'kf_rotation_matrix': R_estimated.tolist(),

#             # Additional coverage / viewpoint metrics
#             'coverage_score': coverage_score,
#             'viewpoint_diff_deg': viewpoint_diff
#         }
#         return pose_data
    
#     # def _visualize_matches(self, frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints):
#     #     anchor_image_gray = cv2.cvtColor(self.anchor_image, cv2.COLOR_BGR2GRAY)
#     #     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     #     inlier_idx = inliers.flatten()
#     #     inlier_mkpts0 = mkpts0[inlier_idx]
#     #     inlier_mkpts1 = mkpts1[inlier_idx]
#     #     inlier_conf = mconf[inlier_idx]
#     #     color = cm.jet(inlier_conf)

#     #     out = make_matching_plot_fast(
#     #         anchor_image_gray,
#     #         frame_gray,
#     #         self.anchor_keypoints_sp,
#     #         frame_keypoints,
#     #         inlier_mkpts0,
#     #         inlier_mkpts1,
#     #         color,
#     #         text=[],
#     #         path=None,
#     #         show_keypoints=self.opt.show_keypoints,
#     #         small_text=[]
#     #     )

#     #     # Show the object (leader) position in camera frame from pose_data
#     #     # E.g. we can just show tvec:
#     #     t_in_cam = pose_data['object_translation_in_cam']
#     #     position_text = (f"Leader in Cam: "
#     #                      f"x={t_in_cam[0]:.3f}, y={t_in_cam[1]:.3f}, z={t_in_cam[2]:.3f}")
#     #     cv2.putText(out, position_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
#     #                 0.7, (255, 0, 0), 2, cv2.LINE_AA)

#     #     return out

#     def _visualize_matches(self, frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints):
#         """
#         Visualize matches between anchor and current frame (used during initialization)
#         """
#         anchor_image_gray = cv2.cvtColor(self.anchor_image, cv2.COLOR_BGR2GRAY)
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         inlier_idx = inliers.flatten()
#         inlier_mkpts0 = mkpts0[inlier_idx]
#         inlier_mkpts1 = mkpts1[inlier_idx]
#         inlier_conf = mconf[inlier_idx]
#         color = cm.jet(inlier_conf)

#         out = make_matching_plot_fast(
#             anchor_image_gray,
#             frame_gray,
#             self.anchor_keypoints_sp,
#             frame_keypoints,
#             inlier_mkpts0,
#             inlier_mkpts1,
#             color,
#             text=[],
#             path=None,
#             show_keypoints=self.opt.show_keypoints,
#             small_text=[]
#         )

#         # Show both raw and filtered pose information
#         # Raw position from PnP
#         t_in_cam = pose_data['object_translation_in_cam']
#         position_text = (f"Raw: x={t_in_cam[0]:.3f}, y={t_in_cam[1]:.3f}, z={t_in_cam[2]:.3f}")
#         cv2.putText(out, position_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (255, 0, 0), 2, cv2.LINE_AA)
        
#         # Filtered position from Kalman filter
#         kf_t = pose_data['kf_translation_vector']
#         kf_text = (f"KF: x={kf_t[0]:.3f}, y={kf_t[1]:.3f}, z={kf_t[2]:.3f}")
#         cv2.putText(out, kf_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
#         # Add initialization mode indicator
#         cv2.putText(out, "INITIALIZATION MODE", (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
#         # Add stats about matches
#         stats_text = f"Inliers: {len(inlier_idx)}/{len(mkpts0)}, Reprojection error: {pose_data['mean_reprojection_error']:.2f}"
#         cv2.putText(out, stats_text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7, (255, 255, 0), 2, cv2.LINE_AA)

#         return out