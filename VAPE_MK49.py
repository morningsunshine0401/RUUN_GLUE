import cv2
import numpy as np
import torch
import time
import argparse
import warnings
import json
import threading
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import queue
import math # Added for quaternion calculations

# --- DEPENDENCY IMPORTS ---
# Suppress warnings for a cleaner console output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 VAPE MK47 Pose Estimator")
try:
    from ultralytics import YOLO
    import timm
    from torchvision import transforms
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    from PIL import Image
    from scipy.spatial import cKDTree
    print("✅ All libraries loaded successfully.")
except ImportError as e:
    print(f"❌ Import error: {e}. Please run 'pip install -r requirements.txt' to install dependencies.")
    exit(1)

# --- DATA STRUCTURES ---
@dataclass
class ProcessingResult:
    """Holds all data for a single processed frame for visualization and logging."""
    frame_id: int
    frame: np.ndarray
    position: Optional[np.ndarray] = None
    quaternion: Optional[np.ndarray] = None
    kf_position: Optional[np.ndarray] = None
    kf_quaternion: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    num_inliers: int = 0
    pose_success: bool = False
    viewpoint_used: Optional[str] = None # Added to log the viewpoint used for pose estimation

@dataclass
class PoseData:
    """A simple container for pose results."""
    position: np.ndarray
    quaternion: np.ndarray
    inliers: int
    reprojection_error: float
    viewpoint: str # The viewpoint that yielded this pose

# --- KALMAN FILTER ---
class LooselyCoupledKalmanFilter:
    """A simple Kalman Filter for smoothing pose data."""
    def __init__(self, dt=1/30.0):
        self.dt = dt
        self.initialized = False
        self.n_states = 13  # [pos(3), vel(3), quat(4), ang_vel(3)]
        self.x = np.zeros(self.n_states)
        self.x[9] = 1.0  # Identity quaternion (w,x,y,z or x,y,z,w depending on convention)
        self.P = np.eye(self.n_states) * 0.1 # Initial covariance
        self.Q = np.eye(self.n_states) * 1e-3 # Process noise covariance
        self.R = np.eye(7) * 1e-4 # Measurement noise covariance (for 3D pos + 4D quat)

    def normalize_quaternion(self, q):
        """Normalizes a quaternion to unit length."""
        norm = np.linalg.norm(q)
        return q / norm if norm > 1e-10 else np.array([0, 0, 0, 1])

    def predict(self):
        """
        Predicts the next state of the system based on the current state and motion model.
        Returns predicted position and quaternion.
        """
        if not self.initialized:
            # If not initialized, prediction is not meaningful, return current state or None
            # Return identity quaternion and zero position as a default uninitialized state
            return np.zeros(3), np.array([0,0,0,1])
        
        dt = self.dt
        
        # State transition matrix F
        # Position updates based on velocity: P_new = P_old + V_old * dt
        F = np.eye(self.n_states)
        F[0:3, 3:6] = np.eye(3) * dt # Update position based on velocity
        
        # Update state vector x
        self.x = F @ self.x
        
        # Quaternion update based on angular velocity (simplified Lie group integration)
        q, w = self.x[6:10], self.x[10:13] # Current quaternion and angular velocity
        
        # Skew-symmetric matrix for quaternion multiplication
        omega_mat = 0.5 * np.array([
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0]
        ])
        self.x[6:10] = self.normalize_quaternion((np.eye(4) + dt * omega_mat) @ q)
        
        # Update covariance matrix P
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x[0:3], self.x[6:10] # Return predicted position and quaternion

    def update(self, position: np.ndarray, quaternion: np.ndarray):
        """
        Updates the Kalman filter state with a new measurement.
        Returns updated (smoothed) position and quaternion.
        """
        # Form the measurement vector [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        # Ensure the incoming quaternion is normalized
        measurement = np.concatenate([position, self.normalize_quaternion(quaternion)])
        
        if not self.initialized:
            # Initialize state with the first measurement
            self.x[0:3] = position
            self.x[6:10] = self.normalize_quaternion(quaternion)
            self.initialized = True
            return self.x[0:3], self.x[6:10] # Return initial state
        
        # Measurement matrix H maps state to measurement
        H = np.zeros((7, self.n_states))
        H[0:3, 0:3] = np.eye(3) # Position part
        H[3:7, 6:10] = np.eye(4) # Quaternion part
        
        # Innovation (measurement residual)
        innovation = measurement - H @ self.x
        
        # Handle quaternion sign ambiguity: if dot product is negative, flip one quaternion
        # to ensure the shortest path is measured.
        if np.dot(measurement[3:7], self.x[6:10]) < 0:
            measurement[3:7] *= -1 # Flip sign of measurement quaternion
            innovation = measurement - H @ self.x # Re-calculate innovation
            
        # Innovation covariance S
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain K
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x += K @ innovation
        self.x[6:10] = self.normalize_quaternion(self.x[6:10]) # Re-normalize quaternion
        
        # Update error covariance
        self.P = (np.eye(self.n_states) - K @ H) @ self.P
        
        return self.x[0:3], self.x[6:10] # Return updated (smoothed) position and quaternion

# --- MAIN ESTIMATOR CLASS ---
class PoseEstimator:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔩 Using device: {self.device}")

        # --- Component Configuration (Baseline) ---
        self.use_yolo = True
        self.use_viewpoint = True
        self.use_kalman_filter = True
        # -------------------------------------------

        self.camera_width, self.camera_height = 1280, 720
        self.all_poses_log = []
        self.running = True

        # --- Pre-filtering related attributes ---
        self.last_orientation: Optional[np.ndarray] = None # Stores the last accepted quaternion

        # Filter thresholds (tune these values based on your specific scenario and expected motion)
        self.ORI_MAX_DIFF_DEG = 30.0  # Max allowed orientation change per frame in degrees
        
        # Re-initialization parameters
        self.rejected_consecutive_frames_count = 0 # Counter for consecutive rejected frames
        self.MAX_REJECTED_FRAMES = 60 # Number of consecutive rejected frames before re-initialization (e.g., 2 seconds at 30 FPS)
        # ----------------------------------------

        self._initialize_input_source()
        self._initialize_models()
        self._initialize_anchor_data()
        
        if self.use_kalman_filter:
            self.kf = LooselyCoupledKalmanFilter()

        # Thread-safe queue for displaying frames
        self.display_queue = queue.Queue(maxsize=2)
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()

        print("✅ Estimator initialized. Starting processing...")

    def _initialize_input_source(self):
        """Initializes the input source (webcam, video, or image folder)."""
        self.video_capture = None
        self.image_files = []
        self.frame_idx = 0
        self.is_video_stream = False

        if self.args.webcam:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                raise IOError("Cannot open webcam.")
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.is_video_stream = True
            print("📹 Using webcam input.")
        elif self.args.video_file:
            if not os.path.exists(self.args.video_file):
                raise FileNotFoundError(f"Video file not found: {self.args.video_file}")
            self.video_capture = cv2.VideoCapture(self.args.video_file)
            self.is_video_stream = True
            print(f"📹 Using video file input: {self.args.video_file}")
        elif self.args.image_dir:
            if not os.path.exists(self.args.image_dir):
                raise FileNotFoundError(f"Image directory not found: {self.args.image_dir}")
            self.image_files = sorted([os.path.join(self.args.image_dir, f) for f in os.listdir(self.args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not self.image_files:
                raise IOError(f"No images found in directory: {self.args.image_dir}")
            print(f"🖼️ Found {len(self.image_files)} images for processing.")
        else:
            raise ValueError("No input source specified. Use --webcam, --video_file, or --image_dir.")

    def _initialize_models(self):
        """Loads all required machine learning models."""
        print("📦 Loading models...")
        self.yolo_model = YOLO("YOLO_best.pt").to(self.device)
        
        self.vp_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=4)
        vp_model_path = 'mobilevit_viewpoint_20250703.pth'
        if os.path.exists(vp_model_path):
            self.vp_model.load_state_dict(torch.load(vp_model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Required viewpoint model not found: {vp_model_path}")
        self.vp_model.eval().to(self.device)
        self.vp_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
        self.class_names = ['NE', 'NW', 'SE', 'SW'] # Corresponds to the output classes of the viewpoint model
        
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        print("   ...models loaded.")

    def _initialize_anchor_data(self):
        """Pre-processes anchor images and their 2D-3D correspondences."""
        print("🛠️  Initializing anchor data...")
        # Define 2D-3D correspondences for each anchor viewpoint
        # These points are specific to your object and anchor images
        ne_anchor_2d = np.array([[924, 148], [571, 115], [398, 31], [534, 133], [544, 141], [341, 219], [351, 228], [298, 240], [420, 83], [225, 538], [929, 291], [794, 381], [485, 569], [826, 305], [813, 264], [791, 285], [773, 271], [760, 289], [830, 225], [845, 233], [703, 308], [575, 361], [589, 373], [401, 469], [414, 481], [606, 454], [548, 399], [521, 510], [464, 451], [741, 380]], dtype=np.float32)
        ne_anchor_3d = np.array([[-0.0, -0.025, -0.24], [0.23, 0.0, -0.113], [0.243, -0.104, 0.0], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [0.23, 0.0, 0.113], [0.206, -0.07, -0.002], [-0.0, -0.025, 0.24], [-0.08, 0.0, -0.156], [-0.09, 0.0, -0.042], [-0.08, 0.0, 0.156], [-0.052, 0.0, -0.097], [-0.029, 0.0, -0.127], [-0.037, 0.0, -0.097], [-0.017, 0.0, -0.092], [-0.023, 0.0, -0.075], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [-0.074, 0.0, 0.074], [-0.019, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.019, 0.0, 0.128], [-0.1, -0.03, 0.0]], dtype=np.float32)
        nw_anchor_2d = np.array([[511, 293], [591, 284], [587, 330], [413, 249], [602, 348], [715, 384], [598, 298], [656, 171], [805, 213], [703, 392], [523, 286], [519, 327], [387, 289], [727, 126], [425, 243], [636, 358], [745, 202], [595, 388], [436, 260], [539, 313], [795, 220], [351, 291], [665, 165], [611, 353], [650, 377], [516, 389], [727, 143], [496, 378], [575, 312], [617, 368], [430, 312], [480, 281], [834, 225], [469, 339], [705, 223], [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]], dtype=np.float32)
        nw_anchor_3d = np.array([[-0.014, 0.0, 0.042], [0.025, -0.014, -0.011], [-0.014, 0.0, -0.042], [-0.014, 0.0, 0.156], [-0.023, 0.0, -0.065], [0.0, 0.0, -0.156], [0.025, 0.0, -0.015], [0.217, 0.0, 0.07], [0.23, 0.0, -0.07], [-0.014, 0.0, -0.156], [0.0, 0.0, 0.042], [-0.057, -0.018, -0.01], [-0.074, -0.0, 0.128], [0.206, -0.07, -0.002], [-0.0, -0.0, 0.156], [-0.017, -0.0, -0.092], [0.217, -0.0, -0.027], [-0.052, -0.0, -0.097], [-0.019, -0.0, 0.128], [-0.035, -0.018, -0.01], [0.217, -0.0, -0.07], [-0.08, -0.0, 0.156], [0.23, 0.0, 0.07], [-0.023, -0.0, -0.075], [-0.029, -0.0, -0.127], [-0.09, -0.0, -0.042], [0.206, -0.055, -0.002], [-0.09, -0.0, -0.015], [0.0, -0.0, -0.015], [-0.037, -0.0, -0.097], [-0.074, -0.0, 0.074], [-0.019, -0.0, 0.074], [0.23, -0.0, -0.113], [-0.1, -0.03, 0.0], [0.17, -0.0, -0.015], [0.23, -0.0, 0.113], [-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [-0.08, -0.0, -0.156]], dtype=np.float32)
        se_anchor_2d = np.array([[415, 144], [1169, 508], [275, 323], [214, 395], [554, 670], [253, 428], [280, 415], [355, 365], [494, 621], [519, 600], [806, 213], [973, 438], [986, 421], [768, 343], [785, 328], [841, 345], [931, 393], [891, 306], [980, 345], [651, 210], [625, 225], [588, 216], [511, 215], [526, 204], [665, 271]], dtype=np.float32)
        se_anchor_3d = np.array([[-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [0.23, 0.0, -0.113], [0.23, 0.0, 0.113], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.206, -0.07, -0.002], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [-0.1, -0.03, 0.0], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.019, 0.0, 0.074], [-0.019, 0.0, 0.128], [-0.074, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.052, 0.0, -0.097], [-0.037, 0.0, -0.097], [-0.029, 0.0, -0.127], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042]], dtype=np.float32)
        sw_anchor_2d = np.array([[650, 312], [630, 306], [907, 443], [814, 291], [599, 349], [501, 386], [965, 359], [649, 355], [635, 346], [930, 335], [843, 467], [702, 339], [718, 321], [930, 322], [727, 346], [539, 364], [786, 297], [1022, 406], [1004, 399], [539, 344], [536, 309], [864, 478], [745, 310], [1049, 393], [895, 258], [674, 347], [741, 281], [699, 294], [817, 494], [992, 281]], dtype=np.float32)
        sw_anchor_3d = np.array([[-0.035, -0.018, -0.01], [-0.057, -0.018, -0.01], [0.217, -0.0, -0.027], [-0.014, -0.0, 0.156], [-0.023, 0.0, -0.065], [-0.014, -0.0, -0.156], [0.234, -0.05, -0.002], [0.0, -0.0, -0.042], [-0.014, -0.0, -0.042], [0.206, -0.055, -0.002], [0.217, -0.0, -0.07], [0.025, -0.014, -0.011], [-0.014, -0.0, 0.042], [0.206, -0.07, -0.002], [0.049, -0.016, -0.011], [-0.029, -0.0, -0.127], [-0.019, -0.0, 0.128], [0.23, -0.0, 0.07], [0.217, -0.0, 0.07], [-0.052, -0.0, -0.097], [-0.175, -0.0, -0.015], [0.23, -0.0, -0.07], [-0.019, -0.0, 0.074], [0.23, -0.0, 0.113], [-0.0, -0.025, 0.24], [-0.0, -0.0, -0.015], [-0.074, -0.0, 0.128], [-0.074, -0.0, 0.074], [0.23, -0.0, -0.113], [0.243, -0.104, 0.0]], dtype=np.float32)
        
        # Paths to your anchor images. Ensure these paths are correct relative to your script.
        anchor_definitions = {
            'NE': {'path': 'NE.png', '2d': ne_anchor_2d, '3d': ne_anchor_3d},
            'NW': {'path': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png', '2d': nw_anchor_2d, '3d': nw_anchor_3d},
            'SE': {'path': 'SE.png', '2d': se_anchor_2d, '3d': se_anchor_3d},
            'SW': {'path': 'Anchor_B.png', '2d': sw_anchor_2d, '3d': sw_anchor_3d}
        }
        
        self.viewpoint_anchors = {}
        for viewpoint, data in anchor_definitions.items():
            path = data['path']
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required anchor image not found: {path}")
            
            # Load and resize anchor image to match camera resolution
            anchor_image_bgr = cv2.resize(cv2.imread(path), (self.camera_width, self.camera_height))
            # Extract SuperPoint features from the anchor image
            anchor_features = self._extract_features_sp(anchor_image_bgr)
            anchor_keypoints = anchor_features['keypoints'][0].cpu().numpy()
            
            # Find correspondences between manually labeled 2D points and extracted SuperPoint features
            # This step is crucial for mapping 2D image points to known 3D object points.
            sp_tree = cKDTree(anchor_keypoints)
            # Query for the closest SuperPoint feature to each manually labeled 2D point
            distances, indices = sp_tree.query(data['2d'], k=1, distance_upper_bound=5.0) # Max 5 pixel distance
            valid_mask = distances != np.inf # Filter out points without a close SuperPoint feature
            
            self.viewpoint_anchors[viewpoint] = {
                'features': anchor_features, # SuperPoint features of the anchor image
                # Map SuperPoint feature index to its corresponding 3D object point
                'map_3d': {idx: pt for idx, pt in zip(indices[valid_mask], data['3d'][valid_mask])}
            }
        print("   ...anchor data initialized.")

    def _get_next_frame(self):
        """Fetches the next frame from the configured input source."""
        if self.is_video_stream:
            ret, frame = self.video_capture.read()
            if not ret:
                self.running = False
                return None
            return frame
        else: # Image directory
            if self.frame_idx < len(self.image_files):
                frame = cv2.imread(self.image_files[self.frame_idx])
                self.frame_idx += 1
                return frame
            else:
                self.running = False
                return None
    
    def run(self):
        """The main processing loop."""
        frame_count = 0
        start_time = time.time()

        while self.running:
            frame = self._get_next_frame()
            if frame is None:
                break
            
            # Process the current frame and get the result
            result = self._process_frame(frame, frame_count)
            
            # Log the pose data for later analysis
            self.all_poses_log.append({
                'frame': frame_count,
                'success': result.pose_success,
                'position': result.position.tolist() if result.position is not None else None,
                'quaternion': result.quaternion.tolist() if result.quaternion is not None else None,
                'kf_position': result.kf_position.tolist() if result.kf_position is not None else None,
                'kf_quaternion': result.kf_quaternion.tolist() if result.kf_quaternion is not None else None,
                'num_inliers': result.num_inliers,
                'viewpoint_used': result.viewpoint_used # Log the viewpoint used
            })

            # Calculate running FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            self._update_display(result, fps)
        
        self.cleanup()

    def _process_frame(self, frame: np.ndarray, frame_id: int) -> ProcessingResult:
        """Runs the full pose estimation pipeline on a single frame, including pre-filtering."""
        current_frame_copy = frame.copy() # Create a copy for processing and visualization
        
        # Initialize result object with default values (no pose success)
        result = ProcessingResult(frame_id=frame_id, frame=current_frame_copy, pose_success=False)
        
        # Predict step for Kalman Filter (always predict, even if measurement is rejected)
        # This provides a continuous estimate even when no new measurements are integrated.
        kf_predicted_pos, kf_predicted_quat = None, None
        if self.use_kalman_filter:
            kf_predicted_pos, kf_predicted_quat = self.kf.predict()
            result.kf_position = kf_predicted_pos
            result.kf_quaternion = kf_predicted_quat
            
        # 1. Object Detection
        bbox = self._yolo_detect(current_frame_copy)
        result.bbox = bbox
        
        # Prepare crop for viewpoint classification
        crop = current_frame_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else current_frame_copy
        initial_viewpoint = self._classify_viewpoint(crop) if self.use_viewpoint else 'NW' # Default if not using VP
        
        # 2. Feature Matching and Pose Estimation (PnP)
        # This function tries to find the best pose among all viewpoints
        best_pose = self._estimate_pose_with_fallback(current_frame_copy, initial_viewpoint, bbox)
        
        # --- Pre-Filtering Checks ---
        is_current_measurement_valid = False
        if best_pose:
            current_position = best_pose.position
            current_quaternion = best_pose.quaternion
            current_viewpoint_used = best_pose.viewpoint # The actual viewpoint that yielded this pose

            # 1. Orientation Jump Check
            if self.last_orientation is not None:
                angle_diff = math.degrees(self.quaternion_angle_diff(self.last_orientation, current_quaternion))
                if angle_diff > self.ORI_MAX_DIFF_DEG:
                    print(f"🚫 Frame {frame_id}: Rejected (Orientation Jump: {angle_diff:.1f}° > {self.ORI_MAX_DIFF_DEG}°)")
                else:
                    is_current_measurement_valid = True # Passed this check
            else:
                # If no previous orientation, this is the first valid measurement, so accept it
                is_current_measurement_valid = True
        
        # --- Apply Measurement (if valid) or use Prediction ---
        if is_current_measurement_valid and best_pose: # Ensure best_pose is not None here
            # Measurement is valid, update Kalman Filter and result
            self.rejected_consecutive_frames_count = 0 # Reset rejection counter

            result.position = best_pose.position
            result.quaternion = best_pose.quaternion
            result.num_inliers = best_pose.inliers
            result.pose_success = True
            result.viewpoint_used = best_pose.viewpoint

            # Update last known good pose for subsequent checks
            self.last_orientation = best_pose.quaternion
            
            if self.use_kalman_filter:
                kf_pos, kf_quat = self.kf.update(best_pose.position, best_pose.quaternion)
                result.kf_position = kf_pos
                result.kf_quaternion = kf_quat
        else:
            # Pose estimation failed or was rejected by pre-filters.
            self.rejected_consecutive_frames_count += 1
            print(f"ℹ️ Frame {frame_id}: No valid pose found or measurement rejected. Consecutive rejections: {self.rejected_consecutive_frames_count}")

            # Check for re-initialization condition
            if self.rejected_consecutive_frames_count >= self.MAX_REJECTED_FRAMES:
                print(f"⚠️ Frame {frame_id}: Exceeded {self.MAX_REJECTED_FRAMES} consecutive rejections. Re-initializing Kalman Filter and last known pose.")
                self.kf.initialized = False # Reset Kalman filter
                self.last_orientation = None # Reset last valid orientation
                self.rejected_consecutive_frames_count = 0 # Reset counter after re-init

            # Use Kalman Filter's prediction for display/logging if available
            result.position = None
            result.quaternion = None
            result.num_inliers = 0
            result.pose_success = False
            result.viewpoint_used = initial_viewpoint # Still log the initial viewpoint attempt for context

            # kf_predicted_pos and kf_predicted_quat are already set from the initial predict call
            # They will be used in result.kf_position and result.kf_quaternion

        return result

    def _estimate_pose_with_fallback(self, frame: np.ndarray, initial_viewpoint: str, bbox: Optional[Tuple]) -> Optional[PoseData]:
        """Tries initial viewpoint, then falls back to others, returning the best valid pose."""
        all_viewpoints = ['NW', 'NE', 'SE', 'SW']
        # Prioritize the viewpoint predicted by the classifier
        viewpoints_to_try = [initial_viewpoint] + [vp for vp in all_viewpoints if vp != initial_viewpoint]
        
        successful_poses = []
        for viewpoint in viewpoints_to_try:
            pose_data = self._solve_for_viewpoint(frame, viewpoint, bbox)
            if pose_data:
                successful_poses.append(pose_data)
        
        if not successful_poses:
            return None
        
        # Return the pose with the most inliers as the "best" candidate
        return max(successful_poses, key=lambda p: p.inliers)

    def _solve_for_viewpoint(self, frame: np.ndarray, viewpoint: str, bbox: Optional[Tuple]) -> Optional[PoseData]:
        """Attempts to solve the pose for a single given viewpoint using PnP."""
        anchor = self.viewpoint_anchors.get(viewpoint)
        if not anchor: return None # No anchor data for this viewpoint
        
        # Crop the frame based on the detected bounding box
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else frame
        # Calculate offset for keypoints if a crop was used
        crop_offset = np.array([bbox[0], bbox[1]]) if bbox else np.array([0, 0])
        
        if crop.size == 0: # Check if crop is empty
            return None
        
        # Extract SuperPoint features from the current frame/crop
        frame_features = self._extract_features_sp(crop)
        
        # Match features between anchor image and current frame
        with torch.no_grad():
            matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
        matches = rbd(matches_dict)['matches'].cpu().numpy()
        
        # Need at least 6 matches for solvePnP
        if len(matches) < 6:
            # print(f"Debug: Not enough matches ({len(matches)}) for viewpoint {viewpoint}")
            return None

        # Build 2D-3D point correspondences for PnP
        points_3d, points_2d = [], []
        for anchor_idx, frame_idx in matches:
            # Only use matches where the anchor keypoint has a known 3D correspondence
            if anchor_idx in anchor['map_3d']:
                points_3d.append(anchor['map_3d'][anchor_idx])
                # Add crop offset to 2D keypoints to get original frame coordinates
                points_2d.append(frame_features['keypoints'][0].cpu().numpy()[frame_idx] + crop_offset)
        
        # Need at least 6 corresponding points for solvePnP
        if len(points_3d) < 6:
            # print(f"Debug: Not enough 3D-2D correspondences ({len(points_3d)}) for viewpoint {viewpoint}")
            return None
        
        # Convert lists to numpy arrays for OpenCV
        points_3d_np = np.array(points_3d, dtype=np.float32)
        points_2d_np = np.array(points_2d, dtype=np.float32)

        # Solve PnP (Perspective-n-Point) to get camera pose
        K, dist_coeffs = self._get_camera_intrinsics()
        try:
            # SOLVEPNP_EPNP is generally robust and fast
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d_np, points_2d_np, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
            )

            if success and len(inliers) > 4:
                obj_inliers = points_3d_np[inliers.flatten()]
                img_inliers = points_2d_np[inliers.flatten()]
                rvec, tvec = cv2.solvePnPRefineVVS(
                    objectPoints=obj_inliers,imagePoints=img_inliers,cameraMatrix=K,distCoeffs=dist_coeffs,rvec=rvec,tvec=tvec
                )

        except cv2.error as e:
            print(f"PnP Error for viewpoint {viewpoint}: {e}")
            return None

        # Check for PnP success and sufficient inliers
        if not success or inliers is None or len(inliers) < 4: # At least 4 inliers for a valid pose
            # print(f"Debug: PnP failed or too few inliers ({len(inliers) if inliers is not None else 0}) for viewpoint {viewpoint}")
            return None

        # Convert rotation vector to rotation matrix, then to quaternion
        R, _ = cv2.Rodrigues(rvec)
        position = tvec.flatten() # Flatten translation vector to 1D array
        quaternion = self._rotation_matrix_to_quaternion(R)
        
        # Calculate reprojection error for quality assessment
        # Project the 3D inlier points back to 2D using the estimated pose
        projected_points, _ = cv2.projectPoints(points_3d_np[inliers.flatten()], rvec, tvec, K, dist_coeffs)
        # Calculate the mean Euclidean distance between projected and detected 2D points
        error = np.mean(np.linalg.norm(points_2d_np[inliers.flatten()].reshape(-1, 1, 2) - projected_points, axis=2))
        
        # Return a PoseData object with all relevant information
        return PoseData(position, quaternion, len(inliers), error, viewpoint)
    
    # --- Helper & Utility Methods for Pre-Filtering ---

    def quaternion_angle_diff(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Computes the smallest angle (in radians) between two unit quaternions.
        This is used for the 'sudden orientation jump' check.
        """
        # Ensure quaternions are unit length
        q1_norm = q1 / np.linalg.norm(q1)
        q2_norm = q2 / np.linalg.norm(q2)
        
        # Compute dot product
        dot = np.dot(q1_norm, q2_norm)
        
        # Clamp dot product to valid acos range [-1, 1] to prevent numerical errors
        dot = max(-1.0, min(1.0, dot))
        
        # Use absolute dot product to get the shortest angle (0 to pi radians)
        angle = 2 * math.acos(abs(dot))
        return angle # in radians

    # --- Existing Helper & Utility Methods ---

    def _yolo_detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detects the object in the frame and returns a bounding box (x1, y1, x2, y2)."""
        results = self.yolo_model(frame, verbose=False, conf=0.5) # conf=0.5 is confidence threshold
        if len(results[0].boxes) > 0:
            # Return the first detected bounding box
            return tuple(map(int, results[0].boxes.xyxy.cpu().numpy()[0]))
        return None

    def _classify_viewpoint(self, crop: np.ndarray) -> str:
        """Classifies the viewpoint from a cropped image using the MobileViT model."""
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return 'NW' # Default viewpoint if crop is empty (e.g., no detection)
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = self.vp_transform(pil_img).unsqueeze(0).to(self.device) # Add batch dimension
        with torch.no_grad():
            # Get model prediction and convert to class name
            output = self.vp_model(tensor)
            predicted_class_idx = torch.argmax(output, dim=1).item()
            return self.class_names[predicted_class_idx]
    
    def _extract_features_sp(self, image_bgr: np.ndarray) -> Dict:
        """Extracts SuperPoint features (keypoints and descriptors) from a BGR image."""
        # Convert BGR to RGB, then to a normalized PyTorch tensor
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            return self.extractor.extract(tensor)

    def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
        """Returns the camera intrinsic matrix (K) and distortion coefficients (dist_coeffs)."""
        # These values are specific to your camera calibration
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        # Assuming no distortion coefficients for simplicity, or provide your calibrated values
        dist_coeffs = None # np.zeros((4, 1), dtype=np.float32) if you have them
        return K, dist_coeffs

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Converts a 3x3 rotation matrix to a quaternion (x, y, z, w)."""
        # This is a standard conversion algorithm
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return np.array([qx, qy, qz, qw]) # Return as (x, y, z, w)

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Converts a quaternion (x, y, z, w) to a 3x3 rotation matrix."""
        # Ensure quaternion is normalized before conversion
        q_norm = q / np.linalg.norm(q)
        x, y, z, w = q_norm
        
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])

    def _update_display(self, result: ProcessingResult, fps: float):
        """Prepares and queues a frame for the display thread with OSD."""
        vis_frame = result.frame # Get the frame from the processing result
        
        # Draw bounding box if detected
        if result.bbox:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Use the smoothed KF pose for visualization if available and valid,
        # otherwise use the raw pose if valid, else nothing.
        display_pos = result.kf_position if result.kf_position is not None else result.position
        display_quat = result.kf_quaternion if result.kf_quaternion is not None else result.quaternion

        if display_pos is not None and display_quat is not None:
            self._draw_axes(vis_frame, display_pos, display_quat)

        # --- On-screen Display (OSD) Text ---
        status_color = (0, 255, 0) if result.pose_success else (0, 0, 255) # Green for success, Red for failure
        status_text = "SUCCESS" if result.pose_success else "TRACKING FAILED"
        
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(vis_frame, f"STATUS: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(vis_frame, f"Inliers: {result.num_inliers}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if result.viewpoint_used:
            cv2.putText(vis_frame, f"Viewpoint: {result.viewpoint_used}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        try:
            # Non-blocking put to avoid delays if display is lagging
            self.display_queue.put_nowait(vis_frame)
        except queue.Full:
            pass # Skip frame if display queue is full

    def _draw_axes(self, frame: np.ndarray, position: np.ndarray, quaternion: np.ndarray):
        """Draws a 3D coordinate axis (X=Red, Y=Green, Z=Blue) on the frame at the estimated pose."""
        try:
            R = self._quaternion_to_rotation_matrix(quaternion)
            rvec, _ = cv2.Rodrigues(R) # Convert rotation matrix back to rotation vector
            tvec = position.reshape(3, 1) # Reshape position to column vector
            K, _ = self._get_camera_intrinsics() # Get camera intrinsics
            
            # Define 3D points for the axes (origin and endpoints of X, Y, Z axes)
            # Length of axes is 0.1 meters (10 cm)
            axis_pts = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
            
            # Project 3D axis points to 2D image points
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, None) # None for dist_coeffs
            img_pts = img_pts.reshape(-1, 2).astype(int) # Reshape and convert to integer coordinates
            
            origin = tuple(img_pts[0]) # Origin of the axes
            
            # Draw lines for X, Y, Z axes
            cv2.line(frame, origin, tuple(img_pts[1]), (0,0,255), 3)  # X-axis (Red)
            cv2.line(frame, origin, tuple(img_pts[2]), (0,255,0), 3)  # Y-axis (Green)
            cv2.line(frame, origin, tuple(img_pts[3]), (255,0,0), 3)  # Z-axis (Blue)
        except (cv2.error, AttributeError, ValueError) as e:
            # print(f"Error drawing axes: {e}") # Log drawing errors if needed
            pass # Ignore drawing errors to keep the pipeline running

    def _display_loop(self):
        """Dedicated thread for rendering frames from the queue to a CV2 window."""
        window_name = "Real-time Pose Estimation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Create resizable window
        cv2.resizeWindow(window_name, 1280, 720) # Set initial window size

        while self.running:
            try:
                # Get frame from queue with a timeout to allow thread to exit gracefully
                frame = self.display_queue.get(timeout=1.0)
                cv2.imshow(window_name, frame)
            except queue.Empty:
                if not self.running: break # Exit if main loop has stopped
                continue # Continue waiting for frames
            
            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False # Signal main loop to stop

    def cleanup(self):
        """Releases resources and saves logs upon shutdown."""
        print("\nShutting down...")
        self.running = False # Ensure all threads know to stop
        
        # Wait for the display thread to finish (with a timeout)
        self.display_thread.join(timeout=2.0)
        
        # Release video capture if it was used
        if self.is_video_stream and self.video_capture:
            self.video_capture.release()
        
        cv2.destroyAllWindows() # Close all OpenCV windows

        # Save the pose log if requested
        if self.args.save_output:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist
            filename = os.path.join(output_dir, f"pose_log_{time.strftime('%Y%m%d-%H%M%S')}.json")
            with open(filename, 'w') as f:
                json.dump(self.all_poses_log, f, indent=4) # Save log in pretty-printed JSON format
            print(f"💾 Pose log saved to {filename}")

def main():
    """Main function to parse arguments and run the Pose Estimator."""
    parser = argparse.ArgumentParser(description="VAPE MK47 - Real-time Pose Estimator")
    
    # Mutually exclusive group for input source (only one can be selected)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--webcam', action='store_true', help='Use webcam as input.')
    group.add_argument('--video_file', type=str, help='Path to a video file.')
    group.add_argument('--image_dir', type=str, help='Path to a directory of images.')
    
    parser.add_argument('--save_output', action='store_true', help='Save the final pose data to a JSON file.')
    args = parser.parse_args()

    try:
        estimator = PoseEstimator(args)
        estimator.run() # Start the main processing loop
    except (IOError, FileNotFoundError) as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C).")
    finally:
        print("✅ Process finished.")

if __name__ == '__main__':
    main()
