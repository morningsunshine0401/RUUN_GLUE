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
import math

# --- DEPENDENCY IMPORTS ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 VAPE MK50 Pose Estimator (No MobileViT)")
try:
    from ultralytics import YOLO
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
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
# class LooselyCoupledKalmanFilter:
#     """A Kalman Filter with a constant acceleration motion model."""
#     def __init__(self, dt=1/30.0):
#         self.dt = dt
#         self.initialized = False
#         # --- MODIFIED: New state vector size for acceleration (16 states) ---
#         self.n_states = 16 # [pos(3), vel(3), acc(3), quat(4), ang_vel(3)]
#         self.x = np.zeros(self.n_states)
#         self.x[9] = 1.0  # Identity quaternion (w,x,y,z or x,y,z,w)
#         self.P = np.eye(self.n_states) * 0.1 # Initial covariance
#         self.Q = np.eye(self.n_states) * 1e-3 # Process noise covariance
#         # --- Measurement is still 7 states ---
#         self.R = np.eye(7) * 1e-4 # Measurement noise covariance (for 3D pos + 4D quat)

#     def normalize_quaternion(self, q):
#         """Normalizes a quaternion to unit length."""
#         norm = np.linalg.norm(q)
#         return q / norm if norm > 1e-10 else np.array([0, 0, 0, 1])

#     def predict(self):
#         """
#         Predicts the next state using a constant acceleration model.
#         Returns predicted position and quaternion.
#         """
#         if not self.initialized:
#             # Return a safe, uninitialized state
#             return np.zeros(3), np.array([0, 0, 0, 1])
        
#         dt = self.dt
        
#         # --- MODIFIED: State transition matrix F with acceleration terms ---
#         F = np.eye(self.n_states)
#         # Position updates based on velocity and acceleration
#         # P_new = P_old + V_old*dt + 0.5*A_old*dt^2
#         F[0:3, 3:6] = np.eye(3) * dt
#         F[0:3, 6:9] = np.eye(3) * 0.5 * dt**2
#         # Velocity updates based on acceleration
#         # V_new = V_old + A_old*dt
#         F[3:6, 6:9] = np.eye(3) * dt
        
#         # Update state vector x
#         self.x = F @ self.x
        
#         # Quaternion update based on angular velocity (simplified)
#         # The quaternion and angular velocity are now at a different index
#         q, w = self.x[9:13], self.x[13:16] 
        
#         # Skew-symmetric matrix for quaternion multiplication
#         omega_mat = 0.5 * np.array([
#             [0, -w[0], -w[1], -w[2]],
#             [w[0], 0, w[2], -w[1]],
#             [w[1], -w[2], 0, w[0]],
#             [w[2], w[1], -w[0], 0]
#         ])
#         self.x[9:13] = self.normalize_quaternion((np.eye(4) + dt * omega_mat) @ q)
        
#         # Update covariance matrix P
#         self.P = F @ self.P @ F.T + self.Q
        
#         # Return predicted position and quaternion (at new indices)
#         return self.x[0:3], self.x[9:13]

#     def update(self, position: np.ndarray, quaternion: np.ndarray):
#         """
#         Updates the Kalman filter state with a new measurement.
#         """
#         measurement = np.concatenate([position, self.normalize_quaternion(quaternion)])
        
#         if not self.initialized:
#             # Initialize state with the first measurement
#             self.x[0:3] = position
#             # --- MODIFIED: Quaternion is at a new index ---
#             self.x[9:13] = self.normalize_quaternion(quaternion)
#             self.initialized = True
#             return self.x[0:3], self.x[9:13]
        
#         # Measurement matrix H maps state to measurement
#         H = np.zeros((7, self.n_states))
#         H[0:3, 0:3] = np.eye(3) # Position part
#         # --- MODIFIED: Quaternion is at a new index ---
#         H[3:7, 9:13] = np.eye(4) # Quaternion part
        
#         innovation = measurement - H @ self.x
        
#         # Handle quaternion sign ambiguity
#         if np.dot(measurement[3:7], self.x[9:13]) < 0:
#             measurement[3:7] *= -1
#             innovation = measurement - H @ self.x
            
#         S = H @ self.P @ H.T + self.R
#         K = self.P @ H.T @ np.linalg.inv(S)
        
#         self.x += K @ innovation
#         self.x[9:13] = self.normalize_quaternion(self.x[9:13])
        
#         self.P = (np.eye(self.n_states) - K @ H) @ self.P
        
#         # Return updated (smoothed) position and quaternion
#         return self.x[0:3], self.x[9:13]


class LooselyCoupledKalmanFilter:
    """A Kalman Filter with a constant jerk motion model for translation."""
    def __init__(self, dt=1/30.0):
        self.dt = dt
        self.initialized = False
        # --- MODIFIED: New state vector size for jerk (19 states) ---
        self.n_states = 19  # [pos(3), vel(3), acc(3), jerk(3), quat(4), ang_vel(3)]
        self.x = np.zeros(self.n_states)
        self.x[12] = 1.0  # Identity quaternion (w,x,y,z or x,y,z,w)
        self.P = np.eye(self.n_states) * 0.1 # Initial covariance
        self.Q = np.eye(self.n_states) * 1e-3 # Process noise covariance
        self.R = np.eye(7) * 1e-4  # Measurement noise covariance (for 3D pos + 4D quat)

    def normalize_quaternion(self, q):
        """Normalizes a quaternion to unit length."""
        norm = np.linalg.norm(q)
        return q / norm if norm > 1e-10 else np.array([0, 0, 0, 1])

    def predict(self):
        """
        Predicts the next state using a constant jerk model.
        Returns predicted position and quaternion.
        """
        if not self.initialized:
            # Return a safe, uninitialized state
            return np.zeros(3), np.array([0, 0, 0, 1])

        dt = self.dt
        
        # --- MODIFIED: State transition matrix F with jerk terms ---
        F = np.eye(self.n_states)
        
        # Position updates based on vel, acc, and jerk: P_new = P_old + V*dt + 0.5*A*dt^2 + (1/6)*J*dt^3
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt**2
        F[0:3, 9:12] = np.eye(3) * (1/6.0) * dt**3
        
        # Velocity updates based on acc and jerk: V_new = V_old + A*dt + 0.5*J*dt^2
        F[3:6, 6:9] = np.eye(3) * dt
        F[3:6, 9:12] = np.eye(3) * 0.5 * dt**2
        
        # Acceleration updates based on jerk: A_new = A_old + J*dt
        F[6:9, 9:12] = np.eye(3) * dt
        
        # Update state vector x
        self.x = F @ self.x
        
        # Quaternion update based on angular velocity (simplified). The indices have changed.
        q, w = self.x[12:16], self.x[16:19]
        
        omega_mat = 0.5 * np.array([
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0]
        ])
        self.x[12:16] = self.normalize_quaternion((np.eye(4) + dt * omega_mat) @ q)
        
        # Update covariance matrix P
        self.P = F @ self.P @ F.T + self.Q
        
        # Return predicted position and quaternion (at new indices)
        return self.x[0:3], self.x[12:16]

    def update(self, position: np.ndarray, quaternion: np.ndarray):
        """
        Updates the Kalman filter state with a new measurement.
        """
        measurement = np.concatenate([position, self.normalize_quaternion(quaternion)])
        
        if not self.initialized:
            self.x[0:3] = position
            # --- MODIFIED: Quaternion is at a new index ---
            self.x[12:16] = self.normalize_quaternion(quaternion)
            self.initialized = True
            return self.x[0:3], self.x[12:16]
        
        # Measurement matrix H maps state to measurement
        H = np.zeros((7, self.n_states))
        H[0:3, 0:3] = np.eye(3) # Position part
        # --- MODIFIED: Quaternion is at a new index ---
        H[3:7, 12:16] = np.eye(4) # Quaternion part
        
        innovation = measurement - H @ self.x
        
        # Handle quaternion sign ambiguity
        if np.dot(measurement[3:7], self.x[12:16]) < 0:
            measurement[3:7] *= -1
            innovation = measurement - H @ self.x
            
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x += K @ innovation
        self.x[12:16] = self.normalize_quaternion(self.x[12:16])
        
        self.P = (np.eye(self.n_states) - K @ H) @ self.P
        
        # Return updated (smoothed) position and quaternion
        return self.x[0:3], self.x[12:16]
    


# --- NEW: High-Frequency Main Thread ---
class MainThread(threading.Thread):
    def __init__(self, processing_queue, pose_data_lock, kf, args):
        super().__init__()
        self.running = True
        self.processing_queue = processing_queue
        self.pose_data_lock = pose_data_lock
        self.kf = kf
        self.args = args

        self.camera_width, self.camera_height = 1280, 720
        self.is_video_stream = False
        self.video_capture = None
        self.image_files = []
        self.frame_idx = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        self._initialize_input_source()
        self.K, self.dist_coeffs = self._get_camera_intrinsics()

    def _initialize_input_source(self):
        """Initializes the input source (webcam, video, or image folder)."""
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

    # In your MainThread class's run method:

    def run(self):
        window_name = "VAPE MK50 - Real-time Pose Estimation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.camera_width, self.camera_height)

        while self.running:
            start_time = time.time()  # Start the timer for FPS calculation and capping

            # Get a new frame from the camera/video source
            frame = self._get_next_frame()
            if frame is None:
                self.running = False
                break

            # 1. Kalman Filter Prediction (continuous, high-frequency)
            predicted_pose_rvec, predicted_pose_tvec, predicted_pose_quat = None, None, None
            with self.pose_data_lock:
                predicted_pose_tvec, predicted_pose_quat = self.kf.predict()
                # Convert quaternion to rvec for drawing
                predicted_pose_rvec, _ = cv2.Rodrigues(self._quaternion_to_rotation_matrix(predicted_pose_quat))

            # 2. Visualization
            vis_frame = frame.copy()
            if predicted_pose_tvec is not None and predicted_pose_quat is not None:
                self._draw_axes(vis_frame, predicted_pose_tvec, predicted_pose_quat)
            
            # Draw OSD
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(vis_frame, "STATUS: PREDICITING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

            # 3. Send frame to the processing thread
            # Keep queue small to avoid lag, drop frames if necessary
            if self.processing_queue.qsize() < 2:
                self.processing_queue.put(frame.copy())

            cv2.imshow(window_name, vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                print("User requested shutdown.")
                break

            # --- NEW: Cap the frame rate ---
            # Enforce a consistent frame rate, e.g., 30 FPS.
            # This is critical for video file playback which can run much faster than 30 FPS.
            frame_rate_cap = 30.0
            time_to_wait = 1.0 / frame_rate_cap - (time.time() - start_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)
        
        self.cleanup()

    def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = None
        return K, dist_coeffs

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        q_norm = q / np.linalg.norm(q)
        x, y, z, w = q_norm
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
    
    def _draw_axes(self, frame: np.ndarray, position: np.ndarray, quaternion: np.ndarray):
        """Draws a 3D coordinate axis (X=Red, Y=Green, Z=Blue) on the frame at the estimated pose."""
        try:
            R = self._quaternion_to_rotation_matrix(quaternion)
            rvec, _ = cv2.Rodrigues(R)
            tvec = position.reshape(3, 1)
            axis_pts = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, self.K, self.dist_coeffs)
            img_pts = img_pts.reshape(-1, 2).astype(int)
            origin = tuple(img_pts[0])
            cv2.line(frame, origin, tuple(img_pts[1]), (0,0,255), 3)  # X-axis (Red)
            cv2.line(frame, origin, tuple(img_pts[2]), (0,255,0), 3)  # Y-axis (Green)
            cv2.line(frame, origin, tuple(img_pts[3]), (255,0,0), 3)  # Z-axis (Blue)
        except (cv2.error, AttributeError, ValueError) as e:
            pass

    def cleanup(self):
        """Releases resources upon shutdown."""
        self.running = False
        if self.is_video_stream and self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()


# --- NEW: Low-Frequency Processing Thread ---
class ProcessingThread(threading.Thread):
    def __init__(self, processing_queue, pose_data_lock, kf, args):
        super().__init__()
        self.running = True
        self.processing_queue = processing_queue
        self.pose_data_lock = pose_data_lock
        self.kf = kf
        self.args = args

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.camera_width, self.camera_height = 1280, 720
        self.all_poses_log = []
        
        # --- Temporal Consistency for Viewpoint Selection ---
        self.current_best_viewpoint = None
        self.viewpoint_quality_threshold = 6  # minimum inliers to keep using current viewpoint
        self.consecutive_failures = 0
        self.max_failures_before_search = 5  # frames before searching all viewpoints again
        # ---------------------------------------------------

        # --- Pre-filtering related attributes ---
        self.last_orientation: Optional[np.ndarray] = None # Stores the last accepted quaternion
        self.ORI_MAX_DIFF_DEG = 30.0  # Max allowed orientation change per frame in degrees
        self.rejected_consecutive_frames_count = 0 # Counter for consecutive rejected frames
        self.MAX_REJECTED_FRAMES = 45 # Number of consecutive rejected frames before re-initialization
        # ----------------------------------------
        
        self.yolo_model = None
        self.extractor = None
        self.matcher = None
        self.viewpoint_anchors = {}

        self._initialize_models()
        self._initialize_anchor_data()
        self.K, self.dist_coeffs = self._get_camera_intrinsics()

    def _initialize_models(self):
        """Loads all required machine learning models."""
        print("📦 Loading models...")
        self.yolo_model = YOLO("yolo_coco_pretrained.pt").to(self.device)
        self.extractor = SuperPoint(max_num_keypoints=4096).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        print("   ...models loaded.")
    
    def _initialize_anchor_data(self):
        """Pre-processes anchor images and their 2D-3D correspondences."""
        print("🛠️  Initializing anchor data...")
        ne_anchor_2d = np.array([[924, 148], [571, 115], [398, 31], [534, 133], [544, 141], [341, 219], [351, 228], [298, 240], [420, 83], [225, 538], [929, 291], [794, 381], [485, 569], [826, 305], [813, 264], [791, 285], [773, 271], [760, 289], [830, 225], [845, 233], [703, 308], [575, 361], [589, 373], [401, 469], [414, 481], [606, 454], [548, 399], [521, 510], [464, 451], [741, 380]], dtype=np.float32)
        ne_anchor_3d = np.array([[-0.0, -0.025, -0.24], [0.23, 0.0, -0.113], [0.243, -0.104, 0.0], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [0.23, 0.0, 0.113], [0.206, -0.07, -0.002], [-0.0, -0.025, 0.24], [-0.08, 0.0, -0.156], [-0.09, 0.0, -0.042], [-0.08, 0.0, 0.156], [-0.052, 0.0, -0.097], [-0.029, 0.0, -0.127], [-0.037, 0.0, -0.097], [-0.017, 0.0, -0.092], [-0.023, 0.0, -0.075], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [-0.074, 0.0, 0.074], [-0.019, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.019, 0.0, 0.128], [-0.1, -0.03, 0.0]], dtype=np.float32)
        nw_anchor_2d = np.array([[511, 293], [591, 284], [587, 330], [413, 249], [602, 348], [715, 384], [598, 298], [656, 171], [805, 213], [703, 392], [523, 286], [519, 327], [387, 289], [727, 126], [425, 243], [636, 358], [745, 202], [595, 388], [436, 260], [539, 313], [795, 220], [351, 291], [665, 165], [611, 353], [650, 377], [516, 389], [727, 143], [496, 378], [575, 312], [617, 368], [430, 312], [480, 281], [834, 225], [469, 339], [705, 223], [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]], dtype=np.float32)
        nw_anchor_3d = np.array([[-0.014, 0.0, 0.042], [0.025, -0.014, -0.011], [-0.014, 0.0, -0.042], [-0.014, 0.0, 0.156], [-0.023, 0.0, -0.065], [0.0, 0.0, -0.156], [0.025, 0.0, -0.015], [0.217, 0.0, 0.07], [0.23, 0.0, -0.07], [-0.014, 0.0, -0.156], [0.0, 0.0, 0.042], [-0.057, -0.018, -0.01], [-0.074, -0.0, 0.128], [0.206, -0.07, -0.002], [-0.0, -0.0, 0.156], [-0.017, -0.0, -0.092], [0.217, -0.0, -0.027], [-0.052, -0.0, -0.097], [-0.019, -0.0, 0.128], [-0.035, -0.018, -0.01], [0.217, -0.0, -0.07], [-0.08, -0.0, 0.156], [0.23, 0.0, 0.07], [-0.023, -0.0, -0.075], [-0.029, -0.0, -0.127], [-0.09, -0.0, -0.042], [0.206, -0.055, -0.002], [-0.09, -0.0, -0.015], [0.0, -0.0, -0.015], [-0.037, -0.0, -0.097], [-0.074, -0.0, 0.074], [-0.019, -0.0, 0.074], [0.23, -0.0, -0.113], [-0.1, -0.03, 0.0], [0.17, -0.0, -0.015], [0.23, -0.0, 0.113], [-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [-0.08, -0.0, -0.156]], dtype=np.float32)
        se_anchor_2d = np.array([[415, 144], [1169, 508], [275, 323], [214, 395], [554, 670], [253, 428], [280, 415], [355, 365], [494, 621], [519, 600], [806, 213], [973, 438], [986, 421], [768, 343], [785, 328], [841, 345], [931, 393], [891, 306], [980, 345], [651, 210], [625, 225], [588, 216], [511, 215], [526, 204], [665, 271]], dtype=np.float32)
        se_anchor_3d = np.array([[-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [0.23, 0.0, -0.113], [0.23, 0.0, 0.113], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.206, -0.07, -0.002], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [-0.1, -0.03, 0.0], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.019, 0.0, 0.074], [-0.019, 0.0, 0.128], [-0.074, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.052, 0.0, -0.097], [-0.037, 0.0, -0.097], [-0.029, 0.0, -0.127], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042]], dtype=np.float32)
        sw_anchor_2d = np.array([[650, 312], [630, 306], [907, 443], [814, 291], [599, 349], [501, 386], [965, 359], [649, 355], [635, 346], [930, 335], [843, 467], [702, 339], [718, 321], [930, 322], [727, 346], [539, 364], [786, 297], [1022, 406], [1004, 399], [539, 344], [536, 309], [864, 478], [745, 310], [1049, 393], [895, 258], [674, 347], [741, 281], [699, 294], [817, 494], [992, 281]], dtype=np.float32)
        sw_anchor_3d = np.array([[-0.035, -0.018, -0.01], [-0.057, -0.018, -0.01], [0.217, -0.0, -0.027], [-0.014, -0.0, 0.156], [-0.023, 0.0, -0.065], [-0.014, -0.0, -0.156], [0.234, -0.05, -0.002], [0.0, -0.0, -0.042], [-0.014, -0.0, -0.042], [0.206, -0.055, -0.002], [0.217, -0.0, -0.07], [0.025, -0.014, -0.011], [-0.014, -0.0, 0.042], [0.206, -0.07, -0.002], [0.049, -0.016, -0.011], [-0.029, -0.0, -0.127], [-0.019, -0.0, 0.128], [0.23, -0.0, 0.07], [0.217, -0.0, 0.07], [-0.052, -0.0, -0.097], [-0.175, -0.0, -0.015], [0.23, -0.0, -0.07], [-0.019, -0.0, 0.074], [0.23, -0.0, 0.113], [-0.0, -0.025, 0.24], [-0.0, -0.0, -0.015], [-0.074, -0.0, 0.128], [-0.074, -0.0, 0.074], [0.23, -0.0, -0.113], [0.243, -0.104, 0.0]], dtype=np.float32)

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
            
            anchor_image_bgr = cv2.resize(cv2.imread(path), (self.camera_width, self.camera_height))
            anchor_features = self._extract_features_sp(anchor_image_bgr)
            anchor_keypoints = anchor_features['keypoints'][0].cpu().numpy()
            sp_tree = cKDTree(anchor_keypoints)
            distances, indices = sp_tree.query(data['2d'], k=1, distance_upper_bound=5.0)
            valid_mask = distances != np.inf
            
            self.viewpoint_anchors[viewpoint] = {
                'features': anchor_features,
                'map_3d': {idx: pt for idx, pt in zip(indices[valid_mask], data['3d'][valid_mask])}
            }
        print("   ...anchor data initialized.")

    def run(self):
        frame_id = 0
        while self.running:
            if not self.processing_queue.empty():
                frame = self.processing_queue.get()
                
                # --- The core pose estimation pipeline logic ---
                result = self._process_frame(frame, frame_id)
                frame_id += 1
                
                self.all_poses_log.append({
                    'frame': result.frame_id,
                    'success': result.pose_success,
                    'position': result.position.tolist() if result.position is not None else None,
                    'quaternion': result.quaternion.tolist() if result.quaternion is not None else None,
                    'kf_position': result.kf_position.tolist() if result.kf_position is not None else None,
                    'kf_quaternion': result.kf_quaternion.tolist() if result.kf_quaternion is not None else None,
                    'num_inliers': result.num_inliers,
                    'viewpoint_used': result.viewpoint_used
                })
            else:
                time.sleep(0.001)

    def _process_frame(self, frame: np.ndarray, frame_id: int) -> ProcessingResult:
        current_frame_copy = frame.copy()
        result = ProcessingResult(frame_id=frame_id, frame=current_frame_copy, pose_success=False)
        
        # Predict step for Kalman Filter (always predict)
        with self.pose_data_lock:
            kf_predicted_pos, kf_predicted_quat = self.kf.predict()
            result.kf_position = kf_predicted_pos
            result.kf_quaternion = kf_predicted_quat
        
        # 1. Object Detection
        bbox = self._yolo_detect(current_frame_copy)
        result.bbox = bbox
        
        # 2. Feature Matching and Pose Estimation (PnP)
        best_pose = self._estimate_pose_with_temporal_consistency(current_frame_copy, bbox)
        
        # --- Pre-Filtering Checks ---
        is_current_measurement_valid = False
        if best_pose:
            current_quaternion = best_pose.quaternion
            if self.last_orientation is not None:
                angle_diff = math.degrees(self.quaternion_angle_diff(self.last_orientation, current_quaternion))
                if angle_diff > self.ORI_MAX_DIFF_DEG:
                    print(f"🚫 Frame {frame_id}: Rejected (Orientation Jump: {angle_diff:.1f}° > {self.ORI_MAX_DIFF_DEG}°)")
                else:
                    is_current_measurement_valid = True
            else:
                is_current_measurement_valid = True
        
        # --- Apply Measurement (if valid) or use Prediction ---
        if is_current_measurement_valid and best_pose:
            self.rejected_consecutive_frames_count = 0
            result.position = best_pose.position
            result.quaternion = best_pose.quaternion
            result.num_inliers = best_pose.inliers
            result.pose_success = True
            result.viewpoint_used = best_pose.viewpoint
            self.last_orientation = best_pose.quaternion
            
            with self.pose_data_lock:
                kf_pos, kf_quat = self.kf.update(best_pose.position, best_pose.quaternion)
                result.kf_position = kf_pos
                result.kf_quaternion = kf_quat
        else:
            self.rejected_consecutive_frames_count += 1
            print(f"ℹ️ Frame {frame_id}: No valid pose found or measurement rejected. Consecutive rejections: {self.rejected_consecutive_frames_count}")
            if self.rejected_consecutive_frames_count >= self.MAX_REJECTED_FRAMES:
                print(f"⚠️ Frame {frame_id}: Exceeded {self.MAX_REJECTED_FRAMES} consecutive rejections. Re-initializing Kalman Filter and last known pose.")
                with self.pose_data_lock:
                    self.kf.initialized = False
                self.last_orientation = None
                self.rejected_consecutive_frames_count = 0
                self.current_best_viewpoint = None
            
            result.position = None
            result.quaternion = None
            result.num_inliers = 0
            result.pose_success = False
            result.viewpoint_used = self.current_best_viewpoint if self.current_best_viewpoint else "Unknown"

        return result

    def _estimate_pose_with_temporal_consistency(self, frame: np.ndarray, bbox: Optional[Tuple]) -> Optional[PoseData]:
        if self.current_best_viewpoint:
            pose_data = self._solve_for_viewpoint(frame, self.current_best_viewpoint, bbox)
            if pose_data and pose_data.inliers >= self.viewpoint_quality_threshold:
                self.consecutive_failures = 0
                return pose_data
            else:
                self.consecutive_failures += 1
                print(f"🔄 Current viewpoint {self.current_best_viewpoint} quality dropped (inliers: {pose_data.inliers if pose_data else 0})")
        
        if (self.current_best_viewpoint is None or self.consecutive_failures >= self.max_failures_before_search):
            print("🔍 Searching for best viewpoint...")
            ordered_viewpoints = self._quick_viewpoint_assessment(frame, bbox)
            successful_poses = []
            for viewpoint in ordered_viewpoints:
                pose_data = self._solve_for_viewpoint(frame, viewpoint, bbox)
                if pose_data:
                    successful_poses.append(pose_data)
                    if pose_data.inliers >= self.viewpoint_quality_threshold * 2:
                        print(f"✅ Found excellent match with {viewpoint} ({pose_data.inliers} inliers)")
                        break
            
            if successful_poses:
                best_pose = max(successful_poses, key=lambda p: (p.inliers, -p.reprojection_error))
                self.current_best_viewpoint = best_pose.viewpoint
                self.consecutive_failures = 0
                print(f"🎯 Selected viewpoint: {best_pose.viewpoint} ({best_pose.inliers} inliers)")
                return best_pose
            else:
                print("❌ No valid poses found in any viewpoint")
                return None
        
        return None

    def _quick_viewpoint_assessment(self, frame: np.ndarray, bbox: Optional[Tuple]) -> List[str]:
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else frame
        if crop.size == 0:
            return ['NW', 'NE', 'SE', 'SW']
        
        frame_features = self._extract_features_sp(crop)
        viewpoint_scores = []
        for viewpoint in ['NE', 'NW', 'SE', 'SW']:
            anchor = self.viewpoint_anchors.get(viewpoint)
            if anchor:
                with torch.no_grad():
                    matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
                matches = rbd(matches_dict)['matches'].cpu().numpy()
                valid_matches = sum(1 for anchor_idx, _ in matches if anchor_idx in anchor['map_3d'])
                viewpoint_scores.append((viewpoint, valid_matches))
        
        viewpoint_scores.sort(key=lambda x: x[1], reverse=True)
        ordered_viewpoints = [vp for vp, score in viewpoint_scores]
        print(f"📊 Viewpoint assessment: {[(vp, score) for vp, score in viewpoint_scores]}")
        return ordered_viewpoints

    def _solve_for_viewpoint(self, frame: np.ndarray, viewpoint: str, bbox: Optional[Tuple]) -> Optional[PoseData]:
        anchor = self.viewpoint_anchors.get(viewpoint)
        if not anchor: return None
        
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else frame
        crop_offset = np.array([bbox[0], bbox[1]]) if bbox else np.array([0, 0])
        
        if crop.size == 0:
            return None
        
        frame_features = self._extract_features_sp(crop)
        
        with torch.no_grad():
            matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
        matches = rbd(matches_dict)['matches'].cpu().numpy()
        
        if len(matches) < 6: return None

        points_3d, points_2d = [], []
        for anchor_idx, frame_idx in matches:
            if anchor_idx in anchor['map_3d']:
                points_3d.append(anchor['map_3d'][anchor_idx])
                points_2d.append(frame_features['keypoints'][0].cpu().numpy()[frame_idx] + crop_offset)
        
        if len(points_3d) < 6: return None
        
        points_3d_np = np.array(points_3d, dtype=np.float32)
        points_2d_np = np.array(points_2d, dtype=np.float32)

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d_np, points_2d_np, self.K, self.dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
            if success and len(inliers) > 4:
                obj_inliers = points_3d_np[inliers.flatten()]
                img_inliers = points_2d_np[inliers.flatten()]
                rvec, tvec = cv2.solvePnPRefineVVS(objectPoints=obj_inliers,imagePoints=img_inliers,cameraMatrix=self.K,distCoeffs=self.dist_coeffs,rvec=rvec,tvec=tvec)
        except cv2.error as e:
            print(f"PnP Error for viewpoint {viewpoint}: {e}")
            return None

        if not success or inliers is None or len(inliers) < 4: return None
        R, _ = cv2.Rodrigues(rvec)
        position = tvec.flatten()
        quaternion = self._rotation_matrix_to_quaternion(R)
        
        projected_points, _ = cv2.projectPoints(points_3d_np[inliers.flatten()], rvec, tvec, self.K, self.dist_coeffs)
        error = np.mean(np.linalg.norm(points_2d_np[inliers.flatten()].reshape(-1, 1, 2) - projected_points, axis=2))
        
        return PoseData(position, quaternion, len(inliers), error, viewpoint)

    def quaternion_angle_diff(self, q1: np.ndarray, q2: np.ndarray) -> float:
        q1_norm = q1 / np.linalg.norm(q1)
        q2_norm = q2 / np.linalg.norm(q2)
        dot = np.dot(q1_norm, q2_norm)
        dot = max(-1.0, min(1.0, dot))
        angle = 2 * math.acos(abs(dot))
        return angle

    def _yolo_detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        confidence_thresholds = [0.3, 0.2, 0.1]
        for conf_thresh in confidence_thresholds:
            results = self.yolo_model(frame, verbose=False, conf=conf_thresh)
            airplane_detections = []
            for box in results[0].boxes:
                if box.cls.item() == 4:
                    confidence = box.conf.item()
                    bbox = box.xyxy.cpu().numpy()[0]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    airplane_detections.append((bbox, confidence, area))
            if airplane_detections:
                best_detection = max(airplane_detections, key=lambda x: x[2])
                print(f"✈️ Found airplane with confidence {best_detection[1]:.3f}")
                return tuple(map(int, best_detection[0]))
        print("⚠️ No aircraft detected")
        return None

    def _extract_features_sp(self, image_bgr: np.ndarray) -> Dict:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            return self.extractor.extract(tensor)

    def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = None
        return K, dist_coeffs

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
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
        return np.array([qx, qy, qz, qw])

    def cleanup(self):
        print("\nShutting down...")
        self.running = False
        if self.args.save_output:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"pose_log_{time.strftime('%Y%m%d-%H%M%S')}.json")
            with open(filename, 'w') as f:
                json.dump(self.all_poses_log, f, indent=4)
            print(f"💾 Pose log saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="VAPE MK50 - Real-time Pose Estimator (No MobileViT)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--webcam', action='store_true', help='Use webcam as input.')
    group.add_argument('--video_file', type=str, help='Path to a video file.')
    group.add_argument('--image_dir', type=str, help='Path to a directory of images.')
    parser.add_argument('--save_output', action='store_true', help='Save the final pose data to a JSON file.')
    args = parser.parse_args()

    try:
        processing_queue = queue.Queue(maxsize=2)
        pose_data_lock = threading.Lock()
        
        kf = LooselyCoupledKalmanFilter()
        
        main_thread = MainThread(processing_queue, pose_data_lock, kf, args)
        processing_thread = ProcessingThread(processing_queue, pose_data_lock, kf, args)
        
        print("Starting VAPE_MK50 in multi-threaded mode...")
        main_thread.start()
        processing_thread.start()
        
        main_thread.join()
        
        print("Stopping processing thread...")
        processing_thread.running = False
        processing_thread.join()
        print("Exiting.")
    
    except (IOError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C).")
    finally:
        print("✅ Process finished.")

if __name__ == '__main__':
    main()