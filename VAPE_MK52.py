# ==================================================================================================
#
#  VAPE MK52 - REAL-TIME 6-DOF POSE ESTIMATOR
#
#  Author: [Your Name/Alias Here]
#  Date: August 12, 2025
#  Description: This script performs real-time 6-DOF (position and orientation) pose estimation
#               of an aircraft using a multi-threaded architecture. It leverages computer vision
#               and deep learning models for object detection, feature matching, and pose calculation,
#               with a Kalman Filter for temporal smoothing and prediction.
#
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
#  1. IMPORTS AND INITIAL SETUP
# --------------------------------------------------------------------------------------------------
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

# --- Dependency Imports ---
# These are the core deep learning and computer vision libraries.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Disable gradient calculations for PyTorch models, as we are only doing inference (no training).
# This significantly speeds up computations and reduces memory usage.
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 VAPE MK52 Pose Estimator (No MobileViT)")
try:
    # YOLO for object detection.
    from ultralytics import YOLO
    # LightGlue for feature matching and SuperPoint for feature extraction.
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    # cKDTree for efficient nearest neighbor searches, used to map 2D anchor points.
    from scipy.spatial import cKDTree
    print("✅ All libraries loaded successfully.")
except ImportError as e:
    print(f"❌ Import error: {e}. Please run 'pip install -r requirements.txt' to install dependencies.")
    exit(1)


# --------------------------------------------------------------------------------------------------
#  2. UTILITY FUNCTIONS
# --------------------------------------------------------------------------------------------------
def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalizes a quaternion to ensure it has a unit length of 1."""
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-10 else np.array([0, 0, 0, 1])


# --------------------------------------------------------------------------------------------------
#  3. DATA STRUCTURES
# --------------------------------------------------------------------------------------------------
# Using dataclasses provides a clean and concise way to group related data.

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
    viewpoint_used: Optional[str] = None

@dataclass
class PoseData:
    """A simple container for the results of a successful pose estimation."""
    position: np.ndarray
    quaternion: np.ndarray
    inliers: int
    reprojection_error: float
    viewpoint: str
    total_matches: int


# --------------------------------------------------------------------------------------------------
#  4. UNSCENTED KALMAN FILTER (UKF)
# --------------------------------------------------------------------------------------------------
class UnscentedKalmanFilter:
    """
    An Unscented Kalman Filter for 6-DOF pose estimation.
    The UKF is used because our system's motion (especially rotation with quaternions) is non-linear,
    and the UKF handles non-linearity more accurately than a standard Extended Kalman Filter (EKF).
    It uses a constant acceleration motion model.
    """
    def __init__(self, dt=1/30.0):
        self.dt = dt  # Time step between filter updates
        self.initialized = False

        # State vector [pos(3), vel(3), acc(3), quat(4), ang_vel(3)]
        self.n = 16  # State size
        # Measurement vector [pos(3), quat(4)]
        self.m = 7   # Measurement size

        # Initialize state vector. Quaternion is initialized to identity (w=1).
        self.x = np.zeros(self.n)
        self.x[9] = 1.0

        # State covariance matrix: our uncertainty about the state.
        self.P = np.eye(self.n) * 0.1

        # UKF parameters (alpha, beta, kappa) control how sigma points are generated.
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n

        # Weights for the sigma points when calculating mean and covariance.
        self.wm = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + self.lambda_)))
        self.wc = self.wm.copy()
        self.wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.wc[0] = self.lambda_ / (self.n + self.lambda_) + (1.0 - self.alpha**2 + self.beta)

        # Noise matrices
        self.Q = np.eye(self.n) * 1e-3  # Process noise: uncertainty in our motion model.
        self.R = np.eye(self.m) * 1e-4  # Measurement noise: uncertainty in our sensor (PnP) readings.

    def _generate_sigma_points(self, x, P):
        """Generates sigma points, which are representative points of the state distribution."""
        sigmas = np.zeros((2 * self.n + 1, self.n))
        # Cholesky decomposition to find the matrix square root of the covariance.
        U = np.linalg.cholesky((self.n + self.lambda_) * P)
        sigmas[0] = x
        for i in range(self.n):
            sigmas[i+1]   = x + U[:, i]
            sigmas[self.n+i+1] = x - U[:, i]
        return sigmas

    def motion_model(self, x_in):
        """
        The non-linear motion model. It predicts the next state based on the current state.
        This is where the "constant acceleration" assumption is implemented.
        """
        dt = self.dt
        x_out = np.zeros_like(x_in)

        pos, vel, acc = x_in[0:3], x_in[3:6], x_in[6:9]
        x_out[0:3] = pos + vel * dt + 0.5 * acc * dt**2  # Position update
        x_out[3:6] = vel + acc * dt                      # Velocity update
        x_out[6:9] = acc                                 # Acceleration is constant

        q, w = x_in[9:13], x_in[13:16]
        # Simplified quaternion integration from angular velocity.
        omega_mat = 0.5 * np.array([
            [0, -w[0], -w[1], -w[2]], [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]], [w[2], w[1], -w[0], 0]
        ])
        q_new = (np.eye(4) + dt * omega_mat) @ q
        x_out[9:13] = normalize_quaternion(q_new)
        x_out[13:16] = w  # Angular velocity is constant

        return x_out

    def predict(self):
        """UKF Prediction Step: Propagates the state and covariance forward in time."""
        if not self.initialized:
            return self.x[0:3], self.x[9:13]

        # 1. Generate sigma points from the current state distribution.
        sigmas = self._generate_sigma_points(self.x, self.P)

        # 2. Propagate each sigma point through the non-linear motion model.
        sigmas_f = np.array([self.motion_model(s) for s in sigmas])

        # 3. Recalculate the mean (predicted state) and covariance from the propagated sigma points.
        x_pred = np.sum(self.wm[:, np.newaxis] * sigmas_f, axis=0)
        P_pred = self.Q.copy()
        for i in range(2 * self.n + 1):
            y = sigmas_f[i] - x_pred
            P_pred += self.wc[i] * np.outer(y, y)

        self.x = x_pred
        self.P = P_pred
        return self.x[0:3], self.x[9:13]

    def hx(self, x_in):
        """The measurement function. It maps a state vector to the measurement space."""
        z = np.zeros(self.m)
        z[0:3] = x_in[0:3]  # Position
        z[3:7] = x_in[9:13] # Quaternion
        return z

    def update(self, position: np.ndarray, quaternion: np.ndarray):
        """UKF Update Step: Corrects the predicted state with a new measurement."""
        measurement = np.concatenate([position, normalize_quaternion(quaternion)])

        if not self.initialized:
            self.x[0:3] = position
            self.x[9:13] = normalize_quaternion(quaternion)
            self.initialized = True
            return self.x[0:3], self.x[9:13]

        # 1. Generate sigma points from the *predicted* state.
        sigmas_f = self._generate_sigma_points(self.x, self.P)

        # 2. Transform sigma points into measurement space using the measurement function.
        sigmas_h = np.array([self.hx(s) for s in sigmas_f])

        # 3. Calculate the predicted measurement and its covariance.
        z_pred = np.sum(self.wm[:, np.newaxis] * sigmas_h, axis=0)
        S = self.R.copy()
        for i in range(2 * self.n + 1):
            y = sigmas_h[i] - z_pred
            S += self.wc[i] * np.outer(y, y)

        # 4. Calculate the cross-covariance between state and measurement.
        P_xz = np.zeros((self.n, self.m))
        for i in range(2 * self.n + 1):
            y_x = sigmas_f[i] - self.x
            y_z = sigmas_h[i] - z_pred
            P_xz += self.wc[i] * np.outer(y_x, y_z)

        # 5. Compute the Kalman Gain and update the state and covariance.
        K = P_xz @ np.linalg.inv(S)
        self.x += K @ (measurement - z_pred)
        self.P -= K @ S @ K.T
        return self.x[0:3], self.x[9:13]


# --------------------------------------------------------------------------------------------------
#  5. HIGH-FREQUENCY MAIN THREAD
# --------------------------------------------------------------------------------------------------
class MainThread(threading.Thread):
    """
    Handles high-frequency, low-latency tasks:
    - Reading frames from the camera/video.
    - Running the KF predict step.
    - Displaying the output video and visualizations.
    - Sending frames to the ProcessingThread.
    """
    def __init__(self, processing_queue, visualization_queue, pose_data_lock, kf, args):
        super().__init__()
        self.running = True
        self.processing_queue = processing_queue
        self.visualization_queue = visualization_queue
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
        """Initializes the input source based on command-line arguments."""
        if self.args.webcam:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened(): raise IOError("Cannot open webcam.")
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.is_video_stream = True
            print("📹 Using webcam input.")
        elif self.args.video_file:
            if not os.path.exists(self.args.video_file): raise FileNotFoundError(f"Video file not found: {self.args.video_file}")
            self.video_capture = cv2.VideoCapture(self.args.video_file)
            self.is_video_stream = True
            print(f"📹 Using video file input: {self.args.video_file}")
        elif self.args.image_dir:
            if not os.path.exists(self.args.image_dir): raise FileNotFoundError(f"Image directory not found: {self.args.image_dir}")
            self.image_files = sorted([os.path.join(self.args.image_dir, f) for f in os.listdir(self.args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not self.image_files: raise IOError(f"No images found in directory: {self.args.image_dir}")
            print(f"🖼️ Found {len(self.image_files)} images for processing.")
        else:
            raise ValueError("No input source specified. Use --webcam, --video_file, or --image_dir.")

    def _get_next_frame(self):
        """Fetches the next frame from the configured input source."""
        if self.is_video_stream:
            ret, frame = self.video_capture.read()
            return frame if ret else None
        else:
            if self.frame_idx < len(self.image_files):
                frame = cv2.imread(self.image_files[self.frame_idx])
                self.frame_idx += 1
                return frame
            return None

    def run(self):
        """The main loop of the thread."""
        window_name = "VAPE MK52 - Real-time Pose Estimation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.camera_width, self.camera_height)

        while self.running:
            loop_start_time = time.time()

            # Get a new frame from the camera/video source.
            frame = self._get_next_frame()
            if frame is None: break

            # 1. Kalman Filter Prediction (runs continuously at high frequency).
            with self.pose_data_lock:
                predicted_pose_tvec, predicted_pose_quat = self.kf.predict()

            # 2. Visualization
            vis_frame = frame.copy()
            if predicted_pose_tvec is not None and predicted_pose_quat is not None:
                self._draw_axes(vis_frame, predicted_pose_tvec, predicted_pose_quat)

            # Draw On-Screen Display (OSD) info.
            elapsed_time = time.time() - self.start_time
            fps = (self.frame_count + 1) / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(vis_frame, "STATUS: PREDICITING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
            self.frame_count += 1

            # 3. Send frame to the processing thread if the queue is not full.
            if self.processing_queue.qsize() < 2:
                self.processing_queue.put(frame.copy())

            # 4. Handle visualization for the feature display window (if enabled).
            if self.args.show:
                try:
                    # Check for new data from the processing thread without blocking.
                    vis_data = self.visualization_queue.get_nowait()
                    kpts, vis_crop = vis_data['kpts'], vis_data['crop']
                    for kpt in kpts:
                        cv2.circle(vis_crop, (int(kpt[0]), int(kpt[1])), 2, (0, 255, 0), -1)
                    cv2.imshow("SuperPoint Features", vis_crop)
                except queue.Empty:
                    pass  # No new data, just continue.

            # 5. Display the main window and check for exit key.
            cv2.imshow(window_name, vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                print("User requested shutdown.")
                break

            # 6. Cap the frame rate to maintain consistency.
            frame_rate_cap = 30.0
            time_to_wait = (1.0 / frame_rate_cap) - (time.time() - loop_start_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        self.cleanup()

    def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
        """Returns the camera intrinsic matrix K and distortion coefficients."""
        # These are hardcoded values for a specific camera.
        # For a different camera, these would need to be found via calibration.
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K, None  # Assuming no lens distortion.

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Converts a quaternion (x, y, z, w) to a 3x3 rotation matrix."""
        q_norm = normalize_quaternion(q)
        x, y, z, w = q_norm
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])

    def _draw_axes(self, frame: np.ndarray, position: np.ndarray, quaternion: np.ndarray):
        """Draws a 3D coordinate axis on the frame at the estimated pose."""
        try:
            R = self._quaternion_to_rotation_matrix(quaternion)
            rvec, _ = cv2.Rodrigues(R) # Convert rotation matrix to rotation vector
            tvec = position.reshape(3, 1)
            # Define the 3D points of the axis lines (length 0.1 meters).
            axis_pts = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
            # Project the 3D points onto the 2D image plane.
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, self.K, self.dist_coeffs)
            img_pts = img_pts.reshape(-1, 2).astype(int)
            origin = tuple(img_pts[0])
            # Draw the lines: X (Red), Y (Green), Z (Blue).
            cv2.line(frame, origin, tuple(img_pts[1]), (0,0,255), 3)
            cv2.line(frame, origin, tuple(img_pts[2]), (0,255,0), 3)
            cv2.line(frame, origin, tuple(img_pts[3]), (255,0,0), 3)
        except (cv2.error, AttributeError, ValueError):
            # Fail silently if there's an error during projection (e.g., invalid pose).
            pass

    def cleanup(self):
        """Releases resources upon shutdown."""
        self.running = False
        if self.is_video_stream and self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------------------
#  6. LOW-FREQUENCY PROCESSING THREAD
# --------------------------------------------------------------------------------------------------
class ProcessingThread(threading.Thread):
    """
    Handles low-frequency, computationally-heavy tasks:
    - YOLO object detection.
    - SuperPoint/LightGlue feature extraction and matching.
    - PnP pose calculation.
    - Running the KF update step.
    """
    def __init__(self, processing_queue, visualization_queue, pose_data_lock, kf, args):
        super().__init__()
        self.running = True
        self.processing_queue = processing_queue
        self.visualization_queue = visualization_queue
        self.pose_data_lock = pose_data_lock
        self.kf = kf
        self.args = args

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.camera_width, self.camera_height = 1280, 720
        self.all_poses_log = []

        # --- Temporal Consistency for Viewpoint Selection ---
        self.current_best_viewpoint = None
        self.viewpoint_quality_threshold = 5
        self.consecutive_failures = 0
        self.max_failures_before_search = 3

        # --- Pre-filtering for Measurement Rejection ---
        self.last_orientation: Optional[np.ndarray] = None
        self.ORI_MAX_DIFF_DEG = 30.0
        self.rejected_consecutive_frames_count = 0
        self.MAX_REJECTED_FRAMES = 10

        self.yolo_model = None
        self.extractor = None
        self.matcher = None
        self.viewpoint_anchors = {}

        self._initialize_models()
        self._initialize_anchor_data()
        self.K, self.dist_coeffs = self._get_camera_intrinsics()

    def _initialize_models(self):
        """Loads all required machine learning models onto the selected device."""
        print("📦 Loading models...")
        self.yolo_model = YOLO("yolo_coco_pretrained.pt").to(self.device)
        self.extractor = SuperPoint(max_num_keypoints=1024*2).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        print("   ...models loaded.")

    def _initialize_anchor_data(self):
        """
        Pre-processes anchor data. For each viewpoint, it loads the anchor image,
        extracts features, and creates a 2D-to-3D map by associating the known 2D
        keypoint locations with their corresponding 3D model points.
        """
        print("🛠️  Initializing anchor data...")
        # NOTE: Anchor 2D/3D points are hardcoded here for simplicity.
        # In a more robust system, this would be loaded from configuration files.
        ne_anchor_2d = np.array([[928, 148],[570, 111],[401, 31],[544, 141],[530, 134],[351, 228],[338, 220],[294, 244],[230, 541],[401, 469],[414, 481],[464, 451],[521, 510],[610, 454],[544, 400],[589, 373],[575, 361],[486, 561],[739, 385],[826, 305],[791, 285],[773, 271],[845, 233],[826, 226],[699, 308],[790, 375]], dtype=np.float32)
        ne_anchor_3d = np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.000, -0.000, 0.156],[-0.014, 0.000, 0.156],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.019, -0.000, 0.074],[-0.014, 0.000, 0.042],[0.000, 0.000, 0.042],[-0.080, -0.000, 0.156],[-0.100, -0.030, 0.000],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.014, 0.000, -0.156],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.042],[-0.090, -0.000, -0.042]], dtype=np.float32)
        nw_anchor_2d = np.array([[511, 293], [591, 284], [587, 330], [413, 249], [602, 348], [715, 384], [598, 298], [656, 171], [805, 213], [703, 392], [523, 286], [519, 327], [387, 289], [727, 126], [425, 243], [636, 358], [745, 202], [595, 388], [436, 260], [539, 313], [795, 220], [351, 291], [665, 165], [611, 353], [650, 377], [516, 389], [727, 143], [496, 378], [575, 312], [617, 368], [430, 312], [480, 281], [834, 225], [469, 339], [705, 223], [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]], dtype=np.float32)
        nw_anchor_3d = np.array([[-0.014, 0.0, 0.042], [0.025, -0.014, -0.011], [-0.014, 0.0, -0.042], [-0.014, 0.0, 0.156], [-0.023, 0.0, -0.065], [0.0, 0.0, -0.156], [0.025, 0.0, -0.015], [0.217, 0.0, 0.07], [0.23, 0.0, -0.07], [-0.014, 0.0, -0.156], [0.0, 0.0, 0.042], [-0.057, -0.018, -0.01], [-0.074, -0.0, 0.128], [0.206, -0.07, -0.002], [-0.0, -0.0, 0.156], [-0.017, -0.0, -0.092], [0.217, -0.0, -0.027], [-0.052, -0.0, -0.097], [-0.019, -0.0, 0.128], [-0.035, -0.018, -0.01], [0.217, -0.0, -0.07], [-0.08, -0.0, 0.156], [0.23, 0.0, 0.07], [-0.023, -0.0, -0.075], [-0.029, -0.0, -0.127], [-0.09, -0.0, -0.042], [0.206, -0.055, -0.002], [-0.09, -0.0, -0.015], [0.0, -0.0, -0.015], [-0.037, -0.0, -0.097], [-0.074, -0.0, 0.074], [-0.019, -0.0, 0.074], [0.23, -0.0, -0.113], [-0.1, -0.03, 0.0], [0.17, -0.0, -0.015], [0.23, -0.0, 0.113], [-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [-0.08, -0.0, -0.156]], dtype=np.float32)
        se_anchor_2d = np.array([[415, 144], [1169, 508], [275, 323], [214, 395], [554, 670], [253, 428], [280, 415], [355, 365], [494, 621], [519, 600], [806, 213], [973, 438], [986, 421], [768, 343], [785, 328], [841, 345], [931, 393], [891, 306], [980, 345], [651, 210], [625, 225], [588, 216], [511, 215], [526, 204], [665, 271]], dtype=np.float32)
        se_anchor_3d = np.array([[-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [0.23, 0.0, -0.113], [0.23, 0.0, 0.113], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.206, -0.07, -0.002], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [-0.1, -0.03, 0.0], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.019, 0.0, 0.074], [-0.019, 0.0, 0.128], [-0.074, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.052, 0.0, -0.097], [-0.037, 0.0, -0.097], [-0.029, 0.0, -0.127], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042]], dtype=np.float32)
        sw_anchor_2d = np.array([[650, 312], [630, 306], [907, 443], [814, 291], [599, 349], [501, 386], [965, 359], [649, 355], [635, 346], [930, 335], [843, 467], [702, 339], [718, 321], [930, 322], [727, 346], [539, 364], [786, 297], [1022, 406], [1004, 399], [539, 344], [536, 309], [864, 478], [745, 310], [1049, 393], [895, 258], [674, 347], [741, 281], [699, 294], [817, 494], [992, 281]], dtype=np.float32)
        sw_anchor_3d = np.array([[-0.035, -0.018, -0.01], [-0.057, -0.018, -0.01], [0.217, -0.0, -0.027], [-0.014, -0.0, 0.156], [-0.023, 0.0, -0.065], [-0.014, -0.0, -0.156], [0.234, -0.05, -0.002], [0.0, -0.0, -0.042], [-0.014, -0.0, -0.042], [0.206, -0.055, -0.002], [0.217, -0.0, -0.07], [0.025, -0.014, -0.011], [-0.014, -0.0, 0.042], [0.206, -0.07, -0.002], [0.049, -0.016, -0.011], [-0.029, -0.0, -0.127], [-0.019, -0.0, 0.128], [0.23, -0.0, 0.07], [0.217, -0.0, 0.07], [-0.052, -0.0, -0.097], [-0.175, -0.0, -0.015], [0.23, -0.0, -0.07], [-0.019, -0.0, 0.074], [0.23, -0.0, 0.113], [-0.0, -0.025, 0.24], [-0.0, -0.0, -0.015], [-0.074, -0.0, 0.128], [-0.074, -0.0, 0.074], [0.23, -0.0, -0.113], [0.243, -0.104, 0.0]], dtype=np.float32)
        w_anchor_2d = np.array([[589, 555],[565, 481],[531, 480],[329, 501],[326, 345],[528, 351],[395, 391],[469, 395],[529, 140],[381, 224],[504, 258],[498, 229],[383, 253],[1203, 100],[1099, 174],[1095, 211],[1201, 439],[1134, 404],[1100, 358],[625, 341],[624, 310],[315, 264]], dtype=np.float32)
        w_anchor_3d = np.array([[-0.000, -0.025, -0.240],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.156],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.015],[-0.014, 0.000, -0.042],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.000, -0.025, 0.240],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.074],[0.243, -0.104, 0.000],[0.206, -0.070, -0.002],[0.206, -0.055, -0.002],[0.230, -0.000, -0.113],[0.217, -0.000, -0.070],[0.217, -0.000, -0.027],[0.025, 0.000, -0.015],[0.025, -0.014, -0.011],[-0.100, -0.030, 0.000]], dtype=np.float32)
        s_anchor_2d = np.array([[14, 243],[1269, 255],[654, 183],[290, 484],[1020, 510],[398, 475],[390, 503],[901, 489],[573, 484],[250, 283],[405, 269],[435, 243],[968, 273],[838, 273],[831, 233],[949, 236]], dtype=np.float32)
        s_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, 0.000, 0.070],[0.217, -0.000, -0.027],[0.000, 0.000, -0.156],[-0.017, -0.000, -0.092],[-0.052, -0.000, -0.097],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128]], dtype=np.float32)
        n_anchor_2d = np.array([[1238, 346],[865, 295],[640, 89],[425, 314],[24, 383],[303, 439],[445, 434],[856, 418],[219, 475],[1055, 450]], dtype=np.float32)
        n_anchor_3d = np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.052, -0.000, -0.097],[-0.080, -0.000, 0.156],[-0.080, -0.000, -0.156]], dtype=np.float32)
        e_anchor_2d = np.array([[696, 165],[46, 133],[771, 610],[943, 469],[921, 408],[793, 478],[781, 420],[793, 520],[856, 280],[743, 284],[740, 245],[711, 248],[74, 520],[134, 465],[964, 309]], dtype=np.float32)
        e_anchor_3d = np.array([[-0.000, -0.025, -0.240],[0.243, -0.104, 0.000],[-0.000, -0.025, 0.240],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.014, 0.000, 0.156],[-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.014, 0.000, -0.156],[0.000, 0.000, -0.156],[0.230, -0.000, 0.113],[0.217, 0.000, 0.070],[-0.100, -0.030, 0.000]], dtype=np.float32)
        sw2_anchor_2d = np.array([[15, 300],[1269, 180],[635, 143],[434, 274],[421, 240],[273, 320],[565, 266],[844, 206],[468, 543],[1185, 466],[565, 506],[569, 530],[741, 491],[1070, 459],[1089, 480],[974, 220],[941, 184],[659, 269],[650, 299],[636, 210],[620, 193]], dtype=np.float32)
        sw2_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.017, -0.000, -0.092],[-0.052, -0.000, -0.097],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.042],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, -0.000, -0.027],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[0.025, -0.014, -0.011],[0.025, 0.000, -0.015],[-0.035, -0.018, -0.010],[-0.057, -0.018, -0.010]], dtype=np.float32)
        se2_anchor_2d = np.array([[48, 216],[1269, 320],[459, 169],[853, 528],[143, 458],[244, 470],[258, 451],[423, 470],[741, 500],[739, 516],[689, 176],[960, 301],[828, 290],[970, 264],[850, 254]], dtype=np.float32)
        se2_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[0.230, -0.000, -0.113],[0.230, 0.000, -0.070],[0.217, -0.000, -0.070],[0.217, -0.000, -0.027],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[-0.100, -0.030, 0.000],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074]], dtype=np.float32)
        su_anchor_2d = np.array([[203, 251],[496, 191],[486, 229],[480, 263],[368, 279],[369, 255],[573, 274],[781, 280],[859, 293],[865, 213],[775, 206],[1069, 326],[656, 135],[633, 241],[629, 204],[623, 343],[398, 668],[463, 680],[466, 656],[761, 706],[761, 681],[823, 709],[616, 666]], dtype=np.float32)
        su_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.017, -0.000, -0.092],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.156],[-0.014, 0.000, -0.042],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.035, -0.018, -0.010],[-0.057, -0.018, -0.010],[0.025, -0.014, -0.011],[0.230, -0.000, -0.113],[0.230, 0.000, -0.070],[0.217, -0.000, -0.070],[0.230, -0.000, 0.070],[0.217, 0.000, 0.070],[0.230, -0.000, 0.113],[0.243, -0.104, 0.000]], dtype=np.float32)
        nu_anchor_2d = np.array([[631, 361],[1025, 293],[245, 294],[488, 145],[645, 10],[803, 146],[661, 188],[509, 365],[421, 364],[434, 320],[509, 316],[779, 360],[784, 321],[704, 398],[358, 393]], dtype=np.float32)
        nu_anchor_3d = np.array([[-0.100, -0.030, 0.000],[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.230, -0.000, 0.113],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.170, -0.000, -0.015],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.090, -0.000, -0.042],[-0.080, -0.000, 0.156]], dtype=np.float32)
        nw2_anchor_2d = np.array([[1268, 328],[1008, 419],[699, 399],[829, 373],[641, 325],[659, 310],[783, 30],[779, 113],[775, 153],[994, 240],[573, 226],[769, 265],[686, 284],[95, 269],[148, 375],[415, 353],[286, 349],[346, 320],[924, 360],[590, 324]], dtype=np.float32)
        nw2_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.042],[-0.052, -0.000, -0.097],[-0.057, -0.018, -0.010],[-0.035, -0.018, -0.010],[0.243, -0.104, 0.000],[0.206, -0.070, -0.002],[0.206, -0.055, -0.002],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.170, -0.000, -0.015],[0.025, -0.014, -0.011],[-0.000, -0.025, 0.240],[-0.080, -0.000, 0.156],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.128],[-0.029, -0.000, -0.127],[-0.100, -0.030, 0.000]], dtype=np.float32)
        ne2_anchor_2d = np.array([[1035, 95],[740, 93],[599, 16],[486, 168],[301, 305],[719, 225],[425, 349],[950, 204],[794, 248],[844, 203],[833, 175],[601, 275],[515, 301],[584, 244],[503, 266]], dtype=np.float32)
        ne2_anchor_3d = np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.080, -0.000, 0.156],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.042],[-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128]], dtype=np.float32)

        anchor_definitions = {
            'NE': {'path': 'NE.png', '2d': ne_anchor_2d, '3d': ne_anchor_3d},
            'NW': {'path': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png', '2d': nw_anchor_2d, '3d': nw_anchor_3d},
            'SE': {'path': 'SE.png', '2d': se_anchor_2d, '3d': se_anchor_3d},
            'W': {'path': 'W.png', '2d': w_anchor_2d, '3d': w_anchor_3d},
            'S': {'path': 'S.png', '2d': s_anchor_2d, '3d': s_anchor_3d},
            'SW': {'path': 'Anchor_B.png', '2d': sw_anchor_2d, '3d': sw_anchor_3d},
            'N': {'path': 'N.png', '2d': n_anchor_2d, '3d': n_anchor_3d},
            'SW2': {'path': 'SW2.png', '2d': sw2_anchor_2d, '3d': sw2_anchor_3d},
            'SE2': {'path': 'SE2.png', '2d': se2_anchor_2d, '3d': se2_anchor_3d},
            'SU': {'path': 'SU.png', '2d': su_anchor_2d, '3d': su_anchor_3d},
            'NU': {'path': 'NU.png', '2d': nu_anchor_2d, '3d': nu_anchor_3d},
            'NW2': {'path': 'NW2.png', '2d': nw2_anchor_2d, '3d': nw2_anchor_3d},
            'NE2': {'path': 'NE2.png', '2d': ne2_anchor_2d, '3d': ne2_anchor_3d},
        }

        self.viewpoint_anchors = {}
        for viewpoint, data in anchor_definitions.items():
            if not os.path.exists(data['path']):
                raise FileNotFoundError(f"Required anchor image not found: {data['path']}")

            anchor_image_bgr = cv2.resize(cv2.imread(data['path']), (self.camera_width, self.camera_height))
            anchor_features = self._extract_features_sp(anchor_image_bgr)
            anchor_keypoints = anchor_features['keypoints'][0].cpu().numpy()

            # Use a KD-Tree for fast lookup of the nearest SuperPoint keypoint to our annotated 2D points.
            sp_tree = cKDTree(anchor_keypoints)
            distances, indices = sp_tree.query(data['2d'], k=1, distance_upper_bound=5.0)
            valid_mask = distances != np.inf

            # Store the features and the mapping from the SuperPoint keypoint index to the 3D point.
            self.viewpoint_anchors[viewpoint] = {
                'features': anchor_features,
                'map_3d': {idx: pt for idx, pt in zip(indices[valid_mask], data['3d'][valid_mask])}
            }
        print("   ...anchor data initialized.")

    def run(self):
        """The main loop of the thread."""
        frame_id = 0
        while self.running:
            if not self.processing_queue.empty():
                frame = self.processing_queue.get()
                result = self._process_frame(frame, frame_id)
                frame_id += 1

                # Log the results for saving later.
                self.all_poses_log.append({
                    'frame': result.frame_id, 'success': result.pose_success,
                    'position': result.position.tolist() if result.position is not None else None,
                    'quaternion': result.quaternion.tolist() if result.quaternion is not None else None,
                    'kf_position': result.kf_position.tolist() if result.kf_position is not None else None,
                    'kf_quaternion': result.kf_quaternion.tolist() if result.kf_quaternion is not None else None,
                    'num_inliers': result.num_inliers, 'viewpoint_used': result.viewpoint_used
                })
            else:
                # Sleep briefly to prevent busy-waiting.
                time.sleep(0.001)

    def _process_frame(self, frame: np.ndarray, frame_id: int) -> ProcessingResult:
        """The core pose estimation pipeline for a single frame."""
        result = ProcessingResult(frame_id=frame_id, frame=frame.copy(), pose_success=False)

        # 1. Object Detection using YOLO.
        bbox = self._yolo_detect(frame)
        result.bbox = bbox

        # 2. Feature Matching and Pose Estimation.
        best_pose = self._estimate_pose_with_temporal_consistency(frame, bbox)

        # 3. Pre-Filtering: Check if the new pose measurement is valid.
        is_valid = False
        if best_pose:
            # Reject measurements that result in a sudden, large jump in orientation.
            if self.last_orientation is not None:
                angle_diff = math.degrees(self.quaternion_angle_diff(self.last_orientation, best_pose.quaternion))
                if angle_diff <= self.ORI_MAX_DIFF_DEG:
                    is_valid = True
                else:
                    print(f"🚫 Frame {frame_id}: Rejected (Orientation Jump: {angle_diff:.1f}° > {self.ORI_MAX_DIFF_DEG}°)")
            else:
                is_valid = True # Accept the first valid measurement.

        # 4. Kalman Filter Update: Update the filter with the new measurement if it's valid.
        if is_valid and best_pose:
            self.rejected_consecutive_frames_count = 0
            result.position, result.quaternion = best_pose.position, best_pose.quaternion
            result.num_inliers, result.pose_success = best_pose.inliers, True
            result.viewpoint_used = best_pose.viewpoint
            self.last_orientation = best_pose.quaternion

            # The lock ensures that the MainThread doesn't try to predict while we are updating.
            with self.pose_data_lock:
                kf_pos, kf_quat = self.kf.update(best_pose.position, best_pose.quaternion)
                result.kf_position, result.kf_quaternion = kf_pos, kf_quat
        else:
            # If no valid pose, increment rejection counter and handle re-initialization if needed.
            self.rejected_consecutive_frames_count += 1
            if self.rejected_consecutive_frames_count >= self.MAX_REJECTED_FRAMES:
                print(f"⚠️ Exceeded {self.MAX_REJECTED_FRAMES} consecutive rejections. Re-initializing KF.")
                with self.pose_data_lock: self.kf.initialized = False
                self.last_orientation = None
                self.current_best_viewpoint = None
                self.rejected_consecutive_frames_count = 0

        return result

    def _estimate_pose_with_temporal_consistency(self, frame: np.ndarray, bbox: Optional[Tuple]) -> Optional[PoseData]:
        """
        Manages viewpoint selection to find the best pose. It tries to stick with the last
        known good viewpoint for stability, but will search all viewpoints if quality drops.
        """
        # If we have a stable viewpoint, try it first.
        if self.current_best_viewpoint:
            pose_data = self._solve_for_viewpoint(frame, self.current_best_viewpoint, bbox)
            if pose_data and pose_data.total_matches >= self.viewpoint_quality_threshold:
                self.consecutive_failures = 0
                return pose_data # Quality is good, stick with this viewpoint.
            else:
                self.consecutive_failures += 1 # Quality dropped.

        # If we don't have a viewpoint or the current one is failing, search for a new one.
        if self.current_best_viewpoint is None or self.consecutive_failures >= self.max_failures_before_search:
            print("🔍 Searching for best viewpoint...")
            # Quickly assess all viewpoints to find the most promising ones first.
            ordered_viewpoints = self._quick_viewpoint_assessment(frame, bbox)
            successful_poses = []
            for viewpoint in ordered_viewpoints:
                pose_data = self._solve_for_viewpoint(frame, viewpoint, bbox)
                if pose_data:
                    successful_poses.append(pose_data)
                    # Early exit if we find a very high-quality match.
                    if pose_data.total_matches >= self.viewpoint_quality_threshold * 2:
                        break

            if successful_poses:
                # Select the best pose based on total matches, then inliers, then reprojection error.
                best_pose = max(successful_poses, key=lambda p: (p.total_matches, p.inliers, -p.reprojection_error))
                self.current_best_viewpoint = best_pose.viewpoint
                self.consecutive_failures = 0
                print(f"🎯 Selected viewpoint: {best_pose.viewpoint} ({best_pose.total_matches} matches, {best_pose.inliers} inliers)")
                return best_pose
        return None

    def _quick_viewpoint_assessment(self, frame: np.ndarray, bbox: Optional[Tuple]) -> List[str]:
        """Quickly matches the frame against all viewpoints to create a ranked list."""
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else frame
        if crop.size == 0: return list(self.viewpoint_anchors.keys())

        frame_features = self._extract_features_sp(crop)
        viewpoint_scores = []
        for viewpoint, anchor in self.viewpoint_anchors.items():
            with torch.no_grad():
                matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
            matches = rbd(matches_dict)['matches'].cpu().numpy()
            # Score is the number of matches that correspond to a known 3D point.
            valid_matches = sum(1 for anchor_idx, _ in matches if anchor_idx in anchor['map_3d'])
            viewpoint_scores.append((viewpoint, valid_matches))

        # Return viewpoints ordered from highest score to lowest.
        viewpoint_scores.sort(key=lambda x: x[1], reverse=True)
        return [vp for vp, score in viewpoint_scores]

    def _solve_for_viewpoint(self, frame: np.ndarray, viewpoint: str, bbox: Optional[Tuple]) -> Optional[PoseData]:
        """Attempts to calculate the pose for a single given viewpoint."""
        anchor = self.viewpoint_anchors.get(viewpoint)
        if not anchor: return None

        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else frame
        if crop.size == 0: return None

        # 1. Extract features from the current frame's crop.
        frame_features = self._extract_features_sp(crop)

        # If visualization is on, send the keypoints and crop to the MainThread.
        if self.args.show and self.visualization_queue.qsize() < 2:
            kpts = frame_features['keypoints'][0].cpu().numpy()
            self.visualization_queue.put({'kpts': kpts, 'crop': crop.copy()})

        # 2. Match features between the anchor image and the current frame.
        with torch.no_grad():
            matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
        matches = rbd(matches_dict)['matches'].cpu().numpy()
        if len(matches) < 6: return None

        # 3. Build the 2D-3D point correspondences for PnP.
        points_3d, points_2d = [], []
        crop_offset = np.array([bbox[0], bbox[1]]) if bbox else np.array([0, 0])
        for anchor_idx, frame_idx in matches:
            if anchor_idx in anchor['map_3d']:
                points_3d.append(anchor['map_3d'][anchor_idx])
                points_2d.append(frame_features['keypoints'][0].cpu().numpy()[frame_idx] + crop_offset)
        if len(points_3d) < 6: return None

        # 4. Solve for the pose using solvePnPRansac.
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(points_3d, dtype=np.float32),
                np.array(points_2d, dtype=np.float32),
                self.K, self.dist_coeffs, reprojectionError=12, confidence=0.9,
                iterationsCount=5000, flags=cv2.SOLVEPNP_EPNP
            )
            # Refine the pose using only the inliers for better accuracy.
            if success and inliers is not None and len(inliers) > 4:
                rvec, tvec = cv2.solvePnPRefineVVS(
                    np.array(points_3d, dtype=np.float32)[inliers.flatten()],
                    np.array(points_2d, dtype=np.float32)[inliers.flatten()],
                    self.K, self.dist_coeffs, rvec, tvec
                )
        except cv2.error as e:
            return None

        if not success or inliers is None or len(inliers) < 4: return None

        # 5. Convert rotation vector to quaternion and calculate reprojection error.
        R, _ = cv2.Rodrigues(rvec)
        position = tvec.flatten()
        quaternion = self._rotation_matrix_to_quaternion(R)
        projected_points, _ = cv2.projectPoints(np.array(points_3d)[inliers.flatten()], rvec, tvec, self.K, self.dist_coeffs)
        error = np.mean(np.linalg.norm(np.array(points_2d)[inliers.flatten()].reshape(-1, 1, 2) - projected_points, axis=2))

        return PoseData(position, quaternion, len(inliers), error, viewpoint, len(points_3d))

    def quaternion_angle_diff(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Calculates the angular difference in radians between two quaternions."""
        dot = np.dot(normalize_quaternion(q1), normalize_quaternion(q2))
        return 2 * math.acos(abs(min(1.0, max(-1.0, dot))))

    def _yolo_detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detects an airplane in the frame using YOLO, returning the bounding box."""
        # Try multiple confidence thresholds to increase chance of detection.
        for conf_thresh in [0.3, 0.2, 0.1]:
            results = self.yolo_model(frame, verbose=False, conf=conf_thresh)
            detections = [box for box in results[0].boxes if box.cls.item() == 4] # Class 4 is airplane in COCO.
            if detections:
                # Return the detection with the largest bounding box area.
                best_box = max(detections, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
                return tuple(map(int, best_box.xyxy.cpu().numpy()[0]))
        return None

    def _extract_features_sp(self, image_bgr: np.ndarray) -> Dict:
        """Extracts SuperPoint features from a single BGR image."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            return self.extractor.extract(tensor)

    def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
        """Returns the camera intrinsic matrix K."""
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K, None

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Converts a 3x3 rotation matrix to a quaternion (x, y, z, w)."""
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
        """Saves the pose log to a JSON file if requested."""
        print("Shutting down...")
        self.running = False
        if self.args.save_output:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"pose_log_{time.strftime('%Y%m%d-%H%M%S')}.json")
            with open(filename, 'w') as f:
                json.dump(self.all_poses_log, f, indent=4)
            print(f"💾 Pose log saved to {filename}")


# --------------------------------------------------------------------------------------------------
#  7. MAIN EXECUTION BLOCK
# --------------------------------------------------------------------------------------------------
def main():
    """Parses arguments, sets up queues and threads, and starts the application."""
    parser = argparse.ArgumentParser(description="VAPE MK52 - Real-time Pose Estimator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--webcam', action='store_true', help='Use webcam as input.')
    group.add_argument('--video_file', type=str, help='Path to a video file.')
    group.add_argument('--image_dir', type=str, help='Path to a directory of images.')
    parser.add_argument('--save_output', action='store_true', help='Save the final pose data to a JSON file.')
    parser.add_argument('--show', action='store_true', help='Show keypoint detections in a separate window.')
    args = parser.parse_args()

    try:
        # Queues for inter-thread communication.
        processing_queue = queue.Queue(maxsize=2)
        visualization_queue = queue.Queue(maxsize=2)
        # A lock to ensure thread-safe access to the shared Kalman Filter object.
        pose_data_lock = threading.Lock()

        # Initialize the Kalman Filter and the two main threads.
        kf = UnscentedKalmanFilter()
        main_thread = MainThread(processing_queue, visualization_queue, pose_data_lock, kf, args)
        processing_thread = ProcessingThread(processing_queue, visualization_queue, pose_data_lock, kf, args)

        print("Starting VAPE_MK52 in multi-threaded mode...")
        main_thread.start()
        processing_thread.start()

        # Wait for the main thread to finish (e.g., user presses 'q').
        main_thread.join()

        # Cleanly shut down the processing thread.
        print("Stopping processing thread...")
        processing_thread.running = False
        processing_thread.join()
        print("Exiting.")

    except (IOError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Process interrupted by user (Ctrl+C).")
    finally:
        print("✅ Process finished.")


if __name__ == '__main__':
    main()

# ==================================================================================================
#
#  VAPE MK52 - WORKFLOW EXPLANATION
#
# ==================================================================================================
#
# This script operates using two main threads to achieve real-time performance by decoupling
# high-frequency I/O from low-frequency, heavy computation.
#
# 1. MainThread (The Conductor 🎵):
#    - Role: Handles all fast, real-time operations.
#    - Responsibilities:
#        - Captures frames from the input source (webcam, video) at a consistent rate (e.g., 30 FPS).
#        - Runs the Kalman Filter's `predict()` step in every single loop. This provides a smooth,
#          continuous pose prediction, making the visualization appear fluid even when the
#          ProcessingThread is busy.
#        - Displays the main video feed with the predicted pose drawn as a 3D axis.
#        - If `--show` is enabled, it checks the `visualization_queue` for data from the
#          ProcessingThread and displays the feature keypoints in a separate window.
#        - Puts new frames onto the `processing_queue` for the other thread to consume.
#
# 2. ProcessingThread (The Powerlifter 💪):
#    - Role: Handles all slow, computationally expensive tasks.
#    - Responsibilities:
#        - Grabs frames from the `processing_queue` whenever it's available.
#        - Performs the core computer vision pipeline:
#            a. YOLO Object Detection: Finds the aircraft in the frame.
#            b. Feature Extraction: Uses SuperPoint to find thousands of keypoints in the detected region.
#            c. Feature Matching: Uses LightGlue to match the frame's keypoints against pre-computed
#               keypoints from various anchor images (different viewpoints of the aircraft).
#            d. Pose Estimation: If enough good matches are found, it uses a PnP (Perspective-n-Point)
#               algorithm to calculate the 6-DOF pose (position and orientation).
#        - Temporal Consistency: It includes logic to ensure the selected viewpoint is stable over
#          time and to filter out noisy pose measurements that are physically improbable.
#        - Kalman Filter Update: A valid, high-quality pose measurement is used to call the Kalman
#          Filter's `update()` step, correcting the prediction with new ground-truth data.
#
# 3. Unscented Kalman Filter (The Oracle 🔮):
#    - Role: To smooth the pose estimation and provide stable predictions.
#    - State: It maintains a belief about the aircraft's state, including its position, velocity,
#      acceleration, and orientation (as a quaternion).
#    - Predict-Update Cycle:
#        - `predict()`: Called rapidly by the MainThread to guess where the aircraft will be next.
#        - `update()`: Called by the ProcessingThread whenever a new, valid pose is calculated. This
#          corrects the filter's belief, improving the accuracy of all future predictions.
#
# Data Flow Summary:
#    - `processing_queue`:     MainThread --(Frames)--> ProcessingThread
#    - `visualization_queue`:  ProcessingThread --(Keypoint Data)--> MainThread
#    - `kf` (Kalman Filter):   A shared object. ProcessingThread writes updates, MainThread reads
#                              predictions. Access is controlled by a `pose_data_lock`.
#
# This architecture is essential for creating a responsive real-time system.
#
# ==================================================================================================