"""
VAPE MK45: Robust Pose Estimator with Advanced Correspondence and Refinement
- No visual tracking; re-localizes the object in every frame for robustness.
- Uses YOLO for initial object detection.
- Employs MobileViT for viewpoint classification to select the correct anchor.
- Sophisticated correspondence matching using SuperPoint features and a cKDTree.
- Advanced pose refinement with correspondence enhancement and VVS.
- Integrated Kalman filter for smoothing the final pose estimate.
- Exports detailed pose data to a JSON file.

Input modes:
- Camera mode: Real-time webcam input
- Video mode: MP4/AVI video file input (NEW!)
- Batch mode: Individual images from directory
"""

import cv2
import numpy as np
import torch
import time
import argparse
import warnings
import signal
import sys
import threading
import json
import csv
import os
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import queue

# New import for robust correspondence matching
from scipy.spatial import cKDTree

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 Starting VAPE MK45 - Robust Edition...")

# Import required libraries
try:
    from ultralytics import YOLO
    import timm
    from torchvision import transforms
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    from PIL import Image
    from scipy.stats import chi2
    print("✅ All libraries loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Simplified State Machine
class TrackingState(Enum):
    DETECTING = "detecting"
    ESTIMATING = "estimating"

@dataclass
class ProcessingResult:
    """Result from processing thread"""
    frame: np.ndarray
    frame_id: int
    timestamp: float
    position: Optional[np.ndarray] = None
    quaternion: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    viewpoint: Optional[str] = None
    num_matches: int = 0
    processing_time: float = 0.0
    pose_data: Optional[Dict] = None
    kf_position: Optional[np.ndarray] = None
    kf_quaternion: Optional[np.ndarray] = None
    measurement_accepted: bool = False
    mahalanobis_distance: float = 0.0

class LooselyCoupledKalmanFilter:
    def __init__(self, dt=1/30.0):
        self.dt = dt
        self.initialized = False
        self.n_states = 13
        self.x = np.zeros(self.n_states)
        self.x[9] = 1.0
        self.P = np.eye(self.n_states) * 0.1
        self.Q = np.eye(self.n_states) * 1e-3
        self.R = np.eye(7) * 1e-4

    def normalize_quaternion(self, q):
        norm = np.linalg.norm(q)
        return q / norm if norm > 1e-10 else np.array([0, 0, 0, 1])

    def predict(self):
        if not self.initialized:
            return None
        px, py, pz = self.x[0:3]
        vx, vy, vz = self.x[3:6]
        qx, qy, qz, qw = self.x[6:10]
        wx, wy, wz = self.x[10:13]
        dt = self.dt
        px_new, py_new, pz_new = px + vx * dt, py + vy * dt, pz + vz * dt
        vx_new, vy_new, vz_new = vx, vy, vz
        q = np.array([qx, qy, qz, qw])
        w = np.array([wx, wy, wz])
        omega_mat = np.array([[0, -wx, -wy, -wz], [wx, 0, wz, -wy], [wy, -wz, 0, wx], [wz, wy, -wx, 0]])
        dq = 0.5 * dt * omega_mat @ q
        q_new = self.normalize_quaternion(q + dq)
        wx_new, wy_new, wz_new = wx, wy, wz
        self.x = np.array([px_new, py_new, pz_new, vx_new, vy_new, vz_new, q_new[0], q_new[1], q_new[2], q_new[3], wx_new, wy_new, wz_new])
        F = np.eye(self.n_states)
        F[0, 3], F[1, 4], F[2, 5] = dt, dt, dt
        self.P = F @ self.P @ F.T + self.Q
        return self.x[0:3], self.x[6:10]

    def update(self, position, quaternion):
        measurement = np.concatenate([position, quaternion])
        if not self.initialized:
            self.x[0:3] = position
            self.x[6:10] = self.normalize_quaternion(quaternion)
            self.initialized = True
            return self.x[0:3], self.x[6:10]
        predicted_measurement = np.array([self.x[0], self.x[1], self.x[2], self.x[6], self.x[7], self.x[8], self.x[9]])
        innovation = measurement - predicted_measurement
        q_meas, q_pred = measurement[3:7], predicted_measurement[3:7]
        if np.dot(q_meas, q_pred) < 0:
            innovation[3:7] = -q_meas - q_pred
        H = np.zeros((7, self.n_states))
        H[0:3, 0:3] = np.eye(3)
        H[3:7, 6:10] = np.eye(4)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ innovation
        self.x[6:10] = self.normalize_quaternion(self.x[6:10])
        I = np.eye(self.n_states)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        return self.x[0:3], self.x[6:10]

    def get_innovation_covariance(self):
        if not self.initialized: return None
        H = np.zeros((7, self.n_states))
        H[0:3, 0:3] = np.eye(3)
        H[3:7, 6:10] = np.eye(4)
        return H @ self.P @ H.T + self.R

    def calculate_mahalanobis_distance(self, position, quaternion):
        if not self.initialized: return 0.0
        measurement = np.concatenate([position, quaternion])
        predicted_measurement = np.array([self.x[0], self.x[1], self.x[2], self.x[6], self.x[7], self.x[8], self.x[9]])
        innovation = measurement - predicted_measurement
        q_meas, q_pred = measurement[3:7], predicted_measurement[3:7]
        if np.dot(q_meas, q_pred) < 0:
            innovation[3:7] = -q_meas - q_pred
        S = self.get_innovation_covariance()
        if S is None: return 0.0
        try:
            return float(np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation))
        except np.linalg.LinAlgError:
            return float('inf')

class OutlierDetector:
    def __init__(self, threshold_chi2_prob=0.99, min_measurements=10):
        self.threshold = chi2.ppf(threshold_chi2_prob, df=7)
        self.min_measurements = min_measurements
        self.history = deque(maxlen=20)

    def is_outlier(self, mahalanobis_distance, kf):
        if not kf.initialized or len(self.history) < self.min_measurements:
            return False
        return mahalanobis_distance > self.threshold

def read_image_index_csv(csv_path):
    entries = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({'index': int(row['Index']), 'timestamp': float(row['Timestamp']), 'filename': row['Filename']})
    return entries

def create_unique_filename(directory, base_filename):
    base_path = os.path.join(directory or ".", base_filename)
    if not os.path.exists(base_path): return base_path
    name, ext = os.path.splitext(base_filename)
    counter = 1
    while True:
        new_path = os.path.join(directory, f"{name}_{counter}{ext}")
        if not os.path.exists(new_path): return new_path
        counter += 1

def convert_to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [convert_to_json_serializable(i) for i in obj]
    return obj

class ThreadSafeFrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_id = 0
        self.timestamp = None
        self.video_frame_number = 0  # For video mode tracking

    def update(self, frame, video_frame_number=None):
        with self.lock:
            self.latest_frame = frame.copy()
            self.frame_id += 1
            self.timestamp = time.perf_counter()
            if video_frame_number is not None:
                self.video_frame_number = video_frame_number
            return self.frame_id

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None: return None, None, None, None
            return self.latest_frame.copy(), self.frame_id, self.timestamp, self.video_frame_number

class PerformanceMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.timings = defaultdict(lambda: deque(maxlen=30))

    def add_timing(self, name: str, duration: float):
        with self.lock:
            self.timings[name].append(duration)

    def get_average(self, name: str) -> float:
        with self.lock:
            if name in self.timings and self.timings[name]:
                return np.mean(list(self.timings[name]))
        return 0.0

class MultiThreadedPoseEstimator:
    def __init__(self, args):
        print("🔧 Initializing VAPE MK45...")
        self.args = args
        self.running = False
        self.threads = []
        self.frame_buffer = ThreadSafeFrameBuffer()
        self.result_queue = queue.Queue(maxsize=1)
        self.perf_monitor = PerformanceMonitor()
        self.state = TrackingState.DETECTING
        self.kf = LooselyCoupledKalmanFilter(dt=1/30.0)
        self.outlier_detector = OutlierDetector()
        self.use_kalman_filter = getattr(args, 'use_kalman_filter', True)
        self.all_poses = []
        self.poses_lock = threading.Lock()
        self.kf_lock = threading.Lock()
        
        # Input mode setup
        self.batch_mode = hasattr(args, 'image_dir') and args.image_dir is not None
        self.video_mode = hasattr(args, 'video_file') and args.video_file is not None
        self.camera_mode = not self.batch_mode and not self.video_mode
        
        # Input-specific variables
        self.image_entries = []
        self.batch_complete = False
        self.video_complete = False
        self.total_video_frames = 0
        self.video_fps = 30.0
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")
        self.camera_width, self.camera_height = 1280, 720
        
        # Determine input mode and initialize accordingly
        if self.batch_mode:
            print("📁 Mode: Batch processing (images from directory)")
            self._init_batch_processing()
        elif self.video_mode:
            print("🎬 Mode: Video file processing")
            self._init_video()
        else:
            print("📹 Mode: Real-time camera")
            self._init_camera()
        
        self._init_models()
        self._init_anchor_data()
        print("✅ VAPE MK45 initialized!")

    def _init_batch_processing(self):
        if hasattr(self.args, 'csv_file') and self.args.csv_file:
            self.image_entries = read_image_index_csv(self.args.csv_file)
            print(f"✅ Loaded {len(self.image_entries)} image entries from CSV")

    def _init_video(self):
        """Initialize video file processing"""
        try:
            if not os.path.exists(self.args.video_file):
                print(f"❌ Video file not found: {self.args.video_file}")
                self.cap = None
                return
            
            self.cap = cv2.VideoCapture(self.args.video_file)
            
            if not self.cap.isOpened():
                print(f"❌ Cannot open video file: {self.args.video_file}")
                self.cap = None
                return
            
            # Get video properties
            self.total_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Test capture
            ret, test_frame = self.cap.read()
            if not ret:
                print("❌ Cannot read from video file")
                self.cap.release()
                self.cap = None
                return
            
            # Reset to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            print(f"✅ Video initialized: {video_width}x{video_height}")
            print(f"   📊 Total frames: {self.total_video_frames}")
            print(f"   📊 FPS: {self.video_fps:.2f}")
            print(f"   📊 Duration: {self.total_video_frames/self.video_fps:.2f} seconds")
            
            self.camera_width = video_width
            self.camera_height = video_height
            
        except Exception as e:
            print(f"❌ Video initialization failed: {e}")
            self.cap = None

    def _init_models(self):
        try:
            print("  📦 Loading YOLO...")
            self.yolo_model = YOLO("yolov8s.pt").to(self.device)
            print("  📦 Loading viewpoint classifier...")
            self.vp_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=4)
            try:
                self.vp_model.load_state_dict(torch.load('mobilevit_viewpoint_20250703.pth', map_location=self.device))
            except FileNotFoundError:
                print("  ⚠️ Viewpoint model file not found, using random weights")
            self.vp_model.eval().to(self.device)
            self.vp_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
            print("  📦 Loading SuperPoint & LightGlue...")
            self.extractor = SuperPoint(max_num_keypoints=512).eval().to(self.device)
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)
            self.class_names = ['NE', 'NW', 'SE', 'SW']
        except Exception as e:
            print(f"❌ Model loading failed: {e}"); raise

    def _init_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened(): raise IOError("Cannot open any camera")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = self.cap.read()
            if not ret: raise IOError("Cannot read from camera")
            self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✅ Camera initialized: {self.camera_width}x{self.camera_height}")
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            self.cap = None

    def _init_anchor_data(self):
        print("🛠️ Initializing anchor data with KDTree...")
        default_anchor_paths = {
            'NE': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png',
            'NW': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png',
            'SE': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png',
            'SW': 'Anchor_B.png'
        }
        # Define 2D/3D points for each viewpoint here...
        default_anchor_2d = np.array([[511, 293], [591, 284], [587, 330], [413, 249], [602, 348], [715, 384], [598, 298], [656, 171], [805, 213], [703, 392], [523, 286], [519, 327], [387, 289], [727, 126], [425, 243], [636, 358], [745, 202], [595, 388], [436, 260], [539, 313], [795, 220], [351, 291], [665, 165], [611, 353], [650, 377], [516, 389], [727, 143], [496, 378], [575, 312], [617, 368], [430, 312], [480, 281], [834, 225], [469, 339], [705, 223], [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]], dtype=np.float32)
        default_anchor_3d = np.array([[-0.014, 0.000, 0.042], [0.025, -0.014, -0.011], [-0.014, 0.000, -0.042], [-0.014, 0.000, 0.156], [-0.023, 0.000, -0.065], [0.000, 0.000, -0.156], [0.025, 0.000, -0.015], [0.217, 0.000, 0.070], [0.230, 0.000, -0.070], [-0.014, 0.000, -0.156], [0.000, 0.000, 0.042], [-0.057, -0.018, -0.010], [-0.074, -0.000, 0.128], [0.206, -0.070, -0.002], [-0.000, -0.000, 0.156], [-0.017, -0.000, -0.092], [0.217, -0.000, -0.027], [-0.052, -0.000, -0.097], [-0.019, -0.000, 0.128], [-0.035, -0.018, -0.010], [0.217, -0.000, -0.070], [-0.080, -0.000, 0.156], [0.230, -0.000, 0.070], [-0.023, -0.000, -0.075], [-0.029, -0.000, -0.127], [-0.090, -0.000, -0.042], [0.206, -0.055, -0.002], [-0.090, -0.000, -0.015], [0.000, -0.000, -0.015], [-0.037, -0.000, -0.097], [-0.074, -0.000, 0.074], [-0.019, -0.000, 0.074], [0.230, -0.000, -0.113], [-0.100, -0.030, 0.000], [0.170, -0.000, -0.015], [0.230, -0.000, 0.113], [-0.000, -0.025, -0.240], [-0.000, -0.025, 0.240], [0.243, -0.104, 0.000], [-0.080, -0.000, -0.156]], dtype=np.float32)
        sw_anchor_2d = np.array([[650, 312], [630, 306], [907, 443], [814, 291], [599, 349], [501, 386], [965, 359], [649, 355], [635, 346], [930, 335], [843, 467], [702, 339], [718, 321], [930, 322], [727, 346], [539, 364], [786, 297], [1022, 406], [1004, 399], [539, 344], [536, 309], [864, 478], [745, 310], [1049, 393], [895, 258], [674, 347], [741, 281], [699, 294], [817, 494], [992, 281]], dtype=np.float32)
        sw_anchor_3d = np.array([[-0.035, -0.018, -0.010], [-0.057, -0.018, -0.010], [ 0.217, -0.000, -0.027], [-0.014, -0.000,  0.156], [-0.023, -0.000, -0.065], [-0.014, -0.000, -0.156], [ 0.234, -0.050, -0.002], [ 0.000, -0.000, -0.042], [-0.014, -0.000, -0.042], [ 0.206, -0.055, -0.002], [ 0.217, -0.000, -0.070], [ 0.025, -0.014, -0.011], [-0.014, -0.000,  0.042], [ 0.206, -0.070, -0.002], [ 0.049, -0.016, -0.011], [-0.029, -0.000, -0.127], [-0.019, -0.000,  0.128], [ 0.230, -0.000,  0.070], [ 0.217, -0.000,  0.070], [-0.052, -0.000, -0.097], [-0.175, -0.000, -0.015], [ 0.230, -0.000, -0.070], [-0.019, -0.000,  0.074], [ 0.230, -0.000,  0.113], [-0.000, -0.025,  0.240], [-0.000, -0.000, -0.015], [-0.074, -0.000,  0.128], [-0.074, -0.000,  0.074], [ 0.230, -0.000, -0.113], [ 0.243, -0.104,  0.000]], dtype=np.float32)
        viewpoint_data = {'NE': {'2d': default_anchor_2d, '3d': default_anchor_3d}, 'NW': {'2d': default_anchor_2d, '3d': default_anchor_3d}, 'SE': {'2d': default_anchor_2d, '3d': default_anchor_3d}, 'SW': {'2d': sw_anchor_2d, '3d': sw_anchor_3d}}

        self.viewpoint_anchors = {}
        for viewpoint, path in default_anchor_paths.items():
            print(f"  📸 Processing anchor for {viewpoint}: {path}")
            anchor_image = self._load_anchor_image(path, viewpoint)
            anchor_features = self._extract_features_from_image(anchor_image)
            anchor_keypoints_sp = anchor_features['keypoints'][0].cpu().numpy()
            
            if len(anchor_keypoints_sp) == 0:
                print(f"  ⚠️ No SuperPoint keypoints found for {viewpoint} anchor. Skipping.")
                continue

            anchor_2d = viewpoint_data[viewpoint]['2d']
            anchor_3d = viewpoint_data[viewpoint]['3d']

            sp_tree = cKDTree(anchor_keypoints_sp)
            distances, indices = sp_tree.query(anchor_2d, k=1)
            
            valid_matches = distances < 5.0
            matched_sp_indices = indices[valid_matches]
            matched_3d_points = anchor_3d[valid_matches]

            print(f"    Found {len(matched_sp_indices)} valid 2D-3D correspondences for {viewpoint}.")

            self.viewpoint_anchors[viewpoint] = {
                'features': anchor_features,
                'all_3d_points': anchor_3d,
                'matched_sp_indices': matched_sp_indices,
                'matched_3d_points': matched_3d_points
            }
        print("✅ Anchor data initialization complete.")

    def _load_anchor_image(self, path, viewpoint):
        try:
            img = cv2.imread(path)
            if img is None: raise FileNotFoundError(f"File not found or could not be read: {path}")
            return cv2.resize(img, (self.camera_width, self.camera_height))
        except Exception as e:
            print(f"  ❌ Failed to load {viewpoint} anchor: {e}. Creating dummy image.")
            dummy = np.full((self.camera_height, self.camera_width, 3), (128, 128, 128), dtype=np.uint8)
            cv2.putText(dummy, f'DUMMY {viewpoint}', (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            return dummy

    def _extract_features_from_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            return self.extractor.extract(tensor)

    def camera_thread(self):
        print("📹 Camera thread started")
        if self.batch_mode:
            self._batch_frame_loader()
        elif self.video_mode:
            self._video_frame_loader()
        else:
            self._camera_frame_capture()
        print("📹 Camera thread stopped")

    def _batch_frame_loader(self):
        for entry in self.image_entries:
            if not self.running: break
            image_path = os.path.join(self.args.image_dir, entry['filename'])
            frame = cv2.imread(image_path)
            if frame is not None:
                buffer_frame_id = self.frame_buffer.update(frame)
                # Simplified metadata handling
                if not hasattr(self, 'current_frame_info'): self.current_frame_info = {}
                self.current_frame_info[buffer_frame_id] = entry
                time.sleep(0.033)
        self.batch_complete = True
        time.sleep(2.0)
        self.running = False

    def _video_frame_loader(self):
        """Load frames from video file"""
        frame_count = 0
        current_video_frame = 0
        
        print(f"🎬 Starting video playback: {self.args.video_file}")
        
        while self.running:
            if self.cap is None:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                print("🎬 Video playback completed")
                self.video_complete = True
                time.sleep(2.0)
                print("📹 Stopping all threads...")
                self.running = False
                break
                
            frame_info = {
                'video_frame_number': current_video_frame,
                'video_timestamp': current_video_frame / self.video_fps,
                'video_file': os.path.basename(self.args.video_file)
            }
            
            buffer_frame_id = self.frame_buffer.update(frame, current_video_frame)
            
            # Store frame info for later retrieval
            if not hasattr(self, 'current_frame_info'):
                self.current_frame_info = {}
            self.current_frame_info[buffer_frame_id] = frame_info
            
            frame_count += 1
            current_video_frame += 1
            
            if frame_count % 300 == 0:
                progress = (current_video_frame / self.total_video_frames) * 100
                print(f"🎬 Video progress: {progress:.1f}% ({current_video_frame}/{self.total_video_frames})")
            
            # Control playback speed to match original video FPS
            # For real-time playback, sleep to maintain video frame rate
            expected_time_per_frame = 1.0 / self.video_fps
            time.sleep(max(0, expected_time_per_frame - 0.01))  # Small buffer for processing
        
        print(f"🎬 Video loading complete: processed {frame_count} frames")

    def _camera_frame_capture(self):
        while self.running:
            if self.cap is None: time.sleep(0.1); continue
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.frame_buffer.update(frame)
            else:
                time.sleep(0.1)

    def processing_thread(self):
        print("⚙️ Processing thread started")
        last_processed_frame_id = -1
        while self.running:
            frame, frame_id, timestamp, video_frame_number = self.frame_buffer.get_latest()
            if frame is None:
                time.sleep(0.01)
                continue
                
            # For batch/video mode, avoid reprocessing same frame
            if (self.batch_mode or self.video_mode) and frame_id == last_processed_frame_id:
                if self.batch_complete or self.video_complete:
                    print("⚙️ Processing complete, stopping processing thread")
                    break
                time.sleep(0.01)
                continue
            
            if frame_id % 30 == 0:
                if self.video_mode:
                    progress = (video_frame_number / self.total_video_frames) * 100 if self.total_video_frames > 0 else 0
                    print(f"🔄 Processing frame {frame_id} (video frame {video_frame_number}, {progress:.1f}%)")
                else:
                    print(f"🔄 Processing frame {frame_id}")
            
            process_start = time.perf_counter()
            result = self._process_frame(frame, frame_id, timestamp)
            result.processing_time = (time.perf_counter() - process_start) * 1000
            last_processed_frame_id = frame_id

            if result.pose_data:
                with self.poses_lock:
                    self.all_poses.append(convert_to_json_serializable(result.pose_data))
            
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
            self.result_queue.put_nowait(result)
        print("⚙️ Processing thread stopped")

    def _process_frame(self, frame, frame_id, timestamp):
        result = ProcessingResult(frame=frame, frame_id=frame_id, timestamp=timestamp)
        frame_info = self.current_frame_info.get(frame_id) if (self.batch_mode or self.video_mode) and hasattr(self, 'current_frame_info') else None

        if self.use_kalman_filter:
            with self.kf_lock:
                pred_result = self.kf.predict()
                if pred_result: result.kf_position, result.kf_quaternion = pred_result

        bbox = self._yolo_detect(frame)
        estimation_frame = frame
        viewpoint_source = "whole_image"

        if bbox:
            result.bbox = bbox
            viewpoint = self._classify_viewpoint(frame, bbox)
            x1, y1, x2, y2 = bbox
            estimation_frame = frame[y1:y2, x1:x2]
            viewpoint_source = "bbox"
        else:
            viewpoint = self._classify_viewpoint_whole_image(frame)

        if estimation_frame.size > 0:
            pos, quat, nm, pose_data = self._estimate_pose_with_data(estimation_frame, viewpoint, bbox, frame_id, frame_info)
            result.position, result.quaternion, result.num_matches, result.pose_data = pos, quat, nm, pose_data
            result.viewpoint = viewpoint

            if self.use_kalman_filter and pos is not None and quat is not None:
                with self.kf_lock:
                    mahal_dist = self.kf.calculate_mahalanobis_distance(pos, quat)
                    result.mahalanobis_distance = mahal_dist
                    if not self.outlier_detector.is_outlier(mahal_dist, self.kf):
                        result.kf_position, result.kf_quaternion = self.kf.update(pos, quat)
                        result.measurement_accepted = True
                    else:
                        result.measurement_accepted = False
        
        return result

    def _yolo_detect(self, frame):
        t_start = time.perf_counter()
        yolo_size = (640, 640)
        yolo_frame = cv2.resize(frame, yolo_size)
        results = self.yolo_model(yolo_frame[..., ::-1], verbose=False, conf=0.5)
        self.perf_monitor.add_timing('yolo_detection', (time.perf_counter() - t_start) * 1000)
        
        if len(results[0].boxes) > 0:
            box = results[0].boxes.xyxy.cpu().numpy()[0]
            scale_x, scale_y = frame.shape[1] / yolo_size[0], frame.shape[0] / yolo_size[1]
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            return max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        return None

    def _classify_viewpoint(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0: return 'NE'
        return self._run_vp_classifier(cropped)

    def _classify_viewpoint_whole_image(self, frame):
        h, w = frame.shape[:2]
        size = min(h, w) // 2
        x1 = (w - size) // 2
        y1 = (h - size) // 2
        cropped = frame[y1:y1+size, x1:x1+size]
        return self._run_vp_classifier(cropped)

    def _run_vp_classifier(self, image):
        t_start = time.perf_counter()
        resized = cv2.resize(image, (128, 128))
        pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        tensor = self.vp_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.vp_model(tensor)
            pred = torch.argmax(logits, dim=1).item()
        self.perf_monitor.add_timing('viewpoint_classification', (time.perf_counter() - t_start) * 1000)
        return self.class_names[pred]

    def _estimate_pose_with_data(self, frame_to_process, viewpoint, bbox, frame_id, frame_info=None):
        t_start = time.perf_counter()
        
        if viewpoint not in self.viewpoint_anchors:
            print(f"  ❌ Viewpoint '{viewpoint}' not found in pre-processed anchors.")
            return None, None, 0, None

        anchor_data = self.viewpoint_anchors[viewpoint]
        anchor_features = anchor_data['features']
        matched_anchor_indices = anchor_data['matched_sp_indices']
        matched_3d_points_map = {idx: pt for idx, pt in zip(matched_anchor_indices, anchor_data['matched_3d_points'])}

        frame_features = self._extract_features_from_image(frame_to_process)
        frame_keypoints = frame_features['keypoints'][0].cpu().numpy()

        with torch.no_grad():
            matches_dict = self.matcher({'image0': anchor_features, 'image1': frame_features})
        
        matches = rbd(matches_dict)['matches'].cpu().numpy()
        num_matches = len(matches)

        if num_matches < 6:
            return None, None, num_matches, self._create_failure_report(frame_id, 'insufficient_matches', num_matches, viewpoint, bbox, frame_info)

        # Find correspondences using pre-computed map
        valid_anchor_sp_indices = matches[:, 0]
        mask = np.isin(valid_anchor_sp_indices, matched_anchor_indices)
        
        if np.sum(mask) < 5:
            return None, None, num_matches, self._create_failure_report(frame_id, 'insufficient_valid_correspondences', num_matches, viewpoint, bbox, frame_info, valid_correspondences=np.sum(mask))

        points_3d = np.array([matched_3d_points_map[i] for i in valid_anchor_sp_indices[mask]])
        points_2d = frame_keypoints[matches[:, 1][mask]]

        if bbox:
            points_2d += np.array([bbox[0], bbox[1]])

        K, dist_coeffs = self._get_camera_intrinsics()
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d.reshape(-1, 1, 3), points_2d.reshape(-1, 1, 2), K, dist_coeffs,
            reprojectionError=6.0, confidence=0.9, iterationsCount=1000, flags=cv2.SOLVEPNP_EPNP)

        if not success or inliers is None or len(inliers) < 4:
            return None, None, num_matches, self._create_failure_report(frame_id, 'pnp_failed', num_matches, viewpoint, bbox, frame_info, num_inliers=len(inliers) if inliers is not None else 0)

        # --- Start Refinement ---
        # 1. Enhance correspondences
        (rvec, tvec), enhanced_3d, enhanced_2d, enhanced_inliers = self.enhance_pose_initialization(
            (rvec, tvec), points_3d[inliers.flatten()], points_2d[inliers.flatten()], viewpoint, frame_to_process)
        
        final_points_3d = enhanced_3d if enhanced_inliers is not None else points_3d[inliers.flatten()]
        final_points_2d = enhanced_2d if enhanced_inliers is not None else points_2d[inliers.flatten()]
        
        # 2. Refine with VVS
        if len(final_points_3d) > 4:
            rvec, tvec = cv2.solvePnPRefineVVS(
                final_points_3d.reshape(-1, 1, 3), final_points_2d.reshape(-1, 1, 2), K, dist_coeffs, rvec, tvec)
        # --- End Refinement ---

        R, _ = cv2.Rodrigues(rvec)
        position = tvec.flatten()
        quaternion = self._rotation_matrix_to_quaternion(R)
        
        duration = (time.perf_counter() - t_start) * 1000
        self.perf_monitor.add_timing('pose_estimation', duration)

        pose_data = self._create_success_report(frame_id, position, quaternion, R, rvec, tvec, num_matches, len(final_points_3d), duration, viewpoint, bbox, frame_info)
        return position, quaternion, num_matches, pose_data

    def enhance_pose_initialization(self, initial_pose, mpts3D, mkpts1, viewpoint, frame, bbox=None):
        rvec, tvec = initial_pose
        K, distCoeffs = self._get_camera_intrinsics()
        
        frame_features = self._extract_features_from_image(frame)
        frame_keypoints = frame_features['keypoints'][0].cpu().numpy()
        
        all_3d_points = self.viewpoint_anchors[viewpoint]['all_3d_points']
        
        projected_points, _ = cv2.projectPoints(all_3d_points, rvec, tvec, K, distCoeffs)
        projected_points = projected_points.reshape(-1, 2)
        
        additional_corrs = []
        for i, model_pt in enumerate(all_3d_points):
            if any(np.all(model_pt == p) for p in mpts3D): continue
            
            proj_pt = projected_points[i]
            distances = np.linalg.norm(frame_keypoints - proj_pt, axis=1)
            min_idx, min_dist = np.argmin(distances), np.min(distances)
            
            if min_dist < 3.0:
                additional_corrs.append((i, min_idx))
        
        if additional_corrs:
            all_3d = np.vstack([mpts3D, all_3d_points[[c[0] for c in additional_corrs]]])
            new_2d_points = frame_keypoints[[c[1] for c in additional_corrs]]
            if bbox:
                new_2d_points += np.array([bbox[0], bbox[1]])
            all_2d = np.vstack([mkpts1, new_2d_points])
            
            success, r, t, inliers = cv2.solvePnPRansac(
                all_3d.reshape(-1, 1, 3), all_2d.reshape(-1, 1, 2), K, distCoeffs,
                rvec=rvec, tvec=tvec, useExtrinsicGuess=True, reprojectionError=4.0, flags=cv2.SOLVEPNP_EPNP)
            
            if success and inliers is not None and len(inliers) >= 6:
                return (r, t), all_3d, all_2d, inliers
        
        return (rvec, tvec), mpts3D, mkpts1, None

    def _get_camera_intrinsics(self):
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K, None

    def _rotation_matrix_to_quaternion(self, R):
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
        return np.array([x, y, z, w])

    def _create_failure_report(self, frame_id, reason, num_matches, viewpoint, bbox, frame_info, **kwargs):
        data = {'frame': int(frame_id), 'pose_estimation_failed': True, 'error_reason': reason, 'num_matches': int(num_matches), 'viewpoint': str(viewpoint), 'bbox': bbox, 'whole_image_estimation': bbox is None, 'method': 'vape_mk45_local_features'}
        data.update(kwargs)
        if frame_info: data.update(convert_to_json_serializable(frame_info))
        return data

    def _create_success_report(self, frame_id, pos, quat, R, rvec, tvec, num_matches, num_inliers, duration, viewpoint, bbox, frame_info):
        data = {'frame': int(frame_id), 'pose_estimation_failed': False, 'method': 'vape_mk45_local_features', 'position': pos.tolist(), 'quaternion': quat.tolist(), 'rotation_matrix': R.tolist(), 'translation_vector': tvec.flatten().tolist(), 'rotation_vector': rvec.flatten().tolist(), 'num_matches': int(num_matches), 'num_inliers': int(num_inliers), 'processing_time_ms': float(duration), 'viewpoint': str(viewpoint), 'bbox': bbox, 'whole_image_estimation': bbox is None}
        if frame_info: data.update(convert_to_json_serializable(frame_info))
        return data

    def display_thread(self):
        print("🖥️ Display thread started")
        if hasattr(self.args, 'no_display') and self.args.no_display:
            while self.running: time.sleep(0.1)
            print("🖥️ Display thread stopped (headless)")
            return

        cv2.namedWindow('VAPE MK45 - Robust Mode', cv2.WINDOW_NORMAL)
        no_result_count = 0
        max_no_result = 50
        
        while self.running:
            try:
                result = self.result_queue.get(timeout=0.1)
                no_result_count = 0
                vis_frame = result.frame.copy()
                
                color = (0, 255, 0) if result.position is not None else (0, 0, 255)
                cv2.putText(vis_frame, f'State: {self.state.value.upper()}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Input mode info
                if self.video_mode:
                    mode_text = f'Video: {os.path.basename(self.args.video_file)}'
                    # Video progress
                    _, _, _, video_frame_number = self.frame_buffer.get_latest()
                    if self.total_video_frames > 0:
                        progress = (video_frame_number / self.total_video_frames) * 100
                        cv2.putText(vis_frame, f'Progress: {progress:.1f}% ({video_frame_number}/{self.total_video_frames})', 
                                    (10, vis_frame.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                elif self.batch_mode:
                    mode_text = 'Batch Mode'
                else:
                    mode_text = 'Camera Mode'
                
                cv2.putText(vis_frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if result.bbox:
                    x1, y1, x2, y2 = result.bbox
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    if result.viewpoint:
                        cv2.putText(vis_frame, f'VP: {result.viewpoint}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Show number of matches
                if result.num_matches > 0:
                    match_color = (0, 255, 0) if result.num_matches > 20 else (0, 165, 255) if result.num_matches > 10 else (0, 0, 255)
                    cv2.putText(vis_frame, f'Matches: {result.num_matches}', (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)

                display_pos = result.kf_position if result.kf_position is not None else result.position
                display_quat = result.kf_quaternion if result.kf_quaternion is not None else result.quaternion

                if display_pos is not None and display_quat is not None:
                    vis_frame = self._draw_axes(vis_frame, display_pos, display_quat)

                # Show processing time
                cv2.putText(vis_frame, f'Processing: {result.processing_time:.1f}ms', (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                # Show pose values
                if display_pos is not None:
                    pos_text = f'Pos: [{display_pos[0]:.3f}, {display_pos[1]:.3f}, {display_pos[2]:.3f}]'
                    cv2.putText(vis_frame, pos_text, (10, vis_frame.shape[0]-40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if display_quat is not None:
                    quat_text = f'Quat: [{display_quat[0]:.3f}, {display_quat[1]:.3f}, {display_quat[2]:.3f}, {display_quat[3]:.3f}]'
                    cv2.putText(vis_frame, quat_text, (10, vis_frame.shape[0]-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Show JSON export info
                with self.poses_lock:
                    json_count = len(self.all_poses)
                cv2.putText(vis_frame, f'JSON Entries: {json_count}', (10, vis_frame.shape[0]-60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow('VAPE MK45 - Robust Mode', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    
            except queue.Empty:
                no_result_count += 1
                
                if (self.batch_mode and self.batch_complete) or (self.video_mode and self.video_complete):
                    if no_result_count > max_no_result:
                        print("🖥️ No more results, stopping display")
                        break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    
        cv2.destroyAllWindows()
        print("🖥️ Display thread stopped")

    def _draw_axes(self, frame, position, quaternion):
        try:
            R = self._quaternion_to_rotation_matrix(quaternion)
            rvec, _ = cv2.Rodrigues(R)
            tvec = position.reshape(3, 1)
            K, distCoeffs = self._get_camera_intrinsics()
            axis_pts = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, distCoeffs)
            img_pts = img_pts.reshape(-1, 2).astype(int)
            
            origin = tuple(img_pts[0])
            cv2.line(frame, origin, tuple(img_pts[1]), (0,0,255), 3) # X
            cv2.line(frame, origin, tuple(img_pts[2]), (0,255,0), 3) # Y
            cv2.line(frame, origin, tuple(img_pts[3]), (255,0,0), 3) # Z
        except Exception:
            pass
        return frame

    def _quaternion_to_rotation_matrix(self, q):
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

    def start(self):
        self.running = True
        self.threads = [
            threading.Thread(target=self.camera_thread, daemon=True),
            threading.Thread(target=self.processing_thread, daemon=True),
            threading.Thread(target=self.display_thread, daemon=True)
        ]
        for t in self.threads: t.start()
        print("✅ All threads started.")

    def stop(self):
        print("🛑 Stopping all threads...")
        self.running = False
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2.0)
        
        if hasattr(self, 'cap') and self.cap: self.cap.release()
        cv2.destroyAllWindows()

        if self.all_poses:
            output_filename = create_unique_filename(self.args.output_dir, 'vape_mk45_pose_results_indoor.json')
            print(f"💾 Saving {len(self.all_poses)} pose records to {output_filename}")
            with open(output_filename, 'w') as f:
                json.dump(self.all_poses, f, indent=4)
        print("✅ Shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description="VAPE MK45 - Robust Pose Estimator with MP4 Support")
    
    # Input mode arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--video_file', type=str, default=None,
                            help='MP4/AVI video file path (enables video mode)')
    input_group.add_argument('--image_dir', type=str, default=None,
                            help='Directory of images for batch processing.')
    
    parser.add_argument('--csv_file', type=str, help='CSV file with image timestamps for batch mode.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save JSON results.')
    parser.add_argument('--no_display', action='store_true', help='Run in headless mode without display.')
    parser.add_argument('--no_kalman_filter', action='store_false', dest='use_kalman_filter', help='Disable the Kalman filter.')
    args = parser.parse_args()

    # Validate arguments
    if args.image_dir and not args.csv_file:
        print("❌ Error: --csv_file is required when using --image_dir")
        sys.exit(1)
    
    if args.video_file and not os.path.exists(args.video_file):
        print(f"❌ Error: Video file not found: {args.video_file}")
        sys.exit(1)

    print("🚀 VAPE MK45 - Robust Pose Estimator")
    print("=" * 70)
    print("📊 METHOD: Local Features (SuperPoint + LightGlue)")
    print("   Advanced correspondence matching with viewpoint classification")
    print("   Kalman filtering for temporal consistency")
    print("=" * 70)
    print("📖 Usage Examples:")
    print("  Camera mode:")
    print("    python VAPE_MK45_Robust.py")
    print()
    print("  Video mode (NEW!):")
    print("    python VAPE_MK45_Robust.py --video_file /path/to/your_video.mp4")
    print()
    print("  Batch mode:")
    print("    python VAPE_MK45_Robust.py --image_dir /path/to/images --csv_file timestamps.csv")
    print("=" * 70)

    estimator = MultiThreadedPoseEstimator(args)
    
    def signal_handler(sig, frame):
        print('SIGINT received, shutting down gracefully.')
        estimator.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    estimator.start()

    try:
        # In camera mode, wait indefinitely. In batch/video mode, wait for completion.
        if estimator.batch_mode or estimator.video_mode:
            estimator.threads[0].join() # Wait for camera (batch/video loader) thread
            estimator.threads[1].join() # Wait for processing thread
        else:
            while estimator.running:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        estimator.stop()

if __name__ == '__main__':
    main()