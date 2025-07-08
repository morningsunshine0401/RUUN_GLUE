"""
VAPE MK46 Robust Fallback: 다단계 백업 전략으로 무조건 포즈 추정
- Level 1: YOLO 실패 → 전체 이미지를 crop으로 사용
- Level 2: MobileViT 실패 → 다른 anchor들 순차 시도  
- Level 3: Anchor 매칭 실패 → 최근 성공 프레임을 임시 anchor로 사용
- Level 4: Dynamic anchor 품질 저하 → 원래 anchor로 복귀
- 목표: 어떤 상황에서도 포즈 추정 성공
- 새로운 기능: 실시간 웹캠 지원 + Match 시각화
"""

import cv2
import queue
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

# Import for robust correspondence matching
from scipy.spatial import cKDTree

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 Starting VAPE MK46 Robust Fallback with Realtime Support + Match Visualization...")

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

# Enhanced State Machine
class TrackingState(Enum):
    DETECTING = "detecting"
    ESTIMATING = "estimating"

class FallbackLevel(Enum):
    NORMAL = "normal"           # YOLO + MobileViT + Original Anchor
    NO_YOLO = "no_yolo"         # Full image + MobileViT + Original Anchor  
    MULTI_ANCHOR = "multi_anchor"  # Try all anchors
    DYNAMIC_ANCHOR = "dynamic_anchor"  # Use recent successful frame
    EMERGENCY = "emergency"     # Last resort

@dataclass
class MatchVisualizationData:
    """Data for match visualization"""
    anchor_image: np.ndarray
    anchor_keypoints: np.ndarray  # 2D keypoints in anchor image
    frame_keypoints: np.ndarray   # 2D keypoints in input frame
    matches: np.ndarray          # Match indices [anchor_idx, frame_idx]
    viewpoint: str
    used_for_pose: np.ndarray    # Boolean mask indicating which matches were used for pose estimation

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
    fallback_level: FallbackLevel = FallbackLevel.NORMAL
    fallback_reason: str = ""
    anchor_type: str = ""               
    num_inliers: int = 0                
    # NEW: Match visualization data
    match_vis_data: Optional[MatchVisualizationData] = None

@dataclass
class DynamicAnchor:
    """Dynamic anchor from successful recent frame"""
    frame_id: int
    features: Dict
    keypoints_2d: np.ndarray
    keypoints_3d: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    num_matches: int
    reprojection_error: float
    quality_score: float
    usage_count: int = 0
    created_timestamp: float = 0.0

class LooselyCoupledKalmanFilter:
    def __init__(self, dt=1/30.0):
        self.dt = dt
        self.initialized = False
        self.n_states = 13
        self.x = np.zeros(self.n_states)
        self.x[9] = 1.0  # quaternion w=1
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
        
        # Constant velocity model
        px_new, py_new, pz_new = px + vx * dt, py + vy * dt, pz + vz * dt
        vx_new, vy_new, vz_new = vx, vy, vz
        
        # Quaternion integration
        q = np.array([qx, qy, qz, qw])
        w = np.array([wx, wy, wz])
        omega_mat = np.array([
            [0, -wx, -wy, -wz], 
            [wx, 0, wz, -wy], 
            [wy, -wz, 0, wx], 
            [wz, wy, -wx, 0]
        ])
        dq = 0.5 * dt * omega_mat @ q
        q_new = self.normalize_quaternion(q + dq)
        wx_new, wy_new, wz_new = wx, wy, wz
        
        self.x = np.array([
            px_new, py_new, pz_new, vx_new, vy_new, vz_new,
            q_new[0], q_new[1], q_new[2], q_new[3], wx_new, wy_new, wz_new
        ])
        
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
            
        predicted_measurement = np.array([
            self.x[0], self.x[1], self.x[2], 
            self.x[6], self.x[7], self.x[8], self.x[9]
        ])
        innovation = measurement - predicted_measurement
        
        # Handle quaternion wraparound
        q_meas, q_pred = measurement[3:7], predicted_measurement[3:7]
        if np.dot(q_meas, q_pred) < 0:
            innovation[3:7] = -q_meas - q_pred
            
        H = np.zeros((7, self.n_states))
        H[0:3, 0:3] = np.eye(3)  # Position
        H[3:7, 6:10] = np.eye(4)  # Quaternion
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ innovation
        self.x[6:10] = self.normalize_quaternion(self.x[6:10])
        
        I = np.eye(self.n_states)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        return self.x[0:3], self.x[6:10]

    def calculate_mahalanobis_distance(self, position, quaternion):
        if not self.initialized: 
            return 0.0
        measurement = np.concatenate([position, quaternion])
        predicted_measurement = np.array([
            self.x[0], self.x[1], self.x[2], 
            self.x[6], self.x[7], self.x[8], self.x[9]
        ])
        innovation = measurement - predicted_measurement
        
        q_meas, q_pred = measurement[3:7], predicted_measurement[3:7]
        if np.dot(q_meas, q_pred) < 0:
            innovation[3:7] = -q_meas - q_pred
            
        H = np.zeros((7, self.n_states))
        H[0:3, 0:3] = np.eye(3)
        H[3:7, 6:10] = np.eye(4)
        S = H @ self.P @ H.T + self.R
        
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
        # Always record the new distance
        self.history.append(mahalanobis_distance)
        # Don't reject until we have enough samples
        if not kf.initialized or len(self.history) < self.min_measurements:
            return False
        # Now decide
        return mahalanobis_distance > self.threshold

class RobustFallbackManager:
    """🛡️ 다단계 백업 전략 관리자"""
    
    def __init__(self, max_dynamic_anchors=5):
        self.current_level = FallbackLevel.NORMAL
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        
        # Dynamic anchor management
        self.dynamic_anchors = deque(maxlen=max_dynamic_anchors)
        self.current_dynamic_anchor = None
        self.dynamic_anchor_usage_count = 0
        self.max_dynamic_usage = 50  # 최대 50프레임까지 사용
        
        # Quality monitoring
        self.recent_matches_history = deque(maxlen=10)
        self.recent_error_history = deque(maxlen=10)
        
        # Anchor trying order for multi-anchor fallback
        self.anchor_try_order = ['NE', 'SE', 'NW', 'SW']
        self.current_anchor_idx = 0
        
    def should_escalate_fallback(self, num_matches, reprojection_error=None):
        """백업 레벨을 올려야 하는지 결정"""
        if num_matches < 6:  # 매칭 부족
            self.consecutive_failures += 1
        elif reprojection_error and reprojection_error > 8.0:  # 에러 큼
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
            return False
            
        return self.consecutive_failures >= self.max_consecutive_failures
    
    def get_next_fallback_level(self):
        """다음 백업 레벨 결정"""
        if self.current_level == FallbackLevel.NORMAL:
            return FallbackLevel.NO_YOLO
        elif self.current_level == FallbackLevel.NO_YOLO:
            return FallbackLevel.MULTI_ANCHOR
        elif self.current_level == FallbackLevel.MULTI_ANCHOR:
            return FallbackLevel.DYNAMIC_ANCHOR
        else:
            return FallbackLevel.EMERGENCY
    
    def escalate_fallback(self, reason=""):
        """백업 레벨 상승"""
        old_level = self.current_level
        self.current_level = self.get_next_fallback_level()
        self.consecutive_failures = 0
        print(f"🔄 Fallback escalated: {old_level.value} → {self.current_level.value} ({reason})")
        
    def recover_fallback(self, num_matches, reprojection_error):
        """백업 레벨 복구 (성공적인 결과가 연속으로 나올 때)"""
        self.recent_matches_history.append(num_matches)
        if reprojection_error:
            self.recent_error_history.append(reprojection_error)
        
        # 최근 성능이 좋으면 레벨 다운
        if (len(self.recent_matches_history) >= 5 and 
            np.mean(list(self.recent_matches_history)) > 15 and
            len(self.recent_error_history) >= 5 and
            np.mean(list(self.recent_error_history)) < 3.0):
            
            if self.current_level != FallbackLevel.NORMAL:
                old_level = self.current_level
                self.current_level = FallbackLevel.NORMAL
                self.current_dynamic_anchor = None  # 리셋
                print(f"🔄 Fallback recovered: {old_level.value} → {self.current_level.value}")
    
    def add_dynamic_anchor(self, anchor: DynamicAnchor):
        """새로운 dynamic anchor 추가"""
        anchor.created_timestamp = time.time()
        self.dynamic_anchors.append(anchor)
        self.current_dynamic_anchor = anchor
        print(f"✅ Added dynamic anchor from frame {anchor.frame_id} (quality: {anchor.quality_score:.2f})")
    
    def get_best_dynamic_anchor(self):
        """가장 좋은 dynamic anchor 반환"""
        if not self.dynamic_anchors:
            return None
        return max(self.dynamic_anchors, key=lambda x: x.quality_score)
    
    def should_update_dynamic_anchor(self, current_quality):
        """현재 dynamic anchor를 업데이트해야 하는지"""
        if not self.current_dynamic_anchor:
            return True
        
        # 품질이 크게 향상되었거나, 사용 횟수가 많아졌을 때
        return (current_quality > self.current_dynamic_anchor.quality_score + 0.2 or
                self.dynamic_anchor_usage_count > self.max_dynamic_usage)

def read_image_index_csv(csv_path):
    entries = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                'index': int(row['Index']), 
                'timestamp': float(row['Timestamp']), 
                'filename': row['Filename']
            })
    return entries

def create_unique_filename(directory, base_filename):
    base_path = os.path.join(directory or ".", base_filename)
    if not os.path.exists(base_path): 
        return base_path
    name, ext = os.path.splitext(base_filename)
    counter = 1
    while True:
        new_path = os.path.join(directory, f"{name}_{counter}{ext}")
        if not os.path.exists(new_path): 
            return new_path
        counter += 1

def convert_to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)): 
        return obj.item()
    if isinstance(obj, np.ndarray): 
        return obj.tolist()
    if isinstance(obj, dict): 
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): 
        return [convert_to_json_serializable(i) for i in obj]
    return obj

class ThreadSafeFrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_id = 0
        self.timestamp = None

    def update(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()
            self.frame_id += 1
            self.timestamp = time.perf_counter()
            return self.frame_id

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None: 
                return None, None, None
            return self.latest_frame.copy(), self.frame_id, self.timestamp

class PerformanceMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.timings = defaultdict(lambda: deque(maxlen=30))
        self.fallback_stats = defaultdict(int)
        self.fps_history = deque(maxlen=30)
        self.last_fps_time = time.time()
        self.fps_frame_count = 0

    def add_timing(self, name: str, duration: float):
        with self.lock:
            self.timings[name].append(duration)
    
    def add_fallback_stat(self, level: FallbackLevel):
        with self.lock:
            self.fallback_stats[level.value] += 1

    def update_fps(self):
        """Update FPS calculation"""
        with self.lock:
            current_time = time.time()
            self.fps_frame_count += 1
            
            if current_time - self.last_fps_time >= 1.0:  # Calculate FPS every second
                fps = self.fps_frame_count / (current_time - self.last_fps_time)
                self.fps_history.append(fps)
                self.fps_frame_count = 0
                self.last_fps_time = current_time

    def get_average_fps(self) -> float:
        with self.lock:
            if self.fps_history:
                return np.mean(list(self.fps_history))
        return 0.0

    def get_average(self, name: str) -> float:
        with self.lock:
            if name in self.timings and self.timings[name]:
                return np.mean(list(self.timings[name]))
        return 0.0
    
    def get_fallback_stats(self):
        with self.lock:
            return dict(self.fallback_stats)

class RobustFallbackPoseEstimator:
    def __init__(self, args):
        print("🔧 Initializing VAPE MK46 Robust Fallback...")
        self.args = args

        # ── ADD THESE ──
        self.kf_missed_count     = 0
        self.kf_missed_threshold = 5    # e.g. reset after 5 consecutive misses
        # ──────────────

        self.running = False
        self.threads = []
        
        # 🛡️ Fallback manager
        self.fallback_manager = RobustFallbackManager()
        
        # Processing mode detection
        self.batch_mode = hasattr(args, 'image_dir') and args.image_dir is not None
        self.sequential_processing = self.batch_mode
        
        # Initialize frame buffer and queues for realtime mode
        if not self.sequential_processing:
            self.frame_buffer = ThreadSafeFrameBuffer()
            self.result_queue = queue.Queue(maxsize=2)  # Increased buffer
        self.display_queue = queue.Queue(maxsize=2)
        
        self.perf_monitor = PerformanceMonitor()
        self.state = TrackingState.DETECTING
        
        # Kalman filter setup
        self.kf = LooselyCoupledKalmanFilter(dt=1/30.0)
        self.kf_initialized = False
        self.outlier_detector = OutlierDetector()
        self.use_kalman_filter = getattr(args, 'use_kalman_filter', True)
        
        self.all_poses = []
        self.poses_lock = threading.Lock()
        self.kf_lock = threading.Lock()
        
        # Image processing setup
        self.image_entries = []
        self.batch_complete = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")
        self.camera_width, self.camera_height = 1280, 720
        
        self._init_models()
        if self.batch_mode: 
            self._init_batch_processing()
        else: 
            self._init_camera()
        self._init_anchor_data()
        
        print("✅ VAPE MK46 Robust Fallback initialized!")

    def _init_batch_processing(self):
        if hasattr(self.args, 'csv_file') and self.args.csv_file:
            self.image_entries = read_image_index_csv(self.args.csv_file)
            self.image_entries.sort(key=lambda x: x['index'])
            print(f"✅ Loaded {len(self.image_entries)} image entries from CSV")
        else:
            if hasattr(self.args, 'image_dir') and os.path.exists(self.args.image_dir):
                image_files = sorted([f for f in os.listdir(self.args.image_dir) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                self.image_entries = [
                    {'index': i, 'timestamp': i/30.0, 'filename': f} 
                    for i, f in enumerate(image_files)
                ]
                print(f"✅ Found {len(self.image_entries)} images in directory")

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
            self.vp_transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
            print("  📦 Loading SuperPoint & LightGlue...")
            self.extractor = SuperPoint(max_num_keypoints=512).eval().to(self.device)
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)
            self.class_names = ['NE', 'NW', 'SE', 'SW']
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

    def _init_camera(self):
        try:
            # Try multiple camera indices
            for cam_idx in [0, 1, 2]:
                self.cap = cv2.VideoCapture(cam_idx)
                if self.cap.isOpened():
                    break
            else:
                raise IOError("Cannot open any camera")
                
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test camera
            ret, frame = self.cap.read()
            if not ret: 
                raise IOError("Cannot read from camera")
                
            self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ Camera initialized: {self.camera_width}x{self.camera_height} @ {actual_fps}fps")
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            self.cap = None
            raise

    def _init_anchor_data(self):
        print("🛠️ Initializing anchor data with KDTree...")
        default_anchor_paths = {
            'NE': 'NE.png',
            'NW': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png',
            'SE': 'SE.png',
            'SW': 'Anchor_B.png'
        }
        
        # Define 2D/3D correspondences
        default_anchor_2d = np.array([
            [511, 293], [591, 284], [587, 330], [413, 249], [602, 348],
            [715, 384], [598, 298], [656, 171], [805, 213], [703, 392],
            [523, 286], [519, 327], [387, 289], [727, 126], [425, 243],
            [636, 358], [745, 202], [595, 388], [436, 260], [539, 313],
            [795, 220], [351, 291], [665, 165], [611, 353], [650, 377],
            [516, 389], [727, 143], [496, 378], [575, 312], [617, 368],
            [430, 312], [480, 281], [834, 225], [469, 339], [705, 223],
            [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]
        ], dtype=np.float32)
        
        default_anchor_3d = np.array([
            [-0.014, 0.000, 0.042], [0.025, -0.014, -0.011], [-0.014, 0.000, -0.042],
            [-0.014, 0.000, 0.156], [-0.023, 0.000, -0.065], [0.000, 0.000, -0.156],
            [0.025, 0.000, -0.015], [0.217, 0.000, 0.070], [0.230, 0.000, -0.070],
            [-0.014, 0.000, -0.156], [0.000, 0.000, 0.042], [-0.057, -0.018, -0.010],
            [-0.074, -0.000, 0.128], [0.206, -0.070, -0.002], [-0.000, -0.000, 0.156],
            [-0.017, -0.000, -0.092], [0.217, -0.000, -0.027], [-0.052, -0.000, -0.097],
            [-0.019, -0.000, 0.128], [-0.035, -0.018, -0.010], [0.217, -0.000, -0.070],
            [-0.080, -0.000, 0.156], [0.230, -0.000, 0.070], [-0.023, -0.000, -0.075],
            [-0.029, -0.000, -0.127], [-0.090, -0.000, -0.042], [0.206, -0.055, -0.002],
            [-0.090, -0.000, -0.015], [0.000, -0.000, -0.015], [-0.037, -0.000, -0.097],
            [-0.074, -0.000, 0.074], [-0.019, -0.000, 0.074], [0.230, -0.000, -0.113],
            [-0.100, -0.030, 0.000], [0.170, -0.000, -0.015], [0.230, -0.000, 0.113],
            [-0.000, -0.025, -0.240], [-0.000, -0.025, 0.240], [0.243, -0.104, 0.000],
            [-0.080, -0.000, -0.156]
        ], dtype=np.float32)
        
        sw_anchor_2d = np.array([
            [650, 312], [630, 306], [907, 443], [814, 291], [599, 349],
            [501, 386], [965, 359], [649, 355], [635, 346], [930, 335],
            [843, 467], [702, 339], [718, 321], [930, 322], [727, 346],
            [539, 364], [786, 297], [1022, 406], [1004, 399], [539, 344],
            [536, 309], [864, 478], [745, 310], [1049, 393], [895, 258],
            [674, 347], [741, 281], [699, 294], [817, 494], [992, 281]
        ], dtype=np.float32)
        
        sw_anchor_3d = np.array([
            [-0.035, -0.018, -0.010], [-0.057, -0.018, -0.010], [0.217, -0.000, -0.027],
            [-0.014, -0.000, 0.156], [-0.023, -0.000, -0.065], [-0.014, -0.000, -0.156],
            [0.234, -0.050, -0.002], [0.000, -0.000, -0.042], [-0.014, -0.000, -0.042],
            [0.206, -0.055, -0.002], [0.217, -0.000, -0.070], [0.025, -0.014, -0.011],
            [-0.014, -0.000, 0.042], [0.206, -0.070, -0.002], [0.049, -0.016, -0.011],
            [-0.029, -0.000, -0.127], [-0.019, -0.000, 0.128], [0.230, -0.000, 0.070],
            [0.217, -0.000, 0.070], [-0.052, -0.000, -0.097], [-0.175, -0.000, -0.015],
            [0.230, -0.000, -0.070], [-0.019, -0.000, 0.074], [0.230, -0.000, 0.113],
            [-0.000, -0.025, 0.240], [-0.000, -0.000, -0.015], [-0.074, -0.000, 0.128],
            [-0.074, -0.000, 0.074], [0.230, -0.000, -0.113], [0.243, -0.104, 0.000]
        ], dtype=np.float32)

        ne_anchor_2d = np.array([
            [924, 148], [571, 115], [398, 31], [534, 133], [544, 141],
            [341, 219], [351, 228], [298, 240], [420, 83], [225, 538],
            [929, 291], [794, 381], [485, 569], [826, 305], [813, 264],
            [791, 285], [773, 271], [760, 289], [830, 225], [845, 233],
            [703, 308], [575, 361], [589, 373], [401, 469], [414, 481],
            [606, 454], [548, 399], [521, 510], [464, 451], [741, 380]
        ], dtype=np.float32)

        ne_anchor_3d = np.array([
            [-0.0, -0.025, -0.24],
            [0.23, 0.0, -0.113],
            [0.243, -0.104, 0.0],
            [0.23, 0.0, -0.07],
            [0.217, 0.0, -0.07],
            [0.23, 0.0, 0.07],
            [0.217, 0.0, 0.07],
            [0.23, 0.0, 0.113],
            [0.206, -0.07, -0.002],
            [-0.0, -0.025, 0.24],
            [-0.08, 0.0, -0.156],
            [-0.09, 0.0, -0.042],
            [-0.08, 0.0, 0.156],
            [-0.052, 0.0, -0.097],
            [-0.029, 0.0, -0.127],
            [-0.037, 0.0, -0.097],
            [-0.017, 0.0, -0.092],
            [-0.023, 0.0, -0.075],
            [0.0, 0.0, -0.156],
            [-0.014, 0.0, -0.156],
            [-0.014, 0.0, -0.042],
            [0.0, 0.0, 0.042],
            [-0.014, 0.0, 0.042],
            [-0.0, 0.0, 0.156],
            [-0.014, 0.0, 0.156],
            [-0.074, 0.0, 0.074],
            [-0.019, 0.0, 0.074],
            [-0.074, 0.0, 0.128],
            [-0.019, 0.0, 0.128],
            [-0.1, -0.03, 0.0]
        ], dtype=np.float32)

        se_anchor_2d = np.array([
            [415, 144], [1169, 508], [275, 323], [214, 395], [554, 670],
            [253, 428], [280, 415], [355, 365], [494, 621], [519, 600],
            [806, 213], [973, 438], [986, 421], [768, 343], [785, 328],
            [841, 345], [931, 393], [891, 306], [980, 345], [651, 210],
            [625, 225], [588, 216], [511, 215], [526, 204], [665, 271]
        ], dtype=np.float32)

        se_anchor_3d = np.array([
            [-0.0, -0.025, -0.24],
            [-0.0, -0.025, 0.24],
            [0.243, -0.104, 0.0],
            [0.23, 0.0, -0.113],
            [0.23, 0.0, 0.113],
            [0.23, 0.0, -0.07],
            [0.217, 0.0, -0.07],
            [0.206, -0.07, -0.002],
            [0.23, 0.0, 0.07],
            [0.217, 0.0, 0.07],
            [-0.1, -0.03, 0.0],
            [-0.0, 0.0, 0.156],
            [-0.014, 0.0, 0.156],
            [0.0, 0.0, 0.042],
            [-0.014, 0.0, 0.042],
            [-0.019, 0.0, 0.074],
            [-0.019, 0.0, 0.128],
            [-0.074, 0.0, 0.074],
            [-0.074, 0.0, 0.128],
            [-0.052, 0.0, -0.097],
            [-0.037, 0.0, -0.097],
            [-0.029, 0.0, -0.127],
            [0.0, 0.0, -0.156],
            [-0.014, 0.0, -0.156],
            [-0.014, 0.0, -0.042]
        ], dtype=np.float32)
        
        viewpoint_data = {
            'NE': {'2d': ne_anchor_2d, '3d': ne_anchor_3d},
            'NW': {'2d': default_anchor_2d, '3d': default_anchor_3d},
            'SE': {'2d': se_anchor_2d, '3d': se_anchor_3d},
            'SW': {'2d': sw_anchor_2d, '3d': sw_anchor_3d}
        }

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

            # Build KDTree correspondence mapping
            sp_tree = cKDTree(anchor_keypoints_sp)
            distances, indices = sp_tree.query(anchor_2d, k=1)
            
            valid_matches = distances < 5.0
            matched_sp_indices = indices[valid_matches]
            matched_3d_points = anchor_3d[valid_matches]

            print(f"    Found {len(matched_sp_indices)} valid 2D-3D correspondences for {viewpoint}.")

            self.viewpoint_anchors[viewpoint] = {
                'features': anchor_features,
                'anchor_image': anchor_image,
                'all_3d_points': anchor_3d,
                'matched_sp_indices': matched_sp_indices,
                'matched_3d_points': matched_3d_points,
                # NEW: Store predefined 2D points for visualization
                'predefined_2d_points': anchor_2d,
                'predefined_3d_points': anchor_3d
            }
        print("✅ Anchor data initialization complete.")

    def _load_anchor_image(self, path, viewpoint):
        try:
            img = cv2.imread(path)
            if img is None: 
                raise FileNotFoundError(f"File not found or could not be read: {path}")
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

    # 🛡️ NEW: Robust fallback processing for batch mode
    def process_all_images_robustly(self):
        """🛡️ 강력한 백업 전략으로 모든 이미지 처리"""
        if not self.batch_mode:
            print("❌ Robust processing is only for batch mode!")
            return

        print(f"🛡️ Starting robust fallback processing of {len(self.image_entries)} images...")
        start_time = time.time()
        processed_count = 0
        success_count = 0
        fallback_usage = defaultdict(int)

        for i, entry in enumerate(self.image_entries):
            if not self.running:
                break

            # Load image
            image_path = os.path.join(self.args.image_dir, entry['filename'])
            if not os.path.exists(image_path):
                print(f"⚠️ Image not found: {image_path}")
                continue

            frame = cv2.imread(image_path)
            if frame is None:
                print(f"⚠️ Failed to load image: {image_path}")
                continue

            # 🛡️ Robust processing with fallback
            frame_id = entry['index']
            timestamp = entry['timestamp']
            
            process_start = time.perf_counter()
            result = self._process_frame_with_fallback(frame, frame_id, timestamp, entry)
            process_end = time.perf_counter()
            
            result.processing_time = (process_end - process_start) * 1000
            processed_count += 1

            # Track fallback usage
            fallback_usage[result.fallback_level.value] += 1
            self.perf_monitor.add_fallback_stat(result.fallback_level)

            # Save pose data
            if result.pose_data:
                with self.poses_lock:
                    self.all_poses.append(convert_to_json_serializable(result.pose_data))
                if not result.pose_data.get('pose_estimation_failed', True):
                    success_count += 1

            # Progress reporting
            if processed_count % 50 == 0 or processed_count == len(self.image_entries):
                elapsed = time.time() - start_time
                fps = processed_count / elapsed if elapsed > 0 else 0
                print(f"📊 Progress: {processed_count}/{len(self.image_entries)} "
                      f"({processed_count/len(self.image_entries)*100:.1f}%) "
                      f"Success: {success_count} "
                      f"FPS: {fps:.1f} "
                      f"Fallback: {result.fallback_level.value}")

            # Show result if display enabled
            if not getattr(self.args, 'no_display', False):
                self._display_result(result)

        total_time = time.time() - start_time
        print(f"✅ Robust fallback processing complete!")
        print(f"   Total images: {len(self.image_entries)}")
        print(f"   Processed: {processed_count}")
        print(f"   Successful poses: {success_count}")
        print(f"   Success rate: {success_count/processed_count*100:.1f}%")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average FPS: {processed_count/total_time:.1f}")
        print(f"🛡️ Fallback statistics:")
        for level, count in fallback_usage.items():
            print(f"   {level}: {count} ({count/processed_count*100:.1f}%)")

    # 🎥 NEW: Realtime webcam processing
    def process_realtime_webcam(self):
        """🎥 실시간 웹캠 처리"""
        print("🎥 Starting realtime webcam processing with fallback support...")
        
        if not self.cap or not self.cap.isOpened():
            print("❌ Camera not available for realtime processing")
            return

        # Initialize display window if not headless
        if not getattr(self.args, 'no_display', False):
            window_name = 'VAPE MK46 Robust Fallback - Realtime'
            if getattr(self.args, 'vis_match', False):
                window_name += ' - Match Visualization'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280 if not getattr(self.args, 'vis_match', False) else 2560, 720)
            
            print("🎮 Controls:")
            print("  'q' - Quit")
            print("  'r' - Reset fallback level to normal")
            print("  's' - Save current statistics")

        frame_count = 0
        success_count = 0
        start_time = time.time()
        last_stats_time = start_time

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Failed to read frame from camera")
                continue

            frame_count += 1
            current_time = time.time()
            
            # Process frame with fallback
            result = self._process_frame_with_fallback(
                frame, frame_count, current_time, None
            )
            
            # Track statistics
            self.perf_monitor.update_fps()
            self.perf_monitor.add_fallback_stat(result.fallback_level)
            
            if result.pose_data and not result.pose_data.get('pose_estimation_failed', True):
                success_count += 1
                
                # Save pose data for realtime mode too
                with self.poses_lock:
                    self.all_poses.append(convert_to_json_serializable(result.pose_data))

            # Display live video with overlays
            if not getattr(self.args, 'no_display', False):
                if getattr(self.args, 'vis_match', False):
                    vis_frame = self._create_match_visualization(result)
                else:
                    vis_frame = self._create_realtime_display(result)
                cv2.imshow(window_name, vis_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('r'):
                # Reset fallback level
                old_level = self.fallback_manager.current_level
                self.fallback_manager.current_level = FallbackLevel.NORMAL
                self.fallback_manager.consecutive_failures = 0
                print(f"🔄 Fallback level reset: {old_level.value} → {FallbackLevel.NORMAL.value}")
            elif key == ord('s'):
                # Save current statistics
                self._save_realtime_statistics()

            # Print periodic statistics
            if current_time - last_stats_time >= 5.0:  # Every 5 seconds
                elapsed = current_time - start_time
                overall_fps = frame_count / elapsed if elapsed > 0 else 0
                avg_fps = self.perf_monitor.get_average_fps()
                success_rate = (success_count / frame_count * 100) if frame_count > 0 else 0
                
                fallback_stats = self.perf_monitor.get_fallback_stats()
                current_level = self.fallback_manager.current_level.value
                
                print(f"📊 Realtime Stats - Frame: {frame_count}, FPS: {avg_fps:.1f}, "
                      f"Success: {success_rate:.1f}%, Current Fallback: {current_level}")
                
                last_stats_time = current_time

        # Close display window
        if not getattr(self.args, 'no_display', False):
            cv2.destroyAllWindows()

        # Final statistics
        total_time = time.time() - start_time
        print(f"🎥 Realtime processing complete!")
        print(f"   Total frames: {frame_count}")
        print(f"   Successful poses: {success_count}")
        print(f"   Success rate: {success_count/frame_count*100:.1f}%")
        print(f"   Average FPS: {frame_count/total_time:.1f}")
        
        fallback_stats = self.perf_monitor.get_fallback_stats()
        print(f"🛡️ Fallback usage:")
        total_fallback_usage = sum(fallback_stats.values())
        for level, count in fallback_stats.items():
            percentage = count / total_fallback_usage * 100 if total_fallback_usage > 0 else 0
            print(f"   {level}: {count} ({percentage:.1f}%)")

    def _process_frame_with_fallback(self, frame, frame_id, timestamp, frame_info):
        """🛡️ 다단계 백업 전략으로 프레임 처리 - 무조건 성공시킴"""
        result = ProcessingResult(frame=frame, frame_id=frame_id, timestamp=timestamp)

        # Kalman filter prediction
        if self.use_kalman_filter:
            with self.kf_lock:
                pred_result = self.kf.predict()
                if pred_result: 
                    result.kf_position, result.kf_quaternion = pred_result

        # 🛡️ Level 1: 정상 처리 (YOLO + MobileViT + Original Anchor)
        success = False
        attempts = []
        
        if self.fallback_manager.current_level == FallbackLevel.NORMAL:
            pos, quat, nm, pose_data, error, reason, match_vis_data = self._try_normal_processing(
                frame, frame_id, frame_info
            )
            attempts.append(f"Normal: {reason}")
            
            if pos is not None:
                success = True
                result.fallback_level = FallbackLevel.NORMAL
                result.match_vis_data = match_vis_data
            elif self.fallback_manager.should_escalate_fallback(nm, error):
                self.fallback_manager.escalate_fallback("Normal processing failed")

        # 🛡️ Level 2: YOLO 없이 (Full Image + MobileViT + Original Anchor)
        if not success and self.fallback_manager.current_level == FallbackLevel.NO_YOLO:
            pos, quat, nm, pose_data, error, reason, match_vis_data = self._try_no_yolo_processing(
                frame, frame_id, frame_info
            )
            attempts.append(f"No-YOLO: {reason}")
            
            if pos is not None:
                success = True
                result.fallback_level = FallbackLevel.NO_YOLO
                result.match_vis_data = match_vis_data
            elif self.fallback_manager.should_escalate_fallback(nm, error):
                self.fallback_manager.escalate_fallback("No-YOLO processing failed")

        # 🛡️ Level 3: 모든 Anchor 시도 (Multi-Anchor)
        if not success and self.fallback_manager.current_level == FallbackLevel.MULTI_ANCHOR:
            pos, quat, nm, pose_data, error, reason, match_vis_data = self._try_multi_anchor_processing(
                frame, frame_id, frame_info
            )
            attempts.append(f"Multi-anchor: {reason}")
            
            if pos is not None:
                success = True
                result.fallback_level = FallbackLevel.MULTI_ANCHOR
                result.match_vis_data = match_vis_data
            elif self.fallback_manager.should_escalate_fallback(nm, error):
                self.fallback_manager.escalate_fallback("Multi-anchor processing failed")

        # 🛡️ Level 4: Dynamic Anchor 사용
        if not success and self.fallback_manager.current_level == FallbackLevel.DYNAMIC_ANCHOR:
            pos, quat, nm, pose_data, error, reason, match_vis_data = self._try_dynamic_anchor_processing(
                frame, frame_id, frame_info
            )
            attempts.append(f"Dynamic: {reason}")
            
            if pos is not None:
                success = True
                result.fallback_level = FallbackLevel.DYNAMIC_ANCHOR
                result.match_vis_data = match_vis_data

        # 🛡️ Level 5: Emergency (최후의 수단)
        if not success:
            pos, quat, nm, pose_data, error, reason, match_vis_data = self._try_emergency_processing(
                frame, frame_id, frame_info
            )
            attempts.append(f"Emergency: {reason}")
            
            if pos is not None:
                success = True
                result.fallback_level = FallbackLevel.EMERGENCY
                result.match_vis_data = match_vis_data
            else:
                # 정말 최후의 수단: 이전 프레임 복사
                if hasattr(self, 'last_successful_pose'):
                    pos, quat = self.last_successful_pose['position'], self.last_successful_pose['quaternion']
                    pose_data = self._create_emergency_pose_data(frame_id, pos, quat, frame_info)
                    success = True
                    result.fallback_level = FallbackLevel.EMERGENCY
                    result.fallback_reason = "Used previous frame"

        # 결과 설정
        if success:
            result.position = pos
            result.quaternion = quat
            result.num_matches = nm
            result.pose_data = pose_data
            result.fallback_reason = "; ".join(attempts)
            
            # 성공한 포즈 저장 (emergency용)
            self.last_successful_pose = {'position': pos, 'quaternion': quat}
            
            # Dynamic anchor 생성 고려
            if result.fallback_level in [FallbackLevel.NORMAL, FallbackLevel.NO_YOLO]:
                self._consider_creating_dynamic_anchor(frame, pos, quat, nm, error, frame_id)
            
            # 성능 회복 모니터링
            if error and nm:
                self.fallback_manager.recover_fallback(nm, error)

        # Kalman filter update
        if success and self.use_kalman_filter and pos is not None and quat is not None:
            with self.kf_lock:
                if not self.kf_initialized:
                    if error and error < 3.0 and nm >= 8:
                        self.kf.x[0:3] = pos
                        self.kf.x[6:10] = self.kf.normalize_quaternion(quat)
                        self.kf.initialized = True
                        self.kf_initialized = True
                        result.kf_position = pos.copy()
                        result.kf_quaternion = quat.copy()
                        result.measurement_accepted = True
                        if not self.batch_mode:  # Don't spam in batch mode
                            print(f"✅ KF initialized at frame {frame_id} with {result.fallback_level.value}")
                    else:
                        result.measurement_accepted = False
                else:
                    mahal_dist = self.kf.calculate_mahalanobis_distance(pos, quat)
                    result.mahalanobis_distance = mahal_dist
                    
                    if not self.outlier_detector.is_outlier(mahal_dist, self.kf):
                        result.kf_position, result.kf_quaternion = self.kf.update(pos, quat)
                        result.measurement_accepted = True
                    else:
                        result.measurement_accepted = False

        # ── NEW: count missed KF updates and auto-reset after threshold ──
        if result.measurement_accepted:
            self.kf_missed_count = 0
        else:
            self.kf_missed_count += 1
            if self.kf_missed_count > self.kf_missed_threshold:
                print("⚠️ KF too long without measurements, resetting filter")
                self.kf.initialized   = False
                self.kf_initialized   = False
                self.kf_missed_count  = 0
        # ───────────────────────────────────────────────────────────────
        return result

    def _try_normal_processing(self, frame, frame_id, frame_info):
        """🛡️ Level 1: 정상 처리"""
        try:
            # YOLO detection
            bbox = self._yolo_detect(frame)
            if bbox is None:
                return None, None, 0, None, None, "YOLO failed", None
            
            # Viewpoint classification
            viewpoint = self._classify_viewpoint(frame, bbox)
            
            # Pose estimation
            pos, quat, nm, pose_data, error, match_vis_data = self._estimate_pose_robust(
                frame, viewpoint, bbox, frame_id, frame_info
            )
            
            if pos is not None:
                return pos, quat, nm, pose_data, error, f"Success with {viewpoint}", match_vis_data
            else:
                return None, None, nm, None, error, f"Pose estimation failed with {viewpoint}", match_vis_data
                
        except Exception as e:
            return None, None, 0, None, None, f"Exception: {str(e)}", None

    def _try_no_yolo_processing(self, frame, frame_id, frame_info):
        """🛡️ Level 2: YOLO 없이 전체 이미지 사용"""
        try:
            if not self.batch_mode:  # Only print in realtime mode
                print(f"🔄 Frame {frame_id}: Trying no-YOLO processing (full image)")
            
            # 전체 이미지를 crop으로 사용
            bbox = None
            
            # Viewpoint classification on whole image
            viewpoint = self._classify_viewpoint_whole_image(frame)
            
            # Pose estimation with full image
            pos, quat, nm, pose_data, error, match_vis_data = self._estimate_pose_robust(
                frame, viewpoint, bbox, frame_id, frame_info
            )
            
            if pos is not None:
                return pos, quat, nm, pose_data, error, f"Success with full image {viewpoint}", match_vis_data
            else:
                return None, None, nm, None, error, f"Failed with full image {viewpoint}", match_vis_data
                
        except Exception as e:
            return None, None, 0, None, None, f"Exception: {str(e)}", None

    def _try_multi_anchor_processing(self, frame, frame_id, frame_info):
        """🛡️ Level 3: 모든 anchor 순차 시도"""
        try:
            if not self.batch_mode:  # Only print in realtime mode
                print(f"🔄 Frame {frame_id}: Trying multi-anchor processing")
            
            # 전체 이미지 사용
            bbox = None
            
            # 모든 viewpoint 시도
            for viewpoint in self.fallback_manager.anchor_try_order:
                if viewpoint not in self.viewpoint_anchors:
                    continue
                    
                pos, quat, nm, pose_data, error, match_vis_data = self._estimate_pose_robust(
                    frame, viewpoint, bbox, frame_id, frame_info
                )
                
                if pos is not None and nm >= 6:  # 최소 요구사항 만족
                    if not self.batch_mode:
                        print(f"✅ Multi-anchor success with {viewpoint}")
                    return pos, quat, nm, pose_data, error, f"Success with {viewpoint}", match_vis_data
            
            return None, None, 0, None, None, "All anchors failed", None
                
        except Exception as e:
            return None, None, 0, None, None, f"Exception: {str(e)}", None

    def _try_dynamic_anchor_processing(self, frame, frame_id, frame_info):
        """🛡️ Level 4: Dynamic anchor 사용"""
        try:
            if not self.batch_mode:
                print(f"🔄 Frame {frame_id}: Trying dynamic anchor processing")
            
            # 가장 좋은 dynamic anchor 선택
            dynamic_anchor = self.fallback_manager.get_best_dynamic_anchor()
            if dynamic_anchor is None:
                return None, None, 0, None, None, "No dynamic anchor available", None
            
            # Dynamic anchor로 포즈 추정
            pos, quat, nm, pose_data, error, match_vis_data = self._estimate_pose_with_dynamic_anchor(
                frame, dynamic_anchor, frame_id, frame_info
            )
            
            if pos is not None:
                self.fallback_manager.dynamic_anchor_usage_count += 1
                return pos, quat, nm, pose_data, error, f"Success with dynamic anchor from frame {dynamic_anchor.frame_id}", match_vis_data
            else:
                return None, None, nm, None, error, "Dynamic anchor failed", match_vis_data
                
        except Exception as e:
            return None, None, 0, None, None, f"Exception: {str(e)}", None

    def _try_emergency_processing(self, frame, frame_id, frame_info):
        """🛡️ Level 5: Emergency processing"""
        try:
            if not self.batch_mode:
                print(f"🚨 Frame {frame_id}: Emergency processing")
            
            # 매우 관대한 조건으로 시도
            for viewpoint in self.fallback_manager.anchor_try_order:
                if viewpoint not in self.viewpoint_anchors:
                    continue
                    
                # 관대한 매개변수로 포즈 추정
                pos, quat, nm, pose_data, error, match_vis_data = self._estimate_pose_emergency(
                    frame, viewpoint, frame_id, frame_info
                )
                
                if pos is not None:
                    return pos, quat, nm, pose_data, error, f"Emergency success with {viewpoint}", match_vis_data
            
            return None, None, 0, None, None, "Emergency failed", None
                
        except Exception as e:
            return None, None, 0, None, None, f"Exception: {str(e)}", None

    def _consider_creating_dynamic_anchor(self, frame, position, quaternion, num_matches, reprojection_error, frame_id):
        """성공적인 프레임으로부터 dynamic anchor 생성 고려"""
        if num_matches < 15 or reprojection_error > 2.0:
            return  # 품질이 충분하지 않음
        
        # Quality score 계산
        quality_score = num_matches / max(1.0, reprojection_error)
        
        # Dynamic anchor 업데이트 고려
        if self.fallback_manager.should_update_dynamic_anchor(quality_score):
            # Extract features for dynamic anchor
            frame_features = self._extract_features_from_image(frame)
            frame_keypoints = frame_features['keypoints'][0].cpu().numpy()
            
            # Create dynamic anchor
            dynamic_anchor = DynamicAnchor(
                frame_id=frame_id,
                features=frame_features,
                keypoints_2d=frame_keypoints,
                keypoints_3d=np.array([]),  # Will be filled when used
                position=position.copy(),
                quaternion=quaternion.copy(),
                num_matches=num_matches,
                reprojection_error=reprojection_error,
                quality_score=quality_score
            )
            
            self.fallback_manager.add_dynamic_anchor(dynamic_anchor)

    def _estimate_pose_with_dynamic_anchor(self, frame, dynamic_anchor, frame_id, frame_info):
        """Dynamic anchor를 사용한 포즈 추정"""
        # Extract features from current frame
        frame_features = self._extract_features_from_image(frame)
        frame_keypoints = frame_features['keypoints'][0].cpu().numpy()
        
        # Match with dynamic anchor
        with torch.no_grad():
            matches_dict = self.matcher({
                'image0': dynamic_anchor.features,
                'image1': frame_features
            })
        
        matches = rbd(matches_dict)['matches'].cpu().numpy()
        num_matches = len(matches)
        
        if num_matches < 8:  # Dynamic anchor는 더 관대하게
            return None, None, num_matches, None, None, None
        
        # 간단한 transformation 추정 (여기서는 기본적인 구현)
        # 실제로는 더 정교한 알고리즘 필요
        position = dynamic_anchor.position
        quaternion = dynamic_anchor.quaternion
        
        pose_data = self._create_dynamic_anchor_pose_data(
            frame_id, position, quaternion, num_matches, dynamic_anchor.frame_id, frame_info
        )
        
        # 가상의 reprojection error
        reprojection_error = 2.0  
        
        # Create match visualization data for dynamic anchor
        # Create a dummy anchor image since we don't have it stored
        dummy_anchor = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        cv2.putText(dummy_anchor, f'Dynamic Anchor {dynamic_anchor.frame_id}', 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        match_vis_data = MatchVisualizationData(
            anchor_image=dummy_anchor,
            anchor_keypoints=dynamic_anchor.keypoints_2d[:len(matches)],
            frame_keypoints=frame_keypoints[matches[:, 1]],
            matches=matches,
            viewpoint=f"Dynamic_{dynamic_anchor.frame_id}",
            used_for_pose=np.ones(len(matches), dtype=bool)
        )
        
        return position, quaternion, num_matches, pose_data, reprojection_error, match_vis_data

    def _estimate_pose_emergency(self, frame, viewpoint, frame_id, frame_info):
        """Emergency 모드 포즈 추정 (매우 관대한 조건)"""
        try:
            anchor_data = self.viewpoint_anchors[viewpoint]
            frame_features = self._extract_features_from_image(frame)
            frame_keypoints = frame_features['keypoints'][0].cpu().numpy()
            
            # Match with relaxed conditions
            with torch.no_grad():
                matches_dict = self.matcher({
                    'image0': anchor_data['features'],
                    'image1': frame_features
                })
            
            matches = rbd(matches_dict)['matches'].cpu().numpy()
            num_matches = len(matches)
            
            if num_matches < 4:  # 매우 관대한 조건
                return None, None, num_matches, None, None, None
            
            # 관대한 PnP 파라미터
            matched_anchor_indices = anchor_data['matched_sp_indices']
            matched_3d_points_map = {idx: pt for idx, pt in zip(matched_anchor_indices, anchor_data['matched_3d_points'])}
            
            valid_anchor_sp_indices = matches[:, 0]
            mask = np.isin(valid_anchor_sp_indices, matched_anchor_indices)
            
            if np.sum(mask) < 4:
                return None, None, num_matches, None, None, None
            
            points_3d = np.array([matched_3d_points_map[i] for i in valid_anchor_sp_indices[mask]])
            points_2d = frame_keypoints[matches[:, 1][mask]]
            
            K, dist_coeffs = self._get_camera_intrinsics()
            
            # 매우 관대한 PnP
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d.reshape(-1, 1, 3), 
                points_2d.reshape(-1, 1, 2), 
                K, dist_coeffs,
                reprojectionError=10.0,  # 매우 관대
                confidence=0.8,          # 낮은 confidence
                iterationsCount=500, 
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if success and inliers is not None and len(inliers) >= 3:  # 매우 관대한
                R, _ = cv2.Rodrigues(rvec)
                position = tvec.flatten()
                quaternion = self._rotation_matrix_to_quaternion(R)
                
                pose_data = self._create_emergency_pose_data(
                    frame_id, position, quaternion, frame_info, viewpoint
                )
                
                # Create match visualization data
                anchor_keypoints_sp = anchor_data['features']['keypoints'][0].cpu().numpy()
                used_for_pose = np.zeros(len(matches), dtype=bool)
                used_for_pose[mask] = True
                
                match_vis_data = MatchVisualizationData(
                    anchor_image=anchor_data['anchor_image'],
                    anchor_keypoints=anchor_keypoints_sp[matches[:, 0]],
                    frame_keypoints=frame_keypoints[matches[:, 1]],
                    matches=matches,
                    viewpoint=viewpoint,
                    used_for_pose=used_for_pose
                )
                
                return position, quaternion, num_matches, pose_data, 5.0, match_vis_data  # 가상의 error
            
            return None, None, num_matches, None, None, None
            
        except Exception as e:
            return None, None, 0, None, None, None

    # 🔧 Common methods (사용 가능한 기존 메서드들)
    def _estimate_pose_robust(self, full_frame, viewpoint, bbox, frame_id, frame_info=None):
        """Robust pose estimation with proper coordinate handling and match visualization"""
        t_start = time.perf_counter()
        
        if viewpoint not in self.viewpoint_anchors:
            return None, None, 0, None, None, None

        anchor_data = self.viewpoint_anchors[viewpoint]
        anchor_features = anchor_data['features']
        matched_anchor_indices = anchor_data['matched_sp_indices']
        matched_3d_points_map = {idx: pt for idx, pt in zip(matched_anchor_indices, anchor_data['matched_3d_points'])}

        # Extract features from crop for efficiency
        if bbox:
            x1, y1, x2, y2 = bbox
            crop = full_frame[y1:y2, x1:x2]
            crop_offset = np.array([x1, y1])
        else:
            crop = full_frame
            crop_offset = np.array([0, 0])

        if crop.size == 0:
            return None, None, 0, None, None, None

        frame_features = self._extract_features_from_image(crop)
        frame_keypoints_crop = frame_features['keypoints'][0].cpu().numpy()

        # Transform to full image coordinates immediately
        frame_keypoints_full = frame_keypoints_crop + crop_offset

        # Match features
        with torch.no_grad():
            matches_dict = self.matcher({'image0': anchor_features, 'image1': frame_features})
        
        matches = rbd(matches_dict)['matches'].cpu().numpy()
        num_matches = len(matches)

        if num_matches < 6:
            return None, None, num_matches, self._create_failure_report(
                frame_id, 'insufficient_matches', num_matches, viewpoint, bbox, frame_info
            ), None, None

        # Find correspondences
        valid_anchor_sp_indices = matches[:, 0]
        mask = np.isin(valid_anchor_sp_indices, matched_anchor_indices)
        
        if np.sum(mask) < 5:
            return None, None, num_matches, self._create_failure_report(
                frame_id, 'insufficient_valid_correspondences', num_matches, viewpoint, bbox, frame_info, 
                valid_correspondences=np.sum(mask)
            ), None, None

        # Use full image coordinates for pose estimation
        points_3d = np.array([matched_3d_points_map[i] for i in valid_anchor_sp_indices[mask]])
        points_2d_full = frame_keypoints_full[matches[:, 1][mask]]

        # Camera intrinsics
        K, dist_coeffs = self._get_camera_intrinsics()
        
        # PnP estimation
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d.reshape(-1, 1, 3), 
            points_2d_full.reshape(-1, 1, 2), 
            K, dist_coeffs,
            reprojectionError=6.0, 
            confidence=0.9, 
            iterationsCount=1000, 
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success or inliers is None or len(inliers) < 4:
            return None, None, num_matches, self._create_failure_report(
                frame_id, 'pnp_failed', num_matches, viewpoint, bbox, frame_info, 
                num_inliers=len(inliers) if inliers is not None else 0
            ), None, None

        # Enhanced correspondences
        (rvec, tvec), enhanced_3d, enhanced_2d, enhanced_inliers = self.enhance_pose_initialization(
            (rvec, tvec), points_3d[inliers.flatten()], points_2d_full[inliers.flatten()], 
            viewpoint, full_frame
        )
        
        final_points_3d = enhanced_3d if enhanced_inliers is not None else points_3d[inliers.flatten()]
        final_points_2d = enhanced_2d if enhanced_inliers is not None else points_2d_full[inliers.flatten()]
        
        # VVS refinement
        if len(final_points_3d) > 4:
            rvec, tvec = cv2.solvePnPRefineVVS(
                final_points_3d.reshape(-1, 1, 3), 
                final_points_2d.reshape(-1, 1, 2), 
                K, dist_coeffs, rvec, tvec
            )

        # Convert to position and quaternion
        R, _ = cv2.Rodrigues(rvec)
        position = tvec.flatten()
        quaternion = self._rotation_matrix_to_quaternion(R)
        
        # Calculate reprojection error
        projected_points, _ = cv2.projectPoints(
            final_points_3d.reshape(-1, 1, 3), rvec, tvec, K, dist_coeffs
        )
        reprojection_errors = np.linalg.norm(
            final_points_2d.reshape(-1, 1, 2) - projected_points, axis=2
        ).flatten()
        mean_reprojection_error = np.mean(reprojection_errors)
        
        duration = (time.perf_counter() - t_start) * 1000
        self.perf_monitor.add_timing('pose_estimation', duration)

        pose_data = self._create_success_report(
            frame_id, position, quaternion, R, rvec, tvec, num_matches, 
            len(final_points_3d), duration, viewpoint, bbox, frame_info,
            mean_reprojection_error
        )
        
        # Create match visualization data
        anchor_keypoints_sp = anchor_features['keypoints'][0].cpu().numpy()
        used_for_pose = np.zeros(len(matches), dtype=bool)
        used_for_pose[mask] = True
        
        match_vis_data = MatchVisualizationData(
            anchor_image=anchor_data['anchor_image'],
            anchor_keypoints=anchor_keypoints_sp[matches[:, 0]],
            frame_keypoints=frame_keypoints_full[matches[:, 1]],
            matches=matches,
            viewpoint=viewpoint,
            used_for_pose=used_for_pose
        )
        
        return position, quaternion, num_matches, pose_data, mean_reprojection_error, match_vis_data

    def enhance_pose_initialization(self, initial_pose, mpts3D, mkpts1, viewpoint, frame):
        """Enhanced pose initialization like MK3"""
        rvec, tvec = initial_pose
        K, distCoeffs = self._get_camera_intrinsics()
        
        frame_features = self._extract_features_from_image(frame)
        frame_keypoints = frame_features['keypoints'][0].cpu().numpy()
        
        all_3d_points = self.viewpoint_anchors[viewpoint]['all_3d_points']
        
        projected_points, _ = cv2.projectPoints(all_3d_points, rvec, tvec, K, distCoeffs)
        projected_points = projected_points.reshape(-1, 2)
        
        additional_corrs = []
        for i, model_pt in enumerate(all_3d_points):
            if any(np.allclose(model_pt, p, atol=1e-6) for p in mpts3D): 
                continue
            
            proj_pt = projected_points[i]
            distances = np.linalg.norm(frame_keypoints - proj_pt, axis=1)
            min_idx, min_dist = np.argmin(distances), np.min(distances)
            
            if min_dist < 3.0:
                additional_corrs.append((i, min_idx))
        
        if additional_corrs:
            all_3d = np.vstack([mpts3D, all_3d_points[[c[0] for c in additional_corrs]]])
            all_2d = np.vstack([mkpts1, frame_keypoints[[c[1] for c in additional_corrs]]])
            
            success, r, t, inliers = cv2.solvePnPRansac(
                all_3d.reshape(-1, 1, 3), all_2d.reshape(-1, 1, 2), K, distCoeffs,
                rvec=rvec, tvec=tvec, useExtrinsicGuess=True, 
                reprojectionError=4.0, flags=cv2.SOLVEPNP_EPNP
            )
            
            if success and inliers is not None and len(inliers) >= 6:
                return (r, t), all_3d, all_2d, inliers
        
        return (rvec, tvec), mpts3D, mkpts1, None

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
        if cropped.size == 0: 
            return 'NE'
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

    def _quaternion_to_rotation_matrix(self, q):
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])

    def _create_failure_report(self, frame_id, reason, num_matches, viewpoint, bbox, frame_info, **kwargs):
        data = {
            'frame': int(frame_id), 
            'pose_estimation_failed': True, 
            'error_reason': reason, 
            'num_matches': int(num_matches), 
            'viewpoint': str(viewpoint), 
            'bbox': bbox, 
            'whole_image_estimation': bbox is None
        }
        data.update(kwargs)
        if frame_info: 
            data.update(convert_to_json_serializable(frame_info))
        return data

    def _create_success_report(self, frame_id, pos, quat, R, rvec, tvec, num_matches, num_inliers, duration, viewpoint, bbox, frame_info, reprojection_error=None):
        data = {
            'frame': int(frame_id), 
            'pose_estimation_failed': False, 
            'position': pos.tolist(), 
            'quaternion': quat.tolist(), 
            'rotation_matrix': R.tolist(), 
            'translation_vector': tvec.flatten().tolist(), 
            'rotation_vector': rvec.flatten().tolist(), 
            'num_matches': int(num_matches), 
            'num_inliers': int(num_inliers), 
            'processing_time_ms': float(duration), 
            'viewpoint': str(viewpoint), 
            'bbox': bbox, 
            'whole_image_estimation': bbox is None,
            'processing_mode': 'robust_fallback'
        }
        if reprojection_error is not None:
            data['mean_reprojection_error'] = float(reprojection_error)
        if frame_info: 
            data.update(convert_to_json_serializable(frame_info))
        return data

    def _create_dynamic_anchor_pose_data(self, frame_id, pos, quat, num_matches, anchor_frame_id, frame_info):
        return {
            'frame': int(frame_id),
            'pose_estimation_failed': False,
            'position': pos.tolist(),
            'quaternion': quat.tolist(),
            'num_matches': int(num_matches),
            'processing_mode': 'dynamic_anchor',
            'dynamic_anchor_frame': int(anchor_frame_id),
            'whole_image_estimation': True
        }

    def _create_emergency_pose_data(self, frame_id, pos, quat, frame_info, viewpoint=None):
        data = {
            'frame': int(frame_id),
            'pose_estimation_failed': False,
            'position': pos.tolist(),
            'quaternion': quat.tolist(),
            'processing_mode': 'emergency',
            'whole_image_estimation': True
        }
        if viewpoint:
            data['viewpoint'] = viewpoint
        if frame_info:
            data.update(convert_to_json_serializable(frame_info))
        return data
    
    def _create_match_visualization(self, result):
        """🎯 Create side-by-side match visualization"""
        if result.match_vis_data is None:
            # Fallback to normal display if no match data
            return self._create_realtime_display(result)
        
        mvd = result.match_vis_data
        
        # Get image dimensions
        h, w = result.frame.shape[:2]
        
        # Create side-by-side canvas
        canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Left side: anchor image
        anchor_img = mvd.anchor_image.copy()
        canvas[:h, :w] = anchor_img
        
        # Right side: current frame
        frame_img = result.frame.copy()
        canvas[:h, w:w*2] = frame_img
        
        # Draw matches as lines between left and right images
        if len(mvd.matches) > 0:
            for i, (anchor_idx, frame_idx) in enumerate(mvd.matches):
                if i >= len(mvd.anchor_keypoints) or i >= len(mvd.frame_keypoints):
                    continue
                    
                # Get keypoint coordinates
                anchor_pt = mvd.anchor_keypoints[i].astype(int)
                frame_pt = mvd.frame_keypoints[i].astype(int)
                
                # Offset frame point to right side of canvas
                frame_pt_canvas = (frame_pt[0] + w, frame_pt[1])
                anchor_pt_canvas = tuple(anchor_pt)
                
                # Color coding: green for points used in pose estimation, red for others
                if i < len(mvd.used_for_pose) and mvd.used_for_pose[i]:
                    color = (0, 255, 0)  # Green for pose estimation matches
                    thickness = 2
                    radius = 4
                else:
                    color = (0, 0, 255)  # Red for other matches
                    thickness = 1
                    radius = 2
                
                # Draw line connecting the matches
                cv2.line(canvas, anchor_pt_canvas, frame_pt_canvas, color, thickness)
                
                # Draw circles at keypoints
                cv2.circle(canvas, anchor_pt_canvas, radius, color, -1)
                cv2.circle(canvas, frame_pt_canvas, radius, color, -1)
        
        # Add text overlays
        # Left side title
        cv2.putText(canvas, f'Anchor: {mvd.viewpoint}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Right side title
        cv2.putText(canvas, f'Frame: {result.frame_id}', (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Match statistics
        total_matches = len(mvd.matches)
        pose_matches = np.sum(mvd.used_for_pose) if mvd.used_for_pose is not None else 0
        
        cv2.putText(canvas, f'Total Matches: {total_matches}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f'Used for Pose: {pose_matches}', (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Fallback level
        fallback_colors = {
            FallbackLevel.NORMAL: (0, 255, 0),
            FallbackLevel.NO_YOLO: (0, 255, 255),
            FallbackLevel.MULTI_ANCHOR: (255, 165, 0),
            FallbackLevel.DYNAMIC_ANCHOR: (255, 0, 255),
            FallbackLevel.EMERGENCY: (0, 0, 255)
        }
        color = fallback_colors.get(result.fallback_level, (255, 255, 255))
        
        cv2.putText(canvas, f'Fallback: {result.fallback_level.value.upper()}', (w + 10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # FPS and processing info
        fps = self.perf_monitor.get_average_fps()
        cv2.putText(canvas, f'FPS: {fps:.1f}', (w + 10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw pose axes on the right side if available
        display_pos = result.kf_position if result.kf_position is not None else result.position
        display_quat = result.kf_quaternion if result.kf_quaternion is not None else result.quaternion

        if display_pos is not None and display_quat is not None:
            canvas = self._draw_axes_on_canvas(canvas, display_pos, display_quat, w, 0)
        
        # Kalman filter status
        if self.use_kalman_filter and result.kf_position is not None:
            kf_color = (0, 255, 0) if result.measurement_accepted else (0, 0, 255)
            cv2.putText(canvas, 'KF: ON' if result.measurement_accepted else 'KF: SKIP', 
                       (w + 10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, kf_color, 2)
        
        # Legend
        legend_y = h - 80
        cv2.putText(canvas, 'Legend:', (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, 'Green: Used for pose', (10, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(canvas, 'Red: Other matches', (10, legend_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Controls
        cv2.putText(canvas, 'Press Q to quit, R to reset fallback, S to save stats', 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return canvas

    def _draw_axes_on_canvas(self, canvas, position, quaternion, x_offset=0, y_offset=0):
        """Draw pose axes on canvas with offset"""
        try:
            R = self._quaternion_to_rotation_matrix(quaternion)
            rvec, _ = cv2.Rodrigues(R)
            tvec = position.reshape(3, 1)
            K, distCoeffs = self._get_camera_intrinsics()
            
            axis_pts = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, distCoeffs)
            img_pts = img_pts.reshape(-1, 2).astype(int)
            
            # Apply offset
            img_pts[:, 0] += x_offset
            img_pts[:, 1] += y_offset
            
            h, w = canvas.shape[:2]
            total_w = w
            points_in_bounds = all(0 <= pt[0] < total_w and 0 <= pt[1] < h for pt in img_pts)
            
            if points_in_bounds:
                origin = tuple(img_pts[0])
                cv2.line(canvas, origin, tuple(img_pts[1]), (0,0,255), 3)  # X - Red
                cv2.line(canvas, origin, tuple(img_pts[2]), (0,255,0), 3)  # Y - Green
                cv2.line(canvas, origin, tuple(img_pts[3]), (255,0,0), 3)  # Z - Blue
                cv2.circle(canvas, origin, 5, (255, 255, 255), -1)
        except Exception:
            pass
        return canvas
    
    def _display_result(self, result):
        """결과 표시 - 배치 모드용"""
        if getattr(self.args, 'vis_match', False):
            vis_frame = self._create_match_visualization(result)
        else:
            vis_frame = self._create_batch_display(result)
        
        # Display the result using queue for batch mode
        try:
            self.display_queue.put_nowait(vis_frame)
        except queue.Full:
            pass # Skip if queue is full (display thread is busy)

    def _create_batch_display(self, result):
        """배치 모드용 기본 디스플레이"""
        vis_frame = result.frame.copy()
        
        # Fallback level에 따른 색상
        fallback_colors = {
            FallbackLevel.NORMAL: (0, 255, 0),        # Green
            FallbackLevel.NO_YOLO: (0, 255, 255),     # Yellow
            FallbackLevel.MULTI_ANCHOR: (255, 165, 0), # Orange
            FallbackLevel.DYNAMIC_ANCHOR: (255, 0, 255), # Magenta
            FallbackLevel.EMERGENCY: (0, 0, 255)      # Red
        }
        
        color = fallback_colors.get(result.fallback_level, (255, 255, 255))
        
        cv2.putText(vis_frame, f'Frame: {result.frame_id}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis_frame, f'Fallback: {result.fallback_level.value.upper()}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show bounding box
        if result.bbox:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

        # Show pose axes
        display_pos = result.kf_position if result.kf_position is not None else result.position
        display_quat = result.kf_quaternion if result.kf_quaternion is not None else result.quaternion

        if display_pos is not None and display_quat is not None:
            vis_frame = self._draw_axes_robust(vis_frame, display_pos, display_quat)

        # Show match info
        if result.num_matches > 0:
            match_color = (0, 255, 0) if result.num_matches > 15 else (0, 165, 255) if result.num_matches > 8 else (0, 0, 255)
            cv2.putText(vis_frame, f'Matches: {result.num_matches}', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)

        # Show Kalman filter status
        if self.use_kalman_filter and result.kf_position is not None:
            kf_color = (0, 255, 0) if result.measurement_accepted else (0, 0, 255)
            cv2.putText(vis_frame, 'KF: ON' if result.measurement_accepted else 'KF: SKIP', 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, kf_color, 2)

        return vis_frame

    def _draw_axes_robust(self, frame, position, quaternion):
        """Draw axes in correct full image coordinates"""
        try:
            R = self._quaternion_to_rotation_matrix(quaternion)
            rvec, _ = cv2.Rodrigues(R)
            tvec = position.reshape(3, 1)
            K, distCoeffs = self._get_camera_intrinsics()
            
            axis_pts = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, distCoeffs)
            img_pts = img_pts.reshape(-1, 2).astype(int)
            
            h, w = frame.shape[:2]
            points_in_bounds = all(0 <= pt[0] < w and 0 <= pt[1] < h for pt in img_pts)
            
            if points_in_bounds:
                origin = tuple(img_pts[0])
                cv2.line(frame, origin, tuple(img_pts[1]), (0,0,255), 3)  # X - Red
                cv2.line(frame, origin, tuple(img_pts[2]), (0,255,0), 3)  # Y - Green
                cv2.line(frame, origin, tuple(img_pts[3]), (255,0,0), 3)  # Z - Blue
                cv2.circle(frame, origin, 5, (255, 255, 255), -1)
        except Exception:
            pass
        return frame

    def _create_realtime_display(self, result):
        """실시간 디스플레이용 프레임 생성"""
        vis_frame = result.frame.copy()
        
        # Fallback level에 따른 색상
        fallback_colors = {
            FallbackLevel.NORMAL: (0, 255, 0),        # Green
            FallbackLevel.NO_YOLO: (0, 255, 255),     # Yellow
            FallbackLevel.MULTI_ANCHOR: (255, 165, 0), # Orange
            FallbackLevel.DYNAMIC_ANCHOR: (255, 0, 255), # Magenta
            FallbackLevel.EMERGENCY: (0, 0, 255)      # Red
        }
        
        color = fallback_colors.get(result.fallback_level, (255, 255, 255))
        
        # Add frame info and fallback level
        cv2.putText(vis_frame, f'Frame: {result.frame_id}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis_frame, f'Fallback: {result.fallback_level.value.upper()}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Add FPS for realtime mode
        fps = self.perf_monitor.get_average_fps()
        cv2.putText(vis_frame, f'FPS: {fps:.1f}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show bounding box
        if result.bbox:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

        # Show pose axes
        display_pos = result.kf_position if result.kf_position is not None else result.position
        display_quat = result.kf_quaternion if result.kf_quaternion is not None else result.quaternion

        if display_pos is not None and display_quat is not None:
            vis_frame = self._draw_axes_robust(vis_frame, display_pos, display_quat)

        # Show match info
        if result.num_matches > 0:
            match_color = (0, 255, 0) if result.num_matches > 15 else (0, 165, 255) if result.num_matches > 8 else (0, 0, 255)
            cv2.putText(vis_frame, f'Matches: {result.num_matches}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)

        # Show Kalman filter status
        if self.use_kalman_filter and result.kf_position is not None:
            kf_color = (0, 255, 0) if result.measurement_accepted else (0, 0, 255)
            cv2.putText(vis_frame, 'KF: ON' if result.measurement_accepted else 'KF: SKIP', 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, kf_color, 2)

        # Show processing time
        if result.processing_time > 0:
            cv2.putText(vis_frame, f'Process: {result.processing_time:.1f}ms', (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add instructions
        instructions = 'Press Q to quit, R to reset fallback, S to save stats'
        if getattr(self.args, 'vis_match', False):
            instructions += ', V to toggle match view'
        cv2.putText(vis_frame, instructions, 
                   (10, vis_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis_frame

    def start(self):
        """메인 실행 함수"""
        self.running = True
        
        if self.sequential_processing:
            print("🛡️ Starting robust fallback processing (batch mode)...")
            if not getattr(self.args, 'no_display', False):
                window_name = 'VAPE MK46 Robust Fallback - Batch Mode'
                if getattr(self.args, 'vis_match', False):
                    window_name += ' - Match Visualization'
                display_thread = threading.Thread(target=self._sequential_display_thread, daemon=True)
                display_thread.start()
                self.threads = [display_thread]
            
            # Main thread does robust processing
            self.process_all_images_robustly()
            
        else:
            print("🎥 Starting realtime webcam processing with fallback support...")
            # No separate display thread for realtime - display directly in main loop
            self.process_realtime_webcam()

    def _sequential_display_thread(self):
        """순차 처리 모드용 디스플레이 스레드"""
        if getattr(self.args, 'no_display', False):
            return
        
        window_name = 'VAPE MK46 Robust Fallback - Batch Mode'
        if getattr(self.args, 'vis_match', False):
            window_name += ' - Match Visualization'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 2560, 720)  # Wider for side-by-side
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
            
        while self.running:
            try:
                vis_frame = self.display_queue.get(timeout=0.1)
                cv2.imshow(window_name, vis_frame)
            except queue.Empty:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
        cv2.destroyAllWindows()

    def _save_realtime_statistics(self):
        """실시간 모드에서 현재 통계 저장"""
        stats = {
            'current_fallback_level': self.fallback_manager.current_level.value,
            'fallback_stats': self.perf_monitor.get_fallback_stats(),
            'average_fps': self.perf_monitor.get_average_fps(),
            'kalman_filter_initialized': self.kf_initialized,
            'total_poses_saved': len(self.all_poses),
            'match_visualization_enabled': getattr(self.args, 'vis_match', False)
        }
        
        stats_filename = create_unique_filename(
            self.args.output_dir, 'realtime_stats.json'
        )
        
        with open(stats_filename, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"💾 Realtime statistics saved to {stats_filename}")

    def stop(self):
        print("🛑 Stopping...")
        self.running = False
        
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2.0)
        
        if hasattr(self, 'cap') and self.cap: 
            self.cap.release()
        cv2.destroyAllWindows()

        # Save results
        if self.all_poses:
            mode_suffix = "batch" if self.batch_mode else "realtime"
            if getattr(self.args, 'vis_match', False):
                mode_suffix += "_with_match_vis"
            output_filename = create_unique_filename(
                self.args.output_dir, f'pose_results_robust_fallback_{mode_suffix}.json'
            )
            print(f"💾 Saving {len(self.all_poses)} pose records to {output_filename}")
            with open(output_filename, 'w') as f:
                json.dump(self.all_poses, f, indent=4)
            
            # Print final fallback statistics
            fallback_stats = self.perf_monitor.get_fallback_stats()
            print(f"🛡️ Final fallback usage:")
            total = sum(fallback_stats.values())
            for level, count in fallback_stats.items():
                percentage = count / total * 100 if total > 0 else 0
                print(f"   {level}: {count} ({percentage:.1f}%)")
                
        print("✅ Shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description="VAPE MK46 Robust Fallback - 다단계 백업 전략 (Batch & Realtime) with Match Visualization")
    parser.add_argument('--image_dir', type=str, help='Directory of images for batch processing.')
    parser.add_argument('--csv_file', type=str, help='CSV file with image timestamps for batch mode.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save JSON results.')
    parser.add_argument('--no_display', action='store_true', help='Run in headless mode without display.')
    parser.add_argument('--no_kalman_filter', action='store_false', dest='use_kalman_filter', help='Disable the Kalman filter.')
    parser.add_argument('--realtime', action='store_true', help='Force realtime mode even if image_dir is provided.')
    parser.add_argument('--vis_match', action='store_true', help='Enable match visualization showing anchor and frame side-by-side with match lines.')
    
    args = parser.parse_args()

    # Override batch mode if realtime is explicitly requested
    if args.realtime:
        args.image_dir = None

    # Print match visualization status
    if args.vis_match:
        print("🎯 Match visualization enabled - showing anchor-frame correspondences")

    estimator = RobustFallbackPoseEstimator(args)
    
    def signal_handler(sig, frame):
        print('\nSIGINT received, shutting down gracefully.')
        estimator.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        estimator.start()
    except KeyboardInterrupt:
        print('\nKeyboard interrupt received.')
    finally:
        estimator.stop()

if __name__ == '__main__':
    main()
