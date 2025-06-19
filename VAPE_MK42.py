"""
VAPE MK42: Simplified Robust Real-time Pose Estimator
- Uses same models as VAPE_MK1 for compatibility
- Single-threaded design for stability
- Enhanced error recovery and monitoring
- Adaptive processing without complex threading
"""

import cv2
import numpy as np
import torch
import time
import argparse
import warnings
import signal
import sys
from collections import defaultdict, deque
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 Starting VAPE MK42 - Simplified Robust Edition...")

# Import required libraries
try:
    from ultralytics import YOLO
    import timm
    from torchvision import transforms
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    from PIL import Image
    from scipy.spatial.distance import cdist
    print("✅ All libraries loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Enhanced State Machine (simplified)
class TrackingState(Enum):
    INITIALIZING = "initializing"
    DETECTING = "detecting" 
    TRACKING = "tracking"
    LOST = "lost"
    RECLASSIFYING = "reclassifying"

@dataclass
class TrackingContext:
    """Context information for current tracking state"""
    frame_id: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    viewpoint: str = "NE"
    num_matches: int = 0
    consecutive_failures: int = 0
    last_detection_frame: int = 0
    last_pose_estimation_frame: int = 0  # Add this for pose estimation frequency control

class PerformanceMonitor:
    """Enhanced performance monitoring"""
    def __init__(self):
        self.timings = defaultdict(lambda: deque(maxlen=30))
        self.fps_history = deque(maxlen=30)
        
    def add_timing(self, name: str, duration: float):
        self.timings[name].append(duration)
        
    def get_average(self, name: str) -> float:
        if name in self.timings and self.timings[name]:
            return np.mean(self.timings[name])
        return 0.0
        
    def print_stats(self, frame_idx):
        if frame_idx % 60 == 0 and frame_idx > 0:
            print(f"\n=== FRAME {frame_idx} PERFORMANCE ===")
            for name, times in self.timings.items():
                if times:
                    # Convert deque to list for slicing
                    times_list = list(times)
                    recent = times_list[-20:] if len(times_list) >= 20 else times_list
                    avg = sum(recent) / len(recent)
                    emoji = "🔴" if avg > 50 else "🟡" if avg > 25 else "🟢"
                    print(f"{emoji} {name:20} | {avg:6.1f}ms")

class TrackingQualityMonitor:
    """Monitor tracking quality for adaptive decisions"""
    def __init__(self, 
                 low_match_threshold=8, 
                 consecutive_low_frames=3,
                 tracking_confidence_threshold=0.6):
        self.low_match_threshold = low_match_threshold
        self.consecutive_low_frames = consecutive_low_frames
        self.tracking_confidence_threshold = tracking_confidence_threshold
        
        self.match_history = deque(maxlen=10)
        self.low_match_count = 0
        self.tracking_confidence = 1.0
        
    def update_matches(self, num_matches):
        self.match_history.append(num_matches)
        
        if num_matches < self.low_match_threshold:
            self.low_match_count += 1
        else:
            self.low_match_count = 0
            
    def update_tracking_confidence(self, confidence):
        self.tracking_confidence = confidence
        
    def should_reclassify_viewpoint(self):
        return self.low_match_count >= self.consecutive_low_frames
        
    def should_redetect(self):
        return self.tracking_confidence < self.tracking_confidence_threshold
        
    def get_average_matches(self):
        if len(self.match_history) > 0:
            return np.mean(list(self.match_history))
        return 0
        
    def get_average_matches(self):
        if len(self.match_history) > 0:
            return np.mean(list(self.match_history))
        return 0


# Loosely Coupled Kalman Filter (same as MK1)
class LooselyCoupledKalmanFilter:
    def __init__(self, dt=1/30.0):
        self.dt = dt
        self.initialized = False
        
        # State: [px, py, pz, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]
        self.n_states = 13
        self.x = np.zeros(self.n_states)
        self.x[9] = 1.0  # quaternion w=1, x=y=z=0
        
        # Covariance matrix
        self.P = np.eye(self.n_states) * 0.1
        
        # Process noise (tuned for stability)
        self.Q = np.eye(self.n_states) * 1e-3 #1e-3
        
        # Measurement noise for [px, py, pz, qx, qy, qz, qw]
        self.R = np.eye(7) * 1e-4 #1e-6 #1e-4
    
    def normalize_quaternion(self, q):
        """Normalize quaternion to unit length"""
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            return q / norm
        else:
            return np.array([0, 0, 0, 1])  # Default quaternion
    
    def predict(self):
        if not self.initialized:
            return None
            
        # Extract state components
        px, py, pz = self.x[0:3]
        vx, vy, vz = self.x[3:6]
        qx, qy, qz, qw = self.x[6:10]
        wx, wy, wz = self.x[10:13]
        
        dt = self.dt
        
        # Simple constant velocity prediction
        px_new = px + vx * dt
        py_new = py + vy * dt
        pz_new = pz + vz * dt
        
        # Velocity remains constant
        vx_new, vy_new, vz_new = vx, vy, vz
        
        # Simple quaternion integration (small angle approximation)
        q = np.array([qx, qy, qz, qw])
        w = np.array([wx, wy, wz])
        
        # Small angle quaternion update: dq = 0.5 * dt * Omega(w) * q
        omega_mat = np.array([
            [0,   -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy,  -wz,  0,   wx],
            [wz,   wy, -wx,  0 ]
        ])
        
        dq = 0.5 * dt * omega_mat @ q
        q_new = q + dq
        q_new = self.normalize_quaternion(q_new)
        
        # Angular velocity remains constant
        wx_new, wy_new, wz_new = wx, wy, wz
        
        # Update state
        self.x = np.array([
            px_new, py_new, pz_new,
            vx_new, vy_new, vz_new,
            q_new[0], q_new[1], q_new[2], q_new[3],
            wx_new, wy_new, wz_new
        ])
        
        # Build Jacobian F (simplified)
        F = np.eye(self.n_states)
        F[0, 3] = dt  # dx/dvx
        F[1, 4] = dt  # dy/dvy
        F[2, 5] = dt  # dz/dvz
        
        # Update covariance
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x[0:3], self.x[6:10]  # position, quaternion
    
    def update(self, position, quaternion):
        """Loosely coupled update with pose measurement"""
        measurement = np.concatenate([position, quaternion])
        
        if not self.initialized:
            # Initialize with first measurement
            self.x[0:3] = position
            self.x[6:10] = self.normalize_quaternion(quaternion)
            self.initialized = True
            return self.x[0:3], self.x[6:10]
        
        # Measurement model: observe position and quaternion directly
        # h(x) = [px, py, pz, qx, qy, qz, qw]
        predicted_measurement = np.array([
            self.x[0], self.x[1], self.x[2],  # position
            self.x[6], self.x[7], self.x[8], self.x[9]  # quaternion
        ])
        
        # Innovation
        innovation = measurement - predicted_measurement
        
        # Handle quaternion wraparound (ensure shortest path)
        q_meas = measurement[3:7]
        q_pred = predicted_measurement[3:7]
        if np.dot(q_meas, q_pred) < 0:
            q_meas = -q_meas
            innovation[3:7] = q_meas - q_pred
        
        # Measurement Jacobian H
        H = np.zeros((7, self.n_states))
        # Position measurements
        H[0, 0] = 1.0  # px
        H[1, 1] = 1.0  # py
        H[2, 2] = 1.0  # pz
        # Quaternion measurements
        H[3, 6] = 1.0  # qx
        H[4, 7] = 1.0  # qy
        H[5, 8] = 1.0  # qz
        H[6, 9] = 1.0  # qw
        
        # Kalman update
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x += K @ innovation
        
        # Normalize quaternion
        self.x[6:10] = self.normalize_quaternion(self.x[6:10])
        
        # Covariance update (Joseph form for numerical stability)
        I = np.eye(self.n_states)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        
        return self.x[0:3], self.x[6:10]


class SimplifiedPoseEstimator:
    """Simplified but robust pose estimator"""
    def __init__(self, args):
        print("🔧 Initializing Simplified VAPE MK42...")
        self.args = args
        self.frame_count = 0
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
        self.quality_monitor = TrackingQualityMonitor()
        
        # System state
        self.state = TrackingState.INITIALIZING
        self.context = TrackingContext()
        self.tracker = None
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")
        
        # Initialize components
        self._init_models()
        self._init_camera()
        self._init_anchor_data()
        
        # Initialize Kalman filter
        self.kf = LooselyCoupledKalmanFilter(dt=1/30.0)
        
        print("✅ Simplified VAPE MK42 initialized!")
        
    def _should_estimate_pose(self, state: TrackingState, context: TrackingContext) -> bool:
        """Determine if we should run pose estimation based on state and performance"""
        pose_intervals = {
            TrackingState.INITIALIZING: 1,      # Every frame
            TrackingState.DETECTING: 1,         # Every frame
            TrackingState.TRACKING: 3,          # Every 3 frames (10 FPS)
            TrackingState.RECLASSIFYING: 1,     # Every frame
            TrackingState.LOST: 1               # Every frame
        }
        
        # If quality is poor, increase frequency
        if context.num_matches < 10 or context.confidence < 0.7:
            pose_intervals[TrackingState.TRACKING] = 2  # Every 2 frames
        
        # If quality is very good, decrease frequency
        if context.num_matches > 20 and context.confidence > 0.9:
            pose_intervals[TrackingState.TRACKING] = 5  # Every 5 frames
        
        interval = pose_intervals.get(state, 3)
        frames_since_last = context.frame_id - context.last_pose_estimation_frame
        
        return frames_since_last >= interval
        
    def _init_models(self):
        """Initialize AI models with MK1 configuration"""
        try:
            # YOLO - same as MK1
            print("  📦 Loading YOLO...")
            self.yolo_model = YOLO("yolov8s.pt")
            if self.device == 'cuda':
                self.yolo_model.to('cuda')
            print("  ✅ YOLO loaded")
                
            # Viewpoint classifier - same as MK1
            print("  📦 Loading viewpoint classifier...")
            self.vp_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=4)
            
            try:
                self.vp_model.load_state_dict(torch.load('mobilevit_viewpoint_twostage_final_2.pth', 
                                                        map_location=self.device))
                print("  ✅ Viewpoint model loaded")
            except FileNotFoundError:
                print("  ⚠️ Viewpoint model file not found, using random weights")
            
            self.vp_model.eval().to(self.device)
            
            # Transform for viewpoint classifier - same as MK1
            self.vp_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
            
            # SuperPoint with same settings as MK1
            print("  📦 Loading SuperPoint & LightGlue...")
            self.extractor = SuperPoint(
                max_num_keypoints=256  # Same as MK1
            ).eval().to(self.device)
            
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)
            print("  ✅ SuperPoint & LightGlue loaded")
            
            # Class names for viewpoint
            self.class_names = ['NE', 'NW', 'SE', 'SW']
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
            
    def _init_camera(self):
        """Initialize camera with error handling"""
        try:
            self.cap = cv2.VideoCapture(0)  # Use default camera
            
            if not self.cap.isOpened():
                print("❌ Cannot open camera 0, trying camera 1...")
                self.cap = cv2.VideoCapture(1)
                
            if not self.cap.isOpened():
                print("❌ Cannot open any camera, using dummy mode")
                self.cap = None
                return
            
            # Set camera properties - same as MK1
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test capture
            ret, test_frame = self.cap.read()
            if not ret:
                print("❌ Cannot read from camera, using dummy mode")
                self.cap.release()
                self.cap = None
                return
                
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✅ Camera initialized: {actual_width}x{actual_height}")
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            self.cap = None
            
    def _init_anchor_data(self):
        """Initialize anchor data for viewpoints - same as MK1"""
        # Load anchor image
        anchor_path = 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png'
        
        try:
            self.anchor_image = cv2.imread(anchor_path)
            if self.anchor_image is None:
                raise FileNotFoundError(f"Could not load anchor image: {anchor_path}")
            
            self.anchor_image = cv2.resize(self.anchor_image, (1280, 720))
            print(f"✅ Anchor image loaded: {anchor_path}")
        except Exception as e:
            print(f"❌ Failed to load anchor image: {e}")
            print("Using dummy anchor image for testing...")
            self.anchor_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Same 2D keypoints as MK1
        self.anchor_2d = np.array([
            [511, 293], [591, 284], [587, 330], [413, 249], [602, 348],
            [715, 384], [598, 298], [656, 171], [805, 213], [703, 392],
            [523, 286], [519, 327], [387, 289], [727, 126], [425, 243],
            [636, 358], [745, 202], [595, 388], [436, 260], [539, 313],
            [795, 220], [351, 291], [665, 165], [611, 353], [650, 377],
            [516, 389], [727, 143], [496, 378], [575, 312], [617, 368],
            [430, 312], [480, 281], [834, 225], [469, 339], [705, 223],
            [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]
        ], dtype=np.float32)
        
        # Same 3D keypoints as MK1
        self.anchor_3d = np.array([
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
        
        # Extract anchor features once
        self._extract_anchor_features()
        
        # Create anchor data for 4 viewpoints (using same data for now)
        self.viewpoint_anchors = {
            'NE': {'2d': self.anchor_2d, '3d': self.anchor_3d, 'features': self.anchor_features},
            'NW': {'2d': self.anchor_2d, '3d': self.anchor_3d, 'features': self.anchor_features},
            'SE': {'2d': self.anchor_2d, '3d': self.anchor_3d, 'features': self.anchor_features},
            'SW': {'2d': self.anchor_2d, '3d': self.anchor_3d, 'features': self.anchor_features}
        }
        
        print("✅ Anchor data initialized for 4 viewpoints")
    
    def _extract_anchor_features(self):
        """Extract SuperPoint features from anchor image once"""
        anchor_rgb = cv2.cvtColor(self.anchor_image, cv2.COLOR_BGR2RGB)
        anchor_tensor = torch.from_numpy(anchor_rgb).float() / 255.0
        anchor_tensor = anchor_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.anchor_features = self.extractor.extract(anchor_tensor)
            
        print("✅ Anchor features extracted")
    
    def _validate_bbox(self, bbox, frame_shape):
        """Validate and fix bounding box coordinates"""
        if bbox is None:
            return None
            
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        
        # Fix coordinates
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Check if bbox is too small
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None
            
        return (int(x1), int(y1), int(x2), int(y2))
    
    def _get_frame(self):
        """Get frame from camera or generate dummy frame"""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
        
        # Generate dummy frame for testing
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Add some fake object in the center for testing
        center_x, center_y = 640, 360
        size = 100
        cv2.rectangle(frame, 
                     (center_x - size, center_y - size), 
                     (center_x + size, center_y + size), 
                     (0, 255, 0), -1)
        
        return frame
        """Get frame from camera or generate dummy frame"""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
        
        # Generate dummy frame for testing
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Add some fake object in the center for testing
        center_x, center_y = 640, 360
        size = 100
        cv2.rectangle(frame, 
                     (center_x - size, center_y - size), 
                     (center_x + size, center_y + size), 
                     (0, 255, 0), -1)
        
        return frame
    
    def _yolo_detect(self, frame):
        """Run YOLO detection - same as MK1"""
        self.perf_monitor.add_timing('yolo_detection', 0)  # Start timing
        t_start = time.time()
        
        # Resize for YOLO (same as MK1)
        yolo_size = (640, 640)
        yolo_frame = cv2.resize(frame, yolo_size)
        
        # Run YOLO
        results = self.yolo_model(yolo_frame[..., ::-1], verbose=False, conf=0.5)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        duration = (time.time() - t_start) * 1000
        self.perf_monitor.add_timing('yolo_detection', duration)
        
        if len(boxes) == 0:
            return None
            
        # Scale bounding box back to original size
        bbox = boxes[0]  # Take first detection
        scale_x = frame.shape[1] / yolo_size[0]
        scale_y = frame.shape[0] / yolo_size[1]
        
        # Scale from center
        center_x = (bbox[0] + bbox[2]) / 2 * scale_x
        center_y = (bbox[1] + bbox[3]) / 2 * scale_y
        width = (bbox[2] - bbox[0]) * scale_x
        height = (bbox[3] - bbox[1]) * scale_y
        
        x1 = int(center_x - width/2)
        y1 = int(center_y - height/2)
        x2 = int(center_x + width/2)
        y2 = int(center_y + height/2)
        
        # Ensure bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        return (x1, y1, x2, y2)
    
    def _init_tracker(self, frame, bbox):
        """Initialize OpenCV tracker - same as MK1"""
        t_start = time.time()

        # pick the right constructor depending on your OpenCV build
        try:
            tracker_ctor = cv2.TrackerCSRT_create
        except AttributeError:
            # OpenCV ≥4.5+ in some pip installs puts trackers in the legacy submodule
            tracker_ctor = cv2.legacy.TrackerCSRT_create

        self.tracker = tracker_ctor()
        
        # Convert bbox to OpenCV format (x, y, w, h)
        x1, y1, x2, y2 = bbox
        w, h = x2-x1, y2-y1
        success = self.tracker.init(frame, (x1, y1, w, h))
        
        duration = (time.time() - t_start) * 1000
        self.perf_monitor.add_timing('tracker_init', duration)
        return success
    
    def _track_object(self, frame):
        """Track object using OpenCV tracker - same as MK1"""
        if self.tracker is None:
            return None, 0.0
            
        t_start = time.time()
        
        success, opencv_bbox = self.tracker.update(frame)
        
        duration = (time.time() - t_start) * 1000
        self.perf_monitor.add_timing('tracking', duration)
        
        if not success:
            return None, 0.0
            
        # Convert back to our format (x1, y1, x2, y2)
        x, y, w, h = opencv_bbox
        bbox = (int(x), int(y), int(x+w), int(y+h))
        
        # Estimate confidence based on bbox size and position consistency
        confidence = self._estimate_tracking_confidence(bbox)
        
        return bbox, confidence
    
    def _estimate_tracking_confidence(self, bbox):
        """Estimate tracking confidence - same as MK1"""
        if self.context.bbox is None:
            return 1.0
            
        # Compare with previous bbox
        x1, y1, x2, y2 = bbox
        px1, py1, px2, py2 = self.context.bbox
        
        # Size change ratio
        current_area = (x2-x1) * (y2-y1)
        prev_area = (px2-px1) * (py2-py1)
        if prev_area > 0:
            size_ratio = min(current_area, prev_area) / max(current_area, prev_area)
        else:
            size_ratio = 0.5
        
        # Position change
        center_x, center_y = (x1+x2)/2, (y1+y2)/2
        prev_center_x, prev_center_y = (px1+px2)/2, (py1+py2)/2
        distance = np.sqrt((center_x-prev_center_x)**2 + (center_y-prev_center_y)**2)
        
        # Confidence based on consistency
        confidence = size_ratio * max(0, 1 - distance/100)  # Penalize large movements
        
        return confidence
    
    def _classify_viewpoint(self, frame, bbox):
        """Classify viewpoint - same as MK1"""
        t_start = time.time()
        
        x1, y1, x2, y2 = bbox
        cropped = frame[y1:y2, x1:x2]
        
        if cropped.size == 0:
            duration = (time.time() - t_start) * 1000
            self.perf_monitor.add_timing('viewpoint_classification', duration)
            return 'NE'
        
        # Resize for classification (same as MK1)
        crop_resized = cv2.resize(cropped, (128, 128))
        
        # Convert to PIL and apply transforms
        img_pil = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
        input_tensor = self.vp_transform(img_pil).unsqueeze(0).to(self.device)
        
        # Predict viewpoint
        with torch.no_grad():
            logits = self.vp_model(input_tensor)
            pred = torch.argmax(logits, dim=1).item()
            viewpoint = self.class_names[pred]
        
        duration = (time.time() - t_start) * 1000
        self.perf_monitor.add_timing('viewpoint_classification', duration)
        return viewpoint
    
    def _estimate_pose(self, cropped_frame, viewpoint):
        """Estimate pose - same as MK1"""
        t_start = time.time()
        
        # Extract features from cropped frame
        frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            frame_features = self.extractor.extract(frame_tensor)
            
        # Match with anchor
        anchor_data = self.viewpoint_anchors[viewpoint]
        
        with torch.no_grad():
            matches_dict = self.matcher({
                'image0': anchor_data['features'],
                'image1': frame_features
            })
        
        # Process matches - same as MK1
        feats0, feats1, matches01 = [rbd(x) for x in [anchor_data['features'], frame_features, matches_dict]]
        kpts0 = feats0["keypoints"].detach().cpu().numpy()
        kpts1 = feats1["keypoints"].detach().cpu().numpy()
        matches = matches01["matches"].detach().cpu().numpy()
        
        num_matches = len(matches)
        self.quality_monitor.update_matches(num_matches)
        
        if num_matches < 6:  # Need minimum matches
            duration = (time.time() - t_start) * 1000
            self.perf_monitor.add_timing('pose_estimation', duration)
            return None, None, num_matches
            
        # Get matched points
        mkpts0 = kpts0[matches[:, 0]]
        mkpts1 = kpts1[matches[:, 1]]
        
        # Map to 3D points
        anchor_2d = anchor_data['2d']
        anchor_3d = anchor_data['3d']
        
        # Find closest 2D points in anchor
        distances = cdist(mkpts0, anchor_2d)
        closest_indices = np.argmin(distances, axis=1)
        valid_mask = np.min(distances, axis=1) < 5.0  # threshold
        
        if np.sum(valid_mask) < 6:
            duration = (time.time() - t_start) * 1000
            self.perf_monitor.add_timing('pose_estimation', duration)
            return None, None, num_matches
            
        # Get valid correspondences
        points_3d = anchor_3d[closest_indices[valid_mask]]
        points_2d = mkpts1[valid_mask]

        if self.context.bbox is not None:
            x1, y1, x2, y2 = self.context.bbox
            crop_offset = np.array([x1, y1])
            points_2d = points_2d + crop_offset  # frame 기준으로 맞춤

        # Solve PnP - same camera parameters as MK1
        K, dist_coeffs = self._get_camera_intrinsics()
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points_3d.reshape(-1, 1, 3),
            imagePoints=points_2d.reshape(-1, 1, 2),
            cameraMatrix=K,
            distCoeffs=dist_coeffs,
            reprojectionError=3.0,
            confidence=0.99,
            iterationsCount=1000,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if not success or len(inliers) < 4:
            duration = (time.time() - t_start) * 1000
            self.perf_monitor.add_timing('pose_estimation', duration)
            return None, None, num_matches
            
        # Convert to position and quaternion - same as MK1
        R, _ = cv2.Rodrigues(rvec)
        position = tvec.flatten()
        
        # Convert rotation matrix to quaternion [x, y, z, w]
        def rotation_matrix_to_quaternion(R):
            trace = np.trace(R)
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2
                w = 0.25 * s
                x = (R[2, 1] - R[1, 2]) / s
                y = (R[0, 2] - R[2, 0]) / s
                z = (R[1, 0] - R[0, 1]) / s
            else:
                if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                    s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                    w = (R[2, 1] - R[1, 2]) / s
                    x = 0.25 * s
                    y = (R[0, 1] + R[1, 0]) / s
                    z = (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2]:
                    s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                    w = (R[0, 2] - R[2, 0]) / s
                    x = (R[0, 1] + R[1, 0]) / s
                    y = 0.25 * s
                    z = (R[1, 2] + R[2, 1]) / s
                else:
                    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                    w = (R[1, 0] - R[0, 1]) / s
                    x = (R[0, 2] + R[2, 0]) / s
                    y = (R[1, 2] + R[2, 1]) / s
                    z = 0.25 * s
            return np.array([x, y, z, w])
        
        quaternion = rotation_matrix_to_quaternion(R)
        
        duration = (time.time() - t_start) * 1000
        self.perf_monitor.add_timing('pose_estimation', duration)
        
        return position, quaternion, num_matches
    
    def _get_camera_intrinsics(self):
        """Get camera intrinsic parameters - same as MK1"""
        # Use same calibrated parameters as MK1
        fx = 1460.10150
        fy = 1456.48915
        cx = 604.85462
        cy = 328.64800
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.array([
            3.56447550e-01, -1.09206851e+01, 1.40564820e-03, 
            -1.10856449e-02, 1.20471120e+02
        ], dtype=np.float32)
        
        return K, None  # Disable distortion for simplicity
    
    def _draw_axes(self, frame, position, quaternion, bbox=None):
        """
        Draw coordinate axes on frame - simplified version based on working code
        
        Args:
            frame: Full frame image
            position: 3D position from pose estimation 
            quaternion: Rotation quaternion from pose estimation
            bbox: Bounding box (x1, y1, x2, y2) where pose was estimated
        """
        if position is None or quaternion is None:
            return frame
        
        try:
            # Convert quaternion to rotation matrix
            def quaternion_to_rotation_matrix(q):
                x, y, z, w = q
                return np.array([
                    [1-2*(y**2+z**2), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2)]
                ])
            
            R = quaternion_to_rotation_matrix(quaternion)
            
            # Convert to rotation vector
            rvec, _ = cv2.Rodrigues(R)
            tvec = position.reshape(3, 1)
            
            # Define axis points - same as working code
            axis_length = 0.1  # 10cm
            axis_points = np.float32([
                [0, 0, 0],           # Origin
                [axis_length, 0, 0], # X-axis
                [0, axis_length, 0], # Y-axis
                [0, 0, axis_length]  # Z-axis
            ])
            
            # Get camera intrinsics - same as working code
            K, distCoeffs = self._get_camera_intrinsics()
            
            # Project axis points directly - no coordinate adjustment
            axis_proj, _ = cv2.projectPoints(axis_points, rvec, tvec, K, distCoeffs)
            axis_proj = axis_proj.reshape(-1, 2).astype(int)
            
            # Draw axes - same colors as working code
            origin = tuple(axis_proj[0])
            x_end = tuple(axis_proj[1])
            y_end = tuple(axis_proj[2])
            z_end = tuple(axis_proj[3])
            
            # Check if points are within frame bounds
            h, w = frame.shape[:2]
            points_in_bounds = all(
                0 <= pt[0] < w and 0 <= pt[1] < h 
                for pt in [origin, x_end, y_end, z_end]
            )
            
            if points_in_bounds:
                # Draw axes - same style as working code
                frame = cv2.line(frame, origin, x_end, (0, 0, 255), 3)    # X - Red
                frame = cv2.line(frame, origin, y_end, (0, 255, 0), 3)    # Y - Green  
                frame = cv2.line(frame, origin, z_end, (255, 0, 0), 3)    # Z - Blue
                
                # Draw origin point
                cv2.circle(frame, origin, 5, (255, 255, 255), -1)
                
                # Labels
                cv2.putText(frame, "X", (x_end[0] + 5, x_end[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, "Y", (y_end[0] + 5, y_end[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "Z", (z_end[0] + 5, z_end[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                # If points are out of bounds, draw simple axes at bbox center
                if bbox is not None:
                    center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                    axis_len = min(bbox[2]-bbox[0], bbox[3]-bbox[1]) // 6
                    
                    cv2.arrowedLine(frame, (center_x, center_y), 
                                   (center_x + axis_len, center_y), (0, 0, 255), 2)
                    cv2.arrowedLine(frame, (center_x, center_y), 
                                   (center_x, center_y - axis_len), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
                    
        except Exception as e:
            # If axis drawing fails, draw simple 2D axes at bbox center
            if bbox is not None:
                center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                axis_len = min(bbox[2]-bbox[0], bbox[3]-bbox[1]) // 6
                
                cv2.arrowedLine(frame, (center_x, center_y), 
                               (center_x + axis_len, center_y), (0, 0, 255), 2)
                cv2.arrowedLine(frame, (center_x, center_y), 
                               (center_x, center_y - axis_len), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
        
        return frame
    
    def run(self):
        """Main processing loop - optimized for fast axis with KF predict"""
        print("🚀 Starting simplified pose estimation")
        cv2.namedWindow('VAPE MK42 - Simplified', cv2.WINDOW_NORMAL)

        fps_timer = time.time()
        fps_count = 0

        try:
            while True:
                self.frame_count += 1
                self.context.frame_id = self.frame_count

                # Capture frame
                frame = self._get_frame()
                fps_count += 1

                # ====== 상태머신 처리 (기존과 동일) ======
                if self.state == TrackingState.INITIALIZING:
                    bbox = self._yolo_detect(frame)
                    if bbox is not None:
                        self.context.bbox = bbox
                        self.context.viewpoint = self._classify_viewpoint(frame, bbox)
                        if self._init_tracker(frame, bbox):
                            self.state = TrackingState.TRACKING
                            print(f"✅ Initialized - Tracking {self.context.viewpoint} viewpoint")
                        else:
                            self.state = TrackingState.DETECTING

                elif self.state == TrackingState.DETECTING:
                    bbox = self._yolo_detect(frame)
                    if bbox is not None:
                        self.context.bbox = bbox
                        self.context.viewpoint = self._classify_viewpoint(frame, bbox)
                        if self._init_tracker(frame, bbox):
                            self.state = TrackingState.TRACKING
                            print(f"✅ Object found - Tracking {self.context.viewpoint} viewpoint")

                elif self.state == TrackingState.TRACKING:
                    bbox, confidence = self._track_object(frame)
                    self.quality_monitor.update_tracking_confidence(confidence)

                    if bbox is not None and confidence > 0.6:
                        self.context.bbox = bbox
                        self.context.confidence = confidence
                        self.context.consecutive_failures = 0

                        if self.quality_monitor.should_reclassify_viewpoint():
                            print("🔄 Low matches detected - Reclassifying viewpoint...")
                            self.state = TrackingState.RECLASSIFYING
                    else:
                        self.context.consecutive_failures += 1
                        if self.context.consecutive_failures > 3:
                            print("❌ Tracking lost - Switching to detection")
                            self.state = TrackingState.LOST
                            self.tracker = None

                elif self.state == TrackingState.RECLASSIFYING:
                    if self.context.bbox is not None:
                        new_viewpoint = self._classify_viewpoint(frame, self.context.bbox)
                        if new_viewpoint != self.context.viewpoint:
                            print(f"🔄 Viewpoint changed: {self.context.viewpoint} → {new_viewpoint}")
                            self.context.viewpoint = new_viewpoint
                        self.quality_monitor.low_match_count = 0
                        self.state = TrackingState.TRACKING

                elif self.state == TrackingState.LOST:
                    bbox = self._yolo_detect(frame)
                    if bbox is not None:
                        self.context.bbox = bbox
                        self.context.viewpoint = self._classify_viewpoint(frame, bbox)
                        if self._init_tracker(frame, bbox):
                            self.state = TrackingState.TRACKING
                            print(f"✅ Object reacquired - Tracking {self.context.viewpoint}")
                        else:
                            self.state = TrackingState.DETECTING

                # ========== KF predict: 항상 매 프레임마다 먼저 실행 ==========
                pred_result = self.kf.predict()
                position, quaternion = None, None
                if pred_result is not None:
                    position, quaternion = pred_result

                # ========== 관측치(pose estimation)가 가능하면 KF update ==========
                if self.context.bbox is not None and self.context.viewpoint is not None:
                    x1, y1, x2, y2 = self.context.bbox
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size > 0:
                        pose_result = self._estimate_pose(cropped, self.context.viewpoint)
                        if pose_result[0] is not None:
                            obs_position, obs_quaternion, num_matches = pose_result
                            self.context.num_matches = num_matches
                            # KF update: 관측 들어온 프레임에서만 수행
                            position, quaternion = self.kf.update(obs_position, obs_quaternion)
                            # (position, quaternion)은 항상 KF의 최신 state

                # ================== Visualization ==================
                vis_frame = frame.copy()

                # Draw current state
                state_colors = {
                    TrackingState.INITIALIZING: (0, 0, 255),    # Red
                    TrackingState.DETECTING: (0, 100, 255),     # Orange
                    TrackingState.TRACKING: (0, 255, 0),        # Green
                    TrackingState.RECLASSIFYING: (255, 255, 0), # Cyan
                    TrackingState.LOST: (0, 0, 200)             # Dark Red
                }
                color = state_colors.get(self.state, (255, 255, 255))
                cv2.putText(vis_frame, f'State: {self.state.value.upper()}', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw bounding box if available
                if self.context.bbox is not None:
                    x1, y1, x2, y2 = self.context.bbox
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    if self.context.viewpoint:
                        cv2.putText(vis_frame, f'VP: {self.context.viewpoint}', (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    # KF의 (position, quaternion)로 axis 그리기 (항상 최신)
                    if position is not None and quaternion is not None:
                        vis_frame = self._draw_axes(vis_frame, position, quaternion, self.context.bbox)

                # Performance & FPS info (기존과 동일)
                cv2.putText(vis_frame, f'Frame: {self.frame_count}', (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                avg_matches = self.quality_monitor.get_average_matches()
                cv2.putText(vis_frame, f'Matches: {avg_matches:.1f}', (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # FPS 계산
                current_time = time.time()
                if current_time - fps_timer > 1.0:
                    time_diff = current_time - fps_timer
                    if time_diff > 0 and fps_count > 0:
                        fps = fps_count / time_diff
                        self.perf_monitor.fps_history.append(fps)
                    fps_count = 0
                    fps_timer = current_time

                if len(self.perf_monitor.fps_history) > 0:
                    fps_list = list(self.perf_monitor.fps_history)
                    avg_fps = np.mean(fps_list[-5:]) if len(fps_list) >= 5 else np.mean(fps_list)
                    fps_color = (0, 255, 0) if avg_fps > 15 else (0, 165, 255) if avg_fps > 10 else (0, 0, 255)
                    cv2.putText(vis_frame, f'FPS: {avg_fps:.1f}', (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

                # Performance timing
                y_offset = 150
                for name in ['yolo_detection', 'tracking', 'viewpoint_classification', 'pose_estimation']:
                    avg_time = self.perf_monitor.get_average(name)
                    if avg_time > 0:
                        time_color = (0, 255, 0) if avg_time < 30 else (0, 165, 255) if avg_time < 50 else (0, 0, 255)
                        cv2.putText(vis_frame, f'{name}: {avg_time:.1f}ms', 
                                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, time_color, 1)
                        y_offset += 15

                # Show pose values if available
                if position is not None:
                    pos_text = f'Pos: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]'
                    cv2.putText(vis_frame, pos_text, (10, vis_frame.shape[0]-40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if quaternion is not None:
                    quat_text = f'Quat: [{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]'
                    cv2.putText(vis_frame, quat_text, (10, vis_frame.shape[0]-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Show frame
                cv2.imshow('VAPE MK42 - Simplified', vis_frame)

                # Print performance stats occasionally
                try:
                    self.perf_monitor.print_stats(self.frame_count)
                except Exception as e:
                    if self.frame_count % 300 == 0:
                        print(f"⚠️ Performance monitoring error: {e}")

                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    
    def _cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            print("  📹 Camera released")
        cv2.destroyAllWindows()
        print("  🖼️ Windows closed")
        print("✅ Cleanup complete")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='VAPE MK42 - Simplified Robust Pose Estimator')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    return parser.parse_args()


if __name__ == "__main__":
    print("🚀 VAPE MK42 - Simplified Robust Real-time Pose Estimator")
    print("=" * 50)
    
    args = parse_args()
    
    # Global reference for signal handler
    estimator = None
    
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        if estimator:
            estimator._cleanup()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        estimator = SimplifiedPoseEstimator(args)
        print("📹 Starting camera feed...")
        print("Press 'q' to quit")
        estimator.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if estimator:
            estimator._cleanup()
        
    print("🏁 Program finished!")