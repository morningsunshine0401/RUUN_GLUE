"""
Advanced Smart Real-time Pose Estimator with Adaptive Intelligence
- Robust detection with adaptive frequency control
- Smart viewpoint classification (only when needed)
- Multi-scale processing and performance optimization
- Advanced failure recovery and situation handling
- XFeat + LighterGlue for superior feature matching
"""

import cv2
import numpy as np
import torch
import time
import argparse
import warnings
from collections import defaultdict, deque
from enum import Enum
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*custom_fwd.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")

# Disable gradients globally for performance
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 Starting Advanced Smart Pose Estimator...")
print("Loading required libraries...")

try:
    from ultralytics import YOLO
    print("✅ YOLO imported")
    
    import timm
    from torchvision import transforms
    print("✅ Vision models imported")
    
    from PIL import Image
    print("✅ PIL imported")
    
    from scipy.spatial.distance import cdist
    print("✅ SciPy imported")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install ultralytics timm torchvision pillow scipy")
    print("For XFeat: pip install kornia kornia-rs --no-deps")
    exit(1)

print("🔄 All libraries loaded successfully!")

# Advanced System States
class SystemState(Enum):
    INITIALIZING = "initializing"
    ACTIVE_DETECTION = "active_detection"
    STABLE_TRACKING = "stable_tracking"
    ADAPTIVE_TRACKING = "adaptive_tracking"
    RECOVERY_MODE = "recovery_mode"
    VIEWPOINT_UPDATE = "viewpoint_update"
    PERFORMANCE_MODE = "performance_mode"

@dataclass
class SystemConfig:
    """Dynamic system configuration"""
    # Detection settings
    detection_interval: int = 5  # frames between detections during tracking
    min_detection_interval: int = 2
    max_detection_interval: int = 15
    
    # Viewpoint settings
    viewpoint_stability_threshold: int = 10  # frames before considering viewpoint stable
    viewpoint_confidence_threshold: float = 0.8
    
    # Tracking settings
    tracking_confidence_threshold: float = 0.7
    tracking_stability_threshold: int = 5
    
    # Performance settings
    target_fps: float = 30.0
    frame_skip_threshold: float = 25.0  # ms
    adaptive_resolution: bool = True
    
    # Matching settings
    min_matches_stable: int = 15
    min_matches_acceptable: int = 8
    min_matches_critical: int = 4

class AdaptivePerformanceManager:
    """Smart performance management with adaptive controls"""
    
    def __init__(self, target_fps=30.0):
        self.target_fps = target_fps
        self.target_frame_time = 1000.0 / target_fps  # ms
        
        self.frame_times = deque(maxlen=30)
        self.processing_times = defaultdict(lambda: deque(maxlen=20))
        self.performance_level = 1.0  # 0.5-1.0 scale
        
        self.frame_skip_count = 0
        self.adaptive_resolution_scale = 1.0
        
    def update_timing(self, component: str, time_ms: float):
        """Update timing for a component"""
        self.processing_times[component].append(time_ms)
        
    def update_frame_time(self, total_time_ms: float):
        """Update total frame processing time"""
        self.frame_times.append(total_time_ms)
        self._update_performance_level()
        
    def _update_performance_level(self):
        """Dynamically adjust performance level based on timing"""
        if not self.frame_times:
            return
            
        avg_frame_time = np.mean(self.frame_times)
        
        if avg_frame_time > self.target_frame_time * 1.5:
            self.performance_level = max(0.5, self.performance_level - 0.1)
        elif avg_frame_time < self.target_frame_time * 0.8:
            self.performance_level = min(1.0, self.performance_level + 0.05)
            
    def should_skip_frame(self) -> bool:
        """Decide if we should skip processing this frame"""
        if not self.frame_times:
            return False
            
        recent_avg = np.mean(list(self.frame_times)[-5:])
        if recent_avg > self.target_frame_time * 2:
            self.frame_skip_count += 1
            return self.frame_skip_count % 2 == 0  # Skip every other frame
        
        self.frame_skip_count = 0
        return False
        
    def get_adaptive_resolution_scale(self) -> float:
        """Get current resolution scale based on performance"""
        if self.performance_level < 0.7:
            return 0.7  # Reduce to 70%
        elif self.performance_level < 0.8:
            return 0.85  # Reduce to 85%
        return 1.0  # Full resolution
        
    def get_detection_interval(self, base_interval: int) -> int:
        """Get adaptive detection interval"""
        if self.performance_level < 0.6:
            return min(base_interval * 2, 20)
        elif self.performance_level < 0.8:
            return min(base_interval + 2, 15)
        return base_interval

class SmartTrackingQualityMonitor:
    """Advanced tracking quality assessment with predictive capabilities"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # Match quality tracking
        self.match_history = deque(maxlen=20)
        self.match_quality_trend = deque(maxlen=10)
        
        # Tracking confidence
        self.tracking_confidence_history = deque(maxlen=15)
        self.tracking_stability_count = 0
        
        # Viewpoint stability
        self.viewpoint_history = deque(maxlen=10)
        self.viewpoint_stable_count = 0
        self.last_viewpoint_change = 0
        
        # Performance metrics
        self.pose_estimation_success_rate = deque(maxlen=20)
        
    def update_matches(self, num_matches: int):
        """Update match count and analyze trends"""
        self.match_history.append(num_matches)
        
        if len(self.match_history) >= 3:
            trend = np.mean(list(self.match_history)[-3:]) - np.mean(list(self.match_history)[-6:-3])
            self.match_quality_trend.append(trend)
            
    def update_tracking_confidence(self, confidence: float):
        """Update tracking confidence with stability analysis"""
        self.tracking_confidence_history.append(confidence)
        
        if confidence > self.config.tracking_confidence_threshold:
            self.tracking_stability_count += 1
        else:
            self.tracking_stability_count = 0
            
    def update_viewpoint(self, viewpoint: str):
        """Track viewpoint changes and stability"""
        if not self.viewpoint_history or self.viewpoint_history[-1] != viewpoint:
            self.viewpoint_history.append(viewpoint)
            self.viewpoint_stable_count = 0
            self.last_viewpoint_change = 0
        else:
            self.viewpoint_stable_count += 1
            
        self.last_viewpoint_change += 1
            
    def update_pose_success(self, success: bool):
        """Track pose estimation success rate"""
        self.pose_estimation_success_rate.append(success)
        
    def get_system_health(self) -> Dict[str, float]:
        """Get comprehensive system health metrics"""
        health = {}
        
        # Match quality health
        if self.match_history:
            avg_matches = np.mean(self.match_history)
            health['match_quality'] = min(1.0, avg_matches / self.config.min_matches_stable)
        else:
            health['match_quality'] = 0.0
            
        # Tracking stability health
        if self.tracking_confidence_history:
            avg_confidence = np.mean(self.tracking_confidence_history)
            health['tracking_stability'] = avg_confidence
        else:
            health['tracking_stability'] = 0.0
            
        # Viewpoint stability health
        health['viewpoint_stability'] = min(1.0, self.viewpoint_stable_count / self.config.viewpoint_stability_threshold)
        
        # Pose estimation health
        if self.pose_estimation_success_rate:
            health['pose_success_rate'] = np.mean(self.pose_estimation_success_rate)
        else:
            health['pose_success_rate'] = 0.0
            
        # Overall system health
        health['overall'] = np.mean(list(health.values()))
        
        return health
        
    def should_trigger_detection(self) -> bool:
        """Smart decision for when to trigger detection"""
        health = self.get_system_health()
        
        # Trigger detection if overall health is poor
        if health['overall'] < 0.6:
            return True
            
        # Trigger if tracking is unstable
        if health['tracking_stability'] < 0.5:
            return True
            
        # Trigger if match quality is degrading
        if len(self.match_quality_trend) >= 3:
            recent_trend = np.mean(list(self.match_quality_trend)[-3:])
            if recent_trend < -2:  # Declining trend
                return True
                
        return False
        
    def should_update_viewpoint(self) -> bool:
        """Decide if viewpoint classification is needed"""
        # Update if we haven't classified in a while and system is unstable
        if self.last_viewpoint_change > 30:  # 30 frames without update
            health = self.get_system_health()
            return health['overall'] < 0.8
            
        # Update if significant match quality drop
        if self.match_history and len(self.match_history) >= 5:
            recent_avg = np.mean(list(self.match_history)[-3:])
            older_avg = np.mean(list(self.match_history)[-8:-5])
            if recent_avg < older_avg * 0.6:  # 40% drop
                return True
                
        return False

class RobustKalmanFilter:
    """Enhanced Kalman filter with outlier rejection and adaptive noise"""
    
    def __init__(self, dt=1/30.0):
        self.dt = dt
        self.initialized = False
        
        # State: [px, py, pz, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]
        self.n_states = 13
        self.x = np.zeros(self.n_states)
        self.x[9] = 1.0  # quaternion w=1
        
        # Covariance matrix
        self.P = np.eye(self.n_states) * 0.1
        
        # Adaptive noise parameters
        self.base_process_noise = 1e-3
        self.base_measurement_noise = 1e-4
        self.Q = np.eye(self.n_states) * self.base_process_noise
        self.R = np.eye(7) * self.base_measurement_noise
        
        # Outlier detection
        self.innovation_history = deque(maxlen=10)
        self.outlier_threshold = 3.0  # Standard deviations
        
    def _is_outlier(self, innovation: np.ndarray) -> bool:
        """Detect measurement outliers using innovation analysis"""
        if len(self.innovation_history) < 3:
            return False
            
        # Calculate Mahalanobis distance
        recent_innovations = np.array(list(self.innovation_history)[-5:])
        mean_innovation = np.mean(recent_innovations, axis=0)
        
        innovation_norm = np.linalg.norm(innovation - mean_innovation)
        historical_std = np.std([np.linalg.norm(inn - mean_innovation) 
                               for inn in recent_innovations])
        
        if historical_std > 0:
            z_score = innovation_norm / historical_std
            return z_score > self.outlier_threshold
            
        return False
        
    def _adapt_noise_parameters(self, innovation: np.ndarray):
        """Adapt noise parameters based on innovation sequence"""
        self.innovation_history.append(innovation)
        
        if len(self.innovation_history) >= 5:
            recent_innovations = np.array(list(self.innovation_history)[-5:])
            innovation_variance = np.var(recent_innovations, axis=0)
            
            # Adapt measurement noise based on innovation variance
            avg_variance = np.mean(innovation_variance)
            if avg_variance > self.base_measurement_noise * 10:
                # High variance - increase measurement noise
                self.R = np.eye(7) * min(self.base_measurement_noise * 5, avg_variance * 0.1)
            else:
                # Low variance - decrease measurement noise
                self.R = np.eye(7) * max(self.base_measurement_noise, avg_variance * 0.5)

    def normalize_quaternion(self, q):
        """Normalize quaternion to unit length"""
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            return q / norm
        else:
            return np.array([0, 0, 0, 1])

    def predict(self):
        """Enhanced prediction with adaptive process noise"""
        if not self.initialized:
            return None
            
        # Extract state components
        px, py, pz = self.x[0:3]
        vx, vy, vz = self.x[3:6]
        qx, qy, qz, qw = self.x[6:10]
        wx, wy, wz = self.x[10:13]
        
        dt = self.dt
        
        # Predict position with velocity
        px_new = px + vx * dt
        py_new = py + vy * dt
        pz_new = pz + vz * dt
        
        # Velocity with damping
        damping = 0.99  # Slight velocity damping
        vx_new, vy_new, vz_new = vx * damping, vy * damping, vz * damping
        
        # Quaternion integration
        q = np.array([qx, qy, qz, qw])
        w = np.array([wx, wy, wz])
        
        omega_mat = np.array([
            [0,   -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy,  -wz,  0,   wx],
            [wz,   wy, -wx,  0 ]
        ])
        
        dq = 0.5 * dt * omega_mat @ q
        q_new = q + dq
        q_new = self.normalize_quaternion(q_new)
        
        # Angular velocity with damping
        wx_new, wy_new, wz_new = wx * damping, wy * damping, wz * damping
        
        # Update state
        self.x = np.array([
            px_new, py_new, pz_new,
            vx_new, vy_new, vz_new,
            q_new[0], q_new[1], q_new[2], q_new[3],
            wx_new, wy_new, wz_new
        ])
        
        # Build Jacobian F
        F = np.eye(self.n_states)
        F[0, 3] = dt  # dx/dvx
        F[1, 4] = dt  # dy/dvy
        F[2, 5] = dt  # dz/dvz
        
        # Adaptive process noise based on motion
        velocity_magnitude = np.linalg.norm([vx, vy, vz])
        angular_velocity_magnitude = np.linalg.norm([wx, wy, wz])
        
        motion_factor = 1 + velocity_magnitude * 10 + angular_velocity_magnitude * 5
        adaptive_Q = self.Q * motion_factor
        
        # Update covariance
        self.P = F @ self.P @ F.T + adaptive_Q
        
        return self.x[0:3], self.x[6:10]

    def update(self, position, quaternion, confidence=1.0):
        """Robust update with outlier rejection and confidence weighting"""
        measurement = np.concatenate([position, quaternion])
        
        if not self.initialized:
            self.x[0:3] = position
            self.x[6:10] = self.normalize_quaternion(quaternion)
            self.initialized = True
            return self.x[0:3], self.x[6:10]

        # Predicted measurement
        predicted_measurement = np.array([
            self.x[0], self.x[1], self.x[2],
            self.x[6], self.x[7], self.x[8], self.x[9]
        ])
        
        # Innovation
        innovation = measurement - predicted_measurement
        
        # Handle quaternion wraparound
        q_meas = measurement[3:7]
        q_pred = predicted_measurement[3:7]
        if np.dot(q_meas, q_pred) < 0:
            q_meas = -q_meas
            innovation[3:7] = q_meas - q_pred
        
        # Outlier detection
        if self._is_outlier(innovation):
            print("⚠️ Outlier detected, skipping update")
            return self.x[0:3], self.x[6:10]
        
        # Adapt noise parameters
        self._adapt_noise_parameters(innovation)
        
        # Measurement Jacobian H
        H = np.zeros((7, self.n_states))
        H[0, 0] = 1.0  # px
        H[1, 1] = 1.0  # py
        H[2, 2] = 1.0  # pz
        H[3, 6] = 1.0  # qx
        H[4, 7] = 1.0  # qy
        H[5, 8] = 1.0  # qz
        H[6, 9] = 1.0  # qw
        
        # Confidence-weighted measurement noise
        R_weighted = self.R / confidence
        
        # Kalman update
        S = H @ self.P @ H.T + R_weighted
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x += K @ innovation
        
        # Normalize quaternion
        self.x[6:10] = self.normalize_quaternion(self.x[6:10])
        
        # Covariance update (Joseph form)
        I = np.eye(self.n_states)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_weighted @ K.T
        
        return self.x[0:3], self.x[6:10]

class AdvancedSmartPoseEstimator:
    def __init__(self, args):
        print("🔧 Initializing Advanced Smart Pose Estimator...")
        self.args = args
        self.config = SystemConfig()
        
        # System components
        self.performance_manager = AdaptivePerformanceManager(self.config.target_fps)
        self.quality_monitor = SmartTrackingQualityMonitor(self.config)
        self.kf = RobustKalmanFilter()
        
        # System state
        self.state = SystemState.INITIALIZING
        self.frame_count = 0
        self.detection_counter = 0
        
        # Tracking state
        self.current_bbox = None
        self.current_viewpoint = None
        self.viewpoint_confidence = 0.0
        self.tracker = None
        
        # Multi-scale processing
        self.processing_scales = [1.0, 0.8, 0.6]
        self.current_scale_idx = 0
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")
        
        # Initialize components
        print("🔄 Loading AI models...")
        self._init_models()
        
        print("🔄 Setting up camera...")
        self._init_camera()
        
        print("🔄 Loading anchor data...")
        self._init_anchor_data()
        
        print("✅ Advanced Smart Pose Estimator initialized!")
        
    def _init_models(self):
        """Initialize all models with error handling"""
        start_time = time.time()
        
        try:
            # YOLO model with multiple size options
            print("  📦 Loading YOLO...")
            model_options = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
            
            for model_name in model_options:
                try:
                    self.yolo_model = YOLO(model_name)
                    if torch.cuda.is_available():
                        self.yolo_model.to('cuda')
                    print(f"  ✅ YOLO loaded: {model_name}")
                    break
                except Exception as e:
                    print(f"  ⚠️ Failed to load {model_name}: {e}")
                    continue
            else:
                raise Exception("Could not load any YOLO model")
            
            # Viewpoint classifier
            print("  📦 Loading viewpoint classifier...")
            self.vp_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=4)
            
            try:
                self.vp_model.load_state_dict(torch.load('mobilevit_viewpoint_twostage_final_2.pth', 
                                                        map_location=self.device))
                print("  ✅ Viewpoint model loaded from file")
            except FileNotFoundError:
                print("  ⚠️ Viewpoint model file not found, using random weights")
            
            self.vp_model.eval().to(self.device)
            
            # Transform for viewpoint classifier
            self.vp_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
            
            # XFeat model with error handling
            print("  📦 Loading XFeat...")
            try:
                self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', 
                                           pretrained=True, top_k=4096)
                print("  ✅ XFeat loaded successfully")
            except Exception as e:
                print(f"  ❌ XFeat loading failed: {e}")
                print("  Please install: pip install kornia kornia-rs --no-deps")
                raise
            
            # Class names
            self.class_names = ['NE', 'NW', 'SE', 'SW']
            
            print(f"✅ All models loaded in {(time.time() - start_time)*1000:.1f}ms")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
        
    def _init_camera(self):
        """Initialize camera with fallback options"""
        camera_indices = [0, 1, 2]  # Try multiple camera indices
        
        for idx in camera_indices:
            try:
                self.cap = cv2.VideoCapture(idx)
                
                if self.cap.isOpened():
                    # Set optimal camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Test capture
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"✅ Camera {idx} initialized: {actual_width}x{actual_height}")
                        return
                        
                self.cap.release()
                
            except Exception as e:
                print(f"❌ Camera {idx} failed: {e}")
                continue
                
        # Fallback to dummy camera
        print("⚠️ No cameras available, using dummy video source")
        self.cap = None
        
    def _init_anchor_data(self):
        """Initialize anchor data with multiple fallback options"""
        anchor_paths = [
            'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png',
            'anchor.png',
            'reference.png'
        ]
        
        for anchor_path in anchor_paths:
            try:
                self.anchor_image = cv2.imread(anchor_path)
                if self.anchor_image is not None:
                    self.anchor_image = cv2.resize(self.anchor_image, (1280, 720))
                    print(f"✅ Anchor image loaded: {anchor_path}")
                    break
            except Exception as e:
                print(f"⚠️ Failed to load {anchor_path}: {e}")
                continue
        else:
            print("⚠️ Using synthetic anchor image")
            # Create a synthetic anchor with features
            self.anchor_image = self._create_synthetic_anchor()
        
        # Define anchor keypoints (same as before)
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
        
        # Extract anchor features
        self._extract_anchor_features()
        
        # Create viewpoint-specific anchors
        self.viewpoint_anchors = {
            'NE': {'2d': self.anchor_2d, '3d': self.anchor_3d, 'features': self.anchor_features},
            'NW': {'2d': self.anchor_2d, '3d': self.anchor_3d, 'features': self.anchor_features},
            'SE': {'2d': self.anchor_2d, '3d': self.anchor_3d, 'features': self.anchor_features},
            'SW': {'2d': self.anchor_2d, '3d': self.anchor_3d, 'features': self.anchor_features}
        }
        
        print("✅ Anchor data initialized for 4 viewpoints")
    
    def _create_synthetic_anchor(self):
        """Create a synthetic anchor image with detectable features"""
        anchor = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Add some geometric patterns
        cv2.rectangle(anchor, (300, 200), (980, 520), (100, 100, 100), -1)
        cv2.rectangle(anchor, (400, 250), (880, 470), (150, 150, 150), 3)
        
        # Add circles
        for i in range(8):
            center = (450 + i * 60, 300 + (i % 2) * 100)
            cv2.circle(anchor, center, 20, (200, 200, 200), -1)
            cv2.circle(anchor, center, 15, (50, 50, 50), 2)
            
        # Add some lines
        for i in range(5):
            y = 280 + i * 30
            cv2.line(anchor, (420, y), (860, y), (180, 180, 180), 2)
            
        return anchor
    
    def _extract_anchor_features(self):
        """Extract XFeat features from anchor image"""
        try:
            anchor_rgb = cv2.cvtColor(self.anchor_image, cv2.COLOR_BGR2RGB)
            self.anchor_features = self.xfeat.detectAndCompute(anchor_rgb, top_k=4096)[0]
            self.anchor_features.update({
                'image_size': (self.anchor_image.shape[1], self.anchor_image.shape[0])
            })
            print("✅ Anchor XFeat features extracted")
        except Exception as e:
            print(f"❌ Anchor feature extraction failed: {e}")
            # Create dummy features for testing
            self.anchor_features = {'keypoints': np.array([[640, 360]]), 'image_size': (1280, 720)}
    
    def _get_frame(self):
        """Get frame with adaptive resolution"""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # Apply adaptive resolution scaling
                scale = self.performance_manager.get_adaptive_resolution_scale()
                if scale < 1.0:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    frame = cv2.resize(frame, (new_w, new_h))
                    # Resize back to maintain consistency
                    frame = cv2.resize(frame, (w, h))
                return frame
        
        # Dummy frame with more realistic content
        frame = self._create_dummy_frame()
        return frame
    
    def _create_dummy_frame(self):
        """Create a realistic dummy frame for testing"""
        frame = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
        
        # Add moving object simulation
        center_x = 640 + int(50 * np.sin(self.frame_count * 0.1))
        center_y = 360 + int(30 * np.cos(self.frame_count * 0.15))
        size = 80 + int(20 * np.sin(self.frame_count * 0.2))
        
        # Draw object with some features
        cv2.rectangle(frame, 
                     (center_x - size, center_y - size), 
                     (center_x + size, center_y + size), 
                     (0, 255, 0), -1)
        
        cv2.circle(frame, (center_x, center_y), size//2, (255, 255, 0), 3)
        
        # Add some noise and features
        for i in range(10):
            x = center_x + np.random.randint(-size, size)
            y = center_y + np.random.randint(-size, size)
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
        
        return frame
    
    def _multi_scale_detection(self, frame):
        """Multi-scale YOLO detection for robustness"""
        start_time = time.time()
        
        best_bbox = None
        best_confidence = 0.0
        
        # Try different scales based on performance level
        scales_to_try = [1.0]
        if self.performance_manager.performance_level > 0.8:
            scales_to_try = [1.0, 0.8]  # Try multiple scales when performance is good
        
        for scale in scales_to_try:
            # Scale frame
            h, w = frame.shape[:2]
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            scaled_frame = cv2.resize(frame, (scaled_w, scaled_h))
            
            # YOLO detection
            try:
                results = self.yolo_model(scaled_frame[..., ::-1], verbose=False)
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    if len(boxes) > 0:
                        # Scale back to original size
                        scale_factor_x = w / scaled_w
                        scale_factor_y = h / scaled_h
                        
                        for bbox, conf in zip(boxes, confidences):
                            if conf > best_confidence:
                                x1 = int(bbox[0] * scale_factor_x)
                                y1 = int(bbox[1] * scale_factor_y)
                                x2 = int(bbox[2] * scale_factor_x)
                                y2 = int(bbox[3] * scale_factor_y)
                                
                                # Ensure bounds
                                x1 = max(0, min(x1, w-1))
                                y1 = max(0, min(y1, h-1))
                                x2 = max(x1+1, min(x2, w))
                                y2 = max(y1+1, min(y2, h))
                                
                                best_bbox = (x1, y1, x2, y2)
                                best_confidence = conf
                                
            except Exception as e:
                print(f"⚠️ Detection failed at scale {scale}: {e}")
                continue
        
        detection_time = (time.time() - start_time) * 1000
        self.performance_manager.update_timing('detection', detection_time)
        
        return best_bbox, best_confidence
    
    def _init_tracker(self, frame, bbox):
        """Initialize tracker with error handling"""
        try:
            start_time = time.time()
            
            # Try different tracker types based on performance
            tracker_types = [cv2.TrackerCSRT_create, cv2.TrackerKCF_create]
            if hasattr(cv2, 'legacy'):
                tracker_types = [cv2.legacy.TrackerCSRT_create, cv2.legacy.TrackerKCF_create]
            
            for tracker_create in tracker_types:
                try:
                    self.tracker = tracker_create()
                    break
                except AttributeError:
                    continue
            else:
                print("❌ No suitable tracker found")
                return False
            
            # Initialize tracker
            x1, y1, x2, y2 = bbox
            w, h = x2-x1, y2-y1
            success = self.tracker.init(frame, (x1, y1, w, h))
            
            init_time = (time.time() - start_time) * 1000
            self.performance_manager.update_timing('tracker_init', init_time)
            
            return success
            
        except Exception as e:
            print(f"❌ Tracker initialization failed: {e}")
            return False
    
    def _robust_tracking(self, frame):
        """Robust tracking with quality assessment"""
        if self.tracker is None:
            return None, 0.0
            
        start_time = time.time()
        
        try:
            success, opencv_bbox = self.tracker.update(frame)
            
            track_time = (time.time() - start_time) * 1000
            self.performance_manager.update_timing('tracking', track_time)
            
            if not success:
                return None, 0.0
                
            # Convert bbox format
            x, y, w, h = opencv_bbox
            bbox = (int(x), int(y), int(x+w), int(y+h))
            
            # Enhanced confidence estimation
            confidence = self._estimate_tracking_confidence(bbox, frame)
            
            return bbox, confidence
            
        except Exception as e:
            print(f"⚠️ Tracking failed: {e}")
            return None, 0.0
    
    def _estimate_tracking_confidence(self, bbox, frame):
        """Enhanced tracking confidence estimation"""
        if self.current_bbox is None:
            return 1.0
            
        # Basic geometric consistency
        x1, y1, x2, y2 = bbox
        px1, py1, px2, py2 = self.current_bbox
        
        # Size consistency
        current_area = (x2-x1) * (y2-y1)
        prev_area = (px2-px1) * (py2-py1)
        if prev_area > 0:
            size_ratio = min(current_area, prev_area) / max(current_area, prev_area)
        else:
            size_ratio = 0.5
        
        # Position consistency
        center_x, center_y = (x1+x2)/2, (y1+y2)/2
        prev_center_x, prev_center_y = (px1+px2)/2, (py1+py2)/2
        distance = np.sqrt((center_x-prev_center_x)**2 + (center_y-prev_center_y)**2)
        
        # Frame dimensions for normalization
        h, w = frame.shape[:2]
        normalized_distance = distance / np.sqrt(w*w + h*h)
        
        # Template matching confidence (basic)
        try:
            if self.current_bbox is not None:
                prev_crop = frame[py1:py2, px1:px2]
                curr_crop = frame[y1:y2, x1:x2]
                
                if prev_crop.size > 0 and curr_crop.size > 0:
                    # Resize to same size for comparison
                    target_size = (64, 64)
                    prev_resized = cv2.resize(prev_crop, target_size)
                    curr_resized = cv2.resize(curr_crop, target_size)
                    
                    # Simple correlation
                    correlation = cv2.matchTemplate(prev_resized, curr_resized, cv2.TM_CCOEFF_NORMED)[0][0]
                    template_confidence = max(0, correlation)
                else:
                    template_confidence = 0.5
            else:
                template_confidence = 0.5
        except:
            template_confidence = 0.5
        
        # Combined confidence
        position_confidence = max(0, 1 - normalized_distance * 10)
        overall_confidence = (size_ratio * 0.3 + 
                            position_confidence * 0.4 + 
                            template_confidence * 0.3)
        
        return np.clip(overall_confidence, 0.0, 1.0)
    
    def _smart_viewpoint_classification(self, frame, bbox):
        """Smart viewpoint classification (only when needed)"""
        start_time = time.time()
        
        try:
            x1, y1, x2, y2 = bbox
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size == 0:
                return self.current_viewpoint or 'NE', 0.0
            
            # Resize for classification
            crop_resized = cv2.resize(cropped, (128, 128))
            
            # Convert and normalize
            img_pil = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
            input_tensor = self.vp_transform(img_pil).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.vp_model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                confidence, pred = torch.max(probabilities, dim=1)
                
                viewpoint = self.class_names[pred.item()]
                confidence_value = confidence.item()
            
            classification_time = (time.time() - start_time) * 1000
            self.performance_manager.update_timing('viewpoint_classification', classification_time)
            
            return viewpoint, confidence_value
            
        except Exception as e:
            print(f"⚠️ Viewpoint classification failed: {e}")
            return self.current_viewpoint or 'NE', 0.0
    
    def _robust_pose_estimation(self, cropped_frame, viewpoint):
        """Robust pose estimation with error handling"""
        try:
            start_time = time.time()
            
            # Feature extraction
            frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            frame_features = self.xfeat.detectAndCompute(frame_rgb, top_k=4096)[0]
            frame_features.update({
                'image_size': (cropped_frame.shape[1], cropped_frame.shape[0])
            })
            
            extraction_time = (time.time() - start_time) * 1000
            self.performance_manager.update_timing('feature_extraction', extraction_time)
            
            # Matching
            start_time = time.time()
            
            anchor_data = self.viewpoint_anchors[viewpoint]
            mkpts_0, mkpts_1 = self.xfeat.match_lighterglue(
                anchor_data['features'], 
                frame_features
            )
            
            matching_time = (time.time() - start_time) * 1000
            self.performance_manager.update_timing('matching', matching_time)
            
            num_matches = len(mkpts_0)
            self.quality_monitor.update_matches(num_matches)
            
            if num_matches < self.config.min_matches_critical:
                return None, None, num_matches
            
            # PnP solving with RANSAC
            start_time = time.time()
            
            # Map to 3D points
            anchor_2d = anchor_data['2d']
            anchor_3d = anchor_data['3d']
            
            # Find correspondences
            distances = cdist(mkpts_0, anchor_2d)
            closest_indices = np.argmin(distances, axis=1)
            
            # Adaptive threshold based on match quality
            if num_matches > self.config.min_matches_stable:
                distance_threshold = 3.0
            elif num_matches > self.config.min_matches_acceptable:
                distance_threshold = 5.0
            else:
                distance_threshold = 8.0
                
            valid_mask = np.min(distances, axis=1) < distance_threshold
            
            if np.sum(valid_mask) < self.config.min_matches_critical:
                return None, None, num_matches
            
            # Get valid correspondences
            points_3d = anchor_3d[closest_indices[valid_mask]]
            points_2d = mkpts_1[valid_mask]
            
            # Camera parameters
            K, dist_coeffs = self._get_camera_intrinsics()
            
            # Adaptive RANSAC parameters
            ransac_iterations = 1000 if num_matches > 20 else 500
            reprojection_error = 2.0 if num_matches > 15 else 4.0
            
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=points_3d.reshape(-1, 1, 3),
                imagePoints=points_2d.reshape(-1, 1, 2),
                cameraMatrix=K,
                distCoeffs=dist_coeffs,
                reprojectionError=reprojection_error,
                confidence=0.99,
                iterationsCount=ransac_iterations,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            pnp_time = (time.time() - start_time) * 1000
            self.performance_manager.update_timing('pnp_solving', pnp_time)
            
            if not success or (inliers is not None and len(inliers) < 4):
                return None, None, num_matches
            
            # Convert to position and quaternion
            R, _ = cv2.Rodrigues(rvec)
            position = tvec.flatten()
            quaternion = self._rotation_matrix_to_quaternion(R)
            
            # Estimate pose confidence
            pose_confidence = self._estimate_pose_confidence(points_3d, points_2d, rvec, tvec, K, inliers)
            
            return position, quaternion, num_matches, pose_confidence
            
        except Exception as e:
            print(f"⚠️ Pose estimation failed: {e}")
            return None, None, 0, 0.0
    
    def _rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion [x, y, z, w]"""
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
    
    def _estimate_pose_confidence(self, points_3d, points_2d, rvec, tvec, K, inliers):
        """Estimate pose estimation confidence"""
        if inliers is None or len(inliers) == 0:
            return 0.0
            
        try:
            # Reprojection error
            projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
            projected_points = projected_points.reshape(-1, 2)
            
            errors = np.linalg.norm(points_2d - projected_points, axis=1)
            mean_error = np.mean(errors)
            
            # Inlier ratio
            inlier_ratio = len(inliers) / len(points_3d)
            
            # Error-based confidence
            error_confidence = max(0, 1 - mean_error / 10.0)  # Normalize by 10 pixels
            
            # Combined confidence
            overall_confidence = (error_confidence * 0.6 + inlier_ratio * 0.4)
            
            return np.clip(overall_confidence, 0.0, 1.0)
            
        except:
            return 0.5
    
    def _get_camera_intrinsics(self):
        """Get camera intrinsic parameters"""
        fx = 1460.10150
        fy = 1456.48915
        cx = 604.85462
        cy = 328.64800
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = None  # Simplified
        
        return K, dist_coeffs
    
    def _state_machine_update(self, frame):
        """Advanced state machine with intelligent transitions"""
        
        # Check if we should skip processing for performance
        if self.performance_manager.should_skip_frame():
            return self.state, None, None
        
        health = self.quality_monitor.get_system_health()
        detection_interval = self.performance_manager.get_detection_interval(self.config.detection_interval)
        
        if self.state == SystemState.INITIALIZING:
            print("🔍 System initializing - First detection...")
            bbox, confidence = self._multi_scale_detection(frame)
            if bbox is not None:
                self.current_bbox = bbox
                viewpoint, vp_confidence = self._smart_viewpoint_classification(frame, bbox)
                self.current_viewpoint = viewpoint
                self.viewpoint_confidence = vp_confidence
                
                if self._init_tracker(frame, bbox):
                    self.state = SystemState.STABLE_TRACKING
                    self.detection_counter = 0
                    print(f"✅ System initialized - Tracking {viewpoint} (conf: {vp_confidence:.2f})")
                else:
                    self.state = SystemState.ACTIVE_DETECTION
        
        elif self.state == SystemState.ACTIVE_DETECTION:
            print("🔍 Active detection mode...")
            bbox, confidence = self._multi_scale_detection(frame)
            if bbox is not None:
                self.current_bbox = bbox
                # Only classify viewpoint if needed
                if self.quality_monitor.should_update_viewpoint():
                    viewpoint, vp_confidence = self._smart_viewpoint_classification(frame, bbox)
                    self.current_viewpoint = viewpoint
                    self.viewpoint_confidence = vp_confidence
                
                if self._init_tracker(frame, bbox):
                    self.state = SystemState.STABLE_TRACKING
                    self.detection_counter = 0
                    print(f"✅ Object acquired - Tracking {self.current_viewpoint}")
        
        elif self.state == SystemState.STABLE_TRACKING:
            # Track object
            bbox, track_confidence = self._robust_tracking(frame)
            self.quality_monitor.update_tracking_confidence(track_confidence)
            
            if bbox is not None and track_confidence > self.config.tracking_confidence_threshold:
                self.current_bbox = bbox
                self.detection_counter += 1
                
                # Periodic detection for verification
                if self.detection_counter >= detection_interval:
                    verify_bbox, verify_confidence = self._multi_scale_detection(frame)
                    if verify_bbox is not None:
                        # Update bbox if detection is more confident
                        if verify_confidence > track_confidence + 0.2:
                            self.current_bbox = verify_bbox
                            if self._init_tracker(frame, verify_bbox):
                                print(f"🔄 Tracker updated with detection (conf: {verify_confidence:.2f})")
                    self.detection_counter = 0
                
                # Check if we need viewpoint update
                if self.quality_monitor.should_update_viewpoint():
                    self.state = SystemState.VIEWPOINT_UPDATE
                    
                # Check system health
                elif health['overall'] < 0.6:
                    self.state = SystemState.ADAPTIVE_TRACKING
                    print("⚠️ System health degraded - Adaptive mode")
                    
            else:
                print("❌ Tracking lost - Recovery mode")
                self.state = SystemState.RECOVERY_MODE
                self.tracker = None
        
        elif self.state == SystemState.ADAPTIVE_TRACKING:
            # More aggressive detection and verification
            self.detection_counter += 1
            
            # Track if tracker exists
            bbox, track_confidence = None, 0.0
            if self.tracker is not None:
                bbox, track_confidence = self._robust_tracking(frame)
                
            # More frequent detection
            if self.detection_counter >= max(1, detection_interval // 2):
                detect_bbox, detect_confidence = self._multi_scale_detection(frame)
                
                # Choose best result
                if detect_bbox is not None:
                    if bbox is None or detect_confidence > track_confidence + 0.1:
                        bbox = detect_bbox
                        if self._init_tracker(frame, bbox):
                            print(f"🔄 Tracker reset in adaptive mode")
                            
                self.detection_counter = 0
            
            if bbox is not None:
                self.current_bbox = bbox
                self.quality_monitor.update_tracking_confidence(max(track_confidence, 0.5))
                
                # Return to stable tracking if health improves
                if health['overall'] > 0.7:
                    self.state = SystemState.STABLE_TRACKING
                    print("✅ System health recovered - Stable tracking")
            else:
                self.state = SystemState.RECOVERY_MODE
        
        elif self.state == SystemState.VIEWPOINT_UPDATE:
            if self.current_bbox is not None:
                viewpoint, vp_confidence = self._smart_viewpoint_classification(frame, self.current_bbox)
                
                if viewpoint != self.current_viewpoint:
                    print(f"🔄 Viewpoint changed: {self.current_viewpoint} → {viewpoint} (conf: {vp_confidence:.2f})")
                    self.current_viewpoint = viewpoint
                    self.viewpoint_confidence = vp_confidence
                
                self.quality_monitor.update_viewpoint(viewpoint)
                
                # Return to appropriate tracking state
                if health['overall'] > 0.7:
                    self.state = SystemState.STABLE_TRACKING
                else:
                    self.state = SystemState.ADAPTIVE_TRACKING
        
        elif self.state == SystemState.RECOVERY_MODE:
            print("🔍 Recovery mode - Aggressive detection...")
            
            # Try detection at multiple scales
            bbox, confidence = self._multi_scale_detection(frame)
            
            if bbox is not None:
                self.current_bbox = bbox
                
                # Classify viewpoint
                viewpoint, vp_confidence = self._smart_viewpoint_classification(frame, bbox)
                self.current_viewpoint = viewpoint
                self.viewpoint_confidence = vp_confidence
                
                if self._init_tracker(frame, bbox):
                    self.state = SystemState.ADAPTIVE_TRACKING  # Start conservatively
                    print(f"✅ Recovery successful - {viewpoint} viewpoint")
                else:
                    self.state = SystemState.ACTIVE_DETECTION
        
        return self.state, self.current_bbox, self.current_viewpoint
    
    def _advanced_visualization(self, frame, position, quaternion, num_matches, system_health):
        """Advanced visualization with system status"""
        vis_frame = frame.copy()
        
        # State color mapping
        state_colors = {
            SystemState.INITIALIZING: (0, 0, 255),      # Red
            SystemState.ACTIVE_DETECTION: (0, 100, 255), # Orange
            SystemState.STABLE_TRACKING: (0, 255, 0),    # Green
            SystemState.ADAPTIVE_TRACKING: (0, 255, 255), # Yellow
            SystemState.RECOVERY_MODE: (0, 0, 200),      # Dark Red
            SystemState.VIEWPOINT_UPDATE: (255, 255, 0), # Cyan
            SystemState.PERFORMANCE_MODE: (255, 0, 255)  # Magenta
        }
        
        color = state_colors.get(self.state, (255, 255, 255))
        
        # System status panel
        panel_height = 200
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0, vis_frame)
        
        # Status text
        y_offset = 30
        line_height = 25
        
        cv2.putText(vis_frame, f'State: {self.state.value.upper()}', (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += line_height
        
        cv2.putText(vis_frame, f'Frame: {self.frame_count}', (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        if self.current_viewpoint:
            cv2.putText(vis_frame, f'Viewpoint: {self.current_viewpoint} ({self.viewpoint_confidence:.2f})', 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(vis_frame, f'Matches: {num_matches}', (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # System health indicators
        health_y = y_offset
        for name, value in system_health.items():
            if name == 'overall':
                continue
            health_color = (0, 255, 0) if value > 0.7 else (0, 255, 255) if value > 0.4 else (0, 0, 255)
            cv2.putText(vis_frame, f'{name}: {value:.2f}', (20, health_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, health_color, 1)
            health_y += 20
        
        # Performance indicators
        perf_level = self.performance_manager.performance_level
        perf_color = (0, 255, 0) if perf_level > 0.8 else (0, 255, 255) if perf_level > 0.6 else (0, 0, 255)
        cv2.putText(vis_frame, f'Performance: {perf_level:.2f}', (20, health_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, perf_color, 1)
        
        # Draw bounding box
        if self.current_bbox is not None:
            x1, y1, x2, y2 = self.current_bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw coordinate axes if pose is available
            if position is not None and quaternion is not None:
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    try:
                        cropped_with_axes = self._draw_coordinate_axes(cropped.copy(), position, quaternion)
                        vis_frame[y1:y2, x1:x2] = cropped_with_axes
                    except:
                        pass
        
        # XFeat/LighterGlue indicator
        cv2.putText(vis_frame, 'XFeat + LighterGlue', (vis_frame.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return vis_frame
    
    def _draw_coordinate_axes(self, frame, position, quaternion):
        """Draw 3D coordinate axes on frame"""
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
            rvec, _ = cv2.Rodrigues(R)
            tvec = position.reshape(3, 1)
            
            # Define axis points
            axis_length = 0.05  # 5cm
            axis_points = np.float32([
                [0, 0, 0],           # Origin
                [axis_length, 0, 0], # X-axis (Red)
                [0, axis_length, 0], # Y-axis (Green)
                [0, 0, axis_length]  # Z-axis (Blue)
            ])
            
            # Project to image
            K, dist_coeffs = self._get_camera_intrinsics()
            
            # Adjust camera matrix for cropped region
            K_cropped = K.copy()
            # Note: This is simplified - in practice you'd need to adjust cx, cy based on crop position
            
            projected, _ = cv2.projectPoints(axis_points, rvec, tvec, K_cropped, dist_coeffs)
            projected = projected.reshape(-1, 2).astype(int)
            
            # Ensure points are within frame bounds
            h, w = frame.shape[:2]
            projected = np.clip(projected, [0, 0], [w-1, h-1])
            
            # Draw axes
            origin = tuple(projected[0])
            
            # X-axis (Red)
            cv2.arrowedLine(frame, origin, tuple(projected[1]), (0, 0, 255), 2, tipLength=0.3)
            
            # Y-axis (Green)
            cv2.arrowedLine(frame, origin, tuple(projected[2]), (0, 255, 0), 2, tipLength=0.3)
            
            # Z-axis (Blue)
            cv2.arrowedLine(frame, origin, tuple(projected[3]), (255, 0, 0), 2, tipLength=0.3)
            
            return frame
            
        except Exception as e:
            print(f"⚠️ Axis drawing failed: {e}")
            return frame
    
    def run(self):
        """Advanced main processing loop"""
        print("🚀 Starting Advanced Smart Pose Estimation")
        
        cv2.namedWindow('Advanced Smart Pose Estimation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Advanced Smart Pose Estimation', 1280, 720)
        
        try:
            while True:
                loop_start_time = time.time()
                
                # Get frame
                frame = self._get_frame()
                if frame is None:
                    continue
                    
                self.frame_count += 1
                
                # State machine update
                state, bbox, viewpoint = self._state_machine_update(frame)
                
                # Pose estimation if we have valid tracking
                position, quaternion, num_matches = None, None, 0
                pose_confidence = 0.0
                
                if bbox is not None and viewpoint is not None:
                    x1, y1, x2, y2 = bbox
                    cropped = frame[y1:y2, x1:x2]
                    
                    if cropped.size > 100:  # Minimum crop size
                        pose_result = self._robust_pose_estimation(cropped, viewpoint)
                        
                        if len(pose_result) == 4:  # With confidence
                            position, quaternion, num_matches, pose_confidence = pose_result
                        else:  # Legacy format
                            position, quaternion, num_matches = pose_result
                            pose_confidence = 0.5
                        
                        self.quality_monitor.update_pose_success(position is not None)
                        
                        if position is not None and quaternion is not None:
                            # Update Kalman filter
                            position, quaternion = self.kf.update(position, quaternion, pose_confidence)
                        else:
                            # Use Kalman prediction only
                            pred_result = self.kf.predict()
                            if pred_result is not None:
                                position, quaternion = pred_result
                
                # Get system health
                system_health = self.quality_monitor.get_system_health()
                
                # Advanced visualization
                vis_frame = self._advanced_visualization(frame, position, quaternion, num_matches, system_health)
                
                # Performance monitoring
                total_frame_time = (time.time() - loop_start_time) * 1000
                self.performance_manager.update_frame_time(total_frame_time)
                
                # FPS display
                if total_frame_time > 0:
                    fps = 1000 / total_frame_time
                    cv2.putText(vis_frame, f'FPS: {fps:.1f}', (vis_frame.shape[1] - 100, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Advanced Smart Pose Estimation', vis_frame)
                
                # Print detailed stats periodically
                if self.frame_count % 100 == 0:
                    self._print_detailed_stats(system_health)
                
                # Exit on 'q'
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):  # Reset system
                    print("🔄 System reset requested")
                    self.state = SystemState.INITIALIZING
                    self.tracker = None
                    self.current_bbox = None
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def _print_detailed_stats(self, system_health):
        """Print detailed system statistics"""
        print(f"\n=== FRAME {self.frame_count} SYSTEM STATUS ===")
        print(f"State: {self.state.value}")
        print(f"Performance Level: {self.performance_manager.performance_level:.2f}")
        
        print("\nSystem Health:")
        for name, value in system_health.items():
            emoji = "🟢" if value > 0.7 else "🟡" if value > 0.4 else "🔴"
            print(f"  {emoji} {name}: {value:.2f}")
        
        print("\nTiming (avg ms):")
        for component, times in self.performance_manager.processing_times.items():
            if times:
                avg_time = np.mean(times)
                emoji = "🔴" if avg_time > 30 else "🟡" if avg_time > 15 else "🟢"
                print(f"  {emoji} {component}: {avg_time:.1f}ms")
        
        if self.performance_manager.frame_times:
            avg_frame_time = np.mean(self.performance_manager.frame_times)
            target_fps = 1000 / avg_frame_time
            print(f"\nOverall FPS: {target_fps:.1f}")
    
    def _cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Smart Real-time Pose Estimator')
    
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')
    
    parser.add_argument('--target-fps', type=float, default=30.0,
                       help='Target FPS (default: 30.0)')
    
    parser.add_argument('--performance-mode', action='store_true',
                       help='Enable aggressive performance optimizations')
    
    return parser.parse_args()

if __name__ == "__main__":
    print("🧠 Advanced Smart Real-time Pose Estimator (XFeat)")
    print("=" * 60)
    print("Features:")
    print("  🎯 Adaptive detection frequency")
    print("  🔍 Smart viewpoint classification")
    print("  📊 Real-time performance monitoring")
    print("  🛡️ Robust failure recovery")
    print("  ⚡ XFeat + LighterGlue matching")
    print("=" * 60)
    
    args = parse_args()
    
    try:
        estimator = AdvancedSmartPoseEstimator(args)
        print("✅ System ready!")
        print("📹 Starting camera feed...")
        print("Controls: 'q' to quit, 'r' to reset")
        estimator.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        import traceback
        traceback.print_exc()

print("🏁 Program finished!")