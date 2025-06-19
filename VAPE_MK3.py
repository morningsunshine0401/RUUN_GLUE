"""
Smart Real-time Pose Estimator with Adaptive Tracking
- Detection + Tracking hybrid approach
- Adaptive viewpoint classification based on matching quality
- Intelligent state management for optimal performance


- Using XFeat + LighterGlue instead of SuperPoint + LightGlue


"""

import cv2
import numpy as np
import torch
import time
import argparse
import warnings
from collections import defaultdict
from enum import Enum

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*custom_fwd.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")

# Disable gradients globally for performance
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 Starting Smart Pose Estimator with XFeat...")
print("Loading required libraries...")

try:
    # Import models
    from ultralytics import YOLO
    print("✅ YOLO imported")
    
    import timm
    from torchvision import transforms
    print("✅ Vision models imported")
    
    # XFeat will be loaded via torch.hub
    print("✅ XFeat will be loaded via torch.hub")
    
    from PIL import Image
    print("✅ PIL imported")
    
    from scipy.spatial.distance import cdist
    print("✅ SciPy imported")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages:")
    print("pip install ultralytics timm torchvision pillow scipy")
    print("pip install kornia kornia-rs --no-deps  # Required for XFeat")
    exit(1)

print("🔄 All libraries loaded successfully!")

# System States
class TrackingState(Enum):
    INITIALIZING = "initializing"
    DETECTING = "detecting" 
    TRACKING = "tracking"
    LOST = "lost"
    RECLASSIFYING = "reclassifying"

# Performance profiler
class PerformanceProfiler:
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timers = {}
        
    def start_timer(self, name):
        self.current_timers[name] = time.time()
        
    def end_timer(self, name):
        if name in self.current_timers:
            elapsed = (time.time() - self.current_timers[name]) * 1000
            self.timings[name].append(elapsed)
            del self.current_timers[name]
            return elapsed
        return 0
    
    def print_stats(self, frame_idx):
        if frame_idx % 60 == 0 and frame_idx > 0:  # Less frequent
            print(f"\n=== FRAME {frame_idx} PERFORMANCE ===")
            for name, times in self.timings.items():
                if times:
                    recent = times[-20:]  # More samples
                    avg = sum(recent) / len(recent)
                    emoji = "🔴" if avg > 30 else "🟡" if avg > 15 else "🟢"
                    print(f"{emoji} {name:20} | {avg:6.1f}ms")

# Tracking quality monitor
class TrackingQualityMonitor:
    def __init__(self, 
                 low_match_threshold=8, 
                 consecutive_low_frames=3,
                 tracking_confidence_threshold=0.6):
        self.low_match_threshold = low_match_threshold
        self.consecutive_low_frames = consecutive_low_frames
        self.tracking_confidence_threshold = tracking_confidence_threshold
        
        self.match_history = []
        self.low_match_count = 0
        self.tracking_confidence = 1.0
        
    def update_matches(self, num_matches):
        self.match_history.append(num_matches)
        if len(self.match_history) > 10:
            self.match_history.pop(0)
            
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
        return np.mean(self.match_history) if self.match_history else 0

# Loosely Coupled Kalman Filter (same as before)
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
        self.Q = np.eye(self.n_states) * 1e-3
        
        # Measurement noise for [px, py, pz, qx, qy, qz, qw]
        self.R = np.eye(7) * 1e-4
    
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


class SmartPoseEstimator:
    def __init__(self, args):
        print("🔧 Initializing SmartPoseEstimator with XFeat...")
        self.args = args
        self.profiler = PerformanceProfiler()
        self.quality_monitor = TrackingQualityMonitor()
        self.frame_count = 0
        
        # System state
        self.state = TrackingState.INITIALIZING
        self.current_bbox = None
        self.current_viewpoint = None
        self.tracker = None
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")
        
        # Initialize models
        print("🔄 Loading AI models...")
        self._init_models()
        
        # Initialize camera
        print("🔄 Setting up camera...")
        self._init_camera()
        
        # Initialize Kalman filter
        print("🔄 Initializing Kalman filter...")
        self.kf = LooselyCoupledKalmanFilter()
        
        # Anchor data (4 viewpoints using same data for now)
        print("🔄 Loading anchor data...")
        self._init_anchor_data()
        
        print("✅ SmartPoseEstimator initialized!")
        
    def _init_models(self):
        """Initialize all models"""
        start_time = time.time()
        
        try:
            # YOLO model
            print("  📦 Loading YOLO...")
            self.yolo_model = YOLO("yolov8s.pt")
            if torch.cuda.is_available():
                self.yolo_model.to('cuda')
            print("  ✅ YOLO loaded")
            
            # Viewpoint classifier
            print("  📦 Loading viewpoint classifier...")
            self.vp_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=4)
            
            try:
                self.vp_model.load_state_dict(torch.load('mobilevit_viewpoint_twostage_final_2.pth', 
                                                        map_location=self.device))
                print("  ✅ Viewpoint model loaded")
            except FileNotFoundError:
                print("  ⚠️  Viewpoint model file not found, using random weights")
            
            self.vp_model.eval().to(self.device)
            
            # Transform for viewpoint classifier
            self.vp_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
            
            # XFeat model
            print("  📦 Loading XFeat...")
            self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', 
                                       pretrained=True, top_k=4096)
            print("  ✅ XFeat loaded")
            
            # Class names for viewpoint
            self.class_names = ['NE', 'NW', 'SE', 'SW']
            
            print(f"✅ All models loaded in {(time.time() - start_time)*1000:.1f}ms")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
        
    def _init_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(0)  # Use default camera
            
            if not self.cap.isOpened():
                print("❌ Cannot open camera 0, trying camera 1...")
                self.cap = cv2.VideoCapture(1)
                
            if not self.cap.isOpened():
                raise ValueError("Could not open any camera")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test capture
            ret, test_frame = self.cap.read()
            if not ret:
                raise ValueError("Cannot read from camera")
                
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✅ Camera initialized: {actual_width}x{actual_height}")
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            print("Creating dummy video source for testing...")
            self.cap = None  # Will use dummy frames
        
    def _init_anchor_data(self):
        """Initialize anchor data"""
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
        
        # 2D keypoints in anchor image
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
        
        # 3D keypoints
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
        """Extract XFeat features from anchor image once"""
        # Convert BGR to RGB for XFeat
        anchor_rgb = cv2.cvtColor(self.anchor_image, cv2.COLOR_BGR2RGB)
        
        # Extract features using XFeat
        self.anchor_features = self.xfeat.detectAndCompute(anchor_rgb, top_k=4096)[0]
        
        # Update with image resolution (required for XFeat)
        self.anchor_features.update({
            'image_size': (self.anchor_image.shape[1], self.anchor_image.shape[0])
        })
            
        print("✅ Anchor XFeat features extracted")
    
    def _get_frame(self):
        """Get frame from camera or generate dummy frame"""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                return frame
        
        # Generate dummy frame for testing
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Add some fake object in the center for testing
        center_x, center_y = 640, 360
        size = 100
        cv2.rectangle(frame, 
                     (center_x - size, center_y - size), 
                     (center_x + size, center_y + size), 
                     (0, 255, 0), -1)
        
        return frame
    
    def _yolo_detect(self, frame):
        """Run YOLO detection"""
        self.profiler.start_timer('yolo_detection')
        
        # Resize for YOLO (smaller for speed)
        yolo_size = (640, 640)  # Standard YOLO input
        yolo_frame = cv2.resize(frame, yolo_size)
        
        # Run YOLO
        results = self.yolo_model(yolo_frame[..., ::-1], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        self.profiler.end_timer('yolo_detection')
        
        if len(boxes) == 0:
            return None
            
        # Scale bounding box back to original size (center-based scaling)
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
        """Initialize OpenCV tracker"""
        self.profiler.start_timer('tracker_init')

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
        
        self.profiler.end_timer('tracker_init')
        return success
    
    def _track_object(self, frame):
        """Track object using OpenCV tracker"""
        if self.tracker is None:
            return None, 0.0
            
        self.profiler.start_timer('tracking')
        
        success, opencv_bbox = self.tracker.update(frame)
        
        self.profiler.end_timer('tracking')
        
        if not success:
            return None, 0.0
            
        # Convert back to our format (x1, y1, x2, y2)
        x, y, w, h = opencv_bbox
        bbox = (int(x), int(y), int(x+w), int(y+h))
        
        # Estimate confidence based on bbox size and position consistency
        confidence = self._estimate_tracking_confidence(bbox)
        
        return bbox, confidence
    
    def _estimate_tracking_confidence(self, bbox):
        """Estimate tracking confidence based on bbox properties"""
        if self.current_bbox is None:
            return 1.0
            
        # Compare with previous bbox
        x1, y1, x2, y2 = bbox
        px1, py1, px2, py2 = self.current_bbox
        
        # Size change ratio
        current_area = (x2-x1) * (y2-y1)
        prev_area = (px2-px1) * (py2-py1)
        size_ratio = min(current_area, prev_area) / max(current_area, prev_area)
        
        # Position change
        center_x, center_y = (x1+x2)/2, (y1+y2)/2
        prev_center_x, prev_center_y = (px1+px2)/2, (py1+py2)/2
        distance = np.sqrt((center_x-prev_center_x)**2 + (center_y-prev_center_y)**2)
        
        # Confidence based on consistency
        confidence = size_ratio * max(0, 1 - distance/100)  # Penalize large movements
        
        return confidence
    
    def _classify_viewpoint(self, frame, bbox):
        """Classify viewpoint of detected object"""
        self.profiler.start_timer('viewpoint_classification')
        
        x1, y1, x2, y2 = bbox
        cropped = frame[y1:y2, x1:x2]
        
        if cropped.size == 0:
            self.profiler.end_timer('viewpoint_classification')
            return 'NE'
        
        # Resize for classification
        crop_resized = cv2.resize(cropped, (128, 128))
        
        # Convert to PIL and apply transforms
        img_pil = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
        input_tensor = self.vp_transform(img_pil).unsqueeze(0).to(self.device)
        
        # Predict viewpoint
        with torch.no_grad():
            logits = self.vp_model(input_tensor)
            pred = torch.argmax(logits, dim=1).item()
            viewpoint = self.class_names[pred]
        
        self.profiler.end_timer('viewpoint_classification')
        return viewpoint
    
    def _estimate_pose(self, cropped_frame, viewpoint):
        """Estimate pose using XFeat + LighterGlue + EPNP"""
        self.profiler.start_timer('xfeat_extraction')
        
        # Convert BGR to RGB for XFeat
        frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        
        # Extract features using XFeat
        frame_features = self.xfeat.detectAndCompute(frame_rgb, top_k=4096)[0]
        
        # Update with image resolution (required for XFeat)
        frame_features.update({
            'image_size': (cropped_frame.shape[1], cropped_frame.shape[0])
        })
        
        self.profiler.end_timer('xfeat_extraction')
        
        # Match with anchor using LighterGlue
        self.profiler.start_timer('lighterglue_matching')
        
        anchor_data = self.viewpoint_anchors[viewpoint]
        
        # Match using XFeat's LighterGlue
        mkpts_0, mkpts_1,_ = self.xfeat.match_lighterglue(
            anchor_data['features'], 
            frame_features
        )
        
        self.profiler.end_timer('lighterglue_matching')
        
        num_matches = len(mkpts_0)
        self.quality_monitor.update_matches(num_matches)
        
        if num_matches < 6:  # Need minimum matches
            return None, None, num_matches
        
        # Map to 3D points
        anchor_2d = anchor_data['2d']
        anchor_3d = anchor_data['3d']
        
        # Find closest 2D points in anchor
        distances = cdist(mkpts_0, anchor_2d)
        closest_indices = np.argmin(distances, axis=1)
        valid_mask = np.min(distances, axis=1) < 5.0  # threshold
        
        if np.sum(valid_mask) < 6:
            return None, None, num_matches
            
        # Get valid correspondences
        points_3d = anchor_3d[closest_indices[valid_mask]]
        points_2d = mkpts_1[valid_mask]
        
        # Solve PnP
        self.profiler.start_timer('pnp_solving')
        
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
        
        self.profiler.end_timer('pnp_solving')
        
        if not success or len(inliers) < 4:
            return None, None, num_matches
            
        # Convert to position and quaternion
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
        
        return position, quaternion, num_matches
    
    def _get_camera_intrinsics(self):
        """Get camera intrinsic parameters"""
        # Use your calibrated parameters
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
    
    def _draw_axes(self, frame, position, quaternion):
        """Draw coordinate axes on frame"""
        if position is None or quaternion is None:
            return frame
        
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
        axis_length = 0.1  # 10cm
        axis_points = np.float32([
            [0, 0, 0],           # Origin
            [axis_length, 0, 0], # X-axis
            [0, axis_length, 0], # Y-axis
            [0, 0, axis_length]  # Z-axis
        ])
        
        # Project to image
        K, dist_coeffs = self._get_camera_intrinsics()
        projected, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist_coeffs)
        projected = projected.reshape(-1, 2).astype(int)
        
        # Draw axes
        origin = tuple(projected[0])
        frame = cv2.line(frame, origin, tuple(projected[1]), (0, 0, 255), 3)  # X - Red
        frame = cv2.line(frame, origin, tuple(projected[2]), (0, 255, 0), 3)  # Y - Green
        frame = cv2.line(frame, origin, tuple(projected[3]), (255, 0, 0), 3)  # Z - Blue
        
        return frame
    
    def run(self):
        """Smart main processing loop with state machine"""
        print("🚀 Starting smart pose estimation with XFeat")
        
        cv2.namedWindow('Smart Pose Estimation (XFeat)', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                self.frame_count += 1
                frame = self._get_frame()
                self.profiler.start_timer('total_frame')
                
                # State machine processing
                position, quaternion = None, None
                
                if self.state == TrackingState.INITIALIZING:
                    print("🔍 Initializing - Detecting first object...")
                    bbox = self._yolo_detect(frame)
                    if bbox is not None:
                        self.current_bbox = bbox
                        self.current_viewpoint = self._classify_viewpoint(frame, bbox)
                        if self._init_tracker(frame, bbox):
                            self.state = TrackingState.TRACKING
                            print(f"✅ Initialized - Tracking {self.current_viewpoint} viewpoint")
                        else:
                            self.state = TrackingState.DETECTING
                    
                    
                elif self.state == TrackingState.DETECTING:
                    print("🔍 Detecting object...")
                    bbox = self._yolo_detect(frame)
                    if bbox is not None:
                        self.current_bbox = bbox
                        self.current_viewpoint = self._classify_viewpoint(frame, bbox)
                        if self._init_tracker(frame, bbox):
                            self.state = TrackingState.TRACKING
                            print(f"✅ Object found - Tracking {self.current_viewpoint} viewpoint")
                    
                elif self.state == TrackingState.TRACKING:
                    bbox, confidence = self._track_object(frame)
                    self.quality_monitor.update_tracking_confidence(confidence)
                    
                    if bbox is not None and confidence > 0.6:
                        self.current_bbox = bbox
                        
                        # Check if we need to reclassify viewpoint
                        if self.quality_monitor.should_reclassify_viewpoint():
                            print("🔄 Low matches detected - Reclassifying viewpoint...")
                            self.state = TrackingState.RECLASSIFYING
                            
                    else:
                        print("❌ Tracking lost - Switching to detection")
                        self.state = TrackingState.LOST
                        self.tracker = None
                
                elif self.state == TrackingState.RECLASSIFYING:
                    if self.current_bbox is not None:
                        new_viewpoint = self._classify_viewpoint(frame, self.current_bbox)
                        if new_viewpoint != self.current_viewpoint:
                            print(f"🔄 Viewpoint changed: {self.current_viewpoint} → {new_viewpoint}")
                            self.current_viewpoint = new_viewpoint
                        self.quality_monitor.low_match_count = 0  # Reset
                        self.state = TrackingState.TRACKING
                
                elif self.state == TrackingState.LOST:
                    print("🔍 Object lost - Redetecting...")
                    bbox = self._yolo_detect(frame)
                    if bbox is not None:
                        self.current_bbox = bbox
                        self.current_viewpoint = self._classify_viewpoint(frame, bbox)
                        if self._init_tracker(frame, bbox):
                            self.state = TrackingState.TRACKING
                            print(f"✅ Object reacquired - Tracking {self.current_viewpoint}")
                        else:
                            self.state = TrackingState.DETECTING
                
                # Pose estimation if we have a bbox
                if self.current_bbox is not None and self.current_viewpoint is not None:
                    x1, y1, x2, y2 = self.current_bbox
                    cropped = frame[y1:y2, x1:x2]
                    
                    if cropped.size > 0:
                        pose_result = self._estimate_pose(cropped, self.current_viewpoint)
                        if pose_result[0] is not None:
                            position, quaternion, num_matches = pose_result
                            position, quaternion = self.kf.update(position, quaternion)
                        else:
                            # Use Kalman prediction
                            pred_result = self.kf.predict()
                            if pred_result is not None:
                                position, quaternion = pred_result
                
                # Visualization
                vis_frame = frame.copy()
                
                # Draw current state
                state_color = {
                    TrackingState.INITIALIZING: (0, 0, 255),    # Red
                    TrackingState.DETECTING: (0, 100, 255),     # Orange
                    TrackingState.TRACKING: (0, 255, 0),        # Green
                    TrackingState.RECLASSIFYING: (255, 255, 0), # Cyan
                    TrackingState.LOST: (0, 0, 200)             # Dark Red
                }
                
                cv2.putText(vis_frame, f'State: {self.state.value.upper()}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color[self.state], 2)
                
                # Add XFeat indicator
                cv2.putText(vis_frame, 'XFeat + LighterGlue', (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw bounding box if available
                if self.current_bbox is not None:
                    x1, y1, x2, y2 = self.current_bbox
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), state_color[self.state], 2)
                    
                    if self.current_viewpoint:
                        cv2.putText(vis_frame, f'VP: {self.current_viewpoint}', (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color[self.state], 2)
                    
                    # Draw coordinate axes on cropped region
                    if position is not None and quaternion is not None:
                        cropped = frame[y1:y2, x1:x2]
                        if cropped.size > 0:
                            cropped_with_axes = self._draw_axes(cropped.copy(), position, quaternion)
                            vis_frame[y1:y2, x1:x2] = cropped_with_axes
                
                # Performance info
                cv2.putText(vis_frame, f'Frame: {self.frame_count}', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                avg_matches = self.quality_monitor.get_average_matches()
                cv2.putText(vis_frame, f'Avg Matches: {avg_matches:.1f}', (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                total_time = self.profiler.end_timer('total_frame')
                if total_time > 0:
                    fps = 1000 / total_time
                    cv2.putText(vis_frame, f'FPS: {fps:.1f}', (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Smart Pose Estimation (XFeat)', vis_frame)
                
                # Print performance stats
                self.profiler.print_stats(self.frame_count)
                
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
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

def parse_args():
    """Parse command line arguments"""
    print("🔄 Parsing arguments...")
    parser = argparse.ArgumentParser(description='Smart Real-time Pose Estimator with XFeat')
    
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    print(f"✅ Arguments parsed: device={args.device}")
    return args

if __name__ == "__main__":
    print("🧠 Smart Real-time Pose Estimator (XFeat)")
    print("=" * 50)
    
    args = parse_args()
    
    try:
        print("🚀 Starting initialization...")
        estimator = SmartPoseEstimator(args)
        print("✅ Estimator ready!")
        print("📹 Starting smart camera feed...")
        print("Press 'q' to quit")
        estimator.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        import traceback
        traceback.print_exc()

print("🏁 Program finished!")