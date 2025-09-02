#!/usr/bin/env python3
"""
RTMPose-based Aircraft Pose Estimator - Research Comparison Version
Based on VAPE MK43 architecture but using global semantic keypoints instead of local features

Comparison: Global Semantic Features (RTMPose) vs Local Features (SuperPoint+LightGlue)
- Same YOLO detection pipeline
- Same PnP + RANSAC pose estimation
- Same multi-threaded architecture
- Same JSON export format
- Different feature extraction method
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

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 Starting RTMPose-based Aircraft Pose Estimator...")

# Import required libraries
try:
    from ultralytics import YOLO
    from mmpose.apis import init_model, inference_topdown
    import mmcv
    print("✅ All libraries loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Enhanced State Machine (same as VAPE)
class TrackingState(Enum):
    INITIALIZING = "initializing"
    DETECTING = "detecting" 
    TRACKING = "tracking"
    LOST = "lost"

@dataclass
class TrackingContext:
    """Context information for current tracking state"""
    frame_id: int = 0
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    num_keypoints: int = 0
    consecutive_failures: int = 0
    last_detection_frame: int = 0

@dataclass
class ProcessingResult:
    """Result from processing thread"""
    frame: np.ndarray
    frame_id: int
    timestamp: float
    position: Optional[np.ndarray] = None
    quaternion: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    num_keypoints: int = 0
    keypoint_confidences: Optional[List[float]] = None
    processing_time: float = 0.0
    pose_data: Optional[Dict] = None

def read_image_index_csv(csv_path):
    """Reads a CSV with columns: Index, Timestamp, Filename"""
    entries = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row['Index'])
            tstamp = float(row['Timestamp'])
            fname = row['Filename']
            entries.append({
                'index': frame_idx,
                'timestamp': tstamp,
                'filename': fname
            })
    return entries

def create_unique_filename(directory, base_filename):
    """Create a unique filename by adding a counter if the file already exists"""
    if not directory:
        directory = "."
    
    base_path = os.path.join(directory, base_filename)
    if not os.path.exists(base_path):
        return base_path
    
    name, ext = os.path.splitext(base_filename)
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_path = os.path.join(directory, new_filename)
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def convert_to_json_serializable(obj):
    """Convert numpy and other non-JSON serializable types to JSON serializable types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def letterbox_black(image, new_size=(512, 384)):
    """
    Resize image with letterboxing to maintain aspect ratio
    Returns: letterboxed image, scale factor, pad_left, pad_top
    """
    h0, w0 = image.shape[:2]
    w_new, h_new = new_size
    
    # Compute scale factor
    r = min(w_new / w0, h_new / h0)
    
    # New unpadded size
    w_unpad = int(round(w0 * r))
    h_unpad = int(round(h0 * r))
    
    # Resize original to (w_unpad, h_unpad)
    resized = cv2.resize(image, (w_unpad, h_unpad), interpolation=cv2.INTER_LINEAR)
    
    # Compute black padding
    pad_w = w_new - w_unpad
    pad_h = h_new - h_unpad
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    # Pad with black
    letterboxed = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    
    return letterboxed, r, pad_left, pad_top

def unletterbox_keypoints(keypoints, r, pad_left, pad_top):
    """Map keypoints from letterboxed coords back to original image coords"""
    if keypoints is None or len(keypoints) == 0:
        return keypoints
    
    kpts_orig = keypoints.copy().astype(np.float32)
    kpts_orig[:, 0] = (kpts_orig[:, 0] - pad_left) / r
    kpts_orig[:, 1] = (kpts_orig[:, 1] - pad_top) / r
    
    return kpts_orig

# Same thread-safe classes as VAPE
class ThreadSafeFrameBuffer:
    """Thread-safe buffer that always stores only the latest frame"""
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
    """Thread-safe performance monitoring"""
    def __init__(self):
        self.lock = threading.Lock()
        self.timings = defaultdict(lambda: deque(maxlen=30))
        self.processing_fps_history = deque(maxlen=30)
        self.camera_fps_history = deque(maxlen=30)
        
    def add_timing(self, name: str, duration: float):
        with self.lock:
            self.timings[name].append(duration)
            
    def add_processing_fps(self, fps: float):
        with self.lock:
            self.processing_fps_history.append(fps)
            
    def add_camera_fps(self, fps: float):
        with self.lock:
            self.camera_fps_history.append(fps)
        
    def get_average(self, name: str) -> float:
        with self.lock:
            if name in self.timings and self.timings[name]:
                return np.mean(list(self.timings[name]))
        return 0.0
        
    def get_processing_fps(self) -> float:
        with self.lock:
            if self.processing_fps_history:
                return np.mean(list(self.processing_fps_history)[-10:])
        return 0.0
        
    def get_camera_fps(self) -> float:
        with self.lock:
            if self.camera_fps_history:
                return np.mean(list(self.camera_fps_history)[-10:])
        return 0.0

class RTMPosePoseEstimator:
    """RTMPose-based pose estimator for research comparison"""
    
    def __init__(self, args):
        print("🔧 Initializing RTMPose-based Aircraft Pose Estimator...")
        self.args = args
        
        # Thread control
        self.running = False
        self.threads = []
        
        # Frame buffer
        self.frame_buffer = ThreadSafeFrameBuffer()
        self.result_queue = queue.Queue(maxsize=1)
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
        
        # System state
        self.state = TrackingState.INITIALIZING
        self.context = TrackingContext()
        self.tracker = None
        self.state_lock = threading.Lock()
        
        # JSON export data
        self.all_poses = []
        self.poses_lock = threading.Lock()
        
        # Batch mode setup
        self.batch_mode = hasattr(args, 'image_dir') and args.image_dir is not None
        self.image_entries = []
        self.batch_complete = False
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")
        
        # Camera setup
        self.camera_width = 1280
        self.camera_height = 720
        
        # Initialize components
        self._init_models()
        if self.batch_mode:
            self._init_batch_processing()
        else:
            self._init_camera()
        self._init_keypoint_3d_coordinates()
        
        print("✅ RTMPose-based Aircraft Pose Estimator initialized!")
        
    def _init_batch_processing(self):
        """Initialize batch processing from CSV file"""
        if hasattr(self.args, 'csv_file') and self.args.csv_file:
            self.image_entries = read_image_index_csv(self.args.csv_file)
            print(f"✅ Loaded {len(self.image_entries)} image entries from CSV")
        else:
            print("⚠️ No CSV file provided for batch processing")
            self.image_entries = []



            
    def _init_models(self):
        """Initialize AI models"""
        try:
            # YOLO (same as VAPE)
            print("  📦 Loading YOLO...")
            #self.yolo_model = YOLO("yolov8s.pt")
            self.yolo_model = YOLO("best.pt")
            if self.device == 'cuda':
                self.yolo_model.to('cuda')
            print("  ✅ YOLO loaded")
            
            # RTMPose model (new component)
            print("  📦 Loading RTMPose model...")
            self.rtmpose_config = getattr(self.args, 'rtmpose_config', 'configs/my_aircraft/rtmpose.py')
            self.rtmpose_checkpoint = getattr(self.args, 'rtmpose_checkpoint', 
                'work_dirs/rtmpose-l_aircraft-384x288_20250901/epoch_580.pth')
                #'work_dirs/rtmpose-l_aircraft-384x288_20250808/best_coco_AP_epoch_80.pth')
            
            self.rtmpose_model = init_model(
                self.rtmpose_config, 
                self.rtmpose_checkpoint, 
                device=self.device
            )
            print("  ✅ RTMPose model loaded")
            
            # Keypoint names (global semantic features)
            self.keypoint_names = ['nose', 'left_wing', 'left_tail', 'tail', 'right_tail', 'right_wing']
            print(f"  ✅ Using {len(self.keypoint_names)} semantic keypoints")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
            
    def _init_camera(self):
        """Initialize camera (same as VAPE)"""
        try:
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                print("❌ Cannot open camera 0, trying camera 1...")
                self.cap = cv2.VideoCapture(1)
                
            if not self.cap.isOpened():
                print("❌ Cannot open any camera")
                self.cap = None
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test capture
            ret, test_frame = self.cap.read()
            if not ret:
                print("❌ Cannot read from camera")
                self.cap.release()
                self.cap = None
                return
                
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✅ Camera initialized: {actual_width}x{actual_height}")
            
            self.camera_width = actual_width
            self.camera_height = actual_height
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            self.cap = None
            
    def _init_keypoint_3d_coordinates(self):
        """Initialize 3D coordinates for semantic keypoints (research data)"""
        # 3D coordinates based on actual aircraft orientation in images
        # Aircraft is oriented sideways: nose left, tail right, wings up/down
        # Coordinate system: X=nose-to-tail, Y=left-to-right-wing, Z=up-down
        self.keypoint_3d_coords = np.array([
            # nose, left_wing, left_tail, tail, right_tail, right_wing
            [-0.200, 0.000, 0.000],    # nose (front center, left in image)
            [-0.000, -0.025, -0.240],   # left_wing (mid-body, left wing, slightly up)
            [0.230, -0.000, -0.113],   # left_tail (rear, left side)
            [0.243, -0.104, 0.000],     # tail (rear center, right in image, up)
            [0.230, -0.000, 0.113],    # right_tail (rear, right side)
            [-0.000, -0.025, 0.240],    # right_wing (mid-body, right wing, slightly up)
        ], dtype=np.float32)
        
        print(f"✅ Initialized 3D coordinates for {len(self.keypoint_3d_coords)} semantic keypoints")
        print("📋 Keypoint 3D coordinates (aircraft frame - sideways orientation):")
        print("    X: nose(-) → tail(+), Y: left_wing(-) → right_wing(+), Z: down(-) → up(+)")
        for i, (name, coord) in enumerate(zip(self.keypoint_names, self.keypoint_3d_coords)):
            print(f"  {name:12}: [{coord[0]:7.3f}, {coord[1]:7.3f}, {coord[2]:7.3f}]")
    
    def _yolo_detect(self, frame):
        """Run YOLO detection (same as VAPE)"""
        t_start = time.perf_counter()
        
        yolo_size = (640, 640)
        yolo_frame = cv2.resize(frame, yolo_size)
        
        results = self.yolo_model(yolo_frame[..., ::-1], verbose=False, conf=0.5)
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
        else:
            boxes = np.array([])
        
        duration = (time.perf_counter() - t_start) * 1000
        self.perf_monitor.add_timing('yolo_detection', duration)
        
        if len(boxes) == 0:
            return None
            
        # Scale bounding box back to original size
        bbox = boxes[0]
        scale_x = frame.shape[1] / yolo_size[0]
        scale_y = frame.shape[0] / yolo_size[1]
        
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
        """Initialize OpenCV tracker with robust fallback"""
        t_start = time.perf_counter()

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w < 16 or h < 16:
            print(f"⚠️ Tracker init skipped: bbox too small ({w}×{h})")
            return False

        # Try multiple tracker options in order of preference
        tracker_constructors = [
            # Modern OpenCV
            lambda: getattr(cv2, 'TrackerCSRT_create', None),
            # Legacy OpenCV
            lambda: getattr(cv2.legacy, 'TrackerCSRT_create', None) if hasattr(cv2, 'legacy') else None,
            # Older OpenCV versions
            lambda: getattr(cv2, 'TrackerKCF_create', None),
            lambda: getattr(cv2.legacy, 'TrackerKCF_create', None) if hasattr(cv2, 'legacy') else None,
            # Even older versions
            lambda: getattr(cv2, 'createTrackerCSRT', None),
            lambda: getattr(cv2, 'createTrackerKCF', None),
        ]
        
        self.tracker = None
        for constructor_func in tracker_constructors:
            try:
                constructor = constructor_func()
                if constructor is not None:
                    self.tracker = constructor()
                    print(f"  ✅ Using tracker: {constructor.__name__}")
                    break
            except Exception as e:
                continue
        
        if self.tracker is None:
            print("  ⚠️ No OpenCV tracker available, using simple bbox tracking")
            # Fallback: store bbox and use simple tracking
            self._simple_tracker_bbox = bbox
            duration = (time.perf_counter() - t_start) * 1000
            self.perf_monitor.add_timing('tracker_init', duration)
            return True
        
        success = self.tracker.init(frame, (x1, y1, w, h))
        
        duration = (time.perf_counter() - t_start) * 1000
        self.perf_monitor.add_timing('tracker_init', duration)
        
        return success
    
    def _track_object(self, frame):
        """Track object using OpenCV tracker with fallback"""
        if self.tracker is None and not hasattr(self, '_simple_tracker_bbox'):
            if not hasattr(self, '_tracker_warning_shown'):
                print("  ⚠️ No tracker available!")
                self._tracker_warning_shown = True
            return None, 0.0
            
        t_start = time.perf_counter()
        
        if self.tracker is not None:
            # Use OpenCV tracker
            success, opencv_bbox = self.tracker.update(frame)
            
            if not success:
                duration = (time.perf_counter() - t_start) * 1000
                self.perf_monitor.add_timing('tracking', duration)
                return None, 0.0
                
            x, y, w, h = opencv_bbox
            bbox = (int(x), int(y), int(x+w), int(y+h))
        else:
            # Use simple bbox tracking (fallback)
            if hasattr(self, '_simple_tracker_bbox'):
                bbox = self._simple_tracker_bbox
                success = True
            else:
                success = False
        
        duration = (time.perf_counter() - t_start) * 1000
        self.perf_monitor.add_timing('tracking', duration)
        
        if not success:
            return None, 0.0
            
        confidence = self._estimate_tracking_confidence(bbox)
        
        return bbox, confidence
    
    def _estimate_tracking_confidence(self, bbox):
        """Estimate tracking confidence (same as VAPE)"""
        with self.state_lock:
            if self.context.bbox is None:
                return 1.0
                
            x1, y1, x2, y2 = bbox
            px1, py1, px2, py2 = self.context.bbox
            
            current_area = (x2-x1) * (y2-y1)
            prev_area = (px2-px1) * (py2-py1)
            if prev_area > 0:
                size_ratio = min(current_area, prev_area) / max(current_area, prev_area)
            else:
                size_ratio = 0.5
            
            center_x, center_y = (x1+x2)/2, (y1+y2)/2
            prev_center_x, prev_center_y = (px1+px2)/2, (py1+py2)/2
            distance = np.sqrt((center_x-prev_center_x)**2 + (center_y-prev_center_y)**2)
            
            confidence = size_ratio * max(0, 1 - distance/100)
            
        return confidence
    
    def _detect_rtmpose_keypoints(self, cropped_frame, bbox, frame_id, frame_info=None):
        """
        Detect aircraft keypoints using RTMPose (NEW METHOD)
        Returns: position, quaternion, num_keypoints, pose_data
        """
        t_start = time.perf_counter()
        
        # Apply letterboxing for RTMPose input
        letterboxed_crop, scale_factor, pad_left, pad_top = letterbox_black(
            cropped_frame, new_size=(512, 384)
        )
        
        # Create bbox for letterboxed image
        letterbox_bbox = np.array([[0, 0, 512, 384]])
        
        # Run RTMPose inference
        try:
            results = inference_topdown(self.rtmpose_model, letterboxed_crop, letterbox_bbox)
        except Exception as e:
            print(f"  ❌ RTMPose inference failed: {e}")
            duration = (time.perf_counter() - t_start) * 1000
            self.perf_monitor.add_timing('rtmpose_detection', duration)
            return None, None, 0, None
        
        if len(results) == 0 or len(results[0].pred_instances.keypoints) == 0:
            print(f"  ❌ No keypoints detected by RTMPose")
            duration = (time.perf_counter() - t_start) * 1000
            self.perf_monitor.add_timing('rtmpose_detection', duration)
            return None, None, 0, None
        
        # Get keypoints and scores
        letterbox_keypoints = results[0].pred_instances.keypoints[0]  # (N, 2)
        scores = results[0].pred_instances.keypoint_scores[0]  # (N,)
        
        # Map keypoints back to cropped image coordinates
        crop_keypoints = unletterbox_keypoints(letterbox_keypoints, scale_factor, pad_left, pad_top)
        
        # Adjust to full image coordinates
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            crop_offset = np.array([x1, y1])
            full_image_keypoints = crop_keypoints + crop_offset
        else:
            full_image_keypoints = crop_keypoints
        
        # Filter keypoints by confidence threshold
        confidence_threshold = 0.1  # Lowered threshold for more keypoints
        valid_mask = scores > confidence_threshold
        valid_keypoints_2d = full_image_keypoints[valid_mask]
        valid_keypoints_3d = self.keypoint_3d_coords[valid_mask]
        valid_scores = scores[valid_mask]
        valid_names = [self.keypoint_names[i] for i in range(len(self.keypoint_names)) if valid_mask[i]]
        
        num_valid_keypoints = len(valid_keypoints_2d)
        
        print(f"  📍 RTMPose detected {num_valid_keypoints}/{len(self.keypoint_names)} valid keypoints")
        
        # Debug: print individual keypoint info
        for i, (name, score, kpt_2d, kpt_3d) in enumerate(zip(valid_names, valid_scores, valid_keypoints_2d, valid_keypoints_3d)):
            print(f"    {name:12}: conf={score:.3f}, 2D=[{kpt_2d[0]:6.1f},{kpt_2d[1]:6.1f}], 3D=[{kpt_3d[0]:6.3f},{kpt_3d[1]:6.3f},{kpt_3d[2]:6.3f}]")
        
        if num_valid_keypoints < 4:  # Need minimum for PnP
            print(f"  ❌ Not enough valid keypoints ({num_valid_keypoints} < 4)")
            duration = (time.perf_counter() - t_start) * 1000
            self.perf_monitor.add_timing('rtmpose_detection', duration)
            
            pose_data = {
                'frame': int(frame_id),
                'pose_estimation_failed': True,
                'num_keypoints': int(num_valid_keypoints),
                'error_reason': 'insufficient_keypoints',
                'processing_time_ms': float(duration),
                'method': 'rtmpose_global_features',
                'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] if bbox else None
            }
            
            if frame_info:
                pose_data.update(convert_to_json_serializable(frame_info))
                
            return None, None, num_valid_keypoints, pose_data
        
        # Debug: Check data shapes and ranges
        print(f"  🔍 Debug - 3D points shape: {valid_keypoints_3d.shape}, 2D points shape: {valid_keypoints_2d.shape}")
        print(f"  🔍 Debug - 2D range: X=[{valid_keypoints_2d[:, 0].min():.1f}, {valid_keypoints_2d[:, 0].max():.1f}], Y=[{valid_keypoints_2d[:, 1].min():.1f}, {valid_keypoints_2d[:, 1].max():.1f}]")
        print(f"  🔍 Debug - 3D range: X=[{valid_keypoints_3d[:, 0].min():.3f}, {valid_keypoints_3d[:, 0].max():.3f}], Y=[{valid_keypoints_3d[:, 1].min():.3f}, {valid_keypoints_3d[:, 1].max():.3f}], Z=[{valid_keypoints_3d[:, 2].min():.3f}, {valid_keypoints_3d[:, 2].max():.3f}]")
        
        # Solve PnP with RANSAC (same as VAPE)
        K, dist_coeffs = self._get_camera_intrinsics()
        
        print(f"  🔍 Camera matrix K:\n{K}")
        
        # Try multiple PnP methods for robustness
        pnp_methods = [
            (cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"),
            (cv2.SOLVEPNP_EPNP, "EPNP"), 
            (cv2.SOLVEPNP_P3P, "P3P"),
            (cv2.SOLVEPNP_DLS, "DLS")
        ]
        
        best_inliers = 0
        best_result = None
        
        for method, method_name in pnp_methods:
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=valid_keypoints_3d.reshape(-1, 1, 3),
                    imagePoints=valid_keypoints_2d.reshape(-1, 1, 2),
                    cameraMatrix=K,
                    distCoeffs=dist_coeffs,
                    reprojectionError=15.0,  # Even more tolerance
                    confidence=0.90,  # Lower confidence for more flexibility
                    iterationsCount=3000,  # More iterations
                    flags=method
                )
                
                num_inliers = len(inliers) if inliers is not None else 0
                print(f"  🔍 {method_name}: success={success}, inliers={num_inliers}")
                
                if success and num_inliers > best_inliers:
                    best_inliers = num_inliers
                    best_result = (success, rvec, tvec, inliers, method_name)
                    
            except Exception as e:
                print(f"  ⚠️ {method_name} failed: {e}")
                continue
        
        if best_result is not None:
            success, rvec, tvec, inliers, method_name = best_result
            print(f"  ✅ Using best method: {method_name} with {best_inliers} inliers")
        else:
            print(f"  ❌ All PnP methods failed")
            success = False
            rvec = tvec = inliers = None
        
        # VVS refinement (same as VAPE)
        if success and inliers is not None and len(inliers) > 3:
            inlier_3d = valid_keypoints_3d[inliers.flatten()].reshape(-1, 1, 3)
            inlier_2d = valid_keypoints_2d[inliers.flatten()].reshape(-1, 1, 2)
            
            rvec, tvec = cv2.solvePnPRefineVVS(
                objectPoints=inlier_3d,
                imagePoints=inlier_2d,
                cameraMatrix=K,
                distCoeffs=dist_coeffs,
                rvec=rvec,
                tvec=tvec
            )
        
        print(f"  🎯 PnP success: {success}, inliers: {len(inliers) if inliers is not None else 0}")
        
        if not success or inliers is None or len(inliers) < 3:
            print(f"  ❌ PnP failed")
            duration = (time.perf_counter() - t_start) * 1000
            self.perf_monitor.add_timing('rtmpose_detection', duration)
            
            pose_data = {
                'frame': int(frame_id),
                'pose_estimation_failed': True,
                'num_keypoints': int(num_valid_keypoints),
                'pnp_success': bool(success),
                'num_inliers': int(len(inliers) if inliers is not None else 0),
                'error_reason': 'pnp_failed',
                'processing_time_ms': float(duration),
                'method': 'rtmpose_global_features',
                'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] if bbox else None
            }
            
            if frame_info:
                pose_data.update(convert_to_json_serializable(frame_info))
                
            return None, None, num_valid_keypoints, pose_data
        
        # Convert to position and quaternion (same as VAPE)
        R, _ = cv2.Rodrigues(rvec)
        position = tvec.flatten()
        quaternion = self._rotation_matrix_to_quaternion(R)
        
        duration = (time.perf_counter() - t_start) * 1000
        self.perf_monitor.add_timing('rtmpose_detection', duration)
        
        # Compute reprojection errors
        projected_points, _ = cv2.projectPoints(
            valid_keypoints_3d[inliers.flatten()].reshape(-1, 1, 3), rvec, tvec, K, dist_coeffs
        )
        reprojection_errors = np.linalg.norm(
            valid_keypoints_2d[inliers.flatten()].reshape(-1, 1, 2) - projected_points, axis=2
        ).flatten()
        mean_reprojection_error = np.mean(reprojection_errors)
        
        # Create comprehensive pose data for JSON export
        pose_data = {
            'frame': int(frame_id),
            'pose_estimation_failed': False,
            'method': 'rtmpose_global_features',  # Important for comparison
            'position': position.tolist(),
            'quaternion': quaternion.tolist(),
            'rotation_matrix': R.tolist(),
            'translation_vector': position.tolist(),
            'rotation_vector': rvec.flatten().tolist(),
            'num_keypoints': int(num_valid_keypoints),
            'num_inliers': int(len(inliers)),
            'inlier_ratio': float(len(inliers) / num_valid_keypoints),
            'reprojection_errors': reprojection_errors.tolist(),
            'mean_reprojection_error': float(mean_reprojection_error),
            'processing_time_ms': float(duration),
            'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] if bbox else None,
            'detected_keypoints': {
                'names': valid_names,
                'confidences': valid_scores[inliers.flatten()].tolist(),
                '2d_coordinates': valid_keypoints_2d[inliers.flatten()].tolist(),
                '3d_coordinates': valid_keypoints_3d[inliers.flatten()].tolist()
            },
            'camera_intrinsics': K.tolist()
        }
        
        if frame_info:
            pose_data.update(convert_to_json_serializable(frame_info))
        
        print(f"  ✅ RTMPose pose estimation successful! Duration: {duration:.1f}ms")
        
        return position, quaternion, num_valid_keypoints, pose_data
    
    def _rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion [x, y, z, w] (same as VAPE)"""
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
    
    def _get_camera_intrinsics(self):
        """Get camera intrinsic parameters (same as VAPE)"""
        fx = 1460.10150
        fy = 1456.48915
        cx = 604.85462
        cy = 328.64800
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = None
        
        return K, dist_coeffs
    
    # Camera and processing threads (same structure as VAPE)
    def camera_thread(self):
        """Thread 1: Capture frames"""
        print("📹 Camera thread started")
        
        if self.batch_mode:
            self._batch_frame_loader()
        else:
            self._camera_frame_capture()
            
        print("📹 Camera thread stopped")
        
    def _batch_frame_loader(self):
        """Load frames from image files in batch mode"""
        frame_count = 0
        
        for entry in self.image_entries:
            if not self.running:
                break
                
            frame_idx = entry['index']
            tstamp = entry['timestamp']
            img_name = entry['filename']
            
            image_path = os.path.join(self.args.image_dir, img_name)
            if not os.path.exists(image_path):
                print(f"⚠️ Image not found: {image_path}")
                continue
                
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"⚠️ Failed to load image: {image_path}")
                continue
            
            frame_info = {
                'original_frame_id': frame_idx,
                'timestamp': tstamp,
                'image_file': img_name
            }
            
            buffer_frame_id = self.frame_buffer.update(frame)
            
            with self.poses_lock:
                if not hasattr(self, 'current_frame_info'):
                    self.current_frame_info = {}
                self.current_frame_info[buffer_frame_id] = frame_info
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"📹 Loaded {frame_count} frames")
                
            time.sleep(0.033)  # ~30 FPS
        
        self.batch_complete = True
        print(f"📹 Batch complete: loaded {frame_count} total frames")
        time.sleep(2.0)
        print("📹 Stopping all threads...")
        self.running = False
            
    def _camera_frame_capture(self):
        """Capture frames from camera in real-time mode"""
        frame_count = 0
        last_fps_time = time.perf_counter()
        fps_frames = 0
        
        while self.running:
            if self.cap is None:
                time.sleep(0.1)
                continue
                
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.frame_buffer.update(frame)
                
                frame_count += 1
                fps_frames += 1
                
                current_time = time.perf_counter()
                if current_time - last_fps_time > 1.0:
                    fps = fps_frames / (current_time - last_fps_time)
                    self.perf_monitor.add_camera_fps(fps)
                    fps_frames = 0
                    last_fps_time = current_time
                    
                    if frame_count % 300 == 0:
                        print(f"📹 Camera FPS: {fps:.1f}")
            else:
                print("⚠️ Camera read failed")
                time.sleep(0.1)
    
    def processing_thread(self):
        """Thread 2: Process latest available frame"""
        print("⚙️ Processing thread started")
        
        last_processing_time = time.perf_counter()
        processing_count = 0
        last_processed_frame_id = -1
        
        while self.running:
            frame, frame_id, timestamp = self.frame_buffer.get_latest()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            if self.batch_mode and frame_id == last_processed_frame_id:
                if self.batch_complete:
                    print("⚙️ Batch processing complete, stopping processing thread")
                    break
                time.sleep(0.01)
                continue
            
            if frame_id % 30 == 0:
                print(f"🔄 Processing frame {frame_id}")
                
            process_start = time.perf_counter()
            result = self._process_frame(frame, frame_id, timestamp)
            process_end = time.perf_counter()
            
            result.processing_time = (process_end - process_start) * 1000
            last_processed_frame_id = frame_id
            
            if result.pose_data:
                with self.poses_lock:
                    self.all_poses.append(result.pose_data)
            
            try:
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                    
                self.result_queue.put_nowait(result)
                
                processing_count += 1
                if process_end - last_processing_time > 1.0:
                    fps = processing_count / (process_end - last_processing_time)
                    self.perf_monitor.add_processing_fps(fps)
                    processing_count = 0
                    last_processing_time = process_end
                    
            except queue.Full:
                pass
                
        print("⚙️ Processing thread stopped")
    
    def _process_frame(self, frame, frame_id, timestamp):
        """Process a single frame using RTMPose pipeline"""
        result = ProcessingResult(
            frame=frame,
            frame_id=frame_id,
            timestamp=timestamp
        )
        
        # Get frame metadata if in batch mode
        frame_info = None
        if self.batch_mode:
            with self.poses_lock:
                if hasattr(self, 'current_frame_info') and frame_id in self.current_frame_info:
                    frame_info = self.current_frame_info[frame_id]
        
        with self.state_lock:
            current_state = self.state
            current_context = self.context
            
        # State machine processing
        if current_state == TrackingState.INITIALIZING:
            bbox = self._yolo_detect(frame)
            if bbox is not None:
                result.bbox = bbox
                print(f"🔄 [INITIALIZING] YOLO detected bbox: {bbox}")
                
                # Use RTMPose for pose estimation
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    pos, quat, num_kpts, pose_data = self._detect_rtmpose_keypoints(crop, bbox, frame_id, frame_info)
                    result.position, result.quaternion, result.num_keypoints = pos, quat, num_kpts
                    result.pose_data = pose_data
                    if pose_data:
                        result.keypoint_confidences = pose_data.get('detected_keypoints', {}).get('confidences', [])
                    
                    if pos is not None:
                        print(f"✅ Initial RTMPose pose: pos={pos.round(3)}, keypoints={num_kpts}")
                    else:
                        print(f"⚠️ Initial RTMPose pose failed, keypoints={num_kpts}")

                tracker_success = self._init_tracker(frame, bbox)
                print(f"🔄 [INITIALIZING] Tracker init success: {tracker_success}")
                
                if tracker_success:
                    with self.state_lock:
                        self.state = TrackingState.TRACKING
                        self.context.bbox = bbox
                        self.context.num_keypoints = result.num_keypoints
                    print(f"✅ Initialized - Tracking with {result.num_keypoints} keypoints, bbox: {bbox}")
                else:
                    # Even if tracker fails, we can still do detection-based tracking
                    print("⚠️ Tracker initialization failed, falling back to detection-only mode")
                    with self.state_lock:
                        self.state = TrackingState.DETECTING
                    
        elif current_state == TrackingState.DETECTING:
            bbox = self._yolo_detect(frame)
            if bbox is not None:
                print(f"🔄 [DETECTING] YOLO bbox={bbox}")
                result.bbox = bbox
                
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    pos, quat, num_kpts, pose_data = self._detect_rtmpose_keypoints(crop, bbox, frame_id, frame_info)
                    result.position, result.quaternion, result.num_keypoints = pos, quat, num_kpts
                    result.pose_data = pose_data
                    if pose_data:
                        result.keypoint_confidences = pose_data.get('detected_keypoints', {}).get('confidences', [])
                    print(f"   → RTMPose keypoints={num_kpts} → {'OK' if pos is not None else 'FAIL'}")
                    
                tracker_success = self._init_tracker(frame, bbox)
                if tracker_success:
                    with self.state_lock:
                        self.state = TrackingState.TRACKING
                        self.context.bbox = bbox
                        self.context.num_keypoints = result.num_keypoints
                    print(f"✅ Object found - Tracking with {result.num_keypoints} keypoints")
                else:
                    # Stay in detection mode if tracker fails
                    print("⚠️ Staying in detection mode (tracker unavailable)")
                    with self.state_lock:
                        self.context.bbox = bbox
                        self.context.num_keypoints = result.num_keypoints
                    
        elif current_state == TrackingState.TRACKING:
            # Try to track, but fall back to detection if needed
            bbox, confidence = self._track_object(frame)
            
            # If tracking fails or confidence is low, try YOLO detection
            if bbox is None or confidence < 0.3:
                print(f"🔄 [TRACKING] Tracker failed (conf={confidence:.2f}), trying YOLO...")
                bbox = self._yolo_detect(frame)
                confidence = 0.5  # Assume decent confidence for YOLO detection
                
                if bbox is not None:
                    # Update simple tracker bbox if we're using fallback tracking
                    if hasattr(self, '_simple_tracker_bbox'):
                        self._simple_tracker_bbox = bbox
            
            if bbox is not None and confidence > 0.2:  # Lower threshold
                result.bbox = bbox
                with self.state_lock:
                    self.context.bbox = bbox
                    self.context.confidence = confidence
                    self.context.consecutive_failures = 0
                    
                # RTMPose pose estimation on current bbox
                x1, y1, x2, y2 = bbox
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    position, quaternion, num_keypoints, pose_data = self._detect_rtmpose_keypoints(cropped, bbox, frame_id, frame_info)
                    result.position = position
                    result.quaternion = quaternion
                    result.num_keypoints = num_keypoints
                    result.pose_data = pose_data
                    if pose_data:
                        result.keypoint_confidences = pose_data.get('detected_keypoints', {}).get('confidences', [])
                    
                    if position is None:
                        print(f"⚠️ RTMPose pose estimation failed - Keypoints: {num_keypoints}")
                    else:
                        if result.frame_id % 30 == 0:
                            print(f"✅ RTMPose pose estimated - Keypoints: {num_keypoints}, Pos: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
            else:
                with self.state_lock:
                    self.context.consecutive_failures += 1
                    if self.context.consecutive_failures > 5:  # More tolerance
                        self.state = TrackingState.LOST
                        self.tracker = None
                        if hasattr(self, '_simple_tracker_bbox'):
                            delattr(self, '_simple_tracker_bbox')
                        print("❌ Tracking lost")
                        
        elif current_state == TrackingState.LOST:
            bbox = self._yolo_detect(frame)
            if bbox is not None:
                result.bbox = bbox
                print(f"🔄 [LOST] Trying to reacquire with bbox: {bbox}")
                
                # Try RTMPose to verify it's a good detection
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    pos, quat, num_kpts, pose_data = self._detect_rtmpose_keypoints(crop, bbox, frame_id, frame_info)
                    result.position, result.quaternion, result.num_keypoints = pos, quat, num_kpts
                    result.pose_data = pose_data
                    
                    # Only reacquire if we get reasonable keypoints
                    if num_kpts >= 3:
                        if self._init_tracker(frame, bbox):
                            with self.state_lock:
                                self.state = TrackingState.TRACKING
                                self.context.bbox = bbox
                                self.context.consecutive_failures = 0
                            print(f"✅ Object reacquired with {num_kpts} keypoints")
                        else:
                            # Go to detection mode if tracker fails
                            with self.state_lock:
                                self.state = TrackingState.DETECTING
                                self.context.bbox = bbox
                                self.context.consecutive_failures = 0
                            print(f"✅ Object reacquired (detection mode) with {num_kpts} keypoints")
                    
        return result
    
    def _draw_axes(self, frame, position, quaternion, bbox=None):
        """Draw coordinate axes on frame (same as VAPE)"""
        if position is None or quaternion is None:
            return frame
        
        try:
            R = self._quaternion_to_rotation_matrix(quaternion)
            rvec, _ = cv2.Rodrigues(R)
            tvec = position.reshape(3, 1)
            
            axis_length = 0.1
            axis_points = np.float32([
                [0, 0, 0],
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, axis_length]
            ])
            
            K, distCoeffs = self._get_camera_intrinsics()
            axis_proj, _ = cv2.projectPoints(axis_points, rvec, tvec, K, distCoeffs)
            axis_proj = axis_proj.reshape(-1, 2).astype(int)
            
            origin = tuple(axis_proj[0])
            x_end = tuple(axis_proj[1])
            y_end = tuple(axis_proj[2])
            z_end = tuple(axis_proj[3])
            
            h, w = frame.shape[:2]
            points_in_bounds = all(
                0 <= pt[0] < w and 0 <= pt[1] < h 
                for pt in [origin, x_end, y_end, z_end]
            )
            
            if points_in_bounds:
                frame = cv2.line(frame, origin, x_end, (0, 0, 255), 3)    # X - Red
                frame = cv2.line(frame, origin, y_end, (0, 255, 0), 3)    # Y - Green  
                frame = cv2.line(frame, origin, z_end, (255, 0, 0), 3)    # Z - Blue
                
                cv2.circle(frame, origin, 5, (255, 255, 255), -1)
                
                cv2.putText(frame, "X", (x_end[0] + 5, x_end[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, "Y", (y_end[0] + 5, y_end[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "Z", (z_end[0] + 5, z_end[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                if bbox is not None:
                    center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                    axis_len = min(bbox[2]-bbox[0], bbox[3]-bbox[1]) // 6
                    
                    cv2.arrowedLine(frame, (center_x, center_y), 
                                   (center_x + axis_len, center_y), (0, 0, 255), 2)
                    cv2.arrowedLine(frame, (center_x, center_y), 
                                   (center_x, center_y - axis_len), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
                    
        except Exception as e:
            if bbox is not None:
                center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                axis_len = min(bbox[2]-bbox[0], bbox[3]-bbox[1]) // 6
                
                cv2.arrowedLine(frame, (center_x, center_y), 
                               (center_x + axis_len, center_y), (0, 0, 255), 2)
                cv2.arrowedLine(frame, (center_x, center_y), 
                               (center_x, center_y - axis_len), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 3, (255, 0, 0), -1)
        
        return frame
    
    def _quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        x, y, z, w = q
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2)]
        ])
    
    def display_thread(self):
        """Thread 3: Display results"""
        print("🖥️ Display thread started")
        
        if hasattr(self.args, 'no_display') and self.args.no_display:
            print("🖥️ Display disabled, running headless")
            while self.running:
                time.sleep(0.1)
            print("🖥️ Display thread stopped (headless)")
            return
        
        cv2.namedWindow('RTMPose Aircraft Pose Estimator', cv2.WINDOW_NORMAL)
        
        displayed_frames = 0
        no_result_count = 0
        max_no_result = 50
        
        while self.running:
            try:
                result = self.result_queue.get(timeout=0.1)
                displayed_frames += 1
                no_result_count = 0
                
                vis_frame = result.frame.copy()
                
                with self.state_lock:
                    state = self.state
                    
                state_colors = {
                    TrackingState.INITIALIZING: (0, 0, 255),
                    TrackingState.DETECTING: (0, 100, 255),
                    TrackingState.TRACKING: (0, 255, 0),
                    TrackingState.LOST: (0, 0, 200)
                }
                color = state_colors.get(state, (255, 255, 255))
                cv2.putText(vis_frame, f'State: {state.value.upper()}', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw bounding box
                if result.bbox is not None:
                    x1, y1, x2, y2 = result.bbox
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis_frame, f'RTMPose: {result.num_keypoints} kpts', (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw axes if pose available
                    if result.position is not None and result.quaternion is not None:
                        vis_frame = self._draw_axes(vis_frame, result.position, 
                                                   result.quaternion, result.bbox)
                
                # Method info
                method_text = 'RTMPose Global Features'
                cv2.putText(vis_frame, method_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Performance info
                processing_fps = self.perf_monitor.get_processing_fps()
                camera_fps = self.perf_monitor.get_camera_fps()
                
                cv2.putText(vis_frame, f'Processing FPS: {processing_fps:.1f}', (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, f'Camera FPS: {camera_fps:.1f}', (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(vis_frame, f'Processing Time: {result.processing_time:.1f}ms', (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                # Keypoint info
                if result.num_keypoints > 0:
                    kpt_color = (0, 255, 0) if result.num_keypoints > 4 else (0, 165, 255) if result.num_keypoints > 2 else (0, 0, 255)
                    cv2.putText(vis_frame, f'Keypoints: {result.num_keypoints}', (10, 180), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, kpt_color, 2)
                
                # Component timings
                y_offset = 210
                for name in ['yolo_detection', 'tracking', 'rtmpose_detection']:
                    avg_time = self.perf_monitor.get_average(name)
                    if avg_time > 0:
                        time_color = (0, 255, 0) if avg_time < 30 else (0, 165, 255) if avg_time < 50 else (0, 0, 255)
                        cv2.putText(vis_frame, f'{name}: {avg_time:.1f}ms', 
                                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, time_color, 1)
                        y_offset += 15
                
                # Show pose values
                if result.position is not None:
                    pos_text = f'Pos: [{result.position[0]:.3f}, {result.position[1]:.3f}, {result.position[2]:.3f}]'
                    cv2.putText(vis_frame, pos_text, (10, vis_frame.shape[0]-40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if result.quaternion is not None:
                    quat_text = f'Quat: [{result.quaternion[0]:.3f}, {result.quaternion[1]:.3f}, {result.quaternion[2]:.3f}, {result.quaternion[3]:.3f}]'
                    cv2.putText(vis_frame, quat_text, (10, vis_frame.shape[0]-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show JSON export info
                if self.batch_mode:
                    with self.poses_lock:
                        json_count = len(self.all_poses)
                    cv2.putText(vis_frame, f'JSON Entries: {json_count}', (10, vis_frame.shape[0]-60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.imshow('RTMPose Aircraft Pose Estimator', vis_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    
            except queue.Empty:
                no_result_count += 1
                
                if self.batch_mode and self.batch_complete and no_result_count > max_no_result:
                    print("🖥️ No more results in batch mode, stopping display")
                    break
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    
        cv2.destroyAllWindows()
        print("🖥️ Display thread stopped")
    
    def save_json_results(self, output_path):
        """Save all pose estimation results to JSON file"""
        with self.poses_lock:
            if os.path.isdir(output_path):
                base_filename = 'rtmpose_aircraft_pose_estimation.json'
                output_path = create_unique_filename(output_path, base_filename)
            else:
                save_dir = os.path.dirname(output_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                base_filename = os.path.basename(output_path)
                output_path = create_unique_filename(save_dir, base_filename)
            
            json_safe_poses = convert_to_json_serializable(self.all_poses)
            
            with open(output_path, 'w') as f:
                json.dump(json_safe_poses, f, indent=4)
            
            print(f"💾 Saved {len(self.all_poses)} RTMPose pose estimation results to {output_path}")
            return output_path
    
    def run(self):
        """Main entry point"""
        print("🚀 Starting RTMPose-based aircraft pose estimation")
        print("📌 Research Method: Global Semantic Features (RTMPose)")
        print("📌 Compare with: Local Features (SuperPoint+LightGlue)")
        if self.batch_mode:
            print(f"📌 Batch mode: Processing {len(self.image_entries)} images from {self.args.image_dir}")
        else:
            print("📌 Camera mode: Real-time processing")
        print("Press 'q' to quit")
        
        self.running = True
        
        camera_thread = threading.Thread(target=self.camera_thread, name="CameraThread")
        processing_thread = threading.Thread(target=self.processing_thread, name="ProcessingThread")
        display_thread = threading.Thread(target=self.display_thread, name="DisplayThread")
        
        self.threads = [camera_thread, processing_thread, display_thread]
        
        for thread in self.threads:
            thread.start()
            print(f"✅ Started {thread.name}")
        
        try:
            if self.batch_mode:
                camera_thread.join()
                print("📹 Camera/loading thread completed")
                
                processing_thread.join()
                print("⚙️ Processing thread completed")
                
                self.running = False
                display_thread.join(timeout=5.0)
                if display_thread.is_alive():
                    print("⚠️ Display thread did not stop cleanly")
                else:
                    print("🖥️ Display thread completed")
            else:
                for thread in self.threads:
                    thread.join()
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            self.running = False
            
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    print(f"⚠️ {thread.name} did not stop cleanly")
                
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
    parser = argparse.ArgumentParser(description='RTMPose-based Aircraft Pose Estimator - Research Comparison')
    
    # RTMPose model arguments
    parser.add_argument('--rtmpose_config', type=str,
                       default='configs/my_aircraft/rtmpose.py',
                       help='RTMPose config file path')
    parser.add_argument('--rtmpose_checkpoint', type=str,
                       #default='work_dirs/rtmpose-l_aircraft-384x288-optimized_20250705/best_coco_AP_epoch_200.pth',
                       default='work_dirs/rtmpose-l_aircraft-384x288_20250901/epoch_580.pth',
                       help='RTMPose checkpoint file path')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    
    # Batch processing arguments
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Folder containing extracted images (enables batch mode)')
    parser.add_argument('--csv_file', type=str, default=None,
                       help='CSV file with columns [Index, Timestamp, Filename]')
    
    # Output arguments
    parser.add_argument('--save_pose', type=str, default='rtmpose_aircraft_pose_results.json',
                       help='Path to save JSON pose estimation results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualization frames')
    
    # Processing arguments
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display visualization window')
    
    return parser.parse_args()


if __name__ == "__main__":
    print("🚀 RTMPose-based Aircraft Pose Estimator - Research Comparison")
    print("=" * 70)
    print("📊 RESEARCH PURPOSE:")
    print("   Compare Global Semantic Features (RTMPose) vs Local Features (SuperPoint+LightGlue)")
    print("   Same pipeline: YOLO → Feature Extraction → PnP+RANSAC → JSON Export")
    print("=" * 70)
    print("📖 Usage Examples:")
    print("  Camera mode:")
    print("    python rtmpose_aircraft_pose_estimator.py --save_pose rtmpose_results.json")
    print()
    print("  Batch mode:")
    print("    python rtmpose_aircraft_pose_estimator.py \\")
    print("      --image_dir /path/to/images \\")
    print("      --csv_file image_index.csv \\")
    print("      --rtmpose_config configs/my_aircraft/rtmpose.py \\")
    print("      --rtmpose_checkpoint work_dirs/rtmpose.../best_coco_AP_epoch_XXX.pth \\")
    print("      --save_pose rtmpose_results.json")
    print("=" * 70)
    
    args = parse_args()
    
    estimator = None
    
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        if estimator:
            estimator.running = False
            if estimator.all_poses:
                output_path = estimator.save_json_results(args.save_pose)
                print(f"💾 Emergency save completed: {output_path}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        estimator = RTMPosePoseEstimator(args)
        estimator.run()
        
        if estimator.all_poses:
            output_path = estimator.save_json_results(args.save_pose)
            print("🔬 Research data saved for comparison with VAPE method!")
        else:
            print("⚠️ No pose data to save")
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        if estimator and estimator.all_poses:
            output_path = estimator.save_json_results(args.save_pose)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        if estimator and estimator.all_poses:
            output_path = estimator.save_json_results(args.save_pose)
    finally:
        if estimator:
            estimator.running = False
            estimator._cleanup()
        
    print("🏁 RTMPose-based pose estimation finished!")
    print("📊 Ready for research comparison with VAPE method!")