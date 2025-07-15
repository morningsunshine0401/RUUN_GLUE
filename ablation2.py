import cv2
import numpy as np
import torch
import time
import argparse
import warnings
import json
import threading
import csv
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import queue # Import queue for display thread

# Import for robust correspondence matching
from scipy.spatial import cKDTree

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 VAPE MK47 Ablation Study Runner")

# Import required libraries
try:
    from ultralytics import YOLO
    import timm
    from torchvision import transforms
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    from PIL import Image
    print("✅ All libraries loaded")
except ImportError as e:
    print(f"❌ Import error: {e}. Please ensure all dependencies are installed.")
    exit(1)

# --- DATA CLASSES ---
@dataclass
class ProcessingResult:
    frame_id: int
    timestamp: float
    frame: Optional[np.ndarray] = None # Added for visualization
    position: Optional[np.ndarray] = None
    quaternion: Optional[np.ndarray] = None
    pose_data: Optional[Dict] = None
    kf_position: Optional[np.ndarray] = None
    kf_quaternion: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None # Added for visualization
    viewpoint: Optional[str] = None # Initial viewpoint estimation result
    final_viewpoint_used: Optional[str] = None # Actual viewpoint used after fallbacks
    fallback_occurred: bool = False # Indicates if a fallback was used
    num_matches: int = 0 # Added for visualization
    # NEW: Match visualization data
    match_vis_data: Optional['MatchVisualizationData'] = None # Forward reference

@dataclass
class MatchVisualizationData:
    """Data for match visualization"""
    anchor_image: np.ndarray
    anchor_keypoints: np.ndarray  # 2D keypoints in anchor image (from SuperPoint)
    frame_keypoints: np.ndarray   # 2D keypoints in input frame (from SuperPoint)
    matches: np.ndarray          # Match indices [anchor_idx, frame_idx]
    viewpoint: str               # The viewpoint of the anchor image being displayed
    used_for_pose: np.ndarray    # Boolean mask indicating which matches were used for pose estimation

# --- KALMAN FILTER ---
class LooselyCoupledKalmanFilter:
    def __init__(self, dt=1/30.0):
        self.dt = dt
        self.initialized = False
        self.n_states = 13 # [px, py, pz, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]
        self.x = np.zeros(self.n_states)
        self.x[9] = 1.0 # Initialize quaternion to [0,0,0,1] (identity rotation)
        
        # P: State Covariance Matrix - represents uncertainty in state estimates
        # Initialized with higher uncertainty (larger values) to allow filter to converge quickly
        self.P = np.eye(self.n_states) * 0.1 
        
        # Q: Process Noise Covariance Matrix - represents uncertainty in the process model (how state changes over time)
        # Higher Q values mean the filter trusts the model less and measurements more.
        # If orientations change too much and KF struggles, consider increasing Q for angular velocity components (indices 10, 11, 12).
        # For example, if self.Q[10,10], self.Q[11,11], self.Q[12,12] were increased.
        self.Q = np.eye(self.n_states) * 1e-3 
        
        # R: Measurement Noise Covariance Matrix - represents uncertainty in measurements
        # Higher R values mean the filter trusts measurements less.
        # Measurements are position (3) and quaternion (4), total 7.
        self.R = np.eye(7) * 1e-4

    def normalize_quaternion(self, q):
        norm = np.linalg.norm(q)
        return q / norm if norm > 1e-10 else np.array([0, 0, 0, 1])

    def predict(self):
        if not self.initialized: return None, None
        
        # State vector components
        px, py, pz, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz = self.x
        dt = self.dt
        
        # Predict position based on velocity
        self.x[0:3] += self.x[3:6] * dt
        
        # Predict quaternion based on angular velocity
        q = np.array([qx, qy, qz, qw])
        w = np.array([wx, wy, wz]) # Angular velocity vector
        
        # Quaternion derivative from angular velocity (simplified for small dt)
        # dq/dt = 0.5 * q * omega_quaternion (where omega_quaternion = [wx, wy, wz, 0])
        # This is a common linear approximation for quaternion propagation in KF.
        # For large, rapid rotations, a more sophisticated non-linear filter (EKF/UKF)
        # or a higher angular velocity process noise (Q) might be needed.
        omega_mat = np.array([[0, -wx, -wy, -wz], 
                              [wx, 0, wz, -wy], 
                              [wy, -wz, 0, wx], 
                              [wz, wy, -wx, 0]])
        q_new = self.normalize_quaternion(q + 0.5 * dt * omega_mat @ q)
        self.x[6:10] = q_new
        
        # State transition matrix F for linear prediction
        F = np.eye(self.n_states)
        F[0:3, 3:6] = np.eye(3) * dt # Position depends on velocity
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x[0:3], self.x[6:10]

    def update(self, position, quaternion):
        measurement = np.concatenate([position, self.normalize_quaternion(quaternion)])
        
        if not self.initialized:
            self.x[0:3], self.x[6:10] = position, measurement[3:7]
            self.initialized = True
            return self.x[0:3], self.x[6:10]
        
        predicted_measurement = np.concatenate([self.x[0:3], self.x[6:10]])
        
        innovation = measurement - predicted_measurement
        # Handle quaternion ambiguity (q and -q represent same rotation)
        if np.dot(measurement[3:7], predicted_measurement[3:7]) < 0:
            innovation[3:7] = -measurement[3:7] - predicted_measurement[3:7] # Choose the closer representation
        
        # Measurement Jacobian H
        H = np.zeros((7, self.n_states))
        H[0:3, 0:3] = np.eye(3) # Position measurement
        H[3:7, 6:10] = np.eye(4) # Quaternion measurement
        
        # Innovation (measurement residual) covariance S
        S = H @ self.P @ H.T + self.R
        
        # Kalman Gain K
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x += K @ innovation
        self.x[6:10] = self.normalize_quaternion(self.x[6:10]) # Re-normalize quaternion after update
        
        # Update covariance
        self.P = (np.eye(self.n_states) - K @ H) @ self.P
        
        return self.x[0:3], self.x[6:10]

# --- HELPER FUNCTIONS ---
def read_image_index_csv(csv_path):
    entries = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({'index': int(row['Index']),'timestamp': float(row['Timestamp']), 'filename': row['Filename']})
    return entries

def convert_to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [convert_to_json_serializable(i) for i in obj]
    return obj

def create_unique_filename(directory, base_filename):
    base_path = Path(directory or ".") / base_filename
    if not base_path.exists(): return str(base_path)
    name, ext = base_path.stem, base_path.suffix
    counter = 1
    while True:
        new_path = base_path.with_name(f"{name}_{counter}{ext}")
        if not new_path.exists(): return str(new_path)
        counter += 1

# --- MAIN ESTIMATOR CLASS (Modified for Ablation) ---
class AblationPoseEstimator:
    def __init__(self, args, ablation_cfg: Dict):
        print("-" * 60)
        print(f"🔧 Initializing Estimator for Experiment: {ablation_cfg['name']}")
        print(f"   - YOLO Enabled: {ablation_cfg['use_yolo']}")
        print(f"   - Viewpoint Estimation Enabled: {ablation_cfg['use_viewpoint']}")
        print(f"   - Matcher: {'SuperPoint/LightGlue' if ablation_cfg['use_superpoint_lightglue'] else 'ORB/BFMatcher'}")
        print(f"   - Kalman Filter Enabled: {ablation_cfg['use_kalman_filter']}")
        print("-" * 60)

        self.args = args
        self.ablation_cfg = ablation_cfg
        self.image_entries = []
        self.video_capture = None # Added for video input
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")
        self.camera_width, self.camera_height = 1280, 720
        self.all_poses = []

        if self.ablation_cfg['use_kalman_filter']:
            self.kf = LooselyCoupledKalmanFilter(dt=1/30.0)

        self._init_models()
        self._init_batch_processing()
        self._init_anchor_data()
        
        # Display queue for visualization
        self.display_queue = queue.Queue(maxsize=1) # Buffer for display frames
        self.running = True # Control display thread
        if not getattr(self.args, 'no_display', False):
            self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
            self.display_thread.start()
            
        print("✅ Estimator initialized successfully!")

    def _init_batch_processing(self):
        if self.args.video_file and os.path.exists(self.args.video_file): # New: Handle video file input
            self.video_capture = cv2.VideoCapture(self.args.video_file)
            if not self.video_capture.isOpened():
                raise ValueError(f"Could not open video file: {self.args.video_file}")
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.image_entries = [{'index': i, 'timestamp': i/fps, 'filename': f"frame_{i:06d}.png"} for i in range(total_frames)]
            print(f"✅ Loaded {len(self.image_entries)} frames from video: {self.args.video_file}.")
        elif self.args.csv_file and os.path.exists(self.args.csv_file):
            self.image_entries = read_image_index_csv(self.args.csv_file)
            self.image_entries.sort(key=lambda x: x['index'])
            print(f"✅ Loaded {len(self.image_entries)} image entries from CSV.")
        elif self.args.image_dir and os.path.exists(self.args.image_dir):
            image_files = sorted([f for f in os.listdir(self.args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            self.image_entries = [{'index': i, 'timestamp': i/30.0, 'filename': f} for i, f in enumerate(image_files)]
            print(f"✅ Found {len(self.image_entries)} images in directory.")
        else:
            raise ValueError("Must provide a valid --image_dir, --csv_file, or --video_file.")


    def _init_models(self):
        try:
            if self.ablation_cfg['use_yolo']:
                print("  📦 Loading YOLO...")
                self.yolo_model = YOLO("YOLO_best.pt").to(self.device)
                #self.yolo_model = YOLO("yolov8s.pt").to(self.device)
            if self.ablation_cfg['use_viewpoint']:
                print("  📦 Loading viewpoint classifier (MobileViT)...")
                self.vp_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=4)
                vp_model_path = 'mobilevit_viewpoint_20250703.pth'
                if os.path.exists(vp_model_path):
                    self.vp_model.load_state_dict(torch.load(vp_model_path, map_location=self.device))
                else:
                    print(f"  ⚠️ Viewpoint model file not found at '{vp_model_path}'.")
                self.vp_model.eval().to(self.device)
                self.vp_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
                self.class_names = ['NE', 'NW', 'SE', 'SW']
            if self.ablation_cfg['use_superpoint_lightglue']:
                print("  📦 Loading SuperPoint & LightGlue...")
                self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
                self.matcher = LightGlue(features="superpoint").eval().to(self.device)
            else:
                print("  📦 Initializing ORB & BFMatcher...")
                self.orb = cv2.ORB_create(nfeatures=2000)
                self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

    def _init_anchor_data(self):
        print("🛠️ Initializing anchor data...")
        
        # Complete 2D and 3D point data from your original VAPE_MK47.py script
        ne_anchor_2d = np.array([[924, 148], [571, 115], [398, 31], [534, 133], [544, 141], [341, 219], [351, 228], [298, 240], [420, 83], [225, 538], [929, 291], [794, 381], [485, 569], [826, 305], [813, 264], [791, 285], [773, 271], [760, 289], [830, 225], [845, 233], [703, 308], [575, 361], [589, 373], [401, 469], [414, 481], [606, 454], [548, 399], [521, 510], [464, 451], [741, 380]], dtype=np.float32)
        ne_anchor_3d = np.array([[-0.0, -0.025, -0.24], [0.23, 0.0, -0.113], [0.243, -0.104, 0.0], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [0.23, 0.0, 0.113], [0.206, -0.07, -0.002], [-0.0, -0.025, 0.24], [-0.08, 0.0, -0.156], [-0.09, 0.0, -0.042], [-0.08, 0.0, 0.156], [-0.052, 0.0, -0.097], [-0.029, 0.0, -0.127], [-0.037, 0.0, -0.097], [-0.017, 0.0, -0.092], [-0.023, 0.0, -0.075], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [-0.074, 0.0, 0.074], [-0.019, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.019, 0.0, 0.128], [-0.1, -0.03, 0.0]], dtype=np.float32)
        nw_anchor_2d = np.array([[511, 293], [591, 284], [587, 330], [413, 249], [602, 348], [715, 384], [598, 298], [656, 171], [805, 213], [703, 392], [523, 286], [519, 327], [387, 289], [727, 126], [425, 243], [636, 358], [745, 202], [595, 388], [436, 260], [539, 313], [795, 220], [351, 291], [665, 165], [611, 353], [650, 377], [516, 389], [727, 143], [496, 378], [575, 312], [617, 368], [430, 312], [480, 281], [834, 225], [469, 339], [705, 223], [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]], dtype=np.float32)
        nw_anchor_3d = np.array([[-0.014, 0.0, 0.042], [0.025, -0.014, -0.011], [-0.014, 0.0, -0.042], [-0.014, 0.0, 0.156], [-0.023, 0.0, -0.065], [0.0, 0.0, -0.156], [0.025, 0.0, -0.015], [0.217, 0.0, 0.07], [0.23, 0.0, -0.07], [-0.014, 0.0, -0.156], [0.0, 0.0, 0.042], [-0.057, -0.018, -0.01], [-0.074, -0.0, 0.128], [0.206, -0.07, -0.002], [-0.0, -0.0, 0.156], [-0.017, -0.0, -0.092], [0.217, -0.0, -0.027], [-0.052, -0.0, -0.097], [-0.019, -0.0, 0.128], [-0.035, -0.018, -0.01], [0.217, -0.0, -0.07], [-0.08, -0.0, 0.156], [0.23, 0.0, 0.07], [-0.023, -0.0, -0.075], [-0.029, -0.0, -0.127], [-0.09, -0.0, -0.042], [0.206, -0.055, -0.002], [-0.09, -0.0, -0.015], [0.0, -0.0, -0.015], [-0.037, -0.0, -0.097], [-0.074, -0.0, 0.074], [-0.019, -0.0, 0.074], [0.23, -0.0, -0.113], [-0.1, -0.03, 0.0], [0.17, -0.0, -0.015], [0.23, -0.0, 0.113], [-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [-0.08, -0.0, -0.156]], dtype=np.float32)
        se_anchor_2d = np.array([[415, 144], [1169, 508], [275, 323], [214, 395], [554, 670], [253, 428], [280, 415], [355, 365], [494, 621], [519, 600], [806, 213], [973, 438], [986, 421], [768, 343], [785, 328], [841, 345], [931, 393], [891, 306], [980, 345], [651, 210], [625, 225], [588, 216], [511, 215], [526, 204], [665, 271]], dtype=np.float32)
        se_anchor_3d = np.array([[-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [0.23, 0.0, -0.113], [0.23, 0.0, 0.113], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.206, -0.07, -0.002], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [-0.1, -0.03, 0.0], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.019, 0.0, 0.074], [-0.019, 0.0, 0.128], [-0.074, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.052, 0.0, -0.097], [-0.037, 0.0, -0.097], [-0.029, 0.0, -0.127], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042]], dtype=np.float32)
        sw_anchor_2d = np.array([[650, 312], [630, 306], [907, 443], [814, 291], [599, 349], [501, 386], [965, 359], [649, 355], [635, 346], [930, 335], [843, 467], [702, 339], [718, 321], [930, 322], [727, 346], [539, 364], [786, 297], [1022, 406], [1004, 399], [539, 344], [536, 309], [864, 478], [745, 310], [1049, 393], [895, 258], [674, 347], [741, 281], [699, 294], [817, 494], [992, 281]], dtype=np.float32)
        sw_anchor_3d = np.array([[-0.035, -0.018, -0.01], [-0.057, -0.018, -0.01], [0.217, -0.0, -0.027], [-0.014, -0.0, 0.156], [-0.023, -0.0, -0.065], [-0.014, -0.0, -0.156], [0.234, -0.05, -0.002], [0.0, -0.0, -0.042], [-0.014, -0.0, -0.042], [0.206, -0.055, -0.002], [0.217, -0.0, -0.07], [0.025, -0.014, -0.011], [-0.014, -0.0, 0.042], [0.206, -0.07, -0.002], [0.049, -0.016, -0.011], [-0.029, -0.0, -0.127], [-0.019, -0.0, 0.128], [0.23, -0.0, 0.07], [0.217, -0.0, 0.07], [-0.052, -0.0, -0.097], [-0.175, -0.0, -0.015], [0.23, -0.0, -0.07], [-0.019, -0.0, 0.074], [0.23, -0.0, 0.113], [-0.0, -0.025, 0.24], [-0.0, -0.0, -0.015], [-0.074, -0.0, 0.128], [-0.074, -0.0, 0.074], [0.23, -0.0, -0.113], [0.243, -0.104, 0.0]], dtype=np.float32)

        # The complete anchor definitions dictionary
        anchor_definitions = {
            'NE': {'path': 'NE.png', '2d': ne_anchor_2d, '3d': ne_anchor_3d},
            'NW': {'path': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png', '2d': nw_anchor_2d, '3d': nw_anchor_3d},
            'SE': {'path': 'SE.png', '2d': se_anchor_2d, '3d': se_anchor_3d},
            'SW': {'path': 'Anchor_B.png', '2d': sw_anchor_2d, '3d': sw_anchor_3d}
        }
        
        self.viewpoint_anchors = {}
        for viewpoint, data in anchor_definitions.items():
            print(f"  📸 Processing anchor for {viewpoint}...")
            anchor_image_bgr = self._load_anchor_image(data['path'], viewpoint) # Pass viewpoint for dummy image
            if anchor_image_bgr is None:
                print(f"     -> Failed to load image for {viewpoint}. Skipping.")
                continue

            if self.ablation_cfg['use_superpoint_lightglue']:
                anchor_features = self._extract_features_sp(anchor_image_bgr)
                anchor_keypoints_sp = anchor_features['keypoints'][0].cpu().numpy()
                if len(anchor_keypoints_sp) == 0:
                    print(f"     -> No SuperPoint keypoints found for {viewpoint}. Skipping.")
                    continue
                sp_tree = cKDTree(anchor_keypoints_sp)
                distances, indices = sp_tree.query(data['2d'], k=1, distance_upper_bound=5.0)
                valid_mask = distances != np.inf
                
                self.viewpoint_anchors[viewpoint] = {
                    'features': anchor_features,
                    'anchor_image': anchor_image_bgr, # Store anchor image for visualization
                    'matched_sp_indices': indices[valid_mask],
                    'matched_3d_points_map': {idx: pt for idx, pt in zip(indices[valid_mask], data['3d'][valid_mask])}
                }
                print(f"     -> Processed with SuperPoint. Found {len(self.viewpoint_anchors[viewpoint]['matched_3d_points_map'])} correspondences.")
            else: # ORB Matcher
                gray_anchor = cv2.cvtColor(anchor_image_bgr, cv2.COLOR_BGR2GRAY)
                kp_orb, des_orb = self.orb.detectAndCompute(gray_anchor, None)
                if des_orb is None:
                    print(f"     -> No ORB features found for {viewpoint}. Skipping.")
                    continue
                orb_kpts_coords = np.array([kp.pt for kp in kp_orb])
                kptree = cKDTree(orb_kpts_coords)
                distances, indices = kptree.query(data['2d'], k=1, distance_upper_bound=8.0)
                valid_mask = distances != np.inf
                
                self.viewpoint_anchors[viewpoint] = {
                    'orb_kp': [kp_orb[i] for i in indices[valid_mask]],
                    'orb_des': des_orb[indices[valid_mask]],
                    'orb_3d_points': data['3d'][valid_mask],
                    'anchor_image': anchor_image_bgr # Store anchor image for visualization
                }
                print(f"     -> Processed with ORB. Found {len(self.viewpoint_anchors[viewpoint]['orb_kp'])} correspondences.")
                
        print("✅ Anchor data initialization complete.")

    def _load_anchor_image(self, path, viewpoint):
        if not os.path.exists(path): 
            print(f"  ❌ Failed to load {viewpoint} anchor: {path} not found. Creating dummy image.")
            dummy = np.full((self.camera_height, self.camera_width, 3), (128, 128, 128), dtype=np.uint8)
            cv2.putText(dummy, f'DUMMY {viewpoint}', (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            return dummy
        return cv2.resize(cv2.imread(path), (self.camera_width, self.camera_height))

    def _extract_features_sp(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad(): return self.extractor.extract(tensor)

    def _detect_features_orb(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(gray, None)

    def run_experiment(self):
        if not self.image_entries: return 0, 0, 0
        print(f"\n▶️ Running Experiment: {self.ablation_cfg['name']}")
        start_time = time.time()
        success_count = 0
        
        total_frames_to_process = len(self.image_entries)

        for i, entry in enumerate(self.image_entries):
            frame = None
            if self.video_capture: # If video input
                ret, frame = self.video_capture.read()
                if not ret:
                    print(f"  Warning: Failed to read frame {i} from video or end of video reached.")
                    break # Exit loop if frame cannot be read
            else: # If image directory or CSV
                image_path = os.path.join(self.args.image_dir, entry['filename']) if self.args.image_dir else entry['filename']
                if not os.path.exists(image_path):
                    print(f"  Warning: Image file not found: {image_path}. Skipping.")
                    continue
                frame = cv2.imread(image_path)
            
            if frame is None:
                continue # Skip if frame couldn't be loaded/read

            result = self._process_frame(frame, entry['index'], entry['timestamp'], entry)
            if result.pose_data and not result.pose_data.get('pose_estimation_failed', True):
                success_count += 1
            self.all_poses.append(convert_to_json_serializable(result.pose_data))
            
            # Display the result
            if not getattr(self.args, 'no_display', False):
                self._display_result(result)

            if (i + 1) % 100 == 0: print(f"  Processed {i+1}/{total_frames_to_process} images...")
        
        total_time = time.time() - start_time
        fps = total_frames_to_process / total_time if total_time > 0 else 0
        success_rate = (success_count / total_frames_to_process) * 100 if total_frames_to_process > 0 else 0

        print(f"✅ Experiment '{self.ablation_cfg['name']}' Complete.")
        print(f"   - Success Rate: {success_rate:.2f}% ({success_count}/{total_frames_to_process})")
        print(f"   - Average FPS: {fps:.2f}")
        
        if self.video_capture:
            self.video_capture.release() # Release video capture object
        
        self.save_results()
        
        # Signal display thread to stop
        self.running = False
        if not getattr(self.args, 'no_display', False) and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0) # Wait for display thread to finish
        cv2.destroyAllWindows() # Ensure all windows are closed

        return success_count, total_frames_to_process, fps

    def _process_frame(self, frame, frame_id, timestamp, frame_info):
        result = ProcessingResult(frame=frame, frame_id=frame_id, timestamp=timestamp) # Pass frame for visualization
        if self.ablation_cfg['use_kalman_filter'] and self.kf.initialized:
            result.kf_position, result.kf_quaternion = self.kf.predict()
        
        bbox = self._yolo_detect(frame) if self.ablation_cfg['use_yolo'] else None
        result.bbox = bbox # Store bbox for visualization
        
        # Determine initial viewpoint
        if self.ablation_cfg['use_viewpoint']:
            if bbox and bbox[3] > bbox[1] and bbox[2] > bbox[0]:
                crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            else:
                crop = frame # Use full frame if no valid bbox or YOLO is disabled
            initial_viewpoint = self._classify_viewpoint(crop)
        else:
            initial_viewpoint = 'NW' # Default if viewpoint estimation is off
        result.viewpoint = initial_viewpoint # Store initial viewpoint for visualization
            
        # Call the refactored _estimate_pose which handles fallbacks and best selection
        pos, quat, pose_data, num_matches, match_vis_data = self._estimate_pose(frame, initial_viewpoint, bbox, frame_id, frame_info)
        
        # Update result with actual viewpoint used and fallback status
        result.final_viewpoint_used = pose_data.get('final_viewpoint_used') if pose_data else None
        result.fallback_occurred = pose_data.get('fallback_occurred', False) if pose_data else False
        result.num_matches = num_matches # Store num_matches for visualization
        result.match_vis_data = match_vis_data # Store match_vis_data for visualization
        
        if pos is not None and self.ablation_cfg['use_kalman_filter']:
            kf_pos, kf_quat = self.kf.update(pos, quat)
            if pose_data and not pose_data.get('pose_estimation_failed', True):
                pose_data['position_kf'] = kf_pos.tolist()
                pose_data['quaternion_kf'] = kf_quat.tolist()
        
        result.position, result.quaternion, result.pose_data = pos, quat, pose_data
        return result

    def _estimate_pose(self, full_frame, initial_viewpoint, bbox, frame_id, frame_info):
        """
        Attempts to estimate pose by trying all available anchors and selecting the best result.
        Crucially, if 'use_viewpoint' is False (i.e., in the 'no_viewpoint' ablation),
        it will *only* attempt pose estimation with the initial_viewpoint ('NW').
        """
        
        # Determine which anchors to try based on the ablation configuration
        if not self.ablation_cfg['use_viewpoint']:
            # If viewpoint estimation is explicitly disabled, only try the initial (hardcoded 'NW') anchor.
            anchors_to_try = [initial_viewpoint]
            print(f"  � 'no_viewpoint' ablation: Only trying anchor '{initial_viewpoint}'. No fallbacks.")
        else:
            # If viewpoint estimation is enabled, try the estimated viewpoint first, then others.
            all_possible_viewpoints = ['NW', 'NE', 'SE', 'SW'] # Define a consistent order
            
            anchors_to_try = [initial_viewpoint]
            for vp in all_possible_viewpoints:
                if vp != initial_viewpoint and vp in self.viewpoint_anchors:
                    anchors_to_try.append(vp)
            
            # Ensure only unique and existing anchors are in the list, maintaining order
            unique_anchors_to_try = []
            seen = set()
            for vp in anchors_to_try:
                if vp not in seen and vp in self.viewpoint_anchors: # Ensure anchor data exists
                    unique_anchors_to_try.append(vp)
                    seen.add(vp)
            anchors_to_try = unique_anchors_to_try # Use the filtered list

        successful_candidates = []

        for current_viewpoint in anchors_to_try: # Iterate through the determined list of anchors
            # Determine if this is a fallback attempt (only relevant if use_viewpoint is True)
            is_fallback_attempt = (current_viewpoint != initial_viewpoint) and self.ablation_cfg['use_viewpoint']
            if is_fallback_attempt:
                print(f"  🔄 Trying fallback anchor: {current_viewpoint}")

            anchor_data, crop_offset = self.viewpoint_anchors[current_viewpoint], np.array([0, 0])
            
            # Re-crop if bbox is available, otherwise use full frame
            if bbox and bbox[3] > bbox[1] and bbox[2] > bbox[0]:
                crop = full_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                crop_offset = np.array([bbox[0], bbox[1]])
            else:
                crop = full_frame
                
            if crop.size == 0:
                print(f"  ⚠️ Skipping {current_viewpoint}: Invalid crop (empty after bbox).")
                continue # Try next anchor if crop is invalid

            if self.ablation_cfg['use_superpoint_lightglue']:
                pos, quat, pose_data, num_matches, match_vis_data = self._estimate_pose_sp(crop, crop_offset, anchor_data, full_frame, current_viewpoint, frame_id, frame_info)
            else:
                pos, quat, pose_data, num_matches, match_vis_data = self._estimate_pose_orb(crop, crop_offset, anchor_data, full_frame, current_viewpoint, frame_id, frame_info)

            # Check if pose estimation was successful with the current anchor
            if pos is not None and pose_data and not pose_data.get('pose_estimation_failed', True):
                # Store successful result
                successful_candidates.append({
                    'viewpoint': current_viewpoint,
                    'pos': pos,
                    'quat': quat,
                    'pose_data': pose_data,
                    'num_matches': num_matches,
                    'match_vis_data': match_vis_data,
                    'num_inliers': pose_data.get('num_inliers', 0),
                    'reprojection_error': pose_data.get('mean_reprojection_error', float('inf'))
                })
                print(f"  ✅ Pose estimated successfully with anchor {current_viewpoint} (Inliers: {pose_data.get('num_inliers', 0)}, Error: {pose_data.get('mean_reprojection_error', float('inf')):.2f}).")
            else:
                print(f"  ❌ Pose estimation failed with anchor {current_viewpoint}. Reason: {pose_data.get('error_reason', 'Unknown')}")

            # IMPORTANT: For 'no_viewpoint' ablation, we only try the initial anchor.
            # If it fails, we don't try others. So, break after the first attempt.
            if not self.ablation_cfg['use_viewpoint']:
                break

        # After trying all relevant anchors, select the best candidate
        if not successful_candidates:
            # If no anchor yielded a successful pose
            failure_report = self._create_failure_report(frame_id, 'all_anchors_failed')
            failure_report['final_viewpoint_used'] = initial_viewpoint # Report the initial one tried
            # Fallback occurred is true only if we tried more than one anchor
            failure_report['fallback_occurred'] = (len(anchors_to_try) > 1 and not self.ablation_cfg['use_viewpoint']) # Corrected logic
            return None, None, failure_report, 0, None
        
        # Sort candidates: primary by max inliers (desc), secondary by min reprojection error (asc)
        best_result = sorted(
            successful_candidates, 
            key=lambda x: (-x['num_inliers'], x['reprojection_error'])
        )[0]

        # Determine if a fallback occurred to get this best result
        # Fallback only occurs if viewpoint estimation is ON and a different anchor was chosen.
        fallback_occurred = (best_result['viewpoint'] != initial_viewpoint) and self.ablation_cfg['use_viewpoint']

        # Update the pose_data dictionary with the final decision
        best_result['pose_data']['final_viewpoint_used'] = best_result['viewpoint']
        best_result['pose_data']['fallback_occurred'] = fallback_occurred

        print(f"  🏆 Best pose selected from anchor: {best_result['viewpoint']} (Inliers: {best_result['num_inliers']}, Error: {best_result['reprojection_error']:.2f})")
        
        return best_result['pos'], best_result['quat'], best_result['pose_data'], best_result['num_matches'], best_result['match_vis_data']
    
    def _estimate_pose_sp(self, crop, crop_offset, anchor_data, full_frame, viewpoint_for_vis, frame_id, frame_info):
        frame_features = self._extract_features_sp(crop)
        with torch.no_grad(): matches_dict = self.matcher({'image0': anchor_data['features'], 'image1': frame_features})
        matches = rbd(matches_dict)['matches'].cpu().numpy()
        num_matches = len(matches) # Get total number of matches
        if num_matches < 6: return None, None, self._create_failure_report(frame_id, 'insufficient_matches'), num_matches, None
            
        valid_anchor_sp_indices = matches[:, 0]
        
        points_3d = []
        points_2d_full = []
        used_for_pose_mask = np.zeros(num_matches, dtype=bool) # For visualization

        for i, (anchor_sp_idx, frame_sp_idx) in enumerate(matches):
            if anchor_sp_idx in anchor_data['matched_3d_points_map']:
                points_3d.append(anchor_data['matched_3d_points_map'][anchor_sp_idx])
                points_2d_full.append(frame_features['keypoints'][0].cpu().numpy()[frame_sp_idx] + crop_offset)
                used_for_pose_mask[i] = True
        
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d_full = np.array(points_2d_full, dtype=np.float32)
        
        if len(points_3d) < 6: return None, None, self._create_failure_report(frame_id, 'insufficient_correspondences_after_mapping'), num_matches, None
            
        pos, quat, pose_data = self._solve_pnp(points_3d, points_2d_full, frame_id, frame_info)

        # Prepare MatchVisualizationData
        anchor_keypoints_sp = anchor_data['features']['keypoints'][0].cpu().numpy()
        frame_keypoints_full_for_vis = frame_features['keypoints'][0].cpu().numpy() + crop_offset # Full frame coords for vis

        match_vis_data = MatchVisualizationData(
            anchor_image=anchor_data['anchor_image'],
            anchor_keypoints=anchor_keypoints_sp[matches[:, 0]], # All matched anchor KPs
            frame_keypoints=frame_keypoints_full_for_vis[matches[:, 1]], # All matched frame KPs
            matches=matches,
            viewpoint=viewpoint_for_vis, # Use the viewpoint currently being attempted for visualization
            used_for_pose=used_for_pose_mask # Mask for points used in PnP
        )
        return pos, quat, pose_data, num_matches, match_vis_data

    def _estimate_pose_orb(self, crop, crop_offset, anchor_data, full_frame, viewpoint_for_vis, frame_id, frame_info):
        kp_frame, des_frame = self._detect_features_orb(crop)
        if des_frame is None or len(des_frame) < 6: return None, None, self._create_failure_report(frame_id, 'insufficient_features'), 0, None
        
        matches = self.bf_matcher.match(anchor_data['orb_des'], des_frame)
        num_matches = len(matches) # Get total number of matches
        if num_matches < 6: return None, None, self._create_failure_report(frame_id, 'insufficient_matches'), num_matches, None
            
        points_3d = np.array([anchor_data['orb_3d_points'][m.queryIdx] for m in matches])
        points_2d_full = np.array([kp_frame[m.trainIdx].pt for m in matches]) + crop_offset

        # For ORB, all matches are considered for PnP initially, so all are "used for pose" for visualization
        used_for_pose_mask = np.ones(num_matches, dtype=bool) 

        pos, quat, pose_data = self._solve_pnp(points_3d, points_2d_full, frame_id, frame_info)

        # Prepare MatchVisualizationData for ORB
        # ORB keypoints are not directly comparable to SuperPoint features dict structure
        # Need to extract ORB keypoints from anchor_data['orb_kp']
        anchor_keypoints_orb_coords = np.array([kp.pt for kp in anchor_data['orb_kp']])
        frame_keypoints_orb_coords = np.array([kp.pt for kp in kp_frame]) + crop_offset # Full frame coords for vis

        match_vis_data = MatchVisualizationData(
            anchor_image=anchor_data['anchor_image'],
            anchor_keypoints=anchor_keypoints_orb_coords[[m.queryIdx for m in matches]], # Matched anchor KPs
            frame_keypoints=frame_keypoints_orb_coords[[m.trainIdx for m in matches]], # Matched frame KPs
            matches=np.array([[m.queryIdx, m.trainIdx] for m in matches]), # Dummy indices for structure
            viewpoint=viewpoint_for_vis, # Use the viewpoint currently being attempted for visualization
            used_for_pose=used_for_pose_mask
        )
        return pos, quat, pose_data, num_matches, match_vis_data

    def _solve_pnp(self, points_3d, points_2d, frame_id, frame_info):
        K, dist_coeffs = self._get_camera_intrinsics()
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        except cv2.error: return None, None, self._create_failure_report(frame_id, 'pnp_cv_error')

        if not success or inliers is None or len(inliers) < 4:
            return None, None, self._create_failure_report(frame_id, 'pnp_failed', num_inliers=len(inliers) if inliers is not None else 0)
        
        R, _ = cv2.Rodrigues(rvec)
        position, quaternion = tvec.flatten(), self._rotation_matrix_to_quaternion(R)
        
        projected, _ = cv2.projectPoints(points_3d[inliers.flatten()], rvec, tvec, K, dist_coeffs)
        error = np.mean(np.linalg.norm(points_2d[inliers.flatten()].reshape(-1, 1, 2) - projected, axis=2))
        
        return position, quaternion, self._create_success_report(frame_id, position, quaternion, len(inliers), error, frame_info)

    def _yolo_detect(self, frame):
        results = self.yolo_model(frame, verbose=False, conf=0.4)
        if len(results[0].boxes) > 0: return tuple(map(int, results[0].boxes.xyxy.cpu().numpy()[0]))
        return None

    def _classify_viewpoint(self, crop):
        # Ensure crop is not empty before processing
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            print("  Warning: Empty crop received for viewpoint classification. Defaulting to 'NW'.")
            return 'NW'
        
        pil_img = Image.fromarray(cv2.cvtColor(cv2.resize(crop, (128, 128)), cv2.COLOR_BGR2RGB))
        tensor = self.vp_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad(): return self.class_names[torch.argmax(self.vp_model(tensor), dim=1).item()]

    def _get_camera_intrinsics(self):
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32), None

    def _rotation_matrix_to_quaternion(self, R):
        # This function converts a rotation matrix to a quaternion.
        # It has been corrected to fix the UnboundLocalError.
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
            
        # The convention is [x, y, z, w]
        return np.array([qx, qy, qz, qw])

    def _quaternion_to_rotation_matrix(self, q):
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])

    def _create_failure_report(self, frame_id, reason, **kwargs):
        return {'frame': frame_id, 'pose_estimation_failed': True, 'error_reason': reason, **kwargs}

    def _create_success_report(self, frame_id, pos, quat, num_inliers, error, frame_info):
        data = {'frame': frame_id, 'pose_estimation_failed': False, 'position': pos.tolist(), 'quaternion': quat.tolist(),
                'num_inliers': num_inliers, 'mean_reprojection_error': error}
        if frame_info: data.update(frame_info)
        return data

    def save_results(self):
        output_filename = create_unique_filename(self.args.output_dir, f"poses_{self.ablation_cfg['name']}.json")
        print(f"💾 Saving {len(self.all_poses)} records to {output_filename}")
        with open(output_filename, 'w') as f:
            json.dump(self.all_poses, f, indent=2)

    def _display_result(self, result):
        """Puts the visualization frame into the display queue."""
        if getattr(self.args, 'vis_match', False):
            vis_frame = self._create_match_visualization(result)
        else:
            vis_frame = self._create_batch_display(result)
        
        try:
            # Clear the queue before putting a new frame to ensure latest frame is displayed
            while not self.display_queue.empty():
                self.display_queue.get_nowait()
            self.display_queue.put_nowait(vis_frame)
        except queue.Full:
            pass # Skip if queue is full (display thread is busy)

    def _display_loop(self):
        """Dedicated thread for displaying frames."""
        window_name = f"Ablation Study - {self.ablation_cfg['name']}"
        if getattr(self.args, 'vis_match', False):
            window_name += " (Match Visualization)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 2560, 720) # Wider for side-by-side
        else:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)

        while self.running:
            try:
                frame_to_display = self.display_queue.get(timeout=0.1)
                cv2.imshow(window_name, frame_to_display)
            except queue.Empty:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False # Signal main thread to stop
                break
        cv2.destroyAllWindows()

    def _create_match_visualization(self, result):
        """🎯 Create side-by-side match visualization"""
        if result.match_vis_data is None:
            # Fallback to normal display if no match data
            return self._create_batch_display(result)
        
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
            for i in range(len(mvd.matches)):
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
        
        # Add initial viewpoint estimation result for the current frame (right side)
        if result.viewpoint:
            cv2.putText(canvas, f'Est. Viewpoint: {result.viewpoint}', (w + 10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add final viewpoint used and fallback status
        if result.final_viewpoint_used:
            fallback_text = " (Fallback)" if result.fallback_occurred else ""
            cv2.putText(canvas, f'Final Anchor: {result.final_viewpoint_used}{fallback_text}', (w + 10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if result.fallback_occurred else (255, 255, 0), 2)

        # Match statistics
        total_matches = result.num_matches
        pose_matches = np.sum(mvd.used_for_pose) if mvd.used_for_pose is not None else 0
        
        cv2.putText(canvas, f'Total Matches: {total_matches}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f'Used for Pose: {pose_matches}', (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Experiment name based colors
        fallback_colors = {
            "baseline": (0, 255, 0), # Green
            "no_yolo": (0, 255, 255), # Yellow
            "no_viewpoint": (255, 165, 0), # Orange
            "orb_matcher": (255, 0, 255), # Magenta
            "no_kalman_filter": (0, 0, 255) # Red
        }
        color = fallback_colors.get(self.ablation_cfg['name'], (255, 255, 255))
        
        cv2.putText(canvas, f'Experiment: {self.ablation_cfg["name"].replace("_", " ").upper()}', (w + 10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw pose axes on the right side if available
        display_pos = result.kf_position if result.kf_position is not None else result.position
        display_quat = result.kf_quaternion if result.kf_quaternion is not None else result.quaternion

        if display_pos is not None and display_quat is not None:
            canvas = self._draw_axes_on_canvas(canvas, display_pos, display_quat, w, 0)
        
        # Kalman filter status
        if self.ablation_cfg['use_kalman_filter'] and result.kf_position is not None:
            kf_color = (0, 255, 0) # Always green if KF is used and position is available
            cv2.putText(canvas, 'KF: ON', (w + 10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, kf_color, 2)
        else:
            cv2.putText(canvas, 'KF: OFF', (w + 10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

        # Controls
        cv2.putText(canvas, 'Press Q to quit', 
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
    
    def _create_batch_display(self, result):
        """Creates a display frame for batch mode without match visualization."""
        vis_frame = result.frame.copy()
        
        # Experiment name based colors
        fallback_colors = {
            "baseline": (0, 255, 0),        # Green
            "no_yolo": (0, 255, 255),     # Yellow
            "no_viewpoint": (255, 165, 0), # Orange
            "orb_matcher": (255, 0, 255), # Magenta
            "no_kalman_filter": (0, 0, 255)      # Red
        }
        
        color = fallback_colors.get(self.ablation_cfg['name'], (255, 255, 255))
        
        cv2.putText(vis_frame, f'Frame: {result.frame_id}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis_frame, f'Experiment: {self.ablation_cfg["name"].replace("_", " ").upper()}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add initial viewpoint estimation result
        if result.viewpoint:
            cv2.putText(vis_frame, f'Est. Viewpoint: {result.viewpoint}', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add final viewpoint used and fallback status
        if result.final_viewpoint_used:
            fallback_text = " (Fallback)" if result.fallback_occurred else ""
            cv2.putText(vis_frame, f'Final Anchor: {result.final_viewpoint_used}{fallback_text}', (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if result.fallback_occurred else (255, 255, 0), 2)

        # Show bounding box
        if result.bbox:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

        # Show pose axes
        display_pos = result.kf_position if result.kf_position is not None else result.position
        display_quat = result.kf_quaternion if result.kf_quaternion is not None else result.quaternion

        if display_pos is not None and display_quat is not None:
            vis_frame = self._draw_axes_on_canvas(vis_frame, display_pos, display_quat)

        # Show match info
        if result.num_matches > 0:
            match_color = (0, 255, 0) if result.num_matches > 15 else (0, 165, 255) if result.num_matches > 8 else (0, 0, 255)
            cv2.putText(vis_frame, f'Matches: {result.num_matches}', (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)

        # Show Kalman filter status
        if self.ablation_cfg['use_kalman_filter'] and result.kf_position is not None:
            kf_color = (0, 255, 0) # Always green if KF is used and position is available
            cv2.putText(vis_frame, 'KF: ON', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, kf_color, 2)
        else:
            cv2.putText(vis_frame, 'KF: OFF', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # Add instructions
        cv2.putText(vis_frame, 'Press Q to quit', 
                   (10, vis_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis_frame


def save_comparison_table(results: List[Dict], output_dir: str):
    """Saves a formatted text file comparing the results of all experiments."""
    filepath = os.path.join(output_dir, "comparison_results.txt")
    
    header = (
        f"{'Experiment':<25} | {'Success Rate (%)':>20} | {'Successful Frames':>20} | {'Total Frames':>15} | {'Avg. FPS':>12}\n"
        f"{'-'*25}-+-{'-'*20}-+-{'-'*20}-+-{'-'*15}-+-{'-'*12}"
    )
    
    lines = [header]
    for r in results:
        rate = (r['success_count'] / r['total_count']) * 100 if r['total_count'] > 0 else 0
        line = (
            f"{r['name']:<25} | {rate:>19.2f} | "
            f"{r['success_count']:>20} | {r['total_count']:>15} | {r['fps']:>11.2f}"
        )
        lines.append(line)
        
    content = "\n".join(lines)
    print("\n" + "="*80)
    print("📊 Ablation Study Comparison Results")
    print("="*80)
    print(content)
    print("="*80)
    
    with open(filepath, 'w') as f:
        f.write("Ablation Study Comparison Results\n")
        f.write("="*100 + "\n")
        f.write(content)
        
    print(f"\n💾 Comparison table saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="VAPE MK47 Ablation Study Runner")
    parser.add_argument('--image_dir', type=str, help='Directory of images for batch processing.')
    parser.add_argument('--csv_file', type=str, help='CSV file with image timestamps.')
    parser.add_argument('--video_file', type=str, help='Path to an MP4 video file for processing.') # New argument
    parser.add_argument('--output_dir', type=str, default='ablation_results', help='Directory to save JSON results.')
    parser.add_argument('--no_display', action='store_true', help='Run in headless mode without display.') # Added
    parser.add_argument('--vis_match', action='store_true', help='Enable match visualization showing anchor and frame side-by-side with match lines.') # Added
    args = parser.parse_args()

    # Ensure at least one input method is provided
    if not (args.image_dir or args.csv_file or args.video_file):
        parser.error("At least one of --image_dir, --csv_file, or --video_file must be provided.")

    os.makedirs(args.output_dir, exist_ok=True)

    experiment_configs = [
        {"name": "baseline", "use_yolo": True, "use_viewpoint": True, "use_superpoint_lightglue": True, "use_kalman_filter": True},
        {"name": "no_yolo", "use_yolo": False, "use_viewpoint": True, "use_superpoint_lightglue": True, "use_kalman_filter": True},
        {"name": "no_viewpoint", "use_yolo": True, "use_viewpoint": False, "use_superpoint_lightglue": True, "use_kalman_filter": True},
        {"name": "orb_matcher", "use_yolo": True, "use_viewpoint": True, "use_superpoint_lightglue": False, "use_kalman_filter": True},
        {"name": "no_kalman_filter", "use_yolo": True, "use_viewpoint": True, "use_superpoint_lightglue": True, "use_kalman_filter": False},
    ]

    all_results = []
    for config in experiment_configs:
        estimator = AblationPoseEstimator(args, ablation_cfg=config)
        try:
            success_count, total_count, fps = estimator.run_experiment()
            all_results.append({
                "name": config['name'],
                "success_count": success_count,
                "total_count": total_count,
                "fps": fps
            })
        except Exception as e:
            print(f"💥 An error occurred during experiment '{config['name']}': {e}")
            import traceback
            traceback.print_exc()
        finally:
            del estimator
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # After all experiments are done, save the final comparison table
    if all_results:
        save_comparison_table(all_results, args.output_dir)

    print("\n🎉 All ablation experiments are complete.")

if __name__ == '__main__':
    main()




# import cv2
# import numpy as np
# import torch
# import time
# import argparse
# import warnings
# import json
# import csv
# import os
# from pathlib import Path
# from dataclasses import dataclass
# from typing import Optional, Tuple, Dict, List

# # Import for robust correspondence matching
# from scipy.spatial import cKDTree

# # Suppress warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# torch.set_grad_enabled(False)
# torch.autograd.set_grad_enabled(False)

# print("🚀 VAPE MK47 Ablation Study Runner")

# # Import required libraries
# try:
#     from ultralytics import YOLO
#     import timm
#     from torchvision import transforms
#     from lightglue import LightGlue, SuperPoint
#     from lightglue.utils import rbd
#     from PIL import Image
#     print("✅ All libraries loaded")
# except ImportError as e:
#     print(f"❌ Import error: {e}. Please ensure all dependencies are installed.")
#     exit(1)

# # --- DATA CLASSES ---
# @dataclass
# class ProcessingResult:
#     frame_id: int
#     timestamp: float
#     position: Optional[np.ndarray] = None
#     quaternion: Optional[np.ndarray] = None
#     pose_data: Optional[Dict] = None
#     kf_position: Optional[np.ndarray] = None
#     kf_quaternion: Optional[np.ndarray] = None

# # --- KALMAN FILTER ---
# class LooselyCoupledKalmanFilter:
#     def __init__(self, dt=1/30.0):
#         self.dt = dt
#         self.initialized = False
#         self.n_states = 13
#         self.x = np.zeros(self.n_states)
#         self.x[9] = 1.0
#         self.P = np.eye(self.n_states) * 0.1
#         self.Q = np.eye(self.n_states) * 1e-3
#         self.R = np.eye(7) * 1e-4

#     def normalize_quaternion(self, q):
#         norm = np.linalg.norm(q)
#         return q / norm if norm > 1e-10 else np.array([0, 0, 0, 1])

#     def predict(self):
#         if not self.initialized: return None, None
#         px, py, pz, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz = self.x
#         dt = self.dt
#         self.x[0:3] += self.x[3:6] * dt
#         q = np.array([qx, qy, qz, qw])
#         w = np.array([wx, wy, wz])
#         omega_mat = np.array([[0, -wx, -wy, -wz], [wx, 0, wz, -wy], [wy, -wz, 0, wx], [wz, wy, -wx, 0]])
#         q_new = self.normalize_quaternion(q + 0.5 * dt * omega_mat @ q)
#         self.x[6:10] = q_new
#         F = np.eye(self.n_states); F[0:3, 3:6] = np.eye(3) * dt
#         self.P = F @ self.P @ F.T + self.Q
#         return self.x[0:3], self.x[6:10]

#     def update(self, position, quaternion):
#         measurement = np.concatenate([position, self.normalize_quaternion(quaternion)])
#         if not self.initialized:
#             self.x[0:3], self.x[6:10] = position, measurement[3:7]
#             self.initialized = True
#             return self.x[0:3], self.x[6:10]
#         predicted_measurement = np.concatenate([self.x[0:3], self.x[6:10]])
#         innovation = measurement - predicted_measurement
#         if np.dot(measurement[3:7], predicted_measurement[3:7]) < 0:
#             innovation[3:7] = -measurement[3:7] - predicted_measurement[3:7]
#         H = np.zeros((7, self.n_states)); H[0:3, 0:3], H[3:7, 6:10] = np.eye(3), np.eye(4)
#         S = H @ self.P @ H.T + self.R
#         K = self.P @ H.T @ np.linalg.inv(S)
#         self.x += K @ innovation
#         self.x[6:10] = self.normalize_quaternion(self.x[6:10])
#         self.P = (np.eye(self.n_states) - K @ H) @ self.P
#         return self.x[0:3], self.x[6:10]

# # --- HELPER FUNCTIONS ---
# def read_image_index_csv(csv_path):
#     entries = []
#     with open(csv_path, 'r', newline='', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             entries.append({'index': int(row['Index']),'timestamp': float(row['Timestamp']), 'filename': row['Filename']})
#     return entries

# def convert_to_json_serializable(obj):
#     if isinstance(obj, (np.integer, np.floating)): return obj.item()
#     if isinstance(obj, np.ndarray): return obj.tolist()
#     if isinstance(obj, dict): return {k: convert_to_json_serializable(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)): return [convert_to_json_serializable(i) for i in obj]
#     return obj

# def create_unique_filename(directory, base_filename):
#     base_path = Path(directory or ".") / base_filename
#     if not base_path.exists(): return str(base_path)
#     name, ext = base_path.stem, base_path.suffix
#     counter = 1
#     while True:
#         new_path = base_path.with_name(f"{name}_{counter}{ext}")
#         if not new_path.exists(): return str(new_path)
#         counter += 1

# # --- MAIN ESTIMATOR CLASS (Modified for Ablation) ---
# class AblationPoseEstimator:
#     def __init__(self, args, ablation_cfg: Dict):
#         print("-" * 60)
#         print(f"🔧 Initializing Estimator for Experiment: {ablation_cfg['name']}")
#         print(f"   - YOLO Enabled: {ablation_cfg['use_yolo']}")
#         print(f"   - Viewpoint Estimation Enabled: {ablation_cfg['use_viewpoint']}")
#         print(f"   - Matcher: {'SuperPoint/LightGlue' if ablation_cfg['use_superpoint_lightglue'] else 'ORB/BFMatcher'}")
#         print(f"   - Kalman Filter Enabled: {ablation_cfg['use_kalman_filter']}")
#         print("-" * 60)

#         self.args = args
#         self.ablation_cfg = ablation_cfg
#         self.image_entries = []
#         self.video_capture = None # Added for video input
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         print(f"🚀 Using device: {self.device}")
#         self.camera_width, self.camera_height = 1280, 720
#         self.all_poses = []

#         if self.ablation_cfg['use_kalman_filter']:
#             self.kf = LooselyCoupledKalmanFilter(dt=1/30.0)

#         self._init_models()
#         self._init_batch_processing()
#         self._init_anchor_data()
#         print("✅ Estimator initialized successfully!")

#     def _init_batch_processing(self):
#         if self.args.video_file and os.path.exists(self.args.video_file): # New: Handle video file input
#             self.video_capture = cv2.VideoCapture(self.args.video_file)
#             if not self.video_capture.isOpened():
#                 raise ValueError(f"Could not open video file: {self.args.video_file}")
#             total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
#             fps = self.video_capture.get(cv2.CAP_PROP_FPS)
#             self.image_entries = [{'index': i, 'timestamp': i/fps, 'filename': f"frame_{i:06d}.png"} for i in range(total_frames)]
#             print(f"✅ Loaded {len(self.image_entries)} frames from video: {self.args.video_file}.")
#         elif self.args.csv_file and os.path.exists(self.args.csv_file):
#             self.image_entries = read_image_index_csv(self.args.csv_file)
#             self.image_entries.sort(key=lambda x: x['index'])
#             print(f"✅ Loaded {len(self.image_entries)} image entries from CSV.")
#         elif self.args.image_dir and os.path.exists(self.args.image_dir):
#             image_files = sorted([f for f in os.listdir(self.args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
#             self.image_entries = [{'index': i, 'timestamp': i/30.0, 'filename': f} for i, f in enumerate(image_files)]
#             print(f"✅ Found {len(self.image_entries)} images in directory.")
#         else:
#             raise ValueError("Must provide a valid --image_dir, --csv_file, or --video_file.")

#     def _init_models(self):
#         try:
#             if self.ablation_cfg['use_yolo']:
#                 print("  📦 Loading YOLO...")
#                 self.yolo_model = YOLO("yolov8s.pt").to(self.device)
#             if self.ablation_cfg['use_viewpoint']:
#                 print("  📦 Loading viewpoint classifier (MobileViT)...")
#                 self.vp_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=4)
#                 vp_model_path = 'mobilevit_viewpoint_20250703.pth'
#                 if os.path.exists(vp_model_path):
#                     self.vp_model.load_state_dict(torch.load(vp_model_path, map_location=self.device))
#                 else:
#                     print(f"  ⚠️ Viewpoint model file not found at '{vp_model_path}'.")
#                 self.vp_model.eval().to(self.device)
#                 self.vp_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)])
#                 self.class_names = ['NE', 'NW', 'SE', 'SW']
#             if self.ablation_cfg['use_superpoint_lightglue']:
#                 print("  📦 Loading SuperPoint & LightGlue...")
#                 self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
#                 self.matcher = LightGlue(features="superpoint").eval().to(self.device)
#             else:
#                 print("  📦 Initializing ORB & BFMatcher...")
#                 self.orb = cv2.ORB_create(nfeatures=2000)
#                 self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#         except Exception as e:
#             print(f"❌ Model loading failed: {e}")
#             raise

#     def _init_anchor_data(self):
#         print("🛠️ Initializing anchor data...")
        
#         # Complete 2D and 3D point data from your original VAPE_MK47.py script
#         ne_anchor_2d = np.array([[924, 148], [571, 115], [398, 31], [534, 133], [544, 141], [341, 219], [351, 228], [298, 240], [420, 83], [225, 538], [929, 291], [794, 381], [485, 569], [826, 305], [813, 264], [791, 285], [773, 271], [760, 289], [830, 225], [845, 233], [703, 308], [575, 361], [589, 373], [401, 469], [414, 481], [606, 454], [548, 399], [521, 510], [464, 451], [741, 380]], dtype=np.float32)
#         ne_anchor_3d = np.array([[-0.0, -0.025, -0.24], [0.23, 0.0, -0.113], [0.243, -0.104, 0.0], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [0.23, 0.0, 0.113], [0.206, -0.07, -0.002], [-0.0, -0.025, 0.24], [-0.08, 0.0, -0.156], [-0.09, 0.0, -0.042], [-0.08, 0.0, 0.156], [-0.052, 0.0, -0.097], [-0.029, 0.0, -0.127], [-0.037, 0.0, -0.097], [-0.017, 0.0, -0.092], [-0.023, 0.0, -0.075], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [-0.074, 0.0, 0.074], [-0.019, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.019, 0.0, 0.128], [-0.1, -0.03, 0.0]], dtype=np.float32)
#         nw_anchor_2d = np.array([[511, 293], [591, 284], [587, 330], [413, 249], [602, 348], [715, 384], [598, 298], [656, 171], [805, 213], [703, 392], [523, 286], [519, 327], [387, 289], [727, 126], [425, 243], [636, 358], [745, 202], [595, 388], [436, 260], [539, 313], [795, 220], [351, 291], [665, 165], [611, 353], [650, 377], [516, 389], [727, 143], [496, 378], [575, 312], [617, 368], [430, 312], [480, 281], [834, 225], [469, 339], [705, 223], [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]], dtype=np.float32)
#         nw_anchor_3d = np.array([[-0.014, 0.0, 0.042], [0.025, -0.014, -0.011], [-0.014, 0.0, -0.042], [-0.014, 0.0, 0.156], [-0.023, 0.0, -0.065], [0.0, 0.0, -0.156], [0.025, 0.0, -0.015], [0.217, 0.0, 0.07], [0.23, 0.0, -0.07], [-0.014, 0.0, -0.156], [0.0, 0.0, 0.042], [-0.057, -0.018, -0.01], [-0.074, -0.0, 0.128], [0.206, -0.07, -0.002], [-0.0, -0.0, 0.156], [-0.017, -0.0, -0.092], [0.217, -0.0, -0.027], [-0.052, -0.0, -0.097], [-0.019, -0.0, 0.128], [-0.035, -0.018, -0.01], [0.217, -0.0, -0.07], [-0.08, -0.0, 0.156], [0.23, 0.0, 0.07], [-0.023, -0.0, -0.075], [-0.029, -0.0, -0.127], [-0.09, -0.0, -0.042], [0.206, -0.055, -0.002], [-0.09, -0.0, -0.015], [0.0, -0.0, -0.015], [-0.037, -0.0, -0.097], [-0.074, -0.0, 0.074], [-0.019, -0.0, 0.074], [0.23, -0.0, -0.113], [-0.1, -0.03, 0.0], [0.17, -0.0, -0.015], [0.23, -0.0, 0.113], [-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [-0.08, -0.0, -0.156]], dtype=np.float32)
#         se_anchor_2d = np.array([[415, 144], [1169, 508], [275, 323], [214, 395], [554, 670], [253, 428], [280, 415], [355, 365], [494, 621], [519, 600], [806, 213], [973, 438], [986, 421], [768, 343], [785, 328], [841, 345], [931, 393], [891, 306], [980, 345], [651, 210], [625, 225], [588, 216], [511, 215], [526, 204], [665, 271]], dtype=np.float32)
#         se_anchor_3d = np.array([[-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [0.23, 0.0, -0.113], [0.23, 0.0, 0.113], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.206, -0.07, -0.002], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [-0.1, -0.03, 0.0], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.019, 0.0, 0.074], [-0.019, 0.0, 0.128], [-0.074, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.052, 0.0, -0.097], [-0.037, 0.0, -0.097], [-0.029, 0.0, -0.127], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042]], dtype=np.float32)
#         sw_anchor_2d = np.array([[650, 312], [630, 306], [907, 443], [814, 291], [599, 349], [501, 386], [965, 359], [649, 355], [635, 346], [930, 335], [843, 467], [702, 339], [718, 321], [930, 322], [727, 346], [539, 364], [786, 297], [1022, 406], [1004, 399], [539, 344], [536, 309], [864, 478], [745, 310], [1049, 393], [895, 258], [674, 347], [741, 281], [699, 294], [817, 494], [992, 281]], dtype=np.float32)
#         sw_anchor_3d = np.array([[-0.035, -0.018, -0.01], [-0.057, -0.018, -0.01], [0.217, -0.0, -0.027], [-0.014, -0.0, 0.156], [-0.023, -0.0, -0.065], [-0.014, -0.0, -0.156], [0.234, -0.05, -0.002], [0.0, -0.0, -0.042], [-0.014, -0.0, -0.042], [0.206, -0.055, -0.002], [0.217, -0.0, -0.07], [0.025, -0.014, -0.011], [-0.014, -0.0, 0.042], [0.206, -0.07, -0.002], [0.049, -0.016, -0.011], [-0.029, -0.0, -0.127], [-0.019, -0.0, 0.128], [0.23, -0.0, 0.07], [0.217, -0.0, 0.07], [-0.052, -0.0, -0.097], [-0.175, -0.0, -0.015], [0.23, -0.0, -0.07], [-0.019, -0.0, 0.074], [0.23, -0.0, 0.113], [-0.0, -0.025, 0.24], [-0.0, -0.0, -0.015], [-0.074, -0.0, 0.128], [-0.074, -0.0, 0.074], [0.23, -0.0, -0.113], [0.243, -0.104, 0.0]], dtype=np.float32)

#         # The complete anchor definitions dictionary
#         anchor_definitions = {
#             'NE': {'path': 'NE.png', '2d': ne_anchor_2d, '3d': ne_anchor_3d},
#             'NW': {'path': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png', '2d': nw_anchor_2d, '3d': nw_anchor_3d},
#             'SE': {'path': 'SE.png', '2d': se_anchor_2d, '3d': se_anchor_3d},
#             'SW': {'path': 'Anchor_B.png', '2d': sw_anchor_2d, '3d': sw_anchor_3d}
#         }
        
#         self.viewpoint_anchors = {}
#         for viewpoint, data in anchor_definitions.items():
#             print(f"  📸 Processing anchor for {viewpoint}...")
#             anchor_image_bgr = self._load_anchor_image(data['path'])
#             if anchor_image_bgr is None:
#                 print(f"     -> Failed to load image for {viewpoint}. Skipping.")
#                 continue

#             if self.ablation_cfg['use_superpoint_lightglue']:
#                 anchor_features = self._extract_features_sp(anchor_image_bgr)
#                 anchor_keypoints_sp = anchor_features['keypoints'][0].cpu().numpy()
#                 if len(anchor_keypoints_sp) == 0:
#                     print(f"     -> No SuperPoint keypoints found for {viewpoint}. Skipping.")
#                     continue
#                 sp_tree = cKDTree(anchor_keypoints_sp)
#                 distances, indices = sp_tree.query(data['2d'], k=1, distance_upper_bound=5.0)
#                 valid_mask = distances != np.inf
                
#                 self.viewpoint_anchors[viewpoint] = {
#                     'features': anchor_features,
#                     'matched_sp_indices': indices[valid_mask],
#                     'matched_3d_points_map': {idx: pt for idx, pt in zip(indices[valid_mask], data['3d'][valid_mask])}
#                 }
#                 print(f"     -> Processed with SuperPoint. Found {len(self.viewpoint_anchors[viewpoint]['matched_3d_points_map'])} correspondences.")
#             else: # ORB Matcher
#                 gray_anchor = cv2.cvtColor(anchor_image_bgr, cv2.COLOR_BGR2GRAY)
#                 kp_orb, des_orb = self.orb.detectAndCompute(gray_anchor, None)
#                 if des_orb is None:
#                     print(f"     -> No ORB features found for {viewpoint}. Skipping.")
#                     continue
#                 orb_kpts_coords = np.array([kp.pt for kp in kp_orb])
#                 kptree = cKDTree(orb_kpts_coords)
#                 distances, indices = kptree.query(data['2d'], k=1, distance_upper_bound=8.0)
#                 valid_mask = distances != np.inf
                
#                 self.viewpoint_anchors[viewpoint] = {
#                     'orb_kp': [kp_orb[i] for i in indices[valid_mask]],
#                     'orb_des': des_orb[indices[valid_mask]],
#                     'orb_3d_points': data['3d'][valid_mask]
#                 }
#                 print(f"     -> Processed with ORB. Found {len(self.viewpoint_anchors[viewpoint]['orb_kp'])} correspondences.")
                
#         print("✅ Anchor data initialization complete.")

#     def _load_anchor_image(self, path):
#         if not os.path.exists(path): return None
#         return cv2.resize(cv2.imread(path), (self.camera_width, self.camera_height))

#     def _extract_features_sp(self, image_bgr):
#         rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#         tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
#         with torch.no_grad(): return self.extractor.extract(tensor)

#     def _detect_features_orb(self, image_bgr):
#         gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#         return self.orb.detectAndCompute(gray, None)

#     def run_experiment(self):
#         if not self.image_entries: return 0, 0, 0
#         print(f"\n▶️ Running Experiment: {self.ablation_cfg['name']}")
#         start_time = time.time()
#         success_count = 0
        
#         total_frames_to_process = len(self.image_entries) # Renamed for clarity

#         for i, entry in enumerate(self.image_entries):
#             frame = None
#             if self.video_capture: # If video input
#                 ret, frame = self.video_capture.read()
#                 if not ret:
#                     print(f"  Warning: Failed to read frame {i} from video or end of video reached.")
#                     break # Exit loop if frame cannot be read
#             else: # If image directory or CSV
#                 image_path = os.path.join(self.args.image_dir, entry['filename']) if self.args.image_dir else entry['filename']
#                 if not os.path.exists(image_path):
#                     print(f"  Warning: Image file not found: {image_path}. Skipping.")
#                     continue
#                 frame = cv2.imread(image_path)
            
#             if frame is None:
#                 continue # Skip if frame couldn't be loaded/read

#             result = self._process_frame(frame, entry['index'], entry['timestamp'], entry)
#             if result.pose_data and not result.pose_data.get('pose_estimation_failed', True):
#                 success_count += 1
#             self.all_poses.append(convert_to_json_serializable(result.pose_data))
#             if (i + 1) % 100 == 0: print(f"  Processed {i+1}/{total_frames_to_process} images...")
        
#         total_time = time.time() - start_time
#         fps = total_frames_to_process / total_time if total_time > 0 else 0
#         success_rate = (success_count / total_frames_to_process) * 100 if total_frames_to_process > 0 else 0

#         print(f"✅ Experiment '{self.ablation_cfg['name']}' Complete.")
#         print(f"   - Success Rate: {success_rate:.2f}% ({success_count}/{total_frames_to_process})")
#         print(f"   - Average FPS: {fps:.2f}")
        
#         if self.video_capture:
#             self.video_capture.release() # Release video capture object
        
#         self.save_results()
#         return success_count, total_frames_to_process, fps

#     def _process_frame(self, frame, frame_id, timestamp, frame_info):
#         result = ProcessingResult(frame_id=frame_id, timestamp=timestamp)
#         if self.ablation_cfg['use_kalman_filter'] and self.kf.initialized:
#             result.kf_position, result.kf_quaternion = self.kf.predict()
        
#         bbox = self._yolo_detect(frame) if self.ablation_cfg['use_yolo'] else None
#         if self.ablation_cfg['use_viewpoint']:
#             # Ensure bbox is not None before attempting to slice
#             if bbox and bbox[3] > bbox[1] and bbox[2] > bbox[0]:
#                 crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#             else:
#                 crop = frame # Use full frame if no valid bbox or YOLO is disabled
#             viewpoint = self._classify_viewpoint(crop)
#         else:
#             viewpoint = 'NW'
            
#         pos, quat, pose_data = self._estimate_pose(frame, viewpoint, bbox, frame_id, frame_info)
        
#         if pos is not None and self.ablation_cfg['use_kalman_filter']:
#             kf_pos, kf_quat = self.kf.update(pos, quat)
#             if pose_data and not pose_data.get('pose_estimation_failed', True):
#                 pose_data['position_kf'] = kf_pos.tolist()
#                 pose_data['quaternion_kf'] = kf_quat.tolist()
        
#         result.position, result.quaternion, result.pose_data = pos, quat, pose_data
#         return result

#     def _estimate_pose(self, full_frame, viewpoint, bbox, frame_id, frame_info):
#         if viewpoint not in self.viewpoint_anchors:
#             return None, None, self._create_failure_report(frame_id, 'invalid_viewpoint')
        
#         anchor_data, crop_offset = self.viewpoint_anchors[viewpoint], np.array([0, 0])
#         # Ensure bbox is valid before cropping
#         if bbox and bbox[3] > bbox[1] and bbox[2] > bbox[0]:
#             crop = full_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#             crop_offset = np.array([bbox[0], bbox[1]])
#         else:
#             crop = full_frame # Use full frame if no valid bbox
            
#         if crop.size == 0: return None, None, self._create_failure_report(frame_id, 'invalid_crop')
        
#         if self.ablation_cfg['use_superpoint_lightglue']:
#             return self._estimate_pose_sp(crop, crop_offset, anchor_data, frame_id, frame_info)
#         else:
#             return self._estimate_pose_orb(crop, crop_offset, anchor_data, frame_id, frame_info)
    
#     def _estimate_pose_sp(self, crop, crop_offset, anchor_data, frame_id, frame_info):
#         frame_features = self._extract_features_sp(crop)
#         with torch.no_grad(): matches_dict = self.matcher({'image0': anchor_data['features'], 'image1': frame_features})
#         matches = rbd(matches_dict)['matches'].cpu().numpy()
#         if len(matches) < 6: return None, None, self._create_failure_report(frame_id, 'insufficient_matches')
            
#         valid_anchor_sp_indices = matches[:, 0]
#         # Only consider anchor points that actually have 3D data defined.
#         # matched_3d_points_map stores keys as indices from the *original* anchor 2D points that mapped to SuperPoint KPs,
#         # so we need to ensure the `valid_anchor_sp_indices` are in those *original* anchor 2D point indices that were valid.
#         # This part of the logic needs to correctly map from matched SuperPoint index back to original anchor 2D index for 3D lookup.
#         # The current `matched_3d_points_map` keys are `indices[valid_mask]` which are SuperPoint keypoint indices,
#         # NOT the original 2D point indices from `data['2d']`. This is a subtle but important distinction.

#         # Correct approach: When creating `matched_3d_points_map`, map SuperPoint keypoint index to its corresponding 3D point.
#         # Then, when doing the lookup, use the SuperPoint keypoint index from `matches[:, 0]`.

#         # Re-evaluating the creation of `matched_3d_points_map` in `_init_anchor_data`:
#         # `indices[valid_mask]` are the INDICES OF SUPERPOINT KEYPOINTS that are close to `data['2d']`.
#         # `data['3d'][valid_mask]` are the 3D points corresponding to the `data['2d']` that were matched.
#         # So, `matched_3d_points_map` maps a SuperPoint keypoint index to a 3D point. This is correct.

#         # So, the check for `mask = np.isin(valid_anchor_sp_indices, list(anchor_data['matched_3d_points_map'].keys()))`
#         # is correctly ensuring that the matched SuperPoint keypoint *index* exists as a key in the map.
        
#         # However, the `points_3d` extraction needs to use the correct mapping.
#         # `valid_anchor_sp_indices` refers to the indices of the SuperPoint keypoints from the anchor image.
#         # `anchor_data['matched_3d_points_map']` maps these SuperPoint keypoint indices to their 3D points.
        
#         points_3d = []
#         points_2d_full = []
        
#         for anchor_sp_idx, frame_sp_idx in matches:
#             if anchor_sp_idx in anchor_data['matched_3d_points_map']:
#                 points_3d.append(anchor_data['matched_3d_points_map'][anchor_sp_idx])
#                 points_2d_full.append(frame_features['keypoints'][0].cpu().numpy()[frame_sp_idx] + crop_offset)
        
#         points_3d = np.array(points_3d, dtype=np.float32)
#         points_2d_full = np.array(points_2d_full, dtype=np.float32)
        
#         if len(points_3d) < 6: return None, None, self._create_failure_report(frame_id, 'insufficient_correspondences_after_mapping')
            
#         return self._solve_pnp(points_3d, points_2d_full, frame_id, frame_info)

#     def _estimate_pose_orb(self, crop, crop_offset, anchor_data, frame_id, frame_info):
#         kp_frame, des_frame = self._detect_features_orb(crop)
#         if des_frame is None or len(des_frame) < 6: return None, None, self._create_failure_report(frame_id, 'insufficient_features')
        
#         matches = self.bf_matcher.match(anchor_data['orb_des'], des_frame)
#         if len(matches) < 6: return None, None, self._create_failure_report(frame_id, 'insufficient_matches')
            
#         points_3d = np.array([anchor_data['orb_3d_points'][m.queryIdx] for m in matches])
#         points_2d_full = np.array([kp_frame[m.trainIdx].pt for m in matches]) + crop_offset
#         return self._solve_pnp(points_3d, points_2d_full, frame_id, frame_info)

#     def _solve_pnp(self, points_3d, points_2d, frame_id, frame_info):
#         K, dist_coeffs = self._get_camera_intrinsics()
#         try:
#             success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
#         except cv2.error: return None, None, self._create_failure_report(frame_id, 'pnp_cv_error')

#         if not success or inliers is None or len(inliers) < 4:
#             return None, None, self._create_failure_report(frame_id, 'pnp_failed', num_inliers=len(inliers) if inliers is not None else 0)
        
#         R, _ = cv2.Rodrigues(rvec)
#         position, quaternion = tvec.flatten(), self._rotation_matrix_to_quaternion(R)
        
#         projected, _ = cv2.projectPoints(points_3d[inliers.flatten()], rvec, tvec, K, dist_coeffs)
#         error = np.mean(np.linalg.norm(points_2d[inliers.flatten()].reshape(-1, 1, 2) - projected, axis=2))
        
#         return position, quaternion, self._create_success_report(frame_id, position, quaternion, len(inliers), error, frame_info)

#     def _yolo_detect(self, frame):
#         # Resize frame to a common YOLO input size if it's too large, for faster detection
#         # This is an optional optimization, but good practice for YOLO.
#         # For simplicity, will use the default YOLO behavior which resizes internally.
#         results = self.yolo_model(frame, verbose=False, conf=0.4)
#         if len(results[0].boxes) > 0: return tuple(map(int, results[0].boxes.xyxy.cpu().numpy()[0]))
#         return None

#     def _classify_viewpoint(self, crop):
#         # Ensure crop is not empty before processing
#         if crop.shape[0] == 0 or crop.shape[1] == 0:
#             print("  Warning: Empty crop received for viewpoint classification. Defaulting to 'NW'.")
#             return 'NW'
        
#         pil_img = Image.fromarray(cv2.cvtColor(cv2.resize(crop, (128, 128)), cv2.COLOR_BGR2RGB))
#         tensor = self.vp_transform(pil_img).unsqueeze(0).to(self.device)
#         with torch.no_grad(): return self.class_names[torch.argmax(self.vp_model(tensor), dim=1).item()]

#     def _get_camera_intrinsics(self):
#         fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
#         return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32), None

#     def _rotation_matrix_to_quaternion(self, R):
#         # This function converts a rotation matrix to a quaternion.
#         # It has been corrected to fix the UnboundLocalError.
#         tr = np.trace(R)
        
#         if tr > 0:
#             S = np.sqrt(tr + 1.0) * 2
#             qw = 0.25 * S
#             qx = (R[2, 1] - R[1, 2]) / S
#             qy = (R[0, 2] - R[2, 0]) / S
#             qz = (R[1, 0] - R[0, 1]) / S
#         elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
#             S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
#             qw = (R[2, 1] - R[1, 2]) / S
#             qx = 0.25 * S
#             qy = (R[0, 1] + R[1, 0]) / S
#             qz = (R[0, 2] + R[2, 0]) / S
#         elif R[1, 1] > R[2, 2]:
#             S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
#             qw = (R[0, 2] - R[2, 0]) / S
#             qx = (R[0, 1] + R[1, 0]) / S
#             qy = 0.25 * S
#             qz = (R[1, 2] + R[2, 1]) / S
#         else:
#             S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
#             qw = (R[1, 0] - R[0, 1]) / S
#             qx = (R[0, 2] + R[2, 0]) / S
#             qy = (R[1, 2] + R[2, 1]) / S
#             qz = 0.25 * S
            
#         # The convention is [x, y, z, w]
#         return np.array([qx, qy, qz, qw])

#     def _create_failure_report(self, frame_id, reason, **kwargs):
#         return {'frame': frame_id, 'pose_estimation_failed': True, 'error_reason': reason, **kwargs}

#     def _create_success_report(self, frame_id, pos, quat, num_inliers, error, frame_info):
#         data = {'frame': frame_id, 'pose_estimation_failed': False, 'position': pos, 'quaternion': quat,
#                 'num_inliers': num_inliers, 'mean_reprojection_error': error}
#         if frame_info: data.update(frame_info)
#         return data

#     def save_results(self):
#         output_filename = create_unique_filename(self.args.output_dir, f"poses_{self.ablation_cfg['name']}.json")
#         print(f"💾 Saving {len(self.all_poses)} records to {output_filename}")
#         with open(output_filename, 'w') as f:
#             json.dump(self.all_poses, f, indent=2)

# def save_comparison_table(results: List[Dict], output_dir: str):
#     """Saves a formatted text file comparing the results of all experiments."""
#     filepath = os.path.join(output_dir, "comparison_results.txt")
    
#     header = (
#         f"{'Experiment':<25} | {'Success Rate (%)':>20} | {'Successful Frames':>20} | {'Total Frames':>15} | {'Avg. FPS':>12}\n"
#         f"{'-'*25}-+-{'-'*20}-+-{'-'*20}-+-{'-'*15}-+-{'-'*12}"
#     )
    
#     lines = [header]
#     for r in results:
#         rate = (r['success_count'] / r['total_count']) * 100 if r['total_count'] > 0 else 0
#         line = (
#             f"{r['name']:<25} | {rate:>19.2f} | "
#             f"{r['success_count']:>20} | {r['total_count']:>15} | {r['fps']:>11.2f}"
#         )
#         lines.append(line)
        
#     content = "\n".join(lines)
#     print("\n" + "="*80)
#     print("📊 Ablation Study Comparison Results")
#     print("="*80)
#     print(content)
#     print("="*80)
    
#     with open(filepath, 'w') as f:
#         f.write("Ablation Study Comparison Results\n")
#         f.write("="*100 + "\n")
#         f.write(content)
        
#     print(f"\n💾 Comparison table saved to {filepath}")

# def main():
#     parser = argparse.ArgumentParser(description="VAPE MK47 Ablation Study Runner")
#     parser.add_argument('--image_dir', type=str, help='Directory of images for batch processing.')
#     parser.add_argument('--csv_file', type=str, help='CSV file with image timestamps.')
#     parser.add_argument('--video_file', type=str, help='Path to an MP4 video file for processing.') # New argument
#     parser.add_argument('--output_dir', type=str, default='ablation_results', help='Directory to save JSON results.')
#     args = parser.parse_args()

#     # Ensure at least one input method is provided
#     if not (args.image_dir or args.csv_file or args.video_file):
#         parser.error("At least one of --image_dir, --csv_file, or --video_file must be provided.")

#     os.makedirs(args.output_dir, exist_ok=True) 

#     experiment_configs = [
#         {"name": "baseline", "use_yolo": True, "use_viewpoint": True, "use_superpoint_lightglue": True, "use_kalman_filter": True},
#         {"name": "no_yolo", "use_yolo": False, "use_viewpoint": True, "use_superpoint_lightglue": True, "use_kalman_filter": True},
#         {"name": "no_viewpoint", "use_yolo": True, "use_viewpoint": False, "use_superpoint_lightglue": True, "use_kalman_filter": True},
#         {"name": "orb_matcher", "use_yolo": True, "use_viewpoint": True, "use_superpoint_lightglue": False, "use_kalman_filter": True},
#         {"name": "no_kalman_filter", "use_yolo": True, "use_viewpoint": True, "use_superpoint_lightglue": True, "use_kalman_filter": False},
#     ]

#     all_results = []
#     for config in experiment_configs:
#         estimator = AblationPoseEstimator(args, ablation_cfg=config)
#         try:
#             success_count, total_count, fps = estimator.run_experiment()
#             all_results.append({
#                 "name": config['name'],
#                 "success_count": success_count,
#                 "total_count": total_count,
#                 "fps": fps
#             })
#         except Exception as e:
#             print(f"💥 An error occurred during experiment '{config['name']}': {e}")
#             import traceback
#             traceback.print_exc()
#         finally:
#             del estimator
#             if torch.cuda.is_available(): torch.cuda.empty_cache()
    
#     # After all experiments are done, save the final comparison table
#     if all_results:
#         save_comparison_table(all_results, args.output_dir)

#     print("\n🎉 All ablation experiments are complete.")

# if __name__ == '__main__':
#     main()