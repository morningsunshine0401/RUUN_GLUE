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

@dataclass
class PoseData:
    """A simple container for pose results."""
    position: np.ndarray
    quaternion: np.ndarray
    inliers: int
    reprojection_error: float
    viewpoint: str

# --- KALMAN FILTER ---
class LooselyCoupledKalmanFilter:
    """A simple Kalman Filter for smoothing pose data."""
    def __init__(self, dt=1/30.0):
        self.dt = dt
        self.initialized = False
        self.n_states = 13  # [pos(3), vel(3), quat(4), ang_vel(3)]
        self.x = np.zeros(self.n_states)
        self.x[9] = 1.0  # Identity quaternion
        self.P = np.eye(self.n_states) * 0.1
        self.Q = np.eye(self.n_states) * 1e-3
        self.R = np.eye(7) * 1e-4

    def normalize_quaternion(self, q):
        norm = np.linalg.norm(q)
        return q / norm if norm > 1e-10 else np.array([0, 0, 0, 1])

    def predict(self):
        if not self.initialized: return None, None
        dt = self.dt
        F = np.eye(self.n_states)
        F[0:3, 3:6] = np.eye(3) * dt
        self.x = F @ self.x
        q, w = self.x[6:10], self.x[10:13]
        omega_mat = 0.5 * np.array([[0, -w[0], -w[1], -w[2]], [w[0], 0, w[2], -w[1]], [w[1], -w[2], 0, w[0]], [w[2], w[1], -w[0], 0]])
        self.x[6:10] = self.normalize_quaternion((np.eye(4) + dt * omega_mat) @ q)
        self.P = F @ self.P @ F.T + self.Q
        return self.x[0:3], self.x[6:10]

    def update(self, position, quaternion):
        measurement = np.concatenate([position, self.normalize_quaternion(quaternion)])
        if not self.initialized:
            self.x[0:3], self.x[6:10] = position, measurement[3:7]
            self.initialized = True
            return self.x[0:3], self.x[6:10]
        
        H = np.zeros((7, self.n_states))
        H[0:3, 0:3], H[3:7, 6:10] = np.eye(3), np.eye(4)
        
        innovation = measurement - H @ self.x
        if np.dot(measurement[3:7], self.x[6:10]) < 0:
            measurement[3:7] *= -1 # Handle quaternion sign ambiguity
            innovation = measurement - H @ self.x

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ innovation
        self.x[6:10] = self.normalize_quaternion(self.x[6:10])
        self.P = (np.eye(self.n_states) - K @ H) @ self.P
        return self.x[0:3], self.x[6:10]

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
        self.class_names = ['NE', 'NW', 'SE', 'SW']
        
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        print("   ...models loaded.")

    def _initialize_anchor_data(self):
        """Pre-processes anchor images and their 2D-3D correspondences."""
        print("🛠️  Initializing anchor data...")
        # Define 2D-3D correspondences for each anchor viewpoint
        ne_anchor_2d = np.array([[924, 148], [571, 115], [398, 31], [534, 133], [544, 141], [341, 219], [351, 228], [298, 240], [420, 83], [225, 538], [929, 291], [794, 381], [485, 569], [826, 305], [813, 264], [791, 285], [773, 271], [760, 289], [830, 225], [845, 233], [703, 308], [575, 361], [589, 373], [401, 469], [414, 481], [606, 454], [548, 399], [521, 510], [464, 451], [741, 380]], dtype=np.float32)
        ne_anchor_3d = np.array([[-0.0, -0.025, -0.24], [0.23, 0.0, -0.113], [0.243, -0.104, 0.0], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [0.23, 0.0, 0.113], [0.206, -0.07, -0.002], [-0.0, -0.025, 0.24], [-0.08, 0.0, -0.156], [-0.09, 0.0, -0.042], [-0.08, 0.0, 0.156], [-0.052, 0.0, -0.097], [-0.029, 0.0, -0.127], [-0.037, 0.0, -0.097], [-0.017, 0.0, -0.092], [-0.023, 0.0, -0.075], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [-0.074, 0.0, 0.074], [-0.019, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.019, 0.0, 0.128], [-0.1, -0.03, 0.0]], dtype=np.float32)
        nw_anchor_2d = np.array([[511, 293], [591, 284], [587, 330], [413, 249], [602, 348], [715, 384], [598, 298], [656, 171], [805, 213], [703, 392], [523, 286], [519, 327], [387, 289], [727, 126], [425, 243], [636, 358], [745, 202], [595, 388], [436, 260], [539, 313], [795, 220], [351, 291], [665, 165], [611, 353], [650, 377], [516, 389], [727, 143], [496, 378], [575, 312], [617, 368], [430, 312], [480, 281], [834, 225], [469, 339], [705, 223], [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]], dtype=np.float32)
        nw_anchor_3d = np.array([[-0.014, 0.0, 0.042], [0.025, -0.014, -0.011], [-0.014, 0.0, -0.042], [-0.014, 0.0, 0.156], [-0.023, 0.0, -0.065], [0.0, 0.0, -0.156], [0.025, 0.0, -0.015], [0.217, 0.0, 0.07], [0.23, 0.0, -0.07], [-0.014, 0.0, -0.156], [0.0, 0.0, 0.042], [-0.057, -0.018, -0.01], [-0.074, -0.0, 0.128], [0.206, -0.07, -0.002], [-0.0, -0.0, 0.156], [-0.017, -0.0, -0.092], [0.217, -0.0, -0.027], [-0.052, -0.0, -0.097], [-0.019, -0.0, 0.128], [-0.035, -0.018, -0.01], [0.217, -0.0, -0.07], [-0.08, -0.0, 0.156], [0.23, 0.0, 0.07], [-0.023, -0.0, -0.075], [-0.029, -0.0, -0.127], [-0.09, -0.0, -0.042], [0.206, -0.055, -0.002], [-0.09, -0.0, -0.015], [0.0, -0.0, -0.015], [-0.037, -0.0, -0.097], [-0.074, -0.0, 0.074], [-0.019, -0.0, 0.074], [0.23, -0.0, -0.113], [-0.1, -0.03, 0.0], [0.17, -0.0, -0.015], [0.23, -0.0, 0.113], [-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [-0.08, -0.0, -0.156]], dtype=np.float32)
        se_anchor_2d = np.array([[415, 144], [1169, 508], [275, 323], [214, 395], [554, 670], [253, 428], [280, 415], [355, 365], [494, 621], [519, 600], [806, 213], [973, 438], [986, 421], [768, 343], [785, 328], [841, 345], [931, 393], [891, 306], [980, 345], [651, 210], [625, 225], [588, 216], [511, 215], [526, 204], [665, 271]], dtype=np.float32)
        se_anchor_3d = np.array([[-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [0.23, 0.0, -0.113], [0.23, 0.0, 0.113], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.206, -0.07, -0.002], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [-0.1, -0.03, 0.0], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.019, 0.0, 0.074], [-0.019, 0.0, 0.128], [-0.074, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.052, 0.0, -0.097], [-0.037, 0.0, -0.097], [-0.029, 0.0, -0.127], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042]], dtype=np.float32)
        sw_anchor_2d = np.array([[650, 312], [630, 306], [907, 443], [814, 291], [599, 349], [501, 386], [965, 359], [649, 355], [635, 346], [930, 335], [843, 467], [702, 339], [718, 321], [930, 322], [727, 346], [539, 364], [786, 297], [1022, 406], [1004, 399], [539, 344], [536, 309], [864, 478], [745, 310], [1049, 393], [895, 258], [674, 347], [741, 281], [699, 294], [817, 494], [992, 281]], dtype=np.float32)
        sw_anchor_3d = np.array([[-0.035, -0.018, -0.01], [-0.057, -0.018, -0.01], [0.217, -0.0, -0.027], [-0.014, -0.0, 0.156], [-0.023, -0.0, -0.065], [-0.014, -0.0, -0.156], [0.234, -0.05, -0.002], [0.0, -0.0, -0.042], [-0.014, -0.0, -0.042], [0.206, -0.055, -0.002], [0.217, -0.0, -0.07], [0.025, -0.014, -0.011], [-0.014, -0.0, 0.042], [0.206, -0.07, -0.002], [0.049, -0.016, -0.011], [-0.029, -0.0, -0.127], [-0.019, -0.0, 0.128], [0.23, -0.0, 0.07], [0.217, -0.0, 0.07], [-0.052, -0.0, -0.097], [-0.175, -0.0, -0.015], [0.23, -0.0, -0.07], [-0.019, -0.0, 0.074], [0.23, -0.0, 0.113], [-0.0, -0.025, 0.24], [-0.0, -0.0, -0.015], [-0.074, -0.0, 0.128], [-0.074, -0.0, 0.074], [0.23, -0.0, -0.113], [0.243, -0.104, 0.0]], dtype=np.float32)
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
            
            # Find correspondences between labeled 2D points and extracted SuperPoint features
            sp_tree = cKDTree(anchor_keypoints)
            distances, indices = sp_tree.query(data['2d'], k=1, distance_upper_bound=5.0)
            valid_mask = distances != np.inf
            
            self.viewpoint_anchors[viewpoint] = {
                'features': anchor_features,
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
            
            result = self._process_frame(frame, frame_count)
            self.all_poses_log.append({
                'frame': frame_count,
                'success': result.pose_success,
                'position': result.position.tolist() if result.position is not None else None,
                'quaternion': result.quaternion.tolist() if result.quaternion is not None else None,
                'kf_position': result.kf_position.tolist() if result.kf_position is not None else None,
                'kf_quaternion': result.kf_quaternion.tolist() if result.kf_quaternion is not None else None,
            })

            # Calculate running FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            self._update_display(result, fps)
        
        self.cleanup()

    def _process_frame(self, frame: np.ndarray, frame_id: int) -> ProcessingResult:
        """Runs the full pose estimation pipeline on a single frame."""
        result = ProcessingResult(frame_id=frame_id, frame=frame.copy())
        
        # Predict step for Kalman Filter
        if self.use_kalman_filter and self.kf.initialized:
            result.kf_position, result.kf_quaternion = self.kf.predict()
            
        # 1. Object Detection
        bbox = self._yolo_detect(frame)
        result.bbox = bbox
        
        # 2. Viewpoint Classification
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else frame
        initial_viewpoint = self._classify_viewpoint(crop)
            
        # 3. Feature Matching and Pose Estimation
        best_pose = self._estimate_pose_with_fallback(frame, initial_viewpoint, bbox)
        
        if best_pose:
            result.position = best_pose.position
            result.quaternion = best_pose.quaternion
            result.num_inliers = best_pose.inliers
            result.pose_success = True
            
            # Update step for Kalman Filter
            if self.use_kalman_filter:
                kf_pos, kf_quat = self.kf.update(best_pose.position, best_pose.quaternion)
                result.kf_position = kf_pos
                result.kf_quaternion = kf_quat
        else:
            # If pose fails, result.pose_success remains False
            pass

        return result

    def _estimate_pose_with_fallback(self, frame: np.ndarray, initial_viewpoint: str, bbox: Optional[Tuple]) -> Optional[PoseData]:
        """Tries initial viewpoint, then falls back to others, returning the best valid pose."""
        all_viewpoints = ['NW', 'NE', 'SE', 'SW']
        # Create a prioritized list of viewpoints to try
        viewpoints_to_try = [initial_viewpoint] + [vp for vp in all_viewpoints if vp != initial_viewpoint]
        
        successful_poses = []
        for viewpoint in viewpoints_to_try:
            pose_data = self._solve_for_viewpoint(frame, viewpoint, bbox)
            if pose_data:
                successful_poses.append(pose_data)
        
        if not successful_poses:
            return None
        
        # Return the pose with the most inliers
        return max(successful_poses, key=lambda p: p.inliers)

    def _solve_for_viewpoint(self, frame: np.ndarray, viewpoint: str, bbox: Optional[Tuple]) -> Optional[PoseData]:
        """Attempts to solve the pose for a single given viewpoint."""
        anchor = self.viewpoint_anchors.get(viewpoint)
        if not anchor: return None
        
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else frame
        crop_offset = np.array([bbox[0], bbox[1]]) if bbox else np.array([0, 0])
        
        if crop.size == 0: return None
        
        # Extract features from the current frame/crop
        frame_features = self._extract_features_sp(crop)
        
        # Match features between anchor and current frame
        with torch.no_grad():
            matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
        matches = rbd(matches_dict)['matches'].cpu().numpy()
        
        if len(matches) < 6: return None

        # Build 2D and 3D point correspondences
        points_3d, points_2d = [], []
        for anchor_idx, frame_idx in matches:
            if anchor_idx in anchor['map_3d']:
                points_3d.append(anchor['map_3d'][anchor_idx])
                points_2d.append(frame_features['keypoints'][0].cpu().numpy()[frame_idx] + crop_offset)
        
        if len(points_3d) < 6: return None
        
        # Solve PnP
        K, dist_coeffs = self._get_camera_intrinsics()
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(points_3d), np.array(points_2d), K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
            )
        except cv2.error:
            return None

        if not success or inliers is None or len(inliers) < 4:
            return None

        R, _ = cv2.Rodrigues(rvec)
        position = tvec.flatten()
        quaternion = self._rotation_matrix_to_quaternion(R)
        
        # Calculate reprojection error for quality assessment
        projected, _ = cv2.projectPoints(np.array(points_3d)[inliers.flatten()], rvec, tvec, K, dist_coeffs)
        error = np.mean(np.linalg.norm(np.array(points_2d)[inliers.flatten()].reshape(-1, 1, 2) - projected, axis=2))
        
        return PoseData(position, quaternion, len(inliers), error, viewpoint)
    
    # --- Helper & Utility Methods ---

    def _yolo_detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detects the object in the frame and returns a bounding box."""
        results = self.yolo_model(frame, verbose=False, conf=0.5)
        if len(results[0].boxes) > 0:
            return tuple(map(int, results[0].boxes.xyxy.cpu().numpy()[0]))
        return None

    def _classify_viewpoint(self, crop: np.ndarray) -> str:
        """Classifies the viewpoint from a cropped image."""
        if crop.shape[0] == 0 or crop.shape[1] == 0: return 'NW' # Default
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = self.vp_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.class_names[torch.argmax(self.vp_model(tensor), dim=1).item()]
    
    def _extract_features_sp(self, image_bgr: np.ndarray) -> Dict:
        """Extracts SuperPoint features from a BGR image."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            return self.extractor.extract(tensor)

    def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32), None

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw, qx, qy, qz = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
        return np.array([qx, qy, qz, qw])

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])

    def _update_display(self, result: ProcessingResult, fps: float):
        """Prepares and queues a frame for the display thread."""
        vis_frame = result.frame
        if result.bbox:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Use the smoothed KF pose for visualization if available
        display_pos = result.kf_position if result.kf_position is not None else result.position
        display_quat = result.kf_quaternion if result.kf_quaternion is not None else result.quaternion

        if display_pos is not None and display_quat is not None:
            self._draw_axes(vis_frame, display_pos, display_quat)

        # --- On-screen Display (OSD) Text ---
        status_color = (0, 255, 0) if result.pose_success else (0, 0, 255)
        status_text = "SUCCESS" if result.pose_success else "TRACKING FAILED"
        
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(vis_frame, f"STATUS: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(vis_frame, f"Inliers: {result.num_inliers}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        try:
            # Non-blocking put to avoid delays
            self.display_queue.put_nowait(vis_frame)
        except queue.Full:
            pass # Skip frame if display is lagging

    def _draw_axes(self, frame, position, quaternion):
        """Draws a 3D coordinate axis on the frame."""
        try:
            R = self._quaternion_to_rotation_matrix(quaternion)
            rvec, _ = cv2.Rodrigues(R)
            tvec = position.reshape(3, 1)
            K, _ = self._get_camera_intrinsics()
            axis_pts = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, None)
            img_pts = img_pts.reshape(-1, 2).astype(int)
            origin = tuple(img_pts[0])
            cv2.line(frame, origin, tuple(img_pts[1]), (0,0,255), 3)  # X-axis (Red)
            cv2.line(frame, origin, tuple(img_pts[2]), (0,255,0), 3)  # Y-axis (Green)
            cv2.line(frame, origin, tuple(img_pts[3]), (255,0,0), 3)  # Z-axis (Blue)
        except (cv2.error, AttributeError):
            pass # Ignore drawing errors

    def _display_loop(self):
        """Dedicated thread for rendering frames from the queue."""
        window_name = "Real-time Pose Estimation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        while self.running:
            try:
                frame = self.display_queue.get(timeout=1.0)
                cv2.imshow(window_name, frame)
            except queue.Empty:
                if not self.running: break
                continue
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

    def cleanup(self):
        """Releases resources and saves logs."""
        print("\nShutting down...")
        self.running = False
        self.display_thread.join(timeout=2.0)
        
        if self.is_video_stream:
            self.video_capture.release()
        
        cv2.destroyAllWindows()

        if self.args.save_output:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"pose_log_{time.strftime('%Y%m%d-%H%M%S')}.json")
            with open(filename, 'w') as f:
                json.dump(self.all_poses_log, f, indent=4)
            print(f"💾 Pose log saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="VAPE MK47 - Real-time Pose Estimator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--webcam', action='store_true', help='Use webcam as input.')
    group.add_argument('--video_file', type=str, help='Path to a video file.')
    group.add_argument('--image_dir', type=str, help='Path to a directory of images.')
    
    parser.add_argument('--save_output', action='store_true', help='Save the final pose data to a JSON file.')
    args = parser.parse_args()

    try:
        estimator = PoseEstimator(args)
        estimator.run()
    except (IOError, FileNotFoundError) as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        print("✅ Process finished.")

if __name__ == '__main__':
    main()