#!/usr/bin/env python3
"""
VAPE MK52 Transformation Visualizer

This script loads a calibration file (produced by ChAruco.py) and visualizes
the transformation by applying it to a live ChArUco pose and comparing it
to a live VAPE pose measurement.
"""

import cv2
import numpy as np
import torch
import time
import json
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import warnings
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_grad_enabled(False)

# Import your existing dependencies
from ultralytics import YOLO
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

# --- ChArUco Helpers (from ChAruco.py) ---
def make_charuco_board(cols, rows, square_len_m, marker_len_m, dict_id=cv2.aruco.DICT_5X5_1000):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(cols, rows, square_len_m, marker_len_m, aruco_dict)
    else:
        board = cv2.aruco.CharucoBoard((cols, rows), square_len_m, marker_len_m, aruco_dict)
    return aruco_dict, board

def make_detectors(aruco_dict, board):
    has_new = hasattr(cv2.aruco, "ArucoDetector") and hasattr(cv2.aruco, "CharucoDetector")
    try:
        aruco_params = cv2.aruco.DetectorParameters()
    except AttributeError:
        aruco_params = cv2.aruco.DetectorParameters_create()
    
    aruco_params.adaptiveThreshWinSizeMin = 5
    aruco_params.adaptiveThreshWinSizeMax = 21
    aruco_params.adaptiveThreshWinSizeStep = 4
    aruco_params.minMarkerPerimeterRate = 0.05
    aruco_params.maxMarkerPerimeterRate = 0.3

    if has_new:
        aruco_det = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        charuco_det = cv2.aruco.CharucoDetector(board)
        return ("new", aruco_det, charuco_det)
    else:
        return ("old", aruco_params, None)

class TransformVisualizer:
    """
    Loads a calibration and visualizes the transform between ChArUco and VAPE poses.
    """
    def __init__(self, video_path: str, calibration_path: str):
        self.video_path = video_path
        self.calibration_path = calibration_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Camera parameters
        self.camera_width, self.camera_height = 1280, 720
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = None

        # Load Calibration
        self.R_to, self.t_to = self._load_calibration()

        # ChArUco setup
        self.aruco_dict, self.board = make_charuco_board(
            cols=5, rows=7, square_len_m=0.035, marker_len_m=0.025
        )
        self.detector_mode, self.aruco_det, self.charuco_det = make_detectors(self.aruco_dict, self.board)

        # VAPE models
        print("🔧 Loading VAPE models...")
        self.yolo_model = YOLO("best.pt").to(self.device)
        self.extractor = SuperPoint(max_num_keypoints=1024*2).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        
        self.viewpoint_anchors = {}
        self._initialize_anchor_data()
        
        print("✅ Visualizer initialized")

    def _load_calibration(self) -> Tuple[np.ndarray, np.ndarray]:
        """Loads the R_to and t_to transformation from the JSON file."""
        print(f"💾 Loading calibration from: {self.calibration_path}")
        if not os.path.exists(self.calibration_path):
            raise FileNotFoundError(f"Calibration file not found: {self.calibration_path}")
        
        with open(self.calibration_path, 'r') as f:
            data = json.load(f)
        
        R_to = np.array(data['R_to'])
        t_to = np.array(data['t_to'])
        
        print("   ✓ R_to and t_to loaded successfully.")
        return R_to, t_to

    def run_visualization(self):
        """Process the video and show the visualization."""
        print(f"🎬 Processing video: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every 2nd frame to run closer to real-time
            if frame_id % 2 == 0:
                vis_frame = self._process_frame(frame)
                cv2.imshow('VAPE vs ChArUco-Predicted Transform', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_id += 1

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Visualization finished.")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Processes a single frame for visualization."""
        
        # 1. Detect ChArUco pose
        charuco_success, R_ct, t_ct = self._detect_charuco_pose(frame)

        # 2. Detect VAPE pose
        vape_success, vape_pos, vape_quat = self._estimate_vape_pose(frame)

        # 3. If both are successful, apply transform and draw comparison
        if charuco_success and vape_success:
            # Predict VAPE pose from ChArUco pose
            R_co_pred = R_ct @ self.R_to
            t_co_pred = (R_ct @ self.t_to) + t_ct
            
            # Draw measured VAPE pose (RED)
            self._draw_axes(frame, vape_pos, vape_quat, label="VAPE (Measured)", color=(0, 0, 255))

            # Draw predicted VAPE pose (BLUE)
            rvec_pred, _ = cv2.Rodrigues(R_co_pred)
            self._draw_axes(frame, t_co_pred, rvec=rvec_pred, label="VAPE (Predicted)", color=(255, 100, 100))
            
            cv2.putText(frame, "STATUS: COMPARING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:
            status = "STATUS: "
            if not charuco_success: status += "CHARUCO NOT FOUND "
            if not vape_success: status += "VAPE NOT FOUND"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        return frame

    def _detect_charuco_pose(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Detects the ChArUco board and returns its pose."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.detector_mode == "new":
            corners, ids, _ = self.aruco_det.detectMarkers(gray)
            if ids is None or len(ids) == 0: return False, None, None
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
            #_, charuco_corners, charuco_ids = self.charuco_det.interpolateCornersCharuco(corners, ids, gray)
        else: # old API
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_det)
            if ids is None or len(ids) == 0: return False, None, None
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)

        if charuco_corners is None or len(charuco_corners) < 6:
            return False, None, None

        pose_ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, self.board, self.K, self.dist_coeffs, None, None
        )

        if pose_ok:
            R_ct, _ = cv2.Rodrigues(rvec)
            return True, R_ct, tvec
        
        return False, None, None

    def _estimate_vape_pose(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimates VAPE pose by finding the best viewpoint match."""
        bbox = self._yolo_detect(frame)
        if not bbox:
            return False, None, None

        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if crop.size == 0: return False, None, None
        
        frame_features = self._extract_features_sp(crop)
        crop_offset = np.array([bbox[0], bbox[1]])
        
        best_pose_data = {'score': -1, 'pos': None, 'quat': None}

        for viewpoint, anchor in self.viewpoint_anchors.items():
            with torch.no_grad():
                matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
            matches = rbd(matches_dict)['matches'].cpu().numpy()
            
            if len(matches) < 6: continue
            
            points_3d, points_2d = [], []
            for anchor_idx, frame_idx in matches:
                if anchor_idx in anchor['map_3d']:
                    points_3d.append(anchor['map_3d'][anchor_idx])
                    points_2d.append(frame_features['keypoints'][0].cpu().numpy()[frame_idx] + crop_offset)
            
            if len(points_3d) < 6: continue
            
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    np.array(points_3d, dtype=np.float32), np.array(points_2d, dtype=np.float32),
                    self.K, self.dist_coeffs, reprojectionError=8.0, confidence=0.99, flags=cv2.SOLVEPNP_EPNP
                )
                if success and inliers is not None and len(inliers) > best_pose_data['score']:
                    R_mat, _ = cv2.Rodrigues(rvec)
                    quat = R.from_matrix(R_mat).as_quat()
                    best_pose_data.update({'score': len(inliers), 'pos': tvec.flatten(), 'quat': quat})
            except cv2.error:
                continue
        
        if best_pose_data['score'] > 0:
            return True, best_pose_data['pos'], best_pose_data['quat']
        
        return False, None, None

    def _draw_axes(self, frame, position, quaternion=None, rvec=None, label="", color=(255,255,255), length=0.1):
        """Draws a 3D coordinate axis on the frame."""
        try:
            if rvec is None and quaternion is not None:
                R_matrix = R.from_quat(quaternion).as_matrix()
                rvec, _ = cv2.Rodrigues(R_matrix)
            
            tvec = position.reshape(3, 1)
            axis_pts = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, self.K, self.dist_coeffs)
            img_pts = img_pts.reshape(-1, 2).astype(int)
            
            origin = tuple(img_pts[0])
            cv2.line(frame, origin, tuple(img_pts[1]), (0, 0, 255), 3)  # X - Red
            cv2.line(frame, origin, tuple(img_pts[2]), (0, 255, 0), 3)  # Y - Green
            cv2.line(frame, origin, tuple(img_pts[3]), (255, 0, 0), 3)  # Z - Blue
            
            cv2.putText(frame, label, (origin[0] + 10, origin[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except:
            pass # Fail silently if projection fails

    def _yolo_detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        names = getattr(self.yolo_model, "names", {0: "iha"})
        inv = {v: k for k, v in names.items()}
        target_id = inv.get("iha", 0)
        results = self.yolo_model(frame, conf=0.25, iou=0.5, max_det=5, classes=[target_id], verbose=False)
        if not results or len(results[0].boxes) == 0: return None
        best = max(results[0].boxes, key=lambda b: float((b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1])))
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().tolist()
        return int(x1), int(y1), int(x2), int(y2)

    def _extract_features_sp(self, image_bgr: np.ndarray) -> Dict:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            return self.extractor.extract(tensor)

    def _initialize_anchor_data(self):
        """Initialize anchor data (copied from ChAruco.py)"""
        print("🛠️ Loading anchor data...")
        anchor_definitions = {
                'NE': {
                    'path': 'NE.png',
                    '2d': np.array([[928, 148],[570, 111],[401, 31],[544, 141],[530, 134],[351, 228],[338, 220],[294, 244],[230, 541],[401, 469],[414, 481],[464, 451],[521, 510],[610, 454],[544, 400],[589, 373],[575, 361],[486, 561],[739, 385],[826, 305],[791, 285],[773, 271],[845, 233],[826, 226],[699, 308],[790, 375]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.000, -0.000, 0.156],[-0.014, 0.000, 0.156],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.019, -0.000, 0.074],[-0.014, 0.000, 0.042],[0.000, 0.000, 0.042],[-0.080, -0.000, 0.156],[-0.100, -0.030, 0.000],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.014, 0.000, -0.156],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.042],[-0.090, -0.000, -0.042]], dtype=np.float32)
                },
                'NW': {
                    'path': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png',
                    '2d': np.array([[511, 293],[591, 284],[587, 330],[413, 249],[602, 348],[715, 384],[598, 298],[656, 171],[805, 213],[703, 392],[523, 286],[519, 327],[387, 289],[727, 126],[425, 243],[636, 358],[745, 202],[595, 388],[436, 260],[539, 313],[795, 220],[351, 291],[665, 165],[611, 353],[650, 377],[516, 389],[727, 143],[496, 378],[575, 312],[617, 368],[430, 312],[480, 281],[834, 225],[469, 339],[705, 223],[637, 156],[816, 414],[357, 195],[752, 77],[642, 451]], dtype=np.float32),
                    '3d': np.array([[-0.014, 0.0, 0.042],[0.025, -0.014, -0.011],[-0.014, 0.0, -0.042],[-0.014, 0.0, 0.156],[-0.023, 0.0, -0.065],[0.0, 0.0, -0.156],[0.025, 0.0, -0.015],[0.217, 0.0, 0.07],[0.23, 0.0, -0.07],[-0.014, 0.0, -0.156],[0.0, 0.0, 0.042],[-0.057, -0.018, -0.01],[-0.074, -0.0, 0.128],[0.206, -0.07, -0.002],[-0.0, -0.0, 0.156],[-0.017, -0.0, -0.092],[0.217, -0.0, -0.027],[-0.052, -0.0, -0.097],[-0.019, -0.0, 0.128],[-0.035, -0.018, -0.01],[0.217, -0.0, -0.07],[-0.08, -0.0, 0.156],[0.23, 0.0, 0.07],[-0.023, -0.0, -0.075],[-0.029, -0.0, -0.127],[-0.09, -0.0, -0.042],[0.206, -0.055, -0.002],[-0.09, -0.0, -0.015],[0.0, -0.0, -0.015],[-0.037, -0.0, -0.097],[-0.074, -0.0, 0.074],[-0.019, -0.0, 0.074],[0.23, -0.0, -0.113],[-0.1, -0.03, 0.0],[0.17, -0.0, -0.015],[0.23, -0.0, 0.113],[-0.0, -0.025, -0.24],[-0.0, -0.025, 0.24],[0.243, -0.104, 0.0],[-0.08, -0.0, -0.156]], dtype=np.float32)
                },
                'SE': {
                    'path': 'SE.png',
                    '2d': np.array([[415, 144],[1169, 508],[275, 323],[214, 395],[554, 670],[253, 428],[280, 415],[355, 365],[494, 621],[519, 600],[806, 213],[973, 438],[986, 421],[768, 343],[785, 328],[841, 345],[931, 393],[891, 306],[980, 345],[651, 210],[625, 225],[588, 216],[511, 215],[526, 204],[665, 271]], dtype=np.float32),
                    '3d': np.array([[-0.0, -0.025, -0.24],[-0.0, -0.025, 0.24],[0.243, -0.104, 0.0],[0.23, 0.0, -0.113],[0.23, 0.0, 0.113],[0.23, 0.0, -0.07],[0.217, 0.0, -0.07],[0.206, -0.07, -0.002],[0.23, 0.0, 0.07],[0.217, 0.0, 0.07],[-0.1, -0.03, 0.0],[-0.0, 0.0, 0.156],[-0.014, 0.0, 0.156],[0.0, 0.0, 0.042],[-0.014, 0.0, 0.042],[-0.019, 0.0, 0.074],[-0.019, 0.0, 0.128],[-0.074, 0.0, 0.074],[-0.074, 0.0, 0.128],[-0.052, 0.0, -0.097],[-0.037, 0.0, -0.097],[-0.029, 0.0, -0.127],[0.0, 0.0, -0.156],[-0.014, 0.0, -0.156],[-0.014, 0.0, -0.042]], dtype=np.float32)
                },
                'SW': {
                    'path': 'Anchor_B.png',
                    '2d': np.array([[650, 312],[630, 306],[907, 443],[814, 291],[599, 349],[501, 386],[965, 359],[649, 355],[635, 346],[930, 335],[843, 467],[702, 339],[718, 321],[930, 322],[727, 346],[539, 364],[786, 297],[1022, 406],[1004, 399],[539, 344],[536, 309],[864, 478],[745, 310],[1049, 393],[895, 258],[674, 347],[741, 281],[699, 294],[817, 494],[992, 281]], dtype=np.float32),
                    '3d': np.array([[-0.035, -0.018, -0.01],[-0.057, -0.018, -0.01],[0.217, -0.0, -0.027],[-0.014, -0.0, 0.156],[-0.023, 0.0, -0.065],[-0.014, -0.0, -0.156],[0.234, -0.05, -0.002],[0.0, -0.0, -0.042],[-0.014, -0.0, -0.042],[0.206, -0.055, -0.002],[0.217, -0.0, -0.07],[0.025, -0.014, -0.011],[-0.014, -0.0, 0.042],[0.206, -0.07, -0.002],[0.049, -0.016, -0.011],[-0.029, -0.0, -0.127],[-0.019, -0.0, 0.128],[0.23, -0.0, 0.07],[0.217, -0.0, 0.07],[-0.052, -0.0, -0.097],[-0.175, -0.0, -0.015],[0.23, -0.0, -0.07],[-0.019, -0.0, 0.074],[0.23, -0.0, 0.113],[-0.0, -0.025, 0.24],[-0.0, -0.0, -0.015],[-0.074, -0.0, 0.128],[-0.074, -0.0, 0.074],[0.23, -0.0, -0.113],[0.243, -0.104, 0.0]], dtype=np.float32)
                },
                'W': {
                    'path': 'W.png',
                    '2d': np.array([[589, 555],[565, 481],[531, 480],[329, 501],[326, 345],[528, 351],[395, 391],[469, 395],[529, 140],[381, 224],[504, 258],[498, 229],[383, 253],[1203, 100],[1099, 174],[1095, 211],[1201, 439],[1134, 404],[1100, 358],[625, 341],[624, 310],[315, 264]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.156],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.015],[-0.014, 0.000, -0.042],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.000, -0.025, 0.240],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.074],[0.243, -0.104, 0.000],[0.206, -0.070, -0.002],[0.206, -0.055, -0.002],[0.230, -0.000, -0.113],[0.217, -0.000, -0.070],[0.217, -0.000, -0.027],[0.025, 0.000, -0.015],[0.025, -0.014, -0.011],[-0.100, -0.030, 0.000]], dtype=np.float32)
                },
                'S': {
                    'path': 'S.png',
                    '2d': np.array([[14, 243],[1269, 255],[654, 183],[290, 484],[1020, 510],[398, 475],[390, 503],[901, 489],[573, 484],[250, 283],[405, 269],[435, 243],[968, 273],[838, 273],[831, 233],[949, 236]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, 0.000, 0.070],[0.217, -0.000, -0.027],[0.000, 0.000, -0.156],[-0.017, -0.000, -0.092],[-0.052, -0.000, -0.097],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128]], dtype=np.float32)
                },
                'N': {
                    'path': 'N.png',
                    '2d': np.array([[1238, 346],[865, 295],[640, 89],[425, 314],[24, 383],[303, 439],[445, 434],[856, 418],[219, 475],[1055, 450]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.052, -0.000, -0.097],[-0.080, -0.000, 0.156],[-0.080, -0.000, -0.156]], dtype=np.float32)
                },
                'SW2': {
                    'path': 'SW2.png',
                    '2d': np.array([[15, 300],[1269, 180],[635, 143],[434, 274],[421, 240],[273, 320],[565, 266],[844, 206],[468, 543],[1185, 466],[565, 506],[569, 530],[741, 491],[1070, 459],[1089, 480],[974, 220],[941, 184],[659, 269],[650, 299],[636, 210],[620, 193]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.017, -0.000, -0.092],[-0.052, -0.000, -0.097],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.042],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, -0.000, -0.027],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[0.025, -0.014, -0.011],[0.025, 0.000, -0.015],[-0.035, -0.018, -0.010],[-0.057, -0.018, -0.010]], dtype=np.float32)
                },
                'SE2': {
                    'path': 'SE2.png',
                    '2d': np.array([[48, 216],[1269, 320],[459, 169],[853, 528],[143, 458],[244, 470],[258, 451],[423, 470],[741, 500],[739, 516],[689, 176],[960, 301],[828, 290],[970, 264],[850, 254]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[0.230, -0.000, -0.113],[0.230, 0.000, -0.070],[0.217, -0.000, -0.070],[0.217, -0.000, -0.027],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[-0.100, -0.030, 0.000],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074]], dtype=np.float32)
                },
                'SU': {
                    'path': 'SU.png',
                    '2d': np.array([[203, 251],[496, 191],[486, 229],[480, 263],[368, 279],[369, 255],[573, 274],[781, 280],[859, 293],[865, 213],[775, 206],[1069, 326],[656, 135],[633, 241],[629, 204],[623, 343],[398, 668],[463, 680],[466, 656],[761, 706],[761, 681],[823, 709],[616, 666]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.017, -0.000, -0.092],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.156],[-0.014, 0.000, -0.042],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.035, -0.018, -0.010],[-0.057, -0.018, -0.010],[0.025, -0.014, -0.011],[0.230, -0.000, -0.113],[0.230, 0.000, -0.070],[0.217, -0.000, -0.070],[0.230, -0.000, 0.070],[0.217, 0.000, 0.070],[0.230, -0.000, 0.113],[0.243, -0.104, 0.000]], dtype=np.float32)
                },
                'NU': {
                    'path': 'NU.png',
                    '2d': np.array([[631, 361],[1025, 293],[245, 294],[488, 145],[645, 10],[803, 146],[661, 188],[509, 365],[421, 364],[434, 320],[509, 316],[779, 360],[784, 321],[704, 398],[358, 393]], dtype=np.float32),
                    '3d': np.array([[-0.100, -0.030, 0.000],[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.230, -0.000, 0.113],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.170, -0.000, -0.015],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.090, -0.000, -0.042],[-0.080, -0.000, 0.156]], dtype=np.float32)
                },
                'NW2': {
                    'path': 'NW2.png',
                    '2d': np.array([[1268, 328],[1008, 419],[699, 399],[829, 373],[641, 325],[659, 310],[783, 30],[779, 113],[775, 153],[994, 240],[573, 226],[769, 265],[686, 284],[95, 269],[148, 375],[415, 353],[286, 349],[346, 320],[924, 360],[590, 324]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.042],[-0.052, -0.000, -0.097],[-0.057, -0.018, -0.010],[-0.035, -0.018, -0.010],[0.243, -0.104, 0.000],[0.206, -0.070, -0.002],[0.206, -0.055, -0.002],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.170, -0.000, -0.015],[0.025, -0.014, -0.011],[-0.000, -0.025, 0.240],[-0.080, -0.000, 0.156],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.128],[-0.029, -0.000, -0.127],[-0.100, -0.030, 0.000]], dtype=np.float32)
                },
                'NE2': {
                    'path': 'NE2.png',
                    '2d': np.array([[1035, 95],[740, 93],[599, 16],[486, 168],[301, 305],[719, 225],[425, 349],[950, 204],[794, 248],[844, 203],[833, 175],[601, 275],[515, 301],[584, 244],[503, 266]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.080, -0.000, 0.156],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.042],[-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128]], dtype=np.float32)
                },
                'E': {
                    'path': 'E.png',
                    '2d': np.array([[696, 165],[46, 133],[771, 610],[943, 469],[921, 408],[793, 478],[781, 420],[793, 520],[856, 280],[743, 284],[740, 245],[711, 248],[74, 520],[134, 465],[964, 309]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[0.243, -0.104, 0.000],[-0.000, -0.025, 0.240],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.014, 0.000, 0.156],[-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.014, 0.000, -0.156],[0.000, 0.000, -0.156],[0.230, -0.000, 0.113],[0.217, 0.000, 0.070],[-0.100, -0.030, 0.000]], dtype=np.float32)
                }
        }
        for viewpoint, data in anchor_definitions.items():
            if os.path.exists(data['path']):
                anchor_image = cv2.resize(cv2.imread(data['path']), (self.camera_width, self.camera_height))
                anchor_features = self._extract_features_sp(anchor_image)
                anchor_keypoints = anchor_features['keypoints'][0].cpu().numpy()
                
                sp_tree = cKDTree(anchor_keypoints)
                distances, indices = sp_tree.query(data['2d'], k=1, distance_upper_bound=5.0)
                valid_mask = distances != np.inf
                
                self.viewpoint_anchors[viewpoint] = {
                    'features': anchor_features,
                    'map_3d': {idx: pt for idx, pt in zip(indices[valid_mask], data['3d'][valid_mask])}
                }
                print(f"   ✓ Loaded {viewpoint} anchor ({np.sum(valid_mask)} points)")

def main():
    parser = argparse.ArgumentParser(description="VAPE MK52 Transformation Visualizer")
    parser.add_argument('--video', required=True, help='Path to input MP4 video for visualization')
    parser.add_argument('--calibration', required=True, help='Path to the calibration JSON file (tag_to_object_ground_truth.json)')
    args = parser.parse_args()

    try:
        visualizer = TransformVisualizer(args.video, args.calibration)
        visualizer.run_visualization()
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
