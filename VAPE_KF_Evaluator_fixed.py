#!/usr/bin/env python3
"""
VAPE MK52 KF-EVALUATOR (Hybrid Version)

This script combines the robust pose estimation pipeline of VAPE_MK52.py
with enhanced evaluation and debugging capabilities.

It provides a comprehensive, real-time evaluation and debugging environment
for the VAPE pose estimation system, including its Kalman Filter, by
implementing the same multi-threaded architecture as VAPE_MK52.py.

It visualizes three key pieces of information simultaneously:
1. Ground Truth Pose (Green): Derived from a ChArUco board and a calibration file.
2. Raw VAPE Measurement (Red): The unfiltered output of the PnP algorithm.
3. Filtered VAPE Pose (Blue): The final, smoothed output from the Unscented Kalman Filter,
   predicted continuously for a smooth visualization.
"""

# --- IMPORTS AND INITIAL SETUP ---
import cv2
import numpy as np
import torch
import time
import argparse
import warnings
import json
import threading
import os
import math
import queue
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_grad_enabled(False)

from ultralytics import YOLO
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

# --- UTILITY & DATASTRUCTURES ---
def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-10 else np.array([0, 0, 0, 1])

@dataclass
class FrameDataPacket:
    """Holds all data passed from ProcessingThread to MainThread."""
    gt_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
    raw_vape_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
    errors: Dict[str, float] = field(default_factory=dict)

@dataclass
class PoseData:
    """A simple container for the results of a successful pose estimation."""
    position: np.ndarray
    quaternion: np.ndarray
    inliers: int
    reprojection_error: float
    viewpoint: str
    total_matches: int
    # For match visualization
    frame_crop: np.ndarray = None
    anchor_image: np.ndarray = None
    matches: np.ndarray = None
    kpts_frame: np.ndarray = None
    kpts_anchor: np.ndarray = None

# --- KALMAN FILTER (from VAPE_MK52.py) ---
class UnscentedKalmanFilter:
    def __init__(self, dt=1/15.0):
        self.dt = dt
        self.initialized = False
        self.n = 16
        self.m = 7
        self.x = np.zeros(self.n)
        self.x[9] = 1.0
        self.P = np.eye(self.n) * 0.1
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        self.wm = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + self.lambda_)))
        self.wc = self.wm.copy()
        self.wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.wc[0] = self.lambda_ / (self.n + self.lambda_) + (1.0 - self.alpha**2 + self.beta)
        self.Q = np.eye(self.n) * 1e-2
        self.R = np.eye(self.m) * 1e-4

    def _generate_sigma_points(self, x, P):
        sigmas = np.zeros((2 * self.n + 1, self.n))
        U = np.linalg.cholesky((self.n + self.lambda_) * P)
        sigmas[0] = x
        for i in range(self.n):
            sigmas[i+1]   = x + U[:, i]
            sigmas[self.n+i+1] = x - U[:, i]
        return sigmas

    def motion_model(self, x_in):
        dt = self.dt
        x_out = np.zeros_like(x_in)
        pos, vel, acc = x_in[0:3], x_in[3:6], x_in[6:9]
        x_out[0:3] = pos + vel * dt + 0.5 * acc * dt**2
        x_out[3:6] = vel + acc * dt
        x_out[6:9] = acc
        q, w = x_in[9:13], x_in[13:16]
        omega_mat = 0.5 * np.array([[0,-w[0],-w[1],-w[2]], [w[0],0,w[2],-w[1]], [w[1],-w[2],0,w[0]], [w[2],w[1],-w[0],0]])
        q_new = (np.eye(4) + dt * omega_mat) @ q
        x_out[9:13] = normalize_quaternion(q_new)
        x_out[13:16] = w
        return x_out

    def predict(self):
        if not self.initialized: return
        sigmas = self._generate_sigma_points(self.x, self.P)
        sigmas_f = np.array([self.motion_model(s) for s in sigmas])
        x_pred = np.sum(self.wm[:, np.newaxis] * sigmas_f, axis=0)
        P_pred = self.Q.copy()
        for i in range(2 * self.n + 1):
            y = sigmas_f[i] - x_pred
            P_pred += self.wc[i] * np.outer(y, y)
        self.x, self.P = x_pred, P_pred

    def hx(self, x_in):
        z = np.zeros(self.m)
        z[0:3] = x_in[0:3]
        z[3:7] = x_in[9:13]
        return z

    def update(self, position: np.ndarray, quaternion: np.ndarray):
        if self.initialized and np.dot(self.x[9:13], quaternion) < 0.0:
            quaternion = -quaternion
        measurement = np.concatenate([position, normalize_quaternion(quaternion)])
        if not self.initialized:
            self.x[0:3], self.x[9:13] = position, normalize_quaternion(quaternion)
            self.initialized = True
            return
        sigmas_f = self._generate_sigma_points(self.x, self.P)
        sigmas_h = np.array([self.hx(s) for s in sigmas_f])
        z_pred = np.sum(self.wm[:, np.newaxis] * sigmas_h, axis=0)
        S = self.R.copy()
        for i in range(2 * self.n + 1):
            y = sigmas_h[i] - z_pred
            S += self.wc[i] * np.outer(y, y)
        P_xz = np.zeros((self.n, self.m))
        for i in range(2 * self.n + 1):
            y_x, y_z = sigmas_f[i] - self.x, sigmas_h[i] - z_pred
            P_xz += self.wc[i] * np.outer(y_x, y_z)
        K = P_xz @ np.linalg.inv(S)
        self.x += K @ (measurement - z_pred)
        self.P -= K @ S @ K.T

# --- CHARUCO HELPERS ---
def make_charuco_board(cols, rows, square_len_m, marker_len_m, dict_id=cv2.aruco.DICT_5X5_1000):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(cols, rows, square_len_m, marker_len_m, aruco_dict)
    else:
        board = cv2.aruco.CharucoBoard((cols, rows), square_len_m, marker_len_m, aruco_dict)
    return aruco_dict, board

def make_detectors(aruco_dict, board):
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(aruco_dict, params), cv2.aruco.CharucoDetector(board)
    else: # Legacy OpenCV
        return params, None

# --- LOW-FREQUENCY PROCESSING THREAD ---
class ProcessingThread(threading.Thread):
    def __init__(self, processing_queue, visualization_queue, pose_data_lock, kf, args):
        super().__init__()
        self.running = True
        self.processing_queue = processing_queue
        self.visualization_queue = visualization_queue
        self.pose_data_lock = pose_data_lock
        self.kf = kf
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.all_errors = []
        self.update_count = 0

        self.current_best_viewpoint = None
        self.viewpoint_quality_threshold = 5
        self.consecutive_failures = 0
        self.max_failures_before_search = 3

        self.last_orientation: Optional[np.ndarray] = None
        self.ORI_MAX_DIFF_DEG = 50
        self.rejected_consecutive_frames_count = 0
        self.MAX_REJECTED_FRAMES = 5

        self._initialize_models()
        self.viewpoint_anchors = {}
        self._initialize_anchor_data()
        self.R_to, self.t_to = self._load_calibration(args.calibration)
        self.K, self.dist_coeffs = self._get_camera_intrinsics()
        
        self.aruco_dict, self.board = make_charuco_board(5, 7, 0.035, 0.025)
        self.aruco_det, self.charuco_det = make_detectors(self.aruco_dict, self.board)

        if self.args.save_matches:
            os.makedirs("output/matches", exist_ok=True)

    def run(self):
        while self.running:
            try:
                frame, frame_id = self.processing_queue.get(timeout=0.1)
                self._process_frame(frame, frame_id)
            except queue.Empty:
                continue
        self.save_results()

    def _calculate_pose_error(self, pose_tvec, pose_quat, gt_tvec, gt_rmat):
        pos_err = np.linalg.norm(pose_tvec - gt_tvec.flatten()) * 100
        
        pose_rmat = R.from_quat(pose_quat).as_matrix()
        R_err_mat = gt_rmat.T @ pose_rmat
        r_err_vec, _ = cv2.Rodrigues(R_err_mat)
        rot_err = np.linalg.norm(r_err_vec) * 180 / np.pi
        return pos_err, rot_err

    def _process_frame(self, frame, frame_id):
        packet = FrameDataPacket()
        error_log_entry = {"frame_id": frame_id}

        gt_success, R_ct, t_ct = self._detect_charuco_pose(frame)
        if gt_success:
            R_co_gt = R_ct @ self.R_to
            t_co_gt = (R_ct @ self.t_to) + t_ct
            packet.gt_pose = (t_co_gt, R_co_gt)

        with self.pose_data_lock:
            kf_pred_pos, kf_pred_quat = self.kf.x[0:3].copy(), self.kf.x[9:13].copy()
            is_kf_initialized = self.kf.initialized

        if gt_success and is_kf_initialized:
            pos_err, rot_err = self._calculate_pose_error(kf_pred_pos, kf_pred_quat, t_co_gt, R_co_gt)
            packet.errors['pred_pos'] = pos_err
            packet.errors['pred_rot'] = rot_err
            error_log_entry['pred_pos_err_cm'] = pos_err
            error_log_entry['pred_rot_err_deg'] = rot_err

        vape_pose_data = self._estimate_vape_pose(frame, frame_id)
        
        is_valid = False
        if vape_pose_data:
            if self.last_orientation is not None:
                angle_diff = math.degrees(self.quaternion_angle_diff(self.last_orientation, vape_pose_data.quaternion))
                if angle_diff <= self.ORI_MAX_DIFF_DEG:
                    is_valid = True
            else:
                is_valid = True

        if is_valid and vape_pose_data:
            self.rejected_consecutive_frames_count = 0
            vape_pos, vape_quat = vape_pose_data.position, vape_pose_data.quaternion
            self.last_orientation = vape_quat
            packet.raw_vape_pose = (vape_pos, R.from_quat(vape_quat).as_matrix())

            if gt_success:
                pos_err, rot_err = self._calculate_pose_error(vape_pos, vape_quat, t_co_gt, R_co_gt)
                packet.errors['meas_pos'] = pos_err
                packet.errors['meas_rot'] = rot_err
                error_log_entry['meas_pos_err_cm'] = pos_err
                error_log_entry['meas_rot_err_deg'] = rot_err

            with self.pose_data_lock:
                self.kf.update(vape_pos, vape_quat)
                if self.kf.initialized:
                    self.update_count += 1
                kf_updated_pos, kf_updated_quat = self.kf.x[0:3], self.kf.x[9:13]

            if gt_success:
                pos_err, rot_err = self._calculate_pose_error(kf_updated_pos, kf_updated_quat, t_co_gt, R_co_gt)
                packet.errors['filt_pos'] = pos_err
                packet.errors['filt_rot'] = rot_err
                error_log_entry['filt_pos_err_cm'] = pos_err
                error_log_entry['filt_rot_err_deg'] = rot_err
        else:
            self.rejected_consecutive_frames_count += 1
            if self.rejected_consecutive_frames_count >= self.MAX_REJECTED_FRAMES:
                with self.pose_data_lock: self.kf.initialized = False
                self.last_orientation = None
                self.current_best_viewpoint = None
                self.rejected_consecutive_frames_count = 0

        if len(error_log_entry) > 1:
            self.all_errors.append(error_log_entry)

        if self.visualization_queue.qsize() < 2:
            self.visualization_queue.put(packet)

    def _detect_charuco_pose(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.charuco_det is not None:
            corners, ids, _ = self.aruco_det.detectMarkers(gray)
            if ids is None: return False, None, None
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_det)
            if ids is None: return False, None, None
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
        
        if charuco_corners is not None and len(charuco_corners) > 4:
            pose_ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.board, self.K, self.dist_coeffs, None, None)
            if pose_ok:
                return True, cv2.Rodrigues(rvec)[0], tvec
        return False, None, None

    def save_results(self):
        if not self.all_errors:
            print("No valid comparisons were made.")
            return
        
        print("\n--- FINAL EVALUATION RESULTS ---")
        df = pd.DataFrame(self.all_errors)
        
        for err_type in ["pred", "meas", "filt"]:
            pos_col, rot_col = f'{err_type}_pos_err_cm', f'{err_type}_rot_err_deg'
            if pos_col in df.columns:
                print(f"\n--- {err_type.upper()} Errors ---")
                mean_pos_err, med_pos_err, std_pos_err = df[pos_col].mean(), df[pos_col].median(), df[pos_col].std()
                mean_rot_err, med_rot_err, std_rot_err = df[rot_col].mean(), df[rot_col].median(), df[rot_col].std()
                print(f"Position Error (cm): Mean={mean_pos_err:.2f}, Median={med_pos_err:.2f}, Std={std_pos_err:.2f}")
                print(f"Rotation Error (deg): Mean={mean_rot_err:.2f}, Median={med_rot_err:.2f}, Std={std_rot_err:.2f}")

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"evaluation_log_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        df.to_csv(filename, index=False)
        print(f"\n💾 Detailed log saved to {filename}")

    def _load_calibration(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return np.array(data['R_to']), np.array(data['t_to'])

    def _get_camera_intrinsics(self):
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32), None

    def _initialize_models(self):
        self.yolo_model = YOLO("best.pt").to(self.device)
        self.extractor = SuperPoint(max_num_keypoints=1024*2).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def _extract_features_sp(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0).to(self.device)/255.0
        with torch.no_grad(): return self.extractor.extract(tensor)

    def _yolo_detect(self, frame):
        names = getattr(self.yolo_model, "names", {0:"iha"})
        inv = {v:k for k,v in names.items()}
        target_id = inv.get("iha", 0)
        results = self.yolo_model(frame, conf=0.25, iou=0.5, max_det=5, classes=[target_id], verbose=False)
        if not results or len(results[0].boxes) == 0: return None
        best = max(results[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
        x1,y1,x2,y2 = best.xyxy[0].cpu().numpy().tolist()
        return int(x1),int(y1),int(x2),int(y2)

    def _estimate_vape_pose(self, frame: np.ndarray, frame_id: int) -> Optional[PoseData]:
        bbox = self._yolo_detect(frame)
        if not bbox: return None
        return self._estimate_pose_with_temporal_consistency(frame, bbox, frame_id)

    def _estimate_pose_with_temporal_consistency(self, frame: np.ndarray, bbox: Optional[Tuple], frame_id: int) -> Optional[PoseData]:
        if self.current_best_viewpoint:
            pose_data = self._solve_for_viewpoint(frame, self.current_best_viewpoint, bbox)
            if pose_data and pose_data.total_matches >= self.viewpoint_quality_threshold:
                self.consecutive_failures = 0
                return pose_data
            else:
                self.consecutive_failures += 1
        if self.current_best_viewpoint is None or self.consecutive_failures >= self.max_failures_before_search:
            ordered_viewpoints = self._quick_viewpoint_assessment(frame, bbox)
            successful_poses = []
            for viewpoint in ordered_viewpoints:
                pose_data = self._solve_for_viewpoint(frame, viewpoint, bbox)
                if pose_data:
                    successful_poses.append(pose_data)
                    if pose_data.total_matches >= self.viewpoint_quality_threshold * 2: break
            if successful_poses:
                best_pose = max(successful_poses, key=lambda p: (p.total_matches, p.inliers, -p.reprojection_error))
                self.current_best_viewpoint = best_pose.viewpoint
                self.consecutive_failures = 0
                if self.args.save_matches:
                    self._save_match_image(best_pose, frame_id)
                return best_pose
        return None

    def _save_match_image(self, pose_data: PoseData, frame_id: int):
        kpts1 = [cv2.KeyPoint(p[0], p[1], 1) for p in pose_data.kpts_anchor]
        kpts2 = [cv2.KeyPoint(p[0], p[1], 1) for p in pose_data.kpts_frame]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(pose_data.matches))]
        
        match_img = cv2.drawMatches(pose_data.anchor_image, kpts1, pose_data.frame_crop, kpts2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        filename = f"output/matches/match_frame_{frame_id:05d}_{pose_data.viewpoint}.png"
        cv2.imwrite(filename, match_img)

    def _quick_viewpoint_assessment(self, frame: np.ndarray, bbox: Optional[Tuple]) -> List[str]:
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else frame
        if crop.size == 0: return list(self.viewpoint_anchors.keys())
        frame_features = self._extract_features_sp(crop)
        viewpoint_scores = []
        for viewpoint, anchor in self.viewpoint_anchors.items():
            with torch.no_grad():
                matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
            matches = rbd(matches_dict)['matches'].cpu().numpy()
            valid_matches = sum(1 for anchor_idx, _ in matches if anchor_idx in anchor['map_3d'])
            viewpoint_scores.append((viewpoint, valid_matches))
        viewpoint_scores.sort(key=lambda x: x[1], reverse=True)
        return [vp for vp, score in viewpoint_scores]

    def _solve_for_viewpoint(self, frame: np.ndarray, viewpoint: str, bbox: Optional[Tuple]) -> Optional[PoseData]:
        anchor = self.viewpoint_anchors.get(viewpoint)
        if not anchor: return None
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox else frame
        if crop.size == 0: return None
        frame_features = self._extract_features_sp(crop)
        with torch.no_grad():
            matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
        matches = rbd(matches_dict)['matches'].cpu().numpy()
        if len(matches) < 6: return None
        points_3d, points_2d, matched_kpts_anchor, matched_kpts_frame, valid_matches_indices = [], [], [], [], []
        crop_offset = np.array([bbox[0], bbox[1]]) if bbox else np.array([0, 0])
        for i, (anchor_idx, frame_idx) in enumerate(matches):
            if anchor_idx in anchor['map_3d']:
                points_3d.append(anchor['map_3d'][anchor_idx])
                points_2d.append(frame_features['keypoints'][0].cpu().numpy()[frame_idx] + crop_offset)
                matched_kpts_anchor.append(anchor['features']['keypoints'][0].cpu().numpy()[anchor_idx])
                matched_kpts_frame.append(frame_features['keypoints'][0].cpu().numpy()[frame_idx])
                valid_matches_indices.append(i)

        if len(points_3d) < 6: return None
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(points_3d, dtype=np.float32), np.array(points_2d, dtype=np.float32), self.K, self.dist_coeffs, reprojectionError=8, confidence=0.95, iterationsCount=7000, flags=cv2.SOLVEPNP_EPNP)
            if success and inliers is not None and len(inliers) > 4:
                rvec, tvec = cv2.solvePnPRefineVVS(np.array(points_3d, dtype=np.float32)[inliers.flatten()], np.array(points_2d, dtype=np.float32)[inliers.flatten()], self.K, self.dist_coeffs, rvec, tvec)
                R_mat, _ = cv2.Rodrigues(rvec)
                position = tvec.flatten()
                quaternion = R.from_matrix(R_mat).as_quat()
                projected_points, _ = cv2.projectPoints(np.array(points_3d)[inliers.flatten()], rvec, tvec, self.K, self.dist_coeffs)
                error = np.mean(np.linalg.norm(np.array(points_2d)[inliers.flatten()].reshape(-1, 1, 2) - projected_points, axis=2))
                
                return PoseData(position, quaternion, len(inliers), error, viewpoint, len(points_3d),
                                frame_crop=crop, anchor_image=anchor['image'], 
                                matches=np.array(valid_matches_indices)[inliers.flatten()],
                                kpts_frame=np.array(matched_kpts_frame)[inliers.flatten()],
                                kpts_anchor=np.array(matched_kpts_anchor)[inliers.flatten()])
        except cv2.error: return None
        return None

    def _initialize_anchor_data(self):
        print("🛠️ Loading anchor data...")
        anchor_definitions = {
            'NE': {'path': 'NE.png', '2d': np.array([[928, 148],[570, 111],[401, 31],[544, 141],[530, 134],[351, 228],[338, 220],[294, 244],[230, 541],[401, 469],[414, 481],[464, 451],[521, 510],[610, 454],[544, 400],[589, 373],[575, 361],[486, 561],[739, 385],[826, 305],[791, 285],[773, 271],[845, 233],[826, 226],[699, 308],[790, 375]], dtype=np.float32), '3d': np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.000, -0.000, 0.156],[-0.014, 0.000, 0.156],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.019, -0.000, 0.074],[-0.014, 0.000, 0.042],[0.000, 0.000, 0.042],[-0.080, -0.000, 0.156],[-0.100, -0.030, 0.000],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.014, 0.000, -0.156],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.042],[-0.090, -0.000, -0.042]], dtype=np.float32)},
            'NW': {'path': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png', '2d': np.array([[511, 293],[591, 284],[587, 330],[413, 249],[602, 348],[715, 384],[598, 298],[656, 171],[805, 213],[703, 392],[523, 286],[519, 327],[387, 289],[727, 126],[425, 243],[636, 358],[745, 202],[595, 388],[436, 260],[539, 313],[795, 220],[351, 291],[665, 165],[611, 353],[650, 377],[516, 389],[727, 143],[496, 378],[575, 312],[617, 368],[430, 312],[480, 281],[834, 225],[469, 339],[705, 223],[637, 156],[816, 414],[357, 195],[752, 77],[642, 451]], dtype=np.float32), '3d': np.array([[-0.014, 0.0, 0.042],[0.025, -0.014, -0.011],[-0.014, 0.0, -0.042],[-0.014, 0.0, 0.156],[-0.023, 0.0, -0.065],[0.0, 0.0, -0.156],[0.025, 0.0, -0.015],[0.217, 0.0, 0.07],[0.23, 0.0, -0.07],[-0.014, 0.0, -0.156],[0.0, 0.0, 0.042],[-0.057, -0.018, -0.01],[-0.074, -0.0, 0.128],[0.206, -0.07, -0.002],[-0.0, -0.0, 0.156],[-0.017, -0.0, -0.092],[0.217, -0.0, -0.027],[-0.052, -0.0, -0.097],[-0.019, -0.0, 0.128],[-0.035, -0.018, -0.01], [0.217, -0.0, -0.07],[-0.08, -0.0, 0.156],[0.23, 0.0, 0.07],[-0.023, -0.0, -0.075],[-0.029, -0.0, -0.127],[-0.09, -0.0, -0.042],[0.206, -0.055, -0.002],[-0.09, -0.0, -0.015],[0.0, -0.0, -0.015],[-0.037, -0.0, -0.097],[-0.074, -0.0, 0.074],[-0.019, -0.0, 0.074],[0.23, -0.0, -0.113],[-0.1, -0.03, 0.0],[0.17, -0.0, -0.015],[0.23, -0.0, 0.113],[-0.0, -0.025, -0.24],[-0.0, -0.025, 0.24],[0.243, -0.104, 0.0],[-0.08, -0.0, -0.156]], dtype=np.float32)},
            'SE': {'path': 'SE.png', '2d': np.array([[415, 144],[1169, 508],[275, 323],[214, 395],[554, 670],[253, 428],[280, 415],[355, 365],[494, 621],[519, 600],[806, 213],[973, 438],[986, 421],[768, 343],[785, 328],[841, 345],[931, 393],[891, 306],[980, 345],[651, 210],[625, 225],[588, 216],[511, 215],[526, 204],[665, 271]], dtype=np.float32), '3d': np.array([[-0.0, -0.025, -0.24],[-0.0, -0.025, 0.24],[0.243, -0.104, 0.0],[0.23, 0.0, -0.113],[0.23, 0.0, 0.113],[0.23, 0.0, -0.07],[0.217, 0.0, -0.07],[0.206, -0.07, -0.002],[0.23, 0.0, 0.07],[0.217, 0.0, 0.07],[-0.1, -0.03, 0.0],[-0.0, 0.0, 0.156],[-0.014, 0.0, 0.156],[0.0, 0.0, 0.042],[-0.014, 0.0, 0.042],[-0.019, 0.0, 0.074],[-0.019, 0.0, 0.128],[-0.074, 0.0, 0.074],[-0.074, 0.0, 0.128],[-0.052, 0.0, -0.097],[-0.037, 0.0, -0.097],[-0.029, 0.0, -0.127],[0.0, 0.0, -0.156],[-0.014, 0.0, -0.156],[-0.014, 0.0, -0.042]], dtype=np.float32)},
            'SW': {'path': 'Anchor_B.png', '2d': np.array([[650, 312],[630, 306],[907, 443],[814, 291],[599, 349],[501, 386],[965, 359],[649, 355],[635, 346],[930, 335],[843, 467],[702, 339],[718, 321],[930, 322],[727, 346],[539, 364],[786, 297],[1022, 406],[1004, 399],[539, 344],[536, 309],[864, 478],[745, 310],[1049, 393],[895, 258],[674, 347],[741, 281],[699, 294],[817, 494],[992, 281]], dtype=np.float32), '3d': np.array([[-0.035, -0.018, -0.01],[-0.057, -0.018, -0.01],[0.217, -0.0, -0.027],[-0.014, -0.0, 0.156],[-0.023, 0.0, -0.065],[-0.014, -0.0, -0.156],[0.234, -0.05, -0.002],[0.0, -0.0, -0.042],[-0.014, -0.0, -0.042],[0.206, -0.055, -0.002],[0.217, -0.0, -0.07],[0.025, -0.014, -0.011],[-0.014, -0.0, 0.042],[0.206, -0.07, -0.002],[0.049, -0.016, -0.011],[-0.029, -0.0, -0.127],[-0.019, -0.0, 0.128],[0.23, -0.0, 0.07],[0.217, -0.0, 0.07],[-0.052, -0.0, -0.097],[-0.175, -0.0, -0.015],[0.23, -0.0, -0.07],[-0.019, -0.0, 0.074],[0.23, -0.0, 0.113],[-0.0, -0.025, 0.24],[-0.0, -0.0, -0.015],[-0.074, -0.0, 0.128],[-0.074, -0.0, 0.074],[0.23, -0.0, -0.113],[0.243, -0.104, 0.0]], dtype=np.float32)},
            'W': {'path': 'W.png', '2d': np.array([[589, 555],[565, 481],[531, 480],[329, 501],[326, 345],[528, 351],[395, 391],[469, 395],[529, 140],[381, 224],[504, 258],[498, 229],[383, 253],[1203, 100],[1099, 174],[1095, 211],[1201, 439],[1134, 404],[1100, 358],[625, 341],[624, 310],[315, 264]], dtype=np.float32), '3d': np.array([[-0.000, -0.025, -0.240],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.156],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.015],[-0.014, 0.000, -0.042],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.000, -0.025, 0.240],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.074],[0.243, -0.104, 0.000],[0.206, -0.070, -0.002],[0.206, -0.055, -0.002],[0.230, -0.000, -0.113],[0.217, -0.000, -0.070],[0.217, -0.000, -0.027],[0.025, 0.000, -0.015],[0.025, -0.014, -0.011],[-0.100, -0.030, 0.000]], dtype=np.float32)},
            'S': {'path': 'S.png', '2d': np.array([[14, 243],[1269, 255],[654, 183],[290, 484],[1020, 510],[398, 475],[390, 503],[901, 489],[573, 484],[250, 283],[405, 269],[435, 243],[968, 273],[838, 273],[831, 233],[949, 236]], dtype=np.float32), '3d': np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, 0.000, 0.070],[0.217, -0.000, -0.027],[0.000, 0.000, -0.156],[-0.017, -0.000, -0.092],[-0.052, -0.000, -0.097],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128]], dtype=np.float32)},
            'N': {'path': 'N.png', '2d': np.array([[1238, 346],[865, 295],[640, 89],[425, 314],[24, 383],[303, 439],[445, 434],[856, 418],[219, 475],[1055, 450]], dtype=np.float32), '3d': np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.052, -0.000, -0.097],[-0.080, -0.000, 0.156],[-0.080, -0.000, -0.156]], dtype=np.float32)},
            'SW2': {'path': 'SW2.png', '2d': np.array([[15, 300],[1269, 180],[635, 143],[434, 274],[421, 240],[273, 320],[565, 266],[844, 206],[468, 543],[1185, 466],[565, 506],[569, 530],[741, 491],[1070, 459],[1089, 480],[974, 220],[941, 184],[659, 269],[650, 299],[636, 210],[620, 193]], dtype=np.float32), '3d': np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.017, -0.000, -0.092],[-0.052, -0.000, -0.097],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.042],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, -0.000, -0.027],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[0.025, -0.014, -0.011],[0.025, 0.000, -0.015],[-0.035, -0.018, -0.010],[-0.057, -0.018, -0.010]], dtype=np.float32)},
            'SE2': {'path': 'SE2.png', '2d': np.array([[48, 216],[1269, 320],[459, 169],[853, 528],[143, 458],[244, 470],[258, 451],[423, 470],[741, 500],[739, 516],[689, 176],[960, 301],[828, 290],[970, 264],[850, 254]], dtype=np.float32), '3d': np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[0.230, -0.000, -0.113],[0.230, 0.000, -0.070],[0.217, -0.000, -0.070],[0.217, -0.000, -0.027],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[-0.100, -0.030, 0.000],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074]], dtype=np.float32)},
            'SU': {'path': 'SU.png', '2d': np.array([[203, 251],[496, 191],[486, 229],[480, 263],[368, 279],[369, 255],[573, 274],[781, 280],[859, 293],[865, 213],[775, 206],[1069, 326],[656, 135],[633, 241],[629, 204],[623, 343],[398, 668],[463, 680],[466, 656],[761, 706],[761, 681],[823, 709],[616, 666]], dtype=np.float32), '3d': np.array([[-0.000, -0.025, -0.240],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.017, -0.000, -0.092],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.156],[-0.014, 0.000, -0.042],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.035, -0.018, -0.010],[-0.057, -0.018, -0.010],[0.025, -0.014, -0.011],[0.230, -0.000, -0.113],[0.230, 0.000, -0.070],[0.217, -0.000, -0.070],[0.230, -0.000, 0.070],[0.217, 0.000, 0.070],[0.230, -0.000, 0.113],[0.243, -0.104, 0.000]], dtype=np.float32)},
            'NU': {'path': 'NU.png', '2d': np.array([[631, 361],[1025, 293],[245, 294],[488, 145],[645, 10],[803, 146],[661, 188],[509, 365],[421, 364],[434, 320],[509, 316],[779, 360],[784, 321],[704, 398],[358, 393]], dtype=np.float32), '3d': np.array([[-0.100, -0.030, 0.000],[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.230, -0.000, 0.113],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.170, -0.000, -0.015],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.090, -0.000, -0.042],[-0.080, -0.000, 0.156]], dtype=np.float32)},
            'NW2': {'path': 'NW2.png', '2d': np.array([[1268, 328],[1008, 419],[699, 399],[829, 373],[641, 325],[659, 310],[783, 30],[779, 113],[775, 153],[994, 240],[573, 226],[769, 265],[686, 284],[95, 269],[148, 375],[415, 353],[286, 349],[346, 320],[924, 360],[590, 324]], dtype=np.float32), '3d': np.array([[-0.000, -0.025, -0.240],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.042],[-0.052, -0.000, -0.097],[-0.057, -0.018, -0.010],[-0.035, -0.018, -0.010],[0.243, -0.104, 0.000],[0.206, -0.070, -0.002],[0.206, -0.055, -0.002],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.170, -0.000, -0.015],[0.025, -0.014, -0.011],[-0.000, -0.025, 0.240],[-0.080, -0.000, 0.156],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.128],[-0.029, -0.000, -0.127],[-0.100, -0.030, 0.000]], dtype=np.float32)},
            'NE2': {'path': 'NE2.png', '2d': np.array([[1035, 95],[740, 93],[599, 16],[486, 168],[301, 305],[719, 225],[425, 349],[950, 204],[794, 248],[844, 203],[833, 175],[601, 275],[515, 301],[584, 244],[503, 266]], dtype=np.float32), '3d': np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.080, -0.000, 0.156],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.042],[-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128]], dtype=np.float32)},
            'E': {'path': 'E.png', '2d': np.array([[696, 165],[46, 133],[771, 610],[943, 469],[921, 408],[793, 478],[781, 420],[793, 520],[856, 280],[743, 284],[740, 245],[711, 248],[74, 520],[134, 465],[964, 309]], dtype=np.float32), '3d': np.array([[-0.000, -0.025, -0.240],[0.243, -0.104, 0.000],[-0.000, -0.025, 0.240],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.014, 0.000, 0.156],[-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.014, 0.000, -0.156],[0.000, 0.000, -0.156],[0.230, -0.000, 0.113],[0.217, 0.000, 0.070],[-0.100, -0.030, 0.000]], dtype=np.float32)}
        }
        for viewpoint, data in anchor_definitions.items():
            if os.path.exists(data['path']):
                anchor_image = cv2.resize(cv2.imread(data['path']), (1280, 720))
                anchor_features = self._extract_features_sp(anchor_image)
                anchor_keypoints = anchor_features['keypoints'][0].cpu().numpy()
                sp_tree = cKDTree(anchor_keypoints)
                distances, indices = sp_tree.query(data['2d'], k=1, distance_upper_bound=5.0)
                valid_mask = distances != np.inf
                self.viewpoint_anchors[viewpoint] = {
                    'image': anchor_image,
                    'features': anchor_features,
                    'map_3d': {idx: pt for idx, pt in zip(indices[valid_mask], data['3d'][valid_mask])}
                }

    def quaternion_angle_diff(self, q1: np.ndarray, q2: np.ndarray) -> float:
        dot = np.dot(normalize_quaternion(q1), normalize_quaternion(q2))
        return 2 * math.acos(abs(min(1.0, max(-1.0, dot))))

# --- HIGH-FREQUENCY MAIN THREAD ---
class MainThread(threading.Thread):
    def __init__(self, processing_queue, visualization_queue, pose_data_lock, kf, args):
        super().__init__()
        self.running = True
        self.processing_queue = processing_queue
        self.visualization_queue = visualization_queue
        self.pose_data_lock = pose_data_lock
        self.kf = kf
        self.args = args
        self.camera_width, self.camera_height = 1280, 720
        self.K, self.dist_coeffs = self._get_camera_intrinsics()
        self.latest_vis_packet = None

        if self.args.save_frames:
            self.frames_out_dir = "output/frames"
            os.makedirs(self.frames_out_dir, exist_ok=True)

    def run(self):
        cap = cv2.VideoCapture(self.args.video)
        if not cap.isOpened(): raise IOError(f"Cannot open video: {self.args.video}")
        
        fps_filter = 0.9
        current_fps = 0.0
        
        frame_id = 0
        while self.running:
            loop_start_time = time.time()

            ret, frame = cap.read()
            if not ret: break

            with self.pose_data_lock:
                self.kf.predict()
                if self.kf.initialized:
                    kf_pos, kf_quat = self.kf.x[0:3], self.kf.x[9:13]
                else:
                    kf_pos, kf_quat = None, None

            if self.processing_queue.qsize() < 2:
                self.processing_queue.put((frame.copy(), frame_id))

            try: 
                self.latest_vis_packet = self.visualization_queue.get_nowait()
            except queue.Empty: 
                pass

            vis_frame = self._draw_visualizations(frame, kf_pos, kf_quat)
            cv2.imshow('VAPE KF Evaluator', vis_frame)

            if self.args.save_frames:
                cv2.imwrite(os.path.join(self.frames_out_dir, f"frame_{frame_id:05d}.png"), vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): self.running = False

            frame_rate_cap = 30.0
            time_to_wait = (1.0 / frame_rate_cap) - (time.time() - loop_start_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            
            frame_id += 1
        
        self.running = False
        cap.release()
        cv2.destroyAllWindows()

    def _draw_visualizations(self, frame, kf_pos, kf_quat):
        vis_frame = frame.copy() 
        
        if self.latest_vis_packet:
            if self.latest_vis_packet.gt_pose:
                gt_pos, gt_R = self.latest_vis_packet.gt_pose
                self._draw_axes(vis_frame, gt_pos, R_mat=gt_R, label="Ground Truth (Green)", color=(0, 255, 0))

            if self.latest_vis_packet.raw_vape_pose:
                raw_pos, raw_R = self.latest_vis_packet.raw_vape_pose
                self._draw_axes(vis_frame, raw_pos, R_mat=raw_R, label="VAPE Raw (Red)", color=(0, 0, 255))
            
            y_offset = 60
            for err_type in [('pred', "Pred"), ('meas', "Meas"), ('filt', "Filt")]:
                pos_key, rot_key = f'{err_type[0]}_pos', f'{err_type[0]}_rot'
                if pos_key in self.latest_vis_packet.errors:
                    p_err = self.latest_vis_packet.errors[pos_key]
                    r_err = self.latest_vis_packet.errors[rot_key]
                    cv2.putText(vis_frame, f"{err_type[1]} Err (cm,deg): {p_err:.2f}, {r_err:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    y_offset += 25

        if kf_pos is not None:
            self._draw_axes(vis_frame, kf_pos, quat=kf_quat, label="VAPE KF (Blue)", color=(255, 0, 0))

        return vis_frame

    def _draw_axes(self, frame, pos, quat=None, R_mat=None, label="", color=(255,255,255), length=0.1):
        try:
            if R_mat is None: R_mat = R.from_quat(quat).as_matrix()
            rvec, _ = cv2.Rodrigues(R_mat)
            img_pts, _ = cv2.projectPoints(np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]]), rvec, pos, self.K, self.dist_coeffs)
            img_pts = img_pts.reshape(-1,2).astype(int)
            cv2.line(frame, tuple(img_pts[0]), tuple(img_pts[1]), (0,0,255), 2)
            cv2.line(frame, tuple(img_pts[0]), tuple(img_pts[2]), (0,255,0), 2)
            cv2.line(frame, tuple(img_pts[0]), tuple(img_pts[3]), (255,0,0), 2)
            cv2.putText(frame, label, (img_pts[0][0], img_pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except:
            pass

    def _get_camera_intrinsics(self):
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32), None

def main():
    parser = argparse.ArgumentParser(description="VAPE MK52 KF Evaluator - Enhanced Debugging")
    parser.add_argument('--video', required=True, help='Path to input MP4 video')
    parser.add_argument('--calibration', required=True, help='Path to the calibration JSON file')
    parser.add_argument('--save-frames', action='store_true', help='Save every visualized frame to output/frames/')
    parser.add_argument('--save-matches', action='store_true', help='Save feature matching images to output/matches/')
    args = parser.parse_args()

    # Import pandas here so it's not a hard dependency for running the main script
    globals()["pd"] = __import__("pandas")

    processing_queue = queue.Queue(maxsize=2)
    visualization_queue = queue.Queue(maxsize=2)
    pose_data_lock = threading.Lock()

    kf = UnscentedKalmanFilter()
    main_thread = MainThread(processing_queue, visualization_queue, pose_data_lock, kf, args)
    processing_thread = ProcessingThread(processing_queue, visualization_queue, pose_data_lock, kf, args)

    print("Starting VAPE KF Evaluator...")
    main_thread.start()
    processing_thread.start()

    main_thread.join()
    processing_thread.running = False
    processing_thread.join()
    print("✅ Evaluation finished.")

if __name__ == '__main__':
    main()