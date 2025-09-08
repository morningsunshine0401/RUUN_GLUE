# -*- coding: utf-8 -*- 
# 파일 이름: VAPE_MK53_Core.py (최종 안정화 버전 - 범용 매칭 시각화 기능 추가) 

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
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
import queue
import math
from collections import deque

# --- NEW: 추가된 임포트 ---
from scipy.spatial.transform import Rotation as R_scipy
from scipy.spatial import cKDTree

# --- Dependency Imports ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

# --- Utility Functions ---
def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-10 else np.array([0.0, 0.0, 0.0, 1.0])

def quaternion_to_xyzw(q_input):
    q = np.array(q_input)
    if abs(q[0]) > abs(q[3]) and abs(q[0]) > 0.7: return np.array([q[1], q[2], q[3], q[0]])
    return q

def quat_mul(a, b):
    x1, y1, z1, w1 = a; x2, y2, z2, w2 = b
    return np.array([ w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2, w1*w2 - x1*x2 - y1*y2 - z1*z2 ])
def quat_conj(q): return np.array([-q[0], -q[1], -q[2], q[3]])
def quat_inv(q): return quat_conj(normalize_quaternion(q))

def quat_to_axis_angle(q):
    qn = normalize_quaternion(q)
    w = float(np.clip(qn[3], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w*w, 0.0))
    axis = np.array([1.0, 0.0, 0.0]) if s < 1e-8 else qn[:3]/s
    return axis, angle

def axis_angle_to_quat(axis, angle):
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    s = np.sin(angle/2.0)
    return normalize_quaternion(np.array([ax[0]*s, ax[1]*s, ax[2]*s, np.cos(angle/2.0)]))

def clamp_quaternion_towards(q_from, q_to, max_deg_per_s, dt):
    if np.isinf(max_deg_per_s): return normalize_quaternion(q_to)
    if np.dot(q_from, q_to) < 0.0: q_to = -q_to
    dq = quat_mul(quat_inv(q_from), q_to)
    axis, ang = quat_to_axis_angle(dq)
    ang_limit = np.deg2rad(max_deg_per_s) * max(dt, 1e-6)
    if ang > ang_limit:
        dq = axis_angle_to_quat(axis, ang_limit)
        q_out = quat_mul(q_from, dq)
    else: q_out = q_to
    return normalize_quaternion(q_out)

def clamp_position_towards(pos_from, pos_to, max_speed_m_per_s, dt):
    if np.isinf(max_speed_m_per_s): return pos_to
    delta_pos = pos_to - pos_from
    distance = np.linalg.norm(delta_pos)
    max_distance = max_speed_m_per_s * max(dt, 1e-6)
    if distance > max_distance:
        direction = delta_pos / distance
        pos_out = pos_from + direction * max_distance
    else: pos_out = pos_to
    return pos_out

def quaternion_angle_diff_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    if q1 is None or q2 is None: return 0.0
    dot = np.dot(normalize_quaternion(q1), normalize_quaternion(q2))
    return 2 * math.degrees(math.acos(abs(min(1.0, max(-1.0, dot)))))

def make_charuco_board(cols, rows, square_len_m, marker_len_m,
                       dict_id=cv2.aruco.DICT_6X6_250):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(cols, rows, square_len_m, marker_len_m, aruco_dict)
    else:
        board = cv2.aruco.CharucoBoard((cols, rows), square_len_m, marker_len_m, aruco_dict)
    return aruco_dict, board

def make_detectors(aruco_dict, board):
    aruco_params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        aruco_det = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        charuco_det = cv2.aruco.CharucoDetector(board)
        return ("new", aruco_det, charuco_det)
    else:
        return ("old", aruco_params, None)


@dataclass
class PoseData:
    position: np.ndarray; quaternion: np.ndarray; inliers: int
    reprojection_error: float; viewpoint: str; total_matches: int

@dataclass
class ProcessingResult:
    frame_id: int; capture_time: float; frame: np.ndarray
    pose_success: bool = False; position: Optional[np.ndarray] = None
    quaternion: Optional[np.ndarray] = None; num_inliers: int = 0
    viewpoint_used: Optional[str] = None; kf_position: Optional[np.ndarray] = None
    kf_quaternion: Optional[np.ndarray] = None; bbox: Optional[Tuple[int, int, int, int]] = None
    detection_time_ms: float = 0.0; feature_time_ms: float = 0.0
    matching_time_ms: float = 0.0; pnp_time_ms: float = 0.0
    kf_time_ms: float = 0.0; is_accepted_by_gate: bool = False
    gate_rejection_reason: str = ""; raw_matches: int = 0
    inlier_ratio: Optional[float] = None; oos_replay_triggered: bool = False
    rate_limited_rot_deg: float = 0.0; rate_limited_pos_m: float = 0.0
    gt_available: bool = False
    gt_position: Optional[np.ndarray] = None
    gt_quaternion: Optional[np.ndarray] = None
    filt_pos_err_cm: Optional[float] = None
    filt_rot_err_deg: Optional[float] = None
    jitter_lin_vel_mps: Optional[float] = None
    jitter_ang_vel_dps: Optional[float] = None
    recovery_frames: int = 0

class UnscentedKalmanFilter:
    def __init__(self, dt=1/15.0):
        self.dt = dt
        self.initialized = False
        self.max_rot_rate_dps = 30.0
        self.max_pos_speed_mps = 1.5
        self.n = 16
        self.m = 7
        self.x = np.zeros(self.n)
        self.x[12] = 1.0
        self.P = np.eye(self.n) * 0.1
        self.P0 = self.P.copy()
        self.alpha = 1e-3; self.beta = 2.0; self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        self.wm = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + self.lambda_)))
        self.wc = self.wm.copy()
        self.wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.wc[0] = self.lambda_ / (self.n + self.lambda_) + (1.0 - self.alpha**2 + self.beta)
        self.Q = np.eye(self.n)
        self.Q[0:3, 0:3] *= 1e-3
        self.Q[3:6, 3:6] *= 1e-2
        self.Q[6:9, 6:9] *= 1e-2
        self.Q[9:13, 9:13] *= 1e-3
        self.Q[13:16, 13:16] *= 1e-2
        self.R = np.eye(self.m) * 1e-4
        self.t_state = None
        self.history = deque(maxlen=200)

    def _push_history(self):
        if self.t_state is not None:
            self.history.append((self.t_state, self.x.copy(), self.P.copy()))

    def _generate_sigma_points(self, x, P):
        sigmas = np.zeros((2 * self.n + 1, self.n))
        P_sym = 0.5 * (P + P.T)
        try: U = np.linalg.cholesky((self.n + self.lambda_) * P_sym)
        except np.linalg.LinAlgError:
            P_jitter = P_sym + 1e-9 * np.eye(self.n)
            try: U = np.linalg.cholesky((self.n + self.lambda_) * P_jitter)
            except np.linalg.LinAlgError:
                U_svd, s, _ = np.linalg.svd(P_jitter)
                U = U_svd @ np.diag(np.sqrt(np.maximum(s, 1e-12))) @ U_svd.T
                U = U * np.sqrt(self.n + self.lambda_)
        sigmas[0] = x
        for i in range(self.n):
            sigmas[i+1] = x + U[:, i]; sigmas[self.n+i+1] = x - U[:, i]
        return sigmas

    def motion_model(self, x_in, dt):
        x_out = np.zeros_like(x_in)
        pos, vel, acc = x_in[0:3], x_in[3:6], x_in[6:9]
        x_out[0:3] = pos + vel * dt + 0.5 * acc * dt**2
        x_out[3:6] = vel + acc * dt
        x_out[6:9] = acc
        q, w = x_in[9:13], x_in[13:16]
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        omega_mat = 0.5 * np.array([ [-qx, -qy, -qz], [qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qw] ])
        q_dot = omega_mat @ w
        q_new = q + dt * q_dot
        x_out[9:13] = normalize_quaternion(q_new)
        x_out[13:16] = w
        return x_out

    def predict(self, dt):
        if not self.initialized: return self.x[0:3], self.x[9:13]
        sigmas = self._generate_sigma_points(self.x, self.P)
        sigmas_f = np.array([self.motion_model(s, dt) for s in sigmas])
        x_pred = np.sum(self.wm[:, np.newaxis] * sigmas_f, axis=0)
        x_pred[9:13] = normalize_quaternion(x_pred[9:13])
        Q_scaled = self.Q * dt + self.Q * (dt**2) * 0.5
        P_pred = Q_scaled.copy()
        for i in range(2 * self.n + 1):
            y = sigmas_f[i] - x_pred
            P_pred += self.wc[i] * np.outer(y, y)
        self.P = 0.5 * (P_pred + P_pred.T)
        self.x = x_pred
        return self.x[0:3], self.x[9:13]

    def hx(self, x_in):
        z = np.zeros(self.m)
        z[0:3] = x_in[0:3]
        z[3:7] = normalize_quaternion(x_in[9:13])
        return z

    def _measurement_update(self, z_pos, z_quat, R):
        if self.initialized and np.dot(self.x[9:13], z_quat) < 0.0: z_quat = -z_quat
        measurement = np.concatenate([z_pos, normalize_quaternion(z_quat)])
        sigmas_f = self._generate_sigma_points(self.x, self.P)
        sigmas_h = np.array([self.hx(s) for s in sigmas_f])
        z_pred = np.sum(self.wm[:, np.newaxis] * sigmas_h, axis=0)
        z_pred[3:7] = normalize_quaternion(z_pred[3:7])
        S = R.copy()
        for i in range(2 * self.n + 1):
            y = sigmas_h[i] - z_pred
            S += self.wc[i] * np.outer(y, y)
        S = 0.5 * (S + S.T)
        P_xz = np.zeros((self.n, self.m))
        for i in range(2 * self.n + 1):
            P_xz += self.wc[i] * np.outer(sigmas_f[i] - self.x, sigmas_h[i] - z_pred)
        try: K = P_xz @ np.linalg.inv(S)
        except np.linalg.LinAlgError: K = P_xz @ np.linalg.pinv(S)
        self.x += K @ (measurement - z_pred)
        self.x[9:13] = normalize_quaternion(self.x[9:13])
        self.P -= K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        min_eigenval = np.min(np.real(np.linalg.eigvals(self.P)))
        if min_eigenval < 1e-12: self.P += (1e-9 - min_eigenval) * np.eye(self.n)

    def update_with_timestamp(self, z_pos, z_quat, t_meas, R=None, t_now=None):
        oos_replay_triggered = False
        if R is None: R = self.R.copy()
        if self.t_state is None:
            self.x[0:3] = z_pos
            self.x[9:13] = normalize_quaternion(z_quat)
            self.P[:] = self.P0
            self.t_state = t_meas
            self.initialized = True
            return self.x[0:3], self.x[9:13], (0,0), oos_replay_triggered
        
        pre_update_pos, pre_update_quat = self.x[0:3].copy(), self.x[9:13].copy()
        
        if t_meas >= self.t_state:
            dt1 = t_meas - self.t_state
            if dt1 > 0: self._push_history(); self.predict(dt1); self.t_state = t_meas
            dt_to_meas = dt1 if dt1 > 0 else 1e-6
            pos_clamped = clamp_position_towards(self.x[0:3], z_pos, self.max_pos_speed_mps, dt_to_meas)
            q_clamped = clamp_quaternion_towards(self.x[9:13], normalize_quaternion(z_quat), self.max_rot_rate_dps, dt_to_meas)
            self._measurement_update(pos_clamped, q_clamped, R)
            if t_now is not None and t_now > t_meas:
                dt2 = t_now - t_meas
                self._push_history(); self.predict(dt2); self.t_state = t_now
        else: # Out-of-sequence measurement
            oos_replay_triggered = True
            valid_history = [(i, t, x, P) for i, (t, x, P) in enumerate(self.history) if t <= t_meas]
            if not valid_history:
                dt1 = max(0, t_meas - self.t_state)
                if dt1 > 0: self._push_history(); self.predict(dt1); self.t_state = t_meas
                dt_to_meas = dt1 if dt1 > 0 else 1e-6
                pos_clamped = clamp_position_towards(self.x[0:3], z_pos, self.max_pos_speed_mps, dt_to_meas)
                q_clamped = clamp_quaternion_towards(self.x[9:13], normalize_quaternion(z_quat), self.max_rot_rate_dps, dt_to_meas)
                self._measurement_update(pos_clamped, q_clamped, R)
            else:
                k, t_k, x_k, P_k = valid_history[-1]
                self.x[:], self.P[:], self.t_state = x_k, P_k, t_k
                dt_to_meas = t_meas - t_k if t_meas > t_k else 1e-6
                if t_meas > t_k: self.predict(dt_to_meas); self.t_state = t_meas
                pos_clamped = clamp_position_towards(self.x[0:3], z_pos, self.max_pos_speed_mps, dt_to_meas)
                q_clamped = clamp_quaternion_towards(self.x[9:13], normalize_quaternion(z_quat), self.max_rot_rate_dps, dt_to_meas)
                self._measurement_update(pos_clamped, q_clamped, R)
                for j in range(k + 1, len(self.history)):
                    t_next = self.history[j][0]
                    dt_replay = t_next - self.t_state
                    if dt_replay > 0: self.predict(dt_replay); self.t_state = t_next
        
        rate_limited_pos_m = np.linalg.norm(self.x[0:3] - pre_update_pos)
        rate_limited_rot_deg = quaternion_angle_diff_deg(self.x[9:13], pre_update_quat)
        return self.x[0:3], self.x[9:13], (rate_limited_rot_deg, rate_limited_pos_m), oos_replay_triggered

    def set_rate_limits(self, max_rotation_dps: float = None, max_position_mps: float = None):
        if max_rotation_dps is not None: self.max_rot_rate_dps = max_rotation_dps
        if max_position_mps is not None: self.max_pos_speed_mps = max_position_mps

    def reset(self):
        self.initialized = False
        self.x = np.zeros(self.n)
        self.x[12] = 1.0
        self.P = self.P0.copy()
        self.t_state = None
        self.history.clear()

    def apply_damping(self, damping_factor=0.9):
        if self.initialized:
            self.x[3:9] *= damping_factor 
            self.x[13:16] *= damping_factor

class VAPE53_Estimator:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.kf = UnscentedKalmanFilter()
        if args.rate_limits.lower() != 'inf,inf':
            rot, pos = [float(x) if x != 'inf' else float('inf') for x in args.rate_limits.split(',')]
            self.kf.set_rate_limits(max_rotation_dps=rot, max_position_mps=pos)
        else: self.kf.set_rate_limits(float('inf'), float('inf'))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.camera_width, self.camera_height = 1280, 720
        
        self.jsonl_fp = None
        self._initialize_logging()
        
        self.last_bbox, self.last_det_frame_id = None, -999999
        self.current_best_viewpoint = None
        self.consecutive_failures = 0
        self.viewpoint_switch_votes = 0
        self.last_valid_orientation: Optional[np.ndarray] = None
        self.prev_successful_pose: Optional[Dict[str, Any]] = None
        self.rejected_consecutive_frames_count = 0
        self.MAX_REJECTED_FRAMES = 5
        self.frames_since_last_success = 0
        
        self.yolo_model, self.extractor, self.matcher = None, None, None
        self.viewpoint_anchors = {}
        self.coarse_4_views = ['NW', 'SW', 'NE', 'SE']

        from lightglue import LightGlue, SuperPoint
        from ultralytics import YOLO

        self._initialize_models(YOLO, SuperPoint, LightGlue)
        self._initialize_anchor_data()
        self.K, self.dist_coeffs = self._get_camera_intrinsics()

        self.R_to, self.t_to = self._load_calibration(args.calibration)
        
        print("Initializing A1 ChArUco board for ground truth...")
        self.aruco_dict, self.board = make_charuco_board(
            cols=7, rows=11,
            square_len_m=0.0765,
            marker_len_m=0.0573,
            dict_id=cv2.aruco.DICT_6X6_250
        )
        self.detector_mode, self.aruco_det, self.charuco_det = make_detectors(self.aruco_dict, self.board)
        print(f"  - Ground truth detection mode: {self.detector_mode}")

        self._initialize_input_source()
        self.display_queue = queue.Queue(maxsize=2) if args.show else None
        if self.args.show:
            self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
            self.display_thread.start()

    def _initialize_input_source(self):
        if self.args.video_file or self.args.webcam:
            self.video_capture = cv2.VideoCapture(0 if self.args.webcam else self.args.video_file)
            if not self.video_capture.isOpened(): raise IOError("Cannot open input source.")
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        elif self.args.image_dir:
            if not os.path.exists(self.args.image_dir): raise FileNotFoundError(f"Image directory not found: {self.args.image_dir}")
            self.image_files = sorted([os.path.join(self.args.image_dir, f) for f in os.listdir(self.args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not self.image_files: raise IOError(f"No images found in directory: {self.args.image_dir}")
        else: raise ValueError("No input source specified.")

    def _get_next_frame(self):
        if self.args.video_file or self.args.webcam:
            ret, frame = self.video_capture.read()
            return frame if ret else None
        else:
            if self.frame_idx < len(self.image_files):
                frame = cv2.imread(self.image_files[self.frame_idx])
                self.frame_idx += 1
                return frame
            return None

    def _load_calibration(self, path):
        with open(path, 'r') as f: data = json.load(f)
        return np.array(data['R_to']), np.array(data['t_to'])

    def _detect_charuco_pose(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.detector_mode == "new":
            corners, ids, _ = self.aruco_det.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_det)
        if ids is None or len(ids) == 0: return False, None, None
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
        if charuco_corners is None or len(charuco_corners) < 4: return False, None, None
        pose_ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.board, self.K, self.dist_coeffs, None, None)
        if pose_ok: return True, cv2.Rodrigues(rvec)[0], tvec
        return False, None, None
        
    # def _calculate_pose_error(self, pose_tvec, pose_quat, gt_tvec, gt_rmat):
    #     pos_err = np.linalg.norm(pose_tvec - gt_tvec.flatten()) * 100
    #     pose_rmat = R_scipy.from_quat(pose_quat).as_matrix()
    #     R_err_mat = gt_rmat.T @ pose_rmat
    #     r_err_vec, _ = cv2.Rodrigues(R_err_mat)
    #     rot_err = np.linalg.norm(r_err_vec) * 180 / np.pi
    #     return pos_err, rot_err
    
    def _calculate_pose_error(self, pose_tvec, pose_quat, gt_tvec, gt_rmat):
        """Calculates the position and rotation error between an estimated pose and ground truth."""
        
        # Calculate the positional error in centimeters
        pos_err = np.linalg.norm(pose_tvec - gt_tvec.flatten()) * 100
        
        # The KF and GT quaternions are in [x, y, z, w] format, which is what scipy expects.
        pose_rmat = R_scipy.from_quat(pose_quat).as_matrix()

        # Calculate the rotational error in degrees
        R_err_mat = gt_rmat.T @ pose_rmat
        r_err_vec, _ = cv2.Rodrigues(R_err_mat)
        rot_err = np.linalg.norm(r_err_vec) * 180 / np.pi
        
        return pos_err, rot_err

    
    def _initialize_logging(self):
        if self.args.log_jsonl:
            out_dir = Path(self.args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self.jsonl_fp = (out_dir / "frames.jsonl").open("w")
            print(f"📝 Logging per-frame data to {out_dir / 'frames.jsonl'}")

    

    def _initialize_models(self, YOLO, SuperPoint, LightGlue):
        print("📦 Loading models...")
        if not self.args.no_det: self.yolo_model = YOLO("best.pt").to(self.device)
        if self.args.matcher in ['lightglue', 'nnrt']:
            self.extractor = SuperPoint(max_num_keypoints=self.args.sp_kpts).eval().to(self.device)
            if self.args.matcher == 'lightglue':
                self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        elif self.args.matcher == 'orb':
            self.orb = cv2.ORB_create(nfeatures=self.args.orb_nfeatures)
        elif self.args.matcher == 'sift':
            self.sift = cv2.SIFT_create(nfeatures=self.args.sift_nfeatures)
        print("   ...models loaded.")

    def _initialize_anchor_data(self):
        print("🛠️ Initializing anchor data...")
        # Hardcoded anchor data (same as before)
        # Hardcoded anchor data (same as before)
        ne_anchor_2d = np.array([[928, 148],[570, 111],[401, 31],[544, 141],[530, 134],[351, 228],[338, 220],[294, 244],[230, 541],[401, 469],[414, 481],[464, 451],[521, 510],[610, 454],[544, 400],[589, 373],[575, 361],[486, 561],[739, 385],[826, 305],[791, 285],[773, 271],[845, 233],[826, 226],[699, 308],[790, 375]], dtype=np.float32)
        ne_anchor_3d = np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.000, -0.000, 0.156],[-0.014, 0.000, 0.156],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.019, -0.000, 0.074],[-0.014, 0.000, 0.042],[0.000, 0.000, 0.042],[-0.080, -0.000, 0.156],[-0.100, -0.030, 0.000],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.014, 0.000, -0.156],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.042],[-0.090, -0.000, -0.042]], dtype=np.float32)
        nw_anchor_2d = np.array([[511, 293], [591, 284], [587, 330], [413, 249], [602, 348], [715, 384], [598, 298], [656, 171], [805, 213], [703, 392], [523, 286], [519, 327], [387, 289], [727, 126], [425, 243], [636, 358], [745, 202], [595, 388], [436, 260], [539, 313], [795, 220], [351, 291], [665, 165], [611, 353], [650, 377], [516, 389], [727, 143], [496, 378], [575, 312], [617, 368], [430, 312], [480, 281], [834, 225], [469, 339], [705, 223], [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]], dtype=np.float32)
        nw_anchor_3d = np.array([[-0.014, 0.0, 0.042], [0.025, -0.014, -0.011], [-0.014, 0.0, -0.042], [-0.014, 0.0, 0.156], [-0.023, 0.0, -0.065], [0.0, 0.0, -0.156], [0.025, 0.0, -0.015], [0.217, 0.0, 0.07], [0.23, 0.0, -0.07], [-0.014, 0.0, -0.156], [0.0, 0.0, 0.042], [-0.057, -0.018, -0.01], [-0.074, -0.0, 0.128], [0.206, -0.07, -0.002], [-0.0, -0.0, 0.156], [-0.017, -0.0, -0.092], [0.217, -0.0, -0.027], [-0.052, -0.0, -0.097], [-0.019, -0.0, 0.128], [-0.035, -0.018, -0.01], [0.217, -0.0, -0.07], [-0.08, -0.0, 0.156], [0.23, 0.0, 0.07], [-0.023, -0.0, -0.075], [-0.029, -0.0, -0.127], [-0.09, -0.0, -0.042], [0.206, -0.055, -0.002], [-0.09, -0.0, -0.015], [0.0, -0.0, -0.015], [-0.037, -0.0, -0.097], [-0.074, -0.0, 0.074], [-0.019, -0.0, 0.074], [0.23, -0.0, -0.113], [-0.1, -0.03, 0.0], [0.17, -0.0, -0.015], [0.23, -0.0, 0.113], [-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [-0.08, -0.0, -0.156]], dtype=np.float32)
        se_anchor_2d = np.array([[415, 144], [1169, 508], [275, 323], [214, 395], [554, 670], [253, 428], [280, 415], [355, 365], [494, 621], [519, 600], [806, 213], [973, 438], [986, 421], [768, 343], [785, 328], [841, 345], [931, 393], [891, 306], [980, 345], [651, 210], [625, 225], [588, 216], [511, 215], [526, 204], [665, 271]], dtype=np.float32)
        se_anchor_3d = np.array([[-0.0, -0.025, -0.24], [-0.0, -0.025, 0.24], [0.243, -0.104, 0.0], [0.23, 0.0, -0.113], [0.23, 0.0, 0.113], [0.23, 0.0, -0.07], [0.217, 0.0, -0.07], [0.206, -0.07, -0.002], [0.23, 0.0, 0.07], [0.217, 0.0, 0.07], [-0.1, -0.03, 0.0], [-0.0, 0.0, 0.156], [-0.014, 0.0, 0.156], [0.0, 0.0, 0.042], [-0.014, 0.0, 0.042], [-0.019, 0.0, 0.074], [-0.019, 0.0, 0.128], [-0.074, 0.0, 0.074], [-0.074, 0.0, 0.128], [-0.052, 0.0, -0.097], [-0.037, 0.0, -0.097], [-0.029, 0.0, -0.127], [0.0, 0.0, -0.156], [-0.014, 0.0, -0.156], [-0.014, 0.0, -0.042]], dtype=np.float32)
        sw_anchor_2d = np.array([[650, 312], [630, 306], [907, 443], [814, 291], [599, 349], [501, 386], [965, 359], [649, 355], [635, 346], [930, 335], [843, 467], [702, 339], [718, 321], [930, 322], [727, 346], [539, 364], [786, 297], [1022, 406], [1004, 399], [539, 344], [536, 309], [864, 478], [745, 310], [1049, 393], [895, 258], [674, 347], [741, 281], [699, 294], [817, 494], [992, 281]], dtype=np.float32)
        sw_anchor_3d = np.array([[-0.035, -0.018, -0.01], [-0.057, -0.018, -0.01], [0.217, -0.0, -0.027], [-0.014, -0.0, 0.156], [-0.023, 0.0, -0.065], [-0.014, -0.0, -0.156], [0.234, -0.05, -0.002], [0.0, -0.0, -0.042], [-0.014, -0.0, -0.042], [0.206, -0.055, -0.002], [0.217, -0.0, -0.07], [0.025, -0.014, -0.011], [-0.014, -0.0, 0.042], [0.206, -0.07, -0.002], [0.049, -0.016, -0.011], [-0.029, -0.0, -0.127], [-0.019, -0.0, 0.128], [0.23, -0.0, 0.07], [0.217, -0.0, 0.07], [-0.052, -0.0, -0.097], [-0.175, -0.0, -0.015], [0.23, -0.0, -0.07], [-0.019, -0.0, 0.074], [0.23, -0.0, 0.113], [-0.0, -0.025, 0.24], [-0.0, -0.0, -0.015], [-0.074, -0.0, 0.128], [-0.074, -0.0, 0.074], [0.23, -0.0, -0.113], [0.243, -0.104, 0.0]], dtype=np.float32)
        w_anchor_2d = np.array([[589, 555],[565, 481],[531, 480],[329, 501],[326, 345],[528, 351],[395, 391],[469, 395],[529, 140],[381, 224],[504, 258],[498, 229],[383, 253],[1203, 100],[1099, 174],[1095, 211],[1201, 439],[1134, 404],[1100, 358],[625, 341],[624, 310],[315, 264]], dtype=np.float32)
        w_anchor_3d = np.array([[-0.000, -0.025, -0.240],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.156],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.015],[-0.014, 0.000, -0.042],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.000, -0.025, 0.240],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.074],[0.243, -0.104, 0.000],[0.206, -0.070, -0.002],[0.206, -0.055, -0.002],[0.230, -0.000, -0.113],[0.217, -0.000, -0.070],[0.217, -0.000, -0.027],[0.025, 0.000, -0.015],[0.025, -0.014, -0.011],[-0.100, -0.030, 0.000]], dtype=np.float32)
        s_anchor_2d = np.array([[14, 243],[1269, 255],[654, 183],[290, 484],[1020, 510],[398, 475],[390, 503],[901, 489],[573, 484],[250, 283],[405, 269],[435, 243],[968, 273],[838, 273],[831, 233],[949, 236]], dtype=np.float32)
        s_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, 0.000, 0.070],[0.217, -0.000, -0.027],[0.000, 0.000, -0.156],[-0.017, -0.000, -0.092],[-0.052, -0.000, -0.097],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128]], dtype=np.float32)
        n_anchor_2d = np.array([[1238, 346],[865, 295],[640, 89],[425, 314],[24, 383],[303, 439],[445, 434],[856, 418],[219, 475],[1055, 450]], dtype=np.float32)
        n_anchor_3d = np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.052, -0.000, -0.097],[-0.080, -0.000, 0.156],[-0.080, -0.000, -0.156]], dtype=np.float32)
        e_anchor_2d = np.array([[696, 165],[46, 133],[771, 610],[943, 469],[921, 408],[793, 478],[781, 420],[793, 520],[856, 280],[743, 284],[740, 245],[711, 248],[74, 520],[134, 465],[964, 309]], dtype=np.float32)
        e_anchor_3d = np.array([[-0.000, -0.025, -0.240],[0.243, -0.104, 0.000],[-0.000, -0.025, 0.240],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.014, 0.000, 0.156],[-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.014, 0.000, -0.156],[0.000, 0.000, -0.156],[0.230, -0.000, 0.113],[0.217, 0.000, 0.070],[-0.100, -0.030, 0.000]], dtype=np.float32)
        sw2_anchor_2d = np.array([[15, 300],[1269, 180],[635, 143],[434, 274],[421, 240],[273, 320],[565, 266],[844, 206],[468, 543],[1185, 466],[565, 506],[569, 530],[741, 491],[1070, 459],[1089, 480],[974, 220],[941, 184],[659, 269],[650, 299],[636, 210],[620, 193]], dtype=np.float32)
        sw2_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.017, -0.000, -0.092],[-0.052, -0.000, -0.097],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.042],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, -0.000, -0.027],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[0.025, -0.014, -0.011],[0.025, 0.000, -0.015],[-0.035, -0.018, -0.010],[-0.057, -0.018, -0.010]], dtype=np.float32)
        se2_anchor_2d = np.array([[48, 216],[1269, 320],[459, 169],[853, 528],[143, 458],[244, 470],[258, 451],[423, 470],[741, 500],[739, 516],[689, 176],[960, 301],[828, 290],[970, 264],[850, 254]], dtype=np.float32)
        se2_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[0.230, -0.000, -0.113],[0.230, 0.000, -0.070],[0.217, -0.000, -0.070],[0.217, -0.000, -0.027],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],[-0.100, -0.030, 0.000],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074]], dtype=np.float32)
        su_anchor_2d = np.array([[203, 251],[496, 191],[486, 229],[480, 263],[368, 279],[369, 255],[573, 274],[781, 280],[859, 293],[865, 213],[775, 206],[1069, 326],[656, 135],[633, 241],[629, 204],[623, 343],[398, 668],[463, 680],[466, 656],[761, 706],[761, 681],[823, 709],[616, 666]], dtype=np.float32)
        su_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.017, -0.000, -0.092],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.156],[-0.014, 0.000, -0.042],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.035, -0.018, -0.010],[-0.057, -0.018, -0.010],[0.025, -0.014, -0.011],[0.230, -0.000, -0.113],[0.230, 0.000, -0.070],[0.217, -0.000, -0.070],[0.230, -0.000, 0.070],[0.217, 0.000, 0.070],[0.230, -0.000, 0.113],[0.243, -0.104, 0.000]], dtype=np.float32)
        nu_anchor_2d = np.array([[631, 361],[1025, 293],[245, 294],[488, 145],[645, 10],[803, 146],[661, 188],[509, 365],[421, 364],[434, 320],[509, 316],[779, 360],[784, 321],[704, 398],[358, 393]], dtype=np.float32)
        nu_anchor_3d = np.array([[-0.100, -0.030, 0.000],[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.230, -0.000, 0.113],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],[0.170, -0.000, -0.015],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.090, -0.000, -0.042],[-0.080, -0.000, 0.156]], dtype=np.float32)
        nw2_anchor_2d = np.array([[1268, 328],[1008, 419],[699, 399],[829, 373],[641, 325],[659, 310],[783, 30],[779, 113],[775, 153],[994, 240],[573, 226],[769, 265],[686, 284],[95, 269],[148, 375],[415, 353],[286, 349],[346, 320],[924, 360],[590, 324]], dtype=np.float32)
        nw2_anchor_3d = np.array([[-0.000, -0.025, -0.240],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.042],[-0.052, -0.000, -0.097],[-0.057, -0.018, -0.010],[-0.035, -0.018, -0.010],[0.243, -0.104, 0.000],[0.206, -0.070, -0.002],[0.206, -0.055, -0.002],[0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.170, -0.000, -0.015],[0.025, -0.014, -0.011],[-0.000, -0.025, 0.240],[-0.080, -0.000, 0.156],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.128],[-0.029, -0.000, -0.127],[-0.100, -0.030, 0.000]], dtype=np.float32)
        ne2_anchor_2d = np.array([[1035, 95],[740, 93],[599, 16],[486, 168],[301, 305],[719, 225],[425, 349],[950, 204],[794, 248],[844, 203],[833, 175],[601, 275],[515, 301],[584, 244],[503, 266]], dtype=np.float32)
        ne2_anchor_3d = np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],[-0.080, -0.000, 0.156],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.042],[-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128]], dtype=np.float32)

        anchor_definitions = {
            'NE': {'path': 'NE.png', '2d': ne_anchor_2d, '3d': ne_anchor_3d},
            'NW': {'path': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png', '2d': nw_anchor_2d, '3d': nw_anchor_3d},
            'SE': {'path': 'SE.png', '2d': se_anchor_2d, '3d': se_anchor_3d},
            'W': {'path': 'W.png', '2d': w_anchor_2d, '3d': w_anchor_3d},
            'S': {'path': 'S.png', '2d': s_anchor_2d, '3d': s_anchor_3d},
            'SW': {'path': 'Anchor_B.png', '2d': sw_anchor_2d, '3d': sw_anchor_3d},
            'N': {'path': 'N.png', '2d': n_anchor_2d, '3d': n_anchor_3d},
            'SW2': {'path': 'SW2.png', '2d': sw2_anchor_2d, '3d': sw2_anchor_3d},
            'SE2': {'path': 'SE2.png', '2d': se2_anchor_2d, '3d': se2_anchor_3d},
            'SU': {'path': 'SU.png', '2d': su_anchor_2d, '3d': su_anchor_3d},
            'NU': {'path': 'NU.png', '2d': nu_anchor_2d, '3d': nu_anchor_3d},
            'NW2': {'path': 'NW2.png', '2d': nw2_anchor_2d, '3d': nw2_anchor_3d},
            'NE2': {'path': 'NE2.png', '2d': ne2_anchor_2d, '3d': ne2_anchor_3d},
            'E': {'path': 'E.png', '2d': e_anchor_2d, '3d': e_anchor_3d},
        }
        self.viewpoint_anchors = {}
        for viewpoint, data in anchor_definitions.items():
            if not os.path.exists(data['path']): continue
            anchor_image_bgr = cv2.resize(cv2.imread(data['path']), (self.camera_width, self.camera_height))
            base_anchor_data = {'image': anchor_image_bgr.copy()}
            if self.args.matcher in ['lightglue', 'nnrt']:
                anchor_features = self._extract_features_sp(anchor_image_bgr)
                anchor_keypoints_np = anchor_features['keypoints'][0].cpu().numpy()
                base_anchor_data['features'] = anchor_features
                if anchor_keypoints_np.shape[0] > 0:
                    sp_tree = cKDTree(anchor_keypoints_np)
                    distances, indices = sp_tree.query(data['2d'], k=1, distance_upper_bound=5.0)
                    valid_mask = distances != np.inf
                    base_anchor_data['map_3d'] = {idx: pt for idx, pt in zip(indices[valid_mask], data['3d'][valid_mask])}
                else: base_anchor_data['map_3d'] = {}
                self.viewpoint_anchors[viewpoint] = base_anchor_data
        print("   ...anchor data initialized.")
        self.all_view_ids = list(self.viewpoint_anchors.keys())
        self.coarse_4_views = [v for v in self.coarse_4_views if v in self.all_view_ids]

    def run(self):
        frame_count = 0
        while self.running:
            frame = self._get_next_frame()
            if frame is None:
                break

            if frame_count % 100 == 0:
                print(f"[INFO] Processing MK53 frame {frame_count}...")
            
            t_capture = time.monotonic()
            result = self._process_frame(frame, frame_count, t_capture)
            
            if self.args.show:
                vis_frame = self._update_display(result)
                
                # Display the main video frame
                cv2.imshow("VAPE MK53 Comparison", vis_frame)
                
                # Display the latest feature match image if it exists
                if self.vis_match_image is not None:
                    cv2.imshow("Feature Matches", self.vis_match_image)

                # Check for 'q' key press to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            
            frame_count += 1
            
        self.cleanup()



    def _sanitize_for_json(self, data: dict) -> dict:
        """Creates a copy of the data dict and makes it JSON serializable."""
        # Create a shallow copy to avoid modifying the original object
        sanitized_data = data.copy()
        
        # Remove the raw image frame, as it's large and not serializable
        if 'frame' in sanitized_data:
            del sanitized_data['frame']

        # Iterate over the items and convert NumPy arrays to lists
        for key, value in sanitized_data.items():
            if isinstance(value, np.ndarray):
                sanitized_data[key] = value.tolist()
        
        return sanitized_data

    # Replace your existing _log_frame function with this corrected version.

    def _log_frame(self, data: dict):
        """Logs a sanitized, JSON-serializable version of the frame data."""
        if self.jsonl_fp:
            # Sanitize the data before attempting to serialize it
            sanitized_dict = self._sanitize_for_json(data)
            self.jsonl_fp.write(json.dumps(sanitized_dict, ensure_ascii=False) + "\n")
            self.jsonl_fp.flush()
    
    def _passes_gates(self, prev_q, new_q, match_ratio):
        """
        Checks if a new pose measurement is valid based on orientation jumps and match quality.
        This is the "Hard Gate" that rejects bad measurements entirely.
        """
        gate_ori_ok, gate_ratio_ok, reason = True, True, ""
        
        if self.args.gate_ori_deg >= 0 and prev_q is not None and new_q is not None:
            angle = quaternion_angle_diff_deg(prev_q, new_q)
            if angle > self.args.gate_ori_deg:
                gate_ori_ok = False
                reason += f"ori_jump({angle:.1f}d) "
                
        if match_ratio < self.args.gate_ratio:
            gate_ratio_ok = False
            reason += f"low_ratio({match_ratio:.2f}) "
            
        # --- THIS IS THE FINAL, CRITICAL CHANGE ---
        # The logic must be a strict "and" to prevent bad data from poisoning the filter.
        passes = (gate_ori_ok and gate_ratio_ok)
        # --- END OF CHANGE ---

        return passes, reason.strip()

    def _process_frame(self, frame: np.ndarray, frame_id: int, t_capture: float):
        result = ProcessingResult(frame_id=frame_id, frame=frame.copy(), capture_time=t_capture)
        
        # --- 1. Ground Truth Pose Calculation ---
        gt_success, R_ct, t_ct = self._detect_charuco_pose(frame)
        if gt_success:
            R_co_gt = R_ct @ self.R_to
            t_co_gt = (R_ct @ self.t_to) + t_ct
            result.gt_available = True
            result.gt_position = t_co_gt.flatten()
            result.gt_quaternion = R_scipy.from_matrix(R_co_gt).as_quat()

        # --- 2. VAPE Pose Estimation Pipeline ---
        t_start_det = time.monotonic()
        bbox_detected = self._yolo_detect(frame)
        result.bbox = bbox_detected
        result.detection_time_ms = (time.monotonic() - t_start_det) * 1000

        best_pose, metrics = self._estimate_pose_with_temporal_consistency(frame, result.bbox)
        result.feature_time_ms = metrics.get("feature_time_ms", 0.0)
        result.matching_time_ms = metrics.get("matching_time_ms", 0.0)
        result.pnp_time_ms = metrics.get("pnp_time_ms", 0.0)
        result.raw_matches = metrics.get("raw_matches", 0)

        # --- 3. Gating Logic (The "Hard Gate") ---
        is_accepted_by_gate = False
        if best_pose:
            match_ratio = best_pose.inliers / best_pose.total_matches if best_pose.total_matches > 0 else 0.0
            result.inlier_ratio = match_ratio
            is_accepted_by_gate, result.gate_rejection_reason = self._passes_gates(
                self.last_valid_orientation, best_pose.quaternion, match_ratio
            )
        result.is_accepted_by_gate = is_accepted_by_gate

        # --- 4. Kalman Filter Update (with internal "Soft Gate") and Final Evaluation ---
        t_start_kf = time.monotonic()
        
        if is_accepted_by_gate and best_pose:
            if self.frames_since_last_success > 0:
                result.recovery_frames = self.frames_since_last_success
            self.frames_since_last_success = 0
            self.rejected_consecutive_frames_count = 0
            
            result.pose_success = True
            result.position, result.quaternion, result.num_inliers, result.viewpoint_used = best_pose.position, best_pose.quaternion, best_pose.inliers, best_pose.viewpoint
            self.last_valid_orientation = best_pose.quaternion
            
            # RESTORED LOGIC: Use the time-aware update, which contains the "Soft Gate"
            kf_pos, kf_quat, _, _ = self.kf.update_with_timestamp(
                best_pose.position, best_pose.quaternion, t_meas=t_capture, R=None, t_now=time.monotonic()
            )
            
            result.kf_position, result.kf_quaternion = kf_pos, kf_quat
                
            if result.gt_available:
                # The quaternion from the KF is already in the correct [x,y,z,w] format for scipy
                pos_err, rot_err = self._calculate_pose_error(result.kf_position, result.kf_quaternion, result.gt_position, R_scipy.from_quat(result.gt_quaternion).as_matrix())
                result.filt_pos_err_cm, result.filt_rot_err_deg = pos_err, rot_err

            if self.prev_successful_pose:
                dt = result.capture_time - self.prev_successful_pose['t_capture']
                if dt > 1e-6:
                    dist = np.linalg.norm(result.kf_position - self.prev_successful_pose['kf_pos'])
                    ang_diff = quaternion_angle_diff_deg(result.kf_quaternion, self.prev_successful_pose['kf_quat'])
                    result.jitter_lin_vel_mps, result.jitter_ang_vel_dps = dist / dt, ang_diff / dt
            self.prev_successful_pose = {'t_capture': result.capture_time, 'kf_pos': result.kf_position, 'kf_quat': result.kf_quaternion}
        else:
            # If the pose is rejected, handle the failure
            self.frames_since_last_success += 1
            self.prev_successful_pose = None
            self.rejected_consecutive_frames_count += 1
            self.kf.apply_damping()
            if self.rejected_consecutive_frames_count >= self.MAX_REJECTED_FRAMES:
                self.kf.reset()
                self.last_valid_orientation, self.current_best_viewpoint, self.rejected_consecutive_frames_count = None, None, 0
                
        result.kf_time_ms = (time.monotonic() - t_start_kf) * 1000
        self._log_frame(result.__dict__)



    def _estimate_pose_with_temporal_consistency(self, frame, bbox):
        pose_metrics = {}
        # First, try the last known good viewpoint
        if self.current_best_viewpoint:
            pose_data, metrics = self._solve_for_viewpoint(frame, self.current_best_viewpoint, bbox)
            pose_metrics.update(metrics)
            if pose_data and pose_data.total_matches >= 5: # Using 5 as the quality threshold from VAPE_MK53_3.py
                self.consecutive_failures = 0
                return pose_data, pose_metrics
            else:
                self.consecutive_failures += 1

        # If the current viewpoint fails, or we don't have one, start a search
        if self.current_best_viewpoint is None or self.consecutive_failures >= self.args.vp_failures:
            if self.args.vp_mode == "coarse4": candidate_viewpoints = self.coarse_4_views
            elif self.args.vp_mode == "fixed": candidate_viewpoints = [self.all_view_ids[self.args.fixed_view]] if self.all_view_ids else []
            else: candidate_viewpoints = self.all_view_ids
            
            print("🔍 Searching for best viewpoint...")
            ordered_viewpoints, assessment_metrics = self._quick_viewpoint_assessment(frame, bbox, candidate_viewpoints)
            pose_metrics.update(assessment_metrics)
            
            successful_poses = []
            # This is the exhaustive search from VAPE_MK53_3.py
            viewpoint_quality_threshold = 5
            for viewpoint in ordered_viewpoints:
                pose_data, metrics = self._solve_for_viewpoint(frame, viewpoint, bbox)
                if pose_data:
                    successful_poses.append(pose_data)
                    if pose_data.total_matches >= viewpoint_quality_threshold * 2:
                        break
            
            if successful_poses:
                best_pose = max(successful_poses, key=lambda p: (p.total_matches, p.inliers, -p.reprojection_error))
                self.current_best_viewpoint = best_pose.viewpoint
                self.consecutive_failures = 0
                print(f"🎯 Selected viewpoint: {best_pose.viewpoint} ({best_pose.total_matches} matches, {best_pose.inliers} inliers)")
                return best_pose, pose_metrics

        return None, pose_metrics

    def _quick_viewpoint_assessment(self, frame, bbox, candidates):
        from lightglue.utils import rbd
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox and bbox[2]>bbox[0] and bbox[3]>bbox[1] else frame
        if crop.size == 0: return self.all_view_ids, {}
        t_start, viewpoint_scores = time.monotonic(), []
        for viewpoint in candidates:
            anchor = self.viewpoint_anchors.get(viewpoint)
            if not anchor or not anchor.get('features'): continue
            with torch.no_grad():
                frame_features = self._extract_features_sp(crop)
                matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
            matches = rbd(matches_dict)['matches'].cpu().numpy()
            valid_matches = sum(1 for anchor_idx, _ in matches if anchor_idx in anchor.get('map_3d', {}))
            viewpoint_scores.append((viewpoint, valid_matches))
        viewpoint_scores.sort(key=lambda x: x[1], reverse=True)
        return [vp for vp, score in viewpoint_scores], {"assessment_time_ms": (time.monotonic() - t_start) * 1000}

    # def _solve_for_viewpoint(self, frame, viewpoint, bbox):
    #     anchor = self.viewpoint_anchors.get(viewpoint)
    #     if not anchor: return None, {}
    #     crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox and bbox[2]>bbox[0] and bbox[3]>bbox[1] else frame
    #     if crop.size == 0: return None, {}
        
    #     t_start_feat = time.monotonic()
    #     matches, raw_count, frame_features, metrics = self._get_features_and_matches(anchor, crop)
    #     t_end_match = time.monotonic()
    #     metrics["feature_time_ms"] = metrics.get("feature_time_ms", (t_end_match - t_start_feat) * 1000)
    #     metrics["matching_time_ms"] = (t_end_match - t_start_feat) * 1000 - metrics["feature_time_ms"]
    #     metrics["raw_matches"] = raw_count
        
    #     if len(matches) < 6: return None, metrics
        
    #     points_3d, points_2d = [], []
    #     crop_offset = np.array([bbox[0], bbox[1]]) if bbox else np.array([0, 0])
        
    #     if self.args.matcher in ['lightglue', 'nnrt']:
    #         kps_frame = frame_features['keypoints'][0].cpu().numpy()
    #         for anchor_idx, frame_idx in matches:
    #             if anchor_idx in anchor.get('map_3d', {}):
    #                 points_3d.append(anchor['map_3d'][anchor_idx])
    #                 points_2d.append(kps_frame[frame_idx] + crop_offset)
        
    #     if len(points_3d) < 6: return None, metrics
        
    #     t_start_pnp = time.monotonic()
    #     success, pnp_result = self._do_pnp(np.array(points_3d), np.array(points_2d), self.K, self.dist_coeffs)
    #     metrics["pnp_time_ms"] = (time.monotonic() - t_start_pnp) * 1000
        
    #     if not success: return None, metrics
    #     rvec, tvec, inliers = pnp_result
    #     if inliers is None or len(inliers) < 4: return None, metrics
        
    #     R, _ = cv2.Rodrigues(rvec)
    #     #position, quaternion = tvec.flatten(), normalize_quaternion(quaternion_to_xyzw(self._rotation_matrix_to_quaternion(R)))
    #     position, quaternion = tvec.flatten(), normalize_quaternion(self._rotation_matrix_to_quaternion(R))
    #     inlier_pts_3d = np.array(points_3d)[inliers.flatten()]
    #     inlier_pts_2d = np.array(points_2d)[inliers.flatten()].reshape(-1, 1, 2)
    #     projected_points, _ = cv2.projectPoints(inlier_pts_3d, rvec, tvec, self.K, self.dist_coeffs)
    #     error = np.mean(np.linalg.norm(inlier_pts_2d - projected_points, axis=2))
    #     return PoseData(position, quaternion, len(inliers), error, viewpoint, len(points_3d)), metrics

    #     # # --- START OF FIX ---
    #     # # 1. Correctly treat the PnP result as the pose of the TAG relative to the camera (ct).
    #     # R_ct, _ = cv2.Rodrigues(rvec)
    #     # t_ct = tvec

    #     # # 2. Apply the calibration transformation (from the JSON file) to get the OBJECT's pose (co).
    #     # #    This aligns the VAPE pipeline with the Ground Truth pipeline.
    #     # R_co = R_ct @ self.R_to
    #     # t_co = (R_ct @ self.t_to.reshape(3, 1)) + t_ct

    #     # # 3. The final, corrected pose for the object.
    #     # position = t_co.flatten()
    #     # quaternion = normalize_quaternion(quaternion_to_xyzw(self._rotation_matrix_to_quaternion(R_co)))
    #     # # --- END OF FIX ---
        
    #     # # Reprojection error is still calculated with the original rvec/tvec from PnP
    #     # inlier_pts_3d = np.array(points_3d)[inliers.flatten()]
    #     # inlier_pts_2d = np.array(points_2d)[inliers.flatten()].reshape(-1, 1, 2)
    #     # projected_points, _ = cv2.projectPoints(inlier_pts_3d, rvec, tvec, self.K, self.dist_coeffs)
    #     # error = np.mean(np.linalg.norm(inlier_pts_2d - projected_points, axis=2))

    #     # return PoseData(position, quaternion, len(inliers), error, viewpoint, len(points_3d)), metrics

    def _solve_for_viewpoint(self, frame, viewpoint, bbox):
        anchor = self.viewpoint_anchors.get(viewpoint)
        if not anchor: return None, {}
        
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] if bbox and bbox[2]>bbox[0] and bbox[3]>bbox[1] else frame
        if crop.size == 0: return None, {}
        
        # Step 1: Get 2D-3D matches (This part is unchanged)
        t_start_feat = time.monotonic()
        matches, raw_count, frame_features, metrics = self._get_features_and_matches(anchor, crop)
        t_end_match = time.monotonic()
        metrics["feature_time_ms"] = metrics.get("feature_time_ms", (t_end_match - t_start_feat) * 1000)
        metrics["matching_time_ms"] = (t_end_match - t_start_feat) * 1000 - metrics["feature_time_ms"]
        metrics["raw_matches"] = raw_count
        
        if len(matches) < 6: return None, metrics
        
        points_3d, points_2d = [], []
        crop_offset = np.array([bbox[0], bbox[1]]) if bbox else np.array([0, 0])
        
        kps_frame = frame_features['keypoints'][0].cpu().numpy()
        for anchor_idx, frame_idx in matches:
            if anchor_idx in anchor.get('map_3d', {}):
                points_3d.append(anchor['map_3d'][anchor_idx])
                points_2d.append(kps_frame[frame_idx] + crop_offset)
        
        if len(points_3d) < 6: return None, metrics
        
        # Step 2: Perform PnP (This part is unchanged)
        t_start_pnp = time.monotonic()
        success, pnp_result = self._do_pnp(np.array(points_3d), np.array(points_2d), self.K, self.dist_coeffs)
        metrics["pnp_time_ms"] = (time.monotonic() - t_start_pnp) * 1000
        
        if not success: return None, metrics
        rvec, tvec, inliers = pnp_result
        if inliers is None or len(inliers) < 4: return None, metrics
        
        # --- START OF THE FINAL FIX ---

        # Step 3: Treat the PnP result as the pose of the TAG relative to the camera (ct).
        R_ct, _ = cv2.Rodrigues(rvec)
        t_ct = tvec

        # Step 4: Apply the calibration transformation to get the final OBJECT pose (co).
        # This makes the VAPE pipeline identical to the Ground Truth pipeline.
        R_co = R_ct @ self.R_to
        t_co = (R_ct @ self.t_to.reshape(3, 1)) + t_ct

        # Step 5: Create the final pose data using the corrected values.
        # The quaternion is now calculated from the correctly transformed rotation matrix (R_co).
        position = t_co.flatten()
        quaternion = normalize_quaternion(self._rotation_matrix_to_quaternion(R_co))
        
        # --- END OF THE FINAL FIX ---
        
        # Reprojection error is still calculated using the original PnP result (rvec, tvec)
        # as it relates to the raw 2D-3D matches.
        inlier_pts_3d = np.array(points_3d)[inliers.flatten()]
        inlier_pts_2d = np.array(points_2d)[inliers.flatten()].reshape(-1, 1, 2)
        projected_points, _ = cv2.projectPoints(inlier_pts_3d, rvec, tvec, self.K, self.dist_coeffs)
        error = np.mean(np.linalg.norm(inlier_pts_2d - projected_points, axis=2))
        
        return PoseData(position, quaternion, len(inliers), error, viewpoint, raw_count), metrics
        
    def _yolo_detect(self, frame):
        if self.yolo_model is None: return None
        names = getattr(self.yolo_model, "names", {0: "iha"}); inv = {v: k for k, v in names.items()}; target_id = inv.get("iha", 0)
        for conf_thresh in (0.30, 0.20, 0.10):
            results = self.yolo_model(frame, imgsz=640, conf=conf_thresh, iou=0.5, max_det=5, classes=[target_id], verbose=False)
            if not results or len(results[0].boxes) == 0: continue
            best = max(results[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1]))
            x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().tolist()
            return int(x1), int(y1), int(x2), int(y2)
        return None

    def _extract_features_sp(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad(): return self.extractor.extract(tensor)

    def _get_camera_intrinsics(self):
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32), None

    def _rotation_matrix_to_quaternion(self, R):
        tr = np.trace(R)
        if tr > 0: S = np.sqrt(tr + 1.0) * 2; qw=0.25*S; qx=(R[2,1]-R[1,2])/S; qy=(R[0,2]-R[2,0])/S; qz=(R[1,0]-R[0,1])/S
        elif (R[0,0]>R[1,1])and(R[0,0]>R[2,2]): S=np.sqrt(1.+R[0,0]-R[1,1]-R[2,2])*2; qw=(R[2,1]-R[1,2])/S; qx=0.25*S; qy=(R[0,1]+R[1,0])/S; qz=(R[0,2]+R[2,0])/S
        elif R[1,1]>R[2,2]: S=np.sqrt(1.+R[1,1]-R[0,0]-R[2,2])*2; qw=(R[0,2]-R[2,0])/S; qx=(R[0,1]+R[1,0])/S; qy=0.25*S; qz=(R[1,2]+R[2,1])/S
        else: S=np.sqrt(1.+R[2,2]-R[0,0]-R[1,1])*2; qw=(R[1,0]-R[0,1])/S; qx=(R[0,2]+R[2,0])/S; qy=(R[1,2]+R[2,1])/S; qz=0.25*S
        return np.array([qx, qy, qz, qw])
    
    def _expand_bbox(self, b, margin, W, H):
        if b is None: return None
        x1, y1, x2, y2 = b
        return int(max(0, x1 - margin)), int(max(0, y1 - margin)), int(min(W - 1, x2 + margin)), int(min(H - 1, y2 + margin))

    def _get_features_and_matches(self, anchor_data, crop_img):
        from lightglue.utils import rbd
        metrics, frame_features_out = {}, None
        t_feat_start = time.monotonic()
        if self.args.matcher in ['lightglue', 'nnrt']:
            frame_features = self._extract_features_sp(crop_img)
            metrics["feature_time_ms"] = (time.monotonic() - t_feat_start) * 1000
            if self.args.matcher == 'lightglue':
                with torch.no_grad(): matches_dict = self.matcher({'image0': anchor_data['features'], 'image1': frame_features})
                matches = rbd(matches_dict)['matches'].cpu().numpy()
            else:
                des_crop, des_anchor = frame_features['descriptors'][0].cpu().numpy().T, anchor_data['features']['descriptors'][0].cpu().numpy().T
                knn_matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_anchor, des_crop, k=2)
                matches = np.array([[m.queryIdx, m.trainIdx] for m, n in knn_matches if m and n and m.distance < self.args.nn_ratio * n.distance], dtype=int)
            raw_count, frame_features_out = len(matches), frame_features
            if self.args.show and self.visualization_queue.qsize() < 2:
                kps_anchor = [cv2.KeyPoint(p[0],p[1],1) for p in anchor_data['features']['keypoints'][0].cpu().numpy()]
                kps_frame = [cv2.KeyPoint(p[0],p[1],1) for p in frame_features['keypoints'][0].cpu().numpy()]
                dmatches = [cv2.DMatch(m[0],m[1],0) for m in matches]
                img_matches = cv2.drawMatches(anchor_data['image'], kps_anchor, crop_img, kps_frame, dmatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                self.visualization_queue.put({'match_image': img_matches})
            return matches, raw_count, frame_features_out, metrics
        return np.empty((0,2),dtype=int),0,None,{}
    
    def _update_display(self, result: ProcessingResult):
        """Draws all visualization overlays onto the frame."""
        vis_frame = result.frame.copy()
        
        # Draw the bounding box
        if result.bbox:
            cv2.rectangle(vis_frame, (result.bbox[0], result.bbox[1]), (result.bbox[2], result.bbox[3]), (0, 255, 0), 2)

        # Draw the final, filtered pose from the Kalman Filter
        if result.kf_position is not None and result.kf_quaternion is not None:
            self._draw_axes(vis_frame, result.kf_position, result.kf_quaternion)
        
        # Draw the ground truth pose in bright, distinct colors for comparison
        if result.gt_available:
            gt_colors = ((255, 255, 0), (255, 0, 255), (0, 255, 255)) # Cyan, Magenta, Yellow
            self._draw_axes(vis_frame, result.gt_position, result.gt_quaternion, color_override=gt_colors)

        # Add status text
        status_color = (0, 255, 0) if result.pose_success else (0, 0, 255)
        cv2.putText(vis_frame, f"STATUS: {'SUCCESS' if result.pose_success else 'FAILED'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        if not result.pose_success and result.gate_rejection_reason:
            cv2.putText(vis_frame, f"REASON: {result.gate_rejection_reason}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        return vis_frame

    def _draw_axes(self, frame, position, quaternion, color_override=None):
        """Draws the 3D axes on the frame for a given pose."""
        try:
            colors = color_override if color_override else ((0, 0, 255), (0, 255, 0), (255, 0, 0)) # BGR: Red, Green, Blue
            
            # Ensure quaternion is in [x, y, z, w] format for scipy
            q = np.asarray(quaternion)
            if q[3] > 0.999: # Handle potential scalar-first format
                q_xyzw = np.array([q[0], q[1], q[2], q[3]])
            else: # Default to assuming it's [w,x,y,z] if unsure, but log a warning
                q_xyzw = np.array([q[1], q[2], q[3], q[0]])

            R = R_scipy.from_quat(q_xyzw).as_matrix()
            rvec, _ = cv2.Rodrigues(R)
            tvec = position.reshape(3, 1)
            
            axis_pts = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, self.K, self.dist_coeffs)
            img_pts = img_pts.reshape(-1, 2).astype(int)
            
            origin = tuple(img_pts[0])
            cv2.line(frame, origin, tuple(img_pts[1]), colors[0], 3) # X-axis
            cv2.line(frame, origin, tuple(img_pts[2]), colors[1], 3) # Y-axis
            cv2.line(frame, origin, tuple(img_pts[3]), colors[2], 3) # Z-axis
        except (cv2.error, AttributeError, ValueError):
            pass

    def _do_pnp(self, pts_3d, pts_2d, K, dist):
        flag = cv2.SOLVEPNP_EPNP if self.args.pnp_solver == 'epnp' else cv2.SOLVEPNP_ITERATIVE
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, K, dist, flags=flag, reprojectionError=self.args.pnp_reproj, iterationsCount=self.args.pnp_iters, confidence=self.args.pnp_conf)
            if not ok or inliers is None: return False, None
            if self.args.pnp_refine and len(inliers) > 4: rvec, tvec = cv2.solvePnPRefineVVS(pts_3d[inliers.flatten()], pts_2d[inliers.flatten()], K, dist, rvec, tvec)
            return True, (rvec, tvec, inliers)
        except cv2.error: return False, None
    
    def _passes_gates(self, prev_q, new_q, match_ratio):
        gate_ori_ok, gate_ratio_ok, reason = True, True, ""
        if self.args.gate_ori_deg >= 0 and prev_q is not None and new_q is not None:
            angle = quaternion_angle_diff_deg(prev_q, new_q)
            if angle > self.args.gate_ori_deg: gate_ori_ok = False; reason += f"ori_jump({angle:.1f}d) "
        if match_ratio < self.args.gate_ratio: gate_ratio_ok = False; reason += f"low_ratio({match_ratio:.2f}) "
        return (gate_ori_ok or gate_ratio_ok) if self.args.gate_logic == 'or' else (gate_ori_ok and gate_ratio_ok), reason.strip()

    def _build_R(self, inliers, reproj_err, latency_ms):
        R = np.eye(7); base_pos_noise, base_quat_noise = 1e-4, 1e-4
        mode = self.args.kf_adaptiveR
        if mode == 'none': return R * base_pos_noise
        inlier_factor = max(0.5, min(3.0, 10.0 / max(1, inliers)))
        error_factor = max(0.5, min(4.0, reproj_err / 2.0)) if reproj_err else 1.0
        latency_factor = 1.0 + self.args.kf_latency_gamma * max(0.0, latency_ms)
        pos_scale = quat_scale = inlier_factor * error_factor * latency_factor if mode == 'full' else inlier_factor * error_factor if mode == 'inliers_reproj' else inlier_factor
        R[0:3, 0:3] *= base_pos_noise * pos_scale; R[3:7, 3:7] *= base_quat_noise * quat_scale
        return R

    def _update_display(self, result):
        vis_frame = result.frame
        if result.bbox: cv2.rectangle(vis_frame, (result.bbox[0], result.bbox[1]), (result.bbox[2], result.bbox[3]), (0, 255, 0), 2)
        display_pos = result.kf_position if result.kf_position is not None else result.position
        display_quat = result.kf_quaternion if result.kf_quaternion is not None else result.quaternion
        if display_pos is not None and display_quat is not None: self._draw_axes(vis_frame, display_pos, display_quat)
        if result.gt_available: self._draw_axes(vis_frame, result.gt_position, result.gt_quaternion, color_override=((255,255,0),(255,0,255),(0,255,255)))
        status_color = (0,255,0) if result.pose_success else (0,0,255)
        cv2.putText(vis_frame, f"STATUS: {'SUCCESS' if result.pose_success else 'FAILED'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        try: self.display_queue.put_nowait(vis_frame)
        except queue.Full: pass

    def _draw_axes(self, frame, position, quaternion, color_override=None):
        try:
            colors = color_override if color_override else ((0,0,255),(0,255,0),(255,0,0))
            R = R_scipy.from_quat(quaternion).as_matrix()
            rvec, _ = cv2.Rodrigues(R)
            tvec = position.reshape(3,1)
            axis_pts = np.float32([[0,0,0],[0.1,0,0],[0,0.1,0],[0,0,0.1]]).reshape(-1,3)
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, self.K, self.dist_coeffs)
            img_pts = img_pts.reshape(-1,2).astype(int)
            origin = tuple(img_pts[0])
            cv2.line(frame, origin, tuple(img_pts[1]), colors[0], 3)
            cv2.line(frame, origin, tuple(img_pts[2]), colors[1], 3)
            cv2.line(frame, origin, tuple(img_pts[3]), colors[2], 3)
        except (cv2.error, AttributeError, ValueError): pass

    def _display_loop(self):
        window_name = "VAPE MK53 - Pose Estimation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.camera_width, self.camera_height)
        while self.running:
            try:
                vis_frame = self.display_queue.get(timeout=1.0)
                cv2.imshow(window_name, vis_frame)
            except queue.Empty:
                if not self.running: break
                continue
            if cv2.waitKey(1) & 0xFF == ord('q'): self.running = False

    def cleanup(self):
        print("Shutting down...")
        self.running = False
        if self.video_capture: self.video_capture.release()
        if self.args.show: cv2.destroyAllWindows()
        if self.jsonl_fp: self.jsonl_fp.close()
        if self.args.output_dir and self.args.total_video_frames > 0:
            meta_path = Path(self.args.output_dir) / "run_metadata.json"
            with open(meta_path, 'w') as f: json.dump({"total_video_frames": self.args.total_video_frames}, f, indent=2)
            print(f"📝 Saved metadata to {meta_path}")

def main():
    parser = argparse.ArgumentParser(description="VAPE MK53 - Core Ablation Logic")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--webcam', action='store_true')
    group.add_argument('--video_file', type=str)
    group.add_argument('--image_dir', type=str)
    parser.add_argument("--output_dir", type=str, default="./run_out")
    parser.add_argument("--log_jsonl", action="store_true")
    parser.add_argument('--show', action='store_true')
    parser.add_argument("--no_det", action="store_true")
    parser.add_argument("--det_every_n", type=int, default=1)
    parser.add_argument("--det_margin_px", type=int, default=20)
    parser.add_argument("--sp_kpts", type=int, default=2048)
    parser.add_argument("--matcher", type=str, default="lightglue", choices=["lightglue", "nnrt", "orb", "sift"])
    parser.add_argument("--nn_ratio", type=float, default=0.75)
    parser.add_argument("--orb_nfeatures", type=int, default=1000)
    parser.add_argument("--sift_nfeatures", type=int, default=1000)
    parser.add_argument("--pnp_solver", type=str, default="epnp", choices=["epnp", "iterative"])
    parser.add_argument("--pnp_refine", type=int, default=1, choices=[0, 1])
    parser.add_argument("--pnp_reproj", type=float, default=8.0)
    parser.add_argument("--pnp_iters", type=int, default=7000)
    parser.add_argument("--pnp_conf", type=float, default=0.95)
    parser.add_argument("--gate_logic", type=str, default="or", choices=["or", "and"])
    parser.add_argument("--gate_ratio", type=float, default=0.55)
    parser.add_argument("--gate_ori_deg", type=float, default=30.0)
    parser.add_argument("--kf_timeaware", type=int, default=1, choices=[0, 1])
    parser.add_argument("--kf_adaptiveR", type=str, default="full", choices=["none", "inliers", "inliers_reproj", "full"])
    parser.add_argument("--kf_latency_gamma", type=float, default=1e-3)
    parser.add_argument("--rate_limits", type=str, default="30,1.5")
    parser.add_argument("--vp_mode", type=str, default="dynamic", choices=["dynamic", "fixed", "coarse4"])
    parser.add_argument("--fixed_view", type=int, default=1)
    parser.add_argument("--vp_switch_hysteresis", type=int, default=2)
    parser.add_argument("--vp_failures", type=int, default=3)
    parser.add_argument("--calibration", type=str, required=True, help="Path to the calibration JSON file.")
    parser.add_argument("--total_video_frames", type=int, default=0, help="Total frames in the source video for accurate metrics.")
    args = parser.parse_args()
    VAPE53_Estimator(args).run()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C).")