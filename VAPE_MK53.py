# ==================================================================================================
#
#  VAPE MK53.1 - REAL-TIME 6-DOF POSE ESTIMATOR (with Latency Compensation)
#
#  Author: [Your Name/Alias Here]
#  Date: August 16, 2025
#  Description: This script performs real-time 6-DOF pose estimation. This version (MK53.1)
#               incorporates review feedback for a more robust, time-aware Unscented Kalman Filter
#               (UKF) that correctly handles vision processing latency using a fixed-lag smoother.
#
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
#  1. IMPORTS AND INITIAL SETUP
# --------------------------------------------------------------------------------------------------
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
import math
from collections import deque

# --- Dependency Imports ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 VAPE MK53.1 Pose Estimator (with Latency Compensation)")
try:
    from ultralytics import YOLO
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    from scipy.spatial import cKDTree
    print("✅ All libraries loaded successfully.")
except ImportError as e:
    print(f"❌ Import error: {e}. Please run 'pip install -r requirements.txt' to install dependencies.")
    exit(1)


# --------------------------------------------------------------------------------------------------
#  2. UTILITY FUNCTIONS
# --------------------------------------------------------------------------------------------------
def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalizes a quaternion to ensure it has a unit length of 1."""
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-10 else np.array([0, 0, 0, 1])

def quat_mul(a, b):
    """Multiply two quaternions (x,y,z,w order)."""
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quat_conj(q):
    """Quaternion conjugate."""
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_inv(q):
    """Quaternion inverse."""
    qn = normalize_quaternion(q)
    return quat_conj(qn)

def quat_to_axis_angle(q):
    """Convert quaternion to axis-angle representation."""
    qn = normalize_quaternion(q)
    w = float(np.clip(qn[3], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w*w, 0.0))
    axis = np.array([1.0, 0.0, 0.0]) if s < 1e-8 else qn[:3]/s
    return axis, angle

def axis_angle_to_quat(axis, angle):
    """Convert axis-angle to quaternion."""
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    s = np.sin(angle/2.0)
    return normalize_quaternion(np.array([ax[0]*s, ax[1]*s, ax[2]*s, np.cos(angle/2.0)]))

def clamp_quaternion_towards(q_from, q_to, max_deg_per_s, dt):
    """Rate-limit quaternion rotation to prevent sudden orientation jumps."""
    if np.dot(q_from, q_to) < 0.0:
        q_to = -q_to
    
    dq = quat_mul(quat_inv(q_from), q_to)
    axis, ang = quat_to_axis_angle(dq)
    
    ang_limit = np.deg2rad(max_deg_per_s) * max(dt, 1e-6)
    
    if ang > ang_limit:
        dq = axis_angle_to_quat(axis, ang_limit)
        q_out = quat_mul(q_from, dq)
    else:
        q_out = q_to
    
    return normalize_quaternion(q_out)

def clamp_position_towards(pos_from, pos_to, max_speed_m_per_s, dt):
    """Rate-limit position movement to prevent sudden position jumps."""
    delta_pos = pos_to - pos_from
    distance = np.linalg.norm(delta_pos)
    
    max_distance = max_speed_m_per_s * max(dt, 1e-6)
    
    if distance > max_distance:
        direction = delta_pos / distance
        pos_out = pos_from + direction * max_distance
    else:
        pos_out = pos_to
    
    return pos_out


# --------------------------------------------------------------------------------------------------
#  3. DATA STRUCTURES
# --------------------------------------------------------------------------------------------------
@dataclass
class ProcessingResult:
    frame_id: int
    frame: np.ndarray
    position: Optional[np.ndarray] = None
    quaternion: Optional[np.ndarray] = None
    kf_position: Optional[np.ndarray] = None
    kf_quaternion: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    num_inliers: int = 0
    pose_success: bool = False
    viewpoint_used: Optional[str] = None

@dataclass
class PoseData:
    position: np.ndarray
    quaternion: np.ndarray
    inliers: int
    reprojection_error: float
    viewpoint: str
    total_matches: int


# --------------------------------------------------------------------------------------------------
#  4. UNSCENTED KALMAN FILTER (UKF)
# --------------------------------------------------------------------------------------------------
class UnscentedKalmanFilter:
    """
    Time-aware UKF for asynchronous measurements with latency.
    """
    def __init__(self):
        self.t_state = None
        self.history = deque(maxlen=200)

        self.max_rot_rate_dps = 45.0
        self.max_pos_speed_mps = 1.5

        self.n = 16
        self.m = 7

        self.x = np.zeros(self.n)
        self.x[9:13] = np.array([0., 0., 0., 1.]) # Correct identity for (x,y,z,w)

        self.P0 = np.eye(self.n) * 0.1
        self.P = self.P0.copy()

        self.alpha, self.beta, self.kappa = 1e-3, 2.0, 0.0
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n

        self.wm = np.full(2 * self.n + 1, 1.0 / (2.0 * (self.n + self.lambda_)))
        self.wc = self.wm.copy()
        self.wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.wc[0] = self.lambda_ / (self.n + self.lambda_) + (1.0 - self.alpha**2 + self.beta)

        self.Q = np.eye(self.n) * 1e-2
        self.R_base = np.eye(self.m) * 1e-4

    def _push_history(self):
        if self.t_state is not None:
            self.history.append((self.t_state, self.x.copy(), self.P.copy()))

    def _generate_sigma_points(self, x, P):
        sigmas = np.zeros((2 * self.n + 1, self.n))
        try:
            U = np.linalg.cholesky((self.n + self.lambda_) * P)
        except np.linalg.LinAlgError:
            try:
                P_jittered = P + 1e-9 * np.eye(self.n)
                U = np.linalg.cholesky((self.n + self.lambda_) * P_jittered)
            except np.linalg.LinAlgError:
                print("⚠️ UKF covariance is not positive-definite. Using identity matrix.")
                U = np.sqrt((self.n + self.lambda_)) * np.eye(self.n)
        sigmas[0] = x
        for i in range(self.n):
            sigmas[i+1]   = x + U[:, i]
            sigmas[self.n+i+1] = x - U[:, i]
        return sigmas

    def motion_model(self, x_in, dt):
        x_out = np.zeros_like(x_in)
        pos, vel, acc = x_in[0:3], x_in[3:6], x_in[6:9]
        x_out[0:3] = pos + vel * dt + 0.5 * acc * dt**2
        x_out[3:6] = vel + acc * dt
        x_out[6:9] = acc
        q, w = x_in[9:13], x_in[13:16]
        omega_mat = 0.5 * np.array([
            [0, -w[0], -w[1], -w[2]], [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]], [w[2], w[1], -w[0], 0]
        ])
        q_new = (np.eye(4) + dt * omega_mat) @ q
        x_out[9:13] = normalize_quaternion(q_new)
        x_out[13:16] = w
        return x_out

    def predict(self, dt):
        sigmas = self._generate_sigma_points(self.x, self.P)
        sigmas_f = np.array([self.motion_model(s, dt) for s in sigmas])
        x_pred = np.sum(self.wm[:, np.newaxis] * sigmas_f, axis=0)
        P_pred = self.Q.copy()
        for i in range(2 * self.n + 1):
            y = sigmas_f[i] - x_pred
            P_pred += self.wc[i] * np.outer(y, y)
        self.x = x_pred
        self.P = P_pred

    def hx(self, x_in):
        z = np.zeros(self.m)
        z[0:3] = x_in[0:3]
        z[3:7] = x_in[9:13]
        return z

    def _measurement_update(self, position, quaternion, R, dt_for_clamping):
        pos_pred, q_pred = self.x[0:3], self.x[9:13]
        if np.dot(q_pred, quaternion) < 0.0:
            quaternion = -quaternion
        pos_meas_clamped = clamp_position_towards(pos_pred, position, self.max_pos_speed_mps, dt_for_clamping)
        q_meas_clamped = clamp_quaternion_towards(q_pred, normalize_quaternion(quaternion), self.max_rot_rate_dps, dt_for_clamping)
        measurement = np.concatenate([pos_meas_clamped, q_meas_clamped])
        sigmas_x = self._generate_sigma_points(self.x, self.P)
        sigmas_h = np.array([self.hx(s) for s in sigmas_x])
        z_pred = np.sum(self.wm[:, np.newaxis] * sigmas_h, axis=0)
        S = R.copy()
        for i in range(2 * self.n + 1):
            y = sigmas_h[i] - z_pred
            S += self.wc[i] * np.outer(y, y)
        P_xz = np.zeros((self.n, self.m))
        for i in range(2 * self.n + 1):
            y_x = sigmas_x[i] - self.x
            y_z = sigmas_h[i] - z_pred
            P_xz += self.wc[i] * np.outer(y_x, y_z)
        K = np.linalg.solve(S.T, P_xz.T).T
        self.x += K @ (measurement - z_pred)
        self.x[9:13] = normalize_quaternion(self.x[9:13])
        self.P -= K @ S @ K.T

    def update_with_timestamp(self, z_pos, z_quat, t_meas, R, t_now=None):
        if self.t_state is None:
            self.x[0:3] = z_pos
            self.x[9:13] = normalize_quaternion(z_quat)
            self.P = self.P0.copy()
            self.t_state = t_meas
            return

        if t_meas >= self.t_state:
            dt1 = t_meas - self.t_state
            if dt1 > 1e-6:
                self._push_history()
                self.predict(dt1)
            self.t_state = t_meas
            self._measurement_update(z_pos, z_quat, R, dt_for_clamping=dt1)
            if t_now is not None and t_now > t_meas:
                dt2 = t_now - t_meas
                if dt2 > 1e-6:
                    self._push_history()
                    self.predict(dt2)
                    self.t_state = t_now
            return

        if not self.history:
            return
        try:
            k = max(i for i, (t, _, _) in enumerate(self.history) if t <= t_meas)
        except ValueError:
            return

        t_k, x_k, P_k = self.history[k]
        self.x[:], self.P[:], self.t_state = x_k, P_k, t_k
        dt_to_meas = t_meas - t_k
        if dt_to_meas > 1e-6:
            self.predict(dt_to_meas)
        self.t_state = t_meas
        self._measurement_update(z_pos, z_quat, R, dt_for_clamping=dt_to_meas)

        for j in range(k + 1, len(self.history)):
            t_next, _, _ = self.history[j]
            dt_replay = t_next - self.t_state
            if dt_replay > 1e-6:
                self._push_history()
                self.predict(dt_replay)
                self.t_state = t_next
        
        if t_now is not None and t_now > self.t_state:
            dt_ff = t_now - self.t_state
            if dt_ff > 1e-6:
                self._push_history()
                self.predict(dt_ff)
                self.t_state = t_now

    def get_visualisation_state(self, t_now: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.t_state is None:
            return None, None
        dt = t_now - self.t_state
        if dt <= 0:
            return self.x[0:3], self.x[9:13]
        x_display = self.motion_model(self.x.copy(), dt)
        return x_display[0:3], x_display[9:13]

# --------------------------------------------------------------------------------------------------
#  5. HIGH-FREQUENCY MAIN THREAD
# --------------------------------------------------------------------------------------------------
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
        self.is_video_stream = False
        self.video_capture = None
        self.image_files = []
        self.frame_idx = 0
        self.frame_count = 0
        self.start_time = time.monotonic()
        self._initialize_input_source()
        self.K, self.dist_coeffs = self._get_camera_intrinsics()

    def _initialize_input_source(self):
        if self.args.webcam:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened(): raise IOError("Cannot open webcam.")
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.is_video_stream = True
        elif self.args.video_file:
            self.video_capture = cv2.VideoCapture(self.args.video_file)
            self.is_video_stream = True
        elif self.args.image_dir:
            self.image_files = sorted([os.path.join(self.args.image_dir, f) for f in os.listdir(self.args.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not self.image_files: raise IOError(f"No images found in directory: {self.args.image_dir}")
        else:
            raise ValueError("No input source specified.")

    def _get_next_frame_with_timestamp(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        capture_time = time.monotonic()
        if self.is_video_stream:
            ret, frame = self.video_capture.read()
            return (frame, capture_time) if ret else (None, None)
        else:
            if self.frame_idx < len(self.image_files):
                frame = cv2.imread(self.image_files[self.frame_idx])
                self.frame_idx += 1
                return frame, capture_time
            return None, None

    def run(self):
        window_name = "VAPE MK53.1 - Real-time Pose Estimation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.camera_width, self.camera_height)
        while self.running:
            loop_start_time = time.monotonic()
            frame, t_capture = self._get_next_frame_with_timestamp()
            if frame is None: break
            with self.pose_data_lock:
                predicted_pose_tvec, predicted_pose_quat = self.kf.get_visualisation_state(time.monotonic())
            vis_frame = frame.copy()
            if predicted_pose_tvec is not None and predicted_pose_quat is not None:
                self._draw_axes(vis_frame, predicted_pose_tvec, predicted_pose_quat)
            elapsed_time = time.monotonic() - self.start_time
            fps = (self.frame_count + 1) / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(vis_frame, "STATUS: PREDICTING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
            self.frame_count += 1
            if self.processing_queue.qsize() < 2:
                self.processing_queue.put((frame.copy(), t_capture))
            if self.args.show:
                try:
                    vis_data = self.visualization_queue.get_nowait()
                    kpts, vis_crop = vis_data['kpts'], vis_data['crop']
                    for kpt in kpts:
                        cv2.circle(vis_crop, (int(kpt[0]), int(kpt[1])), 2, (0, 255, 0), -1)
                    cv2.imshow("SuperPoint Features", vis_crop)
                except queue.Empty:
                    pass
            cv2.imshow(window_name, vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
            time_to_wait = (1.0 / 30.0) - (time.monotonic() - loop_start_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)
        self.cleanup()

    def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K, None

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        q_norm = normalize_quaternion(q)
        x, y, z, w = q_norm
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])

    def _draw_axes(self, frame: np.ndarray, position: np.ndarray, quaternion: np.ndarray):
        try:
            R = self._quaternion_to_rotation_matrix(quaternion)
            rvec, _ = cv2.Rodrigues(R)
            tvec = position.reshape(3, 1)
            axis_pts = np.float32([[0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]]).reshape(-1,3)
            img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, self.K, self.dist_coeffs)
            img_pts = img_pts.reshape(-1, 2).astype(int)
            origin = tuple(img_pts[0])
            cv2.line(frame, origin, tuple(img_pts[1]), (0,0,255), 3)
            cv2.line(frame, origin, tuple(img_pts[2]), (0,255,0), 3)
            cv2.line(frame, origin, tuple(img_pts[3]), (255,0,0), 3)
        except (cv2.error, AttributeError, ValueError):
            pass

    def cleanup(self):
        self.running = False
        if self.is_video_stream and self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------------------
#  6. LOW-FREQUENCY PROCESSING THREAD
# --------------------------------------------------------------------------------------------------
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
        self.camera_width, self.camera_height = 1280, 720
        self.all_poses_log = []
        self.current_best_viewpoint = None
        self.viewpoint_quality_threshold = 5
        self.consecutive_failures = 0
        self.max_failures_before_search = 3
        self.last_orientation: Optional[np.ndarray] = None
        self.ORI_MAX_DIFF_DEG = 40.0
        self.rejected_consecutive_frames_count = 0
        self.MAX_REJECTED_FRAMES = 7
        self.yolo_model, self.extractor, self.matcher = None, None, None
        self.viewpoint_anchors = {}
        self._initialize_models()
        self._initialize_anchor_data()
        self.K, self.dist_coeffs = self._get_camera_intrinsics()

    def _initialize_models(self):
        self.yolo_model = YOLO("best.pt").to(self.device)
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def _initialize_anchor_data(self):
        # (Anchor data definitions remain unchanged)
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
            'NE': {'path': 'NE.png', '2d': ne_anchor_2d, '3d': ne_anchor_3d}, 'NW': {'path': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png', '2d': nw_anchor_2d, '3d': nw_anchor_3d},
            'SE': {'path': 'SE.png', '2d': se_anchor_2d, '3d': se_anchor_3d}, 'W': {'path': 'W.png', '2d': w_anchor_2d, '3d': w_anchor_3d},
            'S': {'path': 'S.png', '2d': s_anchor_2d, '3d': s_anchor_3d}, 'SW': {'path': 'Anchor_B.png', '2d': sw_anchor_2d, '3d': sw_anchor_3d},
            'N': {'path': 'N.png', '2d': n_anchor_2d, '3d': n_anchor_3d}, 'SW2': {'path': 'SW2.png', '2d': sw2_anchor_2d, '3d': sw2_anchor_3d},
            'SE2': {'path': 'SE2.png', '2d': se2_anchor_2d, '3d': se2_anchor_3d}, 'SU': {'path': 'SU.png', '2d': su_anchor_2d, '3d': su_anchor_3d},
            'NU': {'path': 'NU.png', '2d': nu_anchor_2d, '3d': nu_anchor_3d}, 'NW2': {'path': 'NW2.png', '2d': nw2_anchor_2d, '3d': nw2_anchor_3d},
            'NE2': {'path': 'NE2.png', '2d': ne2_anchor_2d, '3d': ne2_anchor_3d}, 'E': {'path': 'E.png', '2d': e_anchor_2d, '3d': e_anchor_3d},
        }
        self.viewpoint_anchors = {}
        for viewpoint, data in anchor_definitions.items():
            if not os.path.exists(data['path']):
                continue
            anchor_image_bgr = cv2.resize(cv2.imread(data['path']), (self.camera_width, self.camera_height))
            anchor_features = self._extract_features_sp(anchor_image_bgr)
            sp_tree = cKDTree(anchor_features['keypoints'][0].cpu().numpy())
            distances, indices = sp_tree.query(data['2d'], k=1, distance_upper_bound=5.0)
            valid_mask = distances != np.inf
            self.viewpoint_anchors[viewpoint] = {
                'features': anchor_features,
                'map_3d': {idx: pt for idx, pt in zip(indices[valid_mask], data['3d'][valid_mask])}
            }

    def run(self):
        frame_id = 0
        while self.running:
            try:
                frame, t_capture = self.processing_queue.get(timeout=0.1)
                result = self._process_frame(frame, frame_id, t_capture)
                frame_id += 1
                self.all_poses_log.append({
                    'frame': result.frame_id, 'success': result.pose_success,
                    'position': result.position.tolist() if result.position is not None else None,
                    'quaternion': result.quaternion.tolist() if result.quaternion is not None else None,
                    'kf_position': result.kf_position.tolist() if result.kf_position is not None else None,
                    'kf_quaternion': result.kf_quaternion.tolist() if result.kf_quaternion is not None else None,
                    'num_inliers': result.num_inliers, 'viewpoint_used': result.viewpoint_used
                })
            except queue.Empty:
                time.sleep(0.001)

    def _process_frame(self, frame: np.ndarray, frame_id: int, t_capture: float) -> ProcessingResult:
        result = ProcessingResult(frame_id=frame_id, frame=frame.copy(), pose_success=False)
        bbox = self._yolo_detect(frame)
        result.bbox = bbox
        best_pose = self._estimate_pose_with_temporal_consistency(frame, bbox)
        is_valid = False
        if best_pose:
            if self.last_orientation is not None:
                angle_diff = math.degrees(self.quaternion_angle_diff(self.last_orientation, best_pose.quaternion))
                if angle_diff <= self.ORI_MAX_DIFF_DEG:
                    is_valid = True
            else:
                is_valid = True
        if is_valid and best_pose:
            self.rejected_consecutive_frames_count = 0
            result.position, result.quaternion = best_pose.position, best_pose.quaternion
            result.num_inliers, result.pose_success = best_pose.inliers, True
            result.viewpoint_used = best_pose.viewpoint
            self.last_orientation = best_pose.quaternion
            t_now = time.monotonic()
            with self.pose_data_lock:
                self.kf.update_with_timestamp(
                    z_pos=best_pose.position, z_quat=best_pose.quaternion,
                    t_meas=t_capture, R=self.kf.R_base, t_now=t_now
                )
                result.kf_position, result.kf_quaternion = self.kf.x[0:3], self.kf.x[9:13]
        else:
            self.rejected_consecutive_frames_count += 1
            if self.rejected_consecutive_frames_count >= self.MAX_REJECTED_FRAMES:
                with self.pose_data_lock:
                    self.kf.t_state = None
                    self.kf.history.clear()
                self.last_orientation = None
                self.current_best_viewpoint = None
                self.rejected_consecutive_frames_count = 0
        return result

    def _estimate_pose_with_temporal_consistency(self, frame: np.ndarray, bbox: Optional[Tuple]) -> Optional[PoseData]:
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
                    if pose_data.total_matches >= self.viewpoint_quality_threshold * 2:
                        break
            if successful_poses:
                best_pose = max(successful_poses, key=lambda p: (p.total_matches, p.inliers, -p.reprojection_error))
                self.current_best_viewpoint = best_pose.viewpoint
                self.consecutive_failures = 0
                return best_pose
        return None

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
        if self.args.show and self.visualization_queue.qsize() < 2:
            self.visualization_queue.put({'kpts': frame_features['keypoints'][0].cpu().numpy(), 'crop': crop.copy()})
        with torch.no_grad():
            matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
        matches = rbd(matches_dict)['matches'].cpu().numpy()
        if len(matches) < 6: return None
        points_3d, points_2d = [], []
        crop_offset = np.array([bbox[0], bbox[1]]) if bbox else np.array([0, 0])
        for anchor_idx, frame_idx in matches:
            if anchor_idx in anchor['map_3d']:
                points_3d.append(anchor['map_3d'][anchor_idx])
                points_2d.append(frame_features['keypoints'][0].cpu().numpy()[frame_idx] + crop_offset)
        if len(points_3d) < 6: return None
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(points_3d, dtype=np.float32), np.array(points_2d, dtype=np.float32),
                self.K, self.dist_coeffs, reprojectionError=8, confidence=0.95,
                iterationsCount=7000, flags=cv2.SOLVEPNP_EPNP
            )
            if success and inliers is not None and len(inliers) > 4:
                rvec, tvec = cv2.solvePnPRefineVVS(
                    np.array(points_3d, dtype=np.float32)[inliers.flatten()],
                    np.array(points_2d, dtype=np.float32)[inliers.flatten()],
                    self.K, self.dist_coeffs, rvec, tvec
                )
        except cv2.error: return None
        if not success or inliers is None or len(inliers) < 4: return None
        R, _ = cv2.Rodrigues(rvec)
        position = tvec.flatten()
        quaternion = self._rotation_matrix_to_quaternion(R)
        projected_points, _ = cv2.projectPoints(np.array(points_3d)[inliers.flatten()], rvec, tvec, self.K, self.dist_coeffs)
        error = np.mean(np.linalg.norm(np.array(points_2d)[inliers.flatten()].reshape(-1, 1, 2) - projected_points, axis=2))
        return PoseData(position, quaternion, len(inliers), error, viewpoint, len(points_3d))

    def quaternion_angle_diff(self, q1: np.ndarray, q2: np.ndarray) -> float:
        dot = np.dot(normalize_quaternion(q1), normalize_quaternion(q2))
        return 2 * math.acos(abs(min(1.0, max(-1.0, dot))))

    def _yolo_detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        names = getattr(self.yolo_model, "names", {0: "iha"})
        inv = {v: k for k, v in names.items()}
        target_id = inv.get("iha", 0)
        for conf_thresh in (0.30, 0.20, 0.10):
            results = self.yolo_model(frame, imgsz=640, conf=conf_thresh, iou=0.5, max_det=5, classes=[target_id], verbose=False)
            if not results or len(results[0].boxes) == 0: continue
            best = max(results[0].boxes, key=lambda b: float((b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1])))
            x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().tolist()
            return int(x1), int(y1), int(x2), int(y2)
        return None

    def _extract_features_sp(self, image_bgr: np.ndarray) -> Dict:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            return self.extractor.extract(tensor)

    def _get_camera_intrinsics(self) -> Tuple[np.ndarray, None]:
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K, None

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2; qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S; qy = (R[0, 2] - R[2, 0]) / S; qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S; qy = (R[0, 1] + R[1, 0]) / S; qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S; qy = 0.25 * S; qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S; qy = (R[1, 2] + R[2, 1]) / S; qz = 0.25 * S
        return np.array([qx, qy, qz, qw])

    def cleanup(self):
        self.running = False
        if self.args.save_output:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"pose_log_{time.strftime('%Y%m%d-%H%M%S')}.json")
            with open(filename, 'w') as f:
                json.dump(self.all_poses_log, f, indent=4)
            print(f"💾 Pose log saved to {filename}")

# --------------------------------------------------------------------------------------------------
#  7. MAIN EXECUTION BLOCK
# --------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VAPE MK53.1 - Real-time Pose Estimator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--webcam', action='store_true', help='Use webcam as input.')
    group.add_argument('--video_file', type=str, help='Path to a video file.')
    group.add_argument('--image_dir', type=str, help='Path to a directory of images.')
    parser.add_argument('--save_output', action='store_true', help='Save the final pose data to a JSON file.')
    parser.add_argument('--show', action='store_true', help='Show keypoint detections in a separate window.')
    args = parser.parse_args()

    main_thread, processing_thread = None, None
    try:
        processing_queue = queue.Queue(maxsize=3)
        visualization_queue = queue.Queue(maxsize=3)
        pose_data_lock = threading.Lock()
        kf = UnscentedKalmanFilter()
        main_thread = MainThread(processing_queue, visualization_queue, pose_data_lock, kf, args)
        processing_thread = ProcessingThread(processing_queue, visualization_queue, pose_data_lock, kf, args)
        main_thread.start()
        processing_thread.start()
        main_thread.join()
    except (IOError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C).")
    finally:
        if main_thread is not None and main_thread.is_alive():
            main_thread.running = False
            main_thread.join(timeout=1)
        if processing_thread is not None and processing_thread.is_alive():
            processing_thread.running = False
            processing_thread.join(timeout=1)
        print("✅ Process finished.")

if __name__ == '__main__':
    main()