#!/usr/bin/env python3
"""
RTMPose-based Aircraft Pose Estimator - Research Comparison Version
- Now with robust, optional Ground Truth comparison using a ChArUco board.
"""

import cv2
import numpy as np
import torch
import time
import argparse
import warnings
import sys
import threading
import json
import os
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import queue

from scipy.spatial.transform import Rotation as R_scipy

warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

print("🚀 Starting RTMPose-based Aircraft Pose Estimator...")

try:
    from ultralytics import YOLO
    from mmpose.apis import init_model, inference_topdown
    import mmcv
    print("✅ All libraries loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def make_charuco_board(cols, rows, square_len_m, marker_len_m, dict_id=cv2.aruco.DICT_6X6_250):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(cols, rows, square_len_m, marker_len_m, aruco_dict)
    else:
        board = cv2.aruco.CharucoBoard((cols, rows), square_len_m, marker_len_m, aruco_dict)
    return aruco_dict, board

def make_detectors(aruco_dict):
    try:
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.adaptiveThreshWinSizeMin = 5
        aruco_params.adaptiveThreshWinSizeMax = 25
        aruco_params.adaptiveThreshWinSizeStep = 4
        aruco_params.minMarkerPerimeterRate = 0.03
        aruco_params.maxMarkerPerimeterRate = 0.4
        aruco_params.minMarkerDistanceRate = 0.05
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        aruco_params.cornerRefinementWinSize = 5
        aruco_params.cornerRefinementMaxIterations = 30
        aruco_params.cornerRefinementMinAccuracy = 0.1
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    except AttributeError:
        print("⚠️ Using legacy ArUco detector API.")
        detector = cv2.aruco.DetectorParameters_create()
    return detector

@dataclass
class ProcessingResult:
    frame: np.ndarray
    frame_id: int
    timestamp: float
    position: Optional[np.ndarray] = None
    quaternion: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    num_keypoints: int = 0
    pose_data: Optional[Dict] = None
    gt_available: bool = False
    gt_position: Optional[np.ndarray] = None
    gt_quaternion: Optional[np.ndarray] = None
    pose_error_pos: Optional[float] = None
    pose_error_rot: Optional[float] = None

class ThreadSafeFrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame, self.frame_id, self.timestamp, self.video_frame_number = None, 0, None, 0
    def update(self, frame, video_frame_number=None):
        with self.lock:
            self.latest_frame, self.frame_id, self.timestamp = frame.copy(), self.frame_id + 1, time.monotonic()
            if video_frame_number is not None:
                self.video_frame_number = video_frame_number
    def get_latest(self):
        with self.lock:
            return (self.latest_frame.copy(), self.frame_id, self.timestamp, self.video_frame_number) if self.latest_frame is not None else (None, -1, -1, -1)

def create_unique_filename(directory, base_filename):
    base_path = os.path.join(directory, base_filename)
    if not os.path.exists(base_path):
        return base_path
    name, ext = os.path.splitext(base_filename)
    counter = 1
    while True:
        new_path = os.path.join(directory, f"{name}_{counter}{ext}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def convert_to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(i) for i in obj]
    return obj

class RTMPosePoseEstimator:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.frame_buffer = ThreadSafeFrameBuffer()
        self.result_queue = queue.Queue(maxsize=2)
        self.all_poses_lock = threading.Lock()
        self.all_poses = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")
        self._init_camera_and_input()
        self._init_models()
        self._init_keypoint_3d_coordinates()
        self._init_ground_truth()
        self._get_camera_intrinsics()
        print("✅ RTMPose-based Aircraft Pose Estimator initialized!")

    def _init_camera_and_input(self):
        self.video_mode = self.args.video_file is not None
        self.camera_mode = self.args.webcam
        self.batch_mode = self.args.image_dir is not None
        self.camera_width, self.camera_height = 1280, 720
        if self.video_mode:
            self.cap = cv2.VideoCapture(self.args.video_file)
            self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elif self.camera_mode:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

    def _init_models(self):
        self.yolo_model = YOLO("best.pt").to(self.device)
        self.rtmpose_model = init_model(self.args.rtmpose_config, self.args.rtmpose_checkpoint, device=self.device)
        self.keypoint_names = ['nose', 'left_wing', 'left_tail', 'tail', 'right_tail', 'right_wing']

    def _init_keypoint_3d_coordinates(self):
        self.keypoint_3d_coords = np.array([[-0.2,0,0], [0,-0.025,-0.24], [0.23,0,-0.113], [0.243,-0.104,0], [0.23,0,0.113], [0,-0.025,0.24]], dtype=np.float32)

    def _init_ground_truth(self):
        self.gt_enabled = False
        self.interpolate_func = None
        if self.args.calibration:
            if hasattr(cv2.aruco, 'interpolateCornersCharuco'):
                self.interpolate_func = cv2.aruco.interpolateCornersCharuco
            elif hasattr(cv2.aruco, 'legacy') and hasattr(cv2.aruco.legacy, 'interpolateCornersCharuco'):
                self.interpolate_func = cv2.aruco.legacy.interpolateCornersCharuco
            else:
                print("❌ CRITICAL: Cannot find 'interpolateCornersCharuco' in your cv2.aruco module.")
                print("   Ground Truth mode is disabled. Please check your OpenCV and opencv-contrib-python installation.")
                return
            try:
                print("🛠️ Initializing Ground Truth system...")
                self.R_to, self.t_to = self._load_calibration(self.args.calibration)
                self.aruco_dict, self.board = make_charuco_board(7, 11, 0.0765, 0.0573)
                self.aruco_detector = make_detectors(self.aruco_dict)
                self.gt_enabled = True
                print("  ✅ Ground Truth enabled.")
            except Exception as e:
                print(f"❌ Failed to initialize Ground Truth system: {e}")
                self.gt_enabled = False

    def _load_calibration(self, path):
        with open(path, 'r') as f: data = json.load(f)
        return np.array(data['R_to']), np.array(data['t_to'])

    def _detect_charuco_pose(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        if ids is None or len(ids) == 0: return False, None, None
        _, charuco_corners, charuco_ids = self.interpolate_func(corners, ids, gray, self.board)
        if charuco_corners is None or len(charuco_corners) < 4: return False, None, None
        pose_ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.board, self.K, self.dist_coeffs, None, None)
        return (True, cv2.Rodrigues(rvec)[0], tvec) if pose_ok else (False, None, None)

    def _calculate_pose_error(self, pose_tvec, pose_quat, gt_tvec, gt_rmat):
        pos_err = np.linalg.norm(pose_tvec - gt_tvec.flatten()) * 100
        pose_rmat = R_scipy.from_quat(pose_quat).as_matrix()
        rot_err = np.linalg.norm(cv2.Rodrigues(gt_rmat.T @ pose_rmat)[0]) * 180 / np.pi
        return pos_err, rot_err

    def _get_camera_intrinsics(self):
        self.K = np.array([[1460.1015, 0, 604.85462], [0, 1456.48915, 328.648], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = None
        return self.K, self.dist_coeffs

    def _process_frame(self, frame, frame_id, timestamp):
        result = ProcessingResult(frame=frame, frame_id=frame_id, timestamp=timestamp)
        if self.gt_enabled:
            gt_success, R_ct, t_ct = self._detect_charuco_pose(frame)
            if gt_success:
                result.gt_available = True
                result.gt_position = (R_ct @ self.t_to + t_ct).flatten()
                result.gt_quaternion = R_scipy.from_matrix(R_ct @ self.R_to).as_quat()
        
        bbox = self._yolo_detect(frame)
        if bbox:
            result.bbox = bbox
            pos, quat, num_kpts, pose_data = self._detect_rtmpose_keypoints(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]], bbox, frame_id, timestamp)
            result.position, result.quaternion, result.num_keypoints, result.pose_data = pos, quat, num_kpts, pose_data

        if result.gt_available and result.position is not None:
            pos_err, rot_err = self._calculate_pose_error(result.position, result.quaternion, result.gt_position, R_scipy.from_quat(result.gt_quaternion).as_matrix())
            result.pose_error_pos, result.pose_error_rot = pos_err, rot_err
            if result.pose_data:
                result.pose_data.update({'gt_pos_xyz': result.gt_position, 'gt_q_xyzw': result.gt_quaternion, 'raw_pos_err_cm': pos_err, 'raw_rot_err_deg': rot_err})
        return result

    def _yolo_detect(self, frame):
        results = self.yolo_model(frame, verbose=False, conf=0.5)
        if not results or len(results[0].boxes) == 0: return None
        return tuple(map(int, results[0].boxes.xyxy[0].cpu().numpy()))

    def _detect_rtmpose_keypoints(self, cropped_frame, bbox, frame_id, t_capture):
        if cropped_frame.size == 0: return (None, None, 0, {})
        letterboxed, r, pad_left, pad_top = self.letterbox_black(cropped_frame)
        results = inference_topdown(self.rtmpose_model, letterboxed, np.array([[0, 0, 384, 288]]))
        
        base_data = {"frame_idx": frame_id, "t_capture": t_capture, "est_pos_xyz": None, "est_quat_xyzw": None, "num_inliers": 0}
        if not results or not results[0].pred_instances.keypoints.any(): return None, None, 0, base_data

        kpts, scores = results[0].pred_instances.keypoints[0], results[0].pred_instances.keypoint_scores[0]
        full_kpts = self.unletterbox_keypoints(kpts, r, pad_left, pad_top) + np.array([bbox[0], bbox[1]])
        valid = scores > 0.1
        if sum(valid) < 4: return None, None, sum(valid), base_data

        K, dist = self._get_camera_intrinsics()
        success, rvec, tvec, inliers = cv2.solvePnPRansac(self.keypoint_3d_coords[valid], full_kpts[valid], K, dist)
        if not success or inliers is None or len(inliers) < 4: return None, None, sum(valid), base_data

        pos, quat = tvec.flatten(), R_scipy.from_matrix(cv2.Rodrigues(rvec)[0]).as_quat()
        base_data.update({"est_pos_xyz": pos, "est_quat_xyzw": quat, "num_inliers": len(inliers)})
        return pos, quat, sum(valid), base_data

    def letterbox_black(self, image, new_size=(384, 288)):
        h, w = image.shape[:2]
        r = min(new_size[1] / h, new_size[0] / w)
        unpad_w, unpad_h = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(image, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = new_size[0] - unpad_w, new_size[1] - unpad_h
        top, left = pad_h // 2, pad_w // 2
        return cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT), r, left, top

    def unletterbox_keypoints(self, kpts, r, pad_l, pad_t):
        if kpts is None: return None
        kpts_orig = kpts.copy().astype(np.float32)
        kpts_orig[:, 0] = (kpts_orig[:, 0] - pad_l) / r
        kpts_orig[:, 1] = (kpts_orig[:, 1] - pad_t) / r
        return kpts_orig

    def save_json_results(self):
        output_path_str = self.args.save_pose
        if self.gt_enabled:
            save_dir = "GT_result"
            os.makedirs(save_dir, exist_ok=True)
            output_path_str = os.path.join(save_dir, os.path.basename(output_path_str))
        
        final_path = create_unique_filename(os.path.dirname(output_path_str) or ".", os.path.basename(output_path_str))
        with self.all_poses_lock:
            with open(final_path, 'w') as f:
                json.dump([convert_to_json_serializable(p) for p in self.all_poses if p], f, indent=4)
            print(f"💾 Saved {len(self.all_poses)} pose results to {final_path}")

    def run(self):
        if self.batch_mode or (not self.args.show):
            # Headless or batch mode processing
            self.processing_loop_headless()
        else:
            # Display mode for video/webcam
            self.run_with_display()
        
        self.save_json_results()
        self._cleanup()

    def camera_loop(self):
        frame_idx = 0
        while self.running:
            if not self.cap or not self.cap.isOpened(): self.running = False; break
            ret, frame = self.cap.read()
            if not ret: self.running = False; break
            self.frame_buffer.update(frame, frame_idx)
            frame_idx += 1
            time.sleep(1/60) # Small sleep to prevent busy-waiting

    def processing_loop_headless(self):
        # This loop is for batch or non-display runs, processes as fast as possible
        frame_idx = 0
        while True:
            if self.video_mode or self.camera_mode:
                if not self.cap or not self.cap.isOpened(): break
                ret, frame = self.cap.read()
                if not ret: break
            elif self.batch_mode:
                if frame_idx >= len(os.listdir(self.args.image_dir)): break
                frame = cv2.imread(os.path.join(self.args.image_dir, sorted(os.listdir(self.args.image_dir))[frame_idx]))
            else: break
            if frame is None: continue
            result = self._process_frame(frame, frame_idx, time.monotonic())
            with self.all_poses_lock: self.all_poses.append(result.pose_data)
            frame_idx += 1
        print("Headless processing finished.")

    def run_with_display(self):
        frame_idx = 0
        while True:
            if not self.cap or not self.cap.isOpened(): break
            ret, frame = self.cap.read()
            if not ret: break
            
            result = self._process_frame(frame, frame_idx, time.monotonic())
            with self.all_poses_lock: self.all_poses.append(result.pose_data)

            vis_frame = result.frame.copy()
            if result.position is not None: self._draw_axes(vis_frame, result.position, result.quaternion, result.bbox)
            if result.gt_available: self._draw_axes(vis_frame, result.gt_position, result.gt_quaternion, result.bbox, color_override=((255,255,0),(255,0,255),(0,255,255)))
            if result.pose_error_pos is not None: cv2.putText(vis_frame, f"Err: {result.pose_error_pos:.1f}cm, {result.pose_error_rot:.1f}deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
            
            cv2.imshow('RTMPose GT Comparison', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            frame_idx += 1

    def _draw_axes(self, frame, position, quaternion, bbox=None, color_override=None):
        try:
            colors = color_override or ((0,0,255),(0,255,0),(255,0,0))
            rvec, _ = cv2.Rodrigues(R_scipy.from_quat(quaternion).as_matrix())
            axis_pts, _ = cv2.projectPoints(np.float32([[0,0,0],[0.1,0,0],[0,0.1,0],[0,0,0.1]]), rvec, position.reshape(3,1), self.K, self.dist_coeffs)
            pts = axis_pts.reshape(-1,2).astype(int)
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), colors[0], 3)
            cv2.line(frame, tuple(pts[0]), tuple(pts[2]), colors[1], 3)
            cv2.line(frame, tuple(pts[0]), tuple(pts[3]), colors[2], 3)
        except Exception:
            if bbox is not None:
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                color = (0, 0, 255) if color_override is None else (0, 255, 255)
                cv2.drawMarker(frame, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    def _cleanup(self):
        if hasattr(self, 'cap') and self.cap: self.cap.release()
        cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description='RTMPose-based Aircraft Pose Estimator with optional Ground Truth')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video_file', type=str, help='MP4/AVI video file path')
    group.add_argument('--image_dir', type=str, help='Folder containing images')
    group.add_argument('--webcam', action='store_true', help='Use webcam input')
    parser.add_argument('--rtmpose_config', type=str, default='configs/my_aircraft/rtmpose.py')
    parser.add_argument('--rtmpose_checkpoint', type=str, default='work_dirs/rtmpose-l_aircraft-384x288_20250901/epoch_580.pth')
    parser.add_argument('--csv_file', type=str, help='CSV file for batch mode')
    parser.add_argument('--save_pose', type=str, default='rtmpose_aircraft_pose_results.json')
    parser.add_argument('--show', action='store_true', help='Show the visualization window.')
    parser.add_argument("--calibration", type=str, default=None, help="Path to calibration JSON file to enable Ground Truth mode.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    estimator = RTMPosePoseEstimator(args)
    estimator.run()
    print("🏁 Process finished.")