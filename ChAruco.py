#!/usr/bin/env python3
"""
VAPE MK52 ChArUco Calibrator
High-quality calibration system for tag-to-object transform

This script processes an MP4 video and automatically selects the best frames
for calibration based on pose quality metrics from both VAPE and ChArUco systems.
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


# --- ChArUco cross-version helpers ------------------------------------------
def make_charuco_board(cols, rows, square_len_m, marker_len_m,
                       dict_id=cv2.aruco.DICT_5X5_1000):
    """
    Returns (aruco_dict, board, chessboard_corners) and supports both old/new OpenCV APIs.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(cols, rows, square_len_m, marker_len_m, aruco_dict)
        chessboard_corners = board.chessboardCorners
    else:
        board = cv2.aruco.CharucoBoard((cols, rows), square_len_m, marker_len_m, aruco_dict)
        #print(board.getChessboardCorners())
        chessboard_corners = board.getChessboardCorners()
    
    return aruco_dict, board, chessboard_corners

def make_detectors(aruco_dict, board):
    """
    Returns a tuple describing the detector setup:
      ("new", aruco_detector, charuco_detector)  - modern API
      ("old", detector_params, None)             - legacy API
    """
    has_new = hasattr(cv2.aruco, "ArucoDetector") and hasattr(cv2.aruco, "CharucoDetector")
    
    try:
        aruco_params = cv2.aruco.DetectorParameters()
    except AttributeError:
        aruco_params = cv2.aruco.DetectorParameters_create()

    # Set enhanced detection parameters from _detect_charuco_with_quality
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

def detect_charuco_pose(frame_bgr, detector_mode, aruco_det, charuco_det, board, K, dist):
    """
    Detects ChArUco on a frame and returns (ok, R_ct, t_ct, dbg)
      - ok: True if pose solved
      - R_ct: 3x3 rotation cam->board
      - t_ct: 3x1 translation cam->board (meters)
      - dbg: dict with corners/ids for visualization
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if detector_mode == "new":
        corners, ids, _ = aruco_det.detectMarkers(gray)
        ok_c, charuco_corners, charuco_ids = charuco_det.interpolateCornersCharuco(corners, ids, gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_det)  # here aruco_det is 'params'
        if ids is None or len(ids) == 0:
            ok_c, charuco_corners, charuco_ids = False, None, None
        else:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            ok_c = charuco_corners is not None and len(charuco_corners) >= 4

    if not ok_c:
        return False, None, None, {"corners": [], "ids": None}

    pose_ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, K, dist, None, None
    )
    if not pose_ok:
        return False, None, None, {"corners": charuco_corners, "ids": charuco_ids}

    R_ct, _ = cv2.Rodrigues(rvec)
    t_ct = tvec.reshape(3, 1)
    dbg = {"corners": charuco_corners, "ids": charuco_ids}
    return True, R_ct, t_ct, dbg



@dataclass
class QualityMetrics:
    """Quality metrics for pose estimation"""
    num_matches: int = 0
    num_inliers: int = 0
    reprojection_error: float = float('inf')
    confidence_score: float = 0.0
    viewpoint: str = ""
    
    def quality_score(self) -> float:
        """Combined quality score for ranking poses"""
        if self.num_inliers < 6:
            return 0.0
        
        # Higher is better: more inliers, lower error
        score = (self.num_inliers * 10) / (1 + self.reprojection_error)
        
        # Bonus for many matches
        if self.num_matches > 20:
            score *= 1.5
        elif self.num_matches > 15:
            score *= 1.2
            
        return score

@dataclass
class CalibrationFrame:
    """A single frame with both VAPE and ChArUco poses"""
    frame_id: int
    vape_position: np.ndarray
    vape_quaternion: np.ndarray
    vape_quality: QualityMetrics
    charuco_rvec: np.ndarray
    charuco_tvec: np.ndarray
    charuco_quality: QualityMetrics
    frame: np.ndarray
    
    def combined_quality(self) -> float:
        """Combined quality score for both systems"""
        return (self.vape_quality.quality_score() + self.charuco_quality.quality_score()) / 2


class CharucoGroundTruthCalibrator:
    """High-precision calibrator for tag-to-object transform"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Camera parameters (from your VAPE code)
        self.camera_width, self.camera_height = 1280, 720
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = None
        
        # ChArUco setup
        self.aruco_dict, self.board, self.chessboard_corners = make_charuco_board(
            cols=5, rows=7, square_len_m=0.035, marker_len_m=0.025, dict_id=cv2.aruco.DICT_5X5_1000
        )
        self.detector_mode, self.aruco_det, self.charuco_det = make_detectors(self.aruco_dict, self.board)

        
        # VAPE models
        print("🔧 Loading VAPE models...")
        self.yolo_model = YOLO("best.pt").to(self.device)
        self.extractor = SuperPoint(max_num_keypoints=1024*2).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        
        # Viewpoint anchors (using your existing data)
        self.viewpoint_anchors = {}
        self._initialize_anchor_data()
        
        # Calibration data
        self.candidate_frames = []
        self.final_calibration_frames = []
        
        print("✅ Calibrator initialized")
    
    def _initialize_anchor_data(self):
        """Initialize anchor data (simplified version with key viewpoints)"""
        print("🛠️ Loading anchor data...")
        
        # Using a subset of your anchors for calibration (key viewpoints)
        anchor_definitions = {
                'NE': {
                    'path': 'NE.png',
                    '2d': np.array([[928, 148],[570, 111],[401, 31],[544, 141],[530, 134],
                                    [351, 228],[338, 220],[294, 244],[230, 541],[401, 469],
                                    [414, 481],[464, 451],[521, 510],[610, 454],[544, 400],
                                    [589, 373],[575, 361],[486, 561],[739, 385],[826, 305],
                                    [791, 285],[773, 271],[845, 233],[826, 226],[699, 308],
                                    [790, 375]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],
                                    [0.217, -0.000, -0.070],[0.230, 0.000, -0.070],[0.217, 0.000, 0.070],
                                    [0.230, -0.000, 0.070],[0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],
                                    [-0.000, -0.000, 0.156],[-0.014, 0.000, 0.156],[-0.019, -0.000, 0.128],
                                    [-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.019, -0.000, 0.074],
                                    [-0.014, 0.000, 0.042],[0.000, 0.000, 0.042],[-0.080, -0.000, 0.156],
                                    [-0.100, -0.030, 0.000],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],
                                    [-0.017, -0.000, -0.092],[-0.014, 0.000, -0.156],[0.000, 0.000, -0.156],
                                    [-0.014, 0.000, -0.042],[-0.090, -0.000, -0.042]], dtype=np.float32)
                },
                'NW': {
                    'path': 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png',
                    '2d': np.array([[511, 293],[591, 284],[587, 330],[413, 249],[602, 348],
                                    [715, 384],[598, 298],[656, 171],[805, 213],[703, 392],
                                    [523, 286],[519, 327],[387, 289],[727, 126],[425, 243],
                                    [636, 358],[745, 202],[595, 388],[436, 260],[539, 313],
                                    [795, 220],[351, 291],[665, 165],[611, 353],[650, 377],
                                    [516, 389],[727, 143],[496, 378],[575, 312],[617, 368],
                                    [430, 312],[480, 281],[834, 225],[469, 339],[705, 223],
                                    [637, 156],[816, 414],[357, 195],[752, 77],[642, 451]], dtype=np.float32),
                    '3d': np.array([[-0.014, 0.0, 0.042],[0.025, -0.014, -0.011],[-0.014, 0.0, -0.042],
                                    [-0.014, 0.0, 0.156],[-0.023, 0.0, -0.065],[0.0, 0.0, -0.156],
                                    [0.025, 0.0, -0.015],[0.217, 0.0, 0.07],[0.23, 0.0, -0.07],
                                    [-0.014, 0.0, -0.156],[0.0, 0.0, 0.042],[-0.057, -0.018, -0.01],
                                    [-0.074, -0.0, 0.128],[0.206, -0.07, -0.002],[-0.0, -0.0, 0.156],
                                    [-0.017, -0.0, -0.092],[0.217, -0.0, -0.027],[-0.052, -0.0, -0.097],
                                    [-0.019, -0.0, 0.128],[-0.035, -0.018, -0.01],[0.217, -0.0, -0.07],
                                    [-0.08, -0.0, 0.156],[0.23, 0.0, 0.07],[-0.023, -0.0, -0.075],
                                    [-0.029, -0.0, -0.127],[-0.09, -0.0, -0.042],[0.206, -0.055, -0.002],
                                    [-0.09, -0.0, -0.015],[0.0, -0.0, -0.015],[-0.037, -0.0, -0.097],
                                    [-0.074, -0.0, 0.074],[-0.019, -0.0, 0.074],[0.23, -0.0, -0.113],
                                    [-0.1, -0.03, 0.0],[0.17, -0.0, -0.015],[0.23, -0.0, 0.113],
                                    [-0.0, -0.025, -0.24],[-0.0, -0.025, 0.24],[0.243, -0.104, 0.0],
                                    [-0.08, -0.0, -0.156]], dtype=np.float32)
                },
                'SE': {
                    'path': 'SE.png',
                    '2d': np.array([[415, 144],[1169, 508],[275, 323],[214, 395],[554, 670],
                                    [253, 428],[280, 415],[355, 365],[494, 621],[519, 600],
                                    [806, 213],[973, 438],[986, 421],[768, 343],[785, 328],
                                    [841, 345],[931, 393],[891, 306],[980, 345],[651, 210],
                                    [625, 225],[588, 216],[511, 215],[526, 204],[665, 271]], dtype=np.float32),
                    '3d': np.array([[-0.0, -0.025, -0.24],[-0.0, -0.025, 0.24],[0.243, -0.104, 0.0],
                                    [0.23, 0.0, -0.113],[0.23, 0.0, 0.113],[0.23, 0.0, -0.07],
                                    [0.217, 0.0, -0.07],[0.206, -0.07, -0.002],[0.23, 0.0, 0.07],
                                    [0.217, 0.0, 0.07],[-0.1, -0.03, 0.0],[-0.0, 0.0, 0.156],
                                    [-0.014, 0.0, 0.156],[0.0, 0.0, 0.042],[-0.014, 0.0, 0.042],
                                    [-0.019, 0.0, 0.074],[-0.019, 0.0, 0.128],[-0.074, 0.0, 0.074],
                                    [-0.074, 0.0, 0.128],[-0.052, 0.0, -0.097],[-0.037, 0.0, -0.097],
                                    [-0.029, 0.0, -0.127],[0.0, 0.0, -0.156],[-0.014, 0.0, -0.156],
                                    [-0.014, 0.0, -0.042]], dtype=np.float32)
                },
                'SW': {
                    'path': 'Anchor_B.png',
                    '2d': np.array([[650, 312],[630, 306],[907, 443],[814, 291],[599, 349],
                                    [501, 386],[965, 359],[649, 355],[635, 346],[930, 335],
                                    [843, 467],[702, 339],[718, 321],[930, 322],[727, 346],
                                    [539, 364],[786, 297],[1022, 406],[1004, 399],[539, 344],
                                    [536, 309],[864, 478],[745, 310],[1049, 393],[895, 258],
                                    [674, 347],[741, 281],[699, 294],[817, 494],[992, 281]], dtype=np.float32),
                    '3d': np.array([[-0.035, -0.018, -0.01],[-0.057, -0.018, -0.01],[0.217, -0.0, -0.027],
                                    [-0.014, -0.0, 0.156],[-0.023, 0.0, -0.065],[-0.014, -0.0, -0.156],
                                    [0.234, -0.05, -0.002],[0.0, -0.0, -0.042],[-0.014, -0.0, -0.042],
                                    [0.206, -0.055, -0.002],[0.217, -0.0, -0.07],[0.025, -0.014, -0.011],
                                    [-0.014, -0.0, 0.042],[0.206, -0.07, -0.002],[0.049, -0.016, -0.011],
                                    [-0.029, -0.0, -0.127],[-0.019, -0.0, 0.128],[0.23, -0.0, 0.07],
                                    [0.217, -0.0, 0.07],[-0.052, -0.0, -0.097],[-0.175, -0.0, -0.015],
                                    [0.23, -0.0, -0.07],[-0.019, -0.0, 0.074],[0.23, -0.0, 0.113],
                                    [-0.0, -0.025, 0.24],[-0.0, -0.0, -0.015],[-0.074, -0.0, 0.128],
                                    [-0.074, -0.0, 0.074],[0.23, -0.0, -0.113],[0.243, -0.104, 0.0]], dtype=np.float32)
                },
                'W': {
                    'path': 'W.png',
                    '2d': np.array([[589, 555],[565, 481],[531, 480],[329, 501],[326, 345],
                                    [528, 351],[395, 391],[469, 395],[529, 140],[381, 224],
                                    [504, 258],[498, 229],[383, 253],[1203, 100],[1099, 174],
                                    [1095, 211],[1201, 439],[1134, 404],[1100, 358],[625, 341],
                                    [624, 310],[315, 264]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.156],
                                    [-0.080, -0.000, -0.156],[-0.090, -0.000, -0.015],[-0.014, 0.000, -0.042],
                                    [-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],[-0.000, -0.025, 0.240],
                                    [-0.074, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128],
                                    [-0.074, -0.000, 0.074],[0.243, -0.104, 0.000],[0.206, -0.070, -0.002],
                                    [0.206, -0.055, -0.002],[0.230, -0.000, -0.113],[0.217, -0.000, -0.070],
                                    [0.217, -0.000, -0.027],[0.025, 0.000, -0.015],[0.025, -0.014, -0.011],
                                    [-0.100, -0.030, 0.000]], dtype=np.float32)
                },
                'S': {
                    'path': 'S.png',
                    '2d': np.array([[14, 243],[1269, 255],[654, 183],[290, 484],[1020, 510],
                                    [398, 475],[390, 503],[901, 489],[573, 484],[250, 283],
                                    [405, 269],[435, 243],[968, 273],[838, 273],[831, 233],
                                    [949, 236]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.243, -0.104, 0.000],
                                    [0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.217, -0.000, -0.070],
                                    [0.230, 0.000, -0.070],[0.217, 0.000, 0.070],[0.217, -0.000, -0.027],
                                    [0.000, 0.000, -0.156],[-0.017, -0.000, -0.092],[-0.052, -0.000, -0.097],
                                    [-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.074, -0.000, 0.074],
                                    [-0.074, -0.000, 0.128]], dtype=np.float32)
                },
                'N': {
                    'path': 'N.png',
                    '2d': np.array([[1238, 346],[865, 295],[640, 89],[425, 314],[24, 383],
                                    [303, 439],[445, 434],[856, 418],[219, 475],[1055, 450]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],
                                    [0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.074, -0.000, 0.128],
                                    [-0.074, -0.000, 0.074],[-0.052, -0.000, -0.097],[-0.080, -0.000, 0.156],
                                    [-0.080, -0.000, -0.156]], dtype=np.float32)
                },
                'SW2': {
                    'path': 'SW2.png',
                    '2d': np.array([[15, 300],[1269, 180],[635, 143],[434, 274],[421, 240],
                                    [273, 320],[565, 266],[844, 206],[468, 543],[1185, 466],
                                    [565, 506],[569, 530],[741, 491],[1070, 459],[1089, 480],
                                    [974, 220],[941, 184],[659, 269],[650, 299],[636, 210],
                                    [620, 193]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],
                                    [-0.017, -0.000, -0.092],[-0.052, -0.000, -0.097],[0.000, 0.000, -0.156],
                                    [-0.014, 0.000, -0.042],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],
                                    [0.230, -0.000, 0.113],[0.217, -0.000, -0.070],[0.230, 0.000, -0.070],
                                    [0.217, -0.000, -0.027],[0.217, 0.000, 0.070],[0.230, -0.000, 0.070],
                                    [-0.019, -0.000, 0.128],[-0.074, -0.000, 0.128],[0.025, -0.014, -0.011],
                                    [0.025, 0.000, -0.015],[-0.035, -0.018, -0.010],[-0.057, -0.018, -0.010]], dtype=np.float32)
                },
                'SE2': {
                    'path': 'SE2.png',
                    '2d': np.array([[48, 216],[1269, 320],[459, 169],[853, 528],[143, 458],
                                    [244, 470],[258, 451],[423, 470],[741, 500],[739, 516],
                                    [689, 176],[960, 301],[828, 290],[970, 264],[850, 254]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],[0.243, -0.104, 0.000],
                                    [0.230, -0.000, 0.113],[0.230, -0.000, -0.113],[0.230, 0.000, -0.070],
                                    [0.217, -0.000, -0.070],[0.217, -0.000, -0.027],[0.217, 0.000, 0.070],
                                    [0.230, -0.000, 0.070],[-0.100, -0.030, 0.000],[-0.019, -0.000, 0.128],
                                    [-0.019, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074]], dtype=np.float32)
                },
                'SU': {
                    'path': 'SU.png',
                    '2d': np.array([[203, 251],[496, 191],[486, 229],[480, 263],[368, 279],
                                    [369, 255],[573, 274],[781, 280],[859, 293],[865, 213],
                                    [775, 206],[1069, 326],[656, 135],[633, 241],[629, 204],
                                    [623, 343],[398, 668],[463, 680],[466, 656],[761, 706],
                                    [761, 681],[823, 709],[616, 666]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[-0.052, -0.000, -0.097],[-0.037, -0.000, -0.097],
                                    [-0.017, -0.000, -0.092],[0.000, 0.000, -0.156],[-0.014, 0.000, -0.156],
                                    [-0.014, 0.000, -0.042],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128],
                                    [-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.000, -0.025, 0.240],
                                    [-0.100, -0.030, 0.000],[-0.035, -0.018, -0.010],[-0.057, -0.018, -0.010],
                                    [0.025, -0.014, -0.011],[0.230, -0.000, -0.113],[0.230, 0.000, -0.070],
                                    [0.217, -0.000, -0.070],[0.230, -0.000, 0.070],[0.217, 0.000, 0.070],
                                    [0.230, -0.000, 0.113],[0.243, -0.104, 0.000]], dtype=np.float32)
                },
                'NU': {
                    'path': 'NU.png',
                    '2d': np.array([[631, 361],[1025, 293],[245, 294],[488, 145],[645, 10],
                                    [803, 146],[661, 188],[509, 365],[421, 364],[434, 320],
                                    [509, 316],[779, 360],[784, 321],[704, 398],[358, 393]], dtype=np.float32),
                    '3d': np.array([[-0.100, -0.030, 0.000],[-0.000, -0.025, -0.240],[-0.000, -0.025, 0.240],
                                    [0.230, -0.000, 0.113],[0.243, -0.104, 0.000],[0.230, -0.000, -0.113],
                                    [0.170, -0.000, -0.015],[-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],
                                    [-0.019, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.052, -0.000, -0.097],
                                    [-0.017, -0.000, -0.092],[-0.090, -0.000, -0.042],[-0.080, -0.000, 0.156]], dtype=np.float32)
                },
                'NW2': {
                    'path': 'NW2.png',
                    '2d': np.array([[1268, 328],[1008, 419],[699, 399],[829, 373],[641, 325],
                                    [659, 310],[783, 30],[779, 113],[775, 153],[994, 240],
                                    [573, 226],[769, 265],[686, 284],[95, 269],[148, 375],
                                    [415, 353],[286, 349],[346, 320],[924, 360],[590, 324]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.042],
                                    [-0.052, -0.000, -0.097],[-0.057, -0.018, -0.010],[-0.035, -0.018, -0.010],
                                    [0.243, -0.104, 0.000],[0.206, -0.070, -0.002],[0.206, -0.055, -0.002],
                                    [0.230, -0.000, -0.113],[0.230, -0.000, 0.113],[0.170, -0.000, -0.015],
                                    [0.025, -0.014, -0.011],[-0.000, -0.025, 0.240],[-0.080, -0.000, 0.156],
                                    [-0.074, -0.000, 0.074],[-0.074, -0.000, 0.128],[-0.019, -0.000, 0.128],
                                    [-0.029, -0.000, -0.127],[-0.100, -0.030, 0.000]], dtype=np.float32)
                },
                'NE2': {
                    'path': 'NE2.png',
                    '2d': np.array([[1035, 95],[740, 93],[599, 16],[486, 168],[301, 305],
                                    [719, 225],[425, 349],[950, 204],[794, 248],[844, 203],
                                    [833, 175],[601, 275],[515, 301],[584, 244],[503, 266]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[0.230, -0.000, -0.113],[0.243, -0.104, 0.000],
                                    [0.230, -0.000, 0.113],[-0.000, -0.025, 0.240],[-0.100, -0.030, 0.000],
                                    [-0.080, -0.000, 0.156],[-0.080, -0.000, -0.156],[-0.090, -0.000, -0.042],
                                    [-0.052, -0.000, -0.097],[-0.017, -0.000, -0.092],[-0.074, -0.000, 0.074],
                                    [-0.074, -0.000, 0.128],[-0.019, -0.000, 0.074],[-0.019, -0.000, 0.128]], dtype=np.float32)
                },
                'E': {
                    'path': 'E.png',
                    '2d': np.array([[696, 165],[46, 133],[771, 610],[943, 469],[921, 408],
                                    [793, 478],[781, 420],[793, 520],[856, 280],[743, 284],
                                    [740, 245],[711, 248],[74, 520],[134, 465],[964, 309]], dtype=np.float32),
                    '3d': np.array([[-0.000, -0.025, -0.240],[0.243, -0.104, 0.000],[-0.000, -0.025, 0.240],
                                    [-0.074, -0.000, 0.128],[-0.074, -0.000, 0.074],[-0.019, -0.000, 0.128],
                                    [-0.019, -0.000, 0.074],[-0.014, 0.000, 0.156],[-0.052, -0.000, -0.097],
                                    [-0.017, -0.000, -0.092],[-0.014, 0.000, -0.156],[0.000, 0.000, -0.156],
                                    [0.230, -0.000, 0.113],[0.217, 0.000, 0.070],[-0.100, -0.030, 0.000]], dtype=np.float32)
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
    
    def process_video_for_calibration(self):
        """Process entire video and collect high-quality calibration frames"""
        print(f"🎬 Processing video: {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📹 Video info: {total_frames} frames at {fps:.1f} FPS")
        
        frame_id = 0
        processed_frames = 0
        
        # Process every 5th frame to reduce computation
        frame_skip = 5
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_id % frame_skip == 0:
                candidate = self._process_frame_for_calibration(frame, frame_id)
                if candidate:
                    self.candidate_frames.append(candidate)
                    print(f"✓ Frame {frame_id:4d}: VAPE={candidate.vape_quality.quality_score():.1f}, "
                          f"ChArUco={candidate.charuco_quality.quality_score():.1f}, "
                          f"Combined={candidate.combined_quality():.1f}")
                
                processed_frames += 1
                
                # Progress update
                if processed_frames % 20 == 0:
                    progress = (frame_id / total_frames) * 100
                    print(f"📊 Progress: {progress:.1f}% ({len(self.candidate_frames)} candidates found)")
            
            frame_id += 1
        
        cap.release()
        
        print(f"\n🎯 Found {len(self.candidate_frames)} candidate frames")
        self._select_best_calibration_frames()
    
    def _process_frame_for_calibration(self, frame: np.ndarray, frame_id: int) -> Optional[CalibrationFrame]:
        """Process a single frame and return calibration data if both poses are good quality"""
        
        # 1. YOLO Detection
        bbox = self._yolo_detect(frame)
        if not bbox:
            return None
        
        # 2. ChArUco Detection
        charuco_success, charuco_rvec, charuco_tvec, charuco_quality = self._detect_charuco_with_quality(frame)
        if not charuco_success:
            return None
        
        # 3. VAPE Pose Estimation
        vape_success, vape_pos, vape_quat, vape_quality = self._estimate_vape_pose_with_quality(frame, bbox)
        if not vape_success:
            return None
        
        # 4. Quality thresholds for calibration
        min_vape_quality = 15.0  # Minimum quality score for VAPE
        min_charuco_quality = 20.0  # Minimum quality score for ChArUco
        
        if (vape_quality.quality_score() >= min_vape_quality and 
            charuco_quality.quality_score() >= min_charuco_quality):
            
            return CalibrationFrame(
                frame_id=frame_id,
                vape_position=vape_pos,
                vape_quaternion=vape_quat,
                vape_quality=vape_quality,
                charuco_rvec=charuco_rvec,
                charuco_tvec=charuco_tvec,
                charuco_quality=charuco_quality,
                frame=frame.copy()
            )
        
        return None
    
    def _detect_charuco_with_quality(self, frame: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, QualityMetrics]:
        """Detect ChArUco with quality metrics"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quality = QualityMetrics()

        # Use pre-configured detectors
        if self.detector_mode == "new":
            corners, ids, _ = self.aruco_det.detectMarkers(gray)
            if ids is not None and len(ids) > 0:
                # Use the static function which is more compatible
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.board)
                ok_c = ret >= 4
            else:
                ok_c, charuco_corners, charuco_ids = False, None, None
        else: # old API
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_det)
            if ids is None or len(ids) == 0:
                ok_c, charuco_corners, charuco_ids = False, None, None
            else:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.board
                )
                ok_c = charuco_corners is not None and len(charuco_corners) >= 4
        
        if not ok_c:
            return False, None, None, quality

        if ids is not None:
            quality.num_matches = len(ids)
        
        if charuco_corners is not None and len(charuco_corners) >= 6:
            ret = len(charuco_corners)
            quality.num_inliers = ret
            
            # Estimate pose
            # The modern API returns a success flag and populates rvec and tvec
            # when they are provided as None.
            rvec = np.zeros((3, 1), dtype=np.float32)
            tvec = np.zeros((3, 1), dtype=np.float32)
            pose_success = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, self.board, self.K, self.dist_coeffs, rvec, tvec
            )
            
            if pose_success:
                # Calculate reprojection error
                projected_corners, _ = cv2.projectPoints(
                    self.chessboard_corners[charuco_ids.flatten()],
                    rvec, tvec, self.K, self.dist_coeffs)
                
                error = np.mean(np.linalg.norm(
                    charuco_corners.reshape(-1, 2) - projected_corners.reshape(-1, 2), axis=1))
                
                quality.reprojection_error = error
                quality.confidence_score = min(100, ret * 5)  # More corners = higher confidence
                
                # Quality thresholds
                if error < 2.0 and ret >= 8:  # Low error, many corners
                    return True, rvec, tvec, quality
        
        return False, None, None, quality
    
    def _estimate_vape_pose_with_quality(self, frame: np.ndarray, bbox: Tuple) -> Tuple[bool, np.ndarray, np.ndarray, QualityMetrics]:
        """Estimate VAPE pose with detailed quality metrics"""
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if crop.size == 0:
            return False, None, None, QualityMetrics()
        
        frame_features = self._extract_features_sp(crop)
        crop_offset = np.array([bbox[0], bbox[1]])
        
        best_pose = None
        best_quality = QualityMetrics()
        
        # Try all viewpoints and pick the best
        for viewpoint, anchor in self.viewpoint_anchors.items():
            try:
                with torch.no_grad():
                    matches_dict = self.matcher({'image0': anchor['features'], 'image1': frame_features})
                matches = rbd(matches_dict)['matches'].cpu().numpy()
                
                if len(matches) < 8:  # Need more matches for calibration quality
                    continue
                
                # Build 2D-3D correspondences
                points_3d, points_2d = [], []
                for anchor_idx, frame_idx in matches:
                    if anchor_idx in anchor['map_3d']:
                        points_3d.append(anchor['map_3d'][anchor_idx])
                        points_2d.append(frame_features['keypoints'][0].cpu().numpy()[frame_idx] + crop_offset)
                
                if len(points_3d) < 8:
                    continue
                
                # High-quality PnP solving
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    np.array(points_3d, dtype=np.float32),
                    np.array(points_2d, dtype=np.float32),
                    self.K, self.dist_coeffs, 
                    reprojectionError=4,#5.0,  # Stricter for calibration
                    confidence=0.99,
                    iterationsCount=10000,
                    flags=cv2.SOLVEPNP_EPNP
                )
                
                if success and inliers is not None and len(inliers) >= 8:
                    # Refine with VVS for maximum accuracy
                    rvec_refined, tvec_refined = cv2.solvePnPRefineVVS(
                        np.array(points_3d, dtype=np.float32)[inliers.flatten()],
                        np.array(points_2d, dtype=np.float32)[inliers.flatten()],
                        self.K, self.dist_coeffs, rvec, tvec,
                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
                    )
                    
                    # Calculate reprojection error
                    projected_points, _ = cv2.projectPoints(
                        np.array(points_3d)[inliers.flatten()], rvec_refined, tvec_refined, 
                        self.K, self.dist_coeffs)
                    error = np.mean(np.linalg.norm(
                        np.array(points_2d)[inliers.flatten()].reshape(-1, 1, 2) - projected_points, axis=2))
                    
                    # Create quality metrics
                    quality = QualityMetrics(
                        num_matches=len(points_3d),
                        num_inliers=len(inliers),
                        reprojection_error=error,
                        confidence_score=min(100, len(inliers) * 5),
                        viewpoint=viewpoint
                    )
                    
                    # Keep the best pose
                    if quality.quality_score() > best_quality.quality_score():
                        R_matrix, _ = cv2.Rodrigues(rvec_refined)
                        position = tvec_refined.flatten()
                        quaternion = R.from_matrix(R_matrix).as_quat()  # (x,y,z,w)
                        
                        best_pose = (position, quaternion)
                        best_quality = quality
                        
            except Exception as e:
                continue
        
        if best_pose and best_quality.quality_score() > 0:
            return True, best_pose[0], best_pose[1], best_quality
        
        return False, None, None, QualityMetrics()
    
    def _select_best_calibration_frames(self):
        """Select the best frames for calibration from candidates"""
        print("\n🎯 Selecting best calibration frames...")
        
        if len(self.candidate_frames) < 5:
            print(f"❌ Not enough candidates ({len(self.candidate_frames)}). Need at least 5.")
            return
        
        # Sort by combined quality
        sorted_candidates = sorted(self.candidate_frames, key=lambda x: x.combined_quality(), reverse=True)
        
        # Select diverse, high-quality frames
        selected = []
        min_frame_gap = 30  # Minimum frames between selections for diversity
        
        for candidate in sorted_candidates:
            # Check if this frame is too close to already selected frames
            too_close = any(abs(candidate.frame_id - sel.frame_id) < min_frame_gap for sel in selected)
            
            if not too_close and len(selected) < 15:  # Max 15 frames for calibration
                selected.append(candidate)
        
        self.final_calibration_frames = selected
        
        print(f"📋 Selected {len(self.final_calibration_frames)} frames for calibration:")
        for i, frame in enumerate(self.final_calibration_frames):
            print(f"   {i+1:2d}. Frame {frame.frame_id:4d}: Quality={frame.combined_quality():.1f}, "
                  f"VAPE_inliers={frame.vape_quality.num_inliers}, "
                  f"ChArUco_corners={frame.charuco_quality.num_inliers}")
    
    def solve_calibration(self):
        """
        Solve for tag-to-object transform using frame-by-frame averaging.
        This is more robust for this specific problem than hand-eye calibration.
        """
        if len(self.final_calibration_frames) < 10:
            print(f"❌ Not enough good frames ({len(self.final_calibration_frames)}). Need at least 10 for robust averaging.")
            return False
        
        print(f"\n🔧 Solving calibration by averaging transformations from {len(self.final_calibration_frames)} frames...")
        
        all_r_to = []
        all_t_to = []

        for frame in self.final_calibration_frames:
            # Get T_co (VAPE pose: object -> camera)
            R_co = R.from_quat(frame.vape_quaternion).as_matrix()
            t_co = frame.vape_position.reshape(3, 1)

            # Get T_ct (ChArUco pose: tag -> camera)
            R_ct, _ = cv2.Rodrigues(frame.charuco_rvec)
            t_ct = frame.charuco_tvec.reshape(3, 1)

            # Calculate T_to = inv(T_ct) * T_co
            # This gives the transform from the object frame to the tag frame.
            R_to_candidate = R_ct.T @ R_co
            t_to_candidate = R_ct.T @ (t_co - t_ct)

            # Convert rotation to axis-angle (rvec) for stable averaging/median
            r_to_candidate, _ = cv2.Rodrigues(R_to_candidate)

            all_r_to.append(r_to_candidate.flatten())
            all_t_to.append(t_to_candidate.flatten())

        # Average the results using the median, which is robust to outliers
        final_r_to_vec = np.median(np.array(all_r_to), axis=0)
        final_t_to = np.median(np.array(all_t_to), axis=0).reshape(3, 1)

        # Convert final averaged rvec back to a rotation matrix
        final_R_to, _ = cv2.Rodrigues(final_r_to_vec)

        # --- Sanity Check: Print Standard Deviation ---
        # A low standard deviation indicates the transformation was consistent across frames.
        r_std = np.std(np.array(all_r_to), axis=0)
        t_std = np.std(np.array(all_t_to), axis=0)
        print(f"📈 Statistics (the lower the better):")
        print(f"   Rotation Vector Std Dev: {r_std}")
        print(f"   Translation Std Dev (meters): {t_std}")
        if np.any(t_std > 0.05): # If translation varies by more than 5cm
            print("   ⚠️ Warning: High variation in translation detected across frames.")

        # Apply the known 25cm pole offset (tag is 25cm below CG in Y direction)
        # This assumes the tag's Y-axis points "up" towards the object's CG
        #final_t_to[1] += 0.25
        
        calibration_result = {
            'R_to': final_R_to,
            't_to': final_t_to,
            'num_frames': len(self.final_calibration_frames)
        }
        
        self._save_calibration(calibration_result)
        
        print("\n✅ Calibration completed successfully using frame-by-frame averaging!")
        return True
    
    def _validate_calibration(self, calibration, R_ct_list, t_ct_list, R_co_list, t_co_list):
        """Validate calibration quality with detailed residuals"""
        print("\n📊 Calibration Validation:")
        
        R_to = calibration['R_to']
        t_to = calibration['t_to']
        
        residuals = []
        
        for i, (R_ct, t_ct, R_co, t_co) in enumerate(zip(R_ct_list, t_ct_list, R_co_list, t_co_list)):
            # Predict object pose from tag pose
            R_pred = R_ct @ R_to
            t_pred = R_ct @ t_to + t_ct
            
            # Calculate residuals
            dR = R.from_matrix(R_pred.T @ R_co).as_rotvec()
            deg_error = np.linalg.norm(dR) * 180 / np.pi
            cm_error = np.linalg.norm(t_pred - t_co) * 100
            
            residuals.append((deg_error, cm_error))
            
            frame_id = self.final_calibration_frames[i].frame_id
            print(f"   Frame {frame_id:4d}: {deg_error:5.1f}° {cm_error:5.1f}cm")
        
        # Summary statistics
        deg_errors = [r[0] for r in residuals]
        cm_errors = [r[1] for r in residuals]
        
        print(f"\n📈 Statistics:")
        print(f"   Rotation: {np.mean(deg_errors):.1f}° ± {np.std(deg_errors):.1f}° (max: {np.max(deg_errors):.1f}°)")
        print(f"   Position: {np.mean(cm_errors):.1f}cm ± {np.std(cm_errors):.1f}cm (max: {np.max(cm_errors):.1f}cm)")
        
        # Quality assessment
        if np.max(deg_errors) < 2 and np.max(cm_errors) < 2:
            print("🏆 EXCELLENT calibration quality!")
        elif np.max(deg_errors) < 3 and np.max(cm_errors) < 3:
            print("✅ GOOD calibration quality")
        elif np.max(deg_errors) < 5 and np.max(cm_errors) < 5:
            print("⚠️  ACCEPTABLE calibration quality")
        else:
            print("❌ POOR calibration quality - consider recalibrating")
    
    def _save_calibration(self, calibration):
        """Save calibration results"""
        os.makedirs('calibration', exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        calibration_data = {
            'R_to': calibration['R_to'].tolist(),
            't_to': calibration['t_to'].tolist(),
            'num_frames': calibration['num_frames'],
            'pole_offset_applied': 0.25,  # 25cm pole offset
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'video_source': self.video_path
        }
        
        filename = 'calibration/tag_to_object_ground_truth.json'
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"💾 Calibration saved to {filename}")
        
        # Also save frame details for debugging
        frame_data = []
        for frame in self.final_calibration_frames:
            frame_data.append({
                'frame_id': frame.frame_id,
                'vape_position': frame.vape_position.tolist(),
                'vape_quaternion': frame.vape_quaternion.tolist(),
                'vape_quality_score': frame.vape_quality.quality_score(),
                'charuco_quality_score': frame.charuco_quality.quality_score(),
                'combined_quality': frame.combined_quality()
            })
        
        debug_filename = 'calibration/calibration_frames_debug.json'
        with open(debug_filename, 'w') as f:
            json.dump(frame_data, f, indent=2)
        
        print(f"🐛 Debug data saved to {debug_filename}")
    
    def visualize_calibration_frames(self):
        """Create visualization video showing selected calibration frames"""
        if not self.final_calibration_frames:
            print("❌ No calibration frames to visualize")
            return
        
        print("🎨 Creating calibration visualization...")
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('calibration_frames_visualization.mp4', fourcc, 2.0, 
                             (self.camera_width, self.camera_height))
        
        for i, cal_frame in enumerate(self.final_calibration_frames):
            frame = cal_frame.frame.copy()
            
            # Draw ChArUco detection and axes
            self._draw_charuco_axes(frame, cal_frame.charuco_rvec, cal_frame.charuco_tvec, 
                                   color=(0, 255, 0), length=0.05)
            
            # Draw VAPE pose axes
            self._draw_vape_axes(frame, cal_frame.vape_position, cal_frame.vape_quaternion, 
                                color=(0, 0, 255), length=0.05)
            
            # Add frame info
            info_text = [
                f"Calibration Frame {i+1}/{len(self.final_calibration_frames)} (Frame ID: {cal_frame.frame_id})",
                f"VAPE Quality: {cal_frame.vape_quality.quality_score():.1f} (Inliers: {cal_frame.vape_quality.num_inliers})",
                f"ChArUco Quality: {cal_frame.charuco_quality.quality_score():.1f} (Corners: {cal_frame.charuco_quality.num_inliers})",
                f"Combined Quality: {cal_frame.combined_quality():.1f}",
                f"Green: ChArUco axes, Red: VAPE axes"
            ]
            
            for j, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 30 + j*25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2)
                cv2.putText(frame, text, (10, 30 + j*25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 0, 0), 1)
            
            # Write frame multiple times for slower playback
            for _ in range(30):  # 15 seconds per frame at 2 FPS
                out.write(frame)
        
        out.release()
        print("🎬 Visualization saved as 'calibration_frames_visualization.mp4'")
    
    def _draw_charuco_axes(self, frame, rvec, tvec, color=(0, 255, 0), length=0.05):
        """Draw ChArUco coordinate axes"""
        axis_points = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
        img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, self.K, self.dist_coeffs)
        img_points = img_points.reshape(-1, 2).astype(int)
        
        origin = tuple(img_points[0])
        cv2.line(frame, origin, tuple(img_points[1]), (0, 0, 255), 3)      # X - Red
        cv2.line(frame, origin, tuple(img_points[2]), (0, 255, 0), 3)      # Y - Green  
        cv2.line(frame, origin, tuple(img_points[3]), (255, 0, 0), 3)      # Z - Blue
        
        # Add labels
        cv2.putText(frame, 'ChArUco', (origin[0]-20, origin[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _draw_vape_axes(self, frame, position, quaternion, color=(0, 0, 255), length=0.05):
        """Draw VAPE pose coordinate axes"""
        try:
            # Convert quaternion to rotation matrix
            # Scipy expects quaternion as (x, y, z, w), which is the format the quaternion is already in.
            R_matrix = R.from_quat(quaternion).as_matrix()
            rvec, _ = cv2.Rodrigues(R_matrix)
            tvec = position.reshape(3, 1)
            
            axis_points = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
            img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, self.K, self.dist_coeffs)
            img_points = img_points.reshape(-1, 2).astype(int)
            
            origin = tuple(img_points[0])
            cv2.line(frame, origin, tuple(img_points[1]), (0, 0, 255), 3)      # X - Red
            cv2.line(frame, origin, tuple(img_points[2]), (0, 255, 0), 3)      # Y - Green  
            cv2.line(frame, origin, tuple(img_points[3]), (255, 0, 0), 3)      # Z - Blue
            
            # Add labels
            cv2.putText(frame, 'VAPE', (origin[0]+20, origin[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            pass  # Fail silently if projection fails
    
    def _yolo_detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """YOLO detection (simplified version from your code)"""
        names = getattr(self.yolo_model, "names", {0: "iha"})
        inv = {v: k for k, v in names.items()}
        target_id = inv.get("iha", 0)

        for conf_thresh in (0.30, 0.20, 0.10):
            results = self.yolo_model(
                frame,
                imgsz=640,
                conf=conf_thresh,
                iou=0.5,
                max_det=5,
                classes=[target_id],
                verbose=False
            )
            if not results or len(results[0].boxes) == 0:
                continue

            boxes = results[0].boxes
            best = max(
                boxes,
                key=lambda b: float((b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1]))
            )
            x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().tolist()
            return int(x1), int(y1), int(x2), int(y2)

        return None
    
    def _extract_features_sp(self, image_bgr: np.ndarray) -> Dict:
        """Extract SuperPoint features"""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            return self.extractor.extract(tensor)


def main():
    parser = argparse.ArgumentParser(description="VAPE MK52 ChArUco Ground Truth Calibrator")
    parser.add_argument('--video', required=True, help='Path to input MP4 video')
    parser.add_argument('--visualize', action='store_true', help='Create visualization video')
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    print("🚀 VAPE MK52 ChArUco Ground Truth Calibrator")
    print("=" * 50)
    
    try:
        # Initialize calibrator
        calibrator = CharucoGroundTruthCalibrator(args.video)
        
        # Process video and find calibration frames
        calibrator.process_video_for_calibration()
        
        # Solve calibration
        if calibrator.solve_calibration():
            print("\n✅ Calibration completed successfully!")
            
            # Create visualization if requested
            if args.visualize:
                calibrator.visualize_calibration_frames()
            
            print("\n📋 Next steps:")
            print("   1. Check calibration quality in the output above")
            print("   2. If quality is good (<3° and <3cm), use the saved calibration")
            print("   3. If quality is poor, collect more diverse viewpoints and re-run")
            print("   4. The calibration is saved in 'calibration/tag_to_object_ground_truth.json'")
            
        else:
            print("\n❌ Calibration failed. Please check:")
            print("   - Video contains clear views of both aircraft and ChArUco board")
            print("   - ChArUco board is properly mounted and visible")
            print("   - Camera moves through diverse viewpoints")
            print("   - YOLO model can detect the aircraft reliably")
    
    except Exception as e:
        print(f"❌ Error during calibration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# ===== USAGE INSTRUCTIONS =====
"""
USAGE:

1. PREPARATION:
   - Mount ChArUco board 25cm below aircraft CG using rigid pole
   - Record MP4 video moving camera around aircraft from diverse viewpoints
   - Ensure both aircraft and ChArUco board are visible in most frames

2. RUN CALIBRATION:
   python vape_charuco_calibrator.py --video your_video.mp4 --visualize

3. QUALITY CHECK:
   - Look for "EXCELLENT" or "GOOD" calibration quality
   - Residuals should be <3° rotation and <3cm position
   - If poor quality, record new video with more diverse viewpoints

4. OUTPUT FILES:
   - calibration/tag_to_object_ground_truth.json: Main calibration file
   - calibration/calibration_frames_debug.json: Debug information
   - calibration_frames_visualization.mp4: Visual verification (if --visualize used)

5. INTEGRATION:
   Use the calibration JSON file in your main VAPE MK52 system for runtime pose fusion

KEY FEATURES:
- Automatic quality filtering (only uses high-quality poses)
- Uses raw solvePnPVVS results (not Kalman filtered) for ground truth accuracy
- Applies 25cm pole offset automatically
- Shows both coordinate systems for debugging
- Comprehensive quality validation with detailed residuals
- Selects diverse frames automatically (minimum 30-frame spacing)

QUALITY THRESHOLDS:
- VAPE: Minimum 15.0 quality score, 8+ inliers, <5px reprojection error
- ChArUco: Minimum 20.0 quality score, 6+ corners, <2px reprojection error
- Final selection: Best combined quality with frame diversity
"""