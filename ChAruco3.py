#!/usr/bin/env python3
"""
VAPE MK52 ChArUco Calibrator - Updated for A1 Board
High-quality calibration system for tag-to-object transform

This script processes an MP4 video and automatically selects the best frames
for calibration based on pose quality metrics from both VAPE and ChArUco systems.

Updated for A1 ChArUco board: 7x11 grid, 76.5mm squares, 57.3mm markers
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
                       dict_id=cv2.aruco.DICT_6X6_250):
    """
    Returns (aruco_dict, board, chessboard_corners) and supports both old/new OpenCV APIs.
    
    Parameters explained:
    - cols, rows: Grid dimensions (7x11 for A1 board = 38 ArUco markers)
    - square_len_m: Physical size of each square in meters (0.0765m = 76.5mm)
    - marker_len_m: Physical size of each ArUco marker in meters (0.0573m = 57.3mm)
    - dict_id: ArUco dictionary (DICT_6X6_250 = 6x6 bit markers, 250 unique IDs)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        # Older OpenCV API (< 4.7)
        board = cv2.aruco.CharucoBoard_create(cols, rows, square_len_m, marker_len_m, aruco_dict)
        chessboard_corners = board.chessboardCorners
    else:
        # Newer OpenCV API (>= 4.7)
        board = cv2.aruco.CharucoBoard((cols, rows), square_len_m, marker_len_m, aruco_dict)
        chessboard_corners = board.getChessboardCorners()
    
    return aruco_dict, board, chessboard_corners

def make_detectors(aruco_dict, board):
    """
    Returns a tuple describing the detector setup:
      ("new", aruco_detector, charuco_detector)  - modern API
      ("old", detector_params, None)             - legacy API
    
    Detection parameters explained:
    - adaptiveThreshWinSizeMin/Max: Window sizes for adaptive thresholding (5-21 pixels)
    - adaptiveThreshWinSizeStep: Step size for window size search (4 pixels)
    - minMarkerPerimeterRate: Minimum marker size relative to image (0.05 = 5% of image width)
    - maxMarkerPerimeterRate: Maximum marker size relative to image (0.3 = 30% of image width)
    """
    has_new = hasattr(cv2.aruco, "ArucoDetector") and hasattr(cv2.aruco, "CharucoDetector")
    
    try:
        aruco_params = cv2.aruco.DetectorParameters()
    except AttributeError:
        aruco_params = cv2.aruco.DetectorParameters_create()

    # Enhanced detection parameters optimized for A1 board at 2-3m distance
    aruco_params.adaptiveThreshWinSizeMin = 5      # Smaller markers need smaller windows
    aruco_params.adaptiveThreshWinSizeMax = 25     # Increased for better detection at distance
    aruco_params.adaptiveThreshWinSizeStep = 4     # Fine-grained search
    aruco_params.minMarkerPerimeterRate = 0.03     # Smaller minimum (markers can appear small at distance)
    aruco_params.maxMarkerPerimeterRate = 0.4      # Larger maximum for close-up detection
    
    # Additional parameters for robust detection
    aruco_params.minMarkerDistanceRate = 0.05      # Minimum distance between markers
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # Sub-pixel accuracy
    aruco_params.cornerRefinementWinSize = 5       # Window size for corner refinement
    aruco_params.cornerRefinementMaxIterations = 30
    aruco_params.cornerRefinementMinAccuracy = 0.1

    if has_new:
        aruco_det = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        charuco_det = cv2.aruco.CharucoDetector(board)
        return ("new", aruco_det, charuco_det)
    else:
        return ("old", aruco_params, None)

def test_charuco_detection(frame_bgr, detector_mode, aruco_det, charuco_det, board, aruco_dict, K, dist):
    """
    Test ChArUco detection on a frame and return detailed debug information.
    Returns (success, debug_info_dict)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    debug_info = {
        'aruco_markers_detected': 0,
        'charuco_corners_found': 0,
        'pose_estimation_success': False,
        'reprojection_error': float('inf'),
        'detection_details': []
    }

    # Step 1: Detect ArUco markers
    if detector_mode == "new":
        corners, ids, rejected = aruco_det.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_det)
    
    if ids is None or len(ids) == 0:
        debug_info['detection_details'].append("No ArUco markers detected")
        return False, debug_info
    
    debug_info['aruco_markers_detected'] = len(ids)
    debug_info['detected_marker_ids'] = ids.flatten().tolist()
    debug_info['detection_details'].append(f"Detected {len(ids)} ArUco markers: {ids.flatten()}")

    # Step 2: Interpolate ChArUco corners - use static function for reliability
    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board)
    ok_c = charuco_corners is not None and len(charuco_corners) >= 4

    if not ok_c:
        debug_info['detection_details'].append("Failed to interpolate ChArUco corners")
        return False, debug_info
    
    debug_info['charuco_corners_found'] = len(charuco_corners)
    debug_info['detection_details'].append(f"Interpolated {len(charuco_corners)} ChArUco corners")

    # Step 3: Estimate pose
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    pose_success = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, K, dist, rvec, tvec)
    
    if not pose_success:
        debug_info['detection_details'].append("Pose estimation failed")
        return False, debug_info
    
    debug_info['pose_estimation_success'] = True
    debug_info['rvec'] = rvec.flatten().tolist()
    debug_info['tvec'] = tvec.flatten().tolist()

    # Step 4: Calculate reprojection error
    try:
        board_corners = board.getChessboardCorners() if hasattr(board, 'getChessboardCorners') else board.chessboardCorners
        projected_corners, _ = cv2.projectPoints(
            board_corners[charuco_ids.flatten()], rvec, tvec, K, dist)
        
        error = np.mean(np.linalg.norm(
            charuco_corners.reshape(-1, 2) - projected_corners.reshape(-1, 2), axis=1))
        debug_info['reprojection_error'] = float(error)
        debug_info['detection_details'].append(f"Reprojection error: {error:.2f} pixels")
        
        # Quality assessment
        if error < 2.0 and len(charuco_corners) >= 8:
            debug_info['quality_assessment'] = "EXCELLENT - Ready for calibration"
        elif error < 3.0 and len(charuco_corners) >= 6:
            debug_info['quality_assessment'] = "GOOD - Usable for calibration"
        elif error < 5.0 and len(charuco_corners) >= 4:
            debug_info['quality_assessment'] = "ACCEPTABLE - May work for calibration"
        else:
            debug_info['quality_assessment'] = "POOR - Not suitable for calibration"
            
    except Exception as e:
        debug_info['detection_details'].append(f"Error calculating reprojection: {e}")
        return False, debug_info

    return True, debug_info

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
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_det)  # here aruco_det is 'params'
    
    if ids is None or len(ids) == 0:
        return False, None, None, {"corners": [], "ids": None}

    # Use static function for ChArUco corner interpolation (more reliable)
    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board)
    
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
    """High-precision calibrator for tag-to-object transform - Updated for A1 board"""
    
    def __init__(self, video_path: str, debug: bool = False):
        self.video_path = video_path
        self.debug = debug
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Camera parameters (from your VAPE code)
        self.camera_width, self.camera_height = 1280, 720
        fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = None
        
        # ChArUco setup - UPDATED FOR A1 BOARD
        print("Initializing A1 ChArUco board configuration:")
        print("  - Grid: 7 columns x 11 rows (38 ArUco markers)")
        print("  - Square size: 76.5mm (0.0765m)")
        print("  - Marker size: 57.3mm (0.0573m)")
        print("  - Dictionary: DICT_6X6_250 (6x6 bit patterns, 250 unique markers)")
        print("  - Board physical size: ~535mm x 841mm")
        
        self.aruco_dict, self.board, self.chessboard_corners = make_charuco_board(
            cols=7, rows=11,                    # A1 board grid dimensions
            square_len_m=0.0765,                # 76.5mm squares in meters
            marker_len_m=0.0573,                # 57.3mm markers in meters
            dict_id=cv2.aruco.DICT_6X6_250      # 6x6 dictionary for robustness
        )
        self.detector_mode, self.aruco_det, self.charuco_det = make_detectors(self.aruco_dict, self.board)
        
        print(f"  - Detection mode: {self.detector_mode}")
        print(f"  - Total chessboard corners available: {len(self.chessboard_corners)}")

        # VAPE models
        print("Loading VAPE models...")
        self.yolo_model = YOLO("best.pt").to(self.device)
        self.extractor = SuperPoint(max_num_keypoints=1024*2).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        
        # Viewpoint anchors (using your existing data)
        self.viewpoint_anchors = {}
        self._initialize_anchor_data()
        
        # Calibration data
        self.candidate_frames = []
        self.final_calibration_frames = []
        
        print("Calibrator initialized for A1 board")
    
    def test_board_detection(self, test_image_path: str = None, live_camera: bool = False):
        """
        Test ChArUco board detection to verify parameters are correct.
        
        Args:
            test_image_path: Path to test image with the board
            live_camera: Use camera for live testing
        """
        print("\n" + "="*50)
        print("CHARUCO BOARD DETECTION TEST")
        print("="*50)
        
        if live_camera:
            print("Starting live camera test. Press 'q' to quit, 's' to save test image.")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera")
                return
                
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Resize to expected resolution
                frame = cv2.resize(frame, (self.camera_width, self.camera_height))
                
                # Test detection
                success, debug_info = test_charuco_detection(
                    frame, self.detector_mode, self.aruco_det, self.charuco_det, 
                    self.board, self.aruco_dict, self.K, self.dist_coeffs
                )
                
                # Draw results on frame
                display_frame = self._draw_detection_results(frame, debug_info, success, self.K, self.dist_coeffs)
                
                cv2.imshow('ChArUco Detection Test', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite('charuco_test_image.jpg', frame)
                    print("Test image saved as 'charuco_test_image.jpg'")
            
            cap.release()
            cv2.destroyAllWindows()
            
        elif test_image_path and os.path.exists(test_image_path):
            print(f"Testing detection on image: {test_image_path}")
            
            frame = cv2.imread(test_image_path)
            if frame is None:
                print("Error: Could not load test image")
                return
                
            # Resize to expected resolution
            frame = cv2.resize(frame, (self.camera_width, self.camera_height))
            
            # Test detection
            success, debug_info = test_charuco_detection(
                frame, self.detector_mode, self.aruco_det, self.charuco_det, 
                self.board, self.aruco_dict, self.K, self.dist_coeffs
            )
            
            # Print detailed results
            self._print_detection_results(debug_info, success)
            
            # Show visual results
            display_frame = self._draw_detection_results(frame, debug_info, success, self.K, self.dist_coeffs)
            cv2.imwrite('charuco_detection_result.jpg', display_frame)
            print("Detection result saved as 'charuco_detection_result.jpg'")
            
            cv2.imshow('ChArUco Detection Result', display_frame)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        else:
            print("Error: No valid test input provided.")
            print("Usage:")
            print("  - For live camera: calibrator.test_board_detection(live_camera=True)")
            print("  - For test image: calibrator.test_board_detection('path/to/image.jpg')")
    
    def _draw_detection_results(self, frame, debug_info, success, K=None, dist=None):
        """Draw detection results on frame for visualization"""
        display_frame = frame.copy()

        # Draw axis if pose was estimated
        if success and K is not None and debug_info.get('pose_estimation_success', False):
            rvec = np.array(debug_info['rvec']).reshape(3, 1)
            tvec = np.array(debug_info['tvec']).reshape(3, 1)
            cv2.drawFrameAxes(display_frame, K, dist, rvec, tvec, 0.1)  # Draw 10cm axes
        
        # Draw text overlay
        y_offset = 30
        line_height = 25
        
        # Title
        title = "ChArUco Board Detection Test - A1 Board (7x11)"
        cv2.putText(display_frame, title, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, title, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        y_offset += line_height * 2
        
        # Status
        status = "SUCCESS" if success else "FAILED"
        color = (0, 255, 0) if success else (0, 0, 255)
        cv2.putText(display_frame, f"Status: {status}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Status: {status}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y_offset += line_height
        
        # Detection details
        info_lines = [
            f"ArUco markers detected: {debug_info.get('aruco_markers_detected', 0)}",
            f"ChArUco corners found: {debug_info.get('charuco_corners_found', 0)}",
            f"Pose estimation: {'OK' if debug_info.get('pose_estimation_success', False) else 'FAILED'}",
            f"Reprojection error: {debug_info.get('reprojection_error', float('inf')):.2f} pixels"
        ]
        
        if 'quality_assessment' in debug_info:
            info_lines.append(f"Quality: {debug_info['quality_assessment']}")
        
        for line in info_lines:
            cv2.putText(display_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(display_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if success else (0, 0, 255), 1)
            y_offset += line_height
        
        return display_frame
    
    def _print_detection_results(self, debug_info, success):
        """Print detailed detection results to console"""
        print(f"\nDetection Results:")
        print(f"  Status: {'SUCCESS' if success else 'FAILED'}")
        print(f"  ArUco markers detected: {debug_info.get('aruco_markers_detected', 0)}")
        
        if 'detected_marker_ids' in debug_info:
            print(f"  Detected marker IDs: {debug_info['detected_marker_ids']}")
            
        print(f"  ChArUco corners found: {debug_info.get('charuco_corners_found', 0)}")
        print(f"  Pose estimation success: {debug_info.get('pose_estimation_success', False)}")
        print(f"  Reprojection error: {debug_info.get('reprojection_error', float('inf')):.3f} pixels")
        
        if 'quality_assessment' in debug_info:
            print(f"  Quality assessment: {debug_info['quality_assessment']}")
        
        print(f"\nDetection details:")
        for detail in debug_info.get('detection_details', []):
            print(f"    - {detail}")
        
        if not success:
            print(f"\nTroubleshooting suggestions:")
            if debug_info.get('aruco_markers_detected', 0) == 0:
                print("    - Check lighting conditions (avoid glare/shadows)")
                print("    - Ensure board is printed clearly at high resolution")
                print("    - Verify camera focus and distance (try 1-3 meters)")
                print("    - Check for board warping or damage")
            elif debug_info.get('charuco_corners_found', 0) < 4:
                print("    - More of the board needs to be visible")
                print("    - Reduce camera angle (view more perpendicular to board)")
                print("    - Check for occlusions covering corner areas")
    
    def _initialize_anchor_data(self):
        """Initialize anchor data (simplified version with key viewpoints)"""
        print("Loading anchor data...")
        
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
                print(f"   Loaded {viewpoint} anchor ({np.sum(valid_mask)} points)")
    
    def process_video_for_calibration(self):
        """Process entire video and collect high-quality calibration frames"""
        print(f"Processing video: {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {total_frames} frames at {fps:.1f} FPS")
        
        frame_id = 0
        processed_frames = 0
        
        # Process every 5th frame to reduce computation
        frame_skip = 5
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_id % frame_skip == 0:
                if self.debug:
                    candidate = self._process_frame_for_calibration_debug(frame, frame_id)
                else:
                    candidate = self._process_frame_for_calibration(frame, frame_id)

                if candidate:
                    self.candidate_frames.append(candidate)
                    if not self.debug:
                        print(f"Frame {frame_id:4d}: VAPE={candidate.vape_quality.quality_score():.1f}, "
                              f"ChArUco={candidate.charuco_quality.quality_score():.1f}, "
                              f"Combined={candidate.combined_quality():.1f}")
                
                processed_frames += 1
                
                # Progress update
                if not self.debug and processed_frames % 20 == 0:
                    progress = (frame_id / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({len(self.candidate_frames)} candidates found)")
            
            frame_id += 1
        
        cap.release()
        
        print(f"\nFound {len(self.candidate_frames)} candidate frames")
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
        
        # 4. Quality thresholds for calibration - UPDATED for A1 board
        min_vape_quality = 15.0  
        min_charuco_quality = 25.0  # Higher threshold for larger, more accurate A1 board
        
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
        """Detect ChArUco with quality metrics - UPDATED for A1 board"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quality = QualityMetrics()

        # Step 1: Detect ArUco markers
        if self.detector_mode == "new":
            corners, ids, _ = self.aruco_det.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_det)
        
        if ids is None or len(ids) == 0:
            return False, None, None, quality

        # Step 2: Interpolate ChArUco corners (use static function for reliability)
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board)
        
        if charuco_corners is None or len(charuco_corners) < 4:
            return False, None, None, quality

        # Step 3: Update quality metrics
        quality.num_matches = len(ids)
        quality.num_inliers = len(charuco_corners)
        
        # Step 4: Estimate pose
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)
        pose_success = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, self.board, self.K, self.dist_coeffs, rvec, tvec
        )
        
        if not pose_success:
            return False, None, None, quality
        
        # Step 5: Calculate reprojection error
        projected_corners, _ = cv2.projectPoints(
            self.chessboard_corners[charuco_ids.flatten()],
            rvec, tvec, self.K, self.dist_coeffs)
        
        error = np.mean(np.linalg.norm(
            charuco_corners.reshape(-1, 2) - projected_corners.reshape(-1, 2), axis=1))
        
        quality.reprojection_error = error
        quality.confidence_score = min(100, len(charuco_corners) * 5)
        
        # Step 6: Quality thresholds for A1 board (larger markers = better accuracy)
        if error < 1.5 and len(charuco_corners) >= 10:
            return True, rvec, tvec, quality
        
        return False, None, None, quality

    # Rest of the methods continue as before...
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
                
                if len(matches) < 8:
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
                    reprojectionError=4,
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
                        quaternion = R.from_matrix(R_matrix).as_quat()
                        
                        best_pose = (position, quaternion)
                        best_quality = quality
                        
            except Exception as e:
                continue
        
        if best_pose and best_quality.quality_score() > 0:
            return True, best_pose[0], best_pose[1], best_quality
        
        return False, None, None, QualityMetrics()

    def _select_best_calibration_frames(self):
        """Select the best frames for calibration from candidates"""
        print("\nSelecting best calibration frames...")
        
        if len(self.candidate_frames) < 5:
            print(f"Not enough candidates ({len(self.candidate_frames)}). Need at least 5.")
            return
        
        # Sort by combined quality
        sorted_candidates = sorted(self.candidate_frames, key=lambda x: x.combined_quality(), reverse=True)
        
        # Select diverse, high-quality frames
        selected = []
        min_frame_gap = 30
        
        for candidate in sorted_candidates:
            too_close = any(abs(candidate.frame_id - sel.frame_id) < min_frame_gap for sel in selected)
            
            if not too_close and len(selected) < 15:
                selected.append(candidate)
        
        self.final_calibration_frames = selected
        
        print(f"Selected {len(self.final_calibration_frames)} frames for calibration:")
        for i, frame in enumerate(self.final_calibration_frames):
            print(f"   {i+1:2d}. Frame {frame.frame_id:4d}: Quality={frame.combined_quality():.1f}, "
                  f"VAPE_inliers={frame.vape_quality.num_inliers}, "
                  f"ChArUco_corners={frame.charuco_quality.num_inliers}")

    def _get_object_3d_model(self) -> np.ndarray:
        """Aggregates all unique 3D anchor points to form a point cloud model."""
        all_points = []
        for anchor in self.viewpoint_anchors.values():
            if 'map_3d' in anchor:
                all_points.extend(list(anchor['map_3d'].values()))
        
        if not all_points:
            return np.array([])
            
        return np.unique(np.array(all_points, dtype=np.float32), axis=0)

    def visualize_results(self, output_path: str = "calibration_visualization.mp4"):
        """
        Creates a video visualizing the calibration result by projecting the 3D object
        model onto the selected calibration frames.
        """
        if not self.final_calibration_frames:
            print("No calibration frames selected. Cannot create visualization.")
            return

        if not hasattr(self, 'calibration_result') or self.calibration_result is None:
            print("Calibration has not been solved. Cannot create visualization.")
            return

        print(f"\nCreating visualization video at: {output_path}")

        # Get calibration and 3D model
        R_to = self.calibration_result['R_to']
        t_to = self.calibration_result['t_to']
        object_3d_points = self._get_object_3d_model()

        if object_3d_points.shape[0] == 0:
            print("Could not build 3D object model from anchors. Aborting visualization.")
            return

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 2, (self.camera_width, self.camera_height))

        for calib_frame in sorted(self.final_calibration_frames, key=lambda f: f.frame_id):
            frame_image = calib_frame.frame.copy()
            
            # --- Get poses ---
            # ChArUco pose (Tag in camera frame)
            R_ct, _ = cv2.Rodrigues(calib_frame.charuco_rvec)
            t_ct = calib_frame.charuco_tvec

            # Original VAPE pose (Object in camera frame)
            R_co_vape = R.from_quat(calib_frame.vape_quaternion).as_matrix()
            rvec_co_vape, _ = cv2.Rodrigues(R_co_vape)
            t_co_vape = calib_frame.vape_position.reshape(3, 1)

            # New ChArUco-derived pose for the object
            R_co_new = R_ct @ R_to
            rvec_co_new, _ = cv2.Rodrigues(R_co_new)
            t_co_new = (R_ct @ t_to) + t_ct

            # --- Visualization ---
            # 1. Draw ChArUco board axes
            cv2.drawFrameAxes(frame_image, self.K, self.dist_coeffs, calib_frame.charuco_rvec, calib_frame.charuco_tvec, 0.1)

            # 2. Project model using ORIGINAL VAPE pose (draw in BLUE)
            vape_projected, _ = cv2.projectPoints(object_3d_points, rvec_co_vape, t_co_vape, self.K, self.dist_coeffs)
            if vape_projected is not None:
                for pt in vape_projected:
                    cv2.circle(frame_image, tuple(pt[0].astype(int)), 3, (255, 0, 0), -1) # BLUE

            # 3. Project model using NEW CHARUCO-DERIVED pose (draw in YELLOW)
            charuco_projected, _ = cv2.projectPoints(object_3d_points, rvec_co_new, t_co_new, self.K, self.dist_coeffs)
            if charuco_projected is not None:
                for pt in charuco_projected:
                    cv2.circle(frame_image, tuple(pt[0].astype(int)), 3, (0, 255, 255), -1) # YELLOW

            # 4. Add text overlay with legend
            cv2.putText(frame_image, f"Frame {calib_frame.frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
            cv2.putText(frame_image, f"Frame {calib_frame.frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
            
            cv2.putText(frame_image, "YELLOW: ChArUco-derived Pose", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
            cv2.putText(frame_image, "YELLOW: ChArUco-derived Pose", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.putText(frame_image, "BLUE: Original VAPE Pose", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
            cv2.putText(frame_image, "BLUE: Original VAPE Pose", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
            
            out.write(frame_image)

        out.release()
        print(f"Visualization complete.")

    def solve_calibration(self):
        """Solve for tag-to-object transform using frame-by-frame averaging"""
        if len(self.final_calibration_frames) < 7:
            print(f"Not enough good frames ({len(self.final_calibration_frames)}). Need at least 7.")
            return False
        
        print(f"\nSolving calibration by averaging transformations from {len(self.final_calibration_frames)} frames...")
        
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
            R_to_candidate = R_ct.T @ R_co
            t_to_candidate = R_ct.T @ (t_co - t_ct)

            # Convert rotation to axis-angle for stable averaging
            r_to_candidate, _ = cv2.Rodrigues(R_to_candidate)

            all_r_to.append(r_to_candidate.flatten())
            all_t_to.append(t_to_candidate.flatten())

        # Average using median for robustness
        final_r_to_vec = np.median(np.array(all_r_to), axis=0)
        final_t_to = np.median(np.array(all_t_to), axis=0).reshape(3, 1)
        final_R_to, _ = cv2.Rodrigues(final_r_to_vec)

        # Statistics
        r_std = np.std(np.array(all_r_to), axis=0)
        t_std = np.std(np.array(all_t_to), axis=0)
        print(f"Statistics (lower is better):")
        print(f"   Rotation Vector Std Dev: {r_std}")
        print(f"   Translation Std Dev (meters): {t_std}")
        if np.any(t_std > 0.05):
            print("   Warning: High variation in translation detected across frames.")
        
        calibration_result = {
            'R_to': final_R_to,
            't_to': final_t_to,
            'num_frames': len(self.final_calibration_frames)
        }
        
        self.calibration_result = calibration_result
        self._save_calibration(calibration_result)
        print("\nCalibration completed successfully using frame-by-frame averaging!")
        return True

    def _save_calibration(self, calibration):
        """Save calibration results"""
        os.makedirs('calibration', exist_ok=True)
        
        calibration_data = {
            'R_to': calibration['R_to'].tolist(),
            't_to': calibration['t_to'].tolist(),
            'num_frames': calibration['num_frames'],
            'board_config': {
                'cols': 7,
                'rows': 11, 
                'square_len_m': 0.0765,
                'marker_len_m': 0.0573,
                'dict_id': 'DICT_6X6_250'
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'video_source': self.video_path
        }
        
        filename = 'calibration/tag_to_object_ground_truth_A1.json'
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Calibration saved to {filename}")

    # # Utility methods
    # def _yolo_detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    #     """YOLO detection"""
    #     names = getattr(self.yolo_model, "names", {0: "iha"})
    #     inv = {v: k for k, v in names.items()}
    #     target_id = inv.get("iha", 0)

    #     for conf_thresh in (0.30, 0.20, 0.10):
    #         results = self.yolo_model(
    #             frame, imgsz=640, conf=conf_thresh, iou=0.5, 
    #             max_det=5, classes=[target_id], verbose=False
    #         )
    #         if not results or len(results[0].boxes) == 0:
    #             continue

    #         boxes = results[0].boxes
    #         best = max(boxes, key=lambda b: float((b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1])))
    #         x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().tolist()
    #         return int(x1), int(y1), int(x2), int(y2)

    #     return None

    # Utility methods
    def _yolo_detect(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """YOLO detection with fallback to full frame"""
        names = getattr(self.yolo_model, "names", {0: "iha"})
        inv = {v: k for k, v in names.items()}
        target_id = inv.get("iha", 0)

        for conf_thresh in (0.30, 0.20, 0.10):
            results = self.yolo_model(
                frame, imgsz=640, conf=conf_thresh, iou=0.5,
                max_det=5, classes=[target_id], verbose=False
            )
            if not results or len(results[0].boxes) == 0:
                continue

            boxes = results[0].boxes
            # choose the largest box (by area)
            best = max(
                boxes,
                key=lambda b: float((b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
            )
            x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().tolist()
            return int(x1), int(y1), int(x2), int(y2)

        # fallback: return whole frame if no detection
        h, w = frame.shape[:2]
        return 0, 0, w, h

    
    def _extract_features_sp(self, image_bgr: np.ndarray) -> Dict:
        """Extract SuperPoint features"""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            return self.extractor.extract(tensor)
        
    def _process_frame_for_calibration_debug(self, frame: np.ndarray, frame_id: int) -> Optional[CalibrationFrame]:
        """Debug version with detailed logging and relaxed thresholds"""
        
        print(f"\n--- DEBUG Frame {frame_id} ---")
        
        # 1. YOLO Detection
        bbox = self._yolo_detect(frame)
        if not bbox:
            print(f"  FAIL: No YOLO detection")
            return None
        print(f"  ✓ YOLO bbox: {bbox}")
        
        # 2. ChArUco Detection  
        charuco_success, charuco_rvec, charuco_tvec, charuco_quality = self._detect_charuco_with_quality_debug(frame)
        if not charuco_success:
            print(f"  FAIL: ChArUco detection failed")
            return None
        print(f"  ✓ ChArUco: corners={charuco_quality.num_inliers}, error={charuco_quality.reprojection_error:.2f}px, score={charuco_quality.quality_score():.1f}")
        
        # 3. VAPE Pose Estimation
        vape_success, vape_pos, vape_quat, vape_quality = self._estimate_vape_pose_with_quality(frame, bbox)
        if not vape_success:
            print(f"  FAIL: VAPE pose estimation failed")
            return None
        print(f"  ✓ VAPE: inliers={vape_quality.num_inliers}, error={vape_quality.reprojection_error:.2f}px, score={vape_quality.quality_score():.1f}")
        
        # 4. RELAXED quality thresholds for debugging
        min_vape_quality = 60#10.0      # Reduced from 15.0
        min_charuco_quality = 100#15.0   # Reduced from 25.0
        
        vape_ok = vape_quality.quality_score() >= min_vape_quality
        charuco_ok = charuco_quality.quality_score() >= min_charuco_quality
        
        print(f"  Quality check: VAPE={vape_ok} ({vape_quality.quality_score():.1f}>={min_vape_quality}), ChArUco={charuco_ok} ({charuco_quality.quality_score():.1f}>={min_charuco_quality})")
        
        if vape_ok and charuco_ok:
            print(f"  ✓ ACCEPT: Frame {frame_id} meets relaxed quality criteria")
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
        else:
            print(f"  REJECT: Quality thresholds not met")
            return None

    def _detect_charuco_with_quality_debug(self, frame: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, QualityMetrics]:
        """Debug version with relaxed thresholds and detailed logging"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quality = QualityMetrics()

        # Step 1: Detect ArUco markers
        if self.detector_mode == "new":
            corners, ids, _ = self.aruco_det.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_det)
        
        if ids is None or len(ids) == 0:
            print(f"    ChArUco: No ArUco markers detected")
            return False, None, None, quality
        
        print(f"    ChArUco: Detected {len(ids)} ArUco markers")

        # Step 2: Interpolate ChArUco corners
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board)
        
        if charuco_corners is None or len(charuco_corners) < 4:
            print(f"    ChArUco: Failed to interpolate corners (got {len(charuco_corners) if charuco_corners is not None else 0})")
            return False, None, None, quality
        
        print(f"    ChArUco: Interpolated {len(charuco_corners)} corners")

        # Step 3: Update quality metrics
        quality.num_matches = len(ids)
        quality.num_inliers = len(charuco_corners)
        
        # Step 4: Estimate pose
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)
        pose_success = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, self.board, self.K, self.dist_coeffs, rvec, tvec
        )
        
        if not pose_success:
            print(f"    ChArUco: Pose estimation failed")
            return False, None, None, quality
        
        print(f"    ChArUco: Pose estimation successful")
        
        # Step 5: Calculate reprojection error
        projected_corners, _ = cv2.projectPoints(
            self.chessboard_corners[charuco_ids.flatten()],
            rvec, tvec, self.K, self.dist_coeffs)
        
        error = np.mean(np.linalg.norm(
            charuco_corners.reshape(-1, 2) - projected_corners.reshape(-1, 2), axis=1))
        
        quality.reprojection_error = error
        quality.confidence_score = min(100, len(charuco_corners) * 5)
        
        print(f"    ChArUco: Reprojection error={error:.2f}px, corners={len(charuco_corners)}")
        
        # Step 6: RELAXED quality thresholds for debugging
        if error < 3.0 and len(charuco_corners) >= 6:  # Much more relaxed
            print(f"    ChArUco: PASS relaxed thresholds")
            return True, rvec, tvec, quality
        else:
            print(f"    ChArUco: FAIL relaxed thresholds (error={error:.2f}, corners={len(charuco_corners)})")
        
        return False, None, None, quality


def main():
    parser = argparse.ArgumentParser(description="VAPE MK52 ChArUco Ground Truth Calibrator - A1 Board")
    parser.add_argument('--video', help='Path to input MP4 video')
    parser.add_argument('--test-detection', action='store_true', help='Test ChArUco detection')
    parser.add_argument('--test-image', help='Path to test image for detection testing')
    parser.add_argument('--live-test', action='store_true', help='Use live camera for detection testing')
    parser.add_argument('--visualize', action='store_true', help='Create visualization video')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug logging for calibration processing')
    args = parser.parse_args()
    
    print("VAPE MK52 ChArUco Ground Truth Calibrator - A1 Board")
    print("=" * 60)
    
    try:
        # Initialize calibrator (always needed)
        calibrator = CharucoGroundTruthCalibrator("", debug=args.debug)
        
        if args.debug:
            print("*** DEBUG MODE ENABLED ***")

        # Test detection if requested
        if args.test_detection or args.test_image or args.live_test:
            if args.live_test:
                calibrator.test_board_detection(live_camera=True)
            elif args.test_image:
                calibrator.test_board_detection(test_image_path=args.test_image)
            else:
                print("Detection test mode. Use --test-image or --live-test")
            return
        
        # Run calibration
        if not args.video:
            print("Error: --video required for calibration")
            return
            
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
        
        calibrator.video_path = args.video
        calibrator.process_video_for_calibration()
        
        if calibrator.solve_calibration():
            print("\nCalibration completed successfully!")

            if args.visualize:
                calibrator.visualize_results()

            print("\nNext steps:")
            print("   1. Check calibration quality in the output above")
            if args.visualize:
                print("   2. Review 'calibration_visualization.mp4' to visually inspect accuracy.")
            print("   3. If quality is good, use the saved 'calibration/tag_to_object_ground_truth_A1.json'")
        else:
            print("\nCalibration failed. Try:")
            print("   - Test detection first: --test-detection --live-test")
            print("   - Check board visibility and quality in video")
            print("   - Ensure diverse camera viewpoints")
            
    except Exception as e:
        print(f"Error during calibration: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()
        