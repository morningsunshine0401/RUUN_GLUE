#!/usr/bin/env python3
"""
Camera Intrinsic Calibration from a Video File using an A1 ChArUco Board.

This script calibrates a camera by processing a pre-recorded video file.
It automatically selects suitable frames based on a skipping interval to ensure
view diversity and processing efficiency.

It is configured for the A1 ChArUco board with the following specs:
- Grid: 7 columns x 11 rows
- Square Size: 76.5mm (0.0765m)
- Marker Size: 57.3mm (0.0573m)
- Dictionary: DICT_6X6_250

How to Use:
1. Run the script from your terminal with the path to your video:
   python calibrate_from_video.py --video /path/to/your_calibration_video.mp4

2. A window will appear showing the video being processed. Detected corners
   will be highlighted in green.

3. The script will automatically collect data from frames at the specified
   interval (e.g., every 15 frames by default).

4. Let the script run through the entire video, or press 'q' to stop early.

5. Once processing is complete, the script will perform the calibration,
   print the results, and save them to 'camera_calibration.npz'.
"""

import cv2
import numpy as np
import os
import argparse

# --- ChArUco Board Configuration (from your ChAruco3.py) ---
CHARUCO_COLS = 7
CHARUCO_ROWS = 11
SQUARE_LENGTH_M = 0.0765  # 76.5mm
MARKER_LENGTH_M = 0.0573  # 57.3mm
ARUCO_DICT_ID = cv2.aruco.DICT_6X6_250
MIN_DETECTED_CORNERS = 10 # Minimum number of corners to accept a frame

# --- Function to create the board (handles OpenCV version differences) ---
def create_charuco_board():
    """Creates the ChArUco board object based on the defined constants."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    
    if hasattr(cv2.aruco, 'CharucoBoard'): # OpenCV >= 4.7
        board = cv2.aruco.CharucoBoard(
            (CHARUCO_COLS, CHARUCO_ROWS), SQUARE_LENGTH_M, MARKER_LENGTH_M, aruco_dict
        )
    else: # Older OpenCV
        board = cv2.aruco.CharucoBoard_create(
            CHARUCO_COLS, CHARUCO_ROWS, SQUARE_LENGTH_M, MARKER_LENGTH_M, aruco_dict
        )
    return aruco_dict, board

def main(args):
    """Main function to run the camera calibration process from a video."""
    
    # --- Initialize ChArUco Board and Detector ---
    print("Initializing A1 ChArUco board and detector...")
    aruco_dict, board = create_charuco_board()

    try:
        aruco_params = cv2.aruco.DetectorParameters()
    except AttributeError:
        aruco_params = cv2.aruco.DetectorParameters_create()

    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_params.adaptiveThreshWinSizeMin = 5
    aruco_params.adaptiveThreshWinSizeMax = 25
    aruco_params.adaptiveThreshWinSizeStep = 4
    aruco_params.minMarkerPerimeterRate = 0.03

    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    else:
        detector = (aruco_dict, aruco_params)

    print("Initialization complete.")

    # --- Start Video Processing ---
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at '{args.video}'")
        return

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {args.video} ({total_frames} frames)")

    # --- Data Collection Loop ---
    all_charuco_corners = []
    all_charuco_ids = []
    frame_size = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Skip frames to ensure diversity and reduce processing time
        if frame_count % args.skip_frames != 0:
            continue
        
        if frame_size is None:
            frame_size = (frame.shape[1], frame.shape[0])
            print(f"Frame resolution detected: {frame_size[0]}x{frame_size[1]}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        if isinstance(detector, tuple): # Old API
            corners, ids, _ = cv2.aruco.detectMarkers(gray, detector[0], parameters=detector[1])
        else: # New API
            corners, ids, _ = detector.detectMarkers(gray)

        display_frame = frame.copy()

        # If markers are found, interpolate and try to capture
        if ids is not None and len(ids) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )

            # If enough corners are found, accept the frame for calibration
            if ret and charuco_corners is not None and len(charuco_corners) >= MIN_DETECTED_CORNERS:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                # Draw detected corners in green
                cv2.aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids, (0, 255, 0))

        # --- UI and Display ---
        progress = f"Frame: {frame_count}/{total_frames}"
        captures = f"Good Frames Found: {len(all_charuco_corners)}"
        cv2.putText(display_frame, progress, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display_frame, captures, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Video Processing for Calibration', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing stopped by user.")
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # --- Perform Calibration ---
    if len(all_charuco_corners) < 15:
        print(f"\n❌ Calibration failed: Only {len(all_charuco_corners)} suitable frames found. At least 15 are recommended.")
        print("Try reducing the '--skip_frames' value or using a video with clearer board views.")
        return

    print(f"\n✅ Processing complete. Calibrating using {len(all_charuco_corners)} selected frames...")
    
    try:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, board, frame_size, None, None
        )

        if not ret:
            print("Calibration failed. Could not compute parameters.")
            return

        # --- Print and Save Results ---
        print("\n✅ Calibration Successful!")
        print("-" * 30)
        print(f"Reprojection Error: {ret:.4f} pixels")
        print(" (A lower error, typically < 1.0, indicates a better calibration)")
        print("-" * 30)
        
        print("Camera Matrix (K):")
        print(camera_matrix)
        print("\nDistortion Coefficients (k1, k2, p1, p2, k3):")
        print(dist_coeffs.ravel())
        print("-" * 30)

        np.savez(
            args.output_file,
            camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, reprojection_error=ret
        )
        print(f"Calibration data saved to '{args.output_file}'")

    except Exception as e:
        print(f"An error occurred during calibration: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from a video file.")
    parser.add_argument("-v", "--video", type=str, required=True,
                        help="Path to the input video file for calibration.")
    parser.add_argument("-o", "--output_file", type=str, default="camera_calibration.npz",
                        help="Path to save the output calibration data (.npz file).")
    parser.add_argument("-s", "--skip_frames", type=int, default=15,
                        help="Number of frames to skip between captures. Lower for shorter videos.")
    args = parser.parse_args()
    main(args)