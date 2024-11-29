import torch
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
import matplotlib.cm as cm

import json
import os

# List to store all pose estimations
all_poses = []

from models.matching import Matching
from models.utils import (AverageTimer, make_matching_plot_fast)

torch.set_grad_enabled(False)

def frame2tensor(frame, device):
    if frame is None:
        raise ValueError('Received an empty frame.')
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        # Image is already grayscale
        gray = frame
    else:
        # Convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Normalize and convert to tensor
    tensor = torch.from_numpy(gray / 255.).float()[None, None].to(device)
    return tensor

# Function to create a unique filename (already defined)
def create_unique_filename(directory, base_filename):
    """
    Creates a unique filename in the given directory by appending a number if the file already exists.
    """
    filename, ext = os.path.splitext(base_filename)
    counter = 1
    new_filename = base_filename

    # Continue incrementing the counter until we find a filename that doesn't exist
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{filename}_{counter}{ext}"
        counter += 1
    
    return new_filename


def init_kalman_filter(kf, n_states, n_measurements, n_inputs, dt):
    """
    Initializes the Kalman Filter parameters.

    Args:
        kf (cv2.KalmanFilter): The Kalman Filter to initialize.
        dt (float): Time interval between measurements.
    """
    #kf.init(n_states, n_measurements, n_inputs, cv2.CV_64F)

    # Transition matrix (A)



    kf.transitionMatrix = np.eye(n_states, dtype=np.float64)
    for i in range(3):
        kf.transitionMatrix[i, i+3] = dt
        kf.transitionMatrix[i, i+6] = 0.5 * dt**2
        kf.transitionMatrix[i+3, i+6] = dt

    for i in range(9, 12):
        kf.transitionMatrix[i, i+3] = dt
        kf.transitionMatrix[i, i+6] = 0.5 * dt**2
        kf.transitionMatrix[i+3, i+6] = dt

    # Measurement matrix (H)
    kf.measurementMatrix = np.zeros((n_measurements, n_states), dtype=np.float64)
    kf.measurementMatrix[0, 0] = 1.0  # x
    kf.measurementMatrix[1, 1] = 1.0  # y
    kf.measurementMatrix[2, 2] = 1.0  # z
    kf.measurementMatrix[3, 9] = 1.0  # roll
    kf.measurementMatrix[4, 10] = 1.0  # pitch
    kf.measurementMatrix[5, 11] = 1.0  # yaw

    # Process noise covariance (Q)
    kf.processNoiseCov = np.eye(n_states, dtype=np.float64) * 1e-5

    # Measurement noise covariance (R)
    kf.measurementNoiseCov = np.eye(n_measurements, dtype=np.float64) * 1e-4

    # Error covariance matrix (P)
    kf.errorCovPost = np.eye(n_states, dtype=np.float64)

    # Initial state
    kf.statePost = np.zeros((n_states, 1), dtype=np.float64)


def rotation_matrix_to_euler_angles(R):
    """
    Converts a rotation matrix to Euler angles (roll, pitch, yaw).
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])  # Roll
        y = np.arctan2(-R[2, 0], sy)      # Pitch
        z = np.arctan2(R[1, 0], R[0, 0])  # Yaw
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])  # Roll
        y = np.arctan2(-R[2, 0], sy)       # Pitch
        z = 0                              # Yaw

    return np.array([x, y, z])

def euler_angles_to_rotation_matrix(theta):
    """
    Converts Euler angles (roll, pitch, yaw) to a rotation matrix.
    """
    roll, pitch, yaw = theta
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = R_z @ R_y @ R_x
    return R


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue Pose Estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to an image directory or movie file')
    parser.add_argument(
        '--anchor', type=str, required=True,
        help='Path to the anchor (reference) image')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference.')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by SuperPoint '
             '(\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius '
             '(Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--save_pose', type=str, default='pose_estimation_research.json',
        help='Path to save pose estimation results in JSON format')

    opt = parser.parse_args()
    print(opt)

    # Adjust resize options
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    # Check if the provided path is a directory, if so, handle the filename
    if os.path.isdir(opt.save_pose):
        # If the path is a directory, append a filename and ensure it's unique
        base_filename = 'pose_estimation.json'
        opt.save_pose = create_unique_filename(opt.save_pose, base_filename)
    else:
        # If the path is a file, ensure it's unique as well
        save_dir = os.path.dirname(opt.save_pose)
        base_filename = os.path.basename(opt.save_pose)
        opt.save_pose = create_unique_filename(save_dir, base_filename)

    # Set device
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    # Initialize the SuperGlue matching model
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints,
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # Load the anchor (reference) image
    anchor_image = cv2.imread(opt.anchor)
    assert anchor_image is not None, 'Failed to load anchor image at {}'.format(opt.anchor)

    # Resize the anchor image if needed
    if len(opt.resize) == 2:
        anchor_image = cv2.resize(anchor_image, tuple(opt.resize))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        h, w = anchor_image.shape[:2]
        scale = opt.resize[0] / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        anchor_image = cv2.resize(anchor_image, new_size)

    # Convert the anchor image to tensor and move to device
    anchor_tensor = frame2tensor(anchor_image, device)

    # Extract keypoints and descriptors from the anchor image using SuperPoint
    anchor_data = matching.superpoint({'image': anchor_tensor})
    anchor_keypoints_sp = anchor_data['keypoints'][0].cpu().numpy()  # Shape: (N, 2)
    anchor_descriptors_sp = anchor_data['descriptors'][0].cpu().numpy()  # Shape: (256, N)
    anchor_scores_sp = anchor_data['scores'][0].cpu().numpy()

    # Provided 2D and 3D keypoints for the anchor image
    anchor_keypoints_2D = np.array([
        [563, 565], 
        [77, 582], 
        [515, 318], 
        [606, 317], 
        [612, 411],
        [515, 414], 
        [420, 434], 
        [420, 465], 
        [618, 455], 
        [500, 123], 
        [418, 153], 
        [417, 204], 
        [417, 243], 
        [502, 279],
        [585, 240],  
        [289, 26],  
        [322, 339], 
        [349, 338], 
        [349, 374], 
        [321, 375],
        [390, 349], 
        [243, 462], 
        [367, 550], 
        [368, 595], 
        [383, 594],
        [386, 549], 
        [779, 518], 
        [783, 570]
        
    ], dtype=np.float32)

    anchor_keypoints_3D = np.array([
        [0.03, -0.165, 0.05],
        [-0.190, -0.165, 0.050],
        [0.010, -0.025, 0.0],
        [0.060, -0.025, 0.0],
        [0.06, -0.080, 0.0],
        [0.010, -0.080, 0.0],
        [-0.035, -0.087, 0.0],
        [-0.035, -0.105, 0.0],
        [0.065, -0.105, 0.0],
        [0.0, 0.045, 0.0],
        [-0.045, 0.078, 0.0],
        [-0.045, 0.046, 0.0],
        [-0.045, 0.023, 0.0],
        [0.0, -0.0, 0.0],
        [0.045, 0.022, 0.0],
        [-0.120, 0.160, 0.0],
        [-0.095, -0.035,0.0],
        [-0.080, -0.035, 0.0],
        [-0.080, -0.055, 0.0],
        [-0.095, -0.055, 0.0],
        [-0.050, -0.040, 0.010],
        [-0.135, -0.100, 0.0],
        [-0.060, -0.155, 0.050],
        [-0.060, -0.175, 0.050],
        [-0.052, -0.175, 0.050],
        [-0.052, -0.155, 0.050],
        [0.135, -0.147, 0.050],
        [0.135, -0.172, 0.050]

    ], dtype=np.float32)


    # # Provided 2D and 3D keypoints for the anchor image
    # anchor_keypoints_2D = np.array([
    #     [563, 565], 
    #     [77, 582], 
    #     [515, 318], 
         
    #     [612, 411],
    #     [515, 414],
    #     [420, 434], 
       
    #     [618, 455], 
    #     [500, 123], 
    #     [418, 153],
         
    #     [417, 243], 
       
    #     [585, 240],  
    #     [289, 26],  
    #     [322, 339], 
         
        
    #     [390, 349],

    #     [243, 462], 
    #     [367, 550], 
        
    #     [779, 518], 
    #     [783, 570]
        
    # ], dtype=np.float32)

    # anchor_keypoints_3D = np.array([
    #     [0.03, -0.165, 0.05],
    #     [-0.190, -0.165, 0.050],
    #     [0.010, -0.025, 0.0],
        
    #     [0.06, -0.080, 0.0],
    #     [0.010, -0.080, 0.0],
    #     [-0.035, -0.087, 0.0],
       
    #     [0.065, -0.105, 0.0],
    #     [0.0, 0.045, 0.0],
    #     [-0.045, 0.078, 0.0],
        
    #     [-0.045, 0.023, 0.0],
       
    #     [0.045, 0.022, 0.0],
    #     [-0.120, 0.160, 0.0],
    #     [-0.095, -0.035,0.0],
        
       
    #     [-0.050, -0.040, 0.010],

    #     [-0.135, -0.100, 0.0],
    #     [-0.060, -0.155, 0.050],
        
    #     [0.135, -0.147, 0.050],
    #     [0.135, -0.172, 0.050]

    # ], dtype=np.float32)





    # # Provided 2D and 3D keypoints for the anchor image
    # anchor_keypoints_2D = np.array([
    #     [494, 605],
    #     [566, 641], 
    #     [603, 557], 
    #     [539, 515], 
    #     [512, 345],
    #     [834, 491], 
    #     [927, 217], 
    #     [707, 44], 
    #     [752, 214], 
    #     [851, 173],
    #     [1069, 509],
    #     [1016, 639], 
    #     [413, 209], 
    #     [325, 298], 
    #     [743, 343],
    #     [541, 407], 
    #     [676, 382]
        
    # ], dtype=np.float32)

    # anchor_keypoints_3D = np.array([
    #     [0.054, -0.007, 0.008],
    #     [0.054, 0.007, 0.008],
    #     [0.038, 0.007, 0.015],
    #     [0.038, -0.007, 0.015],
    #     [0.005, -0.038, 0.008],
    #     [0.005, 0.038, 0.008],
    #     [-0.044, 0.034, 0.035],
    #     [-0.064, -0.034, 0.035],
    #     [-0.044, -0.007, 0.015],
    #     [-0.064, 0.007, 0.015],
    #     [-0.015, 0.080, 0.008],
    #     [0.015, 0.080, 0.008],
    #     [-0.015, -0.080, 0.008],
    #     [0.015, -0.008, 0.008],
    #     [-0.014, 0.007, 0.015],
    #     [0.015, 0.023, 0.008],
    #     [0.000, 0.000, 0.015]
        
    # ], dtype=np.float32)


  

    # Build a KD-Tree of the SuperPoint keypoints
    sp_tree = cKDTree(anchor_keypoints_sp)

    # For each provided 2D keypoint, find the nearest SuperPoint keypoint
    distances, indices = sp_tree.query(anchor_keypoints_2D, k=1)

    # Set a distance threshold to accept matches (e.g., 1 pixels)
    distance_threshold = 1
    valid_matches = distances < distance_threshold

    if not np.any(valid_matches):
        raise ValueError('No matching keypoints found within the distance threshold.')

    # Filter to keep only valid matches
    matched_anchor_indices = indices[valid_matches]
    matched_2D_keypoints = anchor_keypoints_2D[valid_matches]
    matched_3D_keypoints = anchor_keypoints_3D[valid_matches]

    # Get the descriptors for the matched keypoints
    matched_descriptors = anchor_descriptors_sp[:, matched_anchor_indices]
    # Get the keypoints
    matched_anchor_keypoints = anchor_keypoints_sp[matched_anchor_indices]
    # Get the scores
    matched_scores = anchor_scores_sp[matched_anchor_indices]

    # Initialize Kalman Filter parameters
    n_states = 18         # Number of state variables
    n_measurements = 6    # Number of measured variables
    n_inputs = 0          # Number of control variables
    frame_rate = 30       # Adjust this to your video frame rate
    dt = 1 / frame_rate   # Time interval between measurements
    kf = cv2.KalmanFilter(n_states, n_measurements, n_inputs, cv2.CV_64F)
    init_kalman_filter(kf, n_states, n_measurements, n_inputs, dt)
    # Minimum number of inliers to update the Kalman Filter
    min_inliers_kalman = 5

    # Initialize measurement vector
    measurement = np.zeros((n_measurements, 1), dtype=np.float64)

    # Open video file or camera
    cap = cv2.VideoCapture(opt.input)

    # Check if camera opened successfully
    if not cap.isOpened():
        print('Error when opening video file or camera (try different --input?)')
        exit(1)

    # Read the first frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print('Error when reading the first frame (try different --input?)')
        exit(1)

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pose Estimation', 640 * 2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    timer = AverageTimer()
    frame_idx = 0  # Initialize frame counter

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Finished processing video or invalid frame.')
            break
        frame_idx += 1
        print('Frame shape:', frame.shape)  # For debugging
        timer.update('data')

        # Resize the frame if needed
        if len(opt.resize) == 2:
            frame = cv2.resize(frame, tuple(opt.resize))
        elif len(opt.resize) == 1 and opt.resize[0] > 0:
            h, w = frame.shape[:2]
            scale = opt.resize[0] / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            frame = cv2.resize(frame, new_size)

        # Convert the current frame to tensor
        frame_tensor = frame2tensor(frame, device)

        # Extract keypoints and descriptors from the current frame using SuperPoint
        frame_data = matching.superpoint({'image': frame_tensor})
        frame_keypoints = frame_data['keypoints'][0].cpu().numpy()
        frame_descriptors = frame_data['descriptors'][0].cpu().numpy()
        frame_scores = frame_data['scores'][0].cpu().numpy()

        # Prepare data for SuperGlue matching
        input_superglue = {
            'keypoints0': torch.from_numpy(matched_anchor_keypoints).unsqueeze(0).to(device),
            'keypoints1': torch.from_numpy(frame_keypoints).unsqueeze(0).to(device),
            'descriptors0': torch.from_numpy(matched_descriptors).unsqueeze(0).to(device),
            'descriptors1': torch.from_numpy(frame_descriptors).unsqueeze(0).to(device),
            'scores0': torch.from_numpy(matched_scores).unsqueeze(0).to(device),
            'scores1': torch.from_numpy(frame_scores).unsqueeze(0).to(device),
            'image0': anchor_tensor,  # Added line
            'image1': frame_tensor,   # Added line
        }

        # Perform matching with SuperGlue
        pred = matching.superglue(input_superglue)
        timer.update('forward')

        # Retrieve matched keypoints
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

        # Valid matches (exclude unmatched keypoints)
        valid = matches > -1
        mkpts0 = matched_anchor_keypoints[valid]  # Matched keypoints in anchor image
        mkpts1 = frame_keypoints[matches[valid]]  # Matched keypoints in current frame
        mpts3D = matched_3D_keypoints[valid]      # Corresponding 3D points
        mconf = confidence[valid]

        # **Save the total number of matches for analysis**
        total_matches = len(mkpts0)

        print('matched nums\n',len(mkpts0))

        # Proceed only if there are enough matches
        if len(mkpts0) >= 4:
            # Define camera intrinsic parameters (adjust these values to your camera)
            #focal_length = 1410  # Replace with your camera's focal length in pixels
            #cx = frame.shape[1] / 2  # Principal point x-coordinate
            #cy = frame.shape[0] / 2  # Principal point y-coordinate
            
            #################################################################################################################################################################
            # # Real calibration values from XML Webcam!!!!!!!!!!!!!(perspectiveProjWithoutDistortion)
            # focal_length_x = 1526.22  # px
            # focal_length_y = 1531.18  # py
            # cx = 637.98  # Principal point u0
            # cy = 416.04  # Principal point v0

            # # Distortion coefficients from "perspectiveProjWithDistortion" model in the XML
            # distCoeffs = np.array([0.0319171567, 2.38832491, 0.0165411907, -0.00283550938, -1.07711172], dtype=np.float32)
            
            #################################################################################################################################################################
            # Real calibration values from Phone cam!!!!!!!!!!!!!(perspectiveProjWithoutDistortion)
            focal_length_x = 1079.83796  # px
            focal_length_y = 1081.11500  # py
            cx = 627.318141  # Principal point u0
            cy = 332.745740  # Principal point v0

            # Distortion coefficients from "perspectiveProjWithDistortion" model in the XML
            distCoeffs = np.array([0.0314, -0.2847, -0.0105, -0.00005, 1.0391], dtype=np.float32)


            #################################################################################################################################################################


            print(f"Principal point cx: {cx}")
            print(f"Principal point cy: {cy}")

            # Intrinsic camera matrix (K)
            K = np.array([
                [focal_length_x, 0, cx],
                [0, focal_length_y, cy],
                [0, 0, 1]
            ], dtype=np.float32)

            # Convert points to the required shape for solvePnPRansac
            objectPoints = mpts3D.reshape(-1, 1, 3)
            imagePoints = mkpts1.reshape(-1, 1, 2)

            

            # Use cv2.solvePnPRansac for robustness to outliers
            success, rvec_o, tvec_o, inliers = cv2.solvePnPRansac(
                objectPoints=objectPoints,
                imagePoints=imagePoints,
                cameraMatrix=K,
                #distCoeffs=None,
                distCoeffs=distCoeffs,
                reprojectionError=5,#2.0,
                confidence=0.9,#0.95,
                iterationsCount=1500,#500,
                flags=cv2.SOLVEPNP_P3P
            )


            

            # **Calculate inlier ratio**
            if inliers is not None:
                num_inliers = len(inliers)
                inlier_ratio = num_inliers / total_matches
            else:
                num_inliers = 0
                inlier_ratio = 0

            if success and inliers is not None and len(inliers)>=3:
                #####################################################################
                # Use cv2.solvePnPRansac for robustness to outliers
                '''

                rvec, tvec = cv2.solvePnPRefineLM(
                    objectPoints=objectPoints,
                    imagePoints=imagePoints,
                    cameraMatrix=K,
                    #distCoeffs=None,
                    distCoeffs=distCoeffs,
                    rvec=rvec_o,
                    tvec=tvec_o
                )'''

                # Use cv2.solvePnPRansac for robustness to outliers

                objectPoints_inliers = mpts3D[inliers.flatten()].reshape(-1, 1, 3)
                imagePoints_inliers = mkpts1[inliers.flatten()].reshape(-1, 1, 2)
                rvec, tvec = cv2.solvePnPRefineVVS(
                    objectPoints=objectPoints_inliers,
                    imagePoints=imagePoints_inliers,
                    cameraMatrix=K,
                    distCoeffs=distCoeffs,
                    #distCoeffs=distCoeffs,
                    rvec=rvec_o,
                    tvec=tvec_o
                )

                ######################################################################
                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                # Calculate camera position in world coordinates
                camera_position = -R.T @ tvec

                # Ensure objectPoints is reshaped to (N, 1, 3) for cv2.projectPoints
                #objectPoints = mpts3D[inliers].reshape(-1, 1, 3)
                #imagePoints = mkpts1.reshape(-1, 1, 2)

                # **Compute reprojection errors**
                projected_points, _ = cv2.projectPoints(
                    objectPoints=objectPoints_inliers,
                    rvec=rvec,
                    tvec=tvec,
                    cameraMatrix=K,
                    distCoeffs=distCoeffs
                )
                reprojection_errors = np.linalg.norm(imagePoints_inliers- projected_points, axis=2).flatten()
                mean_reprojection_error = np.mean(reprojection_errors)
                std_reprojection_error = np.std(reprojection_errors)

                # Define thresholds
                reprojection_error_threshold = 10  # Adjust based on your data
                max_translation_jump = 7.0         # Adjust based on expected motion (in meters)
                max_rotation_jump = 0.5            # Adjust based on expected rotation (in radians)

                # Kalman Filter Update
                if num_inliers >= min_inliers_kalman:
                    # Convert rotation matrix to Euler angles
                    eulers_measured = rotation_matrix_to_euler_angles(R)

                    # Prepare the measurement vector
                    measurement[0:3, 0] = tvec.flatten()
                    measurement[3:6, 0] = eulers_measured

                    # Predict the next state
                    predicted = kf.predict()

                    # Extract the estimated translation and orientation
                    translation_estimated = predicted[0:3].flatten()
                    eulers_estimated = predicted[9:12].flatten()

                    # Compute changes for temporal consistency
                    translation_change = np.linalg.norm(tvec.flatten() - translation_estimated)
                    rotation_change = np.linalg.norm(eulers_measured - eulers_estimated)

                    # Apply the combined approach
                    if mean_reprojection_error < reprojection_error_threshold:
                        if translation_change < max_translation_jump :#and rotation_change < max_rotation_jump:
                            # Update the Kalman Filter with the measurement
                            kf.correct(measurement)
                        else:
                            print("Skipping Kalman update due to large jump in translation/rotation.")
                    else:
                        print("Skipping Kalman update due to high reprojection error.")

                    # Use the estimated state from Kalman filter for visualization and saving
                    translation_estimated = predicted[0:3].flatten()
                    eulers_estimated = predicted[9:12].flatten()
                    R_estimated = euler_angles_to_rotation_matrix(eulers_estimated)
                    t_estimated = translation_estimated.reshape(3, 1)

                else:
                    print('Not enough inliers for Kalman filter update.')
                    # Predict the next state without correction
                    predicted = kf.predict()
                    translation_estimated = predicted[0:3].flatten()
                    eulers_estimated = predicted[9:12].flatten()
                    R_estimated = euler_angles_to_rotation_matrix(eulers_estimated)
                    t_estimated = translation_estimated.reshape(3, 1)


                # **Save inlier and reprojection error data**
                pose_data = {
                    'frame': frame_idx,  # Current frame index
                    'rotation_matrix': R.tolist(),
                    'translation_vector': tvec.flatten().tolist(),
                    'camera_position': camera_position.flatten().tolist(),
                    'num_inliers': num_inliers,
                    'total_matches': total_matches,
                    'inlier_ratio': inlier_ratio,
                    'reprojection_errors': reprojection_errors.tolist(),
                    'mean_reprojection_error': float(mean_reprojection_error),
                    'std_reprojection_error': float(std_reprojection_error),
                    'inliers': inliers.flatten().tolist(),
                    'mkpts0': mkpts0.tolist(),  # Matched keypoints in anchor image
                    'mkpts1': mkpts1.tolist(),  # Matched keypoints in current frame
                    'mpts3D': mpts3D.tolist(),  # Corresponding 3D points
                    'mconf': mconf.tolist(),    # Matching confidence scores
                    # Kalman Filter refined outputs
                    'kf_translation_vector': translation_estimated.tolist(),  # Refined translation
                    'kf_rotation_matrix': R_estimated.tolist(),  # Refined rotation matrix
                    'kf_euler_angles': eulers_estimated.tolist()  # Refined rotation in Euler angles
                    
                }
                all_poses.append(pose_data)

                # Output the estimated pose
                print('Estimated Camera Pose:')
                print('Rotation Matrix:\n', R)
                print('Translation Vector:\n', tvec.flatten())
                print('Camera Position (World Coordinates):\n', camera_position.flatten())

                # Convert images to grayscale for visualization
                # For anchor_image
                if len(anchor_image.shape) == 2 or anchor_image.shape[2] == 1:
                    anchor_image_gray = anchor_image
                else:
                    anchor_image_gray = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)

                # For frame
                if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                    frame_gray = frame
                else:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Draw the inlier matches
                inlier_idx = inliers.flatten()
                inlier_mkpts0 = mkpts0[inlier_idx]
                inlier_mkpts1 = mkpts1[inlier_idx]
                inlier_conf = mconf[inlier_idx]
                color = cm.jet(inlier_conf)

                # Visualize matches
                out = make_matching_plot_fast(
                    anchor_image_gray,         # Grayscale anchor image
                    frame_gray,                # Grayscale current frame
                    matched_anchor_keypoints,  # kpts0
                    frame_keypoints,           # kpts1
                    inlier_mkpts0,             # mkpts0
                    inlier_mkpts1,             # mkpts1
                    color,                     # color
                    text=[],                   # text
                    path=None,
                    show_keypoints=opt.show_keypoints,
                    small_text=[])

                # Overlay pose information on the frame
                position_text = f'Position: {camera_position.flatten()}'
                cv2.putText(out, position_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2, cv2.LINE_AA)

                if not opt.no_display:
                    cv2.imshow('Pose Estimation', out)
                    if cv2.waitKey(1) == ord('q'):
                        break

                # Save the output frame if needed
                if opt.output_dir is not None:
                    out_file = str(Path(opt.output_dir, f'frame_{frame_idx:06d}.png'))
                    cv2.imwrite(out_file, out)
            else:
                print('PnP pose estimation failed.')
                # **Even if pose estimation fails, save data for analysis**
                pose_data = {
                    'frame': frame_idx,
                    'num_inliers': 0,
                    'total_matches': total_matches,
                    'inlier_ratio': 0,
                    'mean_reprojection_error': None,
                    'std_reprojection_error': None,
                    # Additional data can be added if needed
                }
                all_poses.append(pose_data)
                if not opt.no_display:
                    cv2.imshow('Pose Estimation', frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
        else:
            print('Not enough matches to compute pose.')
            # **Save data when there are not enough matches**
            pose_data = {
                'frame': frame_idx,
                'num_inliers': 0,
                'total_matches': total_matches,
                'inlier_ratio': 0,
                'mean_reprojection_error': None,
                'std_reprojection_error': None,
                # Additional data can be added if needed
            }
            all_poses.append(pose_data)
            if not opt.no_display:
                cv2.imshow('Pose Estimation', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        timer.update('viz')
        timer.print()

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()

    with open(opt.save_pose, 'w') as f:
        json.dump(all_poses, f, indent=4)
    print(f'Pose estimation saved to {opt.save_pose}')
