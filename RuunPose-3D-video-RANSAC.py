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
        '--save_pose', type=str, default='pose_estimation.json',
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
        [385., 152.],
        [167., 153.],
        [195., 259.],
        [407., 159.],
        [127., 268.],
        [438., 168.],
        [206., 272.],
        [300., 124.],
        [343., 166.],
        [501., 278.],
        [444., 318.],
        [474., 150.],
        [337., 108.],
        [103., 173.],
        [389., 174.],
        [165., 241.],
        [163., 284.]
    ], dtype=np.float32)

    anchor_keypoints_3D = np.array([
        [-0.065, -0.007, 0.02],
        [-0.015, -0.077, 0.01],
        [0.04, 0.007, 0.02],
        [-0.045, 0.007, 0.02],
        [0.055, -0.007, 0.01],
        [-0.045, 0.025, 0.035],
        [0.04, 0.007, 0.0],
        [-0.045, -0.025, 0.035],
        [-0.045, -0.007, 0.02],
        [-0.015, 0.077, 0.01],
        [0.015, 0.077, 0.01],
        [-0.065, 0.025, 0.035],
        [-0.065, -0.025, 0.035],
        [0.015, -0.077, 0.01],
        [-0.045, 0.007, 0.02],
        [0.04, -0.007, 0.02],
        [0.055, 0.007, 0.01]
    ], dtype=np.float32)

    # Build a KD-Tree of the SuperPoint keypoints
    sp_tree = cKDTree(anchor_keypoints_sp)

    # For each provided 2D keypoint, find the nearest SuperPoint keypoint
    distances, indices = sp_tree.query(anchor_keypoints_2D, k=1)

    # Set a distance threshold to accept matches (e.g., 5 pixels)
    distance_threshold = 5.0
    valid_matches = distances < distance_threshold

    if not np.any(valid_matches):
        raise ValueError('No matching keypoints found within the distance threshold.')

    # Filter to keep only valid matches
    matched_anchor_indices = indices[valid_matches]
    matched_2D_keypoints = anchor_keypoints_2D[valid_matches]
    matched_3D_keypoints = anchor_keypoints_3D[valid_matches]

    # Get the descriptors for the matched keypoints
    matched_descriptors = anchor_descriptors_sp[:, matched_anchor_indices]
    matched_anchor_keypoints = anchor_keypoints_sp[matched_anchor_indices]
    matched_scores = anchor_scores_sp[matched_anchor_indices]

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

    # Create a window to display the demo
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

        # Refine matches using epipolar geometry
        if len(mkpts0) >= 4:
            # Define camera intrinsic parameters (adjust these values to your camera)
            focal_length = 1410  # Replace with your camera's focal length in pixels
            cx = frame.shape[1] / 2  # Principal point x-coordinate (center of the image)
            cy = frame.shape[0] / 2  # Principal point y-coordinate (center of the image)
            
            # Camera intrinsic matrix K
            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            # Step 1: Calculate the Fundamental or Essential Matrix using RANSAC
            E, inliers = cv2.findEssentialMat(
                mkpts0, mkpts1, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            # Check if the essential matrix computation was successful
            if E is None or inliers is None:
                print("Error: Essential matrix or inliers could not be computed.")
                continue  # Skip to the next frame if no valid matches are found
            # inliers is a mask that marks the inlier matches
            inliers = inliers.ravel().astype(bool)  # Convert to a boolean mask

            # Step 2: Filter the keypoints and 3D points based on the inliers
            inlier_mkpts0 = mkpts0[inliers]  # Inliers in the anchor image
            inlier_mkpts1 = mkpts1[inliers]  # Inliers in the current frame
            inlier_mpts3D = mpts3D[inliers]  # Corresponding 3D points

            if len(inlier_mkpts0) >= 4:
                # Proceed with solvePnPRansac using only the inliers
                objectPoints = inlier_mpts3D.reshape(-1, 1, 3)
                imagePoints = inlier_mkpts1.reshape(-1, 1, 2)

                # Use cv2.solvePnPRansac for robustness to outliers
                success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                    objectPoints=objectPoints,
                    imagePoints=imagePoints,
                    cameraMatrix=K,
                    distCoeffs=None,
                    reprojectionError=2.0,  # Lower reprojection error for accuracy
                    confidence=0.99,
                    iterationsCount=2000,  # Increase iterations for robustness
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success and inliers_pnp is not None:
                    # Convert rotation vector to rotation matrix
                    R, _ = cv2.Rodrigues(rvec)
                    # Calculate camera position in world coordinates
                    camera_position = -R.T @ tvec

                    # Append the estimated pose for this frame to the list
                    pose_data = {
                        'frame': frame_idx,  # Current frame index
                        'rotation_matrix': R.tolist(),
                        'translation_vector': tvec.flatten().tolist(),
                        'camera_position': camera_position.flatten().tolist()
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

                    # Visualize matches and camera pose information
                    inlier_idx = inliers_pnp.flatten()
                    inlier_mkpts0 = inlier_mkpts0[inlier_idx]
                    inlier_mkpts1 = inlier_mkpts1[inlier_idx]
                    inlier_conf = mconf[inliers_pnp.flatten()]
                    color = cm.jet(inlier_conf)

                    # Visualize matches
                    out = make_matching_plot_fast(
                        anchor_image_gray,         # Grayscale anchor image
                        frame_gray,                # Grayscale current frame
                        matched_anchor_keypoints,
                        frame_keypoints,
                        inlier_mkpts0,        # mkpts0 (inlier keypoints from anchor image)
                        inlier_mkpts1,        # mkpts1 (inlier keypoints from current frame)
                        color,                # color (colors for each matched keypoint pair)
                        text=[],              # Additional text, optional
                        path=None,            # Save path, if needed
                        show_keypoints=opt.show_keypoints,  # Option to display keypoints
                        small_text=[]
                    )

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
                    if not opt.no_display:
                        cv2.imshow('Pose Estimation', frame)
                        if cv2.waitKey(1) == ord('q'):
                            break
            else:
                print('Not enough inliers to compute pose.')
                if not opt.no_display:
                    cv2.imshow('Pose Estimation', frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
        else:
            print('Not enough matches to compute pose.')
            if not opt.no_display:
                cv2.imshow('Pose Estimation', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        timer.update('viz')
        timer.print()

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()

    # Save the estimated poses to the JSON file
    with open(opt.save_pose, 'w') as f:
        json.dump(all_poses, f, indent=4)
    print(f'Pose estimation saved to {opt.save_pose}')

