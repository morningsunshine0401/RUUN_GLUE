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

# Function to create a unique filename
def create_unique_filename(directory, base_filename):
    filename, ext = os.path.splitext(base_filename)
    counter = 1
    new_filename = base_filename

    # Continue incrementing the counter until we find a filename that doesn't exist
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{filename}_{counter}{ext}"
        counter += 1
    
    return new_filename

def load_ground_truth_poses(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    gt_poses = {}
    for frame in data['frames']:
        image_name = frame['image']
        for obj in frame['object_poses']:
            if obj['name'] == 'Camera':
                pose = np.array(obj['pose'], dtype=np.float32)
                gt_poses[image_name] = pose
                break  # Assume only one camera pose per frame
    return gt_poses

def rotation_matrix_to_axis_angle(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    return theta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue Pose Estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to an image directory')
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
    parser.add_argument(
        '--ground_truth', type=str, required=True,
        help='Path to the ground truth JSON file')

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

    # # Provided 2D and 3D keypoints for the anchor image
    # anchor_keypoints_2D = np.array([
    #     [342., 193.],
    #     [305., 200.],
    #     [483., 212.],
    #     [447., 218.],
    #     [426., 264.],
    #     [150., 277.],
    #     [ 87., 289.],
    #     [209., 315.],
    #     [293., 343.],
    #     [131., 349.],
    #     [516., 353.],
    #     [167., 360.],
    #     [459., 373.]
    # ], dtype=np.float32)

    # anchor_keypoints_3D = np.array([
    #     [-0.81972,  -0.325801,  5.28664 ],
    #     [-0.60385,  -0.3258,   5.28664 ],
    #     [-0.81972,   0.33811,   5.28079 ],
    #     [-0.60385,   0.33329,   5.28664 ],
    #     [-0.81972,   0.08616,   5.0684  ],
    #     [-0.29107,  -0.83895,   4.96934 ],
    #     [ 0.01734,  -0.83895,   4.9697  ],
    #     [ 0.26951,   0.0838,    5.0646  ],
    #     [-0.11298,   0.0838,    4.89972 ],
    #     [ 0.44813,  -0.0721,    4.96381 ],
    #     [-0.29152,   0.84644,   4.96934 ],
    #     [ 0.44813,   0.0838,    4.96381 ],
    #     [ 0.01759,   0.84644,   4.96699 ]
    # ], dtype=np.float32)

    # # Provided 2D and 3D keypoints for the anchor image
    # anchor_keypoints_2D = np.array([
    #     [272., 138.],
    #     [346., 239.],
    #     [274., 252.],
    #     [309., 272.],
    #     [344., 274.],
    #     [363., 301.],
    #     [291., 304.],
    #     [324., 311.],
    #     [311., 328.],
    #     [343., 336.],
    #     [275., 362.]
    # ], dtype=np.float32)

    # anchor_keypoints_3D = np.array([
    #     [ 0.   ,   0.   ,   5.75  ],
    #     [-0.25 , 0.25  ,  5.25  ],
    #     [ 0.25 ,   0.25,    5.25  ],
    #     [ 0.   ,  0.25 ,   5.1414],
    #     [ 0.    , 0.45 ,   5.1414],
    #     [-0.1414 ,0.45 ,   5.    ],
    #     [ 0.1414 , 0.25,    5.    ],
    #     [ 0.1414 , 0.45,    5.    ],
    #     [ 0.     , 0.25,    4.8586],
    #     [ 0.     , 0.45,    4.8586],
    #     [ 0.25   , 0.25,    4.75  ]
    # ], dtype=np.float32)

    # Provided 2D and 3D keypoints for the anchor image
    anchor_keypoints_2D = np.array([
        [545., 274.],
        [693., 479.],
        [401., 481.],
        [548., 508.],
        [624., 539.],
        [728., 600.],
        [582., 609.],
        [648., 623.],
        [623., 656.],
        [688., 671.],
        [550., 724.]
    ], dtype=np.float32)

    anchor_keypoints_3D = np.array([
        [ 0.  ,    0.  ,    5.75  ],
        [-0.25 ,   0.25,    5.25  ],
        [ 0.25 ,  -0.25,    5.25  ],
        [ 0.25 ,   0.25,    5.25  ],
        [ 0.   ,   0.25,    5.1414],
        [-0.1414,  0.45,    5.    ],
        [ 0.1414,  0.25,    5.    ],
        [ 0.1414,  0.45,    5.    ],
        [ 0.    ,  0.25,    4.8586],
        [ 0.    ,  0.45,    4.8586],
        [ 0.25  ,  0.25,    4.75  ]
    ], dtype=np.float32)

    # Build a KD-Tree of the SuperPoint keypoints
    sp_tree = cKDTree(anchor_keypoints_sp)

    # For each provided 2D keypoint, find the nearest SuperPoint keypoint
    distances, indices = sp_tree.query(anchor_keypoints_2D, k=1)

    # Set a distance threshold to accept matches (e.g., 1 pixels)
    distance_threshold = 1#0.5
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

    # Load ground truth poses
    gt_poses = load_ground_truth_poses(opt.ground_truth)

    # Read a sequence of images from the input directory
    input_images = sorted(list(Path(opt.input).glob('*.png')))
    assert len(input_images) > 0, f'No images found in the directory {opt.input}'

    print(f'Found {len(input_images)} images in directory {opt.input}')

    frame_idx = 0  # Initialize frame counter
    timer = AverageTimer()

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pose Estimation', 640 * 2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    for img_path in input_images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f'Error loading image: {img_path}')
            continue

        frame_name = img_path.name  # Get the image filename
        frame_idx += 1
        print(f'Processing frame {frame_idx}: {frame_name} with shape: {frame.shape}')
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
            'image0': anchor_tensor,
            'image1': frame_tensor,
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

        # Save the total number of matches for analysis
        total_matches = len(mkpts0)

        # Proceed only if there are enough matches
        if len(mkpts0) >= 4:
            # Camera intrinsic parameters (replace with your camera's parameters)
            #focal_length_x = 1333.33333  # px
            #focal_length_y = 1333.33333  # py
            #cx = 319.5  # Principal point u0
            #cy = 239.5  # Principal point v0

            #focal_length_x = 888.888888888888  # px
            #focal_length_y = 888.888888888888  # py


            focal_length_x = 1777.77777777777  # px
            focal_length_y = 1777.77777777777  # py
            cx = 319.5  # Principal point u0
            cy = 239.5  # Principal point v0

            # Distortion coefficients (if any)
            distCoeffs = np.array([0.2728747755008597, -0.25885103136641374, 0, 0], dtype=np.float32)

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
                #distCoeffs=distCoeffs,
                distCoeffs=None,
                reprojectionError=3.0,
                confidence=0.90,
                iterationsCount=1000,
                flags=cv2.SOLVEPNP_P3P
            )

            # Calculate inlier ratio
            if inliers is not None:
                num_inliers = len(inliers)
                inlier_ratio = num_inliers / total_matches
            else:
                num_inliers = 0
                inlier_ratio = 0

            if success and inliers is not None and len(inliers) >= 3:
                # Refine pose
                rvec, tvec = cv2.solvePnPRefineVVS(
                    objectPoints=objectPoints,
                    imagePoints=imagePoints,
                    cameraMatrix=K,
                    distCoeffs=distCoeffs,
                    rvec=rvec_o,
                    tvec=tvec_o
                )

                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                # Calculate camera position in world coordinates
                camera_position = -R.T @ tvec

                # Compute reprojection errors
                objectPoints_inliers = mpts3D[inliers.flatten()].reshape(-1, 1, 3)
                imagePoints_inliers = mkpts1[inliers.flatten()].reshape(-1, 1, 2)

                projected_points, _ = cv2.projectPoints(
                    objectPoints=objectPoints_inliers,
                    rvec=rvec,
                    tvec=tvec,
                    cameraMatrix=K,
                    distCoeffs=None
                )
                reprojection_errors = np.linalg.norm(imagePoints_inliers - projected_points, axis=2).flatten()
                mean_reprojection_error = np.mean(reprojection_errors)
                std_reprojection_error = np.std(reprojection_errors)

                # **Compare with Ground Truth**
                if frame_name in gt_poses:
                    gt_pose = gt_poses[frame_name]
                    gt_R = gt_pose[:3, :3]
                    gt_t = gt_pose[:3, 3]

                    # Compute rotation error
                    rotation_diff = R @ gt_R.T
                    rotation_error = rotation_matrix_to_axis_angle(rotation_diff)
                    # Compute translation error
                    translation_error = np.linalg.norm(tvec.flatten() - gt_t)

                else:
                    print(f'Ground truth pose not found for {frame_name}')
                    rotation_error = None
                    translation_error = None

                # Save pose data
                pose_data = {
                    'frame': frame_idx,
                    'image_name': frame_name,
                    'rotation_matrix': R.tolist(),
                    'translation_vector': tvec.flatten().tolist(),
                    'camera_position': camera_position.flatten().tolist(),
                    'num_inliers': num_inliers,
                    'total_matches': total_matches,
                    'inlier_ratio': inlier_ratio,
                    'reprojection_errors': reprojection_errors.tolist(),
                    'mean_reprojection_error': float(mean_reprojection_error),
                    'std_reprojection_error': float(std_reprojection_error),
                    'rotation_error_rad': float(rotation_error) if rotation_error is not None else None,
                    'translation_error': float(translation_error) if translation_error is not None else None,
                    'inliers': inliers.flatten().tolist(),
                    'mkpts0': mkpts0.tolist(),
                    'mkpts1': mkpts1.tolist(),
                    'mpts3D': mpts3D.tolist(),
                    'mconf': mconf.tolist(),
                }
                all_poses.append(pose_data)

                # Output the estimated pose
                print('Estimated Camera Pose:')
                print('Rotation Matrix:\n', R)
                print('Translation Vector:\n', tvec.flatten())
                print('Camera Position (World Coordinates):\n', camera_position.flatten())
                print(f'Rotation Error (rad): {rotation_error}')
                print(f'Translation Error: {translation_error}')

                # Visualization code (same as before)
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
                pose_data = {
                    'frame': frame_idx,
                    'image_name': frame_name,
                    'num_inliers': 0,
                    'total_matches': total_matches,
                    'inlier_ratio': 0,
                    'mean_reprojection_error': None,
                    'std_reprojection_error': None,
                    'rotation_error_rad': None,
                    'translation_error': None,
                }
                all_poses.append(pose_data)
                # Visualization code (if needed)
                if not opt.no_display:
                    cv2.imshow('Pose Estimation', frame)
                    if cv2.waitKey(1) == ord('q'):
                        break

        else:
            print('Not enough matches to compute pose.')
            pose_data = {
                'frame': frame_idx,
                'image_name': frame_name,
                'num_inliers': 0,
                'total_matches': total_matches,
                'inlier_ratio': 0,
                'mean_reprojection_error': None,
                'std_reprojection_error': None,
                'rotation_error_rad': None,
                'translation_error': None,
            }
            all_poses.append(pose_data)
            # Visualization code (if needed)
            if not opt.no_display:
                cv2.imshow('Pose Estimation', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        timer.update('viz')
        timer.print()

    cv2.destroyAllWindows()

    with open(opt.save_pose, 'w') as f:
        json.dump(all_poses, f, indent=4)
    print(f'Pose estimation saved to {opt.save_pose}')
