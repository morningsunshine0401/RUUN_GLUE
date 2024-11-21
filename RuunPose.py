import torch
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
import matplotlib.cm as cm

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast)

torch.set_grad_enabled(False)

import pyrealsense2 as rs 

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue Pose Estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
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
    # Get the keypoints
    matched_anchor_keypoints = anchor_keypoints_sp[matched_anchor_indices]
    # Get the scores
    matched_scores = anchor_scores_sp[matched_anchor_indices]

    # Initialize VideoStreamer
    vs = VideoStreamer(opt.input, opt.resize, skip=1, image_glob=['*.png', '*.jpg', '*.jpeg'], max_length=1000000)
    frame, ret = vs.next_frame()
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

    while True:
        frame, ret = vs.next_frame()
        if not ret or frame is None:
            print('Finished processing video or invalid frame.')
            break
        print('Frame shape:', frame.shape)  # For debugging
        timer.update('data')

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

        # Proceed only if there are enough matches
        if len(mkpts0) >= 4:
            # Define camera intrinsic parameters (adjust these values to your camera)
            focal_length = 1410  # Replace with your camera's focal length in pixels
            cx = frame.shape[1] / 2  # Principal point x-coordinate
            cy = frame.shape[0] / 2  # Principal point y-coordinate
            print(cx)
            print(cy)

            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)

            
            # Convert points to the required shape for solvePnPRansac
            objectPoints = mpts3D.reshape(-1, 1, 3)
            imagePoints = mkpts1.reshape(-1, 1, 2)

            # Use cv2.solvePnPRansac for robustness to outliers
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=objectPoints,
                imagePoints=imagePoints,
                cameraMatrix=K,
                distCoeffs=None,
                reprojectionError=8.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            '''
            # Convert points to the required shape for solvePnPRansac
            objectPoints = mpts3D.reshape(-1, 1, 3)
            imagePoints = mkpts1.reshape(-1, 1, 2)
            # Use cv2.solvePnPRansac for robustness to outliers
            success, rvec, tvec, inliers = cv2.solvePnP(
                objectPoints=objectPoints,
                imagePoints=imagePoints,
                cameraMatrix=K,
                distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            '''

            if success and inliers is not None:
                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                # Calculate camera position in world coordinates
                camera_position = -R.T @ tvec

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
                    out_file = str(Path(opt.output_dir, f'frame_{vs.i - 1:06d}.png'))
                    cv2.imwrite(out_file, out)
            else:
                print('PnP pose estimation failed.')
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

    cv2.destroyAllWindows()
    vs.cleanup()
