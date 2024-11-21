import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.cm as cm

def frame2gray(frame):
    if frame is None:
        raise ValueError('Received an empty frame.')
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        # Image is already grayscale
        gray = frame
    else:
        # Convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ORB Pose Estimation',
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
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen')

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

    # Convert the anchor image to grayscale
    anchor_gray = frame2gray(anchor_image)

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=5000)

    # Detect keypoints and compute descriptors in the anchor image using ORB
    anchor_keypoints, anchor_descriptors = orb.detectAndCompute(anchor_gray, None)

    # Convert keypoints to numpy array of (x, y) coordinates
    anchor_keypoints_np = np.array([kp.pt for kp in anchor_keypoints], dtype=np.float32)

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

    # Build a KD-Tree of the ORB keypoints
    orb_tree = cKDTree(anchor_keypoints_np)

    # For each provided 2D keypoint, find the nearest ORB keypoint
    distances, indices = orb_tree.query(anchor_keypoints_2D, k=1)

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
    matched_descriptors = anchor_descriptors[matched_anchor_indices]
    # Get the keypoints
    matched_anchor_keypoints = [anchor_keypoints[i] for i in matched_anchor_indices]

    # Initialize video capture
    # If input is '0', '1', etc., treat as camera ID
    try:
        cam_id = int(opt.input)
        cap = cv2.VideoCapture(cam_id)
    except ValueError:
        # Otherwise, treat as video file
        cap = cv2.VideoCapture(opt.input)

    if not cap.isOpened():
        print('Error when opening video stream (try different --input?)')
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

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Finished processing video or invalid frame.')
            break
        frame_count += 1
        print('Frame shape:', frame.shape)  # For debugging

        # Resize the frame if needed
        if len(opt.resize) == 2:
            frame = cv2.resize(frame, tuple(opt.resize))
        elif len(opt.resize) == 1 and opt.resize[0] > 0:
            h, w = frame.shape[:2]
            scale = opt.resize[0] / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            frame = cv2.resize(frame, new_size)

        # Convert the current frame to grayscale
        frame_gray = frame2gray(frame)

        # Detect keypoints and compute descriptors in the current frame using ORB
        frame_keypoints, frame_descriptors = orb.detectAndCompute(frame_gray, None)

        if frame_descriptors is None or len(frame_descriptors) == 0:
            print('No descriptors found in frame.')
            continue

        # Create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors between anchor image and current frame
        matches = bf.match(matched_descriptors, frame_descriptors)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Proceed only if there are enough matches
        if len(matches) >= 4:
            # Get matched keypoints in anchor image and current frame
            mkpts0 = np.array([matched_anchor_keypoints[m.queryIdx].pt for m in matches])
            mkpts1 = np.array([frame_keypoints[m.trainIdx].pt for m in matches])
            mpts3D = np.array([matched_3D_keypoints[m.queryIdx] for m in matches])  # Corresponding 3D points

            # Convert points to the required shape for solvePnPRansac
            objectPoints = mpts3D.reshape(-1, 1, 3)
            imagePoints = mkpts1.reshape(-1, 1, 2)

            # Define camera intrinsic parameters (adjust these values to your camera)
            focal_length = 1410  # Replace with your camera's focal length in pixels
            cx = frame.shape[1] / 2  # Principal point x-coordinate
            cy = frame.shape[0] / 2  # Principal point y-coordinate

            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)

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

                # Draw the inlier matches
                inlier_matches = [matches[i[0]] for i in inliers]
                out = cv2.drawMatches(
                    anchor_image, matched_anchor_keypoints,
                    frame, frame_keypoints,
                    inlier_matches, None,
                    matchColor=(0, 255, 0), singlePointColor=None,
                    matchesMask=None, flags=2)

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
                    out_file = str(Path(opt.output_dir, f'frame_{frame_count - 1:06d}.png'))
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

    cap.release()
    cv2.destroyAllWindows()
