import torch
import argparse
from pathlib import Path
import cv2
import numpy as np
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

# -------------------- Add PointTracker Class --------------------

import numpy as np

class PointTracker(object):
    """Class to manage a fixed memory of points and descriptors that enables
    sparse optical flow point tracking.

    Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
    tracks with maximum length L, where each row corresponds to:
    row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
    """

    def __init__(self, max_length, nn_thresh):
        if max_length < 2:
            raise ValueError('max_length must be greater than or equal to 2.')
        self.maxl = max_length
        self.nn_thresh = nn_thresh
        self.all_pts = []
        for n in range(self.maxl):
            self.all_pts.append(np.zeros((2, 0)))
        self.last_desc = None
        self.tracks = np.zeros((0, self.maxl+2))
        self.track_count = 0
        self.max_score = 9999

    def nn_match_two_way(self, desc1, desc2, nn_thresh):
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.

        Inputs:
          desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
          desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
          nn_thresh - Optional descriptor distance below which is a good match.

        Returns:
          matches - 3xL numpy array, of L matches, where L <= N and each column i is
                    a match of two descriptors, d_i in image 1 and d_j' in image 2:
                    [d_i index, d_j' index, match_score]^T
        """
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        if nn_thresh < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')
        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        # Populate the final 3xN match data structure.
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores
        return matches

    def get_offsets(self):
        """Iterate through list of points and accumulate an offset value. Used to
        index the global point IDs into the list of points.

        Returns
          offsets - N length array with integer offset locations.
        """
        # Compute id offsets.
        offsets = []
        offsets.append(0)
        for i in range(len(self.all_pts)-1):  # Skip last camera size, not needed.
            offsets.append(self.all_pts[i].shape[1])
        offsets = np.array(offsets)
        offsets = np.cumsum(offsets)
        return offsets

    def update(self, pts, desc):
        """Add a new set of point and descriptor observations to the tracker.

        Inputs
          pts - 3xN numpy array of 2D point observations.
          desc - DxN numpy array of corresponding D dimensional descriptors.
        """
        if pts is None or desc is None:
            print('PointTracker: Warning, no points were added to tracker.')
            return
        assert pts.shape[1] == desc.shape[1]
        # Initialize last_desc.
        if self.last_desc is None:
            self.last_desc = np.zeros((desc.shape[0], 0))
        # Remove oldest points, store its size to update ids later.
        remove_size = self.all_pts[0].shape[1]
        self.all_pts.pop(0)
        self.all_pts.append(pts[:2, :])
        # Remove oldest point in track.
        self.tracks = np.delete(self.tracks, 2, axis=1)
        # Update track offsets.
        for i in range(2, self.tracks.shape[1]):
            self.tracks[:, i] -= remove_size
        self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
        offsets = self.get_offsets()
        # Add a new -1 column.
        self.tracks = np.hstack((self.tracks, -1*np.ones((self.tracks.shape[0], 1))))
        # Try to append to existing tracks.
        matched = np.zeros((pts.shape[1])).astype(bool)
        matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
        for match in matches.T:
            # Add a new point to its matched track.
            id1 = int(match[0]) + offsets[-2]
            id2 = int(match[1]) + offsets[-1]
            found = np.argwhere(self.tracks[:, -2] == id1)
            if found.shape[0] > 0:
                matched[int(match[1])] = True
                row = int(found)
                self.tracks[row, -1] = id2
                if self.tracks[row, 1] == self.max_score:
                    # Initialize track score.
                    self.tracks[row, 1] = match[2]
                else:
                    # Update track score with running average.
                    track_len = (self.tracks[row, 2:] != -1).sum() - 1.
                    frac = 1. / float(track_len)
                    self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match[2]
        # Add unmatched tracks.
        new_ids = np.arange(pts.shape[1]) + offsets[-1]
        new_ids = new_ids[~matched]
        new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 2))
        new_tracks[:, -1] = new_ids
        new_num = new_ids.shape[0]
        new_trackids = self.track_count + np.arange(new_num)
        new_tracks[:, 0] = new_trackids
        new_tracks[:, 1] = self.max_score*np.ones(new_ids.shape[0])
        self.tracks = np.vstack((self.tracks, new_tracks))
        self.track_count += new_num  # Update the track count.
        # Remove empty tracks.
        keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
        self.tracks = self.tracks[keep_rows, :]
        # Store the last descriptors.
        self.last_desc = desc.copy()
        return

    def get_tracks(self, min_length):
        """Retrieve point tracks of a given minimum length.
        Input
          min_length - integer >= 1 with minimum track length
        Output
          returned_tracks - M x (2+L) sized matrix storing track indices, where
            M is the number of tracks and L is the maximum track length.
        """
        if min_length < 1:
            raise ValueError('\'min_length\' too small.')
        valid = np.ones((self.tracks.shape[0])).astype(bool)
        good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
        # Remove tracks which do not have an observation in most recent frame.
        not_headless = (self.tracks[:, -1] != -1)
        keepers = np.logical_and.reduce((valid, good_len, not_headless))
        returned_tracks = self.tracks[keepers, :].copy()
        return returned_tracks

    def draw_tracks(self, out, tracks):
        """Visualize tracks all overlaid on a single image.
        Inputs
          out - numpy uint8 image sized HxWx3 upon which tracks are overlaid.
          tracks - M x (2+L) sized matrix storing track info.
        """
        # Store the number of points per camera.
        pts_mem = self.all_pts
        N = len(pts_mem)  # Number of cameras/images.
        # Get offset ids needed to reference into pts_mem.
        offsets = self.get_offsets()
        # Width of track and point circles to be drawn.
        stroke = 1
        # Iterate through each track and draw it.
        for track in tracks:
            clr = cm.jet(int(np.clip(np.floor(track[1]*10), 0, 9))/10.)[:3]*255
            for i in range(N-1):
                if track[i+2] == -1 or track[i+3] == -1:
                    continue
                offset1 = offsets[i]
                offset2 = offsets[i+1]
                idx1 = int(track[i+2]-offset1)
                idx2 = int(track[i+3]-offset2)
                pt1 = pts_mem[i][:, idx1]
                pt2 = pts_mem[i+1][:, idx2]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
                # Draw end points of each track.
                if i == N-2:
                    clr2 = (255, 0, 0)
                    cv2.circle(out, p2, stroke, clr2, -1, lineType=16)

# -------------------- End of PointTracker Class --------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue Pose Estimation with Tracking',
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

    # Initialize the PointTracker
    nn_thresh = 0.7  # Descriptor matching threshold
    max_length = 5   # Maximum length of tracks
    tracker = PointTracker(max_length=max_length, nn_thresh=nn_thresh)

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
    distance_threshold = 15.0
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

    # Visualize the provided 2D keypoints on the anchor image
    anchor_image_vis = anchor_image.copy()
    for pt in anchor_keypoints_2D:
        x, y = pt
        cv2.circle(anchor_image_vis, (int(x), int(y)), 5, (0, 255, 0), -1)
    cv2.imshow('Anchor Image with Provided Keypoints', anchor_image_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

    # Initialize variables for tracking
    track_id_to_3D_point = {}

    # Process the first frame
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
    frame_keypoints = frame_data['keypoints'][0].cpu().numpy()  # Shape: (N, 2)
    frame_descriptors = frame_data['descriptors'][0].cpu().numpy()  # Shape: (256, N)
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

    print('Matched keypoints in anchor image', len(mkpts0))
    print('Matched keypoints in current frame', len(mkpts1))

    print(f'Number of valid matches: {len(mkpts1)}')

    if len(mkpts1) >= 4:
        # Get descriptors of matched keypoints in the current frame
        matched_frame_indices = matches[valid]
        matched_frame_descriptors = frame_descriptors[:, matched_frame_indices]

        # Prepare keypoints and descriptors for the tracker
        pts = np.vstack((mkpts1.T, mconf[np.newaxis, :]))  # Shape: (3, N)
        desc = matched_frame_descriptors  # Shape: (256, N)

        # Update the tracker with the matched keypoints and descriptors
        tracker.update(pts, desc)

        # Build mapping from track IDs to 3D points
        tracks = tracker.get_tracks(min_length=1)
        for i, track in enumerate(tracks):
            track_id = int(track[0])
            track_id_to_3D_point[track_id] = mpts3D[i]

        # Proceed with pose estimation
        # Define camera intrinsic parameters (adjust these values to your camera)
        focal_length = 1410  # Replace with your camera's focal length in pixels
        cx = frame.shape[1] / 2  # Principal point x-coordinate
        cy = frame.shape[0] / 2  # Principal point y-coordinate

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

        if success and inliers is not None:
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            # Calculate camera position in world coordinates
            camera_position = -R.T @ tvec

            # Append the estimated pose for this frame to the list
            pose_data = {
                'frame': frame_idx,
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

            # Visualization code can be added here

        else:
            print('PnP pose estimation failed.')
            #exit(1)

    else:
        print('Not enough matches to initialize tracking.')
        #exit(1)

    # Start processing subsequent frames
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
        frame_keypoints = frame_data['keypoints'][0].cpu().numpy()  # Shape: (N, 2)
        frame_descriptors = frame_data['descriptors'][0].cpu().numpy()  # Shape: (256, N)
        frame_scores = frame_data['scores'][0].cpu().numpy()

        # Prepare keypoints and descriptors for the tracker
        pts = np.vstack((frame_keypoints.T, frame_scores[np.newaxis, :]))  # Shape: (3, N)
        desc = frame_descriptors  # Shape: (256, N)

        # Update the tracker with the new keypoints and descriptors
        tracker.update(pts, desc)

        # Retrieve the tracks
        tracks = tracker.get_tracks(min_length=1)

        # Prepare lists for pose estimation
        matched_keypoints = []
        matched_3D_points = []

        # For each track, get the current keypoint and corresponding 3D point
        for track in tracks:
            track_id = int(track[0])
            keypoint_idx = int(track[-1])
            if keypoint_idx >= 0 and track_id in track_id_to_3D_point:
                keypoint = frame_keypoints[keypoint_idx]
                matched_keypoints.append(keypoint)
                matched_3D_points.append(track_id_to_3D_point[track_id])

        matched_keypoints = np.array(matched_keypoints)
        matched_3D_points = np.array(matched_3D_points)

        # Proceed only if there are enough matches
        if len(matched_keypoints) >= 4:
            # Define camera intrinsic parameters (adjust these values to your camera)
            focal_length = 1410  # Replace with your camera's focal length in pixels
            cx = frame.shape[1] / 2  # Principal point x-coordinate
            cy = frame.shape[0] / 2  # Principal point y-coordinate

            K = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)

            # Convert points to the required shape for solvePnPRansac
            objectPoints = matched_3D_points.reshape(-1, 1, 3)
            imagePoints = matched_keypoints.reshape(-1, 1, 2)

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

                # Append the estimated pose for this frame to the list
                pose_data = {
                    'frame': frame_idx,
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

                # Visualization code can be added here

            else:
                print('PnP pose estimation failed.')
                # Implement reinitialization
                print('Reinitializing tracking.')
                # Perform matching with anchor image

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

                # Retrieve matched keypoints
                matches = pred['matches0'][0].cpu().numpy()
                confidence = pred['matching_scores0'][0].cpu().numpy()

                # Valid matches (exclude unmatched keypoints)
                valid = matches > -1
                mkpts0 = matched_anchor_keypoints[valid]  # Matched keypoints in anchor image
                mkpts1 = frame_keypoints[matches[valid]]  # Matched keypoints in current frame
                mpts3D = matched_3D_keypoints[valid]      # Corresponding 3D points
                mconf = confidence[valid]

                if len(mkpts1) >= 4:
                    # Get descriptors of matched keypoints in the current frame
                    matched_frame_indices = matches[valid]
                    desc = frame_descriptors[:, matched_frame_indices]

                    # Prepare pts for the tracker
                    pts = np.vstack((mkpts1.T, mconf[np.newaxis, :]))  # Shape: (3, N)

                    # Update the tracker
                    tracker.update(pts, desc)

                    # Rebuild mapping from track IDs to 3D points
                    tracks = tracker.get_tracks(min_length=1)
                    track_id_to_3D_point = {}
                    for i, track in enumerate(tracks):
                        track_id = int(track[0])
                        track_id_to_3D_point[track_id] = mpts3D[i]

                    # Proceed with pose estimation using mkpts1 and mpts3D
                    objectPoints = mpts3D.reshape(-1, 1, 3)
                    imagePoints = mkpts1.reshape(-1, 1, 2)

                    # Use cv2.solvePnPRansac
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

                        # Append the estimated pose for this frame to the list
                        pose_data = {
                            'frame': frame_idx,
                            'rotation_matrix': R.tolist(),
                            'translation_vector': tvec.flatten().tolist(),
                            'camera_position': camera_position.flatten().tolist()
                        }
                        all_poses.append(pose_data)

                        # Output the estimated pose
                        print('Estimated Camera Pose (after reinitialization):')
                        print('Rotation Matrix:\n', R)
                        print('Translation Vector:\n', tvec.flatten())
                        print('Camera Position (World Coordinates):\n', camera_position.flatten())

                        # Visualization code can be added here

                    else:
                        print('PnP pose estimation failed during reinitialization.')
                        # You may choose to exit or continue

                else:
                    print('Not enough matches to reinitialize tracking.')
                    # You may choose to exit or continue

        else:
            print('Not enough matches to compute pose.')
            # Implement reinitialization
            print('Reinitializing tracking.')
            # The reinitialization code is the same as above

        timer.update('viz')
        timer.print()

        # Visualization code can be added here
        # Example: Display the frame with keypoints
        if not opt.no_display:
            display_frame = frame.copy()
            for pt in matched_keypoints:
                x, y = pt
                cv2.circle(display_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.imshow('Pose Estimation', display_frame)
            if cv2.waitKey(1) == ord('q'):
                break

        # Save the output frame if needed
        if opt.output_dir is not None:
            out_file = str(Path(opt.output_dir, f'frame_{frame_idx:06d}.png'))
            cv2.imwrite(out_file, frame)

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()

    with open(opt.save_pose, 'w') as f:
        json.dump(all_poses, f, indent=4)
    print(f'Pose estimation saved to {opt.save_pose}')
