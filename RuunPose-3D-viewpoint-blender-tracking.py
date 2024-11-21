import torch
import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.cm as cm
import json
import os
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
from models.matching import Matching
from models.utils import (AverageTimer, make_matching_plot_fast)

# Remove duplicate imports and set torch to not compute gradients
torch.set_grad_enabled(False)

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

class PointTracker(object):
  """ Class to manage a fixed memory of points and descriptors that enables
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
    """ Iterate through list of points and accumulate an offset value. Used to
    index the global point IDs into the list of points.

    Returns
      offsets - N length array with integer offset locations.
    """
    # Compute id offsets.
    offsets = []
    offsets.append(0)
    for i in range(len(self.all_pts)-1): # Skip last camera size, not needed.
      offsets.append(self.all_pts[i].shape[1])
    offsets = np.array(offsets)
    offsets = np.cumsum(offsets)
    return offsets

  def update(self, pts, desc):
    """ Add a new set of point and descriptor observations to the tracker.

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
    self.all_pts.append(pts)
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
      # Add a new point to it's matched track.
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
          # NOTE(dd): this running average can contain scores from old matches
          #           not contained in last max_length track points.
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
    self.track_count += new_num # Update the track count.
    # Remove empty tracks.
    keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
    self.tracks = self.tracks[keep_rows, :]
    # Store the last descriptors.
    self.last_desc = desc.copy()
    return

  def get_tracks(self, min_length):
    """ Retrieve point tracks of a given minimum length.
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
    """ Visualize tracks all overlayed on a single image.
    Inputs
      out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
      tracks - M x (2+L) sized matrix storing track info.
    """
    # Store the number of points per camera.
    pts_mem = self.all_pts
    N = len(pts_mem) # Number of cameras/images.
    # Get offset ids needed to reference into pts_mem.
    offsets = self.get_offsets()
    # Width of track and point circles to be drawn.
    stroke = 1
    # Iterate through each track and draw it.
    for track in tracks:
      clr = myjet[int(np.clip(np.floor(track[1]*10), 0, 9)), :]*255
      for i in range(N-1):
        if track[i+2] == -1 or track[i+3] == -1:
          continue
        offset1 = offsets[i]
        offset2 = offsets[i+1]
        idx1 = int(track[i+2]-offset1)
        idx2 = int(track[i+3]-offset2)
        pt1 = pts_mem[i][:2, idx1]
        pt2 = pts_mem[i+1][:2, idx2]
        p1 = (int(round(pt1[0])), int(round(pt1[1])))
        p2 = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
        # Draw end points of each track.
        if i == N-2:
          clr2 = (255, 0, 0)
          cv2.circle(out, p2, stroke, clr2, -1, lineType=16)

  def get_matched_keypoints(self, frame_idx_1, frame_idx_2):
        """ Retrieve the matched keypoints between two frames.
        Inputs:
            frame_idx_1 - The index of the first frame (0 for the most recent frame)
            frame_idx_2 - The index of the second frame (1 for the previous frame)
        Outputs:
            matched_pts1 - 2D points from the first frame (Nx2 array)
            matched_pts2 - 2D points from the second frame (Nx2 array)
        """
        # Ensure frame indices are within bounds
        assert 0 <= frame_idx_1 < self.maxl
        assert 0 <= frame_idx_2 < self.maxl

        # Extract valid tracks (tracks with points in both frames)
        valid_tracks = self.tracks[:, [2 + frame_idx_1, 2 + frame_idx_2]]
        valid_mask = np.all(valid_tracks != -1, axis=1)

        # Get the indices of valid tracks (where points exist in both frames)
        valid_tracks = valid_tracks[valid_mask]

        # Get offsets for the points
        offsets = self.get_offsets()

        # Extract the corresponding keypoints from the point memory
        matched_pts1 = self.all_pts[frame_idx_1][:, valid_tracks[:, 0].astype(int) - offsets[frame_idx_1]].T
        matched_pts2 = self.all_pts[frame_idx_2][:, valid_tracks[:, 1].astype(int) - offsets[frame_idx_2]].T

        return matched_pts1, matched_pts2





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

# def load_ground_truth_poses(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     gt_poses = {}
#     for frame in data['frames']:
#         image_name = frame['image']
#         for obj in frame['object_poses']:
#             if obj['name'] == 'Camera':
#                 pose = np.array(obj['pose'], dtype=np.float32)
#                 gt_poses[image_name] = pose
#                 break  # Assume only one camera pose per frame
#     return gt_poses


def load_ground_truth_poses(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    gt_poses = {}
    T_blender_to_opencv = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ], dtype=np.float32)
    for frame in data['frames']:
        image_name = frame['image']
        for obj in frame['object_poses']:
            if obj['name'] == 'Camera':
                pose = np.array(obj['pose'], dtype=np.float32)
                # Extract rotation and translation
                R_blender = pose[:3, :3]
                t_blender = pose[:3, 3]
                # Transform rotation and translation
                R_opencv = T_blender_to_opencv @ R_blender
                t_opencv = T_blender_to_opencv @ t_blender
                # Reconstruct pose matrix
                pose_opencv = np.eye(4, dtype=np.float32)
                pose_opencv[:3, :3] = R_opencv
                pose_opencv[:3, 3] = t_opencv
                gt_poses[image_name] = pose_opencv
                break  # Assume only one camera pose per frame
    return gt_poses

def rotation_matrix_to_axis_angle(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    return theta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue Pose Estimation with Viewpoint-based Anchors',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to an image directory')
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
    parser.add_argument(
        '--viewpoint_model_path', type=str, required=True,
        help='Path to the trained viewpoint model')
    parser.add_argument('--min_length', type=int, default=2,
      help='Minimum length of point tracks (default: 2).')
    parser.add_argument('--max_length', type=int, default=5,
      help='Maximum length of point tracks (default: 5).')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
      help='Descriptor matching threshold (default: 0.7).')
    parser.add_argument('--show_extra', action='store_true',
      help='Show extra debug outputs (default: False).')
    parser.add_argument('--display_scale', type=int, default=1,
      help='Factor to scale output visualization (default: 1).')
    
    parser.add_argument('--H', type=int, default=1280,
      help='Input image height (default: 120).')
    parser.add_argument('--W', type=int, default=960,
      help='Input image width (default:160).')

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

    # This class helps merge consecutive point matches into tracks.
    #tracker = PointTracker(5, nn_thresh=0.7)
    tracker = PointTracker(opt.max_length, nn_thresh=opt.nn_thresh)

    # Load the viewpoint classification model
    num_classes = 4  # Number of viewpoint classes
    class_names = ['back', 'front', 'left', 'right']  # Viewpoint class names

    # Load the pre-trained ResNet18 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    # Load the trained weights
    model.load_state_dict(torch.load(opt.viewpoint_model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Function to predict viewpoint
    def predict_viewpoint(model, image_pil):
        image = transform(image_pil).unsqueeze(0)  # Add batch dimension
        image = image.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
            _, preds = torch.max(outputs, 1)
        
        # Get predicted class
        class_idx = preds.item()
        predicted_viewpoint = class_names[class_idx]
        
        # Convert probabilities to percentages
        probabilities_percent = probabilities.cpu().numpy()[0] * 100
        
        return predicted_viewpoint, probabilities_percent

    # Prepare anchor images and their keypoints for each viewpoint
    anchor_image_paths = {
        'front': '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/61.png',
        'back': '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/70.png',
        'left': '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/62.png',
        'right': '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/85.png'
    }

    # Replace the following with your actual 2D and 3D keypoints for each viewpoint
    anchor_keypoints_2D_data = {
        'front': np.array([[ 558.,  269.],
                            [ 856.,  277.],
                            [ 536.,  283.],
                            [ 265.,  449.],
                            [ 225.,  462.],
                            [ 657.,  477.],
                            [1086.,  480.],
                            [ 217.,  481.],
                            [ 567.,  483.],
                            [ 653.,  488.],
                            [1084.,  497.],
                            [1084.,  514.],
                            [ 552.,  551.],
                            [ 640.,  555.]], dtype=np.float32),
        'back': np.array([[ 860.,  388.],
                            [ 467.,  394.],
                            [ 881.,  414.],
                            [ 466.,  421.],
                            [ 668.,  421.],
                            [ 591.,  423.],
                            [1078.,  481.],
                            [ 195.,  494.],
                            [ 183.,  540.],
                            [ 626.,  592.],
                            [ 723.,  592.]], dtype=np.float32),
        'left': np.array([[ 968.,  313.],
                            [1077.,  315.],
                            [1083.,  376.],
                            [ 713.,  402.],
                            [ 688.,  412.],
                            [ 827.,  417.],
                            [ 512.,  436.],
                            [ 472.,  446.],
                            [1078.,  468.],
                            [ 774.,  492.],
                            [ 740.,  493.],
                            [1076.,  506.],
                            [ 416.,  511.],
                            [ 452.,  527.],
                            [ 594.,  594.],
                            [ 560.,  611.],
                            [ 750.,  618.]], dtype=np.float32),
        'right': np.array([[367., 300.],
                            [264., 298.],
                            [279., 357.],
                            [165., 353.],
                            [673., 401.],
                            [559., 409.],
                            [780., 443.],
                            [772., 459.],
                            [209., 443.],
                            [609., 490.],
                            [528., 486.],
                            [867., 515.],
                            [495., 483.],
                            [822., 537.],
                            [771., 543.],
                            [539., 592.],
                            [573., 610.],
                            [386., 604.]], dtype=np.float32)
    }

    anchor_keypoints_3D_data = {
        'front': np.array([[-0.81972, -0.3258 ,  5.28664],
                            [-0.81972,  0.33329,  5.28664],
                            [-0.60385, -0.3258 ,  5.28664],
                            [-0.29107, -0.83895,  4.96934],
                            [-0.04106, -0.83895,  4.995  ],
                            [ 0.26951,  0.0838 ,  5.0646 ],
                            [-0.29152,  0.84644,  4.96934],
                            [ 0.01734, -0.83895,  4.9697 ],
                            [ 0.31038, -0.0721 ,  5.05571],
                            [ 0.31038,  0.07959,  5.05571],
                            [-0.03206,  0.84644,  4.99393],
                            [ 0.01734,  0.84644,  4.9697 ],
                            [ 0.44813, -0.07631,  4.9631 ],
                            [ 0.44813,  0.0838 ,  4.96381]], dtype=np.float32),
        'back': np.array([[-0.60385, -0.3258 ,  5.28664],
                            [-0.60385,  0.33329,  5.28664],
                            [-0.81972, -0.3258 ,  5.28664],
                            [-0.81972,  0.33329,  5.28664],
                            [ 0.26951, -0.07631,  5.0646 ],
                            [ 0.26951,  0.0838 ,  5.0646 ],
                            [-0.29297, -0.83895,  4.96825],
                            [-0.04106,  0.84644,  4.995  ],
                            [-0.29297,  0.84644,  4.96825],
                            [-0.81973,  0.0838 ,  4.99302],
                            [-0.81973, -0.07631,  4.99302]], dtype=np.float32),
        'left': np.array([[ -0.60385, -0.3258   ,5.28664],
                            [-0.81972 ,-0.3258   ,5.28664],
                            [-0.81972 , 0.33329  ,5.28664],
                            [-0.04106 ,-0.83895  ,4.995  ],
                            [ 0.01551 ,-0.83895  ,4.97167],
                            [-0.29107 ,-0.83895  ,4.96934],
                            [ 0.26951 ,-0.07631  ,5.0646 ],
                            [ 0.31038 ,-0.07631  ,5.05571],
                            [-0.81972 , 0.08616  ,5.06584],
                            [-0.26104 , 0.0838   ,5.00304],
                            [-0.1986  , 0.0838   ,5.00304],
                            [-0.81906 , 0.0838   ,4.99726],
                            [ 0.42759 , 0.0838   ,4.94447],
                            [ 0.35674 , 0.0838   ,4.91463],
                            [-0.03206 , 0.84644  ,4.99393],
                            [ 0.01551 , 0.84644  ,4.9717 ],
                            [-0.29152 , 0.84644  ,4.96934]], dtype=np.float32),
        'right': np.array([[-0.60385,  0.33329,  5.28664],
                            [-0.81972,  0.33329,  5.28664],
                            [-0.60385, -0.3258 ,  5.28664],
                            [-0.81972, -0.3258 ,  5.28664],
                            [-0.04106,  0.84644,  4.995  ],
                            [-0.29152,  0.84644,  4.96934],
                            [ 0.26951,  0.0838 ,  5.0646 ],
                            [ 0.26951, -0.07631,  5.0646 ],
                            [-0.81972, -0.07867,  5.06584],
                            [-0.04106, -0.07631,  4.995  ],
                            [-0.1986 , -0.07631,  5.00304],
                            [ 0.44813, -0.07631,  4.96381],
                            [-0.26104, -0.07631,  5.00304],
                            [ 0.35674, -0.07631,  4.91463],
                            [ 0.2674 , -0.07631,  4.89973],
                            [-0.04106, -0.83895,  4.995  ],
                            [ 0.01551, -0.83895,  4.97167],
                            [-0.29152, -0.83895,  4.96934]], dtype=np.float32)
    }

    # Transformation matrix from Blender to OpenCV coordinate system
    T_blender_to_opencv = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ], dtype=np.float32)

    # Transform the anchor_keypoints_3D_data from Blender to OpenCV coordinate system
    for viewpoint in anchor_keypoints_3D_data:
        anchor_kp_3D_blender = anchor_keypoints_3D_data[viewpoint]
        # Apply the transformation
        anchor_kp_3D_opencv = (T_blender_to_opencv @ anchor_kp_3D_blender.T).T
        # Update the dictionary with transformed points
        anchor_keypoints_3D_data[viewpoint] = anchor_kp_3D_opencv

    # Preprocess anchor images and keypoints
    anchor_data = {}
    for viewpoint in class_names:
        # Load the anchor image
        anchor_path = anchor_image_paths[viewpoint]
        anchor_image = cv2.imread(anchor_path)
        assert anchor_image is not None, f'Failed to load anchor image at {anchor_path}'

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
        anchor_sp_data = matching.superpoint({'image': anchor_tensor})
        anchor_keypoints_sp = anchor_sp_data['keypoints'][0].cpu().numpy()  # Shape: (N, 2)
        anchor_descriptors_sp = anchor_sp_data['descriptors'][0].cpu().numpy()  # Shape: (256, N)
        anchor_scores_sp = anchor_sp_data['scores'][0].cpu().numpy()

        # Load the provided 2D and 3D keypoints for the anchor image
        anchor_kp_2D = anchor_keypoints_2D_data[viewpoint]
        anchor_kp_3D = anchor_keypoints_3D_data[viewpoint]

        # Build a KD-Tree of the SuperPoint keypoints
        sp_tree = cKDTree(anchor_keypoints_sp)

        # For each provided 2D keypoint, find the nearest SuperPoint keypoint
        distances, indices = sp_tree.query(anchor_kp_2D, k=1)

        # Set a distance threshold to accept matches (e.g., 1 pixels)
        distance_threshold = 1  # Adjust as needed
        valid_matches = distances < distance_threshold

        if not np.any(valid_matches):
            raise ValueError(f'No matching keypoints found within the distance threshold for viewpoint {viewpoint}')

        # Filter to keep only valid matches
        matched_anchor_indices = indices[valid_matches]
        matched_2D_keypoints = anchor_kp_2D[valid_matches]
        matched_3D_keypoints = anchor_kp_3D[valid_matches]

        # Get the descriptors for the matched keypoints
        matched_descriptors = anchor_descriptors_sp[:, matched_anchor_indices]
        # Get the keypoints
        matched_anchor_keypoints = anchor_keypoints_sp[matched_anchor_indices]
        # Get the scores
        matched_scores = anchor_scores_sp[matched_anchor_indices]

        # Store the data in the anchor_data dictionary
        anchor_data[viewpoint] = {
            'image': anchor_image,
            'tensor': anchor_tensor,
            'keypoints_sp': anchor_keypoints_sp,
            'descriptors_sp': anchor_descriptors_sp,
            'scores_sp': anchor_scores_sp,
            'matched_anchor_keypoints': matched_anchor_keypoints,
            'matched_descriptors': matched_descriptors,
            'matched_scores': matched_scores,
            'matched_3D_keypoints': matched_3D_keypoints
        }

    # Load ground truth poses
    gt_poses = load_ground_truth_poses(opt.ground_truth)

    # Read a sequence of images from the input directory
    input_images = sorted(list(Path(opt.input).glob('*.png')))
    assert len(input_images) > 0, f'No images found in the directory {opt.input}'

    print(f'Found {len(input_images)} images in directory {opt.input}')

    frame_idx = 0  # Initialize frame counter
    timer = AverageTimer()
    all_poses = []

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pose Estimation', 640 * 2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Initialize variables
    cumulative_R = None
    cumulative_t = None
    R_tracking = None
    t_tracking = None


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

        # Convert the frame to PIL Image for viewpoint prediction
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Predict the viewpoint
        predicted_viewpoint, probabilities = predict_viewpoint(model, frame_pil)
        print(f'Predicted viewpoint: {predicted_viewpoint}, probabilities: {probabilities}')

        # Select the anchor data for the predicted viewpoint
        anchor_info = anchor_data[predicted_viewpoint]
        anchor_tensor = anchor_info['tensor']
        matched_anchor_keypoints = anchor_info['matched_anchor_keypoints']
        matched_descriptors = anchor_info['matched_descriptors']
        matched_scores = anchor_info['matched_scores']
        matched_3D_keypoints = anchor_info['matched_3D_keypoints']
        anchor_image = anchor_info['image']  # For visualization

        # Convert the current frame to tensor
        frame_tensor = frame2tensor(frame, device)

        # Extract keypoints and descriptors from the current frame using SuperPoint
        frame_data = matching.superpoint({'image': frame_tensor})
        frame_keypoints = frame_data['keypoints'][0].cpu().numpy()
        frame_descriptors = frame_data['descriptors'][0].cpu().numpy()
        frame_scores = frame_data['scores'][0].cpu().numpy()
        print(f"SuperPoint keypoints: {frame_keypoints.shape}, descriptors: {frame_descriptors.shape}")

        # Handle edge cases: skip the frame if no keypoints or descriptors are detected
        if frame_keypoints.shape[0] == 0 or frame_descriptors.shape[1] == 0:
            print(f"No keypoints or descriptors detected in frame {frame_name}, skipping frame.")
            continue  # Skip to the next frame

        ##################################################################tracking code
        # Font parameters for visualizaton.
        font = cv2.FONT_HERSHEY_DUPLEX
        font_clr = (255, 255, 255)
        font_pt = (4, 12)
        font_sc = 0.4
        frame_keypoints_SP = frame_keypoints.T 
        tracker.update(frame_keypoints_SP, frame_descriptors)
        
        # Get tracks for points which were match successfully across all frames.
        tracks = tracker.get_tracks(opt.min_length)
        #tracks = tracker.get_tracks(2)

        # Primary output - Show point tracks overlayed on top of input image.
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame1 = (frame1.astype('float32') / 255.)
        frame1 = frame1.astype('float32')

        out1 = (np.dstack((frame1, frame1, frame1)) * 255.).astype('uint8')
        tracks[:, 1] /= float(0.7) # Normalize track scores to [0,1].
        tracker.draw_tracks(out1, tracks)
        if opt.show_extra:
            cv2.putText(out1, 'Point Tracks', font_pt, font, font_sc, font_clr, lineType=16)
        
        out1 = cv2.resize(out1, (opt.display_scale*opt.W, opt.display_scale*opt.H))
        cv2.imshow('Tracking', out1)

        # Get the matched keypoints between the most recent frame (frame_idx_1 = 0) and the previous frame (frame_idx_2 = 1)
        matched_pts1, matched_pts2 = tracker.get_matched_keypoints(0, 1)

        # Print or use the matched keypoints as needed
        print("Matched keypoints in frame 1 (p1):", matched_pts1)
        print("Matched keypoints in frame 2 (p2):", matched_pts2)
        print("matched_pts1 shape:", matched_pts1.shape)
        print("matched_pts2 shape:", matched_pts2.shape)


        #############################################################
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
            focal_length_x = 2666.66666666666  # px
            focal_length_y = 2666.66666666666  # py
            cx = 639.5  # Principal point u0
            cy = 479.5  # Principal point v0

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
                # Compute reprojection errors
                objectPoints_inliers = mpts3D[inliers.flatten()].reshape(-1, 1, 3)
                imagePoints_inliers = mkpts1[inliers.flatten()].reshape(-1, 1, 2)

                rvec, tvec = cv2.solvePnPRefineVVS(
                    objectPoints=objectPoints_inliers,
                    imagePoints=imagePoints_inliers,
                    cameraMatrix=K,
                    distCoeffs=None,
                    rvec=rvec_o,
                    tvec=tvec_o
                )

                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                # Calculate camera position in world coordinates
                camera_position = -R.T @ tvec


                ##########################################################################tracking pose estimation code
                # Initialize variables
                if frame_idx == 1:
                    # Initial pose from SuperGlue and PnP
                    initial_R = R.copy()
                    initial_t = tvec.copy()
                    cumulative_R = R.copy()
                    cumulative_t = tvec.copy()
                else:
                    if matched_pts1.shape[0] >= 5 and matched_pts2.shape[0] >= 5:
                        E, mask = cv2.findEssentialMat(matched_pts1, matched_pts2, K)
                        retval, R_rel, t_rel, _ = cv2.recoverPose(E, matched_pts1, matched_pts2, K)

                        if retval > 0:  # Pose recovery was successful
                            # Normalize t_rel to unit length
                            t_rel_normalized = t_rel / np.linalg.norm(t_rel)
                            
                            # Ensure R_rel and cumulative_R have valid shapes before matrix multiplication
                            if R_rel is not None and R_rel.shape == (3, 3) and cumulative_R is not None:
                                cumulative_R = R_rel @ cumulative_R
                                cumulative_t = R_rel @ cumulative_t + t_rel_normalized
                            else:
                                print(f"Skipping pose update due to invalid matrix shapes at frame {frame_idx}")
                        else:
                            print(f"Pose recovery failed for frame {frame_idx}")

                # Compare poses
                # Pose from SuperGlue and PnP
                R_pnp = R
                t_pnp = tvec

                # Check if R_tracking and t_tracking are valid
                if R_tracking is not None and t_tracking is not None:
                    # Pose from Tracking
                    # Compute tracking rotation and translation errors
                    rotation_diff_tracking = R_tracking @ gt_R.T
                    rotation_error_tracking = rotation_matrix_to_axis_angle(rotation_diff_tracking)
                    translation_error_tracking = np.linalg.norm(t_tracking.flatten() - gt_t)
                else:
                    # Set errors to None if tracking is unavailable
                    rotation_error_tracking = None
                    translation_error_tracking = None


                ##########################################################################################################################

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
                print('mean_reprojection_error:',mean_reprojection_error)
                print('std_reprojection_error:',std_reprojection_error)

                # Compare with Ground Truth
                if frame_name in gt_poses:
                    gt_pose = gt_poses[frame_name]
                    gt_R = gt_pose[:3, :3]
                    gt_t = gt_pose[:3, 3]

                    # Compute rotation error
                    rotation_diff = R @ gt_R.T
                    rotation_error = rotation_matrix_to_axis_angle(rotation_diff)
                    # Compute translation error
                    translation_error = np.linalg.norm(tvec.flatten() - gt_t)

                    # Errors for Tracking pose
                    rotation_diff_tracking = R_tracking @ gt_R.T
                    rotation_error_tracking = rotation_matrix_to_axis_angle(rotation_diff_tracking)
                    translation_error_tracking = np.linalg.norm(t_tracking.flatten() - gt_t)

                    # Log or print the errors for comparison
                    print(f'PnP Rotation Error (rad): {rotation_error}')
                    print(f'PnP Translation Error: {translation_error}')
                    print(f'Tracking Rotation Error (rad): {rotation_error_tracking}')
                    print(f'Tracking Translation Error: {translation_error_tracking}')
                    
                    # You can store these errors in your pose_data for later analysis
                    pose_data['rotation_error_tracking_rad'] = float(rotation_error_tracking)
                    pose_data['translation_error_tracking'] = float(translation_error_tracking)

                else:
                    print(f'Ground truth pose not found for {frame_name}')
                    rotation_error = None
                    translation_error = None

                # Save pose data
                pose_data = {
                        'frame': frame_idx,
                        'image_name': frame_name,
                        'predicted_viewpoint': predicted_viewpoint,
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
                        'rotation_matrix_pnp': R_pnp.tolist(),
                        'translation_vector_pnp': t_pnp.flatten().tolist(),
                        'rotation_matrix_tracking': R_tracking.tolist() if R_tracking is not None else None,
                        'translation_vector_tracking': t_tracking.flatten().tolist() if t_tracking is not None else None,
                        'rotation_error_tracking_rad': float(rotation_error_tracking) if rotation_error_tracking is not None else None,
                        'translation_error_tracking': float(translation_error_tracking) if translation_error_tracking is not None else None,
                    }
                all_poses.append(pose_data)

                # Output the estimated pose
                print('Estimated Camera Pose:')
                print('Rotation Matrix:\n', R)
                print('Translation Vector:\n', tvec.flatten())
                print('Camera Position (World Coordinates):\n', camera_position.flatten())
                print(f'Rotation Error (rad): {rotation_error}')
                print(f'Translation Error: {translation_error}')

                # Visualization code
                # Convert images to grayscale for visualization
                anchor_image_gray = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
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
