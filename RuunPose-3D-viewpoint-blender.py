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
