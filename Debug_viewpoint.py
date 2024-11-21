import cv2
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from pathlib import Path

# Ensure interactive mode is off for proper figure updates
plt.ioff()

def load_pose_data(pose_file):
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    return pose_data

def load_ground_truth_poses(gt_file):
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    gt_poses = {}
    # Transformation matrix from Blender to OpenCV coordinate system
    T_blender_to_opencv = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ], dtype=np.float32)
    for frame in gt_data['frames']:
        image_name = Path(frame['image']).name  # Extract filename only
        for obj in frame['object_poses']:
            if obj['name'] == 'Camera':
                pose_matrix_blender = np.array(obj['pose'], dtype=np.float32)
                # Extract rotation and translation
                R_blender = pose_matrix_blender[:3, :3]
                t_blender = pose_matrix_blender[:3, 3]
                # Transform rotation and translation
                R_opencv = T_blender_to_opencv @ R_blender
                t_opencv = T_blender_to_opencv @ t_blender
                # Reconstruct pose matrix
                pose_matrix_opencv = np.eye(4, dtype=np.float32)
                pose_matrix_opencv[:3, :3] = R_opencv
                pose_matrix_opencv[:3, 3] = t_opencv
                gt_poses[image_name] = pose_matrix_opencv
                break  # Assume only one camera pose per frame
    return gt_poses

def visualize_pose_and_matches(pose_data, image_dir, anchor_images_info,
                               K, distCoeffs, gt_poses):
    # Get list of image paths
    image_paths = sorted(list(Path(image_dir).glob('*.png')))
    if not image_paths:
        print('No images found in the directory.')
        return

    total_frames = len(image_paths)
    frame_idx = 0  # Initialize frame counter
    camera_positions_estimated = []
    camera_positions_ground_truth = []

    # Prepare figure for 3D pose visualization
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    ax_image = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[0, 1], projection='3d')

    # Event handling variables
    stop = False

    def map_frame_name(frame_name):
        # Extract the numerical part of the frame name
        base_name = os.path.splitext(frame_name)[0]  # Remove extension
        frame_number = base_name[-4:]  # Get the last 4 digits
        gt_frame_name = f"{int(frame_number):08d}.png"
        print('GT frame name matched with current frame:', gt_frame_name)
        return gt_frame_name

    def update_visualization():
        nonlocal frame_idx, stop

        if frame_idx < 0 or frame_idx >= total_frames:
            print('Frame index out of bounds.')
            return

        # Clear axes
        ax_image.clear()
        ax_3d.clear()

        # Read the current image
        img_path = image_paths[frame_idx]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f'Failed to read image {img_path}.')
            stop = True
            return

        # Get the image filename to match with pose data
        frame_name = img_path.name  # e.g., '00010001.png'
        # Adjust frame_name to match ground truth image names
        gt_frame_name = map_frame_name(frame_name)

        print(f'Current frame_name: {frame_name}')
        print(f'Adjusted ground truth frame name: {gt_frame_name}')
        print(f'Available ground truth frames (first 5): {list(gt_poses.keys())[:5]}')

        # Find the frame data
        frame_data = next((fd for fd in pose_data if fd.get('image_name') == frame_name
                           or fd.get('frame') == frame_idx + 1), None)
        if frame_data is None:
            print(f'Frame data for image {frame_name} not found.')
            return

        # Get the viewpoint used for this frame (assumed to be stored in pose_data)
        predicted_viewpoint = frame_data.get('predicted_viewpoint', None)
        if predicted_viewpoint is None:
            print(f'Predicted viewpoint for frame {frame_name} not found in pose data.')
            return

        # Get anchor info for the predicted viewpoint
        anchor_info = anchor_images_info.get(predicted_viewpoint, None)
        if anchor_info is None:
            print(f'Anchor information for viewpoint "{predicted_viewpoint}" not found.')
            return

        # Load the anchor image
        anchor_image = anchor_info['image']
        anchor_keypoints_2D = anchor_info['keypoints_2D']
        anchor_keypoints_3D = anchor_info['keypoints_3D']

        # Convert anchor keypoints to numpy arrays
        anchor_keypoints_2D = np.array(anchor_keypoints_2D)
        anchor_keypoints_3D = np.array(anchor_keypoints_3D)

        # Get ground truth pose for this frame
        gt_pose = gt_poses.get(gt_frame_name)
        if gt_pose is not None:
            gt_rotation_matrix = np.array(gt_pose[:3, :3])
            gt_translation_vector = np.array(gt_pose[:3, 3])
            gt_camera_position = -gt_rotation_matrix.T @ gt_translation_vector

            # Compute ground truth camera's forward direction in world coordinates
            forward_vector_gt = gt_rotation_matrix.T @ np.array([0, 0, 1])

            # Print the ground truth rotation and translation
            print(f"Ground Truth Rotation Matrix for {gt_frame_name}:\n{gt_rotation_matrix}")
            print(f"Ground Truth Translation Vector for {gt_frame_name}:\n{gt_translation_vector}")
            print(f"Ground Truth Camera Position (World Coordinates): {gt_camera_position}")
        else:
            gt_camera_position = None
            gt_translation_vector = None
            print(f'Ground truth pose for image {gt_frame_name} not found.')

        # Get matched keypoints
        mkpts0 = np.array(frame_data.get('mkpts0', []))  # Anchor keypoints
        mkpts1 = np.array(frame_data.get('mkpts1', []))  # Frame keypoints
        mconf = np.array(frame_data.get('mconf', []))    # Matching confidence scores
        inliers = frame_data.get('inliers', [])
        inliers = np.array(inliers).astype(int)
        num_inliers = frame_data.get('num_inliers', 0)
        total_matches = frame_data.get('total_matches', 0)
        inlier_ratio = frame_data.get('inlier_ratio', 0)
        mean_reproj_error = frame_data.get('mean_reprojection_error', None)
        std_reproj_error = frame_data.get('std_reprojection_error', None)
        rotation_error_rad = frame_data.get('rotation_error_rad', None)
        translation_error = frame_data.get('translation_error', None)
        print('translation_error', translation_error)

        # Prepare keypoints for plotting
        # For anchor image
        anchor_image_gray = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
        # For current frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a combined image
        height1, width1 = anchor_image_gray.shape
        height2, width2 = frame_gray.shape
        max_height = max(height1, height2)
        total_width = width1 + width2
        combined_image = np.zeros((max_height, total_width), dtype=np.uint8)
        combined_image[:height1, :width1] = anchor_image_gray
        combined_image[:height2, width1:total_width] = frame_gray

        # Adjust keypoints for plotting on combined image
        mkpts0_plot = mkpts0.copy()
        mkpts1_plot = mkpts1.copy()

        # Ensure mkpts1_plot is 2-dimensional and has data
        if mkpts1_plot.ndim == 2 and mkpts1_plot.shape[0] > 0:
            # Shift x-coordinate of frame keypoints
            mkpts1_plot[:, 0] += width1  # Shift x-coordinate
        else:
            print(f'Warning: mkpts1_plot is not 2D or is empty for frame {frame_idx}')
            return  # Skip visualization for this frame if mkpts1_plot is invalid

        # Color mapping based on matching confidence scores
        # Normalize confidence scores between 0 and 1
        if len(mconf) > 0:
            mconf_norm = (mconf - mconf.min()) / (mconf.max() - mconf.min() + 1e-8)
            colors = cm.viridis(mconf_norm)
        else:
            colors = []

        # Plot the combined image
        ax_image.imshow(combined_image, cmap='gray')
        ax_image.axis('off')

        # Plot matches
        for i in range(len(mkpts0)):
            x0, y0 = mkpts0_plot[i]
            x1, y1 = mkpts1_plot[i]
            if len(colors) > 0:
                color = colors[i]
            else:
                color = 'yellow'
            linewidth = 1
            linestyle = '-'
            if i in inliers:
                linewidth = 2
            else:
                linestyle = '--'
            ax_image.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth, linestyle=linestyle)

        # Plot keypoints
        ax_image.scatter(mkpts0_plot[:, 0], mkpts0_plot[:, 1], s=30, c='cyan', marker='o', label='Anchor Keypoints')
        ax_image.scatter(mkpts1_plot[:, 0], mkpts1_plot[:, 1], s=30, c='orange', marker='o', label='Frame Keypoints')

        ax_image.set_title(f'Frame {frame_idx + 1}: Matches (Confidence Colored)')
        ax_image.legend(loc='upper right')

        # Reproject points using the estimated pose, only if inliers > 5
        if num_inliers > 5 and 'rotation_matrix' in frame_data and 'translation_vector' in frame_data:
            rotation_matrix = np.array(frame_data.get('rotation_matrix', []))
            translation_vector = np.array(frame_data.get('translation_vector', []))
            print('estimated rotation_matrix:', rotation_matrix)

            if rotation_matrix.size == 9 and translation_vector.size == 3:
                # Reshape rotation matrix to 3x3 if necessary
                rotation_matrix = rotation_matrix.reshape(3, 3)
                translation_vector = translation_vector.reshape(3, 1)

                print('estimated rotation_matrix:', rotation_matrix)

                # Convert rotation matrix to rotation vector using Rodrigues
                rvec, _ = cv2.Rodrigues(rotation_matrix)

                print('estimated rvec:', rvec)

                # Now, rvec is a rotation vector suitable for cv2.projectPoints
                reprojected_points, _ = cv2.projectPoints(anchor_keypoints_3D, rvec, translation_vector, K, distCoeffs)
                reprojected_points = reprojected_points.squeeze()

                # Plot the reprojected points in different colors for inliers and outliers
                for i, (x, y) in enumerate(reprojected_points):
                    if i in inliers:
                        ax_image.scatter(x + width1, y, s=50, c='green', marker='x',
                                         label='Inliers Reprojection' if i == 0 else "")
                    else:
                        ax_image.scatter(x + width1, y, s=50, c='red', marker='x',
                                         label='Outliers Reprojection' if i == 0 else "")

                # Avoid duplicate labels
                handles, labels = ax_image.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_image.legend(by_label.values(), by_label.keys())

                ax_image.set_title(f'Reprojected Points (Frame {frame_idx + 1})')

            else:
                print('Invalid rotation matrix or translation vector dimensions.')
                return

            # Only update 3D plot if number of inliers is greater than 5
            if 'camera_position' in frame_data:
                # Compute estimated camera position
                camera_position_estimated = -rotation_matrix.T @ translation_vector

                # Compute the camera's forward direction in world coordinates
                forward_vector_estimated = rotation_matrix[:, 2]  # Third column of R

                print('forward_vector_estimated:', forward_vector_estimated)

                # Define the rotation matrix for 90 degrees clockwise rotation around x-axis
                R_x_90 = np.array([
                    [1, 0,  0],
                    [0, 0, -1],
                    [0, 1,  0]
                ], dtype=np.float32)

                arrow_length = 1  # Adjust as needed based on your scale

                # Rotate anchor keypoints
                anchor_keypoints_3D_rotated = (R_x_90 @ anchor_keypoints_3D.T).T

                # Plot the rotated anchor 3D points
                ax_3d.scatter(anchor_keypoints_3D_rotated[:, 0], anchor_keypoints_3D_rotated[:, 1], anchor_keypoints_3D_rotated[:, 2],
                              c='b', marker='o', label='Anchor 3D Points')

                # Rotate estimated camera position
                camera_position_estimated_rotated = R_x_90 @ camera_position_estimated.flatten()
                camera_positions_estimated.append(camera_position_estimated_rotated)
                camera_positions_est_np = np.array(camera_positions_estimated)

                # Plot the estimated camera positions
                ax_3d.plot(camera_positions_est_np[:, 0], camera_positions_est_np[:, 1], camera_positions_est_np[:, 2],
                           c='r', marker='o', label='Estimated Camera Trajectory')

                # Plot current estimated camera position
                ax_3d.scatter(camera_position_estimated_rotated[0], camera_position_estimated_rotated[1], camera_position_estimated_rotated[2],
                              c='r', marker='^', s=100, label='Current Estimated Position')

                # Rotate estimated forward vector
                forward_vector_estimated_rotated = R_x_90 @ forward_vector_estimated

                # Plot viewing direction for estimated pose
                start_est = camera_position_estimated_rotated
                ax_3d.quiver(
                    start_est[0], start_est[1], start_est[2],
                    forward_vector_estimated_rotated[0], forward_vector_estimated_rotated[1], forward_vector_estimated_rotated[2],
                    length=arrow_length, color='blue', normalize=True, label='Estimated Viewing Direction'
                )

                # Plot ground truth camera positions
                if gt_translation_vector is not None:
                    # Rotate ground truth camera position
                    gt_camera_position_rotated = R_x_90 @ gt_translation_vector.flatten()
                    camera_positions_ground_truth.append(gt_camera_position_rotated)
                    camera_positions_gt_np = np.array(camera_positions_ground_truth)

                    # Plot ground truth camera trajectory
                    ax_3d.plot(camera_positions_gt_np[:, 0], camera_positions_gt_np[:, 1], camera_positions_gt_np[:, 2],
                               c='g', marker='o', label='Ground Truth Trajectory')

                    # Plot current ground truth camera position
                    ax_3d.scatter(gt_camera_position_rotated[0], gt_camera_position_rotated[1], gt_camera_position_rotated[2],
                                  c='g', marker='^', s=100, label='Current Ground Truth Position')

                    # # Rotate ground truth forward vector
                    # forward_vector_gt_rotated = R_x_90 @ forward_vector_gt

                    # # Plot viewing direction for ground truth pose
                    # start_gt = gt_camera_position_rotated
                    # ax_3d.quiver(
                    #     start_gt[0], start_gt[1], start_gt[2],
                    #     forward_vector_gt_rotated[0], forward_vector_gt_rotated[1], forward_vector_gt_rotated[2],
                    #     length=arrow_length, color='magenta', normalize=True, label='Ground Truth Viewing Direction'
                    # )

                else:
                    print(f'Ground truth pose for image {gt_frame_name} not found.')

                # Combine all positions to get the full range
                all_x = np.concatenate((anchor_keypoints_3D_rotated[:, 0], camera_positions_est_np[:, 0]))
                all_y = np.concatenate((anchor_keypoints_3D_rotated[:, 1], camera_positions_est_np[:, 1]))
                all_z = np.concatenate((anchor_keypoints_3D_rotated[:, 2], camera_positions_est_np[:, 2]))

                if gt_translation_vector is not None:
                    all_x = np.concatenate((all_x, camera_positions_gt_np[:, 0]))
                    all_y = np.concatenate((all_y, camera_positions_gt_np[:, 1]))
                    all_z = np.concatenate((all_z, camera_positions_gt_np[:, 2]))

                # Calculate the ranges
                x_min, x_max = all_x.min(), all_x.max()
                y_min, y_max = all_y.min(), all_y.max()
                z_min, z_max = all_z.min(), all_z.max()
                max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0

                # Calculate midpoints
                mid_x = (x_max + x_min) * 0.5
                mid_y = (y_max + y_min) * 0.5
                mid_z = (z_max + z_min) * 0.5

                # Set the axis limits
                ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
                ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
                ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
                ax_3d.set_box_aspect([1, 1, 1])  # Equal aspect ratio

                # Set labels and title
                ax_3d.set_xlabel('X')
                ax_3d.set_ylabel('Y')
                ax_3d.set_zlabel('Z')
                ax_3d.set_title('Estimated and Ground Truth Camera Poses')

                # To prevent multiple labels in the legend
                handles, labels = ax_3d.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_3d.legend(by_label.values(), by_label.keys())

            else:
                ax_3d.text(0.5, 0.5, 0.5, 'No Pose Data', horizontalalignment='center', verticalalignment='center')
                ax_3d.set_title('Estimated Camera Poses')
        else:
            print(f"Skipping 3D plot: insufficient inliers (num_inliers = {num_inliers}).")

        # Draw the figure
        plt.draw()

        # Print frame information
        print(f'Frame {frame_idx + 1}/{total_frames}')
        print(f'Number of Inliers: {num_inliers}')
        print(f'Total Matches: {total_matches}')
        print(f'Inlier Ratio: {inlier_ratio:.2f}')
        if mean_reproj_error is not None:
            print(f'Mean Reprojection Error: {mean_reproj_error:.2f}')
            print(f'Std Reprojection Error: {std_reproj_error:.2f}')
        if rotation_error_rad is not None:
            rotation_error_deg = np.degrees(rotation_error_rad)
            print(f'Rotation Error: {rotation_error_deg:.2f} degrees')
        if translation_error is not None:
            print(f'Translation Error: {translation_error:.4f}')
        # Matching confidence scores
        if len(mconf) > 0:
            print(f'Matching Confidence Scores - Min: {mconf.min():.2f}, Max: {mconf.max():.2f}, Mean: {mconf.mean():.2f}')
        else:
            print('No Matching Confidence Scores')
        print('Press \'n\' for next frame, \'p\' for previous frame, \'q\' to quit.')

    # Event handler for keyboard input
    def on_key(event):
        nonlocal frame_idx, stop

        if event.key == 'n':
            frame_idx += 1
            if frame_idx >= total_frames:
                frame_idx = total_frames - 1
            update_visualization()
        elif event.key == 'p':
            frame_idx -= 1
            if frame_idx < 0:
                frame_idx = 0
            update_visualization()
        elif event.key == 'q':
            stop = True
            plt.close(fig)

    # Connect the event handler
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial visualization
    update_visualization()

    # Show the figure
    plt.show()

if __name__ == '__main__':
    # Paths and parameters
    pose_file = '/home/runbk0401/SuperGluePretrainedNetwork/pose_estimation_research_54.json'  # Replace with your actual path
    image_dir = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/test/'  # Replace with your actual image directory
    gt_pose_file = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/test/viewpoint_GT.json'  # Replace with your ground truth JSON file

    # Camera intrinsic parameters (replace with your camera's parameters)
    focal_length_x = 2666.66666666666  # px
    focal_length_y = 2666.66666666666  # py
    cx = 639.5  # Principal point u0
    cy = 479.5  # Principal point v0

    # Assuming zero distortion coefficients
    distCoeffs = np.array([0, 0, 0, 0], dtype=np.float32)

    # Intrinsic camera matrix (K)
    K = np.array([
                [focal_length_x, 0, cx],
                [0, focal_length_y, cy],
                [0, 0, 1]
            ], dtype=np.float32)

    # Load the pose data
    pose_data = load_pose_data(pose_file)

    # Load the ground truth poses
    gt_poses = load_ground_truth_poses(gt_pose_file)

    # Prepare anchor images and their keypoints for each viewpoint
    anchor_images_info = {
        'front': {
            'image': cv2.imread('/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/61.png'),
            'keypoints_2D': [
                [558., 269.],
                [856., 277.],
                [536., 283.],
                [265., 449.],
                [225., 462.],
                [657., 477.],
                [1086., 480.],
                [217., 481.],
                [567., 483.],
                [653., 488.],
                [1084., 497.],
                [1084., 514.],
                [552., 551.],
                [640., 555.]
            ],
            'keypoints_3D': [
                [-0.81972, -0.3258, 5.28664],
                [-0.81972, 0.33329, 5.28664],
                [-0.60385, -0.3258, 5.28664],
                [-0.29107, -0.83895, 4.96934],
                [-0.04106, -0.83895, 4.995],
                [0.26951, 0.0838, 5.0646],
                [-0.29152, 0.84644, 4.96934],
                [0.01734, -0.83895, 4.9697],
                [0.31038, -0.0721, 5.05571],
                [0.31038, 0.07959, 5.05571],
                [-0.03206, 0.84644, 4.99393],
                [0.01734, 0.84644, 4.9697],
                [0.44813, -0.07631, 4.9631],
                [0.44813, 0.0838, 4.96381]
            ]
        },
        'back': {
            'image': cv2.imread('/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/70.png'),
            'keypoints_2D': [
                [860., 388.],
                [467., 394.],
                [881., 414.],
                [466., 421.],
                [668., 421.],
                [591., 423.],
                [1078., 481.],
                [195., 494.],
                [183., 540.],
                [626., 592.],
                [723., 592.]
            ],
            'keypoints_3D': [
                [-0.60385, -0.3258, 5.28664],
                [-0.60385, 0.33329, 5.28664],
                [-0.81972, -0.3258, 5.28664],
                [-0.81972, 0.33329, 5.28664],
                [0.26951, -0.07631, 5.0646],
                [0.26951, 0.0838, 5.0646],
                [-0.29297, -0.83895, 4.96825],
                [-0.04106, 0.84644, 4.995],
                [-0.29297, 0.84644, 4.96825],
                [-0.81973, 0.0838, 4.99302],
                [-0.81973, -0.07631, 4.99302]
            ]
        },
        'left': {
            'image': cv2.imread('/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/62.png'),
            'keypoints_2D': [
                [968., 313.],
                [1077., 315.],
                [1083., 376.],
                [713., 402.],
                [688., 412.],
                [827., 417.],
                [512., 436.],
                [472., 446.],
                [1078., 468.],
                [774., 492.],
                [740., 493.],
                [1076., 506.],
                [416., 511.],
                [452., 527.],
                [594., 594.],
                [560., 611.],
                [750., 618.]
            ],
            'keypoints_3D': [
                [-0.60385, -0.3258, 5.28664],
                [-0.81972, -0.3258, 5.28664],
                [-0.81972, 0.33329, 5.28664],
                [-0.04106, -0.83895, 4.995],
                [0.01551, -0.83895, 4.97167],
                [-0.29107, -0.83895, 4.96934],
                [0.26951, -0.07631, 5.0646],
                [0.31038, -0.07631, 5.05571],
                [-0.81972, 0.08616, 5.06584],
                [-0.26104, 0.0838, 5.00304],
                [-0.1986, 0.0838, 5.00304],
                [-0.81906, 0.0838, 4.99726],
                [0.42759, 0.0838, 4.94447],
                [0.35674, 0.0838, 4.91463],
                [-0.03206, 0.84644, 4.99393],
                [0.01551, 0.84644, 4.9717],
                [-0.29152, 0.84644, 4.96934]
            ]
        },
        'right': {
            'image': cv2.imread('/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/85.png'),
            'keypoints_2D': [
                [367., 300.],
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
                [386., 604.]
            ],
            'keypoints_3D': [
                [-0.60385, 0.33329, 5.28664],
                [-0.81972, 0.33329, 5.28664],
                [-0.60385, -0.3258, 5.28664],
                [-0.81972, -0.3258, 5.28664],
                [-0.04106, 0.84644, 4.995],
                [-0.29152, 0.84644, 4.96934],
                [0.26951, 0.0838, 5.0646],
                [0.26951, -0.07631, 5.0646],
                [-0.81972, -0.07867, 5.06584],
                [-0.04106, -0.07631, 4.995],
                [-0.1986, -0.07631, 5.00304],
                [0.44813, -0.07631, 4.96381],
                [-0.26104, -0.07631, 5.00304],
                [0.35674, -0.07631, 4.91463],
                [0.2674, -0.07631, 4.89973],
                [-0.04106, -0.83895, 4.995],
                [0.01551, -0.83895, 4.97167],
                [-0.29152, -0.83895, 4.96934]
            ]
        },
    }

    # Transform anchor keypoints_3D from Blender to OpenCV coordinate system
    T_blender_to_opencv = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ], dtype=np.float32)

    for viewpoint, info in anchor_images_info.items():
        keypoints_3D_blender = np.array(info['keypoints_3D'], dtype=np.float32)
        # Apply the transformation
        keypoints_3D_opencv = (T_blender_to_opencv @ keypoints_3D_blender.T).T
        # Update the keypoints_3D in the dictionary
        info['keypoints_3D'] = keypoints_3D_opencv.tolist()

    # Check that all anchor images were loaded
    for viewpoint, info in anchor_images_info.items():
        if info['image'] is None:
            raise ValueError(f"Failed to load anchor image for viewpoint '{viewpoint}'.")

    # Visualize pose and matches interactively
    visualize_pose_and_matches(
        pose_data, image_dir, anchor_images_info,
        K, distCoeffs, gt_poses
    )
