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
    for frame in gt_data['frames']:
        image_name = Path(frame['image']).name  # Extract filename only
        for obj in frame['object_poses']:
            if obj['name'] == 'Camera':
                pose_matrix = np.array(obj['pose'], dtype=np.float32)
                gt_poses[image_name] = pose_matrix
                break  # Assume only one camera pose per frame
    return gt_poses

def visualize_pose_and_matches(pose_data, image_dir, anchor_image_path,
                               anchor_keypoints_2D, anchor_keypoints_3D, K, distCoeffs, gt_poses):
    # Get list of image paths
    image_paths = sorted(list(Path(image_dir).glob('*.png')))
    if not image_paths:
        print('No images found in the directory.')
        return

    total_frames = len(image_paths)
    frame_idx = 0  # Initialize frame counter
    camera_positions_estimated = []
    camera_positions_ground_truth = []

    # Load the anchor image
    anchor_image = cv2.imread(anchor_image_path)
    assert anchor_image is not None, 'Failed to load anchor image.'

    # Convert anchor keypoints to numpy arrays
    anchor_keypoints_2D = np.array(anchor_keypoints_2D)
    anchor_keypoints_3D = np.array(anchor_keypoints_3D)

    # Prepare figure for 3D pose visualization
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    ax_image = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[0, 1], projection='3d')

    # Event handling variables
    stop = False

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
        # Replace '0001' with '0000' in the filename
        gt_frame_name = frame_name.replace('0001', '0000', 1)  # Replace only the first occurrence

        print(f'Current frame_name: {frame_name}')
        print(f'Adjusted ground truth frame name: {gt_frame_name}')
        print(f'Available ground truth frames (first 5): {list(gt_poses.keys())[:5]}')

        # Find the frame data
        frame_data = next((fd for fd in pose_data if fd.get('image_name') == frame_name
                           or fd.get('frame') == frame_idx + 1), None)
        if frame_data is None:
            print(f'Frame data for image {frame_name} not found.')
            return

        # Get ground truth pose for this frame
        gt_pose = gt_poses.get(gt_frame_name)
        if gt_pose is not None:
            gt_rotation_matrix = np.array(gt_pose[:3, :3])
            gt_translation_vector = np.array(gt_pose[:3, 3])
            gt_camera_position = -gt_rotation_matrix.T @ gt_translation_vector
            
            # Print the ground truth rotation and translation
            print(f"Ground Truth Rotation Matrix for {gt_frame_name}:\n{gt_rotation_matrix}")
            print(f"Ground Truth Translation Vector for {gt_frame_name}:\n{gt_translation_vector}")
            print(f"Ground Truth Camera Position (World Coordinates): {gt_camera_position}")

            # Apply coordinate system transformation to ground truth
            #gt_camera_position[2] *= -1  # Flip Z-axis
            #gt_rotation_matrix[2, :] *= -1
            #gt_rotation_matrix[:, 2] *= -1
        else:
            gt_camera_position = None
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
        if num_inliers > 4 and 'rotation_matrix' in frame_data and 'translation_vector' in frame_data:
            rotation_matrix = np.array(frame_data.get('rotation_matrix', []))
            translation_vector = np.array(frame_data.get('translation_vector', []))
            

            if rotation_matrix.size == 9 and translation_vector.size == 3:
                # Reshape rotation matrix to 3x3 if necessary
                rotation_matrix = rotation_matrix.reshape(3, 3)
                translation_vector = translation_vector.reshape(3, 1)

                # Convert rotation matrix to rotation vector using Rodrigues
                rvec, _ = cv2.Rodrigues(rotation_matrix)

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
                camera_position_estimated = np.array(frame_data['camera_position'])
                camera_positions_estimated.append(camera_position_estimated)

                # Plot the anchor 3D points
                ax_3d.scatter(anchor_keypoints_3D[:, 0], anchor_keypoints_3D[:, 1], anchor_keypoints_3D[:, 2],
                              c='b', marker='o', label='Anchor 3D Points')

                # Plot the estimated camera positions
                camera_positions_est_np = np.array(camera_positions_estimated)
                ax_3d.plot(camera_positions_est_np[:, 0], camera_positions_est_np[:, 1], camera_positions_est_np[:, 2],
                           c='r', marker='o', label='Estimated Camera Trajectory')

                # Plot current estimated camera position
                ax_3d.scatter(camera_position_estimated[0], camera_position_estimated[1], camera_position_estimated[2],
                              c='r', marker='^', s=100, label='Current Estimated Position')

                # Extract viewing direction for estimated pose
                forward_vector_estimated = rotation_matrix[:, 2]  # Third column of R
                arrow_length = 1#0.1  # Adjust as needed based on your scale

                # Plot viewing direction for estimated pose
                start_est = camera_position_estimated
                ax_3d.quiver(
                    start_est[0], start_est[1], start_est[2],
                    forward_vector_estimated[0], forward_vector_estimated[1], forward_vector_estimated[2],
                    length=arrow_length, color='blue', normalize=True, label='Estimated Viewing Direction'
                )

                # Plot ground truth camera positions
                if gt_camera_position is not None:
                    camera_positions_ground_truth.append(gt_camera_position)
                    camera_positions_gt_np = np.array(camera_positions_ground_truth)
                    ax_3d.plot(camera_positions_gt_np[:, 0], camera_positions_gt_np[:, 1], camera_positions_gt_np[:, 2],
                               c='g', marker='o', label='Ground Truth Trajectory')

                    # Plot current ground truth camera position
                    ax_3d.scatter(gt_camera_position[0], gt_camera_position[1], gt_camera_position[2],
                                  c='g', marker='^', s=100, label='Current Ground Truth Position')

                    # Extract viewing direction for ground truth pose
                    forward_vector_gt = gt_rotation_matrix[:, 2]

                    # Plot viewing direction for ground truth pose
                    start_gt = gt_camera_position
                    ax_3d.quiver(
                        start_gt[0], start_gt[1], start_gt[2],
                        forward_vector_gt[0], forward_vector_gt[1], forward_vector_gt[2],
                        length=arrow_length, color='magenta', normalize=True, label='Ground Truth Viewing Direction'
                    )

                else:
                    print(f'Ground truth pose for image {gt_frame_name} not found.')

                # Combine all positions to get the full range
                all_x = np.concatenate((anchor_keypoints_3D[:, 0], camera_positions_est_np[:, 0]))
                all_y = np.concatenate((anchor_keypoints_3D[:, 1], camera_positions_est_np[:, 1]))
                all_z = np.concatenate((anchor_keypoints_3D[:, 2], camera_positions_est_np[:, 2]))

                if gt_camera_position is not None:
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
        # Remove plt.pause() to avoid RuntimeError

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
    pose_file = '/home/runbk0401/SuperGluePretrainedNetwork/Pose_estimation_JSON/pose_estimation_research_41.json'  # Replace with your actual path
    image_dir = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/test/rotated/'  # Replace with your actual image directory
    anchor_image_path = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/w70.png'  # Replace with your actual path
    gt_pose_file = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/test/rotated/viewpoint_GT_rotate.json'  # Replace with your ground truth JSON file

    # Camera intrinsic parameters (replace with your camera's parameters)
    #focal_length_x = 1111.11111  # px
    #focal_length_y = 1111.11111  # py

    #focal_length_x = 888.88888  # px
    #focal_length_y = 888.88888  # py

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

    # # Provide your anchor keypoints
    # anchor_keypoints_2D = [
    #     [342., 193.],
    #     [305., 200.],
    #     [483., 212.],
    #     [447., 218.],
    #     [426., 264.],
    #     [150., 277.],
    #     [87., 289.],
    #     [209., 315.],
    #     [293., 343.],
    #     [131., 349.],
    #     [516., 353.],
    #     [167., 360.],
    #     [459., 373.]
    # ]

    # anchor_keypoints_3D = [
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
    # ]

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
    

    # Load the pose data
    pose_data = load_pose_data(pose_file)

    # Load the ground truth poses
    gt_poses = load_ground_truth_poses(gt_pose_file)

    # Visualize pose and matches interactively
    visualize_pose_and_matches(
        pose_data, image_dir, anchor_image_path,
        anchor_keypoints_2D, anchor_keypoints_3D, K, distCoeffs, gt_poses
    )
