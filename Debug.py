import cv2
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Ensure interactive mode is off for proper figure updates
plt.ioff()

def load_pose_data(pose_file):
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    return pose_data

def visualize_pose_and_matches(pose_data, video_path, anchor_image_path, anchor_keypoints_2D, anchor_keypoints_3D, K, distCoeffs):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error opening video file.')
        return

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

    # Initialize variables
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    camera_positions = []

    # Event handling variables
    next_frame = True
    stop = False

    def update_visualization():
        nonlocal frame_idx, next_frame, stop

        # Clear axes
        ax_image.clear()
        ax_3d.clear()

        # Set OpenCV video capture to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f'Failed to read frame {frame_idx}.')
            stop = True
            return

        # Find the frame data
        frame_data = next((fd for fd in pose_data if fd['frame'] == frame_idx + 1), None)
        if frame_data is None:
            print(f'Frame data for frame {frame_idx + 1} not found.')
            stop = True
            return

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
                        ax_image.scatter(x + width1, y, s=50, c='green', marker='x', label='Inliers Reprojection' if i == 0 else "")
                    else:
                        ax_image.scatter(x + width1, y, s=50, c='red', marker='x', label='Outliers Reprojection' if i == 0 else "")

                # Avoid duplicate labels
                handles, labels = ax_image.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_image.legend(by_label.values(), by_label.keys())

                ax_image.set_title(f'Reprojected Points (Frame {frame_idx + 1})')

            else:
                print('Invalid rotation matrix or translation vector dimensions.')
                return

        # Only update 3D plot if number of inliers is greater than 5
        if num_inliers > 4:
            if 'camera_position' in frame_data:
                camera_position = np.array(frame_data['camera_position'])
                camera_positions.append(camera_position)

                # Plot the anchor 3D points
                ax_3d.scatter(anchor_keypoints_3D[:, 0], anchor_keypoints_3D[:, 1], anchor_keypoints_3D[:, 2],
                            c='b', marker='o', label='Anchor 3D Points')

                # Plot the camera positions
                camera_positions_np = np.array(camera_positions)
                ax_3d.plot(camera_positions_np[:, 0], camera_positions_np[:, 1], camera_positions_np[:, 2],
                        c='r', marker='o', label='Camera Trajectory')

                # Plot current camera position
                ax_3d.scatter(camera_position[0], camera_position[1], camera_position[2],
                            c='g', marker='^', s=100, label='Current Camera Position')

                # Extract viewing direction
                # Forward vector in world coordinates
                forward_vector = rotation_matrix[:, 2]  # Third column of R
                # Define the length of the viewing direction arrow
                arrow_length = 0.05  # Adjust as needed based on your scale

                # Starting point (camera position)
                start = camera_position

                # Ending point (camera position + forward_vector scaled)
                end = start + forward_vector * arrow_length

                # Plot the viewing direction as an arrow (using plot for simplicity)
                ax_3d.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                        c='k', linewidth=2, label='Viewing Direction')

                # Define frustum dimensions (example values)
                frustum_length = 0.1  # Adjust based on your scene's scale
                frustum_width = 0.05
                frustum_height = 0.05

                # Define frustum corners in camera coordinates
                frustum_corners = np.array([
                    [0, 0, 0],  # Camera origin
                    [frustum_width, frustum_height, frustum_length],
                    [-frustum_width, frustum_height, frustum_length],
                    [-frustum_width, -frustum_height, frustum_length],
                    [frustum_width, -frustum_height, frustum_length],
                    [frustum_width, frustum_height, frustum_length]
                ])

                # Transform frustum corners to world coordinates
                frustum_world = (rotation_matrix @ frustum_corners.T).T + camera_position

                # Plot frustum edges
                ax_3d.plot([frustum_world[0,0], frustum_world[1,0]],
                          [frustum_world[0,1], frustum_world[1,1]],
                          [frustum_world[0,2], frustum_world[1,2]], c='k')
                ax_3d.plot([frustum_world[0,0], frustum_world[2,0]],
                          [frustum_world[0,1], frustum_world[2,1]],
                          [frustum_world[0,2], frustum_world[2,2]], c='k')
                ax_3d.plot([frustum_world[0,0], frustum_world[3,0]],
                          [frustum_world[0,1], frustum_world[3,1]],
                          [frustum_world[0,2], frustum_world[3,2]], c='k')
                ax_3d.plot([frustum_world[0,0], frustum_world[4,0]],
                          [frustum_world[0,1], frustum_world[4,1]],
                          [frustum_world[0,2], frustum_world[4,2]], c='k')
                ax_3d.plot([frustum_world[1,0], frustum_world[5,0]],
                          [frustum_world[1,1], frustum_world[5,1]],
                          [frustum_world[1,2], frustum_world[5,2]], c='k')
                ax_3d.plot([frustum_world[2,0], frustum_world[5,0]],
                          [frustum_world[2,1], frustum_world[5,1]],
                          [frustum_world[2,2], frustum_world[5,2]], c='k')
                ax_3d.plot([frustum_world[3,0], frustum_world[5,0]],
                          [frustum_world[3,1], frustum_world[5,1]],
                          [frustum_world[3,2], frustum_world[5,2]], c='k')
                ax_3d.plot([frustum_world[4,0], frustum_world[5,0]],
                          [frustum_world[4,1], frustum_world[5,1]],
                          [frustum_world[4,2], frustum_world[5,2]], c='k')

                # Set labels and limits
                ax_3d.set_xlabel('X')
                ax_3d.set_ylabel('Y')
                ax_3d.set_zlabel('Z')
                ax_3d.set_title('Estimated Camera Poses')

                # To prevent multiple labels in the legend
                handles, labels = ax_3d.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_3d.legend(by_label.values(), by_label.keys())

                # Equal aspect ratio
                if camera_positions_np.shape[0] > 1:
                    ax_3d.set_box_aspect([
                        np.ptp(camera_positions_np[:, 0]),
                        np.ptp(camera_positions_np[:, 1]),
                        np.ptp(camera_positions_np[:, 2])
                    ])
                else:
                    ax_3d.set_box_aspect([1,1,1])

            else:
                ax_3d.text(0.5, 0.5, 0.5, 'No Pose Data', horizontalalignment='center', verticalalignment='center')
                ax_3d.set_title('Estimated Camera Poses')
        else:
            print(f"Skipping 3D plot: insufficient inliers (num_inliers = {num_inliers}).")

        # Draw the figure
        plt.draw()
        plt.pause(0.001)

        # Print frame information
        print(f'Frame {frame_idx + 1}/{total_frames}')
        print(f'Number of Inliers: {num_inliers}')
        print(f'Total Matches: {total_matches}')
        print(f'Inlier Ratio: {inlier_ratio:.2f}')
        if mean_reproj_error is not None:
            print(f'Mean Reprojection Error: {mean_reproj_error:.2f}')
            print(f'Std Reprojection Error: {std_reproj_error:.2f}')
        # Matching confidence scores
        if len(mconf) > 0:
            print(f'Matching Confidence Scores - Min: {mconf.min():.2f}, Max: {mconf.max():.2f}, Mean: {mconf.mean():.2f}')
        else:
            print('No Matching Confidence Scores')
        print('Press \'n\' for next frame, \'p\' for previous frame, \'q\' to quit.')

    # Event handler for keyboard input
    def on_key(event):
        nonlocal frame_idx, next_frame, stop

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

    cap.release()

if __name__ == '__main__':
    # Paths and parameters
    pose_file = '/home/runbk0401/SuperGluePretrainedNetwork/pose_estimation_research.json'  # Replace with your actual path
    video_path = '/home/runbk0401/SuperGluePretrainedNetwork/Ruun_code/20241002_output_translation.avi'          # Replace with your actual path
    anchor_image_path = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/boxcraft/frame_00001.png'       # Replace with your actual path


    # Real calibration values from XML (perspectiveProjWithoutDistortion)
    focal_length_x = 778.38449164772408  # px
    focal_length_y = 780.92121918822045  # py
    cx = 336.10046045116735  # Principal point u0
    cy = 258.73402363943103  # Principal point v0

    # Distortion coefficients from "perspectiveProjWithDistortion" model in the XML
    distCoeffs = np.array([0.2728747755008597, -0.25885103136641374, 0, 0], dtype=np.float32)

    # Intrinsic camera matrix (K)
    K = np.array([
                [focal_length_x, 0, cx],
                [0, focal_length_y, cy],
                [0, 0, 1]
            ], dtype=np.float32)

    # Provide your anchor keypoints
    anchor_keypoints_2D = [
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
    ]

    anchor_keypoints_3D = [
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
    ]

    # Load the pose data
    pose_data = load_pose_data(pose_file)

    # Visualize pose and matches interactively
    visualize_pose_and_matches(
        pose_data, video_path, anchor_image_path, anchor_keypoints_2D, anchor_keypoints_3D, K, distCoeffs
    )
