import cv2
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Ensure interactive mode is off for proper figure updates
plt.ioff()

def load_pose_data(pose_file):
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    return pose_data

def visualize_pose_and_matches(pose_data, video_path, anchor_image_path, anchor_keypoints_2D, anchor_keypoints_3D, K):
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
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2)
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
        ax_image.scatter(mkpts0_plot[:, 0], mkpts0_plot[:, 1], s=10, c='cyan', marker='o')
        ax_image.scatter(mkpts1_plot[:, 0], mkpts1_plot[:, 1], s=10, c='orange', marker='o')

        ax_image.set_title(f'Frame {frame_idx + 1}: Matches (Confidence Colored)')

        # Only update 3D plot if number of inliers is greater than 5
        if num_inliers > 3: #and inlier_ratio > 0.5:
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

                # Set labels and limits
                ax_3d.set_xlabel('X')
                ax_3d.set_ylabel('Y')
                ax_3d.set_zlabel('Z')
                ax_3d.set_title('Estimated Camera Poses')
                ax_3d.legend()

                # Equal aspect ratio
                #ax_3d.set_box_aspect([np.ptp(a) for a in [camera_positions_np[:, 0], camera_positions_np[:, 1], camera_positions_np[:, 2]]])

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
    pose_file = '/home/runbk0401/SuperGluePretrainedNetwork/pose_estimation_research_41.json'  # Replace with your actual path
    video_path = '/home/runbk0401/SuperGluePretrainedNetwork/Ruun_code/steady.avi'          # Replace with your actual path
    anchor_image_path = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/realAnchor.png'       # Replace with your actual path
    

    # Real calibration values from XML (perspectiveProjWithoutDistortion)
    focal_length_x = 1526.22  # px
    focal_length_y = 1531.18  # py
    cx = 637.98  # Principal point u0
    cy = 416.04  # Principal point v0

    # Distortion coefficients from "perspectiveProjWithDistortion" model in the XML
    distCoeffs = np.array([0.2728747755008597, -0.25885103136641374, 0, 0], dtype=np.float32)

    # Intrinsic camera matrix (K)
    K = np.array([
                [focal_length_x, 0, cx],
                [0, focal_length_y, cy],
                [0, 0, 1]
            ], dtype=np.float32)

    # # Provide your anchor keypoints
    # anchor_keypoints_2D = [
    #    [494, 605],
    #     [566, 641], 
    #     [603, 557], 
    #     [539, 515], 
    #     [512, 345],
    #     [834, 491], 
    #     [927, 217], 
    #     [707, 44], 
    #     [752, 214], 
    #     [851, 173],
    #     [1069, 509],
    #     [1016, 639], 
    #     [413, 209], 
    #     [325, 298], 
    #     [743, 343],
    #     [541, 407], 
    #     [676, 382]
        
    # ]

    # anchor_keypoints_3D = [
    #     [0.054, -0.007, 0.008],
    #     [0.054, 0.007, 0.008],
    #     [0.038, 0.007, 0.015],
    #     [0.038, -0.007, 0.015],
    #     [0.005, -0.038, 0.008],
    #     [0.005, 0.038, 0.008],
    #     [-0.044, 0.034, 0.035],
    #     [-0.064, -0.034, 0.035],
    #     [-0.044, -0.007, 0.015],
    #     [-0.064, 0.007, 0.015],
    #     [-0.015, 0.080, 0.008],
    #     [0.015, 0.080, 0.008],
    #     [-0.015, -0.080, 0.008],
    #     [0.015, -0.008, 0.008],
    #     [-0.014, 0.007, 0.015],
    #     [0.015, 0.023, 0.008],
    #     [0.000, 0.000, 0.015]
    # ]

     # Provided 2D and 3D keypoints for the anchor image
    anchor_keypoints_2D = np.array([
        [563, 565], 
        [77, 582], 
        [515, 318], 
        [606, 317], 
        [612, 411],
        [515, 414], 
        [420, 434], 
        [420, 465], 
        [618, 455], 
        [500, 123], 
        [418, 153], 
        [417, 204], 
        [417, 243], 
        [502, 279],
        [585, 240],  
        [289, 26],  
        [322, 339], 
        [349, 338], 
        [349, 374], 
        [321, 375],
        [390, 349], 
        [243, 462], 
        [367, 550], 
        [368, 595], 
        [383, 594],
        [386, 549], 
        [779, 518], 
        [783, 570]
        
    ], dtype=np.float32)

    anchor_keypoints_3D = np.array([
        [0.03, -0.165, 0.05],
        [-0.190, -0.165, 0.050],
        [0.010, -0.025, 0.0],
        [0.060, -0.025, 0.0],
        [0.06, -0.080, 0.0],
        [0.010, -0.080, 0.0],
        [-0.035, -0.087, 0.0],
        [-0.035, -0.105, 0.0],
        [0.065, -0.105, 0.0],
        [0.0, 0.045, 0.0],
        [-0.045, 0.078, 0.0],
        [-0.045, 0.046, 0.0],
        [-0.045, 0.023, 0.0],
        [0.0, -0.0, 0.0],
        [0.045, 0.022, 0.0],
        [-0.120, 0.160, 0.0],
        [-0.095, -0.035,0.0],
        [-0.080, -0.035, 0.0],
        [-0.080, -0.055, 0.0],
        [-0.095, -0.055, 0.0],
        [-0.050, -0.040, 0.010],
        [-0.135, -0.100, 0.0],
        [-0.060, -0.155, 0.050],
        [-0.060, -0.175, 0.050],
        [-0.052, -0.175, 0.050],
        [-0.052, -0.155, 0.050],
        [0.135, -0.147, 0.050],
        [0.135, -0.172, 0.050]

    ], dtype=np.float32)


    # # Provided 2D and 3D keypoints for the anchor image
    # anchor_keypoints_2D = np.array([
    #     [563, 565], 
    #     [77, 582], 
    #     [515, 318], 
         
    #     [612, 411],
    #     [515, 414],
    #     [420, 434], 
       
    #     [618, 455], 
    #     [500, 123], 
    #     [418, 153],
         
    #     [417, 243], 
       
    #     [585, 240],  
    #     [289, 26],  
    #     [322, 339], 
         
        
    #     [390, 349],

    #     [243, 462], 
    #     [367, 550], 
        
    #     [779, 518], 
    #     [783, 570]
        
    # ], dtype=np.float32)

    # anchor_keypoints_3D = np.array([
    #     [0.03, -0.165, 0.05],
    #     [-0.190, -0.165, 0.050],
    #     [0.010, -0.025, 0.0],
        
    #     [0.06, -0.080, 0.0],
    #     [0.010, -0.080, 0.0],
    #     [-0.035, -0.087, 0.0],
       
    #     [0.065, -0.105, 0.0],
    #     [0.0, 0.045, 0.0],
    #     [-0.045, 0.078, 0.0],
        
    #     [-0.045, 0.023, 0.0],
       
    #     [0.045, 0.022, 0.0],
    #     [-0.120, 0.160, 0.0],
    #     [-0.095, -0.035,0.0],
        
       
    #     [-0.050, -0.040, 0.010],

    #     [-0.135, -0.100, 0.0],
    #     [-0.060, -0.155, 0.050],
        
    #     [0.135, -0.147, 0.050],
    #     [0.135, -0.172, 0.050]

    # ], dtype=np.float32)

    # Load the pose data
    pose_data = load_pose_data(pose_file)

    # Visualize pose and matches interactively
    visualize_pose_and_matches(
        pose_data, video_path, anchor_image_path, anchor_keypoints_2D, anchor_keypoints_3D, K
    )