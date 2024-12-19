import cv2
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Ensure interactive mode is off for proper figure updates
plt.ioff()

# GT to World rotation matrix (identity in this case)
R_gt_to_world = np.array([
    [1,  0,  0],
    [0,  1,  0],
    [0,  0,  1]
], dtype=float)

def load_pose_data(pose_file):
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    return pose_data

def load_gt_data(gt_file):
    """
    Load and transform ground truth data from JSON file to World coordinate frame.
    Args:
        gt_file (str): Path to the GT JSON file.
    Returns:
        gt_camera_positions_world_aligned (np.ndarray): GT camera positions in World frame, aligned to origin.
        gt_target_positions_world_aligned (np.ndarray): GT target positions in World frame, aligned to origin.
    """
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
        
    # Extract GT positions
    gt_camera_positions = [np.array([item['camera_position']['X'], 
                                     item['camera_position']['Y'], 
                                     item['camera_position']['Z']]) for item in gt_data]
    gt_target_positions = [np.array([item['target_position']['X'], 
                                     item['target_position']['Y'], 
                                     item['target_position']['Z']]) for item in gt_data]
    
    # Convert GT positions to World coordinate frame
    gt_camera_positions_world = np.dot(np.array(gt_camera_positions), R_gt_to_world.T)
    gt_target_positions_world = np.dot(np.array(gt_target_positions), R_gt_to_world.T)
    
    # Compute translation to align the initial target position to origin
    initial_target_pos = gt_target_positions_world[0]
    translation = -initial_target_pos
    
    # Apply translation to align GT data
    gt_camera_positions_world_aligned = gt_camera_positions_world + translation
    gt_target_positions_world_aligned = gt_target_positions_world + translation
    
    return gt_camera_positions_world_aligned, gt_target_positions_world_aligned

def is_pose_valid(frame_data, position_threshold=3.0, reproj_error_threshold=5.0):
    if 'camera_position' not in frame_data or 'kf_translation_vector' not in frame_data:
        return False

    raw_position = np.array(frame_data['camera_position'])
    kf_position = np.array(frame_data['kf_translation_vector'])

    # Transform kf_position
    kf_position_transformed = kf_position.copy()
    kf_position_transformed[0], kf_position_transformed[1], kf_position_transformed[2] = kf_position[0], -kf_position[2], kf_position[1]

    position_diff = np.linalg.norm(raw_position - kf_position_transformed)
    mean_reproj_error = frame_data.get('mean_reprojection_error', float('inf'))

    if mean_reproj_error > reproj_error_threshold or position_diff > position_threshold:
        print('position_diff:', position_diff)
        return False

    # Update the kf_translation_vector with the transformed values
    frame_data['kf_translation_vector'] = kf_position_transformed.tolist()

    return True


def visualize_pose_and_matches(pose_data, gt_camera_positions, gt_target_positions, 
                               video_path, anchor_image_path, anchor_keypoints_2D, anchor_keypoints_3D, K):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error opening video file.')
        return

    anchor_image = cv2.imread(anchor_image_path)
    assert anchor_image is not None, 'Failed to load anchor image.'

    anchor_keypoints_2D = np.array(anchor_keypoints_2D)
    anchor_keypoints_3D = np.array(anchor_keypoints_3D)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2)
    ax_image = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[0, 1], projection='3d')

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    camera_positions = []
    kf_camera_positions = []

    def update_visualization():
        nonlocal frame_idx

        ax_image.clear()
        ax_3d.clear()

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f'Failed to read frame {frame_idx}.')
            return

        frame_data = next((fd for fd in pose_data if fd['frame'] == frame_idx + 1), None)
        if frame_data is None:
            print(f'Frame data for frame {frame_idx + 1} not found.')
            return

        mkpts0 = np.array(frame_data.get('mkpts0', []))
        mkpts1 = np.array(frame_data.get('mkpts1', []))
        mconf = np.array(frame_data.get('mconf', []))
        inliers = np.array(frame_data.get('inliers', []), dtype=int)

        anchor_image_gray = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Combine images
        height1, width1 = anchor_image_gray.shape
        height2, width2 = frame_gray.shape
        combined_image = np.zeros((max(height1, height2), width1 + width2), dtype=np.uint8)
        combined_image[:height1, :width1] = anchor_image_gray
        combined_image[:height2, width1:] = frame_gray

        mkpts1_plot = mkpts1.copy()
        if mkpts1_plot.ndim == 2 and mkpts1_plot.shape[0] > 0:
            mkpts1_plot[:, 0] += width1

        ax_image.imshow(combined_image, cmap='gray')
        ax_image.axis('off')
        for i in range(len(mkpts0)):
            x0, y0 = mkpts0[i]
            x1, y1 = mkpts1_plot[i]
            color = 'lime' if i in inliers else 'yellow'
            ax_image.plot([x0, x1], [y0, y1], color=color, linewidth=1)
        ax_image.set_title(f'Frame {frame_idx + 1}: Matches')

        if is_pose_valid(frame_data):
            raw_position = np.array(frame_data['camera_position'])
            kf_position = np.array(frame_data['kf_translation_vector'])
            camera_positions.append(raw_position)
            kf_camera_positions.append(kf_position)

            # Convert lists to numpy arrays for plotting
            camera_positions_array = np.array(camera_positions)
            kf_camera_positions_array = np.array(kf_camera_positions)

            # Plot GT Camera Trajectory
            ax_3d.plot(gt_camera_positions[:, 0], gt_camera_positions[:, 1], gt_camera_positions[:, 2],
                       c='blue', marker='.', label='GT Camera Trajectory')

            # Plot GT Target Trajectory
            ax_3d.plot(gt_target_positions[:, 0], gt_target_positions[:, 1], gt_target_positions[:, 2],
                       c='cyan', marker='x', label='GT Target Trajectory')

            # Plot Estimated Camera Trajectory
            ax_3d.plot(camera_positions_array[:, 0], camera_positions_array[:, 1], camera_positions_array[:, 2],
                       c='red', marker='o', label='Estimated Camera Trajectory')

            # Plot Kalman Filter Camera Trajectory
            if len(kf_camera_positions_array) > 0:
                ax_3d.plot(kf_camera_positions_array[:, 0], kf_camera_positions_array[:, 1], kf_camera_positions_array[:, 2],
                           c='green', marker='^', label='KF Camera Trajectory')

            # Setting labels
            ax_3d.set_xlabel('X (World)')
            ax_3d.set_ylabel('Y (World)')
            ax_3d.set_zlabel('Z (World)')

            # Setting a title
            ax_3d.set_title('3D Trajectories')

            # Handling the legend to avoid duplicate labels
            handles, labels = ax_3d.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax_3d.legend(by_label.values(), by_label.keys())

            # Optional: Adjust the view angle for better visualization
            ax_3d.view_init(elev=20., azim=30)

        plt.draw()
        plt.pause(0.001)

    def on_key(event):
        nonlocal frame_idx
        if event.key == 'n':
            frame_idx = min(frame_idx + 1, total_frames - 1)
            update_visualization()
        elif event.key == 'p':
            frame_idx = max(frame_idx - 1, 0)
            update_visualization()
        elif event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_visualization()
    plt.show()
    cap.release()


if __name__ == '__main__':
    # Paths and parameters
    pose_file = '/home/runbk0401/SuperGluePretrainedNetwork/pose_estimation_research_139.json'  # Replace with your actual path
    video_path = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/video/20241217/Opti_Test1.mp4'          # Replace with your actual path
    anchor_image_path = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/realAnchor.png'      # Replace with your actual path
    gt_file = '/home/runbk0401/SuperGluePretrainedNetwork/GT/extracted_positions(1).json'

    # pose_file = '/home/runbk0401/SuperGluePretrainedNetwork/pose_estimation_research_123.json'  # Replace with your actual path
    # video_path = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/video/20241218/20241218_HD.mp4'          # Replace with your actual path
    # anchor_image_path = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/video/20241218/Box_Anchor/Opti_Box_Anchor.png'      # Replace with your actual path
    # gt_file = '/home/runbk0401/SuperGluePretrainedNetwork/GT/20241218/20241218_extracted_positions.json'

    ####################################################################################################
    # # Real calibration values from XML (perspectiveProjWithoutDistortion)
    # focal_length_x = 1526.22  # px
    # focal_length_y = 1531.18  # py
    # cx = 637.98  # Principal point u0
    # cy = 416.04  # Principal point v0

    # # Distortion coefficients from "perspectiveProjWithDistortion" model in the XML
    # distCoeffs = np.array([0.2728747755008597, -0.25885103136641374, 0, 0], dtype=np.float32)
    
    ####################################################################################################

    # Real calibration Phone (perspectiveProjWithoutDistortion)
    focal_length_x = 1195.08491  # px
    focal_length_y = 1354.35538  # py
    cx = 581.022033  # Principal point u0
    cy = 571.458522  # Principal point v0

    # Distortion coefficients from "perspectiveProjWithDistortion" model in the XML
    distCoeffs = np.array([0.10058526, 0.4507094, 0.13687279, -0.01839536, 0.13001669], dtype=np.float32)
   
    ####################################################################################################

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

    # Known 2D and 3D correspondences (pre-defined)
    anchor_keypoints_2D = np.array([
            [563, 565], [77, 582], [515, 318], [606, 317], [612, 411],
            [515, 414], [420, 434], [420, 465], [618, 455], [500, 123],
            [418, 153], [417, 204], [417, 243], [502, 279], [585, 240],
            [289, 26], [322, 339], [349, 338], [349, 374], [321, 375],
            [390, 349], [243, 462], [367, 550], [368, 595], [383, 594],
            [386, 549], [779, 518], [783, 570]
        ], dtype=np.float32)
    # This is a test I changed it into GT coordinate frame 20241218 #########################################
    anchor_keypoints_3D = np.array([
            [0.03, -0.05, -0.165],
            [-0.190, -0.050, -0.165],
            [0.010, -0.0, -0.025],
            [0.060, -0.0, -0.025],
            [0.06, -0.0, -0.080],
            [0.010, -0.0, -0.080],
            [-0.035, -0.0, -0.087],
            [-0.035, -0.0, -0.105],
            [0.065, -0.0, -0.105],
            [0.0, 0.0, 0.045],
            [-0.045, 0.0,0.078 ],
            [-0.045, 0.0, 0.046],
            [-0.045, 0.0, 0.023],
            [0.0, -0.0, 0.0],
            [0.045, 0.0, 0.022],
            [-0.120, 0.0, 0.160],
            [-0.095, -0.0,-0.035],
            [-0.080, -0.0, -0.035],
            [-0.080, -0.0, -0.055],
            [-0.095, -0.0, -0.055],
            [-0.050, -0.010, -0.040],
            [-0.135, -0.0, -0.1],
            [-0.060, -0.050,-0.155],
            [-0.060, -0.050,-0.175],
            [-0.052, -0.050, -0.175],
            [-0.052, -0.050, -0.155],
            [0.135, -0.050, -0.147],
            [0.135, -0.050, -0.172]
        ], dtype=np.float32)
    ###################################################
    # # Provided 2D and 3D keypoints for the anchor image: Box
    # anchor_keypoints_2D = np.array([
    #     [780, 216],  
    # [464, 111],  
    # [258, 276], 
    # [611, 538],  
    # [761, 324],
    # [644, 168],
    # [479, 291] ,
    # [586, 149] ,
    # [550, 182] ,
    # [610, 202] ,
    # [361, 193] ,
    # [319, 298] ,
    # [344, 418] ,
    # [440, 460] ,
    # [502, 489] ,
    # [496, 372]
        
    # ], dtype=np.float32)

    # anchor_keypoints_3D = np.array([
    #         [0.049, 0.045, 0],     
    # [-0.051, 0.045, 0],    
    # [-0.051, -0.044, 0],    
    # [0.049, -0.044, -0.04],     
    # [0.049, 0.045, 0],
    # [0.01, 0.045, 0],
    # [-0.003, -0.023, 0],
    # [-0.001, 0.045, 0],
    # [-0.001, 0.025, 0],
    # [0.001, 0.025, 0],
    # [-0.051, -0.005, 0],
    # [-0.035, -0.044, 0],
    # [-0.035, -0.044, -0.04],
    # [-0.002, -0.044, -0.04], 
    # [0.017, -0.044, -0.04],
    # [0.017, -0.044, 0.0]
    #     ], dtype=np.float32) 
     
    ####################################################3#######################3
    #  # Provided 2D and 3D keypoints for the anchor image : TAIL
    # anchor_keypoints_2D = np.array([
    #     [563, 565], 
    #     [77, 582], 
    #     [515, 318], 
    #     [606, 317], 
    #     [612, 411],
    #     [515, 414], 
    #     [420, 434], 
    #     [420, 465], 
    #     [618, 455], 
    #     [500, 123], 
    #     [418, 153], 
    #     [417, 204], 
    #     [417, 243], 
    #     [502, 279],
    #     [585, 240],  
    #     [289, 26],  
    #     [322, 339], 
    #     [349, 338], 
    #     [349, 374], 
    #     [321, 375],
    #     [390, 349], 
    #     [243, 462], 
    #     [367, 550], 
    #     [368, 595], 
    #     [383, 594],
    #     [386, 549], 
    #     [779, 518], 
    #     [783, 570]
        
    # ], dtype=np.float32)

    # anchor_keypoints_3D = np.array([
    #         [0.03, -0.05, -0.165],
    #         [-0.190, -0.050, -0.165],
    #         [0.010, -0.0, -0.025],
    #         [0.060, -0.0, -0.025],
    #         [0.06, -0.0, -0.080],
    #         [0.010, -0.0, -0.080],
    #         [-0.035, -0.0, -0.087],
    #         [-0.035, -0.0, -0.105],
    #         [0.065, -0.0, -0.105],
    #         [0.0, 0.0, 0.045],
    #         [-0.045, 0.0,0.078 ],
    #         [-0.045, 0.0, 0.046],
    #         [-0.045, 0.0, 0.023],
    #         [0.0, -0.0, 0.0],
    #         [0.045, 0.0, 0.022],
    #         [-0.120, 0.0, 0.160],
    #         [-0.095, -0.0,-0.035],
    #         [-0.080, -0.0, -0.035],
    #         [-0.080, -0.0, -0.055],
    #         [-0.095, -0.0, -0.055],
    #         [-0.050, -0.010, -0.040],
    #         [-0.135, -0.0, -0.1],
    #         [-0.060, -0.050,-0.155],
    #         [-0.060, -0.050,-0.175],
    #         [-0.052, -0.050, -0.175],
    #         [-0.052, -0.050, -0.155],
    #         [0.135, -0.050, -0.147],
    #         [0.135, -0.050, -0.172]
    #     ], dtype=np.float32)

    #############################################################################

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
    gt_camera_positions, gt_target_positions = load_gt_data(gt_file)
    visualize_pose_and_matches(pose_data, gt_camera_positions, gt_target_positions, 
                               video_path, anchor_image_path, anchor_keypoints_2D, anchor_keypoints_3D, K)
