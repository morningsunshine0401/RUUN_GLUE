import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the saved pose estimations (multiple frames)
def load_pose_data(pose_file):
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    return pose_data

# Plot the camera positions and orientations in 3D space for all frames, including anchor points
def plot_camera_poses_and_anchor(pose_data, anchor_points_3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    camera_positions = []

    # Loop through each frame's pose data
    for frame_data in pose_data:
        camera_position = np.array(frame_data['camera_position'])
        camera_positions.append(camera_position)

        # Plot the camera position as a red dot
        ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='r', label=f"Camera Position {frame_data['frame']}" if frame_data == pose_data[0] else None)

    # Convert camera positions to a numpy array for easier plotting
    camera_positions = np.array(camera_positions)

    # Plot the trajectory (lines connecting the camera positions)
    ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], color='black', linestyle='--', label='Camera Trajectory')

    # Plot the anchor 3D points as blue dots
    anchor_points_3D = np.array(anchor_points_3D)
    ax.scatter(anchor_points_3D[:, 0], anchor_points_3D[:, 1], anchor_points_3D[:, 2], c='b', label='Anchor 3D Points')

    # Set labels and limits for better visualization
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add a legend
    ax.legend()

    plt.show()

if __name__ == '__main__':
    # Specify the path to the saved pose data
    #pose_file = '/home/runbk0401/SuperGluePretrainedNetwork/RuunPoseResult/20241002_result/pose_estimation_3.json'
    pose_file = '/home/runbk0401/SuperGluePretrainedNetwork/pose_estimation_research_4.json'
    # Load the pose data (multiple frames)
    pose_data = load_pose_data(pose_file)

    # Define the provided anchor 3D points (replace with your actual anchor 3D points)
    anchor_points_3D = [
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
    
    # Plot the camera poses and anchor points in 3D
    plot_camera_poses_and_anchor(pose_data, anchor_points_3D)

