import json
import numpy as np
import matplotlib.pyplot as plt

# Load the pose data
def load_pose_data(pose_file):
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    return pose_data

# Analyze pose stability over time
def analyze_pose_stability(pose_data):
    frames = []
    camera_positions = []

    for frame_data in pose_data:
        if 'camera_position' in frame_data:
            frames.append(frame_data['frame'])
            camera_position = np.array(frame_data['camera_position'])
            camera_positions.append(camera_position)

    camera_positions = np.array(camera_positions)
    frames = np.array(frames)

    # Compute velocities
    velocities = np.diff(camera_positions, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)

    # Compute accelerations
    accelerations = np.diff(velocities, axis=0)
    accelerations_magnitude = np.linalg.norm(accelerations, axis=1)

    # Plot speeds over time
    plt.figure(figsize=(10, 6))
    plt.plot(frames[1:], speeds, marker='o', label='Speed')
    plt.xlabel('Frame')
    plt.ylabel('Speed (units/frame)')
    plt.title('Camera Speed over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot accelerations over time
    plt.figure(figsize=(10, 6))
    plt.plot(frames[2:], accelerations_magnitude, marker='o', label='Acceleration')
    plt.xlabel('Frame')
    plt.ylabel('Acceleration (units/frame^2)')
    plt.title('Camera Acceleration over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Path to your pose estimation results
    pose_file = 'path_to_your_pose_estimation.json'

    # Load the pose data
    pose_data = load_pose_data(pose_file)

    # Analyze pose stability
    analyze_pose_stability(pose_data)
