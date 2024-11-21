import json
import numpy as np
import matplotlib.pyplot as plt

# Load the pose data
def load_pose_data(pose_file):
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    return pose_data

# Plot reprojection errors over frames
def plot_reprojection_errors(pose_data):
    frames = []
    mean_errors = []
    std_errors = []

    for frame_data in pose_data:
        frames.append(frame_data['frame'])
        mean_error = frame_data.get('mean_reprojection_error')
        std_error = frame_data.get('std_reprojection_error')

        if mean_error is not None:
            mean_errors.append(mean_error)
            std_errors.append(std_error)
        else:
            mean_errors.append(0)
            std_errors.append(0)

    plt.figure(figsize=(10, 6))
    plt.errorbar(frames, mean_errors, yerr=std_errors, fmt='-o', label='Mean Reprojection Error')
    plt.xlabel('Frame')
    plt.ylabel('Reprojection Error (pixels)')
    plt.title('Reprojection Error over Frames')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Path to your pose estimation results
    pose_file = 'path_to_your_pose_estimation.json'

    # Load the pose data
    pose_data = load_pose_data(pose_file)

    # Plot reprojection errors
    plot_reprojection_errors(pose_data)
