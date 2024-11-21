import json
import matplotlib.pyplot as plt

# Load the pose data
def load_pose_data(pose_file):
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    return pose_data

# Plot inlier ratio over frames
def plot_inlier_ratio(pose_data):
    frames = []
    inlier_ratios = []
    num_inliers_list = []
    total_matches_list = []

    for frame_data in pose_data:
        frames.append(frame_data['frame'])
        inlier_ratios.append(frame_data['inlier_ratio'])
        num_inliers_list.append(frame_data['num_inliers'])
        total_matches_list.append(frame_data['total_matches'])

    plt.figure(figsize=(10, 6))
    plt.plot(frames, inlier_ratios, marker='o', label='Inlier Ratio')
    plt.xlabel('Frame')
    plt.ylabel('Inlier Ratio')
    plt.title('Inlier Ratio over Frames')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(frames, num_inliers_list, marker='o', label='Number of Inliers')
    plt.plot(frames, total_matches_list, marker='x', label='Total Matches')
    plt.xlabel('Frame')
    plt.ylabel('Number of Matches')
    plt.title('Number of Inliers and Total Matches over Frames')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Path to your pose estimation results
    pose_file = 'path_to_your_pose_estimation.json'

    # Load the pose data
    pose_data = load_pose_data(pose_file)

    # Plot inlier ratio and number of matches
    plot_inlier_ratio(pose_data)
