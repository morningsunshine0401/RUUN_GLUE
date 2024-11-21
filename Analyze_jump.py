import json
import numpy as np

# Load the pose data
def load_pose_data(pose_file):
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    return pose_data

# Detect frames with sudden jumps in pose estimation
def detect_sudden_jumps(pose_data, position_threshold=0.1, rotation_threshold=10):
    """
    position_threshold: Threshold for position change (in your units, e.g., meters)
    rotation_threshold: Threshold for rotation change (in degrees)
    """
    sudden_jump_frames = []
    prev_position = None
    prev_rotation = None

    for idx, frame_data in enumerate(pose_data):
        frame_idx = frame_data['frame']
        if 'camera_position' not in frame_data or 'rotation_matrix' not in frame_data:
            continue

        position = np.array(frame_data['camera_position'])
        R = np.array(frame_data['rotation_matrix'])

        if prev_position is not None:
            # Compute position difference
            position_diff = np.linalg.norm(position - prev_position)

            # Compute rotation difference
            rotation_diff_matrix = R @ prev_rotation.T
            rotation_angle = np.arccos((np.trace(rotation_diff_matrix) - 1) / 2)
            rotation_angle_deg = np.degrees(rotation_angle)

            # Check if differences exceed thresholds
            if position_diff > position_threshold or rotation_angle_deg > rotation_threshold:
                sudden_jump_frames.append({
                    'frame': frame_idx,
                    'position_diff': position_diff,
                    'rotation_diff_deg': rotation_angle_deg
                })

        prev_position = position
        prev_rotation = R

    return sudden_jump_frames

if __name__ == '__main__':
    # Path to your pose estimation results
    pose_file = 'path_to_your_pose_estimation.json'

    # Load the pose data
    pose_data = load_pose_data(pose_file)

    # Detect sudden jumps
    sudden_jumps = detect_sudden_jumps(pose_data, position_threshold=0.1, rotation_threshold=10)

    # Print frames with sudden jumps
    print("Frames with sudden pose estimation jumps:")
    for jump in sudden_jumps:
        print(f"Frame {jump['frame']}: Position Change = {jump['position_diff']:.3f}, Rotation Change = {jump['rotation_diff_deg']:.2f} degrees")
