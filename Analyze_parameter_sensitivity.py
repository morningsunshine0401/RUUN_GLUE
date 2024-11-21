import subprocess
import itertools
import os

# Define ranges of parameters to test
match_thresholds = [0.1, 0.2, 0.3]
keypoint_thresholds = [0.005, 0.007, 0.009]
nms_radii = [3, 4, 5]

# Path to your script and inputs
pose_estimation_script = 'your_pose_estimation_script.py'
input_video = 'path_to_your_video_input'
anchor_image = 'path_to_anchor_image'
output_base_dir = 'path_to_output_directory'

# Generate all combinations of parameters
param_combinations = list(itertools.product(match_thresholds, keypoint_thresholds, nms_radii))

for idx, (match_thresh, keypoint_thresh, nms_radius) in enumerate(param_combinations):
    output_dir = os.path.join(output_base_dir, f'run_{idx}')
    os.makedirs(output_dir, exist_ok=True)

    # Build the command
    cmd = [
        'python', pose_estimation_script,
        '--input', input_video,
        '--anchor', anchor_image,
        '--output_dir', output_dir,
        '--match_threshold', str(match_thresh),
        '--keypoint_threshold', str(keypoint_thresh),
        '--nms_radius', str(nms_radius),
        '--save_pose', os.path.join(output_dir, 'pose_estimation.json'),
        '--no_display'
    ]

    # Run the pose estimation with the current parameters
    subprocess.run(cmd)

    # You can add code here to analyze the results immediately after each run
