import cv2
import json
import numpy as np
import os

# Load the pose data
def load_pose_data(pose_file):
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    return pose_data

def visualize_matches_and_projections(pose_data, video_path, output_dir, anchor_image_path, anchor_keypoints_2D, anchor_keypoints_3D, K):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error opening video file.')
        return

    # Load the anchor image
    anchor_image = cv2.imread(anchor_image_path)
    assert anchor_image is not None, 'Failed to load anchor image.'
    anchor_keypoints_2D = np.array(anchor_keypoints_2D)
    anchor_keypoints_3D = np.array(anchor_keypoints_3D)

    frame_idx = 0
    for frame_data in pose_data:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1

        # Skip frames if necessary
        if frame_idx != frame_data['frame']:
            continue

        # Get matched keypoints
        mkpts0 = np.array(frame_data.get('mkpts0', []))
        mkpts1 = np.array(frame_data.get('mkpts1', []))
        mpts3D = np.array(frame_data.get('mpts3D', []))
        inliers = frame_data.get('inliers', [])
        inliers = np.array(inliers).astype(int)

        # Draw matches
        matched_image = cv2.drawMatches(
            anchor_image, [cv2.KeyPoint(x[0], x[1], 1) for x in mkpts0],
            frame, [cv2.KeyPoint(x[0], x[1], 1) for x in mkpts1],
            [cv2.DMatch(i, i, 0) for i in range(len(mkpts0))],
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Project 3D points onto the frame using estimated pose
        if frame_data.get('rotation_matrix') is not None:
            R = np.array(frame_data['rotation_matrix'])
            tvec = np.array(frame_data['translation_vector']).reshape(3, 1)
            projected_points, _ = cv2.projectPoints(
                objectPoints=mpts3D[inliers],
                rvec=cv2.Rodrigues(R)[0],
                tvec=tvec,
                cameraMatrix=K,
                distCoeffs=None
            )
            # Draw projected points on the frame
            for pt in projected_points.reshape(-1, 2):
                cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 255, 0), -1)

        # Save or display the frame with matches and projections
        output_path = os.path.join(output_dir, f'frame_{frame_idx:06d}.png')
        cv2.imwrite(output_path, frame)

        # Optionally display the frame
        cv2.imshow('Matched Keypoints and Projections', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Path to your pose estimation results
    pose_file = 'path_to_your_pose_estimation.json'
    video_path = 'path_to_your_video_input'
    output_dir = 'path_to_output_directory'
    anchor_image_path = 'path_to_anchor_image'
    K = np.array([
        [1410, 0, 320],  # Adjust focal length and principal points as per your camera
        [0, 1410, 240],
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

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Visualize matches and projections
    visualize_matches_and_projections(
        pose_data, video_path, output_dir, anchor_image_path,
        anchor_keypoints_2D, anchor_keypoints_3D, K
    )
