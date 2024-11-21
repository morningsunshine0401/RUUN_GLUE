import cv2
import json
import numpy as np
import os

# Load the pose data (same as before)
def load_pose_data(pose_file):
    with open(pose_file, 'r') as f:
        pose_data = json.load(f)
    return pose_data

def visualize_frame(pose_data, frame_number, video_path, anchor_image_path, K, output_dir=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error opening video file.')
        return

    # Load the anchor image
    anchor_image = cv2.imread(anchor_image_path)
    assert anchor_image is not None, 'Failed to load anchor image.'

    # Find the frame data
    frame_data = next((fd for fd in pose_data if fd['frame'] == frame_number), None)
    if frame_data is None:
        print(f'Frame {frame_number} data not found.')
        return

    # Read up to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f'Failed to read frame {frame_number}.')
        return

    # Get matched keypoints
    mkpts0 = np.array(frame_data.get('mkpts0', []))
    mkpts1 = np.array(frame_data.get('mkpts1', []))
    mpts3D = np.array(frame_data.get('mpts3D', []))
    inliers = frame_data.get('inliers', [])
    inliers = np.array(inliers).astype(int)

    # Draw matches (inliers and outliers)
    inlier_mask = np.zeros(len(mkpts0), dtype=bool)
    inlier_mask[inliers] = True

    # Create images with matches
    matches = []
    for i in range(len(mkpts0)):
        matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0))

    # Convert keypoints
    keypoints0 = [cv2.KeyPoint(x[0], x[1], 1) for x in mkpts0]
    keypoints1 = [cv2.KeyPoint(x[0], x[1], 1) for x in mkpts1]

    # Draw matches with inliers in green and outliers in red
    draw_params = dict(
        matchesMask=inlier_mask.tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    matched_image = cv2.drawMatches(
        anchor_image, keypoints0,
        frame, keypoints1,
        matches, None,
        matchColor=(0, 255, 0),  # Inliers in green
        singlePointColor=None,
        matchesMask=inlier_mask.tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Draw outliers in red
    outlier_mask = ~inlier_mask
    if np.any(outlier_mask):
        matched_image = cv2.drawMatches(
            anchor_image, keypoints0,
            frame, keypoints1,
            [matches[i] for i in range(len(matches)) if outlier_mask[i]],
            matched_image,
            matchColor=(0, 0, 255),  # Outliers in red
            singlePointColor=None,
            matchesMask=[1]*outlier_mask.sum(),
            flags=cv2.DrawMatchesFlags_DEFAULT
        )

    # Project 3D points onto the frame using estimated pose
    if frame_data.get('rotation_matrix') is not None and len(inliers) > 0:
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
            cv2.circle(frame, tuple(pt.astype(int)), 5, (255, 0, 0), -1)  # Blue dots

    # Display or save the visualization
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'frame_{frame_number:06d}.png')
        cv2.imwrite(output_path, matched_image)
        print(f'Saved visualization for frame {frame_number} to {output_path}')
    else:
        cv2.imshow(f'Frame {frame_number} Visualization', matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cap.release()

if __name__ == '__main__':
    # Paths and parameters
    pose_file = 'path_to_your_pose_estimation.json'
    video_path = 'path_to_your_video_input'
    anchor_image_path = 'path_to_anchor_image'
    K = np.array([
        [1410, 0, 320],  # Adjust focal length and principal points as per your camera
        [0, 1410, 240],
        [0, 0, 1]
    ], dtype=np.float32)

    # Load the pose data
    pose_data = load_pose_data(pose_file)

    # Frames you want to inspect (e.g., sudden jump frames)
    frames_to_inspect = [10, 20, 30]  # Replace with actual frames with sudden jumps

    # Visualize each frame
    for frame_number in frames_to_inspect:
        visualize_frame(pose_data, frame_number, video_path, anchor_image_path, K)
