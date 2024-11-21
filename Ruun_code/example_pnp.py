import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_scipy

# Provided 3D points in the world coordinate system (For image2)
good_keypoints0_3D = np.array([
    [0.0, 3, 0.72],
    [1.12, -0.53, 0.16],
    [1.12,  2.1, 0.16],
    [1.7, -0.53, 0.06],
    [4.0, 0.08, 0.0],
    [-0.66, 2.4, -0.6],
    [0.66, 2.4, -0.6]
], dtype=np.float32)

# Corresponding 2D projections in the image plane (For image2)
good_keypoints1_2D = np.array([
    [422.0, 203.0],
    [269.0, 234.0],
    [351.0, 232.0],
    [249.0, 236.0],
    [185.0, 243.0],
    [414.0, 268.0],
    [373.0, 268.0]
], dtype=np.float32)

# Camera intrinsic parameters
focal_length = 50  # in mm
sensor_width = 36.0  # in mm
image_width = 1280#640  # in pixels
image_height = 800#480  # in pixels

# Calculate focal lengths in pixels
fx = (focal_length / sensor_width) * image_width
fy = (focal_length / sensor_width) * image_height
cx = image_width / 2  # Principal point x-coordinate
cy = image_height / 2  # Principal point y-coordinate

# Camera intrinsic matrix
K = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
])
print(K)

# Estimate the camera's pose using solvePnP
success, rvec, tvec = cv2.solvePnP(
    good_keypoints0_3D,   # 3D points in the world coordinate system
    good_keypoints1_2D,   # Corresponding 2D image points
    K,                    # Camera intrinsic matrix
    None,                 # Distortion coefficients (assuming no lens distortion)
    flags=cv2.SOLVEPNP_ITERATIVE
)

# Check if the pose estimation was successful
if not success:
    print("Pose estimation failed.")
else:
    # Convert rotation vector to rotation matrix
    R_wc, _ = cv2.Rodrigues(rvec)

    # Estimated camera pose in world coordinates
    camera_position_est = -R_wc.T @ tvec  # Camera position in world coordinates
    R_cw = R_wc.T  # Rotation from camera to world coordinates

    # Ground truth camera pose (example)
    # Assuming the camera is at position (15, 15, 0) and looking at the origin
    camera_position_gt = np.array([15, 15, 0], dtype=np.float32)

    # Compute rotation matrix for ground truth camera looking at the origin
    def compute_camera_rotation(eye, target, up=np.array([0, -1, 0], dtype=np.float32)):
        # Ensure inputs are float64 for precision
        eye = eye.astype(np.float64)
        target = target.astype(np.float64)
        up = up.astype(np.float64)

        # Compute the forward vector (camera's Z-axis)
        forward = (target - eye)
        forward /= np.linalg.norm(forward)

        # Compute the right vector (camera's X-axis)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        # Recompute the true up vector (camera's Y-axis)
        down = np.cross(right, forward)
        down /= np.linalg.norm(down)

        # Construct rotation matrix
        R = np.vstack([right, down, forward]).T
        return R

    R_gt = compute_camera_rotation(camera_position_gt, np.array([0, 0, 0], dtype=np.float32))

    # --- Compute and Print Ground Truth and Estimated Rotations and Translations ---

    # Convert rotation matrices to Euler angles (in degrees)
    euler_gt = R_scipy.from_matrix(R_gt).as_euler('xyz', degrees=True)
    euler_est = R_scipy.from_matrix(R_cw).as_euler('xyz', degrees=True)

    # Print ground truth rotation and translation
    print("\nGround Truth Camera Pose:")
    print("Position (world coordinates):", camera_position_gt)
    print("Rotation (Euler angles in degrees):", euler_gt)

    # Print estimated rotation and translation
    print("\nEstimated Camera Pose:")
    print("Position (world coordinates):", camera_position_est.flatten())
    print("Rotation (Euler angles in degrees):", euler_est)

    # Compute errors
    position_error = camera_position_est.flatten() - camera_position_gt
    rotation_error = euler_est - euler_gt

    # Adjust rotation error to be within [-180, 180] degrees
    rotation_error = (rotation_error + 180) % 360 - 180

    # Print errors
    print("\nErrors:")
    print("Position Error (X, Y, Z):", position_error)
    print("Rotation Error (Roll, Pitch, Yaw):", rotation_error)

    # --- Visualization Part ---

    # Define function to plot camera coordinate frame
    def plot_camera(ax, R, t, color='blue', label='Camera'):
        # Draw camera center
        ax.scatter(t[0], t[1], t[2], color=color, s=100, label=label)

        # Draw camera coordinate axes
        axis_length = 5
        x_axis = -R[:, 0] * axis_length  # Negate X-axis
        y_axis = R[:, 1] * axis_length  # Camera's Y-axis
        z_axis = R[:, 2] * axis_length  # Camera's Z-axis

        # Plot the axes
        origin = t.flatten()
        ax.quiver(*origin, *x_axis, color='red', length=axis_length, normalize=True)
        ax.quiver(*origin, *y_axis, color='green', length=axis_length, normalize=True)
        ax.quiver(*origin, *z_axis, color='blue', length=axis_length, normalize=True)

        # Label the axes
        ax.text(*(origin + x_axis), f'{label} X', color='red')
        ax.text(*(origin + y_axis), f'{label} Y', color='green')
        ax.text(*(origin + z_axis), f'{label} Z', color='blue')

    # Set up the 3D plot
    fig = plt.figure(figsize=(12, 6))

    # Plot the camera poses
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Camera Pose Estimation')

    # Plot the ground truth camera pose
    plot_camera(ax1, R_gt, camera_position_gt, color='green', label='Ground Truth Camera')

    # Plot the estimated camera pose
    plot_camera(ax1, R_cw, camera_position_est, color='red', label='Estimated Camera')

    # Plot the 3D points
    ax1.scatter(good_keypoints0_3D[:, 0], good_keypoints0_3D[:, 1], good_keypoints0_3D[:, 2],
                color='black', s=50, label='3D Points')

    # Set labels
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_zlabel('Z-axis')
    ax1.legend()

    # Set equal aspect ratio
    max_range = np.array([
        good_keypoints0_3D[:, 0].max() - good_keypoints0_3D[:, 0].min(),
        good_keypoints0_3D[:, 1].max() - good_keypoints0_3D[:, 1].min(),
        good_keypoints0_3D[:, 2].max() - good_keypoints0_3D[:, 2].min()
    ]).max() / 2.0

    mid_x = (good_keypoints0_3D[:, 0].max() + good_keypoints0_3D[:, 0].min()) * 0.5
    mid_y = (good_keypoints0_3D[:, 1].max() + good_keypoints0_3D[:, 1].min()) * 0.5
    mid_z = (good_keypoints0_3D[:, 2].max() + good_keypoints0_3D[:, 2].min()) * 0.5

    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    '''
    # Plot the errors
    ax2 = fig.add_subplot(122)
    ax2.set_title('Errors between Ground Truth and Estimated Pose')

    # Prepare data for plotting errors
    labels = ['X', 'Y', 'Z']
    x = np.arange(len(labels))
    width = 0.35

    # Position errors
    ax2.bar(x - width/2, position_error, width, label='Position Error')

    # Rotation errors (Roll, Pitch, Yaw)
    labels_rot = ['Roll', 'Pitch', 'Yaw']
    x_rot = np.arange(len(labels_rot))
    ax2.bar(x_rot + width/2, rotation_error, width, label='Rotation Error')

    ax2.set_xticks(np.concatenate([x - width/2, x_rot + width/2]))
    ax2.set_xticklabels(labels + labels_rot)
    ax2.legend()
    '''
    plt.tight_layout()
    plt.show()










'''
# Provided 3D points in the world coordinate system (For image1)
good_keypoints0_3D = np.array([
    [0.0, 3.6, 0.7],
    [1.1, 2.1, 0.16],
    [-1.1, 2.1, 0.16],
    [1.7, 2.1, 0.06],
    [-1.7, 2.1, 0.06],
    [4.0, 0.08, 0.0],
    [-4.0, 0.08, 0.0],
    [-0.66, 2.4, -0.6],
    [0.66, 2.4, -0.6]
], dtype=np.float32)
'''

'''
# Corresponding 2D projections in the image plane (For image1)
good_keypoints1_2D = np.array([
    [319, 209],
    [276, 232],
    [363, 233],
    [252, 236],
    [387, 237],
    [177, 242],
    [466, 243],
    [343, 264],
    [293, 265]
], dtype=np.float32)
'''