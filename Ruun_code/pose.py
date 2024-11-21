import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Good matches and keypoints provided by you
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
])

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
])

# Convert keypoints to float32 as required by solvePnP
good_keypoints0_3D = good_keypoints0_3D.astype(np.float32)
good_keypoints1_2D = good_keypoints1_2D.astype(np.float32)

# Blender Camera Intrinsics
focal_length = 50  # in mm
sensor_width = 36.0  # standard full-frame width in mm
image_width = 640  # assuming a 640x480 image resolution
image_height = 480

# Calculate focal lengths in pixels
fx = (focal_length / sensor_width) * image_width
fy = (focal_length / sensor_width) * image_height
cx = image_width / 2  # Principal point (assumed to be in the image center)
cy = image_height / 2

# Camera intrinsic matrix based on Blender's values
K_blender = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

# Solve PnP using 3D-2D correspondences (Perspective-n-Point)
success, R_vec, t_vec = cv2.solvePnP(good_keypoints0_3D, good_keypoints1_2D, K_blender, None)

# Convert rotation vector to a rotation matrix
R_est, _ = cv2.Rodrigues(R_vec)

# Print the estimated rotation matrix and translation vector
print("Estimated Rotation Matrix:")
print(R_est)

print("\nEstimated Translation Vector:")
print(t_vec)

# True camera poses (for visualization and comparison)
location_image0 = np.array([0, 15, 0])  # Camera 0 (15 meters behind the aircraft)
location_image1 = np.array([0, 25, 0])  # Camera 1 (25 meters behind the aircraft)

# Euler angles for both images (in degrees) - Blender's output is in XYZ order
rotation_image0 = (269, 179, 0)  # Camera 0 rotation (looking at the aircraft)
rotation_image1 = (269, 179, 0)  # Camera 1 rotation (same rotation)

# Convert Euler angles to rotation matrices
rotation_matrix_image0 = R.from_euler('xyz', rotation_image0, degrees=True).as_matrix()
rotation_matrix_image1 = R.from_euler('xyz', rotation_image1, degrees=True).as_matrix()

print("\nrotation_matrix_image0:")
print(rotation_matrix_image0)
print("\nrotation_matrix_image1:")
print(rotation_matrix_image1)

# Compute the true relative rotation matrix and translation vector
relative_rotation_matrix_true = rotation_matrix_image1 @ rotation_matrix_image0.T
relative_translation_vector_true = location_image1 - location_image0
relative_translation_vector_true_normalized = relative_translation_vector_true / np.linalg.norm(relative_translation_vector_true)

# Print the true relative pose for comparison
print("\nTrue Rotation Matrix:")
print(relative_rotation_matrix_true)

print("\nTrue Translation Vector :")
print(relative_translation_vector_true)

print("\nTrue Translation Vector (normalized):")
print(relative_translation_vector_true_normalized)


# --- Visualization Part ---

# Function to plot camera positions
def plot_camera(ax, R, t, label, color):
    # Draw camera as a point
    ax.scatter(t[0], t[1], t[2], color=color, s=100, label=label)

    # Draw camera orientation using the rotation matrix
    scale = 3  # Scaling factor for arrow length
    
    # Camera forward direction (negative Z-axis in camera frame)
    camera_forward = -R[:, 2]  # The camera looks along the negative Z-axis
    ax.quiver(t[0], t[1], t[2],
              camera_forward[0], camera_forward[1], camera_forward[2],
              length=scale, color=color, arrow_length_ratio=0.1)

    # Camera up direction (Y-axis in camera frame)
    camera_up = R[:, 1]  # Y-axis for camera's 'up' direction
    ax.quiver(t[0], t[1], t[2],
              camera_up[0], camera_up[1], camera_up[2],
              length=1, color='cyan', arrow_length_ratio=0.1)

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Camera Viewpoint Changes')

# Plot true camera poses
plot_camera(ax, rotation_matrix_image0, location_image0, label='Camera 0 (True)', color='blue')
plot_camera(ax, rotation_matrix_image1, location_image1, label='Camera 1 (True)', color='green')

# Plot the estimated camera pose (Camera 1 relative to Camera 0)
location_image1_est = t_vec.ravel() + location_image0
rotation_matrix_est = R_est @ rotation_matrix_image0  # Apply estimated rotation to Camera 0's orientation
plot_camera(ax, rotation_matrix_est, location_image1_est, label='Camera 1 (Estimated)', color='red')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Display legend and plot
ax.legend()
plt.show()
