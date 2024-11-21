import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the new dataset (from the provided .pkl file)
with open('/home/runbk0401/Downloads/output_back_aircraft.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract keypoints
mmkeypoints0 = np.array(data['mkeypoints0_orig'])
mmkeypoints1 = np.array(data['mkeypoints1_orig'])

# Ensure that keypoints are continuous arrays for OpenCV
mmkeypoints0_cont = np.ascontiguousarray(mmkeypoints0, dtype=np.float32)
mmkeypoints1_cont = np.ascontiguousarray(mmkeypoints1, dtype=np.float32)

# Blender Camera Intrinsics
focal_length = 57.79  # in mm
sensor_width = 36.0  # standard full-frame width in mm
image_width = 640  # Assuming a 1920x1080 image resolution from Blender
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

# Compute the essential matrix using Blender's camera intrinsic matrix
E, mask = cv2.findEssentialMat(mmkeypoints0_cont, mmkeypoints1_cont, K_blender, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Recover the relative pose (rotation and translation) from the essential matrix
_, R_est, t_est, mask_pose = cv2.recoverPose(E, mmkeypoints0_cont, mmkeypoints1_cont, K_blender)

# True camera poses for the test (no rotation, different distances along the z-axis)
location_image0 = np.array([0, 0, 15.0])  # 15 meters away
location_image1 = np.array([0, 0, 25.0])  # 25 meters away (along z-axis)

# No rotation, so both rotation matrices are identity matrices
rotation_matrix_image0 = np.eye(3)
rotation_matrix_image1 = np.eye(3)

# Compute the true relative rotation matrix and translation vector
relative_rotation_matrix_true = rotation_matrix_image1 @ rotation_matrix_image0.T
relative_translation_vector_true = location_image1 - location_image0
relative_translation_vector_true_normalized = relative_translation_vector_true / np.linalg.norm(relative_translation_vector_true)

# Scale the estimated translation vector to match the true scale
t_est_scaled = t_est * np.linalg.norm(relative_translation_vector_true)

# Print the estimated and true relative pose
print("Estimated Rotation Matrix:")
print(R_est)

print("\nEstimated Translation Vector (scaled):")
print(t_est_scaled)

print("\nTrue Rotation Matrix:")
print(relative_rotation_matrix_true)

print("\nTrue Translation Vector (normalized):")
print(relative_translation_vector_true_normalized)

# --- Visualization Part ---

# Function to plot camera positions
def plot_camera(ax, R, t, label, color):
    # Draw camera as a point
    ax.scatter(t[0], t[1], t[2], color=color, s=100, label=label)

    # Draw camera orientation using the rotation matrix (as an arrow)
    scale = 0.5  # Scaling factor for arrow length
    camera_direction = R[:, 2]  # Camera looks along the z-axis
    ax.quiver(t[0], t[1], t[2],
              camera_direction[0], camera_direction[1], camera_direction[2],
              length=scale, color=color)

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Camera Viewpoint Changes')

# Plot true camera poses
plot_camera(ax, rotation_matrix_image0, location_image0, label='Camera 0 (True)', color='blue')
plot_camera(ax, rotation_matrix_image1, location_image1, label='Camera 1 (True)', color='green')

# Plot the estimated camera pose (Camera 1 relative to Camera 0)
# We assume that Camera 0 is at the origin with no rotation.
location_image1_est = t_est_scaled.ravel() + location_image0
rotation_matrix_est = R_est @ rotation_matrix_image0  # Apply estimated rotation to Camera 0's orientation
plot_camera(ax, rotation_matrix_est, location_image1_est, label='Camera 1 (Estimated)', color='red')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Display legend and plot
ax.legend()
plt.show()
