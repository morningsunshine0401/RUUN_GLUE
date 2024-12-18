import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
image_path = "1.jpg"  # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 1: Define 3D and 2D Correspondences
object_points = np.array([
    [0.049, 0.045, 0],     
    [-0.051, 0.045, 0],    
    [-0.051, -0.044, 0],    
    [0.049, -0.044, -0.04],     
    [0.049, 0.045, 0],
    [0.01, 0.045, 0],
    [-0.003, -0.023, 0]   
], dtype=np.float32)

image_points = np.array([
    [780, 216],  
    [464, 111],  
    [258, 276], 
    [611, 538],  
    [761, 324],
    [644, 168],
    [479, 291] 
], dtype=np.float32)

# Visualize the input 2D points
image_copy = image.copy()
for point in image_points:
    x, y = int(point[0]), int(point[1])
    cv2.circle(image_copy, (x, y), 5, (255, 0, 0), -1)  # Blue for input points
plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
plt.title("Input 2D Points on Image")
plt.axis("off")
plt.show()

# Step 2: Define Camera Intrinsics
focal_length_x = 1195.08491  
focal_length_y = 1354.35538  
cx = 581.022033  
cy = 571.458522  
camera_matrix = np.array([
    [focal_length_x, 0, cx],
    [0, focal_length_y, cy],
    [0, 0, 1]
], dtype=np.float32)

# Distortion coefficients
use_distortion = True
if use_distortion:
    dist_coeffs = np.array([0.10058526, 0.4507094, 0.13687279, -0.01839536, 0.13001669], dtype=np.float32)
else:
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# Step 3: SolvePnPRansac for pose estimation
success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs)

if success:
    print("Rotation Vector (rvec):\n", rvec)
    print("Translation Vector (tvec):\n", tvec)

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    print("Rotation Matrix (R):\n", R)

    # Function to back-project 2D points onto a plane (e.g., Z = 0)
    def back_project_3d(pixel, R, tvec, camera_matrix, plane_normal, plane_point):
        """Back-project a 2D pixel to a 3D plane."""
        # Step 1: Convert pixel to normalized camera coordinates
        pixel_homog = np.array([pixel[0], pixel[1], 1.0], dtype=np.float32)
        ray_dir_camera = np.linalg.inv(camera_matrix) @ pixel_homog.reshape(3, 1)
        
        # Step 2: Transform ray to world coordinates
        ray_dir_world = R.T @ ray_dir_camera
        ray_dir_world = ray_dir_world.flatten()  # Ensure it's (3,)
        ray_dir_world /= np.linalg.norm(ray_dir_world)  # Normalize

        # Step 3: Compute camera center
        camera_center = -R.T @ tvec.flatten()

        # Step 4: Compute intersection with the plane
        numerator = np.dot(plane_normal, plane_point - camera_center)
        denominator = np.dot(plane_normal, ray_dir_world)
        d = numerator / denominator
        intersection = camera_center + d * ray_dir_world

        return intersection


    # Step 4: Define additional 2D points and back-project them
    additional_2d_points = np.array([
        [586, 149],  # Example 2D points
        [496, 372],
        [374, 250]
    ], dtype=np.float32)

    print("\nBack-Projected 3D Points:")
    plane_normal = np.array([0, 0, 1])  # Plane Z = 0
    plane_point = np.array([0, 0, 0])   # Point on the plane

    for pixel in additional_2d_points:
        point_3d = back_project_3d(pixel, R, tvec, camera_matrix, plane_normal, plane_point)
        print(f"2D Point {pixel} -> 3D Point {point_3d}")

    # Visualize all points (original and new)
    image_result = image.copy()
    for point in additional_2d_points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image_result, (x, y), 5, (0, 0, 255), -1)  # Red for new points

    plt.imshow(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))
    plt.title("Additional 2D Points on Image")
    plt.axis("off")
    plt.show()

else:
    print("solvePnPRansac failed to estimate pose.")
