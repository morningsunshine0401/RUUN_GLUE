import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images (adjust the paths to your actual images)
img0 = cv2.imread('/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/52.png')
img1 = cv2.imread('/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/85.png')

# Verify the image sizes after loading
print("Original Image Sizes:")
print("Image 0:", img0.shape)  # Should output (800, 1280, 3)
print("Image 1:", img1.shape)  # Should output (800, 1280, 3)

# Resize the images to the desired size if necessary
desired_width = 1280
desired_height = 960

if img0.shape[1] != desired_width or img0.shape[0] != desired_height:
    img0 = cv2.resize(img0, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    print("Resized Image 0 to:", img0.shape)

if img1.shape[1] != desired_width or img1.shape[0] != desired_height:
    img1 = cv2.resize(img1, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    print("Resized Image 1 to:", img1.shape)

# Load keypoints and matches
keypoints0 = np.load('/home/runbk0401/SuperGluePretrainedNetwork/dump_match_pairs/match_output/viewpoint/right/keypoints0.npy')
keypoints1 = np.load('/home/runbk0401/SuperGluePretrainedNetwork/dump_match_pairs/match_output/viewpoint/right/keypoints1.npy')
matches = np.load('/home/runbk0401/SuperGluePretrainedNetwork/dump_match_pairs/match_output/viewpoint/right/matches.npy')

# Filter out invalid matches (matches == -1)
valid_matches = matches[matches != -1]
valid_keypoints0 = keypoints0[matches != -1]
valid_keypoints1 = keypoints1[valid_matches]
print("Valid Keypoints in Image 1:")
print(valid_keypoints1)

# Define good matches indices (from your list of good matches)
#good_matches_indices = [0, 1, 2, 4, 13, 14, 15, 17, 23, 36, 55, 56, 59, 65, 71]
good_matches_indices = [0, 2, 3, 5, 6, 9, 10, 14, 16, 22, 26, 28, 29, 30, 32, 34, 39, 42, 45, 47, 48]



# Phase 2: Input 3D coordinates for the good matches
good_keypoints0_3D = []
good_keypoints1_2D = []

for idx, i in enumerate(good_matches_indices):
    kp0 = valid_keypoints0[i]
    kp1 = valid_keypoints1[i]

    # Show the image of good match again for 3D input
    img0_copy = img0.copy()
    img1_copy = img1.copy()

    # Draw the keypoints in both images
    img0_copy = cv2.circle(img0_copy, (int(kp0[0]), int(kp0[1])), 10, (255, 0, 0), 3)
    img1_copy = cv2.circle(img1_copy, (int(kp1[0]), int(kp1[1])), 10, (255, 0, 0), 3)

    # Combine the two images for side-by-side comparison
    combined_img = np.hstack((img0_copy, img1_copy))

    # Create a named window that can be resized
    window_name = f'Good Match {idx+1}: Input 3D Coordinates'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, combined_img.shape[1], combined_img.shape[0])

    # Show the image with keypoints using OpenCV
    cv2.imshow(window_name, combined_img)
    cv2.waitKey(0)  # Wait for the user to input the 3D coordinates

    # Destroy the image window
    cv2.destroyAllWindows()

    # Ask the user for the 3D coordinates of this keypoint in image 0
    x = float(input(f"Enter X coordinate for keypoint {idx+1}: "))
    y = float(input(f"Enter Y coordinate for keypoint {idx+1}: "))
    z = float(input(f"Enter Z coordinate for keypoint {idx+1}: "))

    # Store the 3D coordinates and the corresponding 2D keypoint in image 1
    good_keypoints0_3D.append([x, y, z])
    good_keypoints1_2D.append(kp1)

# Convert good_keypoints0_3D and good_keypoints1_2D to numpy arrays
good_keypoints0_3D = np.array(good_keypoints0_3D, dtype=np.float32)
good_keypoints1_2D = np.array(good_keypoints1_2D, dtype=np.float32)

# Display the final good matches and their 3D points
print(f"\nGood Matches Indices: {good_matches_indices}")
print("\nGood Keypoints 3D in Image 0:")
print(good_keypoints0_3D)

print("\nGood Keypoints 2D in Image 1:")
print(good_keypoints1_2D)

# Save the good matches, 3D points, and corresponding 2D points to files (if needed)
np.save('good_keypoints0_3D.npy', good_keypoints0_3D)
np.save('good_keypoints1_2D.npy', good_keypoints1_2D)
