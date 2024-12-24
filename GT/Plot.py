import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the data from JSON file
with open("extracted_positions(1).json", "r") as file:
    data = json.load(file)

# Extract camera and target positions
camera_positions = []
for entry in data:
    camera_position = entry["camera_position"]
    target_position = entry["target_position"]

    # Calculate camera's relative position to the target
    relative_camera_position = {
        "X": camera_position["X"] - target_position["X"],
        "Y": camera_position["Y"] - target_position["Y"],
        "Z": camera_position["Z"] - target_position["Z"],
    }
    camera_positions.append(relative_camera_position)

# Extract X, Y, Z coordinates for plotting
camera_x = [pos["X"] for pos in camera_positions]
camera_y = [pos["Y"] for pos in camera_positions]
camera_z = [pos["Z"] for pos in camera_positions]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the target at the origin
ax.scatter(0, 0, 0, color='red', label='Target (Origin)', s=100)

# Plot the camera positions
ax.scatter(camera_x, camera_y, camera_z, color='blue', label='Camera Positions', s=50)

# Add labels and legend
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Plot of Camera and Target Positions')
ax.legend()

# Display the plot
plt.show()


################################################################################################


# import json
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# # Load the data from JSON file
# with open("extracted_positions(1).json", "r") as file:
#     data = json.load(file)

# # Define the transformation matrix (consistent with earlier code)
# T = np.array([
#     [1, 0, 0],
#     [0, 0, 1],
#     [0, -1, 0]
# ], dtype=np.float32)

# # Extract camera and target positions
# camera_positions = []
# for entry in data:
#     camera_position = entry["camera_position"]
#     target_position = entry["target_position"]

#     # Calculate camera's relative position to the target
#     relative_camera_position = np.array([
#         camera_position["X"] - target_position["X"],
#         camera_position["Y"] - target_position["Y"],
#         camera_position["Z"] - target_position["Z"]
#     ], dtype=np.float32)

#     # Transform to OpenCV's coordinate frame
#     relative_camera_position_opencv = T @ relative_camera_position

#     # Append the transformed position
#     camera_positions.append({
#         "X": relative_camera_position_opencv[0],
#         "Y": relative_camera_position_opencv[1],
#         "Z": relative_camera_position_opencv[2],
#     })

# # Extract X, Y, Z coordinates for plotting
# camera_x = [pos["X"] for pos in camera_positions]
# camera_y = [pos["Y"] for pos in camera_positions]
# camera_z = [pos["Z"] for pos in camera_positions]

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the target at the origin
# ax.scatter(0, 0, 0, color='red', label='Target (Origin)', s=100)

# # Plot the camera positions
# ax.scatter(camera_x, camera_y, camera_z, color='blue', label='Camera Positions', s=50)

# # Add labels and legend
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# ax.set_title('3D Plot of Camera and Target Positions (OpenCV Frame)')
# ax.legend()

# # Display the plot
# plt.show()
