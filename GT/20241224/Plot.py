import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the data from JSON file
with open("extracted_positions_20241224_test3.json", "r") as file:
    data = json.load(file)

# We want the target's position relative to the camera's frame
# That means: relative_target_position = target_position - camera_position
target_positions_relative = []
distances = []  # To store relative distances

for entry in data:
    cam = entry["camera_position"]
    tgt = entry["target_position"]
    
    # Compute target in camera frame
    rel_tgt = {
        "X": tgt["X"] - cam["X"],
        "Y": tgt["Y"] - cam["Y"],
        "Z": tgt["Z"] - cam["Z"],
    }
    target_positions_relative.append(rel_tgt)
    
    # Compute the Euclidean distance
    distance = np.sqrt(rel_tgt["X"]**2 + rel_tgt["Y"]**2 + rel_tgt["Z"]**2)
    distances.append(distance)

# Extract X, Y, Z for plotting
target_x = [pos["X"] for pos in target_positions_relative]
print("Min X:", min(target_x))
print("Max X:", max(target_x))
target_y = [pos["Y"] for pos in target_positions_relative]
print("Min Y:", min(target_y))
print("Max Y:", max(target_y))
target_z = [pos["Z"] for pos in target_positions_relative]
print("Min Z:", min(target_z))
print("Max Z:", max(target_z))

# Compute distance stats
min_distance = min(distances)
max_distance = max(distances)
avg_distance = sum(distances) / len(distances)

print("Min Distance:", min_distance)
print("Max Distance:", max_distance)
print("Average Distance:", avg_distance)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the camera at the origin
ax.scatter(0, 0, 0, color='red', label='Camera (Origin)', s=100)

# Plot the target positions (in the camera's frame)
ax.scatter(target_x, target_y, target_z, color='blue', label='Target (Relative)', s=50)

# Optionally connect the trajectory with lines (uncomment if desired)
# ax.plot(target_x, target_y, target_z, color='blue', linestyle='--')

# Add labels and legend
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Target Positions Relative to Camera')
ax.legend()

plt.show()

# Plot the relative distances over time
plt.figure(figsize=(10, 6))
plt.plot(distances, label='Relative Distance', color='blue')
plt.axhline(min_distance, color='green', linestyle='--', label=f'Min Distance: {min_distance:.2f}')
plt.axhline(max_distance, color='red', linestyle='--', label=f'Max Distance: {max_distance:.2f}')
plt.axhline(avg_distance, color='orange', linestyle='--', label=f'Avg Distance: {avg_distance:.2f}')
plt.xlabel('Frame Index')
plt.ylabel('Distance (units)')
plt.title('Relative Distance Between Target and Camera')
plt.legend()
plt.grid()
plt.show()


