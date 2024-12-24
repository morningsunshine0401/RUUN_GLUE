################################################################################################### 2D Plot
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryViewer:
    def __init__(self, json_file):
        # 1) Load data
        with open(json_file, 'r') as f:
            self.all_frames = json.load(f)
        if not self.all_frames:
            raise ValueError("JSON file is empty or invalid.")

        # 2) Extract all positions (kf_translation_vector) for the entire trajectory
        #    and all rotations (kf_rotation_matrix).
        self.positions = []
        self.rotations = []
        self.euler_angles = []  # Store roll, pitch, yaw
        
        for frame_data in self.all_frames:
            R_list = frame_data['kf_rotation_matrix']      # 3x3 matrix flattened
            t_list = frame_data['kf_translation_vector']   # 3x1
            R = np.array(R_list, dtype=float).reshape(3, 3)
            t = np.array(t_list, dtype=float).reshape(3)
            self.positions.append(t)
            self.rotations.append(R)

            # Compute Euler angles (roll, pitch, yaw)
            roll, pitch, yaw = self.rotation_matrix_to_euler(R)
            self.euler_angles.append([roll, pitch, yaw])

        self.positions = np.array(self.positions)
        self.euler_angles = np.array(self.euler_angles)

        self.num_frames = len(self.positions)

        # Plot position and rotation over time
        self.plot_positions()
        self.plot_rotations()

    def rotation_matrix_to_euler(self, R):
        """Convert a rotation matrix to Euler angles (roll, pitch, yaw)."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        return roll, pitch, yaw

    def plot_positions(self):
        """Plot x, y, z positions over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.positions[:, 0], label='X Position')
        print("min x : ", min(self.positions[:, 0]))
        print("max x : ", max(self.positions[:, 0]))
        plt.plot(self.positions[:, 1], label='Y Position')
        print("min y : ", min(self.positions[:, 1]))
        print("max y : ", max(self.positions[:, 1]))
        plt.plot(self.positions[:, 2], label='Z Position')
        print("min z : ", min(self.positions[:, 2]))
        print("max z : ", max(self.positions[:, 2]))
        plt.xlabel('Frame Index')
        plt.ylabel('Position')
        plt.title('Positions (X, Y, Z) Over Time')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_rotations(self):
        """Plot roll, pitch, yaw rotations over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.euler_angles[:, 0], label='Roll (rad)')
        plt.plot(self.euler_angles[:, 1], label='Pitch (rad)')
        plt.plot(self.euler_angles[:, 2], label='Yaw (rad)')
        plt.xlabel('Frame Index')
        plt.ylabel('Rotation (Radians)')
        plt.title('Rotations (Roll, Pitch, Yaw) Over Time')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # Example usage:
    json_path = "pose_estimation_research_152.json"  # Replace with your JSON file path
    viewer = TrajectoryViewer(json_path)


################################################################################### 3D Plot

# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # For older Matplotlib versions

# class TrajectoryViewer:
#     def __init__(self, json_file):
#         # 1) Load data
#         with open(json_file, 'r') as f:
#             self.all_frames = json.load(f)
#         if not self.all_frames:
#             raise ValueError("JSON file is empty or invalid.")

#         # 2) Extract all positions (kf_translation_vector) for the entire trajectory
#         #    and all rotations (kf_rotation_matrix).
#         self.positions = []
#         self.rotations = []
        
#         for frame_data in self.all_frames:
#             # Adapt the key names to match your code/JSON:
#             R_list = frame_data['kf_rotation_matrix']      # 3x3 matrix flattened
#             t_list = frame_data['kf_translation_vector']   # 3x1
#             R = np.array(R_list, dtype=float).reshape(3,3)
#             t = np.array(t_list, dtype=float).reshape(3)
#             self.positions.append(t)
#             self.rotations.append(R)

#         self.num_frames = len(self.positions)
        
#         # 3) Current frame index
#         self.idx = 0
        
#         # 4) Create figure & set up for interactive event handling
#         self.fig = plt.figure(figsize=(7,7))
#         self.ax = self.fig.add_subplot(111, projection='3d')
#         self.fig.canvas.mpl_connect('key_press_event', self.on_key)

#         # For nicer interactive experience:
#         plt.ion()  # Turn on interactive mode

#         # Draw the first frame
#         self.update_plot()

#     def on_key(self, event):
#         """Handles key presses:
#            - Press 'n' for the next frame
#            - Press 'b' for the previous frame
#         """
#         if event.key == 'n':
#             # Move to the next frame if possible
#             if self.idx < self.num_frames - 1:
#                 self.idx += 1
#                 self.update_plot()
#             else:
#                 print("Reached last frame.")
#         elif event.key == 'b':
#             # Move to the previous frame if possible
#             if self.idx > 0:
#                 self.idx -= 1
#                 self.update_plot()
#             else:
#                 print("Already at first frame.")

#     def update_plot(self):
#         """Clears the axes and re-draws the scene for the current frame."""
#         self.ax.cla()  # Clear the 3D axes

#         # Axes labeling
#         self.ax.set_title(
#             f"Frame {self.idx}/{self.num_frames - 1} (press 'n' for next, 'b' for back)"
#         )
#         self.ax.set_xlabel('X (cam)')
#         self.ax.set_ylabel('Y (cam)')
#         self.ax.set_zlabel('Z (cam)')

#         # Plot the camera at origin (0, 0, 0)
#         self.ax.scatter(0, 0, 0, color='k', s=50, label='Camera')

#         # Draw camera axes (smaller scale)
#         scale_cam_axes = 0.2
#         self.ax.quiver(
#             0, 0, 0,
#             scale_cam_axes, 0, 0,
#             color='r', arrow_length_ratio=0.2, label='Cam X'
#         )
#         self.ax.quiver(
#             0, 0, 0,
#             0, scale_cam_axes, 0,
#             color='g', arrow_length_ratio=0.2, label='Cam Y'
#         )
#         self.ax.quiver(
#             0, 0, 0,
#             0, 0, scale_cam_axes,
#             color='b', arrow_length_ratio=0.2, label='Cam Z'
#         )

#         # Plot the trajectory from frame 0 up to (and including) the current frame
#         traj_positions = np.array(self.positions[:self.idx+1])
#         self.ax.plot(
#             traj_positions[:, 0],
#             traj_positions[:, 1],
#             traj_positions[:, 2],
#             'k--',
#             label='Trajectory'
#         )

#         # Mark all previous frames in red, and the current frame in magenta
#         if len(traj_positions) > 1:
#             self.ax.scatter(
#                 traj_positions[:-1, 0],
#                 traj_positions[:-1, 1],
#                 traj_positions[:-1, 2],
#                 color='r', s=20, label='Past Frames'
#             )
#         current_pos = self.positions[self.idx]
#         self.ax.scatter(
#             current_pos[0], current_pos[1], current_pos[2],
#             color='m', s=40, label='Current Frame'
#         )

#         # Draw the object's local axes for the current frame
#         local_axis_scale = 0.2
#         local_axis = np.array([
#             [0.0,  0.0,  0.0],  # origin
#             [local_axis_scale, 0.0,         0.0],   # +X
#             [0.0,  local_axis_scale,        0.0],   # +Y
#             [0.0,  0.0,         local_axis_scale],  # +Z
#         ])
#         R = self.rotations[self.idx]
#         t = self.positions[self.idx]

#         # Transform these local-axis points into the camera frame
#         transformed_axis = (R @ local_axis.T).T + t

#         # Plot them
#         self.ax.scatter(
#             transformed_axis[:, 0],
#             transformed_axis[:, 1],
#             transformed_axis[:, 2],
#             color='darkorange', s=30, label='Object local axis'
#         )

#         def plot_quiver(start, end, color):
#             self.ax.quiver(
#                 start[0], start[1], start[2],
#                 end[0] - start[0],
#                 end[1] - start[1],
#                 end[2] - start[2],
#                 color=color, arrow_length_ratio=0.2
#             )

#         # Draw local X, Y, Z from the object's origin
#         origin_pt = transformed_axis[0]
#         x_tip     = transformed_axis[1]
#         y_tip     = transformed_axis[2]
#         z_tip     = transformed_axis[3]
#         plot_quiver(origin_pt, x_tip, 'r')
#         plot_quiver(origin_pt, y_tip, 'g')
#         plot_quiver(origin_pt, z_tip, 'b')

#         # ---------------------------------------------------------------------
#         # Fixed manual axes: from 0 to 3 on each axis
#         # ---------------------------------------------------------------------
#         self.ax.set_xlim(-1/2, 1/2)
#         self.ax.set_ylim(-1/2, 1/2)
#         self.ax.set_zlim(-0.5, 0.5)
        
#         self.ax.legend()
#         self.ax.grid(True)

#         plt.draw()
#         plt.pause(0.01)  # allow GUI event loop to process

#     def show(self):
#         """Display the figure until the window is closed."""
#         plt.show(block=True)


# if __name__ == "__main__":
#     # Example usage:
#     # 1) Provide your JSON file path
#     json_path = "pose_estimation_research_149.json"
    
#     # 2) Initialize the viewer
#     viewer = TrajectoryViewer(json_path)
    
#     # 3) Show the figure & interact:
#     #    - Press 'n' to step forward in frames
#     #    - Press 'b' to step backward
#     #    - Close the window or Ctrl+C to exit
#     viewer.show()

###################################################################################################### Relative distance

# import json
# import numpy as np
# import matplotlib.pyplot as plt


# class TrajectoryViewer:
#     def __init__(self, json_file):
#         # 1) Load data
#         with open(json_file, 'r') as f:
#             self.all_frames = json.load(f)
#         if not self.all_frames:
#             raise ValueError("JSON file is empty or invalid.")

#         # 2) Extract all positions (kf_translation_vector) for the entire trajectory
#         self.positions = []  # Store translation vectors
#         self.distances = []  # Store distances

#         for frame_data in self.all_frames:
#             t_list = frame_data['kf_translation_vector']  # 3x1 translation vector
#             #t_list = frame_data['object_translation_in_cam']  # 3x1 translation vector
#             t = np.array(t_list, dtype=float).reshape(3)  # Convert to numpy array
#             self.positions.append(t)

#             # Calculate the Euclidean distance to the camera
#             distance = np.linalg.norm(t)
#             self.distances.append(distance)

#         self.positions = np.array(self.positions)
#         self.distances = np.array(self.distances)

#         # Plot distance over time
#         self.plot_distances()

#     def plot_distances(self):
#         """Plot relative distance between the camera and the target over time."""
#         min_distance = np.min(self.distances)
#         max_distance = np.max(self.distances)
#         avg_distance = np.mean(self.distances)

#         print(f"Minimum distance: {min_distance:.2f}")
#         print(f"Maximum distance: {max_distance:.2f}")
#         print(f"Average distance: {avg_distance:.2f}")

#         plt.figure(figsize=(10, 6))
#         plt.plot(self.distances, label='Distance to Camera', color='b')
#         plt.axhline(min_distance, color='g', linestyle='--', label=f'Min Distance: {min_distance:.2f}')
#         plt.axhline(max_distance, color='r', linestyle='--', label=f'Max Distance: {max_distance:.2f}')
#         plt.axhline(avg_distance, color='orange', linestyle='--', label=f'Avg Distance: {avg_distance:.2f}')
#         plt.xlabel('Frame Index')
#         plt.ylabel('Distance (units)')
#         plt.title('Relative Distance Between Target and Camera')
#         plt.legend()
#         plt.grid()
#         plt.show()


# if __name__ == "__main__":
#     # Example usage:
#     json_path = "pose_estimation_research_152.json"  # Replace with your JSON file path
#     viewer = TrajectoryViewer(json_path)

