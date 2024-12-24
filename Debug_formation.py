import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For older Matplotlib versions

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
        
        for frame_data in self.all_frames:
            # Adapt the key names to match your code/JSON:
            R_list = frame_data['kf_rotation_matrix']      # 3x3 matrix flattened
            t_list = frame_data['kf_translation_vector']   # 3x1
            R = np.array(R_list, dtype=float).reshape(3,3)
            t = np.array(t_list, dtype=float).reshape(3)
            self.positions.append(t)
            self.rotations.append(R)

        self.num_frames = len(self.positions)
        
        # 3) Current frame index
        self.idx = 0
        
        # 4) Create figure & set up for interactive event handling
        self.fig = plt.figure(figsize=(7,7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # For nicer interactive experience:
        plt.ion()  # Turn on interactive mode

        # Draw the first frame
        self.update_plot()

    def on_key(self, event):
        """Handles key presses:
           - Press 'n' for the next frame
           - Press 'b' for the previous frame
        """
        if event.key == 'n':
            # Move to the next frame if possible
            if self.idx < self.num_frames - 1:
                self.idx += 1
                self.update_plot()
            else:
                print("Reached last frame.")
        elif event.key == 'b':
            # Move to the previous frame if possible
            if self.idx > 0:
                self.idx -= 1
                self.update_plot()
            else:
                print("Already at first frame.")

    def update_plot(self):
        """Clears the axes and re-draws the scene for the current frame."""
        self.ax.cla()  # Clear the 3D axes

        # Axes labeling
        self.ax.set_title(
            f"Frame {self.idx}/{self.num_frames - 1} (press 'n' for next, 'b' for back)"
        )
        self.ax.set_xlabel('X (cam)')
        self.ax.set_ylabel('Y (cam)')
        self.ax.set_zlabel('Z (cam)')

        # Plot the camera at origin (0, 0, 0)
        self.ax.scatter(0, 0, 0, color='k', s=50, label='Camera')

        # Draw camera axes (smaller scale)
        scale_cam_axes = 0.2
        self.ax.quiver(
            0, 0, 0,
            scale_cam_axes, 0, 0,
            color='r', arrow_length_ratio=0.2, label='Cam X'
        )
        self.ax.quiver(
            0, 0, 0,
            0, scale_cam_axes, 0,
            color='g', arrow_length_ratio=0.2, label='Cam Y'
        )
        self.ax.quiver(
            0, 0, 0,
            0, 0, scale_cam_axes,
            color='b', arrow_length_ratio=0.2, label='Cam Z'
        )

        # Plot the trajectory from frame 0 up to (and including) the current frame
        traj_positions = np.array(self.positions[:self.idx+1])
        self.ax.plot(
            traj_positions[:, 0],
            traj_positions[:, 1],
            traj_positions[:, 2],
            'k--',
            label='Trajectory'
        )

        # Mark all previous frames in red, and the current frame in magenta
        if len(traj_positions) > 1:
            self.ax.scatter(
                traj_positions[:-1, 0],
                traj_positions[:-1, 1],
                traj_positions[:-1, 2],
                color='r', s=20, label='Past Frames'
            )
        current_pos = self.positions[self.idx]
        self.ax.scatter(
            current_pos[0], current_pos[1], current_pos[2],
            color='m', s=40, label='Current Frame'
        )

        # Draw the object's local axes for the current frame
        local_axis_scale = 0.2
        local_axis = np.array([
            [0.0,  0.0,  0.0],  # origin
            [local_axis_scale, 0.0,         0.0],   # +X
            [0.0,  local_axis_scale,        0.0],   # +Y
            [0.0,  0.0,         local_axis_scale],  # +Z
        ])
        R = self.rotations[self.idx]
        t = self.positions[self.idx]

        # Transform these local-axis points into the camera frame
        transformed_axis = (R @ local_axis.T).T + t

        # Plot them
        self.ax.scatter(
            transformed_axis[:, 0],
            transformed_axis[:, 1],
            transformed_axis[:, 2],
            color='darkorange', s=30, label='Object local axis'
        )

        def plot_quiver(start, end, color):
            self.ax.quiver(
                start[0], start[1], start[2],
                end[0] - start[0],
                end[1] - start[1],
                end[2] - start[2],
                color=color, arrow_length_ratio=0.2
            )

        # Draw local X, Y, Z from the object's origin
        origin_pt = transformed_axis[0]
        x_tip     = transformed_axis[1]
        y_tip     = transformed_axis[2]
        z_tip     = transformed_axis[3]
        plot_quiver(origin_pt, x_tip, 'r')
        plot_quiver(origin_pt, y_tip, 'g')
        plot_quiver(origin_pt, z_tip, 'b')

        # ---------------------------------------------------------------------
        # Fixed manual axes: from 0 to 3 on each axis
        # ---------------------------------------------------------------------
        self.ax.set_xlim(-1/2, 1/2)
        self.ax.set_ylim(-1/2, 1/2)
        self.ax.set_zlim(-0.5, 0.5)
        
        self.ax.legend()
        self.ax.grid(True)

        plt.draw()
        plt.pause(0.01)  # allow GUI event loop to process

    def show(self):
        """Display the figure until the window is closed."""
        plt.show(block=True)


if __name__ == "__main__":
    # Example usage:
    # 1) Provide your JSON file path
    json_path = "pose_estimation_research_146.json"
    
    # 2) Initialize the viewer
    viewer = TrajectoryViewer(json_path)
    
    # 3) Show the figure & interact:
    #    - Press 'n' to step forward in frames
    #    - Press 'b' to step backward
    #    - Close the window or Ctrl+C to exit
    viewer.show()



# import cv2
# import json
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.cm as cm

# plt.ioff()  # Disable interactive mode

# # Identity since your R_gt_to_world is identity in the snippet
# R_gt_to_world = np.eye(3, dtype=float)

# def load_pose_data(pose_file):
#     with open(pose_file, 'r') as f:
#         pose_data = json.load(f)
#     return pose_data

# def load_gt_data(gt_file):
#     with open(gt_file, 'r') as f:
#         gt_data = json.load(f)
        
#     gt_camera_positions = [np.array([item['camera_position']['X'], 
#                                      item['camera_position']['Y'], 
#                                      item['camera_position']['Z']]) 
#                            for item in gt_data]
#     gt_target_positions = [np.array([item['target_position']['X'], 
#                                      item['target_position']['Y'], 
#                                      item['target_position']['Z']]) 
#                            for item in gt_data]
    
#     # Transform to world frame (here, identity)
#     gt_camera_positions_world = np.dot(np.array(gt_camera_positions), R_gt_to_world.T)
#     gt_target_positions_world = np.dot(np.array(gt_target_positions), R_gt_to_world.T)
    
#     # Align first target to origin
#     initial_target_pos = gt_target_positions_world[0]
#     translation = -initial_target_pos
    
#     gt_camera_positions_world_aligned = gt_camera_positions_world + translation
#     gt_target_positions_world_aligned = gt_target_positions_world + translation
    
#     return gt_camera_positions_world_aligned, gt_target_positions_world_aligned

# def set_axes_equal(ax):
#     """
#     Make axes of 3D plot have equal scale so that spheres/cubes aren’t distorted.
#     """
#     limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
#     centers = np.mean(limits, axis=1)
#     spans = np.max(limits, axis=1) - np.min(limits, axis=1)
#     max_span = np.max(spans)
    
#     bounds = [center - max_span / 2 for center in centers]
#     limits = [(bound, bound + max_span) for bound in bounds]
    
#     ax.set_xlim3d(limits[0])
#     ax.set_ylim3d(limits[1])
#     ax.set_zlim3d(limits[2])

# def is_pose_valid(frame_data, position_threshold=3.0, reproj_error_threshold=5.0):
#     """
#     Decide if this frame’s pose is “valid” based on the difference between raw and KF tvec
#     and the reprojection error. Adjust thresholds as needed.
#     """
#     # We look for 'translation_vector' and 'kf_translation_vector' in the JSON
#     if 'translation_vector' not in frame_data or 'kf_translation_vector' not in frame_data:
#         return False

#     # The raw leader's position in camera frame:
#     leader_pos_raw = np.array(frame_data['translation_vector'], dtype=float)
#     # The Kalman-filtered leader's position in camera frame:
#     leader_pos_kf = np.array(frame_data['kf_translation_vector'], dtype=float)

#     position_diff = np.linalg.norm(leader_pos_raw - leader_pos_kf)
#     mean_reproj_error = frame_data.get('mean_reprojection_error', float('inf'))

#     # You can tune these thresholds:
#     if mean_reproj_error > reproj_error_threshold or position_diff > position_threshold:
#         print(f"Frame {frame_data.get('frame')} - pos_diff: {position_diff:.2f}, "
#               f"mean_reproj_error: {mean_reproj_error:.2f}")
#         return False

#     return True

# def visualize_pose_and_matches(pose_data,
#                                gt_camera_positions,
#                                gt_target_positions,
#                                video_path,
#                                anchor_image_path,
#                                anchor_keypoints_2D,
#                                anchor_keypoints_3D,
#                                K):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print('Error opening video file.')
#         return

#     anchor_image = cv2.imread(anchor_image_path)
#     if anchor_image is None:
#         print('Failed to load anchor image.')
#         return

#     # Prepare 2D/3D anchor keys (not specifically used in 3D plotting, but for reference)
#     anchor_keypoints_2D = np.array(anchor_keypoints_2D, dtype=np.float32)
#     anchor_keypoints_3D = np.array(anchor_keypoints_3D, dtype=np.float32)

#     # Create the figure
#     fig = plt.figure(figsize=(12, 6))
#     gs = fig.add_gridspec(1, 2)
#     ax_image = fig.add_subplot(gs[0, 0])
#     ax_3d = fig.add_subplot(gs[0, 1], projection='3d')

#     frame_idx = 0
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Instead of “camera_positions,” rename to “leader_positions_cam”
#     # because these are the leader’s positions in the camera frame.
#     leader_positions_cam_raw = []
#     leader_positions_cam_kf = []

#     def update_visualization():
#         nonlocal frame_idx
#         ax_image.clear()
#         ax_3d.clear()

#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print(f'Failed to read frame {frame_idx}.')
#             return

#         # Retrieve pose data for this frame
#         frame_data = next((fd for fd in pose_data if fd['frame'] == frame_idx + 1), None)
#         if frame_data is None:
#             print(f'Frame data for frame {frame_idx + 1} not found.')
#             return

#         # Keypoints from JSON
#         mkpts0 = np.array(frame_data.get('mkpts0', []), dtype=np.float32)
#         mkpts1 = np.array(frame_data.get('mkpts1', []), dtype=np.float32)
#         mconf = np.array(frame_data.get('mconf', []), dtype=np.float32)
#         inliers = np.array(frame_data.get('inliers', []), dtype=int)

#         # Combine images side-by-side
#         anchor_gray = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         h1, w1 = anchor_gray.shape
#         h2, w2 = frame_gray.shape
#         comb_h = max(h1, h2)
#         comb_w = w1 + w2
#         combined = np.zeros((comb_h, comb_w), dtype=np.uint8)
#         combined[:h1, :w1] = anchor_gray
#         combined[:h2, w1:] = frame_gray

#         # Adjust mkpts1 horizontally
#         mkpts1_for_plot = mkpts1.copy()
#         mkpts1_for_plot[:, 0] += w1

#         # Plot matches in ax_image
#         ax_image.imshow(combined, cmap='gray')
#         ax_image.axis('off')
#         for i in range(len(mkpts0)):
#             x0, y0 = mkpts0[i]
#             x1, y1 = mkpts1_for_plot[i]
#             color = 'lime' if i in inliers else 'yellow'
#             ax_image.plot([x0, x1], [y0, y1], color=color, linewidth=1)
#         ax_image.set_title(f'Frame {frame_idx + 1}: Matches')

#         # Check validity and plot 3D
#         if is_pose_valid(frame_data):
#             # Raw leader’s pose from solvePnP
#             leader_pos_raw = np.array(frame_data['translation_vector'], dtype=float)
#             # Kalman-filtered pose
#             leader_pos_kf = np.array(frame_data['kf_translation_vector'], dtype=float)

#             leader_positions_cam_raw.append(leader_pos_raw)
#             leader_positions_cam_kf.append(leader_pos_kf)

#             leader_positions_cam_raw_arr = np.array(leader_positions_cam_raw)
#             leader_positions_cam_kf_arr  = np.array(leader_positions_cam_kf)

#             # -------------------------------------------------------------------------
#             # If you want to compare to GT (which is in a "world" frame),
#             # you can still plot them, but note they won't align with your
#             # camera-frame poses unless you do a coordinate transform.
#             # -------------------------------------------------------------------------
#             ax_3d.plot(gt_camera_positions[:, 0],
#                        gt_camera_positions[:, 1],
#                        gt_camera_positions[:, 2],
#                        c='blue', marker='.', label='GT Camera (World)')

#             ax_3d.plot(gt_target_positions[:, 0],
#                        gt_target_positions[:, 1],
#                        gt_target_positions[:, 2],
#                        c='cyan', marker='x', label='GT Target (World)')

#             # Plot the leader's raw positions in the camera frame
#             ax_3d.plot(leader_positions_cam_raw_arr[:, 0],
#                        leader_positions_cam_raw_arr[:, 1],
#                        leader_positions_cam_raw_arr[:, 2],
#                        c='red', marker='o', label='Leader (Raw) in Cam')

#             # Plot the Kalman-filtered leader trajectory in the camera frame
#             ax_3d.plot(leader_positions_cam_kf_arr[:, 0],
#                        leader_positions_cam_kf_arr[:, 1],
#                        leader_positions_cam_kf_arr[:, 2],
#                        c='green', marker='^', label='Leader (KF) in Cam')

#             # Optionally draw coordinate axes at the camera origin:
#             ax_3d.scatter(0, 0, 0, c='magenta', s=40, label='Camera Origin')

#             ax_3d.set_xlabel('X_cam')
#             ax_3d.set_ylabel('Y_cam')
#             ax_3d.set_zlabel('Z_cam')
#             ax_3d.set_title('Leader Pose in Camera Frame')

#             # Avoid legend duplication
#             handles, labels = ax_3d.get_legend_handles_labels()
#             by_label = dict(zip(labels, handles))
#             ax_3d.legend(by_label.values(), by_label.keys())

#             ax_3d.view_init(elev=20., azim=30)

#         set_axes_equal(ax_3d)

#         plt.draw()
#         plt.pause(0.001)

#     def on_key(event):
#         nonlocal frame_idx
#         if event.key == 'n':
#             frame_idx = min(frame_idx + 1, total_frames - 1)
#             update_visualization()
#         elif event.key == 'p':
#             frame_idx = max(frame_idx - 1, 0)
#             update_visualization()
#         elif event.key == 'q':
#             plt.close(fig)

#     fig.canvas.mpl_connect('key_press_event', on_key)
#     update_visualization()
#     plt.show()
#     cap.release()


# if __name__ == '__main__':
#     # Paths and parameters
#     pose_file = '/home/runbk0401/SuperGluePretrainedNetwork/pose_estimation_research_145.json'  # Replace with your actual path
#     video_path = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/video/20241224/20241224_new_cali.mp4'  # Replace with your actual path
#     anchor_image_path = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/realAnchor.png'  # Replace with your actual path
#     gt_file = '/home/runbk0401/SuperGluePretrainedNetwork/GT/extracted_positions(1).json'  # Replace with your actual path

#     ####################################################################################################
#     # Calibration Parameters for Real Calibration Phone (perspectiveProjWithoutDistortion)
#     focal_length_x = 1121.87155
#     focal_length_y = 1125.27185
#     cx = 642.208561
#     cy = 394.971663

#     # Distortion coefficients from "perspectiveProjWithDistortion" model in the XML
#     distCoeffs = np.array([-2.28097367e-03, 1.33152199e+00, 1.09716884e-02, 1.68743767e-03, -8.17039260e+00], dtype=np.float32)

#     ####################################################################################################

#     # Intrinsic camera matrix (K)
#     K = np.array([
#                 [focal_length_x, 0, cx],
#                 [0, focal_length_y, cy],
#                 [0, 0, 1]
#             ], dtype=np.float32)

#     # Provided 2D and 3D keypoints for the anchor image: TAIL (Aligned to GT coordinate frame 20241218)
#     anchor_keypoints_2D = np.array([
#             [563, 565], [77, 582], [515, 318], [606, 317], [612, 411],
#             [515, 414], [420, 434], [420, 465], [618, 455], [500, 123],
#             [418, 153], [417, 204], [417, 243], [502, 279], [585, 240],
#             [289, 26], [322, 339], [349, 338], [349, 374], [321, 375],
#             [390, 349], [243, 462], [367, 550], [368, 595], [383, 594],
#             [386, 549], [779, 518], [783, 570]
#         ], dtype=np.float32)

#     anchor_keypoints_3D = np.array([
#             [0.03, -0.05, -0.165],
#             [-0.190, -0.050, -0.165],
#             [0.010, -0.0, -0.025],
#             [0.060, -0.0, -0.025],
#             [0.06, -0.0, -0.080],
#             [0.010, -0.0, -0.080],
#             [-0.035, -0.0, -0.087],
#             [-0.035, -0.0, -0.105],
#             [0.065, -0.0, -0.105],
#             [0.0, 0.0, 0.045],
#             [-0.045, 0.0,0.078 ],
#             [-0.045, 0.0, 0.046],
#             [-0.045, 0.0, 0.023],
#             [0.0, -0.0, 0.0],
#             [0.045, 0.0, 0.022],
#             [-0.120, 0.0, 0.160],
#             [-0.095, -0.0,-0.035],
#             [-0.080, -0.0, -0.035],
#             [-0.080, -0.0, -0.055],
#             [-0.095, -0.0, -0.055],
#             [-0.050, -0.010, -0.040],
#             [-0.135, -0.0, -0.1],
#             [-0.060, -0.050,-0.155],
#             [-0.060, -0.050,-0.175],
#             [-0.052, -0.050, -0.175],
#             [-0.052, -0.050, -0.155],
#             [0.135, -0.050, -0.147],
#             [0.135, -0.050, -0.172]
#         ], dtype=np.float32)

#     ####################################################################################################

#     # Load the pose data
#     pose_data = load_pose_data(pose_file)
#     gt_camera_positions, gt_target_positions = load_gt_data(gt_file)


#     visualize_pose_and_matches(
#         pose_data,
#         gt_camera_positions,
#         gt_target_positions,
#         video_path,
#         anchor_image_path,
#         anchor_keypoints_2D,
#         anchor_keypoints_3D,
#         K
#     )
