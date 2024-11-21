import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# Ensure interactive mode is off for proper figure updates
plt.ioff()

def interactive_pnp(anchor_image_path, other_image_path, anchor_keypoints_2D, anchor_keypoints_3D):
    # Load the images
    anchor_image = cv2.imread(anchor_image_path)
    other_image = cv2.imread(other_image_path)
    
    assert anchor_image is not None, 'Failed to load anchor image.'
    assert other_image is not None, 'Failed to load other image.'
    
    # Resize images to 640x480
    anchor_image = cv2.resize(anchor_image, (1280, 960))
    other_image = cv2.resize(other_image, (1280, 960))
    
    # Convert images to RGB for matplotlib
    anchor_image_rgb = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2RGB)
    other_image_rgb = cv2.cvtColor(other_image, cv2.COLOR_BGR2RGB)
    
    # Prepare figure and axes
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    ax_anchor = fig.add_subplot(gs[0, 0])
    ax_other = fig.add_subplot(gs[0, 1])
    ax_3d = fig.add_subplot(gs[1, :], projection='3d')
    
    # Display the images
    ax_anchor.imshow(anchor_image_rgb)
    ax_anchor.set_title('Anchor Image')
    ax_anchor.axis('off')
    
    ax_other.imshow(other_image_rgb)
    ax_other.set_title('Other Image')
    ax_other.axis('off')
    
    # Initialize lists to store keypoints and matches
    other_keypoints_2D = []   # 2D keypoints in other image
    matches = []              # List of (anchor_idx, other_idx)
    
    # Plot predefined anchor keypoints
    anchor_keypoints_2D = np.array(anchor_keypoints_2D)
    anchor_circles = ax_anchor.scatter(anchor_keypoints_2D[:, 0], anchor_keypoints_2D[:, 1], s=50, c='cyan', marker='o')
    
    # Variables for interaction
    selected_anchor_idx = [None]  # Use a list to preserve across closures
    selected_other_idx = None
    circles_other = []
    lines = []
    
    #focal_length_x = 778.38449164772408  # px
    #focal_length_y = 780.92121918822045  # py
    #cx = 336.10046045116735  # Principal point u0
    #cy = 258.73402363943103  # Principal point v0
    focal_length_x = 1777.77777  # px
    focal_length_y = 1777.77777  # py
    cx = 319.5  # Principal point u0
    cy = 239.5  # Principal point v0


    # Distortion coefficients from "perspectiveProjWithDistortion" model in the XML
    distCoeffs = np.array([0.2728747755008597, -0.25885103136641374, 0, 0], dtype=np.float32)

    print(f"Principal point cx: {cx}")
    print(f"Principal point cy: {cy}")

    # Intrinsic camera matrix (K)
    K = np.array([
                [focal_length_x, 0, cx],
                [0, focal_length_y, cy],
                [0, 0, 1]
            ], dtype=np.float32)

    
    def update_pose_visualization():
        ax_3d.clear()
        ax_3d.set_title('Estimated Camera Pose')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        
        # Plot anchor keypoints in 3D
        anchor_points_3D = np.array(anchor_keypoints_3D)
        ax_3d.scatter(anchor_points_3D[:, 0], anchor_points_3D[:, 1], anchor_points_3D[:, 2],
                    c='b', marker='o', label='Anchor 3D Points')
        
        # Check if there are enough matches
        if len(matches) >= 4:
            # Get matched points
            object_points = []
            image_points = []
            for anchor_idx, other_idx in matches:
                object_points.append(anchor_keypoints_3D[anchor_idx])
                image_points.append(other_keypoints_2D[other_idx])
            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)
            
            print(f"Object points: {object_points}")
            print(f"Image points: {image_points}")
            
            # Reshape for solvePnPRansac
            object_points = object_points.reshape(-1, 3)
            image_points = image_points.reshape(-1, 2)
            
            # Solve PnPRansac
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=K,
                distCoeffs=None,
                reprojectionError=4.0,
                confidence=0.99,
                iterationsCount=1000,
                #flags=cv2.SOLVEPNP_ITERATIVE
                flags=cv2.SOLVEPNP_P3P
            )
            
            if success and inliers is not None:
                print(f"PnP RANSAC succeeded. rvec: {rvec}, tvec: {tvec}, inliers: {inliers}")
                # Get rotation matrix and camera position
                R, _ = cv2.Rodrigues(rvec)
                camera_position = -R.T @ tvec.reshape(3,)
                
                # Plot camera position
                ax_3d.scatter(camera_position[0], camera_position[1], camera_position[2],
                            c='r', marker='^', s=100, label='Estimated Camera Position')
                
                # Draw lines from camera to points
                for pt in object_points:
                    ax_3d.plot([camera_position[0], pt[0]],
                            [camera_position[1], pt[1]],
                            [camera_position[2], pt[2]],
                            c='gray', linestyle='--')
                
                ax_3d.legend()
                plt.draw()
            else:
                print("PnP RANSAC failed. Unable to estimate the pose.")
                ax_3d.text(0.5, 0.5, 0.5, 'PnP Failed', horizontalalignment='center', verticalalignment='center')
        else:
            print("Not enough matches to compute pose. At least 4 matches are required.")
            ax_3d.text(0.5, 0.5, 0.5, 'Not Enough Matches', horizontalalignment='center', verticalalignment='center')
        
        plt.draw()
    
    # Mouse click event handler for anchor image
    def on_click_anchor(event):
        if event.inaxes != ax_anchor:
            return
        x, y = event.xdata, event.ydata
        print(f"Clicked on anchor image at ({x}, {y})")  # Debugging output
        # Find the nearest existing keypoint
        distances = np.hypot(anchor_keypoints_2D[:, 0] - x, anchor_keypoints_2D[:, 1] - y)
        min_idx = np.argmin(distances)
        if distances[min_idx] < 10:  # Threshold for selecting existing keypoint
            selected_anchor_idx[0] = min_idx  # Preserve in list
            print(f'Selected anchor keypoint index: {selected_anchor_idx[0]}')  # Debugging output
            # Highlight selected keypoint
            ax_anchor.scatter(anchor_keypoints_2D[min_idx, 0], anchor_keypoints_2D[min_idx, 1],
                              s=100, facecolors='none', edgecolors='red')
            plt.draw()
        else:
            selected_anchor_idx[0] = None
            print('No anchor keypoint selected.')  # Debugging output

    # Mouse click event handler for other image
    def on_click_other(event):
        if event.inaxes != ax_other:
            return
        print(f"Clicked on other image at ({event.xdata}, {event.ydata})")  # Debugging output
        if selected_anchor_idx[0] is None:
            print('Please select an anchor keypoint first.')
            return  # Prevent further execution if no anchor keypoint is selected
        
        print(f"Using anchor keypoint index: {selected_anchor_idx[0]}")  # Debugging output
        x, y = event.xdata, event.ydata
        # Add a new keypoint in the other image
        other_keypoints_2D.append((x, y))
        selected_other_idx = len(other_keypoints_2D) - 1
        print(f"Added keypoint at ({x}, {y}) to the other image with index {selected_other_idx}")  # Debugging output
        circle = ax_other.scatter(x, y, s=50, c='orange', marker='o')
        circles_other.append(circle)
        plt.draw()
        # Create a match between the selected keypoints
        matches.append((selected_anchor_idx[0], selected_other_idx))
        print(f"Match added: Anchor keypoint {selected_anchor_idx[0]} -> Other keypoint {selected_other_idx}")  # Debugging output
        # Draw a line connecting the keypoints
        line = matplotlib.lines.Line2D(
            [anchor_keypoints_2D[selected_anchor_idx[0], 0], x + anchor_image.shape[1]],  # Shift x-coordinate for display
            [anchor_keypoints_2D[selected_anchor_idx[0], 1], y],
            transform=fig.transFigure,
            figure=fig,
            color='yellow'
        )
        fig.lines.append(line)
        lines.append(line)
        plt.draw()
        # Reset selection
        selected_anchor_idx[0] = None
        print("Reset selection after match.")  # Debugging output

    # Key press event handler
    def on_key(event):
        if event.key == 'enter':
            print("Enter key pressed. Checking if there are enough matches for pose estimation.")
            if len(matches) >= 4:
                print(f"Number of matches: {len(matches)}. Starting pose estimation.")
                update_pose_visualization()  # Call the pose visualization function
            else:
                print(f"Not enough matches for pose estimation. Only {len(matches)} matches found.")
        elif event.key == 'd':
            # Delete the last match
            if matches:
                matches.pop()
                # Remove the last line
                line = lines.pop()
                fig.lines.remove(line)
                # Remove the last keypoint in other image
                circle = circles_other.pop()
                circle.remove()
                other_keypoints_2D.pop()
                plt.draw()
            print("Deleted the last match.")
        elif event.key == 'r':
            # Reset all matches and keypoints in other image
            matches.clear()
            for line in lines:
                fig.lines.remove(line)
            lines.clear()
            other_keypoints_2D.clear()
            for circle in circles_other:
                circle.remove()
            circles_other.clear()
            plt.draw()
            print("Reset all matches and keypoints.")
        elif event.key == 'q':
            plt.close(fig)
            print("Closed the plot.")

    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_click_anchor)
    fig.canvas.mpl_connect('button_press_event', on_click_other)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("Instructions:")
    print("- Click on an anchor keypoint to select it (it will be highlighted).")
    print("- Click on the other image to add a corresponding keypoint and create a match.")
    print("- Press 'Enter' to compute and visualize the pose estimation.")
    print("- Press 'd' to delete the last match.")
    print("- Press 'r' to reset all matches and keypoints in the other image.")
    print("- Press 'q' to quit.")
    
    plt.show()

if __name__ == '__main__':
    # Paths to images
    anchor_image_path = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/cube/better_frame/0002.png'  # Replace with your anchor image path
    other_image_path = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/cube/better_frame/0040.png'#'/home/runbk0401/Pictures/Webcam/20241002.jpg'    # Replace with your other image path
    
    # Provided 2D and 3D keypoints for the anchor image
    anchor_keypoints_2D = np.array([
        [545., 274.],
        [693., 479.],
        [401., 481.],
        [548., 508.],
        [624., 539.],
        [728., 600.],
        [582., 609.],
        [648., 623.],
        [623., 656.],
        [688., 671.],
        [550., 724.]
    ], dtype=np.float32)

    anchor_keypoints_3D = np.array([
        [ 0.   ,   0.   ,   5.75  ],
        [-0.25 ,   0.25 ,   5.25  ],
        [ 0.25 ,  -0.25 ,   5.25  ],
        [ 0.25 ,   0.25 ,   5.25  ],
        [ 0.    ,  0.25 ,   5.1414],
        [-0.1414,  0.45 ,   5.    ],
        [ 0.1414,  0.25 ,   5.    ],
        [ 0.1414,  0.45 ,   5.    ],
        [ 0.    ,  0.25,    4.8586],
        [ 0.    ,  0.45,    4.8586],
        [ 0.25  ,  0.25,    4.75  ]
    ], dtype=np.float32)
    
    # Run the interactive PnP visualization
    interactive_pnp(anchor_image_path, other_image_path, anchor_keypoints_2D, anchor_keypoints_3D)
