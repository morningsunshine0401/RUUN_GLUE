import numpy as np
import matplotlib.pyplot as plt

# Function to compute rotation matrix to look from a point towards another point
def compute_camera_rotation(eye, target, up=np.array([0, 0, -1], dtype=np.float64)):
    # Ensure inputs are float64
    eye = eye.astype(np.float64)
    target = target.astype(np.float64)
    up = up.astype(np.float64)

    # Compute the forward vector (from eye to target)
    forward = target - eye
    forward /= np.linalg.norm(forward)

    # Compute the right vector
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    # Recompute the up vector
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)

    # Create rotation matrix
    R = np.vstack([right, true_up, forward]).T
    return R


# Function to plot coordinate frame axes
def plot_axes(ax, R, t, axis_length=1.0, label_prefix=''):
    # Create arrays for the axes
    origin = t.flatten()
    x_axis = R[:, 0] * axis_length
    y_axis = R[:, 1] * axis_length
    z_axis = R[:, 2] * axis_length

    # Plot the axes
    ax.quiver(*origin, *x_axis, color='red', length=axis_length, normalize=True)
    ax.quiver(*origin, *y_axis, color='green', length=axis_length, normalize=True)
    ax.quiver(*origin, *z_axis, color='blue', length=axis_length, normalize=True)

    # Label the axes
    ax.text(*(origin + x_axis), f'{label_prefix}X', color='red')
    ax.text(*(origin + y_axis), f'{label_prefix}Y', color='green')
    ax.text(*(origin + z_axis), f'{label_prefix}Z', color='blue')

# Main code
def main():
    # Define the world coordinate frame (origin at (0, 0, 0))
    world_origin = np.array([0, 0, 0], dtype=np.float64)
    R_world = np.eye(3)  # Identity matrix (no rotation)

    # Define the object position (e.g., at the world origin)
    object_position = np.array([0, 0, 0], dtype=np.float64)

    # Define the camera position (e.g., at (15, 15, 0))
    camera_position = np.array([15, 15, 0], dtype=np.float64)

    # Compute the camera rotation matrix to look at the object
    R_camera = compute_camera_rotation(camera_position, object_position)

    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('World and Camera Coordinate Frames')

    # Plot the world coordinate frame at the origin
    plot_axes(ax, R_world, world_origin, axis_length=5, label_prefix='World ')

    # Plot the camera coordinate frame at the camera's position
    plot_axes(ax, R_camera, camera_position, axis_length=5, label_prefix='Camera ')

    # Plot the object position
    ax.scatter(object_position[0], object_position[1], object_position[2],
               color='black', s=100, label='Object')

    # Optionally, plot a line from the camera to the object
    ax.plot([camera_position[0], object_position[0]],
            [camera_position[1], object_position[1]],
            [camera_position[2], object_position[2]],
            color='gray', linestyle='dashed')

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Set aspect ratio
    max_range = np.array([
        camera_position[0], camera_position[1], camera_position[2]
    ]).max() * 1.5

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
