import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Coordinates as a list of lists
coordinates = [
    [-0.014,  0.000,  0.042],
            [ 0.025, -0.014, -0.011],
            [ 0.049, -0.016, -0.011],
            [-0.014,  0.000, -0.042],
            [-0.014,  0.000,  0.156],
            [-0.023,  0.000, -0.065],
            [ 0.000,  0.000, -0.156],
            [ 0.025,  0.000, -0.015],
            [ 0.217,  0.000,  0.070],#
            [ 0.230,  0.000, -0.070],
            [-0.014,  0.000, -0.156],
            [ 0.000,  0.000,  0.042],
            [-0.057, -0.018, -0.010],
            [-0.074, -0.000,  0.128],
            [ 0.206, -0.070, -0.002],
            [-0.000, -0.000,  0.156],
            [-0.017, -0.000, -0.092],
            [ 0.217, -0.000, -0.027],#
            [-0.052, -0.000, -0.097],
            [-0.019, -0.000,  0.128],
            [-0.035, -0.018, -0.010],
            [ 0.217, -0.000, -0.070],
            [-0.080, -0.000,  0.156],
            [ 0.230, -0.000,  0.070],
            [-0.023, -0.000, -0.075],
            [-0.029, -0.000, -0.127],
            [-0.090, -0.000, -0.042],
            [ 0.206, -0.055, -0.002],
            [-0.090, -0.000, -0.015],
            [ 0.000, -0.000, -0.015],
            [-0.037, -0.000, -0.097],
            [-0.074, -0.000,  0.074],
            [-0.019, -0.000,  0.074],
            [ 0.230, -0.000, -0.113],#
            [-0.100, -0.030,  0.000],#
            [ 0.170, -0.000, -0.015],
            [ 0.230, -0.000,  0.113],
            [-0.000, -0.025, -0.240],
            [-0.000, -0.025,  0.240],
            [ 0.243, -0.104,  0.000],
            [-0.080, -0.000, -0.156]
]

# Converting to numpy array for easier manipulation
coordinates = np.array(coordinates)

# Extract x, y, z components
x = coordinates[:, 0]
y = coordinates[:, 1]
z = coordinates[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(x, y, z, c='b', marker='o')

# Labeling axes
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Set the same scale for all axes
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
mid_x = (x.max() + x.min()) * 0.5
mid_y = (y.max() + y.min()) * 0.5
mid_z = (z.max() + z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Title
ax.set_title('3D Scatter Plot of Points')

# Show the plot
plt.show()
