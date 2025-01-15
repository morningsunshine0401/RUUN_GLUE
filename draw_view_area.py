import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define grid for the sphere
phi = np.linspace(0, np.pi, 100)  # Latitude (0 to 180 degrees)
theta = np.linspace(0, 2 * np.pi, 100)  # Longitude (0 to 360 degrees)
phi, theta = np.meshgrid(phi, theta)

# Sphere coordinates
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Define 8 regions based on phi and theta
region = np.zeros_like(phi, dtype=int)

# Upper hemisphere (phi < π/2)
region[(phi < np.pi / 2) & (theta < np.pi / 2)] = 1  # Upper front-right
region[(phi < np.pi / 2) & (theta >= np.pi / 2) & (theta < np.pi)] = 2  # Upper front-left
region[(phi < np.pi / 2) & (theta >= np.pi) & (theta < 3 * np.pi / 2)] = 3  # Upper back-left
region[(phi < np.pi / 2) & (theta >= 3 * np.pi / 2)] = 4  # Upper back-right

# Lower hemisphere (phi >= π/2)
region[(phi >= np.pi / 2) & (theta < np.pi / 2)] = 5  # Lower front-right
region[(phi >= np.pi / 2) & (theta >= np.pi / 2) & (theta < np.pi)] = 6  # Lower front-left
region[(phi >= np.pi / 2) & (theta >= np.pi) & (theta < 3 * np.pi / 2)] = 7  # Lower back-left
region[(phi >= np.pi / 2) & (theta >= 3 * np.pi / 2)] = 8  # Lower back-right

# Color map for the regions with gray tones and one red region
colors = np.empty(region.shape, dtype=object)
colors[region == 1] = '#ff9999'  # Light red for one upper region
colors[region == 2] = '#d9d9d9'  # Light gray
colors[region == 3] = '#bfbfbf'  # Medium light gray
colors[region == 4] = '#a6a6a6'  # Medium gray
colors[region == 5] = '#8c8c8c'  # Medium-dark gray
colors[region == 6] = '#737373'  # Dark gray
colors[region == 7] = '#595959'  # Darker gray
colors[region == 8] = '#404040'  # Darkest gray

# Plot the sphere with transparency
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, facecolors=colors, edgecolor='none', alpha=0.6)  # Transparency with alpha=0.6

# Hide axes
ax.axis('off')

# Set aspect ratio for a perfect sphere
ax.set_box_aspect([1, 1, 1])

# Show plot
plt.show()
