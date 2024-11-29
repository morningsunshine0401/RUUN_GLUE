import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def llh_to_ecef(lat, lon, h):
    """Convert LLH to ECEF coordinates."""
    a = 6378137.0  # WGS-84 semi-major axis
    e2 = 6.69437999014e-3  # WGS-84 first eccentricity squared

    lat, lon = math.radians(lat), math.radians(lon)
    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)

    x = (N + h) * math.cos(lat) * math.cos(lon)
    y = (N + h) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - e2) + h) * math.sin(lat)
    return np.array([x, y, z])

def ecef_to_enu(ecef, ref_ecef, ref_llh):
    """Convert ECEF to ENU coordinates relative to a reference point."""
    lat, lon = math.radians(ref_llh[0]), math.radians(ref_llh[1])

    # Rotation matrix from ECEF to ENU
    R = np.array([
        [-math.sin(lon), math.cos(lon), 0],
        [-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat)],
        [math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat)]
    ])
    enu = R @ (ecef - ref_ecef)
    return enu

def read_llh_file(llh_file):
    """Read an LLH file and extract latitude, longitude, and height."""
    positions = []
    with open(llh_file, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            data = line.strip().split()
            try:
                # Adjust the column indices to match the actual structure
                lat, lon, h = map(float, data[2:5])  # Extract 3rd, 4th, and 5th columns
                positions.append((lat, lon, h))
            except ValueError as e:
                print(f"Error parsing line {line_number}: {e}")  # Debugging: show error
    return positions

def calculate_positions(base_llh, rover_llh_file):
    """Calculate the relative and absolute positions of the rover."""
    # Convert base LLH to ECEF
    base_ecef = llh_to_ecef(*base_llh)

    # Read rover positions
    rover_positions = read_llh_file(rover_llh_file)

    # Calculate relative and absolute positions for the rover
    relative_positions = []
    absolute_positions = []
    for lat, lon, h in rover_positions:
        rover_ecef = llh_to_ecef(lat, lon, h)
        enu = ecef_to_enu(rover_ecef, base_ecef, base_llh)
        relative_positions.append(enu)
        absolute_positions.append(rover_ecef)
    
    return relative_positions, absolute_positions, rover_positions

def plot_positions(relative_positions, absolute_positions, raw_llh):
    """Plot relative, absolute, and raw LLH positions."""
    enu = np.array(relative_positions)
    east, north, up = enu[:, 0], enu[:, 1], enu[:, 2]
    
    absolute = np.array(absolute_positions)
    abs_x, abs_y, abs_z = absolute[:, 0], absolute[:, 1], absolute[:, 2]

    raw_llh = np.array(raw_llh)
    lat, lon, h = raw_llh[:, 0], raw_llh[:, 1], raw_llh[:, 2]

    fig = plt.figure(figsize=(20, 7))

    # Plot Relative Positions
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(east, north, up, label='Relative Rover Trajectory', marker='o')
    ax1.set_xlabel("East (m)")
    ax1.set_ylabel("North (m)")
    ax1.set_zlabel("Up (m)")
    ax1.set_title("Relative Position to Base")
    ax1.legend()

    # Plot Absolute Positions
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(abs_x, abs_y, abs_z, label='Absolute Rover Trajectory', marker='o', color='green')
    ax2.set_xlabel("X (ECEF, m)")
    ax2.set_ylabel("Y (ECEF, m)")
    ax2.set_zlabel("Z (ECEF, m)")
    ax2.set_title("Absolute ECEF Position")
    ax2.legend()

    # Plot Raw LLH
    ax3 = fig.add_subplot(133)
    ax3.plot(lon, lat, label='Lat/Lon Scatter', marker='o', linestyle='')
    ax3.set_xlabel("Longitude (deg)")
    ax3.set_ylabel("Latitude (deg)")
    ax3.set_title("Raw LLH Scatter (Lat/Lon)")
    ax3.legend()

    plt.show()


# Define the base marker LLH coordinates (latitude, longitude, ellipsoidal height)

#base_marker_llh = (37.60028502, 126.86488888, 35.738)  # (latitude, longitude, height in meters)

base_marker_llh = (37.60059782, 126.86587867, 26.091)  # (latitude, longitude, height in meters)

# Filepath to the rover LLH file
rover_llh_file = "/home/runbk0401/SuperGluePretrainedNetwork/assets/RTK/20241129/reach_rover_solution_20241129083905.LLH"
#rover_llh_file = "/home/runbk0401/SuperGluePretrainedNetwork/assets/RTK/20241129/camera/reach_rover_solution_20241129090448.LLH"

# Calculate positions and plot
relative_positions, absolute_positions, raw_llh = calculate_positions(base_marker_llh, rover_llh_file)
plot_positions(relative_positions, absolute_positions, raw_llh)
