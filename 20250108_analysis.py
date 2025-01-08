import json
import matplotlib.pyplot as plt

# Load the JSON data
file_path = "20250107_test2_thresh_15cm.json"
with open(file_path, 'r') as f:
    data = json.load(f)

# Filter the data for frames 430 to 610
filtered_data = [frame for frame in data if 430 <= frame["frame"] <= 610]

# Extract frames and metrics
frames = [frame["frame"] for frame in filtered_data]
total_matches = [frame["total_matches"] for frame in filtered_data]
num_inliers = [frame["num_inliers"] for frame in filtered_data]
inlier_ratio = [frame["inlier_ratio"] for frame in filtered_data]
mean_reprojection_error = [frame["mean_reprojection_error"] for frame in filtered_data]
mean_confidence = [sum(frame["mconf"]) / len(frame["mconf"]) if frame["mconf"] else 0 for frame in filtered_data]

# Plot Total Matches
plt.figure(figsize=(12, 6))
plt.plot(frames, total_matches, marker='o', label="Total Matches")
plt.title("Total Matches for Frames 430 to 610")
plt.xlabel("Frame")
plt.ylabel("Total Matches")
plt.grid()
plt.legend()
plt.show()

# Plot Num Inliers
plt.figure(figsize=(12, 6))
plt.plot(frames, num_inliers, marker='o', color='orange', label="Number of Inliers")
plt.title("Number of Inliers for Frames 430 to 610")
plt.xlabel("Frame")
plt.ylabel("Number of Inliers")
plt.grid()
plt.legend()
plt.show()

# Plot Inlier Ratio
plt.figure(figsize=(12, 6))
plt.plot(frames, inlier_ratio, marker='o', color='green', label="Inlier Ratio")
plt.title("Inlier Ratio for Frames 430 to 610")
plt.xlabel("Frame")
plt.ylabel("Inlier Ratio")
plt.grid()
plt.legend()
plt.show()

# Plot Mean Reprojection Error
plt.figure(figsize=(12, 6))
plt.plot(frames, mean_reprojection_error, marker='o', color='red', label="Mean Reprojection Error")
plt.title("Mean Reprojection Error for Frames 430 to 610")
plt.xlabel("Frame")
plt.ylabel("Mean Reprojection Error")
plt.grid()
plt.legend()
plt.show()

# Plot Mean Confidence
plt.figure(figsize=(12, 6))
plt.plot(frames, mean_confidence, marker='o', color='purple', label="Mean Confidence (mconf)")
plt.title("Mean Confidence for Frames 430 to 610")
plt.xlabel("Frame")
plt.ylabel("Mean Confidence")
plt.grid()
plt.legend()
plt.show()
