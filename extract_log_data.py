import re
import pandas as pd
import matplotlib.pyplot as plt

# Path to your log file
log_file_path = "pose_estimator.log"

# Regular expressions to extract relevant information
frame_pattern = re.compile(r"Frame (\d+):")
inlier_pattern = re.compile(r"inliers=(\d+)")
inlier_ratio_pattern = re.compile(r"inlier_ratio=([\d.]+)")
mean_reproj_pattern = re.compile(r"mean_reprojection_error=([\d.]+)")
viewpoint_diff_pattern = re.compile(r"viewpoint_diff=([\d.]+)°")
mahalanobis_sq_pattern = re.compile(r"mahalanobis_sq=([\d.]+)")
coverage_score_pattern = re.compile(r"coverage_score=([\d.]+)")
translation_change_pattern = re.compile(r"translation_change=([\d.]+)")
orientation_change_pattern = re.compile(r"orientation_change_deg=([\d.]+)°")
alpha_pattern = re.compile(r"Partial blend applied with alpha=([\d.]+)")
max_translation_pattern = re.compile(r"max_translation_jump_adapt=([\d.]+)")
max_orientation_pattern = re.compile(r"max_orientation_jump_adapt_deg=([\d.]+)")

# List to store extracted data
data = []

# Read the log file and extract data
with open(log_file_path, "r") as log_file:
    for line in log_file:
        frame_match = frame_pattern.search(line)
        if frame_match:
            frame_data = {
                "frame": int(frame_match.group(1)),
                "inliers": int(inlier_pattern.search(line).group(1)) if inlier_pattern.search(line) else None,
                "inlier_ratio": float(inlier_ratio_pattern.search(line).group(1)) if inlier_ratio_pattern.search(line) else None,
                "mean_reproj_error": float(mean_reproj_pattern.search(line).group(1)) if mean_reproj_pattern.search(line) else None,
                "viewpoint_diff": float(viewpoint_diff_pattern.search(line).group(1)) if viewpoint_diff_pattern.search(line) else None,
                "mahalanobis_sq": float(mahalanobis_sq_pattern.search(line).group(1)) if mahalanobis_sq_pattern.search(line) else None,
                "coverage_score": float(coverage_score_pattern.search(line).group(1)) if coverage_score_pattern.search(line) else None,
                "translation_change": float(translation_change_pattern.search(line).group(1)) if translation_change_pattern.search(line) else None,
                "orientation_change": float(orientation_change_pattern.search(line).group(1)) if orientation_change_pattern.search(line) else None,
                "alpha": float(alpha_pattern.search(line).group(1).strip(".")) if alpha_pattern.search(line) else None,
                "max_translation_jump": float(max_translation_pattern.search(line).group(1)) if max_translation_pattern.search(line) else None,
                "max_orientation_jump": float(max_orientation_pattern.search(line).group(1)) if max_orientation_pattern.search(line) else None,
                "correction_type": None  # Will be filled later
            }

            # Check for correction reason
            if "Kalman Filter correction accepted due to" in line:
                frame_data["correction_type"] = "Full Correction"

                # Extract reason for full correction
                reason_match = re.search(r"accepted due to (.+?)\.", line)
                frame_data["full_correction_reason"] = reason_match.group(1) if reason_match else "Unknown"

            elif "Partial blend applied with alpha" in line:
                frame_data["correction_type"] = "Partial Blend"
                frame_data["full_correction_reason"] = "Partial Blend"  # Log it as a reason

            elif "Exceeded max skip count" in line:
                frame_data["correction_type"] = "Forced Correction"
                frame_data["full_correction_reason"] = "Forced Correction"  # Log it as a reason

            else:
                frame_data["full_correction_reason"] = None  # No correction was applied


            data.append(frame_data)

# Convert to Pandas DataFrame
df = pd.DataFrame(data)

# Save to CSV and JSON
df.to_csv("extracted_pose_data.csv", index=False)
df.to_json("extracted_pose_data.json", orient="records", indent=4)

# -------------------- Graph Visualization --------------------
plt.figure(figsize=(12, 10))

# Inliers over frames
plt.subplot(3, 2, 1)
plt.plot(df["frame"], df["inliers"], marker='o', linestyle='-', color='b', label="Inliers")
plt.xlabel("Frame")
plt.ylabel("Inliers")
plt.title("Inliers vs. Frame")
plt.legend()

# Coverage Score over frames
plt.subplot(3, 2, 2)
plt.plot(df["frame"], df["coverage_score"], marker='o', linestyle='-', color='g', label="Coverage Score")
plt.xlabel("Frame")
plt.ylabel("Coverage Score")
plt.title("Coverage Score vs. Frame")
plt.legend()

# Mahalanobis Sq over frames
plt.subplot(3, 2, 3)
plt.plot(df["frame"], df["mahalanobis_sq"], marker='o', linestyle='-', color='r', label="Mahalanobis Sq")
plt.xlabel("Frame")
plt.ylabel("Mahalanobis Sq")
plt.title("Mahalanobis Sq vs. Frame")
plt.legend()

# ----- Translation Change -----
plt.subplot(3, 2, 4)
plt.plot(df["frame"], df["translation_change"], marker='o', linestyle='-', color='m', label="Translation Change")
plt.xlabel("Frame")
plt.ylabel("Translation Change")
plt.title("Translation Change vs. Frame")
plt.legend()

# ----- Max Translation Jump -----
plt.subplot(3, 2, 5)
plt.plot(df["frame"], df["max_translation_jump"], linestyle='--', color='r', label="Max Translation Jump")
plt.xlabel("Frame")
plt.ylabel("Max Translation Jump")
plt.title("Max Translation Jump vs. Frame")
plt.ylim(0, df["max_translation_jump"].quantile(0.95))  # Cap extreme values
plt.legend()

# ----- Orientation Change -----
plt.subplot(3, 2, 6)
plt.plot(df["frame"], df["orientation_change"], marker='o', linestyle='-', color='c', label="Orientation Change")
plt.xlabel("Frame")
plt.ylabel("Orientation Change (°)")
plt.title("Orientation Change vs. Frame")
plt.legend()

# ----- Max Orientation Jump -----
plt.figure(figsize=(10, 5))
plt.plot(df["frame"], df["max_orientation_jump"], linestyle='--', color='r', label="Max Orientation Jump")
plt.xlabel("Frame")
plt.ylabel("Max Orientation Jump")
plt.title("Max Orientation Jump vs. Frame")
plt.ylim(0, df["max_orientation_jump"].quantile(0.95))  # Cap extreme values
plt.legend()

# Partial Blend Alpha over frames
plt.subplot(3, 2, 6)
plt.plot(df["frame"], df["alpha"], marker='o', linestyle='-', color='y', label="Partial Blend Alpha")
plt.xlabel("Frame")
plt.ylabel("Alpha")
plt.title("Partial Blend Alpha vs. Frame")
plt.legend()


# Full Correction Frames Visualization
plt.figure(figsize=(12, 6))
full_correction_df = df[df["correction_type"] == "Full Correction"]

if not full_correction_df.empty:
    unique_reasons = full_correction_df["full_correction_reason"].unique()
    colors = plt.cm.get_cmap("tab10", len(unique_reasons))  # Use distinct colors

    for i, reason in enumerate(unique_reasons):
        subset = full_correction_df[full_correction_df["full_correction_reason"] == reason]
        plt.scatter(subset["frame"], [i] * len(subset), label=reason, s=100, color=colors(i), edgecolors="black")

    plt.yticks(range(len(unique_reasons)), unique_reasons)  # Set y-axis labels to reasons

plt.xlabel("Frame")
plt.title("Frames with Full Correction and Their Reasons")
plt.grid(True, linestyle="--", alpha=0.6)  # Add grid for better readability
plt.legend()
plt.show()



# Adjust layout
plt.tight_layout()

# Show graphs
plt.show()
