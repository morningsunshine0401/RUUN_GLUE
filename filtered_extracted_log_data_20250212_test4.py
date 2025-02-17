# import json
# import pandas as pd
# import numpy as np

# # Load JSON data
# json_file_path = "extracted_pose_data.json"
# with open(json_file_path, "r") as f:
#     data = json.load(f)

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Frames with high errors
# high_roll_frames = [80, 85, 92, 116, 117, 119, 124, 128, 134, 149, 153]
# high_pitch_frames = [7, 80, 86, 118, 120, 142, 144, 145, 149, 151, 159]
# high_yaw_frames = [85, 89, 134]

# # Include neighboring frames (+-3)
# def get_neighboring_frames(frames, range_limit=3):
#     expanded_frames = set()
#     for frame in frames:
#         for i in range(-range_limit, range_limit + 1):
#             expanded_frames.add(frame + i)
#     return sorted(expanded_frames)

# all_roll_frames = get_neighboring_frames(high_roll_frames)
# all_pitch_frames = get_neighboring_frames(high_pitch_frames)
# all_yaw_frames = get_neighboring_frames(high_yaw_frames)

# # Orientation error data
# orientation_errors = np.array([
#     [13.128, -11.374, 5.811], [13.975, -10.654, 8.068], [14.268, -10.338, 6.439],
#     [7.921, -12.149, 6.876], [8.680, -5.980, 3.947], [11.209, -8.738, 5.030],
#     [12.520, -16.566, 6.403], [16.154, -11.392, 8.700], [8.525, -5.229, 3.880],
#     [11.279, -3.696, 4.103], [11.600, -8.495, 6.177], [10.747, -4.344, 4.093],
#     [15.350, -9.540, 8.006], [9.917, -4.850, 3.948], [11.158, -6.074, 4.220],
#     [10.340, -4.703, 4.989], [10.367, -6.906, 4.240], [14.013, -12.783, 7.349],
#     [15.279, -9.395, 7.336], [14.377, -9.455, 7.789], [12.606, -5.223, 4.373],
#     [13.192, -9.704, 7.910], [14.839, -8.337, 8.090], [7.134, -3.534, 2.682],
#     [14.320, -9.510, 7.601], [16.780, -9.860, 9.419], [7.645, -5.799, 3.028],
# ])  # Truncated data for example

# # Function to categorize frames
# def categorize_frames(frames, category):
#     categorized_frames = []
#     for frame in frames:
#         frame_data = df[df["frame"] == frame].to_dict(orient='records')
#         if frame_data:
#             frame_data = frame_data[0]
#             frame_index = frame % len(orientation_errors)  # Cycling through the orientation errors
#             frame_data["roll_error"], frame_data["pitch_error"], frame_data["yaw_error"] = orientation_errors[frame_index]
#             frame_data["category"] = category
#             frame_data["is_neighbor"] = frame not in eval(category)
#             categorized_frames.append(frame_data)
#     return categorized_frames

# # Categorize frames
# categorized_data = {
#     "high_roll_frames": categorize_frames(all_roll_frames, "high_roll_frames"),
#     "high_pitch_frames": categorize_frames(all_pitch_frames, "high_pitch_frames"),
#     "high_yaw_frames": categorize_frames(all_yaw_frames, "high_yaw_frames")
# }

# # Convert to DataFrame and display
# final_df = pd.DataFrame(
#     categorized_data["high_roll_frames"] + categorized_data["high_pitch_frames"] + categorized_data["high_yaw_frames"]
# )
# import ace_tools_open as tools
# tools.display_dataframe_to_user(name="Categorized Pose Error Data", dataframe=final_df)

# # Save final extracted data
# with open("categorized_pose_error_data.json", "w") as f:
#     json.dump(categorized_data, f, indent=4)
# final_df.to_csv("categorized_pose_error_data.csv", index=False)











import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the categorized pose error data
json_file_path = "categorized_pose_error_data.json"
with open(json_file_path, "r") as f:
    data = json.load(f)

# Convert JSON dictionary into a list of all frames with a category label
all_data = []
for category, frames in data.items():
    for frame in frames:
        frame["category"] = category  # Label each frame with its category
        all_data.append(frame)

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Ensure category names match correctly
df["category"] = df["category"].replace({
    "high_roll_frames": "high_roll",
    "high_pitch_frames": "high_pitch",
    "high_yaw_frames": "high_yaw"
})

# Categorize frames properly
high_roll_df = df[df["category"] == "high_roll"]
high_pitch_df = df[df["category"] == "high_pitch"]
high_yaw_df = df[df["category"] == "high_yaw"]

# Function to plot each category with frame markers
def plot_category(df, title):
    if df.empty:
        print(f"No data available for {title}")
        return

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)

    # Identify main frames and neighbors
    main_frames = df[df["is_neighbor"] == False]  # Selected frames
    neighbor_frames = df[df["is_neighbor"] == True]  # Neighboring frames

    # Roll/Pitch/Yaw Errors
    axs[0, 0].plot(df["frame"], df["roll_error"], marker='o', linestyle='-', color='b', label="Roll Error")
    axs[0, 0].scatter(main_frames["frame"], main_frames["roll_error"], color='red', marker='x', s=100, label="Main Frame")
    axs[0, 0].set_xlabel("Frame")
    axs[0, 0].set_ylabel("Roll Error")
    axs[0, 0].set_title("Roll Error vs. Frame")
    axs[0, 0].legend()

    axs[0, 1].plot(df["frame"], df["pitch_error"], marker='o', linestyle='-', color='g', label="Pitch Error")
    axs[0, 1].scatter(main_frames["frame"], main_frames["pitch_error"], color='red', marker='x', s=100, label="Main Frame")
    axs[0, 1].set_xlabel("Frame")
    axs[0, 1].set_ylabel("Pitch Error")
    axs[0, 1].set_title("Pitch Error vs. Frame")
    axs[0, 1].legend()

    axs[1, 0].plot(df["frame"], df["yaw_error"], marker='o', linestyle='-', color='r', label="Yaw Error")
    axs[1, 0].scatter(main_frames["frame"], main_frames["yaw_error"], color='red', marker='x', s=100, label="Main Frame")
    axs[1, 0].set_xlabel("Frame")
    axs[1, 0].set_ylabel("Yaw Error")
    axs[1, 0].set_title("Yaw Error vs. Frame")
    axs[1, 0].legend()

    # Inliers, Mahalanobis Sq, Coverage Score
    axs[1, 1].plot(df["frame"], df["inliers"], marker='o', linestyle='-', color='m', label="Inliers")
    axs[1, 1].scatter(main_frames["frame"], main_frames["inliers"], color='red', marker='x', s=100, label="Main Frame")
    axs[1, 1].set_xlabel("Frame")
    axs[1, 1].set_ylabel("Inliers")
    axs[1, 1].set_title("Inliers vs. Frame")
    axs[1, 1].legend()

    axs[2, 0].plot(df["frame"], df["mahalanobis_sq"], marker='o', linestyle='-', color='c', label="Mahalanobis Sq")
    axs[2, 0].scatter(main_frames["frame"], main_frames["mahalanobis_sq"], color='red', marker='x', s=100, label="Main Frame")
    axs[2, 0].set_xlabel("Frame")
    axs[2, 0].set_ylabel("Mahalanobis Sq")
    axs[2, 0].set_title("Mahalanobis Sq vs. Frame")
    axs[2, 0].legend()

    axs[2, 1].plot(df["frame"], df["coverage_score"], marker='o', linestyle='-', color='y', label="Coverage Score")
    axs[2, 1].scatter(main_frames["frame"], main_frames["coverage_score"], color='red', marker='x', s=100, label="Main Frame")
    axs[2, 1].set_xlabel("Frame")
    axs[2, 1].set_ylabel("Coverage Score")
    axs[2, 1].set_title("Coverage Score vs. Frame")
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()

# Plot each category
plot_category(high_roll_df, "High Roll Error Frames")
plot_category(high_pitch_df, "High Pitch Error Frames")
plot_category(high_yaw_df, "High Yaw Error Frames")

