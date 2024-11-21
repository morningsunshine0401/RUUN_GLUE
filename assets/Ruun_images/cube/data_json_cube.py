import numpy as np
import json
import os

# Function to extract object poses from the NPZ file
def extract_object_poses(npz_file, png_file):
    data = np.load(npz_file)
    frame_data = {"image": png_file, "object_poses": []}

    if 'object_poses' in data.files:
        obj_poses = data['object_poses']
        for obj in obj_poses:
            obj_name = obj['name']
            obj_pose = obj['pose'].tolist()  # Convert numpy array to list for JSON serialization
            frame_data["object_poses"].append({
                "name": obj_name,
                "pose": obj_pose
            })
    
    return frame_data

# Main script to process all pairs of PNG and NPZ files
output_data = {"frames": []}
for i in range(1, 41):  # Assuming you have 40 pairs
    npz_file = f"{str(i).zfill(4)}.npz"  # e.g., '0001.npz'
    #png_file = f"{str(i).zfill(8)}.png"  # e.g., '00010001.png'
    png_file = f"{str(i).zfill(4)}.png"  # e.g., '0001.png'


    if os.path.exists(npz_file):
        frame_data = extract_object_poses(npz_file, png_file)
        output_data["frames"].append(frame_data)

# Write the output to a JSON file
with open("weird_cube_object_poses.json", "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print("Object poses saved to object_poses.json")
