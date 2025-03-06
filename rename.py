import os

# Path to your image directory
image_dir = "assets/Ruun_images/viewpoint/Blender/20250305/"

# Loop through files in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".png") and filename.startswith("0001"):
        new_filename = "0000" + filename[4:]  # Replace '0001' with '0000'
        #new_filename = "0000000" + filename[0:]  # Replace '0001' with '0000'
        old_path = os.path.join(image_dir, filename)
        new_path = os.path.join(image_dir, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed {filename} -> {new_filename}")

print("Renaming complete.")
