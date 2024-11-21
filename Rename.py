# import os

# # Base directory containing your train and val images
# base_dir = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/learning/'

# # Function to find the highest numbered file in the directory
# def find_last_number(root):
#     highest_num = 0
#     print(f"Checking directory: {root}")
#     for filename in os.listdir(root):
#         if filename.endswith((".png", ".jpg", ".jpeg")):
#             print(f"Found file: {filename}")
#             # Extract number from the filename
#             try:
#                 num = int(os.path.splitext(filename)[0])
#                 if num > highest_num:
#                     highest_num = num
#             except ValueError:
#                 pass  # Ignore files that don't have a numeric name
#     print(f"Highest number in directory {root}: {highest_num}")
#     return highest_num

# # Walk through all directories and files under base_dir
# for root, dirs, files in os.walk(base_dir):
#     print(f"Processing directory: {root}")
#     # Find the highest numbered file in this directory
#     #last_num = find_last_number(root)
#     last_num = 600
    
#     for filename in files:
#         # Only process image files (e.g., .png, .jpg, .jpeg) that start with 'aug_'
#         if filename.endswith((".png", ".jpg", ".jpeg")) and filename.startswith('aug_'):
#             print(f"Processing file: {filename}")
#             old_path = os.path.join(root, filename)

#             # Increment the last_num to assign a new unique number to the file
#             last_num += 1
#             new_filename = f"{last_num}.png"
#             new_path = os.path.join(root, new_filename)

#             # Safely rename the file
#             try:
#                 print(f"Renaming {old_path} -> {new_path}")
#                 os.rename(old_path, new_path)
#                 print(f"Renamed: {old_path} -> {new_path}")
#             except Exception as e:
#                 print(f"Error renaming {old_path}: {e}")


import os

# Specify the directory where the images are located
directory = "/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/20241118_learn/Resnet50/train/1"

# List all files in the directory
files = os.listdir(directory)

# Loop through the files and rename them
for index, filename in enumerate(files):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Specify image file extensions
        new_name = f"{index + 1}.png"  # Change format as needed
        src = os.path.join(directory, filename)
        dst = os.path.join(directory, new_name)
        os.rename(src, dst)

print(f"Renamed {len(files)} images.")
