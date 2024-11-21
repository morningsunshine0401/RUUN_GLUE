import os
import glob
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image

# Define individual augmentation transformations on tensors with CUDA support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

brightness_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.15),
])

blur_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))], p=1.0),  # Always apply
])

rotation_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomRotation(degrees=10),
])

mild_augmentation_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.15),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))], p=0.3),
    transforms.RandomRotation(degrees=10),
])

# Function to find the highest numbered image in the dataset
def find_highest_numbered_image(base_dir):
    max_number = 0
    for image_file in glob.glob(os.path.join(base_dir, '**', '*.png'), recursive=True):
        try:
            number = int(os.path.basename(image_file).split('.')[0])
            if number > max_number:
                max_number = number
        except ValueError:
            continue
    return max_number

# Function to apply augmentations on CUDA and save images
def apply_and_save_augmentations(image_path, save_dir, start_number):
    with Image.open(image_path).convert('RGB') as image:
        # Convert image to tensor and move to GPU
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

        augmentations = {
            "1": resize_transform,
            "2": brightness_transform,
            "3": blur_transform,
            "4": rotation_transform,
            "5": mild_augmentation_transforms
        }

        for _, transform in augmentations.items():
            # Apply the transform (on CPU for now since transforms are not GPU-accelerated)
            augmented_tensor = transform(image_tensor.cpu()).squeeze(0)
            augmented_image = to_pil_image(augmented_tensor)

            new_image_name = f"{start_number}.png"
            new_image_path = os.path.join(save_dir, new_image_name)
            augmented_image.save(new_image_path)
            print(f"Saved: {new_image_path}")
            start_number += 1

# Main function to handle augmentation only for the 'val' category
def augment_val_data_only(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    start_number = find_highest_numbered_image(base_dir) + 1

    category = 'train'
    category_dir = os.path.join(base_dir, category)

    # Process each viewpoint in the 'val' directory
    for viewpoint in ['1']:
        viewpoint_dir = os.path.join(category_dir, viewpoint)
        if not os.path.exists(viewpoint_dir):
            print(f"Directory not found: {viewpoint_dir}")
            continue

        augmented_viewpoint_dir = os.path.join(output_dir, category, viewpoint)
        os.makedirs(augmented_viewpoint_dir, exist_ok=True)

        image_files = glob.glob(os.path.join(viewpoint_dir, '*.png'))
        for image_file in image_files:
            apply_and_save_augmentations(image_file, augmented_viewpoint_dir, start_number)
            start_number += 5  # Increment for the next batch of augmentations

if __name__ == "__main__":
    # Input base directory (update this path to your dataset path)
    base_dir = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/20241118_learn/Data/'
    
    # Output directory for augmented images
    output_dir = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/20241118_learn/Resnet50/'
    
    # Perform data augmentation only on 'val' category
    augment_val_data_only(base_dir, output_dir)
