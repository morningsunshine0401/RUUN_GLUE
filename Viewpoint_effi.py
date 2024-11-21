import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn.functional as F
import time
import shutil

# Folder to save misclassified images for inspection
misclassified_folder = 'misclassified_images'
os.makedirs(misclassified_folder, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Number of classes (update if different)
num_classes = 4  # 'front', 'back', 'left', 'right'

# Class names
class_names = ['back', 'front', 'left', 'right']

# Ground truth mapping for image sequence (adjusted based on your provided ranges)
def get_ground_truth_viewpoint(image_name):
    image_num = int(image_name[4:8])  # Extract the frame number
    if 25 <= image_num <= 70:
        return 'back'
    elif 71 <= image_num <= 115:
        return 'left'
    elif 116 <= image_num <= 160:
        return 'front'
    else:
        return 'right'

# Load the pre-trained EfficientNet-B0 model
model = models.efficientnet_b0(weights='IMAGENET1K_V1')

# Replace the final layer to match the number of classes
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)

# Load the trained weights from the .pth file
model.load_state_dict(torch.load('viewpoint_model_efficientnet.pth', map_location=device))

# Move the model to the appropriate device
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to make predictions on a new image
def predict_viewpoint(model, image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
        _, preds = torch.max(outputs, 1)
    
    # Get predicted class
    class_idx = preds.item()
    predicted_viewpoint = class_names[class_idx]
    
    # Convert probabilities to percentages
    probabilities_percent = probabilities.cpu().numpy()[0] * 100
    
    return predicted_viewpoint, probabilities_percent

# Function to process a sequence of images and calculate accuracy and error details
def process_images(model, image_folder):
    # Get list of image files in the folder
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])  # Sort to ensure sequential order
    
    # Initialize time tracking and accuracy variables
    start_time = time.time()
    num_frames = 0
    correct_predictions = 0
    total_errors = 0
    
    # Initialize error tracking per category
    category_errors = {class_name: 0 for class_name in class_names}
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        print(f'Processing image: {image_file}')
        
        # Measure the time for each frame
        frame_start_time = time.time()
        
        predicted_viewpoint, probabilities = predict_viewpoint(model, image_path)
        ground_truth_viewpoint = get_ground_truth_viewpoint(image_file)
        
        frame_end_time = time.time()
        frame_duration = frame_end_time - frame_start_time
        
        # Check if the prediction is correct
        if predicted_viewpoint == ground_truth_viewpoint:
            correct_predictions += 1
        else:
            # Track errors per category
            category_errors[ground_truth_viewpoint] += 1
            total_errors += 1
            error_image_path = os.path.join(
                misclassified_folder, f"{image_file}_pred_{predicted_viewpoint}_true_{ground_truth_viewpoint}.png"
            )
            shutil.copy(image_path, error_image_path)
            # Log the details of incorrect predictions
            print(f"Error on frame {image_file}:")
            print(f"  Ground truth: {ground_truth_viewpoint}, Predicted: {predicted_viewpoint}")
            print(f"  Probabilities: {probabilities}")
        
        num_frames += 1
    
    # Calculate overall accuracy and FPS
    total_time = time.time() - start_time
    accuracy = (correct_predictions / num_frames) * 100 if num_frames > 0 else 0
    fps = num_frames / total_time if total_time > 0 else 0
    
    # Print accuracy and error analysis
    print(f'\n--- Results ---')
    print(f'Total time: {total_time:.2f} seconds for {num_frames} frames')
    print(f'FPS: {fps:.2f} frames per second')
    print(f'Accuracy: {accuracy:.2f}% ({correct_predictions}/{num_frames} correct)')
    
    # Print error details
    print(f'\n--- Error Analysis ---')
    for category, errors in category_errors.items():
        print(f"Category '{category}': {errors} errors")
    
    # If errors exist, calculate error percentage
    if total_errors > 0:
        for category, errors in category_errors.items():
            error_percentage = (errors / total_errors) * 100 if total_errors > 0 else 0
            print(f"  {category} error percentage: {error_percentage:.2f}%")

# Example usage
if __name__ == '__main__':
    # Path to the folder containing images
    image_folder = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/test/'  # Replace with your folder path
    
    process_images(model, image_folder)
