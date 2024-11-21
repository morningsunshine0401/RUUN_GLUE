import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn.functional as F
import time
import shutil


#from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights

# Folder to save misclassified images for inspection
misclassified_folder = 'misclassified_images'
os.makedirs(misclassified_folder, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Number of classes (update if different)
num_classes = 4  # 'front', 'back', 'left', 'right'

# Class names
class_names = ['1', '2', '3', '4']

# Ground truth mapping for image sequence (adjusted based on your provided ranges)
def get_ground_truth_viewpoint(image_name):
    image_num = int(image_name[4:8])  # Extract the frame number
    print("Image's name is:", image_num)
    if 1 <= image_num <= 45:
        return '1'
    elif 46 <= image_num <= 90:
        return '2'
    elif 91 <= image_num <= 135:
        return '3'
    else:
        return '4'

# Load the pre-trained ResNet50 model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Replace the final layer to match the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Load the trained weights
model.load_state_dict(torch.load('viewpoint_model_4view_upgrade.pth', map_location=device))

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
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    class_idx = preds.item()
    predicted_viewpoint = class_names[class_idx]
    probabilities_percent = probabilities.cpu().numpy()[0] * 100
    
    return predicted_viewpoint, probabilities_percent

# Function to process a sequence of images and save misclassified results
def process_images(model, image_folder):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
    start_time = time.time()
    num_frames = 0
    correct_predictions = 0
    total_errors = 0
    category_errors = {class_name: 0 for class_name in class_names}
    
    # Log file to record misclassified details
    with open(os.path.join(misclassified_folder, "misclassified_log.txt"), "w") as log_file:
        log_file.write("Misclassified Images Log:\n\n")
        
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
            
            print(f'Processing image: {image_file}')
            
            predicted_viewpoint, probabilities = predict_viewpoint(model, image_path)
            ground_truth_viewpoint = get_ground_truth_viewpoint(image_file)
            
            if predicted_viewpoint == ground_truth_viewpoint:
                correct_predictions += 1
            else:
                category_errors[ground_truth_viewpoint] += 1
                total_errors += 1
                
                # Save misclassified image and log details
                error_image_path = os.path.join(
                    misclassified_folder, f"{image_file}_pred_{predicted_viewpoint}_true_{ground_truth_viewpoint}.png"
                )
                shutil.copy(image_path, error_image_path)
                
                # Log misclassified details with probability scores
                log_file.write(f"Image: {image_file}\n")
                log_file.write(f"  Ground Truth: {ground_truth_viewpoint}\n")
                log_file.write(f"  Predicted: {predicted_viewpoint}\n")
                log_file.write(f"  Probabilities:\n")
                for i, prob in enumerate(probabilities):
                    log_file.write(f"    {class_names[i]}: {prob:.2f}%\n")
                log_file.write("\n")
                
                # Print a summary to console for each misclassified image
                print(f"Misclassified {image_file} - True: {ground_truth_viewpoint}, Pred: {predicted_viewpoint}")
                print(f"  Probabilities: {', '.join([f'{class_names[i]}: {prob:.2f}%' for i, prob in enumerate(probabilities)])}")
            
            num_frames += 1
    
    total_time = time.time() - start_time
    accuracy = (correct_predictions / num_frames) * 100 if num_frames > 0 else 0
    fps = num_frames / total_time if total_time > 0 else 0
    
    print(f'\n--- Results ---')
    print(f'Total time: {total_time:.2f} seconds for {num_frames} frames')
    print(f'FPS: {fps:.2f} frames per second')
    print(f'Accuracy: {accuracy:.2f}% ({correct_predictions}/{num_frames} correct)')
    
    print(f'\n--- Error Analysis ---')
    for category, errors in category_errors.items():
        print(f"Category '{category}': {errors} errors")
    
    if total_errors > 0:
        for category, errors in category_errors.items():
            error_percentage = (errors / total_errors) * 100
            print(f"  {category} error percentage: {error_percentage:.2f}%")

# Example usage
if __name__ == '__main__':
    image_folder = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/20241118_blender_test/'  # Replace with your folder path
    process_images(model, image_folder)
