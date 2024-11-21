import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
from torchvision.models import ResNet50_Weights  # Use ResNet50 weights

# Parameters
num_classes = 4  # Updated to match your 6 viewpoint categories
batch_size = 32
num_epochs = 25
learning_rate = 1e-4

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256)),  # Resize images
        transforms.CenterCrop(224),  # Then center-crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # Normalization for pre-trained models
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256)),  # Resize images
        transforms.CenterCrop(224),  # Then center-crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # Same normalization
                             [0.229, 0.224, 0.225])
    ]),
}

# Datasets
data_dir = '/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/learning6_data/aug2/'  # No augmented data
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# Data loaders
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                             shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# Print dataset sizes
print(f"Training dataset size: {dataset_sizes['train']}")
print(f"Validation dataset size: {dataset_sizes['val']}")

# Model setup
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features

# Replace the final layer with a layer for 6 classes
model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(f"Starting phase: {phase}")
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                print(f"Processing batch with {inputs.size(0)} samples")
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

    print('Training complete')
    torch.save(model.state_dict(), 'viewpoint_model_4view_modified.pth')

# Start training
train_model(model, criterion, optimizer, num_epochs=num_epochs)
