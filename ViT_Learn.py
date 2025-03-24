import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
import torchvision.transforms.functional as TF
import random
import os

# === Custom Gaussian Noise ===
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# === Data Transforms ===
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomAffine(
        degrees=5,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
        shear=0.0
    ),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ], p=0.8),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
    ], p=0.3),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.01),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === Load Dataset ===
train_dataset = datasets.ImageFolder('/media/runbk0401/Storage5/RUUN_GLUE_DATABASE/ViT_Learning/train', transform=transform_train)
val_dataset = datasets.ImageFolder('/media/runbk0401/Storage5/RUUN_GLUE_DATABASE/ViT_Learning/val', transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers =2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers =2)

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('mobilevit_s', pretrained=True, num_classes=4)
model.to(device)

criterion = nn.CrossEntropyLoss()

# === Stage 1: Freeze all but head ===
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.head.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print("\n🔁 Stage 1: Training classifier head only\n")
for epoch in range(15):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f"Stage 1 - Epoch [{epoch+1}/15], Loss: {total_loss:.4f}")

# === Stage 2: Unfreeze all and fine-tune ===
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=5e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

best_accuracy = 0.0

print("\n🔁 Stage 2: Fine-tuning entire model\n")
for epoch in range(15):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f"Stage 2 - Epoch [{epoch+1}/15], Loss: {total_loss:.4f}")

    # === Validation for best model saving ===
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy after Epoch {epoch+1}: {accuracy:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_mobilevit_viewpoint.pth')
        print(f"📌 Best model saved (Val Acc: {accuracy:.2f}%)")

# === Final Evaluation ===
print("\n📈 Final Evaluation on Validation Set:")
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f'\n✅ Final Validation Accuracy: {accuracy:.2f}%')

print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

torch.save(model.state_dict(), 'mobilevit_viewpoint_twostage_final_2.pth')
print("✅ Final model saved as 'mobilevit_viewpoint_twostage_final_2.pth'")
