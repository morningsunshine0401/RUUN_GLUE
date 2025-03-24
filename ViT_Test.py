import cv2
import timm
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time
import numpy as np

# === Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 4

class_names = ['NE', 'NW', 'SE', 'SW']

# === Load Model ===
model = timm.create_model('mobilevit_s', pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load('mobilevit_viewpoint_twostage_final_2.pth', map_location=device))
model.to(device)
model.eval()

# === Preprocessing (no resize here, done via OpenCV for speed) ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🎥 Webcam started. Press 'q' to quit.")

# === FPS Tracking ===
prev_time = time.time()
frame_counter = 0
skip_every_n_frames = 2  # Skip every other frame to reduce load

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % skip_every_n_frames != 0:
        continue

    # Resize frame before feeding into model
    frame_resized = cv2.resize(frame, (256, 256))
    img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
        label = class_names[pred]
        confidence = probs[pred].item()

    # Free GPU memory after each step
    del input_tensor, output
    torch.cuda.empty_cache()

    # Calculate FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # === Overlay Prediction ===
    label_text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # === FPS Display ===
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # === Confidence Bar ===
    bar_x = 10
    bar_y = 100
    bar_width = 200
    bar_height = 20
    conf_bar = int(bar_width * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 2)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_bar, bar_y + bar_height), (0, 255, 0), -1)

    # === Show Frame ===
    cv2.imshow("Aircraft Viewpoint Estimation", frame)

    # Add small delay to reduce CPU/GPU strain
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
