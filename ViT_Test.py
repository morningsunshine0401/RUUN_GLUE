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

print(model)

# === Hook for Feature Map from MobileViT block ===
feature_maps = []

def get_feature_map(module, input, output):
    feature_maps.append(output)

# Register hook on the last MobileViT block's conv_proj
model.stages[4][1].conv_proj.register_forward_hook(get_feature_map)

# === Preprocessing ===
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
skip_every_n_frames = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % skip_every_n_frames != 0:
        continue

    # Resize input to 256x256 for model
    frame_resized = cv2.resize(frame, (256, 256))
    img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    feature_maps.clear()  # Clear previous feature maps

    # === Prediction ===
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
        label = class_names[pred]
        confidence = probs[pred].item()

    # === Process Feature Map for ROI ===
    if feature_maps:
        feat = feature_maps[-1]  # shape: (1, C, H, W)
        feat_mean = feat.mean(dim=1).squeeze().cpu().numpy()  # (H, W)
        feat_norm = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-8)
        feat_resized = cv2.resize(feat_norm, (256, 256))
        feat_resized = (feat_resized * 255).astype(np.uint8)
        
        cv2.imshow("Feature Map RAW?", feat_norm)
        cv2.imshow("Feature Map", feat_resized)


        # Threshold and extract ROI
        _, binary_map = cv2.threshold(feat_resized, int(255 * 0.1), 255, cv2.THRESH_BINARY)
       # _, binary_map = cv2.threshold(feat_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # === Overlay Prediction ===
    label_text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame_resized, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # === FPS ===
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame_resized, f"FPS: {fps:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # === Show Output ===
    cv2.imshow("Aircraft Viewpoint & ROI", frame_resized)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

###################################################################################################

# import cv2
# import timm
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# import time
# import numpy as np

# # === Setup ===
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_classes = 4

# class_names = ['NE', 'NW', 'SE', 'SW']

# # === Load Model ===
# model = timm.create_model('mobilevit_s', pretrained=False, num_classes=num_classes)
# model.load_state_dict(torch.load('mobilevit_viewpoint_twostage_final_2.pth', map_location=device))
# model.to(device)
# model.eval()

# # === Preprocessing (no resize here, done via OpenCV for speed) ===
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
# ])

# # === Webcam Setup ===
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# if not cap.isOpened():
#     print("❌ Cannot open webcam")
#     exit()

# print("🎥 Webcam started. Press 'q' to quit.")

# # === FPS Tracking ===
# prev_time = time.time()
# frame_counter = 0
# skip_every_n_frames = 2  # Skip every other frame to reduce load

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_counter += 1
#     if frame_counter % skip_every_n_frames != 0:
#         continue

#     # Resize frame before feeding into model
#     frame_resized = cv2.resize(frame, (256, 256))
#     img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
#     input_tensor = transform(img_pil).unsqueeze(0).to(device)

#     # Prediction
#     with torch.no_grad():
#         output = model(input_tensor)
#         probs = F.softmax(output, dim=1)[0]
#         pred = torch.argmax(probs).item()
#         label = class_names[pred]
#         confidence = probs[pred].item()

#     # Free GPU memory after each step
#     del input_tensor, output
#     torch.cuda.empty_cache()

#     # Calculate FPS
#     curr_time = time.time()
#     fps = 1.0 / (curr_time - prev_time)
#     prev_time = curr_time

#     # === Overlay Prediction ===
#     label_text = f"{label} ({confidence*100:.1f}%)"
#     cv2.putText(frame, label_text, (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # === FPS Display ===
#     cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

#     # === Confidence Bar ===
#     bar_x = 10
#     bar_y = 100
#     bar_width = 200
#     bar_height = 20
#     conf_bar = int(bar_width * confidence)
#     cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 2)
#     cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_bar, bar_y + bar_height), (0, 255, 0), -1)

#     # === Show Frame ===
#     cv2.imshow("Aircraft Viewpoint Estimation", frame)

#     # Add small delay to reduce CPU/GPU strain
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
