import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (nano version for speed)
#model = YOLO('yolov8n.pt')
model = YOLO("yolov8s.pt")
# Choose source: use 0 for webcam, or provide a path to an image
source = 0  # or e.g., 'test.jpg'

# Open video capture
cap = cv2.VideoCapture(source)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
