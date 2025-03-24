import cv2
import os

def extract_frames(video_path, output_folder, resize_to=None, skip=1):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    if not cap.isOpened():
        print("❌ Failed to open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip == 0:
            if resize_to:
                frame = cv2.resize(frame, resize_to)
            
            filename = os.path.join(output_folder, f"{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ Done! Saved {saved_count} frames to '{output_folder}'")

# === Usage ===
if __name__ == "__main__":
    video_path = "/home/runbk0401/Videos/Webcam/NW_V.webm"  # Change this!
    #output_folder = "/media/runbk0401/Storage3/RUUN_GLUE_DATABASE/ViT_Learning/train/SW/"
    output_folder = "/media/runbk0401/Storage3/RUUN_GLUE_DATABASE/ViT_Learning/val/NW/"
    extract_frames(video_path, output_folder, resize_to=(256, 256), skip=1)
