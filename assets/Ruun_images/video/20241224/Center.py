import cv2

# Intrinsic parameters
cx = 628.078538
cy = 362.156441

# Video input path
video_path = "20241224_test3.mp4"  # Replace with your video file path
output_path = "output_with_dot.mp4"  # Output file path

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object for saving
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit when no more frames are available

    # Draw the dot at (cx, cy)
    dot_color = (0, 0, 255)  # Red in BGR
    dot_radius = 5  # Radius of the dot
    thickness = -1  # Fill the circle
    cv2.circle(frame, (int(cx), int(cy)), dot_radius, dot_color, thickness)

    # Write the frame with the dot
    out.write(frame)

    # Optional: Display the frame
    cv2.imshow("Video with Dot", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video with dot saved to {output_path}")
