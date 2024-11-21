import cv2

# Set video capture device (0 for the default webcam)
cap = cv2.VideoCapture(0)

# Set the frame size to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'MJPG', 'X264'
out = cv2.VideoWriter('20241002_output_rotation_better.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Write the frame to the video file
        out.write(frame)

        # Display the live video feed
        cv2.imshow('Webcam Video', frame)

        # Press 'q' to exit the video capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the capture and writer objects
cap.release()
out.release()

# Close any OpenCV windows
cv2.destroyAllWindows()
