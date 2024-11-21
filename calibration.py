import numpy as np
import cv2 as cv
import glob
import pickle

# Define the dot grid size and termination criteria
dotGridSize = (6, 6)
frameSize = (1280, 720)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((dotGridSize[0] * dotGridSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:dotGridSize[0], 0:dotGridSize[1]].T.reshape(-1, 2)
size_of_dot_spacing_mm = 20
objp = objp * size_of_dot_spacing_mm

# Initialize storage
objpoints = []
imgpoints = []

# Load images
images = glob.glob('assets/Calibration/*.jpg')

for image_path in images:
    print(f"Processing: {image_path}")
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Preprocess the image
    #gray = cv.GaussianBlur(gray, (5, 5), 0)
    #gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # Visualize the preprocessed image
    cv.imshow("Preprocessed Image", gray)
    cv.waitKey(500)

    # Detect dot grid
    ret, centers = cv.findCirclesGrid(gray, dotGridSize, flags=cv.CALIB_CB_SYMMETRIC_GRID)
    if ret:
        print(f"Dot grid detected in {image_path}")
        objpoints.append(objp)
        imgpoints.append(centers)

        # Draw and display the detected grid
        cv.drawChessboardCorners(img, dotGridSize, centers, ret)
        cv.imshow('Dot Grid', img)
        cv.waitKey(500)
    else:
        print(f"Dot grid not detected in {image_path}")

cv.destroyAllWindows()

# Perform calibration
if objpoints and imgpoints:
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    if ret:
        print("Calibration successful!")
        pickle.dump((cameraMatrix, dist), open("calibration.pkl", "wb"))
else:
    print("Calibration failed: No valid grids detected.")


# import cv2 as cv
# import numpy as np

# # Define the dot grid size
# dotGridSize = (6, 6)  # Number of inner grid points

# # Open the default camera
# cap = cv.VideoCapture(0)  # Change to the correct camera index if necessary

# if not cap.isOpened():
#     print("Error: Cannot access the camera.")
#     exit()

# print("Press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame.")
#         break

#     # Convert to grayscale
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # Try to detect the dot grid
#     ret, centers = cv.findCirclesGrid(
#         gray, dotGridSize, flags=cv.CALIB_CB_SYMMETRIC_GRID
#     )

#     # If grid is detected, draw it on the frame
#     if ret:
#         print("Dot grid detected!")
#         cv.drawChessboardCorners(frame, dotGridSize, centers, ret)
#     else:
#         print("No dot grid detected.")

#     # Display the frame
#     cv.imshow("Real-Time Dot Grid Detection", frame)

#     # Quit the loop if 'q' is pressed
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close all OpenCV windows
# cap.release()
# cv.destroyAllWindows()
