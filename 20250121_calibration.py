import cv2
import numpy as np
import os
import pickle

# Parameters
chessboard_size = (7,4)#(8,5)#(9, 6)  # Number of inner corners (columns, rows) in the chessboard
square_size_mm = 20#22  # Size of each square in mm (update based on your board)
frame_size = (1920, 1080)#(1280, 720)  # Resolution of the camera

# Directories
save_dir = "calibration_images"  # Directory to save captured images
os.makedirs(save_dir, exist_ok=True)

# Initialize storage
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size_mm

def capture_images():
    """Capture images from the camera when 'c' is pressed, and finish with 'f'."""
    cap = cv2.VideoCapture(0)  # Open the default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

    print("Press 'c' to capture an image and 'f' to finish capturing.")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save the captured image
            filename = os.path.join(save_dir, f"image_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Captured and saved: {filename}")
            count += 1
        elif key == ord('f'):
            print("Finished capturing images.")
            break

    cap.release()
    cv2.destroyAllWindows()

def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    """Calculate and print the mean reprojection error."""
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    print(f"Mean Reprojection Error: {mean_error}")
    return mean_error

def visualize_reprojected_points(valid_images, objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    """Visualize the reprojected points overlaid on the original calibration images."""
    for i in range(len(valid_images)):
        img = cv2.imread(valid_images[i])
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        for p in imgpoints2:
            cv2.circle(img, (int(p[0][0]), int(p[0][1])), 5, (0, 0, 255), -1)
        cv2.imshow("Reprojected Points", img)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

def undistort_sample_image(camera_matrix, dist_coeffs):
    """Undistort a sample image and display the results."""
    img = cv2.imread(os.path.join(save_dir, "image_000.jpg"))  # Change to any valid image
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Display original and undistorted images
    cv2.imshow("Original", img)
    cv2.imshow("Undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calibrate_camera():
    """Perform camera calibration using the captured images."""
    images = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(".jpg")]

    if len(images) < 10:
        print("Not enough images for calibration. Capture at least 10 images.")
        return

    # Validate images and filter only those with detected grids
    valid_images = []
    for image_path in images:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        # ret, corners = cv2.findChessboardCorners(
        #     gray, chessboard_size,
        #     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        # )
        
        
        # if ret:
        #     valid_images.append(image_path)
        #     objpoints.append(objp)
        #     imgpoints.append(corners)

        if ret:
            # Draw corners
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow("Checkerboard Detection", img)
            cv2.waitKey(300)  # Show for 300ms

            valid_images.append(image_path)
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"Checkerboard not detected in: {image_path}")
    cv2.destroyAllWindows()

    if len(valid_images) < 10:
        print("Not enough valid images for calibration. Ensure grids are properly captured.")
        return

    print("Starting calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None
    )
    if ret:
        print("Calibration successful!")
        print(f"Camera Matrix:\n{camera_matrix}")
        print(f"Distortion Coefficients:\n{dist_coeffs}")
        # Save calibration results
        with open("calibration.pkl", "wb") as f:
            pickle.dump((camera_matrix, dist_coeffs), f)
        print("Calibration data saved to 'calibration.pkl'")

        # Calculate reprojection error
        calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs)

        # Visualize reprojected points
        visualize_reprojected_points(valid_images, objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs)

        # Undistort a sample image
        undistort_sample_image(camera_matrix, dist_coeffs)
    else:
        print("Calibration failed.")

def main():
    capture_images()
    calibrate_camera()

if __name__ == "__main__":
    main()
