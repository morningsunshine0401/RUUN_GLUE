import cv2
import numpy as np

# Image path and keypoints coordinates
image_path = "/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/Anchor.png"  # Replace with your image path
keypoints = np.array([[690.0, 61.0], [861.0, 231.0], [732.0, 83.0], [737.0, 370.0], [935.0, 156.0], [756.0, 367.0], [596.0, 434.0], [723.0, 393.0], [444.0, 222.0], [752.0, 214.0], [1031.0, 495.0], [673.0, 329.0], [806.0, 153.0], [743.0, 343.0], [348.0, 312.0], [662.0, 103.0], [809.0, 272.0], [927.0, 217.0], [707.0, 44.0], [670.0, 500.0], [851.0, 173.0], [494.0, 605.0], [603.0, 557.0], [968.0, 150.0], [539.0, 515.0], [1069.0, 509.0], [1016.0, 639.0], [602.0, 377.0], [565.0, 366.0], [413.0, 209.0], [325.0, 298.0], [672.0, 119.0], [535.0, 617.0], [834.0, 491.0], [692.0, 435.0], [726.0, 110.0], [679.0, 477.0], [778.0, 180.0], [777.0, 461.0], [429.0, 313.0], [1047.0, 498.0], [978.0, 621.0], [736.0, 197.0], [612.0, 589.0], [714.0, 169.0], [815.0, 239.0], [712.0, 407.0], [948.0, 190.0], [346.0, 274.0], [566.0, 641.0], [512.0, 345.0], [336.0, 283.0], [632.0, 559.0], [602.0, 609.0], [697.0, 149.0], [684.0, 382.0], [829.0, 236.0], [976.0, 549.0], [959.0, 173.0], [896.0, 166.0], [785.0, 178.0], [729.0, 64.0], [513.0, 616.0], [541.0, 407.0], [393.0, 228.0], [813.0, 247.0], [738.0, 76.0], [1004.0, 637.0], [726.0, 183.0], [776.0, 332.0], [676.0, 382.0], [880.0, 227.0], [610.0, 594.0], [871.0, 170.0], [648.0, 316.0], [844.0, 234.0], [671.0, 472.0], [765.0, 289.0], [647.0, 473.0], [486.0, 263.0], [935.0, 551.0], [473.0, 330.0], [503.0, 354.0], [750.0, 88.0], [480.0, 288.0], [924.0, 159.0], [863.0, 194.0], [911.0, 183.0], [850.0, 559.0], [495.0, 312.0], [1028.0, 610.0], [771.0, 113.0], [1024.0, 623.0], [780.0, 123.0], [1034.0, 595.0], [417.0, 305.0], [804.0, 390.0], [716.0, 93.0], [528.0, 584.0]])  # Replace with your keypoints

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Ensure keypoints are integers for drawing
keypoints = keypoints.astype(int)

# List to store selected keypoints
selected_keypoints = []

# Callback function for mouse events
def select_keypoint(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Find the nearest keypoint
        distances = np.sqrt((keypoints[:, 0] - x) ** 2 + (keypoints[:, 1] - y) ** 2)
        nearest_idx = np.argmin(distances)
        nearest_point = keypoints[nearest_idx]
        
        # Store selected keypoints
        selected_keypoints.append(nearest_point)
        print(f"Selected keypoint: {nearest_point}")

        # Highlight the selected keypoint on the image
        cv2.circle(image, tuple(nearest_point), 10, (0, 255, 0), -1)
        cv2.imshow("Image with Keypoints", image)

# Draw all keypoints as small red dots
for point in keypoints:
    cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)

# Create a named window and attach the mouse callback
cv2.namedWindow("Image with Keypoints")
cv2.setMouseCallback("Image with Keypoints", select_keypoint)

# Display the image in a loop
while True:
    cv2.imshow("Image with Keypoints", image)
    key = cv2.waitKey(20) & 0xFF  # Use a slight delay to reduce CPU usage

    if key == ord('q'):  # Press 'q' to quit without saving
        print("Exiting without saving selected keypoints.")
        break

    if key == ord('e'):  # Press 'e' to save selected keypoints and exit
        print(f"Saving {len(selected_keypoints)} selected keypoints...")
        # Save the image with selected keypoints to a file
        output_path = "selected_keypoints_image.png"
        cv2.imwrite(output_path, image)
        print(f"Saved annotated image to {output_path}")
        break

cv2.destroyAllWindows()