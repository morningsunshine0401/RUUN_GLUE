import cv2
import numpy as np

image_path = "/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/viewpoint/anchor/Anchor.png"  # Replace with your image path

# Selected keypoints (replace with your selected keypoints)
selected_keypoints = [
    [494, 605], [566, 641], [603, 557], [539, 515], [512, 345],
    [834, 491], [927, 217], [707, 44], [752, 214], [851, 173],
    [1069, 509], [1016, 639], [413, 209], [325, 298], [743, 343],
    [541, 407], [676, 382]
]

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert keypoints to integers (just in case)
selected_keypoints = np.array(selected_keypoints, dtype=int)

# Draw selected keypoints and label them
for idx, point in enumerate(selected_keypoints):
    cv2.circle(image, tuple(point), 8, (0, 255, 255), -1)  # Draw as yellow circles
    label = f"{tuple(point)}"  # Format the label as the coordinate
    cv2.putText(image, label, (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)  # Add text label

# Display the image with selected keypoints
cv2.imshow("Selected Keypoints", image)

# Wait until a key is pressed
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()

# Optionally save the image with selected keypoints and labels
output_path = "annotated_selected_keypoints_with_labels.png"
cv2.imwrite(output_path, image)
print(f"Annotated image with labels saved to {output_path}")
