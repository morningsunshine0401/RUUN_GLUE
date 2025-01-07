import cv2
import numpy as np

# Provide the path to your PNG file
image_path = "Anchor_B.png"

# Read the image
image = cv2.imread(image_path)
assert image is not None, f"Failed to load image at {image_path}"

# Convert BGR image to RGB for displaying with OpenCV
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Coordinates where you want to plot red dots
coordinates = [
    [650, 312], #
[645, 329], #
[630, 306], #
[523, 352], #
[907, 443], #
[586, 344], #
[577, 338], #
[814, 291], #
[599, 349], #
[501, 386], #
[965, 359], #
[649, 355], #
[635, 346], #
[930, 335], #
[843, 467], #
[702, 339], #
[718, 321], #
[930, 322], #
[548, 347], #
[727, 346], #
[539, 364], #
[786, 297], #
[1022, 406], #
[949, 352], #
[952, 322], #
[1004, 399], #
[970, 337], #
[724, 371], #
[539, 344], #
[536, 309], #
[923, 449], #
[864, 478], #
[980, 429], #
[745, 310], #
[1049, 393], #
[895, 258], #
[674, 347], #
[374, 396], #
[741, 281], #
[699, 294], #
[817, 494], #
[992, 281], #
]


# Create a list to store selected points
selected_points = []

# Iterate through each point
for i, point in enumerate(coordinates):
    x, y = point
    temp_image = image_rgb.copy()

    # Draw all selected points so far
    for (sx, sy) in selected_points:
        cv2.circle(temp_image, (sx, sy), radius=5, color=(255, 0, 0), thickness=-1)

    # Highlight the current point being evaluated
    cv2.circle(temp_image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    cv2.putText(temp_image, f"Point: ({x}, {y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Evaluate Points", cv2.cvtColor(temp_image, cv2.COLOR_RGB2BGR))

    # Wait for a key press
    key = cv2.waitKey(0) & 0xFF

    if key == ord('n'):  # Press 'n' to choose the point
        selected_points.append((x, y))
    elif key == ord('k'):  # Press 'k' to skip the point
        continue
    elif key == ord('h'):  # Press 'h' to stop early
        break
    elif key == ord('b'):  # Press 'b' to undo the last choice
        if selected_points:
            selected_points.pop()
            print("Last selection undone.")
            i -= 1  # Go back to the previous point
        else:
            print("No selections to undo.")

cv2.destroyAllWindows()

# Create a new image with only selected points
final_image = image_rgb.copy()
for (x, y) in selected_points:
    cv2.circle(final_image, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
    cv2.putText(final_image, f"({x}, {y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Display the final image
cv2.imshow("Final Image with Selected Points", cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final image
output_path = "image_with_chosen_points_opti8s88.png"
cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
print(f"Final image with chosen points saved to {output_path}")

# Save selected points to a file
selected_points_path = "chosen_points_opti65s65.txt"
with open(selected_points_path, "w") as file:
    for (x, y) in selected_points:
        file.write(f"{x}, {y}\n")
print(f"Chosen points saved to {selected_points_path}")