import cv2
import numpy as np

# Provide the path to your PNG file
image_path = "Anchor2.png"

# Read the image
image = cv2.imread(image_path)
assert image is not None, f"Failed to load image at {image_path}"

# Convert BGR image to RGB for displaying with OpenCV
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Coordinates where you want to plot red dots
coordinates = [
    [511, 293],
[591, 284],
[667, 372],
[610, 269],
[587, 330],
[410, 302],
[413, 249],
[602, 348],
[621, 392],
[715, 384],
[543, 321],
[598, 298],
[590, 358],
[576, 365],
[656, 171],
[805, 213],
[703, 392],
[523, 286],
[519, 327],
[751, 136],
[610, 288],
[535, 329],
[387, 289],
[519, 303],
[727, 126],
[618, 281],
[564, 267],
[425, 243],
[554, 381],
[636, 358],
[616, 412],
[493, 319],
[591, 263],
[745, 202],
[700, 207],
[595, 388],
[436, 260],
[360, 432],
[605, 379],
[420, 306],
[539, 313],
[795, 220],
[415, 291],
[351, 291],
[665, 165],
[611, 353],
[577, 253],
[541, 343],
[585, 383],
[650, 377],
[537, 293],
[516, 389],
[509, 309],
[727, 143],
[553, 337],
[384, 404],
[553, 326],
[755, 96],
[397, 294],
[525, 343],
[496, 378],
[575, 312],
[617, 368],
[341, 271],
[428, 298],
[374, 492],
[589, 368],
[725, 155],
[430, 312],
[621, 273],
[745, 176],
[491, 293],
[772, 410],
[595, 320],
[434, 362],
[341, 462],
[693, 176],
[322, 242],
[480, 281],
[635, 286],
[834, 225],
[802, 436],
[469, 339],
[739, 156],
[373, 416],
[742, 133],
[787, 445],
[448, 351],
[576, 271],
[406, 283],
[740, 167],
[498, 296],
[711, 125],
[551, 362],
[554, 282],
[564, 254],
[355, 443],
[500, 305],
[631, 275],
[676, 213],
[705, 223],
[637, 156],
[752, 464],
[750, 112],
[737, 116],
[323, 256],
[736, 128],
[746, 154],
[620, 173],
[623, 291],
[682, 365],
[816, 414],
[357, 195],
[611, 342],
[334, 279],
[402, 452],
[351, 496],
[482, 302],
[816, 217],
[752, 77],
[748, 193],
[372, 448],
[727, 95],
[395, 229],
[337, 486],
[717, 470],
[734, 145],
[483, 307],
[470, 399],
[744, 117],
[563, 289],
[734, 89],
[332, 217],
[397, 453],
[620, 184],
[781, 245],
[813, 240],
[642, 451],
[325, 231],
[327, 269],
[645, 387],
[479, 388],
[408, 382],
[469, 282],
[482, 375],
[769, 456],
[800, 244],
[391, 397],
[648, 228],
[608, 279],
[421, 294],
[329, 222],
[626, 163],
[377, 437],
[738, 468],
[628, 290],
[788, 208],
[383, 220],
[445, 282],
[554, 348],
[808, 430],
[329, 274],
[768, 243],
[480, 383],
[569, 388],
[695, 469],
[555, 357],
[374, 210],
[720, 107],
[441, 273],
[433, 282],
[399, 390],
[440, 300]

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
output_path = "image_with_chosen_points_opti888.png"
cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
print(f"Final image with chosen points saved to {output_path}")

# Save selected points to a file
selected_points_path = "chosen_points_opti6565.txt"
with open(selected_points_path, "w") as file:
    for (x, y) in selected_points:
        file.write(f"{x}, {y}\n")
print(f"Chosen points saved to {selected_points_path}")