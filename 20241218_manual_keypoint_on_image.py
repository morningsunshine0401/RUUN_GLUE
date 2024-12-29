import cv2
import numpy as np

# Provide the path to your PNG file
image_path = "Anchor_side.png"

# Read the image
image = cv2.imread(image_path)
assert image is not None, f"Failed to load image at {image_path}"

# Convert BGR image to RGB for displaying with OpenCV
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Coordinates where you want to plot red dots
coordinates = [
    [963, 266], [951, 249], [551, 386], [520, 375], [651, 422], [930, 263], [969, 243], [531, 378],
    [954, 233], [502, 330], [950, 263], [537, 396], [543, 376], [605, 386], [828, 338], [801, 332],
    [676, 338], [933, 238], [475, 379], [625, 401], [438, 380], [442, 414], [531, 330], [221, 381],
    [956, 298], [993, 202], [503, 404], [593, 375], [584, 369], [576, 376], [777, 430], [562, 386],
    [597, 398], [491, 331], [636, 340], [572, 482], [864, 335], [592, 360], [488, 445], [639, 363],
    [970, 306], [235, 446], [500, 343], [137, 419], [531, 385], [928, 317], [967, 282], [505, 385],
    [493, 438], [691, 366], [699, 352], [968, 359], [472, 429], [480, 313], [865, 360], [756, 457],
    [933, 252], [532, 356], [975, 233], [825, 374], [152, 390], [657, 405], [498, 304], [766, 439],
    [512, 326], [423, 377], [1059, 384], [505, 351], [951, 314], [489, 306], [986, 234], [492, 275],
    [267, 447], [776, 373], [926, 281], [145, 400], [394, 383], [162, 435], [965, 329], [707, 446],
    [561, 375], [412, 406], [944, 274], [490, 358], [843, 373], [970, 297], [570, 386], [780, 362],
    [827, 324], [186, 364], [959, 388], [626, 481], [971, 389], [672, 352], [968, 320], [409, 315],
    [999, 390], [577, 325], [704, 469], [489, 390], [973, 284], [988, 187], [988, 221], [857, 386],
    [847, 360], [480, 273], [1018, 372], [664, 362], [909, 222], [435, 284], [679, 368], [432, 386],
    [754, 433], [777, 443], [965, 340], [403, 351], [375, 379], [596, 326], [694, 471], [635, 333],
    [221, 445], [221, 391], [437, 368], [989, 216], [978, 263], [951, 190], [417, 396], [185, 440],
    [240, 397], [399, 409], [393, 313], [711, 331], [832, 323], [775, 438], [768, 392], [625, 420],
    [636, 479], [527, 344], [508, 313], [228, 394], [392, 368], [234, 396], [495, 294], [999, 368],
    [662, 442], [631, 428], [380, 377], [403, 312], [511, 464], [494, 387], [369, 317], [555, 479],
    [400, 359], [813, 326], [949, 274], [532, 320]
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