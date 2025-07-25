import cv2
import json
import os
from pathlib import Path

def visualize_reference_annotations(config_path):
    """
    Loads reference image configurations, visualizes keypoints and bounding boxes,
    and checks for common issues with bbox coordinates.

    Args:
        config_path (str): Path to the reference configuration JSON file.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {config_path}")
        return

    references = config.get('references', [])
    if not references:
        print("No 'references' found in the configuration file.")
        return

    print(f"Found {len(references)} reference images in the configuration.")

    for i, ref_data in enumerate(references):
        ref_id = ref_data.get('id', f'unknown_id_{i}')
        image_path_relative = ref_data.get('image_path')
        keypoints = ref_data.get('keypoints', {})
        bbox = ref_data.get('bbox')
        original_size = ref_data.get('original_size')

        if not image_path_relative:
            print(f"Warning: 'image_path' missing for reference {ref_id}. Skipping.")
            continue

        # Construct the full image path. Assuming images are relative to the config file.
        # Adjust this logic if your image paths are structured differently.
        config_dir = Path(config_path).parent
        full_image_path = config_dir / image_path_relative

        if not full_image_path.exists():
            print(f"Warning: Image not found at {full_image_path} for reference {ref_id}. Skipping.")
            continue

        try:
            image = cv2.imread(str(full_image_path))
            if image is None:
                print(f"Warning: Could not load image {full_image_path} for reference {ref_id}. Skipping.")
                continue
        except Exception as e:
            print(f"Error loading image {full_image_path} for reference {ref_id}: {e}. Skipping.")
            continue

        img_height, img_width = image.shape[0], image.shape[1]
        print(f"\nProcessing Reference {ref_id} (Image: {full_image_path.name}) - Size: {img_width}x{img_height}")

        # Create a copy for drawing
        display_image = image.copy()

        # --- Validate and Draw Bounding Box ---
        bbox_valid = True
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            # Basic validation
            if w <= 0 or h <= 0:
                print(f"  Error: Bounding box for {ref_id} has non-positive width or height: w={w}, h={h}")
                bbox_valid = False
            if x < 0 or y < 0:
                print(f"  Error: Bounding box for {ref_id} has negative origin: x={x}, y={y}")
                bbox_valid = False
            if (x + w) > img_width or (y + h) > img_height:
                print(f"  Error: Bounding box for {ref_id} extends beyond image boundaries: "
                      f"({x},{y}) to ({x+w},{y+h}) vs image ({img_width},{img_height})")
                bbox_valid = False

            if bbox_valid:
                # Draw bounding box (Green)
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_image, "BBox", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"  Bounding Box: [{x}, {y}, {w}, {h}] - Valid")
            else:
                print(f"  Bounding Box: {bbox} - Invalid (see errors above)")
        elif bbox:
            print(f"  Warning: Bounding box for {ref_id} is malformed (expected 4 values): {bbox}")
        else:
            print(f"  No bounding box specified for {ref_id}.")

        # --- Draw Keypoints ---
        keypoint_colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255),
                           (0, 165, 255), (255, 255, 0), (128, 0, 128), (0, 128, 0)] # BGR colors
        
        visible_keypoints_count = 0
        for kp_idx, (kp_name, coords) in enumerate(keypoints.items()):
            if coords and len(coords) == 2 and (coords[0] != 0 or coords[1] != 0):
                x, y = int(coords[0]), int(coords[1])
                color = keypoint_colors[kp_idx % len(keypoint_colors)]
                
                # Check if keypoint is within image bounds
                if 0 <= x < img_width and 0 <= y < img_height:
                    cv2.circle(display_image, (x, y), 5, color, -1) # Draw filled circle
                    cv2.putText(display_image, kp_name, (x + 8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    visible_keypoints_count += 1
                else:
                    print(f"  Warning: Keypoint {kp_name} ({x},{y}) for {ref_id} is outside image boundaries ({img_width}x{img_height}).")
            else:
                print(f"  Keypoint {kp_name} for {ref_id} is marked as [0,0] or malformed.")
        print(f"  Total visible keypoints drawn: {visible_keypoints_count}")

        # --- Display Image ---
        window_name = f"Reference: {ref_id} - {full_image_path.name}"
        cv2.imshow(window_name, display_image)
        print("  Press any key to view the next image, or 'q' to quit.")
        
        # Wait for a key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\nVisualization complete.")

if __name__ == "__main__":
    # IMPORTANT: Replace 'reference_config_matcha2.json' with the actual path to your config file.
    # Ensure the image_path values in your JSON are correct relative to the config file's location,
    # or adjust the 'full_image_path' construction logic above.
    config_file_path = 'reference_config_matcha.json' 
    visualize_reference_annotations(config_file_path)
