import json
import argparse

def fix_annotations(input_path, output_path, padding_factor=0.1):
    """
    Corrects bounding boxes in a COCO-style annotation file to ensure they
    contain all associated keypoints.
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    corrected_count = 0

    for ann in annotations:
        keypoints = ann.get('keypoints', [])
        bbox = ann.get('bbox', [])

        if not keypoints or not bbox:
            continue

        # Extract keypoint coordinates
        x_coords = keypoints[0::3]
        y_coords = keypoints[1::3]
        vis = keypoints[2::3]

        # Filter for visible keypoints
        visible_x = [x for x, v in zip(x_coords, vis) if v > 0]
        visible_y = [y for y, v in zip(y_coords, vis) if v > 0]

        if not visible_x or not visible_y:
            continue

        # Find the bounding box of the keypoints
        min_kx = min(visible_x)
        max_kx = max(visible_x)
        min_ky = min(visible_y)
        max_ky = max(visible_y)

        # Original bbox
        x, y, w, h = bbox
        orig_x1, orig_y1 = x, y
        orig_x2, orig_y2 = x + w, y + h

        # Check if correction is needed
        if (min_kx >= orig_x1 and max_kx <= orig_x2 and
                min_ky >= orig_y1 and max_ky <= orig_y2):
            continue

        # Create a new bounding box that encloses both
        new_x1 = min(orig_x1, min_kx)
        new_y1 = min(orig_y1, min_ky)
        new_x2 = max(orig_x2, max_kx)
        new_y2 = max(orig_y2, max_ky)

        # Add padding
        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1
        padding_w = new_w * padding_factor
        padding_h = new_h * padding_factor

        padded_x1 = new_x1 - padding_w / 2
        padded_y1 = new_y1 - padding_h / 2
        padded_x2 = new_x2 + padding_w / 2
        padded_y2 = new_y2 + padding_h / 2
        
        # Ensure the padded box does not go out of image bounds (assuming standard image sizes)
        # A more robust implementation would get image sizes from the 'images' section
        final_x1 = max(0, padded_x1)
        final_y1 = max(0, padded_y1)

        final_w = (padded_x2 - final_x1)
        final_h = (padded_y2 - final_y1)

        # Update the annotation
        ann['bbox'] = [final_x1, final_y1, final_w, final_h]
        ann['area'] = final_w * final_h
        corrected_count += 1

    print("Processed {} annotations.".format(len(annotations)))
    print("Corrected {} bounding boxes.".format(corrected_count))

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print("Saved corrected annotations to {}".format(output_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix COCO bounding boxes to include all keypoints.')
    parser.add_argument('--input', required=True, help='Path to the input COCO annotation file.')
    parser.add_argument('--output', required=True, help='Path to save the corrected annotation file.')
    parser.add_argument('--padding', type=float, default=0.1, help='Padding factor to add around the new bounding box.')
    args = parser.parse_args()

    fix_annotations(args.input, args.output, args.padding)