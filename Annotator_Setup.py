import cv2
import numpy as np
import torch
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_grad_enabled(False)

try:
    from lightglue import SuperPoint
    print("✅ SuperPoint imported successfully")
except ImportError as e:
    print(f"❌ Error importing SuperPoint: {e}")
    print("Please install: pip install lightglue")
    exit(1)


class FixedInteractiveReferenceSetup:
    def __init__(self, device='auto'):
        """
        Initialize interactive reference setup system
        
        Args:
            device: 'auto', 'cuda', or 'cpu'
        """
        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"🚀 Using device: {self.device}")
        
        # Initialize SuperPoint
        print("🔄 Loading SuperPoint...")
        self.extractor = SuperPoint(max_num_keypoints=9192).eval().to(self.device)
        print("✅ SuperPoint loaded!")
        
        self.window_name = "Reference Image Setup - Press 'Q' to quit"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.reference_folder = None
        self.reference_paths = []
        self.current_ref_index = 0
        self.current_image = None
        self.original_width = 0
        self.original_height = 0
        self.scale_factor = 1.0
        self.superpoint_keypoints = None
        self.selected_keypoints = []
        self.current_point_index = 0
        self.num_keypoints = 0
        self.reference_configs = []

        # Bounding box specific variables
        self.bbox_start_point = None
        self.bbox_end_point = None
        self.current_bbox = [] # [x_min, y_min, x_max, y_max]
        self.bbox_mode = False # New mode for bounding box selection


    def mouse_callback(self, event, x, y, flags, param):
        if self.current_image is None:
            return

        if self.bbox_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.bbox_start_point = (x, y)
                self.bbox_end_point = None
                print(f"BBox: Start point selected at ({x}, {y})")
            elif event == cv2.EVENT_LBUTTONUP:
                self.bbox_end_point = (x, y)
                if self.bbox_start_point:
                    x_min = min(self.bbox_start_point[0], self.bbox_end_point[0])
                    y_min = min(self.bbox_start_point[1], self.bbox_end_point[1])
                    x_max = max(self.bbox_start_point[0], self.bbox_end_point[0])
                    y_max = max(self.bbox_start_point[1], self.bbox_end_point[1])
                    self.current_bbox = [x_min, y_min, x_max, y_max]
                    print(f"BBox: Selected [{x_min}, {y_min}, {x_max}, {y_max}]")
                    self.draw_current_state()
                    print("Press 'C' to confirm BBox, 'R' to reset BBox, or 'N' to skip BBox for this image.")
            elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
                if self.bbox_start_point:
                    self.bbox_end_point = (x, y)
                    self.draw_current_state() # Redraw dynamically as user drags
        elif event == cv2.EVENT_LBUTTONDOWN and self.current_point_index < self.num_keypoints:
            # Check if clicked on a SuperPoint keypoint
            if self.superpoint_keypoints is not None:
                # Scale mouse click to original image coordinates for comparison
                click_x_original = x / self.scale_factor
                click_y_original = y / self.scale_factor

                # Find the closest SuperPoint keypoint to the click
                distances = np.linalg.norm(self.superpoint_keypoints - np.array([click_x_original, click_y_original]), axis=1)
                closest_idx = np.argmin(distances)
                
                # If the closest keypoint is within a certain threshold (e.g., 5 pixels scaled)
                if distances[closest_idx] < (5 / self.scale_factor): # Adjust threshold based on display scale
                    kp_original_coords = self.superpoint_keypoints[closest_idx]
                    
                    # Check if this keypoint has already been selected
                    already_selected = False
                    for existing_kp_idx, _ in self.selected_keypoints:
                        if np.array_equal(self.superpoint_keypoints[existing_kp_idx], kp_original_coords):
                            already_selected = True
                            print(f"Keypoint already selected. Choose another one.")
                            break
                    
                    if not already_selected:
                        self.selected_keypoints.append((closest_idx, kp_original_coords))
                        print(f"Keypoint {self.current_point_index + 1} selected at: {kp_original_coords} (Original)")
                        self.current_point_index += 1
                        self.draw_current_state()
                        if self.current_point_index == self.num_keypoints:
                            print(f"All {self.num_keypoints} keypoints selected. Press 'B' to define Bounding Box or 'F' to finish.")
                else:
                    print(f"No SuperPoint keypoint found close enough at ({x},{y}). Please click closer to a red dot.")


    def draw_current_state(self, temp_bbox=None): # Added temp_bbox for dynamic drawing
        if self.current_image is None:
            return

        display_image = self.current_image.copy()

        # Draw all SuperPoint keypoints in red
        if self.superpoint_keypoints is not None:
            for kp_original in self.superpoint_keypoints:
                kp_display = (int(kp_original[0] * self.scale_factor), int(kp_original[1] * self.scale_factor))
                cv2.circle(display_image, kp_display, 1, (0, 0, 255), -1) # Red dots for all detected KPs

        # Draw selected keypoints in numbered colors
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
                  (255, 0, 255), (192, 192, 192), (128, 0, 0), (128, 128, 0), (0, 128, 0)] # More distinct colors
        for i, (kp_idx, kp_original_coords) in enumerate(self.selected_keypoints):
            kp_display = (int(kp_original_coords[0] * self.scale_factor), int(kp_original_coords[1] * self.scale_factor))
            color = colors[i % len(colors)]
            cv2.circle(display_image, kp_display, 7, color, -1)
            cv2.putText(display_image, str(i + 1), (kp_display[0] + 10, kp_display[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if self.bbox_mode:
            instruction_text = "Draw Bounding Box: Click & Drag. Press 'C' to confirm, 'R' to reset, 'N' to skip."
            cv2.putText(display_image, instruction_text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(display_image, instruction_text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Green for BBox mode

            # Draw the bounding box currently being drawn
            if self.bbox_start_point and self.bbox_end_point:
                x1, y1 = self.bbox_start_point
                x2, y2 = self.bbox_end_point
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green rectangle
            elif self.bbox_start_point and not self.bbox_end_point: # Drawing in progress
                cv2.circle(display_image, self.bbox_start_point, 5, (0, 0, 255), -1) # Red start point
            
            # Draw the confirmed bounding box (if any)
            if self.current_bbox: # Display the current confirmed bounding box
                x_min_disp, y_min_disp, x_max_disp, y_max_disp = self.current_bbox
                cv2.rectangle(display_image, (x_min_disp, y_min_disp), (x_max_disp, y_max_disp), (0, 255, 0), 2)
                
        else: # Keypoint mode instructions
            instruction_text = f"Select Keypoint {self.current_point_index + 1}/{self.num_keypoints} (Click on red dot)"
            if self.current_point_index >= self.num_keypoints:
                instruction_text = f"All {self.num_keypoints} keypoints selected. Press 'B' for BBox, 'F' to finish."
            cv2.putText(display_image, instruction_text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(display_image, instruction_text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # Red for keypoint mode

        # Display current reference info
        cv2.putText(display_image, f"Image: {self.reference_paths[self.current_ref_index].name} ({self.current_ref_index + 1}/{len(self.reference_paths)})",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_image, f"Original Size: {self.original_width}x{self.original_height}, Display Scale: {self.scale_factor:.2f}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Controls info
        controls_text = "Controls: 'S'=Skip Keypoint, 'R'=Reset Current, 'B'=BBox Mode, 'F'=Finish Image, 'N'=Next Image, 'Q'=Quit"
        cv2.putText(display_image, controls_text, (20, display_image.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(self.window_name, display_image)


    def _extract_superpoint_keypoints(self, image):
        # Convert image to grayscale and normalize
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray_image.shape
        # Ensure image is float32 and in range [0, 1]
        image_tensor = torch.from_numpy(gray_image / 255.0).float()[None, None].to(self.device)

        # Extract features
        pred = self.extractor({'image': image_tensor})
        # keypoints are in 'image' coordinates, need to be detached from GPU
        keypoints = pred['keypoints'][0].detach().cpu().numpy()
        return keypoints


    def load_next_reference(self):
        if self.current_ref_index >= len(self.reference_paths):
            print("All reference images processed.")
            return False

        image_path = self.reference_paths[self.current_ref_index]
        print(f"\nLoading reference image: {image_path.name}")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path.name}")
            self.current_ref_index += 1
            return self.load_next_reference() # Try next image

        self.original_height, self.original_width = image.shape[:2]
        
        # Determine display scale
        max_dim = 1000 # Max dimension for display
        if max(self.original_height, self.original_width) > max_dim:
            self.scale_factor = max_dim / max(self.original_height, self.original_width)
            display_width = int(self.original_width * self.scale_factor)
            display_height = int(self.original_height * self.scale_factor)
            self.current_image = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
        else:
            self.scale_factor = 1.0
            self.current_image = image.copy()

        # Extract SuperPoint keypoints from the ORIGINAL image for accurate coordinates
        self.superpoint_keypoints = self._extract_superpoint_keypoints(image)
        print(f"Found {len(self.superpoint_keypoints)} SuperPoint keypoints.")

        self.selected_keypoints = []
        self.current_point_index = 0
        self.bbox_start_point = None
        self.bbox_end_point = None
        self.current_bbox = []
        self.bbox_mode = False # Start in keypoint mode
        return True


    def skip_current_keypoint(self):
        if self.current_point_index < self.num_keypoints:
            self.selected_keypoints.append((-1, [0, 0])) # Use -1 index and [0,0] coordinates to signify skipped
            print(f"Keypoint {self.current_point_index + 1} skipped.")
            self.current_point_index += 1
            self.draw_current_state()
            if self.current_point_index == self.num_keypoints:
                print(f"All {self.num_keypoints} keypoints selected. Press 'B' to define Bounding Box or 'F' to finish.")
        else:
            print("All keypoints already selected. Cannot skip.")

    def reset_current_keypoints(self):
        self.selected_keypoints = []
        self.current_point_index = 0
        print("All keypoints reset for current image.")
        self.draw_current_state()

    def reset_current_bbox(self):
        self.bbox_start_point = None
        self.bbox_end_point = None
        self.current_bbox = []
        print("Bounding Box reset.")
        self.draw_current_state()

    def finish_current_reference(self):
        if self.current_point_index < self.num_keypoints:
            print(f"Please select all {self.num_keypoints} keypoints (or skip them) before finishing.")
            return

        if self.bbox_mode:
            print("Please confirm or skip the bounding box first (press 'C' or 'N').")
            return

        print(f"\nProcessing reference image {self.current_ref_index + 1}/{len(self.reference_paths)}...")

        # Prepare keypoints dictionary
        keypoints_dict = {}
        for i in range(self.num_keypoints):
            if i < len(self.selected_keypoints):
                _ , kp_original_coords = self.selected_keypoints[i]
                keypoints_dict[f'keypoint_{i+1}'] = [float(kp_original_coords[0]), float(kp_original_coords[1])]
            else:
                keypoints_dict[f'keypoint_{i+1}'] = [0, 0] # Should not happen if current_point_index check works

        # Convert bounding box coordinates back to original image size
        original_bbox = []
        if self.current_bbox:
            x_min_disp, y_min_disp, x_max_disp, y_max_disp = self.current_bbox
            
            # Convert display coordinates back to original image coordinates
            original_x_min = x_min_disp / self.scale_factor
            original_y_min = y_min_disp / self.scale_factor
            original_x_max = x_max_disp / self.scale_factor
            original_y_max = y_max_disp / self.scale_factor

            # Calculate width and height
            original_width_bbox = original_x_max - original_x_min
            original_height_bbox = original_y_max - original_y_min

            # Store as [x_min, y_min, width, height]
            original_bbox = [float(original_x_min), float(original_y_min), float(original_width_bbox), float(original_height_bbox)]
            print(f"📦 Bounding box (original size): {original_bbox}")

        # Create reference config
        ref_config = {
            'id': f'reference_{self.current_ref_index}',
            'image_path': str(self.reference_paths[self.current_ref_index]),
            'viewpoint': f'viewpoint_{self.current_ref_index}',
            'description': f'Reference image {self.current_ref_index + 1}',
            'keypoints': keypoints_dict,
            'bbox': original_bbox, 
            'original_size': [self.original_width, self.original_height],
            'display_size': [self.current_image.shape[1], self.current_image.shape[0]],
            'scale_factor': self.scale_factor
        }
        self.reference_configs.append(ref_config)
        print(f"✅ Config saved for {self.reference_paths[self.current_ref_index].name}")

        self.current_ref_index += 1
        # Reset bounding box for next image
        self.bbox_start_point = None
        self.bbox_end_point = None
        self.current_bbox = []
        self.bbox_mode = False # Ensure we start with keypoint mode for next image
        self.load_next_reference() # Load the next image automatically


    def save_config(self, output_path="reference_config.json"):
        final_config = {
            "created_at": datetime.now().isoformat(),
            "num_keypoints_per_object": self.num_keypoints,
            "references": self.reference_configs
        }
        with open(output_path, 'w') as f:
            json.dump(final_config, f, indent=4)
        print(f"\n✨ Final configuration saved to: {output_path}")
        return output_path


    def handle_key_press(self, key):
        if key == ord('s') or key == ord('S'): # Skip keypoint
            if not self.bbox_mode:
                self.skip_current_keypoint()
            else:
                print("Cannot skip keypoint in BBox mode. Press 'R' to reset BBox or 'N' to skip BBox.")
        elif key == ord('r') or key == ord('R'): # Reset current
            if self.bbox_mode:
                self.reset_current_bbox()
            else:
                self.reset_current_keypoints()
        elif key == ord('q') or key == ord('Q'): # Quit
            return False 
        elif key == ord('f') or key == ord('F'): # Finish current reference (keypoints + bbox)
            if self.current_point_index < self.num_keypoints:
                print(f"Please select all {self.num_keypoints} keypoints (or skip them) before finishing. Or press 'B' to define BBox if keypoints are done.")
            elif self.bbox_mode:
                 print("Please confirm or skip the bounding box first (press 'C' or 'N').")
            else:
                self.finish_current_reference()
        elif key == ord('n') or key == ord('N'): # Next image (skip all remaining keypoints/bbox and move to next)
            if self.current_point_index < self.num_keypoints:
                print(f"Skipping remaining keypoints and moving to next image for {self.reference_paths[self.current_ref_index].name}...")
                while self.current_point_index < self.num_keypoints:
                    self.skip_current_keypoint()
            if self.bbox_mode:
                print(f"Skipping bounding box and moving to next image for {self.reference_paths[self.current_ref_index].name}...")
                self.current_bbox = [] # Mark as no bbox
                self.bbox_mode = False # Exit BBox mode
            
            # If keypoints were completed and bbox was skipped/completed, now finish the reference
            if self.current_point_index == self.num_keypoints and not self.bbox_mode:
                self.finish_current_reference()
            else: # If N was pressed prematurely
                print("Action 'N' performed. Moving to next reference image if available.")
                self.current_ref_index += 1
                self.load_next_reference()


        elif key == ord('b') or key == ord('B'): # New: Switch to BBox mode
            if self.current_point_index >= self.num_keypoints: # Only allow BBox mode after keypoints
                self.bbox_mode = True
                self.bbox_start_point = None
                self.bbox_end_point = None
                self.current_bbox = [] # Reset for new bbox drawing
                print("Switched to Bounding Box selection mode. Draw your bounding box by clicking and dragging.")
                self.draw_current_state()
            else:
                print(f"Finish selecting keypoints (need {self.num_keypoints - self.current_point_index} more) first before defining bounding box.")
        elif key == ord('c') or key == ord('C'): # New: Confirm BBox
            if self.bbox_mode and self.current_bbox:
                print("Bounding Box confirmed!")
                self.bbox_mode = False # Exit BBox mode after confirmation
                print("Bounding box confirmed. You can now press 'F' to finish this reference image.")
                self.draw_current_state()
            elif self.bbox_mode and not self.current_bbox:
                print("No bounding box drawn yet. Draw one or press 'N' to skip.")
            else:
                print("Not in Bounding Box selection mode. Press 'B' to enter BBox mode.")

        return True # Continue loop


    def run(self, reference_folder, num_keypoints, output_config_path="reference_config.json"):
        self.reference_folder = Path(reference_folder)
        if not self.reference_folder.exists() or not self.reference_folder.is_dir():
            raise ValueError(f"Reference folder not found: {reference_folder}")
        
        self.reference_paths = sorted(list(self.reference_folder.glob("*.jpg")) + 
                                      list(self.reference_folder.glob("*.jpeg")) +
                                      list(self.reference_folder.glob("*.png")))
        if not self.reference_paths:
            raise ValueError(f"No image files found in reference folder: {reference_folder}")
        
        self.num_keypoints = num_keypoints
        print(f"Configuring {len(self.reference_paths)} reference images with {self.num_keypoints} keypoints each.")

        # Load the first image
        if not self.load_next_reference():
            print("No valid reference images to process.")
            return None

        while True:
            self.draw_current_state()
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_key_press(key):
                break
            
            # Check if all images are processed after handling a key press
            if self.current_ref_index >= len(self.reference_paths) and not self.bbox_mode and self.current_point_index >= self.num_keypoints:
                # If the last image was just processed and finished, break the loop
                break

        cv2.destroyAllWindows()
        
        if self.reference_configs:
            return self.save_config(output_config_path)
        else:
            print("No reference configurations were created.")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Reference Setup System for Auto Annotation")
    parser.add_argument('--reference-folder', required=True,
                        help='Folder containing reference images')
    parser.add_argument('--num-keypoints', type=int, required=True,
                        help='Number of keypoints to annotate per object')
    parser.add_argument('--output-config', default='reference_config.json',
                        help='Output path for the reference configuration JSON file')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    print("\n🛩️ Interactive Reference Setup System")
    print("=" * 50)
    
    if args.num_keypoints > 10:
        print("⚠️ Warning: More than 10 keypoints may be difficult to distinguish by color.")
        print("Consider using fewer keypoints or adjusting the color palette.")
    
    try:
        setup = FixedInteractiveReferenceSetup(device=args.device)
        config_path = setup.run(args.reference_folder, args.num_keypoints, args.output_config)
        
        if config_path:
            print(f"\n✅ Setup completed! Config saved to: {config_path}")
            print(f"💡 To use this config with your auto-annotation script:")
            print(f"   python your_auto_annotation_script.py --reference-config {config_path} --input-folder YOUR_IMAGES/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()



# # annotator_config.py - Modified Code

# """
# Fixed Interactive Reference Setup System (Working Skip Key)
# - Universal keypoint annotation tool for any object type
# - Interactive GUI for setting up reference images
# - SuperPoint visualization with click-to-select keypoints
# - Enhanced skip functionality for non-visible keypoints
# - Proper coordinate scaling for different image sizes
# - Automatic config generation for DINOv2 multi-reference system
# - ADDED: Bounding box selection functionality
# """
# import cv2
# import numpy as np
# import torch
# import os
# import json
# import argparse
# from datetime import datetime
# from pathlib import Path
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
# torch.set_grad_enabled(False)

# try:
#     from lightglue import SuperPoint
#     print("✅ SuperPoint imported successfully")
# except ImportError as e:
#     print(f"❌ Error importing SuperPoint: {e}")
#     print("Please install: pip install lightglue")
#     exit(1)


# class FixedInteractiveReferenceSetup:
#     def __init__(self, device='auto'):
#         """
#         Initialize interactive reference setup system
        
#         Args:
#             device: 'auto', 'cuda', or 'cpu'
#         """
#         # Device setup
#         if device == 'auto':
#             self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         else:
#             self.device = device
#         print(f"🚀 Using device: {self.device}")
        
#         # Initialize SuperPoint
#         print("🔄 Loading SuperPoint...")
#         self.extractor = SuperPoint(max_num_keypoints=9192).eval().to(self.device)
#         #self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
#         print("✅ SuperPoint loaded!")
        
#         self.window_name = "Reference Image Setup - Press 'Q' to quit"
#         cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
#         cv2.setMouseCallback(self.window_name, self.mouse_callback)

#         self.reference_folder = None
#         self.reference_paths = []
#         self.current_ref_index = 0
#         self.current_image = None
#         self.original_width = 0
#         self.original_height = 0
#         self.scale_factor = 1.0
#         #self.scale_factor = 2.0
#         self.superpoint_keypoints = None
#         self.selected_keypoints = []
#         self.current_point_index = 0
#         self.num_keypoints = 0
#         self.reference_configs = []

#         # Bounding box specific variables
#         self.bbox_start_point = None
#         self.bbox_end_point = None
#         self.current_bbox = [] # [x_min, y_min, x_max, y_max]
#         self.bbox_mode = False # New mode for bounding box selection


#     def mouse_callback(self, event, x, y, flags, param):
#         if self.current_image is None:
#             return

#         if self.bbox_mode:
#             if event == cv2.EVENT_LBUTTONDOWN:
#                 self.bbox_start_point = (x, y)
#                 self.bbox_end_point = None
#                 print(f"BBox: Start point selected at ({x}, {y})")
#             elif event == cv2.EVENT_LBUTTONUP:
#                 self.bbox_end_point = (x, y)
#                 if self.bbox_start_point:
#                     x_min = min(self.bbox_start_point[0], self.bbox_end_point[0])
#                     y_min = min(self.bbox_start_point[1], self.bbox_end_point[1])
#                     x_max = max(self.bbox_start_point[0], self.bbox_end_point[0])
#                     y_max = max(self.bbox_start_point[1], self.bbox_end_point[1])
#                     self.current_bbox = [x_min, y_min, x_max, y_max]
#                     print(f"BBox: Selected [{x_min}, {y_min}, {x_max}, {y_max}]")
#                     self.draw_current_state()
#                     print("Press 'C' to confirm BBox, 'R' to reset BBox, or 'N' to skip BBox for this image.")
#             elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
#                 if self.bbox_start_point:
#                     self.bbox_end_point = (x, y)
#                     self.draw_current_state() # Redraw dynamically as user drags
#         elif event == cv2.EVENT_LBUTTONDOWN and self.current_point_index < self.num_keypoints:
#             # Check if clicked on a SuperPoint keypoint
#             if self.superpoint_keypoints is not None:
#                 # Scale mouse click to original image coordinates for comparison
#                 click_x_original = x / self.scale_factor
#                 click_y_original = y / self.scale_factor

#                 # Find the closest SuperPoint keypoint to the click
#                 distances = np.linalg.norm(self.superpoint_keypoints - np.array([click_x_original, click_y_original]), axis=1)
#                 closest_idx = np.argmin(distances)
                
#                 # If the closest keypoint is within a certain threshold (e.g., 5 pixels scaled)
#                 if distances[closest_idx] < (5 / self.scale_factor): # Adjust threshold based on display scale
#                     kp_original_coords = self.superpoint_keypoints[closest_idx]
                    
#                     # Check if this keypoint has already been selected
#                     already_selected = False
#                     for existing_kp_idx, _ in self.selected_keypoints:
#                         if np.array_equal(self.superpoint_keypoints[existing_kp_idx], kp_original_coords):
#                             already_selected = True
#                             print(f"Keypoint already selected. Choose another one.")
#                             break
                    
#                     if not already_selected:
#                         self.selected_keypoints.append((closest_idx, kp_original_coords))
#                         print(f"Keypoint {self.current_point_index + 1} selected at: {kp_original_coords} (Original)")
#                         self.current_point_index += 1
#                         self.draw_current_state()
#                         if self.current_point_index == self.num_keypoints:
#                             print(f"All {self.num_keypoints} keypoints selected. Press 'B' to define Bounding Box or 'F' to finish.")
#                 else:
#                     print(f"No SuperPoint keypoint found close enough at ({x},{y}). Please click closer to a red dot.")


#     def draw_current_state(self):
#         if self.current_image is None:
#             return

#         display_image = self.current_image.copy()

#         # Draw all SuperPoint keypoints in red
#         if self.superpoint_keypoints is not None:
#             for kp_original in self.superpoint_keypoints:
#                 kp_display = (int(kp_original[0] * self.scale_factor), int(kp_original[1] * self.scale_factor))
#                 cv2.circle(display_image, kp_display, 1, (0, 0, 255), -1) # Red dots for all detected KPs
#                 #cv2.circle(display_image, kp_display, 3, (0, 0, 255), -1) # Red dots for all detected KPs

#         # Draw selected keypoints in numbered colors
#         colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
#                   (255, 0, 255), (192, 192, 192), (128, 0, 0), (128, 128, 0), (0, 128, 0)] # More distinct colors
#         for i, (kp_idx, kp_original_coords) in enumerate(self.selected_keypoints):
#             kp_display = (int(kp_original_coords[0] * self.scale_factor), int(kp_original_coords[1] * self.scale_factor))
#             color = colors[i % len(colors)]
#             cv2.circle(display_image, kp_display, 7, color, -1)
#             cv2.putText(display_image, str(i + 1), (kp_display[0] + 10, kp_display[1] - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
#         if self.bbox_mode:
#             instruction_text = "Draw Bounding Box: Click & Drag. Press 'C' to confirm, 'R' to reset, 'N' to skip."
#             cv2.putText(display_image, instruction_text, (20, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
#             cv2.putText(display_image, instruction_text, (20, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Green for BBox mode

#             if self.bbox_start_point and self.bbox_end_point:
#                 # Draw the bounding box currently being drawn or the confirmed one
#                 x1, y1 = self.bbox_start_point
#                 x2, y2 = self.bbox_end_point
#                 cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green rectangle
#             elif self.bbox_start_point and not self.bbox_end_point: # Drawing in progress
#                 cv2.circle(display_image, self.bbox_start_point, 5, (0, 0, 255), -1) # Red start point
            
#             if self.current_bbox: # Display the current confirmed bounding box
#                 x_min, y_min, x_max, y_max = self.current_bbox
#                 cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
#         else: # Keypoint mode instructions
#             instruction_text = f"Select Keypoint {self.current_point_index + 1}/{self.num_keypoints} (Click on red dot)"
#             if self.current_point_index >= self.num_keypoints:
#                 instruction_text = f"All {self.num_keypoints} keypoints selected. Press 'B' for BBox, 'F' to finish."
#             cv2.putText(display_image, instruction_text, (20, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
#             cv2.putText(display_image, instruction_text, (20, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # Red for keypoint mode

#         # Display current reference info
#         cv2.putText(display_image, f"Image: {self.reference_paths[self.current_ref_index].name} ({self.current_ref_index + 1}/{len(self.reference_paths)})",
#                     (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         cv2.putText(display_image, f"Original Size: {self.original_width}x{self.original_height}, Display Scale: {self.scale_factor:.2f}",
#                     (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         # Controls info
#         controls_text = "Controls: 'S'=Skip Keypoint, 'R'=Reset Current, 'B'=BBox Mode, 'F'=Finish Image, 'N'=Next Image, 'Q'=Quit"
#         cv2.putText(display_image, controls_text, (20, display_image.shape[0] - 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         cv2.imshow(self.window_name, display_image)


#     def _extract_superpoint_keypoints(self, image):
#         # Convert image to grayscale and normalize
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         h, w = gray_image.shape
#         # Ensure image is float32 and in range [0, 1]
#         image_tensor = torch.from_numpy(gray_image / 255.0).float()[None, None].to(self.device)

#         # Extract features
#         pred = self.extractor({'image': image_tensor})
#         # keypoints are in 'image' coordinates, need to be detached from GPU
#         keypoints = pred['keypoints'][0].detach().cpu().numpy()
#         return keypoints


#     def load_next_reference(self):
#         if self.current_ref_index >= len(self.reference_paths):
#             print("All reference images processed.")
#             return False

#         image_path = self.reference_paths[self.current_ref_index]
#         print(f"\nLoading reference image: {image_path.name}")
#         image = cv2.imread(str(image_path))
#         if image is None:
#             print(f"❌ Could not load image: {image_path.name}")
#             self.current_ref_index += 1
#             return self.load_next_reference() # Try next image

#         self.original_height, self.original_width = image.shape[:2]
        
#         # Determine display scale
#         max_dim = 1000 # Max dimension for display
#         if max(self.original_height, self.original_width) > max_dim:
#             self.scale_factor = max_dim / max(self.original_height, self.original_width)
#             display_width = int(self.original_width * self.scale_factor)
#             display_height = int(self.original_height * self.scale_factor)
#             self.current_image = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
#         else:
#             self.scale_factor = 1.0
#             self.current_image = image.copy()

#         # Extract SuperPoint keypoints from the ORIGINAL image for accurate coordinates
#         self.superpoint_keypoints = self._extract_superpoint_keypoints(image)
#         print(f"Found {len(self.superpoint_keypoints)} SuperPoint keypoints.")

#         self.selected_keypoints = []
#         self.current_point_index = 0
#         self.bbox_start_point = None
#         self.bbox_end_point = None
#         self.current_bbox = []
#         self.bbox_mode = False # Start in keypoint mode
#         return True


#     def skip_current_keypoint(self):
#         if self.current_point_index < self.num_keypoints:
#             self.selected_keypoints.append((-1, [0, 0])) # Use -1 index and [0,0] coordinates to signify skipped
#             print(f"Keypoint {self.current_point_index + 1} skipped.")
#             self.current_point_index += 1
#             self.draw_current_state()
#             if self.current_point_index == self.num_keypoints:
#                 print(f"All {self.num_keypoints} keypoints selected. Press 'B' to define Bounding Box or 'F' to finish.")
#         else:
#             print("All keypoints already selected. Cannot skip.")

#     def reset_current_keypoints(self):
#         self.selected_keypoints = []
#         self.current_point_index = 0
#         print("All keypoints reset for current image.")
#         self.draw_current_state()

#     def reset_current_bbox(self):
#         self.bbox_start_point = None
#         self.bbox_end_point = None
#         self.current_bbox = []
#         print("Bounding Box reset.")
#         self.draw_current_state()

#     def finish_current_reference(self):
#         if self.current_point_index < self.num_keypoints:
#             print(f"Please select all {self.num_keypoints} keypoints (or skip them) before finishing.")
#             return

#         if self.bbox_mode:
#             print("Please confirm or skip the bounding box first (press 'C' or 'N').")
#             return

#         print(f"\nProcessing reference image {self.current_ref_index + 1}/{len(self.reference_paths)}...")

#         # Prepare keypoints dictionary
#         keypoints_dict = {}
#         for i in range(self.num_keypoints):
#             if i < len(self.selected_keypoints):
#                 _ , kp_original_coords = self.selected_keypoints[i]
#                 keypoints_dict[f'keypoint_{i+1}'] = [float(kp_original_coords[0]), float(kp_original_coords[1])]
#             else:
#                 keypoints_dict[f'keypoint_{i+1}'] = [0, 0] # Should not happen if current_point_index check works

#         # Convert bounding box coordinates back to original image size
#         original_bbox = []
#         if self.current_bbox:
#             x_min, y_min, x_max, y_max = self.current_bbox
#             original_x_min = x_min / self.scale_factor
#             original_y_min = y_min / self.scale_factor
#             original_x_max = x_max / self.scale_factor
#             original_y_max = y_max / self.scale_factor
#             original_bbox = [float(original_x_min), float(original_y_min), float(original_x_max), float(original_y_max)]
#             print(f"📦 Bounding box (original size): {original_bbox}")

#         # Create reference config
#         ref_config = {
#             'id': f'reference_{self.current_ref_index}',
#             'image_path': str(self.reference_paths[self.current_ref_index]),
#             'viewpoint': f'viewpoint_{self.current_ref_index}',
#             'description': f'Reference image {self.current_ref_index + 1}',
#             'keypoints': keypoints_dict,
#             'bbox': original_bbox, # ADD THIS LINE
#             'original_size': [self.original_width, self.original_height],
#             'display_size': [self.current_image.shape[1], self.current_image.shape[0]],
#             'scale_factor': self.scale_factor
#         }
#         self.reference_configs.append(ref_config)
#         print(f"✅ Config saved for {self.reference_paths[self.current_ref_index].name}")

#         self.current_ref_index += 1
#         # Reset bounding box for next image
#         self.bbox_start_point = None
#         self.bbox_end_point = None
#         self.current_bbox = []
#         self.bbox_mode = False # Ensure we start with keypoint mode for next image
#         self.load_next_reference() # Load the next image automatically


#     def save_config(self, output_path="reference_config.json"):
#         final_config = {
#             "created_at": datetime.now().isoformat(),
#             "num_keypoints_per_object": self.num_keypoints,
#             "references": self.reference_configs
#         }
#         with open(output_path, 'w') as f:
#             json.dump(final_config, f, indent=4)
#         print(f"\n✨ Final configuration saved to: {output_path}")
#         return output_path


#     def handle_key_press(self, key):
#         if key == ord('s') or key == ord('S'): # Skip keypoint
#             if not self.bbox_mode:
#                 self.skip_current_keypoint()
#             else:
#                 print("Cannot skip keypoint in BBox mode. Press 'R' to reset BBox or 'N' to skip BBox.")
#         elif key == ord('r') or key == ord('R'): # Reset current
#             if self.bbox_mode:
#                 self.reset_current_bbox()
#             else:
#                 self.reset_current_keypoints()
#         elif key == ord('q') or key == ord('Q'): # Quit
#             return False 
#         elif key == ord('f') or key == ord('F'): # Finish current reference (keypoints + bbox)
#             if self.current_point_index < self.num_keypoints:
#                 print(f"Please select all {self.num_keypoints} keypoints (or skip them) before finishing. Or press 'B' to define BBox if keypoints are done.")
#             elif self.bbox_mode:
#                  print("Please confirm or skip the bounding box first (press 'C' or 'N').")
#             else:
#                 self.finish_current_reference()
#         elif key == ord('n') or key == ord('N'): # Next image (skip all remaining keypoints/bbox and move to next)
#             if self.current_point_index < self.num_keypoints:
#                 print(f"Skipping remaining keypoints and moving to next image for {self.reference_paths[self.current_ref_index].name}...")
#                 while self.current_point_index < self.num_keypoints:
#                     self.skip_current_keypoint()
#             if self.bbox_mode:
#                 print(f"Skipping bounding box and moving to next image for {self.reference_paths[self.current_ref_index].name}...")
#                 self.current_bbox = [] # Mark as no bbox
#                 self.bbox_mode = False # Exit BBox mode
            
#             # If keypoints were completed and bbox was skipped/completed, now finish the reference
#             if self.current_point_index == self.num_keypoints and not self.bbox_mode:
#                 self.finish_current_reference()
#             else: # If N was pressed prematurely
#                 print("Action 'N' performed. Moving to next reference image if available.")
#                 self.current_ref_index += 1
#                 self.load_next_reference()


#         elif key == ord('b') or key == ord('B'): # New: Switch to BBox mode
#             if self.current_point_index >= self.num_keypoints: # Only allow BBox mode after keypoints
#                 self.bbox_mode = True
#                 self.bbox_start_point = None
#                 self.bbox_end_point = None
#                 self.current_bbox = [] # Reset for new bbox drawing
#                 print("Switched to Bounding Box selection mode. Draw your bounding box by clicking and dragging.")
#                 self.draw_current_state()
#             else:
#                 print(f"Finish selecting keypoints (need {self.num_keypoints - self.current_point_index} more) first before defining bounding box.")
#         elif key == ord('c') or key == ord('C'): # New: Confirm BBox
#             if self.bbox_mode and self.current_bbox:
#                 print("Bounding Box confirmed!")
#                 self.bbox_mode = False # Exit BBox mode after confirmation
#                 print("Bounding box confirmed. You can now press 'F' to finish this reference image.")
#                 self.draw_current_state()
#             elif self.bbox_mode and not self.current_bbox:
#                 print("No bounding box drawn yet. Draw one or press 'N' to skip.")
#             else:
#                 print("Not in Bounding Box selection mode. Press 'B' to enter BBox mode.")

#         return True # Continue loop


#     def run(self, reference_folder, num_keypoints, output_config_path="reference_config.json"):
#         self.reference_folder = Path(reference_folder)
#         if not self.reference_folder.exists() or not self.reference_folder.is_dir():
#             raise ValueError(f"Reference folder not found: {reference_folder}")
        
#         self.reference_paths = sorted(list(self.reference_folder.glob("*.jpg")) + 
#                                       list(self.reference_folder.glob("*.jpeg")) +
#                                       list(self.reference_folder.glob("*.png")))
#         if not self.reference_paths:
#             raise ValueError(f"No image files found in reference folder: {reference_folder}")
        
#         self.num_keypoints = num_keypoints
#         print(f"Configuring {len(self.reference_paths)} reference images with {self.num_keypoints} keypoints each.")

#         # Load the first image
#         if not self.load_next_reference():
#             print("No valid reference images to process.")
#             return None

#         while True:
#             self.draw_current_state()
#             key = cv2.waitKey(1) & 0xFF
#             if not self.handle_key_press(key):
#                 break
            
#             # Check if all images are processed after handling a key press
#             if self.current_ref_index >= len(self.reference_paths) and not self.bbox_mode and self.current_point_index >= self.num_keypoints:
#                 # If the last image was just processed and finished, break the loop
#                 break

#         cv2.destroyAllWindows()
        
#         if self.reference_configs:
#             return self.save_config(output_config_path)
#         else:
#             print("No reference configurations were created.")
#             return None


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Interactive Reference Setup System for Auto Annotation")
#     parser.add_argument('--reference-folder', required=True,
#                         help='Folder containing reference images')
#     parser.add_argument('--num-keypoints', type=int, required=True,
#                         help='Number of keypoints to annotate per object')
#     parser.add_argument('--output-config', default='reference_config.json',
#                         help='Output path for the reference configuration JSON file')
#     parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
#                         help='Device to use for inference')
    
#     args = parser.parse_args()
    
#     print("\n🛩️ Interactive Reference Setup System")
#     print("=" * 50)
    
#     if args.num_keypoints > 10:
#         print("⚠️ Warning: More than 10 keypoints may be difficult to distinguish by color.")
#         print("Consider using fewer keypoints or adjusting the color palette.")
    
#     try:
#         setup = FixedInteractiveReferenceSetup(device=args.device)
#         config_path = setup.run(args.reference_folder, args.num_keypoints, args.output_config)
        
#         if config_path:
#             print(f"\n✅ Setup completed! Config saved to: {config_path}")
#             print(f"💡 To use this config with your auto-annotation script:")
#             print(f"   python your_auto_annotation_script.py --reference-config {config_path} --input-folder YOUR_IMAGES/")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         import traceback
#         traceback.print_exc()