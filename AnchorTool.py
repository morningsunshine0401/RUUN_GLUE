

"""
Interactive Anchor Correspondence Tool
Creates new anchor 2D-3D correspondences by manually clicking between reference and new anchor images.

Usage:
1. Left side: Reference anchor with known 2D-3D correspondences (red dots)
2. Right side: New anchor image with SuperPoint features (green dots)
3. Click red dot on left, then corresponding green dot on right
4. Press 'q' to finish and export the new anchor data
"""

import cv2
import numpy as np
import torch
import json
import os
from pathlib import Path

# You'll need to have SuperPoint available - adjust import as needed
try:
    from lightglue import SuperPoint
    from lightglue.utils import rbd
    print("✅ SuperPoint loaded successfully")
except ImportError:
    print("❌ SuperPoint not available. Please ensure lightglue is installed.")
    exit(1)

class AnchorCorrespondenceTool:
    def __init__(self, reference_anchor_path, new_anchor_path, reference_viewpoint='SW'):
        """
        Initialize the correspondence tool
        
        Args:
            reference_anchor_path: Path to reference anchor image
            new_anchor_path: Path to new anchor image  
            reference_viewpoint: Which viewpoint to use for reference ('SW', 'NE', etc.)
        """
        self.reference_anchor_path = reference_anchor_path
        self.new_anchor_path = new_anchor_path
        self.reference_viewpoint = reference_viewpoint
        
        # Display settings
        self.display_width = 640
        self.display_height = 480
        self.window_name = "Anchor Correspondence Tool"
        
        # Correspondence data
        self.correspondences = []  # List of (ref_2d, ref_3d, new_2d) tuples
        self.selected_ref_point = None
        self.selected_ref_idx = None
        
        # Visual feedback
        self.ref_point_radius = 5
        self.new_point_radius = 3
        self.selected_color = (0, 255, 255)  # Yellow for selected
        self.ref_color = (0, 0, 255)  # Red for reference points
        self.new_color = (0, 255, 0)  # Green for new anchor features
        self.correspondence_color = (255, 0, 255)  # Magenta for established correspondences
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")
        
        # Initialize SuperPoint
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        
        # Load reference anchor data (from your existing code)
        self._load_reference_data()
        
        # Load and process images
        self._load_images()
        
        print("✅ Anchor Correspondence Tool initialized!")
        print("📖 Instructions:")
        print("   1. Click on RED dot (left side) to select reference point")
        print("   2. Click on GREEN dot (right side) to establish correspondence")
        print("   3. Press 'u' to undo last correspondence")
        print("   4. Press 'r' to reset current selection")
        print("   5. Press 'c' to clear all correspondences")
        print("   6. Press 'q' to finish and export new anchor data")

    def _load_reference_data(self):
        """Load the reference 2D-3D data from your existing anchor definitions"""
        # This matches the data from your code
        if self.reference_viewpoint == 'SW':
            self.ref_2d = np.array([
                [650, 312], [630, 306], [907, 443], [814, 291], [599, 349], 
                [501, 386], [965, 359], [649, 355], [635, 346], [930, 335], 
                [843, 467], [702, 339], [718, 321], [930, 322], [727, 346], 
                [539, 364], [786, 297], [1022, 406], [1004, 399], [539, 344], 
                [536, 309], [864, 478], [745, 310], [1049, 393], [895, 258], 
                [674, 347], [741, 281], [699, 294], [817, 494], [992, 281]
            ], dtype=np.float32)
            
            self.ref_3d = np.array([
                [-0.035, -0.018, -0.010], [-0.057, -0.018, -0.010], [ 0.217, -0.000, -0.027], 
                [-0.014, -0.000,  0.156], [-0.023, -0.000, -0.065], [-0.014, -0.000, -0.156], 
                [ 0.234, -0.050, -0.002], [ 0.000, -0.000, -0.042], [-0.014, -0.000, -0.042], 
                [ 0.206, -0.055, -0.002], [ 0.217, -0.000, -0.070], [ 0.025, -0.014, -0.011], 
                [-0.014, -0.000,  0.042], [ 0.206, -0.070, -0.002], [ 0.049, -0.016, -0.011], 
                [-0.029, -0.000, -0.127], [-0.019, -0.000,  0.128], [ 0.230, -0.000,  0.070], 
                [ 0.217, -0.000,  0.070], [-0.052, -0.000, -0.097], [-0.175, -0.000, -0.015], 
                [ 0.230, -0.000, -0.070], [-0.019, -0.000,  0.074], [ 0.230, -0.000,  0.113], 
                [-0.000, -0.025,  0.240], [-0.000, -0.000, -0.015], [-0.074, -0.000,  0.128], 
                [-0.074, -0.000,  0.074], [ 0.230, -0.000, -0.113], [ 0.243, -0.104,  0.000]
            ], dtype=np.float32)
        else:
            # Default anchor data for other viewpoints
            self.ref_2d = np.array([
                [511, 293], [591, 284], [587, 330], [413, 249], [602, 348], 
                [715, 384], [598, 298], [656, 171], [805, 213], [703, 392], 
                [523, 286], [519, 327], [387, 289], [727, 126], [425, 243], 
                [636, 358], [745, 202], [595, 388], [436, 260], [539, 313], 
                [795, 220], [351, 291], [665, 165], [611, 353], [650, 377], 
                [516, 389], [727, 143], [496, 378], [575, 312], [617, 368], 
                [430, 312], [480, 281], [834, 225], [469, 339], [705, 223], 
                [637, 156], [816, 414], [357, 195], [752, 77], [642, 451]
            ], dtype=np.float32)
            
            self.ref_3d = np.array([
                [-0.014, 0.000, 0.042], [0.025, -0.014, -0.011], [-0.014, 0.000, -0.042], 
                [-0.014, 0.000, 0.156], [-0.023, 0.000, -0.065], [0.000, 0.000, -0.156], 
                [0.025, 0.000, -0.015], [0.217, 0.000, 0.070], [0.230, 0.000, -0.070], 
                [-0.014, 0.000, -0.156], [0.000, 0.000, 0.042], [-0.057, -0.018, -0.010], 
                [-0.074, -0.000, 0.128], [0.206, -0.070, -0.002], [-0.000, -0.000, 0.156], 
                [-0.017, -0.000, -0.092], [0.217, -0.000, -0.027], [-0.052, -0.000, -0.097], 
                [-0.019, -0.000, 0.128], [-0.035, -0.018, -0.010], [0.217, -0.000, -0.070], 
                [-0.080, -0.000, 0.156], [0.230, -0.000, 0.070], [-0.023, -0.000, -0.075], 
                [-0.029, -0.000, -0.127], [-0.090, -0.000, -0.042], [0.206, -0.055, -0.002], 
                [-0.090, -0.000, -0.015], [0.000, -0.000, -0.015], [-0.037, -0.000, -0.097], 
                [-0.074, -0.000, 0.074], [-0.019, -0.000, 0.074], [0.230, -0.000, -0.113], 
                [-0.100, -0.030, 0.000], [0.170, -0.000, -0.015], [0.230, -0.000, 0.113], 
                [-0.000, -0.025, -0.240], [-0.000, -0.025, 0.240], [0.243, -0.104, 0.000], 
                [-0.080, -0.000, -0.156]
            ], dtype=np.float32)

    def _load_images(self):
        """Load and process both anchor images"""
        # Load reference image
        self.ref_image = cv2.imread(self.reference_anchor_path)
        if self.ref_image is None:
            raise FileNotFoundError(f"Could not load reference image: {self.reference_anchor_path}")
        
        # Load new anchor image
        self.new_image = cv2.imread(self.new_anchor_path)
        if self.new_image is None:
            raise FileNotFoundError(f"Could not load new anchor image: {self.new_anchor_path}")
        
        # Resize for display
        self.ref_display = cv2.resize(self.ref_image, (self.display_width, self.display_height))
        self.new_display = cv2.resize(self.new_image, (self.display_width, self.display_height))
        
        # Calculate scaling factors for coordinate conversion
        self.ref_scale_x = self.display_width / self.ref_image.shape[1]
        self.ref_scale_y = self.display_height / self.ref_image.shape[0]
        self.new_scale_x = self.display_width / self.new_image.shape[1] 
        self.new_scale_y = self.display_height / self.new_image.shape[0]
        
        # Scale reference 2D points for display
        self.ref_2d_display = self.ref_2d.copy()
        self.ref_2d_display[:, 0] *= self.ref_scale_x
        self.ref_2d_display[:, 1] *= self.ref_scale_y
        
        # Extract SuperPoint features from new anchor
        self._extract_new_features()

    def _extract_features_from_image(self, image):
        """Extract SuperPoint features from an image"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            return self.extractor.extract(tensor)

    def _extract_new_features(self):
        """Extract SuperPoint features from the new anchor image"""
        print("🔍 Extracting SuperPoint features from new anchor...")
        features = self._extract_features_from_image(self.new_image)
        self.new_keypoints = features['keypoints'][0].cpu().numpy()
        
        # Scale keypoints for display
        self.new_keypoints_display = self.new_keypoints.copy()
        self.new_keypoints_display[:, 0] *= self.new_scale_x
        self.new_keypoints_display[:, 1] *= self.new_scale_y
        
        print(f"✅ Found {len(self.new_keypoints)} SuperPoint features in new anchor")

    def _draw_points(self):
        """Draw all points and correspondences on the combined display"""
        # Create combined display
        combined = np.hstack([self.ref_display.copy(), self.new_display.copy()])
        
        # Draw reference points (red dots)
        for i, (x, y) in enumerate(self.ref_2d_display):
            color = self.selected_color if i == self.selected_ref_idx else self.ref_color
            cv2.circle(combined, (int(x), int(y)), self.ref_point_radius, color, -1)
            cv2.putText(combined, str(i), (int(x+8), int(y-8)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw new anchor features (green dots)
        for i, (x, y) in enumerate(self.new_keypoints_display):
            cv2.circle(combined, (int(x + self.display_width), int(y)), 
                      self.new_point_radius, self.new_color, -1)
        
        # Draw established correspondences (subtle lines that don't block view)
        for i, (ref_2d, ref_3d, new_2d) in enumerate(self.correspondences):
            # Reference point (scaled)
            ref_x = int(ref_2d[0] * self.ref_scale_x)
            ref_y = int(ref_2d[1] * self.ref_scale_y)
            
            # New point (scaled and offset)
            new_x = int(new_2d[0] * self.new_scale_x + self.display_width)
            new_y = int(new_2d[1] * self.new_scale_y)
            
            # Draw subtle dashed line connecting correspondence
            self._draw_dashed_line(combined, (ref_x, ref_y), (new_x, new_y), 
                                 self.correspondence_color, thickness=1, dash_length=5)
            
            # Mark corresponding points with small circles instead of blocking text
            cv2.circle(combined, (ref_x, ref_y), 8, self.correspondence_color, 2)
            cv2.circle(combined, (new_x, new_y), 8, self.correspondence_color, 2)
            
            # Draw small correspondence number near the reference point
            cv2.putText(combined, str(i), (ref_x - 15, ref_y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.correspondence_color, 1)
        
        # Draw instructions
        cv2.putText(combined, "Reference Anchor (click red dots)", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "New Anchor (click green dots)", (self.display_width + 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f"Correspondences: {len(self.correspondences)}", (10, combined.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(combined, "Press: 'q'=finish, 'u'=undo, 'r'=reset, 'c'=clear all", (10, combined.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.selected_ref_point is not None:
            cv2.putText(combined, f"Selected ref point {self.selected_ref_idx}", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.selected_color, 2)
        
        return combined

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=5):
        """Draw a dashed line between two points"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calculate line length and direction
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length == 0:
            return
            
        # Unit vector
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        
        # Draw dashed line
        draw_dash = True
        current_length = 0
        
        while current_length < length:
            start_x = int(x1 + dx * current_length)
            start_y = int(y1 + dy * current_length)
            
            end_length = min(current_length + dash_length, length)
            end_x = int(x1 + dx * end_length)
            end_y = int(y1 + dy * end_length)
            
            if draw_dash:
                cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
            
            draw_dash = not draw_dash
            current_length += dash_length

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for correspondence selection"""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        if x < self.display_width:
            # Click on reference image (left side)
            self._select_reference_point(x, y)
        else:
            # Click on new anchor image (right side)
            self._select_new_point(x - self.display_width, y)

    def _select_reference_point(self, x, y):
        """Select a reference point (red dot) on the left side"""
        click_point = np.array([x, y])
        distances = np.linalg.norm(self.ref_2d_display - click_point, axis=1)
        closest_idx = np.argmin(distances)
        
        if distances[closest_idx] < 15:  # Within 15 pixels
            self.selected_ref_idx = closest_idx
            self.selected_ref_point = self.ref_2d[closest_idx]  # Original coordinates
            print(f"📍 Selected reference point {closest_idx}: {self.selected_ref_point}")

    def _select_new_point(self, x, y):
        """Select a new anchor point (green dot) on the right side"""
        if self.selected_ref_point is None:
            print("⚠️ Please select a reference point (red dot) first!")
            return
        
        click_point = np.array([x, y])
        distances = np.linalg.norm(self.new_keypoints_display - click_point, axis=1)
        closest_idx = np.argmin(distances)
        
        if distances[closest_idx] < 15:  # Within 15 pixels
            new_point = self.new_keypoints[closest_idx]  # Original coordinates
            ref_3d = self.ref_3d[self.selected_ref_idx]
            
            # Add correspondence
            self.correspondences.append((self.selected_ref_point, ref_3d, new_point))
            
            print(f"✅ Added correspondence {len(self.correspondences)-1}:")
            print(f"   Ref 2D: {self.selected_ref_point}")
            print(f"   Ref 3D: {ref_3d}")
            print(f"   New 2D: {new_point}")
            
            # Reset selection
            self.selected_ref_point = None
            self.selected_ref_idx = None

    def _undo_last_correspondence(self):
        """Remove the last added correspondence"""
        if self.correspondences:
            removed = self.correspondences.pop()
            print(f"🔙 Removed correspondence: {removed[0]} -> {removed[2]}")
        else:
            print("⚠️ No correspondences to undo")

    def _reset_selection(self):
        """Reset current selection"""
        self.selected_ref_point = None
        self.selected_ref_idx = None
        print("🔄 Selection reset")

    def _clear_all_correspondences(self):
        """Clear all correspondences"""
        if self.correspondences:
            count = len(self.correspondences)
            self.correspondences.clear()
            self.selected_ref_point = None
            self.selected_ref_idx = None
            print(f"🗑️ Cleared all {count} correspondences")
        else:
            print("⚠️ No correspondences to clear")

    def run(self):
        """Run the interactive correspondence tool"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("🖱️ Tool ready! Click to create correspondences...")
        
        while True:
            # Draw current state
            display = self._draw_points()
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("✅ Finishing correspondence tool...")
                break
            elif key == ord('r'):
                self._reset_selection()
            elif key == ord('u'):
                self._undo_last_correspondence()
            elif key == ord('c'):
                self._clear_all_correspondences()
            elif key == 27:  # Escape
                print("❌ Cancelled")
                return None
        
        cv2.destroyAllWindows()
        return self._export_anchor_data()

    def _export_anchor_data(self):
        """Export the new anchor 2D and 3D data"""
        if not self.correspondences:
            print("⚠️ No correspondences created!")
            return None
        
        # Extract 2D and 3D points
        new_2d = np.array([corr[2] for corr in self.correspondences], dtype=np.float32)
        new_3d = np.array([corr[1] for corr in self.correspondences], dtype=np.float32)
        
        print(f"📊 Created {len(self.correspondences)} correspondences")
        print("\n🎯 New Anchor Data:")
        print("=" * 50)
        
        # Format for code insertion
        print("# New anchor 2D points:")
        print("new_anchor_2d = np.array([")
        for i, pt in enumerate(new_2d):
            print(f"    [{pt[0]:.0f}, {pt[1]:.0f}]{',' if i < len(new_2d)-1 else ''}")
        print("], dtype=np.float32)")
        
        print("\n# New anchor 3D points:")
        print("new_anchor_3d = np.array([")
        for i, pt in enumerate(new_3d):
            print(f"    [{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}]{',' if i < len(new_3d)-1 else ''}")
        print("], dtype=np.float32)")
        
        # Save to JSON file
        output_data = {
            'reference_anchor': self.reference_anchor_path,
            'new_anchor': self.new_anchor_path,
            'reference_viewpoint': self.reference_viewpoint,
            'correspondences_count': len(self.correspondences),
            'new_anchor_2d': new_2d.tolist(),
            'new_anchor_3d': new_3d.tolist(),
            'correspondences': [
                {
                    'ref_2d': corr[0].tolist(),
                    'ref_3d': corr[1].tolist(), 
                    'new_2d': corr[2].tolist()
                }
                for corr in self.correspondences
            ]
        }
        
        output_file = f"new_anchor_data_{Path(self.new_anchor_path).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"\n💾 Saved detailed data to: {output_file}")
        
        return {
            'anchor_2d': new_2d,
            'anchor_3d': new_3d,
            'correspondences': self.correspondences
        }


def main():
    """Main function to run the anchor correspondence tool"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Anchor Correspondence Tool")
    parser.add_argument('--reference', type=str, required=True,
                       help='Path to reference anchor image')
    parser.add_argument('--new_anchor', type=str, required=True,
                       help='Path to new anchor image')
    parser.add_argument('--reference_viewpoint', type=str, default='SW',
                       choices=['SW', 'NE', 'NW', 'SE'],
                       help='Reference viewpoint for 3D data')
    
    args = parser.parse_args()
    
    # Validate files exist
    if not os.path.exists(args.reference):
        print(f"❌ Reference image not found: {args.reference}")
        return
    
    if not os.path.exists(args.new_anchor):
        print(f"❌ New anchor image not found: {args.new_anchor}")
        return
    
    print("🚀 Starting Anchor Correspondence Tool")
    print("=" * 50)
    print(f"📸 Reference: {args.reference}")
    print(f"📸 New anchor: {args.new_anchor}")
    print(f"🎯 Reference viewpoint: {args.reference_viewpoint}")
    print("=" * 50)
    
    try:
        tool = AnchorCorrespondenceTool(
            args.reference, 
            args.new_anchor, 
            args.reference_viewpoint
        )
        
        result = tool.run()
        
        if result is not None:
            print("✅ Anchor correspondence tool completed successfully!")
            print(f"📊 Generated {len(result['correspondences'])} correspondences")
        else:
            print("❌ Tool cancelled or no correspondences created")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()