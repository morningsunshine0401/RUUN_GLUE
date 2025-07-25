import cv2
import numpy as np
import torch
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from scipy.spatial.distance import cdist
import warnings
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_grad_enabled(False)

try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd
    print("✅ LightGlue imported successfully")
except ImportError as e:
    print(f"❌ Error importing LightGlue: {e}")
    print("Please install: pip install lightglue")
    exit(1)

try:
    import timm
    from torchvision import transforms
    print("✅ Vision transformers available")
except ImportError:
    print("⚠️ timm not available - will use fallback feature extraction")
    timm = None


class EnhancedMultiRefAnnotator:
    """
    Initializes the ENHANCED multi-reference annotator with tie-breaking,
    upscaling refinement, and fallback mechanisms.
    """
    def __init__(self, reference_config_path, output_dir="annotations", device='auto', object_name="aircraft", debug=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.object_name = object_name
        self.debug = debug
        self.visualize = False # Default to False, set via argparse

        # ENHANCED RANSAC parameters (Adjusted for higher success rate/quality)
        self.ransac_threshold = 8.0  # Increased from 8.0 for more leniency
        self.ransac_max_trials = 4000
        self.min_matches_for_ransac = 5  # Decreased from 10 to allow RANSAC with fewer matches
        self.ransac_min_samples = 6  # Decreased from 8 to make RANSAC more flexible

        # Keypoint matching parameters (Adjusted for higher success rate/quality)
        self.distance_threshold = 15.0  # Increased from 15.0 for more tolerant keypoint association
        self.min_keypoints_threshold = 4

        # Multi-reference parameters (only for tie-breaking, adjusted for more robust recovery)
        self.enable_multi_ref_tiebreaker = True
        self.score_difference_threshold = 2.0  # Increased from 1.0 to allow more "tied" references
        self.max_tied_refs = 3

        # Upscaling refinement parameters (Adjusted for better refinement)
        self.enable_upscaling_refinement = True
        self.upscale_factor = 2.0
        self.upscale_padding_ratio = 0.1  # Increased from 0.1 to provide more context for upscaling
        self.min_upscale_size = 100
        self.upscale_distance_threshold = 12.0

        # Scoring parameters for reference selection
        self.total_match_weight = 0.6
        self.keypoint_match_weight = 0.4

        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"🚀 Using device: {self.device}")

        # Initialize models and configuration
        print("🔄 Loading AI models...")
        self._init_models()
        print("🔄 Loading reference configuration...")
        self._load_reference_config(reference_config_path)
        self._init_coco_dataset()

    def _init_models(self):
        """Initialize all AI models."""
        self.extractor = SuperPoint(max_num_keypoints=4096).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        print("✅ SuperPoint & LightGlue loaded!")
        
    def _load_reference_config(self, config_path):
        """Load reference images and their keypoints from a JSON config file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.references = {}
        self.num_keypoints = config.get('num_keypoints', 6)
        self.keypoint_names = config.get('keypoint_names', [f'keypoint_{i}' for i in range(self.num_keypoints)])
        
        print(f"🔄 Loading {len(config['references'])} reference images...")
        print(f"📍 Object keypoints: {self.num_keypoints} ({', '.join(self.keypoint_names)})")

        for ref_config in config['references']:
            ref_id = ref_config['id']
            ref_path = ref_config['image_path']
            ref_image = cv2.imread(ref_path)
            if ref_image is None:
                print(f"❌ Could not load reference image: {ref_path}")
                continue

            original_size = ref_config.get('original_size')
            current_size = [ref_image.shape[1], ref_image.shape[0]]
            keypoints = ref_config['keypoints']
            scaled_keypoints = keypoints.copy()
            if original_size and original_size != current_size:
                scale_x = current_size[0] / original_size[0]
                scale_y = current_size[1] / original_size[1]
                for name, coords in keypoints.items():
                    if coords != [0, 0]:
                        scaled_keypoints[name] = [coords[0] * scale_x, coords[1] * scale_y]

            visible_keypoints_count = sum(1 for coords in scaled_keypoints.values() if coords != [0, 0])

            self.references[ref_id] = {
                'image': ref_image,
                'keypoints': scaled_keypoints,
                'viewpoint': ref_config.get('viewpoint', 'unknown'),
                'bbox': ref_config.get('bbox'), # Store bbox from config
                'superpoint_features': self._extract_superpoint_features(ref_image, ref_config.get('bbox')), # Pass bbox for cropping
                'visible_keypoints_count': visible_keypoints_count
            }
        print(f"✅ Loaded {len(self.references)} reference images successfully.")

    def _extract_superpoint_features(self, image, bbox=None):
        """
        Extract SuperPoint features from an image.
        If a bbox is provided, features are extracted from the cropped region,
        and then their coordinates are adjusted back to the original image's frame.
        """
        input_image_for_extractor = image
        crop_offset = (0, 0) # Default offset

        # Add a robust check for bbox validity
        if bbox is not None and len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            # Ensure crop dimensions are positive and within image bounds
            if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= image.shape[1] and (y + h) <= image.shape[0]:
                input_image_for_extractor = image[y:y+h, x:x+w]
                crop_offset = (x, y)
                if self.debug:
                    print(f"   Debug: Cropping reference image to bbox {bbox} for feature extraction.")
            else:
                if self.debug:
                    print(f"   Warning: Invalid bbox dimensions {bbox} for image size {image.shape[:2]}. Using full image.")
        elif bbox is not None and self.debug:
            print(f"   Warning: Malformed bbox {bbox}. Expected 4 values, got {len(bbox)}. Using full image.")

        rgb = cv2.cvtColor(input_image_for_extractor, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.extractor.extract(tensor)
        
        # Adjust keypoint coordinates if a crop was applied
        if crop_offset != (0, 0) and 'keypoints' in features:
            features['keypoints'][:, 0] += crop_offset[0]
            features['keypoints'][:, 1] += crop_offset[1]
            if self.debug:
                print(f"   Debug: Adjusted SuperPoint keypoints by offset {crop_offset}.")

        return features

    def _apply_enhanced_ransac_filtering(self, ref_kpts, input_kpts, matches):
        """Apply a robust RANSAC filter to matches to remove outliers."""
        if len(matches) < self.min_matches_for_ransac:
            return np.array([]), 0.0

        try:
            matched_ref_kpts = ref_kpts[matches[:, 0]]
            matched_input_kpts = input_kpts[matches[:, 1]]

            poly_features = PolynomialFeatures(degree=2, include_bias=True)
            ref_poly = poly_features.fit_transform(matched_ref_kpts)

            ransac_x = RANSACRegressor(random_state=42, residual_threshold=self.ransac_threshold, max_trials=self.ransac_max_trials, min_samples=self.ransac_min_samples)
            ransac_y = RANSACRegressor(random_state=42, residual_threshold=self.ransac_threshold, max_trials=self.ransac_max_trials, min_samples=self.ransac_min_samples)

            ransac_x.fit(ref_poly, matched_input_kpts[:, 0])
            ransac_y.fit(ref_poly, matched_input_kpts[:, 1])

            inlier_mask = np.logical_and(ransac_x.inlier_mask_, ransac_y.inlier_mask_)
            filtered_matches = matches[inlier_mask]
            
            inlier_ratio = np.sum(inlier_mask) / len(matches) if len(matches) > 0 else 0.0

            # Adjusted inlier ratio threshold (from 0.2 to 0.1)
            if len(filtered_matches) < self.min_matches_for_ransac or inlier_ratio < 0.1:
                return np.array([]), 0.0
            
            return filtered_matches, inlier_ratio

        except Exception:
            return np.array([]), 0.0

    def _find_best_references_with_tiebreaker(self, input_image):
        """Select the best reference image, handling ties by returning multiple candidates."""
        input_features = self._extract_superpoint_features(input_image)
        reference_scores = {}
        
        for ref_id, ref_data in self.references.items():
            with torch.no_grad():
                matches_dict = self.matcher({'image0': ref_data['superpoint_features'], 'image1': input_features})
            
            feats0, feats1, matches01 = [rbd(x) for x in [ref_data['superpoint_features'], input_features, matches_dict]]
            matches = matches01["matches"].detach().cpu().numpy()

            raw_lightglue_matches_count = len(matches) # Debugging added here

            if raw_lightglue_matches_count == 0:
                reference_scores[ref_id] = 0
                if self.debug: # Added debug print for zero matches
                    print(f"      Ref {ref_id}: No raw LightGlue matches found.")
                continue
            
            ref_kpts = feats0["keypoints"].detach().cpu().numpy()
            input_kpts = feats1["keypoints"].detach().cpu().numpy()
            
            filtered_matches, _ = self._apply_enhanced_ransac_filtering(ref_kpts, input_kpts, matches)
            
            if len(filtered_matches) == 0:
                reference_scores[ref_id] = 0
                if self.debug: # Added debug print for RANSAC filtering all matches
                    print(f"      Ref {ref_id}: Raw LightGlue matches: {raw_lightglue_matches_count}, RANSAC filtered: {len(filtered_matches)} (all filtered out).")
                continue
            
            matched_ref_kpts = ref_kpts[filtered_matches[:, 0]]
            successful_keypoints = 0
            for name in self.keypoint_names:
                coords = ref_data['keypoints'].get(name)
                if not coords or coords == [0, 0]:
                    continue
                
                distances = cdist(np.array([coords]), matched_ref_kpts)
                if len(distances) > 0 and np.min(distances) < self.distance_threshold:
                    successful_keypoints += 1
            
            combined_score = (self.total_match_weight * len(filtered_matches) + self.keypoint_match_weight * successful_keypoints * 10)
            reference_scores[ref_id] = combined_score
            
            if self.debug: # Added debug print for successful scoring
                print(f"      Ref {ref_id}: Raw LightGlue matches: {raw_lightglue_matches_count}, RANSAC filtered: {len(filtered_matches)}, Score: {combined_score:.2f}")


        if not reference_scores or all(score == 0 for score in reference_scores.values()):
            if self.debug: # Added debug print for no positive scores
                print("      No references found with a score > 0.")
            return None, [], 'no_matches'
        
        max_score = max(reference_scores.values())
        tied_refs = [ref_id for ref_id, score in reference_scores.items() if abs(score - max_score) <= self.score_difference_threshold]
        tied_refs.sort(key=lambda r_id: reference_scores[r_id], reverse=True)
        tied_refs = tied_refs[:self.max_tied_refs]
        
        primary_ref_id = tied_refs[0]
        
        selection_method = 'clear_winner' if len(tied_refs) == 1 else 'tie_detected'
        return primary_ref_id, tied_refs, selection_method

    def _upscale_region_for_refinement(self, image, bbox):
        """Upscale the bounding box region of an image for more precise detection."""
        height, width = image.shape[:2]
        x, y, w, h = [int(v) for v in bbox]

        # Padding calculation (using adjusted upscale_padding_ratio)
        padding_x, padding_y = int(w * self.upscale_padding_ratio), int(h * self.upscale_padding_ratio)
        ext_x1, ext_y1 = max(0, x - padding_x), max(0, y - padding_y)
        ext_x2, ext_y2 = min(width, x + w + padding_x), min(height, y + h + padding_y)

        # Extract the region (first BBox with padding)
        region = image[ext_y1:ext_y2, ext_x1:ext_x2]
        region_h, region_w = region.shape[:2]

        if region_w < self.min_upscale_size or region_h < self.min_upscale_size:
            return None, None

        # Upscaling for increased resolution
        upscaled_w = int(region_w * self.upscale_factor)
        upscaled_h = int(region_h * self.upscale_factor)
        upscaled_region = cv2.resize(region, (upscaled_w, upscaled_h), interpolation=cv2.INTER_CUBIC)
        
        transform_info = {
            'offset_x': ext_x1, 'offset_y': ext_y1, 'upscale_factor': self.upscale_factor
        }
        
        return upscaled_region, transform_info

    def _convert_upscaled_coordinates_to_original(self, keypoints, transform_info):
        """Convert keypoint coordinates from the upscaled region back to the original image frame."""
        if not transform_info:
            return keypoints
            
        converted_keypoints = {}
        factor = transform_info['upscale_factor']
        offset_x, offset_y = transform_info['offset_x'], transform_info['offset_y']
        
        for name, kp in keypoints.items():
            original_x = (kp['x'] / factor) + offset_x
            original_y = (kp['y'] / factor) + offset_y
            
            converted_kp = kp.copy()
            converted_kp.update({'x': float(original_x), 'y': float(original_y)})
            # Carry over enhancement info flags, they will be merged later
            converted_kp['enhancement_info'] = kp.get('enhancement_info', {})
            converted_keypoints[name] = converted_kp
            
        return converted_keypoints

    def _match_keypoints_with_reference(self, ref_id, input_image, distance_threshold=None):
        """Match keypoints between a specific reference and the input image."""
        dist_thresh = distance_threshold if distance_threshold is not None else self.distance_threshold
        ref_data = self.references[ref_id]
        
        input_features = self._extract_superpoint_features(input_image)
        with torch.no_grad():
            matches_dict = self.matcher({'image0': ref_data['superpoint_features'], 'image1': input_features})

        feats0, feats1, matches01 = [rbd(x) for x in [ref_data['superpoint_features'], input_features, matches_dict]]
        matches = matches01["matches"].detach().cpu().numpy()

        if len(matches) == 0:
            return {}, 0.0, []
            
        ref_kpts = feats0["keypoints"].detach().cpu().numpy()
        input_kpts = feats1["keypoints"].detach().cpu().numpy()
        filtered_matches, ransac_ratio = self._apply_enhanced_ransac_filtering(ref_kpts, input_kpts, matches)

        if len(filtered_matches) == 0:
            return {}, 0.0, []

        matched_ref_kpts = ref_kpts[filtered_matches[:, 0]]
        matched_input_kpts = input_kpts[filtered_matches[:, 1]]
        
        matched_object_kpts = {}
        for name in self.keypoint_names:
            coords = ref_data['keypoints'].get(name)
            if not coords or coords == [0, 0]:
                continue
            
            distances = cdist(np.array([coords]), matched_ref_kpts)
            if distances.size > 0:
                min_dist_idx = np.argmin(distances)
                min_dist = distances[0, min_dist_idx]
                
                if min_dist < dist_thresh:
                    matched_input_point = matched_input_kpts[min_dist_idx]
                    matched_object_kpts[name] = {
                        'x': float(matched_input_point[0]),
                        'y': float(matched_input_point[1]),
                        'confidence': float(1.0 - min_dist / dist_thresh),
                        'reference_used': ref_id,
                        'ransac_ratio': float(ransac_ratio)
                    }
        
        return matched_object_kpts, ransac_ratio, matched_input_kpts.tolist()

    def _try_tied_references_for_missing_keypoints(self, tied_ref_ids, input_image, initial_keypoints):
        """If keypoints are missing, try other high-scoring (tied) references to find them."""
        if not self.enable_multi_ref_tiebreaker or len(tied_ref_ids) <= 1:
            return initial_keypoints

        primary_ref_id = tied_ref_ids[0]
        other_tied_refs = tied_ref_ids[1:]
        
        primary_ref_visible = {name for name, coords in self.references[primary_ref_id]['keypoints'].items() if coords != [0,0]}
        missing_keypoints = primary_ref_visible - set(initial_keypoints.keys())
        
        if not missing_keypoints:
            return initial_keypoints

        recovered_keypoints = initial_keypoints.copy()

        for ref_id in other_tied_refs:
            if not missing_keypoints: break
            
            can_recover = missing_keypoints.intersection(
                name for name, coords in self.references[ref_id]['keypoints'].items() if coords != [0, 0]
            )
            if not can_recover: continue

            tied_keypoints, _, _ = self._match_keypoints_with_reference(ref_id, input_image)
            
            for name in can_recover:
                if name in tied_keypoints:
                    kp = tied_keypoints[name]
                    # Ensure enhancement_info is initialized
                    kp['enhancement_info'] = kp.get('enhancement_info', {})
                    kp['enhancement_info']['tied_recovery'] = True
                    recovered_keypoints[name] = kp
                    missing_keypoints.remove(name)

        return recovered_keypoints

    def _apply_contrast_enhancement(self, image):
        """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for robust contrast enhancement."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge the enhanced L-channel with the original A and B channels
        limg = cv2.merge((cl, a, b))
        # Convert back to BGR color space
        enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_image

    def _upscaling_refinement(self, input_image, current_keypoints, all_robust_matches):
        """Perform a refinement step by upscaling the object region and re-matching,
        with an additional contrast enhancement step if keypoints are still missing."""
        debug_info = {}
        if not self.enable_upscaling_refinement or not all_robust_matches:
            return current_keypoints, debug_info
        
        bbox_points = np.array(all_robust_matches)
        x_min, y_min = np.min(bbox_points, axis=0)
        x_max, y_max = np.max(bbox_points, axis=0)
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        debug_info['initial_bbox'] = bbox

        upscaled_image, transform_info = self._upscale_region_for_refinement(input_image, bbox)
        if upscaled_image is None:
            return current_keypoints, debug_info
            
        # Initial attempt with upscaled image
        best_ref_id_upscale, _, _ = self._find_best_references_with_tiebreaker(upscaled_image)
        upscaled_keypoints = {}
        if best_ref_id_upscale:
            upscaled_keypoints, _, _ = self._match_keypoints_with_reference(
                best_ref_id_upscale, upscaled_image, self.upscale_distance_threshold
            )
        
        # Merge current keypoints with those found from initial upscaling
        combined_keypoints_after_upscale = current_keypoints.copy()
        for name, kp in upscaled_keypoints.items():
            # Ensure enhancement_info is initialized
            kp['enhancement_info'] = kp.get('enhancement_info', {})
            kp['enhancement_info']['upscale_refined'] = True # Mark as upscaled
            
            if name not in combined_keypoints_after_upscale or \
               kp['confidence'] > combined_keypoints_after_upscale[name].get('confidence', 0.0):
                combined_keypoints_after_upscale[name] = kp
        
        # Determine which keypoints are still missing after initial upscale attempt
        all_possible_keypoints = set(self.keypoint_names) 
        missing_after_upscale = all_possible_keypoints - set(combined_keypoints_after_upscale.keys())

        # --- New: Contrast Enhancement for Missing Keypoints ---
        if missing_after_upscale and upscaled_image is not None:
            if self.debug:
                print(f"      Debug: Missing keypoints after initial upscale: {missing_after_upscale}. Attempting contrast enhancement.")
            
            enhanced_upscaled_image = self._apply_contrast_enhancement(upscaled_image)
            debug_info['enhanced_upscaled_image'] = enhanced_upscaled_image # For debugging visualization

            best_ref_id_enhanced, _, _ = self._find_best_references_with_tiebreaker(enhanced_upscaled_image)
            enhanced_keypoints_from_ce = {} 
            if best_ref_id_enhanced:
                enhanced_keypoints_from_ce, _, _ = self._match_keypoints_with_reference(
                    best_ref_id_enhanced, enhanced_upscaled_image, self.upscale_distance_threshold
                )
                if enhanced_keypoints_from_ce:
                    if self.debug:
                        print(f"      Debug: Found {len(enhanced_keypoints_from_ce)} keypoints after contrast enhancement.")

                    # Merge newly found keypoints from CE
                    for name, kp in enhanced_keypoints_from_ce.items():
                        # Ensure enhancement_info is initialized
                        kp['enhancement_info'] = kp.get('enhancement_info', {})
                        kp['enhancement_info']['contrast_enhanced_recovery'] = True 
                        kp['enhancement_info']['upscale_refined'] = True # If CE-recovered, it's also implicitly upscaled
                        
                        if name in missing_after_upscale or \
                           (name in combined_keypoints_after_upscale and kp['confidence'] > combined_keypoints_after_upscale[name].get('confidence', 0.0)):
                            combined_keypoints_after_upscale[name] = kp
                            if name in missing_after_upscale: 
                                missing_after_upscale.remove(name)
            
            # Store the keypoints found specifically on the enhanced image for debug visualization
            debug_info['enhanced_upscaled_keypoints'] = enhanced_keypoints_from_ce


        debug_info.update({
            'upscaled_image': upscaled_image, 
            'upscaled_keypoints': upscaled_keypoints # Keypoints from first upscale attempt
        })

        if not combined_keypoints_after_upscale:
            return current_keypoints, debug_info 
            
        final_keypoints = self._convert_upscaled_coordinates_to_original(combined_keypoints_after_upscale, transform_info)
        
        # Ensure enhancement_info is propagated correctly through the conversion
        # The enhancement_info is now carried within each keypoint dictionary
            
        return final_keypoints, debug_info

    def _visualize_enhanced_keypoints(self, image_path, matched_keypoints, best_ref_id, all_robust_matches, debug_info=None):
        """Create and save visualizations of the annotation results, including debug images."""
        try:
            input_image = cv2.imread(str(image_path))
            if input_image is None: return
        except Exception as e:
            print(f"   ⚠️ Visualization failed for {Path(image_path).name}: {e}")
            return
            
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        colors = [cv2.cvtColor(np.uint8([[[int(180 * i / len(self.keypoint_names)), 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0] for i in range(len(self.keypoint_names))]
        color_map = {name: tuple(map(int, color)) for i, (name, color) in enumerate(zip(self.keypoint_names, colors))}

        # --- Main Visualization ---
        output_image_main = input_image.copy()
        if all_robust_matches:
            for x, y in all_robust_matches:
                cv2.circle(output_image_main, (int(x), int(y)), 2, (200, 200, 200), -1) # Gray for raw matches

        if matched_keypoints:
            for name, kp in matched_keypoints.items():
                x, y = int(kp['x']), int(kp['y'])
                color = color_map.get(name, (128, 128, 128))
                
                # Use different markers/colors to denote enhancement origin
                enh_info = kp.get('enhancement_info', {})
                if enh_info.get('contrast_enhanced_recovery', False):
                    cv2.drawMarker(output_image_main, (x, y), color, cv2.MARKER_STAR, 20, 2) # Star for CE
                    cv2.putText(output_image_main, f"{name} (CE)", (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif enh_info.get('upscale_refined', False):
                    cv2.rectangle(output_image_main, (x - 8, y - 8), (x + 8, y + 8), color, -1) # Square for Upscaled
                    cv2.putText(output_image_main, f"{name} (UP)", (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif enh_info.get('tied_recovery', False):
                    pts = np.array([[x, y - 10], [x - 8, y + 8], [x + 8, y + 8]], np.int32)
                    cv2.fillPoly(output_image_main, [pts], color) # Triangle for Tied
                    cv2.putText(output_image_main, f"{name} (TR)", (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else: # Initial detection
                    cv2.circle(output_image_main, (x, y), 8, color, -1) # Circle for Initial
                    cv2.putText(output_image_main, f"{name}", (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        vis_path = vis_dir / f"vis_{Path(image_path).name}"
        cv2.imwrite(str(vis_path), output_image_main)

        # --- DEBUG VISUALIZATIONS (only if --debug is enabled) ---
        if self.debug and debug_info:
            # 1. Visualize the initial bounding box for upscaling
            if 'initial_bbox' in debug_info:
                output_image_bbox = input_image.copy()
                if all_robust_matches:
                    for x, y in all_robust_matches:
                        cv2.circle(output_image_bbox, (int(x), int(y)), 2, (200, 200, 200), -1) # Gray
                
                x, y, w, h = [int(v) for v in debug_info['initial_bbox']]
                cv2.rectangle(output_image_bbox, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue
                cv2.putText(output_image_bbox, "Upscaling BBox", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                debug_path = vis_dir / f"debug_bbox_{Path(image_path).name}"
                cv2.imwrite(str(debug_path), output_image_bbox)

            # 2. Visualize the results from the upscaled image (before CE)
            if 'upscaled_image' in debug_info and 'upscaled_keypoints' in debug_info:
                output_image_upscale = debug_info['upscaled_image'].copy()
                upscaled_kpts = debug_info['upscaled_keypoints']
                
                for name, kp in upscaled_kpts.items():
                    x, y = int(kp['x']), int(kp['y'])
                    color = color_map.get(name, (128, 128, 128))
                    cv2.circle(output_image_upscale, (x, y), 8, color, -1)
                    cv2.putText(output_image_upscale, f"{name} (UP)", (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                debug_path = vis_dir / f"debug_upscale_{Path(image_path).name}" 
                cv2.imwrite(str(debug_path), output_image_upscale)

            # 3. Visualize the results from the contrast-enhanced upscaled image
            if 'enhanced_upscaled_image' in debug_info and 'enhanced_upscaled_keypoints' in debug_info:
                output_image_ce_upscale = debug_info['enhanced_upscaled_image'].copy()
                enhanced_kpts_on_upscale_frame = debug_info['enhanced_upscaled_keypoints']
                
                for name, kp in enhanced_kpts_on_upscale_frame.items():
                    x, y = int(kp['x']), int(kp['y'])
                    color = color_map.get(name, (128, 128, 128))
                    cv2.drawMarker(output_image_ce_upscale, (x, y), color, cv2.MARKER_STAR, 20, 2) 
                    cv2.putText(output_image_ce_upscale, f"{name} (CE)", (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                debug_path = vis_dir / f"debug_ce_upscale_{Path(image_path).name}"
                cv2.imwrite(str(debug_path), output_image_ce_upscale)


    def find_matching_keypoints(self, input_image_path):
        """Main pipeline to find keypoints in an image using multiple enhancement steps."""
        input_image = cv2.imread(str(input_image_path))
        if input_image is None:
            print(f"   ⚠️ Could not load image: {input_image_path.name}")
            return {}, input_image.shape if input_image is not None else (0,0,0), None, [], None

        best_ref_id, tied_ref_ids, selection_method = self._find_best_references_with_tiebreaker(input_image)
        if best_ref_id is None:
            if self.debug:
                print(f"   Debug: No suitable reference found for {input_image_path.name}.")
            return {}, input_image.shape, None, [], None

        initial_keypoints, _, all_robust_matches = self._match_keypoints_with_reference(best_ref_id, input_image)
        
        # Populate initial enhancement info for keypoints found in the primary match
        for name, kp in initial_keypoints.items():
            kp['enhancement_info'] = {'initial_detection': True, 'tied_recovery': False, 'upscale_refined': False, 'contrast_enhanced_recovery': False}

        # This step tries to recover missing ones from other tied references
        # The 'tied_recovery' flag will be set inside _try_tied_references_for_missing_keypoints if successful.
        keypoints_after_tied_recovery = self._try_tied_references_for_missing_keypoints(tied_ref_ids, input_image, initial_keypoints)
        
        # This step takes `keypoints_after_tied_recovery` as its base and attempts upscaling refinement
        # and potentially contrast enhancement within _upscaling_refinement.
        # The enhancement_info flags for 'upscale_refined' and 'contrast_enhanced_recovery' will be set here.
        final_keypoints, upscale_debug_info = self._upscaling_refinement(input_image, keypoints_after_tied_recovery, all_robust_matches)
        
        # Ensure all enhancement info flags are correctly propagated and merged into the final keypoints
        # The flags are already set within the keypoint dictionaries by the refinement functions.
        
        return final_keypoints, input_image.shape, best_ref_id, all_robust_matches, upscale_debug_info

    def _init_coco_dataset(self):
        """Initialize the structure for COCO format annotations."""
        self.coco_dataset = {
            "info": {}, "licenses": [],
            "categories": [{"id": 1, "name": self.object_name, "supercategory": "object", "keypoints": self.keypoint_names, "skeleton": []}],
            "images": [], "annotations": []
        }
        self.image_id = 1
        self.annotation_id = 1

    def create_coco_annotation(self, image_path, matched_keypoints):
        """Create and add a COCO-formatted annotation for a single image."""
        if len(matched_keypoints) < self.min_keypoints_threshold:
            return False

        height, width, _ = cv2.imread(str(image_path)).shape
        
        self.coco_dataset["images"].append({"id": self.image_id, "width": width, "height": height, "file_name": Path(image_path).name})
        
        keypoints_coco = []
        for kp_name in self.keypoint_names:
            kp = matched_keypoints.get(kp_name)
            # COCO format: [x, y, visibility]. Visibility 2=visible, 1=occluded, 0=not labeled
            keypoints_coco.extend([kp['x'], kp['y'], 2] if kp else [0, 0, 0])

        bbox_points = np.array([[kp['x'], kp['y']] for kp in matched_keypoints.values()])
        x_min, y_min = np.min(bbox_points, axis=0)
        x_max, y_max = np.max(bbox_points, axis=0)
        bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

        self.coco_dataset["annotations"].append({
            "id": self.annotation_id, "image_id": self.image_id, "category_id": 1,
            "keypoints": keypoints_coco, "num_keypoints": len(matched_keypoints),
            "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0
        })
        
        self.image_id += 1
        self.annotation_id += 1
        return True

    def process_images(self, input_folder, image_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """Process a folder of images to generate keypoint annotations."""
        input_path = Path(input_folder)
        image_files = [p for ext in image_extensions for p in input_path.glob(f"*{ext}")]
        if not image_files:
            print(f"❌ No images found in {input_folder}")
            return
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n📸 Processing {i}/{len(image_files)}: {image_path.name}")
            try:
                matched_keypoints, _, best_ref_id, all_robust_matches, debug_info = self.find_matching_keypoints(image_path)
                
                if best_ref_id:
                    self.create_coco_annotation(image_path, matched_keypoints)
                
                if self.visualize and best_ref_id:
                    self._visualize_enhanced_keypoints(image_path, matched_keypoints, best_ref_id, all_robust_matches, debug_info)

            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

    def save_annotations(self, filename=None):
        """Save the generated COCO annotations to a JSON file."""
        if not self.coco_dataset["images"]:
            print("⚠️ No annotations generated, skipping save.")
            return

        filename = filename or f"annotations_{self.object_name}.json"
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.coco_dataset, f, indent=4)
        print(f"\n✅ Annotations saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Auto Annotator with Debugging and Contrast Enhancement.')
    parser.add_argument('--reference-config', required=True, help='Path to the reference config JSON.')
    parser.add_argument('--input-folder', required=True, help='Folder with input images.')
    parser.add_argument('--output-dir', default='annotations', help='Output directory.')
    parser.add_argument('--object-name', default='object', help='Object category name.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Inference device.')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug logging and visualizations.')
    parser.add_argument('--visualize', action='store_true', help='Save visualization images.')
    args = parser.parse_args()
    
    try:
        annotator = EnhancedMultiRefAnnotator(
            reference_config_path=args.reference_config,
            output_dir=args.output_dir,
            device=args.device,
            object_name=args.object_name,
            debug=args.debug
        )
        annotator.visualize = args.visualize # Set the visualize flag from args
        annotator.process_images(args.input_folder)
        annotator.save_annotations()
        print("\n🎉 Annotation task finished!")
        
    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
