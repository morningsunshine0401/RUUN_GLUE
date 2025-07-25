
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

class EnhancedMultiRefAnnotator:
    """
    ENHANCED multi-reference annotator with tolerance zones, smart thresholds, and comprehensive statistics.
    """
    def __init__(self, reference_config_path, output_dir="annotations", device='auto', object_name="aircraft", debug=False,
                 # Tolerance zone parameters (NEW)
                 keypoint_tolerance_radius=8.0,
                 # Fallback Strategy Controls
                 enable_sift_fallback=False, sift_edge_threshold=10,
                 enable_orb_fallback=False,
                 enable_patch_fallback=False, patch_size=32,
                 enable_image_enhancement_fallback=False, image_enhancement_type='grayscale',
                 # Bounding Box Strategy (NEW)
                 bbox_strategy='keypoint'): # 'keypoint' or 'total_matches'
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.object_name = object_name
        self.debug = debug

        # --- NEW: Tolerance Zone Configuration ---
        self.keypoint_tolerance_radius = keypoint_tolerance_radius
        print(f"🎯 Keypoint Tolerance Radius: {keypoint_tolerance_radius} pixels")

        # --- Fallback Strategy Configuration ---
        print("--- Fallback Strategy Configuration ---")
        self.enable_sift_fallback = enable_sift_fallback
        self.sift_edge_threshold = sift_edge_threshold
        self.enable_orb_fallback = enable_orb_fallback
        self.enable_patch_fallback = enable_patch_fallback
        self.patch_size = patch_size
        self.enable_image_enhancement_fallback = enable_image_enhancement_fallback
        self.image_enhancement_type = image_enhancement_type

        if enable_sift_fallback: print(f"  - SIFT Fallback: Enabled (Edge Threshold: {sift_edge_threshold})")
        if enable_orb_fallback: print("  - ORB Fallback: Enabled")
        if enable_patch_fallback: print(f"  - Patch Matching Fallback: Enabled (Patch Size: {patch_size}x{patch_size})")
        if enable_image_enhancement_fallback: print(f"  - Image Enhancement Fallback: Enabled (Type: {image_enhancement_type})")
        print("-----------------------------------------")

        # --- NEW: Bounding Box Strategy ---
        self.bbox_strategy = bbox_strategy
        print(f"📦 Bounding Box Strategy: {bbox_strategy}")
        print("-----------------------------------------")

        # ENHANCED RANSAC parameters
        self.ransac_threshold = 6
        self.ransac_max_trials = 4000
        self.min_matches_for_ransac = 6
        self.ransac_min_samples = 6

        # Keypoint matching parameters
        self.distance_threshold = 8
        self.min_keypoints_threshold = 4

        # Multi-reference parameters
        self.enable_multi_ref_tiebreaker = True
        self.score_difference_threshold = 2.0
        self.max_tied_refs = 3

        # Upscaling refinement parameters
        self.enable_upscaling_refinement = True
        self.upscale_factor = 2.0
        self.upscale_padding_ratio = 0.3
        self.min_upscale_size = 100
        self.upscale_distance_threshold = 12.0

        # Scoring parameters for reference selection
        self.total_match_weight = 0.6
        self.keypoint_match_weight = 0.4

        # --- NEW: Statistics tracking ---
        self.stats = {
            'total_images_processed': 0,
            'images_with_annotations': 0,
            'images_failed': 0,
            'total_keypoints_found': 0,
            'keypoints_by_method': {},
            'references_used': {},
            'fallback_usage': {
                'sift': 0, 'orb': 0, 'patch': 0, 'image_enhancement': 0, 
                'tied_recovery': 0, 'upscale_refinement': 0
            }
        }

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
        """Initialize all AI and traditional models based on strategy configuration."""
        # Primary SuperPoint & LightGlue
        self.extractor = SuperPoint(max_num_keypoints=4096*2).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        print("✅ SuperPoint & LightGlue loaded!")

        # Traditional detectors and matcher for fallbacks
        if self.enable_sift_fallback or self.enable_patch_fallback:
            self.sift = cv2.SIFT_create(edgeThreshold=self.sift_edge_threshold)
            print(f"✅ SIFT detector initialized (edgeThreshold={self.sift_edge_threshold})")
        else:
            self.sift = None

        if self.enable_orb_fallback:
            self.orb = cv2.ORB_create(nfeatures=4096)
            print("✅ ORB detector initialized")
        else:
            self.orb = None
        
        if self.enable_sift_fallback or self.enable_orb_fallback or self.enable_patch_fallback or self.enable_image_enhancement_fallback:
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
            print("✅ BFMatcher for traditional methods initialized")
        else:
            self.bf_matcher = None

    def _load_reference_config(self, config_path):
        """Load reference images and their keypoints from a JSON config file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.references = {}
        self.num_keypoints = config.get('num_keypoints', 8)
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
                'bbox': ref_config.get('bbox'),
                'superpoint_features': self._extract_superpoint_features(ref_image, ref_config.get('bbox')),
                'visible_keypoints_count': visible_keypoints_count
            }
            
            # Initialize reference usage tracking
            self.stats['references_used'][ref_id] = 0
            
        print(f"✅ Loaded {len(self.references)} reference images successfully.")

    def _extract_superpoint_features(self, image, bbox=None):
        """Extract SuperPoint features from an image with optional bbox cropping."""
        input_image_for_extractor = image
        crop_offset = (0, 0)

        if bbox is not None and len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
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

            raw_lightglue_matches_count = len(matches)

            if raw_lightglue_matches_count == 0:
                reference_scores[ref_id] = 0
                if self.debug:
                    print(f"      Ref {ref_id}: No raw LightGlue matches found.")
                continue
            
            ref_kpts = feats0["keypoints"].detach().cpu().numpy()
            input_kpts = feats1["keypoints"].detach().cpu().numpy()
            
            filtered_matches, _ = self._apply_enhanced_ransac_filtering(ref_kpts, input_kpts, matches)
            
            if len(filtered_matches) == 0:
                reference_scores[ref_id] = 0
                if self.debug:
                    print(f"      Ref {ref_id}: Raw LightGlue matches: {raw_lightglue_matches_count}, RANSAC filtered: {len(filtered_matches)} (all filtered out).")
                continue
            
            # NEW: Use tolerance zone matching for scoring
            matched_ref_kpts = ref_kpts[filtered_matches[:, 0]]
            successful_keypoints = 0
            for name in self.keypoint_names:
                coords = ref_data['keypoints'].get(name)
                if not coords or coords == [0, 0]:
                    continue
                
                distances = cdist(np.array([coords]), matched_ref_kpts)
                if len(distances) > 0 and np.min(distances) < self.keypoint_tolerance_radius:
                    successful_keypoints += 1
            
            combined_score = (self.total_match_weight * len(filtered_matches) + self.keypoint_match_weight * successful_keypoints * 10)
            reference_scores[ref_id] = combined_score
            
            if self.debug:
                print(f"      Ref {ref_id}: Raw LightGlue matches: {raw_lightglue_matches_count}, RANSAC filtered: {len(filtered_matches)}, Score: {combined_score:.2f}")

        if not reference_scores or all(score == 0 for score in reference_scores.values()):
            if self.debug:
                print("      No references found with a score > 0.")
            return None, [], 'no_matches'
        
        max_score = max(reference_scores.values())
        tied_refs = [ref_id for ref_id, score in reference_scores.items() if abs(score - max_score) <= self.score_difference_threshold]
        tied_refs.sort(key=lambda r_id: reference_scores[r_id], reverse=True)
        tied_refs = tied_refs[:self.max_tied_refs]
        
        primary_ref_id = tied_refs[0]
        
        selection_method = 'clear_winner' if len(tied_refs) == 1 else 'tie_detected'
        return primary_ref_id, tied_refs, selection_method

    def _match_keypoints_with_tolerance_zones(self, ref_id, input_image, distance_threshold=None):
        """
        NEW: Enhanced keypoint matching using tolerance zones around reference keypoints.
        This allows for more robust matching by considering areas around reference points.
        """
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
        
        # NEW: Tolerance zone matching
        matched_object_kpts = {}
        
        # Get all input SuperPoint keypoints for tolerance zone matching
        all_input_kpts = input_kpts
        
        # For each reference keypoint, find the best match within tolerance zone
        for name in self.keypoint_names:
            coords = ref_data['keypoints'].get(name)
            if not coords or coords == [0, 0]:
                continue
            
            # Method 1: Check matches first (more reliable due to feature matching)
            distances_to_matches = cdist(np.array([coords]), matched_ref_kpts)
            if distances_to_matches.size > 0:
                min_dist_idx = np.argmin(distances_to_matches)
                min_dist = distances_to_matches[0, min_dist_idx]
                
                if min_dist < self.keypoint_tolerance_radius:
                    matched_input_point = matched_input_kpts[min_dist_idx]
                    confidence = float(1.0 - min_dist / self.keypoint_tolerance_radius)
                    
                    matched_object_kpts[name] = {
                        'x': float(matched_input_point[0]),
                        'y': float(matched_input_point[1]),
                        'confidence': confidence,
                        'reference_used': ref_id,
                        'ransac_ratio': float(ransac_ratio),
                        'detection_method': 'SuperPoint_Tolerance',
                        'tolerance_distance': float(min_dist)
                    }
                    continue
            
            # Method 2: If no good match found, try to estimate position using transformation
            # This is more experimental and can be enabled if needed
            if len(filtered_matches) >= 4:  # Need minimum points for transformation
                try:
                    # Estimate where this keypoint should appear in the input image
                    src_pts = matched_ref_kpts.reshape(-1, 1, 2).astype(np.float32)
                    dst_pts = matched_input_kpts.reshape(-1, 1, 2).astype(np.float32)
                    
                    # Use affine transformation (more stable than homography for small point sets)
                    if len(filtered_matches) >= 3:
                        M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
                        if M is not None:
                            # Transform the reference keypoint to input image space
                            ref_pt = np.array([[[coords[0], coords[1]]]], dtype=np.float32)
                            transformed_pt = cv2.transform(ref_pt, M)[0][0]
                            
                            # Find closest input keypoint to the estimated position
                            distances_to_all = cdist(np.array([transformed_pt]), all_input_kpts)
                            if distances_to_all.size > 0:
                                closest_idx = np.argmin(distances_to_all)
                                closest_dist = distances_to_all[0, closest_idx]
                                
                                if closest_dist < self.keypoint_tolerance_radius:
                                    closest_input_point = all_input_kpts[closest_idx]
                                    confidence = float(0.5 * (1.0 - closest_dist / self.keypoint_tolerance_radius))  # Lower confidence
                                    
                                    matched_object_kpts[name] = {
                                        'x': float(closest_input_point[0]),
                                        'y': float(closest_input_point[1]),
                                        'confidence': confidence,
                                        'reference_used': ref_id,
                                        'ransac_ratio': float(ransac_ratio),
                                        'detection_method': 'SuperPoint_Transformed',
                                        'tolerance_distance': float(closest_dist)
                                    }
                except Exception as e:
                    if self.debug:
                        print(f"        Failed to estimate position for {name}: {e}")
                    continue
        
        return matched_object_kpts, ransac_ratio, matched_input_kpts.tolist()

    def _match_keypoints_with_reference(self, ref_id, input_image, distance_threshold=None):
        """Wrapper that uses the new tolerance zone matching method."""
        return self._match_keypoints_with_tolerance_zones(ref_id, input_image, distance_threshold)

    def _upscale_region_for_refinement(self, image, bbox):
        """Upscale the bounding box region of an image for more precise detection."""
        height, width = image.shape[:2]
        x, y, w, h = [int(v) for v in bbox]

        padding_x, padding_y = int(w * self.upscale_padding_ratio), int(h * self.upscale_padding_ratio)
        ext_x1, ext_y1 = max(0, x - padding_x), max(0, y - padding_y)
        ext_x2, ext_y2 = min(width, x + w + padding_x), min(height, y + h + padding_y)

        region = image[ext_y1:ext_y2, ext_x1:ext_x2]
        region_h, region_w = region.shape[:2]

        if region_w < self.min_upscale_size or region_h < self.min_upscale_size:
            return None, None

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
            converted_kp.update({'x': float(original_x), 'y': float(original_y), 'upscale_refined': True})
            converted_keypoints[name] = converted_kp
            
        return converted_keypoints

    def _upscaling_refinement(self, input_image, initial_keypoints, all_robust_matches):
        """Perform a refinement step by upscaling the object region and re-matching."""
        debug_info = {}
        if not self.enable_upscaling_refinement or not all_robust_matches:
            return initial_keypoints, debug_info
        
        bbox_points = np.array(all_robust_matches)
        x_min, y_min = np.min(bbox_points, axis=0)
        x_max, y_max = np.max(bbox_points, axis=0)
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        debug_info['initial_bbox'] = bbox

        upscaled_image, transform_info = self._upscale_region_for_refinement(input_image, bbox)
        if upscaled_image is None:
            return initial_keypoints, debug_info
            
        best_ref_id, _, _ = self._find_best_references_with_tiebreaker(upscaled_image) 
        if best_ref_id is None:
            return initial_keypoints, debug_info

        upscaled_keypoints, _, _ = self._match_keypoints_with_reference(
            best_ref_id, upscaled_image, self.upscale_distance_threshold
        )
        debug_info.update({
            'upscaled_image': upscaled_image,
            'upscaled_keypoints': upscaled_keypoints
        })

        if not upscaled_keypoints:
            return initial_keypoints, debug_info

        refined_keypoints = self._convert_upscaled_coordinates_to_original(upscaled_keypoints, transform_info)
        
        final_keypoints = initial_keypoints.copy()
        refined_count = 0
        for name, kp in refined_keypoints.items():
            if name not in final_keypoints or kp['confidence'] > final_keypoints[name].get('confidence', 0.0):
                final_keypoints[name] = kp
                refined_count += 1
        
        # Update statistics
        if refined_count > 0:
            self.stats['fallback_usage']['upscale_refinement'] += 1
        
        return final_keypoints, debug_info

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
        recovered_count = 0

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
                    kp['tied_recovery'] = True
                    recovered_keypoints[name] = kp
                    missing_keypoints.remove(name)
                    recovered_count += 1

        # Update statistics
        if recovered_count > 0:
            self.stats['fallback_usage']['tied_recovery'] += 1

        return recovered_keypoints

    def _get_missing_keypoints(self, found_keypoints, ref_id):
        """Helper to identify keypoints that are visible in the reference but not yet found."""
        ref_data = self.references.get(ref_id)
        if not ref_data: return set()
        ref_visible_kps = {name for name, coords in ref_data['keypoints'].items() if coords != [0, 0]}
        return ref_visible_kps - set(found_keypoints.keys())

    def _find_keypoints_with_traditional_method(self, ref_data, input_image, missing_keypoints, method='SIFT'):
        """FALLBACK: Use SIFT or ORB to find missing keypoints."""
        if method == 'SIFT' and self.sift: detector = self.sift
        elif method == 'ORB' and self.orb: detector = self.orb
        else: return {}, []

        ref_img_gray = cv2.cvtColor(ref_data['image'], cv2.COLOR_BGR2GRAY)
        input_img_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        kp1, des1 = detector.detectAndCompute(ref_img_gray, None)
        kp2, des2 = detector.detectAndCompute(input_img_gray, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2: return {}, []

        matches = self.bf_matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) < self.min_matches_for_ransac: return {}, []

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None: return {}, []
        
        inlier_matches = [m for m, inlier in zip(good_matches, mask.ravel()) if inlier]
        
        found_keypoints = {}
        for name in missing_keypoints:
            ref_kp_coords = ref_data['keypoints'].get(name)
            if not ref_kp_coords or ref_kp_coords == [0, 0]: continue

            ref_cv_kps = np.array([kp1[m.queryIdx].pt for m in inlier_matches])
            distances = cdist(np.array([ref_kp_coords]), ref_cv_kps)
            
            if distances.size > 0 and np.min(distances) < self.keypoint_tolerance_radius:  # Use tolerance radius
                match_idx = np.argmin(distances)
                best_match = inlier_matches[match_idx]
                input_kp = kp2[best_match.trainIdx]
                found_keypoints[name] = {
                    'x': float(input_kp.pt[0]), 'y': float(input_kp.pt[1]),
                    'confidence': 1.0, 'detection_method': f'{method}_Fallback'
                }
        
        robust_input_kps = [kp2[m.trainIdx].pt for m in inlier_matches]
        return found_keypoints, robust_input_kps

    def _find_keypoints_by_patch_matching(self, ref_data, input_image, missing_keypoints):
        """FALLBACK: Use direct patch matching for specific, hard-to-find keypoints."""
        if not self.sift: return {}
        
        input_img_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        input_kps, input_des = self.sift.detectAndCompute(input_img_gray, None)
        if input_des is None: return {}

        found_keypoints = {}
        for name in missing_keypoints:
            ref_kp_coords = ref_data['keypoints'].get(name)
            if not ref_kp_coords or ref_kp_coords == [0, 0]: continue

            x, y = int(ref_kp_coords[0]), int(ref_kp_coords[1])
            h, w = ref_data['image'].shape[:2]
            ps = self.patch_size // 2
            
            y1, y2 = max(0, y - ps), min(h, y + ps)
            x1, x2 = max(0, x - ps), min(w, x + ps)
            if y1 >= y2 or x1 >= x2: continue

            patch = ref_data['image'][y1:y2, x1:x2]
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            
            _, patch_des = self.sift.detectAndCompute(patch_gray, None)
            if patch_des is None: continue

            matches = self.bf_matcher.knnMatch(patch_des, input_des, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if good_matches:
                best_match = sorted(good_matches, key=lambda x: x.distance)[0]
                matched_kp = input_kps[best_match.trainIdx]
                found_keypoints[name] = {
                    'x': float(matched_kp.pt[0]), 'y': float(matched_kp.pt[1]),
                    'confidence': 1.0, 'detection_method': 'Patch_Fallback'
                }
        return found_keypoints

    def _find_keypoints_with_image_enhancement(self, ref_data, input_image, missing_keypoints, enhancement_type):
        """FALLBACK: Apply image enhancement and re-attempt SuperPoint matching."""
        enhanced_image = input_image.copy()
        if enhancement_type == 'grayscale' or enhancement_type == 'both':
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
        
        if enhancement_type == 'clahe' or enhancement_type == 'both':
            if len(enhanced_image.shape) == 3:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                ycrcb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2YCrCb)
                ycrcb[:,:,0] = clahe.apply(ycrcb[:,:,0])
                enhanced_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced_image = clahe.apply(enhanced_image)
                enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

        input_features_enhanced = self._extract_superpoint_features(enhanced_image)
        
        with torch.no_grad():
            matches_dict = self.matcher({'image0': ref_data['superpoint_features'], 'image1': input_features_enhanced})

        feats0, feats1, matches01 = [rbd(x) for x in [ref_data['superpoint_features'], input_features_enhanced, matches_dict]]
        matches = matches01["matches"].detach().cpu().numpy()

        if len(matches) == 0: return {}

        ref_kpts = feats0["keypoints"].detach().cpu().numpy()
        input_kpts_enhanced = feats1["keypoints"].detach().cpu().numpy()
        filtered_matches, _ = self._apply_enhanced_ransac_filtering(ref_kpts, input_kpts_enhanced, matches)

        if len(filtered_matches) == 0: return {}

        matched_ref_kpts = ref_kpts[filtered_matches[:, 0]]
        matched_input_kpts_enhanced = input_kpts_enhanced[filtered_matches[:, 1]]
        
        found_keypoints = {}
        for name in missing_keypoints:
            coords = ref_data['keypoints'].get(name)
            if not coords or coords == [0, 0]:
                continue
            
            distances = cdist(np.array([coords]), matched_ref_kpts)
            if distances.size > 0:
                min_dist_idx = np.argmin(distances)
                min_dist = distances[0, min_dist_idx]
                
                if min_dist < self.keypoint_tolerance_radius:  # Use tolerance radius
                    matched_input_point = matched_input_kpts_enhanced[min_dist_idx]
                    found_keypoints[name] = {
                        'x': float(matched_input_point[0]),
                        'y': float(matched_input_point[1]),
                        'confidence': float(1.0 - min_dist / self.keypoint_tolerance_radius),
                        'detection_method': f'ImageEnhancement_{enhancement_type}'
                    }
        return found_keypoints

    def _update_method_statistics(self, keypoints):
        """Update statistics based on detection methods used."""
        for name, kp in keypoints.items():
            method = kp.get('detection_method', 'Unknown')
            if method not in self.stats['keypoints_by_method']:
                self.stats['keypoints_by_method'][method] = 0
            self.stats['keypoints_by_method'][method] += 1
            
            # Update fallback usage statistics
            if 'SIFT' in method:
                self.stats['fallback_usage']['sift'] += 1
            elif 'ORB' in method:
                self.stats['fallback_usage']['orb'] += 1
            elif 'Patch' in method:
                self.stats['fallback_usage']['patch'] += 1
            elif 'ImageEnhancement' in method:
                self.stats['fallback_usage']['image_enhancement'] += 1

    def _visualize_enhanced_keypoints(self, image_path, matched_keypoints, all_robust_matches, debug_info=None):
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

        output_image_main = input_image.copy()
        if all_robust_matches:
            for x, y in all_robust_matches:
                cv2.circle(output_image_main, (int(x), int(y)), 2, (200, 200, 200), -1)

        if matched_keypoints:
            for name, kp in matched_keypoints.items():
                x, y = int(kp['x']), int(kp['y'])
                color = color_map.get(name, (128, 128, 128))
                method = kp.get('detection_method', 'Unknown')
                tolerance_dist = kp.get('tolerance_distance', 0)
                
                # Different markers based on detection method and refinement
                if kp.get('upscale_refined', False):
                    cv2.rectangle(output_image_main, (x - 8, y - 8), (x + 8, y + 8), color, -1)
                elif kp.get('tied_recovery', False):
                    pts = np.array([[x, y - 10], [x - 8, y + 8], [x + 8, y + 8]], np.int32)
                    cv2.fillPoly(output_image_main, [pts], color)
                elif 'Fallback' in method:
                    if 'Patch' in method:
                        cv2.drawMarker(output_image_main, (x, y), color, markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
                    elif 'SIFT' in method or 'ORB' in method:
                        cv2.drawMarker(output_image_main, (x, y), color, markerType=cv2.MARKER_DIAMOND, markerSize=16, thickness=2)
                    elif 'ImageEnhancement' in method:
                        cv2.drawMarker(output_image_main, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
                else:
                    # SuperPoint methods - show tolerance zone
                    cv2.circle(output_image_main, (x, y), 8, color, -1)
                    if tolerance_dist > 0:
                        # Draw tolerance zone as a light circle
                        cv2.circle(output_image_main, (x, y), int(self.keypoint_tolerance_radius), color, 1)
                
                # Add text with confidence and tolerance distance
                conf_text = f"{name} ({kp.get('confidence', 0.0):.2f}"
                if tolerance_dist > 0:
                    conf_text += f", {tolerance_dist:.1f}px"
                conf_text += ")"
                cv2.putText(output_image_main, conf_text, (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        vis_path = vis_dir / f"vis_{Path(image_path).name}"
        cv2.imwrite(str(vis_path), output_image_main)

        # Debug visualizations
        if self.debug and debug_info:
            if 'initial_bbox' in debug_info:
                output_image_bbox = input_image.copy()
                if all_robust_matches:
                    for x, y in all_robust_matches:
                        cv2.circle(output_image_bbox, (int(x), int(y)), 2, (200, 200, 200), -1)
                
                x, y, w, h = [int(v) for v in debug_info['initial_bbox']]
                cv2.rectangle(output_image_bbox, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(output_image_bbox, "Upscaling BBox", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                debug_path = vis_dir / f"debug_bbox_{Path(image_path).name}"
                cv2.imwrite(str(debug_path), output_image_bbox)

            if 'upscaled_image' in debug_info and 'upscaled_keypoints' in debug_info:
                output_image_upscale = debug_info['upscaled_image'].copy()
                upscaled_kpts = debug_info['upscaled_keypoints']
                
                for name, kp in upscaled_kpts.items():
                    x, y = int(kp['x']), int(kp['y'])
                    color = color_map.get(name, (128, 128, 128))
                    cv2.circle(output_image_upscale, (x, y), 8, color, -1)
                    cv2.putText(output_image_upscale, f"{name}", (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                debug_path = vis_dir / f"debug_upscale_{Path(image_path).name}"
                cv2.imwrite(str(debug_path), output_image_upscale)

    def find_matching_keypoints(self, input_image_path):
        """Main pipeline to find keypoints in an image using multiple enhancement steps."""
        input_image = cv2.imread(str(input_image_path))
        if input_image is None:
            return None, None, None, None, None

        best_ref_id, tied_ref_ids, _ = self._find_best_references_with_tiebreaker(input_image)
        if best_ref_id is None:
            return {}, input_image.shape, None, [], None

        # Update reference usage statistics
        self.stats['references_used'][best_ref_id] += 1

        # Primary detection using SuperPoint with tolerance zones
        initial_keypoints, _, all_robust_matches = self._match_keypoints_with_reference(best_ref_id, input_image)
        
        # Fallback strategies for missing keypoints
        current_keypoints = initial_keypoints.copy()
        missing_kps = self._get_missing_keypoints(current_keypoints, best_ref_id)
        
        # Fallback 1: SIFT
        if missing_kps and self.enable_sift_fallback:
            sift_found, sift_matches = self._find_keypoints_with_traditional_method(self.references[best_ref_id], input_image, missing_kps, 'SIFT')
            current_keypoints.update(sift_found)
            all_robust_matches.extend(sift_matches)
            missing_kps = self._get_missing_keypoints(current_keypoints, best_ref_id)

        # Fallback 2: ORB
        if missing_kps and self.enable_orb_fallback:
            orb_found, orb_matches = self._find_keypoints_with_traditional_method(self.references[best_ref_id], input_image, missing_kps, 'ORB')
            current_keypoints.update(orb_found)
            all_robust_matches.extend(orb_matches)
            missing_kps = self._get_missing_keypoints(current_keypoints, best_ref_id)

        # Fallback 3: Patch Matching
        if missing_kps and self.enable_patch_fallback:
            patch_found = self._find_keypoints_by_patch_matching(self.references[best_ref_id], input_image, missing_kps)
            current_keypoints.update(patch_found)
            missing_kps = self._get_missing_keypoints(current_keypoints, best_ref_id)

        # Fallback 4: Image Enhancement
        if missing_kps and self.enable_image_enhancement_fallback:
            enhanced_found = self._find_keypoints_with_image_enhancement(self.references[best_ref_id], input_image, missing_kps, self.image_enhancement_type)
            current_keypoints.update(enhanced_found)

        # Tied references recovery
        enhanced_keypoints = self._try_tied_references_for_missing_keypoints(tied_ref_ids, input_image, current_keypoints)
        
        # Upscaling refinement
        final_keypoints, upscale_debug_info = self._upscaling_refinement(input_image, enhanced_keypoints, all_robust_matches)
        
        # Update enhancement info
        for name, kp in final_keypoints.items():
            kp['enhancement_info'] = {
                'initial_detection': name in initial_keypoints,
                'tied_recovery': kp.get('tied_recovery', False),
                'upscale_refined': kp.get('upscale_refined', False),
                'detection_method': kp.get('detection_method', 'Unknown')
            }
        
        # Update statistics
        self._update_method_statistics(final_keypoints)
        self.stats['total_keypoints_found'] += len(final_keypoints)
        
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

    def create_coco_annotation(self, image_path, matched_keypoints, ref_id, all_robust_matches):
        """
        NEW: Create and add a COCO-formatted annotation with smart threshold based on reference visibility.
        Updated to support bounding box calculation based on total match results.
        """
        # Get the reference's visible keypoint count
        ref_data = self.references.get(ref_id)
        if not ref_data:
            return False
            
        ref_visible_count = ref_data['visible_keypoints_count']
        
        # NEW: Smart threshold - allow n-1 keypoints if reference has n visible keypoints
        min_required = max(4, ref_visible_count - 2)  # At least 1, but preferably n-1
        
        if len(matched_keypoints) < min_required:
            if self.debug:
                print(f"   Skipping annotation: {len(matched_keypoints)} keypoints found, need at least {min_required} (ref has {ref_visible_count} visible)")
            return False

        height, width, _ = cv2.imread(str(image_path)).shape
        
        self.coco_dataset["images"].append({
            "id": self.image_id, 
            "width": width, 
            "height": height, 
            "file_name": Path(image_path).name,
            "reference_used": ref_id,
            "reference_visible_keypoints": ref_visible_count
        })
        
        keypoints_coco = []
        for kp_name in self.keypoint_names:
            kp = matched_keypoints.get(kp_name)
            keypoints_coco.extend([kp['x'], kp['y'], 2] if kp else [0, 0, 0])

        # --- NEW: Bounding Box Calculation based on strategy ---
        bbox_points = None
        if self.bbox_strategy == 'total_matches' and all_robust_matches:
            bbox_points = np.array(all_robust_matches)
            if self.debug:
                print(f"   Debug: Using 'total_matches' strategy for bbox. Points: {len(bbox_points)}")
        elif matched_keypoints:
            bbox_points = np.array([[kp['x'], kp['y']] for kp in matched_keypoints.values()])
            if self.debug:
                print(f"   Debug: Using 'keypoint' strategy for bbox. Points: {len(bbox_points)}")

        bbox = [0.0, 0.0, 0.0, 0.0] # Default empty bbox
        if bbox_points is not None and len(bbox_points) > 0:
            x_min, y_min = np.min(bbox_points, axis=0)
            x_max, y_max = np.max(bbox_points, axis=0)
            bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
        else:
            if self.debug:
                print("   Warning: Could not calculate bounding box, no valid points.")


        # NEW: Add enhanced annotation metadata
        annotation = {
            "id": self.annotation_id, 
            "image_id": self.image_id, 
            "category_id": 1,
            "keypoints": keypoints_coco, 
            "num_keypoints": len(matched_keypoints),
            "bbox": bbox, 
            "area": bbox[2] * bbox[3], 
            "iscrowd": 0,
            "reference_used": ref_id,
            "keypoint_methods": {name: kp.get('detection_method', 'Unknown') for name, kp in matched_keypoints.items()},
            "keypoint_confidences": {name: kp.get('confidence', 0.0) for name, kp in matched_keypoints.items()}
        }
        
        self.coco_dataset["annotations"].append(annotation)
        
        self.image_id += 1
        self.annotation_id += 1
        return True

    def _print_processing_statistics(self):
        """NEW: Print comprehensive processing statistics."""
        print("\n" + "="*60)
        print("🎯 PROCESSING STATISTICS")
        print("="*60)
        
        # Basic statistics
        total = self.stats['total_images_processed']
        annotated = self.stats['images_with_annotations']
        failed = self.stats['images_failed']
        
        print(f"📊 Images Processed: {total}")
        print(f"✅ Successfully Annotated: {annotated} ({annotated/total*100:.1f}% success rate)" if total > 0 else "✅ Successfully Annotated: 0")
        print(f"❌ Failed to Annotate: {failed}")
        print(f"🔍 Total Keypoints Found: {self.stats['total_keypoints_found']}")
        
        # Reference usage
        print(f"\n📚 Reference Usage:")
        for ref_id, count in self.stats['references_used'].items():
            if count > 0:
                print(f"   - Reference {ref_id}: {count} times")
        
        # Detection methods
        if self.stats['keypoints_by_method']:
            print(f"\n🔧 Detection Methods:")
            for method, count in sorted(self.stats['keypoints_by_method'].items()):
                print(f"   - {method}: {count} keypoints")
        
        # Fallback usage
        fallback_used = any(count > 0 for count in self.stats['fallback_usage'].values())
        if fallback_used:
            print(f"\n🔄 Fallback Strategy Usage:")
            for strategy, count in self.stats['fallback_usage'].items():
                if count > 0:
                    print(f"   - {strategy.replace('_', ' ').title()}: {count} images benefited")
        
        # Tolerance zone effectiveness
        tolerance_methods = ['SuperPoint_Tolerance', 'SuperPoint_Transformed']
        tolerance_count = sum(self.stats['keypoints_by_method'].get(method, 0) for method in tolerance_methods)
        if tolerance_count > 0:
            print(f"\n🎯 Tolerance Zone Effectiveness:")
            print(f"   - {tolerance_count} keypoints found using tolerance zones")
            print(f"   - Tolerance radius: {self.keypoint_tolerance_radius} pixels")
        
        print("="*60)

    def process_images(self, input_folder, image_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """Process a folder of images to generate keypoint annotations with comprehensive statistics."""
        input_path = Path(input_folder)
        image_files = [p for ext in image_extensions for p in input_path.glob(f"*{ext}") ]
        if not image_files:
            print(f"❌ No images found in {input_folder}")
            return
        
        print(f"\n🚀 Starting processing of {len(image_files)} images...")
        
        for i, image_path in enumerate(image_files, 1):
            self.stats['total_images_processed'] += 1
            print(f"📸 Processing {i}/{len(image_files)}: {image_path.name}")
            
            try:
                matched_keypoints, _, best_ref_id, all_robust_matches, debug_info = self.find_matching_keypoints(image_path)
                
                if best_ref_id and matched_keypoints:
                    success = self.create_coco_annotation(image_path, matched_keypoints, best_ref_id, all_robust_matches) # Pass all_robust_matches
                    if success:
                        self.stats['images_with_annotations'] += 1
                        print(f"   ✅ Found {len(matched_keypoints)} keypoints using reference {best_ref_id}")
                    else:
                        print(f"   ⚠️ Insufficient keypoints for COCO annotation ({len(matched_keypoints)} found)")
                        self.stats['images_failed'] += 1
                else:
                    print(f"   ❌ No keypoints detected")
                    self.stats['images_failed'] += 1
                
                if hasattr(self, 'visualize') and self.visualize and best_ref_id:
                    self._visualize_enhanced_keypoints(image_path, matched_keypoints, all_robust_matches, debug_info)

            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
                self.stats['images_failed'] += 1
                if self.debug:
                    import traceback
                    traceback.print_exc()
        
        # Print final statistics
        self._print_processing_statistics()

    def save_annotations(self, filename=None):
        """Save the generated COCO annotations to a JSON file with enhanced metadata."""
        if not self.coco_dataset["images"]:
            print("⚠️ No annotations generated, skipping save.")
            return

        # Add processing statistics to the dataset info
        self.coco_dataset["info"] = {
            "description": f"Auto-generated keypoint annotations for {self.object_name}",
            "processing_date": datetime.now().isoformat(),
            "processing_statistics": self.stats,
            "tolerance_radius": self.keypoint_tolerance_radius,
            "total_images": len(self.coco_dataset["images"]),
            "total_annotations": len(self.coco_dataset["annotations"])
        }

        filename = filename or f"annotations_{self.object_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(self.coco_dataset, f, indent=4)
        
        print(f"✅ Annotations saved to: {output_path}")
        print(f"   📊 {len(self.coco_dataset['images'])} images annotated")
        print(f"   🎯 {sum(ann['num_keypoints'] for ann in self.coco_dataset['annotations'])} total keypoints")

def main():
    parser = argparse.ArgumentParser(description='Ultimate Auto Annotator with tolerance zones and smart thresholds.')
    parser.add_argument('--reference-config', required=True, help='Path to the reference config JSON.')
    parser.add_argument('--input-folder', required=True, help='Folder with input images.')
    parser.add_argument('--output-dir', default='annotations_ultimate', help='Output directory.')
    parser.add_argument('--object-name', default='object', help='Object category name.')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Inference device.')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debug logging and visualizations.')
    parser.add_argument('--visualize', action='store_true', help='Save visualization images.')

    # NEW: Tolerance zone parameters
    parser.add_argument('--tolerance-radius', type=float, default=8.0, help='Tolerance radius for keypoint matching (pixels).')

    # Fallback strategy arguments
    parser.add_argument('--enable-sift-fallback', action='store_true', help='Enable SIFT fallback for missing keypoints.')
    parser.add_argument('--sift-edge-threshold', type=int, default=10, help='SIFT edgeThreshold parameter.')
    parser.add_argument('--enable-orb-fallback', action='store_true', help='Enable ORB fallback for missing keypoints.')
    parser.add_argument('--enable-patch-fallback', action='store_true', help='Enable patch matching fallback.')
    parser.add_argument('--patch-size', type=int, default=32, help='Size of the patch for patch matching fallback.')
    parser.add_argument('--enable-image-enhancement-fallback', action='store_true', help='Enable image enhancement fallback for missing keypoints.')
    parser.add_argument('--image-enhancement-type', type=str, default='grayscale', choices=['grayscale', 'clahe', 'both'], help='Type of image enhancement to apply.')

    # NEW: Bounding Box Strategy Argument
    parser.add_argument('--bbox-strategy', type=str, default='keypoint', choices=['keypoint', 'total_matches'],
                        help='Strategy for bounding box calculation: "keypoint" (strict) or "total_matches" (broader).')
    
    args = parser.parse_args()
    
    annotator = EnhancedMultiRefAnnotator(
        reference_config_path=args.reference_config,
        output_dir=args.output_dir,
        device=args.device,
        object_name=args.object_name,
        debug=args.debug,
        keypoint_tolerance_radius=args.tolerance_radius,  # NEW
        enable_sift_fallback=args.enable_sift_fallback,
        sift_edge_threshold=args.sift_edge_threshold,
        enable_orb_fallback=args.enable_orb_fallback,
        enable_patch_fallback=args.enable_patch_fallback,
        patch_size=args.patch_size,
        enable_image_enhancement_fallback=args.enable_image_enhancement_fallback,
        image_enhancement_type=args.image_enhancement_type,
        bbox_strategy=args.bbox_strategy # NEW
    )
    
    annotator.visualize = args.visualize
    annotator.process_images(args.input_folder)
    annotator.save_annotations()
    print("🎉 Annotation task finished!")

if __name__ == "__main__":
    main()
