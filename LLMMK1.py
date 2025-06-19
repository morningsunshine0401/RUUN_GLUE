"""
ENHANCED ROBUST Multi-Reference Auto Annotator with GROUNDING DINO INTEGRATION
- Grounding DINO for semantic object detection and localization
- RANSAC on ALL SuperPoint matches FIRST (primary filtering step)
- Reference selection based on TOTAL RANSAC-filtered matches
- Semantic bounding box from Grounding DINO (more accurate than feature-based)
- Focused matching within semantically-detected regions
- Object keypoint detection within RANSAC-filtered matches
- Most robust geometric consistency approach
"""

import cv2
import numpy as np
import torch
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from scipy.spatial.distance import cdist, cosine
import warnings
from sklearn.metrics.pairwise import cosine_similarity
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

# Try to import Grounding DINO
try:
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from groundingdino.util import box_ops
    import groundingdino.datasets.transforms as T
    print("✅ Grounding DINO imported successfully")
    GROUNDING_DINO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Grounding DINO not available: {e}")
    print("   Install with: pip install groundingdino-py")
    print("   Or clone: https://github.com/IDEA-Research/GroundingDINO")
    GROUNDING_DINO_AVAILABLE = False


class EnhancedGroundingAnnotator:
    def __init__(self, reference_config_path, output_dir="annotations", device='auto', object_name="object", debug=False):
        """
        Initialize the ENHANCED robust multi-reference auto annotator with GROUNDING DINO
        Key improvements: Semantic object detection + total match-based approach with RANSAC-first pipeline
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.object_name = object_name
        self.debug = debug
        
        # ENHANCED RANSAC parameters for ALL matches (primary filtering)
        self.ransac_enabled = True
        self.ransac_threshold = 8  #12.0  # Tighter threshold for better geometric consistency
        self.ransac_max_trials = 1000 #300  # More trials for better results
        self.min_matches_for_ransac = 10  # Higher minimum for robust filtering
        self.ransac_min_samples = 8  # Minimum samples for affine transformation
        
        # Keypoint matching parameters
        self.distance_threshold = 8 #15.0  # Slightly more lenient after RANSAC filtering
        self.min_keypoints_threshold = 2
        
        # NEW: Total match-based selection parameters
        self.min_total_matches_threshold = 15 #20  # Minimum total matches for reliable reference
        self.total_match_weight = 0.7  # Weight for total matches in selection
        self.keypoint_match_weight = 0.3  # Weight for keypoint matches in selection
        
        # NEW: Grounding DINO parameters
        self.use_grounding_dino = GROUNDING_DINO_AVAILABLE
        self.grounding_dino_enabled = True  # Can be disabled via arguments
        self.grounding_confidence_threshold = 0.3  # Minimum confidence for detection
        self.grounding_text_prompt = None  # Will be set based on object_name
        self.grounding_box_padding_ratio = 0.1  # Padding around Grounding DINO bbox
        self.grounding_min_box_size = 100  # Minimum bounding box size
        
        # Fallback iterative refinement parameters (used if Grounding DINO fails)
        self.enable_iterative_refinement = True
        self.crop_padding_ratio = 0.1  # 10% padding around initial bounding box
        self.min_crop_size = 100  # Minimum crop size to avoid too small regions
        self.max_iterations = 2  # Currently using 2 iterations (initial + refinement)
        
        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"🚀 Using device: {self.device}")
        
        # Initialize models
        print("🔄 Loading AI models...")
        self._init_models()
        
        # Load reference configuration
        print("🔄 Loading reference configuration...")
        self._load_reference_config(reference_config_path)
        
        # Setup Grounding DINO text prompt
        self._setup_grounding_prompt()
        
        # Analyze reference diversity
        if self.debug:
            self._analyze_reference_diversity()
        
        # Initialize COCO dataset
        self._init_coco_dataset()
        
    def _init_models(self):
        """Initialize all models including Grounding DINO"""
        # SuperPoint + LightGlue for keypoint matching
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        print("✅ SuperPoint & LightGlue loaded!")
        
        # Initialize Grounding DINO if available
        if self.use_grounding_dino and self.grounding_dino_enabled:
            self._init_grounding_dino()
        else:
            print("⚠️ Grounding DINO disabled - will use iterative refinement fallback")
        
        # DINOv2 for global feature similarity (fallback only)
        self.use_dinov2 = False
        self.use_efficientnet = False
        
        try:
            # Try to load DINOv2 from torch hub
            print("🔄 Attempting to load DINOv2...")
            self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.dinov2_model.eval().to(self.device)
            self.dinov2_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.use_dinov2 = True
            print("✅ DINOv2 loaded successfully!")
        except Exception as e:
            print(f"⚠️ DINOv2 loading failed: {e}")
            
            # Fallback to EfficientNet if available
            if timm:
                try:
                    print("🔄 Falling back to EfficientNet...")
                    self.feature_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
                    self.feature_model.eval().to(self.device)
                    self.feature_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    self.use_efficientnet = True
                    print("✅ EfficientNet loaded as fallback!")
                except Exception as e2:
                    print(f"⚠️ EfficientNet also failed: {e2}")
                    print("⚠️ Using basic histogram features")
            else:
                print("⚠️ Using basic histogram features")

    def _init_grounding_dino(self):
        """Initialize Grounding DINO model"""
        try:
            print("🔄 Loading Grounding DINO...")
            
            # You might need to adjust these paths based on your Grounding DINO installation
            # These are typical paths for the GroundingDINO repository
            config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"  
            checkpoint_path = "weights/groundingdino_swint_ogc.pth"
            
            # Try to find config and weights in common locations
            possible_config_paths = [
                config_path,
                "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                "./groundingdino/config/GroundingDINO_SwinT_OGC.py",
            ]
            
            possible_weight_paths = [
                checkpoint_path,
                "GroundingDINO/weights/groundingdino_swint_ogc.pth",
                "./weights/groundingdino_swint_ogc.pth",
                "groundingdino_swint_ogc.pth"
            ]
            
            config_found = None
            weights_found = None
            
            for cfg_path in possible_config_paths:
                if os.path.exists(cfg_path):
                    config_found = cfg_path
                    break
                    
            for weight_path in possible_weight_paths:
                if os.path.exists(weight_path):
                    weights_found = weight_path
                    break
            
            if config_found is None or weights_found is None:
                print(f"⚠️ Grounding DINO config or weights not found!")
                print(f"   Config checked: {possible_config_paths}")
                print(f"   Weights checked: {possible_weight_paths}")
                print(f"   Please download from: https://github.com/IDEA-Research/GroundingDINO")
                self.grounding_dino_model = None
                self.use_grounding_dino = False
                return
            
            # Load model
            args = SLConfig.fromfile(config_found)
            args.device = self.device
            model = build_model(args)
            
            checkpoint = torch.load(weights_found, map_location="cpu")
            model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            model.eval().to(self.device)
            
            self.grounding_dino_model = model
            
            # Setup transform
            self.grounding_transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            print("✅ Grounding DINO loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Failed to load Grounding DINO: {e}")
            print("   Will use iterative refinement fallback")
            self.grounding_dino_model = None
            self.use_grounding_dino = False

    def _setup_grounding_prompt(self):
        """Setup text prompt for Grounding DINO based on object name"""
        if self.use_grounding_dino:
            # Create a descriptive prompt for the object
            self.grounding_text_prompt = self.object_name.lower()
            
            # Add common variations to improve detection
            common_variations = {
                'bottle': 'bottle . water bottle . plastic bottle',
                'cup': 'cup . mug . coffee cup . tea cup',
                'phone': 'phone . smartphone . mobile phone . cellphone',
                'laptop': 'laptop . computer . notebook',
                'car': 'car . vehicle . automobile',
                'person': 'person . human . people',
                'dog': 'dog . puppy',
                'cat': 'cat . kitten',
            }
            
            if self.object_name.lower() in common_variations:
                self.grounding_text_prompt = common_variations[self.object_name.lower()]
            
            if self.debug:
                print(f"🎯 Grounding DINO prompt: '{self.grounding_text_prompt}'")

    def detect_object_with_grounding_dino(self, image):
        """
        Use Grounding DINO to detect and localize the target object
        Returns: (bbox, confidence) or (None, 0) if no detection
        """
        if not self.use_grounding_dino or self.grounding_dino_model is None:
            return None, 0
        
        try:
            # Prepare image
            image_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Apply transform
            image_tensor, _ = self.grounding_transform(image_pil, None)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.grounding_dino_model(image_tensor, captions=[self.grounding_text_prompt])
            
            # Process outputs
            prediction_logits = outputs["pred_logits"].sigmoid()[0]  # [num_queries, num_classes]
            prediction_boxes = outputs["pred_boxes"][0]  # [num_queries, 4]
            
            # Filter by confidence
            max_confidence, _ = prediction_logits.max(dim=1)
            confident_mask = max_confidence > self.grounding_confidence_threshold
            
            if not confident_mask.any():
                if self.debug:
                    print(f"   🔍 Grounding DINO: No confident detections (max conf: {max_confidence.max():.3f})")
                return None, 0
            
            # Get best detection
            best_idx = max_confidence.argmax()
            best_confidence = max_confidence[best_idx].item()
            best_box = prediction_boxes[best_idx]
            
            # Convert box from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
            cx, cy, box_w, box_h = best_box.tolist()
            x1 = int((cx - box_w / 2) * w)
            y1 = int((cy - box_h / 2) * h)
            x2 = int((cx + box_w / 2) * w)
            y2 = int((cy + box_h / 2) * h)
            
            # Ensure box is within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Convert to [x, y, width, height] format
            bbox = [x1, y1, x2 - x1, y2 - y1]
            
            # Check minimum box size
            if bbox[2] < self.grounding_min_box_size or bbox[3] < self.grounding_min_box_size:
                if self.debug:
                    print(f"   ⚠️ Grounding DINO: Detection too small ({bbox[2]}x{bbox[3]})")
                return None, 0
            
            if self.debug:
                print(f"   ✅ Grounding DINO: Detected '{self.object_name}' at [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}] "
                      f"(conf: {best_confidence:.3f})")
            
            return bbox, best_confidence
            
        except Exception as e:
            if self.debug:
                print(f"   ⚠️ Grounding DINO detection failed: {e}")
            return None, 0
        
    def _load_reference_config(self, config_path):
        """Load reference images configuration from JSON"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.references = {}
        
        # Get config metadata
        self.num_keypoints = config.get('num_keypoints', 6)
        self.keypoint_names = config.get('keypoint_names', [f'keypoint_{i}' for i in range(self.num_keypoints)])
        
        print(f"🔄 Loading {len(config['references'])} reference images...")
        print(f"📍 Object keypoints: {self.num_keypoints} ({', '.join(self.keypoint_names)})")
        
        for ref_config in config['references']:
            ref_id = ref_config['id']
            ref_path = ref_config['image_path']
            keypoints = ref_config['keypoints']
            viewpoint = ref_config.get('viewpoint', 'unknown')
            description = ref_config.get('description', '')
            
            # Load reference image
            ref_image = cv2.imread(ref_path)
            if ref_image is None:
                print(f"❌ Could not load reference image: {ref_path}")
                continue
            
            # Handle coordinate scaling if needed
            original_size = ref_config.get('original_size', None)
            current_size = [ref_image.shape[1], ref_image.shape[0]]
            
            scaled_keypoints = keypoints.copy()
            if original_size and original_size != current_size:
                scale_x = current_size[0] / original_size[0]
                scale_y = current_size[1] / original_size[1]
                if self.debug:
                    print(f"   📏 Scaling keypoints for {ref_id}: {original_size} → {current_size}")
                
                for name, coords in keypoints.items():
                    if coords != [0, 0]:
                        scaled_keypoints[name] = [coords[0] * scale_x, coords[1] * scale_y]
                        
            # Extract SuperPoint features
            ref_features = self._extract_superpoint_features(ref_image)
            
            # Extract global features for similarity matching (fallback only)
            global_features = self._extract_global_features(ref_image)
            
            self.references[ref_id] = {
                'image': ref_image,
                'image_path': ref_path,
                'keypoints': scaled_keypoints,
                'original_keypoints': keypoints,
                'viewpoint': viewpoint,
                'description': description,
                'superpoint_features': ref_features,
                'global_features': global_features,
                'original_size': original_size,
                'current_size': current_size
            }
            
            visible_count = sum(1 for coords in scaled_keypoints.values() if coords != [0, 0])
            print(f"   ✅ Loaded {ref_id}: {viewpoint} - {visible_count}/{self.num_keypoints} visible keypoints")
        
        if not self.references:
            raise ValueError("No valid reference images loaded!")
            
        print(f"✅ Loaded {len(self.references)} reference images")
        
    def _extract_superpoint_features(self, image):
        """Extract SuperPoint features from image"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.extractor.extract(tensor)
        return features
        
    def _extract_global_features(self, image):
        """Extract global features with enhanced fallback"""
        if self.use_dinov2:
            return self._extract_dinov2_features(image)
        elif self.use_efficientnet:
            return self._extract_efficientnet_features(image)
        else:
            return self._extract_enhanced_histogram_features(image)
            
    def _extract_dinov2_features(self, image):
        """Extract DINOv2 features with error handling"""
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self.dinov2_transform(rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.dinov2_model(tensor)
            return features.cpu().numpy().flatten()
        except Exception as e:
            if self.debug:
                print(f"   ⚠️ DINOv2 extraction failed: {e}")
            return self._extract_enhanced_histogram_features(image)
        
    def _extract_efficientnet_features(self, image):
        """Extract EfficientNet features with error handling"""
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self.feature_transform(rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_model(tensor)
            return features.cpu().numpy().flatten()
        except Exception as e:
            if self.debug:
                print(f"   ⚠️ EfficientNet extraction failed: {e}")
            return self._extract_enhanced_histogram_features(image)
        
    def _extract_enhanced_histogram_features(self, image):
        """Extract enhanced histogram features with multiple color spaces"""
        try:
            # Convert to multiple color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate histograms
            hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
            hist_l = cv2.calcHist([lab], [0], None, [50], [0, 256])
            hist_a = cv2.calcHist([lab], [1], None, [50], [0, 256])
            hist_b = cv2.calcHist([lab], [2], None, [50], [0, 256])
            hist_gray = cv2.calcHist([gray], [0], None, [50], [0, 256])
            
            # Add texture features (LBP-like)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            texture_hist = cv2.calcHist([grad_mag.astype(np.uint8)], [0], None, [50], [0, 256])
            
            # Normalize and concatenate all features
            features = np.concatenate([
                hist_h.flatten() / (hist_h.sum() + 1e-7),
                hist_s.flatten() / (hist_s.sum() + 1e-7),
                hist_v.flatten() / (hist_v.sum() + 1e-7),
                hist_l.flatten() / (hist_l.sum() + 1e-7),
                hist_a.flatten() / (hist_a.sum() + 1e-7),
                hist_b.flatten() / (hist_b.sum() + 1e-7),
                hist_gray.flatten() / (hist_gray.sum() + 1e-7),
                texture_hist.flatten() / (texture_hist.sum() + 1e-7)
            ])
            
            return features
        except Exception as e:
            if self.debug:
                print(f"   ⚠️ Enhanced histogram extraction failed: {e}")
            # Ultimate fallback - basic BGR histogram
            hist_b = cv2.calcHist([image], [0], None, [50], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [50], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [50], [0, 256])
            
            return np.concatenate([
                hist_b.flatten() / (hist_b.sum() + 1e-7),
                hist_g.flatten() / (hist_g.sum() + 1e-7),
                hist_r.flatten() / (hist_r.sum() + 1e-7)
            ])
        
    def _analyze_reference_diversity(self):
        """Analyze diversity between reference images"""
        print(f"\n🔍 Reference diversity analysis:")
        
        ref_ids = list(self.references.keys())
        if len(ref_ids) < 2:
            print("   ⚠️ Only one reference - no diversity analysis possible")
            return
            
        similarities_matrix = np.zeros((len(ref_ids), len(ref_ids)))
        
        for i, ref_id1 in enumerate(ref_ids):
            for j, ref_id2 in enumerate(ref_ids):
                if i == j:
                    similarities_matrix[i, j] = 1.0
                else:
                    feat1 = self.references[ref_id1]['global_features']
                    feat2 = self.references[ref_id2]['global_features']
                    sim = cosine_similarity(
                        feat1.reshape(1, -1),
                        feat2.reshape(1, -1)
                    )[0, 0]
                    similarities_matrix[i, j] = sim
        
        # Check diversity
        off_diagonal = similarities_matrix[np.triu_indices_from(similarities_matrix, k=1)]
        avg_similarity = np.mean(off_diagonal)
        max_similarity = np.max(off_diagonal)
        min_similarity = np.min(off_diagonal)
        
        print(f"   📊 Reference diversity stats:")
        print(f"     Average inter-reference similarity: {avg_similarity:.4f}")
        print(f"     Max inter-reference similarity: {max_similarity:.4f}")
        print(f"     Min inter-reference similarity: {min_similarity:.4f}")
        
        if avg_similarity > 0.95:
            print(f"   ⚠️ WARNING: References are very similar (avg: {avg_similarity:.4f})")
        elif avg_similarity < 0.3:
            print(f"   ✅ Good diversity (avg: {avg_similarity:.4f})")
        else:
            print(f"   ℹ️ Moderate diversity (avg: {avg_similarity:.4f})")

    def _apply_enhanced_ransac_filtering(self, ref_kpts, input_kpts, matches):
        """
        ENHANCED: Apply RANSAC to ALL SuperPoint matches as PRIMARY filtering step
        This is the foundation of the robust approach - geometric consistency first!
        """
        if len(matches) < self.min_matches_for_ransac:
            if self.debug:
                print(f"       🔍 Not enough matches for RANSAC: {len(matches)} < {self.min_matches_for_ransac}")
            return matches, ref_kpts, input_kpts, 0.0
        
        try:
            # Get matched points
            matched_ref_kpts = ref_kpts[matches[:, 0]]
            matched_input_kpts = input_kpts[matches[:, 1]]
            
            if self.debug:
                print(f"       🎯 Applying ENHANCED RANSAC to {len(matches)} raw matches...")
            
            # Use RANSAC with affine transformation model
            # We'll use polynomial features to model more complex transformations
            poly_features = PolynomialFeatures(degree=2, include_bias=True)
            
            # Prepare reference points with polynomial features for robust transformation
            ref_poly = poly_features.fit_transform(matched_ref_kpts)
            
            # Fit transformation for X coordinates
            ransac_x = RANSACRegressor(
                random_state=42,
                residual_threshold=self.ransac_threshold,
                max_trials=self.ransac_max_trials,
                min_samples=self.ransac_min_samples,
                stop_probability=0.99  # High confidence requirement
            )
            
            ransac_x.fit(ref_poly, matched_input_kpts[:, 0])
            inlier_mask_x = np.array(ransac_x.inlier_mask_, dtype=bool)
            
            # Fit transformation for Y coordinates
            ransac_y = RANSACRegressor(
                random_state=42,
                residual_threshold=self.ransac_threshold,
                max_trials=self.ransac_max_trials,
                min_samples=self.ransac_min_samples,
                stop_probability=0.99
            )
            
            ransac_y.fit(ref_poly, matched_input_kpts[:, 1])
            inlier_mask_y = np.array(ransac_y.inlier_mask_, dtype=bool)
            
            # Take intersection of both inlier sets for maximum geometric consistency
            final_inlier_mask = np.logical_and(inlier_mask_x, inlier_mask_y)
            
            # Filter matches based on RANSAC inliers
            filtered_matches = matches[final_inlier_mask]
            
            inlier_count = np.sum(final_inlier_mask)
            outlier_count = len(matches) - inlier_count
            inlier_ratio = inlier_count / len(matches) if len(matches) > 0 else 0.0
            
            if self.debug:
                print(f"       ✅ ENHANCED RANSAC results: {inlier_count} inliers, {outlier_count} outliers")
                print(f"       📊 Inlier ratio: {inlier_ratio:.3f}")
                print(f"       🎯 Transformation scores: X={ransac_x.score(ref_poly, matched_input_kpts[:, 0]):.3f}, "
                      f"Y={ransac_y.score(ref_poly, matched_input_kpts[:, 1]):.3f}")
            
            # Enhanced filtering: Keep only if we have sufficient inliers AND good ratio
            min_inlier_ratio = 0.2  # At least 20% should be inliers
            if inlier_count < self.min_matches_for_ransac or inlier_ratio < min_inlier_ratio:
                if self.debug:
                    print(f"       ⚠️ RANSAC quality too low: {inlier_count} inliers, ratio: {inlier_ratio:.3f}")
                return np.array([]), ref_kpts, input_kpts, 0.0
            
            return filtered_matches, ref_kpts, input_kpts, inlier_ratio
            
        except Exception as e:
            if self.debug:
                print(f"       ⚠️ ENHANCED RANSAC failed: {e}, returning empty matches")
            return np.array([]), ref_kpts, input_kpts, 0.0

    def _total_match_based_reference_selection(self, input_image):
        """
        NEW METHOD: Enhanced reference selection based on TOTAL RANSAC-filtered matches
        This provides the most robust geometric consistency-based selection
        """
        print("🎯 ENHANCED: Total match-based reference selection...")
        
        # Extract features from input image
        input_features = self._extract_superpoint_features(input_image)
        
        reference_scores = {}
        detailed_results = {}
        
        for ref_id, ref_data in self.references.items():
            if self.debug:
                print(f"   🔄 Testing reference: {ref_id}")
            
            # Perform SuperPoint + LightGlue matching
            with torch.no_grad():
                matches_dict = self.matcher({
                    'image0': ref_data['superpoint_features'],
                    'image1': input_features
                })
            
            # Process matches properly using rbd
            feats0, feats1, matches01 = [rbd(x) for x in [
                ref_data['superpoint_features'], input_features, matches_dict
            ]]
            
            ref_kpts = feats0["keypoints"].detach().cpu().numpy()
            input_kpts = feats1["keypoints"].detach().cpu().numpy()
            matches = matches01["matches"].detach().cpu().numpy()
            
            if len(matches) == 0:
                reference_scores[ref_id] = 0
                detailed_results[ref_id] = {
                    'total_matches': 0,
                    'ransac_matches': 0,
                    'ransac_ratio': 0.0,
                    'keypoint_matches': 0,
                    'combined_score': 0.0,
                    'keypoints_found': []
                }
                continue
            
            # Apply ENHANCED RANSAC to ALL matches FIRST (primary filtering)
            filtered_matches, _, _, ransac_ratio = self._apply_enhanced_ransac_filtering(
                ref_kpts, input_kpts, matches
            )
            
            if len(filtered_matches) == 0:
                reference_scores[ref_id] = 0
                detailed_results[ref_id] = {
                    'total_matches': len(matches),
                    'ransac_matches': 0,
                    'ransac_ratio': 0.0,
                    'keypoint_matches': 0,
                    'combined_score': 0.0,
                    'keypoints_found': []
                }
                continue
            
            # Get RANSAC-filtered matched points
            matched_ref_kpts = ref_kpts[filtered_matches[:, 0]]
            matched_input_kpts = input_kpts[filtered_matches[:, 1]]
            
            # NOW find object keypoints within RANSAC-filtered matches
            successful_keypoints = 0
            keypoints_found = []
            
            for name, coords in ref_data['keypoints'].items():
                if coords == [0, 0]:  # Skip invisible keypoints
                    continue
                    
                ref_x, ref_y = coords
                ref_point = np.array([[ref_x, ref_y]])
                
                # Find closest matched reference keypoint within RANSAC-filtered set
                distances = cdist(ref_point, matched_ref_kpts)
                min_distance = np.min(distances) if len(distances) > 0 else float('inf')
                
                if min_distance < self.distance_threshold:
                    successful_keypoints += 1
                    keypoints_found.append(name)
            
            # ENHANCED SCORING: Combine total matches with keypoint matches
            total_match_score = len(filtered_matches)  # Raw count of robust matches
            keypoint_match_score = successful_keypoints  # Object keypoint matches
            
            # Weighted combined score (emphasizing total matches for robustness)
            combined_score = (self.total_match_weight * total_match_score + 
                            self.keypoint_match_weight * keypoint_match_score * 10)  # Scale keypoints
            
            reference_scores[ref_id] = combined_score
            detailed_results[ref_id] = {
                'total_matches': len(matches),
                'ransac_matches': len(filtered_matches),
                'ransac_ratio': ransac_ratio,
                'keypoint_matches': successful_keypoints,
                'combined_score': combined_score,
                'keypoints_found': keypoints_found
            }
            
            if self.debug:
                print(f"     📊 {ref_id}: {len(matches)} → {len(filtered_matches)} (ratio: {ransac_ratio:.3f}) "
                      f"→ {successful_keypoints} keypoints → score: {combined_score:.1f}")
        
        # Select best reference based on combined score
        if not reference_scores or all(score == 0 for score in reference_scores.values()):
            print("   ❌ No valid matches found in any reference!")
            return None, 0, 'total_match_based'
        
        best_ref_id = max(reference_scores.items(), key=lambda x: x[1])[0]
        best_score = reference_scores[best_ref_id]
        
        if self.debug:
            print(f"   🏆 Best reference: {best_ref_id} with combined score: {best_score:.1f}")
            print(f"   📋 All scores: {reference_scores}")
            for ref_id, results in detailed_results.items():
                print(f"     {ref_id}: {results}")
        
        return best_ref_id, best_score, 'total_match_based'

    def find_best_reference(self, input_image, method='total_match_based'):
        """
        Enhanced reference selection with new total match-based method as default
        """
        return self._total_match_based_reference_selection(input_image)

    def _crop_image_with_padding(self, image, bbox, padding_ratio=0.1):
        """
        Crop image with smart padding to focus on object region
        Returns cropped image and transform info for coordinate conversion
        """
        height, width = image.shape[:2]
        x, y, w, h = bbox
        
        # Add padding
        padding_x = max(int(w * padding_ratio), 20)
        padding_y = max(int(h * padding_ratio), 20)
        
        # Calculate crop bounds with padding
        crop_x1 = max(0, int(x - padding_x))
        crop_y1 = max(0, int(y - padding_y))
        crop_x2 = min(width, int(x + w + padding_x))
        crop_y2 = min(height, int(y + h + padding_y))
        
        # Ensure minimum crop size
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1
        
        if crop_width < self.min_crop_size or crop_height < self.min_crop_size:
            if self.debug:
                print(f"       ⚠️ Crop too small ({crop_width}x{crop_height}), using original image")
            return image, None
        
        # Crop the image
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Transform info for coordinate conversion
        transform_info = {
            'crop_x1': crop_x1,
            'crop_y1': crop_y1,
            'crop_x2': crop_x2,
            'crop_y2': crop_y2,
            'crop_width': crop_width,
            'crop_height': crop_height,
            'original_width': width,
            'original_height': height
        }
        
        if self.debug:
            print(f"       ✂️ Cropped region: [{crop_x1}, {crop_y1}, {crop_width}, {crop_height}]")
        
        return cropped, transform_info

    def _convert_coordinates_to_original(self, keypoints, transform_info):
        """
        Convert coordinates from cropped image back to original image coordinates
        """
        if transform_info is None:
            return keypoints
        
        converted_keypoints = {}
        for name, kp in keypoints.items():
            converted_keypoints[name] = {
                'x': float(kp['x'] + transform_info['crop_x1']),
                'y': float(kp['y'] + transform_info['crop_y1']),
                'confidence': kp['confidence'],
                'distance': kp['distance'],
                'reference_used': kp['reference_used'],
                'selection_method': kp['selection_method'],
                'ransac_ratio': kp['ransac_ratio'],
                'detection_method': kp.get('detection_method', 'unknown')
            }
        
        return converted_keypoints

    def _convert_matches_to_original(self, matches, transform_info):
        """
        Convert match coordinates from cropped image back to original image coordinates
        """
        if transform_info is None or matches is None:
            return matches
        
        converted_matches = matches.copy()
        converted_matches[:, 0] += transform_info['crop_x1']  # x coordinates
        converted_matches[:, 1] += transform_info['crop_y1']  # y coordinates
        
        return converted_matches

    def find_matching_keypoints(self, input_image_path, distance_threshold=None):
        """
        ENHANCED keypoint matching with GROUNDING DINO + RANSAC approach:
        1. Grounding DINO: Semantic object detection and localization
        2. Crop: Focus on detected object region
        3. Match: Perform SuperPoint+LightGlue matching in focused region
        4. Fallback: Iterative refinement if Grounding DINO fails
        """
        if distance_threshold is None:
            distance_threshold = self.distance_threshold
            
        # Load input image
        input_image = cv2.imread(str(input_image_path))
        if input_image is None:
            print(f"❌ Could not load image: {input_image_path}")
            return None, None, None, None
            
        # === STAGE 1: Grounding DINO Object Detection ===
        print("   🎯 STAGE 1: Grounding DINO semantic object detection...")
        
        grounding_bbox = None
        grounding_confidence = 0
        detection_method = "none"
        
        if self.use_grounding_dino and self.grounding_dino_enabled:
            grounding_bbox, grounding_confidence = self.detect_object_with_grounding_dino(input_image)
            
            if grounding_bbox is not None:
                print(f"   ✅ Grounding DINO detected '{self.object_name}' with confidence {grounding_confidence:.3f}")
                detection_method = "grounding_dino"
            else:
                print(f"   ⚠️ Grounding DINO failed to detect '{self.object_name}', using iterative fallback")
        else:
            print(f"   ⚠️ Grounding DINO not available, using iterative fallback")
        
        # === STAGE 2: Keypoint Matching in Focused Region ===
        if grounding_bbox is not None:
            print("   🔄 STAGE 2: Focused matching in Grounding DINO region...")
            
            # Crop input image to Grounding DINO detection with padding
            cropped_image, transform_info = self._crop_image_with_padding(
                input_image, grounding_bbox, self.grounding_box_padding_ratio
            )
            
            if transform_info is not None:
                # Find best reference using cropped image
                result = self.find_best_reference(cropped_image, method='total_match_based')
                
                if result is None or result[0] is None:
                    print(f"   ❌ No suitable reference found in Grounding DINO region")
                    return {}, input_image.shape, None, []
                
                best_ref_id, similarity_score, method = result
                best_ref = self.references[best_ref_id]
                
                print(f"   🎯 Best reference: {best_ref_id} ({best_ref['viewpoint']}) - "
                      f"Score: {similarity_score:.3f} via {method}")
                
                # Extract features from cropped image
                cropped_input_features = self._extract_superpoint_features(cropped_image)
                
                # Match cropped image with best reference
                with torch.no_grad():
                    matches_dict = self.matcher({
                        'image0': best_ref['superpoint_features'],
                        'image1': cropped_input_features
                    })
                
                # Process matches
                feats0, feats1, matches01 = [rbd(x) for x in [
                    best_ref['superpoint_features'], cropped_input_features, matches_dict
                ]]
                
                ref_kpts = feats0["keypoints"].detach().cpu().numpy()
                input_kpts = feats1["keypoints"].detach().cpu().numpy()
                matches = matches01["matches"].detach().cpu().numpy()
                
                print(f"   🔗 Found {len(matches)} total feature matches in Grounding DINO region")
                
                if len(matches) > 0:
                    # Apply ENHANCED RANSAC to matches
                    filtered_matches, _, _, ransac_ratio = self._apply_enhanced_ransac_filtering(
                        ref_kpts, input_kpts, matches
                    )
                    
                    if len(filtered_matches) > 0:
                        print(f"   ✅ GROUNDING DINO + RANSAC: {len(filtered_matches)} robust matches (ratio: {ransac_ratio:.3f})")
                        
                        # Get RANSAC-filtered matched points
                        matched_ref_kpts = ref_kpts[filtered_matches[:, 0]]
                        matched_input_kpts = input_kpts[filtered_matches[:, 1]]
                        
                        # Find correspondences for object keypoints
                        matched_object_kpts = {}
                        
                        for name, coords in best_ref['keypoints'].items():
                            if coords == [0, 0]:  # Skip invisible keypoints
                                continue
                                
                            ref_x, ref_y = coords
                            ref_point = np.array([[ref_x, ref_y]])
                            
                            # Find closest matched reference keypoint
                            distances = cdist(ref_point, matched_ref_kpts)
                            min_distance = np.min(distances) if len(distances) > 0 else float('inf')
                            
                            if min_distance < distance_threshold:
                                closest_idx = np.argmin(distances)
                                matched_input_point = matched_input_kpts[closest_idx]
                                
                                matched_object_kpts[name] = {
                                    'x': float(matched_input_point[0]),
                                    'y': float(matched_input_point[1]),
                                    'confidence': float(1.0 - min_distance / distance_threshold),
                                    'distance': float(min_distance),
                                    'reference_used': best_ref_id,
                                    'selection_method': method,
                                    'ransac_ratio': float(ransac_ratio),
                                    'detection_method': detection_method,
                                    'grounding_confidence': float(grounding_confidence)
                                }
                        
                        # Convert keypoints back to original coordinates
                        matched_object_kpts = self._convert_coordinates_to_original(matched_object_kpts, transform_info)
                        
                        # Convert robust matches back to original coordinates
                        final_all_robust_matches = self._convert_matches_to_original(matched_input_kpts, transform_info)
                        
                        final_count = len(matched_object_kpts)
                        total_robust_matches = len(final_all_robust_matches) if final_all_robust_matches is not None else 0
                        
                        print(f"   ✅ GROUNDING DINO RESULTS: {final_count}/{self.num_keypoints} keypoints, {total_robust_matches} total robust matches")
                        
                        return matched_object_kpts, input_image.shape, best_ref_id, final_all_robust_matches
                    else:
                        print(f"   ⚠️ No robust matches in Grounding DINO region after RANSAC")
                else:
                    print(f"   ⚠️ No matches found in Grounding DINO region")
            else:
                print(f"   ⚠️ Grounding DINO crop not possible")
        
        # === FALLBACK: Iterative Refinement Approach ===
        print("   🔄 FALLBACK: Using iterative refinement approach...")
        detection_method = "iterative_fallback"
        
        # Find best reference using full image
        result = self.find_best_reference(input_image, method='total_match_based')
        
        if result is None or result[0] is None:
            print(f"   ❌ No suitable reference found")
            return {}, input_image.shape, None, []
        
        best_ref_id, similarity_score, method = result
        best_ref = self.references[best_ref_id]
        
        print(f"   🎯 Best reference: {best_ref_id} ({best_ref['viewpoint']}) - "
              f"Score: {similarity_score:.3f} via {method}")
        
        # Extract features from input
        input_features = self._extract_superpoint_features(input_image)
        
        # Match with best reference using SuperPoint + LightGlue
        with torch.no_grad():
            matches_dict = self.matcher({
                'image0': best_ref['superpoint_features'],
                'image1': input_features
            })
            
        # Process matches
        feats0, feats1, matches01 = [rbd(x) for x in [
            best_ref['superpoint_features'], input_features, matches_dict
        ]]
        
        ref_kpts = feats0["keypoints"].detach().cpu().numpy()
        input_kpts = feats1["keypoints"].detach().cpu().numpy()
        matches = matches01["matches"].detach().cpu().numpy()
        
        print(f"   🔗 Found {len(matches)} total feature matches")
        
        if len(matches) == 0:
            return {}, input_image.shape, best_ref_id, []
            
        # Apply ENHANCED RANSAC to ALL matches FIRST
        filtered_matches, _, _, ransac_ratio = self._apply_enhanced_ransac_filtering(
            ref_kpts, input_kpts, matches
        )
        
        if len(filtered_matches) == 0:
            print(f"   ⚠️ All matches filtered out by RANSAC")
            return {}, input_image.shape, best_ref_id, []
        
        print(f"   ✅ FALLBACK RANSAC: {len(filtered_matches)} robust matches (ratio: {ransac_ratio:.3f})")
        
        # Get RANSAC-filtered matched points
        matched_ref_kpts = ref_kpts[filtered_matches[:, 0]]
        matched_input_kpts = input_kpts[filtered_matches[:, 1]]
        
        # Store ALL robust match points for bounding box calculation
        all_robust_matches = matched_input_kpts.copy()
        
        # Find correspondences for object keypoints within RANSAC-filtered matches
        matched_object_kpts = {}
        
        for name, coords in best_ref['keypoints'].items():
            # Skip invisible keypoints
            if coords == [0, 0]:
                continue
                
            ref_x, ref_y = coords
            ref_point = np.array([[ref_x, ref_y]])
            
            # Find closest matched reference keypoint within RANSAC-filtered set
            distances = cdist(ref_point, matched_ref_kpts)
            min_distance = np.min(distances) if len(distances) > 0 else float('inf')
            
            if min_distance < distance_threshold:
                closest_idx = np.argmin(distances)
                matched_input_point = matched_input_kpts[closest_idx]
                
                matched_object_kpts[name] = {
                    'x': float(matched_input_point[0]),
                    'y': float(matched_input_point[1]),
                    'confidence': float(1.0 - min_distance / distance_threshold),
                    'distance': float(min_distance),
                    'reference_used': best_ref_id,
                    'selection_method': method,
                    'ransac_ratio': float(ransac_ratio),
                    'detection_method': detection_method,
                    'grounding_confidence': 0.0
                }
        
        final_count = len(matched_object_kpts)
        total_robust_matches = len(all_robust_matches)
        print(f"   ✅ FALLBACK RESULTS: {final_count}/{self.num_keypoints} keypoints, {total_robust_matches} total robust matches")
        
        return matched_object_kpts, input_image.shape, best_ref_id, all_robust_matches
        
    def _init_coco_dataset(self):
        """Initialize COCO dataset structure"""
        skeleton = []
        for i in range(len(self.keypoint_names) - 1):
            skeleton.append([i + 1, i + 2])  # COCO uses 1-based indexing
            
        self.coco_dataset = {
            "info": {
                "description": f"Enhanced Grounding DINO {self.object_name.title()} Keypoint Dataset",
                "version": "7.0",
                "year": datetime.now().year,
                "contributor": "Enhanced Grounding DINO Multi-Reference Auto Annotator",
                "date_created": datetime.now().isoformat(),
                "approach": "Grounding DINO semantic detection + RANSAC-first matching"
            },
            "licenses": [{"id": 1, "name": "Custom License", "url": ""}],
            "images": [],
            "annotations": [],
            "categories": [{
                "id": 1,
                "name": self.object_name,
                "supercategory": "object",
                "keypoints": self.keypoint_names,
                "skeleton": skeleton
            }]
        }
        
        self.image_id = 1
        self.annotation_id = 1
        
    def create_coco_annotation(self, image_path, matched_keypoints, image_shape, best_ref_id, all_robust_matches):
        """
        ENHANCED: Create COCO annotation with Grounding DINO metadata
        """
        if matched_keypoints is None:
            matched_keypoints = {}
            
        height, width = image_shape[:2]
        
        # Determine detection method used
        detection_method = "none"
        grounding_confidence = 0.0
        if matched_keypoints:
            sample_kp = next(iter(matched_keypoints.values()))
            detection_method = sample_kp.get('detection_method', 'unknown')
            grounding_confidence = sample_kp.get('grounding_confidence', 0.0)
        
        # Add image info with enhanced metadata
        image_info = {
            "id": self.image_id,
            "width": width,
            "height": height,
            "file_name": Path(image_path).name,
            "license": 1,
            "date_captured": datetime.now().isoformat(),
            "best_reference_used": best_ref_id,
            "reference_viewpoint": self.references[best_ref_id]['viewpoint'] if best_ref_id else "unknown",
            "feature_extraction_method": "Grounding DINO + SuperPoint+LightGlue with Enhanced RANSAC",
            "selection_method": getattr(self, 'selection_method', 'total_match_based'),
            "total_robust_matches": len(all_robust_matches) if all_robust_matches is not None else 0,
            "grounding_dino_enabled": self.use_grounding_dino and self.grounding_dino_enabled,
            "detection_method": detection_method,
            "grounding_confidence": float(grounding_confidence),
            "grounding_text_prompt": self.grounding_text_prompt if self.use_grounding_dino else None
        }
        self.coco_dataset["images"].append(image_info)
        
        # Create keypoints array
        keypoints = []
        detected_count = 0
        
        for kp_name in self.keypoint_names:
            if kp_name in matched_keypoints:
                kp = matched_keypoints[kp_name]
                keypoints.extend([kp['x'], kp['y'], 2])  # visible
                detected_count += 1
            else:
                keypoints.extend([0, 0, 0])  # undetected
                
        # Calculate bounding box based on ALL robust matches
        if all_robust_matches is not None and len(all_robust_matches) > 0:
            # Use ALL RANSAC-filtered matches for bounding box
            bbox_points = all_robust_matches
            x_min, y_min = np.min(bbox_points, axis=0)
            x_max, y_max = np.max(bbox_points, axis=0)
            
            # Add reasonable padding
            padding = max(20, (x_max - x_min + y_max - y_min) * 0.1)  # Adaptive padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)
            
            # COCO format: [x, y, width, height]
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            bbox = [float(x_min), float(y_min), float(bbox_width), float(bbox_height)]
            area = float(bbox_width * bbox_height)
            
            bbox_source = f"all_robust_matches_{detection_method}"
            
        elif matched_keypoints:
            # Fallback: Use object keypoints if available
            bbox_points = np.array([[kp['x'], kp['y']] for kp in matched_keypoints.values()])
            x_min, y_min = np.min(bbox_points, axis=0)
            x_max, y_max = np.max(bbox_points, axis=0)
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)
            
            # COCO format: [x, y, width, height]
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            bbox = [float(x_min), float(y_min), float(bbox_width), float(bbox_height)]
            area = float(bbox_width * bbox_height)
            
            bbox_source = "object_keypoints"
            
        else:
            # Last resort: Use full image
            bbox = [0.0, 0.0, float(width), float(height)]
            area = float(width * height)
            bbox_source = "full_image"
            
        # Create annotation with enhanced metadata
        annotation = {
            "id": self.annotation_id,
            "image_id": self.image_id,
            "category_id": 1,
            "keypoints": keypoints,
            "num_keypoints": detected_count,
            "bbox": bbox,  # [x, y, width, height] format
            "area": area,
            "iscrowd": 0,
            "reference_used": best_ref_id,
            "reference_viewpoint": self.references[best_ref_id]['viewpoint'] if best_ref_id else "unknown",
            "detection_confidence": np.mean([kp.get('confidence', 0) for kp in matched_keypoints.values()]) if matched_keypoints else 0,
            "selection_method": getattr(self, 'selection_method', 'total_match_based'),
            "ransac_applied": self.ransac_enabled,
            "bbox_source": bbox_source,
            "total_robust_matches": len(all_robust_matches) if all_robust_matches is not None else 0,
            "avg_ransac_ratio": np.mean([kp.get('ransac_ratio', 0) for kp in matched_keypoints.values()]) if matched_keypoints else 0,
            "grounding_dino_enabled": self.use_grounding_dino and self.grounding_dino_enabled,
            "detection_method": detection_method,
            "grounding_confidence": float(grounding_confidence)
        }
        
        self.coco_dataset["annotations"].append(annotation)
        
        method_info = f" ({detection_method}"
        if detection_method == "grounding_dino":
            method_info += f", conf: {grounding_confidence:.3f}"
        method_info += ")"
        
        print(f"   ✅ Detected: {detected_count}/{len(self.keypoint_names)} keypoints using {best_ref_id}{method_info}")
        print(f"   📦 Bounding box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}] "
              f"(area: {area:.1f}, source: {bbox_source})")
        if all_robust_matches is not None:
            print(f"   🎯 Total robust matches used: {len(all_robust_matches)}")
        
        self.image_id += 1
        self.annotation_id += 1
        
        return True
        
    def process_images(self, input_folder, image_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """Process all images with enhanced Grounding DINO + RANSAC approach"""
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
            
        # Get all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
        if not image_files:
            print(f"❌ No images found in {input_folder}")
            return
            
        print(f"🔄 Processing {len(image_files)} images with GROUNDING DINO + RANSAC approach...")
        print(f"🎯 Grounding DINO enabled: {self.use_grounding_dino and self.grounding_dino_enabled}")
        print(f"🎯 Text prompt: '{self.grounding_text_prompt}'")
        print(f"🎯 Confidence threshold: {self.grounding_confidence_threshold}")
        print(f"🎯 RANSAC enabled: {self.ransac_enabled}")
        print(f"🎯 Distance threshold: {self.distance_threshold}")
        
        successful_annotations = 0
        reference_usage = {ref_id: 0 for ref_id in self.references.keys()}
        total_robust_matches_stats = []
        detection_method_stats = {"grounding_dino": 0, "iterative_fallback": 0, "none": 0}
        grounding_confidence_stats = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n📸 Processing {i}/{len(image_files)}: {image_path.name}")
            
            try:
                result = self.find_matching_keypoints(image_path)
                if result[1] is None:
                    continue
                    
                matched_keypoints, image_shape, best_ref_id, all_robust_matches = result
                if best_ref_id:
                    reference_usage[best_ref_id] += 1
                
                if all_robust_matches is not None:
                    total_robust_matches_stats.append(len(all_robust_matches))
                
                # Track detection method statistics
                if matched_keypoints:
                    sample_kp = next(iter(matched_keypoints.values()))
                    detection_method = sample_kp.get('detection_method', 'none')
                    grounding_confidence = sample_kp.get('grounding_confidence', 0.0)
                    
                    detection_method_stats[detection_method] = detection_method_stats.get(detection_method, 0) + 1
                    if detection_method == "grounding_dino":
                        grounding_confidence_stats.append(grounding_confidence)
                
                if self.create_coco_annotation(image_path, matched_keypoints, image_shape, best_ref_id, all_robust_matches):
                    successful_annotations += 1
                    
                    # Optionally visualize
                    if hasattr(self, 'visualize') and self.visualize:
                        self._visualize_grounding_keypoints(image_path, matched_keypoints, best_ref_id, all_robust_matches)
                        
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                continue
                
        # Enhanced statistics with Grounding DINO info
        print(f"\n✅ GROUNDING DINO + RANSAC processing completed!")
        print(f"📊 Results:")
        print(f"   Successfully processed: {successful_annotations}/{len(image_files)} images")
        
        # Reference usage analysis
        unique_refs_used = sum(1 for count in reference_usage.values() if count > 0)
        print(f"   References used: {unique_refs_used}/{len(self.references)}")
        
        print(f"   Reference usage statistics:")
        for ref_id, count in reference_usage.items():
            ref = self.references[ref_id]
            percentage = (count / successful_annotations) * 100 if successful_annotations > 0 else 0
            print(f"     {ref_id} ({ref['viewpoint']}): {count} times ({percentage:.1f}%)")
        
        # Detection method statistics
        print(f"   Detection method usage:")
        for method, count in detection_method_stats.items():
            percentage = (count / successful_annotations) * 100 if successful_annotations > 0 else 0
            print(f"     {method}: {count} images ({percentage:.1f}%)")
        
        # Grounding DINO confidence statistics
        if grounding_confidence_stats:
            avg_conf = np.mean(grounding_confidence_stats)
            median_conf = np.median(grounding_confidence_stats)
            max_conf = np.max(grounding_confidence_stats)
            min_conf = np.min(grounding_confidence_stats)
            
            print(f"   Grounding DINO confidence statistics:")
            print(f"     Average: {avg_conf:.3f}")
            print(f"     Median: {median_conf:.3f}")
            print(f"     Range: {min_conf:.3f} - {max_conf:.3f}")
        
        # Robust matches statistics
        if total_robust_matches_stats:
            avg_matches = np.mean(total_robust_matches_stats)
            median_matches = np.median(total_robust_matches_stats)
            max_matches = np.max(total_robust_matches_stats)
            min_matches = np.min(total_robust_matches_stats)
            
            print(f"   Total robust matches statistics:")
            print(f"     Average: {avg_matches:.1f}")
            print(f"     Median: {median_matches:.1f}")
            print(f"     Range: {min_matches} - {max_matches}")
        
    def save_annotations(self, filename=None):
        """Save COCO annotations with Grounding DINO metadata"""
        if filename is None:
            filename = f"grounding_dino_{self.object_name}_annotations.json"
            
        output_path = self.output_dir / filename
        
        # Add generation metadata
        self.coco_dataset["info"]["generation_stats"] = {
            "total_references": len(self.references),
            "feature_extraction_method": "Grounding DINO + SuperPoint+LightGlue",
            "approach": "Grounding DINO semantic detection + RANSAC-first matching",
            "bounding_box_method": "all_robust_matches_with_grounding_dino",
            "grounding_dino_enabled": self.use_grounding_dino and self.grounding_dino_enabled,
            "grounding_text_prompt": self.grounding_text_prompt if self.use_grounding_dino else None,
            "grounding_confidence_threshold": self.grounding_confidence_threshold,
            "grounding_box_padding_ratio": self.grounding_box_padding_ratio,
            "debug_mode": self.debug,
            "ransac_enabled": self.ransac_enabled,
            "ransac_threshold": self.ransac_threshold,
            "distance_threshold": self.distance_threshold,
            "selection_method": getattr(self, 'selection_method', 'total_match_based'),
            "total_match_weight": self.total_match_weight,
            "keypoint_match_weight": self.keypoint_match_weight
        }
        
        with open(output_path, 'w') as f:
            json.dump(self.coco_dataset, f, indent=2)
            
        print(f"✅ Grounding DINO enhanced annotations saved to: {output_path}")
        
        # Print statistics
        num_images = len(self.coco_dataset["images"])
        num_annotations = len(self.coco_dataset["annotations"])
        print(f"📊 Dataset statistics:")
        print(f"   Images: {num_images}")
        print(f"   Annotations: {num_annotations}")
        print(f"   References available: {len(self.references)}")
        print(f"   Object type: {self.object_name}")
        print(f"   Features: Grounding DINO + SuperPoint+LightGlue with Enhanced RANSAC")
        print(f"   Approach: Semantic detection + RANSAC-first matching")

    def _visualize_grounding_keypoints(self, image_path, matched_keypoints, best_ref_id, all_robust_matches, save_vis=True):
        """
        ENHANCED visualization for Grounding DINO showing detection method
        """
        input_image = cv2.imread(str(image_path))
        ref_image = self.references[best_ref_id]['image'].copy() if best_ref_id else None
        
        # Generate distinct colors for keypoints
        colors = []
        for i in range(len(self.keypoint_names)):
            hue = int(180 * i / len(self.keypoint_names))
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        
        color_map = {name: colors[i] for i, name in enumerate(self.keypoint_names)}
        
        # Draw reference keypoints
        if ref_image is not None and best_ref_id:
            ref_keypoints = self.references[best_ref_id]['keypoints']
            for name, coords in ref_keypoints.items():
                if coords != [0, 0]:
                    ref_x, ref_y = coords
                    color = color_map.get(name, (128, 128, 128))
                    cv2.circle(ref_image, (int(ref_x), int(ref_y)), 8, color, -1)
                    cv2.circle(ref_image, (int(ref_x), int(ref_y)), 15, color, 3)
                    
                    kp_idx = self.keypoint_names.index(name) if name in self.keypoint_names else -1
                    cv2.putText(ref_image, str(kp_idx), (int(ref_x)+20, int(ref_y)-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            ref_info = f"REF: {best_ref_id} ({self.references[best_ref_id]['viewpoint']})"
            cv2.putText(ref_image, ref_info, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            cv2.putText(ref_image, ref_info, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Draw ALL robust matches as small points (background layer)
        if all_robust_matches is not None and len(all_robust_matches) > 0:
            for match_point in all_robust_matches:
                x, y = int(match_point[0]), int(match_point[1])
                # Add bounds checking
                if 0 <= x < input_image.shape[1] and 0 <= y < input_image.shape[0]:
                    cv2.circle(input_image, (x, y), 2, (200, 200, 200), -1)  # Light gray small points
        
        # Determine detection method and confidence
        detection_method = "none"
        grounding_confidence = 0.0
        if matched_keypoints:
            sample_kp = next(iter(matched_keypoints.values()))
            detection_method = sample_kp.get('detection_method', 'unknown')
            grounding_confidence = sample_kp.get('grounding_confidence', 0.0)
        
        # Draw detected object keypoints (foreground layer)
        if not matched_keypoints:
            cv2.putText(input_image, "No keypoints detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            # Draw object keypoints (larger and colored)
            object_keypoint_points = []
            for name, kp in matched_keypoints.items():
                x, y = int(kp['x']), int(kp['y'])
                color = color_map.get(name, (128, 128, 128))
                confidence = kp.get('confidence', 1.0)
                
                cv2.circle(input_image, (x, y), 8, color, -1)
                cv2.circle(input_image, (x, y), 15, color, 3)
                
                kp_idx = self.keypoint_names.index(name) if name in self.keypoint_names else -1
                label = f"{kp_idx} ({confidence:.2f})"
                cv2.putText(input_image, label, (x+20, y-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                object_keypoint_points.append([x, y])
            
            # Draw bounding box
            if all_robust_matches is not None and len(all_robust_matches) > 0:
                bbox_points = np.array(all_robust_matches)
                x_min, y_min = np.min(bbox_points, axis=0)
                x_max, y_max = np.max(bbox_points, axis=0)
                
                # Add padding
                height, width = input_image.shape[:2]
                padding = max(20, (x_max - x_min + y_max - y_min) * 0.1)
                x_min = max(0, int(x_min - padding))
                y_min = max(0, int(y_min - padding))
                x_max = min(width, int(x_max + padding))
                y_max = min(height, int(y_max + padding))
                
                # Color based on detection method
                bbox_color = (0, 255, 255) if detection_method == "grounding_dino" else (0, 255, 0)  # Cyan for Grounding DINO, Green for fallback
                cv2.rectangle(input_image, (x_min, y_min), (x_max, y_max), bbox_color, 3)
                
                # Draw bbox info with detection method
                bbox_w, bbox_h = x_max - x_min, y_max - y_min
                if detection_method == "grounding_dino":
                    method_info = f" (Grounding DINO {grounding_confidence:.3f})"
                else:
                    method_info = f" (Fallback)"
                bbox_info = f"BBox: {bbox_w:.0f}x{bbox_h:.0f}{method_info}"
                cv2.putText(input_image, bbox_info, (x_min, y_min-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
        
        # Enhanced status display with detection method info
        detection_count = len(matched_keypoints) if (matched_keypoints is not None and len(matched_keypoints) > 0) else 0
        total_matches_count = len(all_robust_matches) if all_robust_matches is not None else 0

        cv2.putText(input_image, f"Keypoints: {detection_count}/{len(self.keypoint_names)}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(input_image, f"Keypoints: {detection_count}/{len(self.keypoint_names)}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        cv2.putText(input_image, f"Robust Matches: {total_matches_count}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(input_image, f"Robust Matches: {total_matches_count}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add detection method information
        if detection_method == "grounding_dino":
            method_text = f"Detection: Grounding DINO ({grounding_confidence:.3f})"
            method_color = (0, 255, 255)  # Cyan
        else:
            method_text = f"Detection: Iterative Fallback"
            method_color = (0, 255, 0)  # Green
            
        cv2.putText(input_image, method_text, (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(input_image, method_text, (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, method_color, 1)
        
        # Add approach information
        approach_text = f"Enhanced: Grounding DINO + RANSAC-first"
        cv2.putText(input_image, approach_text, (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(input_image, approach_text, (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        if save_vis and hasattr(self, 'output_dir'):
            vis_dir = self.output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            
            input_vis_path = vis_dir / f"grounding_input_{Path(image_path).name}"
            cv2.imwrite(str(input_vis_path), input_image)
            
            if ref_image is not None:
                target_height = 600
                ref_aspect = ref_image.shape[1] / ref_image.shape[0]
                input_aspect = input_image.shape[1] / input_image.shape[0]
                
                ref_resized = cv2.resize(ref_image, (int(target_height * ref_aspect), target_height))
                input_resized = cv2.resize(input_image, (int(target_height * input_aspect), target_height))
                
                comparison = np.hstack([ref_resized, input_resized])
                separator_x = ref_resized.shape[1]
                cv2.line(comparison, (separator_x, 0), (separator_x, target_height), (255, 255, 255), 3)
                
                comparison_path = vis_dir / f"grounding_comparison_{Path(image_path).name}"
                cv2.imwrite(str(comparison_path), comparison)
                
                print(f"   💾 Saved Grounding DINO visualizations: {input_vis_path.name} & {comparison_path.name}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Grounding DINO Multi-Reference Auto Annotator')
    parser.add_argument('--reference-config', required=True,
                       help='Path to reference configuration JSON file')
    parser.add_argument('--input-folder', required=True,
                       help='Folder containing input images to annotate')
    parser.add_argument('--output-dir', default='annotations',
                       help='Output directory for annotations')
    parser.add_argument('--object-name', default='object',
                       help='Name of the object category for COCO dataset')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--visualize', action='store_true', default=False,
                       help='Save visualization images')
    parser.add_argument('--distance-threshold', type=float, default=15.0,
                       help='Distance threshold for keypoint matching')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug mode with detailed analysis')
    
    # Enhanced RANSAC parameters
    parser.add_argument('--disable-ransac', action='store_true', default=False,
                       help='Disable RANSAC outlier filtering')
    parser.add_argument('--ransac-threshold', type=float, default=12.0,
                       help='RANSAC residual threshold')
    parser.add_argument('--ransac-max-trials', type=int, default=300,
                       help='Maximum RANSAC trials')
    parser.add_argument('--min-matches-for-ransac', type=int, default=10,
                       help='Minimum matches needed for RANSAC')
    
    # Total match-based parameters
    parser.add_argument('--total-match-weight', type=float, default=0.7,
                       help='Weight for total matches in combined scoring')
    parser.add_argument('--keypoint-match-weight', type=float, default=0.3,
                       help='Weight for keypoint matches in combined scoring')
    
    # NEW: Grounding DINO parameters
    parser.add_argument('--disable-grounding-dino', action='store_true', default=False,
                       help='Disable Grounding DINO (use iterative refinement fallback)')
    parser.add_argument('--grounding-confidence-threshold', type=float, default=0.3,
                       help='Minimum confidence threshold for Grounding DINO detection')
    parser.add_argument('--grounding-box-padding-ratio', type=float, default=0.1,
                       help='Padding ratio around Grounding DINO bounding box')
    parser.add_argument('--grounding-min-box-size', type=int, default=100,
                       help='Minimum bounding box size for Grounding DINO detection')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Grounding DINO Multi-Reference Auto Annotator")
    print("=" * 80)
    print("🎯 Grounding DINO semantic detection + RANSAC-first matching")
    print("🧠 Semantic understanding for better object localization")
    print("📦 Bounding boxes based on ALL robust feature matches")
    print("=" * 80)
    
    try:
        annotator = EnhancedGroundingAnnotator(
            reference_config_path=args.reference_config,
            output_dir=args.output_dir,
            device=args.device,
            object_name=args.object_name,
            debug=args.debug
        )
        
        # Set parameters
        annotator.visualize = args.visualize
        annotator.distance_threshold = args.distance_threshold
        annotator.ransac_enabled = not args.disable_ransac
        annotator.ransac_threshold = args.ransac_threshold
        annotator.ransac_max_trials = args.ransac_max_trials
        annotator.min_matches_for_ransac = args.min_matches_for_ransac
        annotator.total_match_weight = args.total_match_weight
        annotator.keypoint_match_weight = args.keypoint_match_weight
        
        # NEW: Grounding DINO parameters
        annotator.grounding_dino_enabled = not args.disable_grounding_dino
        annotator.grounding_confidence_threshold = args.grounding_confidence_threshold
        annotator.grounding_box_padding_ratio = args.grounding_box_padding_ratio
        annotator.grounding_min_box_size = args.grounding_min_box_size
        
        # Process images
        annotator.process_images(args.input_folder)
        
        # Save annotations
        annotator.save_annotations()
        
        print(f"\n🎉 Enhanced Grounding DINO auto annotation completed!")
        print(f"🎯 Key improvements:")
        print(f"   ✅ Grounding DINO semantic object detection")
        print(f"   ✅ RANSAC applied to ALL matches FIRST")
        print(f"   ✅ Reference selection based on total robust matches")
        print(f"   ✅ Focused matching in semantically-detected regions")
        print(f"   ✅ Robust fallback to iterative refinement")
        print(f"   ✅ Enhanced geometric consistency")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()