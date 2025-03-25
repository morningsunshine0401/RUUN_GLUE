import threading
import cv2
import torch
# Disable gradient computation globally
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)
import time
import numpy as np
from scipy.spatial import cKDTree
from utils import (
    frame2tensor,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    normalize_quaternion
)

from KF_MK3 import MultExtendedKalmanFilter

import matplotlib.cm as cm
from models.utils import make_matching_plot_fast
import logging
import timm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Import LightGlue and SuperPoint
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

# Configure logging
logging.basicConfig(
    #level=logging.DEBUG,
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pose_estimator.log"),  # Logs will be saved in this file
        logging.StreamHandler()  # Logs will also be printed to console
    ]
)
logger = logging.getLogger(__name__)

class PoseEstimator:
    def __init__(self, opt, device, kf_mode='auto'):
        self.session_lock = threading.Lock()
        self.opt = opt
        self.device = device
        self.kf_mode = kf_mode  # Store the KF mode
        self.initial_z_set = False  # Flag for first-frame Z override (if desired)
        self.kf_initialized = False  # To track if Kalman filter was ever updated
        self.pred_only = 0

        logger.info("Initializing PoseEstimator with separate SuperPoint and LightGlue models")

        # Initialize SuperPoint and LightGlue models
        self.extractor = SuperPoint(max_num_keypoints=512).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)
        logger.info("Initialized SuperPoint and LightGlue models")

        # Initialize ViT model for viewpoint classification
        self.init_viewpoint_classifier()

        # Define the viewpoint class names
        self.class_names = ['NE', 'NW', 'SE', 'SW']
        
        # Setup anchor data for multiple viewpoints
        self.anchor_data = {}
        self.setup_anchors()

        # We'll store the current anchor viewpoint
        self.current_viewpoint = None
        self.prev_viewpoint = None
        self.viewpoint_confidence = 0.0
        
        # Add these lines:
        self.kf_initialized = False
        self.tracking_3D_points = None
        self.tracking_2D_points = None
        
        # Replace your existing Kalman filter init with MEKF
        # We'll initialize it properly when we get the first good pose
        self.mekf = None

    def init_viewpoint_classifier(self):
        """Initialize the ViT model for viewpoint classification"""
        logger.info("Initializing MobileViT viewpoint classifier")
        
        try:
            # Number of classes for the viewpoint classifier
            num_classes = 4  # NE, NW, SE, SW
            
            # Initialize the model
            self.vit_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=num_classes)
            
            # Load the model weights - if path is provided in options, use it, otherwise use default
            model_path = getattr(self.opt, 'vit_model_path', 'mobilevit_viewpoint_twostage_final_2.pth')
            self.vit_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.vit_model.to(self.device)
            self.vit_model.eval()
            
            # Setup preprocessing for the viewpoint classifier
            self.vit_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
            
            logger.info("MobileViT viewpoint classifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize viewpoint classifier: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def setup_anchors(self):
        """Setup anchor images and features for each viewpoint"""
        logger.info("Setting up multiple anchor images for different viewpoints")
        
        try:
            # Define paths to anchor images for each viewpoint
            # Default to the main anchor if specific ones aren't provided
            anchor_paths = {
                'NE': getattr(self.opt, 'anchor_NE', self.opt.anchor),
                'NW': getattr(self.opt, 'anchor_NW', self.opt.anchor),
                'SE': getattr(self.opt, 'anchor_SE', self.opt.anchor),
                'SW': getattr(self.opt, 'anchor_SW', self.opt.anchor)
            }
            
            # Define 2D and 3D keypoints for each viewpoint
            # For now, we'll use the same keypoints for all viewpoints if specific ones aren't provided
            anchor_keypoints_2D_base = np.array([
                [511, 293], #0
                [591, 284], #
                [587, 330], #
                [413, 249], #
                [602, 348], #
                [715, 384], #
                [598, 298], #
                [656, 171], #
                [805, 213],#
                [703, 392],#10 
                [523, 286],#
                [519, 327],#12
                [387, 289],#13
                [727, 126],# 14
                [425, 243],# 15
                [636, 358],#
                [745, 202],#
                [595, 388],#
                [436, 260],#
                [539, 313], # 20
                [795, 220],# 
                [351, 291],#
                [665, 165],# 
                [611, 353], #
                [650, 377],# 25
                [516, 389],## 
                [727, 143], #
                [496, 378], #
                [575, 312], #
                [617, 368],# 30
                [430, 312], #
                [480, 281], #
                [834, 225], #
                [469, 339], #
                [705, 223], # 35
                [637, 156], 
                [816, 414], 
                [357, 195], 
                [752, 77], 
                [642, 451]
            ], dtype=np.float32)

            anchor_keypoints_3D_base = np.array([
                [-0.014,  0.000,  0.042],
                [ 0.025, -0.014, -0.011],
                [-0.014,  0.000, -0.042],
                [-0.014,  0.000,  0.156],
                [-0.023,  0.000, -0.065],
                [ 0.000,  0.000, -0.156],
                [ 0.025,  0.000, -0.015],
                [ 0.217,  0.000,  0.070],#
                [ 0.230,  0.000, -0.070],
                [-0.014,  0.000, -0.156],
                [ 0.000,  0.000,  0.042],
                [-0.057, -0.018, -0.010],
                [-0.074, -0.000,  0.128],
                [ 0.206, -0.070, -0.002],
                [-0.000, -0.000,  0.156],
                [-0.017, -0.000, -0.092],
                [ 0.217, -0.000, -0.027],#
                [-0.052, -0.000, -0.097],
                [-0.019, -0.000,  0.128],
                [-0.035, -0.018, -0.010],
                [ 0.217, -0.000, -0.070],#
                [-0.080, -0.000,  0.156],
                [ 0.230, -0.000,  0.070],
                [-0.023, -0.000, -0.075],
                [-0.029, -0.000, -0.127],
                [-0.090, -0.000, -0.042],
                [ 0.206, -0.055, -0.002],
                [-0.090, -0.000, -0.015],
                [ 0.000, -0.000, -0.015],
                [-0.037, -0.000, -0.097],
                [-0.074, -0.000,  0.074],
                [-0.019, -0.000,  0.074],
                [ 0.230, -0.000, -0.113],#
                [-0.100, -0.030,  0.000],#
                [ 0.170, -0.000, -0.015],
                [ 0.230, -0.000,  0.113],
                [-0.000, -0.025, -0.240],
                [-0.000, -0.025,  0.240],
                [ 0.243, -0.104,  0.000],
                [-0.080, -0.000, -0.156]
            ], dtype=np.float32)
            
            # Load each anchor image and set up its features
            for viewpoint in self.class_names:
                # Load and resize the anchor image
                anchor_path = anchor_paths[viewpoint]
                anchor_image = cv2.imread(anchor_path)
                if anchor_image is None:
                    logger.warning(f"Failed to load anchor image for viewpoint {viewpoint} at {anchor_path}")
                    logger.warning(f"Falling back to main anchor for {viewpoint}")
                    anchor_image = cv2.imread(self.opt.anchor)
                    if anchor_image is None:
                        raise ValueError(f"Cannot load anchor image from {self.opt.anchor}")
                
                anchor_image = self._resize_image(anchor_image, self.opt.resize)
                
                # Store the anchor image
                self.anchor_data[viewpoint] = {
                    'image': anchor_image,
                    'keypoints_2D': anchor_keypoints_2D_base,  # Use the same for all viewpoints for now
                    'keypoints_3D': anchor_keypoints_3D_base   # Use the same for all viewpoints for now
                }
                
                # Set up anchor features for this viewpoint
                self._set_anchor_features_for_viewpoint(viewpoint)
                
            # Set the initial viewpoint to the first class
            self.current_viewpoint = self.class_names[0]
            logger.info(f"Successfully set up {len(self.anchor_data)} anchor images for different viewpoints")
            
        except Exception as e:
            logger.error(f"Error setting up anchor images: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _set_anchor_features_for_viewpoint(self, viewpoint):
        """Extract SuperPoint features for a specific viewpoint's anchor"""
        anchor_data = self.anchor_data[viewpoint]
        anchor_image = anchor_data['image']
        anchor_keypoints_2D = anchor_data['keypoints_2D']
        anchor_keypoints_3D = anchor_data['keypoints_3D']
        
        try:
            logger.info(f"Extracting features for {viewpoint} anchor image")
            
            # Get the lock for feature extraction
            lock_acquired = self.session_lock.acquire(timeout=10.0)
            if not lock_acquired:
                logger.error(f"Could not acquire session lock for {viewpoint} anchor feature extraction (timeout)")
                return False
            
            try:
                # Process the anchor image with SuperPoint
                with torch.no_grad():
                    anchor_tensor = self._convert_cv2_to_tensor(anchor_image)
                    anchor_feats = self.extractor.extract(anchor_tensor)
                
                # Get the keypoints from SuperPoint
                anchor_keypoints_sp = anchor_feats['keypoints'][0].detach().cpu().numpy()
                
                if len(anchor_keypoints_sp) == 0:
                    logger.error(f"No keypoints detected in {viewpoint} anchor image!")
                    return False
                
                # Build KDTree to match anchor_keypoints_sp -> known anchor_keypoints_2D
                sp_tree = cKDTree(anchor_keypoints_sp)
                distances, indices = sp_tree.query(anchor_keypoints_2D, k=1)
                valid_matches = distances < 5.0  # Increased threshold for "close enough"
                
                if not np.any(valid_matches):
                    logger.error(f"No valid matches found for {viewpoint} between anchor keypoints and 2D points!")
                    return False
                
                # Store the matched indices and 3D keypoints
                matched_anchor_indices = indices[valid_matches]
                matched_3D_keypoints = anchor_keypoints_3D[valid_matches]
                
                # Store all the data in the anchor_data dictionary
                self.anchor_data[viewpoint].update({
                    'feats': anchor_feats,
                    'keypoints_sp': anchor_keypoints_sp,
                    'matched_anchor_indices': matched_anchor_indices,
                    'matched_3D_keypoints': matched_3D_keypoints
                })
                
                logger.info(f"Successfully extracted features for {viewpoint} anchor with {len(matched_anchor_indices)} matched keypoints")
                return True
                
            finally:
                self.session_lock.release()
                
        except Exception as e:
            logger.error(f"Error during {viewpoint} anchor feature extraction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Make sure lock is released if we acquired it and an exception occurred
            try:
                if hasattr(self.session_lock, '_is_owned') and self.session_lock._is_owned():
                    self.session_lock.release()
            except Exception as release_error:
                logger.error(f"Error releasing lock: {release_error}")
            
            return False

    def predict_viewpoint(self, frame):
        """
        Use the MobileViT model to predict the viewpoint of the current frame
        Returns the viewpoint class name and confidence
        """
        try:
            # Resize frame for model input (256x256)
            frame_resized = cv2.resize(frame, (256, 256))
            img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            
            # Apply transformations and prepare for model
            input_tensor = self.vit_transform(img_pil).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.vit_model(input_tensor)
                probs = F.softmax(output, dim=1)[0]
                pred = torch.argmax(probs).item()
                label = self.class_names[pred]
                confidence = probs[pred].item()
            
            # Free GPU memory
            del input_tensor, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return label, confidence
            
        except Exception as e:
            logger.error(f"Error predicting viewpoint: {e}")
            # Return the current viewpoint with zero confidence on error
            return self.current_viewpoint, 0.0

    def reinitialize_anchor(self, new_anchor_path, new_2d_points, new_3d_points):
        """
        Re-load a new anchor image and re-compute relevant data (2D->3D correspondences).
        Called on-the-fly (e.g. after 200 frames).
        """
        try:
            logger.info(f"Re-initializing anchor with new image: {new_anchor_path}")

            # 1. Load new anchor image
            new_anchor_image = cv2.imread(new_anchor_path)
            if new_anchor_image is None:
                logger.error(f"Failed to load new anchor image at {new_anchor_path}")
                raise ValueError(f"Cannot read anchor image from {new_anchor_path}")

            logger.info(f"Successfully loaded anchor image: shape={new_anchor_image.shape}")

            # 2. Resize the image
            new_anchor_image = self._resize_image(new_anchor_image, self.opt.resize)
            logger.info(f"Resized anchor image to {new_anchor_image.shape}")

            # 3. Update anchor image for all viewpoints (with lock)
            lock_acquired = self.session_lock.acquire(timeout=5.0)
            if not lock_acquired:
                logger.error("Could not acquire session lock to update anchor image (timeout)")
                raise TimeoutError("Lock acquisition timed out during anchor update")
                
            try:
                # Determine viewpoint of the new anchor
                viewpoint, _ = self.predict_viewpoint(new_anchor_image)
                
                # Update the anchor image for this viewpoint
                self.anchor_data[viewpoint]['image'] = new_anchor_image
                self.anchor_data[viewpoint]['keypoints_2D'] = new_2d_points
                self.anchor_data[viewpoint]['keypoints_3D'] = new_3d_points
                
                logger.info(f"Anchor image updated for viewpoint {viewpoint}")
            finally:
                self.session_lock.release()

            # 4. Recompute anchor features for this viewpoint
            logger.info(f"Setting anchor features for viewpoint {viewpoint} with {len(new_2d_points)} 2D points and {len(new_3d_points)} 3D points")
            success = self._set_anchor_features_for_viewpoint(viewpoint)
            
            if not success:
                logger.error(f"Failed to set anchor features for viewpoint {viewpoint}")
                raise RuntimeError("Failed to set anchor features")

            logger.info("Anchor re-initialization complete.")
            return True
            
        except Exception as e:
            logger.error(f"Error during anchor reinitialization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _convert_cv2_to_tensor(self, image):
        """Convert OpenCV BGR image to RGB tensor"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to float and normalize to [0, 1]
        image_tensor = torch.from_numpy(image_rgb).float() / 255.0
        # Permute dimensions from (H, W, C) to (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)

    def _set_anchor_features(self, anchor_bgr_image, anchor_keypoints_2D, anchor_keypoints_3D):
        """
        Legacy method for compatibility - delegates to _set_anchor_features_for_viewpoint
        """
        # Determine the viewpoint of this anchor image
        viewpoint, _ = self.predict_viewpoint(anchor_bgr_image)
        
        # Update the anchor data for this viewpoint
        self.anchor_data[viewpoint]['image'] = anchor_bgr_image
        self.anchor_data[viewpoint]['keypoints_2D'] = anchor_keypoints_2D
        self.anchor_data[viewpoint]['keypoints_3D'] = anchor_keypoints_3D
        
        # Extract features
        return self._set_anchor_features_for_viewpoint(viewpoint)

    def _resize_image(self, image, resize):
        logger.debug("Resizing image")
        if len(resize) == 2:
            return cv2.resize(image, tuple(resize))
        elif len(resize) == 1 and resize[0] > 0:
            h, w = image.shape[:2]
            scale = resize[0] / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(image, new_size)
        return image

    def process_frame(self, frame, frame_idx):
        """
        Process a frame to estimate the pose with dynamic viewpoint-based anchor selection
        """
        logger.info(f"Processing frame {frame_idx}")
        start_time = time.time()
        
        # Resize frame to target resolution
        frame = self._resize_image(frame, self.opt.resize)
        
        # Predict the viewpoint of the current frame
        viewpoint, confidence = self.predict_viewpoint(frame)
        
        # Store the viewpoint info for logging
        self.prev_viewpoint = self.current_viewpoint
        self.current_viewpoint = viewpoint
        self.viewpoint_confidence = confidence
        
        logger.info(f"Frame {frame_idx}: Predicted viewpoint {viewpoint} with confidence {confidence:.2f}")
        
        # Ensure gradients are disabled
        with torch.no_grad():
            # Extract features from the frame
            frame_tensor = self._convert_cv2_to_tensor(frame)
            frame_feats = self.extractor.extract(frame_tensor)

        # Get keypoints from current frame
        frame_keypoints = frame_feats["keypoints"][0].detach().cpu().numpy()
        
        # For every frame, perform PnP with the appropriate anchor image based on viewpoint
        anchor_data = self.anchor_data[viewpoint]
        
        # Perform pure PnP estimation using the selected viewpoint anchor
        pnp_pose_data, visualization, mkpts0, mkpts1, mpts3D = self.perform_pnp_estimation_with_viewpoint(
            frame, frame_idx, frame_feats, frame_keypoints, viewpoint
        )
        
        # Check if PnP succeeded
        if pnp_pose_data is None or pnp_pose_data.get('pose_estimation_failed', True):
            logger.warning(f"PnP failed for frame {frame_idx} with viewpoint {viewpoint}")
            
            # If KF initialized, try using prediction
            if self.kf_initialized:
                # Get the prediction from the filter
                x_pred, P_pred = self.mekf.predict()
                
                # Extract prediction state
                position_pred = x_pred[0:3]
                quaternion_pred = x_pred[6:10]
                R_pred = quaternion_to_rotation_matrix(quaternion_pred)
                
                # Create pose data from prediction
                pose_data = {
                    'frame': frame_idx,
                    'kf_translation_vector': position_pred.tolist(),
                    'kf_quaternion': quaternion_pred.tolist(),
                    'kf_rotation_matrix': R_pred.tolist(),
                    'pose_estimation_failed': True,
                    'tracking_method': 'prediction',
                    'viewpoint': viewpoint,
                    'viewpoint_confidence': confidence
                }
                
                # Create simple visualization
                visualization = frame.copy()
                cv2.putText(visualization, f"PnP Failed - Using Prediction ({viewpoint})", 
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                return pose_data, visualization
            else:
                # No KF initialized, return failure
                return {
                    'frame': frame_idx,
                    'pose_estimation_failed': True,
                    'tracking_method': 'failed',
                    'viewpoint': viewpoint,
                    'viewpoint_confidence': confidence
                }, frame
        
        # PnP succeeded - extract pose data
        reprojection_error = pnp_pose_data['mean_reprojection_error']
        num_inliers = pnp_pose_data['num_inliers']
        
        # Extract raw PnP pose
        tvec = np.array(pnp_pose_data['object_translation_in_cam'])
        R = np.array(pnp_pose_data['object_rotation_in_cam'])
        q = rotation_matrix_to_quaternion(R)
        
        # Check if KF is already initialized
        if not self.kf_initialized:
            # Initialize Kalman filter if PnP is good enough
            if reprojection_error < 3.0 and num_inliers >= 6:
                self.mekf = MultExtendedKalmanFilter(dt=1.0/30.0)
                
                # Set initial state
                x_init = np.zeros(self.mekf.n_states)
                x_init[0:3] = tvec  # Position
                x_init[6:10] = q    # Quaternion
                self.mekf.x = x_init
                
                self.kf_initialized = True
                logger.info(f"MEKF initialized with PnP pose (error: {reprojection_error:.2f}, inliers: {num_inliers})")
                
                # Add viewpoint information to the PnP results
                pnp_pose_data['viewpoint'] = viewpoint
                pnp_pose_data['viewpoint_confidence'] = confidence
                
                # Just return the raw PnP results for the first frame
                return pnp_pose_data, visualization
            else:
                # PnP not good enough for initialization
                logger.warning(f"PnP pose not good enough for KF initialization: " +
                            f"error={reprojection_error:.2f}px, inliers={num_inliers}")
                
                # Add viewpoint information to the PnP results
                pnp_pose_data['viewpoint'] = viewpoint
                pnp_pose_data['viewpoint_confidence'] = confidence
                
                return pnp_pose_data, visualization
        
        # If we get here, KF is initialized and PnP succeeded
        # Predict next state
        x_pred, P_pred = self.mekf.predict()
        
        # Process PnP data for KF update if reliable enough
        if reprojection_error < 4.0 and num_inliers >= 5:
            # Extract inlier points for tightly-coupled update
            inlier_indices = np.array(pnp_pose_data['inliers'])
            feature_points = np.array(pnp_pose_data['mkpts1'])[inlier_indices]
            model_points = np.array(pnp_pose_data['mpts3D'])[inlier_indices]
            
            # Create pose measurement for loosely-coupled update
            pose_measurement = np.concatenate([tvec.flatten(), q])
            
            # Get camera parameters
            K, distCoeffs = self._get_camera_intrinsics()
            
            # Choose update method based on kf_mode
            if self.kf_mode == 'L':
                # Always use loosely-coupled
                x_upd, P_upd = self.mekf.update(pose_measurement)
                update_method = "loosely_coupled"
            elif self.kf_mode == 'T':
                # Only use tightly-coupled, with no fallback
                try:
                    x_upd, P_upd = self.mekf.update_tightly_coupled(
                        feature_points, model_points, K, distCoeffs
                    )
                    update_method = "tightly_coupled"
                except Exception as e:
                    # If tightly-coupled fails, use prediction only instead of falling back
                    logger.warning(f"Tightly-coupled update failed, using prediction only: {e}")
                    # Use prediction values directly
                    x_upd = x_pred.copy()
                    P_upd = P_pred.copy()
                    update_method = "tightly_coupled_failed_prediction"
            else:  # 'auto' or any other value
                # Current behavior (try tightly-coupled first, fallback to loosely-coupled)
                try:
                    x_upd, P_upd = self.mekf.update_tightly_coupled(
                        feature_points, model_points, K, distCoeffs
                    )
                    update_method = "tightly_coupled"
                except Exception as e:
                    logger.warning(f"Tightly-coupled update failed, falling back to loosely-coupled: {e}")
                    x_upd, P_upd = self.mekf.update(pose_measurement)
                    update_method = "loosely_coupled_fallback"
            
            # Extract updated pose
            position_upd = x_upd[0:3]
            quaternion_upd = x_upd[6:10]
            R_upd = quaternion_to_rotation_matrix(quaternion_upd)
            
            logger.info(f"Frame {frame_idx}: KF updated with PnP pose " +
                    f"(error: {reprojection_error:.2f}px, inliers: {num_inliers})")
            
            # Create pose data using KF-updated pose
            pose_data = {
                'frame': frame_idx,
                'kf_translation_vector': position_upd.tolist(),
                'kf_quaternion': quaternion_upd.tolist(),
                'kf_rotation_matrix': R_upd.tolist(),
                'raw_pnp_translation': tvec.flatten().tolist(),
                'raw_pnp_rotation': R.tolist(),
                'pose_estimation_failed': False,
                'num_inliers': num_inliers,
                'reprojection_error': reprojection_error,
                'tracking_method': update_method,
                'viewpoint': viewpoint,
                'viewpoint_confidence': confidence,
                'viewpoint_changed': viewpoint != self.prev_viewpoint
            }
            
            # Create visualization with KF pose
            K, distCoeffs = self._get_camera_intrinsics()
            inliers = np.array(pnp_pose_data['inliers'])
            
            # Use all keypoints for visualization
            visualization = frame.copy()

            # Add KF-specific information to visualization
            cv2.putText(visualization, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(visualization, f"Viewpoint: {viewpoint} ({confidence:.2f})", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(visualization, f"Reprojection Error: {reprojection_error:.2f}px", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(visualization, f"Inliers: {num_inliers}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(visualization, f"Method: {update_method}", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Define axis parameters
            axis_length = 0.1  # 10cm for visualization
            axis_points = np.float32([
                [0, 0, 0],
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, axis_length]
            ])

            # DRAW RAW PNP AXES
            rvec_raw, _ = cv2.Rodrigues(R)  # R is the raw PnP rotation
            axis_proj_raw, _ = cv2.projectPoints(axis_points, rvec_raw, tvec.reshape(3, 1), K, distCoeffs)
            axis_proj_raw = axis_proj_raw.reshape(-1, 2)
            origin_raw = tuple(map(int, axis_proj_raw[0]))

            # Draw raw PnP axes with thinner lines
            visualization = cv2.line(visualization, origin_raw, tuple(map(int, axis_proj_raw[1])), (0, 0, 255), 2)  # X-axis (red)
            visualization = cv2.line(visualization, origin_raw, tuple(map(int, axis_proj_raw[2])), (0, 255, 0), 2)  # Y-axis (green)
            visualization = cv2.line(visualization, origin_raw, tuple(map(int, axis_proj_raw[3])), (255, 0, 0), 2)  # Z-axis (blue)

            # Add label for raw PnP axes
            cv2.putText(visualization, "Raw PnP", (origin_raw[0] + 5, origin_raw[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # DRAW KALMAN FILTER AXES
            rvec_upd, _ = cv2.Rodrigues(R_upd)  # R_upd is the Kalman filter rotation
            axis_proj_kf, _ = cv2.projectPoints(axis_points, rvec_upd, position_upd.reshape(3, 1), K, distCoeffs)
            axis_proj_kf = axis_proj_kf.reshape(-1, 2)
            origin_kf = tuple(map(int, axis_proj_kf[0]))

            # Draw KF axes with thicker lines
            visualization = cv2.line(visualization, origin_kf, tuple(map(int, axis_proj_kf[1])), (0, 0, 100), 3)  # X-axis (darker red)
            visualization = cv2.line(visualization, origin_kf, tuple(map(int, axis_proj_kf[2])), (0, 100, 0), 3)  # Y-axis (darker green)
            visualization = cv2.line(visualization, origin_kf, tuple(map(int, axis_proj_kf[3])), (100, 0, 0), 3)  # Z-axis (darker blue)

            # Add label for KF axes
            cv2.putText(visualization, "Kalman Filter", (origin_kf[0] + 5, origin_kf[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Calculate and display rotation difference
            raw_quat = rotation_matrix_to_quaternion(R)
            kf_quat = quaternion_upd
            dot_product = min(1.0, abs(np.dot(raw_quat, kf_quat)))
            angle_diff = np.arccos(dot_product) * 2 * 180 / np.pi
            cv2.putText(visualization, f"Rot Diff: {angle_diff:.1f}°", 
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # If we have a previous loosely-coupled update for comparison
            if hasattr(self, 'last_loose_update') and update_method == 'tightly_coupled':
                # Get the last loose update for comparison
                pos_loose = self.last_loose_update[0:3]
                quat_loose = self.last_loose_update[6:10]
                R_loose = quaternion_to_rotation_matrix(quat_loose)

                # Calculate position difference
                pos_diff = np.linalg.norm(position_upd - pos_loose)
                
                # Draw loose update axes in different colors
                rvec_loose, _ = cv2.Rodrigues(R_loose)
                axis_proj_loose, _ = cv2.projectPoints(axis_points, rvec_loose, pos_loose.reshape(3, 1), K, distCoeffs)
                axis_proj_loose = axis_proj_loose.reshape(-1, 2)
                origin_loose = tuple(map(int, axis_proj_loose[0]))

                # Draw loose axes with yellow/purple/cyan colors
                visualization = cv2.line(visualization, origin_loose, tuple(map(int, axis_proj_loose[1])), (0, 220, 220), 2)  # X-axis (cyan)
                visualization = cv2.line(visualization, origin_loose, tuple(map(int, axis_proj_loose[2])), (220, 220, 0), 2)  # Y-axis (yellow)
                visualization = cv2.line(visualization, origin_loose, tuple(map(int, axis_proj_loose[3])), (220, 0, 220), 2)  # Z-axis (purple)

                # Add label for loose KF
                cv2.putText(visualization, "Loose KF", (origin_loose[0] + 5, origin_loose[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 0), 1)
                
                # Display position difference
                cv2.putText(visualization, f"TC vs LC diff: {pos_diff:.3f}m", 
                            (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Store current update for comparison next frame
            if 'update_method' in locals():
                if update_method == 'loosely_coupled' or update_method == 'loosely_coupled_fallback':
                    self.last_loose_update = x_upd.copy()
                    
            return pose_data, visualization
        else:
            # PnP successful but not reliable enough - use KF prediction only
            logger.warning(f"Frame {frame_idx}: PnP pose not reliable enough for KF update: " +
                        f"error={reprojection_error:.2f}px, inliers={num_inliers}")
            
            # Extract prediction state
            position_pred = x_pred[0:3]
            quaternion_pred = x_pred[6:10]
            R_pred = quaternion_to_rotation_matrix(quaternion_pred)
            
            # Create pose data from prediction
            pose_data = {
                'frame': frame_idx,
                'kf_translation_vector': position_pred.tolist(),
                'kf_quaternion': quaternion_pred.tolist(),
                'kf_rotation_matrix': R_pred.tolist(),
                'raw_pnp_translation': tvec.flatten().tolist(),
                'raw_pnp_rotation': R.tolist(),
                'pose_estimation_failed': False,
                'tracking_method': 'prediction',
                'pnp_result': 'not_reliable_enough',
                'viewpoint': viewpoint,
                'viewpoint_confidence': confidence,
                'viewpoint_changed': viewpoint != self.prev_viewpoint
            }
            
            # Create simple visualization
            visualization = frame.copy()
            cv2.putText(visualization, f"PnP Not Reliable - Using Prediction ({viewpoint})", 
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return pose_data, visualization