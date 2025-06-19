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

# Remove LightGlue imports - we'll use XFeat's built-in LighterGlue
# from lightglue import LightGlue, SuperPoint
# from lightglue.utils import rbd

from collections import defaultdict
import statistics

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

# 성능 프로파일러 클래스 추가
class PoseProfiler:
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timers = {}
        
    def start_timer(self, name):
        self.current_timers[name] = time.time()
        
    def end_timer(self, name):
        if name in self.current_timers:
            elapsed = time.time() - self.current_timers[name]
            self.timings[name].append(elapsed * 1000)
            del self.current_timers[name]
            return elapsed * 1000
        return 0
        
    def print_detailed_stats(self, frame_idx):
        if frame_idx % 20 == 0:  # 20프레임마다
            print(f"\n🎯 === POSE ESTIMATION DETAILS (Frame {frame_idx}) ===")
            for name, times in self.timings.items():
                if not times:
                    continue
                recent = times[-10:]
                avg = statistics.mean(recent)
                if avg > 5:  # 5ms 이상만 표시
                    emoji = "🔴" if avg > 50 else "🟡" if avg > 20 else "🟢"
                    print(f"{emoji} {name:25} | {avg:6.1f}ms")

# Global profiler
pose_profiler = PoseProfiler()


class PoseEstimator:
    def __init__(self, opt, device, kf_mode='auto'):
        self.session_lock = threading.Lock()
        self.opt = opt
        self.device = device
        self.kf_mode = kf_mode  # Store the KF mode
        self.initial_z_set = False  # Flag for first-frame Z override (if desired)
        self.kf_initialized = False  # To track if Kalman filter was ever updated
        self.pred_only = 0

        logger.info("Initializing PoseEstimator with XFeat and LighterGlue models")

        # Load anchor (leader) image
        self.anchor_image = cv2.imread(opt.anchor)
        
        assert self.anchor_image is not None, f'Failed to load anchor image at {opt.anchor}'

        self.anchor_image = cv2.resize(self.anchor_image, (1280, 720))
        print(f"DEBUG AFTER RESIZE: Anchor image shape: {self.anchor_image.shape}")

        logger.info(f"Loaded and resized anchor image from {opt.anchor}")

        # Initialize XFeat model
        start_time = time.time()
        
        # Load XFeat model using torch.hub
        print("Loading XFeat model...")
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=1024)
        self.xfeat.to(device)
        self.xfeat.eval()
        
        # # 반정밀도 최적화 (GPU 메모리 절약) - XFeat may not support half precision
        # if torch.cuda.is_available():
        #     try:
        #         self.xfeat = self.xfeat.half()
        #     except:
        #         logger.warning("XFeat does not support half precision, using full precision")
            
        init_time = (time.time() - start_time) * 1000
        print(f"✅ XFeat initialized in {init_time:.1f}ms")
        logger.info("Initialized XFeat model")

        # Anchor keypoints (same as before)
        anchor_keypoints_2D = np.array([
            [511, 293], [591, 284], [587, 330], [413, 249], [602, 348], [715, 384], [598, 298], [656, 171], [805, 213],
            [703, 392], [523, 286], [519, 327], [387, 289], [727, 126], [425, 243], [636, 358], [745, 202], [595, 388],
            [436, 260], [539, 313], [795, 220], [351, 291], [665, 165], [611, 353], [650, 377], [516, 389], [727, 143],
            [496, 378], [575, 312], [617, 368], [430, 312], [480, 281], [834, 225], [469, 339], [705, 223], [637, 156],
            [816, 414], [357, 195], [752, 77], [642, 451]
        ], dtype=np.float32)

        anchor_keypoints_3D = np.array([
            [-0.014,  0.000,  0.042], [ 0.025, -0.014, -0.011], [-0.014,  0.000, -0.042], [-0.014,  0.000,  0.156],
            [-0.023,  0.000, -0.065], [ 0.000,  0.000, -0.156], [ 0.025,  0.000, -0.015], [ 0.217,  0.000,  0.070],
            [ 0.230,  0.000, -0.070], [-0.014,  0.000, -0.156], [ 0.000,  0.000,  0.042], [-0.057, -0.018, -0.010],
            [-0.074, -0.000,  0.128], [ 0.206, -0.070, -0.002], [-0.000, -0.000,  0.156], [-0.017, -0.000, -0.092],
            [ 0.217, -0.000, -0.027], [-0.052, -0.000, -0.097], [-0.019, -0.000,  0.128], [-0.035, -0.018, -0.010],
            [ 0.217, -0.000, -0.070], [-0.080, -0.000,  0.156], [ 0.230, -0.000,  0.070], [-0.023, -0.000, -0.075],
            [-0.029, -0.000, -0.127], [-0.090, -0.000, -0.042], [ 0.206, -0.055, -0.002], [-0.090, -0.000, -0.015],
            [ 0.000, -0.000, -0.015], [-0.037, -0.000, -0.097], [-0.074, -0.000,  0.074], [-0.019, -0.000,  0.074],
            [ 0.230, -0.000, -0.113], [-0.100, -0.030,  0.000], [ 0.170, -0.000, -0.015], [ 0.230, -0.000,  0.113],
            [-0.000, -0.025, -0.240], [-0.000, -0.025,  0.240], [ 0.243, -0.104,  0.000], [-0.080, -0.000, -0.156]
        ], dtype=np.float32)

        # Set anchor features (run XFeat on anchor, match to known 2D->3D)
        self._set_anchor_features(
            anchor_bgr_image=self.anchor_image,
            anchor_keypoints_2D=anchor_keypoints_2D,
            anchor_keypoints_3D=anchor_keypoints_3D
        )

        # Suppose the anchor was taken at ~ yaw=0°, pitch=-20°, roll=0°, in radians:
        self.anchor_viewpoint_eulers = np.array([0.0, -0.35, 0.0], dtype=np.float32)

        # Add these lines:
        self.kf_initialized = False
        self.tracking_3D_points = None
        self.tracking_2D_points = None
        
        # Replace your existing Kalman filter init with MEKF
        # We'll initialize it properly when we get the first good pose
        self.mekf = None

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
            print("Resized anchor image to {new_anchor_image.shape}\n")

            # 3. Update anchor image (with lock)
            lock_acquired = self.session_lock.acquire(timeout=5.0)
            if not lock_acquired:
                logger.error("Could not acquire session lock to update anchor image (timeout)")
                raise TimeoutError("Lock acquisition timed out during anchor update")
                
            try:
                self.anchor_image = new_anchor_image
                logger.info("Anchor image updated")
            finally:
                self.session_lock.release()

            # 4. Recompute anchor features with the new image and 2D/3D
            logger.info(f"Setting anchor features with {len(new_2d_points)} 2D points and {len(new_3d_points)} 3D points")
            success = self._set_anchor_features(
                anchor_bgr_image=new_anchor_image,
                anchor_keypoints_2D=new_2d_points,
                anchor_keypoints_3D=new_3d_points
            )
            
            if not success:
                logger.error("Failed to set anchor features")
                raise RuntimeError("Failed to set anchor features")

            logger.info("Anchor re-initialization complete.")
            return True
            
        except Exception as e:
            logger.error(f"Error during anchor reinitialization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _convert_cv2_to_xfeat_input(self, image):
        """Convert OpenCV BGR image to RGB numpy array for XFeat"""
        # XFeat expects RGB images in numpy format, not tensors
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    def _set_anchor_features(self, anchor_bgr_image, anchor_keypoints_2D, anchor_keypoints_3D):
        """
        Run XFeat on the anchor image to get anchor_keypoints_xfeat.
        Then match those keypoints to known 2D->3D correspondences via KDTree.
        """
        try:
            # Record the start time
            start_time = time.time()
            logger.info("Starting anchor feature extraction with XFeat...")

            # Print anchor image shape for debugging
            print(f"DEBUG: Anchor image shape: {anchor_bgr_image.shape}")
            expected_width, expected_height = 1280, 720
            
            # Scale keypoints if anchor image is not at expected size
            actual_width, actual_height = anchor_bgr_image.shape[1], anchor_bgr_image.shape[0]
            
            if actual_width != expected_width or actual_height != expected_height:
                scale_x = actual_width / expected_width
                scale_y = actual_height / expected_height
                print(f"DEBUG: Scaling keypoints by {scale_x:.2f}x, {scale_y:.2f}y")
                
                # Create a copy and scale
                scaled_keypoints = anchor_keypoints_2D.copy()
                scaled_keypoints[:, 0] *= scale_x
                scaled_keypoints[:, 1] *= scale_y
                anchor_keypoints_2D = scaled_keypoints
            
            print(f"DEBUG: Using 2D keypoints in range x=[{np.min(anchor_keypoints_2D[:,0]):.1f}-{np.max(anchor_keypoints_2D[:,0]):.1f}], y=[{np.min(anchor_keypoints_2D[:,1]):.1f}-{np.max(anchor_keypoints_2D[:,1]):.1f}]")
            
            # Try to acquire the lock with a timeout
            lock_acquired = self.session_lock.acquire(timeout=10.0)
            if not lock_acquired:
                logger.error("Could not acquire session lock for anchor feature extraction (timeout)")
                return False
            
            try:
                # Convert BGR to RGB for XFeat
                anchor_rgb = self._convert_cv2_to_xfeat_input(anchor_bgr_image)
                
                logger.info("Processing anchor image with XFeat...")
                with torch.no_grad():
                    # XFeat detectAndCompute returns (output, _) where output contains keypoints and descriptors
                    self.anchor_feats = self.xfeat.detectAndCompute(anchor_rgb, top_k=1024)[0]
                    
                    # Update with image size (required for matching)
                    self.anchor_feats.update({'image_size': (anchor_rgb.shape[1], anchor_rgb.shape[0])})
                    
                    logger.info(f"Anchor features extracted in {time.time() - start_time:.3f}s")
                
                # Get anchor keypoints - XFeat returns keypoints in 'keypoints' field
                self.anchor_keypoints_xfeat = self.anchor_feats['keypoints'].detach().cpu().numpy()
                
                if len(self.anchor_keypoints_xfeat) == 0:
                    logger.error("No keypoints detected in anchor image!")
                    return False
                
                # Print shape and sample of keypoints for debugging
                logger.info(f"Anchor keypoints shape: {self.anchor_keypoints_xfeat.shape}")
                if len(self.anchor_keypoints_xfeat) > 5:
                    logger.info(f"First 5 keypoints: {self.anchor_keypoints_xfeat[:5]}")
                
                # Build KDTree to match anchor_keypoints_xfeat -> known anchor_keypoints_2D
                logger.info("Building KDTree for anchor keypoints...")
                xfeat_tree = cKDTree(self.anchor_keypoints_xfeat)
                distances, indices = xfeat_tree.query(anchor_keypoints_2D, k=1)

                # Print distances for debugging
                print(f"DEBUG: KDTree distances min/max/avg: {np.min(distances)}/{np.max(distances)}/{np.mean(distances)}")
                
                valid_matches = distances < 5.0  # Increased threshold for "close enough"
                
                logger.info(f"KDTree query completed in {time.time() - start_time:.3f}s")
                logger.info(f"Valid matches: {sum(valid_matches)} out of {len(anchor_keypoints_2D)}")
                print(f"DEBUG: Valid matches: {sum(valid_matches)} out of {len(anchor_keypoints_2D)}")
                
                # Check if we have any valid matches
                if not np.any(valid_matches):
                    logger.error("No valid matches found between anchor keypoints and 2D points!")
                    # Initialize with empty arrays to prevent None attribute errors
                    self.matched_anchor_indices = np.array([], dtype=np.int64)
                    self.matched_3D_keypoints = np.array([], dtype=np.float32).reshape(0, 3)
                    return False
                
                self.matched_anchor_indices = indices[valid_matches]
                self.matched_3D_keypoints = anchor_keypoints_3D[valid_matches]
                
                logger.info(f"Matched {len(self.matched_anchor_indices)} keypoints to 3D points")
                logger.info(f"Anchor feature extraction completed in {time.time() - start_time:.3f}s")
                
                return True
                
            finally:
                # Always release the lock in the finally block to ensure it gets released
                # even if an exception occurs
                self.session_lock.release()
                logger.debug("Released session lock after anchor feature extraction")
                
        except Exception as e:
            logger.error(f"Error during anchor feature extraction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Make sure lock is released if we acquired it and an exception occurred
            # outside the 'with' block
            try:
                if hasattr(self.session_lock, '_is_owned') and self.session_lock._is_owned():
                    self.session_lock.release()
                    logger.debug("Released session lock after exception")
            except Exception as release_error:
                logger.error(f"Error releasing lock: {release_error}")
            
            return False

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

    def process_frame(self, frame, frame_idx, bbox=None, viewpoint=None):
        """
        Process a frame to estimate the pose using XFeat + LighterGlue
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = frame[y1:y2, x1:x2]
            logger.info(f"Processing frame {frame_idx}")

            print(f"DEBUG: Input frame {frame_idx} shape before resize: {frame.shape}")

            # === 전체 처리 시간 측정 시작 ===
            pose_profiler.start_timer('total_pose_processing')
            
            # === XFeat 특징 추출 ===
            pose_profiler.start_timer('xfeat_total')

            # Ensure gradients are disabled
            with torch.no_grad():
                # Convert frame to RGB for XFeat
                frame_rgb = self._convert_cv2_to_xfeat_input(frame)
                
                # Extract features from the frame using XFeat
                frame_feats = self.xfeat.detectAndCompute(frame_rgb, top_k=1024)[0]
                
                # Update with image size (required for matching)
                frame_feats.update({'image_size': (frame_rgb.shape[1], frame_rgb.shape[0])})

            # Get keypoints from current frame
            frame_keypoints = frame_feats["keypoints"].detach().cpu().numpy()
            xfeat_time = pose_profiler.end_timer('xfeat_total')

            # For every frame, do PnP with the anchor image
            # Perform pure PnP estimation
            pose_profiler.start_timer('pnp_total')
            pnp_pose_data, visualization, mkpts0, mkpts1, mpts3D = self.perform_pnp_estimation(
                frame, frame_idx, frame_feats, frame_keypoints
            )
            pnp_time = pose_profiler.end_timer('pnp_total')
            
            # Check if PnP succeeded
            if pnp_pose_data is None or pnp_pose_data.get('pose_estimation_failed', True):
                logger.warning(f"PnP failed for frame {frame_idx}")
                
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
                        'tracking_method': 'prediction'
                    }
                    
                    # Create simple visualization
                    visualization = frame.copy()
                    cv2.putText(visualization, "PnP Failed - Using Prediction", 
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    return pose_data, visualization
                else:
                    # No KF initialized, return failure
                    return {
                        'frame': frame_idx,
                        'pose_estimation_failed': True,
                        'tracking_method': 'failed'
                    }, frame
            
            # PnP succeeded - extract pose data
            reprojection_error = pnp_pose_data['mean_reprojection_error']
            num_inliers = pnp_pose_data['num_inliers']
            
            # Extract raw PnP pose
            tvec = np.array(pnp_pose_data['object_translation_in_cam'])
            R = np.array(pnp_pose_data['object_rotation_in_cam'])
            q = rotation_matrix_to_quaternion(R)

            pose_profiler.start_timer('kalman_total')
            
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
                    
                    # Just return the raw PnP results for the first frame
                    return pnp_pose_data, visualization
                else:
                    # PnP not good enough for initialization
                    logger.warning(f"PnP pose not good enough for KF initialization: " +
                                f"error={reprojection_error:.2f}px, inliers={num_inliers}")
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
                
                x_upd, P_upd = self.mekf.update(pose_measurement)
                update_method = "loosely_coupled"
                
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
                    'tracking_method': update_method
                }
                
                # Create visualization with KF pose
                K, distCoeffs = self._get_camera_intrinsics()
                inliers = np.array(pnp_pose_data['inliers'])
                
                # Use simplified visualization
                visualization = self._create_simple_visualization(frame, pose_data, frame_idx)

                kf_time = pose_profiler.end_timer('kalman_total')
                total_time = pose_profiler.end_timer('total_pose_processing')

                # 성능 통계 (주기적으로만)
                if frame_idx % 30 == 0:
                    print(f"📊 Frame {frame_idx}: XFeat {xfeat_time:.1f}ms, PnP {pnp_time:.1f}ms, "
                        f"KF {kf_time:.1f}ms, Total {total_time:.1f}ms")
                
                # 상세 프로파일링 출력
                pose_profiler.print_detailed_stats(frame_idx)

                # Draw coordinate axes for both raw PnP and KF results
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
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
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
                    'pnp_result': 'not_reliable_enough'
                }
                
                # Create simple visualization
                visualization = frame.copy()
                cv2.putText(visualization, "PnP Not Reliable - Using Prediction", 
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                return pose_data, visualization

    def _init_kalman_filter(self):
        frame_rate = 30
        dt = 1 / frame_rate
        kf_pose = MultExtendedKalmanFilter(dt)
        return kf_pose

    def _create_simple_visualization(self, frame, pose_data, frame_idx):
        """간단한 시각화 생성 (성능 최적화)"""
        vis = frame.copy()
        
        # 기본 정보만 표시
        cv2.putText(vis, f"Frame: {frame_idx}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if 'reprojection_error' in pose_data:
            error = pose_data['reprojection_error']
            color = (0, 255, 0) if error < 3.0 else (0, 255, 255) if error < 5.0 else (0, 0, 255)
            cv2.putText(vis, f"Error: {error:.1f}px", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        if 'num_inliers' in pose_data:
            inliers = pose_data['num_inliers']
            cv2.putText(vis, f"Inliers: {inliers}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 좌표축 그리기 (간단화)
        try:
            K, distCoeffs = self._get_camera_intrinsics()
            R_upd = np.array(pose_data['kf_rotation_matrix'])
            position_upd = np.array(pose_data['kf_translation_vector'])
            
            axis_length = 0.08  # 더 작게
            axis_points = np.float32([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,axis_length]])
            
            rvec, _ = cv2.Rodrigues(R_upd)
            axis_proj, _ = cv2.projectPoints(axis_points, rvec, position_upd.reshape(3,1), K, distCoeffs)
            axis_proj = axis_proj.reshape(-1, 2)
            
            origin = tuple(map(int, axis_proj[0]))
            
            # 축 그리기 (얇게)
            vis = cv2.line(vis, origin, tuple(map(int, axis_proj[1])), (0, 0, 255), 2)  # X (빨강)
            vis = cv2.line(vis, origin, tuple(map(int, axis_proj[2])), (0, 255, 0), 2)  # Y (초록)
            vis = cv2.line(vis, origin, tuple(map(int, axis_proj[3])), (255, 0, 0), 2)  # Z (파랑)
            
        except:
            pass  # 시각화 실패시 무시
            
        return vis

    def _visualize_tracking(self, frame, feature_points_or_inliers, model_points_or_pose_data, state_or_frame_idx, extra_info=None):
        """
        Unified visualization function for both initialization (PnP) and tracking modes
        """
        # Make a copy for visualization
        vis_img = frame.copy()
        
        # Determine which mode we're in
        if isinstance(model_points_or_pose_data, dict):
            # Initialization/PnP mode
            pose_data = model_points_or_pose_data
            frame_idx = state_or_frame_idx
            inliers = feature_points_or_inliers
            
            # Unpack extra info if provided
            if extra_info is not None and len(extra_info) >= 3:
                mkpts0, mkpts1, mconf = extra_info[:3]
                frame_keypoints = extra_info[3] if len(extra_info) > 3 else None
                
                # Get inlier points
                if inliers is not None:
                    inlier_idx = inliers.flatten()
                    inlier_mkpts0 = mkpts0[inlier_idx]
                    inlier_mkpts1 = mkpts1[inlier_idx]
                    feature_points = inlier_mkpts1
                    
                    # Draw matched keypoints (green)
                    for pt in inlier_mkpts1:
                        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
            
            # Extract pose from pose_data
            if 'kf_rotation_matrix' in pose_data and 'kf_translation_vector' in pose_data:
                R = np.array(pose_data['kf_rotation_matrix'])
                tvec = np.array(pose_data['kf_translation_vector']).reshape(3, 1)
            else:
                R = np.array(pose_data['object_rotation_in_cam'])
                tvec = np.array(pose_data['object_translation_in_cam']).reshape(3, 1)
                print("PNP&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
                
            # Convert to rotation vector for axis visualization
            rvec, _ = cv2.Rodrigues(R)
            
            # Get reprojection error if available
            reprojection_text = ""
            if 'mean_reprojection_error' in pose_data:
                mean_error = pose_data['mean_reprojection_error']
                reprojection_text = f"Reprojection Error: {mean_error:.2f}px"
            
            # Get method info
            method_text = "PnP Initialization" if frame_idx == 1 else "PnP Fallback"
            if 'tracking_method' in pose_data:
                if pose_data['tracking_method'] == 'tracking':
                    method_text = "Tracking"
                elif pose_data['tracking_method'] == 'prediction':
                    method_text = "Prediction Only"
        
        else:
            # Tracking mode
            feature_points = feature_points_or_inliers
            model_points = model_points_or_pose_data
            state = state_or_frame_idx
            frame_idx = extra_info  # In tracking mode, frame_idx is passed in extra_info
            
            # Extract pose from MEKF state
            position = state[0:3]
            quaternion = state[6:10]
            R = quaternion_to_rotation_matrix(quaternion)
            tvec = position.reshape(3, 1)
            rvec, _ = cv2.Rodrigues(R)
            
            # Get camera parameters
            K, distCoeffs = self._get_camera_intrinsics()
            
            # Project model points to check reprojection error
            proj_points, _ = cv2.projectPoints(model_points, rvec, tvec, K, distCoeffs)
            proj_points = proj_points.reshape(-1, 2)
            
            # Calculate average reprojection error
            reprojection_errors = np.linalg.norm(proj_points - feature_points, axis=1)
            mean_error = np.mean(reprojection_errors)
            reprojection_text = f"Reprojection Error: {mean_error:.2f}px"
            
            # Draw feature points (green) and projections (red)
            for i, (feat_pt, proj_pt) in enumerate(zip(feature_points, proj_points)):
                # Draw actual feature point
                cv2.circle(vis_img, (int(feat_pt[0]), int(feat_pt[1])), 3, (0, 255, 0), -1)
                
                # Draw projected point
                cv2.circle(vis_img, (int(proj_pt[0]), int(proj_pt[1])), 2, (0, 0, 255), -1)
                
                # Draw line between them
                cv2.line(vis_img, 
                        (int(feat_pt[0]), int(feat_pt[1])), 
                        (int(proj_pt[0]), int(proj_pt[1])), 
                        (255, 0, 255), 1)
            
            method_text = "Tracking"
        
        # Draw coordinate axes (works for both modes)
        K, distCoeffs = self._get_camera_intrinsics()
        axis_length = 0.1  # 10cm for visualization
        axis_points = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ])
        
        axis_proj, _ = cv2.projectPoints(axis_points, rvec, tvec, K, distCoeffs)
        axis_proj = axis_proj.reshape(-1, 2)
        
        # Draw axes
        origin = tuple(map(int, axis_proj[0]))
        vis_img = cv2.line(vis_img, origin, tuple(map(int, axis_proj[1])), (0, 0, 255), 3)  # X-axis (red)
        vis_img = cv2.line(vis_img, origin, tuple(map(int, axis_proj[2])), (0, 255, 0), 3)  # Y-axis (green)
        vis_img = cv2.line(vis_img, origin, tuple(map(int, axis_proj[3])), (255, 0, 0), 3)  # Z-axis (blue)
        
        # Add telemetry
        cv2.putText(vis_img, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_img, reprojection_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add position info
        if isinstance(model_points_or_pose_data, dict):
            if 'kf_translation_vector' in pose_data:
                pos = pose_data['kf_translation_vector']
            else:
                pos = pose_data['object_translation_in_cam']
        else:
            pos = position
        
        pos_text = f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
        cv2.putText(vis_img, pos_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add method info
        status = "Success"
        if isinstance(model_points_or_pose_data, dict) and pose_data.get('pose_estimation_failed', False):
            status = "Failed"
        
        cv2.putText(vis_img, f"Method: {method_text} ({status})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_img

    def _get_camera_intrinsics(self):
        
        ## Calib_webcam ICUAS LAB 20250124
        focal_length_x = 1460.10150  # fx from the calibrated camera matrix
        focal_length_y = 1456.48915  # fy from the calibrated camera matrix
        cx = 604.85462               # cx from the calibrated camera matrix
        cy = 328.64800               # cy from the calibrated camera matrix

        distCoeffs = np.array(
            [3.56447550e-01, -1.09206851e+01, 1.40564820e-03, -1.10856449e-02, 1.20471120e+02],
            dtype=np.float32
        )

        distCoeffs = None

        K = np.array([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K, distCoeffs

    def enhance_pose_initialization(self, initial_pose, mkpts0, mkpts1, mpts3D, frame):
        """
        Enhance initial pose estimate by finding additional correspondences
        """
        rvec, tvec = initial_pose
        K, distCoeffs = self._get_camera_intrinsics()
        
        # Get all keypoints from the current frame using XFeat
        frame_rgb = self._convert_cv2_to_xfeat_input(frame)
        frame_feats = self.xfeat.detectAndCompute(frame_rgb, top_k=1024)[0]
        frame_keypoints = frame_feats['keypoints'].detach().cpu().numpy()
        
        # Project all 3D model points to find additional correspondences
        all_3d_points = self.matched_3D_keypoints  
        
        # Project all 3D model points to the image plane
        projected_points, _ = cv2.projectPoints(
            all_3d_points, rvec, tvec, K, distCoeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Find additional correspondences
        additional_corrs = []
        for i, model_pt in enumerate(all_3d_points):
            # Skip points already in the initial correspondences
            if model_pt in mpts3D:
                continue
                
            proj_pt = projected_points[i]
            
            # Find the closest feature point
            distances = np.linalg.norm(frame_keypoints - proj_pt, axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            # If close enough, consider it a match
            if min_dist < 3.0:  # Threshold in pixels
                additional_corrs.append((i, min_idx))
        
        # If we found additional correspondences, refine the pose
        if additional_corrs:
            # Combine initial correspondences with new ones
            all_3d = list(mpts3D)
            all_2d = list(mkpts1)
            
            for i3d, i2d in additional_corrs:
                all_3d.append(all_3d_points[i3d])
                all_2d.append(frame_keypoints[i2d])
            
            all_3d = np.array(all_3d)
            all_2d = np.array(all_2d)
            
            # Refine pose using all correspondences
            success, refined_rvec, refined_tvec, inliers = cv2.solvePnPRansac(
                objectPoints=all_3d.reshape(-1, 1, 3),
                imagePoints=all_2d.reshape(-1, 1, 2),
                cameraMatrix=K,
                distCoeffs=distCoeffs,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                reprojectionError=4.0,
                iterationsCount=2000,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if success and inliers is not None and len(inliers) >= 6:
                # Further refine with VVS
                refined_rvec, refined_tvec = cv2.solvePnPRefineVVS(
                    objectPoints=all_3d[inliers].reshape(-1, 1, 3),
                    imagePoints=all_2d[inliers].reshape(-1, 1, 2),
                    cameraMatrix=K,
                    distCoeffs=distCoeffs,
                    rvec=refined_rvec,
                    tvec=refined_tvec
                )
                
                return (refined_rvec, refined_tvec), all_3d, all_2d, inliers
        
        # If no additional correspondences or refinement failed, return original pose
        return (rvec, tvec), mpts3D, mkpts1, None

    def perform_pnp_estimation(self, frame, frame_idx, frame_feats, frame_keypoints):
        """
        Perform pure PnP pose estimation using XFeat + LighterGlue matching.
        """
        # Match features between anchor and frame using XFeat's LighterGlue
        with torch.no_grad():
            with self.session_lock:
                # Use XFeat's match_lighterglue method
                mkpts0, mkpts1,_ = self.xfeat.match_lighterglue(self.anchor_feats, frame_feats)

        # Convert to numpy arrays
        mkpts0 = mkpts0.detach().cpu().numpy()
        mkpts1 = mkpts1.detach().cpu().numpy()
        
        if len(mkpts0) == 0:
            logger.warning(f"No matches found for PnP in frame {frame_idx}")
            return None, None, None, None, None

        # For XFeat, we need to map the matched keypoints to our anchor indices
        # Build a mapping from anchor keypoints to our matched_anchor_indices
        anchor_keypoints = self.anchor_keypoints_xfeat
        
        # Find which anchor keypoints correspond to our known 3D points
        valid_matches = []
        valid_mkpts0 = []
        valid_mkpts1 = []
        valid_3d_points = []
        
        for i, (pt0, pt1) in enumerate(zip(mkpts0, mkpts1)):
            # Find the closest anchor keypoint to pt0
            distances = np.linalg.norm(anchor_keypoints - pt0, axis=1)
            closest_idx = np.argmin(distances)
            
            # Check if this keypoint is in our matched_anchor_indices
            if closest_idx in self.matched_anchor_indices and distances[closest_idx] < 2.0:
                # Get the corresponding 3D point
                anchor_idx_position = np.where(self.matched_anchor_indices == closest_idx)[0]
                if len(anchor_idx_position) > 0:
                    valid_matches.append(i)
                    valid_mkpts0.append(pt0)
                    valid_mkpts1.append(pt1)
                    valid_3d_points.append(self.matched_3D_keypoints[anchor_idx_position[0]])

        if len(valid_matches) < 4:
            logger.warning(f"Not enough valid matches for PnP in frame {frame_idx}: {len(valid_matches)}")
            return None, None, None, None, None

        # Convert to numpy arrays
        mkpts0 = np.array(valid_mkpts0)
        mkpts1 = np.array(valid_mkpts1)
        mpts3D = np.array(valid_3d_points)
        
        # Create dummy confidence scores (XFeat doesn't provide explicit confidence)
        mconf = np.ones(len(mkpts0))

        # Get camera intrinsics
        K, distCoeffs = self._get_camera_intrinsics()
        
        # Prepare data for PnP
        objectPoints = mpts3D.reshape(-1, 1, 3)
        imagePoints = mkpts1.reshape(-1, 1, 2).astype(np.float32)

        # Solve initial PnP
        success, rvec_o, tvec_o, inliers = cv2.solvePnPRansac(
            objectPoints=objectPoints,
            imagePoints=imagePoints,
            cameraMatrix=K,
            distCoeffs=distCoeffs,
            reprojectionError=4,
            confidence=0.99,
            iterationsCount=1000,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success or inliers is None or len(inliers) < 5:
            print("INLIERS ARE :\n", len(inliers) if inliers is not None else 0)
            logger.warning("PnP pose estimation failed or not enough inliers.")
            return None, None, None, None, None

        # Enhance the initial pose by finding additional correspondences
        (rvec, tvec), enhanced_3d, enhanced_2d, enhanced_inliers = self.enhance_pose_initialization(
            (rvec_o, tvec_o), mkpts0, mkpts1, mpts3D, frame
        )

        # If enhancement failed, use the original results
        if enhanced_inliers is None:
            # Use the original results
            objectPoints_inliers = objectPoints[inliers.flatten()]
            imagePoints_inliers = imagePoints[inliers.flatten()]
            final_inliers = inliers
            
            # Refine with VVS
            rvec, tvec = cv2.solvePnPRefineVVS(
                objectPoints=objectPoints_inliers,
                imagePoints=imagePoints_inliers,
                cameraMatrix=K,
                distCoeffs=distCoeffs,
                rvec=rvec_o,
                tvec=tvec_o
            )
        else:
            # Use the enhanced results
            objectPoints_inliers = enhanced_3d[enhanced_inliers.flatten()]
            imagePoints_inliers = enhanced_2d[enhanced_inliers.flatten()]
            final_inliers = enhanced_inliers

        # Convert to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Initialize region counters for coverage score
        regions = {"front-right": 0, "front-left": 0, "back-right": 0, "back-left": 0}

        # Classify points into regions
        for point in objectPoints_inliers[:, 0]:
            if point[0] < 0 and point[2] > 0:  # Front-Right
                regions["front-right"] += 1
            elif point[0] < 0 and point[2] < 0:  # Front-Left
                regions["front-left"] += 1
            elif point[0] > 0 and point[2] > 0:  # Back-Right
                regions["back-right"] += 1
            elif point[0] > 0 and point[2] < 0:  # Back-Left
                regions["back-left"] += 1

        # Calculate coverage score
        total_points = sum(regions.values())
        if total_points > 0:
            used_mconf = mconf[final_inliers.flatten()] if len(final_inliers) > 0 else []
            
            if len(used_mconf) == 0 or np.isnan(used_mconf).any():
                coverage_score = 0
            else:
                # Calculate entropy term
                entropy_sum = 0
                for count in regions.values():
                    if count > 0:
                        proportion = count / total_points
                        entropy_sum += proportion * np.log(proportion)
                
                # Normalize by log(4) as specified in the paper
                normalized_entropy = -entropy_sum / np.log(4)
                
                # Final coverage score (using fixed confidence value for simplicity)
                coverage_score = normalized_entropy
                
                # Ensure score is in valid range [0,1]
                coverage_score = np.clip(coverage_score, 0, 1)
                print(f'Coverage score: {coverage_score:.2f}')
        else:
            coverage_score = 0

        # Compute reprojection errors
        projected_points, _ = cv2.projectPoints(
            objectPoints_inliers, rvec, tvec, K, distCoeffs
        )
        reprojection_errors = np.linalg.norm(imagePoints_inliers - projected_points, axis=2).flatten()
        mean_reprojection_error = np.mean(reprojection_errors)
        std_reprojection_error = np.std(reprojection_errors)

        # Create raw PnP pose_data WITHOUT Kalman filtering
        pose_data = {
            'frame': frame_idx,
            'object_rotation_in_cam': R.tolist(),
            'object_translation_in_cam': tvec.flatten().tolist(),
            'raw_rvec': rvec_o.flatten().tolist(),
            'refined_raw_rvec': rvec.flatten().tolist(),
            'num_inliers': len(final_inliers) if final_inliers is not None else 0,
            'total_matches': len(mkpts0),
            'inlier_ratio': len(final_inliers) / len(mkpts0) if len(mkpts0) > 0 else 0,
            'reprojection_errors': reprojection_errors.tolist(),
            'mean_reprojection_error': float(mean_reprojection_error),
            'std_reprojection_error': float(std_reprojection_error),
            'inliers': final_inliers.flatten().tolist(),
            'mkpts0': mkpts0.tolist(),
            'mkpts1': mkpts1.tolist(),
            'mpts3D': mpts3D.tolist(),
            'mconf': mconf.tolist(),
            'coverage_score': coverage_score,
            'pose_estimation_failed': False,
            'tracking_method': 'pnp'
        }
        
        # Create visualization
        visualization = self._visualize_tracking(
            frame, final_inliers, pose_data, frame_idx, (mkpts0, mkpts1, mconf, frame_keypoints)
        )
        
        # Return the PnP results along with matching data for future use
        return pose_data, visualization, mkpts0, mkpts1, mpts3D