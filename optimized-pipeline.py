import argparse
import torch
torch.set_grad_enabled(False)
import time
from pathlib import Path
import cv2
import queue
import threading
import numpy as np
import onnxruntime as ort
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import os
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimized_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FrameProcessor:
    """Handles frame preprocessing in a separate thread"""
    def __init__(self, resize_dims, max_queue_size=5):
        self.resize_dims = resize_dims
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = None
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def _process_loop(self):
        while self.running:
            try:
                item = self.input_queue.get(timeout=0.1)
                if item is None:
                    self.output_queue.put(None)  # Signal end
                    break
                    
                frame, frame_idx, timestamp, filename = item
                
                # Resize frame
                w_new, h_new = self.resize_dims
                resized = cv2.resize(frame, (w_new, h_new))
                
                # Preprocess (grayscale, normalization, etc.)
                # Create a copy of the original frame for visualization
                resized_vis = resized.copy()
                
                # Convert to grayscale if needed
                if len(resized.shape) == 3:
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                else:
                    gray = resized
                
                # Normalize to [0,1]
                proc = gray.astype(np.float32) / 255.0
                
                # Add batch and channel dimensions
                proc = proc[None, None]
                
                # Output the processed frame
                self.output_queue.put((proc, resized_vis, frame_idx, timestamp, filename))
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in frame processor: {e}")
                continue

class InferenceProcessor:
    """Handles model inference in a separate thread"""
    def __init__(self, onnx_path, device='cuda', max_queue_size=5):
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = None
        self.device = device
        
        # Initialize ONNX Runtime with appropriate providers
        providers = []
        if device == 'cuda':
            providers.append(
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                })
            )
        providers.append("CPUExecutionProvider")
        
        # Load models (consider separate models for SuperPoint and LightGlue)
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Pre-compute anchor features
        self.anchor_features = None
        
    def set_anchor_features(self, anchor_features):
        """Set the pre-computed anchor features"""
        self.anchor_features = anchor_features
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def _process_loop(self):
        while self.running:
            try:
                item = self.input_queue.get(timeout=0.1)
                if item is None:
                    self.output_queue.put(None)  # Signal end
                    break
                    
                proc, resized_vis, frame_idx, timestamp, filename = item
                
                # Create batch with anchor and current frame
                if self.anchor_features is not None:
                    # Concatenate anchor and frame features
                    batch = np.concatenate([self.anchor_features, proc], axis=0).astype(np.float32)
                    
                    # Run inference
                    start_time = time.time()
                    keypoints, matches, mscores = self.session.run(None, {"images": batch})
                    inference_time = time.time() - start_time
                    
                    # Output results
                    result = {
                        'keypoints': keypoints,
                        'matches': matches,
                        'mscores': mscores,
                        'inference_time': inference_time
                    }
                    
                    self.output_queue.put((result, resized_vis, frame_idx, timestamp, filename))
                else:
                    logger.warning(f"Anchor features not set, skipping frame {frame_idx}")
                    
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in inference processor: {e}")
                continue

class PoseEstimator:
    """Handles pose estimation in a separate thread"""
    def __init__(self, camera_matrix, dist_coeffs, anchor_keypoints_3D, matched_anchor_indices, max_queue_size=5):
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = None
        
        self.K = camera_matrix
        self.distCoeffs = dist_coeffs
        self.matched_3D_keypoints = anchor_keypoints_3D
        self.matched_anchor_indices = matched_anchor_indices
        
        # Initialize Kalman Filter
        self.kf_pose = self._init_kalman_filter()
        self.kf_initialized = False
        
    def _init_kalman_filter(self):
        # Initialize KF (simplified here)
        from KF_Q import KalmanFilterPose
        frame_rate = 30
        dt = 1 / frame_rate
        return KalmanFilterPose(dt)
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            
    def _process_loop(self):
        while self.running:
            try:
                item = self.input_queue.get(timeout=0.1)
                if item is None:
                    self.output_queue.put(None)  # Signal end
                    break
                    
                result, resized_vis, frame_idx, timestamp, filename = item
                
                # Extract keypoints and matches
                keypoints = result['keypoints']
                matches = result['matches']
                mscores = result['mscores']
                inference_time = result['inference_time']
                
                # Filter valid matches (matches between anchor and current frame)
                valid_mask = (matches[:, 0] == 0)
                valid_matches = matches[valid_mask]
                
                if len(valid_matches) > 0:
                    mkpts0 = keypoints[0][valid_matches[:, 1]]  # Anchor keypoints
                    mkpts1 = keypoints[1][valid_matches[:, 2]]  # Frame keypoints
                    mconf = mscores[valid_mask]
                    anchor_indices = valid_matches[:, 1]
                    
                    # Filter to known anchor indices (with 3D coordinates)
                    known_mask = np.isin(anchor_indices, self.matched_anchor_indices)
                    mkpts0 = mkpts0[known_mask]
                    mkpts1 = mkpts1[known_mask]
                    mconf = mconf[known_mask]
                    anchor_indices_filtered = anchor_indices[known_mask]
                    
                    # Map anchor indices to 3D points
                    idx_map = {idx: i for i, idx in enumerate(self.matched_anchor_indices)}
                    mpts3D = np.array([
                        self.matched_3D_keypoints[idx_map[aidx]] 
                        for aidx in anchor_indices_filtered if aidx in idx_map
                    ])
                    
                    # Check if we have enough points for pose estimation
                    if len(mpts3D) >= 4:
                        # Perform PnP pose estimation
                        pose_data, visualization = self._estimate_pose(
                            mkpts0, mkpts1, mpts3D, mconf, resized_vis, 
                            frame_idx, keypoints[1]
                        )
                    else:
                        logger.warning(f"Not enough matches for pose (frame {frame_idx})")
                        pose_data = self._use_kalman_prediction(frame_idx)
                        visualization = resized_vis
                else:
                    logger.warning(f"No valid matches for frame {frame_idx}")
                    pose_data = self._use_kalman_prediction(frame_idx)
                    visualization = resized_vis
                
                # Add timestamp and filename to pose data
                pose_data['timestamp'] = timestamp
                pose_data['image_file'] = filename
                
                # Output results
                self.output_queue.put((pose_data, visualization, frame_idx))
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in pose estimator: {e}")
                continue
                
    def _use_kalman_prediction(self, frame_idx):
        """Use Kalman filter prediction when pose estimation fails"""
        x_pred, P_pred = self.kf_pose.predict()
        from utils import quaternion_to_rotation_matrix
        
        # Parse prediction
        translation_estimated = x_pred[0:3]
        q_estimated = x_pred[6:10]
        R_estimated = quaternion_to_rotation_matrix(q_estimated)
        
        return {
            'frame': frame_idx,
            'kf_translation_vector': translation_estimated.tolist(),
            'kf_quaternion': q_estimated.tolist(),
            'kf_rotation_matrix': R_estimated.tolist(),
            'pose_estimation_failed': True
        }
    
    def _estimate_pose(self, mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, frame_keypoints):
        """Estimate pose using PnP and refine with VVS"""
        # Prepare points for PnP
        objectPoints = mpts3D.reshape(-1, 1, 3)
        imagePoints = mkpts1.reshape(-1, 1, 2).astype(np.float32)
        
        # Solve PnP
        success, rvec_o, tvec_o, inliers = cv2.solvePnPRansac(
            objectPoints=objectPoints,
            imagePoints=imagePoints,
            cameraMatrix=self.K,
            distCoeffs=self.distCoeffs,
            reprojectionError=4,
            confidence=0.999,
            iterationsCount=2000,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if not success or inliers is None or len(inliers) < 6:
            logger.warning("PnP pose estimation failed or not enough inliers.")
            return None, frame
        
        # Refine with VVS
        objectPoints_inliers = objectPoints[inliers.flatten()]
        imagePoints_inliers = imagePoints[inliers.flatten()]
        
        rvec, tvec = cv2.solvePnPRefineVVS(
            objectPoints=objectPoints_inliers,
            imagePoints=imagePoints_inliers,
            cameraMatrix=self.K,
            distCoeffs=self.distCoeffs,
            rvec=rvec_o,
            tvec=tvec_o
        )
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Calculate reprojection errors
        projected_points, _ = cv2.projectPoints(
            objectPoints_inliers, rvec, tvec, self.K, self.distCoeffs
        )
        reprojection_errors = np.linalg.norm(imagePoints_inliers - projected_points, axis=2).flatten()
        mean_reprojection_error = np.mean(reprojection_errors)
        std_reprojection_error = np.std(reprojection_errors)
        
        # Update Kalman filter
        from utils import rotation_matrix_to_quaternion
        q_measured = rotation_matrix_to_quaternion(R)
        
        # Build measurement vector z = [px,py,pz, qx,qy,qz,qw]
        tvec = tvec.flatten()
        z_meas = np.array([
            tvec[0], tvec[1], tvec[2],
            q_measured[0], q_measured[1], q_measured[2], q_measured[3]
        ], dtype=np.float64)
        
        # Update Kalman filter
        if not self.kf_initialized:
            # First update
            x_upd, P_upd = self.kf_pose.update(z_meas)
            self.kf_initialized = True
        else:
            # Check if measurement is valid
            if mean_reprojection_error < 4.0 and len(inliers) > 4:
                x_upd, P_upd = self.kf_pose.update(z_meas)
            else:
                # Skip update, just predict
                x_upd, P_upd = self.kf_pose.predict()
        
        # Extract final pose
        px, py, pz = x_upd[0:3]
        qx, qy, qz, qw = x_upd[6:10]
        from utils import quaternion_to_rotation_matrix
        R_estimated = quaternion_to_rotation_matrix([qx, qy, qz, qw])
        
        # Create pose data
        pose_data = {
            'frame': frame_idx,
            'object_rotation_in_cam': R.tolist(),
            'object_translation_in_cam': tvec.flatten().tolist(),
            'raw_rvec': rvec_o.flatten().tolist(),
            'refined_raw_rvec': rvec.flatten().tolist(),
            'num_inliers': len(inliers),
            'total_matches': len(mkpts0),
            'inlier_ratio': len(inliers) / len(mkpts0) if len(mkpts0) > 0 else 0,
            'mean_reprojection_error': float(mean_reprojection_error),
            'std_reprojection_error': float(std_reprojection_error),
            
            # Filtered results from Kalman filter
            'kf_translation_vector': [px, py, pz],
            'kf_quaternion': [qx, qy, qz, qw],
            'kf_rotation_matrix': R_estimated.tolist(),
        }
        
        # Create visualization
        visualization = self._visualize_matches(
            frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints
        )
        
        return pose_data, visualization
        
    def _visualize_matches(self, frame, inliers, mkpts0, mkpts1, mconf, pose_data, frame_keypoints):
        """Create visualization of matches and pose estimation"""
        from models.utils import make_matching_plot_fast
        import matplotlib.cm as cm
        
        # This is just a placeholder - implement actual visualization
        # similar to your existing code
        visualization = frame.copy()
        
        # Show the object position in camera frame
        t_in_cam = pose_data['object_translation_in_cam']
        position_text = (f"Leader in Cam: "
                        f"x={t_in_cam[0]:.3f}, y={t_in_cam[1]:.3f}, z={t_in_cam[2]:.3f}")
        cv2.putText(visualization, position_text, (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        
        return visualization

class OptimizedPipeline:
    """Main pipeline that connects all components"""
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
        logger.info(f'Using device: {self.device}')
        
        # Initialize components
        self.frame_processor = FrameProcessor(args.resize)
        self.inference_processor = InferenceProcessor(
            args.model_path, 
            device=self.device
        )
        
        # Load anchor and set up pose estimator
        self._initialize_anchor()
        
    def _initialize_anchor(self):
        """Initialize the anchor image and 3D points"""
        # Load anchor image
        anchor_image = cv2.imread(self.args.anchor)
        assert anchor_image is not None, f"Failed to load anchor image at {self.args.anchor}"
        
        # Resize anchor image
        w_new, h_new = self.args.resize
        anchor_resized = cv2.resize(anchor_image, (w_new, h_new))
        
        # Convert to grayscale and normalize
        if len(anchor_resized.shape) == 3:
            anchor_gray = cv2.cvtColor(anchor_resized, cv2.COLOR_BGR2GRAY)
        else:
            anchor_gray = anchor_resized
        
        anchor_proc = anchor_gray.astype(np.float32) / 255.0
        anchor_proc = anchor_proc[None, None]  # Add batch and channel dimensions
        
        # Set anchor features for inference processor
        self.inference_processor.set_anchor_features(anchor_proc)
        
        # Set up 2D-3D correspondences for the anchor
        anchor_keypoints_2D = np.array([
            # Your anchor 2D points
            [511, 293], [591, 284], [587, 330], [413, 249],
            # ... rest of your points
        ], dtype=np.float32)
        
        anchor_keypoints_3D = np.array([
            # Your anchor 3D points
            [-0.014, 0.000, 0.042], [0.025, -0.014, -0.011],
            # ... rest of your points
        ], dtype=np.float32)
        
        # For simplicity, let's use dummy camera matrix
        K = np.array([
            [1430.10150, 0, 640.85462],
            [0, 1430.48915, 480.64800],
            [0, 0, 1]
        ], dtype=np.float32)
        
        distCoeffs = np.array([0.3393, 2.0351, 0.0295, -0.0029, -10.9093], dtype=np.float32)
        
        # Set up 2D<->3D keypoint matching (simplified here)
        matched_anchor_indices = np.arange(len(anchor_keypoints_2D))
        
        # Initialize pose estimator
        self.pose_estimator = PoseEstimator(
            K, distCoeffs,
            anchor_keypoints_3D,
            matched_anchor_indices
        )
        
    def start(self):
        """Start all processing threads"""
        self.frame_processor.start()
        self.inference_processor.start()
        self.pose_estimator.start()
        
    def stop(self):
        """Stop all processing threads"""
        self.frame_processor.stop()
        self.inference_processor.stop()
        self.pose_estimator.stop()
        
    def process_frames(self, entries):
        """Process frames from the given entries"""
        all_poses = []
        
        # Prepare output directory if needed
        if self.args.output_dir is not None:
            Path(self.args.output_dir).mkdir(exist_ok=True)
            
        # Prepare display window
        if not self.args.no_display:
            cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Pose Estimation', 640 * 2, 480)
            
        # Start processing pipeline
        self.start()
        
        # Launch result collector thread
        result_thread = threading.Thread(
            target=self._collect_results, 
            args=(all_poses,)
        )
        result_thread.daemon = True
        result_thread.start()
        
        # Feed frames into the pipeline
        for entry in entries:
            frame_idx = entry['index']
            frame_t = entry['timestamp']
            img_name = entry['filename']
            
            image_path = os.path.join(self.args.image_dir, img_name)
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
                
            # Load the image
            frame = cv2.imread(image_path)
            if frame is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue
                
            # Add to processing queue
            self.frame_processor.input_queue.put((frame, frame_idx, frame_t, img_name))
            
        # Signal end of input
        self.frame_processor.input_queue.put(None)
        
        # Wait for processing to finish
        result_thread.join()
        
        # Stop the pipeline
        self.stop()
        
        # Clean up
        if not self.args.no_display:
            cv2.destroyAllWindows()
            
        # Save pose estimation results
        with open(self.args.save_pose, 'w') as f:
            json.dump(all_poses, f, indent=4)
        logger.info(f'Pose estimation results saved to {self.args.save_pose}')
        
        return all_poses
        
    def _collect_results(self, all_poses):
        """Collect and process results from the pipeline"""
        # Connect frame processor to inference processor
        def connect_frame_to_inference():
            while True:
                try:
                    item = self.frame_processor.output_queue.get(timeout=0.1)
                    if item is None:
                        self.inference_processor.input_queue.put(None)
                        break
                    self.inference_processor.input_queue.put(item)
                    self.frame_processor.output_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error connecting frame to inference: {e}")
        
        # Connect inference processor to pose estimator
        def connect_inference_to_pose():
            while True:
                try:
                    item = self.inference_processor.output_queue.get(timeout=0.1)
                    if item is None:
                        self.pose_estimator.input_queue.put(None)
                        break
                    self.pose_estimator.input_queue.put(item)
                    self.inference_processor.output_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error connecting inference to pose: {e}")
        
        # Start connector threads
        frame_to_inference_thread = threading.Thread(target=connect_frame_to_inference)
        inference_to_pose_thread = threading.Thread(target=connect_inference_to_pose)
        frame_to_inference_thread.daemon = True
        inference_to_pose_thread.daemon = True

def read_image_index_csv(csv_path):
    """
    Reads a CSV with columns: Index, Timestamp, Filename
    Returns a list of dicts: [{'index':..., 'timestamp':..., 'filename':...}, ...]
    """
    entries = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row['Index'])
            tstamp = float(row['Timestamp'])
            fname = row['Filename']
            entries.append({
                'index': frame_idx,
                'timestamp': tstamp,
                'filename': fname
            })
    return entries

def create_unique_filename(directory, base_filename):
    """Create a unique filename by appending a counter if file exists"""
    if not directory:
        directory = '.'
    
    path = os.path.join(directory, base_filename)
    if not os.path.exists(path):
        return path
    
    name, ext = os.path.splitext(base_filename)
    counter = 1
    
    while True:
        new_name = f"{name}_{counter}{ext}"
        path = os.path.join(directory, new_name)
        if not os.path.exists(path):
            return path
        counter += 1

def main():
    parser = argparse.ArgumentParser(
        description='Optimized Threaded Pipeline for Pose Estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--image_dir', type=str, required=True,
        help='Folder containing extracted images (PNG/JPG, etc.)')
    parser.add_argument(
        '--csv_file', type=str, required=True,
        help='CSV (image_index.csv) with columns [Index, Timestamp, Filename]')
    parser.add_argument(
        '--anchor', type=str, required=True,
        help='Path to the initial anchor (reference) image')
    parser.add_argument(
        '--model_path', type=str, default="weights/superpoint_lightglue_pipeline_1280x720.onnx",
        help='Path to ONNX model')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (visualization). If None, no output images saved.')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference (width height).')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images in a GUI window.')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force PyTorch to run on CPU even if CUDA is available.')
    parser.add_argument(
        '--save_pose', type=str, default='pose_estimation_optimized.json',
        help='Path to save JSON pose estimation results')
    parser.add_argument(
        '--num_threads', type=int, default=3,
        help='Number of threads to use for the thread pool')
    
    args = parser.parse_args()
    logger.info(f"Parsed options: {args}")
    
    # Handle save_pose argument
    if os.path.isdir(args.save_pose):
        base_filename = 'pose_estimation_optimized.json'
        args.save_pose = create_unique_filename(args.save_pose, base_filename)
    else:
        save_dir = os.path.dirname(args.save_pose)
        base_filename = os.path.basename(args.save_pose)
        args.save_pose = create_unique_filename(save_dir, base_filename)
    
    # Read CSV listing all frames
    entries = read_image_index_csv(args.csv_file)
    if len(entries) == 0:
        logger.error("No entries found in CSV file. Exiting.")
        exit(1)
    
    logger.info(f"Found {len(entries)} frames in {args.csv_file}")
    
    # Initialize and run the optimized pipeline
    pipeline = OptimizedPipeline(args)
    all_poses = pipeline.process_frames(entries)
    
    logger.info(f"Pipeline completed. Processed {len(all_poses)} frames.")

if __name__ == '__main__':
    main()
        frame_to_inference_thread.start()
        inference_to_pose_thread.start()
        
        # Process results from pose estimator
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                item = self.pose_estimator.output_queue.get(timeout=0.1)
                if item is None:
                    break
                    
                pose_data, visualization, frame_idx = item
                
                # Store pose data
                all_poses.append(pose_data)
                frame_count += 1
                
                # Display visualization
                if not self.args.no_display:
                    cv2.imshow('Pose Estimation', visualization)
                    if cv2.waitKey(1) == ord('q'):
                        logger.info('Exiting on user request (q key pressed).')
                        break
                
                # Optionally save the visualization
                if self.args.output_dir:
                    out_file = str(Path(self.args.output_dir, f'frame_{frame_idx:06d}.png'))
                    cv2.imwrite(out_file, visualization)
                
                # Log progress
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {frame_count} frames, current FPS: {fps:.2f}")
                
                self.pose_estimator.output_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing results: {e}")
                continue
        
        # Wait for connector threads to finish
        frame_to_inference_thread.join()
        inference_to_pose_thread.join()
        
        # Log final statistics
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 and frame_count > 0 else 0
        logger.info(f"Processed {frame_count} frames in {elapsed:.2f}s (Average FPS: {fps:.2f})")