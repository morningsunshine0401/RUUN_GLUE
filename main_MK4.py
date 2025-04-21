import cv2
import time
import argparse
import logging
import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import json
import numpy as np
from datetime import datetime
from threaded_pose_estimator import ThreadedPoseEstimator
from utils import create_unique_filename  # Import the same utility used in main_thread.py

# Configure logging (same as your existing code)
logging.basicConfig(
    # level=logging.INFO,
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("webcam_pose_estimator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WebcamPoseEstimator:
    def __init__(self, args):
        # Store arguments
        self.args = args
        
        # Set up device
        if args.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = args.device
        logger.info(f'Using device: {self.device}')
        
        # Initialize pose estimator
        self.pose_estimator = ThreadedPoseEstimator(args, self.device)

        # Apply dual resolution optimization if requested
        if args.dual_resolution:
            try:
                # Get the process_resolution from args
                process_resolution = (args.process_width, args.process_height)
                
                from performance_optimizations import apply_dual_resolution
                self.pose_estimator.pose_estimator = apply_dual_resolution(
                    self.pose_estimator.pose_estimator,
                    process_resolution=process_resolution
                )
                logger.info(f"Applied dual resolution optimization with process resolution {process_resolution}")
            except ImportError:
                logger.warning("Could not import performance_optimizations module.")
                logger.warning("Please make sure performance_optimizations.py is in your path.")
            except Exception as e:
                logger.error(f"Error applying dual resolution optimization: {e}")
        
        # Apply performance optimizations if requested
        if args.optimize:
            try:
                from performance_optimizations import apply_optimizations
                self.pose_estimator.pose_estimator = apply_optimizations(
                    self.pose_estimator.pose_estimator, 
                    level=args.optimization_level
                )
                logger.info(f"Applied {args.optimization_level} performance optimizations")
            except ImportError:
                logger.warning("Could not import performance_optimizations module. "
                               "Run without optimizations.")
            except Exception as e:
                logger.error(f"Error applying optimizations: {e}")
        
        # Initialize video capture
        if args.input is not None:
            # If an input video file is provided, use it instead of the webcam
            logger.info(f"Using input video file: {args.input}")
            self.cap = cv2.VideoCapture(args.input)
        else:
            # Otherwise, use the webcam (via camera_id)
            logger.info(f"Using webcam with camera ID: {args.camera_id}")
            self.cap = cv2.VideoCapture(args.camera_id)
            # Set camera resolution (only relevant for webcam input)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        
        # Check if capture opened successfully
        if not self.cap.isOpened():
            logger.error(f"Error: Could not open capture source: {args.input if args.input else args.camera_id}")
            raise ValueError(f"Could not open capture source: {args.input if args.input else args.camera_id}")
        
        # Get actual capture resolution
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Capture source initialized with resolution: {actual_width}x{actual_height}")
        
        # Initialize frame counter and timing variables
        self.frame_count = 0
        self.processing_fps = 0
        self.last_fps_update = time.time()
        self.start_time = time.time()
        self.fps_history = []  # To track FPS over time
        
        # Create output window
        cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
        
        # Flag to indicate if we're waiting for first result
        self.waiting_for_first_result = True
        
        # Queue to store results that haven't been displayed yet
        self.pending_results = []
        
        # List to store all pose data for consolidated JSON output
        self.all_poses = []
        
        # Create output directories for JSON files if needed
        if args.save_pose_data or args.save_consolidated_json:
            self.results_dir = os.path.join(args.output_dir, 'results')
            os.makedirs(self.results_dir, exist_ok=True)
            logger.info(f"Created output directory for pose data: {self.results_dir}")
            
            # Create session timestamp
            self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"Session timestamp: {self.session_timestamp}")
            
        # Prepare consolidated JSON filename
        if args.save_consolidated_json:
            if os.path.isdir(args.consolidated_json_filename):
                base_filename = 'pose_estimation.json'
                self.consolidated_json_path = create_unique_filename(args.consolidated_json_filename, base_filename)
            else:
                save_dir = os.path.dirname(args.consolidated_json_filename)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                base_filename = os.path.basename(args.consolidated_json_filename)
                if not save_dir:  # If no directory specified, use results_dir
                    save_dir = self.results_dir
                self.consolidated_json_path = create_unique_filename(save_dir, base_filename)
            
            logger.info(f"Will save consolidated pose data to: {self.consolidated_json_path}")
    
    def run(self):
        """Main loop for capturing frames and processing results"""
        logger.info("Starting capture and pose estimation")
        
        try:
            while True:
                # Capture frame from the source (webcam or video file)
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame from source or reached end of video.")
                    break
                
                # Get timestamp
                current_time = time.time()
                
                # Increment frame counter
                self.frame_count += 1
                
                # Skip frames if requested
                if hasattr(self.args, 'skip_frames') and self.args.skip_frames:
                    if self.frame_count % self.args.skip_frames != 0:
                        continue
                
                # Resize frame if needed based on args.resize
                if hasattr(self.args, 'resize') and len(self.args.resize) > 0:
                    if len(self.args.resize) == 2:
                        frame = cv2.resize(frame, tuple(self.args.resize))
                    elif len(self.args.resize) == 1 and self.args.resize[0] > 0:
                        h, w = frame.shape[:2]
                        scale = self.args.resize[0] / max(h, w)
                        new_size = (int(w * scale), int(h * scale))
                        frame = cv2.resize(frame, new_size)
                
                # Submit frame for processing
                self.pose_estimator.process_frame(
                    frame, 
                    self.frame_count, 
                    current_time, 
                    f"frame_{self.frame_count:06d}"
                )
                
                # Process results (non-blocking)
                self.process_results()
                
                # Calculate FPS every 10 frames
                if self.frame_count % 10 == 0:
                    elapsed = time.time() - self.last_fps_update
                    if elapsed > 0:
                        current_fps = 10 / elapsed
                        self.fps_history.append(current_fps)
                        
                        # Keep only the last 5 measurements for smoothing
                        if len(self.fps_history) > 5:
                            self.fps_history.pop(0)
                        
                        # Calculate smoothed FPS
                        self.processing_fps = sum(self.fps_history) / len(self.fps_history)
                        self.last_fps_update = time.time()
                        
                        # Log FPS periodically
                        if self.frame_count % 100 == 0:
                            logger.info(f"Current FPS: {self.processing_fps:.1f}")
                
                # Check for key press to exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or 'q'
                    logger.info("User requested exit")
                    break
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Clean up resources
            self.cleanup()
    
    def process_results(self):
        """Process available results from the pose estimator queue"""
        # Check if there are pending results
        while True:
            try:
                # Get result without blocking (timeout=0.01)
                result = self.pose_estimator.get_result(timeout=0.01)
                if result[0] is None:
                    break
                # Add to pending results
                self.pending_results.append(result)
            except:
                # No more results
                break
        
        # Process the most recent result (if any)
        if self.pending_results:
            pose_data, visualization, frame_idx, frame_t, img_name = self.pending_results[-1]
            self.pending_results = []  # Clear pending results
            self.display_result(pose_data, visualization, frame_idx, frame_t, img_name)
            self.waiting_for_first_result = False
        elif self.waiting_for_first_result:
            # Show waiting message if no results yet
            h, w = 720, 1280
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(blank, "Initializing pose estimation...", (w//4, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Pose Estimation', blank)
    
    def display_result(self, pose_data, visualization, frame_idx, frame_t, img_name):
        """Display pose estimation result"""
        if visualization is None:
            logger.warning(f"No visualization for frame {frame_idx}")
            return
        
        display = visualization.copy()
        cv2.putText(display, f"FPS: {self.processing_fps:.1f}", 
                    (10, display.shape[0] - 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if pose_data and 'tracking_method' in pose_data:
            method = pose_data['tracking_method']
            method_color = (0, 255, 0)
            
            if method == 'pnp':
                method_text = "PnP Estimation"
                method_color = (0, 165, 255)
            elif method == 'tracking':
                method_text = "Tracking"
            elif method == 'prediction':
                method_text = "Prediction Only"
                method_color = (0, 0, 255)
            elif method == 'loosely_coupled':
                method_text = "Loosely Coupled"
                method_color = (255, 165, 0)
            else:
                method_text = f"Method: {method}"
                
            cv2.putText(display, method_text, 
                        (10, display.shape[0] - 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, method_color, 2)
            
            num_matches = pose_data.get('total_matches', "N/A")
            num_inliers = pose_data.get('num_inliers', "N/A")
            matches_text = f"Matches: {num_matches} | Inliers: {num_inliers}"
            cv2.putText(display, matches_text, 
                        (10, display.shape[0] - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if 'mean_reprojection_error' in pose_data:
                error_text = f"Reproj Error: {pose_data['mean_reprojection_error']:.2f}px"
                cv2.putText(display, error_text, 
                            (10, display.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if pose_data and 'scale_factors' in pose_data:
                scale_x, scale_y = pose_data['scale_factors']
                process_res = pose_data.get('processing_resolution',
                                              (int(display.shape[1]/scale_x), int(display.shape[0]/scale_y)))
                res_text = f"Proc Res: {process_res[0]}x{process_res[1]}"
                cv2.putText(display, res_text, 
                            (display.shape[1] - 300, display.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Pose Estimation', display)
        cv2.waitKey(1)
        
        if pose_data and hasattr(self.args, 'save_consolidated_json') and self.args.save_consolidated_json:
            json_data = self._prepare_pose_data_for_json(pose_data)
            json_data['frame_idx'] = frame_idx
            json_data['timestamp'] = frame_t
            json_data['image_name'] = img_name
            self.all_poses.append(json_data)
        
        if hasattr(self.args, 'save_pose_data') and self.args.save_pose_data and pose_data:
            json_filename = os.path.join(
                self.results_dir,
                f"pose_{self.session_timestamp}_{frame_idx:06d}.json"
            )
            json_data = self._prepare_pose_data_for_json(pose_data)
            with open(json_filename, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        if hasattr(self.args, 'save_frames') and self.args.save_frames:
            frames_dir = os.path.join(self.args.output_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            cv2.imwrite(f"{frames_dir}/frame_{self.session_timestamp}_{frame_idx:06d}.jpg", display)
    
    def _prepare_pose_data_for_json(self, pose_data):
        """Prepare pose_data for JSON serialization by converting numpy arrays to lists"""
        json_data = {}
        for key, value in pose_data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                json_data[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                json_data[key] = int(value)
            else:
                json_data[key] = value
        return json_data

    def save_consolidated_json(self):
        """Save all pose data as a single consolidated JSON file"""
        if hasattr(self.args, 'save_consolidated_json') and self.args.save_consolidated_json and self.all_poses:
            logger.info(f"Saving consolidated pose data to {self.consolidated_json_path}")
            with open(self.consolidated_json_path, 'w') as f:
                json.dump(self.all_poses, f, indent=2)
            logger.info(f"Saved {len(self.all_poses)} pose entries to {self.consolidated_json_path}")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources")
        self.save_consolidated_json()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            logger.info("Capture source released")
        if hasattr(self, 'pose_estimator'):
            self.pose_estimator.cleanup()
            logger.info("Pose estimator cleaned up")
        cv2.destroyAllWindows()
        logger.info("OpenCV windows closed")
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        logger.info(f"Processed {self.frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS average)")
        logger.warning(f"Processed {self.frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS average)")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Webcam or Video File Pose Estimation')
    
    # Video source settings
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input video file (if not provided, uses webcam)')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera ID to use if no input video is provided (default: 0)')
    parser.add_argument('--camera_width', type=int, default=1280,
                        help='Camera capture width (default: 1280)')
    parser.add_argument('--camera_height', type=int, default=720,
                        help='Camera capture height (default: 720)')
    
    # Model settings
    parser.add_argument('--anchor', type=str, required=True,
                        help='Path to anchor image for pose estimation')
    parser.add_argument('--resize', type=int, nargs='+', default=[1280, 720],
                        help='Resize the images to this resolution before processing')
    
    # Device settings
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')
    
    # Performance settings
    parser.add_argument('--optimize', action='store_true',
                        help='Apply performance optimizations')
    parser.add_argument('--optimization_level', type=str, default='balanced',
                        choices=['mild', 'balanced', 'aggressive'],
                        help='Optimization aggressiveness level')
    parser.add_argument('--skip_frames', type=int, default=0,
                        help='Process only every n-th frame (0 = process all frames)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output files (default: output)')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save output frames to disk')
    parser.add_argument('--save_pose_data', action='store_true',
                        help='Save individual pose data as JSON files')
    parser.add_argument('--save_consolidated_json', action='store_true',
                        help='Save all pose data as a single consolidated JSON file')
    parser.add_argument('--consolidated_json_filename', type=str, default='pose_estimation.json',
                        help='Filename for consolidated JSON output (default: pose_estimation.json)')
    
    # Additional performance settings
    parser.add_argument('--dual_resolution', action='store_true',
                        help='Use dual resolution processing (high for anchor, low for input)')
    parser.add_argument('--process_width', type=int, default=640,
                        help='Width for processing resolution (default: 640)')
    parser.add_argument('--process_height', type=int, default=480,
                        help='Height for processing resolution (default: 480)')
    parser.add_argument('--show_keypoints', action='store_true',
                        help='Show detected keypoints')
    
    # Kalman filter mode settings
    parser.add_argument('--KF_mode', type=str, default='auto',
                        choices=['L', 'T', 'auto'],
                        help='Kalman filter mode: L for loosely-coupled, T for tightly-coupled, auto for automatic')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create and run the pose estimator
    try:
        estimator = WebcamPoseEstimator(args)
        estimator.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
