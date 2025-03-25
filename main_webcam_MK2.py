# Process the most recent pending result (if any)
        if self.pending_results:
            # Get the most recent result
            pose_data, visualization, frame_idx, frame_t, img_name = self.pending_results[-1]
            
            # Clear the pending results (we only use the most recent)
            self.pending_results = []
            
            # Process this result
            self.display_result(pose_data, visualization, frame_idx, frame_t, img_name)
            
            # No longer waiting for first result
            self.waiting_for_first_result = False
        
        # If still waiting for first result, show waiting message
        elif self.waiting_for_first_result:
            # Create a blank frame with waiting message
            h, w = 720, 1280
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(blank, "Initializing pose estimation...", (w//4, h//2),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display the waiting frame
            cv2.imshow('Pose Estimation', blank)
    
    def display_result(self, pose_data, visualization, frame_idx, frame_t, img_name):
        """Display pose estimation result"""
        # Check if we have visualization
        if visualization is None:
            logger.warning(f"No visualization for frame {frame_idx}")
            return
        
        # Create a copy for display
        display = visualization.copy()
        
        # Add FPS and timing information
        cv2.putText(display, f"FPS: {self.processing_fps:.1f}", 
                  (10, display.shape[0] - 100), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display tracking status
        if pose_data and 'tracking_method' in pose_data:
            method = pose_data['tracking_method']
            method_color = (0, 255, 0)  # Green for tracking
            
            if method == 'pnp':
                method_text = "PnP Estimation"
                method_color = (0, 165, 255)  # Orange
            elif method == 'tracking':
                method_text = "Tracking"
                method_color = (0, 255, 0)    # Green
            elif method == 'prediction':
                method_text = "Prediction Only"
                method_color = (0, 0, 255)    # Red
            elif method == 'loosely_coupled':
                method_text = "Loosely Coupled"
                method_color = (255, 165, 0)  # Blue-ish
            elif method == 'tightly_coupled':
                method_text = "Tightly Coupled"
                method_color = (255, 0, 255)  # Purple
            else:
                method_text = f"Method: {method}"
                
            cv2.putText(display, method_text, 
                      (10, display.shape[0] - 70), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, method_color, 2)
            
            # Display viewpoint information
            if 'viewpoint' in pose_data:
                viewpoint = pose_data['viewpoint']
                confidence = pose_data.get('viewpoint_confidence', 0.0)
                viewpoint_text = f"Viewpoint: {viewpoint} ({confidence:.2f})"
                viewpoint_color = (255, 255, 255)  # White by default
                
                # If viewpoint changed, highlight it
                if pose_data.get('viewpoint_changed', False):
                    viewpoint_color = (0, 255, 255)  # Yellow for changed viewpoint
                
                cv2.putText(display, viewpoint_text, 
                          (10, display.shape[0] - 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, viewpoint_color, 2)
            
            # Display matches information
            num_matches = "N/A"
            num_inliers = "N/A"
            
            if 'total_matches' in pose_data:
                num_matches = pose_data['total_matches']
            if 'num_inliers' in pose_data:
                num_inliers = pose_data['num_inliers']
                
            matches_text = f"Matches: {num_matches} | Inliers: {num_inliers}"
            cv2.putText(display, matches_text, 
                      (10, display.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
            # Display reprojection error if available
            if 'mean_reprojection_error' in pose_data:
                error_text = f"Reproj Error: {pose_data['mean_reprojection_error']:.2f}px"
                cv2.putText(display, error_text, 
                          (display.shape[1] - 300, display.shape[0] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            # Display processing resolution if using dual resolution
            if pose_data and 'scale_factors' in pose_data:
                scale_x, scale_y = pose_data['scale_factors']
                process_res = pose_data.get('processing_resolution', 
                                        (int(display.shape[1]/scale_x), int(display.shape[0]/scale_y)))
                res_text = f"Proc Res: {process_res[0]}x{process_res[1]}"
                cv2.putText(display, res_text, 
                        (display.shape[1] - 300, display.shape[0] - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Only update display every frame to reduce overhead
        if self.frame_count % 1 == 0:
            cv2.imshow('Pose Estimation', display)
            cv2.waitKey(1)
        
        # Store pose data for consolidated JSON output if available
        if pose_data and hasattr(self.args, 'save_consolidated_json') and self.args.save_consolidated_json:
            # Create a copy of pose_data that is fully serializable
            json_data = self._prepare_pose_data_for_json(pose_data)
            
            # Add frame metadata to the pose data
            json_data['frame_idx'] = frame_idx
            json_data['timestamp'] = frame_t
            json_data['image_name'] = img_name
            
            # Add to the list of all poses
            self.all_poses.append(json_data)
        
        # Save individual pose data as JSON
        if hasattr(self.args, 'save_pose_data') and self.args.save_pose_data and pose_data:
            json_filename = os.path.join(
                self.results_dir, 
                f"pose_{self.session_timestamp}_{frame_idx:06d}.json"
            )
            
            # Create a copy of pose_data that is fully serializable
            json_data = self._prepare_pose_data_for_json(pose_data)
            
            with open(json_filename, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        # Optionally save frames
        if hasattr(self.args, 'save_frames') and self.args.save_frames:
            frames_dir = os.path.join(self.args.output_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            cv2.imwrite(f"{frames_dir}/frame_{self.session_timestamp}_{frame_idx:06d}.jpg", display)
    
    def _prepare_pose_data_for_json(self, pose_data):
        """Prepare pose_data for JSON serialization by converting numpy arrays to lists"""
        json_data = {}
        
        for key, value in pose_data.items():
            # Handle numpy arrays and other non-serializable types
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, np.float32) or isinstance(value, np.float64):
                json_data[key] = float(value)
            elif isinstance(value, np.int32) or isinstance(value, np.int64):
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
        
        # Save consolidated JSON before cleaning up other resources
        self.save_consolidated_json()
        
        # Release webcam
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            logger.info("Webcam released")
        
        # Clean up pose estimator
        if hasattr(self, 'pose_estimator'):
            self.pose_estimator.cleanup()
            logger.info("Pose estimator cleaned up")
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        logger.info("OpenCV windows closed")
        
        # Log performance
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        logger.info(f"Processed {self.frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS average)")
        logger.warning(f"Processed {self.frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS average)")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Webcam-based Pose Estimation with Multi-Viewpoint Support')
    
    # Camera settings
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera ID to use (default: 0)')
    parser.add_argument('--camera_width', type=int, default=1280,
                        help='Camera capture width (default: 1280)')
    parser.add_argument('--camera_height', type=int, default=720,
                        help='Camera capture height (default: 720)')
    
    # Model settings
    parser.add_argument('--anchor', type=str, required=True,
                        help='Path to main anchor image for pose estimation')
    parser.add_argument('--anchor_NE', type=str, default=None,
                        help='Path to NE viewpoint anchor image (optional)')
    parser.add_argument('--anchor_NW', type=str, default=None,
                        help='Path to NW viewpoint anchor image (optional)')
    parser.add_argument('--anchor_SE', type=str, default=None,
                        help='Path to SE viewpoint anchor image (optional)')
    parser.add_argument('--anchor_SW', type=str, default=None,
                        help='Path to SW viewpoint anchor image (optional)')
    parser.add_argument('--vit_model_path', type=str, default='mobilevit_viewpoint_twostage_final_2.pth',
                        help='Path to the MobileViT viewpoint classification model')
    parser.add_argument('--resize', type=int, nargs='+', default=[1280, 720],
                    help='Resize the images to this resolution before processing')

    # Camera intrinsics (optional - will use defaults if not provided)
    parser.add_argument('--fx', type=float, default=None,
                        help='Camera focal length in x-direction (optional)')
    parser.add_argument('--fy', type=float, default=None,
                        help='Camera focal length in y-direction (optional)')
    parser.add_argument('--cx', type=float, default=None,
                        help='Camera principal point x-coordinate (optional)')
    parser.add_argument('--cy', type=float, default=None,
                        help='Camera principal point y-coordinate (optional)')
    parser.add_argument('--distCoeffs', type=float, nargs='+', default=None,
                        help='Camera distortion coefficients [k1,k2,p1,p2,k3,...] (optional)')
    
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
    
    # Add under "Performance settings":
    parser.add_argument('--dual_resolution', action='store_true',
                        help='Use dual resolution processing (high for anchor, low for input)')
    parser.add_argument('--process_width', type=int, default=640,
                        help='Width for processing resolution (default: 640)')
    parser.add_argument('--process_height', type=int, default=480,
                        help='Height for processing resolution (default: 480)')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show detected keypoints')
    
    # Add this argument after the optimization/performance settings:
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
    
    # Create and run the webcam pose estimator
    try:
        estimator = WebcamPoseEstimator(args)
        estimator.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())import cv2
import time
import argparse
import logging
import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import json
import numpy as np
from datetime import datetime
from threaded_pose_estimator import ThreadedPoseEstimator
from utils import create_unique_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    #level=logging.WARNING,
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
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(args.camera_id)
        
        # Set camera resolution to 1280x720 or specified resolution
        if hasattr(args, 'camera_width') and hasattr(args, 'camera_height'):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            logger.error(f"Error: Could not open camera {args.camera_id}")
            raise ValueError(f"Could not open camera {args.camera_id}")
        
        # Get actual camera resolution
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Camera initialized with resolution: {actual_width}x{actual_height}")
        
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
        
        # Create output directories for JSON files
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
        logger.info("Starting webcam capture and pose estimation")
        
        try:
            while True:
                # Capture frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame from webcam")
                    break
                
                # Get timestamp
                current_time = time.time()
                
                # Increment frame counter
                self.frame_count += 1
                
                # Skip frames if we're falling behind (if enabled)
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
                    f"webcam_{self.frame_count:06d}"
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
            # Clean up
            self.cleanup()
    
    def process_results(self):
        """Process available results from the pose estimator queue"""
        # Check if there are pending results
        while True:
            try:
                # Get result without blocking (timeout=0.01)
                result = self.pose_estimator.get_result(timeout=0.01)
                if result[0] is None:
                    # No result available or error
                    break
                
                # Add to pending results
                self.pending_results.append(result)
            except:
                # No more results
                break
        
        # Process the most recent