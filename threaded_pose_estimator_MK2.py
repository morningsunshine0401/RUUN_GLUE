import threading
import queue
import time
import logging
import cv2
import atexit
import os
import numpy as np
import torch

# Disable gradient computation globally
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

# Configure logging
logging.basicConfig(
    #level=logging.INFO,
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("threaded_pose_estimator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ThreadedPoseEstimator:
    """Wrapper around PoseEstimator to add threading functionality"""
    def __init__(self, opt, device):
        # This one uses KF_MK4(same as KF_Q, just EKF) with ViT
        from pose_estimator_thread_MK4 import PoseEstimator

        # Pass KF mode to PoseEstimator
        kf_mode = getattr(opt, 'KF_mode', 'auto')
        
        # Initialize the pose estimator
        self.pose_estimator = PoseEstimator(opt, device, kf_mode=kf_mode)
        
        # Add threading components
        self.frame_queue = queue.Queue(maxsize=50)
        self.result_queue = queue.Queue()
        self.running = True
        self.worker_thread = None
        
        # Start worker thread
        self.start_worker()
        
        # Register cleanup
        atexit.register(self.cleanup)
    
    def start_worker(self):
        """Start the worker thread for processing frames"""
        self.worker_thread = threading.Thread(target=self._process_frames_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        logger.info("Started pose estimation worker thread")
    
    def _process_frames_worker(self):
        """Worker function that processes frames from the queue"""
        while self.running:
            try:
                # Get frame data from queue with timeout
                item = self.frame_queue.get(timeout=0.1)
                
                if item is None:
                    break
                
                # Handle anchor reinitialization command
                if isinstance(item, dict) and item.get('command') == 'reinit_anchor':
                    try:
                        logger.info("Worker thread: Processing anchor reinitialization")
                        
                        # Get the completion event from the command
                        completion_event = item.get('completion_event')
                        
                        # Check if anchor file exists
                        new_anchor_path = item['new_anchor_path']
                        if not os.path.exists(new_anchor_path):
                            logger.error(f"Anchor image not found in worker: {new_anchor_path}")
                            if completion_event:
                                completion_event.set()  # Signal completion (failure)
                            self.result_queue.put(('reinit_failed', None, -3, 0, ""))
                            self.frame_queue.task_done()
                            continue
                        
                        reinitialization_completed = False
                        
                        try:
                            # Perform the reinitialization with a timeout via the lock mechanism
                            # The lock timeouts in the pose_estimator_thread.py will handle this
                            success = self.pose_estimator.reinitialize_anchor(
                                new_anchor_path,
                                item['new_2d_points'],
                                item['new_3d_points']
                            )
                            
                            if success:
                                logger.info("Worker thread: Anchor reinitialization complete")
                                reinitialization_completed = True
                                self.result_queue.put(('reinit_complete', None, -2, 0, ""))
                            else:
                                logger.error("Worker thread: Anchor reinitialization failed")
                                self.result_queue.put(('reinit_failed', None, -3, 0, ""))
                        
                        except Exception as e:
                            logger.error(f"Error during anchor reinitialization: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            # Signal error during reinitialization
                            self.result_queue.put(('reinit_failed', None, -3, 0, ""))
                        
                        finally:
                            # Always signal completion regardless of success/failure
                            if completion_event:
                                completion_event.set()
                        
                        self.frame_queue.task_done()
                        continue
                        
                    except Exception as e:
                        logger.error(f"Error in reinitialization handler: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        
                        # Signal error during reinitialization
                        self.result_queue.put(('reinit_failed', None, -3, 0, ""))
                        self.frame_queue.task_done()
                        continue
                
                # Unpack frame data
                frame, frame_idx, frame_t, img_name = item
                
                # Process the frame using the pose estimator
                pose_data, visualization = self.pose_estimator.process_frame(frame, frame_idx)

                # Add timestamp and filename to pose data
                if pose_data:
                    pose_data['timestamp'] = frame_t
                    pose_data['image_file'] = img_name
                
                # Put results in result queue
                self.result_queue.put((pose_data, visualization, frame_idx, frame_t, img_name))
                
                # Mark task as done
                self.frame_queue.task_done()
                
            except queue.Empty:
                # Queue timeout, just continue
                continue
            except Exception as e:
                logger.error(f"Error in worker thread: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Put None in result queue to indicate error
                self.result_queue.put((None, None, -1, 0, ""))
                
                # Mark task as done if we were processing a frame
                try:
                    self.frame_queue.task_done()
                except ValueError:  # If queue was empty
                    pass
    
    def process_frame(self, frame, frame_idx, frame_t, img_name):
        """Add frame to processing queue"""
        self.frame_queue.put((frame, frame_idx, frame_t, img_name))
    
    def reinitialize_anchor(self, new_anchor_path, new_2d_points, new_3d_points):
        """Queue anchor reinitialization to be handled by worker thread and wait for completion"""
        logger.info(f"Queueing anchor reinitialization for {new_anchor_path}")
        
        # Check if the file exists and is readable
        if not os.path.exists(new_anchor_path):
            logger.error(f"Anchor image does not exist: {new_anchor_path}")
            return False
        
        # Try to read the image to verify it's a valid image file
        test_img = cv2.imread(new_anchor_path)
        if test_img is None:
            logger.error(f"Failed to read anchor image: {new_anchor_path}")
            return False
                
        logger.info(f"Verified anchor image exists and is readable: {new_anchor_path}")
        
        # Create a condition variable to signal completion
        completion_event = threading.Event()
        
        # Clear both queues safely
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
                
        # Clear result queue
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
                self.result_queue.task_done()
            except queue.Empty:
                break
        
        # Create and add the reinitialization command
        reinit_data = {
            'command': 'reinit_anchor',
            'new_anchor_path': new_anchor_path,
            'new_2d_points': new_2d_points,
            'new_3d_points': new_3d_points,
            'completion_event': completion_event  # Pass the event to the worker
        }
        
        # Put the reinit command in the queue with high priority
        self.frame_queue.put(reinit_data)
        
        logger.info("Waiting for anchor reinitialization to complete...")
        
        # Wait for the worker to signal completion with timeout
        success = completion_event.wait(timeout=30.0)  # 30 second timeout
        
        if success:
            logger.info("Anchor reinitialization completed successfully")
            return True
        else:
            logger.error("Anchor reinitialization timed out")
            
            # Try to cancel the operation by releasing locks
            if hasattr(self.pose_estimator, 'session_lock') and hasattr(self.pose_estimator.session_lock, '_owner'):
                try:
                    if self.pose_estimator.session_lock._owner:
                        self.pose_estimator.session_lock.release()
                        logger.info("Released session lock after timeout")
                except:
                    pass
                    
            return False

    def get_result(self, timeout=5.0):
        """Get a result from the result queue with timeout"""
        try:
            if self.result_queue.empty():
                # Optionally log if needed
                pass
            start_time = time.time()
            result = self.result_queue.get(timeout=timeout)
            elapsed = time.time() - start_time
            if elapsed > 1.0:  # Only log if waiting took more than 1 second
                logger.info(f"Got result after waiting {elapsed:.2f}s")
            return result
        except queue.Empty:
            # Optionally reduce the logging level here if desired
            return None, None, -1, 0, ""
    
    def cleanup(self):
        """Clean up resources and stop worker thread"""
        logger.info("Cleaning up ThreadedPoseEstimator")
        self.running = False
        
        # Force-release any locks
        if hasattr(self.pose_estimator, 'session_lock') and hasattr(self.pose_estimator.session_lock, '_owner') and self.pose_estimator.session_lock._owner:
            try:
                self.pose_estimator.session_lock.release()
                logger.info("Released session lock during cleanup")
            except:
                pass
        
        # Clear all queues to unblock any waiting threads
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        self.frame_queue.put(None)  # Signal worker to exit
        
        with self.result_queue.mutex:
            self.result_queue.queue.clear()
        
        # Give worker a chance to exit gracefully
        if self.worker_thread and self.worker_thread.is_alive():
            try:
                self.worker_thread.join(timeout=2.0)
            except:
                pass
        
        logger.info("Cleanup complete")