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
    level=logging.DEBUG,
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
        # Import the PoseEstimator class here to avoid circular imports
        from pose_estimator_thread import PoseEstimator
        
        # Initialize the pose estimator
        self.pose_estimator = PoseEstimator(opt, device)
        
        # Add threading components
        self.frame_queue = queue.Queue(maxsize=10)
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
                        
                        # Check if anchor file exists
                        new_anchor_path = item['new_anchor_path']
                        if not os.path.exists(new_anchor_path):
                            logger.error(f"Anchor image not found in worker: {new_anchor_path}")
                            self.result_queue.put(('reinit_failed', None, -3, 0, ""))
                            self.frame_queue.task_done()
                            continue
                        
                        # Set a timeout for the reinitialization
                        reinitialization_start = time.time()
                        reinitialization_timeout = 8.0  # seconds
                        
                        # Perform the reinitialization in a separate thread to detect timeouts
                        def reinitialization_task():
                            try:
                                self.pose_estimator.reinitialize_anchor(
                                    new_anchor_path,
                                    item['new_2d_points'],
                                    item['new_3d_points']
                                )
                                return True
                            except Exception as e:
                                logger.error(f"Error during anchor reinitialization: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                                return False
                        
                        reinitialization_thread = threading.Thread(target=reinitialization_task)
                        reinitialization_thread.daemon = True
                        reinitialization_thread.start()
                        
                        # Wait for the reinitialization to complete with timeout
                        reinitialization_thread.join(timeout=reinitialization_timeout)
                        
                        if reinitialization_thread.is_alive():
                            logger.error(f"Anchor reinitialization timed out after {reinitialization_timeout} seconds")
                            self.result_queue.put(('reinit_failed', None, -3, 0, ""))
                        else:
                            logger.info("Worker thread: Anchor reinitialization complete")
                            self.result_queue.put(('reinit_complete', None, -2, 0, ""))
                        
                        self.frame_queue.task_done()
                        continue
                    except Exception as e:
                        logger.error(f"Error during anchor reinitialization: {e}")
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
                
                # Mark task as done
                self.frame_queue.task_done()
    
    def process_frame(self, frame, frame_idx, frame_t, img_name):
        """Add frame to processing queue"""
        self.frame_queue.put((frame, frame_idx, frame_t, img_name))
    
    def reinitialize_anchor(self, new_anchor_path, new_2d_points, new_3d_points):
        """Queue anchor reinitialization to be handled by worker thread"""
        logger.info(f"Queueing anchor reinitialization for {new_anchor_path}")
        
        # Check if the file exists and is readable
        if not os.path.exists(new_anchor_path):
            logger.error(f"Anchor image does not exist: {new_anchor_path}")
            self.result_queue.put(('reinit_failed', None, -3, 0, ""))
            return
        
        # Try to read the image to verify it's a valid image file
        test_img = cv2.imread(new_anchor_path)
        if test_img is None:
            logger.error(f"Failed to read anchor image: {new_anchor_path}")
            self.result_queue.put(('reinit_failed', None, -3, 0, ""))
            return
            
        logger.info(f"Verified anchor image exists and is readable: {new_anchor_path}")
        
        # Clear any pending items in frame queue 
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
            'new_3d_points': new_3d_points
        }
        
        # Put the reinit command in the queue with high priority
        self.frame_queue.put(reinit_data)
        
        logger.info("Waiting for anchor reinitialization to complete...")

    def get_result(self, timeout=5.0):
        """Get a result from the result queue with timeout"""
        try:
            # Check if there are any results before waiting
            if self.result_queue.empty():
                logger.info(f"Result queue is empty, waiting up to {timeout}s for results...")
            
            start_time = time.time()
            result = self.result_queue.get(timeout=timeout)
            elapsed = time.time() - start_time
            
            if elapsed > 1.0:  # Only log if waiting took more than 1 second
                logger.info(f"Got result after waiting {elapsed:.2f}s")
                
            return result
        except queue.Empty:
            logger.warning(f"Timeout waiting for processing result (timeout={timeout}s)")
            return None, None, -1, 0, ""
    
    def cleanup(self):
        """Clean up resources and stop worker thread"""
        logger.info("Cleaning up ThreadedPoseEstimator")
        self.running = False
        
        # Put None in queue to signal worker to stop
        self.frame_queue.put(None)
        
        # Wait for worker to finish
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                logger.warning("Worker thread did not exit cleanly")

########################################################
# import threading
# import queue
# import time
# import logging
# import cv2
# import atexit
# import os
# import numpy as np
# import torch
# # Disable gradient computation globally
# torch.set_grad_enabled(False)
# torch.autograd.set_grad_enabled(False)

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("threaded_pose_estimator.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# class ThreadedPoseEstimator:
#     """Wrapper around PoseEstimator to add threading functionality"""
#     def __init__(self, opt, device):
#         # Import the PoseEstimator class here to avoid circular imports
#         from pose_estimator_thread import PoseEstimator
        
#         # Initialize the pose estimator
#         self.pose_estimator = PoseEstimator(opt, device)
        
#         # Add threading components
#         self.frame_queue = queue.Queue(maxsize=10)
#         self.result_queue = queue.Queue()
#         self.running = True
#         self.worker_thread = None
        
#         # Start worker thread
#         self.start_worker()
        
#         # Register cleanup
#         atexit.register(self.cleanup)
    
#     def start_worker(self):
#         """Start the worker thread for processing frames"""
#         self.worker_thread = threading.Thread(target=self._process_frames_worker)
#         self.worker_thread.daemon = True
#         self.worker_thread.start()
#         logger.info("Started pose estimation worker thread")
    
#     def _process_frames_worker(self):
#         """Worker function that processes frames from the queue"""
#         while self.running:
#             try:
#                 # Get frame data from queue with timeout
#                 item = self.frame_queue.get(timeout=0.1)
                
#                 if item is None:
#                     break
                
#                 # Handle anchor reinitialization command
#                 if isinstance(item, dict) and item.get('command') == 'reinit_anchor':
#                     try:
#                         logger.info("Worker thread: Processing anchor reinitialization")
                        
#                         # Check if anchor file exists
#                         new_anchor_path = item['new_anchor_path']
#                         if not os.path.exists(new_anchor_path):
#                             logger.error(f"Anchor image not found in worker: {new_anchor_path}")
#                             self.result_queue.put(('reinit_failed', None, -3, 0, ""))
#                             self.frame_queue.task_done()
#                             continue
                        
#                         # Set a timeout for the reinitialization
#                         reinitialization_start = time.time()
#                         reinitialization_timeout = 8.0  # seconds
                        
#                         # Perform the reinitialization in a separate thread to detect timeouts
#                         def reinitialization_task():
#                             try:
#                                 self.pose_estimator.reinitialize_anchor(
#                                     new_anchor_path,
#                                     item['new_2d_points'],
#                                     item['new_3d_points']
#                                 )
#                                 return True
#                             except Exception as e:
#                                 logger.error(f"Error during anchor reinitialization: {e}")
#                                 import traceback
#                                 logger.error(traceback.format_exc())
#                                 return False
                        
#                         reinitialization_thread = threading.Thread(target=reinitialization_task)
#                         reinitialization_thread.daemon = True
#                         reinitialization_thread.start()
                        
#                         # Wait for the reinitialization to complete with timeout
#                         reinitialization_thread.join(timeout=reinitialization_timeout)
                        
#                         if reinitialization_thread.is_alive():
#                             logger.error(f"Anchor reinitialization timed out after {reinitialization_timeout} seconds")
#                             self.result_queue.put(('reinit_failed', None, -3, 0, ""))
#                         else:
#                             logger.info("Worker thread: Anchor reinitialization complete")
#                             self.result_queue.put(('reinit_complete', None, -2, 0, ""))
                        
#                         self.frame_queue.task_done()
#                         continue
#                     except Exception as e:
#                         logger.error(f"Error during anchor reinitialization: {e}")
#                         import traceback
#                         logger.error(traceback.format_exc())
                        
#                         # Signal error during reinitialization
#                         self.result_queue.put(('reinit_failed', None, -3, 0, ""))
#                         self.frame_queue.task_done()
#                         continue
                
#                 # Unpack frame data
#                 frame, frame_idx, frame_t, img_name = item
                
#                 # Process the frame using the pose estimator
#                 pose_data, visualization = self.pose_estimator.process_frame(frame, frame_idx)
                
#                 # Add timestamp and filename to pose data
#                 if pose_data:
#                     pose_data['timestamp'] = frame_t
#                     pose_data['image_file'] = img_name
                
#                 # Put results in result queue
#                 self.result_queue.put((pose_data, visualization, frame_idx, frame_t, img_name))
                
#                 # Mark task as done
#                 self.frame_queue.task_done()
                
#             except queue.Empty:
#                 # Queue timeout, just continue
#                 continue
#             except Exception as e:
#                 logger.error(f"Error in worker thread: {e}")
#                 import traceback
#                 logger.error(traceback.format_exc())
                
#                 # Put None in result queue to indicate error
#                 self.result_queue.put((None, None, -1, 0, ""))
                
#                 # Mark task as done
#                 self.frame_queue.task_done()
    
#     def process_frame(self, frame, frame_idx, frame_t, img_name):
#         """Add frame to processing queue"""
#         self.frame_queue.put((frame, frame_idx, frame_t, img_name))
    
#     def reinitialize_anchor(self, new_anchor_path, new_2d_points, new_3d_points):
#         """Queue anchor reinitialization to be handled by worker thread"""
#         logger.info(f"Queueing anchor reinitialization for {new_anchor_path}")
        
#         # Check if the file exists and is readable
#         if not os.path.exists(new_anchor_path):
#             logger.error(f"Anchor image does not exist: {new_anchor_path}")
#             self.result_queue.put(('reinit_failed', None, -3, 0, ""))
#             return
        
#         # Try to read the image to verify it's a valid image file
#         test_img = cv2.imread(new_anchor_path)
#         if test_img is None:
#             logger.error(f"Failed to read anchor image: {new_anchor_path}")
#             self.result_queue.put(('reinit_failed', None, -3, 0, ""))
#             return
            
#         logger.info(f"Verified anchor image exists and is readable: {new_anchor_path}")
        
#         # Clear result queue
#         while not self.result_queue.empty():
#             try:
#                 self.result_queue.get_nowait()
#                 self.result_queue.task_done()
#             except queue.Empty:
#                 break
        
#         reinit_data = {
#             'command': 'reinit_anchor',
#             'new_anchor_path': new_anchor_path,
#             'new_2d_points': new_2d_points,
#             'new_3d_points': new_3d_points
#         }
#         self.frame_queue.put(reinit_data)
        
#         logger.info("Waiting for anchor reinitialization to complete...")

#     def get_result(self, timeout=5.0):
#         """Get a result from the result queue with timeout"""
#         try:
#             return self.result_queue.get(timeout=timeout)
#         except queue.Empty:
#             logger.warning(f"Timeout waiting for processing result")
#             return None, None, -1, 0, ""
    
#     def cleanup(self):
#         """Clean up resources and stop worker thread"""
#         logger.info("Cleaning up ThreadedPoseEstimator")
#         self.running = False
        
#         # Put None in queue to signal worker to stop
#         self.frame_queue.put(None)
        
#         # Wait for worker to finish
#         if self.worker_thread:
#             self.worker_thread.join(timeout=5.0)
#             if self.worker_thread.is_alive():
#                 logger.warning("Worker thread did not exit cleanly")