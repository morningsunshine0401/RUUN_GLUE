import argparse
import torch
# Disable gradient computation globally
torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)

import time
from pathlib import Path
import cv2
import queue
import threading
import atexit

# Import our new ThreadedPoseEstimator that uses separate models
from threaded_pose_estimator import ThreadedPoseEstimator

from utils import create_unique_filename
from models.utils import AverageTimer
import json
import os
import logging
import numpy as np
import csv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main_separate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
            tstamp    = float(row['Timestamp'])
            fname     = row['Filename']
            entries.append({
                'index': frame_idx,
                'timestamp': tstamp,
                'filename': fname
            })
    return entries

def frame_producer(image_dir, entries, frame_queue, max_queue_size=10):
    """
    Producer function that reads frames and adds them to the queue
    """
    logger.info(f"Starting frame producer, processing {len(entries)} frames")
    
    for entry in entries:
        # Check if queue is almost full, wait if needed
        while frame_queue.qsize() >= max_queue_size - 1:
            time.sleep(0.01)
            
        frame_idx = entry['index']
        frame_t = entry['timestamp']
        img_name = entry['filename']
        
        image_path = os.path.join(image_dir, img_name)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
        
        # Load the image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.warning(f"Failed to load image: {image_path}")
            continue
        
        logger.debug(f"Producer: Adding frame {frame_idx} to queue")
        frame_queue.put((frame, frame_idx, frame_t, img_name))
    
    # Signal end of frames
    logger.info("Producer: All frames have been queued")
    frame_queue.put(None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LightGlue Pose Estimation with Separate Models (Image Folder with Reinit)',
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
        '--save_pose', type=str, default='pose_estimation_separate.json',
        help='Path to save JSON pose estimation results')
    parser.add_argument(
        '--num_threads', type=int, default=2,
        help='Number of worker threads for processing')

    opt = parser.parse_args()
    logger.info(f"Parsed options: {opt}")

    # Handle save_pose argument
    if os.path.isdir(opt.save_pose):
        base_filename = 'pose_estimation.json'
        opt.save_pose = create_unique_filename(opt.save_pose, base_filename)
    else:
        save_dir = os.path.dirname(opt.save_pose)
        base_filename = os.path.basename(opt.save_pose)
        opt.save_pose = create_unique_filename(save_dir, base_filename)

    # Select device
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    logger.info(f'Running inference on device "{device}"')

    # Read CSV listing all frames
    entries = read_image_index_csv(opt.csv_file)
    if len(entries) == 0:
        logger.error("No entries found in CSV file. Exiting.")
        exit(1)

    logger.info(f"Found {len(entries)} frames in {opt.csv_file}")

    # Prepare output directory if needed
    if opt.output_dir is not None:
        logger.info(f'Will write output visualizations to {opt.output_dir}')
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Prepare display window
    if opt.no_display:
        logger.info('Skipping visualization, no GUI window will be shown.')
    else:
        cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pose Estimation', 640 * 2, 480)

    # Create threaded pose estimator
    pose_estimator = ThreadedPoseEstimator(opt, device)
    
    # Create queues for producer-consumer pattern
    frame_queue = queue.Queue(maxsize=10)
    
    # Start producer thread for loading images
    producer_thread = threading.Thread(
        target=frame_producer, 
        args=(opt.image_dir, entries, frame_queue)
    )
    producer_thread.daemon = True
    producer_thread.start()
    
    # Process results
    all_poses = []
    frame_count = 0
    overall_start_time = time.time()
    
    # Handle anchor reinitialization
    anchor_switch_frame = 133#444  # Frame at which to switch anchor
    new_anchor_initialized = False
    
    # Main processing loop
    while True:
        try:
            # Check for 'q' key press (non-blocking)
            key = cv2.waitKey(1)
            if key == ord('q'):
                logger.info('Exiting on user request (q key pressed).')
                break

            # Get frame from producer
            try:
                frame_data = frame_queue.get(timeout=0.1)
            except queue.Empty:
                # No new frames, just continue
                continue

            if frame_data is None:
                # End of frames
                logger.info("Reached end of frames")
                break
            
            frame, frame_idx, frame_t, img_name = frame_data
            start_time = time.time()
            
            logger.debug(f"Processing frame {frame_idx} from {img_name}, timestamp={frame_t:.3f}")
            
            # Check if we need to switch anchor
            if frame_idx == anchor_switch_frame:
                logger.info(f"Switching to a new anchor at frame {frame_idx}...")
                new_anchor_path = os.path.abspath("Anchor_B.png")
                logger.info(f"Using anchor file at: {new_anchor_path}")
                
                # Example new 2D/3D correspondences for the new anchor
                new_2d_points = np.array([
                    [650, 312],
                    [630, 306],
                    [907, 443],
                    [814, 291],
                    [599, 349],
                    [501, 386],
                    [965, 359],
                    [649, 355],
                    [635, 346],
                    [930, 335],
                    [843, 467],
                    [702, 339],
                    [718, 321],
                    [930, 322],
                    [727, 346],
                    [539, 364],
                    [786, 297],
                    [1022, 406],
                    [1004, 399],
                    [539, 344],
                    [536, 309],
                    [864, 478],
                    [745, 310],
                    [1049, 393],
                    [895, 258],
                    [674, 347],
                    [741, 281],
                    [699, 294],
                    [817, 494],
                    [992, 281]
                ], dtype=np.float32)

                new_3d_points = np.array([
                    [-0.035, -0.018, -0.010],
                    [-0.057, -0.018, -0.010],
                    [ 0.217, -0.000, -0.027],
                    [-0.014, -0.000,  0.156],
                    [-0.023, -0.000, -0.065],
                    [-0.014, -0.000, -0.156],
                    [ 0.234, -0.050, -0.002],
                    [ 0.000, -0.000, -0.042],
                    [-0.014, -0.000, -0.042],
                    [ 0.206, -0.055, -0.002],
                    [ 0.217, -0.000, -0.070],
                    [ 0.025, -0.014, -0.011],
                    [-0.014, -0.000,  0.042],
                    [ 0.206, -0.070, -0.002],
                    [ 0.049, -0.016, -0.011],
                    [-0.029, -0.000, -0.127],
                    [-0.019, -0.000,  0.128],
                    [ 0.230, -0.000,  0.070],
                    [ 0.217, -0.000,  0.070],
                    [-0.052, -0.000, -0.097],
                    [-0.175, -0.000, -0.015],
                    [ 0.230, -0.000, -0.070],
                    [-0.019, -0.000,  0.074],
                    [ 0.230, -0.000,  0.113],
                    [-0.000, -0.025,  0.240],
                    [-0.000, -0.000, -0.015],
                    [-0.074, -0.000,  0.128],
                    [-0.074, -0.000,  0.074],
                    [ 0.230, -0.000, -0.113],
                    [ 0.243, -0.104,  0.000]
                ], dtype=np.float32)

                pose_estimator.reinitialize_anchor(
                    new_anchor_path,
                    new_2d_points,
                    new_3d_points
                )
                
                # Wait for confirmation that reinitialization is complete
                try:
                    logger.info("Waiting for anchor reinitialization to complete...")
                    reinit_result = pose_estimator.result_queue.get(timeout=30.0)  # Increased timeout to 30 seconds
                    
                    if reinit_result[0] == 'reinit_complete':
                        logger.info("Anchor reinitialization completed successfully")
                        new_anchor_initialized = True
                    else:
                        logger.error("Anchor reinitialization did not complete correctly")
                        logger.error(f"Reinit result: {reinit_result}")
                except queue.Empty:
                    logger.error("Timeout waiting for anchor reinitialization")
                    
                # Mark this task as done since we've handled the special case
                frame_queue.task_done()
                
                # Skip to the next frame after reinitializing
                logger.info("Skipping to next frame after anchor reinitialization")
                continue
            
            # Process frame
            pose_estimator.process_frame(frame, frame_idx, frame_t, img_name)
            frame_queue.task_done()
            
            # Get result (this will wait for processing to complete)
            pose_data, visualization, result_idx, result_t, result_name = pose_estimator.get_result()
            
            if pose_data is not None:
                all_poses.append(pose_data)
                logger.debug(f'Pose data (frame={result_idx}, time={result_t:.3f}): {pose_data}')
                
                # Show the visualization
                if not opt.no_display and visualization is not None:
                    cv2.imshow('Pose Estimation', visualization)

                # Optionally save the visualization
                if opt.output_dir and visualization is not None:
                    out_file = str(Path(opt.output_dir, f'frame_{result_idx:06d}.png'))
                    cv2.imwrite(out_file, visualization)
            else:
                logger.warning(f"No pose data for frame {result_idx}")
            
            frame_count += 1
            end_time = time.time()
            logger.info(f"Frame {result_idx} processed in {end_time - start_time:.3f}s.")
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected. Exiting...")
            break
        
        except Exception as e:
            logger.error(f"Error in main processing loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Try to continue with next frame
    
    # Wait for producer thread to finish
    producer_thread.join()
    
    # Cleanup threaded pose estimator
    pose_estimator.cleanup()
    
    overall_end_time = time.time()

    # Calculate overall FPS
    total_elapsed_time = overall_end_time - overall_start_time
    if frame_count > 0 and total_elapsed_time > 0:
        total_fps = frame_count / total_elapsed_time
        logger.info(
            f"Processed {frame_count} frames in {total_elapsed_time:.2f}s "
            f"(Total FPS: {total_fps:.2f})"
        )
    else:
        logger.info("No frames were processed or invalid total time.")

    cv2.destroyAllWindows()

    # Save pose estimation results
    with open(opt.save_pose, 'w') as f:
        json.dump(all_poses, f, indent=4)
    logger.info(f'Pose estimation results saved to {opt.save_pose}')

##############################################################################################
# import argparse
# import torch
# # Disable gradient computation globally
# torch.set_grad_enabled(False)
# torch.autograd.set_grad_enabled(False)

# import time
# from pathlib import Path
# import cv2
# import queue
# import threading
# import atexit

# # Import our new ThreadedPoseEstimator that uses separate models
# from threaded_pose_estimator import ThreadedPoseEstimator

# from utils import create_unique_filename
# from models.utils import AverageTimer
# import json
# import os
# import logging
# import numpy as np
# import csv

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,  # Use DEBUG for detailed logs
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("main_separate.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# def read_image_index_csv(csv_path):
#     """
#     Reads a CSV with columns: Index, Timestamp, Filename
#     Returns a list of dicts: [{'index':..., 'timestamp':..., 'filename':...}, ...]
#     """
#     entries = []
#     with open(csv_path, 'r', newline='', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             frame_idx = int(row['Index'])
#             tstamp    = float(row['Timestamp'])
#             fname     = row['Filename']
#             entries.append({
#                 'index': frame_idx,
#                 'timestamp': tstamp,
#                 'filename': fname
#             })
#     return entries

# def frame_producer(image_dir, entries, frame_queue, max_queue_size=10):
#     """
#     Producer function that reads frames and adds them to the queue
#     """
#     logger.info(f"Starting frame producer, processing {len(entries)} frames")
    
#     for entry in entries:
#         # Check if queue is almost full, wait if needed
#         while frame_queue.qsize() >= max_queue_size - 1:
#             time.sleep(0.01)
            
#         frame_idx = entry['index']
#         frame_t = entry['timestamp']
#         img_name = entry['filename']
        
#         image_path = os.path.join(image_dir, img_name)
#         if not os.path.exists(image_path):
#             logger.warning(f"Image not found: {image_path}")
#             continue
        
#         # Load the image
#         frame = cv2.imread(image_path)
#         if frame is None:
#             logger.warning(f"Failed to load image: {image_path}")
#             continue
        
#         logger.debug(f"Producer: Adding frame {frame_idx} to queue")
#         frame_queue.put((frame, frame_idx, frame_t, img_name))
    
#     # Signal end of frames
#     logger.info("Producer: All frames have been queued")
#     frame_queue.put(None)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='LightGlue Pose Estimation with Separate Models (Image Folder with Reinit)',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument(
#         '--image_dir', type=str, required=True,
#         help='Folder containing extracted images (PNG/JPG, etc.)')
#     parser.add_argument(
#         '--csv_file', type=str, required=True,
#         help='CSV (image_index.csv) with columns [Index, Timestamp, Filename]')
#     parser.add_argument(
#         '--anchor', type=str, required=True,
#         help='Path to the initial anchor (reference) image')
#     parser.add_argument(
#         '--output_dir', type=str, default=None,
#         help='Directory where to write output frames (visualization). If None, no output images saved.')
#     parser.add_argument(
#         '--resize', type=int, nargs='+', default=[640, 480],
#         help='Resize the input image before running inference (width height).')
#     parser.add_argument(
#         '--show_keypoints', action='store_true',
#         help='Show detected keypoints')
#     parser.add_argument(
#         '--no_display', action='store_true',
#         help='Do not display images in a GUI window.')
#     parser.add_argument(
#         '--force_cpu', action='store_true',
#         help='Force PyTorch to run on CPU even if CUDA is available.')
#     parser.add_argument(
#         '--save_pose', type=str, default='pose_estimation_separate.json',
#         help='Path to save JSON pose estimation results')
#     parser.add_argument(
#         '--num_threads', type=int, default=2,
#         help='Number of worker threads for processing')

#     opt = parser.parse_args()
#     logger.info(f"Parsed options: {opt}")

#     # Handle save_pose argument
#     if os.path.isdir(opt.save_pose):
#         base_filename = 'pose_estimation.json'
#         opt.save_pose = create_unique_filename(opt.save_pose, base_filename)
#     else:
#         save_dir = os.path.dirname(opt.save_pose)
#         base_filename = os.path.basename(opt.save_pose)
#         opt.save_pose = create_unique_filename(save_dir, base_filename)

#     # Select device
#     device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
#     logger.info(f'Running inference on device "{device}"')

#     # Read CSV listing all frames
#     entries = read_image_index_csv(opt.csv_file)
#     if len(entries) == 0:
#         logger.error("No entries found in CSV file. Exiting.")
#         exit(1)

#     logger.info(f"Found {len(entries)} frames in {opt.csv_file}")

#     # Prepare output directory if needed
#     if opt.output_dir is not None:
#         logger.info(f'Will write output visualizations to {opt.output_dir}')
#         Path(opt.output_dir).mkdir(exist_ok=True)

#     # Prepare display window
#     if opt.no_display:
#         logger.info('Skipping visualization, no GUI window will be shown.')
#     else:
#         cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('Pose Estimation', 640 * 2, 480)

#     # Create threaded pose estimator
#     pose_estimator = ThreadedPoseEstimator(opt, device)
    
#     # Create queues for producer-consumer pattern
#     frame_queue = queue.Queue(maxsize=10)
    
#     # Start producer thread for loading images
#     producer_thread = threading.Thread(
#         target=frame_producer, 
#         args=(opt.image_dir, entries, frame_queue)
#     )
#     producer_thread.daemon = True
#     producer_thread.start()
    
#     # Process results
#     all_poses = []
#     frame_count = 0
#     overall_start_time = time.time()
    
#     # Handle anchor reinitialization
#     anchor_switch_frame = 444  # Frame at which to switch anchor
#     new_anchor_initialized = False
    
#     # Main processing loop
#     while True:
#         # Get frame from producer
#         frame_data = frame_queue.get()
#         if frame_data is None:
#             # End of frames
#             logger.info("Reached end of frames")
#             break
        
#         frame, frame_idx, frame_t, img_name = frame_data
#         start_time = time.time()
        
#         logger.debug(f"Processing frame {frame_idx} from {img_name}, timestamp={frame_t:.3f}")
        
#         # Check if we need to switch anchor
#         if frame_idx == anchor_switch_frame:
#             logger.info(f"Switching to a new anchor at frame {frame_idx}...")
#             new_anchor_path = os.path.abspath("Anchor_B.png")
#             logger.info(f"Using anchor file at: {new_anchor_path}")
            
#             # Example new 2D/3D correspondences for the new anchor
#             new_2d_points = np.array([
#                 [650, 312],
#                 [630, 306],
#                 [907, 443],
#                 [814, 291],
#                 [599, 349],
#                 [501, 386],
#                 [965, 359],
#                 [649, 355],
#                 [635, 346],
#                 [930, 335],
#                 [843, 467],
#                 [702, 339],
#                 [718, 321],
#                 [930, 322],
#                 [727, 346],
#                 [539, 364],
#                 [786, 297],
#                 [1022, 406],
#                 [1004, 399],
#                 [539, 344],
#                 [536, 309],
#                 [864, 478],
#                 [745, 310],
#                 [1049, 393],
#                 [895, 258],
#                 [674, 347],
#                 [741, 281],
#                 [699, 294],
#                 [817, 494],
#                 [992, 281]
#             ], dtype=np.float32)

#             new_3d_points = np.array([
#                 [-0.035, -0.018, -0.010],
#                 [-0.057, -0.018, -0.010],
#                 [ 0.217, -0.000, -0.027],
#                 [-0.014, -0.000,  0.156],
#                 [-0.023, -0.000, -0.065],
#                 [-0.014, -0.000, -0.156],
#                 [ 0.234, -0.050, -0.002],
#                 [ 0.000, -0.000, -0.042],
#                 [-0.014, -0.000, -0.042],
#                 [ 0.206, -0.055, -0.002],
#                 [ 0.217, -0.000, -0.070],
#                 [ 0.025, -0.014, -0.011],
#                 [-0.014, -0.000,  0.042],
#                 [ 0.206, -0.070, -0.002],
#                 [ 0.049, -0.016, -0.011],
#                 [-0.029, -0.000, -0.127],
#                 [-0.019, -0.000,  0.128],
#                 [ 0.230, -0.000,  0.070],
#                 [ 0.217, -0.000,  0.070],
#                 [-0.052, -0.000, -0.097],
#                 [-0.175, -0.000, -0.015],
#                 [ 0.230, -0.000, -0.070],
#                 [-0.019, -0.000,  0.074],
#                 [ 0.230, -0.000,  0.113],
#                 [-0.000, -0.025,  0.240],
#                 [-0.000, -0.000, -0.015],
#                 [-0.074, -0.000,  0.128],
#                 [-0.074, -0.000,  0.074],
#                 [ 0.230, -0.000, -0.113],
#                 [ 0.243, -0.104,  0.000]
#             ], dtype=np.float32)

#             pose_estimator.reinitialize_anchor(
#                 new_anchor_path,
#                 new_2d_points,
#                 new_3d_points
#             )
#             # Wait for confirmation that reinitialization is complete
#             try:
#                 reinit_result = pose_estimator.result_queue.get(timeout=10.0)
#                 if reinit_result[0] == 'reinit_complete':
#                     logger.info("Anchor reinitialization completed successfully")
#                     new_anchor_initialized = True
#                 else:
#                     logger.error("Anchor reinitialization did not complete correctly")
#             except queue.Empty:
#                 logger.error("Timeout waiting for anchor reinitialization")
        
#         # Process frame
#         pose_estimator.process_frame(frame, frame_idx, frame_t, img_name)
#         frame_queue.task_done()
        
#         # Get result (this will wait for processing to complete)
#         pose_data, visualization, result_idx, result_t, result_name = pose_estimator.get_result()
        
#         if pose_data is not None:
#             all_poses.append(pose_data)
#             logger.debug(f'Pose data (frame={result_idx}, time={result_t:.3f}): {pose_data}')
            
#             # Show the visualization
#             if not opt.no_display and visualization is not None:
#                 cv2.imshow('Pose Estimation', visualization)
#                 if cv2.waitKey(1) == ord('q'):
#                     logger.info('Exiting on user request (q key pressed).')
#                     break

#             # Optionally save the visualization
#             if opt.output_dir and visualization is not None:
#                 out_file = str(Path(opt.output_dir, f'frame_{result_idx:06d}.png'))
#                 cv2.imwrite(out_file, visualization)
#         else:
#             logger.warning(f"No pose data for frame {result_idx}")
        
#         frame_count += 1
#         end_time = time.time()
#         logger.info(f"Frame {result_idx} processed in {end_time - start_time:.3f}s.")
    
#     # Wait for producer thread to finish
#     producer_thread.join()
    
#     # Cleanup threaded pose estimator
#     pose_estimator.cleanup()
    
#     overall_end_time = time.time()

#     # Calculate overall FPS
#     total_elapsed_time = overall_end_time - overall_start_time
#     if frame_count > 0 and total_elapsed_time > 0:
#         total_fps = frame_count / total_elapsed_time
#         logger.info(
#             f"Processed {frame_count} frames in {total_elapsed_time:.2f}s "
#             f"(Total FPS: {total_fps:.2f})"
#         )
#     else:
#         logger.info("No frames were processed or invalid total time.")

#     cv2.destroyAllWindows()

#     # Save pose estimation results
#     with open(opt.save_pose, 'w') as f:
#         json.dump(all_poses, f, indent=4)
#     logger.info(f'Pose estimation results saved to {opt.save_pose}')