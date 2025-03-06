import argparse
import torch
torch.set_grad_enabled(False)

import time
from pathlib import Path
import cv2
import json
import os
import logging
import numpy as np
import csv

# Import the improved tracking-based pose estimator
# (Assuming you save the improved implementation in this file)
from pose_estimator_track import PoseEstimatorWithTracking
from utils import create_unique_filename
from models.utils import AverageTimer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main_improved_tracking.log"),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Improved Feature Tracking Pose Estimation',
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
        '--save_pose', type=str, default='pose_estimation_improved_tracking.json',
        help='Path to save JSON pose estimation results')
    parser.add_argument(
        '--min_tracked_points', type=int, default=5,
        help='Minimum number of tracked points before reinitializing')
    parser.add_argument(
        '--max_tracking_error', type=float, default=5.0,
        help='Maximum allowed optical flow error')
    parser.add_argument(
        '--anchor_switch_frames', type=str, default="133",
        help='Comma-separated list of frame indices at which to switch anchor')

    opt = parser.parse_args()
    logger.info(f"Parsed options: {opt}")

    # Parse anchor switch frames
    anchor_switch_frames = [int(x) for x in opt.anchor_switch_frames.split(',') if x.strip()]
    logger.info(f"Will switch anchors at frames: {anchor_switch_frames}")

    # Handle save_pose argument
    if os.path.isdir(opt.save_pose):
        base_filename = 'pose_estimation_improved_tracking.json'
        opt.save_pose = create_unique_filename(opt.save_pose, base_filename)
    else:
        save_dir = os.path.dirname(opt.save_pose)
        base_filename = os.path.basename(opt.save_pose)
        opt.save_pose = create_unique_filename(save_dir, base_filename)

    # Select device
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    logger.info(f'Running inference on device "{device}"')

    # Initialize the improved PoseEstimator with tracking
    pose_estimator = PoseEstimatorWithTracking(opt, device)
    
    # Update tracking parameters from command line if provided
    pose_estimator.min_tracked_points = opt.min_tracked_points
    pose_estimator.max_tracking_error = opt.max_tracking_error
    
    all_poses = []

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
        cv2.namedWindow('Improved Pose Estimation (Tracking)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Improved Pose Estimation (Tracking)', 640 * 2, 480)

    timer = AverageTimer()
    overall_start_time = time.time()
    frame_count = 0
    
    # Tracking statistics
    init_frames = 0
    tracking_frames = 0
    kalman_only_frames = 0
    tracking_mode = False
    current_tracking_streak = 0
    max_tracking_streak = 0
    
    # Define anchor switching data - using the same anchor data as in original code
    anchor_switch_data = {
        133: {
            "path": "Anchor_B.png",
            "points_2d": np.array([
                [650, 312], [630, 306], [907, 443], [814, 291], [599, 349],
                [501, 386], [965, 359], [649, 355], [635, 346], [930, 335],
                [843, 467], [702, 339], [718, 321], [930, 322], [727, 346],
                [539, 364], [786, 297], [1022, 406], [1004, 399], [539, 344],
                [536, 309], [864, 478], [745, 310], [1049, 393], [895, 258],
                [674, 347], [741, 281], [699, 294], [817, 494], [992, 281]
            ], dtype=np.float32),
            "points_3d": np.array([
                [-0.035, -0.018, -0.010], [-0.057, -0.018, -0.010],
                [ 0.217, -0.000, -0.027], [-0.014, -0.000,  0.156],
                [-0.023, -0.000, -0.065], [-0.014, -0.000, -0.156],
                [ 0.234, -0.050, -0.002], [ 0.000, -0.000, -0.042],
                [-0.014, -0.000, -0.042], [ 0.206, -0.055, -0.002],
                [ 0.217, -0.000, -0.070], [ 0.025, -0.014, -0.011],
                [-0.014, -0.000,  0.042], [ 0.206, -0.070, -0.002],
                [ 0.049, -0.016, -0.011], [-0.029, -0.000, -0.127],
                [-0.019, -0.000,  0.128], [ 0.230, -0.000,  0.070],
                [ 0.217, -0.000,  0.070], [-0.052, -0.000, -0.097],
                [-0.175, -0.000, -0.015], [ 0.230, -0.000, -0.070],
                [-0.019, -0.000,  0.074], [ 0.230, -0.000,  0.113],
                [-0.000, -0.025,  0.240], [-0.000, -0.000, -0.015],
                [-0.074, -0.000,  0.128], [-0.074, -0.000,  0.074],
                [ 0.230, -0.000, -0.113], [ 0.243, -0.104,  0.000]
            ], dtype=np.float32)
        }
    }

    # Process each frame
    for entry in entries:
        frame_count += 1

        frame_idx = entry['index']
        frame_t   = entry['timestamp']
        img_name  = entry['filename']

        image_path = os.path.join(opt.image_dir, img_name)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            # Even if image is missing, get Kalman prediction
            x_pred, P_pred = pose_estimator.kf_pose.predict()
            kf_translation = x_pred[0:3]
            kf_quaternion = x_pred[6:10]
            pose_data = {
                'frame': frame_idx,
                'timestamp': frame_t,
                'image_file': img_name,
                'kf_translation_vector': kf_translation.tolist(),
                'kf_quaternion': kf_quaternion.tolist(),
                'pose_estimation_failed': True,
                'estimation_method': 'kalman_prediction_only',
                'reason': 'image_not_found'
            }
            all_poses.append(pose_data)
            kalman_only_frames += 1
            continue

        # Load the image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.warning(f"Failed to load image: {image_path}")
            # Even if image failed to load, get Kalman prediction
            x_pred, P_pred = pose_estimator.kf_pose.predict()
            kf_translation = x_pred[0:3]
            kf_quaternion = x_pred[6:10]
            pose_data = {
                'frame': frame_idx,
                'timestamp': frame_t,
                'image_file': img_name,
                'kf_translation_vector': kf_translation.tolist(),
                'kf_quaternion': kf_quaternion.tolist(),
                'pose_estimation_failed': True,
                'estimation_method': 'kalman_prediction_only',
                'reason': 'image_load_failed'
            }
            all_poses.append(pose_data)
            kalman_only_frames += 1
            continue

        start_time = time.time()
        logger.debug(f"Processing frame {frame_idx} from {image_path}, timestamp={frame_t:.3f}")

        # Check if we need to switch anchors at this frame
        if frame_idx in anchor_switch_data:
            logger.info(f"Switching to a new anchor at frame {frame_idx}...")
            anchor_data = anchor_switch_data[frame_idx]
            
            pose_estimator.reinitialize_anchor(
                anchor_data["path"],
                anchor_data["points_2d"],
                anchor_data["points_3d"]
            )
            logger.info(f"Anchor switched successfully at frame {frame_idx}")

        # Process frame (pose estimation with tracking)
        pose_data, visualization = pose_estimator.process_frame(frame, frame_idx)

        # Always add timestamp and filename to pose data
        pose_data['timestamp'] = frame_t
        pose_data['image_file'] = img_name
        
        # Track statistics about estimation method
        if 'estimation_method' in pose_data:
            if pose_data['estimation_method'] == 'initialization':
                init_frames += 1
                # Reset tracking streak if we were in tracking mode
                if tracking_mode:
                    if current_tracking_streak > max_tracking_streak:
                        max_tracking_streak = current_tracking_streak
                    current_tracking_streak = 0
                tracking_mode = False
            elif pose_data['estimation_method'] == 'tracking':
                tracking_frames += 1
                current_tracking_streak += 1
                tracking_mode = True
            elif pose_data['estimation_method'] == 'kalman_prediction_only':
                kalman_only_frames += 1
                # Don't reset tracking streak for Kalman-only frames as we might 
                # recover tracking in the next frame
        elif 'tracking_mode' in pose_data:
            # For backward compatibility with older code
            if pose_data['tracking_mode']:
                tracking_frames += 1
                current_tracking_streak += 1
                tracking_mode = True
            else:
                init_frames += 1
                if current_tracking_streak > max_tracking_streak:
                    max_tracking_streak = current_tracking_streak
                current_tracking_streak = 0
                tracking_mode = False

        # Add the pose data to our collection
        all_poses.append(pose_data)
        logger.debug(f'Pose data (frame={frame_idx}, time={frame_t:.3f}): {pose_data}')

        # Show the visualization
        if not opt.no_display and visualization is not None:
            # Add tracking mode text overlay
            if 'estimation_method' in pose_data:
                if pose_data['estimation_method'] == 'initialization':
                    mode_text = "INITIALIZATION MODE"
                elif pose_data['estimation_method'] == 'tracking':
                    mode_text = "TRACKING MODE"
                else:
                    mode_text = "KALMAN PREDICTION ONLY"
            else:
                mode_text = "TRACKING MODE" if tracking_mode else "INITIALIZATION MODE"
                
            cv2.putText(visualization, f"Mode: {mode_text}", 
                       (30, visualization.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2, cv2.LINE_AA)
                       
            cv2.imshow('Improved Pose Estimation (Tracking)', visualization)
            if cv2.waitKey(1) == ord('q'):
                logger.info('Exiting on user request (q key pressed).')
                break

        # Optionally save the visualization
        if opt.output_dir and visualization is not None:
            out_file = str(Path(opt.output_dir, f'frame_{frame_idx:06d}.png'))
            cv2.imwrite(out_file, visualization)

        end_time = time.time()
        logger.info(f"Frame {frame_idx} processed in {end_time - start_time:.3f}s.")

    overall_end_time = time.time()

    # Update max tracking streak one last time if we end in tracking mode
    if current_tracking_streak > max_tracking_streak:
        max_tracking_streak = current_tracking_streak

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
        
    # Report tracking statistics
    logger.info(f"Initialization frames: {init_frames}")
    logger.info(f"Tracking frames: {tracking_frames}")
    logger.info(f"Kalman-prediction-only frames: {kalman_only_frames}")
    logger.info(f"Max consecutive tracking streak: {max_tracking_streak}")
    
    total_frames_with_pose = init_frames + tracking_frames + kalman_only_frames
    if total_frames_with_pose > 0:
        init_percentage = (init_frames / total_frames_with_pose) * 100
        tracking_percentage = (tracking_frames / total_frames_with_pose) * 100
        kalman_percentage = (kalman_only_frames / total_frames_with_pose) * 100
        
        logger.info(f"Initialization percentage: {init_percentage:.2f}%")
        logger.info(f"Tracking percentage: {tracking_percentage:.2f}%")
        logger.info(f"Kalman-only percentage: {kalman_percentage:.2f}%")
        
        # Calculate approximate processing time saved
        approx_time_per_init = 0.2  # 200ms for initialization (example)
        approx_time_per_track = 0.02  # 20ms for tracking (example)
        approx_time_per_kalman = 0.01  # 10ms for Kalman only (example)
        
        naive_time = frame_count * approx_time_per_init
        actual_time = (init_frames * approx_time_per_init + 
                      tracking_frames * approx_time_per_track +
                      kalman_only_frames * approx_time_per_kalman)
        
        logger.info(f"Estimated processing time saved: {naive_time - actual_time:.2f}s")
        logger.info(f"Estimated speed improvement: {naive_time / actual_time:.2f}x")

    cv2.destroyAllWindows()

    # Save pose estimation results
    with open(opt.save_pose, 'w') as f:
        json.dump(all_poses, f, indent=4)
    logger.info(f'Pose estimation results saved to {opt.save_pose}')
            