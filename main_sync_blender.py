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

# Import PoseEstimator from your chosen implementation
from pose_estimator_Pixel_blender import PoseEstimator
from utils import create_unique_filename
from models.utils import AverageTimer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main_LG.log"),
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
    if csv_path and csv_path.lower() != 'none':
        try:
            import csv
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
            logger.info(f"Loaded {len(entries)} entries from CSV file {csv_path}")
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
    return entries

def get_image_entries_from_json(json_path):
    """
    Extract image filenames from a JSON file containing ground truth poses
    Returns a list of dicts: [{'index':..., 'timestamp':..., 'filename':...}, ...]
    """
    entries = []
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle JSON with "frames" structure
        if 'frames' in data:
            for i, frame in enumerate(data['frames']):
                # Extract filename from the frame data
                image_name = frame['image']
                entries.append({
                    'index': i + 1,  # 1-based index
                    'timestamp': float(i) / 30.0,  # Assuming 30 fps
                    'filename': image_name
                })
            logger.info(f"Extracted {len(entries)} entries from JSON file {json_path}")
        else:
            logger.warning(f"No 'frames' key found in JSON file {json_path}")
    except Exception as e:
        logger.error(f"Error extracting entries from JSON: {str(e)}")
    
    return entries

def list_image_entries_from_directory(image_dir):
    """
    List all image files in a directory
    Returns a list of dicts: [{'index':..., 'timestamp':..., 'filename':...}, ...]
    """
    entries = []
    
    # List of common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    try:
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
            image_files.extend(list(Path(image_dir).glob(f'*{ext.upper()}')))
        
        # Sort files naturally (so frame10 comes after frame9, not frame1)
        image_files.sort()
        
        for i, file_path in enumerate(image_files):
            entries.append({
                'index': i + 1,  # 1-based index
                'timestamp': float(i) / 30.0,  # Assuming 30 fps
                'filename': file_path.name
            })
        
        logger.info(f"Found {len(entries)} images in directory {image_dir}")
    except Exception as e:
        logger.error(f"Error listing images from directory: {str(e)}")
    
    return entries

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LightGlue Pose Estimation (Image Folder with Reinit)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--image_dir', type=str, required=True,
        help='Folder containing extracted images (PNG/JPG, etc.)')
    parser.add_argument(
        '--csv_file', type=str, required=False, default=None,
        help='CSV (image_index.csv) with columns [Index, Timestamp, Filename]. Optional if using --ground_truth.')
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
        '--save_pose', type=str, default='pose_estimation_research.json',
        help='Path to save JSON pose estimation results')
    parser.add_argument(
        '--ground_truth', type=str, default=None,
        help='Path to ground truth JSON file for pose comparison')

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

    # Initialize PoseEstimator
    pose_estimator = PoseEstimator(opt, device)
    all_poses = []

    # Load ground truth data if provided explicitly
    if opt.ground_truth:
        pose_estimator.gt_poses = pose_estimator.load_ground_truth_poses(opt.ground_truth)
        logger.info(f"Loaded ground truth from {opt.ground_truth}")
    else:
        # Try to use the CSV file as ground truth source too
        try:
            if opt.csv_file and opt.csv_file.lower() != 'none':
                pose_estimator.gt_poses = pose_estimator.load_ground_truth_poses(opt.csv_file)
                if pose_estimator.gt_poses:
                    logger.info(f"Using CSV file for ground truth: {opt.csv_file}")
        except Exception as e:
            logger.warning(f"Could not load ground truth from CSV: {str(e)}")
            pose_estimator.gt_poses = {}

    # Read entries (either from CSV, or from ground truth JSON, or from image directory)
    entries = []
    if opt.csv_file and opt.csv_file.lower() != 'none':
        entries = read_image_index_csv(opt.csv_file)
        
    # If CSV is missing or empty, try to get entries from ground truth JSON
    if not entries and opt.ground_truth:
        entries = get_image_entries_from_json(opt.ground_truth)
        
    # If still no entries, just list all image files in the directory
    if not entries:
        entries = list_image_entries_from_directory(opt.image_dir)
        
    if len(entries) == 0:
        logger.error("No image entries found. Exiting.")
        exit(1)

    logger.info(f"Found {len(entries)} frames to process")

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

    timer = AverageTimer()
    overall_start_time = time.time()
    frame_count = 0

    # Directory for debug visualizations
    #debug_dir = "debug_projections"
    #Path(debug_dir).mkdir(exist_ok=True)

    for entry in entries:
        frame_count += 1

        frame_idx = entry['index']
        frame_t = entry['timestamp']
        img_name = entry['filename']

        image_path = os.path.join(opt.image_dir, img_name)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue

        # Load the image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.warning(f"Failed to load image: {image_path}")
            continue

        start_time = time.time()
        logger.debug(f"Processing frame {frame_idx} from {image_path}, timestamp={frame_t:.3f}")

        # ---------------------------------------------------------
        # Anchor reinitialize handling (customize as needed)
        # ---------------------------------------------------------
        if frame_idx == 400:  # Example frame number for reinitialization
            logger.info("Switching to a new anchor...")
            new_anchor_path = "Anchor_B.png"  # Update this path as needed
            
            # Define 2D and 3D correspondences for the new anchor
            # (your existing points here)
            new_2d_points = np.array([
                [650, 312],
                [630, 306],
                # ... more points
            ], dtype=np.float32)

            new_3d_points = np.array([
                [-0.035, -0.018, -0.010],
                [-0.057, -0.018, -0.010],
                # ... more points
            ], dtype=np.float32)

            # Reinitialize anchor (only if needed)
            # pose_estimator.reinitialize_anchor(
            #     new_anchor_path,
            #     new_2d_points,
            #     new_3d_points
            # )
        # ---------------------------------------------------------

        # Process frame (pose estimation)
        pose_data, visualization = pose_estimator.process_frame(frame, frame_idx, img_name)

        # Store image metadata
        pose_data['timestamp'] = frame_t
        pose_data['image_file'] = img_name
        all_poses.append(pose_data)
        logger.debug(f'Pose data (frame={frame_idx}, time={frame_t:.3f}): {pose_data}')

        # Show ground truth comparison if available
        if 'rotation_error_rad' in pose_data and 'translation_error' in pose_data:
            rot_err = pose_data['rotation_error_rad']
            trans_err = pose_data['translation_error']
            logger.info(f"Ground Truth Comparison - Rotation Error: {rot_err:.4f} rad, Translation Error: {trans_err:.4f}")
            
            # Add error information to visualization
            if visualization is not None:
                error_text = f"Rot Err: {rot_err:.4f} rad, Trans Err: {trans_err:.4f}"
                cv2.putText(visualization, error_text, (10, visualization.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Show the visualization
        if not opt.no_display and visualization is not None:
            cv2.imshow('Pose Estimation', visualization)
            if cv2.waitKey(1) == ord('q'):
                logger.info('Exiting on user request (q key pressed).')
                break

        # Save debug visualization
        #debug_file = os.path.join(debug_dir, f'debug_{frame_idx:04d}.png')
        #if visualization is not None:
        #    cv2.imwrite(debug_file, visualization)

        # Optionally save the regular visualization
        if opt.output_dir and visualization is not None:
            out_file = str(Path(opt.output_dir, f'frame_{frame_idx:06d}.png'))
            cv2.imwrite(out_file, visualization)

        end_time = time.time()
        logger.info(f"Frame {frame_idx} processed in {end_time - start_time:.3f}s.")

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

    # Calculate error statistics if we had ground truth comparisons
    if any('rotation_error_rad' in pose for pose in all_poses):
        rot_errors = [pose['rotation_error_rad'] for pose in all_poses if 'rotation_error_rad' in pose]
        trans_errors = [pose['translation_error'] for pose in all_poses if 'translation_error' in pose]
        
        mean_rot_error = np.mean(rot_errors)
        mean_trans_error = np.mean(trans_errors)
        
        logger.info(f"Ground Truth Comparison Summary:")
        logger.info(f"Mean Rotation Error: {mean_rot_error:.4f} rad")
        logger.info(f"Mean Translation Error: {mean_trans_error:.4f}")

    # Save pose estimation results
    with open(opt.save_pose, 'w') as f:
        json.dump(all_poses, f, indent=4)
    logger.info(f'Pose estimation results saved to {opt.save_pose}')