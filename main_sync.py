import argparse
import torch
torch.set_grad_enabled(False)

import time
from pathlib import Path
import cv2

#ORB
#from pose_estimator_ORB_ICUAS import PoseEstimator

# Default
#from pose_estimator_ICUAS import PoseEstimator

# 20250210 KF testing version 
#from pose_estimator_20250210_KF_test import PoseEstimator

## Adaptive code
#from pose_estimator_ICUAS_20250122 import PoseEstimator

### Adaptive 2
#from pose_estimator_adaptive_20250210 import PoseEstimator

### Adaptive 3
#from pose_estimator_adaptive_20250218 import PoseEstimator


### Quaternion 1
from pose_estimator_Q import PoseEstimator

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
        description='LightGlue Pose Estimation (Image Folder with Reinit)',
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
        '--save_pose', type=str, default='pose_estimation_research.json',
        help='Path to save JSON pose estimation results')

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

    timer = AverageTimer()
    overall_start_time = time.time()
    frame_count = 0

    for entry in entries:
        frame_count += 1

        frame_idx = entry['index']
        frame_t   = entry['timestamp']
        img_name  = entry['filename']

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
        # EXAMPLE: Switch anchor after frame 520 (Adjust as needed)
        # ---------------------------------------------------------
        if frame_idx == 133:#100:##133:#120:#55:#144:#60:#144:#135:#144:#152:#3000:#133:#112:
            logger.info("Switching to a new anchor after frame 520...")
            new_anchor_path = "Anchor_B.png"
            #new_anchor_path = "assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png"

            # # Example anchor keypoints (2D -> 3D). Adapt these to your scenario.
            # new_2d_points = np.array([
            #     [511, 293], [591, 284], [610, 269], [587, 330], [413, 249],
            #     [602, 348], [715, 384], [598, 298], [656, 171], [805, 213],
            #     [703, 392], [523, 286], [519, 327], [387, 289], [727, 126],
            #     [425, 243], [636, 358], [745, 202], [595, 388], [436, 260],
            #     [539, 313], [795, 220], [351, 291], [665, 165], [611, 353],
            #     [650, 377], [516, 389], [727, 143], [496, 378], [575, 312],
            #     [617, 368], [430, 312], [480, 281], [834, 225], [469, 339],
            #     [705, 223], [637, 156], [816, 414], [357, 195], [752, 77],
            #     [642, 451]
            # ], dtype=np.float32)

            # new_3d_points = np.array([
            #     [-0.014, 0.000,  0.042], [ 0.025, -0.014, -0.011], [ 0.049, -0.016, -0.011],
            #     [-0.014, 0.000, -0.042], [-0.014,  0.000,  0.156], [-0.023,  0.000, -0.065],
            #     [ 0.000,  0.000, -0.156], [ 0.025,  0.000, -0.015], [ 0.217,  0.000,  0.070],
            #     [ 0.230,  0.000, -0.070], [-0.014,  0.000, -0.156], [ 0.000,  0.000,  0.042],
            #     [-0.057, -0.018, -0.010], [-0.074, -0.000,  0.128], [ 0.206, -0.070, -0.002],
            #     [-0.000, -0.000,  0.156], [-0.017, -0.000, -0.092], [ 0.217, -0.000, -0.027],
            #     [-0.052, -0.000, -0.097], [-0.019, -0.000,  0.128], [-0.035, -0.018, -0.010],
            #     [ 0.217, -0.000, -0.070], [-0.080, -0.000,  0.156], [ 0.230, -0.000,  0.070],
            #     [-0.023, -0.000, -0.075], [-0.029, -0.000, -0.127], [-0.090, -0.000, -0.042],
            #     [ 0.206, -0.055, -0.002], [-0.090, -0.000, -0.015], [ 0.000, -0.000, -0.015],
            #     [-0.037, -0.000, -0.097], [-0.074, -0.000,  0.074], [-0.019, -0.000,  0.074],
            #     [ 0.230, -0.000, -0.113], [-0.100, -0.030,  0.000], [ 0.170, -0.000, -0.015],
            #     [ 0.230, -0.000,  0.113], [-0.000, -0.025, -0.240], [-0.000, -0.025,  0.240],
            #     [ 0.243, -0.104,  0.000], [-0.080, -0.000, -0.156]
            # ], dtype=np.float32)

            # Example new 2D/3D correspondences for the new anchor
            # You must define these for your anchor
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
        # ---------------------------------------------------------

        # Process frame (pose estimation)
        pose_data, visualization = pose_estimator.process_frame(frame, frame_idx)

        # If we got pose data, store it in our final array
        #if pose_data:
            # Also store the image timestamp in the pose_data
        pose_data['timestamp']  = frame_t
        pose_data['image_file'] = img_name
        all_poses.append(pose_data)
        logger.debug(f'Pose data (frame={frame_idx}, time={frame_t:.3f}): {pose_data}')

        # Show the visualization
        if not opt.no_display and visualization is not None:
            cv2.imshow('Pose Estimation', visualization)
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

    #pose_estimator.kf_pose.save_debug_info("kalman_debug_adaptive.json")

    # Save pose estimation results
    with open(opt.save_pose, 'w') as f:
        json.dump(all_poses, f, indent=4)
    logger.info(f'Pose estimation results saved to {opt.save_pose}')
