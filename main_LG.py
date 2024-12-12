import argparse
import torch
torch.set_grad_enabled(False)

import time
from pathlib import Path
import cv2
from pose_estimator_LG import PoseEstimator
from utils import create_unique_filename
from models.utils import AverageTimer
import json
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Use DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main_LG.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LightGlue Pose Estimation (ONNX)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to an image directory or movie file')
    parser.add_argument(
        '--anchor', type=str, required=True,
        help='Path to the anchor (reference) image')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference.')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--save_pose', type=str, default='pose_estimation_research.json',
        help='Path to save pose estimation results in JSON format')

    opt = parser.parse_args()
    logger.info(f"Parsed options: {opt}")

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        logger.info(f'Will resize to {opt.resize[0]}x{opt.resize[1]} (WxH)')
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        logger.info(f'Will resize max dimension to {opt.resize[0]}')
    elif len(opt.resize) == 1:
        logger.info('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    if os.path.isdir(opt.save_pose):
        base_filename = 'pose_estimation.json'
        opt.save_pose = create_unique_filename(opt.save_pose, base_filename)
    else:
        save_dir = os.path.dirname(opt.save_pose)
        base_filename = os.path.basename(opt.save_pose)
        opt.save_pose = create_unique_filename(save_dir, base_filename)

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    logger.info(f'Running inference on device "{device}"')

    pose_estimator = PoseEstimator(opt, device)
    all_poses = []

    cap = cv2.VideoCapture(opt.input)
    if not cap.isOpened():
        logger.error('Error when opening video file or camera (try different --input?)')
        exit(1)

    if opt.output_dir is not None:
        logger.info(f'Will write outputs to {opt.output_dir}')
        Path(opt.output_dir).mkdir(exist_ok=True)

    if opt.no_display:
        logger.info('Skipping visualization, will not show a GUI.')
    else:
        cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pose Estimation', 640 * 2, 480)

    timer = AverageTimer()
    overall_start_time = time.time()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.info('Finished processing video or invalid frame.')
            break

        frame_idx += 1
        start_time = time.time()  # Start timing

        logger.debug(f"Processing frame {frame_idx} with shape {frame.shape}")

        # Log time for data preparation
        timer.update('data')
        data_prep_time = time.time()

        pose_data, visualization = pose_estimator.process_frame(frame, frame_idx)
        pose_time = time.time()

        if pose_data:
            all_poses.append(pose_data)
            logger.debug(f'Pose data for frame {frame_idx}: {pose_data}')

        if not opt.no_display and visualization is not None:
            cv2.imshow('Pose Estimation', visualization)
            if cv2.waitKey(1) == ord('q'):
                logger.info('Exiting on user request (q key pressed).')
                break

        if opt.output_dir is not None and visualization is not None:
            out_file = str(Path(opt.output_dir, f'frame_{frame_idx:06d}.png'))
            cv2.imwrite(out_file, visualization)
            logger.debug(f'Saved visualization to {out_file}')

        viz_time = time.time()

        # Log elapsed times
        logger.info(
            f"Frame {frame_idx} timings: Data Prep: {data_prep_time - start_time:.3f}s, "
            f"Pose Estimation: {pose_time - data_prep_time:.3f}s, "
            f"Visualization: {viz_time - pose_time:.3f}s"
        )

        timer.update('viz')
    overall_end_time = time.time()

    # Calculate and log total FPS
    total_elapsed_time = overall_end_time - overall_start_time
    if frame_idx > 0 and total_elapsed_time > 0:
        total_fps = frame_idx / total_elapsed_time
        logger.info(f"Processed {frame_idx} frames in {total_elapsed_time:.2f}s (Total FPS: {total_fps:.2f})")
    else:
        logger.info("No frames were processed or invalid total time.")

    cap.release()
    cv2.destroyAllWindows()

    # Save pose estimation results
    with open(opt.save_pose, 'w') as f:
        json.dump(all_poses, f, indent=4)
    logger.info(f'Pose estimation results saved to {opt.save_pose}')
