# main.py
#This code is used to estimate pose of an obkect with a predefined anchor image. This code was used for real life videoes

import argparse
import torch
torch.set_grad_enabled(False)

from pathlib import Path
import cv2
from pose_estimator import PoseEstimator
from utils import create_unique_filename
from models.utils import AverageTimer
import json
import os

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue Pose Estimation',
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
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by SuperPoint '
             '(\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius '
             '(Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

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
    print(opt)

    # Adjust resize options
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    # Check if the provided path is a directory, if so, handle the filename
    if os.path.isdir(opt.save_pose):
        base_filename = 'pose_estimation.json'
        opt.save_pose = create_unique_filename(opt.save_pose, base_filename)
    else:
        save_dir = os.path.dirname(opt.save_pose)
        base_filename = os.path.basename(opt.save_pose)
        opt.save_pose = create_unique_filename(save_dir, base_filename)

    # Set device
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    # Initialize PoseEstimator
    pose_estimator = PoseEstimator(opt, device)
    all_poses = []

    # Open video file or camera
    cap = cv2.VideoCapture(opt.input)

    # Check if camera opened successfully
    if not cap.isOpened():
        print('Error when opening video file or camera (try different --input?)')
        exit(1)

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pose Estimation', 640 * 2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    timer = AverageTimer()
    frame_idx = 0

    # Start the timer
    overall_start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Finished processing video or invalid frame.')
            break
        frame_idx += 1
        print('Frame shape:', frame.shape)
        timer.update('data')  # Keep using timer for step-level timing

        pose_data, visualization = pose_estimator.process_frame(frame, frame_idx)

        if pose_data:
            all_poses.append(pose_data)

        if not opt.no_display:
            cv2.imshow('Pose Estimation', visualization)
            if cv2.waitKey(1) == ord('q'):
                break

        # Save the output frame if needed
        if opt.output_dir is not None and visualization is not None:
            out_file = str(Path(opt.output_dir, f'frame_{frame_idx:06d}.png'))
            cv2.imwrite(out_file, visualization)

        timer.update('viz')  # Step-level timing
        timer.print()  # Optional: Print step-level timing

    # End the timer
    overall_end_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

    # Calculate and print average FPS
    total_time = overall_end_time - overall_start_time
    if frame_idx > 0 and total_time > 0:
        fps = frame_idx / total_time
        print(f'Processed {frame_idx} frames in {total_time:.2f} seconds (Average FPS: {fps:.2f})')
    else:
        print('No frames were processed or invalid total time.')

    # Save pose estimation results
    with open(opt.save_pose, 'w') as f:
        json.dump(all_poses, f, indent=4)
    print(f'Pose estimation saved to {opt.save_pose}')