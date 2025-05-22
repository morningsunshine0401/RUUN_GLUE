from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast)

# Override the frame2tensor function to fix channel ordering.
# This version converts an image to grayscale so that the final tensor shape is [1,1,H,W].
def frame2tensor(image, device):
    # If the image has 3 channels, convert it to grayscale.
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert image to float32 and normalize to [0, 1]
    image = image.astype('float32') / 255.0
    # After conversion, image shape is (H, W). Add a channel dimension to get (H, W, 1)
    image = image[..., None]
    # Transpose from (H, W, 1) to (1, 1, H, W): move the channel to the front and add a batch dimension.
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device)

torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo with optional fixed anchor image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, or path to an image directory or movie file')
    parser.add_argument(
        '--anchor', type=str, default=None,
        help='Path to an anchor image file. If provided, the anchor image will be used for matching.')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')
    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, resize to the exact dimensions, if one number, resize the max dimension, if -1, do not resize')
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.25,
        help='SuperGlue match threshold')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    # Process resize arguments for printing.
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

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device "{}"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    # Initialize VideoStreamer (for incoming webcam frames)
    vs = VideoStreamer(opt.input, opt.resize, opt.skip, opt.image_glob, opt.max_length)
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = 0

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', 640*2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the new anchor (only if no --anchor provided)\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()

    # Set up the anchor image (fixed if provided, or use the first webcam frame)
    if opt.anchor is not None:
        anchor = cv2.imread(opt.anchor)
        if anchor is None:
            raise ValueError("Error reading the anchor image from '{}'".format(opt.anchor))
        # Optionally resize the anchor image to match the desired dimensions
        if len(opt.resize) == 2:
            anchor = cv2.resize(anchor, (opt.resize[0], opt.resize[1]))
        elif len(opt.resize) == 1 and opt.resize[0] > 0:
            h, w = anchor.shape[:2]
            scale = opt.resize[0] / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            anchor = cv2.resize(anchor, (new_w, new_h))
        anchor_tensor = frame2tensor(anchor, device)
        anchor_data = matching.superpoint({'image': anchor_tensor})
        anchor_data = {k+'0': anchor_data[k] for k in keys}
        anchor_data['image0'] = anchor_tensor
        anchor_frame = anchor
        print("Using fixed anchor image from '{}'".format(opt.anchor))
    else:
        # (Already used the first frame as anchor above)
        anchor_data = last_data
        anchor_frame = last_frame

    # Main loop: match the fixed anchor image against incoming webcam frames.
    while True:
        frame, ret = vs.next_frame()
        if not ret:
            print('No new frames available. Restarting from the first frame...')
            vs = VideoStreamer(opt.input, opt.resize, opt.image_glob, opt.skip, opt.max_length)
            frame, ret = vs.next_frame()
            if not ret:
                print('Still no frames available. Exiting.')
                break

        timer.update('data')
        stem0 = "anchor"
        stem1 = vs.i - 1

        frame_tensor = frame2tensor(frame, device)
        pred = matching({**anchor_data, 'image1': frame_tensor})
        # Get 2D keypoints on the anchor image.
        kpts0 = anchor_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        timer.update('forward')

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {}:{:06}'.format(stem0, stem1),
        ]
        # Convert images to grayscale for visualization,
        # so that their shapes are (H, W) and compatible with the plotting function.
        vis_anchor = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY) if len(anchor_frame.shape) == 3 else anchor_frame
        vis_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        out = make_matching_plot_fast(
            vis_anchor, vis_frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text)

        if not opt.no_display:
            cv2.imshow('SuperGlue matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':
                # Only allow updating the anchor if no fixed anchor was provided.
                if opt.anchor is None:
                    anchor_data = {k+'0': pred[k+'1'] for k in keys}
                    anchor_data['image0'] = frame_tensor
                    anchor_frame = frame
                    print("Anchor updated to current frame.")
                else:
                    print("Fixed anchor image in use. Cannot update anchor with 'n'.")
            elif key in ['e', 'r']:
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_threshold']))
            elif key in ['d', 'f']:
                d = 0.05 * (-1 if key == 'd' else 1)
                matching.superglue.config['match_threshold'] = min(max(
                    0.05, matching.superglue.config['match_threshold']+d), .95)
                print('\nChanged the match threshold to {:.2f}'.format(
                    matching.superglue.config['match_threshold']))
            elif key == 'k':
                opt.show_keypoints = not opt.show_keypoints

        timer.update('viz')
        timer.print()

        if opt.output_dir is not None:
            # Save the matching visualization.
            stem = 'matches_{}_{}'.format(stem0, str(stem1).zfill(6))
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)
            
            # Save the 2D coordinates of the anchor keypoints to a text file.
            kp_file = str(Path(opt.output_dir, f'anchor_keypoints_{stem1:06d}.txt'))
            np.savetxt(kp_file, kpts0, fmt='%.2f')
            print('Writing anchor keypoints to {}'.format(kp_file))

    cv2.destroyAllWindows()
    vs.cleanup()
    
