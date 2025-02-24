from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    
    opt = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running inference on device "{device}"')
    
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    vs = VideoStreamer(opt.input, opt.resize, 1, ['*.png', '*.jpg'], 1000000)
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = 0  # Automatically set first frame as anchor
    print("First frame set as anchor automatically.")
    
    while True:
        frame, ret = vs.next_frame()
        if not ret:
            print('Finished processing.')
            break
        
        frame_tensor = frame2tensor(frame, device)
        pred = matching({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])
        
        text = ['SuperGlue', f'Keypoints: {len(kpts0)}:{len(kpts1)}', f'Matches: {len(mkpts0)}']
        
        out = make_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text, path=None, show_keypoints=True)
        
        cv2.imshow('SuperGlue matches', out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    vs.cleanup()
