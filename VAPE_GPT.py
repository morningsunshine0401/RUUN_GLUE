import argparse
import cv2
import torch
from ultralytics import YOLO
from GPT_PE import PoseEstimator  # 방금 올려주신 클래스 정의 파일

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--anchor', type=str, required=True,
                   help='Path to anchor image')
    p.add_argument('--resize', nargs=2, type=int, default=[1280,720],
                   help='Frame resize resolution')
    p.add_argument('--kf_mode', choices=['auto','L','T'], default='L',
                   help='Kalman filter mode')
    return p.parse_args()

def main():
    opt = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) PoseEstimator 인스턴스
    estimator = PoseEstimator(opt, device, kf_mode=opt.kf_mode)

    # 2) YOLO 모델 (검출만 담당)
    yolo = YOLO('yolov8n.pt').to(device)

    cap = cv2.VideoCapture(0)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # (a) 리사이즈
        frame_resized = cv2.resize(frame, tuple(opt.resize))

        # (b) YOLO 검출 → 첫 bbox만 사용
        small = cv2.resize(frame_resized, (640,640))
        results = yolo(small[...,::-1], verbose=False)
        if len(results[0].boxes) == 0:
            cv2.imshow('Pose', frame_resized)
        else:
            x1,y1,x2,y2 = results[0].boxes.xyxy[0].cpu().numpy()
            # 스케일 업 (중심점 기준)
            sx, sy = opt.resize[0]/640, opt.resize[1]/640
            cx, cy = (x1+x2)/2*sx, (y1+y2)/2*sy
            bw, bh = (x2-x1)/2*sx, (y2-y1)/2*sy
            x1n, y1n = int(cx-bw), int(cy-bh)
            x2n, y2n = int(cx+bw), int(cy+bh)
            x1n, y1n = max(0,x1n), max(0,y1n)
            x2n, y2n = min(opt.resize[0],x2n), min(opt.resize[1],y2n)

            # (c) PoseEstimator 처리
            bbox = (x1n, y1n, x2n, y2n)
            pose_data, vis = estimator.process_frame(frame_resized, frame_idx, bbox=bbox)

            # (d) 결과 표시
            cv2.imshow('Pose', vis)

        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
