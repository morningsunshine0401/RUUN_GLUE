# main_MK42.py - 성능 최적화 및 프로파일링 버전

import cv2
import time
import argparse
import logging
import torch
import os
from collections import defaultdict
import statistics

# 성능 프로파일러 클래스
class PerformanceProfiler:
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timers = {}
        self.frame_count = 0
        
    def start_timer(self, name):
        self.current_timers[name] = time.time()
        
    def end_timer(self, name):
        if name in self.current_timers:
            elapsed = time.time() - self.current_timers[name]
            self.timings[name].append(elapsed * 1000)
            del self.current_timers[name]
            return elapsed * 1000
        return 0
    
    def print_stats(self):
        if self.frame_count % 10 == 0 and self.frame_count > 0:
            print(f"\n📊 === FRAME {self.frame_count} PERFORMANCE ===")
            total_avg = 0
            for name, times in self.timings.items():
                if not times:
                    continue
                recent_times = times[-10:]
                avg_ms = statistics.mean(recent_times)
                max_ms = max(recent_times)
                
                if name in ['total_frame']:
                    continue
                    
                if avg_ms > 50:
                    emoji = "🔴"
                elif avg_ms > 20:
                    emoji = "🟡"
                else:
                    emoji = "🟢"
                    
                print(f"{emoji} {name:20} | {avg_ms:6.1f}ms avg | {max_ms:6.1f}ms max")
                
                if name in ['yolo_detection', 'viewpoint_classification', 'superpoint_total', 'pnp_total']:
                    total_avg += avg_ms
            
            if total_avg > 0:
                fps = 1000 / total_avg
                print(f"📈 Estimated FPS: {fps:.1f}")
            print("=" * 50)

# Global profiler
profiler = PerformanceProfiler()

from ultralytics import YOLO
import timm
from torchvision import transforms
import torch.nn.functional as F
import json
import numpy as np
from datetime import datetime
from thread_MK42 import ThreadedPoseEstimator
from utils import create_unique_filename

# GPU 최적화 설정
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"  # 비동기 실행 허용
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# 모델 로드 (한 번만)
print("🔄 Loading models...")
start_time = time.time()

# YOLO 모델 로드
yolo_model = YOLO("yolov8s.pt")
if torch.cuda.is_available():
    yolo_model.to('cuda')
    yolo_model.model.model[-1].export = True  # 더 빠른 추론
    yolo_model.model.model[-1].format = 'engine'  # TensorRT 스타일
print(f"✅ YOLO loaded in {(time.time() - start_time)*1000:.1f}ms")

# 뷰포인트 분류 모델 로드
start_time = time.time()
num_classes = 4
class_names = ['NE', 'NW', 'SE', 'SW']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vp_model = timm.create_model('mobilevit_s', pretrained=False, num_classes=num_classes)
vp_model.load_state_dict(torch.load('mobilevit_viewpoint_twostage_final_2.pth', map_location=device))
vp_model.eval()
vp_model.to(device)

# # 반정밀도 최적화 (GPU 메모리 절약 및 속도 향상)
# if torch.cuda.is_available():
#     vp_model = vp_model.half()  # FP16
    
print(f"✅ Viewpoint model loaded in {(time.time() - start_time)*1000:.1f}ms")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 앵커 데이터
common_anchor_path = 'assets/Ruun_images/viewpoint/anchor/20241226/Anchor2.png'

anchor_keypoints_2D = np.array([
    [511, 293], [591, 284], [587, 330], [413, 249], [602, 348], [715, 384], [598, 298], [656, 171], [805, 213],
    [703, 392], [523, 286], [519, 327], [387, 289], [727, 126], [425, 243], [636, 358], [745, 202], [595, 388],
    [436, 260], [539, 313], [795, 220], [351, 291], [665, 165], [611, 353], [650, 377], [516, 389], [727, 143],
    [496, 378], [575, 312], [617, 368], [430, 312], [480, 281], [834, 225], [469, 339], [705, 223], [637, 156],
    [816, 414], [357, 195], [752, 77], [642, 451]
], dtype=np.float32)

anchor_keypoints_3D = np.array([
    [-0.014,  0.000,  0.042], [ 0.025, -0.014, -0.011], [-0.014,  0.000, -0.042], [-0.014,  0.000,  0.156],
    [-0.023,  0.000, -0.065], [ 0.000,  0.000, -0.156], [ 0.025,  0.000, -0.015], [ 0.217,  0.000,  0.070],
    [ 0.230,  0.000, -0.070], [-0.014,  0.000, -0.156], [ 0.000,  0.000,  0.042], [-0.057, -0.018, -0.010],
    [-0.074, -0.000,  0.128], [ 0.206, -0.070, -0.002], [-0.000, -0.000,  0.156], [-0.017, -0.000, -0.092],
    [ 0.217, -0.000, -0.027], [-0.052, -0.000, -0.097], [-0.019, -0.000,  0.128], [-0.035, -0.018, -0.010],
    [ 0.217, -0.000, -0.070], [-0.080, -0.000,  0.156], [ 0.230, -0.000,  0.070], [-0.023, -0.000, -0.075],
    [-0.029, -0.000, -0.127], [-0.090, -0.000, -0.042], [ 0.206, -0.055, -0.002], [-0.090, -0.000, -0.015],
    [ 0.000, -0.000, -0.015], [-0.037, -0.000, -0.097], [-0.074, -0.000,  0.074], [-0.019, -0.000,  0.074],
    [ 0.230, -0.000, -0.113], [-0.100, -0.030,  0.000], [ 0.170, -0.000, -0.015], [ 0.230, -0.000,  0.113],
    [-0.000, -0.025, -0.240], [-0.000, -0.025,  0.240], [ 0.243, -0.104,  0.000], [-0.080, -0.000, -0.156]
], dtype=np.float32)

viewpoint_to_anchor_path = {
    'NE': common_anchor_path, 'NW': common_anchor_path,
    'SE': common_anchor_path, 'SW': common_anchor_path
}

anchor_kp2d = {
    'NE': anchor_keypoints_2D, 'NW': anchor_keypoints_2D,
    'SE': anchor_keypoints_2D, 'SW': anchor_keypoints_2D
}

anchor_kp3d = {
    'NE': anchor_keypoints_3D, 'NW': anchor_keypoints_3D,
    'SE': anchor_keypoints_3D, 'SW': anchor_keypoints_3D
}

# 로깅 설정
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebcamPoseEstimator:
    def __init__(self, args):
        self.args = args
        self.profiler = PerformanceProfiler()
        
        # 디바이스 설정
        if args.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = args.device
        logger.info(f'Using device: {self.device}')
        
        # 포즈 추정기 초기화
        self.pose_estimator = ThreadedPoseEstimator(args, self.device)
        
        # 카메라 초기화
        if args.input is not None:
            logger.info(f"Using input video file: {args.input}")
            self.cap = cv2.VideoCapture(args.input)
        else:
            logger.info(f"Using webcam with camera ID: {args.camera_id}")
            self.cap = cv2.VideoCapture(args.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
            # 버퍼 크기 줄이기 (지연 시간 감소)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # FPS 제한
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open capture source")
        
        # 실제 해상도 확인
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Capture resolution: {actual_width}x{actual_height}")
        
        # 프레임 관련 변수들
        self.frame_count = 0
        self.processing_fps = 0
        self.last_fps_update = time.time()
        self.start_time = time.time()
        self.fps_history = []
        
        # 창 생성
        cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
        
        # 결과 처리
        self.waiting_for_first_result = True
        self.pending_results = []
        self.all_poses = []

        # 캐시 변수들 (성능 최적화)
        self.detection_cache = {
            'bbox': None,
            'viewpoint': None,
            'age': 0,
            'max_age': 15,  # 더 긴 캐시 (더 빠르게)
            'last_frame_processed': -1
        }
        
        self.last_match_count = 0
        self.low_match_frames = 0
        self.force_viewpoint_update = False
        
        # 프레임 스킵 최적화
        self.actual_skip_frames = max(1, args.skip_frames if hasattr(args, 'skip_frames') and args.skip_frames else 1)
        
        print(f"🚀 Initialization complete. Skip frames: {self.actual_skip_frames}, Cache age: {self.detection_cache['max_age']}")

    def run(self):
        """최적화된 메인 루프"""
        logger.info("Starting optimized pose estimation")
        
        last_anchor_path = common_anchor_path
        frame_process_times = []
        
        try:
            while True:
                loop_start = time.time()
                
                # === 프레임 캡처 (가장 빠르게) ===
                profiler.start_timer('frame_capture')
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                profiler.end_timer('frame_capture')
                
                self.frame_count += 1
                
                # === 적극적인 프레임 스킵 ===
                if self.frame_count % self.actual_skip_frames != 0:
                    continue
                
                profiler.start_timer('total_processing')
                
                # === 프레임 전처리 (필요시에만) ===
                profiler.start_timer('frame_preprocessing')
                if hasattr(self.args, 'resize') and len(self.args.resize) > 0:
                    if len(self.args.resize) == 2:
                        # 더 빠른 리사이징 (INTER_LINEAR 대신 INTER_NEAREST 사용 고려)
                        frame = cv2.resize(frame, tuple(self.args.resize), interpolation=cv2.INTER_LINEAR)
                    elif len(self.args.resize) == 1 and self.args.resize[0] > 0:
                        h, w = frame.shape[:2]
                        scale = self.args.resize[0] / max(h, w)
                        new_size = (int(w * scale), int(h * scale))
                        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
                profiler.end_timer('frame_preprocessing')

                # === 검출 (캐시 사용) ===
                profiler.start_timer('detection_total')
                bbox, viewpoint_label = self.get_detection_with_cache_optimized(frame)
                profiler.end_timer('detection_total')
                
                if bbox is None:
                    continue
                    
                x1, y1, x2, y2 = bbox
                
                # === 앵커 업데이트 (필요시에만) ===
                anchor_path = viewpoint_to_anchor_path[viewpoint_label]
                if anchor_path != last_anchor_path:
                    profiler.start_timer('anchor_update')
                    self.pose_estimator.reinitialize_anchor(
                        anchor_path, anchor_kp2d[viewpoint_label], anchor_kp3d[viewpoint_label]
                    )
                    last_anchor_path = anchor_path
                    profiler.end_timer('anchor_update')

                # === 포즈 추정기로 전달 ===
                profiler.start_timer('pose_queue')
                self.pose_estimator.process_frame(
                    frame, self.frame_count, time.time(),
                    f"frame_{self.frame_count:06d}",
                    bbox=(x1, y1, x2, y2), viewpoint=viewpoint_label
                )
                profiler.end_timer('pose_queue')
                            
                # === 결과 처리 ===
                profiler.start_timer('result_processing')
                self.process_results_optimized()
                profiler.end_timer('result_processing')
                
                total_processing_time = profiler.end_timer('total_processing')
                
                # === FPS 계산 ===
                if self.frame_count % 10 == 0:
                    elapsed = time.time() - self.last_fps_update
                    if elapsed > 0:
                        current_fps = 10 / elapsed
                        self.fps_history.append(current_fps)
                        
                        if len(self.fps_history) > 5:
                            self.fps_history.pop(0)
                        
                        self.processing_fps = sum(self.fps_history) / len(self.fps_history)
                        self.last_fps_update = time.time()

                # === 성능 통계 출력 ===
                self.profiler.frame_count = self.frame_count
                self.profiler.print_stats()
                
                # === 느린 프레임 경고 ===
                loop_total_time = (time.time() - loop_start) * 1000
                frame_process_times.append(loop_total_time)
                
                if loop_total_time > 100:  # 100ms 이상
                    print(f"⚠️  SLOW LOOP {self.frame_count}: {loop_total_time:.1f}ms (processing: {total_processing_time:.1f}ms)")
                
                # === 키 입력 체크 (non-blocking) ===
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    logger.info("User requested exit")
                    break
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # === 최종 성능 리포트 ===
            if frame_process_times:
                avg_time = statistics.mean(frame_process_times)
                max_time = max(frame_process_times)
                min_time = min(frame_process_times)
                
                print(f"\n🏁 FINAL PERFORMANCE REPORT")
                print(f"📊 Total frames processed: {len(frame_process_times)}")
                print(f"⏱️  Average loop time: {avg_time:.1f}ms")
                print(f"⚡ Fastest loop: {min_time:.1f}ms")
                print(f"🐌 Slowest loop: {max_time:.1f}ms")
                print(f"📈 Theoretical max FPS: {1000/avg_time:.1f}")
                
                # 병목 구간 분석
                print(f"\n🔍 BOTTLENECK ANALYSIS:")
                for name, times in profiler.timings.items():
                    if times and len(times) > 5:
                        avg_ms = statistics.mean(times)
                        if avg_ms > 20:
                            print(f"🔴 {name}: {avg_ms:.1f}ms average (SLOW)")
                        elif avg_ms > 10:
                            print(f"🟡 {name}: {avg_ms:.1f}ms average")
            
            self.cleanup()

    def get_detection_with_cache_optimized(self, frame):
        """최적화된 검출 캐시"""
        original_h, original_w = frame.shape[:2]
        
        need_new_detection = (
            self.detection_cache['bbox'] is None or 
            self.detection_cache['age'] >= self.detection_cache['max_age'] or
            self.force_viewpoint_update
        )
        
        if need_new_detection:
            # === YOLO 검출 ===
            profiler.start_timer('yolo_detection')
            
            # 더 작은 YOLO 해상도 (속도 우선)
            yolo_w, yolo_h = 640,640#320, 180  # 416,234에서 감소
            yolo_frame = cv2.resize(frame, (yolo_w, yolo_h), interpolation=cv2.INTER_NEAREST)
            
            # GPU에서 직접 처리
            results = yolo_model(yolo_frame[..., ::-1], verbose=False)  # verbose=False로 로그 줄이기
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            yolo_time = profiler.end_timer('yolo_detection')
            
            if len(boxes) > 0:
                # 바운딩박스 스케일링
                profiler.start_timer('bbox_scaling')
                scale_x = original_w / yolo_w
                scale_y = original_h / yolo_h
                x1, y1, x2, y2 = boxes[0]
                bbox = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))
                profiler.end_timer('bbox_scaling')
                
                # === 뷰포인트 분류 ===
                profiler.start_timer('viewpoint_classification')
                viewpoint_label = self.classify_viewpoint_optimized(frame, bbox)
                vp_time = profiler.end_timer('viewpoint_classification')
                
                # 캐시 업데이트
                self.detection_cache.update({
                    'bbox': bbox,
                    'viewpoint': viewpoint_label,
                    'age': 0,
                    'last_frame_processed': self.frame_count
                })
                
                self.force_viewpoint_update = False
                
                if self.frame_count % 20 == 0:  # 로그 빈도 줄이기
                    print(f"🔄 New detection: YOLO {yolo_time:.1f}ms, VP {vp_time:.1f}ms -> {viewpoint_label}")
                
            else:
                return None, None
                
        else:
            # 캐시된 검출 사용
            bbox = self.detection_cache['bbox']
            viewpoint_label = self.detection_cache['viewpoint']
            self.detection_cache['age'] += 1
        
        return bbox, viewpoint_label

    def classify_viewpoint_optimized(self, frame, bbox):
        """최적화된 뷰포인트 분류"""
        x1, y1, x2, y2 = bbox
        
        # 크롭
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return 'NE'
        
        # 더 작은 해상도로 리사이징 (96x96으로 더 축소)
        crop_resized = cv2.resize(cropped, (96, 96), interpolation=cv2.INTER_LINEAR)
        
        # PIL 변환 (최적화)
        from PIL import Image
        img_pil = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
        
        # 반정밀도 텐서 생성
        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        # if torch.cuda.is_available():
        #     input_tensor = input_tensor.half()  # FP16

        # 모델 추론
        with torch.no_grad():
            vp_logits = vp_model(input_tensor)
            pred = torch.argmax(vp_logits, dim=1).item()
            viewpoint_label = class_names[pred]
            
        return viewpoint_label

    def process_results_optimized(self):
        """최적화된 결과 처리"""
        # 논블로킹으로 결과 수집
        new_results = []
        while True:
            try:
                result = self.pose_estimator.get_result(timeout=0.001)  # 매우 짧은 타임아웃
                if result[0] is None:
                    break
                new_results.append(result)
            except:
                break
        
        # 가장 최신 결과만 처리 (중간 결과들은 버려서 지연 시간 감소)
        if new_results:
            pose_data, visualization, frame_idx, frame_t, img_name = new_results[-1]
            self.display_result_optimized(pose_data, visualization, frame_idx, frame_t, img_name)
            self.waiting_for_first_result = False
        elif self.waiting_for_first_result:
            # 초기화 메시지 (간단하게)
            h, w = 480, 640  # 더 작은 창
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(blank, "Initializing...", (w//4, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Pose Estimation', blank)

    def display_result_optimized(self, pose_data, visualization, frame_idx, frame_t, img_name):
        """최적화된 결과 표시"""
        if visualization is None:
            return
        
        display = visualization.copy()
        
        # 필수 정보만 표시 (텍스트 렌더링 최소화)
        cache_age = self.detection_cache['age']
        viewpoint = self.detection_cache.get('viewpoint', 'Unknown')
        
        # 텍스트 정보 (최소한으로)
        y_pos = display.shape[0] - 120
        cv2.putText(display, f"FPS: {self.processing_fps:.1f}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(display, f"Cache: {cache_age}/{self.detection_cache['max_age']} | VP: {viewpoint}", 
                    (10, y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if pose_data:
            num_matches = pose_data.get('total_matches', 0)
            self.update_match_tracking(num_matches)
            
            match_color = (0, 0, 255) if num_matches < 5 else (255, 255, 255)
            cv2.putText(display, f"Matches: {num_matches}", (10, y_pos + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, match_color, 1)
        
        # 화면 표시
        cv2.imshow('Pose Estimation', display)
        cv2.waitKey(1)

    def update_match_tracking(self, num_matches):
        """매치 추적 업데이트"""
        self.last_match_count = num_matches
        
        if num_matches < 5:
            self.low_match_frames += 1
            if self.low_match_frames >= 3:
                self.force_viewpoint_update = True
                self.low_match_frames = 0
                print(f"🔄 Forcing viewpoint update due to low matches: {num_matches}")
        else:
            self.low_match_frames = 0

    def cleanup(self):
        """리소스 정리"""
        logger.info("Cleaning up resources")
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'pose_estimator'):
            self.pose_estimator.cleanup()
        cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description='Optimized Webcam Pose Estimation')
    
    # 비디오 소스 설정
    parser.add_argument('--input', type=str, default=None, help='Input video file path')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera ID')
    parser.add_argument('--camera_width', type=int, default=1280, help='Camera width')
    parser.add_argument('--camera_height', type=int, default=720, help='Camera height')
    
    # 모델 설정
    parser.add_argument('--anchor', type=str, required=True, help='Anchor image path')
    parser.add_argument('--resize', type=int, nargs='+', default=[640, 480], 
                        help='Resize resolution (default: 640 480 for speed)')
    
    # 디바이스 설정
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    
    # 성능 설정
    parser.add_argument('--skip_frames', type=int, default=2, 
                        help='Process every N-th frame (default: 2 for speed)')
    
    # Kalman filter 설정
    parser.add_argument('--KF_mode', type=str, default='L', choices=['L', 'T', 'auto'],
                        help='Kalman filter mode (L=loosely-coupled for speed)')
    
    return parser.parse_args()

if __name__ == "__main__":
    print("🚀 Starting Optimized Pose Estimation")
    
    args = parse_args()
    
    try:
        estimator = WebcamPoseEstimator(args)
        estimator.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())



