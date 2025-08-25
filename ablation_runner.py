# 파일 이름: ablation_runner_gt.py
import subprocess
import argparse
from pathlib import Path

# --- CHANGED: 최종 6개 실험 구성으로 변경 ---
RUN_CONFIGS = {
    'A_Baseline': {
        '--matcher': 'lightglue', '--det_every_n': 1, '--kf_timeaware': 1, '--kf_adaptiveR': 'full',
        '--gate_logic': 'or', '--gate_ori_deg': 30.0, '--gate_ratio': 0.75,
        '--rate_limits': '30,1.5', '--vp_mode': 'dynamic'
    },
    'I_LegacyFilter': { # 칼만 필터 핵심 기능 비활성화
        '--kf_timeaware': 0, '--kf_adaptiveR': 'none', '--rate_limits': 'inf,inf'
    },
    'B_NoDetector': { # YOLO 검출기 미사용
        '--no_det': True
    },
    'F_ORB': { # 고전적인 ORB 특징점 사용
        '--matcher': 'orb', '--orb_nfeatures': 1024, '--nn_ratio': 0.75
    },
    'L_FixedViewpoint': { # 단일 뷰포인트 고정
        '--vp_mode': 'fixed', '--fixed_view': 1
    },
    'N_Coarse4Viewpoints': { # 4개 주요 뷰포인트만 사용
        '--vp_mode': 'coarse4'
    }
}

def main():
    parser = argparse.ArgumentParser(description="Ablation study orchestrator with GT evaluation")
    parser.add_argument("--python", type=str, default="python3", help="Python executable.")
    parser.add_argument("--script", type=str, required=True, help="Path to the core logic script (VAPE_MK53_Core_GT.py).")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video.")
    # --- NEW: GT 보정 파일 인자를 필수로 받음 ---
    parser.add_argument("--calibration", type=str, required=True, help="Path to the calibration JSON file.")
    parser.add_argument("--out", type=str, default="./ablation_results_gt", help="Root output directory.")
    parser.add_argument('--show', action='store_true', help="Enable visualization windows for each run.")
    args = parser.parse_args()

    root_out_dir = Path(args.out)
    root_out_dir.mkdir(parents=True, exist_ok=True)
    run_ids = list(RUN_CONFIGS.keys()) # --- CHANGED: 고정된 6개 실험만 실행

    for i, run_id in enumerate(run_ids):
        print("-" * 80); print(f"▶️  RUN {i+1}/{len(run_ids)}: [{run_id}]"); print("-" * 80)
        run_dir = root_out_dir / run_id
        run_dir.mkdir(exist_ok=True)

        # A_Baseline을 기본으로, 각 실험의 설정을 덮어씀
        config = RUN_CONFIGS['A_Baseline'].copy()
        if run_id != 'A_Baseline': config.update(RUN_CONFIGS[run_id])

        cmd = [
            args.python, args.script,
            '--video_file', args.video,
            '--calibration', args.calibration, # --- NEW: GT 보정 파일 경로 전달 ---
            '--output_dir', str(run_dir),
            '--log_jsonl'
        ]
        if args.show: cmd.append('--show')

        for key, value in config.items():
            if isinstance(value, bool):
                if value: cmd.append(key)
            else:
                cmd.extend([key, str(value)])
        
        print(f"Executing command:\n{' '.join(cmd)}\n")
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Run [{run_id}] finished successfully.")
        except (subprocess.CalledProcessError, KeyboardInterrupt) as e:
            print(f"❌ Run [{run_id}] failed or was interrupted. Stopping study.")
            break
            
    print("\n🎉 Ablation runs complete! Now run the analysis script on the results.")

if __name__ == "__main__":
    main()

# # 파일 이름: ablation_runner.py
# import subprocess
# import argparse
# from pathlib import Path

# # --- 실험 설정 (이 부분은 그대로 유지) ---
# RUN_CONFIGS = {
#     'A_Baseline': {
#         '--matcher': 'lightglue', '--det_every_n': 1, '--kf_timeaware': 1, '--kf_adaptiveR': 'full',
#         '--gate_logic': 'or', '--gate_ori_deg': 30.0, '--gate_ratio': 0.75,
#         '--rate_limits': '30,1.5', '--vp_mode': 'dynamic'
#     },
#     'B_NoDetector': {'--no_det': True},
#     'C_DetectorCadence5': {'--det_every_n': 5, '--det_margin_px': 20},
#     'D_SP_NNRT': {'--matcher': 'nnrt', '--nn_ratio': 0.75},
#     'F_ORB': {'--matcher': 'orb', '--orb_nfeatures': 1024, '--nn_ratio': 0.75},
#     'I_LegacyFilter': {'--kf_timeaware': 0, '--kf_adaptiveR': 'none', '--rate_limits': 'inf,inf'},
#     'J_NoGating': {'--gate_ori_deg': -1, '--gate_ratio': 0.0},
#     'K_NoRateLimits': {'--rate_limits': 'inf,inf'},
#     'L_FixedViewpoint': {'--vp_mode': 'fixed', '--fixed_view': 0},
#     'N_Coarse4Viewpoints': {'--vp_mode': 'coarse4'}
# }
# MINIMAL_SET = ['A_Baseline', 'B_NoDetector', 'C_DetectorCadence5', 'F_ORB', 'I_LegacyFilter', 'J_NoGating', 'N_Coarse4Viewpoints']

# def main():
#     parser = argparse.ArgumentParser(description="Ablation study orchestrator with GT evaluation")
#     parser.add_argument("--python", type=str, default="python3", help="Python executable.")
#     parser.add_argument("--script", type=str, required=True, help="Path to the core logic script (VAPE_MK53_Core_GT.py).")
#     parser.add_argument("--video", type=str, required=True, help="Path to the input video.")
#     ### GT 통합: calibration 파일 경로를 인자로 받음 ###
#     parser.add_argument("--calibration", type=str, required=True, help="Path to the calibration JSON file.")
#     parser.add_argument("--out", type=str, default="./ablation_results_gt", help="Root output directory.")
#     parser.add_argument("--runs", type=str, default="all", choices=["minimal", "all"])
#     parser.add_argument('--show', action='store_true', help="Enable visualization windows for each run.")
#     args = parser.parse_args()

#     root_out_dir = Path(args.out)
#     root_out_dir.mkdir(parents=True, exist_ok=True)
#     run_ids = MINIMAL_SET if args.runs == "minimal" else list(RUN_CONFIGS.keys())

#     for i, run_id in enumerate(run_ids):
#         print("-" * 80); print(f"▶️  RUN {i+1}/{len(run_ids)}: [{run_id}]"); print("-" * 80)
#         run_dir = root_out_dir / run_id
#         run_dir.mkdir(exist_ok=True)

#         config = RUN_CONFIGS['A_Baseline'].copy()
#         if run_id != 'A_Baseline': config.update(RUN_CONFIGS[run_id])

#         cmd = [
#             args.python, args.script,
#             '--video_file', args.video,
#             '--calibration', args.calibration, # GT 보정 파일 경로 전달
#             '--output_dir', str(run_dir),
#             '--log_jsonl'
#         ]
#         if args.show: cmd.append('--show')

#         for key, value in config.items():
#             if isinstance(value, bool):
#                 if value: cmd.append(key)
#             else:
#                 cmd.extend([key, str(value)])
        
#         print(f"Executing command:\n{' '.join(cmd)}\n")
#         try:
#             subprocess.run(cmd, check=True)
#             print(f"✅ Run [{run_id}] finished successfully.")
#         except (subprocess.CalledProcessError, KeyboardInterrupt) as e:
#             print(f"❌ Run [{run_id}] failed or was interrupted. Stopping study.")
#             break
            
#     print("\n🎉 Ablation runs complete! Now run the analysis script on the results.")

# if __name__ == "__main__":
#     main()