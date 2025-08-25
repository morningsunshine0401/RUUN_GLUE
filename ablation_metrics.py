# 파일 이름: analyze_ablation_gt.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

def analyze_run(run_dir: Path) -> dict:
    """한 실험 폴더의 `frames.jsonl` 파일을 분석하여 4가지 지표를 계산합니다."""
    jsonl_path = run_dir / "frames.jsonl"
    if not jsonl_path.exists():
        return None

    # Load data
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        if not lines: return None
        data = [json.loads(line) for line in lines]
    
    df = pd.DataFrame(data)
    if df.empty: return None

    results = {"Experiment": run_dir.name}

    # --- 1. Accuracy @ (5cm, 5°) ---
    gt_df = df[df['gt_available'] == True].copy()
    if not gt_df.empty:
        accurate_frames = gt_df[
            (gt_df['filt_pos_err_cm'] <= 10) & (gt_df['filt_rot_err_deg'] <= 10)
        ]
        accuracy = (len(accurate_frames) / len(gt_df)) * 100
        results['Accuracy @ (5cm, 5°)'] = f"{accuracy:.2f}%"
    else:
        results['Accuracy @ (5cm, 5°)'] = "N/A"

    # --- 2. Robustness: Tracking Success Rate (%) ---
    attempted_frames = df[df['num_matches'] > 0]
    if not attempted_frames.empty:
        successful_frames = attempted_frames[attempted_frames['accepted'] == True]
        success_rate = (len(successful_frames) / len(attempted_frames)) * 100
        results['Tracking Success Rate'] = f"{success_rate:.2f}%"
    else:
        results['Tracking Success Rate'] = "0.00%"

    # --- 3. Stability: Jitter ---
    jitter_df = df.dropna(subset=['jitter_lin_vel_mps', 'jitter_ang_vel_dps'])
    if not jitter_df.empty:
        pos_jitter = jitter_df['jitter_lin_vel_mps'].std()
        rot_jitter = jitter_df['jitter_ang_vel_dps'].std()
        results['Jitter (Pos, Rot)'] = f"{pos_jitter:.3f} m/s, {rot_jitter:.2f} deg/s"
    else:
        results['Jitter (Pos, Rot)'] = "N/A"

    # --- 4. Speed: Latency (ms) ---
    if 'vision_latency_ms' in df.columns:
        avg_latency = df['vision_latency_ms'].mean()
        results['Latency (ms)'] = f"{avg_latency:.2f}"
    else:
        results['Latency (ms)'] = "N/A"
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze Ablation Study Results with Ground Truth")
    parser.add_argument("results_dir", type=str, help="Path to the root directory of ablation results (e.g., ./ablation_results_gt).")
    args = parser.parse_args()
    #parser.add_argument("run_out4", type=str, help="Path to the root directory of ablation results (e.g., ./ablation_results_gt).")
    #args = parser.parse_args()

    root_dir = Path(args.results_dir)
    if not root_dir.is_dir():
        print(f"Error: Directory not found at {root_dir}")
        return

    all_results = []
    # 실험 순서를 정의
    run_order = ['A_Baseline', 'I_LegacyFilter', 'B_NoDetector', 'F_ORB', 'L_FixedViewpoint', 'N_Coarse4Viewpoints']
    
    for run_name in run_order:
        run_dir = root_dir / run_name
        if run_dir.is_dir():
            print(f"Analyzing {run_name}...")
            run_metrics = analyze_run(run_dir)
            if run_metrics:
                all_results.append(run_metrics)
    
    if not all_results:
        print("No results found to analyze.")
        return

    summary_df = pd.DataFrame(all_results)
    print("\n\n--- ABLATION STUDY SUMMARY ---")
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv(root_dir / "ablation_summary.csv", index=False)
    print(f"\nSummary saved to {root_dir / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()