# 파일 이름: ablation_metrics.py (RecursionError Fix)
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

def analyze_run(run_dir: Path) -> dict:
    jsonl_path = run_dir / "frames.jsonl"
    if not jsonl_path.exists(): return None

    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        if not lines: return None
        data = [json.loads(line) for line in lines]
    
    df = pd.DataFrame(data)
    if df.empty: return None

    results = {"Experiment": run_dir.name}
    
    meta_path = run_dir / "run_metadata.json"
    total_video_frames = 0
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            total_video_frames = metadata.get("total_video_frames", 0)

    total_processed_frames = len(df)
    denominator = total_video_frames if total_video_frames > 0 else total_processed_frames
    if total_video_frames == 0:
        print(f"  [Warning] 'run_metadata.json' not found in {run_dir.name}. Uptime metric will be based on PROCESSED frames ({total_processed_frames}) only.")

    # 1. Accuracy (cm, deg)
    gt_df = df[df['gt_available'] == True].copy()
    if not gt_df.empty:
        # --- FIX 1: Ensure columns are numeric before comparison ---
        gt_df['filt_pos_err_cm'] = pd.to_numeric(gt_df['filt_pos_err_cm'], errors='coerce')
        gt_df['filt_rot_err_deg'] = pd.to_numeric(gt_df['filt_rot_err_deg'], errors='coerce')
        accurate_frames = gt_df[(gt_df['filt_pos_err_cm'] <= 5) & (gt_df['filt_rot_err_deg'] <= 5)]
        accuracy = (len(accurate_frames) / len(gt_df)) * 100
        results['Accuracy (cm, deg)'] = f"{accuracy:.2f}%"
    else:
        results['Accuracy (cm, deg)'] = "N/A"

    # 2. Tracking Uptime (%)
    if denominator > 0:
        successful_frames = df[df['accepted'] == True]
        uptime_rate = (len(successful_frames) / denominator) * 100
        results['Tracking Uptime (%)'] = f"{uptime_rate:.2f}%"
    else:
        results['Tracking Uptime (%)'] = "0.00%"

    # 3. Stability (m/s, deg/s)
    jitter_df = df.dropna(subset=['jitter_lin_vel_mps', 'jitter_ang_vel_dps'])
    if not jitter_df.empty and len(jitter_df) > 1:
        pos_jitter = jitter_df['jitter_lin_vel_mps'].std()
        rot_jitter = jitter_df['jitter_ang_vel_dps'].std()
        results['Stability (m/s, deg/s)'] = f"{pos_jitter:.3f}, {rot_jitter:.2f}"
    else:
        results['Stability (m/s, deg/s)'] = "N/A"

    # 4. Latency (ms)
    if 'vision_latency_ms' in df.columns:
        avg_latency = df['vision_latency_ms'].mean()
        results['Latency (ms)'] = f"{avg_latency:.2f}"
    else:
        results['Latency (ms)'] = "N/A"
        
    # --- FIX 2: Handle empty DataFrame for efficiency calculation ---
    # 5. Match Efficiency (%)
    accepted_df = df[df['accepted'] == True].copy()
    if not accepted_df.empty and accepted_df['num_matches'].sum() > 0:
        # Convert to numeric, coercing errors to NaN which are ignored by mean()
        num_inliers = pd.to_numeric(accepted_df['num_inliers'], errors='coerce')
        num_matches = pd.to_numeric(accepted_df['num_matches'], errors='coerce')
        
        # Calculate efficiency safely
        efficiency = np.divide(num_inliers, num_matches, out=np.zeros_like(num_inliers, dtype=float), where=num_matches!=0)
        avg_efficiency = np.nanmean(efficiency) * 100
        results['Match Efficiency (%)'] = f"{avg_efficiency:.2f}%"
    else:
        results['Match Efficiency (%)'] = "N/A"

    # 6. Recovery Time (frames)
    if 'recovery_frames' in df.columns:
        recovery_df = df[df['recovery_frames'] > 0]
        if not recovery_df.empty:
            avg_recovery = recovery_df['recovery_frames'].mean()
            results['Recovery Time (frames)'] = f"{avg_recovery:.2f}"
        else:
            results['Recovery Time (frames)'] = "0.00"
    else:
        results['Recovery Time (frames)'] = "N/A"

    # 7. Performance Breakdown (ms)
    det_ms = df['det_ms'].mean()
    feat_match_ms = (df['feature_ms'] + df['match_ms']).mean()
    pnp_ms = df['pnp_ms'].mean()
    kf_ms = df['kf_ms'].mean()
    results['Perf. Breakdown (ms)'] = f"D:{det_ms:.1f} F+M:{feat_match_ms:.1f} PnP:{pnp_ms:.1f} KF:{kf_ms:.1f}"
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze Ablation Study Results with Advanced Metrics")
    parser.add_argument("results_dir", type=str, help="Path to the root directory of ablation results.")
    args = parser.parse_args()

    root_dir = Path(args.results_dir)
    if not root_dir.is_dir():
        print(f"Error: Directory not found at {root_dir}")
        return

    all_results = []
    run_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    
    for run_dir in run_dirs:
        print(f"Analyzing {run_dir.name}...")
        run_metrics = analyze_run(run_dir)
        if run_metrics:
            all_results.append(run_metrics)
    
    if not all_results:
        print("No results found to analyze.")
        return

    summary_df = pd.DataFrame(all_results)
    
    column_order = [
        'Experiment', 'Accuracy (cm, deg)', 'Tracking Uptime (%)', 
        'Stability (m/s, deg/s)', 'Match Efficiency (%)', 'Recovery Time (frames)', 
        'Perf. Breakdown (ms)'
    ]
    summary_df = summary_df.reindex(columns=column_order)

    print("\n\n--- ABLATION STUDY SUMMARY (ADVANCED METRICS) ---")
    print(summary_df.to_string(index=False))
    
    summary_csv_path = root_dir / "ablation_summary_advanced.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary saved to {summary_csv_path}")

if __name__ == "__main__":
    main()
