# 파일 이름: ablation_metrics.py (RecursionError Fix)
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

# --- UTILITY FUNCTION (can be placed outside or inside analyze_run) ---
def calculate_add_score(est_R, est_T, gt_R, gt_T, model_points):
    """Calculates the ADD score for a single frame."""
    points_est = (est_R @ model_points.T).T + est_T
    points_gt = (gt_R @ model_points.T).T + gt_T
    distances = np.linalg.norm(points_est - points_gt, axis=1)
    return np.mean(distances)
# ----------------------------------------------------------------------

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

    # --- REPLACEMENT FOR "1. Accuracy" section ---
    # 1. Accuracy (Sparse ADD cm)
    # Filter for frames that were accepted AND have ground truth available
    gt_df = df[(df['accepted'] == True) & (df['gt_available'] == True)].copy()
    # Ensure all necessary pose and point data exists for the calculation
    gt_df.dropna(subset=['gt_pos_xyz', 'gt_q_xyzw', 'kf_pos_xyz', 'kf_q_xyzw', 'pnp_points_3d'], inplace=True)

    if not gt_df.empty:
        from scipy.spatial.transform import Rotation as R_scipy
        
        add_scores_cm = []
        # Your diameter is 60cm, so the threshold is 6cm
        add_threshold_cm = 6 

        for _, row in gt_df.iterrows():
            # Load the sparse 3D model points used for this frame's PnP
            model_points = np.array(row['pnp_points_3d'])
            if model_points.shape[0] == 0: continue

            # Load Ground Truth Pose
            gt_tvec = np.array(row['gt_pos_xyz'])
            gt_quat = np.array(row['gt_q_xyzw'])
            gt_rmat = R_scipy.from_quat(gt_quat).as_matrix()

            # Load Estimated Pose (from Kalman Filter)
            est_tvec = np.array(row['kf_pos_xyz'])
            est_quat = np.array(row['kf_q_xyzw'])
            est_rmat = R_scipy.from_quat(est_quat).as_matrix()

            # Calculate the ADD score (in meters) and convert to cm
            score_m = calculate_add_score(est_rmat, est_tvec, gt_rmat, gt_tvec, model_points)
            add_scores_cm.append(score_m * 100)
        
        if add_scores_cm:
            # Calculate the mean error
            mean_add_error = np.mean(add_scores_cm)
            results['Mean ADD-P Error (cm)'] = f"{mean_add_error:.2f}"
            
            # Calculate the percentage of frames below the threshold (e.g., "ADD < 10% diam")
            correct_frames = np.sum(np.array(add_scores_cm) <= add_threshold_cm)
            accuracy_percent = (correct_frames / len(gt_df)) * 100
            results[f'Accuracy (ADD-P < {add_threshold_cm:.0f}cm)'] = f"{accuracy_percent:.2f}%"
        else:
            results['Mean ADD-P Error (cm)'] = "N/A"
            results[f'Accuracy (ADD-P < {add_threshold_cm:.0f}cm)'] = "N/A"
    else:
        results['Mean ADD-P Error (cm)'] = "N/A"
        results[f'Accuracy (ADD-P < 6cm)'] = "N/A"

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
    
    # column_order = [
    #     'Experiment', 'Accuracy (cm, deg)', 'Tracking Uptime (%)', 
    #     'Stability (m/s, deg/s)', 'Match Efficiency (%)', 'Recovery Time (frames)', 
    #     'Perf. Breakdown (ms)'
    # ]

    # --- AFTER ---
    column_order = [
        'Experiment', 'Accuracy (ADD-P < 6cm)', 'Mean ADD-P Error (cm)', 
        'Tracking Uptime (%)', 'Stability (m/s, deg/s)', 
        'Match Efficiency (%)', 'Recovery Time (frames)', 
        'Perf. Breakdown (ms)'
    ]
    # You can remove 'Mean ADD-P Error (cm)' if the table becomes too wide
    summary_df = summary_df.reindex(columns=column_order)

    print("\n\n--- ABLATION STUDY SUMMARY (ADVANCED METRICS) ---")
    print(summary_df.to_string(index=False))
    
    summary_csv_path = root_dir / "ablation_summary_advanced.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary saved to {summary_csv_path}")

if __name__ == "__main__":
    main()
