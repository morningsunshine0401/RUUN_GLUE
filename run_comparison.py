# File: analyze_only.py (Final Fix for Shape and Index Errors)
import os
import pandas as pd
import numpy as np

# --- File Definitions ---
MK48_OUTPUT_FILE = os.path.join("output", "VAPE_MK48_comparison_results.json")
MK53_OUTPUT_DIR = "run_out_20250908_slow_view_change"
MK53_OUTPUT_FILE = os.path.join(MK53_OUTPUT_DIR, "frames.jsonl")

def analyze_results(df, name):
    """Calculates all metrics for a single run's dataframe."""
    results = {"ALGORITHM": name}
    
    # --- Overall Performance ---
    if 'pose_success' in df.columns:
        results["Tracking Uptime (%)"] = df['pose_success'].mean() * 100
    else:
        results["Tracking Uptime (%)"] = np.nan

    # Standardize column names
    df.rename(columns={
        'detection_time_ms': 'det_ms', 'feature_time_ms': 'feat_ms',
        'matching_time_ms': 'match_ms', 'pnp_time_ms': 'pnp_ms',
        'kf_time_ms': 'kf_ms',
    }, inplace=True)
    
    if 'vp_ms' not in df.columns: df['vp_ms'] = 0 
    
    time_cols = ['det_ms', 'vp_ms', 'feat_ms', 'match_ms', 'pnp_ms', 'kf_ms']
    for col in time_cols:
        if col not in df.columns: df[col] = 0
    
    df['total_time'] = df[time_cols].sum(axis=1)
    results["Avg. Processing Time (ms/frame)"] = df['total_time'].mean()
    results["Perf. Breakdown (D/VP/F+M/PnP/KF)"] = f"{df['det_ms'].mean():.1f}/{df['vp_ms'].mean():.1f}/{(df['feat_ms'] + df['match_ms']).mean():.1f}/{df['pnp_ms'].mean():.1f}/{df['kf_ms'].mean():.1f}"

    # --- Robustness ---
    if 'recovery_frames' in df.columns:
        recovery_frames = df[df['recovery_frames'] > 0]['recovery_frames']
        results["Avg. Recovery Time (frames)"] = recovery_frames.mean() if not recovery_frames.empty else 0
    else:
        results["Avg. Recovery Time (frames)"] = "N/A"

    # --- Accuracy & Efficiency ---
    if 'pose_success' in df.columns and 'gt_available' in df.columns:
        
        ok_frames = df[(df['pose_success'] == True) & (df['gt_available'] == True)].copy()

        if not ok_frames.empty and 'filt_pos_err_cm' in ok_frames.columns and 'filt_rot_err_deg' in ok_frames.columns:
            results["Strict Accuracy (5cm, 5° %)"] = ((ok_frames['filt_pos_err_cm'] <= 5) & (ok_frames['filt_rot_err_deg'] <= 5)).mean() * 100
            results["Mean Pos Error (cm)"] = ok_frames['filt_pos_err_cm'].mean()
            results["Median Pos Error (cm)"] = ok_frames['filt_pos_err_cm'].median()
            results["P95 Pos Error (cm)"] = ok_frames['filt_pos_err_cm'].quantile(0.95)
            results["StdDev Pos Error (cm)"] = ok_frames['filt_pos_err_cm'].std()
            results["Mean Rot Error (deg)"] = ok_frames['filt_rot_err_deg'].mean()
            results["Median Rot Error (deg)"] = ok_frames['filt_rot_err_deg'].median()
            results["P95 Rot Error (deg)"] = ok_frames['filt_rot_err_deg'].quantile(0.95)
            results["StdDev Rot Error (deg)"] = ok_frames['filt_rot_err_deg'].std()
            
            if 'jitter_lin_vel_mps' in ok_frames.columns: results["Positional Jitter (StdDev m/s)"] = ok_frames['jitter_lin_vel_mps'].std()
            if 'jitter_ang_vel_dps' in ok_frames.columns: results["Rotational Jitter (StdDev deg/s)"] = ok_frames['jitter_ang_vel_dps'].std()
            
            match_col = 'num_matches' if 'num_matches' in ok_frames.columns else 'raw_matches'
            if match_col in ok_frames.columns and 'num_inliers' in ok_frames.columns:
                efficient_frames = ok_frames[ok_frames[match_col] > 0]
                if not efficient_frames.empty:
                    results["Match Efficiency (%)"] = (efficient_frames['num_inliers'] / efficient_frames[match_col]).mean() * 100
            
            if 'num_inliers' in ok_frames.columns: results["Avg. Inlier Count"] = ok_frames['num_inliers'].mean()
    return results

def main():
    try:
        print(f"📊 Loading results from {MK48_OUTPUT_FILE} and {MK53_OUTPUT_FILE}...")
        df48 = pd.read_json(MK48_OUTPUT_FILE)
        df53 = pd.read_json(MK53_OUTPUT_FILE, lines=True)
        print("✅ Results loaded successfully.")
        
        df48.reset_index(drop=True, inplace=True)
        df53.reset_index(drop=True, inplace=True)

    except FileNotFoundError as e:
        print(f"\n❌ Error: Cannot find results file. Please run the scripts first.")
        print(f"Details: {e}")
        return

    # --- NEW, ROBUST FIX for duplicate columns ---
    # The definitive success metric from MK53 is 'is_accepted_by_gate'.
    if 'is_accepted_by_gate' in df53.columns:
        # If an old 'pose_success' column also exists, drop it FIRST to avoid duplication.
        if 'pose_success' in df53.columns:
            df53 = df53.drop(columns=['pose_success'])
        
        # Now, safely rename the correct column.
        df53.rename(columns={'is_accepted_by_gate': 'pose_success'}, inplace=True)
    # --- END OF FIX ---
    
    res48 = analyze_results(df48.copy(), "VAPE_MK48")
    res53 = analyze_results(df53.copy(), "VAPE_MK53")

    summary_df = pd.DataFrame([res48, res53]).set_index("ALGORITHM").T.fillna('N/A')
    summary_df = summary_df.round(2)
    print("\n" + "="*70)
    print("📊 FINAL COMPARISON RESULTS")
    print("="*70)
    print(summary_df.to_string())
    print("="*70)

if __name__ == "__main__":
    main()

