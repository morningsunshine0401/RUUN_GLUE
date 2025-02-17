import json
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

def analyze_boundary(json_path, start_frame, end_frame, transition_frame, window=5):
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter frames
    frames = [d for d in data if start_frame <= d['frame'] <= end_frame]
    
    # Create dataframe
    metrics = pd.DataFrame([{
        'frame': d['frame'],
        'matches': d['total_matches'],
        'success_rate': d['inlier_ratio'] * 100,
        'coverage': d['coverage_score'],
        'view_diff': d['viewpoint_diff_deg'],
        'phase': 'Transition' if abs(d['frame'] - transition_frame) <= 1 
                else 'Pre' if d['frame'] < transition_frame 
                else 'Post'
    } for d in frames])
    
    # Calculate statistics for transition window
    trans_mask = metrics['frame'].between(
        transition_frame - window, 
        transition_frame + window
    )
    transition_stats = metrics[trans_mask].describe()
    
    print("\nBoundary Analysis (±5 frames around transition)")
    print("----------------------------------------------")
    print(f"Match Count:    {transition_stats['matches']['mean']:.1f} ± {transition_stats['matches']['std']:.1f}")
    print(f"Success Rate:   {transition_stats['success_rate']['mean']:.1f}% ± {transition_stats['success_rate']['std']:.1f}%")
    print(f"Coverage Score: {transition_stats['coverage']['mean']:.3f} ± {transition_stats['coverage']['std']:.3f}")
    print(f"View Diff:      {transition_stats['view_diff']['mean']:.1f}° ± {transition_stats['view_diff']['std']:.1f}°")
    
    # Print detailed transition metrics
    print("\nDetailed Metrics Around Transition")
    print("----------------------------------")
    transition_detail = metrics[trans_mask].sort_values('frame')
    print(tabulate(transition_detail, headers='keys', floatfmt='.2f', tablefmt='pipe'))
    
    return metrics

# Usage
if __name__ == "__main__":
    json_path = "20250124_ICUAS1_cali_6.json"
    metrics = analyze_boundary(json_path, 124, 165, 144)
    
    # Example plots if needed
    plt.figure(figsize=(12, 4))
    plt.plot(metrics['frame'], metrics['matches'], 'o-', label='Matches')
    plt.plot(metrics['frame'], metrics['coverage']*10, 's-', label='Coverage x10')
    plt.axvline(x=144, color='r', linestyle='--', label='Transition')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frame')
    plt.title('Metrics Across Viewpoint Transition')
    plt.show()