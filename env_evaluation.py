import json
import pandas as pd
import numpy as np

def analyze_illumination_metrics(json_paths, condition_names):
   """
   Analyze metrics across different illumination conditions
   
   Args:
       json_paths (list): List of paths to JSON files for each illumination condition
       condition_names (list): Names of illumination conditions matching json_paths
   """
   results = []
   
   for json_path, condition in zip(json_paths, condition_names):
       with open(json_path, 'r') as f:
           data = json.load(f)
           
       # Calculate metrics per frame
       frame_metrics = []
       for frame in data:
           metrics = {
               'frame': frame['frame'],
               'condition': condition,
               'detected_features': len(frame.get('mkpts0', [])),
               'total_matches': frame['total_matches'],
               'inlier_matches': frame['num_inliers'],
               'success_rate': frame['inlier_ratio'] * 100,
               'coverage_score': frame['coverage_score']
           }
           frame_metrics.append(metrics)
           
       # Calculate statistics
       df = pd.DataFrame(frame_metrics)
       stats = {
           'condition': condition,
           'avg_detections': df['detected_features'].mean(),
           'std_detections': df['detected_features'].std(),
           'avg_matches': df['total_matches'].mean(),
           'std_matches': df['total_matches'].std(),
           'avg_success': df['success_rate'].mean(),
           'std_success': df['success_rate'].std(),
           'avg_coverage': df['coverage_score'].mean(),
           'std_coverage': df['coverage_score'].std()
       }
       results.append(stats)
   
   # Create summary table
   summary_df = pd.DataFrame(results)
   print("\nIllumination Condition Analysis")
   print("------------------------------")
   print(summary_df.round(2).to_string())
   
   return summary_df

# Usage example:
json_paths = [
   'ORB_sync_boundary.json',
   'ORB_sync_varying.json', 
   'ORB_sync_dark.json'#'20250123_ICUAS_2_dark_2_cov.json'
]
conditions = ['Wide Base Line ORB', 'Varying ORB', 'Low Light ORB']

# # Usage example:
# json_paths = [
#    '20250123_ICUAS_2_boundary_hard_3_cov.json',
#    '20250123_ICUAS_2_varying_1_cov.json', 
#    '20250123_ICUAS_2_dark.json'#'20250123_ICUAS_2_dark_2_cov.json'
# ]
# conditions = ['Wide Base Line', 'Varying', 'Low Light']

metrics = analyze_illumination_metrics(json_paths, conditions)