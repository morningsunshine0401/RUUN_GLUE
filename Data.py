import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, f_oneway
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the JSON data

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

# Specify the file path to your JSON file
data_path = '/home/runbk0401/SuperGluePretrainedNetwork/Pose_estimation_JSON/pose_estimation_research_58.json'
data = load_data(data_path)

# Extract confidence scores and inlier ratios
data['mean_mconf'] = data['mconf'].apply(np.mean)  # Averaging confidence scores per frame
data['inlier_ratio'] = data['inlier_ratio']  # Assuming 'inlier_ratio' column exists

# Filter out frames with missing reprojection errors
filtered_data = data.dropna(subset=['mean_reprojection_error'])

# 1. Correlation Analysis between mean confidence and inlier ratios
# This analysis helps us understand if there’s a relationship between the confidence score (mconf)
# and the inlier ratio. If the correlation is positive, higher confidence should correspond to a higher
# inlier ratio, meaning more accurate pose estimations. A negative correlation implies the opposite.
correlation_coefficient, p_value = pearsonr(filtered_data['mean_mconf'], filtered_data['inlier_ratio'])
print("Correlation Coefficient between Confidence and Inlier Ratio:", correlation_coefficient)
print("P-value:", p_value)
# Interpretation: If the p-value is low (usually <0.05), the correlation is statistically significant.
# A positive correlation coefficient suggests confidence is associated with a higher inlier ratio (more accurate matches),
# while a negative value suggests the opposite. If the value is close to 0, there’s no clear relationship.


###########################################################################################################################################################################



# 2. Regression Analysis: Predicting reprojection error
# This regression predicts how different factors (inlier ratio, mean confidence score, and viewpoint) 
# influence reprojection error. Significant predictors can reveal which variables are most important 
# in reducing error and improving pose estimation.
formula = 'mean_reprojection_error ~ inlier_ratio + mean_mconf + C(predicted_viewpoint)'  # 'predicted_viewpoint' assumed categorical
model = smf.ols(formula=formula, data=filtered_data).fit()
print(model.summary())


# Adding an interaction term between number of inliers and inlier ratio to analyze their combined effect
formula = 'mean_reprojection_error ~ num_inliers * inlier_ratio + mean_mconf + C(predicted_viewpoint)'
model = smf.ols(formula=formula, data=filtered_data).fit()
print(model.summary())


# Interpretation:
# - The R-squared value tells us the percentage of variance in reprojection error explained by our predictors.
# - Significant predictors (p < 0.05) have coefficients suggesting how each variable impacts reprojection error.
# - For instance, if the coefficient for `inlier_ratio` is positive, it suggests that higher inlier ratios increase reprojection error.
# - A negative coefficient for `mean_mconf` would imply that higher confidence scores help reduce reprojection error, thus improving accuracy.
# - `predicted_viewpoint` factors show how different viewpoints influence reprojection error relative to a reference viewpoint.


# 3. Plotting Number of Inliers vs Reprojection Error
plt.figure(figsize=(10, 6))
plt.scatter(filtered_data['num_inliers'], filtered_data['mean_reprojection_error'], alpha=0.6)
plt.xlabel('Number of Inliers')
plt.ylabel('Mean Reprojection Error')
plt.title('Scatter Plot of Number of Inliers vs Mean Reprojection Error')
plt.show()

# 4. Visualization for Combined Effect
# Using a 3D scatter plot to visualize the combined effect of number of inliers and inlier ratio on reprojection error
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filtered_data['num_inliers'], filtered_data['inlier_ratio'], filtered_data['mean_reprojection_error'], alpha=0.6)
ax.set_xlabel('Number of Inliers')
ax.set_ylabel('Inlier Ratio')
ax.set_zlabel('Mean Reprojection Error')
ax.set_title('Combined Effect of Number of Inliers and Inlier Ratio on Reprojection Error')
plt.show()

###########################################################################################################################################################################


# 3. ANOVA Test for Viewpoints and reprojection error
# ANOVA tests if there are significant differences in reprojection error across viewpoints.
# If significant, this result suggests that some viewpoints yield higher (or lower) errors than others.
f_stat, p_value = f_oneway(*(filtered_data[filtered_data['predicted_viewpoint'] == vp]['mean_reprojection_error'] for vp in filtered_data['predicted_viewpoint'].unique()))
print("F-statistic for reprojection error across viewpoints:", f_stat)
print("P-value for ANOVA test:", p_value)
# Interpretation:
# - A low p-value (<0.05) suggests significant differences in reprojection error across viewpoints.
# - This means certain angles or viewpoints may impact pose estimation accuracy more than others.
# - If significant, this insight could guide adjustments or optimizations based on viewpoint in future models.

# 4. Outlier Detection for mean reprojection error
# Z-scores identify frames with unusually high reprojection errors (outliers).
# Frames with Z-scores above 3 or below -3 deviate significantly from the average reprojection error,
# which could indicate challenging conditions or areas where the model performs poorly.
filtered_data['z_score_reprojection'] = (filtered_data['mean_reprojection_error'] - filtered_data['mean_reprojection_error'].mean()) / filtered_data['mean_reprojection_error'].std()
outliers = filtered_data[abs(filtered_data['z_score_reprojection']) > 3]  # Z-score threshold
print("Outliers based on Mean Reprojection Error:")
print(outliers[['frame', 'mean_reprojection_error']])
# Interpretation:
# - Outliers highlight frames with errors much larger than typical values, suggesting specific issues in those frames.
# - Investigating these frames can provide clues about conditions that challenge the model, guiding areas for refinement.

# Visualization of reprojection error distribution
# A histogram of reprojection errors shows the spread of errors across frames.
# A tighter distribution around low values suggests more consistent accuracy, while a wide spread indicates variability.
plt.figure(figsize=(10, 6))
plt.hist(filtered_data['mean_reprojection_error'], bins=20, edgecolor='black')
plt.xlabel('Mean Reprojection Error')
plt.ylabel('Frequency')
plt.title('Distribution of Mean Reprojection Error')
plt.show()
# Interpretation:
# - This plot helps visualize how common certain error levels are.
# - A peak at lower reprojection error values would indicate high accuracy, while a broad distribution or multiple peaks might suggest variability in model performance.


###########################################################################################################################################################################


# 2. Impact of Total Matches on Pose Accuracy
# Group by low, medium, high total matches
data['match_category'] = pd.cut(data['total_matches'], bins=[0, 5, 10, 20], labels=['Low', 'Medium', 'High'])
match_accuracy = data.groupby('match_category').agg({'mean_reprojection_error': 'mean', 'inlier_ratio': 'mean'}).reset_index()

# Display match accuracy
print("Match Accuracy by Total Matches:")
print(match_accuracy)

# Plot results for impact of total matches
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(match_accuracy['match_category'], match_accuracy['mean_reprojection_error'])
plt.xlabel('Total Matches Category')
plt.ylabel('Mean Reprojection Error')
plt.title('Mean Reprojection Error by Match Count Category')

plt.subplot(1, 2, 2)
plt.bar(match_accuracy['match_category'], match_accuracy['inlier_ratio'])
plt.xlabel('Total Matches Category')
plt.ylabel('Inlier Ratio')
plt.title('Inlier Ratio by Match Count Category')
plt.show()



########################################################################################################################################################
# Load data from JSON
with open('/home/runbk0401/SuperGluePretrainedNetwork/Pose_estimation_JSON/pose_estimation_research_58.json', 'r') as f:
    data = json.load(f)

# Prepare DataFrames
frames_data = []
keypoints_data = []

for frame_data in data:
    frame = frame_data["frame"]
    num_inliers = frame_data.get("num_inliers", 0)
    total_matches = frame_data.get("total_matches", 0)
    inlier_ratio = frame_data.get("inlier_ratio", 0)
    mean_confidence = np.mean(frame_data.get("mconf", [])) if frame_data.get("mconf") else 0
    
    frames_data.append({
        "frame": frame,
        "num_inliers": num_inliers,
        "total_matches": total_matches,
        "inlier_ratio": inlier_ratio,
        "mean_confidence": mean_confidence
    })

    if frame_data.get("mconf") and frame_data.get("inliers") is not None:
        for idx, conf in enumerate(frame_data["mconf"]):
            is_inlier = 1 if idx in frame_data["inliers"] else 0
            keypoints_data.append({
                "frame": frame,
                "confidence": conf,
                "is_inlier": is_inlier
            })

# DataFrames
df_frames = pd.DataFrame(frames_data)
df_keypoints = pd.DataFrame(keypoints_data)

# 1. Correlation between confidence and inlier status (for individual keypoints)
conf_inlier_corr, conf_inlier_pval = pearsonr(df_keypoints["confidence"], df_keypoints["is_inlier"])
print("Correlation between Confidence and Inlier Status:")
print(f"Correlation: {conf_inlier_corr}, P-value: {conf_inlier_pval}\n")

# 2. Correlation between the number of inliers and mean confidence
inlier_mean_conf_corr, inlier_mean_conf_pval = pearsonr(df_frames["num_inliers"], df_frames["mean_confidence"])
print("Correlation between Number of Inliers and Mean Confidence Score:")
print(f"Correlation: {inlier_mean_conf_corr}, P-value: {inlier_mean_conf_pval}\n")

# 3. Correlation between total matches and number of inliers
matches_inliers_corr, matches_inliers_pval = pearsonr(df_frames["total_matches"], df_frames["num_inliers"])
print("Correlation between Total Matches and Number of Inliers:")
print(f"Correlation: {matches_inliers_corr}, P-value: {matches_inliers_pval}\n")


