import json
import numpy as np
from collections import defaultdict
from itertools import combinations

# -------------------------- Configuration --------------------------
JSON_FILENAME = "20250203_test2_analysis_cali_diff.json"  # JSON file with pose estimation results
REPROJ_ERROR_THRESHOLD = 4.0  # threshold to flag a match as “problematic”

# -------------------------- Define Template Points --------------------------
# List of 2D template feature points (order must match your system)
template_points = [
    [511, 293],  # Index 0
    [591, 284],  # Index 1
    [610, 269],  # Index 2
    [587, 330],  # Index 3
    [413, 249],  # Index 4
    [602, 348],  # Index 5
    [715, 384],  # Index 6
    [598, 298],  # Index 7
    [656, 171],  # Index 8
    [805, 213],  # Index 9
    [703, 392],  # Index 10
    [523, 286],  # Index 11
    [519, 327],  # Index 12
    [387, 289],  # Index 13
    [727, 126],  # Index 14
    [425, 243],  # Index 15
    [636, 358],  # Index 16
    [745, 202],  # Index 17
    [595, 388],  # Index 18
    [436, 260],  # Index 19
    [539, 313],  # Index 20
    [795, 220],  # Index 21
    [351, 291],  # Index 22
    [665, 165],  # Index 23
    [611, 353],  # Index 24
    [650, 377],  # Index 25
    [516, 389],  # Index 26
    [727, 143],  # Index 27
    [496, 378],  # Index 28
    [575, 312],  # Index 29
    [617, 368],  # Index 30
    [430, 312],  # Index 31
    [480, 281],  # Index 32
    [834, 225],  # Index 33
    [469, 339],  # Index 34
    [705, 223],  # Index 35
    [637, 156],  # Index 36
    [816, 414],  # Index 37
    [357, 195],  # Index 38
    [752, 77],   # Index 39
    [642, 451]   # Index 40
]

# -------------------------- Load Error Data --------------------------
error_data_str = """
-2.87508159420280	2.76612895845545	3.09096145763310
-2.18370240023830	-0.0720614141875102	3.41867228867466
-3.62240079202531	0.665479369823793	4.03659048405178
-3.88911021388846	1.53819650051909	2.80630954608538
-1.26306022676583	1.65090411478012	2.64101902184585
-2.33482864107223	1.42691247663274	3.55213750256083
7.59239837690399	-0.405623535396472	6.78358860074081
1.67033497660428	1.54130280901352	4.40973420161992
-1.88548856718047	1.29872389559992	4.05177710597985
-3.42724890763146	1.90304108579658	3.32177213838082
0.968725247814871	-1.91437500328280	6.64989120501970
-1.54413868693222	-0.702128120344810	5.68546430020666
-4.06770254715183	1.33493475145258	4.94031213775460
-4.65542804396181	1.87064679863588	3.85737886314775
-5.11062290638625	1.97891514772921	3.36000780760973
-5.03171110304320	2.18959157500223	2.58576975377389
-4.30431449313412	2.25735025779410	2.53364447640583
-5.18616946129831	2.23346796608692	3.61149691049088
-4.82461073001199	1.96013145756026	3.51194543347354
-3.66266141290216	1.76524710865251	3.18321971456496
-1.99249686759641	0.854809982137164	3.16362502831114
-2.03583396542012	0.789989718999286	3.35641118205419
-3.07359606153410	1.38234809234779	3.16558531435012
-3.08171824901512	1.71569434456704	2.77400713905385
-3.06118135259860	2.71067374164890	1.97814424511355
-3.42422593557755	2.43058479734075	2.10453713545822
-1.77255729582969	1.55141353399108	2.59002129810387
-1.15190873161276	0.920188276918703	2.98070054230155
1.63652830813646	-0.906801981398660	5.08887363663462
-0.627336602479248	-0.128810112612845	4.63843845281951
-0.349661255367581	-0.0342231147301767	4.12415960030801
-1.63040086701707	0.836425346443513	3.52902439615950
"""

def parse_error_data(error_str):
    errors = []
    for line in error_str.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            roll = float(parts[0])
            pitch = float(parts[1])
            yaw = float(parts[2])
            errors.append((roll, pitch, yaw))
        except ValueError:
            continue
    return errors

error_list = parse_error_data(error_data_str)
print(f"Parsed {len(error_list)} error entries.\n")

# -------------------------- Load Pose Estimation JSON Data --------------------------
with open(JSON_FILENAME, "r") as f:
    frames_data = json.load(f)

# -------------------------- Identify Problematic Matches --------------------------
# We will count the frequency of matches that exceed the REPROJ_ERROR_THRESHOLD
problematic_frequency = defaultdict(int)

for idx, frame in enumerate(frames_data):
    reproj_errors = frame.get("reprojection_errors", [])
    # (mconf and inliers are available if you want to filter further)
    for i, error in enumerate(reproj_errors):
        if error > REPROJ_ERROR_THRESHOLD:
            problematic_frequency[i] += 1

print("\n=== Frequency of Problematic Matches (Reproj Error > {:.2f}) ===".format(REPROJ_ERROR_THRESHOLD))
if problematic_frequency:
    for index_val, count in sorted(problematic_frequency.items(), key=lambda x: x[1], reverse=True):
        # Get the corresponding template point (if exists)
        point_str = str(template_points[index_val]) if index_val < len(template_points) else "N/A"
        print(f"Match index {index_val} ({point_str}) appeared with high error {count} time(s).")
else:
    print("No problematic matches found based on the reprojection error threshold.")

# -------------------------- List Frames with High Yaw Error --------------------------
HIGH_YAW_THRESHOLD = 4.5  # Adjust as needed

print("\n=== Frames with High Yaw Error (Abs(Yaw Error) > {:.2f}) ===".format(HIGH_YAW_THRESHOLD))
for idx, frame in enumerate(frames_data):
    if idx < len(error_list):
        roll, pitch, yaw = error_list[idx]
        if abs(yaw) > HIGH_YAW_THRESHOLD:
            print(f"Frame {frame.get('frame', idx+1)}: Yaw Error = {abs(yaw):.3f}")
            # Get the indices of matches with high reprojection error
            high_error_indices = [i for i, e in enumerate(frame.get("reprojection_errors", [])) if e > REPROJ_ERROR_THRESHOLD]
            # Map indices to template point coordinates
            high_error_points = [template_points[i] if i < len(template_points) else "N/A" for i in high_error_indices]
            print("   High error match indices:", high_error_indices)
            print("   Corresponding template points:", high_error_points)

# -------------------------- List Frames with High Roll Error --------------------------
HIGH_ROLL_THRESHOLD = 3  # Adjust as needed

print("\n=== Frames with High Roll Error (Abs(Roll Error) > {:.2f}) ===".format(HIGH_ROLL_THRESHOLD))
for idx, frame in enumerate(frames_data):
    if idx < len(error_list):
        roll, pitch, yaw = error_list[idx]
        if abs(roll) > HIGH_ROLL_THRESHOLD:
            print(f"Frame {frame.get('frame', idx+1)}: Roll Error = {abs(roll):.3f}")
            high_error_indices = [i for i, e in enumerate(frame.get("reprojection_errors", [])) if e > REPROJ_ERROR_THRESHOLD]
            high_error_points = [template_points[i] if i < len(template_points) else "N/A" for i in high_error_indices]
            print("   High error match indices:", high_error_indices)
            print("   Corresponding template points:", high_error_points)

# -------------------------- List Frames with High Pitch Error --------------------------
HIGH_PITCH_THRESHOLD = 2  # Adjust as needed

print("\n=== Frames with High Pitch Error (Abs(Pitch Error) > {:.2f}) ===".format(HIGH_PITCH_THRESHOLD))
for idx, frame in enumerate(frames_data):
    if idx < len(error_list):
        roll, pitch, yaw = error_list[idx]
        if abs(pitch) > HIGH_PITCH_THRESHOLD:
            print(f"Frame {frame.get('frame', idx+1)}: Pitch Error = {abs(pitch):.3f}")
            high_error_indices = [i for i, e in enumerate(frame.get("reprojection_errors", [])) if e > REPROJ_ERROR_THRESHOLD]
            high_error_points = [template_points[i] if i < len(template_points) else "N/A" for i in high_error_indices]
            print("   High error match indices:", high_error_indices)
            print("   Corresponding template points:", high_error_points)



# # -------------------------- Analyze Pairwise Combinations --------------------------
# # Here we will look at the inlier match indices per frame and record the absolute yaw error.
# # (You can modify the code to analyze roll or pitch instead, or even consider triples, etc.)
# pair_yaw_errors = defaultdict(list)

# for idx, frame in enumerate(frames_data):
#     # Get the yaw error for this frame (use absolute value)
#     if idx < len(error_list):
#         _, _, yaw = error_list[idx]
#         yaw_error = abs(yaw)
#     else:
#         yaw_error = 0
#     inliers = frame.get("inliers", [])
    
#     # We require at least 2 inliers to form a pair
#     if len(inliers) < 2:
#         continue
    
#     # For each pair of inlier match indices, record the yaw error
#     for pair in combinations(sorted(inliers), 2):
#         pair_yaw_errors[pair].append(yaw_error)

# # Compute the average yaw error for each pair combination
# pair_avg_yaw_error = {pair: np.mean(errors) for pair, errors in pair_yaw_errors.items()}

# # Optionally, sort the pairs by average yaw error (highest first)
# sorted_pairs = sorted(pair_avg_yaw_error.items(), key=lambda x: x[1], reverse=True)

# print("=== Pairwise Inlier Combinations and Average Yaw Error ===")
# for pair, avg_err in sorted_pairs:
#     freq = len(pair_yaw_errors[pair])
#     print(f"Pair {pair} appeared in {freq} frame(s) with an average yaw error of {avg_err:.3f}°")
