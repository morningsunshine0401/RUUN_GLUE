import json
from collections import defaultdict

# -------------------- Configuration / Thresholds --------------------
# Adjust these thresholds as needed:
ROLL_ERROR_THRESHOLD = 3.0
PITCH_ERROR_THRESHOLD = 2.0
YAW_ERROR_THRESHOLD = 4.0

# JSON file with frame analysis data:
JSON_FILENAME = "20250203_test2_analysis_cali_diff.json"

# -------------------- Provided Template Features --------------------
# This is the complete list of 2D keypoints (features) from the template image.
provided_template_features = [
    [511, 293],
    [591, 284],
    [610, 269],
    [587, 330],
    [413, 249],
    [602, 348],
    [715, 384],
    [598, 298],
    [656, 171],
    [805, 213],
    [703, 392],
    [523, 286],
    [519, 327],
    [387, 289],
    [727, 126],
    [425, 243],
    [636, 358],
    [745, 202],
    [595, 388],
    [436, 260],
    [539, 313],
    [795, 220],
    [351, 291],
    [665, 165],
    [611, 353],
    [650, 377],
    [516, 389],
    [727, 143],
    [496, 378],
    [575, 312],
    [617, 368],
    [430, 312],
    [480, 281],
    [834, 225],
    [469, 339],
    [705, 223],
    [637, 156],
    [816, 414],
    [357, 195],
    [752, 77],
    [642, 451]
]

# -------------------- Error Data as Multi-line String --------------------
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

# -------------------- Helper Function to Parse Error Data --------------------
def parse_error_data(error_str):
    """
    Parse the provided multi-line error string into a list of (roll, pitch, yaw) tuples.
    """
    errors = []
    for line in error_str.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
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

# Parse the error data string
error_list = parse_error_data(error_data_str)
print(f"Parsed {len(error_list)} error entries.\n")

# -------------------- Load JSON Data --------------------
try:
    with open(JSON_FILENAME, "r") as jf:
        frames_data = json.load(jf)
except Exception as e:
    print(f"Error loading JSON file '{JSON_FILENAME}': {e}")
    exit(1)

if len(frames_data) != len(error_list):
    print(f"Warning: Number of frames in JSON ({len(frames_data)}) does not match number of error entries ({len(error_list)}).")

# -------------------- Optional: Frequency Analysis --------------------
# Count the frequency of outlier match indices in frames with high error.
outlier_frequency = defaultdict(int)

# -------------------- Process Each Frame --------------------
for idx, frame in enumerate(frames_data):
    frame_number = frame.get("frame", idx + 1)
    
    # Get corresponding error values (if available)
    if idx < len(error_list):
        roll_error, pitch_error, yaw_error = error_list[idx]
    else:
        roll_error = pitch_error = yaw_error = None

    print(f"\n=== Frame {frame_number} ===")
    if roll_error is not None:
        print(f"Pose Errors -> Roll: {roll_error:.4f}, Pitch: {pitch_error:.4f}, Yaw: {yaw_error:.4f}")
    else:
        print("No error data available for this frame.")

    # Extract match information from JSON
    total_matches = frame.get("total_matches", "N/A")
    num_inliers = frame.get("num_inliers", "N/A")
    inliers = frame.get("inliers", [])
    reproj_errors = frame.get("reprojection_errors", [])
    mconf = frame.get("mconf", [])
    
    print(f"Total Matches: {total_matches}, Inliers: {num_inliers}")

    # Print details for each match
    print("Match details (index, status, reprojection error, confidence):")
    for i, (r_err, conf) in enumerate(zip(reproj_errors, mconf)):
        status = "Inlier" if i in inliers else "Outlier"
        print(f"  Index {i:2d}: {status:7s} | Reproj Error: {r_err:6.3f} | Confidence: {conf:5.3f}")

    # -------------------- Template Feature Usage --------------------
    # Check which template keypoints were used for pose estimation.
    # Prefer the JSON key 'mkpts0' if present; otherwise, use the provided template features.
    if "mkpts0" in frame:
        print("\nTemplate features (mkpts0) and their usage:")
        for i, pt in enumerate(frame["mkpts0"]):
            usage = "Used" if i in inliers else "Not used"
            print(f"  Index {i:2d}: {pt} -> {usage}")
    else:
        print("\nTemplate features (mkpts0) not found in frame; using provided template features:")
        for i, pt in enumerate(provided_template_features):
            # Without per-frame usage info, mark as 'N/A'
            print(f"  Index {i:2d}: {pt} -> N/A")

    # -------------------- Analysis for Each Axis --------------------
    high_error = False  # Flag to mark if any axis exceeds its threshold

    # Check roll error
    if roll_error is not None and abs(roll_error) > ROLL_ERROR_THRESHOLD:
        print("\n>> High roll error detected.")
        high_error = True
        # List outlier matches (sorted by reprojection error)
        outlier_matches = [(i, reproj_errors[i]) for i in range(len(reproj_errors)) if i not in inliers]
        outlier_matches_sorted = sorted(outlier_matches, key=lambda x: x[1], reverse=True)
        print("   Outlier matches (sorted by reprojection error):")
        for mi, err in outlier_matches_sorted:
            print(f"     Index {mi:2d} -> Reproj Error: {err:.3f}")

    # Check pitch error
    if pitch_error is not None and abs(pitch_error) > PITCH_ERROR_THRESHOLD:
        print("\n>> High pitch error detected.")
        high_error = True
        # Calculate average reprojection error for inliers
        if inliers:
            inlier_errors = [reproj_errors[i] for i in inliers if i < len(reproj_errors)]
            avg_inlier_error = sum(inlier_errors) / len(inlier_errors)
            print(f"   Average inlier reprojection error: {avg_inlier_error:.3f}")
        else:
            print("   No inlier data available.")

    # Check yaw error
    if yaw_error is not None and abs(yaw_error) > YAW_ERROR_THRESHOLD:
        print("\n>> High yaw error detected.")
        high_error = True
        # Calculate average reprojection error for outliers
        outlier_errors = [reproj_errors[i] for i in range(len(reproj_errors)) if i not in inliers]
        if outlier_errors:
            avg_outlier_error = sum(outlier_errors) / len(outlier_errors)
            print(f"   Average outlier reprojection error: {avg_outlier_error:.3f}")
        else:
            print("   No outlier data available.")

    # -------------------- Optional Additional Analysis --------------------
    # If this frame has high error, update frequency counts for outlier match indices.
    if high_error:
        for i in range(len(reproj_errors)):
            if i not in inliers:
                outlier_frequency[i] += 1

# -------------------- Summary of Additional Analysis --------------------
print("\n=== Summary of Outlier Index Frequency in High Error Frames ===")
if outlier_frequency:
    sorted_outliers = sorted(outlier_frequency.items(), key=lambda x: x[1], reverse=True)
    for index_val, count in sorted_outliers:
        print(f"Index {index_val:2d} appeared as an outlier in high error frames {count} time(s).")
else:
    print("No high error frames or outlier frequency data collected.")
