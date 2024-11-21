import re
import numpy as np

# Input and output file paths
input_file = "keypoint_2d.txt"  # Replace with your actual file name
output_file = "keypoints_fixed.py"

# Read the contents of the input file
with open(input_file, "r") as f:
    data = f.read()

# Use regular expressions to extract all keypoint pairs
matches = re.findall(r'\[\s*([\d\.]+)\s+([\d\.]+)\s*\]', data)

# Convert matches to a properly formatted NumPy array string
keypoints = np.array([[float(x), float(y)] for x, y in matches])

# Save the cleaned keypoints to a Python file
with open(output_file, "w") as f:
    f.write("import numpy as np\n")
    f.write(f"keypoints = np.array({keypoints.tolist()})\n")

print(f"Formatted keypoints saved to {output_file}")
