# RUUN_GLUE
Pose estimation using deep features and matchers.
#<img src="assets/20241115_arch_1.png" width="960" height="480">
#<img src="assets/20241115_arch_2.png" width="960" height="480">

# Pose Estimation Project Guide

This guide provides an overview of the pose estimation project, explaining the purpose of each script and detailing the functions within them. The explanations focus on the inputs, outputs, and intuitive purposes of the functions to help you understand and navigate the codebase effectively.

---

## Overview

The project consists of the following scripts:

- **`main.py`**: The main script that orchestrates the pose estimation process.
- **`utils.py`**: Contains utility functions used across the project.
- **`kalman_filter.py`**: Implements a Kalman filter for smoothing pose estimates.
- **`pose_estimator.py`**: Contains the `PoseEstimator` class, which handles keypoint detection, matching, and pose estimation.

---

## Script Details

### 1. `main.py`

#### **Purpose**
This script serves as the entry point of the pose estimation process. It parses command-line arguments, initializes the `PoseEstimator`, processes the input video or image sequence, and handles visualization and output.

#### **Key Components**

- **Argument Parsing**: Parses command-line arguments for input video, anchor image, output directory, and various parameters related to the SuperPoint and SuperGlue models.
- **Device Setup**: Determines whether to run on CPU or GPU based on availability and user input.
- **PoseEstimator Initialization**: Creates an instance of the `PoseEstimator` class with the specified options.
- **Video Processing Loop**: Reads and processes frames using the `PoseEstimator`.
- **Visualization and Output**: Displays pose estimation results and saves the output data.

#### **Main Flow**

1. **Parse Arguments**
   - **Inputs**: Command-line arguments.
   - **Outputs**: Configuration object (`opt`).
   - **Purpose**: Configures the pose estimation process.

2. **Set Device**
   - Determines CPU/GPU availability.
   - Moves computation to the selected device.

3. **Initialize PoseEstimator**
   - **Inputs**: `opt`, `device`.
   - **Outputs**: An instance of `PoseEstimator`.

4. **Open Video Capture**
   - **Inputs**: Video file path or camera index.
   - **Outputs**: A `cv2.VideoCapture` object.

5. **Processing Loop**
   - Reads frames, processes each frame with `PoseEstimator`, visualizes results, and saves outputs.

6. **Cleanup**
   - Releases video capture, closes OpenCV windows, and saves pose data to a JSON file.

---

### 2. `utils.py`

#### **Purpose**
Provides utility functions used across the project, such as image preprocessing, unique filename generation, and rotation conversions.

#### **Functions**

1. **`frame2tensor(frame, device)`**
   - **Purpose**: Converts an image frame to a normalized tensor.
   - **Inputs**: 
     - `frame`: Input image (NumPy array).
     - `device`: Target device (`'cpu'` or `'cuda'`).
   - **Outputs**: 
     - Normalized PyTorch tensor.
   - **Description**: Converts to grayscale, normalizes pixel values, and prepares tensor for the neural network.

2. **`create_unique_filename(directory, base_filename)`**
   - **Purpose**: Generates a unique filename in the directory.
   - **Inputs**:
     - `directory`: Target directory.
     - `base_filename`: Desired filename.
   - **Outputs**:
     - Unique filename.
   - **Description**: Appends a counter to ensure uniqueness if the base filename already exists.

3. **`rotation_matrix_to_euler_angles(R)`**
   - **Purpose**: Converts a rotation matrix to Euler angles.
   - **Inputs**:
     - `R`: 3x3 rotation matrix.
   - **Outputs**:
     - Euler angles (`[roll, pitch, yaw]`).
   - **Description**: Handles singular cases when `sy` is close to zero.

4. **`euler_angles_to_rotation_matrix(theta)`**
   - **Purpose**: Converts Euler angles to a rotation matrix.
   - **Inputs**:
     - `theta`: Euler angles (`[roll, pitch, yaw]`).
   - **Outputs**:
     - Rotation matrix.
   - **Description**: Constructs the matrix by combining rotations around X, Y, and Z axes.

---

### 3. `kalman_filter.py`

#### **Purpose**
Implements a Kalman filter for smoothing pose estimates, reducing noise, and providing stability by combining current measurements and predicted states.

#### **Class: `KalmanFilterPose`**

1. **Initialization**
   - **Inputs**:
     - `dt`: Time step between frames.
     - `n_states`: Number of state variables (default = 18).
     - `n_measurements`: Number of measurement variables (default = 6).
   - **Description**: Sets up matrices for transitions, measurements, and noise covariance.

2. **Methods**

   - **`_init_kalman_filter(self)`**
     - **Purpose**: Initializes Kalman filter matrices.
     - **Inputs**: None.
     - **Outputs**: None (modifies class attributes).

   - **`predict(self)`**
     - **Purpose**: Predicts the next state.
     - **Inputs**: None.
     - **Outputs**:
       - Predicted translation vector.
       - Predicted Euler angles.
   
   - **`correct(self, tvec, R)`**
     - **Purpose**: Updates state estimate with new measurements.
     - **Inputs**:
       - `tvec`: Translation vector.
       - `R`: Rotation matrix.
     - **Outputs**: None (updates state).

---

### 4. `pose_estimator.py`

#### **Purpose**
Handles the entire pose estimation process, including feature extraction, matching, pose computation, and Kalman filtering.

#### **Class: `PoseEstimator`**

1. **Initialization**
   - **Inputs**:
     - `opt`: Parsed options.
     - `device`: Target device.
   - **Description**: Initializes models, anchor image, Kalman filter, and configurations.

2. **Methods**

   - **`_init_matching(self)`**
     - **Purpose**: Sets up SuperPoint and SuperGlue models.
   
   - **`_load_anchor_image(self)`**
     - **Purpose**: Loads and preprocesses the anchor image.

   - **`_extract_anchor_features(self)`**
     - **Purpose**: Extracts keypoints and descriptors using SuperPoint.
   
   - **`_match_anchor_keypoints(self)`**
     - **Purpose**: Matches keypoints with known 3D points.

   - **`_init_kalman_filter(self)`**
     - **Purpose**: Initializes the Kalman filter.

   - **`process_frame(self, frame, frame_idx)`**
     - **Purpose**: Processes a frame for pose estimation.

   - **`estimate_pose(self, mkpts0, mkpts1, mpts3D, mconf, frame, frame_idx, frame_keypoints)`**
     - **Purpose**: Estimates camera pose using matched keypoints.

   - **`_get_camera_intrinsics(self)`**
     - **Purpose**: Provides intrinsic parameters.

   - **`_kalman_filter_update(self, ...)`**
     - **Purpose**: Updates Kalman filter and constructs pose data.

   - **`_visualize_matches(self, ...)`**
     - **Purpose**: Visualizes matches and pose information.

---

## Summary

This codebase provides a robust pipeline for estimating object poses in video sequences using SuperPoint and SuperGlue for feature matching, PnP algorithms for pose computation, and Kalman filtering for smoothing. The modular design facilitates easy modification and extension while maintaining clarity and separation of concerns across different stages of the pipeline.
