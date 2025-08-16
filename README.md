# VAPE MK53 - Real-time 6-DOF Aircraft Pose Estimator

**Enhanced with Timestamp Support for Vision Latency Correction**

<img src="assets/demo_pose_estimation.gif" width="960" height="480">

## Overview

VAPE MK53 is a state-of-the-art real-time 6-DOF (6 Degrees of Freedom) pose estimation system designed specifically for aircraft tracking. It combines classical robotics techniques with modern deep learning to achieve robust, low-latency pose estimation with proper handling of vision processing delays.

### Key Features

- 🚁 **Real-time Aircraft Tracking**: Specialized for aircraft pose estimation with 14 viewpoint-specific anchors
- ⏱️ **Timestamp-Aware Processing**: Canonical VIO/SLAM approach for handling vision latency
- 🧠 **Enhanced Unscented Kalman Filter**: Variable-dt prediction with fixed-lag buffer for out-of-sequence measurements
- 🎯 **Multi-threaded Architecture**: Optimized for both low-latency display (30 FPS) and accurate processing
- 🔧 **Physics-Based Filtering**: Rate limiting prevents impossible orientation/position jumps
- 📊 **Adaptive Viewpoint Selection**: Intelligent switching between 14 pre-computed viewing angles

---

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   MainThread    │    │ ProcessingThread │
│   (30 FPS)      │    │   (Variable)     │
│                 │    │                  │
│ • Camera capture│    │ • YOLO detection │
│ • Timestamp     │ ┌──│ • Feature match  │
│ • Visualization │ │  │ • Pose estimation│
│ • UKF prediction│ │  │ • UKF update     │
└─────────────────┘ │  └─────────────────┘
         │           │           │
         └─── Queues + Locks ────┘
                   │
            ┌─────────────┐
            │   Enhanced  │
            │     UKF     │
            │(Timestamp-  │
            │   Aware)    │
            └─────────────┘
```

### Multi-Threading Design

- **MainThread**: High-frequency capture and display (30 FPS) with immediate timestamp recording
- **ProcessingThread**: AI-heavy computation (YOLO + SuperPoint + LightGlue + PnP) with timestamp-aware updates
- **Enhanced UKF**: Handles measurements at correct historical times with variable-dt motion models

---

## Technical Innovation

### Timestamp-Aware Vision Processing

Unlike traditional pose estimation systems that suffer from vision latency, VAPE MK53 implements the canonical VIO/SLAM approach:

1. **Immediate Timestamp Capture**: `t_capture = time.monotonic()` recorded the moment frames are obtained
2. **Latency-Corrected Updates**: UKF processes measurements at their actual capture time, not processing time
3. **Fixed-Lag Buffer**: 200-frame history enables handling of out-of-sequence measurements
4. **Variable-dt Motion Model**: Adapts to actual time intervals instead of assuming fixed frame rates

### Enhanced Unscented Kalman Filter

**State Vector (16D):**
```python
# [0:3]   - Position (x, y, z)
# [3:6]   - Velocity (vx, vy, vz)  
# [6:9]   - Acceleration (ax, ay, az)
# [9:13]  - Quaternion (qx, qy, qz, qw)
# [13:16] - Angular velocity (wx, wy, wz)
```

**Key Features:**
- **dt-Scaled Process Noise**: `Q_scaled = Q * dt + Q * (dt²) * 0.5`
- **Quaternion Normalization**: Prevents numerical drift
- **Rate Limiting**: Physics-based constraints prevent impossible motions
- **Robust Covariance**: SVD fallback for numerical stability

---

## Computer Vision Pipeline

### 1. Multi-Scale Object Detection
- **YOLO v8**: Custom trained on aircraft ("iha" class)
- **Adaptive Thresholding**: 0.30 → 0.20 → 0.10 confidence cascade
- **Largest-Box Selection**: Focuses on primary aircraft target

### 2. Deep Feature Extraction & Matching
- **SuperPoint**: CNN-based keypoint detector (up to 2048 keypoints)
- **LightGlue**: Attention-based feature matching with early termination
- **14 Viewpoint Anchors**: Pre-computed reference images for different viewing angles

### 3. Robust Pose Estimation
- **EPnP + RANSAC**: Initial pose estimation with outlier rejection
- **VVS Refinement**: Virtual Visual Servoing for sub-pixel accuracy
- **Temporal Consistency**: Viewpoint selection with failure recovery

### 4. Intelligent Viewpoint Management
```python
viewpoints = ['NE', 'NW', 'SE', 'SW', 'E', 'W', 'N', 'S', 
              'NE2', 'NW2', 'SE2', 'SW2', 'SU', 'NU']
```
- **Temporal Consistency**: Stick with working viewpoint
- **Adaptive Search**: Switch when current viewpoint fails
- **Quality Metrics**: Match count, inlier count, reprojection error

---

## Installation

### Requirements

**Python Version**: 3.11+

**Hardware Requirements**:
- NVIDIA GPU with CUDA 12.2+ (recommended)
- 8GB+ RAM
- USB camera or video input

### Dependencies

```bash
# Core Dependencies
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Computer Vision & AI
pip install ultralytics>=8.0.0
pip install lightglue
pip install opencv-python>=4.8.0

# Scientific Computing
pip install numpy>=1.24.0
pip install scipy>=1.11.0

# Utilities
pip install matplotlib>=3.7.0
```

### Required Files

1. **YOLO Model**: `best.pt` (trained aircraft detection model)
2. **Anchor Images**: 14 viewpoint reference images (NE.png, NW.png, etc.)
3. **Input Video**: Your aircraft footage for processing

---

## Usage

### Basic Usage

```bash
# Real-time webcam processing
python3 VAPE_MK53_3.py --webcam --show

# Video file processing with feature visualization
python3 VAPE_MK53_3.py --video_file your_video.mp4 --show

# Image sequence processing
python3 VAPE_MK53_3.py --image_dir ./images/ --save_output

# Custom rate limiting for different scenarios
python3 VAPE_MK53_3.py --video_file fast_maneuvers.mp4 --max_rotation_dps 60 --max_position_mps 3.0
```

### Command Line Options

```bash
# Input Sources (required, mutually exclusive)
--webcam              # Use webcam input
--video_file PATH     # Process video file
--image_dir PATH      # Process image sequence

# Visualization Options
--show               # Show SuperPoint keypoint detections
--save_output        # Save pose data to JSON file

# UKF Tuning Parameters
--max_rotation_dps   # Maximum rotation rate (default: 30°/s)
--max_position_mps   # Maximum position speed (default: 1.5 m/s)
```

### Rate Limiting Presets

```python
# Handheld/Walking Around Aircraft
kf.set_rate_limits(max_rotation_dps=30.0, max_position_mps=1.5)

# Fast Movements/Drone Footage  
kf.set_rate_limits(max_rotation_dps=60.0, max_position_mps=3.0)

# Stable Tripod/Fixed Camera
kf.set_rate_limits(max_rotation_dps=15.0, max_position_mps=0.5)
```

---

## Output

### Real-time Display

- **Main Window**: Video with 3D coordinate axes overlaid on aircraft
- **Feature Window** (with `--show`): SuperPoint keypoints visualization
- **Console Output**: Timing, viewpoint selection, and rejection statistics

### Saved Data (with `--save_output`)

```json
{
  "frame": 42,
  "success": true,
  "position": [x, y, z],
  "quaternion": [qx, qy, qz, qw],
  "kf_position": [x_filtered, y_filtered, z_filtered],
  "kf_quaternion": [qx_f, qy_f, qz_f, qw_f],
  "num_inliers": 25,
  "viewpoint_used": "NW",
  "capture_time": 1234567.890
}
```

---

## Performance Characteristics

### Timing Analysis
```
🕒 Frame captured at t=1234.567
🔬 Processing latency: 125.3ms  
🎯 Total system latency: 167.8ms (capture→display)
⏭️ UKF predicting forward: 0.083s for proper temporal fusion
```

### Typical Performance
- **Main Thread**: 30 FPS (display)
- **Processing Thread**: 5-15 FPS (AI processing)
- **System Latency**: 100-200ms (capture to pose update)
- **Memory Usage**: ~2GB GPU, ~1GB RAM

---

## Algorithm Details

### Critical Timing Flow

```
t=0.000: Frame captured, t_capture recorded
t=0.033: Frame sent to processing queue  
t=0.080: YOLO detection completes
t=0.120: Feature matching finishes
t=0.125: UKF.update_with_timestamp(measurement, t_capture=0.000)
         ↳ Filter predicts back to t=0.000
         ↳ Applies measurement at correct time
         ↳ Fast-forwards to t=0.125 for display
```

### UKF Prediction Process

1. **Generate 33 Sigma Points** around current state estimate
2. **Propagate through Motion Model** (constant acceleration)
3. **Recombine with Weights** to get predicted mean and covariance
4. **dt-Scaled Process Noise** reflects uncertainty growth over time
5. **Quaternion Normalization** prevents numerical drift

### Physics-Based Validation

```python
# Orientation rate limiting
max_angle_change = max_rotation_dps * dt
if angle_diff > max_angle_change:
    reject_measurement("Orientation jump too large")

# Position rate limiting  
max_distance = max_position_mps * dt
if movement_distance > max_distance:
    reject_measurement("Position jump too large")
```

---

## Troubleshooting

### Common Issues

**1. YOLO Detection Failures**
```
🚫 No aircraft detected in frame
```
- Check lighting conditions
- Verify aircraft is clearly visible
- Try different confidence thresholds

**2. Excessive Rejections**
```
🚫 Frame 147: Rejected (Orientation Jump: 34.6° > 30°)
⚠️ Exceeded 5 consecutive rejections. Re-initializing KF.
```
- Increase rate limits for faster movements
- Check for motion blur or poor lighting
- Verify anchor images match aircraft type

**3. GPU Memory Issues**
```
CUDA out of memory
```
- Reduce video resolution
- Use CPU mode: set `device = 'cpu'`
- Close other GPU applications

**4. Missing Anchor Images**
```
FileNotFoundError: Required anchor image not found: NE.png
```
- Ensure all 14 viewpoint images are present
- Check file naming convention matches exactly

---

## Camera Calibration

For accurate pose estimation, replace the default camera intrinsics in `_get_camera_intrinsics()`:

```python
def _get_camera_intrinsics(self):
    # Replace with your camera's calibration data
    fx, fy, cx, cy = 1460.10150, 1456.48915, 604.85462, 328.64800
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K, None  # Add distortion coefficients if needed
```

---

## Research Applications

This system implements cutting-edge techniques from:

- **Visual-Inertial Odometry (VIO)**
- **Simultaneous Localization and Mapping (SLAM)**
- **Real-time Computer Vision**
- **Robust State Estimation**

### Academic Contributions

1. **Timestamp-Aware Pose Estimation**: Proper handling of vision processing latency
2. **Multi-threaded UKF Architecture**: Optimized for both accuracy and latency
3. **Adaptive Viewpoint Management**: Robust to viewing angle changes
4. **Physics-Based Measurement Validation**: Prevents impossible state transitions

---

## Development and Debugging

### Adding Debug Output

```python
# In MainThread.run() - Monitor capture timing
t_capture = time.monotonic()
print(f"🕒 CAPTURE: Frame {self.frame_count} at t={t_capture:.3f}")

# In ProcessingThread._process_frame() - Track latency
latency_ms = (time.monotonic() - t_capture) * 1000
print(f"🔬 PROCESS: Frame {frame_id}, latency={latency_ms:.1f}ms")

# In UKF.update_with_timestamp() - Monitor filter decisions
if t_meas >= self.t_state:
    print(f"⏭️ UKF: Predicting forward {dt:.3f}s")
else:
    print(f"⏮️ UKF: Out-of-sequence {abs(dt)*1000:.1f}ms late")
```

### VS Code Debugging Setup

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "VAPE MK53 Debug",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/VAPE_MK53_3.py",
            "args": ["--video_file", "test_video.mp4", "--show"],
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

---

## License

This project builds upon and extends [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) by Magic Leap, Inc. The original SuperGlue components are licensed under the terms provided by Magic Leap.

### Modifications and Enhancements

- Multi-threaded timestamp-aware architecture
- Enhanced Unscented Kalman Filter with variable-dt
- Aircraft-specific YOLO integration
- Viewpoint management system
- Physics-based measurement validation
- Real-time performance optimizations

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{vape_mk53_2025,
  title={VAPE MK53: Real-time 6-DOF Aircraft Pose Estimator with Timestamp Support},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]/VAPE_MK53}
}
```

**Original SuperGlue Citation:**
```bibtex
@inproceedings{sarlin20superglue,
  title={SuperGlue: Learning Feature Matching with Graph Neural Networks},
  author={Sarlin, Paul-Edouard and DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={CVPR},
  year={2020}
}
```

---

## Contributing

Contributions are welcome! Areas of interest:

- Additional aircraft viewpoint anchors
- Performance optimizations
- Extended camera support
- Improved motion models
- Better visualization options

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This system represents state-of-the-art real-time pose estimation combining classical robotics (Enhanced UKF) with modern deep learning (YOLO, SuperPoint, LightGlue). The timestamp-aware architecture follows canonical VIO/SLAM practices used in production robotics systems.
