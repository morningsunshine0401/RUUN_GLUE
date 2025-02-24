# utils.py

import cv2
import torch
torch.set_grad_enabled(False)

import numpy as np
import os

def frame2tensor(frame, device):
    if frame is None:
        raise ValueError('Received an empty frame.')
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(gray / 255.).float()[None, None].to(device)
    return tensor

def create_unique_filename(directory, base_filename):
    filename, ext = os.path.splitext(base_filename)
    counter = 1
    new_filename = base_filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{filename}_{counter}{ext}"
        counter += 1
    return new_filename

def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])  # Roll
        y = np.arctan2(-R[2, 0], sy)      # Pitch
        z = np.arctan2(R[1, 0], R[0, 0])  # Yaw
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])  # Roll
        y = np.arctan2(-R[2, 0], sy)       # Pitch
        z = 0                              # Yaw
    return np.array([x, y, z])

def euler_angles_to_rotation_matrix(theta):
    roll, pitch, yaw = theta
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = R_z @ R_y @ R_x
    return R

def preprocess_image_for_lightglue(img):
    # If needed, resizing is done in PoseEstimator before this call
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # shape: (H,W) -> (1,1,H,W)
    gray = gray[None, None]
    return gray

# ---------------------------------------------------------------------
#         QUATERNION UTILITY FUNCTIONS
# ---------------------------------------------------------------------

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion [x, y, z, w].
    """
    q = np.zeros(4, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        # Otherwise, find the largest diagonal element and proceed
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s
    return q

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion [x, y, z, w] to a 3x3 rotation matrix.
    """
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1.0 - 2.0*(yy + zz),     2.0*(xy - wz),         2.0*(xz + wy)],
        [2.0*(xy + wz),           1.0 - 2.0*(xx + zz),   2.0*(yz - wx)],
        [2.0*(xz - wy),           2.0*(yz + wx),         1.0 - 2.0*(xx + yy)]
    ], dtype=np.float64)
    return R

def normalize_quaternion(q):
    """
    Ensure quaternion has unit length.
    """
    norm_q = np.linalg.norm(q)
    if norm_q < 1e-15:
        # Fallback to identity if somehow zero length
        return np.array([0, 0, 0, 1], dtype=q.dtype)
    return q / norm_q

