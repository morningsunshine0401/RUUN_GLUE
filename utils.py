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

