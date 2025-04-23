#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Kalman Filter Analysis Tool

What it does
------------
1. Generates a constant-velocity & constant-rate ground-truth trajectory
2. Initializes the EKF with the exact state at t = 0
3. Allows custom measurement patterns:
   - Measurements at specified intervals (e.g., every 5th step)
   - Custom measurement noise for position and orientation
4. Tracks and logs detailed filter state (gains, covariances, etc.)
5. Plots comprehensive visualizations including:
   - Ground truth trajectory
   - Noisy measurements
   - KF estimates
   - Error analysis
   - Covariance evolution
   - Kalman gain analysis

Run examples
-----------
python3 kf_analysis_tool.py --meas-interval 5 --noise-pos 0.1 --noise-quat 0.01
python3 kf_analysis_tool.py --meas-interval 3 --noise-pos 0.05 --track-gains
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from KF_MK3 import MultExtendedKalmanFilter

# ── quaternion helpers ───────────────────────────────────────────────
def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def integrate_quat(q, w_body, dt):
    w_norm = np.linalg.norm(w_body)
    if w_norm < 1e-12:
        return q
    ang = w_norm * dt
    axis = w_body / w_norm
    dq = np.hstack([axis*np.sin(ang/2), np.cos(ang/2)])
    qn = quat_mul(dq, q)
    return qn / np.linalg.norm(qn)

def quat2euler(q):
    x, y, z, w = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])

def quat_angle(q1, q2):
    dot = np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2*np.arccos(dot) * 180/np.pi

# ── synthetic ground truth with complex trajectory ───────────────────
def make_truth(dt, n_steps, v0, w_body, complex_trajectory=True):
    p = np.zeros((n_steps, 3))
    q = np.zeros((n_steps, 4))
    q[0, 3] = 1.0  # Initial quaternion is [0,0,0,1]
    
    # Initialize velocity and angular velocity vectors (will be dynamic in complex mode)
    v = np.zeros((n_steps, 3))
    w = np.zeros((n_steps, 3))
    
    # Initial values
    v[0] = np.array(v0)
    w[0] = np.array(w_body)
    
    if not complex_trajectory:
        # Simple constant-velocity and constant-rate model
        for k in range(1, n_steps):
            p[k] = p[k-1] + v0 * dt
            q[k] = integrate_quat(q[k-1], w_body, dt)
    else:
        # Complex trajectory with velocity and angular rate changes
        for k in range(1, n_steps):
            # Time-varying parameters for dynamic trajectory
            t = k * dt
            
            # 1. Create velocity variations - combination of original velocity and sinusoidal variations
            if k > n_steps // 3 and k < 2 * n_steps // 3:
                # Create a more dramatic turn in the middle third of the trajectory
                v[k, 0] = v0[0] + 0.5 * np.sin(t * 0.5)
                v[k, 1] = v0[1] + 0.5 * np.cos(t * 0.5)
                v[k, 2] = v0[2] + 0.2 * np.sin(t * 1.5)
            else:
                # Normal variations at beginning and end
                v[k, 0] = v0[0] + 0.1 * np.sin(t * 0.2)
                v[k, 1] = v0[1] + 0.1 * np.cos(t * 0.2)
                v[k, 2] = v0[2] + 0.05 * np.sin(t * 0.3)
                
            # 2. Create angular velocity variations
            # Base angular rate changes with time
            w[k, 0] = w_body[0] + 2.0 * np.sin(t * 0.3)
            w[k, 1] = w_body[1] + 1.0 * np.cos(t * 0.2)
            w[k, 2] = w_body[2] + 3.0 * np.sin(t * 0.1)
            
            # Add a sudden rotation maneuver around 2/3 of the way through
            if abs(k - int(2 * n_steps / 3)) < n_steps / 20:
                w[k, 0] += 5.0 * np.exp(-((k - 2 * n_steps / 3) / (n_steps / 40))**2)
                w[k, 2] += 8.0 * np.exp(-((k - 2 * n_steps / 3) / (n_steps / 40))**2)
                
            # 3. Update position and orientation
            p[k] = p[k-1] + v[k] * dt
            q[k] = integrate_quat(q[k-1], w[k], dt)
    
    return p, q, v, w

# ── main routine ────────────────────────────────────────────────────
def run(args):
    n_steps = int(args.time / args.dt) + 1
    
    # Generate ground truth trajectory with complex motion
    p_gt, q_gt, v_gt, w_gt = make_truth(args.dt, n_steps,
                                      np.array(args.vel),
                                      np.deg2rad(args.rate),
                                      complex_trajectory=args.complex_trajectory)
    
    # Initialize Kalman filter with the exact state
    kf = MultExtendedKalmanFilter(args.dt)
    kf.x[0:3] = p_gt[0]                # Position
    kf.x[3:6] = v_gt[0]                # Linear velocity 
    kf.x[6:10] = q_gt[0]               # Quaternion
    kf.x[10:13] = w_gt[0]              # Angular velocity
    
    # If custom process noise is provided, set it
    if args.process_noise is not None:
        kf.Q = np.eye(kf.n_states) * args.process_noise
    
    # If custom measurement noise is provided, set it
    if args.meas_noise_r is not None:
        kf.R = np.eye(kf.n_measurements) * args.meas_noise_r
    
    # Arrays to store results
    pos_pred = np.zeros((n_steps, 3))
    vel_pred = np.zeros((n_steps, 3))
    eul_pred = np.zeros((n_steps, 3))
    w_pred = np.zeros((n_steps, 3))
    
    # Arrays to store measurements (including noise)
    pos_meas = np.zeros((n_steps, 3))
    q_meas = np.zeros((n_steps, 4))
    meas_provided = np.zeros(n_steps, dtype=bool)
    
    # Arrays for error logging
    pos_err_log = np.zeros(n_steps)
    ang_err_log = np.zeros(n_steps)
    
    # Arrays for KF state tracking
    if args.track_gains:
        # For Kalman gains (simplified - we'll track norm per measurement component)
        K_norms = np.zeros((n_steps, 7))  # 7 = num measurements
        
        # For covariance tracking (diagonals only for simplicity)
        P_diag_log = np.zeros((n_steps, kf.n_states))
        
        # Innovation tracking
        innovation_log = np.zeros((n_steps, 7))
        S_diag_log = np.zeros((n_steps, 7))  # Innovation covariance diagonals
    
    # Initialize first step
    pos_pred[0] = p_gt[0]
    vel_pred[0] = v_gt[0]
    eul_pred[0] = quat2euler(q_gt[0])
    w_pred[0] = w_gt[0]
    pos_err_log[0] = 0.0
    ang_err_log[0] = 0.0
    
    # Generate all noisy measurements in advance 
    # (but we'll only use them at specified intervals)
    for k in range(n_steps):
        pos_meas[k] = p_gt[k] + np.random.randn(3) * args.noise_pos
        quat_noise = np.random.randn(4) * args.noise_quat
        q_meas[k] = q_gt[k] + quat_noise
        q_meas[k] = q_meas[k] / np.linalg.norm(q_meas[k])  # Re-normalize
    
    # Main simulation loop
    for k in range(1, n_steps):
        # Always predict
        kf.predict()
        
        # Apply measurement update only at specified intervals
        if k % args.meas_interval == 0:
            meas_provided[k] = True
            meas = np.hstack([pos_meas[k], q_meas[k]])
            
            # For tracking gains - store pre-update values
            if args.track_gains:
                z_pred = kf._h(kf.x)
                H = kf._H_jacobian(kf.x)
                
                # Calculate innovation and innovation covariance
                y = meas - z_pred
                
                # Handle quaternion difference correctly
                if np.dot(meas[3:7], z_pred[3:7]) < 0:
                    y[3:7] = -meas[3:7] - z_pred[3:7]
                
                innovation_log[k] = y
                
                # Calculate innovation covariance S
                S = H @ kf.P @ H.T + kf.R
                S_diag_log[k] = np.diag(S)
                
                # Calculate Kalman gain and store its norm for each measurement component
                K = kf.P @ H.T @ np.linalg.inv(S)
                for i in range(7):
                    K_norms[k, i] = np.linalg.norm(K[:, i])
            
            # Perform the update
            kf.update(meas)
        
        # Get the current state estimate
        x_pred, P_pred = kf.get_state()
        
        # Log state and errors
        pos_pred[k] = x_pred[:3]
        vel_pred[k] = x_pred[3:6]
        eul_pred[k] = quat2euler(x_pred[6:10])
        w_pred[k] = x_pred[10:13]
        
        pos_err_log[k] = np.linalg.norm(pos_pred[k] - p_gt[k])
        ang_err_log[k] = quat_angle(x_pred[6:10], q_gt[k])
        
        if args.track_gains:
            P_diag_log[k] = np.diag(P_pred)
        
        # Print status for select steps
        if k < 5 or k % 50 == 0 or k > n_steps-5:
            print(f"k={k:3d}  GT pos={p_gt[k]}  Pred pos={pos_pred[k]}  |err|={pos_err_log[k]:.3e}")
    
    # ── VISUALIZATION ────────────────────────────────────────────────────────
    k_arr = np.arange(n_steps)
    
    # 1) Position components
    print("VISSSSSSSSSSSS\n")
    fig_pos, ax_pos = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, lbl in enumerate(['x', 'y', 'z']):
        ax_pos[i].plot(k_arr, p_gt[:, i], 'k-', label='Ground Truth')
        ax_pos[i].plot(k_arr, pos_pred[:, i], 'b-', label='KF Estimate')
        
        # Plot measurements where available
        meas_indices = np.where(meas_provided)[0]
        ax_pos[i].plot(meas_indices, pos_meas[meas_indices, i], 'ro', 
                      markersize=4, label='Measurements')
        
        ax_pos[i].set_ylabel(f'{lbl} (m)')
        ax_pos[i].grid(ls=':')
        if i == 0:
            ax_pos[i].legend()
    
    ax_pos[-1].set_xlabel('Step')
    fig_pos.suptitle(f'Position (Measurements every {args.meas_interval} steps)')
    plt.tight_layout()
    
    # 1b) Velocity components
    fig_vel, ax_vel = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, lbl in enumerate(['vx', 'vy', 'vz']):
        ax_vel[i].plot(k_arr, v_gt[:, i], 'k-', label='Ground Truth')
        ax_vel[i].plot(k_arr, vel_pred[:, i], 'b-', label='KF Estimate')
        
        ax_vel[i].set_ylabel(f'{lbl} (m/s)')
        ax_vel[i].grid(ls=':')
        if i == 0:
            ax_vel[i].legend()
    
    ax_vel[-1].set_xlabel('Step')
    fig_vel.suptitle('Velocity')
    plt.tight_layout()
    
    # 2) Orientation (RPY)
    eul_gt = np.array([quat2euler(q) for q in q_gt])
    eul_meas = np.array([quat2euler(q) for q in q_meas])
    
    fig_eul, ax_eul = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, lbl in enumerate(['roll', 'pitch', 'yaw']):
        ax_eul[i].plot(k_arr, np.rad2deg(eul_gt[:, i]), 'k-', label='Ground Truth')
        ax_eul[i].plot(k_arr, np.rad2deg(eul_pred[:, i]), 'b-', label='KF Estimate')
        
        # Plot measurements where available
        meas_indices = np.where(meas_provided)[0]
        ax_eul[i].plot(meas_indices, np.rad2deg(eul_meas[meas_indices, i]), 'ro', 
                      markersize=4, label='Measurements')
        
        ax_eul[i].set_ylabel(f'{lbl} (deg)')
        ax_eul[i].grid(ls=':')
        if i == 0:
            ax_eul[i].legend()
    
    ax_eul[-1].set_xlabel('Step')
    fig_eul.suptitle('Orientation')
    plt.tight_layout()
    
    # 2b) Angular Velocity 
    fig_angvel, ax_angvel = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, lbl in enumerate(['wx', 'wy', 'wz']):
        ax_angvel[i].plot(k_arr, np.rad2deg(w_gt[:, i]), 'k-', label='Ground Truth')
        ax_angvel[i].plot(k_arr, np.rad2deg(w_pred[:, i]), 'b-', label='KF Estimate')
        
        ax_angvel[i].set_ylabel(f'{lbl} (deg/s)')
        ax_angvel[i].grid(ls=':')
        if i == 0:
            ax_angvel[i].legend()
    
    ax_angvel[-1].set_xlabel('Step')
    fig_angvel.suptitle('Angular Velocity')
    plt.tight_layout()
    
    # 3) Error curves
    fig_err, ax_err = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax_err[0].plot(k_arr, pos_err_log, 'r-')
    ax_err[0].set_ylabel('|pos err| (m)')
    ax_err[0].grid(ls=':')
    
    # Mark measurement points on error plot
    meas_indices = np.where(meas_provided)[0]
    ax_err[0].plot(meas_indices, pos_err_log[meas_indices], 'ko', markersize=3)
    
    # Calculate and display RMSE for position
    pos_rmse = np.sqrt(np.mean(pos_err_log**2))
    ax_err[0].axhline(y=pos_rmse, color='m', linestyle='--', alpha=0.7)
    ax_err[0].text(0.02, 0.95, f'RMSE: {pos_rmse:.4f} m', 
                  transform=ax_err[0].transAxes, 
                  bbox=dict(facecolor='white', alpha=0.7))
    
    ax_err[1].plot(k_arr, ang_err_log, 'g-')
    ax_err[1].set_ylabel('ang err (deg)')
    ax_err[1].set_xlabel('Step')
    ax_err[1].grid(ls=':')
    
    # Mark measurement points on error plot
    ax_err[1].plot(meas_indices, ang_err_log[meas_indices], 'ko', markersize=3)
    
    # Calculate and display RMSE for orientation
    ang_rmse = np.sqrt(np.mean(ang_err_log**2))
    ax_err[1].axhline(y=ang_rmse, color='m', linestyle='--', alpha=0.7)
    ax_err[1].text(0.02, 0.95, f'RMSE: {ang_rmse:.4f} deg',
                  transform=ax_err[1].transAxes,
                  bbox=dict(facecolor='white', alpha=0.7))
    
    fig_err.suptitle('Position and Orientation Errors')
    plt.tight_layout()
    
    # 4) 3D trajectory
    fig3d = plt.figure(figsize=(10, 8))
    ax3 = fig3d.add_subplot(111, projection='3d')
    ax3.plot(p_gt[:, 0], p_gt[:, 1], p_gt[:, 2], 'k-', lw=2, label='Ground Truth')
    ax3.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], 'b-', lw=1.5, label='KF Estimate')
    
    # Plot measurements in 3D
    meas_indices = np.where(meas_provided)[0]
    ax3.scatter(pos_meas[meas_indices, 0], 
               pos_meas[meas_indices, 1], 
               pos_meas[meas_indices, 2], 
               c='r', marker='o', s=20, label='Measurements')
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D Trajectory')
    ax3.legend()
    plt.tight_layout()
    
    # 5) Kalman Filter Analysis Plots (if tracking gains)
    if args.track_gains:
        # Plot Covariance evolution
        fig_cov, ax_cov = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Position covariance
        ax_cov[0].set_title('Position Covariance')
        ax_cov[0].plot(k_arr, P_diag_log[:, 0], 'r-', label='$P_{xx}$')
        ax_cov[0].plot(k_arr, P_diag_log[:, 1], 'g-', label='$P_{yy}$')
        ax_cov[0].plot(k_arr, P_diag_log[:, 2], 'b-', label='$P_{zz}$')
        ax_cov[0].set_ylabel('Covariance')
        ax_cov[0].grid(ls=':')
        ax_cov[0].legend()
        
        # Velocity covariance
        ax_cov[1].set_title('Velocity Covariance')
        ax_cov[1].plot(k_arr, P_diag_log[:, 3], 'r-', label='$P_{vx}$')
        ax_cov[1].plot(k_arr, P_diag_log[:, 4], 'g-', label='$P_{vy}$')
        ax_cov[1].plot(k_arr, P_diag_log[:, 5], 'b-', label='$P_{vz}$')
        ax_cov[1].set_ylabel('Covariance')
        ax_cov[1].grid(ls=':')
        ax_cov[1].legend()
        
        # Quaternion and angular velocity
        ax_cov[2].set_title('Orientation & Angular Velocity Covariance')
        ax_cov[2].plot(k_arr, P_diag_log[:, 6], 'r-', label='$P_{qx}$')
        ax_cov[2].plot(k_arr, P_diag_log[:, 7], 'g-', label='$P_{qy}$')
        ax_cov[2].plot(k_arr, P_diag_log[:, 8], 'b-', label='$P_{qz}$')
        ax_cov[2].plot(k_arr, P_diag_log[:, 9], 'c-', label='$P_{qw}$')
        ax_cov[2].plot(k_arr, P_diag_log[:, 10], 'm-', label='$P_{wx}$')
        ax_cov[2].plot(k_arr, P_diag_log[:, 11], 'y-', label='$P_{wy}$')
        ax_cov[2].plot(k_arr, P_diag_log[:, 12], 'k-', label='$P_{wz}$')
        ax_cov[2].set_ylabel('Covariance')
        ax_cov[2].set_xlabel('Step')
        ax_cov[2].grid(ls=':')
        ax_cov[2].legend()
        
        plt.tight_layout()
        
        # Kalman Gain Analysis
        fig_gain, ax_gain = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Position-related gains
        ax_gain[0].set_title('Position Kalman Gains (Norm)')
        for i in range(3):
            ax_gain[0].plot(k_arr, K_norms[:, i], 
                           label=f'K for {"xyz"[i]}')
        ax_gain[0].set_ylabel('Gain Magnitude')
        ax_gain[0].grid(ls=':')
        ax_gain[0].legend()
        
        # Orientation-related gains
        ax_gain[1].set_title('Orientation Kalman Gains (Norm)')
        for i in range(3, 7):
            ax_gain[1].plot(k_arr, K_norms[:, i], 
                           label=f'K for q{"xyzw"[i-3]}')
        ax_gain[1].set_ylabel('Gain Magnitude')
        ax_gain[1].set_xlabel('Step')
        ax_gain[1].grid(ls=':')
        ax_gain[1].legend()
        
        plt.tight_layout()
        
        # Innovation Analysis
        fig_innov, ax_innov = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Position innovations
        ax_innov[0].set_title('Position Innovations')
        for i in range(3):
            ax_innov[0].plot(k_arr, innovation_log[:, i], 
                            label=f'Innovation {"xyz"[i]}')
            
            # Add 3-sigma bounds based on innovation covariance
            sigma = np.sqrt(S_diag_log[:, i])
            ax_innov[0].plot(k_arr, 3*sigma, 'k--', alpha=0.3)
            ax_innov[0].plot(k_arr, -3*sigma, 'k--', alpha=0.3)
            
        ax_innov[0].set_ylabel('Innovation')
        ax_innov[0].grid(ls=':')
        ax_innov[0].legend()
        
        # Orientation innovations
        ax_innov[1].set_title('Quaternion Innovations')
        for i in range(3, 7):
            ax_innov[1].plot(k_arr, innovation_log[:, i], 
                            label=f'Innovation q{"xyzw"[i-3]}')
            
            # Add 3-sigma bounds
            sigma = np.sqrt(S_diag_log[:, i])
            ax_innov[1].plot(k_arr, 3*sigma, 'k--', alpha=0.3)
            ax_innov[1].plot(k_arr, -3*sigma, 'k--', alpha=0.3)
            
        ax_innov[1].set_ylabel('Innovation')
        ax_innov[1].set_xlabel('Step')
        ax_innov[1].grid(ls=':')
        ax_innov[1].legend()
        
        plt.tight_layout()
    
    # 6) Compute and print performance metrics
    print("\n========== PERFORMANCE METRICS ==========")
    print(f"Position RMSE: {pos_rmse:.4f} m")
    print(f"Orientation RMSE: {ang_rmse:.4f} degrees")
    
    # Calculate more detailed metrics
    pos_mae = np.mean(pos_err_log)
    pos_max_err = np.max(pos_err_log)
    ang_mae = np.mean(ang_err_log)
    ang_max_err = np.max(ang_err_log)
    
    print(f"Position MAE: {pos_mae:.4f} m")
    print(f"Position Max Error: {pos_max_err:.4f} m")
    print(f"Orientation MAE: {ang_mae:.4f} degrees")
    print(f"Orientation Max Error: {ang_max_err:.4f} degrees")
    
    # Compute error metrics between measurements
    if meas_indices.size > 1:
        between_meas_pos_errors = []
        between_meas_ang_errors = []
        
        for i in range(len(meas_indices)-1):
            start_idx = meas_indices[i]
            end_idx = meas_indices[i+1]
            segment_pos_errors = pos_err_log[start_idx:end_idx]
            segment_ang_errors = ang_err_log[start_idx:end_idx]
            
            # Calculate max error between these two measurements
            between_meas_pos_errors.append(np.max(segment_pos_errors))
            between_meas_ang_errors.append(np.max(segment_ang_errors))
        
        avg_between_meas_pos_err = np.mean(between_meas_pos_errors)
        max_between_meas_pos_err = np.max(between_meas_pos_errors)
        avg_between_meas_ang_err = np.mean(between_meas_ang_errors)
        max_between_meas_ang_err = np.max(between_meas_ang_errors)
        
        print("\n--- Between Measurements Error Analysis ---")
        print(f"Average Max Position Error: {avg_between_meas_pos_err:.4f} m")
        print(f"Worst-case Position Error: {max_between_meas_pos_err:.4f} m")
        print(f"Average Max Orientation Error: {avg_between_meas_ang_err:.4f} degrees")
        print(f"Worst-case Orientation Error: {max_between_meas_ang_err:.4f} degrees")
    
    # Calculate velocity tracking performance
    vel_err_log = np.sqrt(np.sum((vel_pred - v_gt)**2, axis=1))
    vel_rmse = np.sqrt(np.mean(vel_err_log**2))
    vel_mae = np.mean(vel_err_log)
    vel_max_err = np.max(vel_err_log)
    
    print("\n--- Velocity Tracking Performance ---")
    print(f"Velocity RMSE: {vel_rmse:.4f} m/s")
    print(f"Velocity MAE: {vel_mae:.4f} m/s")
    print(f"Velocity Max Error: {vel_max_err:.4f} m/s")
    
    # Calculate angular velocity tracking performance
    w_err_log = np.sqrt(np.sum((w_pred - w_gt)**2, axis=1))
    w_rmse = np.sqrt(np.mean(w_err_log**2))
    w_mae = np.mean(w_err_log)
    w_max_err = np.max(w_err_log)
    
    print("\n--- Angular Velocity Tracking Performance ---")
    print(f"Angular Velocity RMSE: {np.rad2deg(w_rmse):.4f} deg/s")
    print(f"Angular Velocity MAE: {np.rad2deg(w_mae):.4f} deg/s")
    print(f"Angular Velocity Max Error: {np.rad2deg(w_max_err):.4f} deg/s")
    
    # 7) Plot additional performance metrics
    fig_vel_err = plt.figure(figsize=(12, 5))
    plt.plot(k_arr, vel_err_log, 'b-')
    plt.axhline(y=vel_rmse, color='m', linestyle='--', alpha=0.7)
    plt.ylabel('|velocity error| (m/s)')
    plt.xlabel('Step')
    plt.grid(ls=':')
    plt.title('Velocity Error Magnitude')
    plt.text(0.02, 0.95, f'RMSE: {vel_rmse:.4f} m/s', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    
    fig_w_err = plt.figure(figsize=(12, 5))
    plt.plot(k_arr, np.rad2deg(w_err_log), 'g-')
    plt.axhline(y=np.rad2deg(w_rmse), color='m', linestyle='--', alpha=0.7)
    plt.ylabel('|angular velocity error| (deg/s)')
    plt.xlabel('Step')
    plt.grid(ls=':')
    plt.title('Angular Velocity Error Magnitude')
    plt.text(0.02, 0.95, f'RMSE: {np.rad2deg(w_rmse):.4f} deg/s', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    
    # Return data for further analysis if needed
    return {
        'ground_truth': {
            'position': p_gt,
            'velocity': v_gt,
            'quaternion': q_gt,
            'euler': eul_gt,
            'angular_velocity': w_gt
        },
        'measurements': {
            'position': pos_meas,
            'quaternion': q_meas,
            'provided_at': meas_indices
        },
        'kf_estimate': {
            'position': pos_pred,
            'velocity': vel_pred,
            'euler': eul_pred,
            'angular_velocity': w_pred
        },
        'errors': {
            'position': pos_err_log,
            'angle': ang_err_log
        },
        'kf_analysis': {
            'K_norms': K_norms if args.track_gains else None,
            'P_diag': P_diag_log if args.track_gains else None,
            'innovation': innovation_log if args.track_gains else None,
            'S_diag': S_diag_log if args.track_gains else None
        }
    }

# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dt", type=float, default=1/30, help="time step (s)")
    ap.add_argument("--time", type=float, default=10.0, help="total time (s)")
    ap.add_argument("--vel", nargs=3, type=float, default=[0.2, -0.1, 0.0],
                    help="initial body velocity (m/s)")
    ap.add_argument("--rate", nargs=3, type=float, default=[2.0, 0.0, 5.0],
                    help="initial body rates (deg/s)")
    
    # Trajectory type
    ap.add_argument("--complex-trajectory", action="store_true", default=True,
                    help="use complex trajectory with varying velocity and angular rates")
    ap.add_argument("--simple-trajectory", dest="complex_trajectory", action="store_false",
                    help="use simple trajectory with constant velocity and angular rates")
    
    # Measurement settings
    ap.add_argument("--meas-interval", type=int, default=5,
                    help="interval between measurements (steps)")
    ap.add_argument("--noise-pos", type=float, default=0.05,
                    help="σ position measurement noise (m)")
    ap.add_argument("--noise-quat", type=float, default=0.01,
                    help="σ quaternion measurement noise")
    
    # Filter tuning parameters (optional)
    ap.add_argument("--process-noise", type=float, default=None,
                    help="σ process noise (Q diagonal elements)")
    ap.add_argument("--meas-noise-r", type=float, default=None,
                    help="σ measurement noise (R diagonal elements)")
    
    # Analysis options
    ap.add_argument("--track-gains", action="store_true",
                    help="track and plot Kalman gains and covariances")
    ap.add_argument("--save-plots", action="store_true",
                    help="save plots to files")
    
    args = ap.parse_args()
    results = run(args)
    plt.show()
