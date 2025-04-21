#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process‑model checker for MultExtendedKalmanFilter (warm‑up edition).

What it does
------------
1. Generates a constant‑velocity & constant‑rate ground‑truth trajectory.
2. Initialises the EKF with the exact state at t = 0.
3. For the first `--warmup` steps:
       update() with *noise‑free* (or user‑specified noise) measurements.
   For the remaining steps:
       predict() only – no updates.
4. Prints the first / last few predictions and plots GT vs. KF.

Run examples
------------
python3 kf_process_check.py                 # default: predict‑only
python3 kf_process_check.py --warmup 5      # perfect updates for k=0…4
python3 kf_process_check.py --warmup 3 --noise-pos 0.01
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
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
    ang  = w_norm * dt
    axis = w_body / w_norm
    dq   = np.hstack([axis*np.sin(ang/2), np.cos(ang/2)])
    qn   = quat_mul(dq, q)
    return qn / np.linalg.norm(qn)

def quat2euler(q):
    x, y, z, w = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])

def quat_angle(q1, q2):
    dot = np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2*np.arccos(dot) * 180/np.pi

# ── synthetic ground truth ───────────────────────────────────────────
def make_truth(dt, n_steps, v0, w_body):
    p = np.zeros((n_steps, 3))
    q = np.zeros((n_steps, 4));  q[0, 3] = 1.0
    for k in range(1, n_steps):
        p[k] = p[k-1] + v0 * dt
        q[k] = integrate_quat(q[k-1], w_body, dt)
    return p, q

# ── main routine ────────────────────────────────────────────────────
def run(args):
    n_steps = int(args.time / args.dt) + 1
    p_gt, q_gt = make_truth(args.dt, n_steps,
                            np.array(args.vel),
                            np.deg2rad(args.rate))

    kf = MultExtendedKalmanFilter(args.dt)
    kf.x[0:3]  = p_gt[0]
    kf.x[6:10] = q_gt[0]

    pos_pred   = np.zeros((n_steps, 3))
    eul_pred   = np.zeros((n_steps, 3))
    pos_err_log, ang_err_log = [], []

    pos_pred[0] = p_gt[0]
    eul_pred[0] = quat2euler(q_gt[0])
    pos_err_log.append(0.0)
    ang_err_log.append(0.0)

    for k in range(1, n_steps):
        kf.predict()

        # ---------- warm‑up updates (noise‑free or noisy as chosen) -----
        if k < args.warmup:
            meas = np.hstack([
                p_gt[k] + np.random.randn(3)*args.noise_pos,
                q_gt[k] + np.random.randn(4)*args.noise_quat
            ])
            meas[3:7] /= np.linalg.norm(meas[3:7])
            kf.update(meas)

        # ---------- logging -------------------------------------------
        x_pred, _ = kf.get_state()
        pos_pred[k] = x_pred[:3]
        eul_pred[k] = quat2euler(x_pred[6:10])

        pos_err_log.append(np.linalg.norm(pos_pred[k] - p_gt[k]))
        ang_err_log.append(quat_angle(x_pred[6:10], q_gt[k]))

        if k < 5 or k > n_steps-5:
            print(f"k={k:3d}  GT pos={p_gt[k]}  Pred pos={pos_pred[k]}  |err|={pos_err_log[-1]:.3e}")

    # ------------- PLOTS (same layout as before) ----------------------
    k_arr = np.arange(n_steps)

    # 1) position components
    fig_pos, ax_pos = plt.subplots(3,1, figsize=(10,7), sharex=True)
    for i,lbl in enumerate(['x','y','z']):
        ax_pos[i].plot(k_arr, p_gt[:,i],  'k-', label='truth')
        ax_pos[i].plot(k_arr, pos_pred[:,i], 'b--', label='KF predict')
        ax_pos[i].set_ylabel(lbl+' (m)'); ax_pos[i].grid(ls=':')
        if i == 0: ax_pos[i].legend()
    ax_pos[-1].set_xlabel('step')
    fig_pos.suptitle(f'Position (warm‑up updates: {args.warmup})')
    plt.tight_layout()

    # 2) orientation (RPY)
    eul_gt = np.array([quat2euler(q) for q in q_gt])
    fig_eul, ax_eul = plt.subplots(3,1, figsize=(10,7), sharex=True)
    for i,lbl in enumerate(['roll','pitch','yaw']):
        ax_eul[i].plot(k_arr, np.rad2deg(eul_gt[:,i]),  'k-', label='truth')
        ax_eul[i].plot(k_arr, np.rad2deg(eul_pred[:,i]),'b--', label='KF predict')
        ax_eul[i].set_ylabel(lbl+' (deg)'); ax_eul[i].grid(ls=':')
        if i == 0: ax_eul[i].legend()
    ax_eul[-1].set_xlabel('step')
    fig_eul.suptitle('Orientation')
    plt.tight_layout()

    # 3) error curves
    fig_err, ax_err = plt.subplots(2,1, figsize=(8,5), sharex=True)
    ax_err[0].plot(k_arr, pos_err_log, 'r-')
    ax_err[0].set_ylabel('|pos err| (m)'); ax_err[0].grid(ls=':')
    ax_err[1].plot(k_arr, ang_err_log, 'g-')
    ax_err[1].set_ylabel('ang err (deg)'); ax_err[1].set_xlabel('step')
    ax_err[1].grid(ls=':')
    plt.tight_layout()

    # 4) 3‑D trajectory
    fig3d = plt.figure(figsize=(6,5))
    ax3 = fig3d.add_subplot(111, projection='3d')
    ax3.plot(p_gt[:,0], p_gt[:,1], p_gt[:,2], 'k-', lw=2, label='truth')
    ax3.plot(pos_pred[:,0], pos_pred[:,1], pos_pred[:,2], 'b--', lw=1.5, label='KF predict')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.set_title('3‑D trajectory')
    ax3.legend()
    plt.tight_layout(); plt.show()

# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dt",   type=float, default=1/30, help="time step (s)")
    ap.add_argument("--time", type=float, default=10.0, help="total time (s)")
    ap.add_argument("--vel",  nargs=3, type=float, default=[0.2,-0.1,0.0],
                    help="constant body velocity (m/s)")
    ap.add_argument("--rate", nargs=3, type=float, default=[2.0,0.0,5.0],
                    help="body rates (deg/s)")
    # NEW arguments
    ap.add_argument("--warmup", type=int, default=0,
                    help="number of initial steps where a perfect (noise‑free) "
                         "measurement update is applied")
    ap.add_argument("--noise-pos",  type=float, default=0.0,
                    help="σ position noise during warm‑up (0 = perfect)")
    ap.add_argument("--noise-quat", type=float, default=0.0,
                    help="σ quaternion component noise during warm‑up (0 = perfect)")
    args = ap.parse_args()
    run(args)
