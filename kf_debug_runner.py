#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# kf_debug_runner.py  – visual + covariance / gain diagnostics
"""
Visual & statistical debug harness for MultExtendedKalmanFilter.

Options
-------
--update-mod N   perform a measurement update every N steps
--update-phase P start updates at step P  (0 ≤ P < N)
--jac            run Jacobian finite‑difference test only

Example
-------
python3 kf_debug_runner.py --update-mod 2 --update-phase 0
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D       # noqa: F401
from scipy.stats import chi2
from KF_MK3 import MultExtendedKalmanFilter

# ───────────────────── quaternion helpers ────────────────────────
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
    wn = np.linalg.norm(w_body)
    if wn < 1e-12:
        return q
    ang, axis = wn*dt, w_body/wn
    dq = np.hstack([axis*np.sin(ang/2), np.cos(ang/2)])
    return quat_mul(dq, q)

def quat2euler(q):
    x, y, z, w = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])

def quat_angle(q1, q2):
    dot = np.clip(abs(np.dot(q1, q2)), 0.0, 1.0)
    return 2*np.arccos(dot) * 180/np.pi

# ───────────────────── synthetic truth generator ──────────────────
def make_truth(dt, n_steps, v0, w_body):
    p = np.zeros((n_steps,3))
    q = np.zeros((n_steps,4)); q[0,3] = 1.0
    for k in range(1, n_steps):
        p[k] = p[k-1] + v0*dt
        q[k] = integrate_quat(q[k-1], w_body, dt)
    return p, q

# ───────────────── Jacobian finite‑difference check ───────────────
def check_jacobians(kf, eps=1e-6):
    print("⇒ Jacobian finite‑difference check")
    x0,_ = kf.get_state()
    kf.predict(); P1 = kf.P.copy()
    F_an = (P1 - np.eye(kf.n_states)*0.1) @ np.linalg.inv(kf.P)
    kf.x, kf.P = x0.copy(), np.eye(kf.n_states)*0.1
    F_num = np.zeros_like(F_an)
    for i in range(kf.n_states):
        dx = np.zeros(kf.n_states); dx[i] = eps
        kf.x = x0 + dx; x_plus,_ = kf.predict()
        kf.x, kf.P = x0.copy(), np.eye(kf.n_states)*0.1
        x_nom,_ = kf.predict()
        F_num[:,i] = (x_plus - x_nom)/eps
    print(f"   max|F_an - F_num| = {np.max(np.abs(F_an-F_num)):.2e}")

    H_an = kf._H_jacobian(x0); H_num = np.zeros_like(H_an); z0 = kf._h(x0)
    for i in range(kf.n_states):
        dx = np.zeros(kf.n_states); dx[i]=eps
        H_num[:,i] = (kf._h(x0+dx)-z0)/eps
    print(f"   max|H_an - H_num| = {np.max(np.abs(H_an-H_num)):.2e}")

# ───────────────────────── main simulator ────────────────────────
def run_sim(args):
    n = int(args.time/args.dt)+1
    p_gt,q_gt = make_truth(args.dt,n,np.array(args.vel),np.deg2rad(args.rate))
    kf = MultExtendedKalmanFilter(args.dt)

    # storage -------------------------------------------------------
    pos_gt  = np.zeros((n,3)); pos_meas = np.zeros((n,3)); pos_kf = np.zeros((n,3))
    eul_gt  = np.zeros((n,3)); eul_meas = np.zeros((n,3)); eul_kf = np.zeros((n,3))
    nis_log, nees_log, traceP_pred, traceP_upd = [],[],[],[]
    var_pred_pos = np.zeros((n,3)); var_upd_pos = np.full((n,3), np.nan)
    K_pos_log    = np.full((n,3), np.nan)
    ang_err_log  = []
    # full covariance history
    P_pred_hist = []                # NEW  list of 13×13 np.array
    P_upd_hist  = []                # NEW  ''          (None when no update)

    # full Kalman gain history
    K_hist      = []                # NEW  list of 13×7 (None when no update)


    # loop ----------------------------------------------------------
    for k in range(n):
        # truth
        pos_gt[k] = p_gt[k];  eul_gt[k] = quat2euler(q_gt[k])

        # measurement
        meas = np.hstack([p_gt[k]+np.random.randn(3)*args.noise_pos,
                          q_gt[k]+np.random.randn(4)*args.noise_quat])
        meas[3:7] /= np.linalg.norm(meas[3:7])
        pos_meas[k] = meas[:3];  eul_meas[k] = quat2euler(meas[3:7])

        # prediction
        kf.predict(); P_prior = kf.P.copy()
        P_pred_hist.append(P_prior.copy())          # NEW

        traceP_pred.append(np.trace(P_prior)); var_pred_pos[k]=np.diag(P_prior)[:3]

        # decide update
        do_upd = (k % args.update_mod == args.update_phase)
        if do_upd:
            z_pred = kf._h(kf.x)
            y = meas - z_pred
            if np.dot(meas[3:7], z_pred[3:7])<0: y[3:7]*=-1
            H = kf._H_jacobian(kf.x)
            S = H @ P_prior @ H.T + kf.R
            K = P_prior @ H.T @ np.linalg.inv(S)
            K_hist.append(K.copy())                     # NEW
            K_pos_log[k] = K[0:3,0]
            nis_log.append(float(y.T @ np.linalg.inv(S) @ y))
            #print(f"k={k:3d}  K[0,0] = {K[0,0]:.3e}   K row0 = {K[0,:7]}")
            kf.update(meas)
            P_upd_hist.append(kf.P.copy())              # NEW
            err_state = np.hstack([p_gt[k]-kf.x[:3], q_gt[k]-kf.x[6:10], np.zeros(6)])
            nees_log.append(float(err_state.T @ np.linalg.inv(kf.P) @ err_state))
            traceP_upd.append(np.trace(kf.P)); var_upd_pos[k]=np.diag(kf.P)[:3]
        else:
            K_hist.append(None)                     # NEW
            P_upd_hist.append(None)          
            nis_log.append(np.nan); nees_log.append(np.nan)
            traceP_upd.append(np.nan)

        # state log
        pos_kf[k]=kf.x[:3]; eul_kf[k]=quat2euler(kf.x[6:10])
        ang_err_log.append(quat_angle(kf.x[6:10], q_gt[k]))

    k_arr=np.arange(n)

    # CSV dump ------------------------------------------------------
    hdr=["k","nis","nees","trP_pred","trP_upd",
         "varPx_pred","varPy_pred","varPz_pred",
         "varPx_upd","varPy_upd","varPz_upd",
         "Kx","Ky","Kz","pos_err","ang_err"]
    rows=[",".join(hdr)]
    for k in range(n):
        rows.append(",".join(map(str,[
            k, nis_log[k], nees_log[k], traceP_pred[k], traceP_upd[k],
            *var_pred_pos[k], *var_upd_pos[k],
            *K_pos_log[k], np.linalg.norm(pos_kf[k]-pos_gt[k]), ang_err_log[k]
        ])))
    Path(args.out_csv).write_text("\n".join(rows))
    print("⇒ CSV log saved to", Path(args.out_csv).resolve())

    # ---------- PLOTS ---------- -----------------------------------
    plt.figure(figsize=(12,4))
    valid = np.isfinite(nis_log)
    plt.subplot(131); plt.title("NIS")
    plt.plot(k_arr[valid], np.array(nis_log)[valid],'bo-',ms=3,lw=.8)
    plt.hlines(chi2.ppf([0.05,0.95],7),0,n-1,'r','dashed'); plt.xlabel('step')
    valid = np.isfinite(nees_log)
    plt.subplot(132); plt.title("NEES")
    plt.plot(k_arr[valid], np.array(nees_log)[valid],'go-',ms=3,lw=.8)
    plt.hlines(chi2.ppf([0.05,0.95],kf.n_states),0,n-1,'r','dashed'); plt.xlabel('step')
    plt.subplot(133); plt.title("trace(P)")
    plt.plot(k_arr, traceP_pred,'k-',label='pred')
    plt.plot(k_arr, traceP_upd ,'b--',label='upd'); plt.xlabel('step'); plt.legend()
    plt.tight_layout(); plt.show()

    # Position variance
    figv,axv=plt.subplots(3,1,figsize=(9,6),sharex=True)
    for i,lbl in enumerate(['σ²x','σ²y','σ²z']):
        axv[i].plot(k_arr, var_pred_pos[:,i],'k-',label='pred')
        axv[i].plot(k_arr, var_upd_pos[:,i],'b--',label='upd')
        axv[i].set_ylabel(lbl); axv[i].grid(ls=':'); 
        if i==0: axv[i].legend()
    axv[-1].set_xlabel('step'); figv.suptitle('Position variances'); plt.tight_layout(); plt.show()

    # ── ALL state variances diag(P) ───────────────────────────────────
    state_labels = ['px','py','pz','vx','vy','vz',
                    'qx','qy','qz','qw','wx','wy','wz']
    fig_all, axes = plt.subplots(4,4, figsize=(12,9), sharex=True)
    for i in range(13):
        r,c = divmod(i,4)
        ax = axes[r,c]
        ax.plot(k_arr, [P[i,i] for P in P_pred_hist], 'k-', label='pred')
        ax.plot(k_arr, [P_upd_hist[k][i,i] if P_upd_hist[k] is not None else np.nan
                        for k in range(n)], 'b--', label='upd')
        ax.set_ylabel(state_labels[i]); ax.grid(ls=':')
        if i==0: ax.legend()
    # hide the empty subplot (row=3,col=1)
    axes[3][1].set_visible(False); axes[3][2].set_visible(False); axes[3][3].set_visible(False)
    axes[-1][0].set_xlabel('step')
    fig_all.suptitle('All state variances diag(P)'); plt.tight_layout(); plt.show()


    # ── Kalman gain (rows: px,vx,qw vs Δpx) ───────────────────────────
    figK, axK = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    gain_rows = [0, 3, 9]                  # px ,  vx ,  qw
    labelsK   = ['K_px,Δpx', 'K_vx,Δpx', 'K_qw,Δpx']

    for j, row in enumerate(gain_rows):
        # build 1‑D array with NaNs on non‑update steps
        vals = np.array([
            K_hist[k][row, 0] if K_hist[k] is not None else np.nan
            for k in range(n)
        ])

        mask = np.isfinite(vals)           # keep only real numbers
        axK[j].plot(k_arr[mask], vals[mask],
                    'm-',  lw=1.5, marker='o', ms=4)
        axK[j].set_ylabel(labelsK[j])
        axK[j].grid(ls=':')

    axK[-1].set_xlabel('step')
    figK.suptitle('Selected Kalman‑gain entries')
    plt.tight_layout(); plt.show()


    # # HEATMAP
    # gain_mat = np.full((n, 13*7), np.nan)
    # for k in range(n):
    #     if K_hist[k] is not None:
    #         gain_mat[k, :] = K_hist[k].flatten()

    # plt.figure(figsize=(10,5))
    # plt.imshow(gain_mat, aspect='auto', origin='lower',
    #         interpolation='nearest', cmap='viridis')
    # plt.colorbar(label='Kalman gain value')
    # plt.title('Kalman gain, flattened (row‑major)')
    # plt.xlabel('row*7 + col'); plt.ylabel('step k')
    # plt.tight_layout(); plt.show()


    # ── position components ──────────────────────────────────────────
    fig_pos, ax_pos = plt.subplots(3,1, figsize=(10,7), sharex=True)
    for i,lbl in enumerate(['x','y','z']):
        ax_pos[i].plot(k_arr, pos_gt[:,i],   'k-',  label='truth')
        ax_pos[i].plot(k_arr, pos_meas[:,i], 'r.',  ms=3, alpha=.6, label='meas')
        ax_pos[i].plot(k_arr, pos_kf[:,i],   'b--', label='KF')
        ax_pos[i].set_ylabel(lbl+' (m)'); ax_pos[i].grid(ls=':')
        if i==0: ax_pos[i].legend(ncol=3, fontsize=9)
    ax_pos[-1].set_xlabel('step'); fig_pos.suptitle('Position'); plt.tight_layout(); plt.show()

    # ── orientation (roll‑pitch‑yaw) ─────────────────────────────────
    fig_eul, ax_eul = plt.subplots(3,1, figsize=(10,7), sharex=True)
    for i,lbl in enumerate(['roll','pitch','yaw']):
        ax_eul[i].plot(k_arr, np.rad2deg(eul_gt[:,i]),   'k-',  label='truth')
        ax_eul[i].plot(k_arr, np.rad2deg(eul_meas[:,i]), 'r.',  ms=3, alpha=.6, label='meas')
        ax_eul[i].plot(k_arr, np.rad2deg(eul_kf[:,i]),   'b--', label='KF')
        ax_eul[i].set_ylabel(lbl+' (deg)'); ax_eul[i].grid(ls=':')
        if i==0: ax_eul[i].legend(ncol=3, fontsize=9)
    ax_eul[-1].set_xlabel('step'); fig_eul.suptitle('Orientation (RPY)')
    plt.tight_layout(); plt.show()

    # ── orientation error ────────────────────────────────────────────
    plt.figure(figsize=(8,3))
    plt.plot(k_arr, ang_err_log, 'g-')
    plt.title('KF orientation error')
    plt.ylabel('angle (deg)'); plt.xlabel('step'); plt.grid(ls=':')
    plt.tight_layout(); plt.show()

    # ── 3‑D trajectory ───────────────────────────────────────────────
    fig3d = plt.figure(figsize=(6,5))
    ax3 = fig3d.add_subplot(111, projection='3d')
    ax3.plot(pos_gt[:,0], pos_gt[:,1], pos_gt[:,2], 'k-', lw=2, label='truth')
    ax3.scatter(pos_meas[:,0], pos_meas[:,1], pos_meas[:,2], c='r', s=10,
                alpha=.4, label='meas')
    ax3.plot(pos_kf[:,0], pos_kf[:,1], pos_kf[:,2], 'b--', lw=1.5, label='KF')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.set_title('3‑D trajectory'); ax3.legend()
    plt.tight_layout(); plt.show()

# ─────────────────────────────── CLI ──────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--jac", action="store_true", help="Jacobian check only")
    ap.add_argument("--dt", type=float, default=1/30)
    ap.add_argument("--time", type=float, default=10.0)
    ap.add_argument("--vel", nargs=3, type=float, default=[0.2,-0.1,0.0])
    ap.add_argument("--rate",nargs=3,type=float,default=[2.0,0.0,5.0])
    ap.add_argument("--noise_pos",  type=float, default=0.003)
    ap.add_argument("--noise_quat", type=float, default=np.deg2rad(0.15))
    ap.add_argument("--out_csv", default="kf_debug_log.csv")
    ap.add_argument("--update-mod",   type=int, default=1)
    ap.add_argument("--update-phase", type=int, default=0)
    args = ap.parse_args()
    if args.jac:
        check_jacobians(MultExtendedKalmanFilter(args.dt)); return
    args.update_phase %= max(1,args.update_mod)
    run_sim(args)

if __name__ == "__main__":
    main()