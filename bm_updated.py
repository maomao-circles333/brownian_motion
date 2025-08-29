import numpy as np
import matplotlib.pyplot as plt
import os

# TODO 1: plot final positions across different runs. ✅
# TODO 2: compare this to the case of using extrinsic mean, for the same parameters. ✅
# TODO 3: Added plotting code for d>3 ✅
init_seed = 30
def ensure_3d(Z):
    """
    Z: (m, q) with q<=3  -> (m,3) by zero-padding extra columns.
    """
    Z = np.asarray(Z, float)
    m, q = Z.shape if Z.ndim == 2 else (1, Z.size)
    if q == 3:
        return Z
    if q == 2:
        return np.c_[Z, np.zeros((m,1))]
    if q == 1:
        return np.c_[Z, np.zeros((m,2))]
    # q == 0 (degenerate): return all zeros
    return np.zeros((m,3), dtype=float)

# ---------- Sphere normalization ----------
def normalize_rows(X, eps=1e-12):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, eps)

def exp_map_sphere(x, v, eps=1e-12):
    nv = np.linalg.norm(v)
    if nv < eps:
        y = x + v
        return y / np.linalg.norm(y)
    return np.cos(nv)*x + np.sin(nv)*(v/nv)

def log_map_sphere(x, y, eps=1e-12):
    # x,y unit; returns tangent vector at x pointing to y
    c = float(np.clip(np.dot(x, y), -1.0, 1.0))
    th = np.arccos(c)
    if th < eps:
        return np.zeros_like(x)
    u = y - c*x
    su = np.linalg.norm(u)
    if su < eps:
        return np.zeros_like(x)   # avoid NaN near antipodal
    return (th/su) * u

def log_map_sphere_batch(u, X, eps=1e-12):
    """
    u: (d,), X: (n,d)
    returns (n,d) tangent rows at u
    """
    c = np.clip(X @ u, -1.0, 1.0)            # (n,)
    th = np.arccos(c)                        # (n,)
    Uperp = X - c[:, None] * u[None, :]      # (n,d)
    su = np.linalg.norm(Uperp, axis=1, keepdims=True)  # (n,1)
    fac = th[:, None] / np.maximum(su, eps)  # (n,1)
    V = fac * Uperp                          # (n,d)
    V[th < eps] = 0.0
    return V
    # the sum of the rows of V is the gradient vector

def intrinsic_mean_refine(u_prev, points, iters=2, step=1.0):
    u = u_prev.copy()
    for _ in range(iters):
        V = log_map_sphere_batch(u, points)   # (n,d)
        v = V.mean(axis=0)                    # (d,)
        if np.linalg.norm(v) < 1e-12:
            break
        u = exp_map_sphere(u, step*v)
    return u

def intrinsic_mean_Sd(points, tol=1e-10, max_iter=50):
    # warm start: normalized extrinsic mean
    u = normalize_rows(points.mean(axis=0, keepdims=True))[0]
    for _ in range(max_iter):
        V = log_map_sphere_batch(u, points)
        v = V.mean(axis=0)
        if np.linalg.norm(v) <= tol:
            break
        u = exp_map_sphere(u, v)
    return u

# ---------- PCA helpers (for plotting when d>3) ----------
def pca_fit(X, k=3):
    """
    X: (m,d) data matrix. Returns (mu, W) where
      - mu: (d,) mean
      - W:  (d,k) top-k principal directions (columns, orthonormal)
    """
    X = np.asarray(X, float)
    mu = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    W = Vt[:k].T  # (d,k)
    return mu, W

def pca_transform(Y, mu, W):
    """Project rows of Y onto PCA axes: (Y - mu) @ W  -> (m,k)."""
    return (Y - mu) @ W

# ------ Dynamics on sphere --------
def dynamics(x, b=1.0, Q=None, K=None, V_mat=None):
    n, d = x.shape
    Q = np.eye(d) if Q is None else Q
    K = np.eye(d) if K is None else K
    V_mat = np.eye(d) if V_mat is None else V_mat

    q_all = (Q @ x.T).T          # (n,d)
    k_all = (K @ x.T).T          # (n,d)
    v_all = (V_mat @ x.T).T      # (n,d)

    inner = q_all @ k_all.T      # (n,n)
    w = np.exp(b * inner)
    w /= w.sum(axis=1, keepdims=True)
    V = w @ v_all                # (n,d)

    # project onto tangent at each x[i]
    return V - (np.sum(V * x, axis=1, keepdims=True)) * x

# ---------- Simulation with intrinsic-mean-modulated noise ----------
def simulate_intrinsic_noise(n, d, T, dt, b, Q, K, V_mat, sigma0,
                             runs=100, mean_refine_steps=2,
                             init_seed=init_seed, noise_seed_base=1000,
                             store_stride=1):  
    steps = int(T/dt)

    # FIXED: number of stored frames (always include first)
    kept = steps // store_stride + 1
    all_traj = np.zeros((runs, kept, n, d), dtype=float) 

    # Fixed initialization across runs
    rng_init = np.random.RandomState(init_seed)
    x0 = rng_init.randn(n, d).astype(float)
    x0 = normalize_rows(x0)

    for r in range(runs):
        rng = np.random.RandomState(noise_seed_base + r)
        x = x0.copy()
        traj = np.zeros((kept, n, d), dtype=float)  # CHANGED: kept instead of steps+1
        traj[0] = x
        frame = 1  

        u_mean = intrinsic_mean_Sd(x, tol=1e-12, max_iter=100)
        print(u_mean)
        for k in range(steps):
            # Dynamics (deterministic part)
            dx_det = dynamics(x, b, Q, K, V_mat)

            # Update intrinsic mean a little each step (dt small)
            u_mean = intrinsic_mean_refine(u_mean, x, iters=mean_refine_steps, step=1.0)

            # Intrinsic distance (geodesic) on the unit sphere
            c = np.clip(np.sum(x * u_mean[None, :], axis=1, keepdims=True), -1.0, 1.0)  # (n,1)
            theta = np.arccos(c)                                                         # (n,1)
            #amp = sigma0 * theta
            amp = sigma0 * np.linalg.norm(x - u_mean[None, :], axis=1, keepdims=True)  # (n,1)
            

            # Tangent Gaussian increment
            rnd = rng.randn(n, d)
            noise_tan = rnd - (np.sum(rnd * x, axis=1, keepdims=True)) * x

            # Euler–Maruyama + reproject to sphere
            x = x + dt * dx_det + np.sqrt(dt) * (amp * noise_tan)
            x = normalize_rows(x)

            # FIXED: store only every store_stride-th step
            if ((k + 1) % store_stride == 0) or (k == steps - 1):
                if frame < kept:
                    traj[frame] = x
                    frame += 1

        all_traj[r] = traj

    # Average trajectory over runs  (replaced by intrinsic mean over runs)
    # mean_traj = all_traj.mean(axis=0)  # (kept, n, d)

    # intrinsic mean over runs (per time, per agent) with warm start + incremental refine
    # a final stricter refine on the last frame
    mean_traj = np.zeros((kept, n, d), dtype=float)
    # warm start at t=0 for each agent
    for i in range(n):
        pts0 = all_traj[:, 0, i, :]                       # (runs, d)
        mean_traj[0, i] = intrinsic_mean_Sd(pts0, tol=1e-10, max_iter=50)
    # incremental refine across stored frames
    for t in range(1, kept):
        for i in range(n):
            pts_t = all_traj[:, t, i, :]                  # (runs, d)
            u_prev = mean_traj[t-1, i]
            mean_traj[t, i] = intrinsic_mean_refine(u_prev, pts_t, iters=mean_refine_steps, step=1.0)
    # final stricter refine on the last stored frame
    for i in range(n):
        pts_last = all_traj[:, -1, i, :]
        mean_traj[-1, i] = intrinsic_mean_Sd(pts_last, tol=1e-12, max_iter=200)

    final_positions = all_traj[:, -1, :, :]  # (runs, n, d)

    return mean_traj, final_positions

def euclidean_distance(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.linalg.norm(x - y)

# ----------  Convergence check & plots ----------
if __name__ == "__main__":
    n, d = 10, 3
    T, dt = 600.0, 0.005
    # b, sigma0 = 1.0, 0.2
    b, sigma0 = 6.0, 0.05
    runs = 10
    
    init_seed = 30
    outdir = "plots"  
    os.makedirs(outdir, exist_ok=True)

    Q = np.eye(d); K = np.eye(d); V_mat = np.eye(d)

    mean_traj, final_positions = simulate_intrinsic_noise(
        n, d, T, dt, b, Q, K, V_mat, sigma0,
        runs=runs, mean_refine_steps=2,
        init_seed=init_seed, noise_seed_base=1000,
        store_stride=100  # keep every 100th frame 
    )

    steps = mean_traj.shape[0]
    times = np.linspace(0, T, steps)

    # Convergence on mean 
    delta = np.zeros(steps)
    for k in range(steps):
        P = mean_traj[k]                     # (n,d)
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        delta[k] = D.max()

    threshold = 1e-2
    if delta[-1] < threshold:
        final_mean = mean_traj[-1]           # (n,d)
        center_raw = final_mean.mean(axis=0) # no projection
        norm = np.linalg.norm(center_raw)
        if norm > 0:
            consensus_point = center_raw / norm  # on S^{d-1}
        else:
            consensus_point = center_raw
        print("Converged.")
        print("Consensus mean vector:", center_raw)
    else:
        print("System did not converge.")

    # Plot max pairwise distance of the mean
    fig1 = plt.figure(figsize=(7,4))
    plt.plot(times, delta, lw=1.5, label="Max pairwise distance (mean)")
    plt.axhline(threshold, color='red', ls='--', label="Threshold 1e-2")
    plt.yscale('log')
    plt.xlabel("Time"); plt.ylabel("Max pairwise distance")
    plt.title("Convergence of Averaged Trajectories")
    plt.grid(True); plt.legend(); plt.tight_layout(); 
    fig1.savefig(os.path.join(outdir, f"convergence_seed{init_seed}_n=10_d=3_smallsigma.png"),
                 dpi=200, bbox_inches="tight")
    #fig1.savefig("test0.png", dpi=200, bbox_inches="tight")
    plt.show()
    
    # 3D plot of averaged trajectories
    if d >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        # PCA
        if d == 3:
            mu_mt, W_mt = None, None
        else:
            all_pts = mean_traj.reshape(-1, d)           # ((steps)*n, d)
            mu_mt, W_mt = pca_fit(all_pts, k=3)          # (d,), (d,3)

        fig2 = plt.figure(figsize=(6,6))
        ax = fig2.add_subplot(projection='3d')

        # sphere for reference
        u = np.linspace(0, np.pi, 30); v = np.linspace(0, 2*np.pi, 30)
        xu = np.outer(np.sin(u), np.cos(v))
        yu = np.outer(np.sin(u), np.sin(v))
        zu = np.outer(np.cos(u), np.ones_like(v))
        ax.plot_wireframe(xu, yu, zu, color='lightgray', alpha=0.5, linewidth=0.5)

        # Plot each agent's averaged trajectory
        for i in range(n):
            if d == 3:
                pts3 = mean_traj[:, i]  # (steps, 3)
                ax.plot(pts3[:,0], pts3[:,1], pts3[:,2], lw=1.2)
            else:
                xs, ys, zs = [], [], []
                for kidx in range(steps):
                    P = mean_traj[kidx]                        # (n,d)
                    P3 = pca_transform(P, mu_mt, W_mt)        # (n,3)
                    xs.append(P3[i,0]); ys.append(P3[i,1]); zs.append(P3[i,2])
                ax.plot(xs, ys, zs, lw=1.2)

        # Start/end scatters
        if d == 3:
            P0, Pend = mean_traj[0], mean_traj[-1]             # (n,3)
        else:
            P0  = pca_transform(mean_traj[0],  mu_mt, W_mt)    # (n,3)
            Pend= pca_transform(mean_traj[-1], mu_mt, W_mt)    # (n,3)

        ax.scatter(P0[:,0],  P0[:,1],  P0[:,2],
                   color='blue', s=25, label='Mean start (PCA)')
        ax.scatter(Pend[:,0], Pend[:,1], Pend[:,2],
                   color='red',  s=25, label='Mean end (PCA)')

        ax.set_box_aspect([1,1,1])
        ax.set_title("Averaged Trajectories (PCA if d>3)")
        ax.legend(); plt.tight_layout()
        fig2.savefig(os.path.join(outdir, f"avg_traj_3d_seed{init_seed}_n=10_d=3_smallsigma.png"),
                     dpi=200, bbox_inches="tight")
        plt.show()
    # if d > 3 use PCA
    if d >= 3:
        particle_idx = 0
        pts = final_positions[:, particle_idx, :]  # (runs, d)

        if d == 3:
            pts3 = pts
            raw_mean = pts3.mean(axis=0)
        else:
            mu_fp, W_fp = pca_fit(pts, k=3)                        # W_fp: (d, k_eff<=3)
            pts_low = pca_transform(pts, mu_fp, W_fp)              # (runs, k_eff)
            pts3 = ensure_3d(pts_low)                              # (runs, 3)

            raw_mean_full = pts.mean(axis=0, keepdims=True)        # (1, d)
            raw_mean_low  = pca_transform(raw_mean_full, mu_fp, W_fp)  # (1, k_eff)
            raw_mean = ensure_3d(raw_mean_low)[0]                  # (3,)

        fig3 = plt.figure(figsize=(12, 6))
        for j, zoom in enumerate([False, True]):
            ax = fig3.add_subplot(1, 2, j+1, projection='3d')

            ax.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2],
                       s=32, alpha=0.35, depthshade=False,
                       label=f'Final positions (particle {particle_idx})', zorder=3)

            ax.scatter([raw_mean[0]], [raw_mean[1]], [raw_mean[2]],
                       marker='x', s=120, color='red', label='Mean (PCA)',
                       depthshade=False, zorder=9)

            ax.set_box_aspect([1, 1, 1])
            ax.set_proj_type('ortho')
            ax.view_init(elev=22, azim=35)

            if zoom:
                mins = pts3.min(axis=0); maxs = pts3.max(axis=0)
                center = 0.5 * (mins + maxs)
                r = 0.5 * np.max(maxs - mins)
                pad = 1.10
                r = max(float(r) * pad, 0.05)  # if cluster is very tight
                ax.set_xlim(center[0] - r, center[0] + r)
                ax.set_ylim(center[1] - r, center[1] + r)
                ax.set_zlim(center[2] - r, center[2] + r)
                ax.set_title("Final positions — zoomed (PCA if d>3)")
            else:
                ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
                ax.set_title("Final positions — full sphere reference (PCA if d>3)")

            ax.legend(loc='upper left')

        plt.tight_layout()
        fig3.savefig(os.path.join(outdir, f"final_positions_particle{particle_idx}_seed{init_seed}_n=10_d=3_smallsigma.png"),
                     dpi=200, bbox_inches="tight")
        plt.show()
