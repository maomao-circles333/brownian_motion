import numpy as np
import matplotlib.pyplot as plt
import os
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

# adaptive
def simulate_intrinsic_noise(n, d, T, dt, b, Q, K, V_mat, sigma0,
                             runs=100, mean_refine_steps=2,
                             init_seed=init_seed, noise_seed_base=1000):
    steps = int(T/dt)

    # Fixed initialization across runs
    rng_init = np.random.RandomState(init_seed)
    x0 = rng_init.randn(n, d).astype(float)
    x0 = normalize_rows(x0)

    all_traj = np.zeros((runs, steps+1, n, d), dtype=float)

    for r in range(runs):
        rng = np.random.RandomState(noise_seed_base + r)
        x = x0.copy()
        traj = np.zeros((steps+1, n, d), dtype=float)
        traj[0] = x

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
            amp = sigma0 * theta

            # Tangent Gaussian increment
            rnd = rng.randn(n, d)
            noise_tan = rnd - (np.sum(rnd * x, axis=1, keepdims=True)) * x

            # Euler–Maruyama + reproject to sphere
            x = x + dt * dx_det + np.sqrt(dt) * (amp * noise_tan)
            x = normalize_rows(x)

            traj[k+1] = x

        all_traj[r] = traj

    # Average trajectory across runs 
    mean_traj = all_traj.mean(axis=0)  # (steps+1, n, d)
    final_positions = all_traj[:, -1, :, :]  # (runs, n, d)
    return mean_traj, final_positions

# homogeneous
def simulate_homogeneous_noise(n, d, T, dt, b, Q, K, V_mat, sigma0,
                               runs=100, mean_refine_steps=2,
                               init_seed=init_seed, noise_seed_base=1000):
    steps = int(T/dt)

    # Same fixed initialization across runs as adaptive version
    rng_init = np.random.RandomState(init_seed)
    x0 = rng_init.randn(n, d).astype(float)
    x0 = normalize_rows(x0)

    all_traj = np.zeros((runs, steps+1, n, d), dtype=float)

    for r in range(runs):
        rng = np.random.RandomState(noise_seed_base + r)
        x = x0.copy()
        traj = np.zeros((steps+1, n, d), dtype=float)
        traj[0] = x

        u_mean = intrinsic_mean_Sd(x, tol=1e-12, max_iter=100)
        for k in range(steps):
            dx_det = dynamics(x, b, Q, K, V_mat)
            u_mean = intrinsic_mean_refine(u_mean, x, iters=mean_refine_steps, step=1.0)

            # Homogeneous sigma0 (Euler–Maruyama)
            amp = sigma0  # scalar

            rnd = rng.randn(n, d)
            noise_tan = rnd - (np.sum(rnd * x, axis=1, keepdims=True)) * x

            x = x + dt * dx_det + np.sqrt(dt) * (amp * noise_tan)
            x = normalize_rows(x)

            traj[k+1] = x

        all_traj[r] = traj

    mean_traj = all_traj.mean(axis=0)         # (steps+1, n, d)
    final_positions = all_traj[:, -1, :, :]   # (runs, n, d)
    return mean_traj, final_positions

def euclidean_distance(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.linalg.norm(x - y)
# Plots
if __name__ == "__main__":
    n, d = 3, 3
    T, dt = 200.0, 0.005
    b, sigma0 = 1.0, 0.1
    runs = 20
    
    init_seed = 30
    outdir = "plots"  
    os.makedirs(outdir, exist_ok=True)

    Q = np.eye(d); K = np.eye(d); V_mat = np.eye(d)


    mean_traj_A, final_positions_A = simulate_intrinsic_noise(
        n, d, T, dt, b, Q, K, V_mat, sigma0,
        runs=runs, mean_refine_steps=2,
        init_seed=init_seed, noise_seed_base=1000
    )
    mean_traj_B, final_positions_B = simulate_homogeneous_noise(
        n, d, T, dt, b, Q, K, V_mat, sigma0,
        runs=runs, mean_refine_steps=2,
        init_seed=init_seed, noise_seed_base=1000
    )

    n_frames = mean_traj_A.shape[0]
    times = np.linspace(0, T, n_frames)
    # plot 1: convergence plot
    delta_A = np.zeros(n_frames)
    delta_B = np.zeros(n_frames)
    for k in range(n_frames):
        PA = mean_traj_A[k]                     # (n,d)
        PB = mean_traj_B[k]
        DA = np.linalg.norm(PA[:, None, :] - PA[None, :, :], axis=2)
        DB = np.linalg.norm(PB[:, None, :] - PB[None, :, :], axis=2)
        delta_A[k] = DA.max()
        delta_B[k] = DB.max()

    threshold = 1e-2
    fig1, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    # Left: intrinsic-adaptive
    axs[0].plot(times, delta_A, lw=1.5, label="Max pairwise distance (unprojected mean)")
    axs[0].axhline(threshold, color='red', ls='--', label="Threshold 1e-2")
    axs[0].set_yscale('log'); axs[0].grid(True)
    axs[0].set_xlabel("Time"); axs[0].set_ylabel("Max pairwise distance")
    axs[0].set_title("Convergence — Intrinsic-adaptive")
    axs[0].legend()
    # Right: homogeneous
    axs[1].plot(times, delta_B, lw=1.5, label="Max pairwise distance (unprojected mean)")
    axs[1].axhline(threshold, color='red', ls='--', label="Threshold 1e-2")
    axs[1].set_yscale('log'); axs[1].grid(True)
    axs[1].set_xlabel("Time")
    axs[1].set_title("Convergence — Homogeneous")
    axs[1].legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(outdir, f"convergence_compare_seed{init_seed}_metastable.png"),
                 dpi=200, bbox_inches="tight")
    plt.show()

    # plot 2: 3D averaged trajectories 
    if d == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig2 = plt.figure(figsize=(12, 6))
        axL = fig2.add_subplot(1, 2, 1, projection='3d')
        axR = fig2.add_subplot(1, 2, 2, projection='3d')

        # Helper to draw a reference sphere
        def draw_sphere(ax):
            u = np.linspace(0, np.pi, 30); v = np.linspace(0, 2*np.pi, 30)
            xu = np.outer(np.sin(u), np.cos(v))
            yu = np.outer(np.sin(u), np.sin(v))
            zu = np.outer(np.cos(u), np.ones_like(v))
            ax.plot_wireframe(xu, yu, zu, color='lightgray', alpha=0.5, linewidth=0.5)

        # Left: Intrinsic-adaptive
        draw_sphere(axL)
        for i in range(n):
            pts = mean_traj_A[:, i]
            axL.plot(pts[:,0], pts[:,1], pts[:,2], lw=1.2)
        axL.scatter(mean_traj_A[0,:,0],  mean_traj_A[0,:,1],  mean_traj_A[0,:,2],
                    color='blue', s=25, label='Mean start (unprojected)')
        axL.scatter(mean_traj_A[-1,:,0], mean_traj_A[-1,:,1], mean_traj_A[-1,:,2],
                    color='red', s=25, label='Mean end (unprojected)')
        axL.set_box_aspect([1,1,1])
        axL.set_title("Averaged Trajectories — Intrinsic-adaptive")
        axL.legend()

        # Right: Homogeneous
        draw_sphere(axR)
        for i in range(n):
            pts = mean_traj_B[:, i]
            axR.plot(pts[:,0], pts[:,1], pts[:,2], lw=1.2)
        axR.scatter(mean_traj_B[0,:,0],  mean_traj_B[0,:,1],  mean_traj_B[0,:,2],
                    color='blue', s=25, label='Mean start (unprojected)')
        axR.scatter(mean_traj_B[-1,:,0], mean_traj_B[-1,:,1], mean_traj_B[-1,:,2],
                    color='red', s=25, label='Mean end (unprojected)')
        axR.set_box_aspect([1,1,1])
        axR.set_title("Averaged Trajectories — Homogeneous")
        axR.legend()

        plt.tight_layout()
        fig2.savefig(os.path.join(outdir, f"avg_traj_3d_compare_seed{init_seed}_metastable.png"),
                     dpi=200, bbox_inches="tight")
        plt.show()

        # final positions plot
        particle_idx = 0
        ptsA = final_positions_A[:, particle_idx, :]  # (runs, 3)
        ptsB = final_positions_B[:, particle_idx, :]  # (runs, 3)

        meanA = ptsA.mean(axis=0)
        meanB = ptsB.mean(axis=0)

        fig3 = plt.figure(figsize=(7, 6))
        ax = fig3.add_subplot(projection='3d')

        # Sphere reference
        u = np.linspace(0, np.pi, 60)
        v = np.linspace(0, 2*np.pi, 60)
        xs = np.outer(np.sin(u), np.cos(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.cos(u), np.ones_like(v))
        ax.plot_surface(xs, ys, zs, rstride=2, cstride=2, linewidth=0,
                        alpha=0.08, shade=False, color='gray', zorder=0)
        ax.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.25, linewidth=0.4, zorder=0)

        ax.scatter(ptsA[:, 0], ptsA[:, 1], ptsA[:, 2],
                s=32, alpha=0.35, depthshade=False,
                label='Final positions A (intrinsic-adaptive)', zorder=3, color='C0')
        ax.scatter(ptsB[:, 0], ptsB[:, 1], ptsB[:, 2],
                s=32, alpha=0.35, depthshade=False,
                label='Final positions B (homogeneous)', zorder=3, color='C1')

        # Means (unprojected)
        ax.scatter([meanA[0]], [meanA[1]], [meanA[2]],
                marker='x', s=120, color='C0', label='Mean A (unprojected)', depthshade=False, zorder=9)
        ax.scatter([meanB[0]], [meanB[1]], [meanB[2]],
                marker='x', s=120, color='C1', label='Mean B (unprojected)', depthshade=False, zorder=9)

        # Auto-zoom to include both clouds
        pts_all = np.vstack([ptsA, ptsB])
        mins = pts_all.min(axis=0); maxs = pts_all.max(axis=0)
        center = 0.5 * (mins + maxs)
        r = 0.5 * np.max(maxs - mins)
        r = max(float(r) * 1.10, 0.05)

        ax.set_xlim(center[0] - r, center[0] + r)
        ax.set_ylim(center[1] - r, center[1] + r)
        ax.set_zlim(center[2] - r, center[2] + r)

        ax.set_box_aspect([1, 1, 1])
        ax.set_proj_type('ortho')
        ax.view_init(elev=22, azim=35)
        ax.set_title("Final positions — zoomed (A vs B)")
        ax.legend(loc='upper left')

        plt.tight_layout()
        fig3.savefig(os.path.join(outdir, f"final_positions_particle{particle_idx}_compare_seed{init_seed}_metastable.png"),
                    dpi=200, bbox_inches="tight")
        plt.show()
