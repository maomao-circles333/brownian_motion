import numpy as np
import matplotlib.pyplot as plt

# TODO 1: plot final positions across different runs. ✅
# TODO 2: compare this to the case of using extrinsic mean, for the same parameters. ✅

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
                             init_seed=30, noise_seed_base=1000):
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
        for k in range(steps):
            # Dynamics (deterministic part)
            dx_det = dynamics(x, b, Q, K, V_mat)

            # Update intrinsic mean a little each step (dt small)
            u_mean = intrinsic_mean_refine(u_mean, x, iters=mean_refine_steps, step=1.0)

            # Per-particle noise coefficient: sigma0 * ||x_i - u_mean||_2
            # TODO 2: change l^2 norm to intrinsic norm  ✅
            #amp = sigma0 * np.linalg.norm(x - u_mean[None, :], axis=1, keepdims=True)  # (n,1)
            
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

    # Average trajectory across runs (raw mean, NOT projected to the sphere)
    mean_traj = all_traj.mean(axis=0)  # (steps+1, n, d)

    final_positions = all_traj[:, -1, :, :]  # (runs, n, d)

    return mean_traj, final_positions

def euclidean_distance(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.linalg.norm(x - y)

# ----------  Convergence check & plots ----------
if __name__ == "__main__":
    n, d = 3, 3
    T, dt = 10.0, 0.005
    b, sigma0 = 1.0, 0.2
    runs = 20

    Q = np.eye(d); K = np.eye(d); V_mat = np.eye(d)

    mean_traj, final_positions = simulate_intrinsic_noise(
        n, d, T, dt, b, Q, K, V_mat, sigma0,
        runs=runs, mean_refine_steps=2,
        init_seed=31, noise_seed_base=1000
    )

    steps = mean_traj.shape[0]
    times = np.linspace(0, T, steps)

    # Convergence on mean (max pairwise distance among agent means)
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
        print("Consensus mean vector (unprojected):", center_raw)
    else:
        print("System did not converge.")

    # Plot max pairwise distance of the RAW mean
    plt.figure(figsize=(7,4))
    plt.plot(times, delta, lw=1.5, label="Max pairwise distance (unprojected mean)")
    plt.axhline(threshold, color='red', ls='--', label="Threshold 1e-2")
    plt.yscale('log')
    plt.xlabel("Time"); plt.ylabel("Max pairwise distance")
    plt.title("Convergence of Averaged (Unprojected) Trajectories")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# 3D plot of averaged trajectories 
    if d == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')

        # Optional: sphere for reference
        u = np.linspace(0, np.pi, 30); v = np.linspace(0, 2*np.pi, 30)
        xu = np.outer(np.sin(u), np.cos(v))
        yu = np.outer(np.sin(u), np.sin(v))
        zu = np.outer(np.cos(u), np.ones_like(v))
        ax.plot_wireframe(xu, yu, zu, color='lightgray', alpha=0.5, linewidth=0.5)

        for i in range(n):
            pts = mean_traj[:, i]  # unprojected
            ax.plot(pts[:,0], pts[:,1], pts[:,2], lw=1.2)

        ax.scatter(mean_traj[0,:,0],  mean_traj[0,:,1],  mean_traj[0,:,2],
                   color='blue', s=25, label='Mean start (unprojected)')
        ax.scatter(mean_traj[-1,:,0], mean_traj[-1,:,1], mean_traj[-1,:,2],
                   color='red', s=25, label='Mean end (unprojected)')
        ax.set_box_aspect([1,1,1])
        ax.set_title("Averaged Trajectories (unprojected mean)")
        ax.legend(); plt.tight_layout(); plt.show()


# ===== Final positions (one particle across runs): full sphere + zoom
# ===== unprojected mean across runs (red) =====
if d == 3:
    particle_idx = 0
    pts = final_positions[:, particle_idx, :]  # (runs, 3)

    # Unprojected Euclidean mean (may lie inside the sphere)
    raw_mean = pts.mean(axis=0)

    fig = plt.figure(figsize=(12, 6))
    for j, zoom in enumerate([False, True]):
        ax = fig.add_subplot(1, 2, j+1, projection='3d')

        # Sphere reference 
        u = np.linspace(0, np.pi, 60)
        v = np.linspace(0, 2*np.pi, 60)
        xs = np.outer(np.sin(u), np.cos(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.cos(u), np.ones_like(v))
        ax.plot_surface(xs, ys, zs, rstride=2, cstride=2, linewidth=0,
                        alpha=0.08, shade=False, color='gray', zorder=0)
        ax.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.25, linewidth=0.4, zorder=0)

        # Final positions cluster
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=32, alpha=0.35, depthshade=False,
                   label=f'Final positions (particle {particle_idx})', zorder=3)

        # mean
        ax.scatter([raw_mean[0]], [raw_mean[1]], [raw_mean[2]],
                   marker='x', s=120, color='red', label='Mean (unprojected)',
                   depthshade=False, zorder=9)

        ax.set_box_aspect([1, 1, 1])
        ax.set_proj_type('ortho')
        ax.view_init(elev=22, azim=35)

        if zoom:
            # Auto-zoom to the data cloud with padding
            mins = pts.min(axis=0); maxs = pts.max(axis=0)
            center = 0.5 * (mins + maxs)
            r = 0.5 * np.max(maxs - mins)
            pad = 1.10
            r = max(float(r) * pad, 0.05)  # if cluster is very tight
            ax.set_xlim(center[0] - r, center[0] + r)
            ax.set_ylim(center[1] - r, center[1] + r)
            ax.set_zlim(center[2] - r, center[2] + r)
            ax.set_title("Final positions zoomed")
        else:
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
            ax.set_title("Final positions — full sphere reference")

        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
