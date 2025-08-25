import numpy as np
import matplotlib.pyplot as plt
import numpy as np
#TODO 1: plot final positions across different runs.
#TODO 2: compare this to the case of using extrinsic mean, for the same parameters.
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
    # x,y unit; 
    # returns tangent vector at x pointing to y
    c = float(np.clip(np.dot(x, y), -1.0, 1.0))
    # for numerical stability
    th = np.arccos(c)
    if th < eps: 
        return np.zeros_like(x)
    u = y - c*x
    su = np.linalg.norm(u)
    if su < eps: 
        return np.zeros_like(x)   # avoid NaN near antipodal 
    return (th/su) * u

def log_map_sphere_batch(u, X, eps=1e-12):
    # u: (d,1),
    # X: (n,d)
    # returns (n,d) tangent rows at u
    c = np.clip(X @ u, -1.0, 1.0)            # shape (n,1)
    th = np.arccos(c)                        # shape (n,1)
    Uperp = X - c[:, None]*u[None, :]        # shape (n,d)
    su = np.linalg.norm(Uperp, axis=1, keepdims=True)
    fac = th[:, None] / np.maximum(su, eps)
    V = fac * Uperp
    # v_i = log_u (x_i)
    V[th < eps] = 0.0
    return V
    # the sum of the rows of V is the gradient vector

def intrinsic_mean_refine(u_prev, points, iters=2, step=1.0):
    u = u_prev.copy()
    for _ in range(iters):
        V = log_map_sphere_batch(u, points)   # (n,d)
        v = V.mean(axis=0)                    # (d,1)
        if np.linalg.norm(v) < 1e-12: break
        u = exp_map_sphere(u, step*v)
    return u

def intrinsic_mean_Sd(points, tol=1e-10, max_iter=50):
    # used only for the first step 
    # warm start
    u = normalize_rows(points.mean(axis=0, keepdims=True))[0]
    for _ in range(max_iter):
        V = log_map_sphere_batch(u, points)
        v = V.mean(axis=0)
        if np.linalg.norm(v) <= tol: break
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

        #u_mean = intrinsic_mean_Sd(x)  # warm start
        u_mean = intrinsic_mean_Sd(x, tol=1e-12, max_iter=100) 
        # print(u_mean)
        for k in range(steps):
            # Dynamics with no noise
            dx_det = dynamics(x, b, Q, K, V_mat)

            # Update intrinsic mean 
            # this works for small mean_refine_steps because dt is small
            u_mean = intrinsic_mean_refine(u_mean, x, iters=mean_refine_steps, step=1.0)

            # Per-particle noise coefficient: sigma0 * ||x_i - u_mean||_2
            # TODO 3: change l^2 norm to intrinsic norm
            amp = sigma0 * np.linalg.norm(x - u_mean[None, :], axis=1, keepdims=True)  # (n,1)

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
    return mean_traj


def euclidean_distance(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.linalg.norm(x - y)

# ----------  Convergence check ----------
if __name__ == "__main__":
    n, d = 3, 3
    T, dt = 10.0, 0.005
    b, sigma0 = 1.0, 0.2
    runs = 20

    Q = np.eye(d); K = np.eye(d); V_mat = np.eye(d)

    mean_traj = simulate_intrinsic_noise(n, d, T, dt, b, Q, K, V_mat, sigma0,
                                         runs=runs, mean_refine_steps=2,
                                         init_seed=31, noise_seed_base=1000)

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
            consensus_point = center_raw / norm  # on S^{d-1}, for reporting
        else:
            consensus_point = center_raw
        print("Converged.")
        print("Consensus mean vector (unprojected):", center_raw)
    else:
        print("System did not converge.")

    # Plot max pairwise distance of the RAW mean
    plt.figure(figsize=(7,4))
    plt.plot(times, delta, lw=1.5, label="Max pairwise distance (unprojected) mean)")
    plt.axhline(threshold, color='red', ls='--', label="Threshold 1e-2")
    plt.yscale('log')
    plt.xlabel("Time"); plt.ylabel("Max pairwise distance")
    plt.title("Convergence of Averaged (Unprojected) Trajectories")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    # 3D plot of RAW averaged trajectories (will lie inside the unit ball)
    if d == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')

        # Optional: sphere wireframe for reference
        u = np.linspace(0, np.pi, 30); v = np.linspace(0, 2*np.pi, 30)
        xu = np.outer(np.sin(u), np.cos(v))
        yu = np.outer(np.sin(u), np.sin(v))
        zu = np.outer(np.cos(u), np.ones_like(v))
        ax.plot_wireframe(xu, yu, zu, color='lightgray', alpha=0.5, linewidth=0.5)

        for i in range(n):
            pts = mean_traj[:, i]  # RAW
            ax.plot(pts[:,0], pts[:,1], pts[:,2], lw=1.2)

        ax.scatter(mean_traj[0,:,0],  mean_traj[0,:,1],  mean_traj[0,:,2],
                   color='blue', s=25, label='Mean start (unprojected)')
        ax.scatter(mean_traj[-1,:,0], mean_traj[-1,:,1], mean_traj[-1,:,2],
                   color='red', s=25, label='Mean end (unprojected)')
        ax.set_box_aspect([1,1,1])
        ax.set_title("Averaged Trajectories (unprojected mean)")
        ax.legend(); plt.show()
