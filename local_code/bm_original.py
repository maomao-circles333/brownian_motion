# Attention-style consensus dynamics on S^{d-1}, noise versus deterministic.
# Here in the noise calculation, we use amp to control the strength of the noise, and amp is calculated by d(xi, C) where C is the intrinsic mean (we use the same C across particles). 
#
# Plots:
#   (1) Convergence: max pairwise geodesic distance across agents vs time
#       — aggregated over runs (mean with 10–90% band) in *degrees*, log-scale in y-axis.
#   (2) Per-particle mean trajectories (mean over runs)
# Remarks: Need to tweak the TMAX deterministic run dependeing on params. Right now, in comparing the noisy vs. deterministic the deterministic consensus is computed with its own cutoff time.
# Diagnostic: prints the deviation of the trajectory from S^{d-1}: dev = abs(norm-1)
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  

# Platform (CPU)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")  

import jax
import jax.numpy as jnp
from jax import random, lax

N_AGENTS   = 32
DIM        = 3
BETA       = 1.0
DT         = 1e-3
TMAX       = 100.0
THRESHOLD  = 1e-3
SIGMA      = 0.1    
RUNS       = 100
STORE_STRIDE = 10
MEAN_UPDATE_STRIDE = 5    
MEAN_REFINE_STEPS  = 2    
INIT_SEED  = 201
NOISE_SEED = 999

# numpy helpers
def geodesic_angle_np(u, v):
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.arccos(c))

def norm_last(x, eps=1e-7):
    nrm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(nrm, eps)

def softmax_last(a):
    amax = jnp.max(a, axis=-1, keepdims=True)
    z = jnp.exp(a - amax)
    return z / jnp.sum(z, axis=-1, keepdims=True)

def dynamics_batched(X, b=1.0):
    # X: (R,n,d), Q=K=V=I_d
    inner = X @ jnp.swapaxes(X, -1, -2)  # (R,n,n)
    w = softmax_last(b * inner)          # (R,n,n)
    V = w @ X                            # (R,n,d)
    proj = jnp.sum(V * X, axis=-1, keepdims=True) * X
    return V - proj

def _log_map_batch_at_u(u, X):
    """
    u: (..., d)
    X: (..., n, d)  or  (..., d)   # both supported
    Returns: same leading shape as X (tangent vectors at u)
    """
    u_exp = jnp.expand_dims(u, axis=-2)          # (..., 1, d)
    c  = jnp.clip(jnp.sum(X * u_exp, axis=-1), -1.0, 1.0)    # (..., n) or (...)
    th = jnp.arccos(c)                                       # (..., n) or (...)
    Uperp = X - c[..., None] * u_exp                         # (..., n, d) or (..., d)
    su = jnp.linalg.norm(Uperp, axis=-1, keepdims=True)      # (..., n, 1) or (..., 1)
    fac = th[..., None] / jnp.maximum(su, 1e-7)              # (..., n, 1) or (..., 1)
    fac = jnp.where(th[..., None] < 1e-7, 0.0, fac)
    return fac * Uperp

def _exp_map_at_u(u, v):
    nv = jnp.linalg.norm(v)
    y = jnp.cos(nv)*u + jnp.sin(nv)*(v / jnp.maximum(nv, 1e-7))
    y_small = norm_last(u + v)
    return jnp.where(nv < 1e-7, y_small, y)

def intrinsic_mean_sd_jax(points, tol=1e-10, max_iter=100):
    """
    points: (m,d) on S^{d-1}
    Returns u: (d,)
    """
    u0 = norm_last(jnp.mean(points, axis=0))
    def cond_fun(state):
        u, g, k = state
        return jnp.logical_and(jnp.linalg.norm(g) > tol, k < max_iter)
    def body_fun(state):
        u, g, k = state
        V  = _log_map_batch_at_u(u, points)
        g2 = jnp.mean(V, axis=0)
        u2 = _exp_map_at_u(u, g2)
        return (u2, g2, k+1)
    g0 = jnp.mean(_log_map_batch_at_u(u0, points), axis=0)
    u_final, _, _ = lax.while_loop(cond_fun, body_fun, (u0, g0, 0))
    return norm_last(u_final)   # ensure exactly on S^{d-1}

# batched over leading axes: points: (B, m, d) -> (B, d)
intrinsic_mean_sd_batch = jax.jit(jax.vmap(intrinsic_mean_sd_jax, in_axes=(0, None, None)))

# ---------- compute C ----------------
@jax.jit
def compute_global_C_from_X(X, mean_refine_steps_unused):
    """
    X: (R, n, d)
    Returns: C: (R, d) — global intrinsic (Karcher) mean per run (same for all agents)
    """
    return intrinsic_mean_sd_batch(X, 1e-10, 100)  # (R,d)

# ---- Simulation (stores only at store_stride, but does not affect computation)
def simulate_intrinsic_noise_jax(n, d, T, dt, b, sigma0,
                                 runs=1, mean_refine_steps=1,
                                 init_key=None, noise_key=None,
                                 store_stride=100, mean_update_stride=5,
                                 same_init_across_runs=True):
    """
    Returns:
      traj_kept: (kept, runs, n, d)
      U_kept   : (kept, runs, d)  (intrinsic mean over agents per run at stored frames)
    """
    steps = int(float(T) / float(dt))
    kept  = steps // int(store_stride) + 1

    if init_key is None:  init_key  = random.PRNGKey(0)
    if noise_key is None: noise_key = random.PRNGKey(1)

    if same_init_across_runs:
        x0 = random.normal(init_key, (n, d), dtype=jnp.float32)
        X0 = norm_last(jnp.broadcast_to(x0, (runs, n, d)))
    else:
        X0 = norm_last(random.normal(init_key, (runs, n, d), dtype=jnp.float32))

    # Strong solve initially
    U0 = intrinsic_mean_sd_batch(X0, 1e-10, 100)  # (R,d)

    C_cache0 = compute_global_C_from_X(X0, mean_refine_steps)  # (R,d)

    traj0 = (jnp.zeros((kept, runs, n, d), dtype=jnp.float32)
             .at[0].set(X0))
    Uk0   = (jnp.zeros((kept, runs, d), dtype=jnp.float32)
             .at[0].set(U0))
    frame0 = jnp.array(1, dtype=jnp.int32)

    dt32    = jnp.float32(dt)
    sqrt_dt = jnp.sqrt(dt32)
    b32     = jnp.float32(b)
    sigma0_ = jnp.float32(sigma0)

    @jax.jit
    def run():
        def step_fn(carry, k):
            X, U, C_cache, key, traj, Ukr, frame = carry
            # Deterministic part
            dX = dynamics_batched(X, b=b32)

            # Refine the run-wise mean U every few steps 
            def refine(U_in):
                def one_refine(u):
                    V  = _log_map_batch_at_u(u, X)   # (R,n,d)
                    g  = jnp.mean(V, axis=1)         # (R,d)
                    def step(ui, gi): return norm_last(_exp_map_at_u(ui, gi))
                    return jax.vmap(step)(u, g)
                def body(u, _): return one_refine(u), None
                u_out, _ = lax.scan(body, U_in, xs=None, length=mean_refine_steps)
                return u_out

            do_refine = (k % MEAN_UPDATE_STRIDE) == 0
            U_new = lax.cond(do_refine, refine, lambda u: u, U)  # (R,d)

            def recompute_C(_):
                return compute_global_C_from_X(X, mean_refine_steps)  # (R,d)
            do_update_C = (k % MEAN_UPDATE_STRIDE) == 0
            C_used = lax.cond(do_update_C, recompute_C, lambda _: C_cache, operand=None)  # (R,d)

            # Tangent Gaussian noise 
            key, sk = random.split(key)
            rnd = random.normal(sk, X.shape, dtype=jnp.float32)
            proj = jnp.sum(rnd * X, axis=-1, keepdims=True) * X
            noise_tan = rnd - proj
            
            c_i = jnp.clip(jnp.sum(X * C_used[:, None, :], axis=-1, keepdims=True), -1.0, 1.0)  # (R,n,1)
            theta_i = jnp.arccos(c_i)                                                           # (R,n,1)
            amp = sigma0_ * theta_i                                                             # (R,n,1)

            # Ito correction 
            X_next = X + dt32 * (dX - 0.5 * (amp**2) * (d - 1) * X) \
                       + sqrt_dt * (amp * noise_tan)
            X_next = norm_last(X_next)
            

           
            C_next = C_used

            # Store at stride
            store_now = ((k + 1) % STORE_STRIDE == 0) | (k == steps - 1)

            def store_path(args):
                traj_in, Ukr_in, frame_in, Xsnap, Usnap = args
                traj_out = lax.dynamic_update_slice(traj_in, Xsnap[None, ...], (frame_in, 0, 0, 0))
                Ukr_out  = lax.dynamic_update_slice(Ukr_in, Usnap[None, ...], (frame_in, 0, 0))
                return traj_out, Ukr_out, frame_in + 1

            def skip_path(args):
                traj_in, Ukr_in, frame_in, *_ = args
                return traj_in, Ukr_in, frame_in

            traj_new, Ukr_new, frame_new = lax.cond(
                store_now, store_path, skip_path, (traj, Ukr, frame, X_next, U_new)
            )
            return (X_next, U_new, C_next, key, traj_new, Ukr_new, frame_new), None

        carry0 = (X0, U0, C_cache0, random.PRNGKey(NOISE_SEED), traj0, Uk0, frame0)
        (Xf, Uf, Cfin, keyf, trajf, Ukrf, framef), _ = lax.scan(
            step_fn, carry0, xs=jnp.arange(steps, dtype=jnp.int32)
        )
        return trajf, Ukrf  # (kept, R, n, d), (kept, R, d)

    return run()

@jax.jit
def _diameter_single_set(X):  # X: (n,d)
    G = jnp.clip(X @ jnp.swapaxes(X, -1, -2), -1.0, 1.0)
    return jnp.max(jnp.arccos(G))

_diam_per_run  = jax.vmap(_diameter_single_set, in_axes=0)  # (R,n,d)->(R,)
_diam_per_time = jax.vmap(_diam_per_run, in_axes=0)         # (kept,R,n,d)->(kept,R)

def diameter_time_series_deg_jax(traj_kept):
    # traj_kept: (kept, R, n, d)
    diam = _diam_per_time(traj_kept)              # (kept, R) in radians
    diam_deg = jnp.degrees(diam)
    mean_diam = jnp.mean(diam_deg, axis=1)
    # safer than percentile on some JAX builds
    p10 = jnp.quantile(diam_deg, 0.10, axis=1, method="linear")
    p90 = jnp.quantile(diam_deg, 0.90, axis=1, method="linear")
    return diam_deg, mean_diam, p10, p90

# mean trajectory over runs using stored U_kept (each U is already per-run agent mean)
def mean_trajectory_over_runs_from_U(U_kept):
    # U_kept: (kept, R, d)
    return intrinsic_mean_sd_batch(U_kept, 1e-10, 100)  # (kept, d)

# per-agent mean trajectories (vectorized intrinsic mean over runs for each (t,i))
def mean_trajectories_per_agent_jax(traj_kept):
    kept, R, n, d = traj_kept.shape
    Xi = jnp.transpose(traj_kept, (0, 2, 1, 3)).reshape(kept * n, R, d)  # (kept*n, R, d)
    means = jax.jit(jax.vmap(intrinsic_mean_sd_jax, in_axes=(0, None, None)))(
        Xi, 1e-10, 100
    )  # (kept*n, d)
    return means.reshape(kept, n, d)
# PCA
def pca_project_3d_global(points_Kxd):
    mu = points_Kxd.mean(axis=0, keepdims=True)
    X = points_Kxd - mu
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V3 = Vt[:3].T  # (d,3)
    def proj(Y):
        Yc = Y - mu
        return Yc @ V3
    return proj

def main():
    init_key  = random.PRNGKey(INIT_SEED)

    # Deterministic (sigma=0) for reference
    # * Need to tweek TMAX for every different parameter setting
    traj_det, U_det = simulate_intrinsic_noise_jax(
        n=N_AGENTS, d=DIM, T=TMAX*5, dt=DT, b=BETA, sigma0=0.0,
        runs=1, mean_refine_steps=MEAN_REFINE_STEPS,
        init_key=init_key, noise_key=random.PRNGKey(0),
        store_stride=STORE_STRIDE, mean_update_stride=MEAN_UPDATE_STRIDE,
        same_init_across_runs=True
    )
    traj_det_np = np.array(traj_det)  # (kept, 1, n, d)
    u_det = np.array(U_det)[-1, 0]    # final per-run mean as deterministic direction

    # Noisy runs (shared init)
    traj_noisy, U_noisy = simulate_intrinsic_noise_jax(
        n=N_AGENTS, d=DIM, T=TMAX, dt=DT, b=BETA, sigma0=SIGMA,
        runs=RUNS, mean_refine_steps=MEAN_REFINE_STEPS,
        init_key=init_key, noise_key=random.PRNGKey(NOISE_SEED),
        store_stride=STORE_STRIDE, mean_update_stride=MEAN_UPDATE_STRIDE,
        same_init_across_runs=True
    )

    # --- Convergence curve in degrees (JAX) ---
    diam_deg, mean_diam_deg, p10_deg, p90_deg = diameter_time_series_deg_jax(traj_noisy)
    diam_deg_np = np.array(diam_deg)
    mean_diam_deg_np = np.array(mean_diam_deg)
    p10_deg_np = np.array(p10_deg)
    p90_deg_np = np.array(p90_deg)

    kept = diam_deg_np.shape[0]
    times = np.arange(kept, dtype=float) * (STORE_STRIDE * DT)
    times[-1] = TMAX

    # --- Mean trajectory over runs (from stored U_noisy) ---
    M_global = np.array(mean_trajectory_over_runs_from_U(U_noisy))  # (kept, d)

    # --- Per-agent mean trajectories 
    M_agents = np.array(mean_trajectories_per_agent_jax(traj_noisy))  # (kept, n, d)

    # --- Drift vs deterministic point  ---
    drift = geodesic_angle_np(u_det, M_global[-1])
    drift_deg = np.degrees(drift)

    # ---------- Diagnostics: norms & NaN/Inf checks ----------
    def _stats_unit_norm(name, arr, axis=-1):
        norms = np.linalg.norm(arr, axis=axis)
        dev   = np.abs(norms - 1.0)
        print(f"[diag] {name}: max|‖·‖-1|={dev.max():.3e},  mean|‖·‖-1|={dev.mean():.3e}")
        if not np.all(np.isfinite(arr)):
            bad = np.argwhere(~np.isfinite(arr))
            print(f"[diag] {name}: Found NaN/Inf at indices (showing first 5): {bad[:5]}")

    _stats_unit_norm("traj_noisy (states)", traj_noisy, axis=-1)
    _stats_unit_norm("U_noisy (per-run means)", U_noisy, axis=-1)
    _stats_unit_norm("M_global (mean over runs)", M_global, axis=-1)
    _stats_unit_norm("M_agents (per-agent mean over runs)", M_agents, axis=-1)

    # Optional: assert if anything is off beyond a tight tolerance
    MAX_DEV = 5e-4
    assert np.all(np.abs(np.linalg.norm(traj_noisy, axis=-1) - 1.0) < MAX_DEV), \
        "Some raw states deviate from unit sphere more than tolerance."
    assert np.all(np.abs(np.linalg.norm(M_agents,    axis=-1) - 1.0) < MAX_DEV), \
        "Some per-agent means deviate from unit sphere more than tolerance."

    # ---------- Optional visual-only re-normalization before plotting ----------
    M_agents = M_agents / np.maximum(np.linalg.norm(M_agents, axis=-1, keepdims=True), 1e-12)
    M_global = M_global / max(np.linalg.norm(M_global[-1]), 1e-12)

    # --- Print consensus (first time the diameter crosses the threshold) ---
    diam_rad_per_time_run = np.array(_diam_per_time(traj_noisy))
    thr_rad = THRESHOLD

    if RUNS == 1:
        hits = np.flatnonzero(diam_rad_per_time_run[:, 0] <= thr_rad)
        if hits.size > 0:
            t_idx = int(hits[0])
            t_hit = times[t_idx]
            consensus_vec = np.array(U_noisy)[t_idx, 0]  # per-run intrinsic mean at hit time
            print(f"[consensus] Converged (≤ {np.degrees(thr_rad):.3f}°) at t={t_hit:.6g}. "
                  f"Consensus ≈ {consensus_vec}")
        else:
            print(f"[consensus] Did not converge to ≤ {np.degrees(thr_rad):.3f}° by T={TMAX}.")
    else:
        first_idx = np.full(RUNS, -1, dtype=int)
        for r in range(RUNS):
            hits_r = np.flatnonzero(diam_rad_per_time_run[:, r] <= thr_rad)
            if hits_r.size > 0:
                first_idx[r] = int(hits_r[0])
        converged_runs = np.flatnonzero(first_idx >= 0)
        num_conv = converged_runs.size
        print(f"[consensus] Converged runs: {num_conv}/{RUNS} "
              f"(threshold ≤ {np.degrees(thr_rad):.3f}°).")
        if num_conv > 0:
            U_np = np.array(U_noisy)  # (kept, R, d)
            cons_list = [U_np[first_idx[r], r] for r in converged_runs]
            cons_arr = np.stack(cons_list, axis=0)
            cons_global = np.array(intrinsic_mean_sd_jax(jnp.array(cons_arr)))
            print(f"[consensus] Example run {int(converged_runs[0])} consensus ≈ {cons_arr[0]}")
            print(f"[consensus] Intrinsic mean of all converged runs ≈ {cons_global}")
        else:
            print(f"[consensus] No runs reached the threshold by T={TMAX}.")

    # PLOTTING

    # (1) Convergence plot (degrees, log y)
    eps_deg = 1e-6
    yl = np.clip(mean_diam_deg_np, eps_deg, None)
    y10 = np.clip(p10_deg_np, eps_deg, None)
    y90 = np.clip(p90_deg_np, eps_deg, None)

    fig1, ax1 = plt.subplots(figsize=(7.6, 4.8))
    ax1.plot(times, yl, label="Mean diameter (deg)", linewidth=2.0)
    ax1.fill_between(times, y10, y90, alpha=0.25, label="10-90% band", linewidth=0.0)
    ax1.axhline(np.degrees(THRESHOLD), linestyle="--", linewidth=1.25, label="Threshold (deg)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Max pairwise geodesic distance (deg) [log]")
    ax1.set_title("Convergence: geodesic diameter (degrees, log scale)")
    ax1.legend(loc="upper right")
    ax1.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax1.xaxis.set_major_locator(MaxNLocator(8))

    # (2) Per-particle mean trajectories 
    cmap = plt.cm.get_cmap("tab20", N_AGENTS)

    if DIM == 3:
        fig2 = plt.figure(figsize=(7.4, 7.0))
        ax2 = fig2.add_subplot(111, projection="3d")

        # sphere wireframe
        u = np.linspace(0, 2*np.pi, 80)
        v = np.linspace(0, np.pi, 40)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax2.plot_wireframe(xs, ys, zs, linewidth=0.25, alpha=0.35)

        for i in range(N_AGENTS):
            curve = M_agents[:, i, :]  # (kept, 3)
            color_i = cmap(i % cmap.N)
            ax2.plot(curve[:,0], curve[:,1], curve[:,2],
                     linewidth=1.8, alpha=0.95, color=color_i)
            # haloed cross at end
            ax2.plot([curve[-1,0]], [curve[-1,1]], [curve[-1,2]],
                     marker="x", markersize=11, mew=3.0, color="white",
                     linestyle="None", zorder=10)
            ax2.plot([curve[-1,0]], [curve[-1,1]], [curve[-1,2]],
                     marker="x", markersize=8,  mew=1.8, color=color_i,
                     linestyle="None", zorder=11)

        ax2.set_box_aspect([1,1,1])
        ax2.set_xlim([-1.05, 1.05]); ax2.set_ylim([-1.05, 1.05]); ax2.set_zlim([-1.05, 1.05])
        ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
        dev_agents = np.abs(np.linalg.norm(M_agents, axis=-1) - 1.0).max()
        ax2.set_title(f"Per-particle mean trajectories on $S^2$ (end = x)\nmax dev |‖·‖-1| = {dev_agents:.2e}")
        ax2.xaxis.pane.fill = False; ax2.yaxis.pane.fill = False; ax2.zaxis.pane.fill = False
        ax2.grid(False)

    else:
        all_points = M_agents.reshape(-1, DIM)  # (kept*n, d)
        proj = pca_project_3d_global(all_points)

        fig2 = plt.figure(figsize=(7.4, 7.0))
        ax2 = fig2.add_subplot(111, projection="3d")

        for i in range(N_AGENTS):
            curve_d = M_agents[:, i, :]        # (kept,d)
            curve3  = proj(curve_d)            # (kept,3)
            color_i = cmap(i % cmap.N)
            ax2.plot(curve3[:,0], curve3[:,1], curve3[:,2],
                     linewidth=1.8, alpha=0.95, color=color_i)
            # haloed cross at end
            ax2.plot([curve3[-1,0]], [curve3[-1,1]], [curve3[-1,2]],
                     marker="x", markersize=11, mew=3.0, color="white",
                     linestyle="None", zorder=10)
            ax2.plot([curve3[-1,0]], [curve3[-1,1]], [curve3[-1,2]],
                     marker="x", markersize=8,  mew=1.8, color=color_i,
                     linestyle="None", zorder=11)

        ax2.set_box_aspect([1,1,1])
        ax2.set_title("Per-particle mean trajectories (PCA->3D, end = x)")
        ax2.grid(False)

    # ---- Print summary & show ----
    print("\n=== SUMMARY ===")
    print(f"n={N_AGENTS}, d={DIM}, beta={BETA}, dt={DT}, Tmax={TMAX}, threshold={THRESHOLD}")
    print(f"sigma={SIGMA}, runs={RUNS}, store_stride={STORE_STRIDE}")
    print(f"Drift vs deterministic (rad): {drift:.6e}, (deg): {np.degrees(drift):.6e}")
    print(f"Stored frames: {kept}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
