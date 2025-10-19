#!/usr/bin/env python3
# Computes, for each init and sigma:
#   - Deterministic consensus point (sigma=0) under intrinsic dynamics
#   - Noisy consensus across runs (intrinsic mean of per-run consensus)
#   - DRIFT = geodesic angle between (noisy consensus) and (deterministic consensus)
# Uses shard scheduling over initializations. Saves per-init drift in NPZ.
#
# Intrinsic dynamics: softmax drift on the sphere; tangent Gaussian noise with
# amplitude = sigma * theta(U, X_i)
# n=32, d=3, dt=0.005, beta=5.0, Tmax=3000, threshold=1e-2.

import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax

# ---------- helper functions ----
def normalize_rows_np(X, eps=1e-12):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, eps)

def geodesic_angle_np(u, v):
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.arccos(c))

def log_map_sphere_batch_np(u, X, eps=1e-12):
    # u: (d,), X: (n,d) -> (n,d) tangent rows at u
    c  = np.clip((X * u[None, :]).sum(axis=1), -1.0, 1.0)
    th = np.arccos(c)
    Uperp = X - c[:, None] * u[None, :]
    su = np.linalg.norm(Uperp, axis=1, keepdims=True)
    fac = th[:, None] / np.maximum(su, eps)
    fac = np.where(th[:, None] < eps, 0.0, fac)
    return fac * Uperp

def exp_map_sphere_np(u, v, eps=1e-12):
    nv = np.linalg.norm(v)
    if nv < eps:
        y = u + v
        n = np.linalg.norm(y)
        return y/n if n > 0 else y
    return np.cos(nv)*u + np.sin(nv)*(v/nv)

def intrinsic_mean_refine_np(u_prev, points, iters=2, step=1.0):
    u = u_prev.copy()
    for _ in range(iters):
        V = log_map_sphere_batch_np(u, points)
        g = V.mean(axis=0)
        if np.linalg.norm(g) < 1e-12:
            break
        u = exp_map_sphere_np(u, step*g)
    return u

def intrinsic_mean_Sd_np(points, tol=1e-10, max_iter=100):
    u = normalize_rows_np(points.mean(axis=0, keepdims=True))[0]
    for _ in range(max_iter):
        V = log_map_sphere_batch_np(u, points)
        g = V.mean(axis=0)
        if np.linalg.norm(g) <= tol:
            break
        u = exp_map_sphere_np(u, g)
    return u
def norm_last(x, eps=1e-7):
    nrm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(nrm, eps)

def softmax_last(a):
    amax = jnp.max(a, axis=-1, keepdims=True)
    z = jnp.exp(a - amax)
    return z / jnp.sum(z, axis=-1, keepdims=True)

def dynamics_batched(X, b=1.0):
    # X: (R,n,d)
    R, n, d = X.shape
    # Q=K=V=I_d
    XQ = X
    XK = X
    XV = X
    inner = XQ @ jnp.swapaxes(XK, -1, -2)  # (R,n,n)
    w = softmax_last(b * inner)            # (R,n,n)
    V = w @ XV                             # (R,n,d)
    proj = jnp.sum(V * X, axis=-1, keepdims=True) * X
    return V - proj

def simulate_intrinsic_noise_jax(n, d, T, dt, b, sigma0,
                                 runs=1, mean_refine_steps=1,
                                 init_key=None, noise_key=None,
                                 store_stride=100, mean_update_stride=5,
                                 same_init_across_runs=True):
    """
    Returns traj_kept: (kept, runs, n, d)
    RMK: We keep stored frames at multiples of store_stride for efficient storage.
    """
    # Python constants 
    # not tracers
    steps = int(float(T) / float(dt))
    kept  = steps // int(store_stride) + 1

    if init_key is None:  init_key  = random.PRNGKey(0)
    if noise_key is None: noise_key = random.PRNGKey(1)

    if same_init_across_runs:
        x0 = random.normal(init_key, (n, d), dtype=jnp.float32)
        X0 = norm_last(jnp.broadcast_to(x0, (runs, n, d)))
    else:
        X0 = norm_last(random.normal(init_key, (runs, n, d), dtype=jnp.float32))

    U0 = norm_last(jnp.mean(X0, axis=1))  # (R,d)
    traj0  = jnp.zeros((kept, runs, n, d), dtype=jnp.float32).at[0].set(X0)
    frame0 = jnp.array(1, dtype=jnp.int32)

    dt32    = jnp.float32(dt)
    sqrt_dt = jnp.sqrt(dt32)
    b32     = jnp.float32(b)
    sigma0_ = jnp.float32(sigma0)

    @jax.jit
    def run():
        def step_fn(carry, k):
            X, U, key, traj, frame = carry

            dX = dynamics_batched(X, b=b32)

            # intrinsic mean refinement over agents per run
            def refine(U_in):
                def body(u, _):
                    # log map at u toward each agent, average in tangent, exp back
                    c  = jnp.clip(jnp.sum(X * u[:, None, :], axis=-1), -1.0, 1.0)
                    th = jnp.arccos(c)
                    Uperp = X - c[..., None] * u[:, None, :]
                    su = jnp.linalg.norm(Uperp, axis=-1, keepdims=True)
                    fac = th[..., None] / jnp.maximum(su, 1e-7)
                    fac = jnp.where(th[..., None] < 1e-7, 0.0, fac)
                    V = jnp.mean(fac * Uperp, axis=1)  # (R,d)
                    # exp map
                    nv = jnp.linalg.norm(V, axis=-1, keepdims=True)
                    small = nv < 1e-7
                    y = jnp.cos(nv)*u + jnp.sin(nv)*(V / jnp.maximum(nv, 1e-7))
                    y_small = norm_last(u + V)
                    u2 = jnp.where(small, y_small, y)
                    return norm_last(u2), None
                u_out, _ = lax.scan(body, U_in, xs=None, length=mean_refine_steps)
                return u_out

            do_refine = (k % mean_update_stride) == 0
            U_new = lax.cond(do_refine, refine, lambda u: u, U)

            # tangent Gaussian noise scaled
            key, sk = random.split(key)
            rnd = random.normal(sk, X.shape, dtype=jnp.float32)
            proj = jnp.sum(rnd * X, axis=-1, keepdims=True) * X
            noise_tan = rnd - proj
            c = jnp.clip(jnp.sum(X * U_new[:, None, :], axis=-1), -1.0, 1.0)  # (R,n)
            theta = jnp.arccos(c)[..., None]                                   # (R,n,1)
            amp = sigma0_ * theta

            X_next = X + dt32 * (dX - 0.5*amp**2*(d-1)*X) + sqrt_dt * (amp * noise_tan)
            X_next = norm_last(X_next)

            store_now = ((k + 1) % store_stride == 0) | (k == steps - 1)

            def store_path(args):
                traj_in, frame_in, Xsnap = args
                traj_out = lax.dynamic_update_slice(traj_in, Xsnap[None, ...], (frame_in, 0, 0, 0))
                return traj_out, frame_in + 1

            def skip_path(args):
                traj_in, frame_in, _ = args
                return traj_in, frame_in

            traj_new, frame_new = lax.cond(store_now, store_path, skip_path, (traj, frame, X_next))
            return (X_next, U_new, key, traj_new, frame_new), None

        carry0 = (X0, U0, noise_key, traj0, frame0)
        (Xf, Uf, keyf, trajf, framef), _ = lax.scan(
            step_fn, carry0, xs=jnp.arange(steps, dtype=jnp.int32)
        )
        return trajf  # (kept, runs, n, d)

    return run()

# ---------- numpy helpers
def diameter_of_agents(X):
    # X: (n,d)
    Xn = normalize_rows_np(X)
    G = np.clip(Xn @ Xn.T, -1.0, 1.0)
    return float(np.arccos(G).max())

def deterministic_consensus(traj_kept, threshold):
    # traj_kept: (kept, 1, n, d)
    kept = traj_kept.shape[0]
    for t in range(kept):
        X = np.array(traj_kept[t, 0])     # (n,d)
        if diameter_of_agents(X) <= threshold:
            u = intrinsic_mean_Sd_np(X)
            return u, 1
    # not converged by Tmax: use last frame mean
    u = intrinsic_mean_Sd_np(np.array(traj_kept[-1, 0]))
    return u, 0

def noisy_consensus_across_runs(traj_kept, threshold):
    # traj_kept: (kept, R, n, d)
    kept, R = traj_kept.shape[0], traj_kept.shape[1]
    first_idx = np.full(R, -1, dtype=int)
    for r in range(R):
        for t in range(kept):
            if diameter_of_agents(np.array(traj_kept[t, r])) <= threshold:
                first_idx[r] = t
                break
    vecs = []
    for r in range(R):
        if first_idx[r] >= 0:
            X = np.array(traj_kept[first_idx[r], r])  # (n,d)
            vecs.append(intrinsic_mean_Sd_np(X))
    if len(vecs) == 0:
        return None, 0
    V = np.vstack(vecs)
    u_noise = intrinsic_mean_Sd_np(V)
    return u_noise, len(vecs)

# ---------- Command line arguments ----------
def parse_args():
    ap = argparse.ArgumentParser()
    # Dynamics
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--b", type=float, default=4.0)     
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--Tmax", type=float, default=3000.0)
    ap.add_argument("--threshold", type=float, default=1e-2)

    # Sigma sweep
    ap.add_argument("--sigma_min", type=float, default=0.0)
    ap.add_argument("--sigma_max", type=float, default=0.5)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--sigma_idx", type=int, required=True)

    # Sharding over initializations
    ap.add_argument("--tot_inits", type=int, default=100)
    ap.add_argument("--runs_per_init", type=int, default=100)
    ap.add_argument("--inits_per_task", type=int, required=True)
    ap.add_argument("--shard_idx", type=int, required=True)

    # Storage / mean refinement cadence
    ap.add_argument("--store_stride", type=int, default=100)
    ap.add_argument("--mean_update_stride", type=int, default=5)
    ap.add_argument("--mean_refine_steps", type=int, default=1)

    # Seeds
    ap.add_argument("--init_seed", type=int, default=123)
    ap.add_argument("--noise_seed", type=int, default=999)

    # IO
    ap.add_argument("--outdir", type=str, default="out_cc_drift")
    ap.add_argument("--jobname", type=str, default="cc_drift")
    return ap.parse_args()

def main():
    # Ensure CPU usage by default
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "0")

    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Resolve sigma for this task
    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.bins, dtype=np.float64)
    if not (0 <= args.sigma_idx < args.bins):
        raise SystemExit("sigma_idx out of range")
    sigma0 = float(sigmas[args.sigma_idx])

    # Shard range
    start = args.shard_idx * args.inits_per_task
    end = min(start + args.inits_per_task, args.tot_inits)
    if start >= args.tot_inits:
        print(f"[SKIP] shard {args.shard_idx}: no work (start={start} >= tot_inits={args.tot_inits}).")
        return

    # Results accumulators
    init_indices = []
    drifts = []
    det_converged_flags = []
    runs_converged_counts = []

    for i_global in range(start, end):
        # deterministic (sigma=0) with same init key
        key_init_det = random.fold_in(random.PRNGKey(args.init_seed), i_global)
        traj_det = simulate_intrinsic_noise_jax(
            n=args.n, d=args.d, T=args.Tmax, dt=args.dt, b=args.b, sigma0=0.0,
            runs=1, mean_refine_steps=args.mean_refine_steps,
            init_key=key_init_det, noise_key=random.PRNGKey(0),
            store_stride=args.store_stride, mean_update_stride=args.mean_update_stride,
            same_init_across_runs=True
        )
        traj_det_np = np.array(traj_det)  # (kept, 1, n, d)
        u_det, det_flag = deterministic_consensus(traj_det_np, args.threshold)

        # noisy runs (same init across runs; different noise)
        key_init_noise  = random.fold_in(random.PRNGKey(args.init_seed),  i_global)
        key_noise_noise = random.fold_in(random.PRNGKey(args.noise_seed), i_global)

        traj_noisy = simulate_intrinsic_noise_jax(
            n=args.n, d=args.d, T=args.Tmax, dt=args.dt, b=args.b, sigma0=sigma0,
            runs=args.runs_per_init, mean_refine_steps=args.mean_refine_steps,
            init_key=key_init_noise, noise_key=key_noise_noise,
            store_stride=args.store_stride, mean_update_stride=args.mean_update_stride,
            same_init_across_runs=True
        )
        traj_noisy_np = np.array(traj_noisy)  # (kept, R, n, d)
        u_noise, n_conv_runs = noisy_consensus_across_runs(traj_noisy_np, args.threshold)

        if u_noise is None:
            drift = np.nan
        else:
            drift = geodesic_angle_np(u_det, u_noise)

        init_indices.append(i_global)
        drifts.append(float(drift))
        det_converged_flags.append(int(det_flag))
        runs_converged_counts.append(int(n_conv_runs))

    # Convert to arrays
    init_indices = np.asarray(init_indices, dtype=np.int32)
    drifts = np.asarray(drifts, dtype=np.float32)
    det_converged_flags = np.asarray(det_converged_flags, dtype=np.int32)
    runs_converged_counts = np.asarray(runs_converged_counts, dtype=np.int32)

    finite = np.isfinite(drifts)
    shard_min = float(np.nanmin(drifts)) if np.any(finite) else float("nan")
    shard_med = float(np.nanmedian(drifts)) if np.any(finite) else float("nan")
    shard_max = float(np.nanmax(drifts)) if np.any(finite) else float("nan")

    # Save shard in file
    fname = f"{args.jobname}_sig{args.sigma_idx:03d}_shard{args.shard_idx:03d}.npz"
    path = os.path.join(args.outdir, fname)
    np.savez(
        path,
        sigma=np.float32(sigma0),
        sigma_idx=np.int32(args.sigma_idx),
        shard_idx=np.int32(args.shard_idx),
        tot_inits=np.int32(args.tot_inits),
        runs_per_init=np.int32(args.runs_per_init),
        inits_per_task=np.int32(args.inits_per_task),
        init_indices=init_indices,
        drift_values=drifts,                 # radians
        det_converged=det_converged_flags,   # 1 if deterministic reached threshold
        runs_converged=runs_converged_counts,# number of converged noisy runs (0..R)
        # meta
        n=np.int32(args.n), d=np.int32(args.d),
        b=np.float32(args.b), dt=np.float32(args.dt),
        Tmax=np.float32(args.Tmax), threshold=np.float32(args.threshold),
        store_stride=np.int32(args.store_stride),
        mean_update_stride=np.int32(args.mean_update_stride),
        mean_refine_steps=np.int32(args.mean_refine_steps),
        shard_min_drift=np.float32(shard_min),
        shard_median_drift=np.float32(shard_med),
        shard_max_drift=np.float32(shard_max),
    )
    print(f"[OK] Saved shard: {path} (σ={sigma0:.6f}, inits {start}..{end-1}, "
          f"min_drift={shard_min:.4e}, median={shard_med:.4e}, max_drift={shard_max:.4e})")

if __name__ == "__main__":
    main()
