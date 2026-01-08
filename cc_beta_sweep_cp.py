#!/usr/bin/env python3
"""
Fix sigma0, sweep beta, record ||c_rho(Tmax)|| for the MEAN TRAJECTORY (mean over runs).
Sharded over initializations; writes per-(beta_idx, shard_idx) CSV.

Folder convention (similar spirit as before):
  out_cp_beta_sweep/d{d}_n{n}_sig{sigma}/beta_sweep_beta{beta_idx}_shard{shard}.csv

Each row corresponds to ONE initialization (mean trajectory averaged over runs_per_init).
"""

import os, argparse, csv, math
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax, jit
def make_beta_grid_hybrid(beta_min, beta_max, beta_bins, beta_break=1.0, beta_lin_bins=21):
    """
    Hybrid grid on [beta_min, beta_max] with total length beta_bins:

      - linear on [beta_min, beta_break] with beta_lin_bins points
      - log on [beta_break, beta_max] with (beta_bins - beta_lin_bins) points

    Requirements:
      beta_min <= beta_break <= beta_max
      beta_lin_bins >= 2, beta_bins >= 3, beta_lin_bins < beta_bins
      beta_break > 0 is required for the log part (since logspace needs >0).
    """
    if not (beta_min <= beta_break <= beta_max):
        raise ValueError("Need beta_min <= beta_break <= beta_max")
    if beta_bins < 3:
        raise ValueError("beta_bins must be >= 3")
    if beta_lin_bins < 2 or beta_lin_bins >= beta_bins:
        raise ValueError("Need 2 <= beta_lin_bins < beta_bins")
    if beta_break <= 0.0:
        raise ValueError("beta_break must be > 0 for log spacing")

    n_lin = int(beta_lin_bins)
    n_log = int(beta_bins - beta_lin_bins)

    beta_lin = np.linspace(beta_min, beta_break, n_lin, dtype=np.float64)
    beta_log = np.logspace(np.log10(beta_break), np.log10(beta_max), n_log, dtype=np.float64)

    # avoid duplicating beta_break
    betas = np.concatenate([beta_lin, beta_log[1:]]) if n_log > 0 else beta_lin
    if len(betas) != beta_bins:
        # If due to n_log=1 etc, enforce exact length
        betas = np.unique(betas)
        if len(betas) > beta_bins:
            betas = betas[:beta_bins]
        elif len(betas) < beta_bins:
            # pad with a few extra linear points at the end if ever needed
            pad = np.linspace(betas[-1], beta_max, beta_bins - len(betas) + 1, dtype=np.float64)[1:]
            betas = np.concatenate([betas, pad])

    return betas

# ---------------- Sphere helpers ----------------
def norm_last(x, eps=1e-7):
    nrm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(nrm, eps)

def softmax_last(a):
    amax = jnp.max(a, axis=-1, keepdims=True)
    z = jnp.exp(a - amax)
    return z / jnp.sum(z, axis=-1, keepdims=True)

def log_map_sphere_generic(U, X, eps=1e-7):
    c  = jnp.clip(jnp.sum(X * U[..., None, :], axis=-1), -1.0, 1.0)
    th = jnp.arccos(c)
    Uperp = X - c[..., None] * U[..., None, :]
    su = jnp.linalg.norm(Uperp, axis=-1, keepdims=True)
    fac = th[..., None] / jnp.maximum(su, eps)
    fac = jnp.where(th[..., None] < eps, 0.0, fac)
    return fac * Uperp

def exp_map_sphere_generic(U, V, eps=1e-7):
    nv = jnp.linalg.norm(V, axis=-1, keepdims=True)
    small = nv < eps
    Y = jnp.cos(nv) * U + jnp.sin(nv) * (V / jnp.maximum(nv, eps))
    Y_small = norm_last(U + V, eps=eps)
    Y = jnp.where(small, Y_small, Y)
    return norm_last(Y, eps=eps)

def intrinsic_mean_refine_runs_per_agent(M_agents, X_runs_agents, iters=1, step=1.0):
    """
    Intrinsic mean across runs for each agent.
    M_agents: (n,d)
    X_runs_agents: (R,n,d)
    """
    Xnr = jnp.swapaxes(X_runs_agents, 0, 1)  # (n,R,d)
    def one_step(Uc):
        V = log_map_sphere_generic(Uc, Xnr)   # (n,R,d)
        g = jnp.mean(V, axis=-2)              # (n,d)
        return exp_map_sphere_generic(Uc, step * g)
    def body(Uc, _):
        return one_step(Uc), None
    U_out, _ = lax.scan(body, M_agents, None, length=iters)
    return U_out

def geodesic_diameter_points(P):
    G = jnp.clip(jnp.einsum("id,jd->ij", P, P), -1.0, 1.0)
    Theta = jnp.arccos(G)
    return jnp.max(Theta)

# ---------------- Drift dynamics (attention style, Q=K=V=I) ----------------
def dynamics_batched(X, beta=1.0):
    inner = X @ jnp.swapaxes(X, -1, -2)     # (R,n,n)
    w = softmax_last(jnp.float32(beta) * inner)
    Vt = w @ X                              # (R,n,d)
    proj = jnp.sum(Vt * X, axis=-1, keepdims=True) * X
    return Vt - proj                        # tangent

# ---------------- Mean-trajectory simulation: record stats at Tmax ----------------
def meantraj_stats_at_Tmax(
    n, d, T, dt, beta, sigma0,
    runs=100,
    threshold=1e-3,
    check_stride=10,
    mean_refine_steps=2,
    init_seed=123, noise_seed=999,
):
    """
    Simulate R runs, maintain per-agent intrinsic mean over runs (mean trajectory).
    Returns:
      (t_hit, converged, cp_norm_Tmax, diam_Tmax)
    """
    steps = int(T / dt)
    dt32, sqrt_dt = jnp.float32(dt), jnp.sqrt(jnp.float32(dt))
    beta32 = jnp.float32(beta)
    sigma32 = jnp.float32(sigma0)

    # homogeneous tangent noise, with amp = sqrt(2)*sigma
    amp = jnp.sqrt(jnp.float32(2.0)) * sigma32

    key_init  = random.PRNGKey(init_seed)
    key_noise = random.PRNGKey(noise_seed)

    # shared init across runs
    x0 = random.normal(key_init, (n, d), dtype=jnp.float32)
    X0 = norm_last(jnp.broadcast_to(x0, (runs, n, d)))  # (R,n,d)

    # initialize mean trajectory at t=0
    M0 = intrinsic_mean_refine_runs_per_agent(
        jnp.mean(jnp.swapaxes(X0, 0, 1), axis=-2),  # Euclidean start (n,d)
        X0, iters=10, step=1.0
    )

    t0    = jnp.array(jnp.inf, dtype=jnp.float32)
    done0 = jnp.array(False)

    def step_fn(carry, k):
        X, M, key, t_hit, done = carry

        dX = dynamics_batched(X, beta=beta32)

        key, sk = random.split(key)
        rnd = random.normal(sk, X.shape, dtype=jnp.float32)
        proj = jnp.sum(rnd * X, axis=-1, keepdims=True) * X
        noise_tan = rnd - proj

        # Euler–Maruyama + Itô correction
        X_next = X + dt32 * (dX - 0.5 * (amp**2) * (d - 1) * X) + sqrt_dt * (amp * noise_tan)
        X_next = norm_last(X_next)

        def do_check_path(args):
            Xc, Mc, thit = args
            M_new = intrinsic_mean_refine_runs_per_agent(Mc, Xc, iters=mean_refine_steps, step=1.0)
            diam = geodesic_diameter_points(M_new)
            crossed = (diam <= threshold) & (~done)
            t_now = (k + 1) * dt32
            th_new = jnp.where(crossed, t_now, thit)
            return Xc, M_new, th_new

        do_check = ((k + 1) % check_stride == 0) | (k == steps - 1)
        X_out, M_out, t_hit_out = lax.cond(
            do_check, do_check_path, lambda args: args, operand=(X_next, M, t_hit)
        )

        done_out = done | (t_hit_out < jnp.inf)
        return (X_out, M_out, key, t_hit_out, done_out), None

    @jit
    def run():
        carry0 = (X0, M0, key_noise, t0, done0)
        (Xf, Mf, keyf, t_hit, donef), _ = lax.scan(
            step_fn, carry0, xs=jnp.arange(steps, dtype=jnp.int32)
        )
        # final stats at Tmax using the mean trajectory Mf (n,d)
        cp = jnp.mean(Mf, axis=0)                  # (d,)
        cp_norm = jnp.linalg.norm(cp).astype(jnp.float32)
        diam_T = geodesic_diameter_points(Mf).astype(jnp.float32)
        return t_hit, donef, cp_norm, diam_T

    t_hit, done, cp_norm, diam_T = run()
    t_hit_f = float(t_hit) if bool(done) else -1.0
    return t_hit_f, bool(done), float(cp_norm), float(diam_T)

# ---------------- CLI / main ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    # dynamics
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--Tmax", type=float, default=500.0)
    ap.add_argument("--threshold", type=float, default=1e-3)

    # beta sweep
    ap.add_argument("--beta_min", type=float, default=0.1)
    ap.add_argument("--beta_max", type=float, default=8.0)
    ap.add_argument("--beta_bins", type=int, default=60)
    ap.add_argument("--beta_break", type=float, default=1.0)
    ap.add_argument("--beta_lin_bins", type=int, default=21)
    ap.add_argument("--beta_idx", type=int, required=True)
    
    # fixed sigma
    ap.add_argument("--sigma0", type=float, required=True)

    # work partitioning
    ap.add_argument("--run_cp", type=int, default=50)          # number of inits
    ap.add_argument("--runs_per_init", type=int, default=100)  # runs averaged for mean traj
    ap.add_argument("--inits_per_task", type=int, required=True)
    ap.add_argument("--shard_idx", type=int, required=True)

    # mean/check controls
    ap.add_argument("--check_stride", type=int, default=10)
    ap.add_argument("--mean_refine_steps", type=int, default=2)

    # seeds
    ap.add_argument("--init_seed", type=int, default=123)
    ap.add_argument("--noise_seed", type=int, default=999)

    # IO
    ap.add_argument("--outroot", type=str, default="out_cp_beta_sweep")
    ap.add_argument("--jobname", type=str, default="beta_sweep")
    return ap.parse_args()

def main():
    args = parse_args()

    # choose beta for this task
    betas = np.linspace(args.beta_min, args.beta_max, args.beta_bins, dtype=np.float64)
    # Choose hybrid grid for beta near zero
    betas = make_beta_grid_hybrid(
	beta_min=args.beta_min,
    	beta_max=args.beta_max,
    	beta_bins=args.beta_bins,
    	beta_break=args.beta_break,
    	beta_lin_bins=args.beta_lin_bins,
	)

    if not (0 <= args.beta_idx < args.beta_bins):
        raise SystemExit("beta_idx out of range")
    beta = float(betas[args.beta_idx])

    # shard range over inits
    start = args.shard_idx * args.inits_per_task
    end = min(start + args.inits_per_task, args.run_cp)
    if start >= args.run_cp:
        print(f"[SKIP] shard {args.shard_idx}: no work (start={start} >= run_cp={args.run_cp}).")
        return

    # folder: d, n, sigma
    sig_tag = f"{args.sigma0:.3f}".rstrip("0").rstrip(".")
    folder = os.path.join(args.outroot, f"d{args.d}_n{args.n}_sig{sig_tag}")
    os.makedirs(folder, exist_ok=True)

    # output CSV name
    csv_name = f"{args.jobname}_beta{args.beta_idx:03d}_shard{args.shard_idx:03d}.csv"
    csv_path = os.path.join(folder, csv_name)

    # atomic write (prevents empty/partial files)
    tmp_path = csv_path + ".tmp"

    header = [
        "d","n","sigma0","beta","beta_idx",
        "Tmax","dt","threshold",
        "runs_mean","run_cp",
        "init_idx_global",
        "T_hit","converged",
        "cp_norm_Tmax",
        "diam_Tmax",
    ]

    with open(tmp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for i_global in range(start, end):
            t_hit, conv, cp_norm, diam_T = meantraj_stats_at_Tmax(
                n=args.n, d=args.d, T=args.Tmax, dt=args.dt,
                beta=beta, sigma0=args.sigma0,
                runs=args.runs_per_init,
                threshold=args.threshold,
                check_stride=args.check_stride,
                mean_refine_steps=args.mean_refine_steps,
                init_seed=args.init_seed + i_global,
                noise_seed=args.noise_seed + i_global,
            )
            w.writerow([
                args.d, args.n, args.sigma0, beta, args.beta_idx,
                args.Tmax, args.dt, args.threshold,
                args.runs_per_init, args.run_cp,
                i_global,
                t_hit, int(conv),
                cp_norm,
                diam_T,
            ])

    os.replace(tmp_path, csv_path)
    print(f"[OK] wrote {csv_path} (beta={beta:.6g}, sigma0={args.sigma0}, inits {start}..{end-1})")

if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "0")
    main()
