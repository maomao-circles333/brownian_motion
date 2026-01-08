#!/usr/bin/env python3
# Sweep sigma0 and record ||c_p|| and related stats for the *mean trajectory*:
#   mean trajectory M(t) = (M_i(t))_{i=1..n}, where M_i(t) is intrinsic mean across runs of agent i.
#
# For each init:
#   - simulate R runs with shared init (X0), homogeneous tangent noise
#   - maintain per-agent intrinsic mean across runs M(t) online (refined every check_stride)
#   - record:
#       T_hit (first time diam(M)<=threshold, else -1)
#       ||c_p|| at Tmax, where c_p = (1/n) sum_i M_i(Tmax) in R^d
#       diam(M) at Tmax
#       distance of CoM: angle between intrinsic CoM across agents at t=0 and t=Tmax
#         CoM here = intrinsic (Karcher) mean across agents of M(t)
#
# Output: CSV per shard: one row per init.

import os, argparse, csv
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax, jit

# ---------------- Sphere ops ----------------
def norm_last(x, eps=1e-7):
    nrm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(nrm, eps)

def softmax_last(a):
    amax = jnp.max(a, axis=-1, keepdims=True)
    z = jnp.exp(a - amax)
    return z / jnp.sum(z, axis=-1, keepdims=True)

def log_map_sphere_generic(U, X, eps=1e-7):
    """
    U: (..., d)      base points on S^{d-1}
    X: (..., m, d)   points on S^{d-1}, same leading dims as U
    -> (..., m, d)   tangent rows at U toward each point in X
    """
    c  = jnp.clip(jnp.sum(X * U[..., None, :], axis=-1), -1.0, 1.0)  # (..., m)
    th = jnp.arccos(c)                                               # (..., m)
    Uperp = X - c[..., None] * U[..., None, :]                       # (..., m, d)
    su = jnp.linalg.norm(Uperp, axis=-1, keepdims=True)              # (..., m, 1)
    fac = th[..., None] / jnp.maximum(su, eps)                       # (..., m, 1)
    fac = jnp.where(th[..., None] < eps, 0.0, fac)
    return fac * Uperp

def exp_map_sphere_generic(U, V, eps=1e-7):
    """
    U, V: (..., d) -> (..., d), with V tangent at U
    """
    nv = jnp.linalg.norm(V, axis=-1, keepdims=True)
    small = nv < eps
    Y = jnp.cos(nv) * U + jnp.sin(nv) * (V / jnp.maximum(nv, eps))
    Y_small = norm_last(U + V, eps=eps)
    Y = jnp.where(small, Y_small, Y)
    return norm_last(Y, eps=eps)

def geodesic_diameter_points(P):
    """
    P: (n, d) points on S^{d-1} -> scalar geodesic diameter (rad)
    """
    G = jnp.clip(jnp.einsum("id,jd->ij", P, P), -1.0, 1.0)   # (n,n)
    Theta = jnp.arccos(G)
    return jnp.max(Theta)

def intrinsic_mean_across_runs_per_agent_refine(M_agents, X_runs_agents, iters=1, step=1.0):
    """
    Refine intrinsic mean across *runs* for each agent.
    M_agents: (n, d)
    X_runs_agents: (R, n, d)
    Returns: (n, d)
    """
    # rearrange to (n, R, d)
    Xnr = jnp.swapaxes(X_runs_agents, 0, 1)

    def one_step(Uc):
        V = log_map_sphere_generic(Uc, Xnr)   # (n,R,d)
        g = jnp.mean(V, axis=-2)              # mean over R -> (n,d)
        return exp_map_sphere_generic(Uc, step * g)

    def body(Uc, _):
        return one_step(Uc), None

    U_out, _ = lax.scan(body, M_agents, None, length=iters)
    return U_out

def intrinsic_mean_across_agents_refine(u, P, iters=5, step=1.0):
    """
    Karcher mean across agents (points P: (n,d)), refined from initial u: (d,).
    Returns u_out: (d,)
    """
    def one_step(uc):
        # log map of all points at base uc
        V = log_map_sphere_generic(uc, P[None, ...])[0]  # (n,d)
        g = jnp.mean(V, axis=0)                          # (d,)
        return exp_map_sphere_generic(uc, step * g)
    def body(uc, _):
        return one_step(uc), None
    u_out, _ = lax.scan(body, u, None, length=iters)
    return u_out

def geodesic_angle(u, v):
    c = jnp.clip(jnp.sum(u * v), -1.0, 1.0)
    return jnp.arccos(c)

# ---------------- Drift dynamics (attention-style, Q=K=V=I) ----------------
def dynamics_batched(X, b=1.0):
    """
    X: (R, n, d) -> tangent field (R, n, d)
    """
    inner = X @ jnp.swapaxes(X, -1, -2)   # (R,n,n)
    w = softmax_last(b * inner)           # (R,n,n)
    Vt = w @ X                            # (R,n,d)
    proj = jnp.sum(Vt * X, axis=-1, keepdims=True) * X
    return Vt - proj

# ---------------- Per-init simulation (online mean-trajectory stats) --------
def meantraj_stats_one_init(
    n, d, T, dt, b, sigma0,
    runs=100,
    threshold=1e-3,
    check_stride=10,
    mean_refine_steps=2,
    init_seed=0, noise_seed=0,
    init_mean_refine_iters=10,
    com_refine_iters=8,
):
    """
    Returns:
      t_hit (float32, -1 if not hit),
      cp_norm_T (float32),
      diam_T (float32),
      com_dist (float32)  # intrinsic CoM across agents at t=0 vs t=T
    """
    steps = int(T / dt)
    dt32 = jnp.float32(dt)
    sqrt_dt = jnp.sqrt(dt32)
    b32 = jnp.float32(b)
    sigma32 = jnp.float32(sigma0)
    amp = jnp.sqrt(jnp.float32(2.0)) * sigma32  # homogeneous amplitude

    key_init  = random.PRNGKey(init_seed)
    key_noise = random.PRNGKey(noise_seed)

    # shared init across runs
    x0 = random.normal(key_init, (n, d), dtype=jnp.float32)
    X0 = norm_last(jnp.broadcast_to(x0, (runs, n, d)))  # (R,n,d)

    # initial mean trajectory M0 across runs per agent
    M0_euc = norm_last(jnp.mean(X0, axis=0))  # (n,d) Euclidean mean across runs (same as x0, but keep generic)
    M0 = intrinsic_mean_across_runs_per_agent_refine(
        M0_euc, X0, iters=init_mean_refine_iters, step=1.0
    )

    # intrinsic CoM across agents at t=0 (Karcher mean across agents)
    u0_init = norm_last(jnp.mean(M0, axis=0))
    u0 = intrinsic_mean_across_agents_refine(u0_init, M0, iters=com_refine_iters, step=1.0)

    t_hit0 = jnp.float32(-1.0)

    def step_fn(carry, k):
        X, M, key, t_hit = carry

        # drift
        dX = dynamics_batched(X, b=b32)

        # homogeneous tangent noise
        key, sk = random.split(key)
        rnd = random.normal(sk, X.shape, dtype=jnp.float32)
        proj = jnp.sum(rnd * X, axis=-1, keepdims=True) * X
        noise_tan = rnd - proj

        # Euler–Maruyama + Itô correction
        X_next = X + dt32 * (dX - 0.5 * (amp**2) * (d - 1) * X) + sqrt_dt * (amp * noise_tan)
        X_next = norm_last(X_next)

        # check & refine mean-trajectory at stride
        do_check = ((k + 1) % check_stride == 0) | (k == steps - 1)

        def do_check_path(args):
            Xc, Mc, thit = args
            M_new = intrinsic_mean_across_runs_per_agent_refine(
                Mc, Xc, iters=mean_refine_steps, step=1.0
            )
            diam = geodesic_diameter_points(M_new)
            crossed = (diam <= jnp.float32(threshold)) & (thit < 0.0)
            t_now = (k + 1) * dt32
            th_new = jnp.where(crossed, t_now, thit)
            return Xc, M_new, th_new

        X_out, M_out, t_hit_out = lax.cond(
            do_check, do_check_path, lambda args: args, operand=(X_next, M, t_hit)
        )

        return (X_out, M_out, key, t_hit_out), None

    @jit
    def run():
        carry0 = (X0, M0, key_noise, t_hit0)
        (Xf, Mf, keyf, t_hit), _ = lax.scan(step_fn, carry0, xs=jnp.arange(steps, dtype=jnp.int32))

        # finalize stats at Tmax using Mf (mean-trajectory config)
        cp = jnp.mean(Mf, axis=0)                 # Euclidean center of mass in R^d
        cp_norm = jnp.linalg.norm(cp)

        diam_T = geodesic_diameter_points(Mf)

        uT_init = norm_last(jnp.mean(Mf, axis=0))
        uT = intrinsic_mean_across_agents_refine(uT_init, Mf, iters=com_refine_iters, step=1.0)

        com_dist = geodesic_angle(u0, uT)

        return t_hit, cp_norm, diam_T, com_dist

    t_hit, cp_norm_T, diam_T, com_dist = run()
    return float(t_hit), float(cp_norm_T), float(diam_T), float(com_dist)

# ---------------- CLI / main (saves CSV) -----------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--Tmax", type=float, default=500.0)
    ap.add_argument("--threshold", type=float, default=1e-3)

    # sigma sweep
    ap.add_argument("--sigma_min", type=float, default=0.0)
    ap.add_argument("--sigma_max", type=float, default=2.0)
    ap.add_argument("--bins", type=int, default=237)
    ap.add_argument("--sigma_idx", type=int, required=True)

    # averaging structure
    ap.add_argument("--runs_mean", type=int, default=100)  # runs per init for mean trajectory
    ap.add_argument("--run_cp", type=int, default=50)      # total inits (mean trajectories) to average over

    # sharding over inits
    ap.add_argument("--inits_per_task", type=int, required=True)
    ap.add_argument("--shard_idx", type=int, required=True)

    ap.add_argument("--check_stride", type=int, default=10)
    ap.add_argument("--mean_refine_steps", type=int, default=2)
    ap.add_argument("--init_mean_refine_iters", type=int, default=10)
    ap.add_argument("--com_refine_iters", type=int, default=8)

    # seeds (base)
    ap.add_argument("--init_seed", type=int, default=123)
    ap.add_argument("--noise_seed", type=int, default=999)

    # IO
    ap.add_argument("--outroot", type=str, default="out_cp_sweep")
    ap.add_argument("--jobname", type=str, default="cp_sweep")
    return ap.parse_args()

def main():

    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "0")

    args = parse_args()

    # folder per (d,n,beta)
    folder = os.path.join(args.outroot, f"d{args.d}_n{args.n}_b{args.beta:g}")
    os.makedirs(folder, exist_ok=True)

    # sigma value for this task
    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.bins, dtype=np.float64)
    if not (0 <= args.sigma_idx < args.bins):
        raise SystemExit("sigma_idx out of range")
    sigma0 = float(sigmas[args.sigma_idx])

    # shard range over inits: total inits = run_cp
    tot_inits = int(args.run_cp)
    start = args.shard_idx * args.inits_per_task
    end = min(start + args.inits_per_task, tot_inits)
    if start >= tot_inits:
        print(f"[SKIP] shard {args.shard_idx}: no work (start={start} >= run_cp={tot_inits}).")
        return

    # output CSV per shard
    csv_name = f"{args.jobname}_sig{args.sigma_idx:03d}_shard{args.shard_idx:03d}.csv"
    csv_path = os.path.join(folder, csv_name)

    header = [
        "init_idx",
        "d", "n", "beta",
        "T_max", "dt",
        "sigma0", "threshold",
        "runs_mean", "run_cp", 
	# How many runs to average for mean trajectory
	# How many initializations of the mean trajectory to compute the 
	# average cp in the plot
        "T_hit",
        "cp_norm_Tmax",
        "diam_Tmax",
        "com_dist_init_to_Tmax",
        "check_stride",
        "mean_refine_steps",
        "init_mean_refine_iters",
        "com_refine_iters",
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for i_global in range(start, end):
            # distinct seeds per init
            init_seed = args.init_seed + i_global
            noise_seed = args.noise_seed + i_global

            t_hit, cp_norm_T, diam_T, com_dist = meantraj_stats_one_init(
                n=args.n, d=args.d, T=args.Tmax, dt=args.dt, b=args.beta,
                sigma0=sigma0,
                runs=args.runs_mean,
                threshold=args.threshold,
                check_stride=args.check_stride,
                mean_refine_steps=args.mean_refine_steps,
                init_seed=init_seed, noise_seed=noise_seed,
                init_mean_refine_iters=args.init_mean_refine_iters,
                com_refine_iters=args.com_refine_iters,
            )

            #  -1 if not converged
            # (already -1 if not hit)
            w.writerow([
                i_global,
                args.d, args.n, args.beta,
                args.Tmax, args.dt,
                sigma0, args.threshold,
                args.runs_mean, args.run_cp,
                t_hit,
                cp_norm_T,
                diam_T,
                com_dist,
                args.check_stride,
                args.mean_refine_steps,
                args.init_mean_refine_iters,
                args.com_refine_iters,
            ])

    print(f"[OK] Saved CSV: {csv_path} (d={args.d}, n={args.n}, beta={args.beta}, sigma={sigma0:.6f}, inits {start}..{end-1})")

if __name__ == "__main__":
    main()
