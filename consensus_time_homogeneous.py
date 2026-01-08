#!/usr/bin/env python3
# Homogeneous tangent noise + intrinsic *mean trajectory* consensus time (no big storage).
# SDE per agent: dX = f(X) dt + sigma * P_X dB  - sigma^2 (d-1) X dt
# Implementation detail: we use amp = sqrt(2)*sigma, so correction = -0.5*amp^2*(d-1) X.

import os, argparse, numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax, jit

# ---------------- Helpers (sphere ops) ----------------
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
    X: (..., n, d)   points on S^{d-1}, same leading dims as U
    -> (..., n, d)   tangent rows at U toward each point in X
    """
    c  = jnp.clip(jnp.sum(X * U[..., None, :], axis=-1), -1.0, 1.0)  # (..., n)
    th = jnp.arccos(c)                                               # (..., n)
    Uperp = X - c[..., None] * U[..., None, :]                       # (..., n, d)
    su = jnp.linalg.norm(Uperp, axis=-1, keepdims=True)              # (..., n, 1)
    fac = th[..., None] / jnp.maximum(su, eps)                       # (..., n, 1)
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

def intrinsic_mean_refine_runs_per_agent(M_agents, X_runs_agents, iters=1, step=1.0):
    """
    Refine intrinsic mean across *runs* for each agent.
    M_agents: (n, d)
    X_runs_agents: (R, n, d)   points across runs for each agent
    Returns: (n, d)
    """
    Xnr = jnp.swapaxes(X_runs_agents, 0, 1)  # (n, R, d)
    def one_step(Uc):
        V = log_map_sphere_generic(Uc, Xnr)   # Uc: (n,d), Xnr: (n,R,d) -> (n,R,d)
        g = jnp.mean(V, axis=-2)              # (n,d)
        return exp_map_sphere_generic(Uc, step * g)
    def body(Uc, _):
        return one_step(Uc), None
    U_out, _ = lax.scan(body, M_agents, None, length=iters)
    return U_out

def geodesic_diameter_points(P):
    """
    P: (n, d) points on S^{d-1} -> scalar geodesic diameter (rad)
    """
    G = jnp.clip(jnp.einsum("id,jd->ij", P, P), -1.0, 1.0)   # (n,n)
    Theta = jnp.arccos(G)
    return jnp.max(Theta)

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

# ---------------- Online mean-trajectory first-hit (homogeneous noise) -------
def mean_traj_first_hit_online(
    n, d, T, dt, b, sigma0,
    runs=100,
    threshold=1e-2,
    check_stride=50,
    mean_refine_steps=2,
    init_seed=30, noise_seed=1000,
):
    """
    Simulate R runs with homogeneous tangent noise.
    Maintain only the per-agent intrinsic *mean trajectory* across runs.
    Every 'check_stride', refine the mean and check its geodesic diameter.
    Returns: (t_hit: float32, converged: bool)
    """
    steps = int(T / dt)
    dt32, sqrt_dt = jnp.float32(dt), jnp.sqrt(jnp.float32(dt))
    b32 = jnp.float32(b)
    sigma32 = jnp.float32(sigma0)
    amp = jnp.sqrt(jnp.float32(2.0)) * sigma32  # homogeneous amplitude

    key_init  = random.PRNGKey(init_seed)
    key_noise = random.PRNGKey(noise_seed)

    # shared X0 across runs
    x0 = random.normal(key_init, (n, d), dtype=jnp.float32)
    X0 = norm_last(jnp.broadcast_to(x0, (runs, n, d)))  # (R,n,d)

    # Initialize M_agents at t=0 (intrinsic mean across runs per agent)
    M0 = intrinsic_mean_refine_runs_per_agent(
        jnp.mean(jnp.swapaxes(X0, 0, 1), axis=-2),  # (n,d) Euclidean start
        X0, iters=10, step=1.0
    )

    # carry: X, M_agents, key, t_hit, done
    t0    = jnp.array(jnp.inf, dtype=jnp.float32)
    done0 = jnp.array(False)

    def step_fn(carry, k):
        X, M, key, t_hit, done = carry

        # --- Drift
        dX = dynamics_batched(X, b=b32)

        # --- Homogeneous tangent noise
        key, sk = random.split(key)
        rnd = random.normal(sk, X.shape, dtype=jnp.float32)
        proj = jnp.sum(rnd * X, axis=-1, keepdims=True) * X
        noise_tan = rnd - proj   # P_X * dB

        # Euler–Maruyama + Itô correction (scalar amp)
        X_next = X + dt32 * (dX - 0.5 * (amp**2) * (d - 1) * X) + sqrt_dt * (amp * noise_tan)
        X_next = norm_last(X_next)

        # --- At check_stride: refine M_agents (mean across runs per agent) and test diameter
        def do_check_path(args):
            Xc, Mc, thit = args
            M_new = intrinsic_mean_refine_runs_per_agent(Mc, Xc, iters=mean_refine_steps, step=1.0)
            diam = geodesic_diameter_points(M_new)      # scalar
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
        return t_hit, donef

    t_hit, done = run()
    return float(t_hit), bool(done)

# ---------------- CLI / main (sharded sweep; saves NPZ) -----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    # Dynamics / geometry
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--b", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--Tmax", type=float, default=3000.0)
    ap.add_argument("--threshold", type=float, default=1e-2)

    # Sigma sweep
    ap.add_argument("--sigma_min", type=float, default=0.0)
    ap.add_argument("--sigma_max", type=float, default=0.5)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--sigma_idx", type=int, required=True)

    # Work partitioning (shards)
    ap.add_argument("--tot_inits", type=int, default=100)
    ap.add_argument("--runs_per_init", type=int, default=100)
    ap.add_argument("--inits_per_task", type=int, required=True)
    ap.add_argument("--shard_idx", type=int, required=True)

    # Mean / check controls
    ap.add_argument("--check_stride", type=int, default=10)
    ap.add_argument("--mean_refine_steps", type=int, default=2)

    # Seeding
    ap.add_argument("--init_seed", type=int, default=123)
    ap.add_argument("--noise_seed", type=int, default=999)

    # IO
    ap.add_argument("--outdir", type=str, default="out_cc_sweep_homo")
    ap.add_argument("--jobname", type=str, default="cc_sweep_homo")
    return ap.parse_args()

def main():
    args = parse_args()

    # sigma for this task
    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.bins, dtype=np.float64)
    if not (0 <= args.sigma_idx < args.bins):
        raise SystemExit("sigma_idx out of range")
    sigma0 = float(sigmas[args.sigma_idx])

    # shard range
    start = args.shard_idx * args.inits_per_task
    end = min(start + args.inits_per_task, args.tot_inits)
    os.makedirs(args.outdir, exist_ok=True)
    if start >= args.tot_inits:
        print(f"[SKIP] shard {args.shard_idx}: no work (start={start} >= tot_inits={args.tot_inits}).")
        return

    init_indices, times, flags = [], [], []

    for i_global in range(start, end):
        # Distinct seeds per init (keeps X0/noise independent across inits)
        t_hit, conv = mean_traj_first_hit_online(
            n=args.n, d=args.d, T=args.Tmax, dt=args.dt, b=args.b, sigma0=sigma0,
            runs=args.runs_per_init,
            threshold=args.threshold,
            check_stride=args.check_stride,
            mean_refine_steps=args.mean_refine_steps,
            init_seed=args.init_seed + i_global,
            noise_seed=args.noise_seed + i_global,
        )
        init_indices.append(i_global)
        times.append(np.float32(t_hit))
        flags.append(np.int32(1 if conv else 0))

    init_indices = np.asarray(init_indices, dtype=np.int32)
    times = np.asarray(times, dtype=np.float32)
    flags = np.asarray(flags, dtype=np.int32)

    # shard stats over converged inits
    if np.any(flags == 1):
        conv_times = times[flags == 1]
        shard_min_time    = float(np.min(conv_times))
        shard_max_time    = float(np.max(conv_times))
        shard_median_time = float(np.median(conv_times))
    else:
        shard_min_time = shard_max_time = shard_median_time = float("nan")

    # save shard
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
        meantraj_times=times,             # consensus time of INTRINSIC mean trajectory (per init)
        meantraj_converged=flags,         # 1 if converged by Tmax, else 0
        t_hit=times,                      # alias
        hit_mask=flags.astype(bool),      # alias
        # metadata
        n=np.int32(args.n), d=np.int32(args.d),
        b=np.float32(args.b), dt=np.float32(args.dt),
        Tmax=np.float32(args.Tmax), threshold=np.float32(args.threshold),
        check_stride=np.int32(args.check_stride),
        mean_refine_steps=np.int32(args.mean_refine_steps),
    )
    print(f"[OK] Saved shard: {path} (σ={sigma0:.6f}, inits {start}..{end-1}, "
          f"min={shard_min_time:.6g}, median={shard_median_time:.6g}, max={shard_max_time:.6g})")

if __name__ == "__main__":
    # Use jax on CPU by default
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "0")
    main()
