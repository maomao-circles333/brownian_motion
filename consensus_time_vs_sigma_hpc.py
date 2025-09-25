#!/usr/bin/env python3
# Intrinsic noise + intrinsic *mean trajectory (across runs)* consensus time.
# To save memory: computes trajectory on server, stores only per-init scalars
import os, argparse, numpy as np
import matplotlib
matplotlib.use("Agg")  # use on cluster
import jax
import jax.numpy as jnp
from jax import random, lax, jit

# ---- helper functions -----
def norm_last(x, eps=1e-7):
    nrm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(nrm, eps)

def softmax_last(a):
    amax = jnp.max(a, axis=-1, keepdims=True)
    z = jnp.exp(a - amax)
    return z / jnp.sum(z, axis=-1, keepdims=True)

def log_map_sphere_generic(U, X, eps=1e-7):
    """
    U: (..., d)      center(s) on S^{d-1}
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
    # Arrange to (..., n, d) with ... = (n,) centers — we want, for each agent i,
    # center U[i] toward R points X_runs_agents[:, i, :]
    Xnr = jnp.swapaxes(X_runs_agents, 0, 1)  # (n, R, d)
    def one_step(Uc):
        V = log_map_sphere_generic(Uc, Xnr)   # Uc: (n,d), Xnr: (n,R,d) -> (n,R,d)
        g = jnp.mean(V, axis=-2)              # (n,d)
        return exp_map_sphere_generic(Uc, step * g)
    def body(Uc, _):
        return one_step(Uc), None
    U_out, _ = lax.scan(body, M_agents, None, length=iters)
    return U_out

def intrinsic_mean_refine_agents_per_run(U_run, X_agents, iters=1, step=1.0):
    """
    Refine per-run mean across *agents*.
    U_run: (R, d)
    X_agents: (R, n, d)
    Returns: (R, d)
    """
    def one_step(Uc):
        V = log_map_sphere_generic(Uc, X_agents)  # Uc: (R,d), X_agents: (R,n,d) -> (R,n,d)
        g = jnp.mean(V, axis=-2)                  # (R,d)
        return exp_map_sphere_generic(Uc, step * g)
    def body(Uc, _):
        return one_step(Uc), None
    U_out, _ = lax.scan(body, U_run, None, length=iters)
    return U_out

def geodesic_diameter_points(P):
    """
    P: (n, d) points on S^{d-1} -> scalar geodesic diameter (rad)
    """
    G = jnp.clip(jnp.einsum("id,jd->ij", P, P), -1.0, 1.0)   # (n,n)
    Theta = jnp.arccos(G)
    return jnp.max(Theta)

# dynamics
def dynamics_batched(X, b=1.0, Q=None, K=None, Vmat=None):
    """
    X: (R, n, d) -> tangent field (R, n, d)
    """
    R, n, d = X.shape
    Q = jnp.eye(d, dtype=X.dtype) if Q is None else jnp.asarray(Q, dtype=X.dtype)
    K = jnp.eye(d, dtype=X.dtype) if K is None else jnp.asarray(K, dtype=X.dtype)
    V = jnp.eye(d, dtype=X.dtype) if Vmat is None else jnp.asarray(Vmat, dtype=X.dtype)

    XQ = X @ Q.T
    XK = X @ K.T
    XV = X @ V.T
    inner = XQ @ jnp.swapaxes(XK, -1, -2)   # (R,n,n)
    w = softmax_last(b * inner)             # (R,n,n)
    Vt = w @ XV                              # (R,n,d)

    proj = jnp.sum(Vt * X, axis=-1, keepdims=True) * X
    return Vt - proj

# ---------------- computes first-hit (to threshold) time of a mean trajectory on cluster
def mean_traj_first_hit_online(
    n, d, T, dt, b, sigma0,
    runs=100,
    threshold=1e-2,
    check_stride=50,
    mean_update_stride=5,
    mean_refine_steps=2,
    init_seed=30, noise_seed=1000,
    NOISE_SCHEME="angle_to_run_mean",   # "angle_to_run_mean" or "isotropic"
    Q=None, K=None, Vmat=None
):
    """
    Simulate R runs; keep two means:
      - U_run (R,d): per-run mean across agents, used for noise amplitude if desired
      - M_agents (n,d): intrinsic mean across runs per agent (the *mean trajectory*)
    At every 'check_stride', refine M_agents and check geodesic diameter.
    Returns: (t_hit: float32, converged: bool)
    """
    steps = int(T / dt)
    dt32, sqrt_dt = jnp.float32(dt), jnp.sqrt(jnp.float32(dt))
    b32 = jnp.float32(b)
    sigma32 = jnp.float32(sigma0)

    key_init  = random.PRNGKey(init_seed)
    key_noise = random.PRNGKey(noise_seed)

    # shared X0 across runs 
    x0 = random.normal(key_init, (n, d), dtype=jnp.float32)
    X0 = norm_last(jnp.broadcast_to(x0, (runs, n, d)))  # (R,n,d)

    # Initialize U_run at t=0: per-run strict mean across agents (a few steps)
    U0 = jnp.mean(X0, axis=1)
    U0 = norm_last(U0)
    U0 = intrinsic_mean_refine_agents_per_run(U0, X0, iters=5, step=1.0)  # warm start, strict solve
    # Initialize M_agents at t=0: strict intrinsic mean across runs per agent
    # do so in small batches
    M0 = intrinsic_mean_refine_runs_per_agent(jnp.mean(jnp.swapaxes(X0,0,1), axis=-2),  # (n,d) euclid start
                                              X0, iters=10, step=1.0)
    # carry: X, U_run, M_agents, key, t_hit, done
    t0   = jnp.array(jnp.inf, dtype=jnp.float32)
    done0= jnp.array(False)

    def step_fn(carry, k):
        X, U, M, key, t_hit, done = carry

        # --- Deterministic part
        dX = dynamics_batched(X, b=b32, Q=Q, K=K, Vmat=Vmat)

        # --- Noise part
        key, sk = random.split(key)
        rnd = random.normal(sk, X.shape, dtype=jnp.float32)
        proj = jnp.sum(rnd * X, axis=-1, keepdims=True) * X
        noise_tan = rnd - proj

        if NOISE_SCHEME == "angle_to_run_mean":
            # amplitude depends on geodesic angle to the per-run mean U (R,d)
            c = jnp.clip(jnp.sum(X * U[:, None, :], axis=-1), -1.0, 1.0)  # (R,n)
            theta = jnp.arccos(c)[..., None]                               # (R,n,1)
            amp = sigma32 * theta
        else:
            amp = sigma32

        X_next = norm_last(X + dt32 * dX + sqrt_dt * (amp * noise_tan))

        # --- at mean_update_stride steps, update U_run (per-run mean across agents) for noise
        def do_mean_U(U_in):
            return intrinsic_mean_refine_agents_per_run(U_in, X_next, iters=1, step=1.0)
        do_u = (k % mean_update_stride) == 0
        U_next = lax.cond(do_u, do_mean_U, lambda u: u, U)

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
            do_check,
            do_check_path,
            lambda args: args,
            operand=(X_next, M, t_hit)
        )

        done_out = done | (t_hit_out < jnp.inf)
        return (X_out, U_next, M_out, key, t_hit_out, done_out), None

    @jit
    def run():
        carry0 = (X0, U0, M0, key_noise, t0, done0)
        (Xf, Uf, Mf, keyf, t_hit, donef), _ = lax.scan(
            step_fn, carry0, xs=jnp.arange(steps, dtype=jnp.int32)
        )
        return t_hit, donef

    t_hit, done = run()
    return float(t_hit), bool(done)

# ---------------- CLI and main (sharded sweep; saves data in NPZ files per shard)
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

    # Work partitioning 
    ap.add_argument("--tot_inits", type=int, default=100)
    ap.add_argument("--runs_per_init", type=int, default=100)
    ap.add_argument("--inits_per_task", type=int, required=True)
    ap.add_argument("--shard_idx", type=int, required=True)

    # Intrinsic mean & check controls
    ap.add_argument("--check_stride", type=int, default=50)
    ap.add_argument("--mean_update_stride", type=int, default=5)
    ap.add_argument("--mean_refine_steps", type=int, default=2)

    # Seeding
    ap.add_argument("--init_seed", type=int, default=123)
    ap.add_argument("--noise_seed", type=int, default=999)

    # IO
    ap.add_argument("--outdir", type=str, default="out_cc_sweep")
    ap.add_argument("--jobname", type=str, default="cc_sweep")
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
            mean_update_stride=args.mean_update_stride,
            mean_refine_steps=args.mean_refine_steps,
            init_seed=args.init_seed + i_global,
            noise_seed=args.noise_seed + i_global,
            NOISE_SCHEME="angle_to_run_mean",  # since we want per-agent noise
            Q=None, K=None, Vmat=None
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
        shard_min_time=np.float32(shard_min_time),
        shard_max_time=np.float32(shard_max_time),
        shard_median_time=np.float32(shard_median_time),
        # metadata
        n=np.int32(args.n), d=np.int32(args.d),
        b=np.float32(args.b), dt=np.float32(args.dt),
        Tmax=np.float32(args.Tmax), threshold=np.float32(args.threshold),
        check_stride=np.int32(args.check_stride),
        mean_update_stride=np.int32(args.mean_update_stride),
        mean_refine_steps=np.int32(args.mean_refine_steps),
    )
    print(f"[OK] Saved shard: {path} (σ={sigma0:.6f}, inits {start}..{end-1}, "
          f"min={shard_min_time:.6g}, median={shard_median_time:.6g}, max={shard_max_time:.6g})")

# --------------Deprecated-------------
# stored the full trajectory data (kept, runs, n, d) per init and then
# computed the mean-trajectory consensus. 
"""
def simulate_intrinsic_noise_jax(...):
    # returns traj_kept: (kept, runs, n, d)
    ...

def consensus_time_of_mean_trajectory(traj_kept, dt, store_stride, threshold):
    # builds mean_traj (kept, n, d) and finds first stored time with diameter <= threshold
    ...
"""

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # keep jax on cpu, no gpu modules needed for now
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "0")
    main()
