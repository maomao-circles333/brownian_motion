from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from numpy.polynomial.legendre import leggauss


def sphere_area(n: int) -> float:
    return 2.0 * math.pi ** ((n + 1) / 2.0) / math.gamma((n + 1) / 2.0)


def _safe_sqrt(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(x, 0.0))

# J_m(z)/|S^m|

def J_over_Sm(m: int, z: np.ndarray) -> np.ndarray:
    """
    J_m(z)/|S^m| where J_m(z) = \int_{S^m} exp(z * omega_1) dS(omega).

    For m=1 (d=3), equals I0(z) and we use numpy.i0
    For m != 1
      J_m(z)/|S^m| = Gamma((m+1)/2) * (2/z)^{(m-1)/2} * I_{(m-1)/2}(z),  z>0,
    with limit=1 at z=0.
    """
    z = np.asarray(z, dtype=float)
    if m == 1:
        return np.i0(z)

    nu = 0.5 * (m - 1)

    out = np.empty_like(z)
    small = np.abs(z) < 1e-12
    out[small] = 1.0

    zz = z[~small]
    try:
        from scipy.special import iv, gamma
        # Avoid underflow
        out[~small] = gamma((m + 1) / 2.0) * (2.0 / zz) ** nu * iv(nu, zz)
    except Exception:
        import mpmath as mp
        g = float(mp.gamma((m + 1) / 2.0))
        vals = []
        for val in zz:
            vals.append(g * (2.0 / val) ** nu * float(mp.besseli(nu, val)))
        out[~small] = np.array(vals, dtype=float)
    return out


#
def build_kernel_matrix(t: np.ndarray, beta: float, d: int) -> np.ndarray:
    """
    Matrix K_{ij} approximating k_d(t_i,t_j)
    """
    t = np.asarray(t, dtype=float)
    r = _safe_sqrt(1.0 - t**2)
    TS = np.outer(t, t)
    RR = np.outer(r, r)

    m = d - 2 
    return np.exp(beta * TS) * J_over_Sm(m, beta * RR)

# --------------------------
# Quadrature for mu_d
# --------------------------
def mu_d_quadrature_legendre(M: int, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use Gauss-Legendre nodes on [-1,1]
    Returns nodes t and weights w_mu approximating the integral of g with respect mu_d.
    """
    t, w = leggauss(M)
    alpha = 0.5 * (d - 3)
    w_mu = sphere_area(d - 2) * w * (1.0 - t**2) ** alpha
    return t.astype(float), w_mu.astype(float)


# Fixed-point map
def apply_Kd(f: np.ndarray, K: np.ndarray, w_mu: np.ndarray) -> np.ndarray:
    """u = K_d[f] at nodes: u_i = Σ_j K_{ij} f_j w_mu_j."""
    return K @ (f * w_mu)

# T = G(K_d)
def T_map(f: np.ndarray, K: np.ndarray, w_mu: np.ndarray, eps: float, beta: float) -> np.ndarray:
    u = apply_Kd(f, K, w_mu)
    a = u / (eps * beta)
    a = a - np.max(a)
    ea = np.exp(a)
    Z = float(np.sum(ea * w_mu))
    return ea / Z


def normalize_density(f: np.ndarray, w_mu: np.ndarray) -> np.ndarray:
    f = np.maximum(f, 0.0)
    Z = float(np.sum(f * w_mu))
    if Z <= 0:
        raise ValueError("Normalization failed.")
    return f / Z


def init_density(t: np.ndarray, w_mu: np.ndarray, d: int,
                 kind: str = "cap", strength: float = 4.0, seed: int = 0) -> np.ndarray:
    if kind == "uniform":
        f = np.ones_like(t) / sphere_area(d - 1)
        return normalize_density(f, w_mu)
    if kind == "cap":
        return normalize_density(np.exp(strength * t), w_mu)
    if kind == "dipole":
        return normalize_density(np.exp(strength * t) + np.exp(-strength * t), w_mu)
    if kind == "random":
        rng = np.random.default_rng(seed)
        z = rng.gamma(shape=1.0, scale=1.0, size=t.shape[0])
        return normalize_density(z, w_mu)
    raise ValueError(f"Unknown init kind: {kind}")


@dataclass
class FixedPointResult:
    t: np.ndarray
    w_mu: np.ndarray
    f: np.ndarray
    u: np.ndarray
    residual_hist: np.ndarray
    iters: int
    converged: bool


def solve_fixed_point(d: int, beta: float, eps: float,
                      M: int = 400, max_iter: int = 5000, tol: float = 1e-12,
                      damping: float = 1.0,
                      init: str = "cap", init_strength: float = 4.0, seed: int = 0) -> FixedPointResult:
    t, w_mu = mu_d_quadrature_legendre(M, d)
    K = build_kernel_matrix(t, beta, d)

    f = init_density(t, w_mu, d, kind=init, strength=init_strength, seed=seed)
    residuals = []

    f_prev = f.copy()
    for k in range(max_iter):
        f_new = T_map(f, K, w_mu, eps, beta)
        if damping != 1.0:
            f_new = (1.0 - damping) * f + damping * f_new

        r = float(np.sum(np.abs(f_new - f) * w_mu))
        residuals.append(r)

        f = f_new
        if r < tol:
            u = apply_Kd(f, K, w_mu)
            return FixedPointResult(t=t, w_mu=w_mu, f=f, u=u,
                                    residual_hist=np.array(residuals),
                                    iters=k + 1, converged=True)

        f_prev = f

    u = apply_Kd(f, K, w_mu)
    return FixedPointResult(t=t, w_mu=w_mu, f=f, u=u,
                            residual_hist=np.array(residuals),
                            iters=max_iter, converged=False)

# Plotting
def order_parameter_m(t: np.ndarray, f: np.ndarray, w_mu: np.ndarray) -> float:
    return float(np.sum(t * f * w_mu))


def plot_profile(result: FixedPointResult, beta: float, eps: float, d: int, outpath: str) -> None:
    t = result.t
    f = result.f
    f_unif = np.ones_like(f) / sphere_area(d - 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(t, f, linewidth=2.0, label=r"$f_\star(t)$")
    ax.plot(t, f_unif, linestyle="--", linewidth=1.6, label="uniform")
    ax.set_xlabel(r"$t=\langle x,n\rangle$")
    ax.set_ylabel(r"$f(t)$")
    ax.set_title(fr"Fixed point solution, $d={d}$, $\beta={beta:g}$, $\varepsilon={eps:g}$")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_potential(result: FixedPointResult, beta: float, eps: float, d: int, outpath: str) -> None:
    t = result.t
    u = result.u

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(t, u, linewidth=2.0, label=r"$K_d[f_\star](t)$")
    ax.set_xlabel(r"$t=\langle x,n\rangle$")
    ax.set_ylabel(r"$K_d[f_\star](t)$")
    ax.set_title(fr"Mean-field potential, $d={d}$, $\beta={beta:g}$, $\varepsilon={eps:g}$")
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_convergence(result: FixedPointResult, beta: float, eps: float, d: int, outpath: str) -> None:
    r = result.residual_hist
    it = np.arange(1, r.size + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.semilogy(it, np.maximum(r, 1e-300), linewidth=2.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|f^{(k+1)}-f^{(k)}\|_{L^1(\mu_d)}$")
    ax.set_title(fr"Fixed-point residuals, $d={d}$, $\beta={beta:g}$, $\varepsilon={eps:g}$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def plot_sphere_density_d3(result: FixedPointResult, beta: float, eps: float, outpath: str,
                           n_theta: int = 200, n_phi: int = 400) -> None:
    """
    d=3 visualization: color the sphere S^2 by rho(x)=f(x·n), with n taken as the z-axis.
    Adds a colorbar and explicitly indicates which pole has the maximum.
    """
    import matplotlib as mpl

    t_nodes = result.t
    f_nodes = result.f

    def f_of_t(tval):
        return np.interp(tval, t_nodes, f_nodes)

    # Sphere grid
    theta = np.linspace(0.0, math.pi, n_theta)
    phi   = np.linspace(0.0, 2.0 * math.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)

    rho = f_of_t(Z)

    rho_min = float(np.min(rho))
    rho_max = float(np.max(rho))
    rho_rng = rho_max - rho_min

    cmap = plt.cm.berlin

    # normalization
    scale = max(1.0, abs(rho_min), abs(rho_max))
    tiny = 1e-12 * scale

    if rho_rng <= tiny:
        vmin, vmax = rho_min - tiny, rho_max + tiny
        uniform_note = " (uniform)"
    else:
        vmin, vmax = rho_min, rho_max
        uniform_note = ""

    norm = Normalize(vmin=vmin, vmax=vmax)
    facecolors = cmap(norm(rho))

    # --- figure ---
    fig = plt.figure(figsize=(7.4, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        rstride=1, cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,   
    )

    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    # Set colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  ## required by mpl
    cbar = fig.colorbar(
        sm, ax=ax,
        shrink=0.75, pad=0.03, fraction=0.05
    )
    cbar.set_label(r"$\rho(x)=f_\star(\langle x,n\rangle)$")

    rho_north = float(f_of_t(1.0))   # z = +1  (t = +1)
    rho_south = float(f_of_t(-1.0))  # z = -1  (t = -1)

    if rho_north > rho_south + tiny:
        pole_note = "max at north pole (t = +1)"
    elif rho_south > rho_north + tiny:
        pole_note = "max at south pole (t = −1)"
    else:
        pole_note = "equal at both poles"

    ax.set_title(
        fr"$\rho(x)=f_\star(\langle x,n\rangle)$ on $S^2$,  "
        fr"$\beta={beta:g}$, $\varepsilon={eps:g}$"
        + "\n"
        + fr"$\min \rho={rho_min:.6g}$, $\max \rho={rho_max:.6g}$"
        + uniform_note
        + "\n"
        + pole_note
    )

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_sphere_density_d3_deprecated(result: FixedPointResult, beta: float, eps: float, outpath: str,
                           n_theta: int = 200, n_phi: int = 400) -> None:
    """
    d=3 visualization: color the sphere S^2 by rho(x)=f(x·n), with n taken as the z-axis.

    """
    t_nodes = result.t
    f_nodes = result.f

    def f_of_t(tval: np.ndarray) -> np.ndarray:
        # linear interpolation on the fixed-point grid
        return np.interp(tval, t_nodes, f_nodes)

    theta = np.linspace(0.0, math.pi, n_theta)
    phi   = np.linspace(0.0, 2.0 * math.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)

    rho = f_of_t(Z)
    rho_min = float(np.min(rho))
    rho_max = float(np.max(rho))
    rho_rng = rho_max - rho_min

    cmap = plt.cm.berlin

    scale = max(1.0, abs(rho_max), abs(rho_min))
    if rho_rng <= 1e-12 * scale:
        facecolors = cmap(np.full_like(rho, 0.5))  # constant color
        uniform_note = " (uniform)"
    else:
        norm = Normalize(vmin=rho_min, vmax=rho_max)
        facecolors = cmap(norm(rho))
        uniform_note = ""

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, Z,
        rstride=1, cstride=1,
        facecolors=facecolors,
        linewidth=0.0,
        antialiased=False,
        shade=False,   
    )

    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.set_title(
        fr"$\rho(x)=f_\star(\langle x,n\rangle)$ on $S^2$,  $\beta={beta:g}$, $\varepsilon={eps:g}$" + "\n"
        + fr"$\min \rho={rho_min:.16g}$, $\max \rho={rho_max:.16g}$" + uniform_note
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--M", type=int, default=400)
    ap.add_argument("--max_iter", type=int, default=5000)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--damping", type=float, default=1.0)
    ap.add_argument("--init", type=str, default="random", choices=["uniform", "cap", "dipole", "random"])
    ap.add_argument("--init_strength", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="fixed_point_out_beta1_eps0p3")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    result = solve_fixed_point(
        d=args.d,
        beta=args.beta,
        eps=args.eps,
        M=args.M,
        max_iter=args.max_iter,
        tol=args.tol,
        damping=args.damping,
        init=args.init,
        init_strength=args.init_strength,
        seed=args.seed,
    )

    m = order_parameter_m(result.t, result.f, result.w_mu)
    print(f"[fixed point] converged={result.converged} iters={result.iters} |m|={abs(m):.6e}")

    plot_profile(result, args.beta, args.eps, args.d, os.path.join(args.outdir, "profile_f.pdf"))
    plot_potential(result, args.beta, args.eps, args.d, os.path.join(args.outdir, "potential_Kf.pdf"))
    plot_convergence(result, args.beta, args.eps, args.d, os.path.join(args.outdir, "convergence_residual.pdf"))

    if args.d == 3:
        plot_sphere_density_d3(result, args.beta, args.eps, os.path.join(args.outdir, "sphere_density.png"))


if __name__ == "__main__":
    main()
