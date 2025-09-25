#!/usr/bin/env python3
import os, argparse, numpy as np
import matplotlib.pyplot as plt
from glob import glob

def load_npzs(indir, jobname):
    pat = os.path.join(indir, f"{jobname}_sig*_shard*.npz")
    files = sorted(glob(pat))
    if not files:
        raise SystemExit(f"No NPZ files found at pattern: {pat}")
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="out_cc_sweep")
    ap.add_argument("--jobname", type=str, default="cc_sweep")
    ap.add_argument("--out_png_a", type=str, default="consensus_time_vs_sigma.png")
    ap.add_argument("--out_png_b", type=str, default="t_hit_pie.png")
    args = ap.parse_args()

    files = load_npzs(args.indir, args.jobname)

    # Group by sigma
    by_sigma = {}
    meta = None
    Tmax_seen = None

    for f in files:
        d = np.load(f, allow_pickle=True)
        s = float(d["sigma"])
        if "t_hit" in d.files and "hit_mask" in d.files:
            t_hit = np.array(d["t_hit"]).astype(float)
            hit_mask = np.array(d["hit_mask"]).astype(bool)
            # If shape is (inits, runs), flatten
            if t_hit.ndim > 1:
                t_hit = t_hit.reshape(-1)
                hit_mask = hit_mask.reshape(-1)
        else:
            # per_init (deprecated)
            t_hit = np.array(d["meantraj_times"]).astype(float).reshape(-1)
            hit_mask = np.array(d["meantraj_converged"]).astype(bool).reshape(-1)

        by_sigma.setdefault(s, []).append((t_hit, hit_mask))

        if meta is None:
            meta = {
                "n": int(d["n"]),
                "d": int(d["d"]),
                "b": float(d["b"]),
                "dt": float(d["dt"]),
                "threshold": float(d["threshold"]),
            }
            Tmax_seen = float(d["Tmax"])

    S = np.array(sorted(by_sigma.keys()))
    med = np.full_like(S, np.nan, dtype=float)
    q25 = np.full_like(S, np.nan, dtype=float)
    q75 = np.full_like(S, np.nan, dtype=float)

    # collect all converged times for the pie plot
 
    # -------- Plot: time vs sigma --------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    have = np.isfinite(q25) & np.isfinite(q75)
    if np.any(have):
        ax.fill_between(S[have], q25[have], q75[have], alpha=0.35, label="IQR (25–75%)")
    have_m = np.isfinite(med)
    if np.any(have_m):
        ax.plot(S[have_m], med[have_m], lw=2.0, label="Median")

    ax.set_xlim(0.0 - 0.005, 0.5 + 0.005)
    ax.set_ylim(0, (Tmax_seen or 0) * 1.02)
    ax.set_xlabel(r"$\sigma_0$")
    ax.set_ylabel("Consensus time (s)")
    ax.set_title("Consensus time vs noise")

    # Caption with metadata
    if meta:
        cap = f"(n={meta['n']}, d={meta['d']}, β={meta['b']:.3g}, dt={meta['dt']:.3g}, threshold={meta['threshold']:.3g} rad)"
        fig.text(0.5, -0.02, cap, ha="center", va="top")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(args.out_png_a, dpi=220, bbox_inches="tight")
    print(f"[saved] {args.out_png_a}")

    
if __name__ == "__main__":
    main()
