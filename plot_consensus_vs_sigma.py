#!/usr/bin/env python3
import os
import argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

def load_npzs(indir, jobname):
    pat = os.path.join(indir, f"{jobname}_sig*_shard*.npz")
    files = sorted(glob(pat))
    if not files:
        raise SystemExit(f"No NPZ files found at pattern: {pat}")
    return files

def read_shard(path):
    """
    Expected keys:
      NEW: 'sigma', 't_hit' (float), 'hit_mask' (bool)
      OLD: 'meantraj_times', 'meantraj_converged'
    Returns:
      sigma (float),
      t_hit (1D float array),
      hit_mask (1D bool array),
      meta (dict with n,d,b,dt,Tmax,threshold)
    """
    d = np.load(path, allow_pickle=True)
    sigma = float(d["sigma"])

    if "t_hit" in d.files and "hit_mask" in d.files:
        t_hit = np.array(d["t_hit"], dtype=float)
        hit_mask = np.array(d["hit_mask"], dtype=bool)
        if t_hit.ndim > 1:      # (inits, runs) -> flatten
            t_hit = t_hit.reshape(-1)
            hit_mask = hit_mask.reshape(-1)
    else:
        # deprecated format: per-init stats
        t_hit = np.array(d["meantraj_times"], dtype=float).reshape(-1)
        hit_mask = np.array(d["meantraj_converged"], dtype=bool).reshape(-1)

    meta = dict(
        n = int(d.get("n", -1)),
        d = int(d.get("d", -1)),
        b = float(d.get("b", np.nan)),
        dt = float(d.get("dt", np.nan)),
        #Tmax = float(d.get("Tmax", np.nan)),
	Tmax = 60,
        threshold = float(d.get("threshold", np.nan)),
    )
    return sigma, t_hit, hit_mask, meta

def aggregate_by_sigma(files):
    by_sigma = {}
    meta_one = None
    Tmax_seen = None
    total_runs = 0
    total_converged = 0

    for p in files:
        sigma, t_hit, hit_mask, meta = read_shard(p)
        by_sigma.setdefault(sigma, []).append((t_hit, hit_mask))
        if meta_one is None:
            meta_one = meta
        # keep largest Tmax (just for y-limit)
        if np.isfinite(meta.get("Tmax", np.nan)):
            Tmax_seen = float(meta["Tmax"]) if Tmax_seen is None else max(Tmax_seen, float(meta["Tmax"]))
        total_runs      += t_hit.size
        total_converged += int(hit_mask.sum())

    return by_sigma, meta_one, Tmax_seen, total_runs, total_converged

def compute_stats(by_sigma):
    """
    For each sigma, gather all *converged* times and compute median/IQR.
    Returns:
      S   : (K,) sorted sigma values
      med : (K,) median time (NaN if no converged time at that sigma)
      q25 : (K,) 25th percentile (NaN if none)
      q75 : (K,) 75th percentile (NaN if none)
      conv_rate : (K,) fraction converged (0..1) at that sigma
      counts    : (K,2) array of [num_converged, num_total]
    """
    S = np.array(sorted(by_sigma.keys()), dtype=float)
    med = np.full(S.shape, np.nan, dtype=float)
    q25 = np.full(S.shape, np.nan, dtype=float)
    q75 = np.full(S.shape, np.nan, dtype=float)
    conv_rate = np.zeros_like(S, dtype=float)
    counts = np.zeros((S.size, 2), dtype=int)

    for i, s in enumerate(S):
        # collect across all shards for this sigma
        times_s = []
        num_conv = 0
        num_tot  = 0
        for (t_hit, hit_mask) in by_sigma[s]:
            num_conv += int(hit_mask.sum())
            num_tot  += int(hit_mask.size)
            # only include converged times in stats
            if t_hit.size:
                times_s.extend(t_hit[hit_mask].tolist())
        counts[i, 0] = num_conv
        counts[i, 1] = num_tot
        conv_rate[i] = (num_conv / num_tot) if num_tot else np.nan

        if times_s:
            arr = np.asarray(times_s, float)
            med[i] = np.median(arr)
            q25[i] = np.percentile(arr, 95)
            q75[i] = np.percentile(arr, 5)

    return S, med, q25, q75, conv_rate, counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="out_cc_sweep",
                    help="Directory containing <jobname>_sig*_shard*.npz")
    ap.add_argument("--jobname", type=str, default="cc_sweep",
                    help="Filename prefix used when saving shards")
    ap.add_argument("--out_png_a", type=str, default="consensus_time_vs_sigma.png",
                    help="Output path for the median/IQR plot")
    args = ap.parse_args()

    files = load_npzs(args.indir, args.jobname)
    by_sigma, meta, Tmax_seen, total_runs, total_converged = aggregate_by_sigma(files)
    S, med, q25, q75, conv_rate, counts = compute_stats(by_sigma)

    # -------- Plot: consensus time vs sigma (median + IQR) --------
    fig, ax = plt.subplots(figsize=(8.2, 4.7))

    have_iqr = np.isfinite(q25) & np.isfinite(q75)
    if np.any(have_iqr):
        ax.fill_between(S[have_iqr], q25[have_iqr], q75[have_iqr],
                        alpha=0.35, label="trimmed(5%-95%)", linewidth=0)

    have_med = np.isfinite(med)
    if np.any(have_med):
        ax.plot(S[have_med], med[have_med], lw=2.2, label="Median")

    # axes cosmetics
    xpad = 0.01 * (S.max() - S.min() + 1e-9) if S.size else 0.01
    ax.set_xlim((S.min() - xpad) if S.size else -0.01, (S.max() + xpad) if S.size else 0.51)
    if Tmax_seen is not None and np.isfinite(Tmax_seen):
        ax.set_ylim(0, Tmax_seen * 1.02)

    ax.set_xlabel(r"$\sigma_0$")
    ax.set_ylabel("Consensus time (s)")
    ax.set_title("Consensus time vs noise (median and trimmed data across shards/runs)")

    # Caption with metadata (if present)
    if meta:
        cap = f"(n={meta['n']}, d={meta['d']}, β={meta['b']:.3g}, dt={meta['dt']:.3g}, threshold={meta['threshold']:.3g} rad)"
        fig.text(0.5, -0.02, cap, ha="center", va="top")

    ax.grid(True, alpha=0.3)

    # legend only if something is labeled
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_png_a) or ".", exist_ok=True)
    fig.savefig(args.out_png_a, dpi=220, bbox_inches="tight")
    print(f"[saved] {args.out_png_a}")

    # -------- Console summary --------
    print("\nSigma  median   q25     q75     conv%   (conv/total)")
    for s, m, a, b, r, (c, t) in zip(S, med, q25, q75, conv_rate, counts):
        ms = f"{m:7.3f}" if np.isfinite(m) else "   NaN  "
        aS = f"{a:7.3f}" if np.isfinite(a) else "   NaN  "
        bS = f"{b:7.3f}" if np.isfinite(b) else "   NaN  "
        rp = f"{100*r:6.2f}%" if np.isfinite(r) else "  NaN  "
        print(f"{s:4.3f}  {ms}  {aS}  {bS}  {rp}   ({c}/{t})")

    print(f"\nTOTAL: converged {total_converged} / {total_runs} "
          f"({100.0*total_converged/total_runs:.2f}% )")

if __name__ == "__main__":
    main()
