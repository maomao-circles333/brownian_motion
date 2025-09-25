import os, glob
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "out_cc_sweep"     # npz files
OUT_DIR     = os.path.join(RESULTS_DIR, "plots_pies")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_SIGMAS = [0.1, 0.2, 0.3]
BINS = [(0.0, 30.0), (30.0, 100.0), (100.0, np.inf)]

def load_all_npz(results_dir):
    """Load all npz files and return a list of dicts with keys:
       'sigma', 't_hit' (1D array), 'hit_mask' (1D bool array),
       optional: 'threshold', 'dt', 'Tmax', 'n', 'd', 'b'."""
    entries = []
    for path in glob.glob(os.path.join(results_dir, "*.npz")):
        try:
            with np.load(path, allow_pickle=True) as z:
                # Required keys:
                sigma    = float(z["sigma"])
                t_hit    = np.array(z["t_hit"])
                hit_mask = np.array(z["hit_mask"], dtype=bool)

                # Optional metadata:
                meta = {}
                for key, default in [("threshold", np.nan),
                                     ("dt", np.nan),
                                     ("Tmax", np.nan),
                                     ("n", np.nan),
                                     ("d", np.nan),
                                     ("b", np.nan)]:
                    meta[key] = float(z[key]) if key in z else default

                entries.append({
                    "path": path,
                    "sigma": sigma,
                    "t_hit": t_hit,
                    "hit_mask": hit_mask,
                    **meta
                })
        except Exception as e:
            print(f"[warn] Skipping {path}: {e}")
    return entries

def pick_closest(entries, target_sigma, tol=1e9):
    """Return the subset of entries whose sigma is closest to target_sigma.
       If multiple files share that closest distance, include all of them."""
    if not entries:
        return []

    sigmas = np.array([e["sigma"] for e in entries], dtype=float)
    diffs = np.abs(sigmas - float(target_sigma))
    if diffs.size == 0:
        return []

    mindiff = diffs.min()
    if mindiff > tol:
        return []
    mask = (diffs == mindiff)
    return [entries[i] for i in np.where(mask)[0]]

def aggregate_times(entries_for_sigma):
    """Combine all runs across the selected NPZ files for one sigma.
       Returns (times, hit_mask, meta) where times is 1D array and hit_mask is 1D bool."""
    if not entries_for_sigma:
        return np.array([]), np.array([], dtype=bool), {}

    times_list = []
    hits_list  = []
    # Use the first file’s metadata for annotating the plot
    meta = {
        "sigma": entries_for_sigma[0]["sigma"],
        "threshold": entries_for_sigma[0].get("threshold", np.nan),
        "dt": entries_for_sigma[0].get("dt", np.nan),
        "n": entries_for_sigma[0].get("n", np.nan),
        "d": entries_for_sigma[0].get("d", np.nan),
        "b": entries_for_sigma[0].get("b", np.nan),
        "Tmax": entries_for_sigma[0].get("Tmax", np.nan),
    }

    for e in entries_for_sigma:
        t_hit = e["t_hit"].ravel()
        hit   = e["hit_mask"].ravel()

        # Sanity check: equal length
        L = min(t_hit.shape[0], hit.shape[0])
        times_list.append(t_hit[:L])
        hits_list.append(hit[:L])

    times = np.concatenate(times_list, axis=0) if times_list else np.array([])
    hits  = np.concatenate(hits_list,  axis=0) if hits_list  else np.array([], dtype=bool)

    # Non-converged -> count in the last bin (>= 100).leave times as is
    # use hit_mask to decide when binning
    return times, hits, meta

def bin_counts(times, hit_mask, bins):
    """Return counts for (0,30), [30,100), >=100. Non-hits go to >=100."""
    counts = np.zeros(len(bins), dtype=int)
    # Converged runs:
    if times.size:
        for i, (lo, hi) in enumerate(bins):
            if np.isinf(hi):
                sel = (hit_mask & (times >= lo))
            else:
                sel = (hit_mask & (times > lo) & (times < hi))
            counts[i] = int(np.sum(sel))
    # Add non-converged to the last bin:
    counts[-1] += int(np.sum(~hit_mask))
    return counts

def make_pie(counts, labels, title, subtitle, outfile):
    total = counts.sum()
    if total == 0:
        print(f"[warn] No data to plot for {outfile}")
        return

    explode = [0.03, 0.03, 0.03]
    colors  = ["#b19cd9", "#8a2be2", "#5d3fd3"]  # soft purple -> deep purple
    autopct = lambda p: f"{p:.1f}%\n({int(round(p*total/100))})" if p > 0 else ""

    fig, ax = plt.subplots(figsize=(5.3, 5.3), dpi=150)
    wedges, texts, autotexts = ax.pie(
        counts, explode=explode, labels=labels, colors=colors,
        autopct=autopct, pctdistance=0.75, startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.2)
    )
    ax.axis("equal")  # circle
    ax.set_title(title, fontsize=11, pad=14)
    
    ax.text(0.5, 1.02, subtitle, ha="center", va="bottom", transform=ax.transAxes, fontsize=9, color="#444")

    plt.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {outfile}")

def main():
    entries = load_all_npz(RESULTS_DIR)
    if not entries:
        print(f"[error] No NPZ files found under {RESULTS_DIR}")
        return

    for target in TARGET_SIGMAS:
        group = pick_closest(entries, target_sigma=target)
        if not group:
            print(f"[warn] No files near sigma={target}")
            continue

        times, hits, meta = aggregate_times(group)
        counts = bin_counts(times, hits, BINS)

        # Labels & metadata
        bin_labels = [f"(0, 30)", f"[30, 100)", f"≥ 100"]
        sigma_used = meta.get("sigma", float('nan'))
        dt = meta.get("dt", float('nan'))
        n  = int(meta.get("n", np.nan)) if not np.isnan(meta.get("n", np.nan)) else "?"
        d  = int(meta.get("d", np.nan)) if not np.isnan(meta.get("d", np.nan)) else "?"
        b  = meta.get("b", float('nan'))
        thr= meta.get("threshold", float('nan'))

        title = f"Convergence time distribution\n(closest σ to {target:g} → σ={sigma_used:g})"
        subtitle = f"n={n}, d={d}, β={b:g}, dt={dt:g}, threshold={thr:g}, N={hits.size}"

        outfile = os.path.join(OUT_DIR, f"pie_sigma_{target:.3f}_used_{sigma_used:.3f}.png")
        make_pie(counts, bin_labels, title, subtitle, outfile)

if __name__ == "__main__":
    main()
