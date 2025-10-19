#!/usr/bin/env python3
import os, re, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAME_RE = re.compile(r"^(?P<job>.+)_sig(?P<sigidx>\d+)_shard(?P<shard>\d+)\.npz$")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="out_cc_drift")
    ap.add_argument("--jobname", type=str, default="cc_drift",
                    help="Loads files like cc_drift_sig###_shard###.npz")
    ap.add_argument("--plotdir", type=str, default="plots")
    ap.add_argument("--plot_prefix", type=str, default="cc_drift_band",
                    help="Prefix for output CSV/fig files; suffixes _geodesic_deg and _euclidean are added")
    ap.add_argument("--color", type=str, default="#32a852", help="Color for band/median")
    ap.add_argument("--alpha", type=float, default=0.25, help="Opacity for shaded band")
    ap.add_argument("--band", choices=["p10p90","iqr"], default="p10p90",
                    help="p10p90 = 10–90%% band (default); iqr = Q1–Q3 (code kept commented)")
    ap.add_argument("--verbose", action="store_true")
    # Annotation defaults
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--tmax", type=float, default=500.0)
    ap.add_argument("--threshold", type=float, default=1e-2)
    return ap.parse_args()

def load_shard(path):
    base = os.path.basename(path)
    m = NAME_RE.match(base)
    if not m:
        raise ValueError(f"Bad filename: {base}")
    sidx = int(m.group("sigidx"))
    d = np.load(path)
    sigma = float(d["sigma"]) if "sigma" in d.files else np.nan
    idxs  = d["init_indices"].astype(int)
    drifts = d["drift_values"].astype(float)   # radians
    return dict(sigma_idx=sidx, sigma=sigma, init_indices=idxs, drifts=drifts)

def stats(values, mode="p10p90"):
    """Return (low, median, high) per requested band on a 1D array."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    if mode == "p10p90":
        p10, med, p90 = np.percentile(v, [10, 50, 90])
        return float(p10), float(med), float(p90)
    # --- IQR kept for reference ---
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return float(q1), float(med), float(q3)

def aggregate_per_sigma(args):
    """Read all shards and return dict mapping sigma_idx -> dict with:
       'sigma', 'geo_rad_vals' (array of drifts in radians)
    """
    pattern = os.path.join(args.outdir, f"{args.jobname}_sig*_shard*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files found matching {pattern}")

    per_sigma = {}
    for f in files:
        try:
            rec = load_shard(f)
        except Exception as e:
            print(f"[WARN] skipping {os.path.basename(f)}: {e}")
            continue
        sidx = rec["sigma_idx"]
        per_sigma.setdefault(sidx, {"sigma": [], "geo_rad_vals": []})
        per_sigma[sidx]["sigma"].append(rec["sigma"])
        per_sigma[sidx]["geo_rad_vals"].append((rec["init_indices"], rec["drifts"]))

    # consolidate maps and relax sigma consistency to a warning
    for sidx, blob in per_sigma.items():
        sigs = [s for s in blob["sigma"] if not np.isnan(s)]
        if sigs and (np.max(sigs) - np.min(sigs) > 1e-9):
            print(f"[WARN] sigma_idx={sidx:03d}: mixed sigma values "
                  f"(min={np.min(sigs):.8g}, max={np.max(sigs):.8g}); using mean.")
        sigma = float(np.mean(sigs)) if sigs else np.nan
        # deduplicate by init index
        m = {}
        for idxs, vals in blob["geo_rad_vals"]:
            for k, v in zip(idxs, vals):
                m[int(k)] = float(v)
        vals = np.array([m[k] for k in sorted(m.keys())], dtype=float)
        vals = vals[np.isfinite(vals)]
        per_sigma[sidx] = {"sigma": sigma, "geo_rad_vals": vals}
    return per_sigma

def make_series(per_sigma, mode):
    """Compute (sigma, low, med, high) arrays for both metrics:
       - geodesic in degrees (convert samples, then take percentiles)
       - euclidean chord length (convert samples via 2*sin(theta/2), then percentiles)
    """
    rows_geo = []
    rows_euc = []
    for sidx in sorted(per_sigma.keys()):
        sigma = per_sigma[sidx]["sigma"]
        rad = per_sigma[sidx]["geo_rad_vals"]  # radians per init

        if rad.size == 0:
            rows_geo.append((sigma, sidx, np.nan, np.nan, np.nan))
            rows_euc.append((sigma, sidx, np.nan, np.nan, np.nan))
            continue

        # Geodesic in degrees
        deg = np.rad2deg(rad)
        g_low, g_med, g_high = stats(deg, mode=mode)
        rows_geo.append((sigma, sidx, g_low, g_med, g_high))

        # Euclidean chord distance on unit sphere
        euc = 2.0 * np.sin(0.5 * rad)
        e_low, e_med, e_high = stats(euc, mode=mode)
        rows_euc.append((sigma, sidx, e_low, e_med, e_high))

    def pack(rows):
        arr = np.array(rows, dtype=float)
        sigmas = arr[:, 0]
        sidxs  = arr[:, 1].astype(int)
        lows, meds, highs = arr[:, 2], arr[:, 3], arr[:, 4]
        order = np.argsort(np.where(np.isnan(sigmas), sidxs, sigmas))
        return sigmas[order], sidxs[order], lows[order], meds[order], highs[order]

    return pack(rows_geo), pack(rows_euc)

def annotate(ax, args):
    info = (fr"$n={args.n}$, $d={args.d}$, $\beta={args.beta}$, $\Delta t={args.dt}$"
            + "\n" + fr"$T_{{\max}}={args.tmax}$, $\varepsilon={args.threshold}$")
    ax.text(0.98, 0.98, info, transform=ax.transAxes, ha="right", va="top", fontsize=9)

def save_csv(path, sigmas, lows, meds, highs, band):
    header = "sigma,P10,median,P90" if band == "p10p90" else "sigma,Q1,median,Q3"
    np.savetxt(path, np.column_stack([sigmas, lows, meds, highs]),
               delimiter=",", header=header, comments="")

def plot_band(path_png, path_pdf, sigmas, lows, meds, highs, args,
              ylabel, title_suffix, label_band):
    plt.figure(figsize=(7.2, 4.4))
    plt.fill_between(sigmas, lows, highs, color=args.color, alpha=args.alpha, label=label_band)
    plt.plot(sigmas, meds, color=args.color, lw=2.2, label="Median")
    plt.xlabel(r"$\sigma$")
    plt.ylabel(ylabel)
    plt.title(title_suffix)
    annotate(plt.gca(), args)
    plt.tight_layout()
    plt.savefig(path_png, dpi=200); print(f"[OK] Saved: {path_png}")
    plt.savefig(path_pdf);          print(f"[OK] Saved: {path_pdf}")
    plt.close()

def main():
    args = parse_args()

    # Ensure output dirs exist (plots dir was missing before)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.plotdir, exist_ok=True)

    per_sigma = aggregate_per_sigma(args)

    # Build series for both metrics
    (sig_g, sidx_g, low_g, med_g, high_g), (sig_e, sidx_e, low_e, med_e, high_e) = \
        make_series(per_sigma, mode=args.band)

    # Save CSVs
    band_lab = "P10-P90" if args.band == "p10p90" else "Q1-Q3"
    csv_geo = os.path.join(args.plotdir, f"{args.plot_prefix}_deg_n=32d=3.csv")
    csv_euc = os.path.join(args.plotdir, f"{args.plot_prefix}_euclidean_n=32d=3.csv")
    save_csv(csv_geo, sig_g, low_g, med_g, high_g, args.band)
    save_csv(csv_euc, sig_e, low_e, med_e, high_e, args.band)
    print(f"[OK] Saved: {csv_geo}")
    print(f"[OK] Saved: {csv_euc}")

    # Plots
    label_band = "10–90% band" if args.band == "p10p90" else "IQR (Q1–Q3)"

    # Geodesic (degrees)
    png_g = os.path.join(args.plotdir, f"{args.plot_prefix}_deg_n=32d=3_cbo.png")
    pdf_g = os.path.join(args.plotdir, f"{args.plot_prefix}_deg_n=32d=3.pdf")
    plot_band(
        png_g, pdf_g, sig_g, low_g, med_g, high_g, args,
        ylabel="Drift from deterministic consensus (degrees)",
        title_suffix=f"Drift vs. $\\sigma$ — Geodesic ({band_lab}) with median",
        label_band=label_band
    )

    # Euclidean chord (unit sphere)
    png_e = os.path.join(args.plotdir, f"{args.plot_prefix}_euclidean_n=32d=3.png")
    pdf_e = os.path.join(args.plotdir, f"{args.plot_prefix}_euclidean_n=32d=3.pdf")
    plot_band(
        png_e, pdf_e, sig_e, low_e, med_e, high_e, args,
        ylabel="Drift from deterministic consensus (Euclidean chord, unit sphere)",
        title_suffix=f"Drift vs. $\\sigma$ — Euclidean chord ({band_lab}) with median",
        label_band=label_band
    )

if __name__ == "__main__":
    main()
