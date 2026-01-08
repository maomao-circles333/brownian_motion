#!/usr/bin/env python3
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os, glob
import pandas as pd

def load_folder(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        raise RuntimeError(f"No CSVs found in {folder}")

    dfs = []
    bad = 0
    for p in paths:
        # skip empty files
        try:
            if os.path.getsize(p) == 0:
                bad += 1
                continue
        except OSError:
            bad += 1
            continue

        # skip files that are mid-write / malformed
        try:
            dfp = pd.read_csv(p)
            if dfp.shape[1] == 0:
                bad += 1
                continue
            dfs.append(dfp)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            bad += 1
            continue

    if not dfs:
        raise RuntimeError(f"All CSVs in {folder} were empty/invalid (count={len(paths)}).")

    df = pd.concat(dfs, ignore_index=True)
    if bad > 0:
        print(f"[WARN] Skipped {bad} empty/invalid CSV(s) in {folder}")
    return df

def aggregate_by_sigma(df):
    # group by sigma0: mean over inits
    df = df.copy()
    df["cp_norm_sq"] = df["cp_norm_Tmax"]**2
    g = df.groupby("sigma0", as_index=False)
    out = g.agg(
        # Optional: square the norm
        cp_sq_mean=("cp_norm_sq", "mean"),
        cp_mean=("cp_norm_Tmax", "mean"),
	cp_sq_std=("cp_norm_sq", "std"),
        n_samples=("cp_norm_sq", "count"),
        cp_std=("cp_norm_Tmax", "std"),
        diam_mean=("diam_Tmax", "mean"),
        conv_rate=("T_hit", lambda x: np.mean(np.array(x) >= 0.0)),
        hit_median=("T_hit", lambda x: np.median([t for t in x if t >= 0.0]) if np.any(np.array(x) >= 0.0) else np.nan),
        comdist_mean=("com_dist_init_to_Tmax", "mean"),
    )
    return out.sort_values("sigma0")

def main():
    outroot = "out_cp_sweep"
    dims = [2, 3, 5, 10, 100]
    n = 100
    beta = 1.0

    plt.figure()
    for d in dims:
        folder = os.path.join(outroot, f"d{d}_n{n}_b{beta:g}")
        df = load_folder(folder)
        agg = aggregate_by_sigma(df)
	# Optional: plot squared
        plt.plot(agg["sigma0"].values, agg["cp_mean"].values, label=f"d={d}")
        # optional band:
        plt.fill_between(agg["sigma0"].values,
         
	                 agg["cp_mean"].values - agg["cp_std"].values,
 	                 agg["cp_mean"].values + agg["cp_std"].values,
  	                 alpha=0.2)

    plt.xlabel(r"$\sigma_0$")
    plt.ylabel(r"$\|c_p(T_{\max})\|$")
    plt.title(r"$\|c_p(T_{\max})\|$ vs $\sigma_0$ (mean over 100 mean-trajectories)"
		rf"($n={n},\,\beta={beta}$)"
		 )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cp_norm_vs_sigma_n=100_b=1_filled.png", dpi=200)
    
if __name__ == "__main__":
    main()

