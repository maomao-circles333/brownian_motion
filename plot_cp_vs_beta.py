#!/usr/bin/env python3
import os, glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_folder(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        raise RuntimeError(f"No CSVs found in {folder}")

    dfs = []
    skipped = 0
    for p in paths:
        try:
            if os.path.getsize(p) == 0:
                skipped += 1
                continue
        except OSError:
            skipped += 1
            continue
        try:
            dfs.append(pd.read_csv(p))
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            skipped += 1
            continue

    if not dfs:
        raise RuntimeError(f"All CSVs in {folder} were empty/invalid.")
    if skipped:
        print(f"[WARN] Skipped {skipped} empty/invalid CSV(s) in {folder}")

    return pd.concat(dfs, ignore_index=True)

def aggregate_by_beta(df):
    g = df.groupby("beta", as_index=False)
    out = g.agg(
        cp_mean=("cp_norm_Tmax", "mean"),
        cp_std=("cp_norm_Tmax", "std"),
        n_samples=("cp_norm_Tmax", "count"),
    ).sort_values("beta")
    return out

def main():
    outroot = "out_cp_beta_sweep"
    n = 100

    sigmas = [0.01, 0.05, 0.2]
    dims = [2, 5, 20, 100]

    DO_FILL = True

    for sigma0 in sigmas:
        sig_tag = f"{sigma0:.3f}".rstrip("0").rstrip(".")
        plt.figure()

        for d in dims:
            folder = os.path.join(outroot, f"d{d}_n{n}_sig{sig_tag}")
            if not os.path.isdir(folder):
                print(f"[SKIP] missing folder: {folder}")
                continue

            df = load_folder(folder)
            if "beta" not in df.columns or "cp_norm_Tmax" not in df.columns:
                print(f"[SKIP] missing columns in {folder}")
                continue

            agg = aggregate_by_beta(df)
            x = agg["beta"].to_numpy()
            y = agg["cp_mean"].to_numpy()
            s = np.nan_to_num(agg["cp_std"].to_numpy(), nan=0.0)

            plt.plot(x, y, label=f"d={d}")
            if DO_FILL:
                plt.fill_between(x, y - s, y + s, alpha=0.2)

            print(f"[INFO] sigma={sigma0} d={d}: betas={len(x)}, "
                  f"samples per beta min/med/max="
                  f"{agg['n_samples'].min():.0f}/"
                  f"{agg['n_samples'].median():.0f}/"
                  f"{agg['n_samples'].max():.0f}")

        plt.xlabel(r"$\beta$")
        plt.ylabel(r"$\|c_{\rho}(T_{\max})\|$")
        plt.title(rf"$\|c_{{\rho}}(T_{{\max}})\|$ vs $\beta$  ($n={n},\ \sigma_0={sigma0}$)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        outpath = os.path.join(outroot, f"cp_norm_vs_beta_sig{sig_tag}.png")
        plt.savefig(outpath, dpi=200)
        print("Saved:", os.path.abspath(outpath))

if __name__ == "__main__":
    main()
