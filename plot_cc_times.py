#!/usr/bin/env python3
import os, glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_shard(path):
    d = np.load(path, allow_pickle=False)
    # tolerate both old/new field names
    if "meantraj_times" in d.files:
        times = np.array(d["meantraj_times"])
    elif "t_hit" in d.files:
        times = np.array(d["t_hit"])
    else:
        raise KeyError(f"{path}: no 'meantraj_times' or 't_hit'")

    if "meantraj_converged" in d.files:
        flags = np.array(d["meantraj_converged"]).astype(bool)
    elif "hit_mask" in d.files:
        flags = np.array(d["hit_mask"]).astype(bool)
    else:
        # derive from times if not present
        flags = np.isfinite(times)

    if "sigma" in d.files:
        sigma = float(np.array(d["sigma"]).item())
    else:
        # last resort: derive from filename like *_sig072_*
        base = os.path.basename(path)
        sigma = None
        if "_sig" in base:
            try:
                sig_idx = int(base.split("_sig")[1].split("_")[0])
                # if bins metadata exists, we could map; otherwise keep None
                sigma = None
            except Exception:
                pass

    meta = {}
    for k in ["n","d","b","dt","Tmax","threshold"]:
        if k in d.files: meta[k] = np.array(d[k]).item()
    for k in ["check_stride","mean_update_stride","mean_refine_steps","sigma_idx","shard_idx"]:
        if k in d.files: meta[k] = int(np.array(d[k]).item())
    return sigma, meta.get("sigma_idx", None), times, flags, meta

def main(outdir="out_cc_sweep", out_png="consensus_time_vs_sigma.png"):
    files = sorted(glob.glob(os.path.join(outdir, "*.npz")))
    if not files:
        print(f"[warn] No .npz files found in {outdir}")
        return

    # aggregate by sigma (float if present), else by sigma_idx
    by_key = {}
    have_sigma_float = True
    for f in files:
        try:
            sigma, sigma_idx, times, flags, meta = load_shard(f)
        except Exception as e:
            print(f"[skip] {f}: {e}")
            continue
        key = sigma if sigma is not None else sigma_idx
        if key is None:
            print(f"[skip] {f}: cannot infer sigma or sigma_idx")
            continue
        if sigma is None:
            have_sigma_float = False
        if key not in by_key:
            by_key[key] = {"times": [], "flags": [], "metas": []}
        by_key[key]["times"].append(times)
        by_key[key]["flags"].append(flags)
        by_key[key]["metas"].append(meta)

    if not by_key:
        print("[warn] No usable shards after parsing. Nothing to plot.")
        return

    # combine per key
    keys_sorted = sorted(by_key.keys())
    sig_axis = []
    frac_conv = []
    med = []
    q25 = []
    q75 = []
    first_meta = None

    for k in keys_sorted:
        T = np.concatenate(by_key[k]["times"])
        F = np.concatenate(by_key[k]["flags"]).astype(bool)
        # meta for caption (just take first present)
        if first_meta is None:
            mlist = by_key[k]["metas"]
            if mlist: first_meta = mlist[0]

        # convergence fraction
        frac_conv.append(np.mean(F) if F.size else np.nan)

        # times among converged and finite
        mask = F & np.isfinite(T)
        if np.any(mask):
            tconv = T[mask]
            med.append(np.median(tconv))
            q25.append(np.percentile(tconv, 25))
            q75.append(np.percentile(tconv, 75))
        else:
            med.append(np.nan); q25.append(np.nan); q75.append(np.nan)

        # x-axis: prefer actual sigma if available, else use rank order (index)
        sig_axis.append(float(k) if have_sigma_float else len(sig_axis))

    sig_axis = np.array(sig_axis, dtype=float)
    med = np.array(med, dtype=float)
    q25 = np.array(q25, dtype=float)
    q75 = np.array(q75, dtype=float)
    frac_conv = np.array(frac_conv, dtype=float)

    # sort by sigma axis
    order = np.argsort(sig_axis)
    sig_axis, med, q25, q75, frac_conv = sig_axis[order], med[order], q25[order], q75[order], frac_conv[order]

    # plot
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    any_artist = False

    have_band = np.isfinite(q25) & np.isfinite(q75)
    if np.any(have_band):
        ax.fill_between(sig_axis[have_band], q25[have_band], q75[have_band], alpha=0.5, label="IQR (25–75%)")
        any_artist = True

    have_med = np.isfinite(med)
    if np.any(have_med):
        ax.plot(sig_axis[have_med], med[have_med], lw=2.0, label="Median time")
        any_artist = True

    # light gray spans where ANY run failed at that sigma
    if np.any(frac_conv < 1.0):
        dx = (sig_axis.max() - sig_axis.min()) / max(len(sig_axis) - 1, 1) if len(sig_axis) > 1 else 0.02
        for s, fr in zip(sig_axis, frac_conv):
            if fr < 1.0:
                ax.axvspan(s - 0.5*dx, s + 0.5*dx, color="lightgray", alpha=0.25, lw=0)

    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("Consensus time (mean trajectory)")
    ttl = "Consensus time vs σ"
    if first_meta:
        bits = []
        for k in ["n","d","b","dt","threshold","Tmax"]:
            if k in first_meta:
                val = first_meta[k]
                if isinstance(val, float):
                    bits.append(f"{k}={val:g}")
                else:
                    bits.append(f"{k}={val}")
        if bits:
            ttl += "   [" + ", ".join(bits) + "]"
    ax.set_title(ttl)
    ax.grid(True, alpha=0.3)
    if any_artist:
        ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    out_png = os.path.join(outdir, "consensus_time_vs_sigma.png")
    fig.savefig(out_png, dpi=220)
    print(f"[saved] {out_png}")

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "out_cc_sweep"
    main(outdir=outdir)
