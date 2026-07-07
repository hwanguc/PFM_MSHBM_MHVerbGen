"""
plot_group_network_size.py

Group-level visualisation of MS-HBM individual network sizes. Produces one
horizontal bar chart per group (DLD, HSL, TD) -- each bar = group mean % cortical
surface for a network, error bar = SEM across subjects -- plus a combined
3-group comparison chart.

Variant selects which MS-HBM set to plot (full / icafix).

Inputs:
    results/network_size_<variant>/group_network_size_long.csv
    res0urces/networks_meta.csv  (network id/label/colour)

Outputs (in results/network_size_<variant>/):
    group_networksize_DLD.png, group_networksize_HSL.png, group_networksize_TD.png
    group_networksize_compare.png   (3 groups side by side)

## Author: Han Wang
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
META = f"{PROJECT_DIR}/res0urces/networks_meta.csv"

GROUPS = {                       # label -> (nice title, output tag)
    "DLD": ("DLD (BL)", "DLD"),
    "HSL": ("HSL (BH)", "HSL"),
    "TD":  ("Control (BT)", "TD"),
}
GROUP_COLORS = {"DLD": "#d63031", "HSL": "#2ca02c", "TD": "#0984e3"}
EXCLUDE_NETWORKS = {"Noise"}

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["full", "icafix"], default="full")
args = ap.parse_args()
NS_DIR = f"{PROJECT_DIR}/results/network_size_{args.variant}"
LONG = f"{NS_DIR}/group_network_size_long.csv"

long = pd.read_csv(LONG)
meta = pd.read_csv(META)
order = meta[~meta["network_label"].isin(EXCLUDE_NETWORKS)].sort_values("network_id")
labels = order["network_label"].tolist()
colors = order["hex"].tolist()
ids = order["network_id"].tolist()


def summarise(group_key):
    g = long[long["group"] == group_key]
    subs = g["subject"].nunique()
    means, sems = [], []
    for nid in ids:
        vals = g.loc[g["network_id"] == nid, "network_size_pct"].to_numpy(float)
        means.append(np.nanmean(vals) if vals.size else np.nan)
        sems.append(np.nanstd(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0)
    return np.array(means), np.array(sems), subs


def plot_group(group_key, title, tag):
    means, sems, n = summarise(group_key)
    y = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(6.5, 8))
    ax.barh(y, means, xerr=sems, color=colors, edgecolor="black", linewidth=0.4,
            error_kw=dict(ecolor="black", elinewidth=0.8, capsize=2.5))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% of cortical surface", fontsize=11)
    ax.set_title(f"MS-HBM network size — {title}  [{args.variant}]\n"
                 f"(group mean ± SEM, n = {n})", fontsize=12)
    ax.grid(axis="x", alpha=0.25); ax.set_axisbelow(True)
    for yi, m, s in zip(y, means, sems):
        if not np.isnan(m):
            ax.text(m + s + 0.1, yi, f"{m:.1f}", va="center", fontsize=8)
    plt.tight_layout()
    out = f"{NS_DIR}/group_networksize_{tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}  (n={n})")


def plot_compare():
    present = [g for g in GROUPS if (long["group"] == g).any()]
    stats = {g: summarise(g) for g in present}
    y = np.arange(len(ids))
    h = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(7.5, 9))
    for k, g in enumerate(present):
        means, sems, n = stats[g]
        off = (k - (len(present) - 1) / 2) * h
        ax.barh(y + off, means, height=h, xerr=sems, color=GROUP_COLORS[g],
                edgecolor="black", linewidth=0.3, label=f"{g} (n={n})",
                error_kw=dict(ecolor="0.3", elinewidth=0.6, capsize=1.5))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% of cortical surface", fontsize=11)
    ax.set_title(f"MS-HBM network size by group (mean ± SEM)  [{args.variant}]", fontsize=12)
    ax.grid(axis="x", alpha=0.25); ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = f"{NS_DIR}/group_networksize_compare.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


os.makedirs(NS_DIR, exist_ok=True)
for gk, (title, tag) in GROUPS.items():
    if (long["group"] == gk).any():
        plot_group(gk, title, tag)
    else:
        print(f"WARNING: no subjects with group '{gk}' in {LONG}; skipping.")
plot_compare()
