"""
plot_group_network_size.py

Group-level visualisation of MS-HBM individual network sizes, adapted from the
single-subject bar plot in run_subject_mshbm.m. Produces one horizontal bar
chart per group (DLD/BL and TD/control), where each bar is the group mean
% cortical surface for a network and the error bar is the standard error of the
mean (SEM) across subjects.

Inputs:
    results/network_size/group_network_size_long.csv  (from combine_network_size.py)
    res0urces/networks_meta.csv  (network id/label/colour, exported from MS-HBM priors)

Outputs:
    results/network_size/group_networksize_DLD.png
    results/network_size/group_networksize_TD.png

## Author: Han Wang
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
NS_DIR = f"{PROJECT_DIR}/results/network_size"

LONG = f"{NS_DIR}/group_network_size_long.csv"
META = f"{PROJECT_DIR}/res0urces/networks_meta.csv"

# Groups to plot: spreadsheet label -> (nice title, output tag)
GROUPS = {
    "DLD": ("DLD (BL)", "DLD"),
    "TD":  ("Control (BT)", "TD"),
}
EXCLUDE_NETWORKS = {"Noise"}   # drop the unassigned/noise label from the figure

# ------------------------------------------------------------
long = pd.read_csv(LONG)
meta = pd.read_csv(META)

# network plotting order = network_id ascending, minus excluded labels
order = meta[~meta["network_label"].isin(EXCLUDE_NETWORKS)].sort_values("network_id")
labels = order["network_label"].tolist()
colors = order["hex"].tolist()
ids = order["network_id"].tolist()


def summarise(group_key):
    """Return mean and SEM per network (in `ids` order) for one group."""
    g = long[long["group"] == group_key]
    subs = g["subject"].nunique()
    means, sems = [], []
    for nid in ids:
        vals = g.loc[g["network_id"] == nid, "network_size_pct"].to_numpy(dtype=float)
        means.append(np.nanmean(vals) if vals.size else np.nan)
        sems.append(np.nanstd(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0)
    return np.array(means), np.array(sems), subs


def plot_group(group_key, title, tag):
    means, sems, n = summarise(group_key)
    y = np.arange(len(ids))

    fig, ax = plt.subplots(figsize=(6.5, 8))
    ax.barh(y, means, xerr=sems, color=colors, edgecolor="black", linewidth=0.4,
            error_kw=dict(ecolor="black", elinewidth=0.8, capsize=2.5))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()                       # network_id 1 at the top
    ax.set_xlabel("% of cortical surface", fontsize=11)
    ax.set_title(f"MS-HBM network size — {title}\n(group mean ± SEM, n = {n})",
                 fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # value labels at bar ends
    for yi, m, s in zip(y, means, sems):
        if not np.isnan(m):
            ax.text(m + s + 0.1, yi, f"{m:.1f}", va="center", fontsize=8)

    plt.tight_layout()
    out = f"{NS_DIR}/group_networksize_{tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}  (n={n})")


os.makedirs(NS_DIR, exist_ok=True)
for gk, (title, tag) in GROUPS.items():
    if (long["group"] == gk).any():
        plot_group(gk, title, tag)
    else:
        print(f"WARNING: no subjects with group '{gk}' in {LONG}; skipping.")
