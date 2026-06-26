"""
stats_group_connectivity.py

Group comparison (DLD vs TD) of the frontostriatal FC tiles produced per subject
by connectivity/2_run_subject_connectivity_analysis_cab-np.py (the 3x3
subcortical x cortical matrix saved as <sub>_03_frontostriatal_FC_3_rest_antstri.csv).

Pipeline
--------
1. Gather: read every BL (DLD) + BT (TD) subject in the spreadsheet (excl. 513BT),
   load its 3x3 frontostriatal r-matrix, melt to long, and Fisher-z transform each
   tile (z = arctanh(r)). A long table is written out as a byproduct so no separate
   combine script is needed.
2. Test per tile (9 edges): Welch's t on the Fisher-z values (the z-transform turns
   the bounded r into an ~unbounded, variance-stabilised quantity, so a t-test is
   legitimate); Mann-Whitney as a rank-based robustness check; Cohen's d on z;
   Benjamini-Hochberg FDR across the 9 tiles.
3. Figures:
   - group_frontostriatal_mean.png  : two descriptive heatmaps (DLD, TD), tile =
     group mean r with +/-SD annotated (descriptive, in r).
   - group_frontostriatal_diff_tstat.png : one heatmap, tile colour & number = t
     (DLD - TD; red +, blue -), stars = uncorrected p, bold box = survives FDR.

Why r for the descriptive panels but z for the test: r is the interpretable unit to
read off a tile, but its sampling variance depends on the true value, so the
between-group inference is done on Fisher-z and only the displayed means are r.

Input:  results/connectivity_outputs/sub-*/sub-*_03_frontostriatal_FC_3_rest_antstri.csv
        dat_verbgen_scqsdq_subsample.xlsx  (group labels)
Output: results/connectivity_outputs/group_frontostriatal_long.csv
        results/connectivity_outputs/group_frontostriatal_stats.csv
        results/connectivity_outputs/group_frontostriatal_mean.png
        results/connectivity_outputs/group_frontostriatal_diff_tstat.png

## Author: Han Wang
"""

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
CONN_DIR = f"{PROJECT_DIR}/results/connectivity_outputs"
XLSX = ("/home/hanwang/Documents/Data/ucl/gos_ich/verb_gen_krishnan/"
        "behavioural_scq_sdq/dat_verbgen_scqsdq_subsample.xlsx")
SUFFIX = "_3_rest_antstri"

SUBCORTICAL = ["NAcc", "Caudate", "Putamen"]   # rows
CORTICAL = ["ACC", "AI", "LPFC"]               # cols
GROUP_COLORS = {"DLD": "#d63031", "TD": "#0984e3"}


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


# ============================================================
# 1. Gather per-subject frontostriatal tiles -> long (z-transformed)
# ============================================================
beh = pd.read_excel(XLSX)[["code", "group"]].copy()
beh["code"] = beh["code"].astype(str)
keep = (beh["code"].str.endswith(("BL", "BT"))) & (beh["code"] != "513BT")
cohort = beh[keep].set_index("code")["group"].to_dict()   # code -> DLD/TD

rows = []
missing = []
for code, group in cohort.items():
    sub = f"sub-{code}"
    f = f"{CONN_DIR}/{sub}/{sub}_03_frontostriatal_FC{SUFFIX}.csv"
    hits = glob.glob(f)
    if not hits:
        missing.append(sub)
        continue
    m = pd.read_csv(hits[0], index_col="subcortical")
    for sc in SUBCORTICAL:
        for ct in CORTICAL:
            r = float(m.loc[sc, ct])
            rows.append(dict(subject=sub, code=code, group=group,
                             subcortical=sc, cortical=ct, edge=f"{sc}-{ct}",
                             r=r, z=np.arctanh(np.clip(r, -0.999999, 0.999999))))

long = pd.DataFrame(rows)
if missing:
    print(f"WARNING: {len(missing)} subjects had no frontostriatal CSV: {missing}")
n_dld = long[long.group == "DLD"]["subject"].nunique()
n_td = long[long.group == "TD"]["subject"].nunique()
print(f"Loaded {long['subject'].nunique()} subjects: DLD={n_dld}, TD={n_td}")
long.to_csv(f"{CONN_DIR}/group_frontostriatal_long.csv", index=False)
print(f"Saved: {CONN_DIR}/group_frontostriatal_long.csv")

# ============================================================
# 2. Per-tile group comparison (test on Fisher-z)
# ============================================================
res = []
for sc in SUBCORTICAL:
    for ct in CORTICAL:
        d = long[(long.subcortical == sc) & (long.cortical == ct)]
        zd = d[d.group == "DLD"]["z"].to_numpy()
        zt = d[d.group == "TD"]["z"].to_numpy()
        rd = d[d.group == "DLD"]["r"].to_numpy()
        rt = d[d.group == "TD"]["r"].to_numpy()
        t, p = stats.ttest_ind(zd, zt, equal_var=False)             # Welch on z
        u, pu = stats.mannwhitneyu(zd, zt, alternative="two-sided")
        # Cohen's d on z (pooled SD)
        sp = np.sqrt(((len(zd)-1)*zd.var(ddof=1) + (len(zt)-1)*zt.var(ddof=1))
                     / (len(zd)+len(zt)-2))
        d_eff = (zd.mean() - zt.mean()) / sp if sp > 0 else np.nan
        res.append(dict(edge=f"{sc}-{ct}", subcortical=sc, cortical=ct,
                        mean_r_DLD=rd.mean(), sd_r_DLD=rd.std(ddof=1),
                        mean_r_TD=rt.mean(), sd_r_TD=rt.std(ddof=1),
                        mean_z_DLD=zd.mean(), mean_z_TD=zt.mean(),
                        t=t, p=p, mannwhitney_p=pu, cohen_d=d_eff))

stat = pd.DataFrame(res)
# BH-FDR across the 9 tiles
order = np.argsort(stat["p"].to_numpy())
ranks = np.empty_like(order); ranks[order] = np.arange(1, len(stat)+1)
stat["p_fdr"] = np.minimum(1, stat["p"] * len(stat) / ranks)
# enforce monotonicity of BH q-values
q_sorted = stat["p_fdr"].to_numpy()[order]
q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
stat.loc[order, "p_fdr"] = q_sorted

stat.to_csv(f"{CONN_DIR}/group_frontostriatal_stats.csv", index=False)
print(f"Saved: {CONN_DIR}/group_frontostriatal_stats.csv\n")
print("Per-tile DLD vs TD (t on Fisher-z; sign = DLD - TD):")
print(stat[["edge", "mean_r_DLD", "mean_r_TD", "t", "p", "p_fdr",
            "mannwhitney_p", "cohen_d"]].round(3).to_string(index=False))
n_sig = (stat["p"] < .05).sum(); n_fdr = (stat["p_fdr"] < .05).sum()
print(f"\nTiles p<.05 uncorrected: {n_sig}/9 | survive FDR: {n_fdr}/9")


def grid(series_map, sc_list=SUBCORTICAL, ct_list=CORTICAL):
    """Build a (subcortical x cortical) array from an edge->value mapping."""
    a = np.full((len(sc_list), len(ct_list)), np.nan)
    for i, sc in enumerate(sc_list):
        for j, ct in enumerate(ct_list):
            a[i, j] = series_map[f"{sc}-{ct}"]
    return a


smap = stat.set_index("edge")

# ============================================================
# 3a. Descriptive heatmaps: group mean r (+/- SD), one panel per group
# ============================================================
mean_r = {g: grid({e: smap.loc[e, f"mean_r_{g}"] for e in smap.index}) for g in ["DLD", "TD"]}
sd_r = {g: grid({e: smap.loc[e, f"sd_r_{g}"] for e in smap.index}) for g in ["DLD", "TD"]}
vmax = np.nanmax([np.abs(mean_r["DLD"]), np.abs(mean_r["TD"])])
vmax = np.ceil(vmax * 10) / 10

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("white")
for ax, g in zip(axes, ["DLD", "TD"]):
    im = ax.imshow(mean_r[g], cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(CORTICAL))); ax.set_yticks(range(len(SUBCORTICAL)))
    ax.set_xticklabels(CORTICAL, fontweight="bold")
    ax.set_yticklabels(SUBCORTICAL, fontweight="bold")
    n = n_dld if g == "DLD" else n_td
    ax.set_title(f"{g} (n={n})", color=GROUP_COLORS[g], fontweight="bold")
    for i in range(len(SUBCORTICAL)):
        for j in range(len(CORTICAL)):
            c = "white" if abs(mean_r[g][i, j]) > 0.3 else "black"
            ax.text(j, i, f"{mean_r[g][i, j]:+.2f}\n±{sd_r[g][i, j]:.2f}",
                    ha="center", va="center", fontsize=10, color=c)
    ax.set_xlabel("Cortical zone")
axes[0].set_ylabel("Subcortical")
cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.04)
cbar.set_label("Group mean Pearson's r")
fig.suptitle("Frontostriatal FC — group means (tile: mean r ± SD across subjects)",
             fontsize=13)
plt.savefig(f"{CONN_DIR}/group_frontostriatal_mean.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved: {CONN_DIR}/group_frontostriatal_mean.png")

# ============================================================
# 3b. Difference heatmap: t-stat (DLD - TD), stars = p, box = survives FDR
# ============================================================
T = grid({e: smap.loc[e, "t"] for e in smap.index})
P = grid({e: smap.loc[e, "p"] for e in smap.index})
Q = grid({e: smap.loc[e, "p_fdr"] for e in smap.index})
tmax = max(np.nanmax(np.abs(T)), 1e-6)
norm = TwoSlopeNorm(vmin=-tmax, vcenter=0, vmax=tmax)

fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(T, cmap="RdBu_r", norm=norm, aspect="equal")
ax.set_xticks(range(len(CORTICAL))); ax.set_yticks(range(len(SUBCORTICAL)))
ax.set_xticklabels(CORTICAL, fontweight="bold")
ax.set_yticklabels(SUBCORTICAL, fontweight="bold")
for i in range(len(SUBCORTICAL)):
    for j in range(len(CORTICAL)):
        col = "white" if abs(T[i, j]) > 0.6 * tmax else "black"
        ax.text(j, i, f"{T[i, j]:+.2f}{stars(P[i, j])}",
                ha="center", va="center", fontsize=12, color=col)
        if Q[i, j] < 0.05:                                  # survives FDR
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                   edgecolor="black", linewidth=3))
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("t  (DLD − TD, on Fisher-z)")
ax.set_xlabel("Cortical zone"); ax.set_ylabel("Subcortical")
ax.set_title("Frontostriatal FC: DLD − TD\n"
             "red = DLD>TD, blue = DLD<TD;  * p<.05 ** p<.01 *** p<.001 (uncorr.)\n"
             "bold box = survives BH-FDR", fontsize=11)
plt.savefig(f"{CONN_DIR}/group_frontostriatal_diff_tstat.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {CONN_DIR}/group_frontostriatal_diff_tstat.png")
