"""
stats_group_connectivity_emotional.py

EXPLORATORY: does SDQ-emotional symptom level relate to frontostriatal FC, and
does that relationship differ by group (DLD vs TD)? One simple OLS per tile of
the 3x3 frontostriatal matrix (9 models):

    z ~ C(group, Treatment('TD')) * emotional_c          [TD = reference]

  * DV = Fisher-z FC (z = arctanh(r)); variance-stabilised so OLS residuals are
    better behaved than on the bounded r. (At these low r values z ~ r, but z
    keeps this consistent with the group t-test in stats_group_connectivity.py.)
  * emotional_c = SDQ emotional, mean-centred (so the group term is the DLD-TD
    gap at average symptom level).
  * The interaction = DLD slope - TD slope (emotional->FC). BH-FDR across 9 tiles.

This mirrors the salience-size x emotional interaction analysis; same caveats
apply (n=36, exploratory, TD emotional floored). Reads the long table emitted by
stats_group_connectivity.py.

Input:  results/connectivity_outputs/group_frontostriatal_long.csv
        dat_verbgen_scqsdq_subsample.xlsx  (emotional)
Output: results/connectivity_outputs/group_frontostriatal_emotional_stats.csv
        results/connectivity_outputs/group_frontostriatal_emotional_interaction.png
        results/connectivity_outputs/group_frontostriatal_emotional_scatter.png

## Author: Han Wang
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
import statsmodels.formula.api as smf

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
CONN_DIR = f"{PROJECT_DIR}/results/connectivity_outputs"
XLSX = ("/home/hanwang/Documents/Data/ucl/gos_ich/verb_gen_krishnan/"
        "behavioural_scq_sdq/dat_verbgen_scqsdq_subsample.xlsx")

SUBCORTICAL = ["NAcc", "Caudate", "Putamen"]   # rows
CORTICAL = ["ACC", "AI", "LPFC"]               # cols
GROUP_COLORS = {"DLD": "#d63031", "TD": "#0984e3"}
INTER = "C(group, Treatment('TD'))[T.DLD]:emotional_c"


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


# ------------------------------------------------------------
# Data: long FC (z) + emotional
# ------------------------------------------------------------
long = pd.read_csv(f"{CONN_DIR}/group_frontostriatal_long.csv")
beh = pd.read_excel(XLSX)[["code", "emotional"]].copy()
beh["code"] = beh["code"].astype(str)
long = long.merge(beh, on="code", how="left").dropna(subset=["emotional"])
emo_mean = long.drop_duplicates("code")["emotional"].mean()
long["emotional_c"] = long["emotional"] - emo_mean
print(f"{long['subject'].nunique()} subjects; mean SDQ-emotional = {emo_mean:.2f}")

# ------------------------------------------------------------
# 9 OLS models (one per tile)
# ------------------------------------------------------------
res = []
for sc in SUBCORTICAL:
    for ct in CORTICAL:
        d = long[(long.subcortical == sc) & (long.cortical == ct)]
        fit = smf.ols("z ~ C(group, Treatment('TD')) * emotional_c", data=d).fit()
        sl_td = fit.params["emotional_c"]
        sl_dld = sl_td + fit.params[INTER]
        res.append(dict(
            edge=f"{sc}-{ct}", subcortical=sc, cortical=ct,
            slope_TD=sl_td, slope_DLD=sl_dld,
            inter_beta=fit.params[INTER], inter_t=fit.tvalues[INTER],
            inter_p=fit.pvalues[INTER],
            emo_main_p=fit.pvalues["emotional_c"],
            group_main_p=fit.pvalues["C(group, Treatment('TD'))[T.DLD]"],
            r2=fit.rsquared))

stat = pd.DataFrame(res)
# BH-FDR across the 9 interaction p-values
o = np.argsort(stat["inter_p"].to_numpy())
ranks = np.empty(len(o), int); ranks[o] = np.arange(1, len(o)+1)
q = np.minimum(1, stat["inter_p"].to_numpy() * len(o) / ranks)
q[o] = np.minimum.accumulate(q[o][::-1])[::-1]
stat["inter_p_fdr"] = q

stat.to_csv(f"{CONN_DIR}/group_frontostriatal_emotional_stats.csv", index=False)
print(f"Saved: {CONN_DIR}/group_frontostriatal_emotional_stats.csv\n")
print("Per-tile  z ~ group * emotional  (slopes = emotional->FC; sign of interaction = DLD - TD):")
print(stat[["edge", "slope_TD", "slope_DLD", "inter_beta", "inter_t",
            "inter_p", "inter_p_fdr", "r2"]].round(3).to_string(index=False))
print(f"\nInteraction p<.05 uncorrected: {(stat['inter_p']<.05).sum()}/9 | "
      f"survive FDR: {(stat['inter_p_fdr']<.05).sum()}/9")


def grid(col):
    a = np.full((len(SUBCORTICAL), len(CORTICAL)), np.nan)
    m = stat.set_index("edge")
    for i, sc in enumerate(SUBCORTICAL):
        for j, ct in enumerate(CORTICAL):
            a[i, j] = m.loc[f"{sc}-{ct}", col]
    return a


# ------------------------------------------------------------
# Figure 1: interaction t-stat heatmap
# ------------------------------------------------------------
T, P, Q = grid("inter_t"), grid("inter_p"), grid("inter_p_fdr")
tmax = max(np.nanmax(np.abs(T)), 1e-6)
norm = TwoSlopeNorm(vmin=-tmax, vcenter=0, vmax=tmax)

fig, ax = plt.subplots(figsize=(6.2, 5.4))
im = ax.imshow(T, cmap="RdBu_r", norm=norm, aspect="equal")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(CORTICAL, fontweight="bold")
ax.set_yticklabels(SUBCORTICAL, fontweight="bold")
for i in range(3):
    for j in range(3):
        col = "white" if abs(T[i, j]) > 0.6 * tmax else "black"
        ax.text(j, i, f"{T[i, j]:+.2f}{stars(P[i, j])}", ha="center", va="center",
                fontsize=12, color=col)
        if Q[i, j] < 0.05:
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                   edgecolor="black", linewidth=3))
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("interaction t  (DLD slope − TD slope)")
ax.set_xlabel("Cortical zone"); ax.set_ylabel("Subcortical")
ax.set_title("Group × SDQ-emotional interaction on frontostriatal FC\n"
             "red = DLD slope > TD, blue = DLD slope < TD;  * p<.05 ** .01 *** .001 (uncorr.)\n"
             "bold box = survives BH-FDR", fontsize=10.5)
plt.savefig(f"{CONN_DIR}/group_frontostriatal_emotional_interaction.png",
            dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved: {CONN_DIR}/group_frontostriatal_emotional_interaction.png")

# ------------------------------------------------------------
# Figure 2: 3x3 scatter small-multiples (emotional vs FC z, per group + OLS line)
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(11, 9.5), sharex=True, sharey=True)
for i, sc in enumerate(SUBCORTICAL):
    for j, ct in enumerate(CORTICAL):
        ax = axes[i, j]
        d = long[(long.subcortical == sc) & (long.cortical == ct)]
        for g in ["DLD", "TD"]:
            dg = d[d.group == g]
            ax.scatter(dg["emotional"], dg["z"], s=28, alpha=0.85,
                       color=GROUP_COLORS[g], edgecolor="k", linewidth=0.3,
                       label=g)
            if len(dg) > 2:
                b = np.polyfit(dg["emotional"], dg["z"], 1)
                xs = np.linspace(dg["emotional"].min(), dg["emotional"].max(), 30)
                ax.plot(xs, np.polyval(b, xs), color=GROUP_COLORS[g], lw=1.6)
        pv = stat.set_index("edge").loc[f"{sc}-{ct}", "inter_p"]
        ax.set_title(f"{sc}-{ct}  (int p={pv:.3f})", fontsize=9)
        ax.grid(alpha=0.2)
        if i == 2:
            ax.set_xlabel("SDQ emotional")
        if j == 0:
            ax.set_ylabel(f"{sc}\nFC (Fisher z)", fontsize=9)
axes[0, 0].legend(fontsize=8, loc="best")
fig.suptitle("Frontostriatal FC vs SDQ-emotional, by group (per-tile OLS lines)",
             fontsize=13)
plt.tight_layout()
plt.savefig(f"{CONN_DIR}/group_frontostriatal_emotional_scatter.png",
            dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {CONN_DIR}/group_frontostriatal_emotional_scatter.png")
