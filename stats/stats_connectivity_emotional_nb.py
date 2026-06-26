"""
stats_connectivity_emotional_nb.py

Flip of stats_group_connectivity_emotional.py to match the salience-size model
(stats_emotional_salience_nb_interaction.py): the brain measure is now the
PREDICTOR and SDQ-emotional the OUTCOME, one model per frontostriatal tile:

    emotional ~ C(group, Treatment('TD')) * FCz_c          [TD = reference]

  * Outcome = SDQ-emotional (bounded 0-10 count, floored in TD) -> negative
    binomial (log link), exactly as for salience size. The interaction = DLD
    slope - TD slope of FC->emotional on the log-count scale.
  * Predictor = tile FC in Fisher-z (FCz), mean-centred per tile (so the group
    term is the DLD-TD emotional gap at that tile's mean connectivity).
  * Interaction tested by likelihood-ratio (refit without it), the headline test
    used for salience; BH-FDR across the 9 tiles. Exploratory (n=36).

Input:  results/connectivity_outputs/group_frontostriatal_long.csv
        dat_verbgen_scqsdq_subsample.xlsx  (emotional)
Output: results/connectivity_outputs/connectivity_emotional_nb_stats.csv
        results/connectivity_outputs/connectivity_emotional_nb_interaction.png
        results/connectivity_outputs/connectivity_emotional_nb_scatter.png

## Author: Han Wang
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
CONN_DIR = f"{PROJECT_DIR}/results/connectivity_outputs"
XLSX = ("/home/hanwang/Documents/Data/ucl/gos_ich/verb_gen_krishnan/"
        "behavioural_scq_sdq/dat_verbgen_scqsdq_subsample.xlsx")

SUBCORTICAL = ["NAcc", "Caudate", "Putamen"]
CORTICAL = ["ACC", "AI", "LPFC"]
GROUP_COLORS = {"DLD": "#d63031", "TD": "#0984e3"}
INTER = "C(group, Treatment('TD'))[T.DLD]:FCz_c"
RNG = np.random.default_rng(0)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


# ------------------------------------------------------------
# Data: long FC (z) + emotional outcome
# ------------------------------------------------------------
long = pd.read_csv(f"{CONN_DIR}/group_frontostriatal_long.csv")
beh = pd.read_excel(XLSX)[["code", "emotional"]].copy()
beh["code"] = beh["code"].astype(str)
long = long.merge(beh, on="code", how="left").dropna(subset=["emotional"])
long["emotional"] = long["emotional"].round().astype(int).clip(0, 10)
long = long.rename(columns={"z": "FCz"})
print(f"{long['subject'].nunique()} subjects")

# ------------------------------------------------------------
# 9 NB models (one per tile)
# ------------------------------------------------------------
res = []
fits = {}
for sc in SUBCORTICAL:
    for ct in CORTICAL:
        d = long[(long.subcortical == sc) & (long.cortical == ct)].copy()
        d["FCz_c"] = d["FCz"] - d["FCz"].mean()
        try:
            full = smf.negativebinomial(
                "emotional ~ C(group, Treatment('TD')) * FCz_c", data=d).fit(disp=0)
            red = smf.negativebinomial(
                "emotional ~ C(group, Treatment('TD')) + FCz_c", data=d).fit(disp=0)
            lr = 2 * (full.llf - red.llf)
            lr_p = stats.chi2.sf(lr, 1)
            sl_td = full.params["FCz_c"]
            sl_dld = sl_td + full.params[INTER]
            fits[(sc, ct)] = (full, d)
            res.append(dict(
                edge=f"{sc}-{ct}", subcortical=sc, cortical=ct,
                slope_TD=sl_td, slope_DLD=sl_dld,
                inter_beta=full.params[INTER], inter_z=full.tvalues[INTER],
                inter_wald_p=full.pvalues[INTER], inter_lr_p=lr_p,
                alpha=full.params["alpha"]))
        except Exception as e:
            res.append(dict(edge=f"{sc}-{ct}", subcortical=sc, cortical=ct,
                            slope_TD=np.nan, slope_DLD=np.nan, inter_beta=np.nan,
                            inter_z=np.nan, inter_wald_p=np.nan, inter_lr_p=np.nan,
                            alpha=np.nan))
            print(f"  NB failed for {sc}-{ct}: {e}")

stat = pd.DataFrame(res)
# BH-FDR across the 9 interaction LR p-values
p = stat["inter_lr_p"].to_numpy()
o = np.argsort(p)
ranks = np.empty(len(o), int); ranks[o] = np.arange(1, len(o)+1)
q = np.minimum(1, p * len(o) / ranks)
q[o] = np.minimum.accumulate(q[o][::-1])[::-1]
stat["inter_lr_p_fdr"] = q

stat.to_csv(f"{CONN_DIR}/connectivity_emotional_nb_stats.csv", index=False)
print(f"Saved: {CONN_DIR}/connectivity_emotional_nb_stats.csv\n")
print("Per-tile NB  emotional ~ group * FCz  (slopes = FC->emotional, log-count; "
      "interaction sign = DLD - TD):")
print(stat[["edge", "slope_TD", "slope_DLD", "inter_beta", "inter_z",
            "inter_wald_p", "inter_lr_p", "inter_lr_p_fdr"]].round(3).to_string(index=False))
print(f"\nInteraction LR p<.05 uncorrected: {(stat['inter_lr_p']<.05).sum()}/9 | "
      f"survive FDR: {(stat['inter_lr_p_fdr']<.05).sum()}/9")


def grid(col):
    a = np.full((3, 3), np.nan)
    m = stat.set_index("edge")
    for i, sc in enumerate(SUBCORTICAL):
        for j, ct in enumerate(CORTICAL):
            a[i, j] = m.loc[f"{sc}-{ct}", col]
    return a


# ------------------------------------------------------------
# Figure 1: interaction heatmap (Wald z; stars = LR p; box = survives FDR)
# ------------------------------------------------------------
Z, PLR, Q = grid("inter_z"), grid("inter_lr_p"), grid("inter_lr_p_fdr")
zmax = max(np.nanmax(np.abs(Z)), 1e-6)
norm = TwoSlopeNorm(vmin=-zmax, vcenter=0, vmax=zmax)

fig, ax = plt.subplots(figsize=(6.4, 5.6))
im = ax.imshow(Z, cmap="RdBu_r", norm=norm, aspect="equal")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(CORTICAL, fontweight="bold")
ax.set_yticklabels(SUBCORTICAL, fontweight="bold")
for i in range(3):
    for j in range(3):
        col = "white" if abs(Z[i, j]) > 0.6 * zmax else "black"
        ax.text(j, i, f"{Z[i, j]:+.2f}{stars(PLR[i, j])}", ha="center", va="center",
                fontsize=12, color=col)
        if Q[i, j] < 0.05:
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                   edgecolor="black", linewidth=3))
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("interaction Wald z  (DLD slope − TD slope)")
ax.set_xlabel("Cortical zone"); ax.set_ylabel("Subcortical")
ax.set_title("NB: SDQ-emotional ~ group × frontostriatal FC\n"
             "red = DLD slope > TD, blue = DLD slope < TD;  stars = LR p "
             "(* .05 ** .01 *** .001)\nbold box = survives BH-FDR", fontsize=10.5)
plt.savefig(f"{CONN_DIR}/connectivity_emotional_nb_interaction.png",
            dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved: {CONN_DIR}/connectivity_emotional_nb_interaction.png")

# ------------------------------------------------------------
# Figure 2: 3x3 scatter (x=FC z, y=emotional) + NB predicted-mean curves
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(11, 9.5), sharey=True)
for i, sc in enumerate(SUBCORTICAL):
    for j, ct in enumerate(CORTICAL):
        ax = axes[i, j]
        if (sc, ct) not in fits:
            ax.set_visible(False); continue
        full, d = fits[(sc, ct)]
        zmean = d["FCz"].mean()
        for g in ["DLD", "TD"]:
            dg = d[d.group == g]
            yj = dg["emotional"] + RNG.uniform(-0.15, 0.15, len(dg))
            ax.scatter(dg["FCz"], yj, s=26, alpha=0.85, color=GROUP_COLORS[g],
                       edgecolor="k", linewidth=0.3, label=g)
            xs = np.linspace(dg["FCz"].min(), dg["FCz"].max(), 40)
            xs_c = xs - zmean
            if g == "TD":
                eta = full.params["Intercept"] + full.params["FCz_c"] * xs_c
            else:
                eta = (full.params["Intercept"]
                       + full.params["C(group, Treatment('TD'))[T.DLD]"]
                       + (full.params["FCz_c"] + full.params[INTER]) * xs_c)
            ax.plot(xs, np.exp(eta), color=GROUP_COLORS[g], lw=1.8)
        plr = stat.set_index("edge").loc[f"{sc}-{ct}", "inter_lr_p"]
        ax.set_title(f"{sc}-{ct}  (int LR p={plr:.3f})", fontsize=9)
        ax.set_ylim(-0.5, 10.5); ax.grid(alpha=0.2)
        if i == 2:
            ax.set_xlabel(f"{ct}\nFC (Fisher z)", fontsize=9)
        if j == 0:
            ax.set_ylabel("SDQ emotional", fontsize=9)
axes[0, 0].legend(fontsize=8, loc="best")
fig.suptitle("NB-predicted SDQ-emotional vs frontostriatal FC, by group "
             "(points y-jittered)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{CONN_DIR}/connectivity_emotional_nb_scatter.png",
            dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {CONN_DIR}/connectivity_emotional_nb_scatter.png")
