"""
stats_connectivity_emotional_nb.py

Mood-coupling model matching the salience-size analysis
(stats_emotional_salience_nb_interaction.py): the brain measure is the PREDICTOR
and SDQ-emotional the OUTCOME, one negative-binomial model per frontostriatal
tile, now with the 3-level group factor (TD reference) over all 144 subjects:

    emotional ~ C(group, Treatment('TD')) * FCz_c        [TD = reference]

  * Outcome = SDQ-emotional (bounded 0-10 count, floored in TD) -> negative
    binomial (log link), as for salience size.
  * Predictor = tile FC in Fisher-z (FCz), mean-centred per tile, so the group
    terms are the group-vs-TD emotional gaps at that tile's mean connectivity.
  * Two interaction terms now: [T.DLD]:FCz_c and [T.HSL]:FCz_c (each a
    group-vs-TD difference in the FC->emotional slope). The headline
    "interaction" test is a JOINT likelihood-ratio test of BOTH interaction
    terms (2 df: full vs additive), BH-FDR across the 9 tiles. Exploratory.

Input:  results/connectivity_outputs/group_frontostriatal_long.csv
        dat_verbgen_analysis_144.csv  (emotional, group)
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
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

warnings.filterwarnings("ignore")
NPERM = 2000

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
CONN_DIR = f"{PROJECT_DIR}/results/connectivity_outputs"
LISTCSV = ("/home/hanwang/Documents/Data/verb_gen_krishnan/"
           "behavioural_scq_sdq/dat_verbgen_analysis_144.csv")

SUBCORTICAL = ["NAcc", "Caudate", "Putamen"]
CORTICAL = ["ACC", "AI", "LPFC"]
GROUPS = ["DLD", "HSL", "TD"]
GROUP_COLORS = {"DLD": "#d63031", "HSL": "#2ca02c", "TD": "#0984e3"}
INTER = {"DLD": "C(group, Treatment('TD'))[T.DLD]:FCz_c",
         "HSL": "C(group, Treatment('TD'))[T.HSL]:FCz_c"}
MAIN = {"DLD": "C(group, Treatment('TD'))[T.DLD]",
        "HSL": "C(group, Treatment('TD'))[T.HSL]"}
CONTRASTS = ["DLD", "HSL"]   # vs TD reference
RNG = np.random.default_rng(0)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def pois_joint_lr(dd):
    """Joint-interaction LR (deviance drop, full vs additive) under a Poisson
    working model -- the statistic for the Freedman-Lane permutation."""
    f = smf.glm("emotional ~ C(group, Treatment('TD')) * FCz_c",
                data=dd, family=sm.families.Poisson()).fit()
    r = smf.glm("emotional ~ C(group, Treatment('TD')) + FCz_c",
                data=dd, family=sm.families.Poisson()).fit()
    return r.deviance - f.deviance


def freedman_lane_p(d):
    """Distribution-free p for the joint group x FC interaction (permute the
    reduced-model residuals, recompute the Poisson joint-interaction LR)."""
    lr_obs = pois_joint_lr(d)
    red = smf.glm("emotional ~ C(group, Treatment('TD')) + FCz_c",
                  data=d, family=sm.families.Poisson()).fit()
    mu = red.fittedvalues.to_numpy()
    resid = d["emotional"].to_numpy() - mu
    dp = d.copy()
    n_ge = n_val = 0
    for _ in range(NPERM):
        dp["emotional"] = np.clip(np.round(mu + resid[RNG.permutation(len(resid))]), 0, None)
        try:
            n_ge += pois_joint_lr(dp) >= lr_obs
            n_val += 1
        except Exception:
            continue
    return (1 + n_ge) / (1 + n_val)


# ------------------------------------------------------------
# Data: long FC (z) + emotional outcome
# ------------------------------------------------------------
long = pd.read_csv(f"{CONN_DIR}/group_frontostriatal_long.csv")
beh = pd.read_csv(LISTCSV)[["code", "emotional"]].copy()
beh["code"] = beh["code"].astype(str)
long["code"] = long["code"].astype(str)
long = long.merge(beh, on="code", how="left").dropna(subset=["emotional"])
long["emotional"] = long["emotional"].round().astype(int).clip(0, 10)
long = long.rename(columns={"z": "FCz"})
print(f"{long['subject'].nunique()} subjects | groups: "
      + long.drop_duplicates('subject')['group'].value_counts().to_dict().__str__())

# ------------------------------------------------------------
# 9 NB models (one per tile), 3-level group
# ------------------------------------------------------------
res, fits = [], {}
for sc in SUBCORTICAL:
    for ct in CORTICAL:
        d = long[(long.subcortical == sc) & (long.cortical == ct)].copy()
        d["FCz_c"] = d["FCz"] - d["FCz"].mean()
        try:
            full = smf.negativebinomial(
                "emotional ~ C(group, Treatment('TD')) * FCz_c", data=d).fit(disp=0, maxiter=1000)
            red = smf.negativebinomial(
                "emotional ~ C(group, Treatment('TD')) + FCz_c", data=d).fit(disp=0, maxiter=1000)
            lr = 2 * (full.llf - red.llf)
            lr_p = stats.chi2.sf(lr, 2)          # 2 df: both interaction terms
            rec = dict(edge=f"{sc}-{ct}", subcortical=sc, cortical=ct,
                       slope_TD=full.params["FCz_c"],
                       slope_DLD=full.params["FCz_c"] + full.params[INTER["DLD"]],
                       slope_HSL=full.params["FCz_c"] + full.params[INTER["HSL"]],
                       inter_beta_DLD=full.params[INTER["DLD"]],
                       inter_z_DLD=full.tvalues[INTER["DLD"]],
                       inter_wald_p_DLD=full.pvalues[INTER["DLD"]],
                       inter_beta_HSL=full.params[INTER["HSL"]],
                       inter_z_HSL=full.tvalues[INTER["HSL"]],
                       inter_wald_p_HSL=full.pvalues[INTER["HSL"]],
                       inter_lr_p=lr_p, alpha=full.params["alpha"],
                       inter_perm_p=freedman_lane_p(d))
            fits[(sc, ct)] = (full, d)
            res.append(rec)
        except Exception as e:
            res.append(dict(edge=f"{sc}-{ct}", subcortical=sc, cortical=ct,
                            inter_lr_p=np.nan))
            print(f"  NB failed for {sc}-{ct}: {e}")

stat = pd.DataFrame(res)
# BH-FDR across the 9 joint-interaction LR p-values
p = stat["inter_lr_p"].to_numpy()
o = np.argsort(p)
ranks = np.empty(len(o), int); ranks[o] = np.arange(1, len(o) + 1)
q = np.minimum(1, p * len(o) / ranks)
q[o] = np.minimum.accumulate(q[o][::-1])[::-1]
stat["inter_lr_p_fdr"] = q
# BH-FDR on the distribution-free permutation p as well
pp = stat["inter_perm_p"].to_numpy()
op = np.argsort(pp)
rp = np.empty(len(op), int); rp[op] = np.arange(1, len(op) + 1)
qp = np.minimum(1, pp * len(op) / rp)
qp[op] = np.minimum.accumulate(qp[op][::-1])[::-1]
stat["inter_perm_p_fdr"] = qp

stat.to_csv(f"{CONN_DIR}/connectivity_emotional_nb_stats.csv", index=False)
print(f"Saved: {CONN_DIR}/connectivity_emotional_nb_stats.csv\n")
print("Per-tile NB  emotional ~ group * FCz  (slopes = FC->emotional, log-count):")
print(stat[["edge", "slope_TD", "slope_DLD", "slope_HSL",
            "inter_lr_p", "inter_lr_p_fdr", "inter_perm_p", "inter_perm_p_fdr"]]
      .round(3).to_string(index=False))
print(f"\nJoint interaction — LR p<.05: {(stat['inter_lr_p']<.05).sum()}/9 "
      f"(FDR {(stat['inter_lr_p_fdr']<.05).sum()}/9)  |  "
      f"permutation p<.05: {(stat['inter_perm_p']<.05).sum()}/9 "
      f"(FDR {(stat['inter_perm_p_fdr']<.05).sum()}/9)")


def grid(col):
    a = np.full((3, 3), np.nan)
    m = stat.set_index("edge")
    for i, sc in enumerate(SUBCORTICAL):
        for j, ct in enumerate(CORTICAL):
            a[i, j] = m.loc[f"{sc}-{ct}", col] if col in m.columns else np.nan
    return a


# ------------------------------------------------------------
# Figure 1: interaction heatmaps, one panel per group-vs-TD contrast
# (Wald z; stars = that term's Wald p; box = tile's joint LR survives FDR)
# ------------------------------------------------------------
Q = grid("inter_lr_p_fdr")
Zs = {g: grid(f"inter_z_{g}") for g in CONTRASTS}
Ps = {g: grid(f"inter_wald_p_{g}") for g in CONTRASTS}
zmax = max(np.nanmax([np.abs(Zs[g]) for g in CONTRASTS]), 1e-6)
norm = TwoSlopeNorm(vmin=-zmax, vcenter=0, vmax=zmax)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
for ax, g in zip(axes, CONTRASTS):
    im = ax.imshow(Zs[g], cmap="RdBu_r", norm=norm, aspect="equal")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CORTICAL, fontweight="bold")
    ax.set_yticklabels(SUBCORTICAL, fontweight="bold")
    ax.set_title(f"{g} slope − TD slope", fontsize=12, fontweight="bold",
                 color=GROUP_COLORS[g])
    for i in range(3):
        for j in range(3):
            col = "white" if abs(Zs[g][i, j]) > 0.6 * zmax else "black"
            ax.text(j, i, f"{Zs[g][i, j]:+.2f}{stars(Ps[g][i, j])}",
                    ha="center", va="center", fontsize=12, color=col)
            if Q[i, j] < 0.05:
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                       edgecolor="black", linewidth=3))
    ax.set_xlabel("Cortical zone")
axes[0].set_ylabel("Subcortical")
cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
cbar.set_label("interaction Wald z  (group slope − TD slope)")
fig.suptitle("NB: SDQ-emotional ~ group × frontostriatal FC  (TD reference)\n"
             "red = group slope > TD, blue = < TD;  stars = Wald p; "
             "bold box = tile's joint interaction survives BH-FDR", fontsize=11)
plt.savefig(f"{CONN_DIR}/connectivity_emotional_nb_interaction.png",
            dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved: {CONN_DIR}/connectivity_emotional_nb_interaction.png")

# ------------------------------------------------------------
# Figure 2: 3x3 scatter (x=FC z, y=emotional) + NB predicted-mean curves, 3 groups
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(11, 9.5), sharey=True)
for i, sc in enumerate(SUBCORTICAL):
    for j, ct in enumerate(CORTICAL):
        ax = axes[i, j]
        if (sc, ct) not in fits:
            ax.set_visible(False); continue
        full, d = fits[(sc, ct)]
        zmean = d["FCz"].mean()
        # delta-method 95% CI on the predicted mean (log-count scale, exp back)
        di = full.model.data.design_info
        kf = len(di.column_names)
        beta = np.asarray(full.params)[:kf]          # alpha is the last param
        cov = np.asarray(full.cov_params())[:kf, :kf]
        for g in GROUPS:
            dg = d[d.group == g]
            yj = dg["emotional"] + RNG.uniform(-0.15, 0.15, len(dg))
            ax.scatter(dg["FCz"], yj, s=22, alpha=0.8, color=GROUP_COLORS[g],
                       edgecolor="k", linewidth=0.3, label=g)
            xs = np.linspace(dg["FCz"].min(), dg["FCz"].max(), 40)
            grid_df = pd.DataFrame({"group": g, "FCz": xs, "FCz_c": xs - zmean})
            X = np.asarray(patsy.dmatrix(di, grid_df))
            eta = X @ beta
            se = np.sqrt(np.einsum("ij,jk,ik->i", X, cov, X))
            ax.fill_between(xs, np.exp(eta - 1.96 * se), np.exp(eta + 1.96 * se),
                            color=GROUP_COLORS[g], alpha=0.13, lw=0)
            ax.plot(xs, np.exp(eta), color=GROUP_COLORS[g], lw=1.8)
        plr = stat.set_index("edge").loc[f"{sc}-{ct}", "inter_lr_p"]
        ax.set_title(f"{sc}-{ct}  (joint int LR p={plr:.3f})", fontsize=9)
        ax.set_ylim(-0.5, 10.5); ax.grid(alpha=0.2)
        if i == 2:
            ax.set_xlabel(f"{ct}\nFC (Fisher z)", fontsize=9)
        if j == 0:
            ax.set_ylabel("SDQ emotional", fontsize=9)
axes[0, 0].legend(fontsize=8, loc="best")
fig.suptitle("NB-predicted SDQ-emotional vs frontostriatal FC, by group "
             "(shaded = 95% CI on predicted mean; points y-jittered)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{CONN_DIR}/connectivity_emotional_nb_scatter.png",
            dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {CONN_DIR}/connectivity_emotional_nb_scatter.png")
