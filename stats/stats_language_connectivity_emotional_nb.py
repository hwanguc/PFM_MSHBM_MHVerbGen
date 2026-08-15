"""
stats_language_connectivity_emotional_nb.py

## Author: Han Wang
### 2026-08-14: Initial version (language frontal-putamen arm).

Mood-coupling model for the SINGLE language frontal-putamen edge, matching the
striatal arm's stats_connectivity_emotional_nb.py: brain FC is the PREDICTOR and
SDQ-emotional the OUTCOME, one negative-binomial model, 3-level group (TD ref):

    emotional ~ C(group, Treatment('TD')) * FCz_c        [TD = reference]

Edge: Language-14_L-Ctx (L_44, pars opercularis) <-> Language-14_L-Putamen
      (medial/anterior left putamen).

  * Outcome = SDQ-emotional (0-10 count) -> negative binomial (log link).
  * Predictor = edge FC in Fisher-z (FCz), mean-centred, so group terms are the
    group-vs-TD emotional gaps at the mean connectivity.
  * Two interaction terms: [T.DLD]:FCz_c and [T.HSL]:FCz_c. Headline test is the
    JOINT likelihood-ratio test of BOTH (2 df: full vs additive). Also a
    distribution-free Freedman-Lane permutation p for that joint interaction.
    Single edge -> no multiple-comparison correction across tiles.

Input:  results/language_connectivity_outputs/group_language_putamen_long.csv
        dat_verbgen_analysis_144.csv  (emotional, group)
Output: results/language_connectivity_outputs/language_connectivity_emotional_nb_stats.csv
        results/language_connectivity_outputs/language_connectivity_emotional_nb_scatter.png
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

warnings.filterwarnings("ignore")
NPERM = 2000

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
LANG_DIR = f"{PROJECT_DIR}/results/language_connectivity_outputs"
LISTCSV = ("/home/hanwang/Documents/Data/verb_gen_krishnan/"
           "behavioural_scq_sdq/dat_verbgen_analysis_144.csv")

GROUPS = ["DLD", "HSL", "TD"]
GROUP_COLORS = {"DLD": "#d63031", "HSL": "#2ca02c", "TD": "#0984e3"}
INTER = {"DLD": "C(group, Treatment('TD'))[T.DLD]:FCz_c",
         "HSL": "C(group, Treatment('TD'))[T.HSL]:FCz_c"}
MAIN = {"DLD": "C(group, Treatment('TD'))[T.DLD]",
        "HSL": "C(group, Treatment('TD'))[T.HSL]"}
CONTRASTS = ["DLD", "HSL"]
RNG = np.random.default_rng(0)
EDGE = "L-Putamen(Language) <-> L_44 (pars opercularis)"


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def pois_joint_lr(dd):
    f = smf.glm("emotional ~ C(group, Treatment('TD')) * FCz_c",
                data=dd, family=sm.families.Poisson()).fit()
    r = smf.glm("emotional ~ C(group, Treatment('TD')) + FCz_c",
                data=dd, family=sm.families.Poisson()).fit()
    return r.deviance - f.deviance


def freedman_lane_p(d):
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
# Data: long edge (z) + emotional outcome
# ------------------------------------------------------------
long = pd.read_csv(f"{LANG_DIR}/group_language_putamen_long.csv")
beh = pd.read_csv(LISTCSV)[["code", "emotional"]].copy()
beh["code"] = beh["code"].astype(str)
long["code"] = long["code"].astype(str)
d = long.merge(beh, on="code", how="left").dropna(subset=["emotional"])
d["emotional"] = d["emotional"].round().astype(int).clip(0, 10)
d = d.rename(columns={"z": "FCz"})
d["FCz_c"] = d["FCz"] - d["FCz"].mean()
counts = d["group"].value_counts().to_dict()
print(f"{len(d)} subjects | " + ", ".join(f"{g}={counts.get(g,0)}" for g in GROUPS))

# ------------------------------------------------------------
# One NB model, 3-level group
# ------------------------------------------------------------
full = smf.negativebinomial(
    "emotional ~ C(group, Treatment('TD')) * FCz_c", data=d).fit(disp=0)
red = smf.negativebinomial(
    "emotional ~ C(group, Treatment('TD')) + FCz_c", data=d).fit(disp=0)
lr = 2 * (full.llf - red.llf)
lr_p = stats.chi2.sf(lr, 2)
perm_p = freedman_lane_p(d)

rec = dict(edge=EDGE,
           slope_TD=full.params["FCz_c"],
           slope_DLD=full.params["FCz_c"] + full.params[INTER["DLD"]],
           slope_HSL=full.params["FCz_c"] + full.params[INTER["HSL"]],
           inter_beta_DLD=full.params[INTER["DLD"]], inter_z_DLD=full.tvalues[INTER["DLD"]],
           inter_wald_p_DLD=full.pvalues[INTER["DLD"]],
           inter_beta_HSL=full.params[INTER["HSL"]], inter_z_HSL=full.tvalues[INTER["HSL"]],
           inter_wald_p_HSL=full.pvalues[INTER["HSL"]],
           inter_lr_chi2=lr, inter_lr_p=lr_p, inter_perm_p=perm_p,
           alpha=full.params["alpha"])
stat = pd.DataFrame([rec])
stat.to_csv(f"{LANG_DIR}/language_connectivity_emotional_nb_stats.csv", index=False)
print(f"Saved: {LANG_DIR}/language_connectivity_emotional_nb_stats.csv\n")

print(f"NB  emotional ~ group * FCz   ({EDGE})")
print(f"  FC->emotional slope (log-count):  TD={rec['slope_TD']:+.3f}  "
      f"DLD={rec['slope_DLD']:+.3f}  HSL={rec['slope_HSL']:+.3f}")
print(f"  interaction [DLD-TD]: beta={rec['inter_beta_DLD']:+.3f}  z={rec['inter_z_DLD']:+.2f}  "
      f"Wald p={rec['inter_wald_p_DLD']:.3f} {stars(rec['inter_wald_p_DLD'])}")
print(f"  interaction [HSL-TD]: beta={rec['inter_beta_HSL']:+.3f}  z={rec['inter_z_HSL']:+.2f}  "
      f"Wald p={rec['inter_wald_p_HSL']:.3f} {stars(rec['inter_wald_p_HSL'])}")
print(f"  JOINT interaction LR chi2(2)={lr:.3f}, p={lr_p:.4f} {stars(lr_p)}  |  "
      f"Freedman-Lane perm p={perm_p:.4f} {stars(perm_p)}")

# ------------------------------------------------------------
# Predicted mean + 95% CI per group (delta method on linear predictor):
# var(eta) = x' Cov(beta) x on the log-count scale, back-transformed by exp().
# Mirrors the network-size NB scripts (stats_emotional_*_nb_interaction.py).
# ------------------------------------------------------------
zmean = d["FCz"].mean()
design_info = full.model.data.design_info
kf = len(design_info.column_names)
beta = np.asarray(full.params)[:kf]          # mean-structure betas (alpha is last)
cov = np.asarray(full.cov_params())[:kf, :kf]


def predict_band(group, n=120):
    dg = d[d.group == group]
    xs = np.linspace(dg["FCz"].min(), dg["FCz"].max(), n)
    grid = pd.DataFrame({"group": group, "FCz": xs, "FCz_c": xs - zmean})
    X = np.asarray(patsy.dmatrix(design_info, grid))
    eta = X @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", X, cov, X))
    return xs, np.exp(eta), np.exp(eta - 1.96 * se), np.exp(eta + 1.96 * se)


# ------------------------------------------------------------
# Figure: scatter (x=FC z, y=emotional) + NB predicted mean +/- 95% CI, 3 groups
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.6))
pred_rows = []
for g in GROUPS:
    dg = d[d.group == g]
    c = GROUP_COLORS[g]
    xs, mu, lo, hi = predict_band(g)
    ax.fill_between(xs, lo, hi, color=c, alpha=0.15, lw=0)
    ax.plot(xs, mu, color=c, lw=2.2, label=f"{g} (n={counts.get(g,0)})")
    yj = dg["emotional"] + RNG.uniform(-0.15, 0.15, len(dg))
    ax.scatter(dg["FCz"], yj, s=30, alpha=0.8, color=c,
               edgecolor="k", linewidth=0.3)
    for xv, mv, lv, hv in zip(xs, mu, lo, hi):
        pred_rows.append(dict(group=g, FCz=round(xv, 4), mean=round(mv, 4),
                              ci_lo=round(lv, 4), ci_hi=round(hv, 4)))
ax.set_ylim(-0.5, 10.5)
ax.set_xlabel("Language frontal-putamen FC (Fisher z)")
ax.set_ylabel("SDQ emotional")
ax.legend(fontsize=9, loc="best")
ax.grid(alpha=0.2)
ax.set_title("NB-predicted SDQ-emotional vs language frontal-putamen FC, by group\n"
             "shaded = 95% CI on predicted mean;  "
             f"joint interaction LR p={lr_p:.3f} {stars(lr_p)}  "
             f"(perm p={perm_p:.3f})", fontsize=11)
plt.tight_layout()
plt.savefig(f"{LANG_DIR}/language_connectivity_emotional_nb_scatter.png",
            dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved: {LANG_DIR}/language_connectivity_emotional_nb_scatter.png")

pd.DataFrame(pred_rows).to_csv(
    f"{LANG_DIR}/language_connectivity_emotional_nb_predband.csv", index=False)
print(f"Saved: {LANG_DIR}/language_connectivity_emotional_nb_predband.csv")
