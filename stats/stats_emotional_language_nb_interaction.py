"""
stats_emotional_language_nb_interaction.py

Negative-binomial model of SDQ-emotional symptoms on Language-network size with a
group x language interaction, now with the 3-level group factor (TD reference)
over all 144 subjects:

    emotional ~ C(group, Treatment('TD')) * language_c      (log link, NB2)

The log link keeps the mean non-negative (handles the floor at 0 without a curve)
and the NB dispersion soaks up the DLD over-dispersion. TD's language slope is the
Lynch-style positive control (higher language -> more emotional symptoms in
controls); the two interaction terms test whether DLD and HSL depart from it.

Robustness of the interaction:
  (A) NB joint likelihood-ratio test of BOTH interaction terms (2 df).
  (B) Freedman-Lane permutation using a Poisson working model, statistic = the
      joint interaction LR (deviance drop, full vs additive) -- distribution-free.

Variant selects which MS-HBM set the Language size comes from (full / icafix).

Input:  results/network_size_<variant>/group_network_size_long.csv (Language size)
        dat_verbgen_analysis_144.csv                                (emotional)
Output: results/network_size_<variant>/emotional_language_nb_interaction.csv
        results/network_size_<variant>/emotional_language_interaction_robustness.csv
        results/network_size_<variant>/emotional_vs_language_nb.png

## Author: Han Wang
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import patsy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
LISTCSV = ("/home/hanwang/Documents/Data/verb_gen_krishnan/"
           "behavioural_scq_sdq/dat_verbgen_analysis_144.csv")
GROUPS = ["DLD", "HSL", "TD"]
GROUP_COLORS = {"DLD": "#d63031", "HSL": "#2ca02c", "TD": "#0984e3"}
INTER = {"DLD": "C(group, Treatment('TD'))[T.DLD]:language_c",
         "HSL": "C(group, Treatment('TD'))[T.HSL]:language_c"}
MAIN = {"DLD": "C(group, Treatment('TD'))[T.DLD]",
        "HSL": "C(group, Treatment('TD'))[T.HSL]"}
RNG = np.random.default_rng(0)

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["full", "icafix"], default="full")
args = ap.parse_args()
NS_DIR = f"{PROJECT_DIR}/results/network_size_{args.variant}"

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
beh = pd.read_csv(LISTCSV)[["code", "emotional"]]
beh["code"] = beh["code"].astype(str)
long = pd.read_csv(f"{NS_DIR}/group_network_size_long.csv")
long["code"] = long["code"].astype(str)
sal = long[long["network_label"] == "Language"][["code", "group", "network_size_pct"]]
dat = (sal.merge(beh, on="code", how="left")
          .rename(columns={"network_size_pct": "language"})
          .dropna(subset=["emotional", "language"]))
dat = dat[dat["group"].isin(GROUPS)].copy()
dat["emotional"] = dat["emotional"].round().astype(int).clip(0, 10)
sal_mean = dat["language"].mean()
dat["language_c"] = dat["language"] - sal_mean
print(f"[{args.variant}] n={len(dat)} | "
      + dat["group"].value_counts().to_dict().__str__())

# ------------------------------------------------------------
# (A) NB interaction model, 3-level group
# ------------------------------------------------------------
F_FULL = "emotional ~ C(group, Treatment('TD')) * language_c"
F_RED = "emotional ~ C(group, Treatment('TD')) + language_c"
nb_full = smf.negativebinomial(F_FULL, data=dat).fit(disp=0, maxiter=1000)
nb_red = smf.negativebinomial(F_RED, data=dat).fit(disp=0, maxiter=1000)
lr_stat = 2 * (nb_full.llf - nb_red.llf)
lr_p = stats.chi2.sf(lr_stat, 2)              # joint: both interaction terms

sl = {"TD": nb_full.params["language_c"]}
sl["DLD"] = sl["TD"] + nb_full.params[INTER["DLD"]]
sl["HSL"] = sl["TD"] + nb_full.params[INTER["HSL"]]

print("=" * 70)
print(f"NB: emotional ~ group * language_c  [TD ref]  (variant={args.variant})")
print("=" * 70)
print("Language slope (log-count) per group:")
for g in GROUPS:
    print(f"  {g}: {sl[g]:+.4f}  (RR={np.exp(sl[g]):.3f} per +1% cortex)")
print(f"TD slope (positive control) Wald p = {nb_full.pvalues['language_c']:.4g}")
for g in ["DLD", "HSL"]:
    print(f"  interaction {g}-TD = {nb_full.params[INTER[g]]:+.4f}, "
          f"Wald p = {nb_full.pvalues[INTER[g]]:.4g}")
print(f"Joint interaction LR chi2(2) = {lr_stat:.3f}, p = {lr_p:.4g}")

# ------------------------------------------------------------
# (B) Freedman-Lane permutation (Poisson working model; joint LR statistic)
# ------------------------------------------------------------
def pois_joint_lr(data):
    f = smf.glm(F_FULL, data=data, family=sm.families.Poisson()).fit()
    r = smf.glm(F_RED, data=data, family=sm.families.Poisson()).fit()
    return r.deviance - f.deviance          # = 2*(llf_full - llf_red)

lr_obs = pois_joint_lr(dat)
pois_red = smf.glm(F_RED, data=dat, family=sm.families.Poisson()).fit()
mu_red = pois_red.fittedvalues.to_numpy()
resid = dat["emotional"].to_numpy() - mu_red
NPERM = 2000
dperm = dat.copy()
n_ge = n_valid = 0
for _ in range(NPERM):
    dperm["emotional"] = np.clip(np.round(mu_red + resid[RNG.permutation(len(resid))]), 0, None)
    try:
        lrp = pois_joint_lr(dperm)
    except Exception:
        continue
    n_valid += 1
    n_ge += lrp >= lr_obs
p_perm = (1 + n_ge) / (1 + n_valid)

print("\n" + "-" * 70)
print("ROBUSTNESS — joint group x language interaction")
print(f"  (A) NB likelihood-ratio        chi2(2) = {lr_stat:5.3f}   p = {lr_p:.4f}")
print(f"  (B) Freedman-Lane permutation  ({n_valid} perms)      p = {p_perm:.4f}")

pd.DataFrame([
    dict(test="NB_joint_LR_chi2_2df", statistic=round(lr_stat, 3), p=round(lr_p, 4)),
    dict(test="Freedman_Lane_permutation", statistic=n_valid, p=round(p_perm, 4)),
]).to_csv(f"{NS_DIR}/emotional_language_interaction_robustness.csv", index=False)
print(f"Saved: {NS_DIR}/emotional_language_interaction_robustness.csv")

# ------------------------------------------------------------
# Predicted mean + 95% CI per group (delta method on linear predictor)
# ------------------------------------------------------------
design_info = nb_full.model.data.design_info
k = len(design_info.column_names)
beta = np.asarray(nb_full.params)[:k]
cov = np.asarray(nb_full.cov_params())[:k, :k]


def predict_band(group, n=200):
    d = dat[dat["group"] == group]
    xs = np.linspace(d["language"].min(), d["language"].max(), n)
    grid = pd.DataFrame({"group": group, "language": xs, "language_c": xs - sal_mean})
    X = np.asarray(patsy.dmatrix(design_info, grid))
    eta = X @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", X, cov, X))
    return xs, np.exp(eta), np.exp(eta - 1.96 * se), np.exp(eta + 1.96 * se)


pred_rows = []
fig, ax = plt.subplots(figsize=(7.4, 5.6))
for g in GROUPS:
    d = dat[dat["group"] == g]
    c = GROUP_COLORS[g]
    xs, mu, lo, hi = predict_band(g)
    ax.fill_between(xs, lo, hi, color=c, alpha=0.15, lw=0)
    ax.plot(xs, mu, color=c, lw=2.4,
            label=f"{g} (n={len(d)}): RR={np.exp(sl[g]):.2f}/+1%")
    yj = d["emotional"] + RNG.uniform(-0.12, 0.12, len(d))
    ax.scatter(d["language"], yj, color=c, edgecolor="k", linewidth=0.3,
               s=34, zorder=3, alpha=0.85)
    for xv, mv, lv, hv in zip(xs, mu, lo, hi):
        pred_rows.append(dict(group=g, language=round(xv, 4), mean=round(mv, 4),
                              ci_lo=round(lv, 4), ci_hi=round(hv, 4)))

ax.set_xlabel("Language network size (% cortical surface)")
ax.set_ylabel("SDQ emotional symptoms (0-10)")
ax.set_ylim(-0.5, 10.5)
ax.set_title("Emotional symptoms vs Language size — NB fit (mean ± 95% CI)\n"
             f"joint group × language interaction LR p = {lr_p:.3f} "
             f"(variant={args.variant})", fontsize=12)
ax.grid(alpha=0.25); ax.set_axisbelow(True)
ax.legend(title="NB predicted mean", fontsize=9, loc="upper center")
plt.tight_layout()
out_png = f"{NS_DIR}/emotional_vs_language_nb.png"
plt.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close()

pd.DataFrame(pred_rows).to_csv(
    f"{NS_DIR}/emotional_language_nb_interaction.csv", index=False)
print(f"\nSaved: {out_png}")
print(f"Saved: {NS_DIR}/emotional_language_nb_interaction.csv")
