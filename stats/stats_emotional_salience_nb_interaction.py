"""
stats_emotional_salience_nb_interaction.py

Single negative-binomial model of SDQ-emotional symptoms on Salience-network
size WITH a group x salience interaction (TD = reference):

    emotional ~ C(group, Treatment('TD')) * salience_c      (log link, NB2)

The log link makes the mean non-negative (no curve needed for the floor at 0)
and the NB dispersion soaks up the DLD over-dispersion. The interaction term
tests whether the salience -> emotional slope differs between DLD and TD.

The figure shows BOTH groups on one panel (like emotional_by_group.png): the
NB-predicted mean curve per group plus a 95% confidence band for the mean,
obtained by the delta method -- se(eta)=sqrt(x' Cov(beta) x) on the linear
predictor, then exp(eta +/- 1.96*se) back through the log link. (This is a CI
for the expected count, not a prediction interval for individuals.)

Input:  results/network_size/group_network_size_long.csv  (Salience size)
        dat_verbgen_scqsdq_subsample.xlsx                  (emotional)
Output: results/network_size/emotional_salience_nb_interaction.csv  (predictions)
        results/network_size/emotional_vs_salience_nb.png

## Author: Han Wang
"""

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
NS_DIR = f"{PROJECT_DIR}/results/network_size"
XLSX = ("/home/hanwang/Documents/Data/ucl/gos_ich/verb_gen_krishnan/"
        "behavioural_scq_sdq/dat_verbgen_scqsdq_subsample.xlsx")
GROUP_COLORS = {"DLD": "#d63031", "TD": "#0984e3"}
RNG = np.random.default_rng(0)

# ------------------------------------------------------------
# Data (same construction as the other emotional-salience scripts)
# ------------------------------------------------------------
beh = pd.read_excel(XLSX)[["code", "emotional"]]
long = pd.read_csv(f"{NS_DIR}/group_network_size_long.csv")
sal = long[long["network_label"] == "Salience"][["code", "group", "network_size_pct"]]
dat = (sal.merge(beh, on="code", how="left")
          .rename(columns={"network_size_pct": "salience"})
          .dropna(subset=["emotional", "salience"]))
dat = dat[dat["group"].isin(["DLD", "TD"])].copy()
dat["emotional"] = dat["emotional"].round().astype(int).clip(0, 10)
sal_mean = dat["salience"].mean()
dat["salience_c"] = dat["salience"] - sal_mean      # global mean-centre

# ------------------------------------------------------------
# (A) NB interaction model
# ------------------------------------------------------------
FORMULA = "emotional ~ C(group, Treatment('TD')) * salience_c"
nb_full = smf.negativebinomial(FORMULA, data=dat).fit(disp=0)
nb_red = smf.negativebinomial(
    "emotional ~ C(group, Treatment('TD')) + salience_c", data=dat).fit(disp=0)

inter = "C(group, Treatment('TD'))[T.DLD]:salience_c"
lr_stat = 2 * (nb_full.llf - nb_red.llf)
lr_p = stats.chi2.sf(lr_stat, 1)

print("=" * 68)
print("NEGATIVE BINOMIAL:  emotional ~ group * salience_c   [TD = reference]")
print("=" * 68)
print(nb_full.summary())
sl_td = nb_full.params["salience_c"]
sl_dld = nb_full.params["salience_c"] + nb_full.params[inter]
print(f"\nSalience slope (log-count):  TD = {sl_td:+.4f} (RR={np.exp(sl_td):.3f}),  "
      f"DLD = {sl_dld:+.4f} (RR={np.exp(sl_dld):.3f})")
print(f"Interaction (DLD - TD) = {nb_full.params[inter]:+.4f}, "
      f"Wald p = {nb_full.pvalues[inter]:.4g}")
print(f"LR test of interaction: chi2(1) = {lr_stat:.3f}, p = {lr_p:.4g}")

# (B) Freedman-Lane permutation (Poisson working model: fast & convergence-stable)

F_FULL = "emotional ~ C(group, Treatment('TD')) * salience_c"
F_RED = "emotional ~ C(group, Treatment('TD')) + salience_c"


pois_full = smf.glm(F_FULL, data=dat, family=sm.families.Poisson()).fit()
pois_red = smf.glm(F_RED, data=dat, family=sm.families.Poisson()).fit()
z_obs = pois_full.tvalues[inter]
mu_red = pois_red.fittedvalues.to_numpy()
resid = dat["emotional"].to_numpy() - mu_red       # response-scale residuals
NPERM = 2000
dat_perm = dat.copy()
n_ge = n_valid = 0
for _ in range(NPERM):
    dat_perm["emotional"] = np.clip(
        np.round(mu_red + resid[RNG.permutation(len(resid))]), 0, None)
    try:
        zp = smf.glm(F_FULL, data=dat_perm,
                     family=sm.families.Poisson()).fit().tvalues[inter]
    except Exception:
        continue
    n_valid += 1
    n_ge += abs(zp) >= abs(z_obs)
p_perm = (1 + n_ge) / (1 + n_valid)

print("\n" + "-" * 68)
print("ROBUSTNESS — two tests of the group x salience interaction")
print("-" * 68)
print(f"  (A) NB likelihood-ratio        chi2(1) = {lr_stat:5.3f}   p = {lr_p:.4f}")
print(f"  (B) Freedman-Lane permutation  ({n_valid} perms)    p = {p_perm:.4f}")

pd.DataFrame([
    dict(test="NB_likelihood_ratio", statistic=round(lr_stat, 3), p=round(lr_p, 4)),
    dict(test="Freedman_Lane_permutation", statistic=n_valid, p=round(p_perm, 4)),
]).to_csv(f"{NS_DIR}/emotional_salience_interaction_robustness.csv", index=False)
print(f"Saved: {NS_DIR}/emotional_salience_interaction_robustness.csv")

# ------------------------------------------------------------
# Predicted mean + 95% CI per group (delta method on the linear predictor)
# ------------------------------------------------------------
design_info = nb_full.model.data.design_info
k = len(design_info.column_names)              # regression cols (alpha excluded)
beta = np.asarray(nb_full.params)[:k]
cov = np.asarray(nb_full.cov_params())[:k, :k]


def predict_band(group, n=200):
    d = dat[dat["group"] == group]
    xs = np.linspace(d["salience"].min(), d["salience"].max(), n)
    grid = pd.DataFrame({"group": group, "salience": xs, "salience_c": xs - sal_mean})
    X = np.asarray(patsy.dmatrix(design_info, grid))
    eta = X @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", X, cov, X))   # diag(X Cov X')
    return xs, np.exp(eta), np.exp(eta - 1.96 * se), np.exp(eta + 1.96 * se)


pred_rows = []
fig, ax = plt.subplots(figsize=(7, 5.5))
for g in ["DLD", "TD"]:
    d = dat[dat["group"] == g]
    c = GROUP_COLORS[g]
    xs, mu, lo, hi = predict_band(g)
    ax.fill_between(xs, lo, hi, color=c, alpha=0.18, lw=0)
    ax.plot(xs, mu, color=c, lw=2.4,
            label=f"{g} (n={len(d)}): RR={np.exp(sl_dld if g=='DLD' else sl_td):.2f}/+1%")
    yj = d["emotional"] + RNG.uniform(-0.12, 0.12, len(d))      # tiny jitter for ties
    ax.scatter(d["salience"], yj, color=c, edgecolor="k", linewidth=0.3,
               s=42, zorder=3, alpha=0.9)
    for xv, mv, lv, hv in zip(xs, mu, lo, hi):
        pred_rows.append(dict(group=g, salience=round(xv, 4), mean=round(mv, 4),
                              ci_lo=round(lv, 4), ci_hi=round(hv, 4)))

ax.set_xlabel("Salience network size (% cortical surface)")
ax.set_ylabel("SDQ emotional symptoms (0-10)")
ax.set_ylim(-0.5, 10.5)
ax.set_title("Emotional symptoms vs Salience size — NB fit (mean ± 95% CI)\n"
             f"group × salience interaction LR p = {lr_p:.3f}", fontsize=12)
ax.grid(alpha=0.25); ax.set_axisbelow(True)
ax.legend(title="NB predicted mean", fontsize=9, loc="upper center")
ax.text(0.99, 0.01, "points y-jittered ±0.12 for visibility",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.4")
plt.tight_layout()
out_png = f"{NS_DIR}/emotional_vs_salience_nb.png"
plt.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close()

pd.DataFrame(pred_rows).to_csv(
    f"{NS_DIR}/emotional_salience_nb_interaction.csv", index=False)
print(f"\nSaved: {out_png}")
print(f"Saved: {NS_DIR}/emotional_salience_nb_interaction.csv")
