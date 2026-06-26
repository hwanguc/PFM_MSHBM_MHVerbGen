"""
stats_emotional_salience_interaction.py

Formally test whether the relationship between Salience-network size and SDQ
emotional symptoms differs between DLD and TD (a group x salience interaction).

Three approaches:
  1. OLS regression  emotional ~ group * salience_c   (salience mean-centred).
     The interaction coefficient IS the slope difference (DLD slope - TD slope);
     reported with classical and HC3 heteroscedasticity-robust SEs, since the
     outcome is a skewed bounded count with a floor in TD.
  2. Fisher r-to-z   comparing the two independent within-group correlations.
  3. Bootstrap CI    on the slope difference (resampling within each group) as a
     distribution-free robustness check.

Input:  results/network_size/group_network_size_long.csv  (Salience size)
        dat_verbgen_scqsdq_subsample.xlsx                  (emotional)
Output: console summary (+ reuses scatter from stats_emotional_salience.py)

## Author: Han Wang
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
NS_DIR = f"{PROJECT_DIR}/results/network_size"
XLSX = ("/home/hanwang/Documents/Data/ucl/gos_ich/verb_gen_krishnan/"
        "behavioural_scq_sdq/dat_verbgen_scqsdq_subsample.xlsx")
RNG = np.random.default_rng(0)
NBOOT = 10000

# ------------------------------------------------------------
beh = pd.read_excel(XLSX)[["code", "emotional"]]
long = pd.read_csv(f"{NS_DIR}/group_network_size_long.csv")
sal = long[long["network_label"] == "Salience"][["code", "group", "network_size_pct"]]
dat = sal.merge(beh, on="code", how="left").rename(columns={"network_size_pct": "salience"})
dat = dat[dat["group"].isin(["DLD", "TD"])].copy()
dat["salience_c"] = dat["salience"] - dat["salience"].mean()

# ------------------------------------------------------------
# 1. OLS interaction model (TD = reference)
# ------------------------------------------------------------
print("=" * 66)
print("(1) OLS:  emotional ~ C(group) * salience_c     [TD = reference]")
print("=" * 66)
model = smf.ols("emotional ~ C(group, Treatment('TD')) * salience_c", data=dat)
fit = model.fit()
fit_hc3 = model.fit(cov_type="HC3")

inter = "C(group, Treatment('TD'))[T.DLD]:salience_c"
print(fit.summary().tables[1])
print(f"\nInteraction term = slope(DLD) - slope(TD):")
print(f"  beta = {fit.params[inter]:+.3f}")
print(f"  classical: t = {fit.tvalues[inter]:.3f}, p = {fit.pvalues[inter]:.4g}")
print(f"  HC3 robust: t = {fit_hc3.tvalues[inter]:.3f}, p = {fit_hc3.pvalues[inter]:.4g}")
ci = fit.conf_int().loc[inter]
print(f"  95% CI (classical): [{ci[0]:+.3f}, {ci[1]:+.3f}]")

# per-group slopes for context
sl_td = fit.params["salience_c"]
sl_dld = fit.params["salience_c"] + fit.params[inter]
print(f"\n  Implied slopes (emotional points per +1% salience):  TD = {sl_td:+.3f},  DLD = {sl_dld:+.3f}")
print(f"  Model R^2 = {fit.rsquared:.3f}")

# ------------------------------------------------------------
# 2. Fisher r-to-z on the two independent correlations
# ------------------------------------------------------------
print("\n" + "=" * 66)
print("(2) Fisher r-to-z: difference between within-group correlations")
print("=" * 66)
res = {}
for g in ["DLD", "TD"]:
    d = dat[dat["group"] == g]
    r, p = stats.pearsonr(d["salience"], d["emotional"])
    res[g] = (r, len(d))
    print(f"  {g}: r = {r:+.3f}  (n = {len(d)})")
r1, n1 = res["DLD"]; r2, n2 = res["TD"]
z1, z2 = np.arctanh(r1), np.arctanh(r2)
se = np.sqrt(1/(n1-3) + 1/(n2-3))
z = (z1 - z2) / se
p_fisher = 2 * stats.norm.sf(abs(z))
print(f"  z = {z:.3f},  p = {p_fisher:.4g}  (two-sided)")

# ------------------------------------------------------------
# 3. Bootstrap CI on slope difference (resample within group)
# ------------------------------------------------------------
print("\n" + "=" * 66)
print("(3) Bootstrap CI on slope difference (DLD - TD), within-group resampling")
print("=" * 66)
def slope(d):
    return np.polyfit(d["salience"], d["emotional"], 1)[0]
diffs = np.empty(NBOOT)
dld_d = dat[dat["group"] == "DLD"]; td_d = dat[dat["group"] == "TD"]
for b in range(NBOOT):
    bd = dld_d.iloc[RNG.integers(0, len(dld_d), len(dld_d))]
    bt = td_d.iloc[RNG.integers(0, len(td_d), len(td_d))]
    diffs[b] = slope(bd) - slope(bt)
lo, hi = np.percentile(diffs, [2.5, 97.5])
p_boot = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
print(f"  slope diff (point) = {slope(dld_d) - slope(td_d):+.3f}")
print(f"  bootstrap 95% CI = [{lo:+.3f}, {hi:+.3f}]")
print(f"  bootstrap two-sided p (CI excludes 0?) = {p_boot:.4g}"
      + ("  -> excludes 0" if lo > 0 or hi < 0 else "  -> includes 0"))

print("\nNOTE: TD emotional scores are floor-bounded (mostly 0, max 2), so all of")
print("these rest on a thin TD signal. Treat a significant interaction as")
print("suggestive pending replication; a count GLM (Poisson/NB) is a further check.")
