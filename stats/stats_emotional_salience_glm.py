"""
stats_emotional_salience_glm.py

Test the Salience-size -> SDQ-emotional relationship with models that respect
the bounded, floored response, instead of OLS / a hand-fit exponential:

  * NEGATIVE BINOMIAL (count, log link).  E[y] = exp(b0 + b1*salience).
    - The log link makes the mean non-negative by construction, so the "floor"
      at 0 needs no curve. Crucially, the exp() link means the apparent
      e^{+x} rise in TD is just a LINEAR salience effect on the log scale --
      this model tests that effect directly (b1, no extra curvature granted).
    - The dispersion parameter alpha soaks up the DLD over-dispersion (0s next
      to 9s/10s) that a Poisson would choke on. We LR-test alpha>0 to justify NB.

  * BETA-BINOMIAL (bounded 0..10, the actual data-generating process).
    SDQ-emotional = sum of 5 items each 0..2 => k successes out of n=10.
    p = inv_logit(b0 + b1*salience); a Beta(theta) mixing prior adds the
    over-dispersion. This is the only family here that honours BOTH the floor
    (0) and the ceiling (10). We LR-test against a plain binomial.

For each model we fit (a) within DLD and within TD, and (b) a pooled
group * salience interaction (the formal "does the slope differ by group?"),
with a likelihood-ratio test of the interaction.

NOTE on the bounds you raised:
  - Response 0..10  -> handled by NB (floor only) and beta-binomial (floor+ceiling).
  - Predictor salience in [0,1] -> as a PREDICTOR boundedness needs no special
    model; we keep it in % and mean-centre it. Over the observed ~5-9% window
    exp(b*x) is nearly linear, which is why the earlier LOESS/exp barely beat a
    line. Coordinates: 'salience' is % cortical surface (network_size_pct).

Input:  results/network_size/group_network_size_long.csv  (Salience size)
        dat_verbgen_scqsdq_subsample.xlsx                  (emotional)
Output: results/network_size/emotional_salience_glm.csv
        results/network_size/emotional_vs_salience_glm.png

## Author: Han Wang
"""

import warnings
import numpy as np
import pandas as pd
import patsy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit
from scipy.stats import betabinom
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.base.model import GenericLikelihoodModel

warnings.filterwarnings("ignore")

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
NS_DIR = f"{PROJECT_DIR}/results/network_size"
XLSX = ("/home/hanwang/Documents/Data/ucl/gos_ich/verb_gen_krishnan/"
        "behavioural_scq_sdq/dat_verbgen_scqsdq_subsample.xlsx")
GROUP_COLORS = {"DLD": "#d63031", "TD": "#0984e3"}
N_TRIALS = 10            # SDQ emotional = 5 items x (0,1,2) = 0..10

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
dat["emotional"] = dat["emotional"].round().astype(int).clip(0, N_TRIALS)
dat["salience_c"] = dat["salience"] - dat["salience"].mean()   # global centre

rows = []   # tidy results for CSV


# ============================================================
# Beta-binomial GLM via maximum likelihood (GenericLikelihoodModel)
#   mu_i = inv_logit(x_i'beta);  k_i ~ BetaBinom(n=10, a=mu*theta, b=(1-mu)*theta)
#   theta = exp(log_theta) > 0  (theta->inf -> plain binomial)
#   intra-class corr rho = 1 / (theta + 1)
# ============================================================
class BetaBinomGLM(GenericLikelihoodModel):
    def __init__(self, endog, exog, n_trials=N_TRIALS, **kwds):
        self.n_trials = n_trials
        super().__init__(endog, exog, extra_params_names=["log_theta"], **kwds)

    def nloglikeobs(self, params):
        beta = params[:-1]
        theta = np.exp(params[-1])
        mu = np.clip(expit(np.asarray(self.exog) @ beta), 1e-6, 1 - 1e-6)
        a, b = mu * theta, (1 - mu) * theta
        return -betabinom.logpmf(np.asarray(self.endog).ravel(), self.n_trials, a, b)

    def fit(self, start_params=None, **kwds):
        if start_params is None:
            p0 = np.clip(np.mean(self.endog) / self.n_trials, 1e-2, 1 - 1e-2)
            start_params = np.zeros(self.exog.shape[1] + 1)
            start_params[0] = np.log(p0 / (1 - p0))   # intercept ~ logit(rate)
            start_params[-1] = np.log(5.0)            # mild over-dispersion
        kwds.setdefault("method", "bfgs")
        kwds.setdefault("maxiter", 2000)
        kwds.setdefault("disp", 0)
        return super().fit(start_params=start_params, **kwds)


def fit_betabinom(formula, data):
    y, X = patsy.dmatrices(formula, data, return_type="dataframe")
    res = BetaBinomGLM(y.values.ravel(), X.values).fit()
    names = list(X.columns) + ["log_theta"]
    return res, names, list(X.columns)


def lr_test(llf_full, llf_red, df):
    stat = 2 * (llf_full - llf_red)
    return stat, stats.chi2.sf(stat, df)


# ============================================================
# (1) NEGATIVE BINOMIAL
# ============================================================
print("=" * 70)
print("(1) NEGATIVE BINOMIAL   emotional ~ salience   (log link)")
print("=" * 70)

for g in ["DLD", "TD"]:
    d = dat[dat["group"] == g]
    nb = smf.negativebinomial("emotional ~ salience_c", data=d).fit(disp=0)
    po = smf.poisson("emotional ~ salience_c", data=d).fit(disp=0)
    b = nb.params["salience_c"]; p = nb.pvalues["salience_c"]
    ci = nb.conf_int().loc["salience_c"]
    od_stat, od_p = lr_test(nb.llf, po.llf, 1)       # boundary test (alpha=0)
    print(f"\n--- {g} (n={len(d)}) ---")
    print(f"  salience slope (log-count) b = {b:+.4f}, p = {p:.4g}")
    print(f"  rate ratio per +1% salience  = {np.exp(b):.3f}  "
          f"95% CI [{np.exp(ci[0]):.3f}, {np.exp(ci[1]):.3f}]")
    print(f"  dispersion alpha = {nb.params['alpha']:.3f}  "
          f"(NB>Poisson LR chi2={od_stat:.2f}, p~{od_p/2:.4g})")
    rows.append(dict(model="NB", scope=g, n=len(d), term="salience_c",
                     estimate=round(b, 4), p=round(p, 4),
                     effect=f"RR={np.exp(b):.3f}"))

# pooled interaction
nb_full = smf.negativebinomial(
    "emotional ~ C(group, Treatment('TD')) * salience_c", data=dat).fit(disp=0)
nb_red = smf.negativebinomial(
    "emotional ~ C(group, Treatment('TD')) + salience_c", data=dat).fit(disp=0)
inter = "C(group, Treatment('TD'))[T.DLD]:salience_c"
lr_stat, lr_p = lr_test(nb_full.llf, nb_red.llf, 1)
print("\n--- pooled group x salience interaction (NB) ---")
print(f"  interaction b = {nb_full.params[inter]:+.4f}, "
      f"Wald p = {nb_full.pvalues[inter]:.4g}")
print(f"  LR test (full vs no-interaction): chi2 = {lr_stat:.3f}, p = {lr_p:.4g}")
rows.append(dict(model="NB", scope="interaction", n=len(dat), term=inter,
                 estimate=round(nb_full.params[inter], 4),
                 p=round(nb_full.pvalues[inter], 4), effect=f"LRp={lr_p:.4g}"))


# ============================================================
# (2) BETA-BINOMIAL
# ============================================================
print("\n" + "=" * 70)
print("(2) BETA-BINOMIAL   emotional/10 ~ salience   (logit link, floor+ceiling)")
print("=" * 70)

for g in ["DLD", "TD"]:
    d = dat[dat["group"] == g]
    res, names, xcols = fit_betabinom("emotional ~ salience_c", d)
    # plain binomial GLM for the over-dispersion LR test
    yb = np.c_[d["emotional"].values, N_TRIALS - d["emotional"].values]
    Xb = patsy.dmatrices("emotional ~ salience_c", d, return_type="dataframe")[1]
    binom = sm.GLM(yb, Xb, family=sm.families.Binomial()).fit()
    j = names.index("salience_c")
    b, se = res.params[j], res.bse[j]
    p = 2 * stats.norm.sf(abs(b / se))
    theta = np.exp(res.params[-1]); rho = 1 / (theta + 1)
    od_stat, od_p = lr_test(res.llf, binom.llf, 1)   # boundary test (theta->inf)
    print(f"\n--- {g} (n={len(d)}) ---")
    print(f"  salience slope (logit-rate) b = {b:+.4f}, p = {p:.4g}")
    print(f"  theta = {theta:.2f}  (intra-class rho = {rho:.3f}; "
          f"BB>Binom LR chi2={od_stat:.2f}, p~{od_p/2:.4g})")
    rows.append(dict(model="BetaBinom", scope=g, n=len(d), term="salience_c",
                     estimate=round(b, 4), p=round(p, 4), effect=f"rho={rho:.3f}"))

# pooled interaction
f_full = "emotional ~ C(group, Treatment('TD')) * salience_c"
f_red = "emotional ~ C(group, Treatment('TD')) + salience_c"
res_full, names_full, _ = fit_betabinom(f_full, dat)
res_red, _, _ = fit_betabinom(f_red, dat)
ji = names_full.index(inter)
bI, seI = res_full.params[ji], res_full.bse[ji]
pI = 2 * stats.norm.sf(abs(bI / seI))
lr_stat_bb, lr_p_bb = lr_test(res_full.llf, res_red.llf, 1)
print("\n--- pooled group x salience interaction (Beta-binomial) ---")
print(f"  interaction b = {bI:+.4f}, Wald p = {pI:.4g}")
print(f"  LR test (full vs no-interaction): chi2 = {lr_stat_bb:.3f}, p = {lr_p_bb:.4g}")
rows.append(dict(model="BetaBinom", scope="interaction", n=len(dat), term=inter,
                 estimate=round(bI, 4), p=round(pI, 4), effect=f"LRp={lr_p_bb:.4g}"))

pd.DataFrame(rows).to_csv(f"{NS_DIR}/emotional_salience_glm.csv", index=False)
print(f"\nSaved: {NS_DIR}/emotional_salience_glm.csv")


# ============================================================
# Figure: fitted mean curves (on the 0..10 scale) from both models
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
xc_mean = dat["salience"].mean()
for ax, g in zip(axes, ["DLD", "TD"]):
    d = dat[dat["group"] == g]
    c = GROUP_COLORS[g]
    xs = np.linspace(d["salience"].min(), d["salience"].max(), 200)
    xs_c = xs - xc_mean
    # NB predicted mean = exp(eta)
    nb = smf.negativebinomial("emotional ~ salience_c", data=d).fit(disp=0)
    nb_mu = np.exp(nb.params["Intercept"] + nb.params["salience_c"] * xs_c)
    # BB predicted mean = n * inv_logit(eta)
    res, names, _ = fit_betabinom("emotional ~ salience_c", d)
    bb_mu = N_TRIALS * expit(res.params[0] + res.params[1] * xs_c)

    ax.scatter(d["salience"], d["emotional"], color=c, edgecolor="k",
               linewidth=0.3, s=45, zorder=3, label="subjects")
    ax.plot(xs, nb_mu, color=c, lw=2.2, ls="-", label="NB mean")
    ax.plot(xs, bb_mu, color="k", lw=1.8, ls="--", label="Beta-binom mean")
    ax.set_title(f"{g} (n={len(d)})")
    ax.set_xlabel("Salience network size (% cortical surface)")
    ax.grid(alpha=0.25); ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="best")
axes[0].set_ylabel("SDQ emotional symptoms (0-10)")
axes[0].set_ylim(-0.5, 10.5)
fig.suptitle("Emotional symptoms vs Salience size — NB & beta-binomial fits",
             fontsize=13)
plt.tight_layout()
out_png = f"{NS_DIR}/emotional_vs_salience_glm.png"
plt.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out_png}")

print("\nREAD-OUT: the salience effect within each group, and the group x salience")
print("interaction, are what to report. Both models agree the floor needs no")
print("special curve -- the link supplies the shape. TD floor is still thin, so")
print("treat as a robustness check / hypothesis-generating for the larger study.")
