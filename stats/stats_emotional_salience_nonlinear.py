"""
stats_emotional_salience_nonlinear.py

EXPLORATORY: is the emotional-symptoms vs Salience-network-size relationship
non-linear, and does the curvature differ by group? The scatter hints at an
increasing (concave-up, ~e^{+x}) trend in TD and a decreasing (~e^{-x}) trend
in DLD. Here we go beyond the straight-line fit in stats_emotional_salience.py.

Per group (DLD, TD) we fit and compare:
  - linear            emotional ~ b0 + b1*x
  - quadratic         emotional ~ b0 + b1*x + b2*x^2   (curvature sign = b2)
  - exponential       emotional ~ a*exp(b*x)           (b>0 = e^{+x}, b<0 = e^{-x})
  - LOESS             local regression smoother (assumption-free shape)

Because n is small (~16 DLD / 20 TD) and the in-sample fit of a flexible curve
is optimistic, every parametric model is scored with leave-one-out cross-
validated RMSE (honest out-of-sample error) alongside in-sample R^2 / AIC.
A curvature term that does not improve LOO-CV over the straight line is
overfitting, not signal.

Input:  results/network_size/group_network_size_long.csv  (Salience size)
        dat_verbgen_scqsdq_subsample.xlsx                  (emotional)
Output: results/network_size/emotional_salience_nonlinear.csv
        results/network_size/emotional_vs_salience_nonlinear.png

## Author: Han Wang
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
NS_DIR = f"{PROJECT_DIR}/results/network_size"
XLSX = ("/home/hanwang/Documents/Data/ucl/gos_ich/verb_gen_krishnan/"
        "behavioural_scq_sdq/dat_verbgen_scqsdq_subsample.xlsx")
GROUP_COLORS = {"DLD": "#d63031", "TD": "#0984e3"}
LOESS_FRAC = 0.8          # large frac -> smooth, since n per group is small
RNG = np.random.default_rng(0)

# ------------------------------------------------------------
# Data (same construction as stats_emotional_salience.py)
# ------------------------------------------------------------
beh = pd.read_excel(XLSX)[["code", "emotional"]]
long = pd.read_csv(f"{NS_DIR}/group_network_size_long.csv")
sal = long[long["network_label"] == "Salience"][["code", "group", "network_size_pct"]]
dat = sal.merge(beh, on="code", how="left").rename(columns={"network_size_pct": "salience"})
dat = dat[dat["group"].isin(["DLD", "TD"])].dropna(subset=["emotional", "salience"]).copy()


# ------------------------------------------------------------
# Model definitions: each returns predict(xnew) given (x, y)
# ------------------------------------------------------------
def fit_poly(x, y, deg):
    b = np.polyfit(x, y, deg)
    return lambda xn: np.polyval(b, xn), deg + 1, b


def fit_exp(x, y):
    """emotional ~ a*exp(b*x); robust-ish init, fall back if it won't converge."""
    p0 = [max(y.mean(), 0.1), 0.0]
    try:
        popt, _ = curve_fit(lambda xx, a, b: a * np.exp(b * xx), x, y,
                            p0=p0, maxfev=20000)
        return (lambda xn: popt[0] * np.exp(popt[1] * xn)), 2, popt
    except Exception:
        return None, 2, None


def metrics(y, yhat, k):
    """In-sample R^2 and AIC (Gaussian) for n obs, k params."""
    n = len(y)
    rss = np.sum((y - yhat) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - rss / tss if tss > 0 else np.nan
    sigma2 = rss / n
    aic = n * np.log(sigma2 + 1e-12) + 2 * k if sigma2 > 0 else -np.inf
    return r2, aic, rss


def loo_rmse(x, y, fitter):
    """Leave-one-out CV RMSE for a fitter(x_train, y_train)->predict func."""
    n = len(y)
    err = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        pred, _, params = fitter(x[m], y[m])
        if pred is None or params is None:
            return np.nan
        err.append((y[i] - pred(x[i])) ** 2)
    return np.sqrt(np.mean(err))


# ------------------------------------------------------------
# Fit & score per group
# ------------------------------------------------------------
rows = []
fits = {}      # group -> dict of dense curves for plotting
print("=" * 70)
print("NON-LINEAR EXPLORATION: emotional ~ f(Salience size), per group")
print(f"(LOESS frac = {LOESS_FRAC}; LOO-CV RMSE = honest out-of-sample error)")
print("=" * 70)

candidates = {
    "linear":      lambda x, y: fit_poly(x, y, 1),
    "quadratic":   lambda x, y: fit_poly(x, y, 2),
    "exponential": fit_exp,
}

for g in ["DLD", "TD"]:
    d = dat[dat["group"] == g].sort_values("salience")
    x = d["salience"].to_numpy(float)
    y = d["emotional"].to_numpy(float)
    xs = np.linspace(x.min(), x.max(), 200)
    fits[g] = {"x": x, "y": y, "xs": xs}

    print(f"\n--- {g} (n={len(d)}) ---")
    print(f"{'model':<12}{'k':>3}{'R2':>9}{'AIC':>10}{'LOO-RMSE':>11}   note")
    base_loo = None
    for name, fitter in candidates.items():
        pred, k, params = fitter(x, y)
        if pred is None:
            print(f"{name:<12}{'-':>3}{'fit failed':>9}")
            continue
        yhat = pred(x)
        r2, aic, _ = metrics(y, yhat, k)
        loo = loo_rmse(x, y, fitter)
        fits[g][name] = pred(xs)
        note = ""
        if name == "linear":
            base_loo = loo
        elif base_loo is not None and not np.isnan(loo):
            note = ("improves on linear" if loo < base_loo
                    else "WORSE than linear (overfit)")
        if name == "quadratic" and params is not None:
            note += f"  curv b2={params[0]:+.4f}"
        if name == "exponential" and params is not None:
            shape = "e^+x (concave-up)" if params[1] > 0 else "e^-x (concave-down)"
            note += f"  b={params[1]:+.4f} -> {shape}"
        print(f"{name:<12}{k:>3}{r2:>9.3f}{aic:>10.2f}{loo:>11.3f}   {note}")
        rows.append(dict(group=g, n=len(d), model=name, k=k,
                         r2=round(r2, 4), aic=round(aic, 3),
                         loo_rmse=round(loo, 4),
                         param_b=(round(params[1], 5) if name == "exponential"
                                  else (round(params[0], 5) if name == "quadratic"
                                        else np.nan))))

    # LOESS smoother (no params to count; report as shape only)
    lo = lowess(y, x, frac=LOESS_FRAC, return_sorted=True)
    fits[g]["loess_x"], fits[g]["loess_y"] = lo[:, 0], lo[:, 1]

    # Spearman (monotonic) as a rank-based sanity check on direction
    rho, rho_p = stats.spearmanr(x, y)
    print(f"Spearman (monotonic) rho = {rho:+.3f}, p = {rho_p:.4g}")
    rows.append(dict(group=g, n=len(d), model="spearman", k=np.nan,
                     r2=np.nan, aic=np.nan, loo_rmse=np.nan, param_b=round(rho, 4)))

out_csv = f"{NS_DIR}/emotional_salience_nonlinear.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# ------------------------------------------------------------
# Figure: scatter + LOESS + exponential, one panel per group
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, g in zip(axes, ["DLD", "TD"]):
    f = fits[g]
    c = GROUP_COLORS[g]
    ax.scatter(f["x"], f["y"], color=c, edgecolor="k", linewidth=0.3, s=45,
               zorder=3, label="subjects")
    if "linear" in f:
        ax.plot(f["xs"], f["linear"], color="grey", lw=1.2, ls="--", label="linear")
    if "exponential" in f:
        ax.plot(f["xs"], f["exponential"], color=c, lw=1.6, ls=":",
                label="exponential")
    ax.plot(f["loess_x"], f["loess_y"], color=c, lw=2.4,
            label=f"LOESS (frac={LOESS_FRAC})")
    ax.set_title(f"{g} (n={len(f['x'])})")
    ax.set_xlabel("Salience network size (% cortical surface)")
    ax.grid(alpha=0.25); ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="best")
axes[0].set_ylabel("SDQ emotional symptoms (0-10)")
fig.suptitle("Emotional symptoms vs Salience size — non-linear exploration",
             fontsize=13)
plt.tight_layout()
out_png = f"{NS_DIR}/emotional_vs_salience_nonlinear.png"
plt.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out_png}")

print("\nCAVEAT: n is small and the SDQ-emotional scale is a floored count "
      "(TD mostly 0).\nIn-sample R^2 always rises with curvature; trust LOO-RMSE. "
      "Treat any\ncurvature as hypothesis-generating for the larger study, not "
      "confirmatory.")
