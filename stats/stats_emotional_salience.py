"""
stats_emotional_salience.py

Two questions about the SDQ emotional-symptoms score ("emotional"):

(1) Group comparison of emotional symptoms (one-way ANOVA, main effect of group)
    across DLD / TD / HSL. Because the score is a skewed bounded count with very
    unequal group variances, we also report Levene's test, Welch's ANOVA and the
    Kruskal-Wallis test, plus Tukey HSD post-hoc and the focused DLD-vs-TD test.

(2) Correlation between emotional symptoms and Salience-network size, computed
    separately within DLD and within TD (the two imaged groups). Pearson and
    Spearman, with a scatter + per-group regression lines.

Inputs:
    dat_verbgen_scqsdq_subsample.xlsx  (code, group, emotional)
    results/network_size/group_network_size_long.csv  (Salience size)
Outputs:
    results/network_size/emotional_salience_corr.csv
    results/network_size/emotional_by_group.png
    results/network_size/emotional_vs_salience.png

## Author: Han Wang
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
NS_DIR = f"{PROJECT_DIR}/results/network_size"
XLSX = ("/home/hanwang/Documents/Data/ucl/gos_ich/verb_gen_krishnan/"
        "behavioural_scq_sdq/dat_verbgen_scqsdq_subsample.xlsx")
GROUP_COLORS = {"DLD": "#d63031", "TD": "#0984e3", "HSL": "#00b894"}


def welch_anova(groups):
    """Welch's one-way ANOVA (does not assume equal variances)."""
    k = len(groups)
    n = np.array([len(g) for g in groups])
    m = np.array([np.mean(g) for g in groups])
    v = np.array([np.var(g, ddof=1) for g in groups])
    w = n / v
    mbar = np.sum(w * m) / np.sum(w)
    num = np.sum(w * (m - mbar) ** 2) / (k - 1)
    denom = 1 + (2 * (k - 2) / (k ** 2 - 1)) * np.sum((1 - w / np.sum(w)) ** 2 / (n - 1))
    F = num / denom
    df2 = (k ** 2 - 1) / (3 * np.sum((1 - w / np.sum(w)) ** 2 / (n - 1)))
    p = stats.f.sf(F, k - 1, df2)
    return F, k - 1, df2, p


# ------------------------------------------------------------
beh = pd.read_excel(XLSX)[["code", "group", "emotional"]].dropna(subset=["emotional"])

print("=" * 64)
print("(1) EMOTIONAL SYMPTOMS BY GROUP")
print("=" * 64)
print(beh.groupby("group")["emotional"].agg(["count", "mean", "std", "median"]).round(3).to_string())

groups = {g: d["emotional"].to_numpy(float) for g, d in beh.groupby("group")}
arrs = list(groups.values())

F, p = stats.f_oneway(*arrs)
print(f"\nClassic one-way ANOVA:  F({len(arrs)-1},{len(beh)-len(arrs)}) = {F:.3f},  p = {p:.4g}")
lev_stat, lev_p = stats.levene(*arrs)
print(f"Levene equal-variance:  W = {lev_stat:.3f},  p = {lev_p:.4g}"
      + ("  (variances UNEQUAL -> prefer Welch/KW)" if lev_p < 0.05 else ""))
wF, wdf1, wdf2, wp = welch_anova(arrs)
print(f"Welch's ANOVA:          F({wdf1},{wdf2:.1f}) = {wF:.3f},  p = {wp:.4g}")
H, kp = stats.kruskal(*arrs)
print(f"Kruskal-Wallis:         H = {H:.3f},  p = {kp:.4g}")

print("\nTukey HSD post-hoc (pairwise):")
tuk = pairwise_tukeyhsd(beh["emotional"], beh["group"])
print(tuk.summary())

dld, td = groups["DLD"], groups["TD"]
t_stat, t_p = stats.ttest_ind(dld, td, equal_var=False)
u_stat, u_p = stats.mannwhitneyu(dld, td, alternative="two-sided")
print(f"\nFocused DLD vs TD:  Welch t = {t_stat:.3f}, p = {t_p:.4g} | "
      f"Mann-Whitney p = {u_p:.4g}")

# ------------------------------------------------------------
print("\n" + "=" * 64)
print("(2) EMOTIONAL vs SALIENCE-NETWORK SIZE  (within group)")
print("=" * 64)
long = pd.read_csv(f"{NS_DIR}/group_network_size_long.csv")
sal = long[long["network_label"] == "Salience"][["code", "group", "network_size_pct"]]
sal = sal.rename(columns={"network_size_pct": "salience_pct"})
dat = sal.merge(beh[["code", "emotional"]], on="code", how="left")

rows = []
for g in ["DLD", "TD"]:
    d = dat[dat["group"] == g]
    r, rp = stats.pearsonr(d["salience_pct"], d["emotional"])
    rho, rhop = stats.spearmanr(d["salience_pct"], d["emotional"])
    rows.append(dict(group=g, n=len(d), pearson_r=r, pearson_p=rp,
                     spearman_rho=rho, spearman_p=rhop,
                     emo_var=np.var(d["emotional"], ddof=1)))
    print(f"\n{g} (n={len(d)}):")
    print(f"  Pearson  r = {r:+.3f}, p = {rp:.4g}")
    print(f"  Spearman rho = {rho:+.3f}, p = {rhop:.4g}")
# combined (pooled) for reference
r, rp = stats.pearsonr(dat["salience_pct"], dat["emotional"])
rows.append(dict(group="ALL", n=len(dat), pearson_r=r, pearson_p=rp,
                 spearman_rho=np.nan, spearman_p=np.nan, emo_var=np.var(dat["emotional"], ddof=1)))
pd.DataFrame(rows).to_csv(f"{NS_DIR}/emotional_salience_corr.csv", index=False)
print(f"\nSaved: {NS_DIR}/emotional_salience_corr.csv")

# ------------------------------------------------------------
# Figures
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5))
order = ["DLD", "TD", "HSL"]
data_box = [groups[g] for g in order if g in groups]
labs = [g for g in order if g in groups]
parts = ax.boxplot(data_box, labels=labs, showmeans=True, patch_artist=True)
for patch, g in zip(parts["boxes"], labs):
    patch.set_facecolor(GROUP_COLORS[g]); patch.set_alpha(0.5)
# jittered points
for i, g in enumerate(labs, start=1):
    y = groups[g]; x = np.random.normal(i, 0.05, len(y))
    ax.scatter(x, y, color=GROUP_COLORS[g], edgecolor="k", linewidth=0.3, s=22, zorder=3)
ax.set_ylabel("SDQ emotional symptoms (0-10)")
ax.set_title("Emotional symptoms by group")
plt.tight_layout(); plt.savefig(f"{NS_DIR}/emotional_by_group.png", dpi=200, bbox_inches="tight"); plt.close()
print(f"Saved: {NS_DIR}/emotional_by_group.png")

fig, ax = plt.subplots(figsize=(6, 5))
for g in ["DLD", "TD"]:
    d = dat[dat["group"] == g]
    ax.scatter(d["salience_pct"], d["emotional"], color=GROUP_COLORS[g],
               edgecolor="k", linewidth=0.3, s=40, label=f"{g} (n={len(d)})")
    if len(d) > 2:
        b = np.polyfit(d["salience_pct"], d["emotional"], 1)
        xs = np.linspace(d["salience_pct"].min(), d["salience_pct"].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=GROUP_COLORS[g], lw=1.5)
ax.set_xlabel("Salience network size (% cortical surface)")
ax.set_ylabel("SDQ emotional symptoms (0-10)")
ax.set_title("Emotional symptoms vs Salience network size")
ax.legend(); plt.tight_layout()
plt.savefig(f"{NS_DIR}/emotional_vs_salience.png", dpi=200, bbox_inches="tight"); plt.close()
print(f"Saved: {NS_DIR}/emotional_vs_salience.png")
