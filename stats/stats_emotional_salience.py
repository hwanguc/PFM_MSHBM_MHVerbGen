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
    dat_verbgen_analysis_144.csv  (code, group, emotional)
    results/network_size_<variant>/group_network_size_long.csv  (Salience size)
Outputs (in results/network_size_<variant>/):
    emotional_salience_corr.csv, emotional_by_group.png, emotional_vs_salience.png

Usage:
    python3 stats/stats_emotional_salience.py                 # full variant
    python3 stats/stats_emotional_salience.py --variant icafix

## Author: Han Wang
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
LISTCSV = ("/home/hanwang/Documents/Data/verb_gen_krishnan/"
           "behavioural_scq_sdq/dat_verbgen_analysis_144.csv")
GROUP_COLORS = {"DLD": "#d63031", "HSL": "#2ca02c", "TD": "#0984e3"}
GROUPS = ["DLD", "HSL", "TD"]

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["full", "icafix"], default="full")
args = ap.parse_args()
NS_DIR = f"{PROJECT_DIR}/results/network_size_{args.variant}"

# ------------------------------------------------------------
beh = pd.read_csv(LISTCSV)[["code", "group", "emotional"]].dropna(subset=["emotional"])
beh["code"] = beh["code"].astype(str)

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
wa = pg.welch_anova(data=beh, dv="emotional", between="group").iloc[0]
print(f"Welch's ANOVA:          F({wa['ddof1']:.0f},{wa['ddof2']:.1f}) = {wa['F']:.3f},  "
      f"p = {wa['p_unc']:.4g}")
H, kp = stats.kruskal(*arrs)
print(f"Kruskal-Wallis:         H = {H:.3f},  p = {kp:.4g}")

print("\nGames-Howell post-hoc (pairwise, unequal variance):")
gh = pg.pairwise_gameshowell(data=beh, dv="emotional", between="group")
print(gh[["A", "B", "diff", "se", "T", "pval", "hedges"]].round(4).to_string(index=False))

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
for g in GROUPS:
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
order = ["DLD", "HSL", "TD"]
data_box = [groups[g] for g in order if g in groups]
labs = [g for g in order if g in groups]
parts = ax.boxplot(data_box, showmeans=True, patch_artist=True)
ax.set_xticks(range(1, len(labs) + 1)); ax.set_xticklabels(labs)
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
for g in GROUPS:
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
