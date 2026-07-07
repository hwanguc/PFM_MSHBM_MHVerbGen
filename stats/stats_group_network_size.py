"""
stats_group_network_size.py

Between-group comparison of MS-HBM cortical network sizes across the three
verb-gen groups (DLD, HSL, TD), one test per network (Noise excluded), on the
chosen MS-HBM variant.

Per network:
  OMNIBUS (does group matter at all?)
    1. Welch's ANOVA          - unequal-variance, the 3-group generalisation of
                                the Welch t used before and the primary omnibus
                                (parallels the connectivity analysis).
    2. Beta-regression LR     - network size is a proportion of cortical surface,
                                so a logit-link beta GLM is the principled model;
                                likelihood-ratio test of the 3-level group factor
                                (2 df), TD reference. Sizes squeezed into (0,1).
    3. Kruskal-Wallis         - rank-based robustness backup.
  Benjamini-Hochberg FDR across networks on the Welch omnibus p (beta-LR FDR
  also reported).
  PROTECTED POST-HOC (only where Welch omnibus p<.05):
    Games-Howell (DLD-TD, HSL-TD, DLD-HSL; Hedges g) + Dunn (Holm) backup.

NOTE (compositional data): per-subject sizes sum to 100%, so networks are not
independent; per-network tests are a screen. A fully principled joint analysis
would use Dirichlet regression / CLR transforms.

Usage:
    python3 stats/stats_group_network_size.py                 # full variant
    python3 stats/stats_group_network_size.py --variant icafix

Input:  results/network_size_<variant>/group_network_size_long.csv
Output: results/network_size_<variant>/group_stats_3group.csv
        results/network_size_<variant>/group_posthoc_3group.csv

## Author: Han Wang
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.othermod.betareg import BetaModel
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import scikit_posthocs as sp

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
GROUPS = ["DLD", "HSL", "TD"]
CONTRASTS = [("DLD", "TD"), ("HSL", "TD"), ("DLD", "HSL")]
EXCLUDE = {"Noise"}
ALPHA = 0.05

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["full", "icafix"], default="full")
args = ap.parse_args()
NS_DIR = f"{PROJECT_DIR}/results/network_size_{args.variant}"
LONG = f"{NS_DIR}/group_network_size_long.csv"
OUT = f"{NS_DIR}/group_stats_3group.csv"
OUT_POST = f"{NS_DIR}/group_posthoc_3group.csv"


def beta_lr_group(sub):
    """Beta-regression LR test of the 3-level group factor (TD ref) on a network.
    Returns (lr_p, beta_DLD, p_DLD, beta_HSL, p_HSL) on the logit scale."""
    d = sub.copy()
    y = d["network_size_pct"].to_numpy(float) / 100.0
    n = len(y)
    y = (y * (n - 1) + 0.5) / n                      # squeeze into (0,1)
    dld = (d["group"] == "DLD").to_numpy(float)
    hsl = (d["group"] == "HSL").to_numpy(float)
    Xf = np.column_stack([np.ones(n), dld, hsl])     # TD = reference
    X0 = np.ones((n, 1))
    try:
        ff = BetaModel(y, Xf).fit(disp=0)
        f0 = BetaModel(y, X0).fit(disp=0)
        lr_p = stats.chi2.sf(2 * (ff.llf - f0.llf), 2)
        return (lr_p, float(ff.params[1]), float(ff.pvalues[1]),
                float(ff.params[2]), float(ff.pvalues[2]))
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)


long = pd.read_csv(LONG)
long = long[~long["network_label"].isin(EXCLUDE)]
long = long[long["group"].isin(GROUPS)]

rows, post = [], []
for (nid, label), g in long.groupby(["network_id", "network_label"], sort=True):
    arrs = {grp: g.loc[g["group"] == grp, "network_size_pct"].to_numpy(float)
            for grp in GROUPS}
    if any(len(a) < 2 for a in arrs.values()):
        continue

    wa = pg.welch_anova(data=g, dv="network_size_pct", between="group").iloc[0]
    F, p_w, np2 = float(wa["F"]), float(wa["p_unc"]), float(wa["np2"])
    H, p_kw = stats.kruskal(*arrs.values())
    lr_p, b_dld, p_dld, b_hsl, p_hsl = beta_lr_group(g)

    rec = dict(network_id=nid, network_label=label,
               mean_DLD=arrs["DLD"].mean(), mean_HSL=arrs["HSL"].mean(),
               mean_TD=arrs["TD"].mean(),
               sd_DLD=arrs["DLD"].std(ddof=1), sd_HSL=arrs["HSL"].std(ddof=1),
               sd_TD=arrs["TD"].std(ddof=1),
               F=F, welch_p=p_w, eta2=np2, kw_p=p_kw,
               beta_lr_p=lr_p, beta_DLD=b_dld, beta_p_DLD=p_dld,
               beta_HSL=b_hsl, beta_p_HSL=p_hsl)
    rows.append(rec)

    gh = pg.pairwise_gameshowell(data=g, dv="network_size_pct", between="group")
    gh_lu = {(a, b): r for (a, b), r in gh.set_index(["A", "B"]).iterrows()}
    dunn = sp.posthoc_dunn(g, val_col="network_size_pct", group_col="group",
                           p_adjust="holm")
    for A, B in CONTRASTS:
        row = gh_lu[(A, B)] if (A, B) in gh_lu else gh_lu[(B, A)]
        sign = 1.0 if (A, B) in gh_lu else -1.0
        post.append(dict(network_id=nid, network_label=label, contrast=f"{A}-{B}",
                         diff=sign * float(row["diff"]),
                         hedges=sign * float(row["hedges"]),
                         gh_p=float(row["pval"]), dunn_p=float(dunn.loc[A, B])))

df = pd.DataFrame(rows)
post = pd.DataFrame(post)

# FDR across networks on Welch omnibus p and on beta-LR p
for col, pre in [("welch_p", "welch"), ("beta_lr_p", "beta_lr"), ("kw_p", "kw")]:
    p = df[col].to_numpy(float); ok = ~np.isnan(p)
    fdr = np.full_like(p, np.nan)
    if ok.any():
        fdr[ok] = multipletests(p[ok], method="fdr_bh")[1]
    df[f"{pre}_p_fdr"] = fdr
df["omnibus_sig"] = df["welch_p"] < ALPHA

post = post.merge(df[["network_label", "welch_p", "welch_p_fdr", "omnibus_sig"]]
                  .rename(columns={"welch_p": "omnibus_p",
                                   "welch_p_fdr": "omnibus_p_fdr"}),
                  on="network_label", how="left")
post["protected"] = post["omnibus_sig"]

df = df.sort_values("welch_p").reset_index(drop=True)
df.to_csv(OUT, index=False)
post.to_csv(OUT_POST, index=False)
print(f"[{args.variant}] Saved: {OUT}")
print(f"[{args.variant}] Saved: {OUT_POST}\n")

pd.set_option("display.width", 220, "display.max_columns", 40)
show = ["network_label", "mean_DLD", "mean_HSL", "mean_TD", "F", "welch_p",
        "welch_p_fdr", "eta2", "beta_lr_p", "beta_lr_p_fdr", "kw_p"]
print(f"All networks (variant={args.variant}, sorted by Welch omnibus p):")
print(df[show].round(4).to_string(index=False))

n_sig = int((df["welch_p"] < ALPHA).sum())
n_fdr = int((df["welch_p_fdr"] < ALPHA).sum())
print(f"\nNetworks Welch omnibus p<.05: {n_sig} | survive FDR: {n_fdr}")
if n_sig:
    print("\nProtected Games-Howell (omnibus-significant networks):")
    prot = post[post.protected]
    print(prot[["network_label", "contrast", "diff", "hedges", "gh_p", "dunn_p"]]
          .round(4).to_string(index=False))
else:
    print("  -> No network reaches a significant omnibus; post-hoc not licensed.")
