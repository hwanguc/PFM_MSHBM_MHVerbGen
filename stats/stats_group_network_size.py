"""
stats_group_network_size.py

Test whether MS-HBM cortical network sizes differ between the DLD (BL) and
TD/control (BT) groups, for every network.

Three complementary tests per network (Noise excluded):
  1. Welch's t-test            - parametric, unequal variance.
  2. Mann-Whitney U            - nonparametric robustness check.
  3. Beta regression           - the appropriate model for a response that is a
                                 proportion bounded in (0, 1). Network size is a
                                 fraction of cortical surface, so a logit-link
                                 beta GLM is preferable to a raw t-test (and to
                                 logistic regression, which is for binary 0/1
                                 outcomes, not continuous proportions). The group
                                 coefficient is on the log-odds scale; positive =
                                 larger in DLD.

Multiple comparisons across the 20 networks are controlled with Benjamini-
Hochberg FDR (q-values); Bonferroni is also reported.

NOTE (compositional data): per-subject network sizes sum to 100%, so the
networks are not independent. Per-network tests are a reasonable screen, but a
fully principled joint analysis would use Dirichlet regression or centred
log-ratio (CLR) transforms. Interpret single-network results with that caveat.

Input:  results/network_size/group_network_size_long.csv
Output: results/network_size/group_stats_DLD_vs_TD.csv (+ console summary)

## Author: Han Wang
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.othermod.betareg import BetaModel
from statsmodels.stats.multitest import multipletests

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
NS_DIR = f"{PROJECT_DIR}/results/network_size"
LONG = f"{NS_DIR}/group_network_size_long.csv"
OUT = f"{NS_DIR}/group_stats_DLD_vs_TD.csv"

GROUP_POS = "DLD"   # positive effect direction
GROUP_NEG = "TD"
EXCLUDE = {"Noise"}
ALPHA = 0.05


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan


def beta_group_effect(pct_pos, pct_neg):
    """Beta regression of proportion ~ group. Returns (coef, p) for the group
    term (log-odds scale, positive = larger in GROUP_POS)."""
    y = np.concatenate([pct_pos, pct_neg]) / 100.0          # -> (0,1)
    g = np.concatenate([np.ones(len(pct_pos)), np.zeros(len(pct_neg))])
    X = np.column_stack([np.ones_like(g), g])               # const + group
    try:
        res = BetaModel(y, X).fit(disp=0)
        return float(res.params[1]), float(res.pvalues[1])
    except Exception as e:                                   # convergence etc.
        return np.nan, np.nan


long = pd.read_csv(LONG)
long = long[~long["network_label"].isin(EXCLUDE)]

rows = []
for (nid, label), g in long.groupby(["network_id", "network_label"], sort=True):
    pos = g.loc[g["group"] == GROUP_POS, "network_size_pct"].to_numpy(float)
    neg = g.loc[g["group"] == GROUP_NEG, "network_size_pct"].to_numpy(float)
    if len(pos) < 2 or len(neg) < 2:
        continue
    t_stat, t_p = stats.ttest_ind(pos, neg, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    b_coef, b_p = beta_group_effect(pos, neg)
    rows.append(dict(
        network_id=nid, network_label=label,
        mean_DLD=np.mean(pos), mean_TD=np.mean(neg),
        diff_DLD_minus_TD=np.mean(pos) - np.mean(neg),
        sd_DLD=np.std(pos, ddof=1), sd_TD=np.std(neg, ddof=1),
        cohens_d=cohens_d(pos, neg),
        t_stat=t_stat, t_p=t_p,
        mwu_p=u_p,
        beta_coef=b_coef, beta_p=b_p,
    ))

df = pd.DataFrame(rows)

# Multiple-comparison correction across networks, per test
for col, pre in [("t_p", "t"), ("mwu_p", "mwu"), ("beta_p", "beta")]:
    p = df[col].to_numpy(float)
    ok = ~np.isnan(p)
    fdr = np.full_like(p, np.nan)
    bonf = np.full_like(p, np.nan)
    if ok.any():
        fdr[ok] = multipletests(p[ok], method="fdr_bh")[1]
        bonf[ok] = multipletests(p[ok], method="bonferroni")[1]
    df[f"{pre}_p_fdr"] = fdr
    df[f"{pre}_p_bonf"] = bonf

df = df.sort_values("beta_p").reset_index(drop=True)
df.to_csv(OUT, index=False)
print(f"Saved: {OUT}\n")

# Console summary
pd.set_option("display.width", 200, "display.max_columns", 30)
show = ["network_label", "mean_DLD", "mean_TD", "diff_DLD_minus_TD", "cohens_d",
        "t_p", "t_p_fdr", "mwu_p", "beta_coef", "beta_p", "beta_p_fdr"]
print("All networks (sorted by beta-regression p):")
print(df[show].round(4).to_string(index=False))

print(f"\nSignificant at FDR q < {ALPHA}:")
any_sig = False
for pre, name in [("t", "Welch t-test"), ("mwu", "Mann-Whitney U"), ("beta", "Beta regression")]:
    sig = df[df[f"{pre}_p_fdr"] < ALPHA]
    labs = sig["network_label"].tolist()
    print(f"  {name:16s}: {labs if labs else 'none'}")
    any_sig = any_sig or bool(labs)
if not any_sig:
    print("  -> No network survives FDR correction in any test "
          "(uncorrected p-values are in the CSV).")
