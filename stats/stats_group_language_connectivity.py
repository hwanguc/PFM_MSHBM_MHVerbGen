"""
stats_group_language_connectivity.py

## Author: Han Wang
### 2026-08-14: Initial version (language frontal-putamen arm).

Between-group comparison of the SINGLE language frontal-putamen FC edge

    Language-14_L-Ctx (L_44, pars opercularis) <-> Language-14_L-Putamen
    (medial/anterior left putamen)

across the three verb-gen groups (DLD, HSL, TD), mirroring the striatal arm's
stats_group_connectivity.py but for one edge (so no 9-tile FDR).

Design (n up to 144: DLD=53, HSL=27, TD=64)
-------------------------------------------
1. Gather every analysed subject's per-subject edge CSV
   (<sub>_language_putamen_FC.csv), merge group, Fisher-z (already stored as z).
2. OMNIBUS on Fisher-z:
     - Welch's ANOVA (unequal variance / unequal n) -> F, p, partial eta^2, omega^2.
     - Kruskal-Wallis as a rank-based robustness backup.
3. Protected POST-HOC (interpret only if omnibus p<.05 -- Fisher-protected):
     - Games-Howell (unequal-variance pairwise) with Hedges g: DLD-TD, HSL-TD, DLD-HSL.
     - Dunn (Holm) rank-based backup.
Test on Fisher-z, display r.

Input:  results/language_connectivity_outputs/sub-*/sub-*_language_putamen_FC.csv
        dat_verbgen_analysis_144.csv  (code -> group)
Output: results/language_connectivity_outputs/group_language_putamen_long.csv
        results/language_connectivity_outputs/group_language_putamen_omnibus.csv
        results/language_connectivity_outputs/group_language_putamen_posthoc.csv
        results/language_connectivity_outputs/group_language_putamen.png
"""

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg
import scikit_posthocs as sp

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
LANG_DIR = f"{PROJECT_DIR}/results/language_connectivity_outputs"
LISTCSV = ("/home/hanwang/Documents/Data/verb_gen_krishnan/"
           "behavioural_scq_sdq/dat_verbgen_analysis_144.csv")

GROUPS = ["DLD", "HSL", "TD"]                    # display order (clinical gradient)
GROUP_COLORS = {"DLD": "#d63031", "HSL": "#2ca02c", "TD": "#0984e3"}
CONTRASTS = [("DLD", "TD"), ("HSL", "TD"), ("DLD", "HSL")]
EDGE = "L-Putamen(Language) <-> L_44 (pars opercularis)"


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def omega_sq(groups):
    k = len(groups); n = sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_t = ss_b + ss_w
    ms_w = ss_w / (n - k)
    denom = ss_t + ms_w
    return (ss_b - (k - 1) * ms_w) / denom if denom > 0 else np.nan


# ============================================================
# 1. Gather per-subject edges -> long
# ============================================================
beh = pd.read_csv(LISTCSV)[["code", "group"]].copy()
beh["code"] = beh["code"].astype(str)
code2group = beh.set_index("code")["group"].to_dict()

rows, missing = [], []
for code, group in code2group.items():
    sub = f"sub-{code}"
    hits = glob.glob(f"{LANG_DIR}/{sub}/{sub}_language_putamen_FC.csv")
    if not hits:
        missing.append(sub); continue
    m = pd.read_csv(hits[0]).iloc[0]
    r = float(m["r"]); z = float(m["z"])
    rows.append(dict(subject=sub, code=code, group=group, r=r, z=z))

long = pd.DataFrame(rows)
if missing:
    print(f"WARNING: {len(missing)} subjects had no language edge CSV: {missing}")
counts = long["group"].value_counts().to_dict()
print(f"Loaded {len(long)} subjects: " + ", ".join(f"{g}={counts.get(g,0)}" for g in GROUPS))
long.to_csv(f"{LANG_DIR}/group_language_putamen_long.csv", index=False)
print(f"Saved: {LANG_DIR}/group_language_putamen_long.csv")

# ============================================================
# 2. Omnibus (Welch ANOVA + Kruskal) on Fisher-z
# ============================================================
arrs = [long[long.group == g]["z"].to_numpy() for g in GROUPS]
wa = pg.welch_anova(data=long, dv="z", between="group").iloc[0]
F, p_w, np2 = float(wa["F"]), float(wa["p_unc"]), float(wa["np2"])
ddof1, ddof2 = float(wa["ddof1"]), float(wa["ddof2"])
H, p_kw = stats.kruskal(*arrs)
w2 = omega_sq(arrs)

rec = dict(edge=EDGE, F=F, ddof1=ddof1, ddof2=ddof2, p=p_w,
           eta2=np2, omega2=w2, kw_H=H, kw_p=p_kw)
for g in GROUPS:
    gr = long[long.group == g]["r"]
    rec[f"mean_r_{g}"] = gr.mean(); rec[f"sd_r_{g}"] = gr.std(ddof=1); rec[f"n_{g}"] = len(gr)
omni = pd.DataFrame([rec])
omni.to_csv(f"{LANG_DIR}/group_language_putamen_omnibus.csv", index=False)

print(f"\nOmnibus Welch ANOVA on Fisher-z:")
print(f"  F({ddof1:.0f},{ddof2:.1f}) = {F:.3f},  p = {p_w:.4f} {stars(p_w)}"
      f"   eta^2={np2:.3f}  omega^2={w2:.3f}")
print(f"  Kruskal-Wallis H = {H:.3f}, p = {p_kw:.4f} {stars(p_kw)}")
print("  Group mean r (+/- SD):  " +
      "  ".join(f"{g}={rec[f'mean_r_{g}']:+.3f}±{rec[f'sd_r_{g}']:.3f}(n{rec[f'n_{g}']})"
               for g in GROUPS))

# ============================================================
# 3. Protected post-hoc (Games-Howell + Dunn)
# ============================================================
gh = pg.pairwise_gameshowell(data=long, dv="z", between="group")
gh_lu = {(a, b): row for (a, b), row in gh.set_index(["A", "B"]).iterrows()}
dunn = sp.posthoc_dunn(long, val_col="z", group_col="group", p_adjust="holm")

post = []
for A, B in CONTRASTS:
    row = gh_lu.get((A, B)) if (A, B) in gh_lu else gh_lu.get((B, A))
    sign = 1.0 if (A, B) in gh_lu else -1.0
    post.append(dict(contrast=f"{A}-{B}", diff_z=sign * float(row["diff"]),
                     hedges=sign * float(row["hedges"]), gh_T=sign * float(row["T"]),
                     gh_p=float(row["pval"]), dunn_p=float(dunn.loc[A, B])))
post = pd.DataFrame(post)
post["omnibus_p"] = p_w
post["protected"] = p_w < .05
post.to_csv(f"{LANG_DIR}/group_language_putamen_posthoc.csv", index=False)
print(f"\nSaved: {LANG_DIR}/group_language_putamen_omnibus.csv")
print(f"Saved: {LANG_DIR}/group_language_putamen_posthoc.csv")

if p_w < .05:
    print("\nProtected Games-Howell (omnibus significant):")
    print(post[["contrast", "diff_z", "hedges", "gh_p", "dunn_p"]].round(3).to_string(index=False))
else:
    print("\nOmnibus n.s. -> post-hoc not licensed (Games-Howell/Dunn computed but not interpreted):")
    print(post[["contrast", "hedges", "gh_p", "dunn_p"]].round(3).to_string(index=False))

# ============================================================
# 4. Figure: per-group distribution of the edge (r), box + jittered points + mean
# ============================================================
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(6.4, 5.2))
data_by_g = [long[long.group == g]["r"].to_numpy() for g in GROUPS]
bp = ax.boxplot(data_by_g, positions=range(len(GROUPS)), widths=0.55,
                showfliers=False, patch_artist=True, zorder=1)
for patch, g in zip(bp["boxes"], GROUPS):
    patch.set_facecolor(GROUP_COLORS[g]); patch.set_alpha(0.25)
for med in bp["medians"]:
    med.set_color("black")
for k, g in enumerate(GROUPS):
    y = data_by_g[k]
    x = k + rng.uniform(-0.16, 0.16, len(y))
    ax.scatter(x, y, s=26, color=GROUP_COLORS[g], edgecolor="k", linewidth=0.3,
               alpha=0.85, zorder=3)
    ax.scatter([k], [y.mean()], marker="D", s=70, color="white",
               edgecolor=GROUP_COLORS[g], linewidth=2, zorder=4)
ax.axhline(0, color="grey", lw=0.8, ls="--")
ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels([f"{g}\n(n={counts.get(g,0)})" for g in GROUPS], fontweight="bold")
for tick, g in zip(ax.get_xticklabels(), GROUPS):
    tick.set_color(GROUP_COLORS[g])
ax.set_ylabel("Language frontal-putamen FC (Pearson's r)")
ax.set_title(f"Language frontal-putamen connectivity by group\n"
             f"{EDGE}\nWelch F({ddof1:.0f},{ddof2:.1f})={F:.2f}, p={p_w:.3f} {stars(p_w)}"
             f"  (ω²={w2:.3f})", fontsize=11)
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(f"{LANG_DIR}/group_language_putamen.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved: {LANG_DIR}/group_language_putamen.png")
