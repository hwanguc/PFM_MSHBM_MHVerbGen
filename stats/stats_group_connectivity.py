"""
stats_group_connectivity.py

Between-group comparison of the 9 frontostriatal FC tiles (3 subcortical x 3
cortical) across the three verb-gen groups (DLD, HSL, TD) using the per-subject
matrices from connectivity/2_run_subject_connectivity_analysis_cab-np.py
(<sub>_03_frontostriatal_FC_3_rest_antstri.csv).

Design (n = 144: DLD=53, HSL=27, TD=64)
---------------------------------------
1. Gather every analysed subject's 3x3 frontostriatal r-matrix, melt to long,
   Fisher-z each tile (z = arctanh r). Long table written as a byproduct.
2. Per tile (9 edges), OMNIBUS on Fisher-z:
     - Welch's ANOVA (does NOT assume equal variance -> robust to DLD's larger
       spread + the unequal n; the 3-group generalisation of the Welch t used
       in the 2-group version). Reports F, p, partial eta^2, and (classically
       computed) omega^2.
     - Kruskal-Wallis as a rank-based robustness backup.
     - Benjamini-Hochberg FDR across the 9 tiles on the Welch omnibus p.
3. Protected POST-HOC (only interpreted where the tile's omnibus p<.05 --
   Fisher-protected logic, valid for 3 groups):
     - Games-Howell (unequal-variance pairwise, Welch-consistent): DLD-TD,
       HSL-TD, DLD-HSL with Hedges g.
     - Dunn (Holm) as the rank-based backup.
   Games-Howell/Dunn are computed for all tiles but flagged `protected` so the
   interpretation respects the gating.
4. Figures:
     - group_frontostriatal_mean.png    : 3 descriptive panels (DLD, HSL, TD),
       tile = group mean r +/- SD (descriptive, in r).
     - group_frontostriatal_omnibusF.png: omnibus Welch F per tile, stars = p,
       bold box = survives BH-FDR.
     - group_frontostriatal_pairwise_g.png : 3 panels (DLD-TD, HSL-TD, DLD-HSL),
       tile = Hedges g (Games-Howell), stars = GH p; tiles whose omnibus is not
       significant are greyed (post-hoc not licensed there).

Test on Fisher-z, display r: r is the interpretable unit, but its sampling
variance depends on the true value, so inference is on z and only means are r.

Input:  results/connectivity_outputs/sub-*/sub-*_03_frontostriatal_FC_3_rest_antstri.csv
        dat_verbgen_analysis_144.csv  (code -> group)
Output: results/connectivity_outputs/group_frontostriatal_long.csv
        results/connectivity_outputs/group_frontostriatal_omnibus.csv
        results/connectivity_outputs/group_frontostriatal_posthoc.csv
        results/connectivity_outputs/group_frontostriatal_mean.png
        results/connectivity_outputs/group_frontostriatal_omnibusF.png
        results/connectivity_outputs/group_frontostriatal_pairwise_g.png

## Author: Han Wang
"""

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
import pingouin as pg
import scikit_posthocs as sp

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
CONN_DIR = f"{PROJECT_DIR}/results/connectivity_outputs"
LISTCSV = ("/home/hanwang/Documents/Data/verb_gen_krishnan/"
           "behavioural_scq_sdq/dat_verbgen_analysis_144.csv")
SUFFIX = "_3_rest_antstri"

SUBCORTICAL = ["NAcc", "Caudate", "Putamen"]   # rows
CORTICAL = ["ACC", "AI", "LPFC"]               # cols
GROUPS = ["DLD", "HSL", "TD"]                  # display order (clinical gradient)
GROUP_COLORS = {"DLD": "#d63031", "HSL": "#2ca02c", "TD": "#0984e3"}
# pairwise contrasts of interest (A - B)
CONTRASTS = [("DLD", "TD"), ("HSL", "TD"), ("DLD", "HSL")]


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def omega_sq(groups):
    """Classic one-way omega^2 effect size from a list of arrays."""
    k = len(groups)
    n = sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_t = ss_b + ss_w
    df_b = k - 1
    ms_w = ss_w / (n - k)
    denom = ss_t + ms_w
    return (ss_b - df_b * ms_w) / denom if denom > 0 else np.nan


# ============================================================
# 1. Gather per-subject frontostriatal tiles -> long (z-transformed)
# ============================================================
beh = pd.read_csv(LISTCSV)[["code", "group"]].copy()
beh["code"] = beh["code"].astype(str)
code2group = beh.set_index("code")["group"].to_dict()

rows, missing = [], []
for code, group in code2group.items():
    sub = f"sub-{code}"
    hits = glob.glob(f"{CONN_DIR}/{sub}/{sub}_03_frontostriatal_FC{SUFFIX}.csv")
    if not hits:
        missing.append(sub)
        continue
    m = pd.read_csv(hits[0], index_col="subcortical")
    for sc in SUBCORTICAL:
        for ct in CORTICAL:
            r = float(m.loc[sc, ct])
            rows.append(dict(subject=sub, code=code, group=group,
                             subcortical=sc, cortical=ct, edge=f"{sc}-{ct}",
                             r=r, z=np.arctanh(np.clip(r, -0.999999, 0.999999))))

long = pd.DataFrame(rows)
if missing:
    print(f"WARNING: {len(missing)} subjects had no frontostriatal CSV: {missing}")
counts = long.drop_duplicates("subject")["group"].value_counts().to_dict()
print(f"Loaded {long['subject'].nunique()} subjects: "
      + ", ".join(f"{g}={counts.get(g,0)}" for g in GROUPS))
long.to_csv(f"{CONN_DIR}/group_frontostriatal_long.csv", index=False)
print(f"Saved: {CONN_DIR}/group_frontostriatal_long.csv")

# ============================================================
# 2. Per-tile omnibus (Welch ANOVA + Kruskal-Wallis) on Fisher-z
# 3. Per-tile protected post-hoc (Games-Howell + Dunn)
# ============================================================
omni, post = [], []
for sc in SUBCORTICAL:
    for ct in CORTICAL:
        edge = f"{sc}-{ct}"
        d = long[(long.subcortical == sc) & (long.cortical == ct)].copy()
        arrs = [d[d.group == g]["z"].to_numpy() for g in GROUPS]

        wa = pg.welch_anova(data=d, dv="z", between="group").iloc[0]
        F, p_w, np2 = float(wa["F"]), float(wa["p_unc"]), float(wa["np2"])
        H, p_kw = stats.kruskal(*arrs)
        w2 = omega_sq(arrs)
        rec = dict(edge=edge, subcortical=sc, cortical=ct,
                   F=F, p=p_w, eta2=np2, omega2=w2, kw_H=H, kw_p=p_kw)
        for g in GROUPS:
            gr = d[d.group == g]["r"]
            rec[f"mean_r_{g}"] = gr.mean()
            rec[f"sd_r_{g}"] = gr.std(ddof=1)
        omni.append(rec)

        # Games-Howell (all tiles; protection applied at interpretation)
        gh = pg.pairwise_gameshowell(data=d, dv="z", between="group")
        gh_lu = {(a, b): row for (a, b), row in
                 gh.set_index(["A", "B"]).iterrows()}
        dunn = sp.posthoc_dunn(d, val_col="z", group_col="group", p_adjust="holm")
        for A, B in CONTRASTS:
            row = gh_lu.get((A, B)) if (A, B) in gh_lu else gh_lu.get((B, A))
            sign = 1.0 if (A, B) in gh_lu else -1.0   # flip if pingouin ordered (B,A)
            post.append(dict(
                edge=edge, subcortical=sc, cortical=ct, contrast=f"{A}-{B}",
                diff_z=sign * float(row["diff"]),
                hedges=sign * float(row["hedges"]),
                gh_T=sign * float(row["T"]), gh_p=float(row["pval"]),
                dunn_p=float(dunn.loc[A, B])))

omni = pd.DataFrame(omni)
post = pd.DataFrame(post)

# BH-FDR across the 9 tiles on the Welch omnibus p
order = np.argsort(omni["p"].to_numpy())
ranks = np.empty(len(omni), int); ranks[order] = np.arange(1, len(omni) + 1)
q = np.minimum(1, omni["p"].to_numpy() * len(omni) / ranks)
q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
q_full = np.empty_like(q); q_full[order] = q_sorted
omni["p_fdr"] = q_full
omni["omnibus_sig"] = omni["p"] < .05          # gates the protected post-hoc

# tag post-hoc rows with whether their tile's omnibus licenses interpretation
post = post.merge(omni[["edge", "p", "p_fdr", "omnibus_sig"]]
                  .rename(columns={"p": "omnibus_p", "p_fdr": "omnibus_p_fdr"}),
                  on="edge", how="left")
post["protected"] = post["omnibus_sig"]

omni.to_csv(f"{CONN_DIR}/group_frontostriatal_omnibus.csv", index=False)
post.to_csv(f"{CONN_DIR}/group_frontostriatal_posthoc.csv", index=False)
print(f"Saved: {CONN_DIR}/group_frontostriatal_omnibus.csv")
print(f"Saved: {CONN_DIR}/group_frontostriatal_posthoc.csv\n")

print("Per-tile omnibus (Welch ANOVA on Fisher-z):")
print(omni[["edge", "mean_r_DLD", "mean_r_HSL", "mean_r_TD",
            "F", "p", "p_fdr", "eta2", "omega2", "kw_p"]]
      .round(3).to_string(index=False))
n_sig = int((omni["p"] < .05).sum()); n_fdr = int((omni["p_fdr"] < .05).sum())
print(f"\nTiles omnibus p<.05: {n_sig}/9 | survive FDR: {n_fdr}/9")
if n_sig:
    print("\nProtected Games-Howell for omnibus-significant tiles:")
    print(post[post.protected][["edge", "contrast", "diff_z", "hedges",
                                "gh_p", "dunn_p"]].round(3).to_string(index=False))
else:
    print("\nNo tile reaches a significant omnibus; post-hoc not licensed.")


def grid(mapping, sc_list=SUBCORTICAL, ct_list=CORTICAL):
    a = np.full((len(sc_list), len(ct_list)), np.nan)
    for i, sc in enumerate(sc_list):
        for j, ct in enumerate(ct_list):
            a[i, j] = mapping.get(f"{sc}-{ct}", np.nan)
    return a


osmap = omni.set_index("edge")

# ============================================================
# 4a. Descriptive: group mean r (+/- SD), one panel per group
# ============================================================
mean_r = {g: grid({e: osmap.loc[e, f"mean_r_{g}"] for e in osmap.index}) for g in GROUPS}
sd_r = {g: grid({e: osmap.loc[e, f"sd_r_{g}"] for e in osmap.index}) for g in GROUPS}
vmax = np.ceil(np.nanmax([np.abs(mean_r[g]) for g in GROUPS]) * 10) / 10

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("white")
for ax, g in zip(axes, GROUPS):
    im = ax.imshow(mean_r[g], cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(CORTICAL))); ax.set_yticks(range(len(SUBCORTICAL)))
    ax.set_xticklabels(CORTICAL, fontweight="bold")
    ax.set_yticklabels(SUBCORTICAL, fontweight="bold")
    ax.set_title(f"{g} (n={counts.get(g,0)})", color=GROUP_COLORS[g], fontweight="bold")
    for i in range(len(SUBCORTICAL)):
        for j in range(len(CORTICAL)):
            c = "white" if abs(mean_r[g][i, j]) > 0.3 else "black"
            ax.text(j, i, f"{mean_r[g][i, j]:+.2f}\n±{sd_r[g][i, j]:.2f}",
                    ha="center", va="center", fontsize=9, color=c)
    ax.set_xlabel("Cortical zone")
axes[0].set_ylabel("Subcortical")
cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
cbar.set_label("Group mean Pearson's r")
fig.suptitle("Frontostriatal FC — group means (tile: mean r ± SD across subjects)",
             fontsize=13)
plt.savefig(f"{CONN_DIR}/group_frontostriatal_mean.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved: {CONN_DIR}/group_frontostriatal_mean.png")

# ============================================================
# 4b. Omnibus F heatmap: stars = omnibus p, bold box = survives FDR
# ============================================================
Fg = grid({e: osmap.loc[e, "F"] for e in osmap.index})
Pg = grid({e: osmap.loc[e, "p"] for e in osmap.index})
Qg = grid({e: osmap.loc[e, "p_fdr"] for e in osmap.index})
fmax = max(np.nanmax(Fg), 1e-6)

fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(Fg, cmap="viridis", vmin=0, vmax=fmax, aspect="equal")
ax.set_xticks(range(len(CORTICAL))); ax.set_yticks(range(len(SUBCORTICAL)))
ax.set_xticklabels(CORTICAL, fontweight="bold")
ax.set_yticklabels(SUBCORTICAL, fontweight="bold")
for i in range(len(SUBCORTICAL)):
    for j in range(len(CORTICAL)):
        col = "white" if Fg[i, j] < 0.6 * fmax else "black"
        ax.text(j, i, f"{Fg[i, j]:.2f}{stars(Pg[i, j])}",
                ha="center", va="center", fontsize=12, color=col)
        if Qg[i, j] < 0.05:
            ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                   edgecolor="red", linewidth=3))
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Welch ANOVA F (on Fisher-z)")
ax.set_xlabel("Cortical zone"); ax.set_ylabel("Subcortical")
ax.set_title("Frontostriatal FC: 3-group omnibus (DLD/HSL/TD)\n"
             "* p<.05 ** p<.01 *** p<.001 (uncorr.);  red box = survives BH-FDR",
             fontsize=11)
plt.savefig(f"{CONN_DIR}/group_frontostriatal_omnibusF.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {CONN_DIR}/group_frontostriatal_omnibusF.png")

# ============================================================
# 4c. Pairwise Hedges g heatmaps (3 contrasts); grey where omnibus n.s.
# ============================================================
pmap = post.set_index(["contrast", "edge"])
gmax = max(np.nanmax(np.abs(post["hedges"])), 1e-6)
norm = TwoSlopeNorm(vmin=-gmax, vcenter=0, vmax=gmax)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, (A, B) in zip(axes, CONTRASTS):
    G = grid({e: pmap.loc[(f"{A}-{B}", e), "hedges"] for e in osmap.index})
    Pp = grid({e: pmap.loc[(f"{A}-{B}", e), "gh_p"] for e in osmap.index})
    Prot = grid({e: float(pmap.loc[(f"{A}-{B}", e), "protected"]) for e in osmap.index})
    im = ax.imshow(G, cmap="RdBu_r", norm=norm, aspect="equal")
    ax.set_xticks(range(len(CORTICAL))); ax.set_yticks(range(len(SUBCORTICAL)))
    ax.set_xticklabels(CORTICAL, fontweight="bold")
    ax.set_yticklabels(SUBCORTICAL, fontweight="bold")
    ax.set_title(f"{A} − {B}", fontsize=12, fontweight="bold")
    for i in range(len(SUBCORTICAL)):
        for j in range(len(CORTICAL)):
            if Prot[i, j] < 0.5:                      # omnibus n.s. -> not licensed
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, facecolor="white",
                                       alpha=0.6, edgecolor="none"))
                txt, col = f"{G[i, j]:+.2f}", "0.6"
            else:
                txt, col = f"{G[i, j]:+.2f}{stars(Pp[i, j])}", \
                    ("white" if abs(G[i, j]) > 0.6 * gmax else "black")
            ax.text(j, i, txt, ha="center", va="center", fontsize=11, color=col)
    ax.set_xlabel("Cortical zone")
axes[0].set_ylabel("Subcortical")
cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
cbar.set_label("Hedges g (Games–Howell, on Fisher-z)")
fig.suptitle("Frontostriatal FC pairwise contrasts (protected: greyed = omnibus n.s.)\n"
             "red = first group higher, blue = lower;  * p<.05 ** p<.01 *** p<.001 (Games–Howell)",
             fontsize=12)
plt.savefig(f"{CONN_DIR}/group_frontostriatal_pairwise_g.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {CONN_DIR}/group_frontostriatal_pairwise_g.png")
