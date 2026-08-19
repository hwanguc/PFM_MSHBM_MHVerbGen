"""
make_poster_figures.py

Poster-ready SVG versions of the three headline figures. This script does NOT
touch the analysis pipeline; it re-fits the same models and re-styles the plots
for print. Text is kept as text in the SVG (svg.fonttype='none') so the labels
stay editable in Illustrator/Inkscape. Fonts are unified across all three
figures via rcParams (DejaVu Sans, matching Fig 1); where a p-value is bold,
only its number is emphasised, drawn with real Text objects (offsetbox) rather
than mathtext so the glyphs use the same text engine/spacing as Fig 1.

  Fig 1 — SDQ-emotional by group (box + jittered points).
          No title; y-axis "SDQ emotional symptoms (0-10)"; TD flier
          (open-circle outlier) markers hidden; mean triangles drawn white.
  Fig 2 — FC -> SDQ-emotional, one figure holding BOTH networks: the 3x3
          salience fronto-striatal grid (left, light-blue block) and the single
          language frontal-putamen edge (right, light-red block, ~square).
          Per-panel subtitle "<edge> (interaction: [FDR Adj] p=.xxx)" (number
          bold when significant) + in-panel "β(DLD-TD)=.., p=.." annotation.
  Fig 3 — SDQ-emotional vs network size, salience | language side by side
          (NB predicted mean +/- 95% CI). No title; y-axis as above; x-axis
          "<Network> network size (% cortical surface)"; legend = coloured
          lines DLD / HSL / TD only. Each panel annotates the joint interaction
          p and the TD (reference) slope: "interaction: p=.xxx" / "β(TD)=..".

Network-size variant for Fig 3: icafix (per project decision, n=142).

Outputs (poster/): fig1_emotional_by_group.svg, fig2_fc_scatter.svg,
                   fig3_network_size.svg

Run with the project venv:  .venv/bin/python stats/make_poster_figures.py

## Author: Han Wang
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import patsy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnchoredOffsetbox
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # hush Arial-not-found
# Unified typography for every figure: Arial. Arial is not installed on this
# Linux box, so metrics are taken from Liberation Sans (metric-compatible with
# Arial) and the SVG font-family lists 'Arial' first -> it renders as Arial on
# any machine that has it. Bold emphasis (significant p-numbers) is produced
# with real text objects, NOT mathtext, so every glyph shares one text engine.
matplotlib.rcParams.update({
    "svg.fonttype": "none",            # keep SVG text editable
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
LISTCSV = ("/home/hanwang/Documents/Data/verb_gen_krishnan/"
           "behavioural_scq_sdq/dat_verbgen_analysis_144.csv")
CONN_DIR = f"{PROJECT_DIR}/results/connectivity_outputs"
LANG_DIR = f"{PROJECT_DIR}/results/language_connectivity_outputs"
VARIANT = "icafix"
NS_DIR = f"{PROJECT_DIR}/results/network_size_{VARIANT}"
OUT_DIR = f"{PROJECT_DIR}/poster"
os.makedirs(OUT_DIR, exist_ok=True)

GROUPS = ["DLD", "HSL", "TD"]
GROUP_COLORS = {"DLD": "#d63031", "HSL": "#2ca02c", "TD": "#0984e3"}
SUBCORTICAL = ["NAcc", "Caudate", "Putamen"]
CORTICAL = ["ACC", "AI", "LPFC"]
INTER = {"DLD": "C(group, Treatment('TD'))[T.DLD]:FCz_c",
         "HSL": "C(group, Treatment('TD'))[T.HSL]:FCz_c"}
YLAB = "SDQ emotional symptoms (0-10)"
RNG = np.random.default_rng(0)


# ------------------------------------------------------------
# text helpers. A "rich" string is a list of (text, bold) segments; it is
# rendered with real Text objects (offsetbox) so bold and regular glyphs share
# one text engine -> identical font to Fig 1 (no mathtext).
# ------------------------------------------------------------
def pfmt(p):
    return "<.001" if p < .001 else f"{p:.3f}".lstrip("0")


def stat_seg(prefix, p, suffix=""):
    """[(text, bold)] for '<prefix>p=.037<suffix>', bolding ONLY the number when
    p<.05. Everything except the number is merged into ONE run (prefix + 'p=')
    so the only segment boundary sits where there is no space -- SVG renderers
    trim trailing whitespace inside a <text> run, which would otherwise open a
    gap before 'p=' at a segment boundary."""
    sig = p < .05
    op, num = ("p<", ".001") if p < .001 else ("p=", pfmt(p))
    segs = [(prefix + op, False), (num, sig)]
    if suffix:
        segs.append((suffix, False))
    return segs


def _line(segs, fs):
    return HPacker(pad=0, sep=0, align="baseline", children=[
        TextArea(t, textprops=dict(fontsize=fs, color="black",
                                   fontweight="bold" if b else "normal"))
        for t, b in segs])


def _box(lines, fs, align):
    return VPacker(pad=0, sep=2, align=align,
                   children=[_line(s, fs) for s in lines])


def place_title(ax, lines, fs):
    """Centred, multi-line title just above the axes (real text, partial bold)."""
    ao = AnchoredOffsetbox(loc="lower center", child=_box(lines, fs, "center"),
                           pad=0, borderpad=0.25, frameon=False,
                           bbox_to_anchor=(0.5, 1.0), bbox_transform=ax.transAxes)
    ax.add_artist(ao)


def place_annot(ax, lines, fs):
    """Top-left in-panel annotation in a light rounded box (real text)."""
    ao = AnchoredOffsetbox(loc="upper left", child=_box(lines, fs, "left"),
                           pad=0.3, borderpad=0.4, frameon=True,
                           bbox_to_anchor=(0.0, 1.0), bbox_transform=ax.transAxes)
    ao.patch.set(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.6, alpha=0.85)
    ax.add_artist(ao)


def band(full, d, xcol, xc_col, group, xmean, n=140):
    """delta-method NB predicted mean + 95% CI for one group (log scale, exp back)."""
    di = full.model.data.design_info
    k = len(di.column_names)
    beta = np.asarray(full.params)[:k]
    cov = np.asarray(full.cov_params())[:k, :k]
    dg = d[d.group == group]
    xs = np.linspace(dg[xcol].min(), dg[xcol].max(), n)
    grid = pd.DataFrame({"group": group, xcol: xs, xc_col: xs - xmean})
    X = np.asarray(patsy.dmatrix(di, grid))
    eta = X @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", X, cov, X))
    return xs, np.exp(eta), np.exp(eta - 1.96 * se), np.exp(eta + 1.96 * se)


# ============================================================
# Behavioural table (shared)
# ============================================================
beh = pd.read_csv(LISTCSV)
beh["code"] = beh["code"].astype(str)
beh_emo = beh[["code", "group", "emotional"]].dropna(subset=["emotional"])


# ============================================================
# FIG 1 — SDQ-emotional by group
# ============================================================
groups = {g: d["emotional"].to_numpy(float) for g, d in beh_emo.groupby("group")}
order = [g for g in GROUPS if g in groups]

fig, ax = plt.subplots(figsize=(5.6, 5.0))
data_box = [groups[g] for g in order]
bp = ax.boxplot(
    data_box, showmeans=True, patch_artist=True, widths=0.6,
    medianprops=dict(color="#e17000", lw=1.8),
    whiskerprops=dict(color="black", lw=1.2),
    capprops=dict(color="black", lw=1.2),
    boxprops=dict(color="black", lw=1.2),
    meanprops=dict(marker="^", markerfacecolor="white", markeredgecolor="black",
                   markersize=9, markeredgewidth=1.2, zorder=6),
    flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="black",
                    markersize=6))
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels(order, fontsize=13)
for patch, g in zip(bp["boxes"], order):
    patch.set_facecolor(GROUP_COLORS[g]); patch.set_alpha(0.45)
if "TD" in order:                                    # hide TD outlier markers only
    bp["fliers"][order.index("TD")].set_visible(False)
for i, g in enumerate(order, start=1):               # jittered raw points
    y = groups[g]
    x = np.random.normal(i, 0.055, len(y))
    ax.scatter(x, y, color=GROUP_COLORS[g], edgecolor="k", linewidth=0.3,
               s=24, zorder=4, alpha=0.9)
ax.set_ylabel(YLAB)
ax.set_ylim(-0.6, 10.6)
fig.tight_layout()
f1 = f"{OUT_DIR}/fig1_emotional_by_group.svg"
fig.savefig(f1, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {f1}")


# ============================================================
# FIG 2 — FC -> SDQ-emotional : salience 3x3 + language single edge
# ============================================================
# --- salience: refit 9 tiles + BH-FDR on the joint-interaction LR p ---
long = pd.read_csv(f"{CONN_DIR}/group_frontostriatal_long.csv")
long["code"] = long["code"].astype(str)
long = long.merge(beh_emo[["code", "emotional"]], on="code", how="left").dropna(subset=["emotional"])
long["emotional"] = long["emotional"].round().astype(int).clip(0, 10)
long = long.rename(columns={"z": "FCz"})

sal_fit, sal_p, sal_beta, sal_waldp = {}, {}, {}, {}
for sc in SUBCORTICAL:
    for ct in CORTICAL:
        d = long[(long.subcortical == sc) & (long.cortical == ct)].copy()
        d["FCz_c"] = d["FCz"] - d["FCz"].mean()
        full = smf.negativebinomial(
            "emotional ~ C(group, Treatment('TD')) * FCz_c", data=d).fit(disp=0, maxiter=1000)
        red = smf.negativebinomial(
            "emotional ~ C(group, Treatment('TD')) + FCz_c", data=d).fit(disp=0, maxiter=1000)
        sal_fit[(sc, ct)] = (full, d)
        sal_p[(sc, ct)] = stats.chi2.sf(2 * (full.llf - red.llf), 2)
        sal_beta[(sc, ct)] = full.params[INTER["DLD"]]
        sal_waldp[(sc, ct)] = full.pvalues[INTER["DLD"]]
keys = [(sc, ct) for sc in SUBCORTICAL for ct in CORTICAL]
pv = np.array([sal_p[k] for k in keys])
o = np.argsort(pv)
ranks = np.empty(len(o), int); ranks[o] = np.arange(1, len(o) + 1)
q = np.minimum(1, pv * len(o) / ranks)
q[o] = np.minimum.accumulate(q[o][::-1])[::-1]
sal_fdr = {k: q[i] for i, k in enumerate(keys)}

# --- language: refit the single edge ---
llong = pd.read_csv(f"{LANG_DIR}/group_language_putamen_long.csv")
llong["code"] = llong["code"].astype(str)
dl = llong.merge(beh_emo[["code", "emotional"]], on="code", how="left").dropna(subset=["emotional"])
dl["emotional"] = dl["emotional"].round().astype(int).clip(0, 10)
dl = dl.rename(columns={"z": "FCz"})
dl["FCz_c"] = dl["FCz"] - dl["FCz"].mean()
lfull = smf.negativebinomial(
    "emotional ~ C(group, Treatment('TD')) * FCz_c", data=dl).fit(disp=0, maxiter=1000)
lred = smf.negativebinomial(
    "emotional ~ C(group, Treatment('TD')) + FCz_c", data=dl).fit(disp=0, maxiter=1000)
lang_lr_p = stats.chi2.sf(2 * (lfull.llf - lred.llf), 2)
lang_beta = lfull.params[INTER["DLD"]]
lang_waldp = lfull.pvalues[INTER["DLD"]]


def draw_fc_panel(ax, full, d, xmean, title_lines, beta_dld, waldp_dld,
                  title_fs=10, annot_fs=9.5):
    for g in GROUPS:
        dg = d[d.group == g]
        c = GROUP_COLORS[g]
        yj = dg["emotional"] + RNG.uniform(-0.15, 0.15, len(dg))
        ax.scatter(dg["FCz"], yj, s=20, alpha=0.8, color=c,
                   edgecolor="k", linewidth=0.3)
        xs, mu, lo, hi = band(full, d, "FCz", "FCz_c", g, xmean)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.13, lw=0)
        ax.plot(xs, mu, color=c, lw=2.0)
    ax.set_ylim(-0.5, 10.5)
    ax.grid(alpha=0.18)
    ax.set_axisbelow(True)
    place_title(ax, title_lines, title_fs)
    annot = [stat_seg(f"DLD-TD slope (β)={beta_dld:+.2f}, ", waldp_dld)]
    place_annot(ax, annot, annot_fs)


fig = plt.figure(figsize=(17.5, 9.6))
fig.set_facecolor("white")
sf_sal, sf_lang = fig.subfigures(1, 2, width_ratios=[3, 1.5], wspace=0.02)
sf_sal.set_facecolor("white")       # no block tint; networks cued by title colour
sf_lang.set_facecolor("white")
sf_sal.suptitle("Salience Network: Fronto-striatal FC",
                fontsize=17, fontweight="bold", color="#1f4e79", y=0.985)
sf_lang.suptitle("Language Network: Frontal-putamen FC",
                 fontsize=17, fontweight="bold", color="#a02020", y=0.985)

ax_sal = sf_sal.subplots(3, 3, sharey=True,
                         gridspec_kw=dict(top=0.90, bottom=0.10, left=0.085,
                                          right=0.985, hspace=0.52, wspace=0.14))
for i, sc in enumerate(SUBCORTICAL):
    for j, ct in enumerate(CORTICAL):
        full, d = sal_fit[(sc, ct)]
        title_lines = [
            [(f"{sc}-{ct}", False)],
            stat_seg("(interaction: FDR Adj ", sal_fdr[(sc, ct)], ")"),
        ]
        draw_fc_panel(ax_sal[i, j], full, d, d["FCz"].mean(), title_lines,
                      sal_beta[(sc, ct)], sal_waldp[(sc, ct)], title_fs=9.5)
sf_sal.supxlabel("Fronto-striatal FC (Fisher z)", fontsize=14)
sf_sal.supylabel(YLAB, fontsize=14)

ax_lang = sf_lang.subplots(1, 1)
ax_lang.set_box_aspect(1.18)         # near-square panel, vertically centred
lang_title = [stat_seg("left POp-putamen (interaction: ", lang_lr_p, ")")]
draw_fc_panel(ax_lang, lfull, dl, dl["FCz"].mean(), lang_title,
              lang_beta, lang_waldp, title_fs=13, annot_fs=11)
ax_lang.set_xlabel("Frontal-putamen FC (Fisher z)", fontsize=14)
ax_lang.set_ylabel(YLAB, fontsize=14)
handles = [Line2D([0], [0], color=GROUP_COLORS[g], lw=3, label=g) for g in GROUPS]
ax_lang.legend(handles=handles, loc="upper right", framealpha=0.9)

f2 = f"{OUT_DIR}/fig2_fc_scatter.svg"
fig.savefig(f2, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {f2}")


# ============================================================
# FIG 3 — SDQ-emotional vs network size : salience | language
# ============================================================
nlong = pd.read_csv(f"{NS_DIR}/group_network_size_long.csv")
nlong["code"] = nlong["code"].astype(str)


def size_fit(label):
    col = label.lower()
    s = nlong[nlong["network_label"] == label][["code", "group", "network_size_pct"]]
    d = (s.merge(beh[["code", "emotional"]], on="code", how="left")
           .rename(columns={"network_size_pct": col})
           .dropna(subset=["emotional", col]))
    d = d[d["group"].isin(GROUPS)].copy()
    d["emotional"] = d["emotional"].round().astype(int).clip(0, 10)
    xmean = d[col].mean()
    d[col + "_c"] = d[col] - xmean
    full = smf.negativebinomial(
        f"emotional ~ C(group, Treatment('TD')) * {col}_c", data=d).fit(disp=0, maxiter=1000)
    red = smf.negativebinomial(
        f"emotional ~ C(group, Treatment('TD')) + {col}_c", data=d).fit(disp=0, maxiter=1000)
    lr_p = stats.chi2.sf(2 * (full.llf - red.llf), 2)
    td_beta = full.params[f"{col}_c"]          # TD is the reference -> its slope
    td_p = full.pvalues[f"{col}_c"]
    return full, d, xmean, col, lr_p, td_beta, td_p


fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
for ax, label in zip(axes, ["Salience", "Language"]):
    full, d, xmean, col, lr_p, td_beta, td_p = size_fit(label)
    for g in GROUPS:
        dg = d[d.group == g]
        c = GROUP_COLORS[g]
        xs, mu, lo, hi = band(full, d, col, col + "_c", g, xmean, n=200)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(xs, mu, color=c, lw=2.4, label=g)
        yj = d.loc[d.group == g, "emotional"] + RNG.uniform(-0.12, 0.12, len(dg))
        ax.scatter(dg[col], yj, color=c, edgecolor="k", linewidth=0.3,
                   s=30, zorder=3, alpha=0.85)
    ax.set_xlabel(f"{label} network size (% cortical surface)")
    ax.set_ylim(-0.5, 10.5)
    ax.grid(alpha=0.25); ax.set_axisbelow(True)
    annot = [
        stat_seg("interaction: ", lr_p),
        stat_seg(f"TD slope (β)={td_beta:.2f}, ", td_p),
    ]
    place_annot(ax, annot, 10.5)
axes[0].set_ylabel(YLAB)
axes[1].legend(loc="upper right")          # legend on the right (Language) panel
fig.tight_layout()
f3 = f"{OUT_DIR}/fig3_network_size.svg"
fig.savefig(f3, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {f3}")
print("\nDone. Poster SVGs written to poster/")
