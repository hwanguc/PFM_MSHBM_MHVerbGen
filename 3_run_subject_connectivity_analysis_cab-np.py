"""
single_subject.py

## Author: Han Wang
### 8 May 2026: Initial version

Visualise FC between selected ROIs for one subject, using the FC matrix
output from 02_run_subject.sh.

ROIs are specified using Glasser parcel names. The script automatically
translates them to CAB-NP names by reading the official label key file
(GLASSERLABELNAME column).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# CONFIG
# ============================================================
SUBJ = "100307"
PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"

FC_FILE       = f"{PROJECT_DIR}/derivatives/fc/sub-{SUBJ}_FC.txt"
CABNP_LABELS  = f"{PROJECT_DIR}/derivatives/cabnp_labels.txt"
CABNP_KEY     = f"{PROJECT_DIR}/atlas/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"

OUTDIR = f"{PROJECT_DIR}/results/sub-{SUBJ}"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# ROI definitions — using Glasser names (translated automatically)
# ============================================================
ROI_GROUPS_GLASSER = {
    # ============================================================
    # ACC (5 parcels per hemisphere) — from sciencedirect S1053811920305450
    # Perigenual + subgenual + dorsal ACC subdivisions
    # ============================================================
    "ACC":  ["L_s32_ROI",     "R_s32_ROI",
             "L_p32_ROI",     "R_p32_ROI",
             "L_a24_ROI",     "R_a24_ROI",
             "L_p24_ROI",     "R_p24_ROI",
             "L_d32_ROI",     "R_d32_ROI"],

    # ============================================================
    # AI (2 parcels per hemisphere) — from PMC6032992
    # Anterior insula
    # ============================================================
    "AI":   ["L_AVI_ROI",     "R_AVI_ROI",
             "L_AAIC_ROI",    "R_AAIC_ROI"],

    # ============================================================
    # LPFC — from Assem et al. 2020 + PMC11761636
    # Core lateral frontal MD areas (excluding AVI, kept in AI):
    #   a9-46v, p9-46v, 8C, IFJp, i6-8
    # Plus explicitly named in the paper: 44, 45
    # ============================================================
    "LPFC": ["L_a9-46v_ROI",  "R_a9-46v_ROI",
             "L_p9-46v_ROI",  "R_p9-46v_ROI",
             "L_8C_ROI",      "R_8C_ROI",
             "L_IFJp_ROI",    "R_IFJp_ROI",
             "L_i6-8_ROI",    "R_i6-8_ROI",
             "L_44_ROI",      "R_44_ROI",
             "L_45_ROI",      "R_45_ROI"],
}




# Subcortical: aggregate all parcels for each anatomical structure
# (CAB-NP splits each structure across multiple networks)
SUBCORTICAL_PATTERNS = {
    "NAcc":    ["_L-Accumbens", "_R-Accumbens"],
    "Caudate": ["_L-Caudate",   "_R-Caudate"],
    "Putamen": ["_L-Putamen",   "_R-Putamen"],
}

CORTICAL_REGIONS    = list(ROI_GROUPS_GLASSER.keys())
SUBCORTICAL_REGIONS = list(SUBCORTICAL_PATTERNS.keys())
REGION_ORDER        = CORTICAL_REGIONS + SUBCORTICAL_REGIONS

REGION_COLORS = {
    "ACC":     "#d63031",
    "AI":      "#0984e3",
    "LPFC":    "#fdcb6e",
    "NAcc":    "#6c5ce7",
    "Caudate": "#00b894",
    "Putamen": "#e17055",
}

# ============================================================
# Load CAB-NP label key (the source of truth for Glasser→CAB-NP)
# ============================================================
print(f"Loading CAB-NP label key: {CABNP_KEY}")
key_df = pd.read_csv(CABNP_KEY, sep="\t")
print(f"  {len(key_df)} parcels")

# Build Glasser → CAB-NP lookup
glasser_to_cabnp = dict(zip(
    key_df.loc[key_df["GLASSERLABELNAME"].notna(), "GLASSERLABELNAME"],
    key_df.loc[key_df["GLASSERLABELNAME"].notna(), "LABEL"]
))

# Translate cortical ROIs
print("\nTranslating Glasser → CAB-NP names:")
ROI_GROUPS_CABNP = {}
for region, glasser_names in ROI_GROUPS_GLASSER.items():
    cabnp_names = []
    for gname in glasser_names:
        if gname in glasser_to_cabnp:
            cname = glasser_to_cabnp[gname]
            cabnp_names.append(cname)
            print(f"  {gname:18s} → {cname}")
        else:
            print(f"  WARNING: {gname} not in label key!")
    ROI_GROUPS_CABNP[region] = cabnp_names

# ============================================================
# Load FC matrix and parcel order from cabnp_labels.txt
# ============================================================
print(f"\nLoading FC matrix: {FC_FILE}")
fc_z = np.loadtxt(FC_FILE)
print(f"  Shape: {fc_z.shape}")

fc_r = np.tanh(fc_z)
fc_r[np.isinf(fc_r)] = np.nan
np.fill_diagonal(fc_r, np.nan)

print(f"\nLoading parcel order: {CABNP_LABELS}")
labels = []
with open(CABNP_LABELS) as f:
    lines = [l.strip() for l in f if l.strip()]
# Format alternates: name, "key R G B A"
for i in range(0, len(lines), 2):
    labels.append(lines[i])
print(f"  {len(labels)} parcels")

if len(labels) != fc_r.shape[0]:
    raise ValueError(f"Label count ({len(labels)}) != matrix dim ({fc_r.shape[0]})")

# ============================================================
# Map ROI names → matrix indices
# ============================================================
def find_indices(target_patterns, labels, mode="exact"):
    """mode='exact': full label match. mode='contains': substring match."""
    indices = []
    for tname in target_patterns:
        if mode == "exact":
            matches = [i for i, l in enumerate(labels) if l == tname]
        else:
            matches = [i for i, l in enumerate(labels) if tname in l]
        if not matches:
            print(f"  WARNING: pattern '{tname}' ({mode}) not found!")
        indices.extend(matches)
    return sorted(set(indices))

print("\nMapping ROI groups to matrix indices:")
roi_indices = {}

# Cortical: exact match against CAB-NP names
for region, cabnp_names in ROI_GROUPS_CABNP.items():
    idx = find_indices(cabnp_names, labels, mode="exact")
    roi_indices[region] = idx
    print(f"  {region:8s} (cortical):    {len(idx)} parcels")

# Subcortical: substring match (aggregate across networks)
for region, patterns in SUBCORTICAL_PATTERNS.items():
    idx = find_indices(patterns, labels, mode="contains")
    roi_indices[region] = idx
    print(f"  {region:8s} (subcortical): {len(idx)} parcels")

# ============================================================
# Build ordered parcel list (cortical → subcortical)
# ============================================================
ordered_indices = []
ordered_parcel_labels = []

for region in REGION_ORDER:
    for i in roi_indices[region]:
        ordered_indices.append(i)
        ordered_parcel_labels.append(labels[i])

sub_fc = fc_r[np.ix_(ordered_indices, ordered_indices)]
n_parcels = len(ordered_indices)
print(f"\nSubmatrix shape: {sub_fc.shape}")

# ============================================================
# FIGURE 1: Per-parcel heatmap
# ============================================================
print("\n[1/3] Plotting per-parcel heatmap...")

fig, ax = plt.subplots(figsize=(15, 13))
cmap = plt.cm.RdBu_r.copy()
cmap.set_bad("white")

vmax = 0.6
im = ax.imshow(sub_fc, cmap=cmap, vmin=-vmax, vmax=vmax,
               aspect="equal", interpolation="nearest")

ax.set_xticks(range(n_parcels))
ax.set_yticks(range(n_parcels))
ax.set_xticklabels(ordered_parcel_labels, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(ordered_parcel_labels, fontsize=7)

ax.set_xticks(np.arange(-0.5, n_parcels, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_parcels, 1), minor=True)
ax.grid(which="minor", color="grey", linewidth=0.2)
ax.tick_params(which="minor", bottom=False, left=False)

boundaries = []
cum = 0
for region in REGION_ORDER:
    cum += len(roi_indices[region])
    boundaries.append(cum)
for b in boundaries[:-1]:
    ax.axhline(b - 0.5, color="black", linewidth=1.2)
    ax.axvline(b - 0.5, color="black", linewidth=1.2)

n_cortical = sum(len(roi_indices[r]) for r in CORTICAL_REGIONS)
ax.axhline(n_cortical - 0.5, color="black", linewidth=2.2)
ax.axvline(n_cortical - 0.5, color="black", linewidth=2.2)

prev = 0
for region in REGION_ORDER:
    n = len(roi_indices[region])
    if n == 0:
        continue
    center = prev + n / 2 - 0.5
    ax.text(center, -3, region, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=REGION_COLORS[region])
    ax.add_patch(Rectangle((prev - 0.5, -1.5), n, 0.7,
                           facecolor=REGION_COLORS[region],
                           edgecolor="none", clip_on=False))
    prev += n

cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cbar.set_label("Pearson's r", fontsize=11)
ax.set_title(f"Subject {SUBJ} — Per-parcel FC", fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/01_full_ROI_FC_1.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/01_full_ROI_FC_1.png")

# ============================================================
# FIGURE 2: Collapsed-by-region heatmap (6×6)
# ============================================================
print("\n[2/3] Plotting collapsed (region × region) heatmap...")

n_regions = len(REGION_ORDER)
collapsed = np.full((n_regions, n_regions), np.nan)

for i, r1 in enumerate(REGION_ORDER):
    for j, r2 in enumerate(REGION_ORDER):
        if i == j:
            continue
        idx1 = roi_indices[r1]
        idx2 = roi_indices[r2]
        if not idx1 or not idx2:
            continue
        sub = fc_z[np.ix_(idx1, idx2)]
        collapsed[i, j] = np.tanh(np.nanmean(sub))

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(collapsed, cmap=cmap, vmin=-vmax, vmax=vmax,
               aspect="equal", interpolation="nearest")

ax.set_xticks(range(n_regions))
ax.set_yticks(range(n_regions))
ax.set_xticklabels(REGION_ORDER, rotation=45, ha="right", fontsize=11)
ax.set_yticklabels(REGION_ORDER, fontsize=11)

for tick, region in zip(ax.get_xticklabels(), REGION_ORDER):
    tick.set_color(REGION_COLORS[region])
    tick.set_fontweight("bold")
for tick, region in zip(ax.get_yticklabels(), REGION_ORDER):
    tick.set_color(REGION_COLORS[region])
    tick.set_fontweight("bold")

n_cort = len(CORTICAL_REGIONS)
ax.axhline(n_cort - 0.5, color="black", linewidth=2)
ax.axvline(n_cort - 0.5, color="black", linewidth=2)

for i in range(n_regions):
    for j in range(n_regions):
        if not np.isnan(collapsed[i, j]):
            color = "white" if abs(collapsed[i, j]) > 0.3 else "black"
            ax.text(j, i, f"{collapsed[i, j]:+.2f}",
                    ha="center", va="center", fontsize=10, color=color)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson's r", fontsize=11)

ax.set_title(f"Subject {SUBJ} — Region-level FC\n(values averaged in Fisher-z space)",
             fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/02_collapsed_FC_1.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/02_collapsed_FC_1.png")

# ============================================================
# FIGURE 3: Frontostriatal heatmap (cortex × subcortex only)
# ============================================================
print("\n[3/3] Plotting frontostriatal-only heatmap...")

frontostriatal = np.full((len(SUBCORTICAL_REGIONS), len(CORTICAL_REGIONS)), np.nan)

for i, sub_r in enumerate(SUBCORTICAL_REGIONS):
    for j, cort_r in enumerate(CORTICAL_REGIONS):
        idx_s = roi_indices[sub_r]
        idx_c = roi_indices[cort_r]
        if not idx_s or not idx_c:
            continue
        sub = fc_z[np.ix_(idx_s, idx_c)]
        frontostriatal[i, j] = np.tanh(np.nanmean(sub))

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(frontostriatal, cmap=cmap, vmin=-vmax, vmax=vmax,
               aspect="equal", interpolation="nearest")

ax.set_xticks(range(len(CORTICAL_REGIONS)))
ax.set_yticks(range(len(SUBCORTICAL_REGIONS)))
ax.set_xticklabels(CORTICAL_REGIONS, fontsize=12, fontweight="bold")
ax.set_yticklabels(SUBCORTICAL_REGIONS, fontsize=12, fontweight="bold")

for tick, region in zip(ax.get_xticklabels(), CORTICAL_REGIONS):
    tick.set_color(REGION_COLORS[region])
for tick, region in zip(ax.get_yticklabels(), SUBCORTICAL_REGIONS):
    tick.set_color(REGION_COLORS[region])

for i in range(len(SUBCORTICAL_REGIONS)):
    for j in range(len(CORTICAL_REGIONS)):
        if not np.isnan(frontostriatal[i, j]):
            color = "white" if abs(frontostriatal[i, j]) > 0.3 else "black"
            ax.text(j, i, f"{frontostriatal[i, j]:+.2f}",
                    ha="center", va="center", fontsize=12, color=color)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson's r", fontsize=11)
ax.set_xlabel("Cortical zone", fontsize=11)
ax.set_ylabel("Subcortical structure", fontsize=11)
ax.set_title(f"Subject {SUBJ} — Frontostriatal FC", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/03_frontostriatal_FC_1.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTDIR}/03_frontostriatal_FC_1.png")

# ============================================================
# Save numeric values + Lynch-style edges printout
# ============================================================
collapsed_df = pd.DataFrame(collapsed, index=REGION_ORDER, columns=REGION_ORDER)
collapsed_df.to_csv(f"{OUTDIR}/sub-{SUBJ}_FC_values_1.csv")
print(f"\n  Saved: {OUTDIR}/sub-{SUBJ}_FC_values_1.csv")

print("\n" + "=" * 50)
print("Lynch-style frontostriatal edges (Pearson's r):")
print("=" * 50)
for sub_r in SUBCORTICAL_REGIONS:
    for cort_r in CORTICAL_REGIONS:
        i = REGION_ORDER.index(sub_r)
        j = REGION_ORDER.index(cort_r)
        val = collapsed[i, j]
        marker = " ← Lynch primary" if (sub_r, cort_r) == ("NAcc", "ACC") else ""
        print(f"  {sub_r:8s} ↔ {cort_r:5s}: r = {val:+.3f}{marker}")

print(f"\nDone. All outputs in: {OUTDIR}/")