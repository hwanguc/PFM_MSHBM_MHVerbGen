"""
build_language_putamen_rois.py

## Author: Han Wang
### 2026-08-14: Initial version (language frontal-putamen arm).

Define + visually confirm the TWO ROIs for the language frontal-putamen
connectivity analysis, straight from the STOCK CAB-NP atlas (no custom carving):

  * Left pars opercularis  = `Language-14_L-Ctx`      (key 74; Glasser L_44_ROI)
  * Medial/ant. putamen    = `Language-14_L-Putamen`  (key 6140)

These are CAB-NP's Language-network (NETWORKKEY 6) "14" component in cortex and in
putamen. The putamen parcel is the only putamen parcel CAB-NP assigns to Language,
is LEFT-only, and sits in medial/anterior putamen -- matching subnetwork #3
("Med/Ant Putamen", converging with the language network) in Gordon et al. (2021,
Cereb Cortex, bhab387), Figures 1-3.

Outputs (nothing here is on the analysis critical path -- it is a QC/definition step):
  * atlas/CABNP_language14_rois.dlabel.nii   -- stock atlas masked to the 2 ROIs
                                                (drop into wb_view for a surface+vol check)
  * results/language_connectivity_outputs/roi_language_putamen_check.png
                                                -- MNI slices through the putamen ROI

Run with the nibabel venv:
    /home/hanwang/Apps/Programming/matlab-proj/pfm-nsi/.venv/bin/python \
        connectivity/build_language_putamen_rois.py
"""

import os
import numpy as np
import nibabel as nib
from nibabel.cifti2 import cifti2_axes as cax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
SRC = f"{PROJECT_DIR}/atlas/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii"
OUT_DLABEL = f"{PROJECT_DIR}/atlas/CABNP_language14_rois.dlabel.nii"
OUT_DIR = f"{PROJECT_DIR}/results/language_connectivity_outputs"
OUT_PNG = f"{OUT_DIR}/roi_language_putamen_check.png"

# The two ROIs (CAB-NP LABEL name -> key)
CORTEX_KEY = 74      # Language-14_L-Ctx  == Glasser L_44_ROI (pars opercularis)
PUTAMEN_KEY = 6140   # Language-14_L-Putamen

os.makedirs(OUT_DIR, exist_ok=True)

dl = nib.load(SRC)
lab_ax = dl.header.get_axis(0)
bm = dl.header.get_axis(1)
keys = dl.get_fdata()[0].astype(int)
names = np.array(bm.name)
label_dict = dict(lab_ax.label[0])   # key -> (name, rgba)

# ------------------------------------------------------------
# Report the two parcels
# ------------------------------------------------------------
print("=" * 66)
print("Language frontal-putamen ROIs (stock CAB-NP)")
print("=" * 66)
for key in (CORTEX_KEY, PUTAMEN_KEY):
    sel = keys == key
    nm = label_dict[key][0]
    n = int(sel.sum())
    struct = sorted(set(names[sel]))
    line = f"  key {key:5d}  {nm:24s}  n_grayord={n:4d}  {struct[0]}"
    v = bm.voxel[sel]
    good = v[:, 0] >= 0
    if good.any():
        mni = nib.affines.apply_affine(bm.affine, v[good])
        c = mni.mean(0)
        line += f"\n              MNI centroid ({c[0]:+.1f}, {c[1]:+.1f}, {c[2]:+.1f}) [+y anterior]"
    print(line)

# ------------------------------------------------------------
# 1. Masked dlabel (only the 2 ROIs kept; everything else -> 0/???)
# ------------------------------------------------------------
new_keys = np.zeros_like(keys)
new_keys[keys == CORTEX_KEY] = CORTEX_KEY
new_keys[keys == PUTAMEN_KEY] = PUTAMEN_KEY
new_label = {0: label_dict.get(0, ("???", (0, 0, 0, 0))),
             CORTEX_KEY: (label_dict[CORTEX_KEY][0], (1.0, 0.55, 0.0, 1.0)),
             PUTAMEN_KEY: (label_dict[PUTAMEN_KEY][0], (0.0, 0.6, 0.6, 1.0))}
new_lab_ax = cax.LabelAxis(lab_ax.name, [new_label])
img = nib.Cifti2Image(new_keys[None, :], header=(new_lab_ax, bm),
                      nifti_header=dl.nifti_header)
img.to_filename(OUT_DLABEL)
print(f"\nSaved masked dlabel (open in wb_view): {OUT_DLABEL}")

# ------------------------------------------------------------
# 2. Volume slices through the putamen ROI (MNI), for a quick visual check
# ------------------------------------------------------------
vol_shape = bm.volume_shape             # (i,j,k) of the CIFTI volume space
mask = np.zeros(vol_shape, dtype=float)
sel = keys == PUTAMEN_KEY
ijk = bm.voxel[sel]
ijk = ijk[ijk[:, 0] >= 0]
for i, j, k in ijk:
    mask[i, j, k] = 1.0
# centroid voxel (for slice selection)
ci, cj, ck = ijk.mean(0).round().astype(int)
mni_c = nib.affines.apply_affine(bm.affine, ijk).mean(0)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
# axial (k), coronal (j), sagittal (i) through centroid
planes = [("axial z", mask[:, :, ck].T, ci, cj),
          ("coronal y", mask[:, ck * 0 + cj, :].T, ci, ck),
          ("sagittal x", mask[ci, :, :].T, cj, ck)]
titles = [f"axial  z={mni_c[2]:+.0f}", f"coronal  y={mni_c[1]:+.0f}",
          f"sagittal  x={mni_c[0]:+.0f}"]
for ax, (nm, sl, a, b), t in zip(axes, planes, titles):
    ax.imshow(sl, origin="lower", cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(sl == 0, sl), origin="lower",
              cmap="autumn", vmin=0, vmax=1, alpha=0.9)
    ax.set_title(t, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Language-14_L-Putamen ROI (medial/anterior LEFT putamen)\n"
             f"{int(sel.sum())} voxels, MNI centroid "
             f"({mni_c[0]:+.0f}, {mni_c[1]:+.0f}, {mni_c[2]:+.0f})",
             fontsize=12)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved putamen ROI check image: {OUT_PNG}")
