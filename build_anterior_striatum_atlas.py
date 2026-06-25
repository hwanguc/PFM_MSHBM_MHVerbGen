"""
build_anterior_striatum_atlas.py

## Author: Han Wang
### 25 Jun 2026: Initial version

Build a custom CAB-NP atlas in which the *anterior* (precommissural) Caudate and
Putamen voxels are relabelled into single bilateral head ROIs:

    L-Caudate-head, R-Caudate-head, L-Putamen-head, R-Putamen-head

"Anterior / head" is defined as MNI Y >= Y_THRESH. The anterior commissure sits
at Y = 0 in MNI152, so this is the precommissural striatum; Y >= 4 keeps the
caudate head bulge + anterior putamen while trimming the body/genu/tail, matching
the anterior striatal salience foci reported by Lynch et al. (2024,
https://www.nature.com/articles/s41586-024-07805-2).

Only Caudate/Putamen voxels are touched. Cortical surface parcels (ACC, AI, LPFC,
...) and all other subcortical structures (incl. NAcc) keep their stock CAB-NP
labels — the relabel is gated by anatomical structure, and cortex is surface-based
(no Y coordinate) so the Y threshold never applies to it.
"""

import nibabel as nib
import numpy as np
from nibabel.cifti2 import cifti2_axes as cax

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
SRC = f"{PROJECT_DIR}/atlas/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii"
OUT = f"{PROJECT_DIR}/atlas/CABNP_anteriorStriatum_Y4.dlabel.nii"
Y_THRESH = 4  # MNI Y (mm); anterior commissure at Y=0, so this is precommissural

# (CIFTI structure substring, new ROI name, new label key, RGBA in 0-1)
NEW_DEFS = [
    ("CAUDATE_LEFT",  "L-Caudate-head", 12171, (0.0,  0.72, 0.58, 1.0)),
    ("CAUDATE_RIGHT", "R-Caudate-head", 12172, (0.0,  0.72, 0.58, 1.0)),
    ("PUTAMEN_LEFT",  "L-Putamen-head", 12173, (0.88, 0.44, 0.33, 1.0)),
    ("PUTAMEN_RIGHT", "R-Putamen-head", 12174, (0.88, 0.44, 0.33, 1.0)),
]

dl = nib.load(SRC)
lab_ax = dl.header.get_axis(0)
bm = dl.header.get_axis(1)

data = dl.get_fdata().copy()            # (1, 91282)
keys = data[0].astype(int)
names = bm.name
Y = nib.affines.apply_affine(bm.affine, bm.voxel)[:, 1]   # surface vertices -> placeholder, never selected below

label_dict = dict(lab_ax.label[0])      # key -> (name, rgba)
for struct, newname, newkey, rgba in NEW_DEFS:
    sel = np.array([struct in n for n in names]) & (Y >= Y_THRESH)
    keys[sel] = newkey
    label_dict[newkey] = (newname, rgba)
    print(f"{newname:16s} key={newkey} voxels={sel.sum()}")

data[0] = keys
new_lab_ax = cax.LabelAxis(lab_ax.name, [label_dict])
img = nib.Cifti2Image(data, header=(new_lab_ax, bm), nifti_header=dl.nifti_header)
img.to_filename(OUT)
print(f"saved {OUT}")
