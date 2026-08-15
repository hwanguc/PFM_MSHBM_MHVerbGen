"""
2_run_subject_language_analysis_cab-np.py

## Author: Han Wang
### 2026-08-14: Initial version (language frontal-putamen arm).

Extract the single language frontal-putamen FC edge for one subject from the
stock-CAB-NP Fisher-z FC matrix produced by
1_run_subject_language_extraction_cab-np.sh:

    Language-14_L-Ctx  (= Glasser L_44, left pars opercularis)
        <->  Language-14_L-Putamen  (medial/anterior left putamen)

Gordon et al. (2021, Cereb Cortex bhab387) subnetwork #3: medial/anterior
putamen converging with the (left-lateralised) language network.

The individual "connectivity map" here is a single edge, so this writes a
one-row per-subject CSV (mirroring the striatal arm's
<sub>_03_frontostriatal_FC_*.csv, which the group scripts glob over).

Input:  derivatives/fc_cabnp_stock/<sub>_FC.txt   (718x718 Fisher-z)
        derivatives/cabnp_stock_labels.txt         (parcel order, one per line)
Output: results/language_connectivity_outputs/<sub>/<sub>_language_putamen_FC.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"

# The one edge: (cortical CAB-NP name, subcortical CAB-NP name)
CORTICAL = "Language-14_L-Ctx"        # Glasser L_44 (pars opercularis)
SUBCORTICAL = "Language-14_L-Putamen"  # medial/anterior left putamen
CORTICAL_LABEL = "L_44"                # short display name
SUBCORTICAL_LABEL = "L_Putamen_lang"   # short display name

parser = argparse.ArgumentParser(
    description="Language frontal-putamen FC edge for one subject (stock CAB-NP).")
parser.add_argument("subject", nargs="?", default="sub-509BT",
                    help="subject id, e.g. sub-509BT (default: sub-509BT)")
parser.add_argument("--outbase", default="results/language_connectivity_outputs",
                    help="output dir is <PROJECT_DIR>/<outbase>/<subject>")
args = parser.parse_args()
SUBJ = args.subject

FC_FILE = f"{PROJECT_DIR}/derivatives/fc_cabnp_stock/{SUBJ}_FC.txt"
LABELS = f"{PROJECT_DIR}/derivatives/cabnp_stock_labels.txt"
OUTDIR = f"{PROJECT_DIR}/{args.outbase}/{SUBJ}"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------------------------------------
# Load parcel order + FC matrix
# ------------------------------------------------------------
with open(LABELS) as f:
    labels = [l.strip() for l in f if l.strip()]

fc_z = np.loadtxt(FC_FILE)
if fc_z.shape[0] != len(labels):
    raise ValueError(f"Label count ({len(labels)}) != matrix dim ({fc_z.shape[0]})")

try:
    i = labels.index(CORTICAL)
    j = labels.index(SUBCORTICAL)
except ValueError as e:
    raise ValueError(f"ROI parcel not found in {LABELS}: {e}")

z = float(fc_z[i, j])
r = float(np.tanh(z))

# ------------------------------------------------------------
# Write the one-row per-subject edge CSV
# ------------------------------------------------------------
out = pd.DataFrame([dict(subject=SUBJ,
                         edge=f"{SUBCORTICAL_LABEL}-{CORTICAL_LABEL}",
                         cortical=CORTICAL_LABEL, subcortical=SUBCORTICAL_LABEL,
                         cortical_cabnp=CORTICAL, subcortical_cabnp=SUBCORTICAL,
                         r=r, z=z)])
out_csv = f"{OUTDIR}/{SUBJ}_language_putamen_FC.csv"
out.to_csv(out_csv, index=False)
print(f"{SUBJ}: L_44 <-> L-Putamen(Language)  r={r:+.4f}  z={z:+.4f}")
print(f"  Saved: {out_csv}")
