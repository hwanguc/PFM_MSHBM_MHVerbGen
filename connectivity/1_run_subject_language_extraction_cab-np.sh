#!/usr/bin/env bash

## Author: Han Wang
### 2026-08-14: Initial version (language frontal-putamen arm).

### Extracts whole-brain parcellated connectivity for one subject using the
### STOCK CAB-NP atlas (CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_
### parcels_LR.dlabel.nii), for the language frontal-putamen analysis.
###
### Why the stock atlas (not the anterior-striatum one used by the striatal arm):
### the language putamen ROI is the CAB-NP Language-network putamen parcel
### `Language-14_L-Putamen` (7 voxels, MNI ~ -20,+4,+5 = medial/anterior LEFT
### putamen; Gordon et al. 2021 subnetwork #3). The anterior-striatum atlas
### relabels precommissural (Y>=4) putamen voxels into L/R-Putamen-head, which
### cannibalises part of that parcel -- so a clean edge needs the stock atlas.
###
### Output FC matrix goes to a SEPARATE derivatives dir so the striatal arm's
### derivatives/fc/ is never touched.

set -e

# ============================================================
# CONFIG
# ============================================================
SUBJ=$1
# Preprocessed data + MS-HBM outputs live on the 2T internal drive (2026-07).
# NB: the path contains a space ("Data 001 2T") so every expansion must be quoted.
DATA_ROOT="/run/media/hanwang/Data 001 2T/hanwang/Documents/Data/verb_gen_krishnan/processed"
# Rest-only run (same run the striatal connectivity arm uses): first 25 vols of
# the session dropped + first 10s of each rest block removed (HRF spill-over).
DTSERIES="${DATA_ROOT}/${SUBJ}/MNINonLinear/Results/rfMRI_VERBGEN_AP_rest/rfMRI_VERBGEN_AP_rest_Atlas_hp2000_clean.dtseries.nii"

PROJECT_DIR=/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen
# Stock CAB-NP atlas (unmodified) -- keeps every CAB-NP parcel incl. the
# Language-network putamen parcel intact.
ATLAS=${PROJECT_DIR}/atlas/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii
OUTDIR=${PROJECT_DIR}/derivatives/fc_cabnp_stock

mkdir -p "${OUTDIR}"

# Sanity checks
if [ -z "${SUBJ}" ]; then
    echo "ERROR: No subject ID provided."
    echo "Usage: $0 SUBJECT_ID   (e.g. $0 sub-509BT)"
    exit 1
fi
if [ ! -f "${DTSERIES}" ]; then
    echo "ERROR: dtseries not found: ${DTSERIES}"
    exit 1
fi
if [ ! -f "${ATLAS}" ]; then
    echo "ERROR: atlas not found: ${ATLAS}"
    exit 1
fi

echo "=== Subject ${SUBJ} (language arm, stock CAB-NP) ==="

# ============================================================
# Parcellate using the stock CAB-NP atlas
# ============================================================
echo "[1/3] Parcellating..."
wb_command -cifti-parcellate \
    "${DTSERIES}" \
    "${ATLAS}" \
    COLUMN \
    "${OUTDIR}/${SUBJ}.ptseries.nii"

# ============================================================
# Correlate (Fisher-z)
# ============================================================
echo "[2/3] Computing FC matrix..."
wb_command -cifti-correlation \
    "${OUTDIR}/${SUBJ}.ptseries.nii" \
    "${OUTDIR}/${SUBJ}.pconn.nii" \
    -fisher-z

# ============================================================
# Export to text
# ============================================================
echo "[3/3] Exporting to text..."
wb_command -cifti-convert -to-text \
    "${OUTDIR}/${SUBJ}.pconn.nii" \
    "${OUTDIR}/${SUBJ}_FC.txt"

echo "Done: ${OUTDIR}/${SUBJ}_FC.txt"
