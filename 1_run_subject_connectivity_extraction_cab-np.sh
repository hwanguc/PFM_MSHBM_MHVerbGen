#!/usr/bin/env bash

## Author: Han Wang
### 7 May 2026: Initial version


### This script extracts the whole brain and ROI connectivity (nucleus accumbens, aneterior cingulate cortex, and basal ganglia) for the HCP pre-processed data after ICA+FIX pipeline for each subject and saves the results in a csv file.

#!/bin/bash
set -e

# ============================================================
# CONFIG
# ============================================================
SUBJ=$1
# Krishnan et al. (2021) data processed with adapted HCP pipeline (MSMSulc, no MSMAll)
#DTSERIES=/home/hanwang/Documents/Data/ucl/gos_ich/verb_gen_krishnan/processed/${SUBJ}/MNINonLinear/Results/rfMRI_VERBGEN_AP/rfMRI_VERBGEN_AP_Atlas_hp2000_clean.dtseries.nii
DTSERIES=/home/hanwang/Documents/Data/ucl/gos_ich/verb_gen_krishnan/processed/${SUBJ}/MNINonLinear/Results/rfMRI_VERBGEN_AP_rest/rfMRI_VERBGEN_AP_rest_Atlas_hp2000_clean.dtseries.nii


PROJECT_DIR=/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen
ATLAS=${PROJECT_DIR}/atlas/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii
OUTDIR=${PROJECT_DIR}/derivatives/fc

mkdir -p ${OUTDIR}

# Sanity checks
if [ -z "${SUBJ}" ]; then
    echo "ERROR: No subject ID provided."
    echo "Usage: $0 SUBJECT_ID"
    echo "Example: $0 sub-509BT"
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

echo "=== Subject ${SUBJ} ==="

# ============================================================
# Parcellate using the CAB-NP atlas (https://github.com/ColeLab/ColeAnticevicNetPartition/blob/master/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii)
# ============================================================
echo "[1/3] Parcellating..."
wb_command -cifti-parcellate \
    ${DTSERIES} \
    ${ATLAS} \
    COLUMN \
    ${OUTDIR}/${SUBJ}.ptseries.nii

# ============================================================
# Correlate (Fisher-z)
# ============================================================
echo "[2/3] Computing FC matrix..."
wb_command -cifti-correlation \
    ${OUTDIR}/${SUBJ}.ptseries.nii \
    ${OUTDIR}/${SUBJ}.pconn.nii \
    -fisher-z

# ============================================================
# Export to text
# ============================================================
echo "[3/3] Exporting to text..."
wb_command -cifti-convert -to-text \
    ${OUTDIR}/${SUBJ}.pconn.nii \
    ${OUTDIR}/${SUBJ}_FC.txt

echo "Done: ${OUTDIR}/${SUBJ}_FC.txt"