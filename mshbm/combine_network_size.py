"""
combine_network_size.py

Combine the per-subject MS-HBM network-size CSVs (written by run_subject_mshbm.m
into results/network_size_<variant>/<sub>_networksize.csv) into one long-format
CSV, annotated with each subject's group (DLD / TD / HSL) from the canonical
analysis table.

Variant selects which MS-HBM set to combine:
  full   -> results/network_size_full   (25-vol-cut rfMRI_VERBGEN_AP_full)
  icafix -> results/network_size_icafix (all-vols rfMRI_VERBGEN_AP)

Output: results/network_size_<variant>/group_network_size_long.csv
    columns: subject, code, group, network_id, network_label, network_size_pct

Usage:
    python3 mshbm/combine_network_size.py                 # full
    python3 mshbm/combine_network_size.py --variant icafix

## Author: Han Wang
"""

import argparse
import glob
import os
import pandas as pd

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
LISTCSV = ("/home/hanwang/Documents/Data/verb_gen_krishnan/"
           "behavioural_scq_sdq/dat_verbgen_analysis_144.csv")

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["full", "icafix"], default="full")
args = ap.parse_args()

NS_DIR = f"{PROJECT_DIR}/results/network_size_{args.variant}"
OUT = f"{NS_DIR}/group_network_size_long.csv"

# code -> group lookup from the analysis table
beh = pd.read_csv(LISTCSV)
code2group = dict(zip(beh["code"].astype(str), beh["group"].astype(str)))

files = sorted(glob.glob(f"{NS_DIR}/sub-*_networksize.csv"))
if not files:
    raise SystemExit(f"No per-subject CSVs found in {NS_DIR}. "
                     f"Run run_group_mshbm(Variant='{args.variant}') first.")

frames = []
for f in files:
    df = pd.read_csv(f)
    df["code"] = df["subject"].str.replace("^sub-", "", regex=True)
    df["group"] = df["code"].map(code2group)
    frames.append(df)

long = pd.concat(frames, ignore_index=True)
long = long[["subject", "code", "group", "network_id", "network_label", "network_size_pct"]]
long = long.sort_values(["group", "subject", "network_id"]).reset_index(drop=True)

os.makedirs(NS_DIR, exist_ok=True)
long.to_csv(OUT, index=False)

n_sub = long["subject"].nunique()
print(f"[{args.variant}] Combined {len(files)} subjects ({n_sub} unique) -> {OUT}")
print("Subjects per group:")
print(long.drop_duplicates("subject")["group"].value_counts().to_string())
if long["group"].isna().any():
    missing = long.loc[long["group"].isna(), "code"].unique()
    print(f"WARNING: no group found for: {list(missing)}")
