"""
combine_network_size.py

Combine the per-subject MS-HBM network-size CSVs (written by run_subject_mshbm.m
into results/network_size/<sub>_networksize.csv) into a single long-format CSV,
annotated with each subject's group (DLD / TD / HSL) from the behavioural
spreadsheet.

Output: results/network_size/group_network_size_long.csv
    columns: subject, code, group, network_id, network_label, network_size_pct

## Author: Han Wang
"""

import glob
import os
import pandas as pd

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
NS_DIR = f"{PROJECT_DIR}/results/network_size"
XLSX = ("/home/hanwang/Documents/Data/ucl/gos_ich/verb_gen_krishnan/"
        "behavioural_scq_sdq/dat_verbgen_scqsdq_subsample.xlsx")
OUT = f"{NS_DIR}/group_network_size_long.csv"

# ------------------------------------------------------------
# code -> group lookup from the spreadsheet
# ------------------------------------------------------------
beh = pd.read_excel(XLSX)
code2group = dict(zip(beh["code"].astype(str), beh["group"].astype(str)))

# ------------------------------------------------------------
# Gather per-subject CSVs
# ------------------------------------------------------------
files = sorted(glob.glob(f"{NS_DIR}/sub-*_networksize.csv"))
if not files:
    raise SystemExit(f"No per-subject CSVs found in {NS_DIR}. Run run_group_mshbm first.")

frames = []
for f in files:
    df = pd.read_csv(f)
    # subject like "sub-584BL" -> code "584BL"
    df["code"] = df["subject"].str.replace("^sub-", "", regex=True)
    df["group"] = df["code"].map(code2group)
    frames.append(df)

long = pd.concat(frames, ignore_index=True)
long = long[["subject", "code", "group", "network_id", "network_label", "network_size_pct"]]
long = long.sort_values(["group", "subject", "network_id"]).reset_index(drop=True)

os.makedirs(NS_DIR, exist_ok=True)
long.to_csv(OUT, index=False)

n_sub = long["subject"].nunique()
print(f"Combined {len(files)} subjects ({n_sub} unique) -> {OUT}")
print("Subjects per group:")
print(long.drop_duplicates("subject")["group"].value_counts().to_string())
if long["group"].isna().any():
    missing = long.loc[long["group"].isna(), "code"].unique()
    print(f"WARNING: no group found in spreadsheet for: {list(missing)}")
