"""
build_analysis_table.py

Build the single canonical analysis table for the full verb-gen MS-HBM /
connectivity dataset (n = 144): the 36-subject original subsample (BL/BT,
513BT excluded) + the 509BT demo participant + the 107 newly pre-processed
subjects (99 batch101_lab + 8 HSL_lab, 512BT/664BT excluded as pre-processing
failures).

Group is derived from the code suffix: BL -> DLD, BT -> TD, BH -> HSL.

SDQ / SCQ scores (incl. `emotional`) are joined from the full behavioural
source `BOLD_data_SDQ_SCQ_27032025.xlsx`, which covers all 144 subjects.

Output: dat_verbgen_analysis_144.csv in the behavioural dir, with one row per
analysed subject:
    code, subject, group, source, sex, age,
    emotional, conduct, hyperactivity, peer_problems, prosocial,
    total_difficulties, scq_total

## Author: Han Wang
"""

import os
import pandas as pd

BDIR = "/home/hanwang/Documents/Data/verb_gen_krishnan/behavioural_scq_sdq"
SUBSAMPLE = os.path.join(BDIR, "dat_verbgen_scqsdq_subsample.xlsx")
MANIFEST = os.path.join(BDIR, "dat_verbgen_pre-processed.csv")
SDQ_SRC = os.path.join(BDIR, "BOLD_data_SDQ_SCQ_27032025.xlsx")
OUT = os.path.join(BDIR, "dat_verbgen_analysis_144.csv")

SUFFIX_TO_GROUP = {"BL": "DLD", "BT": "TD", "BH": "HSL"}


def group_from_code(code: str) -> str:
    return SUFFIX_TO_GROUP.get(str(code).strip()[-2:], "UNKNOWN")


def main() -> None:
    # ---- original 37: subsample BL/BT excl 513BT (36) + 509BT demo ----
    sub = pd.read_excel(SUBSAMPLE)
    sub["code"] = sub["code"].astype(str).str.strip()
    done36 = sub.loc[
        (sub["code"].str.endswith(("BL", "BT"))) & (sub["code"] != "513BT"),
        "code",
    ].tolist()
    original = [(c, "subsample36") for c in done36] + [("509BT", "demo509")]

    # ---- 107 newly pre-processed (99 batch + 8 HSL) ----
    man = pd.read_csv(MANIFEST)
    man["code"] = man["code"].astype(str).str.strip()
    new107 = list(zip(man["code"], man["source"]))

    rows = original + new107
    df = pd.DataFrame(rows, columns=["code", "source"]).drop_duplicates("code")
    df["group"] = df["code"].map(group_from_code)
    df["subject"] = "sub-" + df["code"]

    # ---- join SDQ/SCQ scores ----
    sdq = pd.read_excel(SDQ_SRC)
    sdq["code"] = sdq["code"].astype(str).str.strip()
    keep = [
        "code", "sex", "age", "emotional", "conduct", "hyperactivity",
        "peer_problems", "prosocial", "total_difficulties", "scq_total",
    ]
    sdq = sdq[[c for c in keep if c in sdq.columns]].drop_duplicates("code")
    out = df.merge(sdq, on="code", how="left")

    # ---- sanity checks ----
    assert out["group"].isin(["DLD", "TD", "HSL"]).all(), "unmapped group(s)"
    miss = out.loc[out["emotional"].isna(), "code"].tolist()
    assert not miss, f"missing emotional for: {miss}"

    cols = [
        "code", "subject", "group", "source", "sex", "age",
        "emotional", "conduct", "hyperactivity", "peer_problems",
        "prosocial", "total_difficulties", "scq_total",
    ]
    out = out[[c for c in cols if c in out.columns]].sort_values(
        ["group", "code"]
    ).reset_index(drop=True)
    out.to_csv(OUT, index=False)

    print(f"Wrote {OUT}: n = {len(out)}")
    print(out["group"].value_counts().to_dict())
    print(out["source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
