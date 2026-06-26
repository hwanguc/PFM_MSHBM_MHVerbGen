"""
run_group_connectivity.py

Batch driver for the CAB-NP anterior-striatum connectivity pipeline over every
DLD (BL) and TD (BT) subject in the verb-gen subsample (excluding 513BT = 36
subjects). For each subject it runs, in order:

  1. 1_run_subject_connectivity_extraction_cab-np.sh  -> parcellate the rest
     dtseries with the anterior-striatum atlas and export a Fisher-z FC matrix
     to derivatives/fc/<sub>_FC.txt   (the slow wb_command step)
  2. 2_run_subject_connectivity_analysis_cab-np.py     -> the three FC figures
     plus the fig-2 (collapsed region x region) and fig-3 (frontostriatal)
     CSVs, written to results/connectivity_outputs/<sub>/

One subject failing does not abort the batch. Progress is streamed live to
results/connectivity_outputs/run_group_connectivity_live.log (tail -f it), and a
per-subject success/failure summary is written to run_group_connectivity_log.csv.

Usage:
    python3 connectivity/run_group_connectivity.py              # all 36, re-extract
    python3 connectivity/run_group_connectivity.py --skip-existing   # skip extraction
                                                                     # if FC.txt exists
    python3 connectivity/run_group_connectivity.py --subjects sub-509BT sub-584BL

## Author: Han Wang
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
import pandas as pd

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
CONN_DIR = f"{PROJECT_DIR}/connectivity"
XLSX = ("/home/hanwang/Documents/Data/ucl/gos_ich/verb_gen_krishnan/"
        "behavioural_scq_sdq/dat_verbgen_scqsdq_subsample.xlsx")
OUTBASE = f"{PROJECT_DIR}/results/connectivity_outputs"
FC_TXT = f"{PROJECT_DIR}/derivatives/fc/{{}}_FC.txt"

EXTRACT_SH = f"{CONN_DIR}/1_run_subject_connectivity_extraction_cab-np.sh"
ANALYSE_PY = f"{CONN_DIR}/2_run_subject_connectivity_analysis_cab-np.py"


def subject_list():
    """BL (DLD) + BT (TD) subjects from the spreadsheet, excluding 513BT."""
    codes = pd.read_excel(XLSX)["code"].astype(str)
    keep = codes[(codes.str.endswith("BL") | codes.str.endswith("BT"))
                 & (codes != "513BT")]
    return ["sub-" + c for c in keep]


def run_stream(cmd, logf):
    """Run cmd, streaming combined stdout/stderr to both console and logf (live)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        logf.write(line)
        logf.flush()
    proc.wait()
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip the extraction step when <sub>_FC.txt already exists")
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="explicit subject list (default: all BL+BT from the xlsx)")
    args = ap.parse_args()

    subjects = args.subjects if args.subjects else subject_list()
    os.makedirs(OUTBASE, exist_ok=True)
    live_path = f"{OUTBASE}/run_group_connectivity_live.log"

    results = []
    with open(live_path, "w") as logf:
        def log(msg):
            print(msg); logf.write(msg + "\n"); logf.flush()

        log(f"# run_group_connectivity  started {datetime.now():%Y-%m-%d %H:%M:%S}")
        log(f"# {len(subjects)} subjects | skip_existing={args.skip_existing}")

        for i, sub in enumerate(subjects, 1):
            log(f"\n========== [{i}/{len(subjects)}] {sub} ==========")
            stage, ok, err = "extraction", True, ""
            try:
                # 1) extraction (skippable)
                if args.skip_existing and os.path.isfile(FC_TXT.format(sub)):
                    log(f"  [skip] FC.txt exists -> skipping extraction for {sub}")
                else:
                    rc = run_stream(["bash", EXTRACT_SH, sub], logf)
                    if rc != 0:
                        raise RuntimeError(f"extraction exited {rc}")

                # 2) analysis (always; fast, regenerates figures + CSVs)
                stage = "analysis"
                rc = run_stream(["python3", ANALYSE_PY, sub], logf)
                if rc != 0:
                    raise RuntimeError(f"analysis exited {rc}")

            except Exception as e:
                ok, err = False, f"{stage}: {e}"
                log(f"  FAILED {sub} -> {err}")
            results.append(dict(subject=sub, succeeded=ok, failed_stage="" if ok else stage,
                                error=err))

        # summary
        df = pd.DataFrame(results)
        n_ok = int(df["succeeded"].sum())
        log("\n================ SUMMARY ================")
        log(f"Succeeded: {n_ok}/{len(subjects)}")
        if n_ok < len(subjects):
            for _, r in df[~df["succeeded"]].iterrows():
                log(f"  {r['subject']}: {r['error']}")
        log_csv = f"{OUTBASE}/run_group_connectivity_log.csv"
        df.to_csv(log_csv, index=False)
        log(f"Wrote run log: {log_csv}")
        log(f"Done {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
