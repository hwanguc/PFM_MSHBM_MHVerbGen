"""
run_group_language_connectivity.py

## Author: Han Wang
### 2026-08-14: Initial version (language frontal-putamen arm).

Batch driver for the language frontal-putamen connectivity arm over every
analysed subject (n=144: DLD/BL, TD/BT, HSL/BH) in dat_verbgen_analysis_144.csv.
Per subject, in order:

  1. 1_run_subject_language_extraction_cab-np.sh -> parcellate the rest dtseries
     with the STOCK CAB-NP atlas, Fisher-z FC to derivatives/fc_cabnp_stock/
     <sub>_FC.txt  (the slow wb_command step)
  2. 2_run_subject_language_analysis_cab-np.py    -> pull the single
     L_44 <-> Language-14_L-Putamen edge to
     results/language_connectivity_outputs/<sub>/<sub>_language_putamen_FC.csv

One subject failing does not abort the batch. Mirrors run_group_connectivity.py.

Usage:
    python3 connectivity/run_group_language_connectivity.py               # all 144
    python3 connectivity/run_group_language_connectivity.py --skip-existing
    python3 connectivity/run_group_language_connectivity.py --subjects sub-509BT
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
import pandas as pd

PROJECT_DIR = "/home/hanwang/Apps/Programming/matlab-proj/PFM_MSHBM_MHVerbGen"
CONN_DIR = f"{PROJECT_DIR}/connectivity"
LISTCSV = ("/home/hanwang/Documents/Data/verb_gen_krishnan/"
           "behavioural_scq_sdq/dat_verbgen_analysis_144.csv")
OUTBASE = f"{PROJECT_DIR}/results/language_connectivity_outputs"
FC_TXT = f"{PROJECT_DIR}/derivatives/fc_cabnp_stock/{{}}_FC.txt"

EXTRACT_SH = f"{CONN_DIR}/1_run_subject_language_extraction_cab-np.sh"
ANALYSE_PY = f"{CONN_DIR}/2_run_subject_language_analysis_cab-np.py"


def subject_list():
    codes = pd.read_csv(LISTCSV)["code"].astype(str)
    return ["sub-" + c for c in codes]


def run_stream(cmd, logf):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush()
        logf.write(line); logf.flush()
    proc.wait()
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip extraction when <sub>_FC.txt already exists")
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="explicit subject list (default: all 144 from the table)")
    args = ap.parse_args()

    subjects = args.subjects if args.subjects else subject_list()
    os.makedirs(OUTBASE, exist_ok=True)
    live_path = f"{OUTBASE}/run_group_language_connectivity_live.log"

    results = []
    with open(live_path, "w") as logf:
        def log(msg):
            print(msg); logf.write(msg + "\n"); logf.flush()

        log(f"# run_group_language_connectivity started {datetime.now():%Y-%m-%d %H:%M:%S}")
        log(f"# {len(subjects)} subjects | skip_existing={args.skip_existing}")

        for i, sub in enumerate(subjects, 1):
            log(f"\n========== [{i}/{len(subjects)}] {sub} ==========")
            stage, ok, err = "extraction", True, ""
            try:
                if args.skip_existing and os.path.isfile(FC_TXT.format(sub)):
                    log(f"  [skip] FC.txt exists -> skipping extraction for {sub}")
                else:
                    rc = run_stream(["bash", EXTRACT_SH, sub], logf)
                    if rc != 0:
                        raise RuntimeError(f"extraction exited {rc}")
                stage = "analysis"
                rc = run_stream([sys.executable, ANALYSE_PY, sub], logf)
                if rc != 0:
                    raise RuntimeError(f"analysis exited {rc}")
            except Exception as e:
                ok, err = False, f"{stage}: {e}"
                log(f"  FAILED {sub} -> {err}")
            results.append(dict(subject=sub, succeeded=ok,
                                failed_stage="" if ok else stage, error=err))

        df = pd.DataFrame(results)
        n_ok = int(df["succeeded"].sum())
        log("\n================ SUMMARY ================")
        log(f"Succeeded: {n_ok}/{len(subjects)}")
        if n_ok < len(subjects):
            for _, r in df[~df["succeeded"]].iterrows():
                log(f"  {r['subject']}: {r['error']}")
        log_csv = f"{OUTBASE}/run_group_language_connectivity_log.csv"
        df.to_csv(log_csv, index=False)
        log(f"Wrote run log: {log_csv}")
        log(f"Done {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
