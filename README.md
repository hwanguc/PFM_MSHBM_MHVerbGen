# PFM_MSHBM_MHVerbGen

**Author: Han Wang (2025–2026)**

This repository contains the code for precision functional mapping (PFM; [Gordon et al., 2017](https://www.cell.com/neuron/fulltext/S0896-6273(17)30613-X)) and functional connectivity (FC) analyses for data from [Krishnan et al. (2021)](https://www.sciencedirect.com/science/article/pii/S1053811920310843). The PFM is based on a multi-session hierarchical Bayesian model (MS-HBM) pipeline developed by [Kong et al. (2019)](https://academic.oup.com/cercor/article/29/6/2533/5033556?login=false). The pipeline maps brain functional networks in individual participants whose resting-state data has been pre-processed using the [Human Connectome Project (HCP) pipeline](https://github.com/Washington-University/HCPpipelines) with denoising via [ICA-FIX](https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/fix.html).

The whole-brain FC analysis uses the parcellation defined by the Cole-Anticevic Brain Network Atlas (CAB-NP; [Ji et al., 2019](https://www.sciencedirect.com/science/article/abs/pii/S1053811918319657); [GitHub repo](https://github.com/ColeLab/ColeAnticevicNetPartition)). Region-of-interest (ROI) FC analysis uses nodes defined on the functional cortical atlas from [Glasser et al. (2016)](https://www.nature.com/articles/nature18933), which underpins the cortical parcels in Ji et al. (2019).

The motivating question is the frontostriatal salience network and its relationship to mood, following the salience-network expansion in depression reported by [Lynch et al. (2024)](https://www.nature.com/articles/s41586-024-07805-2). A parallel arm examines the **language network's** frontal-putamen connectivity (subnetwork #3 of [Gordon et al., 2021](https://doi.org/10.1093/cercor/bhab387)), motivated by the elevated emotional difficulties seen in developmental language disorder (DLD).

---

## Cohort & data

Subjects are the verb-generation cohort; the `code` column suffix gives the group: **BL = DLD**, **BT = TD/control**, **BH = HSL**. Analyses cover the **full processed cohort (n = 144: DLD 53, HSL 27, TD 64)** — the original 36-subject subsample + the `509BT` demo participant + 107 newly pre-processed subjects (99 `batch101` + 8 `HSL`; `512BT`/`664BT` excluded as pre-processing failures, `513BT` excluded as failed QC / replaced by `675BT`). The canonical subject list **and** SDQ scores are assembled by `stats/build_analysis_table.py` into `dat_verbgen_analysis_144.csv` (group from the code suffix; SDQ from `BOLD_data_SDQ_SCQ_27032025.xlsx`), which every driver/stats script reads. Group comparisons are **3-group (DLD / HSL / TD)**.

Pre-processed subject data **and** MS-HBM outputs live on a 2T internal drive at `/run/media/hanwang/Data 001 2T/hanwang/Documents/Data/verb_gen_krishnan/processed/<sub-CODE>/` (folder name = `sub-` + code). Because that mount path contains a space — which breaks CBIG's unquoted shell `mkdir` calls — the MS-HBM `BaseDir` points at a **no-space symlink** `~/verbgen_processed → …/processed`. Update `BaseDir` (`run_subject_mshbm.m`) / `DATA_ROOT` (`1_run_subject_connectivity_extraction_cab-np.sh`) if the data moves.

**Dependencies:** MATLAB (tested R2025a) with the [CBIG](https://github.com/ThomasYeoLab/CBIG) functions on the path (see `Paths{3}` in `run_subject_mshbm.m`); [Connectome Workbench](https://www.humanconnectome.org/software/connectome-workbench) (`wb_command`); Python 3 with `nibabel`, `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `openpyxl`, `pingouin`, `scikit-posthocs`. A project virtualenv is provided: `python -m venv .venv && .venv/bin/pip install -r requirements.txt` — all Python commands below assume `.venv/bin/python`.

---

## Pipeline A — Group-level MS-HBM individual network mapping

Maps individual functional networks per subject and aggregates cortical network sizes to the group level.

MS-HBM is run in **two comparable variants** (a `Variant` argument) so their network-size results can be compared side by side:
- **`full`** — run `rfMRI_VERBGEN_AP_full` (session minus the first 25 volumes, dropped because the noise-cancelling headphones were still adapting) → per-subject `mshbm_output/`, CSVs in `results/network_size_full/`.
- **`icafix`** — run `rfMRI_VERBGEN_AP` (standard ICA+FIX, all volumes) → per-subject `mshbm_output_pre25vol/`, CSVs in `results/network_size_icafix/`.

| Script | Role |
| --- | --- |
| `mshbm/run_subject_mshbm.m` | Run MS-HBM for **one** subject/variant; write per-subject cortical network sizes to `results/network_size_<variant>/<sub>_networksize.csv`. `Variant` ∈ {`full`,`icafix`}; `SkipIfDone` re-exports the CSV without re-fitting. |
| `mshbm/run_group_mshbm.m` | Batch driver: read `dat_verbgen_analysis_144.csv`, loop over all 144 subjects, call `run_subject_mshbm` for each. Args: `Variant`, `SkipIfDone` (resumable), `Exclude` (skip named subjects). |
| `mshbm/combine_network_size.py` | Merge per-subject CSVs into one long table annotated with group (`--variant full|icafix`). |
| `mshbm/plot_group_network_size.py` | Group-level bar charts of network size (mean ± SEM), one per group + a 3-group comparison (`--variant`). |
| `stats/stats_group_network_size.py` | 3-group test per network: Welch ANOVA + Games–Howell (protected), beta-regression LR, Kruskal–Wallis/Dunn; BH-FDR across networks (`--variant`). |

**Usage**

```matlab
% MATLAB — run from the mshbm/ folder (or addpath it) so the two .m files
% resolve each other; both use an absolute ProjectDir internally.
cd mshbm        % or: addpath('mshbm')

% single subject / variant (runs the EM fit, writes its CSV)
run_subject_mshbm('sub-509BT', 'Variant', 'full')

% Whole cohort, one call per variant (resumable: skips subjects already fitted).
% Exclude drops subjects whose fit diverges (e.g. the noise-degenerate icafix ones).
run_group_mshbm('Variant', 'full')
run_group_mshbm('Variant', 'icafix', 'Exclude', {'sub-587BH','sub-616BT'})
```

```bash
# Python — after the MATLAB batch finishes (run from the repo root), per variant
.venv/bin/python mshbm/combine_network_size.py   --variant full   # -> results/network_size_full/group_network_size_long.csv
.venv/bin/python mshbm/plot_group_network_size.py --variant full   # -> group_networksize_{DLD,HSL,TD,compare}.png
.venv/bin/python stats/stats_group_network_size.py --variant full  # -> group_stats_3group.csv (+ group_posthoc_3group.csv)
# ...repeat with --variant icafix
```

**Notes**
- MS-HBM here is **cortex-only** (`Structures = {'CORTEX_LEFT','CORTEX_RIGHT'}`). Network sizes are % of cortical surface; per subject they sum to 100% across the 21 networks (incl. `Noise`, which is excluded from the plots).
- `run_group_mshbm.m` writes a `run_group_mshbm_log.csv` (per-subject success/failure). Stream live progress with `tail -f results/network_size_<variant>/run_group_mshbm_live.log`.
- A handful of `icafix` (all-volume) fits diverge to NaN because of the noisy early volumes (`sub-587BH`, `sub-616BT`); they fit fine in `full` and are dropped from `icafix` via `Exclude`, so `icafix` = 142 and `full` = 144.
- `res0urces/networks_meta.csv` (network id / label / colour, exported from the MS-HBM priors) drives the plot colours and labels.

**Exploratory mood analyses** (in `stats/`, using the SDQ scores; group is the 3-level DLD/HSL/TD factor, TD reference):
- `stats_emotional_salience.py` — group comparison of SDQ emotional symptoms (ANOVA + Welch + Kruskal-Wallis + Games–Howell) and within-group correlation of emotional symptoms with Salience-network size (`--variant`).
- `stats_emotional_salience_nb_interaction.py` — the headline NB group × Salience interaction model with predicted-mean ± 95% CI figure, plus the joint-interaction LR and Freedman–Lane permutation tests (`--variant`).
- `stats_emotional_language_nb_interaction.py` — the same NB group × network-size interaction model for the **Language** network (predicted-mean ± 95% CI, joint-interaction LR + Freedman–Lane permutation; `--variant`).
- `stats_emotional_salience_interaction.py`, `_nonlinear.py`, `_glm.py` — earlier 2-group exploratory variants (OLS/HC3 + bootstrap; non-linear shape w/ LOO-CV; bounded-response NB / beta-binomial) kept for reference; not part of the current 3-group / full-cohort analysis.

---

## Pipeline B — CAB-NP connectivity with an anterior-striatum mask

Extracts and visualises frontostriatal FC, with the Caudate and Putamen restricted to their **anterior (head) portions** to match the anterior striatal foci in Lynch et al. (2024).

| Script | Role |
| --- | --- |
| `connectivity/build_anterior_striatum_atlas.py` | Build `atlas/CABNP_anteriorStriatum_Y4.dlabel.nii` — stock CAB-NP with anterior (precommissural, MNI **Y ≥ 4**) Caudate/Putamen voxels relabelled into single `L/R-Caudate-head` and `L/R-Putamen-head` ROIs. Cortical parcels and NAcc are left identical to stock CAB-NP. |
| `connectivity/1_run_subject_connectivity_extraction_cab-np.sh` | Parcellate a subject's dtseries with the custom atlas (`wb_command`) and export a Fisher-z FC matrix to `derivatives/fc/<sub>_FC.txt`. |
| `connectivity/2_run_subject_connectivity_analysis_cab-np.py` | ROI FC analysis/figures: ACC, anterior insula (AI), lateral PFC (LPFC) cortical zones × NAcc / anterior-Caudate / anterior-Putamen. Takes the subject id as a CLI arg. |
| `connectivity/run_group_connectivity.py` | Batch driver: loop the extraction + analysis over all 144 subjects (from `dat_verbgen_analysis_144.csv`), writing per-subject outputs to `results/connectivity_outputs/<sub>/`. |

**Usage**

```bash
# (run from the repo root)
# 1) Build the custom atlas once (only needed if it doesn't exist / threshold changes)
.venv/bin/python connectivity/build_anterior_striatum_atlas.py

# --- single subject ---
# 2) Extract the FC matrix (uses the custom atlas)
./connectivity/1_run_subject_connectivity_extraction_cab-np.sh sub-509BT
# 3) Analyse / plot ROI FC for that subject
.venv/bin/python connectivity/2_run_subject_connectivity_analysis_cab-np.py sub-509BT

# --- whole cohort (144 subjects) ---
.venv/bin/python connectivity/run_group_connectivity.py                 # re-extract everyone
.venv/bin/python connectivity/run_group_connectivity.py --skip-existing # skip extraction if FC.txt exists
# stream progress:  tail -f results/connectivity_outputs/run_group_connectivity_live.log
```

**Per-subject outputs** (in `results/connectivity_outputs/<sub>/`, suffix `_3_rest_antstri`):
- `01_full_ROI_FC…png` — per-parcel heatmap; `02_collapsed_FC…png` — region×region; `03_frontostriatal_FC…png` — subcortical×cortical.
- `<sub>_02_collapsed_FC…csv` — the fig-2 6×6 region matrix (Pearson r, Fisher-z averaged).
- `<sub>_03_frontostriatal_FC…csv` — the fig-3 subcortical(rows)×cortical(cols) matrix, for group analysis.

**Group-level analysis** (in `stats/`, over the 9 frontostriatal tiles; all testing is done in **Fisher-z** and only displayed means are back-transformed to r):
- `stats/stats_group_connectivity.py` — 3-group (DLD/HSL/TD) per tile: Welch ANOVA omnibus + protected Games–Howell (Hedges g), Kruskal–Wallis/Dunn backup, BH-FDR across tiles. Gathers the per-subject `…_03_frontostriatal…csv` into `group_frontostriatal_long.csv` (no separate combine step), then writes the group-mean panels, the omnibus-F heatmap, and the pairwise-g contrast heatmaps.
- `stats/stats_connectivity_emotional_nb.py` — group × FC interaction predicting SDQ-emotional, one **negative-binomial** model per tile (`emotional ~ C(group, Treatment('TD'))*FCz_c`, 3-level group), matching the salience-size NB model; joint-interaction LR + Freedman–Lane permutation, BH-FDR, interaction heatmaps (per group vs TD) + per-tile NB-predicted scatter with delta-method **95% CI bands**.

```bash
.venv/bin/python stats/stats_group_connectivity.py          # 3-group difference per tile
.venv/bin/python stats/stats_connectivity_emotional_nb.py   # group x FC -> emotional (NB)
```

**Notes**
- "Anterior / head" = precommissural striatum (the anterior commissure is at MNI Y = 0). The 2 mm grid means the effective cut keeps voxels at Y = 4 and forward. See `results/striatum_anterior_mask_check.png` for the threshold rationale.
- Restricting to the anterior head substantially strengthens the Caudate/Putamen frontostriatal edges relative to the whole-structure ROIs, while leaving cortical and NAcc FC unchanged.

---

## Pipeline C — CAB-NP language-network frontal-putamen connectivity

Extracts the single **left language-network frontal-putamen FC edge** highlighted by [Gordon et al. (2021)](https://doi.org/10.1093/cercor/bhab387) (corticostriatal subnetwork #3, the "medial/anterior putamen" component their parcellation places inside the language network): left pars opercularis (Glasser `L_44`) ↔ medial/anterior left putamen. Both nodes are **stock** CAB-NP Language-network parcels, so nothing is carved — but this arm must use the **stock** CAB-NP atlas, *not* the Pipeline-B anterior-striatum atlas (which relabels the precommissural putamen voxels and would cannibalise this parcel).

| Script | Role |
| --- | --- |
| `connectivity/build_language_putamen_rois.py` | QC/definition: identify the two stock CAB-NP Language parcels — cortical `Language-14_L-Ctx` (key 74 = Glasser `L_44`, pars opercularis) and subcortical `Language-14_L-Putamen` (key 6140, medial/anterior left putamen, MNI centroid ≈ −20/+4/+5). Writes `atlas/CABNP_language14_rois.dlabel.nii` (2-ROI mask for `wb_view`) + an MNI-slice check PNG. |
| `connectivity/1_run_subject_language_extraction_cab-np.sh` | Parcellate a subject's rest dtseries with the **stock** CAB-NP atlas (`wb_command`) → 718×718 Fisher-z FC to `derivatives/fc_cabnp_stock/<sub>_FC.txt` (parcel order in `derivatives/cabnp_stock_labels.txt`). |
| `connectivity/2_run_subject_language_analysis_cab-np.py` | Pull the single `L_44` ↔ `Language-14_L-Putamen` edge → `results/language_connectivity_outputs/<sub>/<sub>_language_putamen_FC.csv`. |
| `connectivity/run_group_language_connectivity.py` | Batch driver: extraction + analysis over all 144 subjects. |

**Usage**

```bash
# (run from the repo root)
# 1) Define / QC the two ROIs once (writes the masked dlabel + a slice-check PNG)
../pfm-nsi/.venv/bin/python connectivity/build_language_putamen_rois.py   # needs nibabel

# --- single subject ---
./connectivity/1_run_subject_language_extraction_cab-np.sh sub-509BT
.venv/bin/python connectivity/2_run_subject_language_analysis_cab-np.py sub-509BT

# --- whole cohort (144 subjects) ---
.venv/bin/python connectivity/run_group_language_connectivity.py                 # re-extract everyone
.venv/bin/python connectivity/run_group_language_connectivity.py --skip-existing # skip extraction if FC.txt exists
```

**Group-level analysis** (in `stats/`; a single edge, so no multiple-comparison correction; testing in Fisher-z, means shown in r):
- `stats/stats_group_language_connectivity.py` — 3-group (DLD/HSL/TD) Welch ANOVA on the edge + Kruskal–Wallis, protected Games–Howell (Hedges g) / Dunn; box-and-points figure.
- `stats/stats_language_connectivity_emotional_nb.py` — group × FC interaction predicting SDQ-emotional, one **negative-binomial** model (`emotional ~ C(group, Treatment('TD'))*FCz_c`); joint-interaction LR + Freedman–Lane permutation; scatter with delta-method **95% CI bands** on each group's predicted mean (`…_predband.csv`).

```bash
.venv/bin/python stats/stats_group_language_connectivity.py           # 3-group difference on the edge
.venv/bin/python stats/stats_language_connectivity_emotional_nb.py    # group x FC -> emotional (NB)
```

**Note**
- `build_language_putamen_rois.py` and any other `nibabel` steps use the sibling **pfm-nsi** venv (`../pfm-nsi/.venv/bin/python`); the project `.venv` has no `nibabel`.

---

## Interactive notebooks

`notebooks/` holds notebook versions of the two connectivity arms — one per network, each with three sections (between-group connectivity comparison, group × FC → SDQ-emotional NB, group × network-size → SDQ-emotional NB):
- `notebooks/salience_network_analyses.ipynb`
- `notebooks/language_network_analyses.ipynb`

They are assembled from the validated `stats/` scripts (figures render inline; each cell still writes its PNG/CSV to `results/`). Select the project `.venv` as the kernel.

---

## Repository layout

```
mshbm/         Pipeline A — individual MS-HBM network mapping (MATLAB + Python)
connectivity/  Pipelines B & C — CAB-NP connectivity (anterior-striatum salience + language frontal-putamen)
stats/         group-level + mood statistics; build_analysis_table.py -> the 144-subject master table
notebooks/     interactive notebook versions of the salience & language stats arms
res0urces/     helper functions + CIFTI read/write (from the MSCcodebase)
archived/      superseded scripts / pre-cut results kept for reference
atlas/         custom CAB-NP dlabels (anterior-striatum + language-network ROIs)
results/       per-subject and group outputs (mostly git-ignored; group figures + FINDINGS_summary.md tracked)
.venv/         project virtualenv (git-ignored; see requirements.txt)
```

A written summary of the current full-cohort (n = 144) findings — behavioural, network-size (both variants), and connectivity — is kept in **`results/FINDINGS_summary.md`**.

**_./res0urces/_** — helper functions and CIFTI read/write, from the [MSCcodebase](https://github.com/MidnightScanClub/MSCcodebase).

**_./archived/_** — superseded scripts kept for reference (e.g. the original single-subject `run_subject_mshbm`, and `stats_group_connectivity_emotional.py` — the OLS `FCz ~ group*emotional` flip, dropped in favour of the salience-consistent NB `emotional ~ group*FCz` in `stats/stats_connectivity_emotional_nb.py`).
