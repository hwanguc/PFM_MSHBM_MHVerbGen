# PFM_MSHBM_MHVerbGen

**Author: Han Wang (2025)**

This repository contains the code for precision functional mapping (PFM; [Gordon et al., 2017](https://www.cell.com/neuron/fulltext/S0896-6273(17)30613-X)) and functional connectivity (FC) analyses for data from [Krishnan et al. (2021)](https://www.sciencedirect.com/science/article/pii/S1053811920310843). The PFM is based on a multi-session hierarchical Bayesian model (MS-HBM) pipeline developed by [Kong et al. (2019)](https://academic.oup.com/cercor/article/29/6/2533/5033556?login=false). The pipeline maps brain functional networks in individual participants whose resting-state data has been pre-processed using the [Human Connectome Project (HCP) pipeline](https://github.com/Washington-University/HCPpipelines) with denoising via [ICA-FIX](https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/fix.html).

The whole-brain FC analysis uses the parcellation defined by the Cole-Anticevic Brain Network Atlas (CAB-NP; [Ji et al., 2019](https://www.sciencedirect.com/science/article/abs/pii/S1053811918319657); [GitHub repo](https://github.com/ColeLab/ColeAnticevicNetPartition)). Region-of-interest (ROI) FC analysis uses nodes defined on the functional cortical atlas from [Glasser et al. (2016)](https://www.nature.com/articles/nature18933), which underpins the cortical parcels in Ji et al. (2019).

The motivating question is the frontostriatal salience network and its relationship to mood, following the salience-network expansion in depression reported by [Lynch et al. (2024)](https://www.nature.com/articles/s41586-024-07805-2).

---

## Cohort & data

Subjects are the verb-generation sub-sample defined in `dat_verbgen_scqsdq_subsample.xlsx` (`code` column; group suffixes: **BL = DLD**, **BT = TD/control**, **BH = HSL**). The DLD-vs-control analyses use all `BL` + `BT` subjects, excluding `513BT` (failed QC, replaced by `675BT`) = 16 DLD + 20 TD.

Pre-processed subject data lives at `/media/hanwang/Data/Data/ucl/gos_ich/verb_gen_krishnan/processed/<sub-CODE>/` (folder name = `sub-` + spreadsheet code). Update the `BaseDir`/`DTSERIES` paths at the top of the scripts if the data moves.

**Dependencies:** MATLAB (tested R2025a) with the [CBIG](https://github.com/ThomasYeoLab/CBIG) functions on the path (see `Paths{3}` in `run_subject_mshbm.m`); [Connectome Workbench](https://www.humanconnectome.org/software/connectome-workbench) (`wb_command`); Python 3 with `nibabel`, `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `openpyxl`.

---

## Pipeline A — Group-level MS-HBM individual network mapping

Maps individual functional networks per subject and aggregates cortical network sizes to the group level.

| Script | Role |
| --- | --- |
| `mshbm/run_subject_mshbm.m` | Run MS-HBM for **one** subject; write per-subject cortical network sizes to `results/network_size/<sub>_networksize.csv`. |
| `mshbm/run_group_mshbm.m` | Batch driver: read the spreadsheet, loop over all BL+BT subjects (excl. 513BT), call `run_subject_mshbm` for each. |
| `mshbm/combine_network_size.py` | Merge per-subject CSVs into one long table annotated with group. |
| `mshbm/plot_group_network_size.py` | Group-level bar charts of network size (mean ± SEM), one per group. |
| `stats/stats_group_network_size.py` | DLD-vs-TD test per network (Welch t, Mann-Whitney, beta regression; BH-FDR corrected). |

**Usage**

```matlab
% MATLAB — run from the mshbm/ folder (or addpath it) so the two .m files
% resolve each other; both use an absolute ProjectDir internally.
cd mshbm        % or: addpath('mshbm')

% single subject (runs the EM fit, writes its CSV)
run_subject_mshbm('sub-509BT')

% Re-export the CSV from an existing fit without re-running EM
run_subject_mshbm('sub-509BT', 'SkipIfDone', true)

% Whole BL+BT cohort (resumable: skips subjects already fitted)
run_group_mshbm
```

```bash
# Python — after the MATLAB batch finishes (run from the repo root)
python3 mshbm/combine_network_size.py     # -> results/network_size/group_network_size_long.csv
python3 mshbm/plot_group_network_size.py  # -> results/network_size/group_networksize_{DLD,TD}.png
python3 stats/stats_group_network_size.py # -> results/network_size/group_stats_DLD_vs_TD.csv
```

**Notes**
- MS-HBM here is **cortex-only** (`Structures = {'CORTEX_LEFT','CORTEX_RIGHT'}`). Network sizes are % of cortical surface; per subject they sum to 100% across the 21 networks (incl. `Noise`, which is excluded from the plots).
- `run_group_mshbm.m` writes a `run_group_mshbm_log.csv` (per-subject success/failure). Stream live progress with `tail -f results/network_size/run_group_mshbm_live.log`.
- `res0urces/networks_meta.csv` (network id / label / colour, exported from the MS-HBM priors) drives the plot colours and labels.

**Exploratory mood analyses** (in `stats/`, using the SDQ scores in the spreadsheet):
- `stats_emotional_salience.py` — group comparison of SDQ emotional symptoms (ANOVA + Welch + Kruskal-Wallis + Tukey) and within-group correlation of emotional symptoms with Salience-network size.
- `stats_emotional_salience_interaction.py` — formal group × Salience-size interaction (OLS w/ HC3, Fisher r-to-z, bootstrap).
- `stats_emotional_salience_nonlinear.py` — non-linear shape exploration (linear vs quadratic vs exponential vs LOESS) with leave-one-out CV.
- `stats_emotional_salience_glm.py` — bounded-response models that respect the SDQ floor (negative binomial; beta-binomial for the 0–10 ceiling), per group and as an interaction.
- `stats_emotional_salience_nb_interaction.py` — the headline NB group × Salience interaction model with predicted-mean ± 95% CI figure, plus LR and permutation tests of the interaction.

---

## Pipeline B — CAB-NP connectivity with an anterior-striatum mask

Extracts and visualises frontostriatal FC, with the Caudate and Putamen restricted to their **anterior (head) portions** to match the anterior striatal foci in Lynch et al. (2024).

| Script | Role |
| --- | --- |
| `connectivity/build_anterior_striatum_atlas.py` | Build `atlas/CABNP_anteriorStriatum_Y4.dlabel.nii` — stock CAB-NP with anterior (precommissural, MNI **Y ≥ 4**) Caudate/Putamen voxels relabelled into single `L/R-Caudate-head` and `L/R-Putamen-head` ROIs. Cortical parcels and NAcc are left identical to stock CAB-NP. |
| `connectivity/1_run_subject_connectivity_extraction_cab-np.sh` | Parcellate a subject's dtseries with the custom atlas (`wb_command`) and export a Fisher-z FC matrix to `derivatives/fc/<sub>_FC.txt`. |
| `connectivity/2_run_subject_connectivity_analysis_cab-np.py` | ROI FC analysis/figures: ACC, anterior insula (AI), lateral PFC (LPFC) cortical zones × NAcc / anterior-Caudate / anterior-Putamen. Takes the subject id as a CLI arg. |
| `connectivity/run_group_connectivity.py` | Batch driver: loop the extraction + analysis over all BL+BT subjects (excl. 513BT), writing per-subject outputs to `results/connectivity_outputs/<sub>/`. |

**Usage**

```bash
# (run from the repo root)
# 1) Build the custom atlas once (only needed if it doesn't exist / threshold changes)
python3 connectivity/build_anterior_striatum_atlas.py

# --- single subject ---
# 2) Extract the FC matrix (uses the custom atlas)
./connectivity/1_run_subject_connectivity_extraction_cab-np.sh sub-509BT
# 3) Analyse / plot ROI FC for that subject
python3 connectivity/2_run_subject_connectivity_analysis_cab-np.py sub-509BT

# --- whole BL+BT cohort (36 subjects, ~3 min) ---
python3 connectivity/run_group_connectivity.py                 # re-extract everyone
python3 connectivity/run_group_connectivity.py --skip-existing # skip extraction if FC.txt exists
# stream progress:  tail -f results/connectivity_outputs/run_group_connectivity_live.log
```

**Per-subject outputs** (in `results/connectivity_outputs/<sub>/`, suffix `_3_rest_antstri`):
- `01_full_ROI_FC…png` — per-parcel heatmap; `02_collapsed_FC…png` — region×region; `03_frontostriatal_FC…png` — subcortical×cortical.
- `<sub>_02_collapsed_FC…csv` — the fig-2 6×6 region matrix (Pearson r, Fisher-z averaged).
- `<sub>_03_frontostriatal_FC…csv` — the fig-3 subcortical(rows)×cortical(cols) matrix, for group analysis.

**Group-level analysis** (in `stats/`, over the 9 frontostriatal tiles; all testing is done in **Fisher-z** and only displayed means are back-transformed to r):
- `stats/stats_group_connectivity.py` — DLD-vs-TD per tile (Welch t on Fisher-z, Mann-Whitney, Cohen's d, BH-FDR). Gathers the per-subject `…_03_frontostriatal…csv` into `group_frontostriatal_long.csv` (so no separate combine step), then writes group-mean and DLD−TD t-stat heatmaps.
- `stats/stats_connectivity_emotional_nb.py` — group × FC interaction predicting SDQ-emotional, one **negative-binomial** model per tile (`emotional ~ C(group)*FCz_c`), matching the salience-size NB model; LR-tested interaction, BH-FDR, interaction heatmap + per-tile NB-predicted scatter.

```bash
python3 stats/stats_group_connectivity.py          # group difference per tile
python3 stats/stats_connectivity_emotional_nb.py   # group x FC -> emotional (NB)
```

**Notes**
- "Anterior / head" = precommissural striatum (the anterior commissure is at MNI Y = 0). The 2 mm grid means the effective cut keeps voxels at Y = 4 and forward. See `results/striatum_anterior_mask_check.png` for the threshold rationale.
- Restricting to the anterior head substantially strengthens the Caudate/Putamen frontostriatal edges relative to the whole-structure ROIs, while leaving cortical and NAcc FC unchanged.

---

## Repository layout

```
mshbm/         Pipeline A — individual MS-HBM network mapping (MATLAB + Python)
connectivity/  Pipeline B — CAB-NP anterior-striatum connectivity
stats/         group-level and exploratory mood statistics
res0urces/     helper functions + CIFTI read/write (from the MSCcodebase)
archived/      superseded scripts kept for reference
atlas/         custom CAB-NP anterior-striatum dlabel
results/       per-subject and group outputs (mostly git-ignored)
```

**_./res0urces/_** — helper functions and CIFTI read/write, from the [MSCcodebase](https://github.com/MidnightScanClub/MSCcodebase).

**_./archived/_** — superseded scripts kept for reference (e.g. the original single-subject `run_subject_mshbm`, and `stats_group_connectivity_emotional.py` — the OLS `FCz ~ group*emotional` flip, dropped in favour of the salience-consistent NB `emotional ~ group*FCz` in `stats/stats_connectivity_emotional_nb.py`).
