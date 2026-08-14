# Code for "The Jensen Gaps of Directional Diffusion"

Analysis and figure-generation code for *The Jensen Gaps of Directional
Diffusion: Model-Free Microstructural White Matter Markers of Aging*.

The logarithmic Jensen gap measures directional diffusion nonuniformity in two
domains — the diffusivity-domain gap `J_ln = ln(ADC_1/ADC_0)` (arithmetic vs
geometric mean of the per-direction diffusivities) and the signal-domain gap
`J_ln^S = ln<S'> - <ln S'>` — directly from a multi-direction acquisition, with
no tensor fit. This repository contains every script used to compute the gaps
and produce the figures and reported statistics, starting from the public HCP-A
data.

## Repository layout
```
gap_computation/   raw DWI -> directional diffusivities -> Holder means -> gaps
tbss_pipeline/      voxelwise TBSS (gap maps -> skeleton -> randomise; full + agesex designs)
figure_scripts/     one script per manuscript figure
analysis/           reported statistics
```

## Setup

### 1. Get the code
```
git clone https://github.com/snhwang/paper-jensen-gap-code.git
cd paper-jensen-gap-code
```

### 2. Python environment (for gap computation, figures, statistics)
Python 3.11+ is recommended.
```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # numpy, pandas, scipy, statsmodels, matplotlib, nibabel
```

### 3. Install FSL (for preprocessing and the TBSS pipeline)
The gap computation, figures, and statistics are pure Python. Steps 1 and 4
(atlas registration and voxelwise TBSS) additionally require **FSL**, the FMRIB
Software Library. Install it from the official instructions —
<https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation> — and confirm that
`randomise`, `fslmaths`, `applywarp`, and `tbss_*` are on your `PATH`.

### 4. Obtain the dataset (HCP-A / AABC, access-restricted)
The HCP-Aging (HCA) / Adult Aging Brain Connectome (AABC) diffusion data are
**not** redistributed here. They are obtained from **AABC Release 2** via the
BALSA repository, after registering as a qualified researcher and agreeing to
the AABC Data Use Terms:
- BALSA data repository (download hub): <https://balsa.wustl.edu>
- AABC consortium (access + Data Use Terms): <https://agingadultbrainconnectome.wustl.edu>
- Study / release overview: <https://www.humanconnectome.org/study/hcp-lifespan-aging/data-releases>

See Step 0 below for exactly which files are needed and how to lay them out.

## End-to-end reproduction

### Step 0 — Obtain the data (HCP-A / AABC, access-restricted)
No data are redistributed here. The diffusion images come from **AABC Release 2**
of the HCP-Aging (HCA) / Adult Aging Brain Connectome (AABC) study (cross-sectional
V1 plus longitudinal V2--V4 visits, ages 36--90+), distributed through the **BALSA**
repository under the AABC Data Use Terms:
- Study / release overview: <https://www.humanconnectome.org/study/hcp-lifespan-aging/data-releases>
- BALSA repository (the download hub): <https://balsa.wustl.edu>
- AABC consortium (access details + Data Use Terms): <https://agingadultbrainconnectome.wustl.edu>

Access requires certifying that you are a qualified researcher at an academic,
non-profit, or government institution, registering on BALSA with a valid
institutional email, and agreeing to use the data only for non-commercial
research. Once approved, download from BALSA:
- the **preprocessed diffusion** "HCP package" (eddy/motion/distortion-corrected
  DWI, `bvals`/`bvecs`, and brain mask) for the HCA/AABC imaging visits, and
- the per-visit **non-imaging demographics** (`AABC2_subjects_*.csv`, providing
  `age_open` and `sex`).

Lay these out as described in Step 1 below.

### Step 1 — Per-session preprocessing (standard FSL/HCP, not in this repo)
For each session, lay out a directory `$DTI_OUTPUT_DIR/<session_id>/` with:
```
inputs/dwi.nii.gz, inputs/dwi.bval, inputs/dwi.bvec, inputs/nodif_brain_mask.nii.gz
processed/atlas/jhu_labels_registered.nii.gz
```
The atlas labels are the JHU ICBM-DTI-81 atlas registered into each session's
native diffusion space: register the subject FA to `FMRIB58_FA_1mm` with FSL
FLIRT (affine) then FNIRT (nonlinear), and carry the JHU labels back through the
inverse warp (nearest-neighbour). This uses standard FSL tools only.

### Step 2 — Build the cohort manifest
```
python gap_computation/build_n1379_manifest.py        # -> HCP/manifest_n1379_b{1500,3000}.tsv
```
One representative session per subject, with age and sex (N = 1,379).

### Step 3 — Compute the gaps (the method)
```
python gap_computation/batch_gaps_hcp.py              # -> HCP/cr_gaps_long.csv
python gap_computation/gap_voxel_maps.py <session> <bval>   # -> per-session voxel maps (Fig 1)
```
`batch_gaps_hcp.py` forms `D_i = -(1/b) ln(S_i/S_0)` per direction, computes the
Holder means (`ADC_0,1,2,...` and `M*_S`) and the Jensen gaps in both domains,
and averages over the 16 JHU ROIs into `cr_gaps_long.csv` (one row per
session x shell x ROI) — the table every figure/stat script reads.
`gap_voxel_maps.py` writes the same quantities as voxelwise NIfTI maps for one
session (used for Figure 1 and as the per-subject input to TBSS).

### Step 3b — Attenuation gap `J_f` (Section 6.3, Fig 8)
```
python gap_computation/batch_jf_hcp.py                # -> HCP/cr_jf_long.csv
```
`J_f = ln<f> - <ln f>` over the per-direction attenuation ratios
`f_i = D_i(3000)/D_i(1500)`. It tests the same b-flatness null as
`Delta_b J_ln` but contrasts each direction before averaging rather than
differencing two aggregated gaps, which is what lets it survive voxelwise.

This needs both shells at once, so unlike `batch_gaps_hcp.py` it cannot run per
shell. The two shells do **not** share a gradient ordering: index-paired vectors
sit about 67 degrees apart. Directions are paired by mutual nearest neighbour in
`|cos|` and only near-exact reciprocal pairs are kept, giving 68 of 93 for the
HCP-A scheme. Sessions whose schemes cannot be matched are reported and skipped
(17 of 2,759).

### Step 4 — Voxelwise TBSS (FSL; for Figs 5-6)
Generate per-subject native gap maps, carry them into FMRIB58 standard space,
run standard TBSS to build the skeletonized 4D, then randomise:
```
# 1. per-subject native J_ln_D / J_dir maps (same math as batch_gaps_hcp.py):
python gap_computation/gap_voxel_maps.py <session> <bval>
# 2. warp each map to standard space with that subject's FA->FMRIB58 warp
#    (FSL applywarp), then run standard TBSS (tbss_1_preproc ... tbss_skeleton)
#    to produce  tbss_age_b{bval}_n1379/stats/all_{J_ln_D,J_dir,FA}_skeletonised.nii.gz
# 3. sex-adjusted "agesex" design (the reported figures) + FWE randomise:
python tbss_pipeline/build_design_agesex.py 1500          # design_agesex_b{bval}.txt / contrast_agesex.txt
Text2Vest HCP/design_agesex_b1500.txt HCP/design_agesex_b1500.mat   # FSL: text -> .mat/.con
Text2Vest HCP/contrast_agesex.txt     HCP/design_agesex.con
bash tbss_pipeline/_deploy_agesex.sh 1500 <host>          # ship 4D + design (run in WSL)
bash tbss_pipeline/dgx_randomise_agesex_n1379.sh 1500     # randomise: 5000 perms, TFCE, FWE
# 4. (robustness) FULL 10-EV confound-adjusted design; maps near-identical (Dice 0.98-1.00):
python analysis/motion_robustness.py                      # writes HCP/quality_{,b3000_}n1379.csv
python tbss_pipeline/build_design_full.py                 # design_full_b{bval}.txt / contrast_full.txt
Text2Vest HCP/design_full_b1500.txt HCP/design_full_b1500.mat
Text2Vest HCP/contrast_full.txt     HCP/contrast_full.con
bash tbss_pipeline/_deploy_full.sh 1500 <host>
bash tbss_pipeline/dgx_randomise_full.sh 1500 --parallel
```
The reported figures use the sex-adjusted "agesex" design (age + age^2 + sex +
age*sex); the contrast (tstat1 = +age, tstat2 = -age) is the sex-adjusted
linear-age effect, and `build_dual_gap_tbss_panel.py` reads the resulting
`{J_ln_D,J_dir,FA}_agesex_tfce_corrp_tstat{1,2}.nii.gz`. As a **robustness** check
the full design appends acquisition site (3 dummies), head motion, and the eddy
outlier fraction (10 EVs, matching the ROI covariates); the age maps are
near-identical (Dice 0.98-1.00 vs the `*_full_*` outputs). Because the b1500 and
b3000 manifests select different visits for 275 subjects, `build_design_full.py`
reads each shell's motion/outlier from its own quality file. These steps require
an FSL install (`applywarp`, `tbss_*`, `randomise`, `Text2Vest`, `fslmaths`). The
lab run parallelized steps 1-2 across a compute cluster; that orchestration is
not included here, but the operations are standard FSL applied to the
`gap_voxel_maps.py` outputs.

```
# 4a. tract coverage/bias screen -> writes skeleton_screened_mask.nii.gz, the
#     inference mask + reporting denominator used by the TBSS figures:
python tbss_pipeline/screen_tracts.py ~/tbss_age_b1500_n1379 1500
python tbss_pipeline/screen_tracts.py ~/tbss_age_b3000_n1379 3000

# 4b. (beyond-tensor) does an age effect survive removing FA/MD/AD/RD voxelwise?
#    residualize the metric on [1,FA,MD,AD,RD] per voxel, then randomise the age
#    contrast on the residual (full 10-EV design, TFCE, over skeleton_good_mask):
NPERM=5000 bash tbss_pipeline/dgx_randomise_beyondtensor.sh 1500 J_ln_D   # diffusivity gap
NPERM=5000 bash tbss_pipeline/dgx_randomise_beyondtensor.sh 1500 J_dir    # signal gap (= J_ln_S)
# -> jlnd_rig_/jdir_rig_ tfce_corrp maps; localize_beyondtensor.py prints the
#    FWE voxel counts, % of usable skeleton, survivor partial-r, and JHU labels.
```
This is the test behind the Limitations statement that the gaps' aging signal is
largely shared with the tensor. `make_resid_tensor.py` does the per-voxel
regression (FA/MD/AD/RD are collinear by construction, so `ZtZ` carries a small
ridge); the same test is applied to the info-theory descriptors in the companion
paper using the identical `skeleton_good_mask.nii.gz`, so results are directly
comparable across the two papers. Requires the tensor scalars
(`all_{FA,MD,AD,RD}_skeletonised.nii.gz`) alongside the gap 4D in the TBSS stats
dir; both come from the standard FSL DTI fit.

### Step 5 — Figures
```
python figure_scripts/plot_jensen_gap_panel.py      # Fig 1  (voxel maps)
python figure_scripts/build_gap_spectrum.py         # Fig 2  (rung-pair spectrum)
python figure_scripts/build_gap_multidim.py         # Fig 3 (heatmap), Fig 4 (SCR scatter)
python figure_scripts/build_dual_gap_tbss_panel.py  # Figs 5, 6 (TBSS: signal gap, diffusivity gap, FA)
python figure_scripts/build_gap_beyondtensor.py     # Fig 7  (beyond-tensor TBSS; needs step 4b maps)
python figure_scripts/build_aging_master.py         # Fig 8  (b-signature, lifespan, longitudinal)
python figure_scripts/build_gaussianity_map_figure.py  # Fig 8 (voxelwise J_f, Delta_b, DKI)
```
`build_gaussianity_map_figure.py` recomputes `J_f` from the raw DWI using
`batch_jf_hcp.py`'s own constants and direction matching, so the published map
cannot drift from the cohort table. It also prints the partial-volume control
(`J_f` against ADC deciles and FA strata) and the DKI agreement quoted in
Section 10.2. It expects DKI maps for the example session in
`HCP/voxel_maps/`.

### Step 6 — Reported statistics
```
python analysis/headline_significance.py            # cross-sectional + spectrum significance
python analysis/gap_fa_correlation.py               # gap vs FA, spatial + across-subject
python analysis/jf_age_association.py               # J_f vs age, 16 ROIs, vs Delta_b J_ln
python analysis/jf_split_half_reliability.py        # voxelwise reliability of both indices
JG_DATA=/path/to/tree python analysis/reproduce_stats.py   # SELF-TEST: verify every scalar vs the paper
JG_DATA=/path/to/tree HCPA_ZIP_GLOB="H:/HCA*_DiffusionRecommended.zip" \
  python analysis/motion_robustness.py                     # Sec 7.1 motion+site+outlier robustness, both shells
                                                            #   (reads the HCP-A zips; also writes HCP/quality_{,b3000_}n1379.csv)
JG_DATA=/path/to/tree python analysis/tbss_map_stats.py    # Sec 7.4 TBSS scalars from the randomise maps:
                                                            #   Dice(agesex,full) robustness + corona-radiata pos:neg ratios
                                                            #   (CR ratios need the FSL JHU atlas; set JHU_ATLAS if $FSLDIR is unset)
```
`reproduce_stats.py` is a **self-test**: it recomputes every ROI-level scalar
quoted in the manuscript (cross-sectional, FA, within-sex, age x sex interaction,
spectrum, longitudinal slopes, b-signature, gap non-Gaussianity), prints each as
`[OK]`/`[FAIL]` against the paper value, and **exits non-zero if any check fails**
-- so it doubles as a regression gate. The 56 "90 or older" participants are coded
as age 90 via `clean_age()` and retained (N = 1,379), matching the paper.
`tbss_map_stats.py` covers the map-derived scalars: the full-vs-sex-adjusted Dice
(0.98-1.00) and the corona-radiata positive-to-negative voxel ratios (signal gap
~1.2:1, FA ~0.45:1 at b=1500).

## Paths and data root
The figure and gap-computation scripts run from the analysis tree and use paths
relative to it: `HCP/` for the metric tables and `tbss_age_b{b}_n1379/stats/` for
the TBSS outputs. Figures are written to `figures/` (and also copied to a sibling
`paper-jensen-gap/figures/` only if that directory happens to exist). Either place
your data under that layout or edit the `BASE`/path constants at the top of each
script.

The `analysis/` scripts instead take the data root from the **`JG_DATA`**
environment variable (`reproduce_stats.py`, `headline_significance.py`,
`gap_fa_correlation.py`, `motion_robustness.py`, `tbss_map_stats.py`,
`jf_age_association.py`, `jf_split_half_reliability.py`). `JG_DATA` is also
honoured by `figure_scripts/build_gaussianity_map_figure.py`, which additionally
reads the raw DWI through `DTI_OUTPUT_DIR`.
`motion_robustness.py` additionally uses `HCPA_ZIP_GLOB` (the raw HCP-A eddy-log
zips), and `tbss_map_stats.py` uses `JHU_ATLAS` (the JHU corona-radiata atlas;
defaults to the one in `$FSLDIR`).

## Statistical conventions
- Reported age associations are **sex-adjusted** primary: partial correlation
  controlling for sex (cross-sectional, spectrum), sex as a covariate (mixed
  model), or the sex-adjusted linear-age contrast (voxelwise TBSS, Figs 5-6). A
  full-covariate robustness check (adding acquisition site + head motion + eddy
  outlier fraction) shifts the ROI correlations by at most 0.013 at both shells
  (`analysis/motion_robustness.py`) and leaves the TBSS maps near-identical (Dice
  0.98-1.00; `tbss_pipeline/build_design_full.py`).
- Cross-sectional and longitudinal analyses use the **voxelwise-gap mean** per
  ROI; the spectrum heatmap (Figs 2-3) uses the **log-ratio of the ROI-mean
  Holder means**. The two differ by Jensen's inequality, so one is fixed within
  each analysis.
- **Age:** participant age is from the AABC demographics; the HIPAA "90 or older"
  top-code (n = 56) is coded as **90 and retained**, so every analysis runs on the
  same **N = 1,379** cohort. This is centralized in `clean_age()`
  (`analysis/reproduce_stats.py`); reading the raw string age instead would coerce
  those 56 to NaN and silently drop them.

## Notes
This is a curated, code-only snapshot for transparency. The scripts are provided
to document the exact computation; the input data are access-restricted, and
Steps 1 and 4 assume a standard FSL/HCP preprocessing environment and (for TBSS)
a compute cluster.

## License
The code in this repository is released under the MIT License (see `LICENSE`).
The HCP-A / AABC input data are **not** covered by this license and remain
subject to the AABC Data Use Terms.
