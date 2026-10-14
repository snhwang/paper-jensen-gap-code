# Analysis code for the revised manuscript

Every number, figure and table of the revised manuscript and its supplement is
produced by a script in this folder. This file says which script, and from what.

## Quick start

```bash
cd revision
python run_all.py --list     # the steps, in dependency order
python run_all.py            # rebuild everything except the slow native passes
python run_all.py --heavy    # include the native passes (about 5 h, disk-bound)
```

`run_all.py` stops at the first failing step. `--from NAME` resumes from a step.

## Inputs

The scripts read data that the original pipeline produced. They are not in this
repository and must not be, under the AABC/HCP-A Data Use Terms. The pipeline
that produces them is in the other folders of this repository (see the
top-level README).

| Input | Where (`config.py`) | Produced by |
|---|---|---|
| Diffusion data, AABC Release 2 | `JG_DTI`/session_*/ (`inputs/dwi.bval`, `processed/dwi_raw.nii.gz`, `dwi_raw.bvec`, `mask.nii.gz`, `atlas/jhu_labels_registered.nii.gz`) | BALSA download, minimally preprocessed; JHU labels registered by the original pipeline (FSL FLIRT + FNIRT, inverse warp) |
| Per-session ROI means of the gaps and Hölder means | `JG_HCP`/cr_gaps_long.csv | `../gap_computation/batch_gaps_hcp.py` |
| Download zips of the first visits (eddy output inside, for head motion and outliers) | `JG_ZIPS`/<Subject>_V1_MR_DiffusionRecommended.zip | BALSA download |
| J_f per session (paired directions, earlier preprint) | `JG_HCP`/cr_jf_long.csv | `../gap_computation/batch_jf_hcp.py` |
| Demographics (age, sex, site, scanner, dMRI flag) | `JG_HCP`/AABC2_subjects_*.csv | BALSA |
| Example participant, original maps (J_f) | `JG_OLDMAPS` | `../figure_scripts/build_gaussianity_map_figure.py` |
| JHU labels and FMRIB58 template (1 mm) | `JG_ATLAS` | FSL |

All paths live in `config.py` and can be overridden with environment variables
(`JG_DATA`, `JG_HCP`, `JG_DTI`, `JG_MAPS`, `JG_OLDMAPS`, `JG_ATLAS`, `JG_ZIPS`).
By default they are looked for under `../data/` (`HCP/`, `dti_output/`,
`atlases/`, `zips/`).

Software used: Python 3.14, numpy, pandas, scipy, statsmodels, nibabel,
DIPY 1.12 (tensor fits, spherical harmonics) and matplotlib.

## Outputs

| Folder | Content | In git? |
|---|---|---|
| `out/` | intermediate tables and text summaries | no |
| `figures/` | main-text figures (PDF, with PNG previews) | no |
| `figures_supp/` | supplementary figures | no |
| `tables_supp/` | LaTeX tables of the supplement | no |

Figure PDFs are reproducible byte for byte. `config.py` fixes `SOURCE_DATE_EPOCH`.

## Steps

| Step | Script | Reads | Writes | Time |
|---|---|---|---|---|
| cohort | `v1_roi_stats.py` | cr_gaps_long, demographics, b-values | `v1_cohort_sessions.csv`, `A_cross_sectional.csv`, `B_decomposition.csv`, `C_bdependence.csv`, `E_spectrum_voxelwise.csv`, `summary.txt` | 1 min |
| motion | `motion_covariates.py` | download zips of the cohort (two eddy text files each) | `motion.csv` | 30 s |
| native | `native_pass.py` | raw DWI of the cohort | `native_pass.csv` | 2 h |
| post_native | `post_native.py` | `native_pass.csv` | `fa_correlations.csv`, `beyond_tensor.csv`, `bdep_snr.csv`, `noise_corrected.csv`, `post_native_summary.txt` | 1 min |
| long_sessions | `long_sessions.py` | cr_gaps_long, demographics, b-values | `long_sessions.csv` (with `days_from_V1`, the exact visit date offset), `long_todo_sessions.csv` | 1 min |
| native_long | `native_pass.py --sessions out/long_todo_sessions.csv --tag long` | raw DWI | `native_pass_long.csv` | 2 h |
| longitudinal | `longitudinal.py` | `long_sessions.csv`, `native_pass*.csv` | `longitudinal.csv`, `longitudinal.txt` | 1 min |
| jf_pass | `jf_pass.py --sample 300` | raw DWI of 300 random cohort sessions | `jf_pass.csv` | 30 min |
| jf_stats | `jf_stats.py` | `jf_pass.csv`, `C_bdependence.csv` | `jf_stats.csv`, `jf_stats.txt` | seconds |
| example_maps | `example_maps.py` | raw DWI of the example session | voxel maps in `JG_MAPS` | 30 s |
| example_stats | `example_stats.py` | example maps, b-vectors | `example_stats.txt` | seconds |
| check_pairing | `check_pairing.py` | example DWI and maps | `check_pairing.txt` | 30 s |
| gminmax_noise | `gminmax_noise.py` | example DWI | `gminmax_noise.txt` | 30 s |
| theory_table | `theory_table.py` | none (simulation) | `tables_supp/tab_accuracy.tex` | seconds |
| sensitivity | `sensitivity.py` | cr_gaps_long, `motion.csv`, `native_pass.csv` | `tables_supp/tab_sens.tex`, `sensitivity.csv`, `sensitivity.txt` | 1 min |
| fig1 to fig5_6 | `fig1_schematic.py`, `fig2_maps.py`, `fig3_fa_relation.py`, `fig_heatmaps.py`, `fig_region.py` | see each script's docstring | `figures/`, `figures_supp/`, `fig4_age_correlations.csv`, `fig5_decomposition.csv` | 1 min |
| supp_tables | `supp_tables.py` | `fig4_age_correlations.csv`, `beyond_tensor.csv`, `C_bdependence.csv`, `noise_corrected.csv`, `jf_stats.csv` | `tables_supp/tab_age.tex`, `tab_beyond.tex`, `tab_bdep.tex`, `tab_jf.tex` | seconds |

The native passes write a `*_partial.csv` as they go and resume from it if
interrupted. Delete it to start over.

`check_heavy_outputs.py [N]` reruns the first N sessions (default 3) of each
stored slow output (`native_pass.csv`, `native_pass_long.csv`, `jf_pass.csv`)
with the current scripts and compares every value. Run it after editing
`native_pass.py` or `jf_pass.py`, to decide whether the slow passes must be
rerun. It takes under a minute.

## Where each result comes from

### Main text

| Item | Source |
|---|---|
| Section 3.1: release size (1,396 participants, 2,878 visits, 1,379 with dMRI), N = 1,354, age, sex, sites, HCP-A/AABC split | `v1_roi_stats.py` → `summary.txt` |
| Section 3.1: eigenvalues of M (0.341, 0.334, 0.325; 0.341, 0.331, 0.329) | `example_stats.py` → `example_stats.txt` |
| Section 3.1: ADC_1 within 1% of MD (median 0.991) | `post_native.py` → `post_native_summary.txt` |
| Section 3.5.2: simulation over-predicts in 63 of 64 cells, median 3% (J_ln, b = 1500) and 12% (b = 3000), the "tensor + noise (DW residual)" excess rows | `post_native.py` → `post_native_summary.txt` |
| Section 4.1: ADC_0, D_SM about 5% below ADC_1; ρ(J_ln, FA) 0.994/0.991; ρ = 0.998 and ratio 0.906 for the signal gap; 30,878 voxels | `example_stats.py` → `example_stats.txt` |
| Section 4.1: ρ with FA across participants (0.83 to 0.98; 0.30 to 0.92) | `post_native.py` → `fa_correlations.csv` |
| Section 4.2: r with age of the gaps, FA and MD; 180 of 192 significant | `fig_heatmaps.py main` → `fig4_age_correlations.csv` (same values in `A_cross_sectional.csv`) |
| Section 4.2: noise inflation 11–12% and 16–17%; SNR vs age r = +0.64; noise-corrected r | `post_native.py` → `post_native_summary.txt`, `noise_corrected.csv` |
| Section 4.2: covariates (at most 0.011), top-coded ages (at most 0.001) | `motion_covariates.py`, then `sensitivity.py` → `sensitivity.csv`, `sensitivity.txt` |
| Section 4.3: MD and anisotropy terms, 13/16 and 15/16 signs, r = 0.90/0.96 | `v1_roi_stats.py` → `B_decomposition.csv`; `fig_region.py` → `fig5_decomposition.csv` |
| Section 4.4: J_ln lower at b = 3000 (12–30%, >99% of participants), Δ_b vs age | `v1_roi_stats.py` → `C_bdependence.csv` |
| Section 4.4: noise-corrected change (14–31%) and Δ_b vs age; SNR-covariate r = +0.11 | `post_native.py` → `noise_corrected.csv`, `bdep_snr.csv` |
| Section 4.5: within-person rates, measured and noise-corrected; participants, sessions, follow-up, scanner changes. Time is the exact interval from `days_from_V1`, not the released whole-year ages | `longitudinal.py` → `longitudinal.txt` |
| Section 4.6: uncorrected excess 7.5%, r = −0.60, SNR covariate r = −0.01; residual r ranges with both noise estimates | `post_native.py` → `beyond_tensor.csv`, `post_native_summary.txt` |
| Figure 1 | `fig1_schematic.py` (theory only, no data) |
| Figure 2 | `example_maps.py`, then `fig2_maps.py` |
| Figure 3 | `example_maps.py`, `post_native.py`, then `fig3_fa_relation.py` |
| Figure 4 | `fig_heatmaps.py main` |
| Figure 5 | `fig_region.py` |
| Figure 6 | `fig_region.py` (panels c, d use `longitudinal.csv`) |
| Table 1 | theory, no data |

### Supplement

| Item | Source |
|---|---|
| S1 and Table S1 | `theory_table.py` → `tab_accuracy.tex` |
| S2 and Figure S1 | `v1_roi_stats.py` (`E_spectrum_voxelwise.csv`), then `fig_heatmaps.py spectrum` |
| S2: earlier-preprint values (−0.600 vs −0.472; +0.354 vs +0.301) | the submitted analysis: `../analysis/reproduce_stats.py` |
| S2: G_min,max noise share (20% vs 12%) | `gminmax_noise.py` → `gminmax_noise.txt` |
| S3 and Table S2 | `motion_covariates.py`, then `sensitivity.py` → `tab_sens.tex`, `sensitivity.txt` (head motion vs age r = +0.13); noise numbers from `post_native.py` |
| Tables S3, S4, S5 | `supp_tables.py` → `tab_age.tex`, `tab_beyond.tex`, `tab_bdep.tex` |
| S5 and Table S6 | `jf_pass.py`, `jf_stats.py` → `tab_jf.tex`, `jf_stats.txt` (14/16 below the paired null, mismatch 8–18%, 3/16 above the identical-direction null by at most 3%, expected J_f about 3e-4 against noise about 5e-3); pairing angles (2–8°, median 6°) and mismatch in the example from `check_pairing.py` |
