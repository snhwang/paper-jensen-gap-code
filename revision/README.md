# Analysis code for the revised manuscript

Every number, figure and table of the revised manuscript and its supplement is
produced by a script in this folder. This file says which script, and from what.

## Quick start

```bash
cd revision
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py --list     # the steps, in dependency order
python run_all.py            # rebuild everything except the slow native passes
python run_all.py --heavy    # include the native passes (about 9 h, disk-bound)
```

`requirements.txt` pins the package versions of the paper's run (Python
3.14.2). Python 3.11 or later can install them. The scripts read the data
described under Inputs below. `run_all.py` stops at the first failing step.
`--from NAME` resumes from a step.

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

## Outputs

| Folder | Content | In git? |
|---|---|---|
| `out/` | intermediate tables and text summaries | the text summaries and the 15 group-level tables (`GROUP_TABLES` in `config.py`) yes. The participant-level tables **no**, because they carry participant or session identifiers |
| `figures/` | main-text figures (PDF, with PNG previews) | no |
| `figures_supp/` | supplementary figures | no |
| `tables_supp/` | LaTeX tables of the supplement | yes |

Figure PDFs are reproducible byte for byte with the package versions of
`requirements.txt`. `config.py` fixes `SOURCE_DATE_EPOCH`. After a rerun,
`git diff` shows any change in the committed tables and summaries.

Without access to the data, the reported numbers can be checked against the
committed tables. Four steps need no data at all: `fig1_schematic.py`,
`fig_configurations.py`, `theory_table.py` and `crossing_sim.py`.
`supp_tables.py` rebuilds Supplementary Tables S3 and S4 from the committed
tables. Every other step starts from the participant data.

## Steps

| Step | Script | Reads | Writes | Time |
|---|---|---|---|---|
| cohort | `v1_roi_stats.py` | cr_gaps_long, demographics, b-values | `v1_cohort_sessions.csv`, `A_cross_sectional.csv`, `B_decomposition.csv`, `C_bdependence.csv`, `E_spectrum_voxelwise.csv`, `summary.txt` | 1 min |
| motion | `motion_covariates.py` | download zips of the cohort (two eddy text files each) | `motion.csv` | 30 s |
| native | `native_pass.py` | raw DWI of the cohort | `native_pass.csv` | 2 h |
| split_half_masks | `split_half_masks.py` | b-values and b-vectors of the example session | `split_half_masks.npz` (committed, next to the script) | 3 min |
| split_half | `split_half_pass.py` | raw DWI of the cohort, `split_half_masks.npz` | `split_half.csv` | 1.5 h |
| split_half_stats | `split_half_stats.py` | `split_half.csv`, `motion.csv` | `split_half_regions.csv`, `split_half_stats.txt` | seconds |
| floor_null | `floor_null_pass.py` | raw DWI of the cohort, `split_half_masks.npz` | `floor_null.csv` | 3 h |
| floor_null_stats | `floor_null_stats.py` | `floor_null.csv`, `motion.csv` | `floor_null_noise.csv`, `floor_null_regions.csv`, `floor_null_stats.txt` | seconds |
| floor_null_sim | `floor_null_sim.py --sample 300` | raw DWI of 300 random cohort sessions, `floor_null.csv` | `floor_null_sim.csv`, `floor_null_sim.txt` | 30 min |
| post_native | `post_native.py` | `native_pass.csv` | `fa_correlations.csv`, `beyond_tensor.csv`, `bdep_snr.csv`, `noise_corrected.csv`, `post_native_summary.txt` | 1 min |
| long_sessions | `long_sessions.py` | cr_gaps_long, demographics, b-values | `long_sessions.csv` (with `days_from_V1`, the exact visit date offset), `long_todo_sessions.csv` | 1 min |
| native_long | `native_pass.py --sessions out/long_todo_sessions.csv --tag long` | raw DWI | `native_pass_long.csv` | 2 h |
| longitudinal | `longitudinal.py` | `long_sessions.csv`, `native_pass*.csv` | `longitudinal.csv`, `longitudinal_by_age.csv`, `longitudinal.txt` | 1 min |
| jf_pass | `jf_pass.py --sample 300` | raw DWI of 300 random cohort sessions | `jf_pass.csv` | 30 min |
| jf_stats | `jf_stats.py` | `jf_pass.csv`, `C_bdependence.csv` | `jf_stats.csv`, `jf_stats.txt` | seconds |
| example_maps | `example_maps.py` | raw DWI of the example session | voxel maps in `JG_MAPS` | 30 s |
| example_stats | `example_stats.py` | example maps, b-vectors | `example_stats.txt` | seconds |
| check_pairing | `check_pairing.py` | example DWI and maps | `check_pairing.txt` | 30 s |
| theory_table | `theory_table.py` | none (simulation) | `tables_supp/tab_accuracy.tex` | seconds |
| crossing_sim | `crossing_sim.py` | nothing (theory) | `crossing_sim.txt` | 5 s |
| sensitivity | `sensitivity.py` | cr_gaps_long, `motion.csv` | `tables_supp/tab_sens.tex`, `sensitivity.csv`, `sensitivity.txt` | 1 min |
| fig1 to fig5_6 | `fig1_schematic.py`, `fig_configurations.py`, `fig2_maps.py`, `fig3_fa_relation.py`, `fig_heatmaps.py`, `fig_region.py` | see each script's docstring | `figures/`, `figures_supp/`, `fig_configurations.txt`, `fig4_age_correlations.csv`, `fig5_decomposition.csv`, `lifespan_shape.txt` | 7 min (6 for the GAM bootstrap) |
| supp_tables | `supp_tables.py` | `fig4_age_correlations.csv`, `C_bdependence.csv`, `split_half_regions.csv` | `tables_supp/tab_age.tex`, `tab_bdep.tex`, `tab_jf.tex` | seconds |

The native passes write a `*_partial.csv` as they go and resume from it if
interrupted. Delete it to start over.

`check_heavy_outputs.py` checks the slow outputs in two ways.
- `python check_heavy_outputs.py [N]` reruns the first N sessions (default 3)
  of each stored slow output (`native_pass.csv`, `native_pass_long.csv`,
  `jf_pass.csv`, `split_half.csv`, `floor_null.csv`) with the current scripts
  and compares every value. It takes a few minutes. Run it after editing
  `native_pass.py`, `jf_pass.py`, `split_half_pass.py` or `floor_null_pass.py`.
- `python check_heavy_outputs.py --paper ROOT`, after a complete
  `run_all.py --heavy`, compares every session of the slow outputs with those
  of the paper's source tree ROOT. It also requires the text summaries, the
  supplementary tables and the figure PDFs to have identical content. It
  ignores only line endings, PDF compression and the sign of zero, which
  differ between Windows and Linux builds.

The public copy was rerun from scratch this way on 2026-10-14, on Linux
(Ubuntu, Python 3.12, the same package versions as the paper's Windows run). Every session matched the paper's
outputs to within 5e-11, and every summary, table and figure was identical in
content.

It was rerun again on 2026-11-16, after the noise-floor nulls were added, on
Linux (Ubuntu, Python 3.14, the pinned versions). Every session of the slow
outputs matched to within 2.5e-7 (relative), and every summary, group-level
table and supplementary table was identical in content. One figure differed.
Figure 2 (`fig_configurations.pdf`) draws its 3D surfaces as an embedded bitmap,
and on Linux 260 of its 1.45 million preview pixels differ by one level of 255.
Redrawn on Windows from the same code, it is identical in content to the
paper's, so the difference comes from the platform's floating-point arithmetic,
not the code. Its numbers (`fig_configurations.txt`) are the same on both.

## Where each result comes from

### Main text

| Item | Source |
|---|---|
| Section 2.5: crossing fibres, gap over the fitted tensor's gap 1.19 (b = 1500) and 1.67 (b = 3000) at 90 degrees, 1.00 for one population (the noise comparison the script also prints was cut from the text on 2026-11-15) | `crossing_sim.py` → `crossing_sim.txt` |
| Section 3.1: release size (1,396 participants, 2,878 visits, 1,379 with dMRI), N = 1,354, age, sex, sites, HCP-A/AABC split | `v1_roi_stats.py` → `summary.txt` |
| Section 3.1: eigenvalues of M (0.341, 0.334, 0.325; 0.341, 0.331, 0.329) | `example_stats.py` → `example_stats.txt` |
| Section 3.1: ADC_1 against the tensor MD (median ratio 0.994 at b = 1500, 0.985 at b = 3000) | `post_native.py` → `post_native_summary.txt` |
| Section 3.2: the limits on S' change fewer than 0.01% of the white-matter signals of the example participant, and no gap is negative | `example_stats.py` → `example_stats.txt` |
| Section 4.1: ADC_0, D_SM about 5% below ADC_1; ρ(J_ln, FA) 0.994/0.991; ρ = 0.998 and ratio 0.906 for the signal gap; 30,878 voxels | `example_stats.py` → `example_stats.txt` |
| Section 4.1: ρ with FA across participants (0.83 to 0.98; 0.30 to 0.92) | `post_native.py` → `fa_correlations.csv` |
| Section 4.2: r with age of the gaps, FA and MD; 180 of 192 significant | `fig_heatmaps.py main` → `fig4_age_correlations.csv` (same values in `A_cross_sectional.csv`) |
| Section 4.2: covariates (at most 0.011), top-coded ages (at most 0.001) | `motion_covariates.py`, then `sensitivity.py` → `sensitivity.csv`, `sensitivity.txt` |
| Section 4.3: MD and anisotropy terms, 13/16 and 15/16 signs, r = 0.90/0.96, least-squares slope of observed on predicted 0.47/0.59 ("about half") | `v1_roi_stats.py` → `B_decomposition.csv`; `fig_region.py` → `fig5_decomposition.csv` |
| Section 4.3: course over age in the corona radiata, MD term +0.24 to +1.52 %/y and anisotropy term −0.23 to −0.67 %/y (ages 36–54 against 70–90), signal gap lowest at age 53.5 (GAM curve of Figure 7b), and the GAM test against a straight line, with p from a parametric bootstrap of 2000 data sets (signal gap p = 0.0005 at both shells, i.e. none of the simulated statistics as large, reported as p < 0.001; diffusivity gap p = 0.0195 and 0.0960 at b = 1500 and 3000, reported as 0.02 and 0.10). The text rounds the minimum to "about age 54", the abstract and Discussion to "the mid-50s" | `fig_region.py` → `lifespan_shape.txt` |
| Section 4.4: J_ln lower at b = 3000 (12–30%, >99% of participants), Δ_b vs age | `v1_roi_stats.py` → `C_bdependence.csv` |
| Section 4.4 and Table S4: change of the diffusivity gap against the noise-floor nulls (corona radiata: null 12.7 to 15.3 of 15.8 points, excess 0.5 to 3.1; remaining decrease significant in 6 to 10 regions, reversed in 5 to 9, always ec, ptr, cingulum, tapetum; excess r with age +0.36 to +0.61 over all 18 nulls) | `floor_null_pass.py` → `floor_null_stats.py` → `floor_null_stats.txt` (section "Paper"), `floor_null_regions.csv`; `supp_tables.py` → `tab_bdep.tex` |
| Section 4.4 and S4: null error in simulated data (tensor 0.6 to 2.7 points, corona radiata 1.0; directional up to 2.5, corona radiata 0.9), corrected excess 2.3 to 3.1 points and r with age +0.59 to +0.62 in the corona radiata, change of r −0.13 to +0.04 | `floor_null_sim.py --sample 300` → `floor_null_sim.txt` |
| S4: calibration raise (4 to 5% corona radiata, 12 to 16% corpus callosum), uncalibrated nulls (15/16, 3.2 to 6.5 points), four-channel noise (no excess in the corona radiata), SH order (≤4%), ventricle σ ratio (1.04, 0.99 to 1.10, r with age −0.41), b=0 σ (3 to 90% higher), fixed n_eff (age r 0.03 to 0.04 lower in the corona radiata, up to 0.14 elsewhere) | `floor_null_stats.py` → `floor_null_stats.txt`, `floor_null_noise.csv` |
| Section 4.5: within-person rates as measured; participants, sessions, follow-up, scanner changes. Time is the exact interval from `days_from_V1`, not the released whole-year ages. The same file also gives visits per participant and age at the last visit, which the paper does not report | `longitudinal.py` → `longitudinal.txt` |
| Section 4.5: rate of the signal gap by age at the first visit (−0.17, +0.12, +0.42 %/y at 45, 60, 75, interaction p < 1e-38), per-person slopes (−0.17, +0.13, +0.43), and the age dependence of the diffusivity gap's rate, measured (interaction p = 3.1e-6, reported as p < 1e-5) and with the noise inflation removed (p = 0.13). The noise inflation per voxel is the tensor+noise gap minus the tensor gap (`native_pass.py`, columns `*_TN` and `*_T`) | `longitudinal.py` → `longitudinal.txt`, `longitudinal_by_age.csv` |
| Section 3.5.2: signal-to-noise ratio against age (r = +0.64, corona radiata) | `post_native.py` → `post_native_summary.txt` |
| Section 3.5.2: n_eff from the ventricle floor at b = 3000 (4.15, 5th to 95th percentile 3.67 to 4.65, 1,351 sessions with a ventricle mask; r with age +0.19); S4: ventricle floor mean 2.66 sigma and mean/SD 3.90, and the reference values of averaged Rician and four-channel noncentral chi noise | `floor_null_pass.py` → `floor_null_stats.py` → `floor_null_stats.txt` |
| Figure 1 | `fig1_schematic.py` (theory only, no data) |
| Figure 2 (six simulated voxels at two b-values), and Section 2.6: Δ_b J_ln +30% for the restricted fibre, −16% for the dispersed bundle, zero for random fibres although their ADC falls (0.655 to 0.581). The same file checks two statements of Section 2.3: one component's signal gap equals the linear-minus-isotropic encoding contrast, and the voxel-to-component ratio approaches the squared order parameter only at low b (0.20 against 0.25 for the crossing at b = 1500) | `fig_configurations.py` → `fig_configurations.txt` (theory only, no data) |
| Figure 3 | `example_maps.py`, then `fig2_maps.py` |
| Figure 4 | `example_maps.py`, `post_native.py`, then `fig3_fa_relation.py` |
| Figure 5 | `fig_heatmaps.py main` |
| Figure 6 | `fig_region.py` |
| Figure 7 | `fig_region.py` (panel b: GAM curves with pointwise 95% intervals; panels c, d use `longitudinal.csv` and `longitudinal_by_age.csv`) |
| Table 1 | theory, no data |

### Supplement

| Item | Source |
|---|---|
| S1 and Table S1 | `theory_table.py` → `tab_accuracy.tex` |
| S2 and Figure S1 | `v1_roi_stats.py` (`E_spectrum_voxelwise.csv`), then `fig_heatmaps.py spectrum` |
| S3 and Table S2 | `motion_covariates.py`, then `sensitivity.py` → `tab_sens.tex`, `sensitivity.txt` (head motion vs age r = +0.13) |
| Tables S3 and S4 | `supp_tables.py` → `tab_age.tex`, `tab_bdep.tex` |
| Not in the paper (S5 and Table S5 were cut on 2026-11-15) | J_f on paired directions (mismatch 8–18%, random 300): `jf_pass.py`, `jf_stats.py` → `jf_stats.txt`; pairing angles (2–8°) from `check_pairing.py`. J_f mostly noise (noise share 51–76%): `split_half_pass.py` → `split_half_stats.py` → `split_half_stats.txt` |
