"""Robustness of the CR_all age correlations to head motion + site + outliers (Section 7.1).

Shows the sex-adjusted J_ln / J_lnS age correlations in CR_all shift by at most
0.013 at BOTH shells (b=1500 and b=3000) after also adjusting for in-scanner head
motion, acquisition site, and the eddy outlier fraction, and reports how motion
tracks age.

Per-subject quality metrics, read from each HCP-A DiffusionRecommended zip
(T1w/Diffusion/eddylogs/):
  motion  = mean FRAMEWISE (relative-to-previous, column 2) restricted-movement RMS
            (the field-standard motion QC; column 1 is cumulative drift and barely
            tracks age -- the framewise column is the one that does, r=0.16,
            matching the info-theory companion paper).
  outlier = mean of eddy_outlier_map (scan x slice 0/1, header line skipped)
            = fraction of slice-volumes flagged as signal-dropout outliers.
Site: from the AABC2 demographics.

Side effect: writes the per-shell quality caches HCP/quality_n1379.csv (b1500)
and HCP/quality_b3000_n1379.csv (b3000), which tbss_pipeline/build_design_full.py
consumes to build the 10-EV confound-adjusted TBSS design.

Env: JG_DATA (data root with HCP/...), HCPA_ZIP_GLOB (glob for the zips;
     comma-separated globs allowed, default "H:/HCA*_DiffusionRecommended.zip").
"""
import os, glob, io, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

DATA = Path(os.environ.get("JG_DATA", "."))
HCP = DATA / "HCP"
zips = {}
for gpat in os.environ.get("HCPA_ZIP_GLOB", "H:/HCA*_DiffusionRecommended.zip").split(","):
    for p in glob.glob(gpat.strip()):
        zips[Path(p).name] = p

ED = "T1w/Diffusion/eddylogs/eddy_unwarped_images"


def quality(subj, visit):
    """(framewise motion RMS, outlier fraction) for one subject, or (nan, nan)."""
    p = zips.get(f"{subj}_{visit}_MR_DiffusionRecommended.zip")
    if not p:
        return np.nan, np.nan
    try:
        z = zipfile.ZipFile(p)
        base = f"{subj}_{visit}_MR/{ED}"
        mot = float(np.mean(np.loadtxt(io.BytesIO(z.read(base + ".eddy_restricted_movement_rms")))[:, 1]))
        out = float(np.loadtxt(io.BytesIO(z.read(base + ".eddy_outlier_map")), skiprows=1).mean())
        return mot, out
    except Exception:
        return np.nan, np.nan


demo = pd.read_csv(glob.glob(str(HCP / "AABC2_subjects_*.csv"))[0], low_memory=False)
demo["subject_id"] = demo.id_event.str.rsplit("_", n=1).str[0]
demo["visit"] = demo.id_event.str.rsplit("_", n=1).str[1]
g = pd.read_csv(HCP / "cr_gaps_long.csv")


def partial_r(df, y, covs):
    d = df.dropna(subset=[y, "age"] + [c for c in covs if not c.startswith("C(")]).copy()
    parts = [np.ones(len(d))]
    for c in covs:
        parts.append(pd.get_dummies(d[c[2:-1]], drop_first=True).astype(float).values
                     if c.startswith("C(") else d[c].values.reshape(-1, 1))
    X = np.column_stack(parts)
    res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    return stats.pearsonr(res(d[y].values), res(d["age"].values))[0]


# The manifest picks the best-QC visit independently per shell, so the b=1500 and
# b=3000 cohorts differ for 275 subjects. Each shell's covariates are therefore
# pulled from its own sessions (paper reports both shells adjusted).
FULL = ["C(sex)", "motion", "C(site)", "outlier"]
PAPER = {1500: "J_ln -0.47->-0.46, J_lnS +0.30->+0.30",
         3000: "J_ln -0.42->-0.41, J_lnS +0.35->+0.35"}
# Shell-specific quality cache filename (also consumed by
# tbss_pipeline/build_design_full.py to build the 10-EV TBSS design).
QUAL_CSV = {1500: "quality_n1379.csv", 3000: "quality_b3000_n1379.csv"}
for bval in (1500, 3000):
    man = pd.read_csv(HCP / f"manifest_n1379_b{bval}.tsv", sep="\t")
    man["age"] = pd.to_numeric(man["age"], errors="coerce")
    man[["motion", "outlier"]] = [quality(s, v) for s, v in zip(man.subject_id, man.visit)]
    man[["subject_id", "visit", "motion", "outlier"]].to_csv(HCP / QUAL_CSV[bval], index=False)
    man = man.merge(demo[["subject_id", "visit", "site"]], on=["subject_id", "visit"], how="left")
    j = g[(g.shell_bval == bval) & (g.ROI == "CR_all")].merge(
        man[["session_id", "age", "sex", "motion", "outlier", "site"]],
        left_on="Session_ID", right_on="session_id", how="inner")
    for c in ("J_ln_D_mean", "J_ln_S_mean"):
        j[c] = pd.to_numeric(j[c], errors="coerce")
    print(f"\nb={bval}: quality for {man.motion.notna().sum()}/{len(man)} ; "
          f"motion vs age r={j['motion'].corr(j['age']):+.3f}, "
          f"outlier vs age r={j['outlier'].corr(j['age']):+.3f}")
    print(f"  CR_all age partial r  (sex -> sex+motion+site+outlier ; paper {PAPER[bval]}):")
    for y, nm in [("J_ln_D_mean", "J_ln"), ("J_ln_S_mean", "J_lnS")]:
        print(f"    {nm:5s}  {partial_r(j, y, ['C(sex)']):+.3f}  ->  {partial_r(j, y, FULL):+.3f}")
