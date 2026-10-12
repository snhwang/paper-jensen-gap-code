"""Head motion and eddy outlier fraction for the first-visit cohort.

These are the motion covariates of the sensitivity analysis (sensitivity.py,
Supplementary Section S3). They come from the eddy output that the HCP
preprocessing stores in each session's download zip
(<Subject>_V1_MR_DiffusionRecommended.zip, AABC Release 2 on BALSA). Only the
two small text files below are read from each zip. Nothing is extracted.

  motion_rel     mean of column 2 of eddy_restricted_movement_rms, the RMS
                 displacement of each volume relative to the previous volume
                 (mm), disregarding translation along the phase-encoding
                 direction, which eddy cannot separate from eddy currents
  outlier_frac   fraction of entries equal to 1 in eddy_outlier_map, which has
                 one row per volume and one column per slice (1 = slice flagged
                 as an outlier and replaced by eddy)

eddy runs once per session on all shells together, so both values are the same
for b = 1500 and b = 3000.

Inputs:  out/v1_cohort_sessions.csv (v1_roi_stats.py), the download zips in
         config.ZIPS (environment variable JG_ZIPS)
Output:  out/motion.csv (Subject_ID, motion_rel, outlier_frac)
Usage:   python motion_covariates.py
"""
import sys
import zipfile

import numpy as np
import pandas as pd

from config import OUT, ZIPS


def member(z, suffix):
    names = [n for n in z.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"{z.filename}: {len(names)} members end with {suffix}")
    return z.read(names[0]).decode().splitlines()


def covariates(path):
    with zipfile.ZipFile(path) as z:
        rms = np.array([[float(x) for x in ln.split()]
                        for ln in member(z, "eddy_restricted_movement_rms") if ln.strip()])
        flags = [int(x) for ln in member(z, "eddy_outlier_map")
                 if (p := ln.split()) and all(x in ("0", "1") for x in p) for x in p]
    return rms[:, 1].mean(), np.mean(flags)


cohort = pd.read_csv(OUT / "v1_cohort_sessions.csv")
rows, missing = [], []
for i, sid in enumerate(cohort.Subject_ID, 1):
    path = ZIPS / f"{sid}_V1_MR_DiffusionRecommended.zip"
    if not path.exists():
        missing.append(sid)
        continue
    motion, outliers = covariates(path)
    rows.append(dict(Subject_ID=sid, motion_rel=motion, outlier_frac=outliers))
    if i % 200 == 0:
        print(f"{i}/{len(cohort)}", flush=True)
if missing:
    sys.exit(f"no download zip in {ZIPS} for {len(missing)} participants, e.g. {missing[:5]}")
m = pd.DataFrame(rows)
m.to_csv(OUT / "motion.csv", index=False)
print(f"{len(m)} participants: motion_rel median {m.motion_rel.median():.3f} mm, "
      f"outlier_frac median {m.outlier_frac.median():.4f}")
