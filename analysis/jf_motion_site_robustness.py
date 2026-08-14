"""Motion, site and quality robustness for the J_f age association.

The manuscript reports that the gap age correlations shift by at most 0.013 when
head motion, acquisition site and eddy outlier fraction are partialled out. J_f
is a new reported result and is held to the same standard here.

J_f warrants this check more than the single-shell gaps do. It is a ratio ACROSS
shells, f_i = D_i(3000)/D_i(1500), and b=3000 is acquired later in the scan where
motion and dropout are worse. Motion also rises with age. So motion could inflate
directional spread in f preferentially in older participants and manufacture part
of the age association. This tests that directly.

Model mirrors robustness_motion_site.py exactly:
    unadjusted   J_f ~ age | sex
    adjusted     J_f ~ age | sex + motion (absolute + framewise) + site + outlier_frac
Site is categorical (four sites), one-hot with the first level dropped.

Output: printed table over the 16 JHU ROIs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import os
BASE = Path(os.environ.get("JG_DATA", Path(__file__).resolve().parent))
HCP = BASE / "HCP"
DEMOG = HCP / "AABC2_subjects_2026_02_05_14_29_11.csv"

ROIS = ["CR_all", "ACR", "SCR", "PCR", "Body", "Genu", "Splenium", "alic", "plic",
        "rlic", "ec", "slf", "ptr", "cingulum_cc", "ss", "tapetum"]
PRIMARY = "CR_all"


def residualize(y, X):
    Xd = np.column_stack([np.ones(len(X)), X.values.astype(float)]) if len(X.columns) \
        else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return y - Xd @ beta


def partial_r(df, metric, covars):
    sub = df[["age", metric] + covars].dropna()
    parts = []
    for c in covars:
        if pd.api.types.is_numeric_dtype(sub[c]):
            parts.append(sub[[c]].astype(float))
        else:
            parts.append(pd.get_dummies(sub[c], prefix=c, drop_first=True).astype(float))
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=sub.index)
    ry = residualize(sub[metric].to_numpy(float), X)
    ra = residualize(sub["age"].to_numpy(float), X)
    r, _ = stats.pearsonr(ra, ry)
    dof = len(sub) - 2 - X.shape[1]
    t = r * np.sqrt(dof / max(1e-12, 1 - r * r))
    return r, 2 * stats.t.sf(abs(t), dof), len(sub)


def load_site():
    d = pd.read_csv(DEMOG, low_memory=False)
    d["Subject_ID"] = d["id_event"].str.rsplit("_", n=1).str[0]
    d["visit"] = d["id_event"].str.rsplit("_", n=1).str[1]
    return (d[d["visit"] == "V1"][["Subject_ID", "site"]]
            .drop_duplicates("Subject_ID"))


def main():
    jf = pd.read_csv(HCP / "cr_jf_long.csv")
    man = pd.read_csv(HCP / "manifest_n1379_b1500.tsv", sep="\t")
    man["age"] = pd.to_numeric(
        man["age"].astype(str).str.replace("90 or older", "90", regex=False),
        errors="coerce")

    # The manuscript specifies FRAMEWISE motion. eddy_restricted_movement_rms has
    # both an absolute and a relative (framewise) column, and only the framewise
    # one tracks age (r=0.155 vs 0.037). Adjusting on the absolute column alone
    # would barely control the confound. Both are included here, which is
    # stricter than the manuscript's single covariate.
    mot = pd.read_csv(HCP / "motion_rms_n1379.csv")[
        ["subject_id", "visit", "motion_rms", "motion_rms_relprev"]]
    out = pd.read_csv(HCP / "eddy_outliers_n1379.csv")[
        ["subject_id", "visit", "outlier_frac"]]
    site = load_site()

    d = (jf.merge(man[["session_id", "age", "sex"]],
                  left_on="Session_ID", right_on="session_id")
           .merge(mot, left_on=["Subject_ID", "Visit"],
                  right_on=["subject_id", "visit"], how="left")
           .merge(out, left_on=["Subject_ID", "Visit"],
                  right_on=["subject_id", "visit"], how="left")
           .merge(site, left_on="Subject_ID", right_on="Subject_ID", how="left"))
    d["J_f_mean"] = pd.to_numeric(d["J_f_mean"], errors="coerce")

    cr = d[d.ROI == PRIMARY]
    print(f"sessions: {cr.Session_ID.nunique():,}")
    for c in ("motion_rms", "motion_rms_relprev", "outlier_frac", "site"):
        print(f"  {c:13s} missing {int(cr[c].isna().sum()):4d}   "
              f"{'levels: ' + str(sorted(cr[c].dropna().unique())) if c == 'site' else ''}")
    mm = cr[["motion_rms_relprev", "age"]].dropna()
    mr, mp = stats.pearsonr(mm["motion_rms_relprev"].values, mm["age"].values)
    print(f"  motion vs age: r={mr:+.3f} (p={mp:.1e})\n")

    print(f"{'ROI':14s} {'unadjusted':>12s} {'adjusted':>12s} {'shift':>8s}")
    shifts = []
    for roi in ROIS:
        s = d[d.ROI == roi]
        r0, _, n0 = partial_r(s, "J_f_mean", ["sex"])
        r1, p1, n1 = partial_r(s, "J_f_mean",
                               ["sex", "motion_rms", "motion_rms_relprev", "site", "outlier_frac"])
        shifts.append(abs(r1 - r0))
        print(f"  {roi:12s} {r0:+.3f}       {r1:+.3f}      {abs(r1-r0):.3f}   "
              f"p={p1:.1e}  N={n1:,}")

    cr_adj = partial_r(d[d.ROI == PRIMARY], "J_f_mean",
                       ["sex", "motion_rms", "motion_rms_relprev", "site", "outlier_frac"])[0]
    print(f"\nmax |shift| across the 16 ROIs: {max(shifts):.3f}")
    print(f"CR_all adjusted: r={cr_adj:+.3f}")


if __name__ == "__main__":
    main()
