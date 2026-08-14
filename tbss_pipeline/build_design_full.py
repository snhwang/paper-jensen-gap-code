"""
Build the FULL confound-adjusted TBSS design for the N=1379 randomise re-run:
age + age^2 + sex + age*sex + acquisition site + head motion + eddy outlier
fraction. This is the design behind the reported TBSS figures (Figs 5-6); it is
the sex-adjusted "agesex" design (build_design_agesex.py) with five nuisance EVs
appended, so the two are directly comparable (the maps are near-identical, Dice
0.98-1.00).

Reads the EXISTING manifest (manifest_n1379_b{bval}.tsv) so design.mat row k
matches TBSS 4D volume k, exactly as build_design_agesex.py does.

Design columns (10 EVs, one row per subject; every nuisance column demeaned):
   1  intercept   = 1
   2  age_c       = age - mean(age)
   3  age2_c      = age_c^2 - mean(age_c^2)
   4  sex_c       = sex_code - mean(sex_code)          (F=0, M=1)
   5  age_x_sex   = age_c * sex_c, demeaned
   6..k  site     = one demeaned indicator per non-reference site (4 sites -> 3)
   k+1 motion     = mean framewise eddy restricted-movement RMS, demeaned
   k+2 outlier    = mean eddy outlier fraction, demeaned

Contrast (Text2Vest input): the adjusted linear-age effect only,
  tstat1 = +age   tstat2 = -age   (a 0 on every nuisance column).

Motion + outlier are read from the SHELL-SPECIFIC quality files, because the
b1500 and b3000 manifests select different visits for 275 subjects, so each
shell's covariates must come from its own sessions:
    b1500 -> HCP/quality_n1379.csv        (subject_id, visit, motion, outlier)
    b3000 -> HCP/quality_b3000_n1379.csv
Both are written by analysis/motion_robustness.py. Site is subject-level, taken
from the AABC demographics (AABC2_subjects_*.csv).

Outputs (per bval):
  HCP/design_full_b{bval}.txt   (Text2Vest input -> design.mat)
  HCP/contrast_full.txt         (Text2Vest input -> design.con; shell-agnostic)

The .txt files become FSL .mat/.con via Text2Vest (see README Step 4):
  Text2Vest HCP/design_full_b1500.txt HCP/design_full_b1500.mat
  Text2Vest HCP/contrast_full.txt     HCP/contrast_full.con
"""
import csv
import glob
import math
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
HCP = BASE / "HCP"
QUALITY = {1500: "quality_n1379.csv", 3000: "quality_b3000_n1379.csv"}


def _mean(x):
    return sum(x) / len(x)


def _demean(x):
    m = _mean(x)
    return [v - m for v in x]


def _corr(a, b):
    ma, mb = _mean(a), _mean(b)
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(len(a)))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((x - mb) ** 2 for x in b))
    return num / (da * db) if da * db else float("nan")


def load_site():
    """Subject_ID -> acquisition site, from the AABC demographics (V1 row)."""
    path = glob.glob(str(HCP / "AABC2_subjects_*.csv"))[0]
    d = pd.read_csv(path, low_memory=False)
    d["subject_id"] = d["id_event"].str.rsplit("_", n=1).str[0]
    d["visit"] = d["id_event"].str.rsplit("_", n=1).str[1]
    d = d[d["visit"] == "V1"].drop_duplicates("subject_id")
    return dict(zip(d["subject_id"], d["site"]))


def load_quality(bval):
    q = pd.read_csv(HCP / QUALITY[bval])
    return dict(zip(q.subject_id, q.motion)), dict(zip(q.subject_id, q.outlier))


def build(bval, site_map, site_levels):
    sid, age, sex = [], [], []
    with open(HCP / f"manifest_n1379_b{bval}.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sid.append(r["subject_id"])
            age.append(float(r["age"]))
            sex.append(r["sex"].strip().upper())
    n = len(sid)
    mot_map, out_map = load_quality(bval)

    miss = [s for s in sid if s not in site_map or s not in mot_map or s not in out_map]
    if miss:
        raise SystemExit(f"ABORT b={bval}: {len(miss)} subjects missing a covariate, e.g. {miss[:5]}")

    age_c = _demean(age)
    age2_c = _demean([x * x for x in age_c])
    sex_c = _demean([1.0 if s == "M" else 0.0 for s in sex])
    inter_c = _demean([age_c[i] * sex_c[i] for i in range(n)])
    motion_c = _demean([float(mot_map[s]) for s in sid])
    out_c = _demean([float(out_map[s]) for s in sid])
    ref = site_levels[0]
    dummies = [_demean([1.0 if site_map[s] == lvl else 0.0 for s in sid]) for lvl in site_levels[1:]]

    cols = [[1.0] * n, age_c, age2_c, sex_c, inter_c] + dummies + [motion_c, out_c]
    n_ev = len(cols)
    design = HCP / f"design_full_b{bval}.txt"
    with open(design, "w") as f:
        for i in range(n):
            f.write(" ".join(f"{cols[j][i]:.6f}" for j in range(n_ev)) + "\n")

    print(f"b={bval}: n={n} EVs={n_ev} covar-file={QUALITY[bval]} site ref='{ref}'")
    print(f"  collinearity w/ age: motion={_corr(age_c, motion_c):+.3f} outlier={_corr(age_c, out_c):+.3f} "
          + " ".join(f"site[{site_levels[k + 1]}]={_corr(age_c, dummies[k]):+.3f}" for k in range(len(dummies))))
    print(f"  wrote {design} ({n} x {n_ev})")
    return n_ev


def main():
    site = load_site()
    site_levels = sorted(set(site.values()))
    print(f"sites ({len(site_levels)}): {site_levels}")
    n_ev = None
    for bval in (1500, 3000):
        n_ev = build(bval, site, site_levels)
    con = HCP / "contrast_full.txt"
    with open(con, "w") as f:
        plus = ["0"] * n_ev
        plus[1] = "1"
        minus = ["0"] * n_ev
        minus[1] = "-1"
        f.write(" ".join(plus) + "\n")
        f.write(" ".join(minus) + "\n")
    print(f"\ncontrasts (tstat1 +age, tstat2 -age) -> {con} ({n_ev} cols)")


if __name__ == "__main__":
    main()
