"""
Build the field-standard age + age^2 + sex + age*sex design for the N=1379
TBSS randomise re-run.

Reads the EXISTING manifest (manifest_n1379_b{bval}.tsv) so the row order is
identical to the order the TBSS 4D volumes were built in (alphabetical by
subject_id) -- design.mat row k therefore matches volume k, exactly as the
current age-only design does.

Design columns (one row per subject):
  1  intercept   = 1
  2  age_c       = age - mean(age)
  3  age2_c      = age_c^2 - mean(age_c^2)         (demeaned quadratic)
  4  sex_c       = sex_code - mean(sex_code)        (F=0, M=1)
  5  age_x_sex   = age_c * sex_c, demeaned          (linear age-by-sex)

Contrasts (Text2Vest input), 8 rows -> randomise tstat1..8:
  tstat1 +age      tstat2 -age
  tstat3 +age^2    tstat4 -age^2
  tstat5 +sex(M>F) tstat6 -sex(F>M)
  tstat7 +age*sex  tstat8 -age*sex

Outputs (per bval):
  HCP/design_agesex_b{bval}.txt   (Text2Vest input -> design.mat)
  HCP/contrast_agesex.txt         (Text2Vest input -> design.con; shell-agnostic)
"""
import csv
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent

CONTRASTS = [
    ("0 1 0 0 0",  "+age (linear, adjusted)"),
    ("0 -1 0 0 0", "-age"),
    ("0 0 1 0 0",  "+age^2 (nonlinear)"),
    ("0 0 -1 0 0", "-age^2"),
    ("0 0 0 1 0",  "+sex (M>F)"),
    ("0 0 0 -1 0", "-sex (F>M)"),
    ("0 0 0 0 1",  "+age*sex"),
    ("0 0 0 0 -1", "-age*sex"),
]


def _mean(x):
    return sum(x) / len(x)


def _corr(a, b):
    ma, mb = _mean(a), _mean(b)
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(len(a)))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((x - mb) ** 2 for x in b))
    return num / (da * db) if da * db else float("nan")


def build(bval):
    man = BASE / "HCP" / f"manifest_n1379_b{bval}.tsv"
    rows = []
    with open(man) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append((r["subject_id"], float(r["age"]), r["sex"].strip().upper()))
    n = len(rows)

    ages = [a for _, a, _ in rows]
    mean_age = _mean(ages)
    age_c = [a - mean_age for a in ages]

    age2 = [x * x for x in age_c]
    mean_age2 = _mean(age2)
    age2_c = [x - mean_age2 for x in age2]

    sex_code = [1.0 if s == "M" else 0.0 for _, _, s in rows]  # F=0, M=1
    mean_sex = _mean(sex_code)
    sex_c = [s - mean_sex for s in sex_code]

    inter = [age_c[i] * sex_c[i] for i in range(n)]
    mean_int = _mean(inter)
    inter_c = [x - mean_int for x in inter]

    design = BASE / "HCP" / f"design_agesex_b{bval}.txt"
    with open(design, "w") as f:
        for i in range(n):
            f.write(f"1 {age_c[i]:.6f} {age2_c[i]:.6f} {sex_c[i]:.6f} {inter_c[i]:.6f}\n")

    n_m = int(sum(sex_code))
    print(f"b={bval}: n={n}  (M={n_m}, F={n - n_m}, {100 * mean_sex:.1f}% M)  mean_age={mean_age:.4f}")
    print(f"  centered-col means (want ~0): "
          f"age_c={_mean(age_c):+.2e} age2_c={_mean(age2_c):+.2e} "
          f"sex_c={_mean(sex_c):+.2e} int={_mean(inter_c):+.2e}")
    print(f"  collinearity check: corr(age,age^2)={_corr(age_c, age2_c):+.3f}  "
          f"corr(age,sex)={_corr(age_c, sex_c):+.3f}  corr(age,age*sex)={_corr(age_c, inter_c):+.3f}")
    print(f"  wrote {design}  ({n} rows x 5 cols)")
    return age_c


def main():
    con = BASE / "HCP" / "contrast_agesex.txt"
    with open(con, "w") as f:
        for row, _ in CONTRASTS:
            f.write(row + "\n")
    print("contrast -> tstat mapping:")
    for i, (row, desc) in enumerate(CONTRASTS, 1):
        print(f"  tstat{i}: {desc:28s} [{row}]")
    print(f"  wrote {con}\n")

    for bval in (1500, 3000):
        build(bval)


if __name__ == "__main__":
    main()
