"""
Build the N=1379 one-visit-per-subject manifest for TBSS at b=1500 (or 3000).

For each unique HCP-A subject:
  1. Pick V1 if its cache_b{bval} is complete; else V2/V3/V4 in order.
  2. Record (subject_id, visit, session_id, age, sex).

Outputs (per bval):
  HCP/manifest_n1379_b{bval}.tsv     (subject_id, session_id, visit, age, sex)
  HCP/design_n1379_b{bval}.txt       (Text2Vest input: intercept + demeaned age)
  HCP/contrast_n1379.txt             (Text2Vest input: +age, -age contrasts)
  HCP/sessions_rsync_b{bval}.txt     (one session_id per line, for rsync filter)

The subject order in the manifest IS the order in which the TBSS 4D will be
built (alphabetical by subject_id), so design.mat row k matches volume k.
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("DTI_OUTPUT_DIR", str(Path("Q:/dti_output"))))

EXPECTED_CACHE_FILES = [
    "FA.nii.gz", "MD.nii.gz", "AD.nii.gz", "RD.nii.gz",
    "ADC_lin.nii.gz", "ADC_geo.nii.gz", "J_dir.nii.gz", "H_ang_norm.nii.gz",
    "R_aniso.nii.gz", "H_SH.nii.gz", "H_aniso.nii.gz", "CDCI.nii.gz",
    "H_W.nii.gz", "H_scale.nii.gz",
    "gamma_avg.nii.gz", "gamma_scale0.nii.gz",
    "gamma_scale1.nii.gz", "gamma_scale2.nii.gz", "PR_pos_avg.nii.gz",
]


def cache_complete(session_id, bval):
    d = OUTPUT_DIR / session_id / "processed" / f"cr_metric_cache_b{bval}"
    if not d.is_dir():
        return False
    return all((d / f).exists() for f in EXPECTED_CACHE_FILES)


def visit_key(v):
    return int(v[1:]) if v[1:].isdigit() else 99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bval", type=int, required=True, choices=[1500, 3000])
    args = ap.parse_args()
    bval = args.bval

    csv_path = BASE / "HCP" / f"cr_metrics_long_b{bval}.csv"

    # Group visits by subject; keep age + sex
    visits_by_subj = defaultdict(list)
    age_sex_by_visit = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row.get("FA_mean") or row.get("error"):
                continue
            sid = row["Subject_ID"]
            visit = row["Visit"]
            session = row["Session_ID"]
            try:
                age = float(row["Age"].split()[0].rstrip("+"))
            except (ValueError, AttributeError):
                continue
            sex = row.get("Sex", "")
            k = (sid, visit, session)
            if k not in age_sex_by_visit:
                age_sex_by_visit[k] = (age, sex)
                visits_by_subj[sid].append((visit, session))

    # Pick one visit per subject (V1 preferred, fallback to next available with complete cache)
    chosen = []
    skipped_no_cache = []
    for sid, vs in sorted(visits_by_subj.items()):
        vs_sorted = sorted(vs, key=lambda x: visit_key(x[0]))
        picked = None
        for v, s in vs_sorted:
            if cache_complete(s, bval):
                picked = (v, s)
                break
        if picked is None:
            skipped_no_cache.append(sid)
            continue
        v, s = picked
        age, sex = age_sex_by_visit[(sid, v, s)]
        chosen.append((sid, s, v, age, sex))

    n = len(chosen)
    if n == 0:
        print("ERROR: zero chosen subjects -- caches may not be ready yet")
        return

    mean_age = sum(c[3] for c in chosen) / n
    print(f"b={bval}: {n} unique subjects with complete cache "
          f"(skipped {len(skipped_no_cache)} without any cached visit)")
    print(f"  mean age = {mean_age:.4f}  range = {min(c[3] for c in chosen):.1f}..{max(c[3] for c in chosen):.1f}")
    if skipped_no_cache:
        print(f"  skipped first 5: {skipped_no_cache[:5]}")

    # Write manifest
    manifest_path = BASE / "HCP" / f"manifest_n1379_b{bval}.tsv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["subject_id", "session_id", "visit", "age", "sex"])
        for sid, s, v, age, sex in chosen:
            w.writerow([sid, s, v, age, sex])
    print(f"wrote {manifest_path}")

    # Write design.txt (input to FSL Text2Vest)
    design_path = BASE / "HCP" / f"design_n1379_b{bval}.txt"
    with open(design_path, "w") as f:
        for sid, s, v, age, sex in chosen:
            f.write(f"1\t{age - mean_age:.6f}\n")
    print(f"wrote {design_path}  ({n} rows)")

    # Write contrast.txt (same for both bvals -- two contrasts: +age, -age)
    contrast_path = BASE / "HCP" / "contrast_n1379.txt"
    with open(contrast_path, "w") as f:
        f.write("0 1\n0 -1\n")
    print(f"wrote {contrast_path}")

    # Write session list for rsync (one session per line)
    sess_path = BASE / "HCP" / f"sessions_rsync_b{bval}.txt"
    with open(sess_path, "w") as f:
        for sid, s, v, age, sex in chosen:
            f.write(f"{s}\n")
    print(f"wrote {sess_path}")


if __name__ == "__main__":
    main()
