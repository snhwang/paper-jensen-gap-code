"""Tract coverage/bias screen, and construction of the screened inference mask.

Why this exists
---------------
TBSS keeps only skeleton voxels present in EVERY participant. At N = 1,379 that
is severe: a voxel present in a fraction p of participants survives with
probability p^N, so deep periventricular and inferior structures are pruned.
Relaxing the criterion is not a remedy, because the missingness is age-dependent
(genu coverage correlates with age at r = +0.24) and randomise treats an absent
voxel as FA = 0 rather than as missing -- admitting partially covered voxels
would let age-related dropout generate spurious age effects.

So instead we (a) screen every JHU label for coverage and coverage-vs-age bias,
(b) name findings only in labels that pass, and (c) remove voxels of labels that
FAIL the screen from the inference mask, giving the "screened skeleton" used for
the voxelwise analyses and figures.

Screen (measured on the registered, pre-skeletonisation maps, so coverage is a
property of the data rather than of the skeleton projection):
    coverage   = mean per-participant fraction of that label's skeleton voxels
                 carrying valid (non-zero, finite) data
    bias       = Pearson r between that per-participant fraction and age
    verdict    ADEQUATE  coverage >= 99.5% and |r| < 0.10
               marginal  coverage >= 97%   and |r| < 0.15
               EXCLUDE   otherwise
Reported tracts additionally require >= 200 skeleton voxels, so that a tract mean
is not carried by a few dozen atlas-boundary voxels. With HCP-A b=1500 this gives
12 reported labels: body of the corpus callosum; bilateral SLF, SCR, PCR, PTR and
cingulate cingulum; and right sagittal stratum.

Outputs
-------
  stdout                              the full 48-label screen table
  <tbss_dir>/skeleton_screened_mask.nii.gz
                                      good mask minus EXCLUDE-verdict labels;
                                      the inference mask + reporting denominator
                                      for build_dual_gap_tbss_panel.py and
                                      build_gap_beyondtensor.py

Note the mask removes only EXCLUDE labels, not "marginal" ones; marginal tracts
stay in the analysis but are reported cautiously. Unlabelled skeleton (~79% of
the good mask, white matter the JHU atlas does not name) is RETAINED -- it passed
the identical all-subjects criterion and is described by location, not discarded
for lacking an atlas name.

Usage:  python screen_tracts.py <tbss_dir> <bval> [--subjects N]
  e.g.  python screen_tracts.py ~/tbss_age_b1500_n1379 1500
"""
import csv
import os
import sys

import numpy as np
import nibabel as nib

# JHU ICBM-DTI-81 label names (1-48)
NAMES = {
    1: "MCP", 2: "PCT", 3: "Genu CC", 4: "Body CC", 5: "Splenium CC", 6: "Fornix",
    7: "CST R", 8: "CST L", 9: "ML R", 10: "ML L", 11: "ICP R", 12: "ICP L",
    13: "SCP R", 14: "SCP L", 15: "CP R", 16: "CP L", 17: "ALIC R", 18: "ALIC L",
    19: "PLIC R", 20: "PLIC L", 21: "RLIC R", 22: "RLIC L", 23: "ACR R", 24: "ACR L",
    25: "SCR R", 26: "SCR L", 27: "PCR R", 28: "PCR L", 29: "PTR R", 30: "PTR L",
    31: "SS R", 32: "SS L", 33: "EC R", 34: "EC L", 35: "Cing(cing) R",
    36: "Cing(cing) L", 37: "Cing(hipp) R", 38: "Cing(hipp) L", 39: "Fx/ST R",
    40: "Fx/ST L", 41: "SLF R", 42: "SLF L", 43: "SFO R", 44: "SFO L",
    45: "UF R", 46: "UF L", 47: "Tapetum R", 48: "Tapetum L",
}
MIN_VOX = 200          # a tract mean needs enough skeleton voxels to be stable
COV_ADEQ, BIAS_ADEQ = 99.5, 0.10
COV_MARG, BIAS_MARG = 97.0, 0.15
N_SAMPLE = 250         # participants sampled for the coverage/bias estimate


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        sys.exit(__doc__)
    D, B = os.path.expanduser(args[0]), args[1]
    n_sample = N_SAMPLE
    if "--subjects" in sys.argv:
        n_sample = int(sys.argv[sys.argv.index("--subjects") + 1])
    home = os.path.expanduser("~")
    fsldir = os.environ.get("FSLDIR", home + "/fsl")

    order = [l.strip() for l in open(f"{home}/jlnd_order_b{B}.txt") if l.strip()]
    ages = {}
    with open(f"{home}/manifest_n1379_b{B}.tsv") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                ages[row["subject_id"]] = float(
                    str(row["age"]).replace("90 or older", "90"))
            except ValueError:
                pass

    jhu = np.asarray(nib.load(
        fsldir + "/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz").dataobj).astype(int)
    skel_img = nib.load(D + "/stats/mean_FA_skeleton_mask.nii.gz")
    skel = np.asarray(skel_img.dataobj) > 0

    ids = [k for k in NAMES if ((jhu == k) & skel).sum() >= 50]
    masks = {k: (jhu == k) & skel for k in ids}
    cov = {k: [] for k in ids}
    age_list = []

    step = max(1, len(order) // n_sample)
    for s in order[::step]:
        f = f"{D}/FA/{s}_FA_FA_to_target.nii.gz"      # registered, pre-skeletonisation
        if not os.path.exists(f) or s not in ages:
            continue
        a = np.asarray(nib.load(f).dataobj)
        for k in ids:
            cov[k].append(100.0 * float((a[masks[k]] > 0).mean()))
        age_list.append(ages[s])
    age = np.array(age_list)
    print(f"screen on {len(age)} sampled participants, age {age.min():.0f}-{age.max():.0f}\n",
          flush=True)

    rows, excluded, reported = [], [], []
    for k in ids:
        c = np.array(cov[k])
        r = 0.0 if c.std() == 0 else float(np.corrcoef(c, age)[0, 1])
        nv = int(masks[k].sum())
        mean_c = float(c.mean())
        if mean_c >= COV_ADEQ and abs(r) < BIAS_ADEQ:
            verdict = "ADEQUATE"
            if nv >= MIN_VOX:
                reported.append(k)
        elif mean_c >= COV_MARG and abs(r) < BIAS_MARG:
            verdict = "marginal"
        else:
            verdict = "EXCLUDE"
            excluded.append(k)
        rows.append((NAMES[k], nv, mean_c, r, verdict))

    print(f"{'tract':16s} {'skelvox':>7s} {'cov%':>6s} {'r(cov,age)':>10s}  verdict")
    for nm, nv, c, r, v in sorted(rows, key=lambda x: (-x[2], abs(x[3]))):
        print(f"{nm:16s} {nv:>7,} {c:>5.1f}% {r:>+10.3f}  {v}")

    print(f"\nreported tracts (ADEQUATE and >= {MIN_VOX} voxels): "
          + ", ".join(NAMES[k] for k in reported))
    print(f"excluded from inference mask ({len(excluded)} labels): "
          + ", ".join(NAMES[k] for k in excluded))

    # screened inference mask = good mask minus EXCLUDE-verdict labels
    good_p = D + "/skeleton_good_mask.nii.gz"
    if not os.path.exists(good_p):
        print(f"\n[skip] {good_p} not found; run make_resid_tensor.py first "
              f"(it writes the good mask). Screen table above is still valid.")
        return
    gimg = nib.load(good_p)
    good = np.asarray(gimg.dataobj) > 0
    screened = good & ~np.isin(jhu, excluded)
    out = D + "/skeleton_screened_mask.nii.gz"
    nib.save(nib.Nifti1Image(screened.astype(np.uint8), gimg.affine, gimg.header), out)
    print(f"\ngood mask {int(good.sum()):,} -> screened {int(screened.sum()):,} "
          f"(removed {int(good.sum() - screened.sum()):,}, "
          f"{100 * (good.sum() - screened.sum()) / good.sum():.1f}%)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
