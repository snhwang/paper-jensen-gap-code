"""TBSS-map-derived scalars quoted in the manuscript, computed from the FWE
randomise outputs (Section 7.4 / Figs 5-6). Two quantities:

  (1) Confound robustness -- Dice overlap between the sex-adjusted "agesex" and
      the full 10-EV "full" FWE-significant skeleton masks, per metric and
      contrast. Paper: near-identical, Dice 0.98-1.00. (No atlas needed.)

  (2) Corona-radiata crossing-fiber reversal -- the positive-to-negative ratio
      of FWE-significant skeleton voxels inside the JHU corona radiata (the same
      CR_all ROI as the scalar analysis, JHU-ICBM labels 23-28). Paper (b=1500):
      the signal gap J_dir tips positive-age (~1.2:1) while FA's positive cluster
      is outweighed by negative-age voxels (~0.45:1).

Reads the FWE maps {metric}_{agesex,full}_tfce_corrp_tstat{1,2}.nii.gz and the
skeleton mask from  $JG_DATA/tbss_age_b{bval}_n1379/stats/ .

Env:
  JG_DATA     data root (default: current dir); holds tbss_age_b{bval}_n1379/stats/
  JHU_ATLAS   JHU-ICBM label atlas in FMRIB58 1mm space (for the CR ratios);
              default $FSLDIR/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz. If it
              is missing, the Dice section still runs and the CR section is skipped.
"""
import os
from pathlib import Path

import numpy as np
import nibabel as nib

DATA = Path(os.environ.get("JG_DATA", "."))
THR = 0.95                      # FWE p < 0.05  (1 - p_corr > 0.95)
METRICS = ["J_dir", "J_ln_D", "FA"]
CR_LABELS = [23, 24, 25, 26, 27, 28]   # JHU-ICBM corona radiata (ant/sup/post x R/L)


def stats_dir(bval):
    return DATA / f"tbss_age_b{bval}_n1379" / "stats"


def load(p):
    return nib.load(str(p)).get_fdata() if Path(p).exists() else None


def sig(sdir, metric, design, t):
    a = load(sdir / f"{metric}_{design}_tfce_corrp_tstat{t}.nii.gz")
    return None if a is None else a > THR


def jhu_atlas_path():
    p = os.environ.get("JHU_ATLAS")
    if p:
        return Path(p)
    fsldir = os.environ.get("FSLDIR", "")
    return Path(fsldir) / "data" / "atlases" / "JHU" / "JHU-ICBM-labels-1mm.nii.gz"


def dice_robustness():
    print("=== (1) confound robustness: Dice(agesex, full) ; paper 0.98-1.00 ===")
    for bval in (1500, 3000):
        sdir = stats_dir(bval)
        skel = load(sdir / "mean_FA_skeleton_mask.nii.gz")
        if skel is None:
            print(f"  b={bval}: no skeleton mask, skip")
            continue
        skel = skel > 0
        print(f"  b={bval}:")
        for m in METRICS:
            for t, name in ((1, "+age"), (2, "-age")):
                a, f = sig(sdir, m, "agesex", t), sig(sdir, m, "full", t)
                if a is None or f is None:
                    print(f"    {m:7s} tstat{t} {name:5s}: maps missing")
                    continue
                a, f = a & skel, f & skel
                na, nf = int(a.sum()), int(f.sum())
                d = 2 * int((a & f).sum()) / (na + nf) if (na + nf) else 1.0
                print(f"    {m:7s} tstat{t} {name:5s}: agesex={na:6d} full={nf:6d}  Dice={d:.3f}")


def cr_reversal_ratio():
    atlas = jhu_atlas_path()
    print(f"\n=== (2) corona-radiata pos:neg ratio (agesex, b=1500) ; "
          f"paper J_dir ~1.2:1, FA ~0.45:1 ===")
    if not atlas.exists():
        print(f"  JHU atlas not found ({atlas}); set JHU_ATLAS to enable. Skipping.")
        return
    sdir = stats_dir(1500)
    skel = load(sdir / "mean_FA_skeleton_mask.nii.gz")
    jhu = load(atlas)
    if skel is None or jhu is None:
        print("  skeleton mask or atlas missing, skip")
        return
    skel = skel > 0
    if jhu.shape != skel.shape:
        print(f"  atlas {jhu.shape} != skeleton {skel.shape} (not same space), skip")
        return
    cr = skel & np.isin(jhu, CR_LABELS)
    print(f"  corona radiata (JHU {CR_LABELS}): {int(cr.sum()):,} skeleton voxels")
    for m in METRICS:
        p = sig(sdir, m, "agesex", 1)
        n = sig(sdir, m, "agesex", 2)
        if p is None or n is None:
            print(f"    {m:7s}: maps missing")
            continue
        cp, cn = int((p & cr).sum()), int((n & cr).sum())
        ratio = f"{cp / cn:.2f}:1" if cn else "inf"
        print(f"    {m:7s} CR+={cp:5d} CR-={cn:5d}  pos:neg = {ratio}")


if __name__ == "__main__":
    dice_robustness()
    cr_reversal_ratio()
