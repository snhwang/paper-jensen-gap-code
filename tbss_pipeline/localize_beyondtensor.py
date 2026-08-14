"""Summarize beyond-tensor randomise output.

Reports, per age direction: the number of FWE p<0.05 voxels, that count as a
percentage of the usable skeleton, max(1-p), the survivor partial-r
(median/max), and the top JHU white-matter labels.

df = N - 14: 10 EVs in the full design plus the 5 tensor regressors
[1,FA,MD,AD,RD] removed by make_resid_tensor.py, less the shared intercept.

Usage: python localize_beyondtensor.py <tbss_dir> <prefix> [metric_label]
  e.g. python localize_beyondtensor.py ~/tbss_age_b1500_n1379 jlnd_rig J_ln_D
"""
import collections
import os
import sys

import numpy as np
import nibabel as nib

if len(sys.argv) < 3:
    sys.exit(__doc__)
D, PREFIX = sys.argv[1], sys.argv[2]
LABEL = sys.argv[3] if len(sys.argv) > 3 else PREFIX

FSLDIR = os.environ.get("FSLDIR", os.path.expanduser("~/fsl"))
JHU = f"{FSLDIR}/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz"
lab = np.asarray(nib.load(JHU).dataobj).astype(int)

good = np.asarray(nib.load(D + "/skeleton_good_mask.nii.gz").dataobj) > 0
ng = int(good.sum())

resid = f"{D}/resid_{LABEL}_tensor.nii.gz"
n = int(nib.load(resid).shape[3]) if os.path.exists(resid) else 1379
df = n - 14

for c, direction in [(1, "+age"), (2, "-age")]:
    cp = np.asarray(nib.load(f"{D}/{PREFIX}_tfce_corrp_tstat{c}.nii.gz").dataobj)
    t = np.asarray(nib.load(f"{D}/{PREFIX}_tstat{c}.nii.gz").dataobj)
    sig = cp > 0.95
    ns = int(sig.sum())
    print(f"\n=== {direction} (residual {LABEL}): FWE p<0.05 voxels = {ns} "
          f"({100 * ns / ng:.1f}% of {ng} usable)  max(1-p)={cp.max():.4f}")
    if ns:
        ar = np.abs(t[sig] / np.sqrt(t[sig] ** 2 + df))
        print(f"    survivor partial-r median={np.median(ar):.3f} max={ar.max():.3f}")
        ls = lab[sig]
        for lv, ct in collections.Counter(ls[ls > 0]).most_common(6):
            print(f"      label {lv}: {ct}")
