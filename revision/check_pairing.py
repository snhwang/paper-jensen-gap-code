"""Does direction mismatch between the shells create J_f, or Delta_b J_ln, under
the Gaussian null? Example participant, atlas white matter.

For each voxel, one Gaussian tensor (fitted to b=0 plus b=1500) predicts the
directional ADC on both shells' directions. Then
  J_f null      = Jensen gap of D_pred(g3_i) / D_pred(g1_i) over the paired
                  directions used for J_f (mutual nearest, |cos| > 0.99)
  Delta_b null  = J_ln of D_pred on the b=3000 set minus on the b=1500 set
Both would be zero if the shells shared their directions exactly.

Usage:  python check_pairing.py
"""

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from scipy.stats import spearmanr

from config import DTI, EXAMPLE_SESSION as SES, MAPS, OUT
from config import OLDMAPS as OLD

SRC = DTI / SES


bvals = np.loadtxt(SRC / "inputs" / "dwi.bval").ravel()
bvecs = np.loadtxt(SRC / "processed" / "dwi_raw.bvec")
if bvecs.shape[0] != 3:
    bvecs = bvecs.T
wm = nib.load(MAPS / "wm_jhu_mask.nii.gz").get_fdata() > 0
dwi = np.asarray(nib.load(SRC / "processed" / "dwi_raw.nii.gz").dataobj, dtype=np.float32)
X = dwi[wm].astype(np.float64)
del dwi
is0 = bvals < 50
i1 = np.where(np.abs(bvals - 1500) < 100)[0]
i3 = np.where(np.abs(bvals - 3000) < 100)[0]

keep = is0 | (np.abs(bvals - 1500) < 100)
fit = TensorModel(gradient_table(bvals[keep], bvecs=bvecs[:, keep].T, b0_threshold=50)).fit(X[:, keep])
Q = fit.quadratic_form


def dpred(idx):
    g = bvecs[:, idx].T
    return np.einsum("ij,njk,ik->ni", g, Q, g)


def jgap(v):
    lv = np.log(np.maximum(v, 1e-9))
    return np.log(np.maximum(v.mean(-1), 1e-9)) - lv.mean(-1)


C = np.abs(bvecs[:, i1].T @ bvecs[:, i3])
b3 = C.argmax(1)
b1 = C.argmax(0)
pairs = np.array([(a, b3[a]) for a in range(len(i1)) if b1[b3[a]] == a and C[a, b3[a]] > 0.99])
ang = np.degrees(np.arccos(np.clip(C[pairs[:, 0], pairs[:, 1]], 0, 1)))
lines = [f"{len(pairs)} pairs used for J_f (mutual nearest, |cos| > 0.99): angle "
         f"{ang.min():.1f} to {ang.max():.1f} deg, median {np.median(ang):.1f} deg"]
print(lines[-1])

D1, D3 = dpred(i1), dpred(i3)
jf_null = jgap(D3[:, pairs[:, 1]] / np.maximum(D1[:, pairs[:, 0]], 1e-9))
dJ_null = jgap(D3) - jgap(D1)
jl1 = jgap(D1)

jf = nib.load(OLD / f"{SES}_Jf_attenuation_gap.nii.gz").get_fdata()[wm]
dJ = (nib.load(MAPS / "J_ln_b3000.nii.gz").get_fdata() - nib.load(MAPS / "J_ln_b1500.nii.gz").get_fdata())[wm]
jl_meas = nib.load(MAPS / "J_ln_b1500.nii.gz").get_fdata()[wm]
ok = np.isfinite(jf) & (jf > 0)
lines.append(f"J_f measured median {np.median(jf[ok]):.4f}; J_f under Gaussian null with "
             f"these pairs median {np.median(jf_null[ok]):.4f} "
             f"({np.median(jf_null[ok] / jf[ok]):.0%} of measured); "
             f"Spearman(J_f, J_f null) {spearmanr(jf[ok], jf_null[ok])[0]:.2f}; "
             f"Spearman(J_f null, FA) {spearmanr(jf_null[ok], fit.fa[ok])[0]:.2f}")
lines.append(f"Delta_b J_ln measured median {np.median(dJ):+.4f} "
             f"({np.median(dJ / jl_meas):+.1%} of J_ln); null from direction sets median "
             f"{np.median(dJ_null):+.5f} ({np.median(dJ_null / jl1):+.2%}), "
             f"IQR {np.percentile(dJ_null / jl1, [25, 75]).round(4)}")
print("\n".join(lines[1:]))
(OUT / "check_pairing.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
