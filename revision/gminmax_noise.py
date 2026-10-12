"""How much of the extremal rung gap G_min,max = ln(max D / min D) is noise?

Supplementary Section S2 leaves G_min,max out of the spectrum. This script
supports that choice. In the example participant's white matter at b = 1500, it
compares, for G_min,max and for the diffusivity gap J_ln = G_0,1, the measured
value with the value from the fitted tensor alone and from the tensor plus
simulated Rician noise (noise level from the residual of an order-6
spherical-harmonic fit, as in native_pass.py). The noise share is
(tensor+noise - tensor) / measured.

Output:  out/gminmax_noise.txt
Usage:   python gminmax_noise.py
"""
import nibabel as nib
import numpy as np
from dipy.core.geometry import cart2sphere
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.reconst.shm import real_sh_descoteaux

from config import DTI, EXAMPLE_SESSION, MAPS, OUT

src = DTI / EXAMPLE_SESSION
bvals = np.loadtxt(src / "inputs" / "dwi.bval").ravel()
bvecs = np.loadtxt(src / "processed" / "dwi_raw.bvec")
if bvecs.shape[0] != 3:
    bvecs = bvecs.T
wm = nib.load(MAPS / "wm_jhu_mask.nii.gz").get_fdata() > 0
dwi = np.asarray(nib.load(src / "processed" / "dwi_raw.nii.gz").dataobj, dtype=np.float32)
X = dwi[wm].astype(np.float64)
del dwi
is0 = bvals < 50
sh = np.abs(bvals - 1500) < 100
b = float(bvals[sh].mean())
g = bvecs[:, sh].T
S0 = X[:, is0].mean(-1)
_, th, ph = cart2sphere(*g.T)
B = real_sh_descoteaux(6, th, ph, legacy=False)[0]
res = X[:, sh] - X[:, sh] @ np.linalg.pinv(B).T @ B.T
sig = np.sqrt((res ** 2).sum(-1) / (sh.sum() - B.shape[1]))


def gaps(S, s0):
    D = np.maximum(-np.log(np.clip(S / s0[:, None], 1e-2, 1)) / b, 1e-6)
    return np.log(D.mean(-1)) - np.log(D).mean(-1), np.log(D.max(-1) / D.min(-1))


keep = is0 | sh
fit = TensorModel(gradient_table(bvals[keep], bvecs=bvecs[:, keep].T,
                                 b0_threshold=50)).fit(X[:, keep])
Dt = np.maximum(np.einsum("ij,njk,ik->ni", g, fit.quadratic_form, g), 1e-6)
clean = S0[:, None] * np.exp(-b * Dt)
rng = np.random.default_rng(0)
noisy = np.hypot(clean + rng.normal(0, 1, clean.shape) * sig[:, None],
                 rng.normal(0, 1, clean.shape) * sig[:, None])
s0n = S0 + rng.normal(0, 1, S0.shape) * sig / np.sqrt(is0.sum())

meas = gaps(X[:, sh], S0)
tens = gaps(clean, S0)
tn = gaps(noisy, s0n)
ok = (S0 > 0) & (sig > 0)
lines = [f"example participant, white matter, b={b:.0f}, {ok.sum()} voxels"]
for i, name in enumerate(("J_ln = G_0,1", "G_min,max")):
    share = np.median((tn[i][ok] - tens[i][ok]) / meas[i][ok])
    lines.append(f"{name}: median measured {np.median(meas[i][ok]):.4f}, tensor "
                 f"{np.median(tens[i][ok]):.4f}, tensor+noise {np.median(tn[i][ok]):.4f}; "
                 f"noise share of measured value {100 * share:.0f}%")
print("\n".join(lines))
(OUT / "gminmax_noise.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
