"""Voxel maps of the example participant for Figures 1 and 2 of the revision.

Recomputes, from the minimally preprocessed DWI of one session and with one set
of definitions, every quantity the two figures show.

  per shell b in {1500, 3000}
    S'_i = clip(S_i / S_0, 1e-2, 1),  D_i = -ln S'_i / b   (b = shell mean)
    ADC_1 = <D>,  ADC_0 = exp<ln max(D, 1e-6)>,  D_SM = -ln<S'> / b
    J_ln  = ln ADC_1 - <ln D>,  J_lnS = ln<S'> - <ln S'>      (clipped at 0)
    tensor (DIPY WLS on b0 + this shell): FA, MD and the gaps of the
    tensor-predicted directional ADCs g_i^T D g_i on the same directions.

Uniform direction weights, as in the cohort analysis. Maps are participant-level
images, so they are written outside the repository.

Usage:  python example_maps.py
"""
import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

from config import DTI, EXAMPLE_SESSION, MAPS

SESSION = EXAMPLE_SESSION
SRC = DTI / SESSION
OUT = MAPS
OUT.mkdir(parents=True, exist_ok=True)

img = nib.load(SRC / "processed" / "dwi_raw.nii.gz")
dwi = np.asarray(img.dataobj, dtype=np.float32)
bvals = np.loadtxt(SRC / "inputs" / "dwi.bval").ravel()
bvecs = np.loadtxt(SRC / "processed" / "dwi_raw.bvec")
if bvecs.shape[0] != 3:
    bvecs = bvecs.T
mask = nib.load(SRC / "processed" / "mask.nii.gz").get_fdata() > 0
aff = img.affine
is_b0 = bvals < 50
S0 = dwi[..., is_b0].mean(-1)
vox = mask & (S0 > 0)
print(f"{vox.sum()} brain voxels, {is_b0.sum()} b0 volumes")


def save(arr, name):
    out = np.zeros(mask.shape, np.float32)
    out[vox] = arr
    nib.save(nib.Nifti1Image(out, aff), OUT / f"{name}.nii.gz")


def gaps(D, S):
    """Return ADC_1, ADC_0, J_ln, J_lnS for rows of directional D and S'."""
    adc1 = D.mean(-1)
    lnD = np.log(np.maximum(D, 1e-6))
    adc0 = np.exp(lnD.mean(-1))
    jln = np.maximum(np.log(np.maximum(adc1, 1e-6)) - lnD.mean(-1), 0)
    jls = np.maximum(np.log(S.mean(-1)) - np.log(S).mean(-1), 0)
    return adc1, adc0, jln, jls


s0v = S0[vox]
for target in (1500, 3000):
    sh = np.abs(bvals - target) < 100
    b = float(bvals[sh].mean())
    g = bvecs[:, sh].T
    S = np.clip(dwi[vox][:, sh] / s0v[:, None], 1e-2, 1.0)
    D = -np.log(S) / b
    adc1, adc0, jln, jls = gaps(D, S)
    dsm = -np.log(S.mean(-1)) / b

    keep = is_b0 | sh
    gt = gradient_table(bvals[keep], bvecs=bvecs[:, keep].T, b0_threshold=50)
    fit = TensorModel(gt).fit(dwi[..., keep], mask=vox)
    q = fit.quadratic_form[vox]                       # (n, 3, 3)
    Dt = np.einsum("ij,njk,ik->ni", g, q, g)          # tensor-predicted D_i
    Dt = np.maximum(Dt, 0)
    St = np.exp(-b * Dt)
    _, _, jln_t, jls_t = gaps(Dt, St)

    for arr, nm in ((adc1, "ADC_1"), (adc0, "ADC_0"), (dsm, "D_SM"), (jln, "J_ln"),
                    (jls, "J_lnS"), (fit.fa[vox], "FA"), (fit.md[vox], "MD"),
                    (jln_t, "J_ln_tensor"), (jls_t, "J_lnS_tensor")):
        save(arr, f"{nm}_b{target}")
    print(f"b={target} (mean {b:.0f}, {sh.sum()} directions) done")

# Atlas white-matter mask for the scatter panels (JHU labels, registered)
lab = SRC / "processed" / "atlas" / "jhu_labels_registered.nii.gz"
wm = (nib.load(lab).get_fdata() > 0) & vox
nib.save(nib.Nifti1Image(wm.astype(np.uint8), aff), OUT / "wm_jhu_mask.nii.gz")
print(f"JHU white matter: {wm.sum()} voxels. Wrote {OUT}")
