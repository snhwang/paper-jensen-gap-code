"""Residualize a skeletonised metric on the voxelwise diffusion tensor.

The "beyond-tensor" test: at each skeleton voxel, regress the metric across
subjects on [1, FA, MD, AD, RD] and keep the residual. The residual 4D is then
fed to randomise (dgx_randomise_beyondtensor.sh), so any age effect that
survives is one the tensor scalars cannot account for.

Used for both gaps -- J_ln_D (diffusivity domain) and J_dir (signal domain,
= J_ln_S) -- and mirrors the identical test applied to the info-theory
descriptors in the companion paper, reusing the SAME skeleton_good_mask.nii.gz
(the usable-skeleton subset) so all metrics are voxel-for-voxel comparable.

Note the design is per-voxel: FA/MD/AD/RD vary voxel to voxel, so this is
V separate 1379x5 regressions, not one shared fit. Solved in 3,000-voxel chunks
with a 1e-5 ridge on ZtZ for conditioning (FA/MD/AD/RD are collinear -- all are
functions of the same three eigenvalues -- so ZtZ is near-singular by
construction; the ridge makes the solve stable without materially changing the
residual).

Usage: python make_resid_tensor.py <tbss_dir> <metric>
  e.g. python make_resid_tensor.py ~/tbss_age_b1500_n1379 J_ln_D
Writes <tbss_dir>/resid_<metric>_tensor.nii.gz
"""
import sys

import numpy as np
import nibabel as nib

if len(sys.argv) != 3:
    sys.exit(__doc__)
D, METRIC = sys.argv[1], sys.argv[2]
S = D + "/stats"

mimg = nib.load(D + "/skeleton_good_mask.nii.gz")
mask = np.asarray(mimg.dataobj) > 0
V = int(mask.sum())


def load(name):
    """Skeleton-masked (V, nsubj) float32 for all_<name>_skeletonised.nii.gz."""
    a = np.asarray(nib.load(f"{S}/all_{name}_skeletonised.nii.gz").dataobj, dtype=np.float32)
    o = a[mask]
    del a
    return o


C = load(METRIC)
FA, MD, AD, RD = load("FA"), load("MD"), load("AD"), load("RD")
n = C.shape[1]
print(f"n={n} good voxels={V} {METRIC} std>0 in {int((C.std(1) > 0).sum())}", flush=True)

resid = np.zeros_like(C)
one = np.ones(n, np.float32)
for s in range(0, C.shape[0], 3000):
    e = min(s + 3000, C.shape[0])
    c = e - s
    Z = np.empty((c, n, 5), np.float32)
    Z[:, :, 0] = one
    Z[:, :, 1] = FA[s:e]
    Z[:, :, 2] = MD[s:e]
    Z[:, :, 3] = AD[s:e]
    Z[:, :, 4] = RD[s:e]
    ZtZ = np.einsum('cni,cnj->cij', Z, Z) + 1e-5 * np.eye(5, dtype=np.float32)[None]
    b = np.linalg.solve(ZtZ, np.einsum('cni,cn->ci', Z, C[s:e])[..., None])[..., 0]
    resid[s:e] = C[s:e] - np.einsum('cni,ci->cn', Z, b)

out = np.zeros(mask.shape + (n,), np.float32)
out[np.where(mask)] = resid
nib.save(nib.Nifti1Image(out, mimg.affine, mimg.header), f"{D}/resid_{METRIC}_tensor.nii.gz")
print(f"wrote resid_{METRIC}_tensor.nii.gz", flush=True)
