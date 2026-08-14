"""Split-half reliability of the voxelwise non-Gaussianity indices.

Both Delta_b J_ln and J_f test the same null (Prop. 6.1 b-flatness). The claim
in the manuscript is that only J_f survives at voxel scale. This measures that
directly and with the FINAL pipeline (mutual nearest-neighbour direction
matching), so the reported numbers match the maps that are actually published.

Design: the matched direction PAIRS are split into two disjoint halves. Each
half yields an independent estimate of each index from independent gradient
directions on the same voxels. The correlation between halves across white
matter voxels is the reliability. Splitting pairs (not volumes) keeps the two
shells matched inside each half, which is what both indices require.

Reported per index: Pearson r and Spearman rho between halves, averaged over
several random splits so the answer does not depend on one arbitrary partition.

Output: printed table.
"""
import numpy as np
import nibabel as nib
from scipy import stats

import os
import sys
from pathlib import Path

# Reuse the cohort script's constants and direction matching, so reliability is
# measured on exactly the pipeline that produces the published maps.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gap_computation"))
import batch_jf_hcp as bj  # noqa: E402

DATA = Path(os.environ.get("JG_DATA", Path(__file__).resolve().parent))
SESSION = "session_20260222_031250"
N_SPLITS = 20
SEED = 0


def gap(D):
    return np.maximum(np.log(np.clip(D.mean(-1), 1e-12, None)) - np.log(D).mean(-1), 0.0)


def jf(D1, D2):
    f = D2 / D1
    return np.maximum(np.log(np.clip(f.mean(-1), 1e-12, None)) - np.log(f).mean(-1), 0.0)


def main():
    src = bj.OUTPUT_DIR / SESSION
    bvals = np.loadtxt(src / "inputs" / "dwi.bval").astype(float)
    bvecs = np.loadtxt(src / "inputs" / "dwi.bvec").astype(float)
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    img = nib.load(str(src / "processed" / "dwi_raw.nii.gz"))
    dwi = img.get_fdata().astype(np.float32)

    r = np.round(bvals / 500) * 500
    sel = {bj.B1: (r == bj.B1), bj.B2: (r == bj.B2)}
    b0_mean = dwi[..., bvals < 50].mean(-1)

    v1, v2 = bvecs[sel[bj.B1]], bvecs[sel[bj.B2]]
    v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    cos = np.abs(v1 @ v2.T)
    j, back = cos.argmax(1), cos.argmax(0)
    idx = np.arange(len(v1))
    keep = (cos[idx, j] > bj.COS_MIN) & (back[j] == idx)
    n = int(keep.sum())
    print(f"matched direction pairs: {n}")

    D1 = bj._shell_D(dwi, bvals, sel[bj.B1], b0_mean)[..., keep]
    D2 = bj._shell_D(dwi, bvals, sel[bj.B2], b0_mean)[..., j[keep]]
    adc1 = D1.mean(-1)

    mk = nib.load(str(DATA / "HCP" / "voxel_maps" /
                      f"{SESSION}_DKI_MK.nii.gz")).get_fdata()
    wm = (adc1 > 0.2e-3) & (adc1 < 1.1e-3) & (mk > 0)
    print(f"WM voxels: {int(wm.sum()):,}\n")

    rng = np.random.default_rng(SEED)
    acc = {"Delta_b_J_ln": [], "J_f": []}
    for _ in range(N_SPLITS):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2: 2 * (n // 2)]

        dA = (gap(D2[..., a]) - gap(D1[..., a]))[wm]
        dB = (gap(D2[..., b]) - gap(D1[..., b]))[wm]
        acc["Delta_b_J_ln"].append((stats.pearsonr(dA, dB).statistic,
                                    stats.spearmanr(dA, dB).statistic))

        jA = jf(D1[..., a], D2[..., a])[wm]
        jB = jf(D1[..., b], D2[..., b])[wm]
        acc["J_f"].append((stats.pearsonr(jA, jB).statistic,
                           stats.spearmanr(jA, jB).statistic))

    print(f"split-half reliability over {N_SPLITS} random splits "
          f"({n // 2} directions per half):")
    print(f"  {'index':16s} {'Pearson r':>18s} {'Spearman rho':>18s}")
    for k, v in acc.items():
        arr = np.array(v)
        print(f"  {k:16s} {arr[:,0].mean():8.3f} +/- {arr[:,0].std():.3f}"
              f" {arr[:,1].mean():10.3f} +/- {arr[:,1].std():.3f}")

    print("\nNote: half-length estimates are noisier than the full-length maps,")
    print("so these are LOWER bounds on the reliability of the published maps.")


if __name__ == "__main__":
    main()
