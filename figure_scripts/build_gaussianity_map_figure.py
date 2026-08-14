"""Voxelwise non-Gaussianity maps for the manuscript's b-signature section.

The manuscript establishes the b-signature at ROI scale: the diffusivity gap is
b-invariant under Gaussian diffusion (Prop. 6.1), it falls 22.5% from b=1500 to
b=3000 in 16/16 ROIs, and that shift attenuates with age (r=+0.53 in CR_all).
It shows no MAP, and this figure explains why one was missing and supplies it.

Delta_b J_ln = J_ln(b2) - J_ln(b1) contrasts two already-AGGREGATED gaps. The
shells are ~0.97 correlated, so the shared anatomy cancels and the residual
fights unchanged noise. Averaging thousands of voxels inside an ROI suppresses
that, which is why the ROI-scale result is solid, but the same contrast is too
noisy to read voxel by voxel.

Contrasting PER DIRECTION first and aggregating second removes the
cancellation. Applying the paper's own Jensen construction to the per-direction
attenuation ratios f_i = D_i(b2)/D_i(b1):

    J_f = ln<f> - <ln f>   >= 0

J_f is zero exactly when attenuation is direction-independent, i.e. exactly the
b-flatness of Prop. 6.1. Same test, same framework, no fit, no b=0, but it
survives at voxel scale.

Panel (c) is the DKI comparison. Sec. 10.2 argues the gap b-signature and DKI
are complementary probes; this quantifies that instead of asserting it.

CAVEAT on the DKI side: this acquisition has two non-zero shells, the MINIMUM
for DKI, so MK here is less well conditioned than a 3-shell protocol would give.
It is a reference, not an optimised DKI analysis.

Outputs:
  figures/gaussianity_maps.{png,pdf}
  HCP/voxel_maps/<session>_Jf_attenuation_gap.nii.gz   (regenerated, mutual match)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LogNorm
from scipy import stats

import os
import sys

# Reuse the cohort script's constants and direction matching so the published
# map cannot drift from the numbers computed across the cohort.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gap_computation"))
import batch_jf_hcp as bj  # noqa: E402

DATA = Path(os.environ.get("JG_DATA", Path(__file__).resolve().parent))
VOX = DATA / "HCP" / "voxel_maps"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

SESSION = "session_20260222_031250"     # same participant as Figure 1
SLICE_Z = 60


def axial(a, z):
    return np.flipud(a[:, :, z].T)


def cmap_black(name):
    c = plt.get_cmap(name).copy()
    c.set_bad("black")
    return c


def compute_maps():
    """J_f, Delta_b J_ln and ADC_1, using batch_jf_hcp's exact conventions."""
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

    v1 = bvecs[sel[bj.B1]]
    v2 = bvecs[sel[bj.B2]]
    v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    cos = np.abs(v1 @ v2.T)
    j, back = cos.argmax(1), cos.argmax(0)
    idx = np.arange(len(v1))
    keep = (cos[idx, j] > bj.COS_MIN) & (back[j] == idx)
    print(f"  matched directions: {int(keep.sum())}")

    D1 = bj._shell_D(dwi, bvals, sel[bj.B1], b0_mean)[..., keep]
    D2 = bj._shell_D(dwi, bvals, sel[bj.B2], b0_mean)[..., j[keep]]

    f = D2 / D1
    J_f = np.maximum(np.log(np.clip(f.mean(-1), 1e-12, None)) - np.log(f).mean(-1), 0.0)

    def gap(D):
        return np.maximum(np.log(np.clip(D.mean(-1), 1e-12, None)) - np.log(D).mean(-1), 0.0)

    delta = gap(D2) - gap(D1)
    return J_f, delta, D1.mean(-1), img


def main():
    print("computing maps...", flush=True)
    J_f, delta, adc1, img = compute_maps()

    nib.save(nib.Nifti1Image(np.nan_to_num(J_f).astype(np.float32), img.affine,
                             img.header),
             str(VOX / f"{SESSION}_Jf_attenuation_gap.nii.gz"))

    mk = nib.load(str(VOX / f"{SESSION}_DKI_MK.nii.gz")).get_fdata().astype(float)

    # WM mask chosen on ADC and MK validity, independent of both contrasts
    wm = (adc1 > 0.2e-3) & (adc1 < 1.1e-3) & (mk > 0) & np.isfinite(J_f) & np.isfinite(delta)
    print(f"  WM voxels: {int(wm.sum()):,}")

    d, jf, k = delta[wm], J_f[wm], mk[wm]
    r_jf, p_jf = stats.pearsonr(jf, k)
    rho_jf, _ = stats.spearmanr(jf, k)
    r_d, _ = stats.pearsonr(d, k)
    rho_d, _ = stats.spearmanr(d, k)
    r_jd, _ = stats.pearsonr(jf, d)
    print(f"  J_f      vs MK : r={r_jf:+.3f}  rho={rho_jf:+.3f}  (p={p_jf:.1e})")
    print(f"  Delta_b  vs MK : r={r_d:+.3f}  rho={rho_d:+.3f}")
    print(f"  J_f  vs Delta_b: r={r_jd:+.3f}")
    print(f"  J_f  median {np.median(jf):.5f}  IQR "
          f"[{np.percentile(jf,25):.5f},{np.percentile(jf,75):.5f}]")

    # --- partial-volume control ------------------------------------------------
    # J_f is brightest periventricularly, which is also where CSF partial volume
    # sits. PV RAISES ADC and LOWERS FA, so a PV artefact would make J_f grow with
    # ADC and shrink in dense WM. Both go the other way, so the pattern is tissue.
    fa = nib.load(str(VOX / f"{SESSION}_DKI_FA_dki.nii.gz")).get_fdata().astype(float)
    a = adc1[wm]
    print(f"  PV control: rho(J_f, ADC) = {stats.spearmanr(jf, a).statistic:+.3f} "
          f"(PV would be positive)")
    rho_hi = float("nan")
    for th in (0.0, 0.3, 0.5):
        m = wm & (fa > th)
        rr = stats.spearmanr(J_f[m], mk[m]).statistic
        print(f"    FA>{th:.1f}  n={int(m.sum()):7,}  J_f median "
              f"{np.median(J_f[m]):.5f}  rho(J_f,MK)={rr:+.3f}")
        if th == 0.5:
            rho_hi = rr

    bm = axial(wm, SLICE_Z)
    dz = np.ma.masked_where(~bm, axial(delta, SLICE_Z))
    jz = np.ma.masked_where(~bm, axial(J_f, SLICE_Z))
    kz = np.ma.masked_where(~bm, axial(mk, SLICE_Z))
    dlim = float(np.nanpercentile(np.abs(dz.compressed()), 98))
    jhi = float(np.nanpercentile(jz.compressed(), 98))
    khi = float(np.nanpercentile(kz.compressed(), 98))

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6))
    panels = [
        (axes[0, 0], dz, cmap_black("RdBu_r"), -dlim, dlim,
         r"(a) $\Delta_b J_{\ln}$, voxelwise",
         "contrast of aggregates: noise-dominated"),
        (axes[0, 1], jz, cmap_black("inferno"), 0.0, jhi,
         r"(b) $J_f = \ln\langle f\rangle - \langle \ln f\rangle$",
         "same $b$-flatness null, contrasted per direction"),
        (axes[1, 0], kz, cmap_black("inferno"), 0.0, khi,
         "(c) DKI mean kurtosis", "reference, two-shell fit"),
    ]
    for ax, im_, cm, lo, hi, title, note in panels:
        h = ax.imshow(im_, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(note, fontsize=8.5, color="0.3")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.02)

    # hexbin renders as slivers here because x and y differ by ~100x in scale,
    # so use an explicit 2D histogram with a log colour scale instead.
    ax = axes[1, 1]
    jhi_s = float(np.percentile(jf, 99.5))
    ax.hist2d(np.clip(k, 0, 2.0), np.clip(jf, 0, jhi_s),
              bins=[np.linspace(0, 2.0, 90), np.linspace(0, jhi_s, 90)],
              norm=LogNorm(), cmap="Blues")
    ax.set_xlabel("DKI mean kurtosis")
    ax.set_ylabel(r"$J_f$")
    ax.set_title(f"(d) $J_f$ vs DKI in white matter\n"
                 rf"Spearman $\rho={rho_jf:+.2f}$ (all WM), "
                 rf"$\rho={rho_hi:+.2f}$ (FA$>$0.5)", fontsize=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        try:
            fig.savefig(FIG / f"gaussianity_maps.{ext}", dpi=200, bbox_inches="tight")
        except PermissionError:
            print(f"  [skip] locked: gaussianity_maps.{ext}")
    plt.close(fig)
    print(f"wrote {FIG / 'gaussianity_maps'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
