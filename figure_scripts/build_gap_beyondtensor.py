"""Beyond-tensor TBSS montage for the directional Jensen gaps (b=1500 + b=3000).

Companion to build_dual_gap_tbss_panel.py, in the same visual style (mean-FA
background, same axial slices, yellow skeleton, red=+age / blue=-age), so the
raw age maps and the tensor-residualized maps can be read side by side.

After FA/MD/AD/RD are regressed out voxelwise (make_resid_tensor.py) the
linear-age contrast is run on the residual; inference is restricted to the
screened skeleton (usable minus screen-failing tracts). One row per gap x shell; rows whose randomise maps are not yet
on disk are skipped, so this renders the diffusivity gap alone until the
signal-gap run lands, then both.

Reads, per shell, from tbss_age_b{bval}_n1379/ (inference restricted to the
screened skeleton: usable minus tracts failing the coverage/age-bias screen):
    {jlnd,jdir}_rig_tfce_corrp_tstat{1,2}.nii.gz   (dgx_randomise_beyondtensor.sh)
    skeleton_screened_mask.nii.gz
    stats/mean_FA.nii.gz
Writes figures/tbss_beyondtensor_gap.{png,pdf} and copies into ../paper-jensen-gap/figures.
"""
from pathlib import Path

import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
SLICES = [60, 70, 80, 90, 100, 110]  # axial z in FMRIB58 1mm, matching the dual-gap panel
THR = 0.95  # FWE p < 0.05

# Presentation only; the statistics are identical either way.
#   False  whole brain with the excluded skeleton painted
#   True   conventional TBSS: analysed skeleton over mean FA, no exclusion layer
CONVENTIONAL = bool(int(os.environ.get("TBSS_CONVENTIONAL", "0")))
SKEL_CONV = "#00b400"

# (randomise prefix, row label, shell); rendered only if the maps exist.
ROWS = [
    ("jlnd_rig", r"$J_{\ln}$ (diffusivity)", 1500),
    ("jlnd_rig", r"$J_{\ln}$ (diffusivity)", 3000),
    ("jdir_rig", r"$J_{\mathrm{dir}}$ (signal)", 1500),
    ("jdir_rig", r"$J_{\mathrm{dir}}$ (signal)", 3000),
]


def load(p):
    return nib.load(str(p)).get_fdata()


def ax_slice(a3, z):
    return np.flipud(a3[:, :, z].T)


def tdir(bval):
    return BASE / f"tbss_age_b{bval}_n1379"


def present(prefix, bval):
    d = tdir(bval)
    return (d / f"{prefix}_tfce_corrp_tstat1.nii.gz").exists() and \
           (d / f"{prefix}_tfce_corrp_tstat2.nii.gz").exists()


def overlay(ax, mask3d, z, color, alpha):
    m = ax_slice(mask3d, z)
    if m.any():
        ax.imshow(np.ma.masked_where(~m, m.astype(float)),
                  cmap=mcolors.ListedColormap([color]), vmin=0, vmax=1,
                  alpha=alpha, interpolation="nearest")


def main():
    rows = [r for r in ROWS if present(r[0], r[2])]
    if not rows:
        raise SystemExit("no beyond-tensor maps found (randomise still running?)")
    # whole-brain anatomical background (standard FMRIB58 FA template)
    tmpl = load(BASE / "atlases" / "FMRIB58_FA_1mm.nii.gz")
    trng = (0, np.percentile(tmpl[tmpl > 0.05], 96))
    fig, axes = plt.subplots(len(rows), len(SLICES),
                             figsize=(2.15 * len(SLICES), 2.4 * len(rows)),
                             squeeze=False)
    NOTANA = "#8a6bbf"  # muted purple = skeleton present but NOT analysed (excluded)

    for ri, (prefix, label, bval) in enumerate(rows):
        d = tdir(bval)
        good = load(d / "skeleton_screened_mask.nii.gz") > 0            # analysed (screened)
        # mean_FA is ZERO at ~44% of mean_FA_skeleton_mask, so those voxels were
        # never real skeleton for this cohort. Using the raw mask as the baseline
        # inflates the excluded layer about sevenfold with template artifact.
        full = ((load(d / "stats" / "mean_FA_skeleton_mask.nii.gz") > 0)
                & (load(d / "stats" / "mean_FA.nii.gz") > 0))
        notana = full & ~good                                          # real skeleton not analysed
        # STATISTICS come from the raw corrp on the screened skeleton -- these
        # are the percentages the manuscript reports.
        sig_pos = (load(d / f"{prefix}_tfce_corrp_tstat1.nii.gz") > THR) & good
        sig_neg = (load(d / f"{prefix}_tfce_corrp_tstat2.nii.gz") > THR) & good
        ng = int(good.sum())
        print(f"{label} b{bval}: -age {100 * int(sig_neg.sum()) / ng:.1f}%  "
              f"+age {100 * int(sig_pos.sum()) / ng:.1f}%  (analysed {ng:,})")

        # DISPLAY uses tbss_fill (FSL's sanctioned thickening: grows along tract
        # structure rather than isotropically, so it cannot bleed into CSF/GM).
        # It deliberately covers more voxels than the inference, so it must not
        # be used for the statistics above.
        pos_d = load(d / "filled" / f"{prefix}_1_fill.nii.gz") > 0
        neg_d = load(d / "filled" / f"{prefix}_2_fill.nii.gz") > 0
        nonsig = binary_dilation(good & ~sig_pos & ~sig_neg, iterations=1)
        notana_d = binary_dilation(notana, iterations=1)

        for ci, z in enumerate(SLICES):
            ax = axes[ri][ci]
            ax.imshow(ax_slice(tmpl, z), cmap="gray", vmin=trng[0], vmax=trng[1])
            if CONVENTIONAL:
                # Standard TBSS presentation: the analysed skeleton over mean FA,
                # with no layer for voxels outside the analysis domain. Coverage
                # and the tract screen are reported in the Methods instead.
                overlay(ax, nonsig, z, SKEL_CONV, 0.55)
            else:
                overlay(ax, nonsig, z, "#c9c900", 0.40)  # analysed, n.s.
                overlay(ax, notana_d, z, NOTANA, 0.55)   # excluded skeleton
            overlay(ax, pos_d, z, "#ff2020", 0.95)   # +age
            overlay(ax, neg_d, z, "#2a6bff", 0.95)   # -age
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"z={z}", fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{label}\n$b={bval}$", fontsize=10)

    # legend
    from matplotlib.patches import Patch
    if CONVENTIONAL:
        handles = [Patch(fc="#ff2020", label="+age (FWE $p{<}0.05$)"),
                   Patch(fc="#2a6bff", label="$-$age"),
                   Patch(fc=SKEL_CONV, label="skeleton, n.s.")]
    else:
        handles = [Patch(fc="#ff2020", label="+age (FWE $p{<}0.05$)"),
                   Patch(fc="#2a6bff", label="$-$age"),
                   Patch(fc="#d8d800", label="analysed, n.s."),
                   Patch(fc=NOTANA, label="skeleton not analysed")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    stem = "tbss_beyondtensor_gap" + ("_conv" if CONVENTIONAL else "")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / stem}.{{png,pdf}}")

    # Only the default style is copied into the manuscript. The conventional
    # variant is written locally for comparison and is promoted deliberately,
    # not as a side effect of rendering it.
    paper_fig = BASE.parent / "paper-jensen-gap" / "figures"
    if paper_fig.exists() and not CONVENTIONAL:
        import shutil
        for ext in ("png", "pdf"):
            shutil.copy2(FIG / f"tbss_beyondtensor_gap.{ext}",
                         paper_fig / f"tbss_beyondtensor_gap.{ext}")
        print("  copied into paper-jensen-gap/figures")


if __name__ == "__main__":
    main()
