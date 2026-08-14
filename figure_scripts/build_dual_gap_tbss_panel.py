"""Build the dual-gap + FA TBSS panel: signal-domain J_dir vs diffusivity-domain
J_ln_D vs FA (DTI reference) age contrasts, on the n=1379 HCP-A skeleton.

Maps come from the sex-adjusted "agesex" TBSS design (age + age^2 + sex +
age*sex). The reported contrast is the sex-adjusted LINEAR AGE effect:
  tstat1 = +age  (red),  tstat2 = -age  (blue),  both at FWE p<0.05 (corrp>0.95).
(Robustness: adding acquisition site + head motion + eddy outlier fraction, the
full 10-EV maps are near-identical, Dice 0.98-1.00; see compare_agesex_full.py.)
Skeleton trace yellow on the mean_FA background. Voxels dilated for visibility.

FA is the standard DTI anisotropy metric and is rendered as a comparison row.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

BASE = Path(__file__).resolve().parent
STATS = {b: BASE / f"tbss_age_b{b}_n1379" / "stats" for b in (1500, 3000)}
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

SLICES = [60, 70, 80, 90, 100, 110]  # axial z in FMRIB58 1mm
THR = 0.95  # FWE p < 0.05


def load(p):
    return nib.load(str(p)).get_fdata()


def overlay_mask(t1_corrp, t2_corrp, skel, dilate_iter=2):
    sig_pos = (t1_corrp > THR) & (skel > 0)
    sig_neg = (t2_corrp > THR) & (skel > 0)
    sig_pos_d = binary_dilation(sig_pos, iterations=dilate_iter)
    sig_neg_d = binary_dilation(sig_neg, iterations=dilate_iter)
    skel_d = binary_dilation(skel > 0, iterations=1)
    return sig_pos_d, sig_neg_d, skel_d


def ax_slice(a3, z):
    return np.flipud(a3[:, :, z].T)  # radiological-ish display


NOTANA = "#8a6bbf"  # skeleton present but NOT analysed (failed the tract screen)


def paint(ax, m3, z, color, alpha):
    m = ax_slice(m3, z)
    if m.any():
        ax.imshow(np.ma.masked_where(~m, m.astype(float)),
                  cmap=mcolors.ListedColormap([color]), vmin=0, vmax=1,
                  alpha=alpha, interpolation="nearest")


SKEL_CONV = "#00b400"  # conventional TBSS renders the analysed skeleton in green


def render_row(axes_row, tmpl, trng, notana, nonsig, pos, neg, ylabel,
               with_titles=False, conventional=False):
    for col, z in enumerate(SLICES):
        ax = axes_row[col]
        ax.imshow(ax_slice(tmpl, z), cmap="gray", vmin=trng[0], vmax=trng[1])
        if conventional:
            # Standard TBSS presentation: the analysed skeleton over mean FA,
            # with no layer for voxels outside the analysis domain. The skeleton
            # IS the domain, so marking its complement is not the convention.
            # Coverage and the tract screen are reported in the Methods instead.
            paint(ax, nonsig, z, SKEL_CONV, 0.55)
        else:
            paint(ax, nonsig, z, "#c9c900", 0.40)   # analysed, n.s.
            paint(ax, notana, z, NOTANA, 0.55)      # excluded skeleton
        paint(ax, pos, z, "#ff2020", 0.95)      # +age
        paint(ax, neg, z, "#2a6bff", 0.95)      # -age
        ax.set_xticks([]); ax.set_yticks([])
        if with_titles:
            ax.set_title(f"z={z}", fontsize=8)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=10)


def panel(bval, conventional=False):
    S = STATS[bval]
    cp = lambda m, t: load(S / f"{m}_agesex_tfce_corrp_tstat{t}.nii.gz")  # 1=+age, 2=-age
    # whole-brain anatomical background so results are anatomically locatable
    tmpl = load(BASE / "atlases" / "FMRIB58_FA_1mm.nii.gz")
    trng = (0, np.percentile(tmpl[tmpl > 0.05], 96))
    good = load(S.parent / "skeleton_screened_mask.nii.gz") > 0  # analysed (screened)
    # mean_FA_skeleton_mask holds 137,832 voxels, but mean_FA is ZERO at ~44% of
    # them, so those were never real skeleton for this cohort. Using the raw mask
    # as the baseline inflates the excluded layer about sevenfold with template
    # artifact. Restrict to voxels that carry actual cohort data.
    full = (load(S / "mean_FA_skeleton_mask.nii.gz") > 0) & (load(S / "mean_FA.nii.gz") > 0)
    notana = binary_dilation(full & ~good, iterations=1)

    rows = []
    for metric, label in (("J_dir", r"$J_{\mathrm{dir}}$ (signal)"),
                          ("J_ln_D", r"$J_{\ln}$ (diffusivity)"),
                          ("FA", "FA (DTI ref.)")):
        # STATISTICS: raw corrp on the screened skeleton.
        sp = (cp(metric, 1) > THR) & good
        sn = (cp(metric, 2) > THR) & good
        # DISPLAY: tbss_fill, matching the beyond-tensor panel and what the
        # caption states. Isotropic binary_dilation was used here previously,
        # which can bleed across tract boundaries into CSF and gray matter;
        # tbss_fill grows the result along local tract structure instead.
        pos_d = load(S.parent / "filled" / f"{metric}_agesex_1_fill.nii.gz") > 0
        neg_d = load(S.parent / "filled" / f"{metric}_agesex_2_fill.nii.gz") > 0
        rows.append((binary_dilation(good & ~sp & ~sn, iterations=1),
                     pos_d, neg_d, f"{label}\n$b={bval}$"))

    fig, axes = plt.subplots(len(rows), len(SLICES),
                             figsize=(2.15 * len(SLICES), 2.4 * len(rows)))
    for r, (nonsig, pos, neg, ylabel) in enumerate(rows):
        render_row(axes[r], tmpl, trng, notana, nonsig, pos, neg, ylabel,
                   with_titles=(r == 0), conventional=conventional)
    from matplotlib.patches import Patch
    if conventional:
        handles = [Patch(fc="#ff2020", label="+age (FWE $p{<}0.05$)"),
                   Patch(fc="#2a6bff", label="$-$age"),
                   Patch(fc=SKEL_CONV, label="skeleton, n.s.")]
    else:
        handles = [Patch(fc="#ff2020", label="+age (FWE $p{<}0.05$)"),
                   Patch(fc="#2a6bff", label="$-$age"),
                   Patch(fc="#c9c900", label="analysed, n.s."),
                   Patch(fc=NOTANA, label="skeleton not analysed")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


def main():
    missing = [S / f"{m}_agesex_tfce_corrp_tstat1.nii.gz"
               for b, S in STATS.items() for m in ("J_dir", "J_ln_D", "FA")
               if not (S / f"{m}_agesex_tfce_corrp_tstat1.nii.gz").exists()]
    if missing:
        raise SystemExit("agesex maps not ready yet (randomise still running?):\n  "
                         + "\n  ".join(str(p) for p in missing))
    # Two presentations of the identical statistics:
    #   screened      whole brain, excluded skeleton painted (current figures)
    #   conventional  analysed skeleton over mean FA, no exclusion layer
    for conv in (False, True):
        suffix = "_conv" if conv else ""
        for bval in (1500, 3000):
            fig = panel(bval, conventional=conv)
            for ext in ("png", "pdf"):
                out = FIG / f"tbss_dual_gap_b{bval}{suffix}.{ext}"
                fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {FIG / f'tbss_dual_gap_b{bval}{suffix}'}.{{png,pdf}}")

    PAPER_FIG = BASE.parent / "paper-jensen-gap" / "figures"
    if PAPER_FIG.exists():
        import shutil
        for bval in (1500, 3000):
            for ext in ("png", "pdf"):
                src = FIG / f"tbss_dual_gap_b{bval}.{ext}"
                shutil.copy2(src, PAPER_FIG / src.name)
        print("  copied into paper-jensen-gap/figures")


if __name__ == "__main__":
    main()
