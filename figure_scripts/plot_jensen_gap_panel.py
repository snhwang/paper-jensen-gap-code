"""Render Figure 1 for paper-jensen-gap.

Three panels for one representative HCP-A subject (internal de-identified
session session_20260222_031250) at b=1500:
  ADC_1  arithmetic mean of directional diffusivities (clinical trace), GRAYSCALE
  ADC_0  geometric mean of directional diffusivities,                   GRAYSCALE
  J_ln_D = ln(ADC_1/ADC_0)  diffusivity-domain Jensen gap,              COLOR

ADC maps share one grayscale window; the gap is shown in color. Background
outside the brain is black. Writes figures/jensen_gap_adc_panel.{png,pdf} and
copies into paper-jensen-gap/figures/.
"""
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

BASE = Path(__file__).resolve().parent
VOX = BASE / "HCP" / "voxel_maps"
FIG = BASE / "figures"
PAPER_FIG = BASE.parent / "paper-jensen-gap" / "figures"

SESSION = "session_20260222_031250"
BVAL = 1500
SLICE_Z = 60
GAP_CMAP = "viridis"   # gap shown in color (change freely)
GAP_VMAX = 0.10        # display clip for J_ln_D (~p97; makes WM tracts vivid)


def load(name):
    return nib.load(str(VOX / f"{SESSION}_{name}_b{BVAL}.nii.gz")).get_fdata().astype(float)


def axial(a, z):
    return a[:, :, z].T


def cmap_black(name):
    c = plt.get_cmap(name).copy()
    c.set_bad("black")
    return c


def main():
    adc1, adc0, gap = load("ADC_1"), load("ADC_0"), load("J_ln_D")
    brain = adc1 > 0
    # window the ADC maps to the tissue range (clip bright CSF at the top and
    # extend the low end) so white-matter contrast is not crushed into the dark.
    lo, hi = np.percentile(np.concatenate([adc1[brain], adc0[brain]]), [2, 82])
    vmin, vmax = lo - 0.3 * (hi - lo), hi

    bm = axial(brain, SLICE_Z)
    s1 = np.ma.masked_where(~bm, axial(adc1, SLICE_Z))
    s0 = np.ma.masked_where(~bm, axial(adc0, SLICE_Z))
    sg = np.ma.masked_where(~bm, axial(gap, SLICE_Z))

    specs = [
        (s1, r"$\mathrm{ADC}_1$ (arithmetic, clinical trace)", cmap_black("gray"), vmin, vmax),
        (s0, r"$\mathrm{ADC}_0$ (geometric mean of $D$)",      cmap_black("gray"), vmin, vmax),
        (sg, r"$J_{\ln}=\ln(\mathrm{ADC}_1/\mathrm{ADC}_0)$",  cmap_black(GAP_CMAP), 0.0, GAP_VMAX),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (img, title, cmap, lo, hi) in zip(axes, specs):
        ax.set_facecolor("black")
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    fig.tight_layout()

    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        out = FIG / f"jensen_gap_adc_panel.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", out)
    plt.close(fig)

    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        src = FIG / f"jensen_gap_adc_panel.{ext}"
        if src.exists():
            shutil.copy2(src, PAPER_FIG / f"jensen_gap_adc_panel.{ext}")
    print("copied into", PAPER_FIG)


if __name__ == "__main__":
    main()
