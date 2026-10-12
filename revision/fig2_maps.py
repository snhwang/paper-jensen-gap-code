"""Figure 2: directional summaries, FA and the two gaps in the example participant.

Panels (a)-(c) show ADC_1, ADC_0 and D_SM on one grey/white-matter window in
which cerebrospinal fluid saturates. Panels (d)-(f) show FA, the
diffusivity gap and the signal gap.

Input:   the example-participant maps written by example_maps.py (config.MAPS)
Output:  figures/fig2_maps.pdf (and .png preview)
Usage:   python fig2_maps.py
"""
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from config import FIGS, MAPS

matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 8, "pdf.fonttype": 42,
                            "font.family": "DejaVu Sans"})
Z = 60                      # axial slice
B = 1500
ADC_WIN = (0.3, 1.3)        # 10^-3 mm^2/s. Grey and white matter, CSF saturates.


def load(name):
    return nib.load(MAPS / f"{name}.nii.gz").get_fdata()


def axial(a):
    return a[:, :, Z].T          # radiological: participant right on viewer left


def cmap_bad(name, bad="black"):
    c = plt.get_cmap(name).copy()
    c.set_bad(bad)
    return c


m = {k: load(f"{k}_b{B}") for k in ("ADC_1", "ADC_0", "D_SM", "FA", "J_ln", "J_lnS")}
brain = m["ADC_1"] > 0


def crop(img):
    """Trim empty rows and columns around the brain on the display slice."""
    bm = axial(brain)
    r = np.where(bm.any(1))[0]
    c = np.where(bm.any(0))[0]
    return img[r[0] - 2:r[-1] + 3, c[0] - 2:c[-1] + 3]


def show(ax, vol, cmap, lo, hi, title):
    im = ax.imshow(crop(np.ma.masked_where(~axial(brain), axial(vol))), cmap=cmap,
                   vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def cbar(fig, ax, im, label, ticks=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=ticks)
    cb.set_label(label)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)


fig, axs = plt.subplots(2, 3, figsize=(6.9, 5.0), constrained_layout=True)
grey = cmap_bad("gray")
gapc = cmap_bad("viridis")
for ax, k, t in zip(axs[0], ("ADC_1", "ADC_0", "D_SM"),
                    (r"(a) $\mathrm{ADC}_1$, trace ADC",
                     r"(b) $\mathrm{ADC}_0$, geometric mean",
                     r"(c) $D_{\mathrm{SM}}$, spherical mean")):
    im = show(ax, 1e3 * m[k], grey, *ADC_WIN, t)
cbar(fig, axs[0, 2], im, r"$10^{-3}\,\mathrm{mm^2/s}$")
im = show(axs[1, 0], m["FA"], grey, 0, 1, "(d) FA")
cbar(fig, axs[1, 0], im, "FA", ticks=[0, 0.5, 1])
im = show(axs[1, 1], m["J_ln"], gapc, 0, 0.10, r"(e) $J_{\ln}$, diffusivity gap")
cbar(fig, axs[1, 1], im, r"$J_{\ln}$", ticks=[0, 0.05, 0.10])
im = show(axs[1, 2], m["J_lnS"], gapc, 0, 0.10, r"(f) $J_{\ln}^{S}$, signal gap")
cbar(fig, axs[1, 2], im, r"$J_{\ln}^{S}$", ticks=[0, 0.05, 0.10])
fig.savefig(FIGS / "fig2_maps.pdf")
fig.savefig(FIGS / "fig2_maps.png", dpi=200)
plt.close(fig)
print("wrote", FIGS / "fig2_maps.pdf")
