"""Figure 3: the gaps and the tensor metrics.

(a) J_ln against FA in the example participant's white matter, with the
leading-order curve 2 FA^2/(15 - 10 FA^2) and the exact gap of prolate and
planar Gaussian tensors. (b) J_lnS against (b MD)^2 J_ln. (c, d) Spearman
correlation of each gap with FA across participants, by region, shown on the
JHU regions over the FMRIB58 template.

Inputs:  example-participant maps (example_maps.py), out/fa_correlations.csv
         (post_native.py), config.ATLAS (JHU labels, FMRIB58)
Output:  figures/fig3_fa_relation.pdf (and .png preview)
Usage:   python fig3_fa_relation.py
"""
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from config import ATLAS as ATL, FIGS, MAPS, OUT, ROI_DEFS

LABELS = {k: v for k, v in ROI_DEFS.items() if k != "CR_all"}
F = pd.read_csv(OUT / "fa_correlations.csv")
matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 8, "pdf.fonttype": 42})
B = 1500
L = lambda n: nib.load(MAPS / f"{n}.nii.gz").get_fdata()
fa_v, jl_v, js_v, md_v = (L(f"{k}_b{B}") for k in ("FA", "J_ln", "J_lnS", "MD"))
wm = (L("wm_jhu_mask") > 0) & (fa_v > 0) & (jl_v > 0)
fa, jl, js, md = fa_v[wm], jl_v[wm], js_v[wm], md_v[wm]

n = 20000
k = np.arange(n) + 0.5
ph = np.arccos(1 - 2 * k / n)
th = np.pi * (1 + 5 ** 0.5) * k
g2 = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1) ** 2


def curve(f, tmax):
    out = []
    for t in np.linspace(0, tmax, 120):
        lam = f(t)
        dd = g2 @ lam
        mu = lam.mean()
        out.append((np.sqrt(1.5 * ((lam - mu) ** 2).sum() / (lam ** 2).sum()),
                    np.log(dd.mean()) - np.log(dd).mean()))
    return np.array(out)


pro = curve(lambda t: np.array([1 + 2 * t, 1 - t, 1 - t]), 0.9)
pla = curve(lambda t: np.array([1 + t, 1 + t, 1 - 2 * t]), 0.45)
fg = np.linspace(0, 0.9, 200)

fig = plt.figure(figsize=(6.9, 6.6), layout="constrained")
top, mid, bot = fig.subfigures(3, 1, height_ratios=[1.25, 1, 1])
ta = top.subplots(1, 2)
ax = ta[0]
ax.hist2d(fa, jl, bins=[np.linspace(0, 0.9, 90), np.linspace(0, 0.2, 90)], norm=LogNorm(),
          cmap="Greys", rasterized=True)
ax.plot(fg, 2 * fg ** 2 / (15 - 10 * fg ** 2), color="#1f5fa8", lw=1.2,
        label=r"$2\,\mathrm{FA}^2/(15-10\,\mathrm{FA}^2)$")
ax.plot(*pro.T, color="#c2410c", lw=1.2, ls="--", label="tensor, prolate")
ax.plot(*pla.T, color="#15803d", lw=1.2, ls=":", label="tensor, planar")
ax.set(xlim=(0, 0.9), ylim=(0, 0.2), xlabel="FA", ylabel=r"$J_{\ln}$")
ax.set_title(r"(a) $J_{\ln}$ against FA, one participant")
ax.legend(frameon=False, loc="upper left", fontsize=6.5)
ax = ta[1]
x = (B * 1e-3 * md * 1e3) ** 2 * jl
ax.hist2d(np.log10(x), np.log10(js), bins=90, range=[[-2.6, -0.6], [-2.6, -0.6]],
          norm=LogNorm(), cmap="Greys", rasterized=True)
ax.plot([-2.6, -0.6], [-2.6, -0.6], color="#1f5fa8", lw=1.0)
ax.text(-0.75, -0.95, "equality", color="#1f5fa8", fontsize=6.5, ha="right", rotation=45)
tk = [0.003, 0.01, 0.03, 0.1, 0.3]
ax.set_xticks(np.log10(tk), [str(t) for t in tk])
ax.set_yticks(np.log10(tk), [str(t) for t in tk])
ax.set(xlabel=r"$(b\,\mathrm{MD})^2\,J_{\ln}$", ylabel=r"$J_{\ln}^{S}$")
ax.set_title(r"(b) $J_{\ln}^{S}$ against $(b\,\mathrm{MD})^2J_{\ln}$")
for a in ta:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

lab = nib.load(ATL / "JHU-ICBM-labels-1mm.nii.gz").get_fdata()
tmpl = nib.load(ATL / "FMRIB58_FA_1mm.nii.gz").get_fdata()
Fb = F[F.shell == B].set_index("ROI")
ZS = (63, 80, 100)                                # MNI z = -9, +8, +28
cmap = plt.get_cmap("viridis").copy()
for sub, col, tag, title in ((mid, "rho_Jln_FA", "(c)", r"$J_{\ln}$"),
                             (bot, "rho_JlnS_FA", "(d)", r"$J_{\ln}^{S}$")):
    vol = np.full(lab.shape, np.nan)
    for roi, ls in LABELS.items():
        vol[np.isin(lab, ls)] = Fb.loc[roi, col]
    axs = sub.subplots(1, 3)
    for ax, z in zip(axs, ZS):
        bg = tmpl[:, 20:200, z].T
        ax.imshow(bg, cmap="gray", vmin=0, vmax=8000, origin="lower")
        im = ax.imshow(np.ma.masked_invalid(vol[:, 20:200, z].T), cmap=cmap, vmin=0, vmax=1,
                       origin="lower", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"MNI $z={z - 72:+d}$ mm", fontsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
    sub.suptitle(f"{tag} Spearman correlation of {title} with FA across participants, "
                 f"$b={B}\\,\\mathrm{{s/mm^2}}$", fontsize=8)
    cb = sub.colorbar(im, ax=axs, fraction=0.03, pad=0.02)
    cb.set_label(r"$\rho$")
    cb.outline.set_linewidth(0.4)
fig.savefig(FIGS / "fig3_fa_relation.pdf", dpi=300)
fig.savefig(FIGS / "fig3_fa_relation.png", dpi=200)
print("wrote", FIGS / "fig3_fa_relation.pdf")
