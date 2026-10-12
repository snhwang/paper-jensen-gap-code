"""Figure 1 of the revision: the two Jensen gaps for a voxel with two directions.

Adapted from the ASFNR 2026 poster schematic. (a) diffusivity gap on the
logarithm of the directional ADCs, (b) signal gap on the logarithm of the
normalized signals of the same two directions at b = 1500 s/mm^2.

Usage:  python fig_schematic.py
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config import FIGS

matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 8, "pdf.fonttype": 42,
                            "axes.spines.top": False, "axes.spines.right": False,
                            "axes.linewidth": 0.6})
BLUE, ORANGE, INK, MUTED = "#1f5fa8", "#c2410c", "0.15", "0.6"

D = np.array([0.3, 1.7])                 # 10^-3 mm^2/s, across and along fibres
B = 1.5                                   # b in 10^3 s/mm^2, so b*D is dimensionless

fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9), constrained_layout=True)


def panel(ax, x, xs, y_of, mean_x, mean_y, gap_label, xlab, ylab, title, ticks):
    ax.plot(xs, y_of(xs), color=INK, lw=1.6)
    ax.plot(x, y_of(x), color=BLUE, lw=1.0, ls="--")
    ax.scatter(x, y_of(x), s=28, color=BLUE, zorder=4)
    ax.scatter([mean_x], [y_of(mean_x)], s=26, marker="s", color=INK, zorder=5)
    ax.scatter([mean_x], [mean_y], s=26, marker="s", color=BLUE, zorder=5)
    ax.annotate("", xy=(mean_x, mean_y + 0.03), xytext=(mean_x, y_of(mean_x) - 0.03),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.3))
    ax.text(mean_x + 0.03 * (xs[-1] - xs[0]), (mean_y + y_of(mean_x)) / 2, gap_label,
            color=ORANGE, va="center", fontsize=8.5, fontweight="bold")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xticks(*ticks)
    return ax


# (a) diffusivity domain
adc1 = D.mean()
adc0 = np.exp(np.log(D).mean())
ax = panel(axs[0], D, np.linspace(0.22, 1.95, 200), np.log, adc1, np.log(D).mean(),
           rf"$J_{{\ln}}={np.log(adc1 / adc0):.2f}$",
           r"directional ADC $D_i$ ($10^{-3}\,\mathrm{mm^2/s}$)", r"$\ln D$",
           "(a) Diffusivity gap",
           ([0.3, adc0, adc1, 1.7], ["0.3", f"{adc0:.2f}", "1.0", "1.7"]))
ax.text(adc1 - 0.04, np.log(adc1) + 0.12, r"$\ln\mathrm{ADC}_1$", ha="right", fontsize=7.5)
ax.text(adc1 + 0.04, np.log(D).mean() - 0.14, r"$\ln\mathrm{ADC}_0=\langle\ln D\rangle$",
        color=BLUE, fontsize=7.5)
ax.plot([adc0, adc0, adc1], [-1.45, np.log(D).mean(), np.log(D).mean()], color=MUTED,
        lw=0.6, ls=":")
ax.plot([adc1, adc1], [-1.45, np.log(D).mean()], color=MUTED, lw=0.6, ls=":")
ax.text(0.33, np.log(0.3) - 0.05, "across fibres", color=BLUE, fontsize=7, va="top")
ax.text(1.68, np.log(1.7) + 0.12, "along fibres", color=BLUE, fontsize=7, ha="right")
ax.set_ylim(-1.45, 0.85)

# (b) signal domain, same two directions
S = np.exp(-B * D)
ms = S.mean()
dsm = -np.log(ms) / B
ax = panel(axs[1], S, np.linspace(0.05, 0.75, 200), np.log, ms, np.log(S).mean(),
           rf"$J_{{\ln}}^{{S}}={np.log(ms) - np.log(S).mean():.2f}$",
           r"normalized signal $S_i'=e^{-bD_i}$, $b=1500\,\mathrm{s/mm^2}$",
           r"$\ln S'$", "(b) Signal gap",
           ([S[1], ms, S[0]], [f"{S[1]:.2f}", f"{ms:.2f}", f"{S[0]:.2f}"]))
ax.text(ms - 0.015, np.log(ms) + 0.18, r"$\ln\langle S'\rangle=-b\,D_{\mathrm{SM}}$",
        ha="right", fontsize=7.5)
ax.text(ms + 0.02, np.log(S).mean() - 0.22, r"$\langle\ln S'\rangle=-b\,\mathrm{ADC}_1$",
        color=BLUE, fontsize=7.5)
ax.text(S[1] + 0.02, np.log(S[1]) - 0.05, "along fibres", color=BLUE, fontsize=7, va="top")
ax.text(S[0] - 0.01, np.log(S[0]) + 0.15, "across fibres", color=BLUE, fontsize=7, ha="right")
ax.set_ylim(-3.0, -0.2)
fig.savefig(FIGS / "fig1_schematic.pdf")
fig.savefig(FIGS / "fig1_schematic.png", dpi=200)
print(f"J_ln {np.log(adc1 / adc0):.3f}, ADC0 {adc0:.3f}; S {S.round(3)}, <S'> {ms:.3f}, "
      f"D_SM {dsm:.3f}, J_lnS {np.log(ms) - np.log(S).mean():.3f}")
