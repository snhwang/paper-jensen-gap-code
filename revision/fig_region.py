"""Figures 5 and 6 of the revision (region level).

Figure 5: why the two gaps change with age in opposite directions. The age slope
of ln J_lnS is split into the tensor-model terms 2 ln(b ADC_1) and ln J_ln.
Figure 6: (a) dependence on b, (b) lifespan trajectories in the corona radiata,
(c, d) within-person change, with the rates of longitudinal.py.

Inputs:  config.HCP tables (cr_gaps_long.csv), out/v1_cohort_sessions.csv
         (v1_roi_stats.py), out/long_sessions.csv (long_sessions.py),
         out/native_pass*.csv (native_pass.py), out/longitudinal.csv
Outputs: figures/fig5_decomposition.pdf, figures/fig6_aging.pdf (.png previews),
         out/fig5_decomposition.csv (slopes plotted in Figure 5)
Usage:   python fig_region.py
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from config import FIGS, HCP, NAMES, OUT, ROIS

matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 8, "pdf.fonttype": 42,
                            "axes.spines.top": False, "axes.spines.right": False,
                            "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                            "ytick.major.width": 0.6})
BLUE, ORANGE, INK, MUTED = "#1f5fa8", "#c2410c", "0.15", "0.55"


coh = pd.read_csv(OUT / "v1_cohort_sessions.csv")
g = pd.read_csv(HCP / "cr_gaps_long.csv", low_memory=False)
for c in ("J_ln_D_mean", "J_ln_S_mean", "ADC_1_mean"):
    g[c] = pd.to_numeric(g[c], errors="coerce")
v1 = g.merge(coh, on=["Subject_ID", "Session_ID"])


def slope(y, age, sex):
    X = np.column_stack([np.ones_like(age), age, (sex == "M").astype(float)])
    return 100 * np.linalg.lstsq(X, y, rcond=None)[0][1]          # % per year


rows = []
for b in (1500, 3000):
    for roi in ROIS:
        s = v1[(v1.shell_bval == b) & (v1.ROI == roi)]
        a, x = s.age.values, s.sex.values
        rows.append(dict(shell=b, ROI=roi,
                         obs=slope(np.log(s.J_ln_S_mean.values), a, x),
                         md=slope(2 * np.log(b * s.ADC_1_mean.values), a, x),
                         aniso=slope(np.log(s.J_ln_D_mean.values), a, x)))
D = pd.DataFrame(rows)
D["pred"] = D.md + D.aniso

# --------------------------------------------------------------------------
# Figure 4
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(6.9, 4.0), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1])
ax = fig.add_subplot(gs[0])
d = D[D.shell == 1500].sort_values("obs")
y = np.arange(len(d))
ax.barh(y, d.md, height=0.62, color=ORANGE, label=r"MD term, $2\,\ln(b\,\mathrm{ADC}_1)$")
ax.barh(y, d.aniso, height=0.62, color=BLUE, label=r"anisotropy term, $\ln J_{\ln}$")
ax.scatter(d.pred, y, marker="D", s=14, facecolor="white", edgecolor=INK, lw=0.8,
           zorder=3, label="sum of the two terms")
ax.scatter(d.obs, y, marker="o", s=18, color=INK, zorder=4,
           label=r"observed, $\ln J_{\ln}^{S}$")
ax.axvline(0, color=INK, lw=0.6)
ax.set_yticks(y, [NAMES[r] for r in d.ROI])
ax.set_xlabel("age slope (% per year), $b=1500\\,\\mathrm{s/mm^2}$")
ax.set_title("(a) Terms of the signal-gap age slope")
ax.grid(axis="x", color="0.9", lw=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2,
          bbox_to_anchor=(0.45, -0.13), columnspacing=1.0, handlelength=1.4)

ax = fig.add_subplot(gs[1])
for b, mk, col in ((1500, "o", BLUE), (3000, "s", ORANGE)):
    q = D[D.shell == b]
    r = np.corrcoef(q.pred, q.obs)[0, 1]
    ax.scatter(q.pred, q.obs, marker=mk, s=18, color=col, edgecolor="white", lw=0.5,
               label=rf"$b={b}$, $r={r:.2f}$", zorder=3)
ax.legend(frameon=False, fontsize=6.5, loc="lower right")
lim = [-0.6, 0.8]
ax.plot(lim, lim, color=MUTED, lw=0.8)
ax.axhline(0, color="0.85", lw=0.5)
ax.axvline(0, color="0.85", lw=0.5)
ax.text(0.75, 0.55, "equality", color=MUTED, fontsize=6.5, ha="right", rotation=45)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect("equal")
ax.set_xlabel("predicted slope (% per year)")
ax.set_ylabel(r"observed slope of $\ln J_{\ln}^{S}$ (% per year)")
ax.set_title("(b) Predicted vs observed")
fig.savefig(FIGS / "fig5_decomposition.pdf")
fig.savefig(FIGS / "fig5_decomposition.png", dpi=200)
plt.close(fig)
D.to_csv(OUT / "fig5_decomposition.csv", index=False)

# --------------------------------------------------------------------------
# Figure 6
# --------------------------------------------------------------------------

fig, axs = plt.subplots(2, 2, figsize=(6.0, 5.2), constrained_layout=True)
axs = axs.ravel()

ax = axs[0]
p = v1[v1.shell_bval.isin([1500, 3000])].groupby(["ROI", "shell_bval"]).J_ln_D_mean.mean().unstack()
ax.scatter(p[1500], p[3000], s=14, color=BLUE, edgecolor="white", lw=0.5, zorder=3)
mx = 0.16
ax.plot([0, mx], [0, mx], color=MUTED, lw=0.8)
ax.text(0.14, 0.152, "$y=x$", color=MUTED, fontsize=6.5, ha="right")
ax.set_xlim(0, mx)
ax.set_ylim(0, mx)
ax.set_aspect("equal")
ax.set_xlabel(r"$J_{\ln}$, $b=1500$")
ax.set_ylabel(r"$J_{\ln}$, $b=3000$")
ax.set_title("(a) Dependence on $b$")

ax = axs[1]
s = v1[(v1.shell_bval == 1500) & (v1.ROI == "CR_all")]
for c, col, lab in (("J_ln_D_mean", BLUE, r"$J_{\ln}$"), ("J_ln_S_mean", ORANGE, r"$J_{\ln}^{S}$")):
    yv = 100 * (s[c] / s[c].mean() - 1)
    ax.scatter(s.age, yv, s=1.5, color=col, alpha=0.18, lw=0, rasterized=True)
    lo = lowess(yv, s.age, frac=0.4)
    ax.plot(lo[:, 0], lo[:, 1], color=col, lw=1.8, label=lab)
ax.axhline(0, color="0.8", lw=0.5)
ax.set_ylim(-45, 45)
ax.set_xlabel("age (years)")
ax.set_ylabel("% from cohort mean")
ax.set_title("(b) Lifespan, corona radiata")
ax.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.2)

# Panels (c, d): the sessions and rates of longitudinal.py, so the figure and the
# text share one source. Trajectories show the measured gap at b = 1500.
L = pd.read_csv(OUT / "long_sessions.csv").query("standard")
nat = pd.concat([pd.read_csv(OUT / "native_pass.csv"),
                 pd.read_csv(OUT / "native_pass_long.csv")], ignore_index=True)
nat = nat[(nat.ROI == "CR_all") & (nat.shell_bval == 1500)].drop_duplicates("Session_ID")
L = L.merge(nat[["Session_ID", "J_ln", "J_lnS"]], on="Session_ID")
L = L[L.groupby("Subject_ID").Visit.transform("nunique") >= 2].copy()
# x position: age at the first included visit plus the exact time since it, as
# in longitudinal.py (released ages are whole years and top-coded at 90).
first = L.groupby("Subject_ID").days_from_V1.transform("min")
age_bl = L.loc[L.days_from_V1 == first].set_index("Subject_ID").age.reindex(L.Subject_ID)
L["x"] = age_bl.values + (L.days_from_V1 - first) / 365.25
rates = pd.read_csv(OUT / "longitudinal.csv").query("shell == 1500").set_index(["gap", "kind"])
for ax, c, col, lab, tag in ((axs[2], "J_ln", BLUE, r"$J_{\ln}$", "(c)"),
                             (axs[3], "J_lnS", ORANGE, r"$J_{\ln}^{S}$", "(d)")):
    y = 100 * (L[c] / L[c].mean() - 1)
    for _, q in L.assign(y=y).sort_values("x").groupby("Subject_ID"):
        ax.plot(q.x, q.y, color=col, lw=0.3, alpha=0.18)
    rm, rn = rates.loc[(c, "measured")], rates.loc[(c, "noise removed")]
    ax.text(0.03, 0.04, f"within person, %/y\nmeasured {rm.rate:+.2f} "
            f"[{rm.ci_lo:+.2f}, {rm.ci_hi:+.2f}]\nnoise removed {rn.rate:+.2f} "
            f"[{rn.ci_lo:+.2f}, {rn.ci_hi:+.2f}]", transform=ax.transAxes, fontsize=6,
            color=INK)
    ax.set_ylim(-45, 45)
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set_xlabel("age (years)")
    ax.set_title(f"{tag} Within person, {lab}")
    print(f"{c}: {L.Subject_ID.nunique()} participants, {len(L)} sessions")
axs[2].set_ylabel("% from cohort mean")
fig.savefig(FIGS / "fig6_aging.pdf", dpi=300)
fig.savefig(FIGS / "fig6_aging.png", dpi=200)
plt.close(fig)
print(D[D.shell == 1500].round(3).to_string())
