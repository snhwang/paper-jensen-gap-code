"""Figures 6 and 7 of the revision (region level).

Figure 6: why the two gaps change with age in opposite directions. The age slope
of ln J_lnS is split into the tensor-model terms 2 ln(b ADC_1) and ln J_ln.
Figure 7: (a) dependence on b, (b) lifespan course in the corona radiata: means in
age bins (36-39, 40-44, ..., 85-90), GAM curves (mssm, REML) with pointwise 95% intervals and the
linear fits they are tested against (GLRT, p from a parametric bootstrap; this step
takes several minutes),
(c, d) within-person change,
with the rates of longitudinal.py. The GAM tests and the course over age of the
decomposition terms (Section 4.3) go to out/lifespan_shape.txt.

Inputs:  config.HCP tables (cr_gaps_long.csv), out/v1_cohort_sessions.csv
         (v1_roi_stats.py), out/long_sessions.csv (long_sessions.py),
         out/native_pass*.csv (native_pass.py), out/longitudinal.csv and
         out/longitudinal_by_age.csv (longitudinal.py)
Outputs: figures/fig5_decomposition.pdf, figures/fig6_aging.pdf (.png previews),
         out/fig5_decomposition.csv (slopes plotted in Figure 6), out/lifespan_shape.txt
Usage:   python fig_region.py
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mssm import models as mm
from mssm.src.python.compare import compare_CDL
from mssm.src.python.utils import approx_smooth_p_values

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

N_BOOT = 2000


def fit_age(d, age_term):
    m = mm.GAMM(mm.Formula(mm.lhs("y"), [mm.i(), age_term, mm.l(["male"])], data=d, print_warn=False),
                mm.Gaussian())
    m.fit(progress_bar=False)
    return m


def glrt(d):
    """GAM and linear fits of d, and the generalized likelihood ratio test of the GAM
    against the linear model (conditional on the smoothing parameter)."""
    gam, lin = fit_age(d, mm.f(["age"])), fit_age(d, mm.l(["age"]))
    return gam, lin, compare_CDL(gam, lin, correct_V=False, perform_GLRT=True, correct_t1=True, seed=0)


def gam_age(s, col):
    """GAM of one gap (percent from its cohort mean) on age in one region and shell: a
    penalized spline of age (mssm defaults, 10 B-spline basis functions, second-order
    difference penalty, REML) with sex as a centred covariate. Returns the curve and its
    pointwise 95% interval on GRID at the average sex, the straight line of the linear
    model, the edf and Wood (2013) p-value of s(age), and the test against the linear
    model. Ignoring uncertainty in the smoothing parameter makes the GLRT's chi-squared
    p-value too small, and correcting for it makes it too large (Krause et al. 2025,
    footnote 12). Its p-value is therefore taken from a parametric bootstrap instead:
    N_BOOT data sets are simulated from the linear fit by resampling its residuals, and
    p = (1 + number of simulated statistics >= the observed one) / (1 + N_BOOT)."""
    d = pd.DataFrame(dict(y=100 * (s[col].values / s[col].mean() - 1), age=s.age.values.astype(float),
                          male=(s.sex.values == "M") - (s.sex.values == "M").mean()))
    gam, lin, obs = glrt(d)
    mu = lin.predict([0, 1, 2], d)[0]
    res = d.y.values - mu
    rng = np.random.default_rng(0)
    sim = np.empty(N_BOOT)
    for k in range(N_BOOT):
        sim[k] = glrt(d.assign(y=mu + rng.choice(res, len(res))))[2]["test_stat"]
    n_ge = int((sim >= obs["test_stat"]).sum())
    grid = pd.DataFrame(dict(age=GRID, male=0.0))
    pred, _, half = gam.predict([0, 1], grid, ci=True)
    return dict(curve=pred, lo=pred - half, hi=pred + half, line=lin.predict([0, 1], grid)[0],
                edf=gam.term_edf[0], p_smooth=approx_smooth_p_values(gam)[0][0], stat=obs["test_stat"],
                p_chi2=obs["p"], n_ge=n_ge, p_boot=(1 + n_ge) / (1 + N_BOOT), aic_gain=-obs["aic_diff"])


def fmt_p(p):
    if p < 0.001:
        return "$p<0.001$"
    return f"$p={p:.3f}$" if p < 0.01 else f"$p={p:.2f}$"


# The course over age in the corona radiata (Section 4.3 and Figure 7b): a GAM of each
# gap on age, the age at the minimum of the signal gap's curve, and the slopes of the
# decomposition in age bands.
GRID = np.arange(36, 90.01, 0.5)
BANDS = ((36, 55), (55, 70), (70, 91))
GAMS = {}
shape = ["Corona radiata. GAM of each gap on age (sex-adjusted): edf and p of s(age), and the GLRT "
         f"against a linear age term, with its p from a parametric bootstrap ({N_BOOT} data sets "
         "simulated from the linear fit) and, for reference, from chi-squared. Then sex-adjusted "
         "slopes in %/y of the decomposition by age band"]
for b in (1500, 3000):
    s = v1[(v1.shell_bval == b) & (v1.ROI == "CR_all")]
    for col, name in (("J_ln_D_mean", "J_ln"), ("J_ln_S_mean", "J_lnS")):
        r = GAMS[(b, col)] = gam_age(s, col)
        shape.append(f"b={b} {name:5s}: edf {r['edf']:.2f}, s(age) p={r['p_smooth']:.1e}, "
                     f"GLRT statistic {r['stat']:.2f}, bootstrap p={r['p_boot']:.4f} "
                     f"({r['n_ge']} of {N_BOOT} simulated statistics as large), chi-squared p={r['p_chi2']:.1e}, "
                     f"AIC gain over linear {r['aic_gain']:.1f}")
    shape.append(f"b={b}: signal gap lowest at age {GRID[np.argmin(GAMS[(b, 'J_ln_S_mean')]['curve'])]:.1f}")
    for a0, a1 in BANDS:
        q = s[(s.age >= a0) & (s.age < a1)]
        a, x = q.age.values, q.sex.values
        o, m_, n_ = (slope(np.log(q.J_ln_S_mean.values), a, x), slope(2 * np.log(b * q.ADC_1_mean.values), a, x),
                     slope(np.log(q.J_ln_D_mean.values), a, x))
        shape.append(f"  ages {a0}-{a1 - 1} (n={len(q)}): observed {o:+.2f}, MD term {m_:+.2f}, "
                     f"anisotropy term {n_:+.2f}, predicted {m_ + n_:+.2f}")
(OUT / "lifespan_shape.txt").write_text("\n".join(shape) + "\n", encoding="utf-8")
print("\n".join(shape))

# --------------------------------------------------------------------------
# Figure 6
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
# Figure 7
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
# The band is for the mean curve, so the data are shown on the same footing: means in
# Age bins (36-39, then 5-year bins, 85-90 last) with their 95% CIs (individuals scatter far more widely, panels c, d).
# The dashed line is the linear model that the GAM is tested against.
EDGES = (36, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 91)
bins = pd.cut(s.age, EDGES, right=False)
for k, (c, col, lab) in enumerate((("J_ln_D_mean", BLUE, r"$J_{\ln}$"), ("J_ln_S_mean", ORANGE, r"$J_{\ln}^{S}$"))):
    yv = 100 * (s[c] / s[c].mean() - 1)
    q = pd.DataFrame(dict(age=s.age, y=yv)).groupby(bins, observed=True).agg(
        age=("age", "mean"), m=("y", "mean"), sd=("y", "std"), n=("y", "size"))
    ax.errorbar(q.age, q.m, yerr=1.96 * q.sd / np.sqrt(q.n), fmt="o", ms=2.6, color=col, mec="white",
                mew=0.4, elinewidth=0.7, capsize=0, zorder=3)
    r = GAMS[(1500, c)]
    ax.fill_between(GRID, r["lo"], r["hi"], color=col, alpha=0.3, lw=0)
    ax.plot(GRID, r["curve"], color=col, lw=1.6, label=lab)
    ax.plot(GRID, r["line"], color=col, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.03, 0.17 - 0.08 * k, f"{lab} {fmt_p(r['p_boot'])}, edf {r['edf']:.1f}",
            transform=ax.transAxes, fontsize=6, color=col)
ax.text(0.03, 0.25, "GAM against straight line:", transform=ax.transAxes, fontsize=6, color=INK)
ax.axhline(0, color="0.8", lw=0.5)
ax.set_ylim(-20, 15)
ax.set_xlabel("age (years)")
ax.set_ylabel("% from cohort mean")
ax.set_title("(b) Lifespan, corona radiata")
ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2, handlelength=1.2)

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
# The model's within-person change over 10 years of follow-up, from first-visit ages
# of 45, 60 and 75 and the model's mean level there. For the signal gap the rate
# depends on age at the first visit, so panel (d) uses the rate at each age and shows
# it, with its 95% CI, as a curve over first-visit age in an inset. The diffusivity
# gap's rate does not, once its noise inflation is removed (longitudinal.txt), so
# panel (c) uses the single rate at all three ages and gives its CI in the label.
# (The CI of a rate spans about 1 point of the y axis over 10 years, too narrow to
# show on the trajectories themselves.)
by_age = pd.read_csv(OUT / "longitudinal_by_age.csv").query(
    "shell == 1500 and kind == 'measured' and method == 'mixed'")
SEG_AGES = (45, 60, 75)


def segment(ax, a, level, rate):
    ax.plot([a, a + 10], [level, level + 10 * rate], color=INK, lw=1.4, solid_capstyle="round",
            zorder=4)


for ax, c, col, lab, tag in ((axs[2], "J_ln", BLUE, r"$J_{\ln}$", "(c)"),
                             (axs[3], "J_lnS", ORANGE, r"$J_{\ln}^{S}$", "(d)")):
    y = 100 * (L[c] / L[c].mean() - 1)
    for _, q in L.assign(y=y).sort_values("x").groupby("Subject_ID"):
        ax.plot(q.x, q.y, color=col, lw=0.3, alpha=0.18)
    ra = by_age[by_age.gap == c]
    seg = ra[ra.age.isin(SEG_AGES)]
    if c == "J_ln":
        rm = rates.loc[(c, "measured")]
        for r in seg.itertuples():
            segment(ax, r.age, r.level, rm.rate)
        ax.text(0.03, 0.04, f"within person, %/y\n{rm.rate:+.2f} [{rm.ci_lo:+.2f}, {rm.ci_hi:+.2f}]",
                transform=ax.transAxes, fontsize=6, color=INK)
    else:
        for r in seg.itertuples():
            segment(ax, r.age, r.level, r.rate)
        ins = ax.inset_axes([0.14, 0.66, 0.34, 0.28])
        ins.fill_between(ra.age, ra.ci_lo, ra.ci_hi, color=col, alpha=0.3, lw=0)
        ins.plot(ra.age, ra.rate, color=col, lw=1.2)
        ins.axhline(0, color=MUTED, lw=0.5)
        ins.plot(seg.age, seg.rate, "o", color=INK, ms=2.2, zorder=3)
        ins.set_xlim(40, 85)
        ins.set_xticks([45, 60, 75])
        ins.tick_params(labelsize=5.5, length=2, pad=1)
        ins.set_xlabel("age at first visit", fontsize=5.5, labelpad=1)
        ins.set_ylabel("rate, %/y", fontsize=5.5, labelpad=1)
        ins.patch.set_alpha(0.85)
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
