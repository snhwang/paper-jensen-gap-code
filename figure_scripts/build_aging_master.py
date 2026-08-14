"""Aging-characterization master figure.

(a) b-signature: per-ROI scatter of J_ln_D(b=1500) vs J_ln_D(b=3000).
(b) Lifespan trajectory: J_ln(D) and J_ln(S) vs age in CR_all, twin axes, LOWESS.
(c,d) Within-subject longitudinal trajectories of J_ln (c) and J_ln^S (d) across
      visits in CR_all, with the linear mixed-effects within-subject slope.
"""
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAVE_LOWESS = True
except ImportError:
    HAVE_LOWESS = False

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"; FIG.mkdir(exist_ok=True)
PAPER_FIG = BASE.parent / "paper-jensen-gap" / "figures"

ROIS = ["CR_all", "ACR", "SCR", "PCR", "Body", "Genu", "Splenium",
        "alic", "plic", "rlic", "ec", "slf", "ptr",
        "cingulum_cc", "ss", "tapetum"]

JD = r"$J_{\ln}$"
JS = r"$J_{\ln}^{S}$"


def panel_a_bsignature(ax, g):
    pts = []
    for roi in ROIS:
        s = g[g.ROI == roi]
        m1 = pd.to_numeric(s[s.shell_bval == 1500].J_ln_D_mean, errors="coerce").mean()
        m3 = pd.to_numeric(s[s.shell_bval == 3000].J_ln_D_mean, errors="coerce").mean()
        pts.append((roi, m1, m3))
    pts = pd.DataFrame(pts, columns=["roi", "b1500", "b3000"])
    mx = pts[["b1500", "b3000"]].max().max() * 1.05
    ax.plot([0, mx], [0, mx], "-", color="gray", lw=1, alpha=0.7, label="y = x (Gaussian)")
    ax.scatter(pts.b1500, pts.b3000, s=40, c="#1f4e8a", edgecolors="black", lw=0.5, zorder=5)
    # label only the extremes (2 lowest, 3 highest by J_ln) to avoid the cluster
    ordered = pts.sort_values("b1500")
    extremes = pd.concat([ordered.head(2), ordered.tail(3)])
    for _, r in extremes.iterrows():
        right = r.b1500 > 0.6 * pts.b1500.max()
        ax.annotate(r.roi, (r.b1500, r.b3000), fontsize=7,
                    xytext=(-5, -1) if right else (6, -1), textcoords="offset points",
                    ha="right" if right else "left", va="center")
    pct = 100.0 * (pts.b3000 - pts.b1500).mean() / pts.b1500.mean()
    ax.set_xlabel(f"{JD} at $b=1500$", fontsize=10)
    ax.set_ylabel(f"{JD} at $b=3000$", fontsize=10)
    ax.set_title(f"(a) b-signature  (per-ROI cohort means)\nmean shift {pct:+.1f}% $\\Rightarrow$ non-Gaussian", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)


def partial_r(val, age, sex):
    """Partial correlation of val vs age controlling for binary sex."""
    df = pd.DataFrame({"v": val, "a": age, "s": sex}).dropna()
    sd = pd.get_dummies(df["s"], drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(df)), sd.values])
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(res(df["v"].values), res(df["a"].values))[0, 1])


def panel_b_lifespan(ax, g, man):
    s = g[(g.shell_bval == 1500) & (g.ROI == "CR_all")]
    j = s.merge(man[["session_id", "age", "sex"]], left_on="Session_ID", right_on="session_id", how="inner")
    for c in ("age", "J_ln_D_mean", "J_ln_S_mean"):
        j[c] = pd.to_numeric(j[c], errors="coerce")
    j = j.dropna(subset=["age", "J_ln_D_mean", "J_ln_S_mean", "sex"])
    ax.scatter(j.age, j.J_ln_D_mean, s=4, c="#1f4e8a", alpha=0.25)
    ax2 = ax.twinx()
    ax2.scatter(j.age, j.J_ln_S_mean, s=4, c="#c0392b", alpha=0.25)
    if HAVE_LOWESS:
        for arr, axi, color in [(j.J_ln_D_mean.values, ax, "#1f4e8a"), (j.J_ln_S_mean.values, ax2, "#c0392b")]:
            lo = lowess(arr, j.age.values, frac=0.3, return_sorted=True)
            axi.plot(lo[:, 0], lo[:, 1], color=color, lw=2.4)
    rD = partial_r(j.J_ln_D_mean, j.age, j.sex); rS = partial_r(j.J_ln_S_mean, j.age, j.sex)
    ax.set_xlabel("Age (y)", fontsize=10)
    ax.set_ylabel(f"{JD} (diffusivity)", color="#1f4e8a", fontsize=10)
    ax2.set_ylabel(f"{JS} (signal)", color="#c0392b", fontsize=10)
    ax.tick_params(axis="y", labelcolor="#1f4e8a"); ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax.set_title(f"(b) Lifespan trajectory, CR_all  ($b=1500$)\n{JD}: $r={rD:+.2f}$   {JS}: $r={rS:+.2f}$", fontsize=10)
    ax.grid(alpha=0.25)


def load_age_map():
    """Per-visit age (years) keyed by (Subject_ID, Visit) from AABC2 demographics."""
    import glob
    a = pd.read_csv(glob.glob(str(BASE / "HCP" / "AABC2_subjects_*.csv"))[0], low_memory=False)
    a = a.dropna(subset=["id_event"]).copy()
    a["Subject_ID"] = a.id_event.str.rsplit("_", n=1).str[0]
    a["Visit"] = a.id_event.str.rsplit("_", n=1).str[1]
    a["age_yr"] = pd.to_numeric(a.age_open.astype(str).str.replace("90 or older", "90", regex=False),
                                errors="coerce")
    return a[["Subject_ID", "Visit", "age_yr", "sex"]].dropna()


def within_slope(sl, col):
    """Within-subject age slope per year from a mixed model, adjusted for sex."""
    import statsmodels.formula.api as smf
    m = smf.mixedlm(f"{col} ~ age_yr + C(sex)", sl, groups=sl["Subject_ID"]).fit(method="lbfgs")
    return m.params["age_yr"], m.pvalues["age_yr"], m.params["Intercept"]


def panel_c_traj(ax, sl, col, color, name, tag):
    slope, p, b0 = within_slope(sl, col)
    # faint individual trajectories (each subject's visits connected), subsampled
    subs = sl.Subject_ID.drop_duplicates().tolist()
    for sub in subs[::max(1, len(subs) // 200)]:
        grp = sl[sl.Subject_ID == sub].sort_values("age_yr")
        ax.plot(grp.age_yr, grp[col], color=color, alpha=0.12, lw=0.6, marker="o", ms=2)
    # mixed-model population (within-subject) trend over the observed age range
    xa = np.array([sl.age_yr.min(), sl.age_yr.max()])
    ax.plot(xa, b0 + slope * xa, color="black", lw=2.6, zorder=6, label="within-subject trend")
    pct = 100 * slope / sl[col].mean()
    ax.set_xlabel("Age (y)", fontsize=10)
    ax.set_ylabel(name, fontsize=10)
    ax.set_title(f"({tag}) {name} within-subject\nslope $={pct:+.2f}\\%$/y, $p={p:.1g}$", fontsize=9)
    ax.grid(alpha=0.25); ax.legend(loc="best", fontsize=7)
    return slope, p


def main():
    g = pd.read_csv(BASE / "HCP" / "cr_gaps_long.csv")
    man1500 = pd.read_csv(BASE / "HCP" / "manifest_n1379_b1500.tsv", sep="\t")

    # longitudinal subset: CR_all, b=1500, subjects with >=2 visits, with per-visit age
    amap = load_age_map()
    s = g[(g.shell_bval == 1500) & (g.ROI == "CR_all")].merge(amap, on=["Subject_ID", "Visit"], how="left")
    s["J_ln_D_mean"] = pd.to_numeric(s.J_ln_D_mean, errors="coerce")
    s["J_ln_S_mean"] = pd.to_numeric(s.J_ln_S_mean, errors="coerce")
    s = s.dropna(subset=["J_ln_D_mean", "J_ln_S_mean", "age_yr", "sex", "Subject_ID"])
    sl = s[s.groupby("Subject_ID").age_yr.transform("count") >= 2].copy()
    n_sub = sl.Subject_ID.nunique()

    # Compact 2x2 layout (a,b top; c,d bottom) so the figure is ~square and fits
    # one page with its caption. Width ~= text width so LaTeX barely scales it.
    fig = plt.figure(figsize=(7.0, 6.6))
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0]); panel_a_bsignature(ax_a, g)
    ax_b = fig.add_subplot(gs[0, 1]); panel_b_lifespan(ax_b, g, man1500)
    ax_c = fig.add_subplot(gs[1, 0]); sD, pD = panel_c_traj(ax_c, sl, "J_ln_D_mean", "#1f4e8a", JD, "c")
    ax_d = fig.add_subplot(gs[1, 1]); sS, pS = panel_c_traj(ax_d, sl, "J_ln_S_mean", "#c0392b", JS, "d")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"aging_master.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {FIG / 'aging_master'}.{{png,pdf}}")
    print(f"  within-subject (n={n_sub}): J_ln slope/visit={sD:+.5f} p={pD:.2e} | "
          f"J_ln^S slope/visit={sS:+.5f} p={pS:.2e}")
    if PAPER_FIG.exists():
        for ext in ("png", "pdf"):
            shutil.copy2(FIG / f"aging_master.{ext}", PAPER_FIG / f"aging_master.{ext}")
        print("  copied into paper-jensen-gap/figures")


if __name__ == "__main__":
    main()
