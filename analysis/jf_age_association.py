"""Does the per-direction attenuation gap J_f track age like Delta_b J_ln does?

The manuscript's b-signature aging result rests on Delta_b J_ln, which is
reliable at region scale. J_f is introduced as the voxelwise form of the same
b-flatness test. If J_f also carries the age association, the two agree at both
scales and the choice of estimator does not drive the finding. This checks that.

Both indices come from cr_jf_long.csv, so they are computed on IDENTICAL voxels,
directions and clamps. Delta_b there is recomputed alongside J_f rather than
taken from cr_gaps_long.csv, which removes any masking or matching confound from
the comparison.

Sign note: more non-Gaussian means MORE positive J_f and MORE negative Delta_b.
So a NEGATIVE r for J_f vs age and a POSITIVE r for Delta_b vs age both mean the
same thing, that white matter becomes more b-flat and less non-Gaussian with age.

Output: printed table plus figures/jf_age_association.{png,pdf}
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(os.environ.get("JG_DATA", Path(__file__).resolve().parent))
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

ROIS = ["CR_all", "ACR", "SCR", "PCR", "Body", "Genu", "Splenium", "alic", "plic",
        "rlic", "ec", "slf", "ptr", "cingulum_cc", "ss", "tapetum"]
PRIMARY = "CR_all"


def partial_r(v, age, sex):
    """Sex-adjusted partial correlation: residualise both on [1, sex]."""
    d = pd.DataFrame({"v": v, "a": age, "s": sex}).dropna()
    X = np.column_stack([np.ones(len(d)),
                         pd.get_dummies(d["s"], drop_first=True).astype(float).values])
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    r, _ = stats.pearsonr(res(d["v"].values), res(d["a"].values))
    df = len(d) - 3
    t = r * np.sqrt(df / max(1 - r * r, 1e-16))
    return r, 2 * stats.t.sf(abs(t), df), len(d)


def main():
    jf = pd.read_csv(BASE / "HCP" / "cr_jf_long.csv")
    man = pd.read_csv(BASE / "HCP" / "manifest_n1379_b1500.tsv", sep="\t")
    # HIPAA top-code: "90 or older" -> 90, retained so N stays at 1,379
    man["age"] = pd.to_numeric(
        man["age"].astype(str).str.replace("90 or older", "90", regex=False),
        errors="coerce")

    jf = jf.merge(man[["session_id", "age", "sex"]],
                  left_on="Session_ID", right_on="session_id")
    print(f"sessions with J_f and phenotype: {jf.Session_ID.nunique():,}")
    print(f"age range: {jf.age.min():.0f} to {jf.age.max():.0f}\n")

    rows = []
    for roi in ROIS:
        s = jf[jf.ROI == roi]
        rj, pj, n = partial_r(pd.to_numeric(s.J_f_mean, errors="coerce").values,
                              s.age.values, s.sex.values)
        rd, pd_, _ = partial_r(
            pd.to_numeric(s.Delta_b_J_ln_mean, errors="coerce").values,
            s.age.values, s.sex.values)
        rows.append((roi, rj, pj, rd, pd_, n))

    print(f"{'ROI':14s} {'J_f vs age':>22s} {'Delta_b vs age':>22s}")
    for roi, rj, pj, rd, pd_, n in rows:
        print(f"  {roi:12s} r={rj:+.3f} p={pj:.1e}   r={rd:+.3f} p={pd_:.1e}   N={n:,}")

    rj_all = np.array([r[1] for r in rows])
    rd_all = np.array([r[3] for r in rows])
    print(f"\nJ_f     : {(rj_all < 0).sum()}/16 negative, "
          f"range {rj_all.min():+.3f} to {rj_all.max():+.3f}")
    print(f"Delta_b : {(rd_all > 0).sum()}/16 positive, "
          f"range {rd_all.min():+.3f} to {rd_all.max():+.3f}")
    print(f"agreement across ROIs: r = {stats.pearsonr(rj_all, -rd_all).statistic:+.3f}"
          "  (J_f vs sign-flipped Delta_b)")

    # ---- figure ------------------------------------------------------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    s = jf[jf.ROI == PRIMARY]
    v = pd.to_numeric(s.J_f_mean, errors="coerce")
    r, p, n = partial_r(v.values, s.age.values, s.sex.values)
    axA.scatter(s.age, v, s=7, alpha=0.30, color="#c0392b", edgecolors="none")
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        lo = lowess(v.values, s.age.values, frac=0.5, return_sorted=True)
        axA.plot(lo[:, 0], lo[:, 1], color="#7b241c", lw=2.6)
    except Exception:
        b = np.polyfit(s.age, v, 1)
        xs = np.linspace(s.age.min(), s.age.max(), 100)
        axA.plot(xs, np.polyval(b, xs), color="#7b241c", lw=2.6)
    axA.set_xlabel("Age (years)")
    axA.set_ylabel(r"$J_f$")
    axA.set_title(f"(a) $J_f$ vs age, {PRIMARY}\n"
                  f"sex-adjusted $r={r:+.2f}$, $p={p:.1e}$, $N={n:,}$", fontsize=10)
    axA.text(0.02, 0.04, "lower means more $b$-flat, less non-Gaussian",
             transform=axA.transAxes, fontsize=8, color="0.3")

    order = sorted(rows, key=lambda x: x[1])
    y = np.arange(len(order))
    axB.barh(y, [x[1] for x in order], color="#c0392b", alpha=0.85, height=0.68)
    axB.set_yticks(y); axB.set_yticklabels([x[0] for x in order], fontsize=8.5)
    axB.axvline(0, color="0.35", lw=1.2)
    axB.set_xlabel(r"sex-adjusted partial $r$ ($J_f$ vs age)")
    axB.set_title("(b) All 16 JHU ROIs", fontsize=10)
    for yi, rr in zip(y, [x[1] for x in order]):
        axB.text(rr - 0.012, yi, f"{rr:+.2f}", va="center", ha="right",
                 fontsize=7.5, color="0.25")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        try:
            fig.savefig(FIG / f"jf_age_association.{ext}", dpi=200, bbox_inches="tight")
        except PermissionError:
            print(f"  [skip] locked: jf_age_association.{ext}")
    plt.close(fig)
    print(f"\nwrote {FIG / 'jf_age_association'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
