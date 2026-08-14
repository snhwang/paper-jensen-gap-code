"""Empirical gap spectrum: age correlations of Holder rung-pair gaps in both
domains and shells, on the n=1379 HCP-A cohort.

Output: figures/gap_spectrum_age.{png,pdf}
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"; FIG.mkdir(exist_ok=True)

# (low rung, high rung, label) — diffusivity domain (ADC_r)
DIFF = [("ADC_m1", "ADC_0",  r"$G_{-1,0}$"),
        ("ADC_0",  "ADC_1",  r"$G_{0,1}=J_{\ln}$"),
        ("ADC_1",  "ADC_2",  r"$G_{1,2}$"),
        ("ADC_m1", "ADC_2",  r"$G_{-1,2}$"),
        ("ADC_min","ADC_max",r"$G_{\mathrm{min},\mathrm{max}}$")]
# signal-domain rungs (M_r(S'))
SIG  = [("Mm1_S",  "M0_S",  r"$G^{S}_{-1,0}$"),
        ("M0_S",   "M1_S",  r"$G^{S}_{0,1}=J_{\ln}^{S}$"),
        ("M1_S",   "M2_S",  r"$G^{S}_{1,2}$"),
        ("Mm1_S",  "M2_S",  r"$G^{S}_{-1,2}$"),
        ("M_min_S","M_max_S",r"$G^{S}_{\mathrm{min},\mathrm{max}}$")]


def partial_r(gap, age, sex):
    """Pearson partial correlation of gap vs age, controlling for (binary) sex."""
    df = pd.DataFrame({"g": gap, "a": age, "s": sex}).dropna()
    sd = pd.get_dummies(df["s"], drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(df)), sd.values])
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(res(df["g"].values), res(df["a"].values))[0, 1])


def age_r(g, manifest, bval, roi, lo_col, hi_col):
    gb = g[g.shell_bval == bval]
    j = gb.merge(manifest[["session_id", "age", "sex"]],
                 left_on="Session_ID", right_on="session_id", how="inner")
    s = j[j.ROI == roi].copy()
    lo = pd.to_numeric(s[f"{lo_col}_mean"], errors="coerce").clip(lower=1e-12)
    hi = pd.to_numeric(s[f"{hi_col}_mean"], errors="coerce").clip(lower=1e-12)
    gap = np.log(hi / lo)
    return partial_r(gap.values, s["age"].values, s["sex"].values)


def main():
    g = pd.read_csv(BASE / "HCP" / "cr_gaps_long.csv")
    mans = {b: pd.read_csv(BASE / "HCP" / f"manifest_n1379_b{b}.tsv", sep="\t")
            for b in (1500, 3000)}
    for b in mans:
        mans[b]["age"] = pd.to_numeric(mans[b]["age"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    labels = [d[2] for d in DIFF]
    x = np.arange(len(labels))
    for ax, roi in zip(axes, ("CR_all", "SCR")):
        rD1500 = [age_r(g, mans[1500], 1500, roi, lo, hi) for lo, hi, _ in DIFF]
        rD3000 = [age_r(g, mans[3000], 3000, roi, lo, hi) for lo, hi, _ in DIFF]
        rS1500 = [age_r(g, mans[1500], 1500, roi, lo, hi) for lo, hi, _ in SIG]
        rS3000 = [age_r(g, mans[3000], 3000, roi, lo, hi) for lo, hi, _ in SIG]
        ax.axhline(0, color="gray", lw=0.8, alpha=0.7)
        ax.plot(x, rD1500, "o-",  color="#1f4e8a", lw=2, ms=7, label=r"diffusivity, $b=1500$")
        ax.plot(x, rD3000, "s--", color="#1f4e8a", lw=2, ms=6, label=r"diffusivity, $b=3000$")
        ax.plot(x, rS1500, "o-",  color="#c0392b", lw=2, ms=7, label=r"signal, $b=1500$")
        ax.plot(x, rS3000, "s--", color="#c0392b", lw=2, ms=6, label=r"signal, $b=3000$")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, rotation=12)
        ax.set_title(f"{roi}", fontsize=11)
        ax.set_xlabel("rung-pair gap (narrow $\\to$ extremal)", fontsize=10)
        ax.grid(alpha=0.25)
        if roi == "CR_all":
            ax.set_ylabel(r"Pearson $r$ vs age", fontsize=11)
            ax.legend(loc="best", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"gap_spectrum_age.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'gap_spectrum_age'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
