"""Reproduce and VERIFY the ROI-level scalar statistics in the manuscript.

    Covers the gap definitions, the cross-sectional and longitudinal age
    associations, the gap spectrum and the FA comparisons. It does NOT cover the
    attenuation gap J_f, the small-gap expansion accuracy or the age top-coding
    sensitivity; each of those has its own script, listed in the README table.

Each computed value is checked against the number quoted in the paper and marked
[ OK ] or [FAIL]; the script exits non-zero if anything fails, so it doubles as a
regression test. All age associations are SEX-ADJUSTED partial correlations
(residualize the metric and age on [intercept, sex], Pearson of the residuals),
or sex as a covariate in the mixed model (longitudinal). The 56 participants
whose age is HIPAA-masked as "90 or older" are coded as age 90 and retained via
clean_age(), so every analysis runs on N = 1,379 (the paper's convention) --
using the string age directly would silently drop them and shift, e.g., FA.

Data root is taken from the JG_DATA environment variable (default: current dir),
and its HCP/ subfolder must contain:
    HCP/cr_gaps_long.csv
    HCP/manifest_n1379_b{1500,3000}.tsv
    HCP/cr_metrics_long_b{1500,3000}_n1379.csv      (for FA)
    HCP/AABC2_subjects_*.csv                         (per-visit age/sex)

Usage:
    JG_DATA=/path/to/analysis_tree python reproduce_stats.py

The gap-spectrum figure and the voxelwise TBSS scalars (Dice, corona-radiata
ratios) have their own scripts; this driver covers every ROI-level scalar.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DATA = Path(os.environ.get("JG_DATA", "."))
HCP = DATA / "HCP"
RTOL = 0.015          # correlations quoted to 2-3 dp: pass within 0.015, catch real errors
_FAILS = []


def clean_age(s):
    """Numeric age with the HIPAA "90 or older" top-code mapped to 90 (retained)."""
    return pd.to_numeric(s.astype(str).str.replace("90 or older", "90", regex=False),
                         errors="coerce")


def check(label, computed, expected, tol=RTOL, fmt="{:+.3f}"):
    ok = abs(computed - expected) <= tol
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label:36s} computed {fmt.format(computed):>9s}"
          f"   paper {fmt.format(expected):>8s}")
    if not ok:
        _FAILS.append(label)


def check_int(label, computed, expected):
    ok = int(computed) == int(expected)
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label:36s} computed {int(computed):>9d}   paper {int(expected):>8d}")
    if not ok:
        _FAILS.append(label)


def check_true(label, ok, detail=""):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label:36s} {detail}")
    if not ok:
        _FAILS.append(label)


g = pd.read_csv(HCP / "cr_gaps_long.csv")
mans = {b: pd.read_csv(HCP / f"manifest_n1379_b{b}.tsv", sep="\t") for b in (1500, 3000)}
for b in mans:
    mans[b]["age"] = clean_age(mans[b]["age"])


def partial_r(v, age, sex):
    d = pd.DataFrame({"v": pd.to_numeric(v, errors="coerce"), "a": age, "s": sex}).dropna()
    X = np.column_stack([np.ones(len(d)), pd.get_dummies(d["s"], drop_first=True).astype(float).values])
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    r, p = stats.pearsonr(res(d["v"].values), res(d["a"].values))
    return r, p, len(d)


def merged(bval):
    return (g[g.shell_bval == bval]
            .merge(mans[bval][["session_id", "age", "sex"]],
                   left_on="Session_ID", right_on="session_id", how="inner"))


def col_r(bval, roi, col):
    s = merged(bval); s = s[s.ROI == roi]
    return partial_r(s[col].values, s["age"].values, s["sex"].values)[0]


def logmean_r(bval, roi, lo, hi):
    s = merged(bval); s = s[s.ROI == roi]
    L = pd.to_numeric(s[lo + "_mean"], errors="coerce").clip(lower=1e-12)
    H = pd.to_numeric(s[hi + "_mean"], errors="coerce").clip(lower=1e-12)
    return partial_r(np.log(H / L).values, s["age"].values, s["sex"].values)[0]


print("=" * 74 + "\nCROSS-SECTIONAL, sex-adjusted partial r  (Section 7.1)\n" + "=" * 74)
XSEC = {(1500, "CR_all"): (-0.472, +0.301), (1500, "SCR"): (-0.334, +0.325),
        (3000, "CR_all"): (-0.415, +0.354), (3000, "SCR"): (-0.255, +0.372)}
for (bval, roi), (eJ, eS) in XSEC.items():
    check(f"b{bval} {roi} J_ln",  col_r(bval, roi, "J_ln_D_mean"), eJ)
    check(f"b{bval} {roi} J_lnS", col_r(bval, roi, "J_ln_S_mean"), eS)
check("b1500 CR_all ADC_1 (trace)", col_r(1500, "CR_all", "ADC_1_mean"), +0.71)

print("\nFA vs age, sex-adjusted  (Section 7.1)")
for b, eFA in ((1500, -0.54), (3000, -0.45)):
    d = pd.read_csv(HCP / f"cr_metrics_long_b{b}_n1379.csv"); s = d[d.ROI == "CR_all"]
    r = partial_r(s.FA_mean.values, clean_age(s.Age).values, s.Sex.values)
    check(f"FA CR_all b{b}", r[0], eFA)
    check_int(f"  n (90+ retained) b{b}", r[2], 1379)

print("\nWITHIN-SEX dissociation, CR_all b1500  (Section 7.1)")
s = merged(1500); s = s[s.ROI == "CR_all"]
for sx, (eJ, eS) in (("F", (-0.52, +0.27)), ("M", (-0.41, +0.34))):
    d = s[s.sex == sx]
    for c, nm, e in (("J_ln_D_mean", "J_ln", eJ), ("J_ln_S_mean", "J_lnS", eS)):
        v = pd.to_numeric(d[c], errors="coerce"); ok = v.notna() & d.age.notna()
        check(f"sex={sx} {nm}", stats.pearsonr(v[ok], d.age[ok])[0], e)

print("\nAGE x SEX interaction, CR_all b1500  (Section 7.1)")
s = merged(1500); s = s[s.ROI == "CR_all"].copy()
s["age_c"] = s["age"] - s["age"].mean()
pv = {}
for col in ("J_ln_D_mean", "J_ln_S_mean"):
    s[col] = pd.to_numeric(s[col], errors="coerce")
    m = smf.ols(f"{col} ~ age_c * C(sex)", s.dropna(subset=[col, "age_c", "sex"])).fit()
    pv[col] = m.pvalues[[t for t in m.params.index if ":" in t][0]]
check_true("J_ln interaction significant", pv["J_ln_D_mean"] < 0.01, f"p={pv['J_ln_D_mean']:.3f} (paper ~0.004)")
check_true("J_lnS interaction n.s.",       pv["J_ln_S_mean"] > 0.05, f"p={pv['J_ln_S_mean']:.3f} (paper n.s.)")

print("\nSPECTRUM log-ratio of ROI-mean Holder means  (Section 7.2)")
check("CR_all G_0,1 (=J_ln)",      logmean_r(1500, "CR_all", "ADC_0", "ADC_1"),   -0.600)
check("CR_all G_min,max",          logmean_r(1500, "CR_all", "ADC_min", "ADC_max"), -0.724)
check("CR_all G^S_0,1 (=J_lnS)",   logmean_r(1500, "CR_all", "M0_S", "M1_S"),    +0.354)
check("CR_all G^S_min,max",        logmean_r(1500, "CR_all", "M_min_S", "M_max_S"), +0.086)
check("SCR G_min,max",             logmean_r(1500, "SCR", "ADC_min", "ADC_max"), -0.688)

print("\nLONGITUDINAL mixed model, CR_all b1500  (Section 7.5)")
amap = pd.read_csv(glob.glob(str(HCP / "AABC2_subjects_*.csv"))[0], low_memory=False).dropna(subset=["id_event"]).copy()
amap["Subject_ID"] = amap.id_event.str.rsplit("_", n=1).str[0]
amap["Visit"] = amap.id_event.str.rsplit("_", n=1).str[1]
amap["age_yr"] = clean_age(amap.age_open)
sl = g[(g.shell_bval == 1500) & (g.ROI == "CR_all")].merge(
    amap[["Subject_ID", "Visit", "age_yr", "sex"]], on=["Subject_ID", "Visit"], how="left")
for c in ("J_ln_D_mean", "J_ln_S_mean"):
    sl[c] = pd.to_numeric(sl[c], errors="coerce")
sl = sl.dropna(subset=["J_ln_D_mean", "J_ln_S_mean", "age_yr", "sex"])
sl = sl[sl.groupby("Subject_ID").age_yr.transform("count") >= 2]
for c, nm, e in (("J_ln_D_mean", "J_ln", -0.42), ("J_ln_S_mean", "J_lnS", +0.17)):
    m = smf.mixedlm(f"{c} ~ age_yr + C(sex)", sl, groups=sl.Subject_ID).fit(method="lbfgs")
    check(f"{nm} within-subject %/yr", 100 * m.params["age_yr"] / sl[c].mean(), e, tol=0.03, fmt="{:+.2f}")
check_int("longitudinal n subjects", sl.Subject_ID.nunique(), 861)

print("\nb-SIGNATURE  (Section 7.5)")
ROIS = ["CR_all", "ACR", "SCR", "PCR", "Body", "Genu", "Splenium", "alic", "plic",
        "rlic", "ec", "slf", "ptr", "cingulum_cc", "ss", "tapetum"]
roi_mean = lambda bval, roi: pd.to_numeric(
    g[(g.shell_bval == bval) & (g.ROI == roi)].J_ln_D_mean, errors="coerce").mean()
a = np.array([[roi_mean(1500, r), roi_mean(3000, r)] for r in ROIS])
check("b1500->b3000 mean shift %", 100 * (a[:, 1] - a[:, 0]).mean() / a[:, 0].mean(), -22.5,
      tol=1.0, fmt="{:+.1f}")
check_int("ROIs below y=x", int((a[:, 1] < a[:, 0]).sum()), 16)

print("\nGAP NON-GAUSSIANITY, Delta_b J_ln vs age  (Section 7.5)")
ng = {}
for roi in ROIS:
    s = g[g.ROI == roi].copy()
    s["J_ln_D_mean"] = pd.to_numeric(s.J_ln_D_mean, errors="coerce")
    piv = s.pivot_table(index="Session_ID", columns="shell_bval",
                        values="J_ln_D_mean").dropna(subset=[1500, 3000])
    piv["delta"] = piv[3000] - piv[1500]
    j = piv.merge(mans[1500][["session_id", "age", "sex"]],
                  left_on="Session_ID", right_on="session_id", how="inner")
    ng[roi] = partial_r(j["delta"].values, j["age"].values, j["sex"].values)[0]
check("CR_all Delta_b J_ln", ng["CR_all"], +0.53)
check_true("16-ROI range +0.28..+0.59", +0.26 <= min(ng.values()) and max(ng.values()) <= +0.61,
           f"[{min(ng.values()):+.3f}, {max(ng.values()):+.3f}]")

print("\n" + "=" * 74)
if _FAILS:
    print(f"FAILED: {len(_FAILS)} check(s) did not match the paper: {_FAILS}")
    sys.exit(1)
print("ALL CHECKS PASSED. Every reproduced value matches the manuscript.")
