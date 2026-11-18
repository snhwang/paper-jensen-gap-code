"""The noise measures, and the noise-floor nulls compared.

Reads out/floor_null.csv (floor_null_pass.py). For the first-visit cohort:
  - Noise measures. By region, the median over sessions of sigma_sh4/sigma_sh6,
    sigma_sh8/sigma_sh6 (how much the SH order matters), sigma_b0/sigma_sh6 (what the b=0
    repeats add), sigma_sh6/sigma_cr (how the noise varies over the brain) and the b=0
    signal-to-noise ratio S0/sigma_sh6. By session, the ventricle floor (mean/SD, n_eff,
    mean/sigma_cr, sigma_vent/sigma_cr) and the background, median and 5th to 95th
    percentile, with the sex-adjusted r of n_eff and of sigma_vent/sigma_cr with age and
    the partial r (sex, age) of n_eff with eddy's outlier fraction.
  - Nulls. For each of the 18 nulls and each region, as split_half_stats.py: the null's
    change of the diffusivity gap from b=1500 to b=3000 as % of the data's b=1500 cohort
    mean ("floor"), the excess of the data's change over the null's per participant ("excess",
    so change = floor + excess, one-sample t, Bonferroni over 16 regions) and its
    sex-adjusted r with age, for the exact and split-half estimators (SH order 6), the
    split-half J_f of the null and the data's excess over it, and for the calibrated nulls
    the mean raise of the b=3000 level over the measured ratio.

Writes out/floor_null_noise.csv and out/floor_null_regions.csv (group-level) and
out/floor_null_stats.txt.
Usage:  python floor_null_stats.py [--partial]
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from config import OUT

ROIS = ["CR_all", "ACR", "SCR", "PCR", "Genu", "Body", "Splenium", "alic", "plic", "rlic", "ec",
        "slf", "ptr", "ss", "cingulum_cc", "tapetum"]
NULLS = [f"{p}{m}{c}" for p, m in (("T", "c4"), ("T", "a4"), ("T", "aN"), ("T", "aNv"), ("D", "c4"),
                                   ("D", "a4"), ("D", "aN"), ("D", "aNv"), ("D", "aNb")) for c in "uc"]
o = "_o6"
thr = 0.05 / 16

coh = pd.read_csv(OUT / "v1_cohort_sessions.csv")
mot = pd.read_csv(OUT / "motion.csv")
src = OUT / ("floor_null_partial.csv" if "--partial" in sys.argv else "floor_null.csv")
d = pd.read_csv(src).merge(coh, on="Session_ID").merge(mot, on="Subject_ID", how="left")
d["male"] = (d.sex == "M").astype(float)
N = d.Session_ID.nunique()


def partial(y, x, covs):
    X = np.column_stack([np.ones(len(y))] + covs)
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    r = np.corrcoef(ry, rx)[0, 1]
    df = len(y) - X.shape[1] - 1
    return r, 2 * stats.t.sf(abs(r) * np.sqrt(df / (1 - r * r)), df)


def q595(s):
    return f"median {s.median():.2f}, 5th to 95th percentile {s.quantile(0.05):.2f} to {s.quantile(0.95):.2f}"


def span(s):
    return f"{s.min():+.2f} to {s.max():+.2f}"


L = [f"{src.name}: {N} sessions; SH order 6 unless stated; Bonferroni over 16 regions (p < {thr:.4f})", ""]

# Noise measures ---------------------------------------------------------------------------
noise = []
for roi in ROIS:
    q = d[d.ROI == roi]
    noise.append(dict(ROI=roi, sh4_over_sh6=(q.sigma_sh4 / q.sigma_sh6).median(),
                      sh8_over_sh6=(q.sigma_sh8 / q.sigma_sh6).median(),
                      b0_over_sh6=(q.sigma_b0 / q.sigma_sh6).median(),
                      sigma_over_cr_core=(q.sigma_sh6 / q.sigma_cr).median(),
                      snr0=(q.S0 / q.sigma_sh6).median()))
noise = pd.DataFrame(noise).set_index("ROI")
noise.to_csv(OUT / "floor_null_noise.csv")
L.append("Noise per region, medians over sessions of the region means")
L.append(noise.round(3).to_string())
L.append(f"sigma_sh4/sigma_sh6 {noise.sh4_over_sh6.min():.3f} to {noise.sh4_over_sh6.max():.3f}, "
         f"sigma_sh8/sigma_sh6 {noise.sh8_over_sh6.min():.3f} to {noise.sh8_over_sh6.max():.3f}, "
         f"sigma_b0/sigma_sh6 {noise.b0_over_sh6.min():.2f} to {noise.b0_over_sh6.max():.2f}, "
         f"region sigma / corona radiata core {noise.sigma_over_cr_core.min():.2f} to "
         f"{noise.sigma_over_cr_core.max():.2f}")
f = d.drop_duplicates("Session_ID")
L += ["", "Per session (one rician image: mean/SD 1.91, mean 1.25 sigma; four averaged: 3.83, 2.51; "
          "four-channel noncentral chi: 3.94, 2.74)"]
for c in ("vent_mean_over_sd", "n_eff", "vent_mean_over_sigma", "sigma_vent_over_sigma", "bg_b0_over_b3000",
          "bg_mean_over_sd", "bg_mean_over_sigma"):
    L.append(f"  {c}: {q595(f[c].dropna())}")
L.append(f"  ventricle voxels: median {f.vent_n_voxels.median():.0f}; background voxels: median "
         f"{f.bg_n_voxels.median():.0f}")
a, m = f.age.values, [f.male.values]
for c in ("n_eff", "sigma_vent_over_sigma"):
    r, p = partial(f[c].values, a, m)
    L.append(f"  r of {c} with age (sex-adjusted) {r:+.3f}, p = {p:.1e}")
w = f.dropna(subset=["outlier_frac"])
r, p = partial(w.n_eff.values, w.outlier_frac.values, [w.male.values, w.age.values])
L.append(f"  partial r of n_eff with eddy's outlier fraction (sex, age) {r:+.3f}, p = {p:.1e}")

# Nulls ------------------------------------------------------------------------------------
e = d.copy()
e["dB_exact"] = e.Jln3000_exact - e.Jln1500_exact
e["dB_split"] = e[f"Jln3000_split{o}"] - e[f"Jln1500_split{o}"]
rows = []
for t in NULLS:
    nd = {"exact": e[f"{t}_Jln3000_exact"] - e[f"{t}_Jln1500_exact"],
          "split": e[f"{t}_Jln3000_split{o}"] - e[f"{t}_Jln1500_split{o}"]}
    for roi in ROIS:
        k = (e.ROI == roi).values
        q = e[k]
        a, m = q.age.values, [q.male.values]
        row = dict(null=t, ROI=roi)
        for est in ("exact", "split"):
            J1 = q[f"Jln1500_{est}" if est == "exact" else f"Jln1500_split{o}"].mean()
            ex = q[f"dB_{est}"] - nd[est][k]
            row[f"change_{est}"] = 100 * q[f"dB_{est}"].mean() / J1
            row[f"floor_{est}"] = 100 * nd[est][k].mean() / J1
            row[f"excess_{est}"] = 100 * ex.mean() / J1
            row[f"p_excess_{est}"] = stats.ttest_1samp(ex, 0).pvalue
            row[f"r_age_null_{est}"] = partial(nd[est][k].values, a, m)[0]
            row[f"r_age_excess_{est}"], row[f"p_age_excess_{est}"] = partial(ex.values, a, m)
        jx = q[f"Jf_split{o}"] - q[f"{t}_Jf_split{o}"]
        row.update(Jf_full=q[f"Jf_full{o}"].mean(), Jf_split=q[f"Jf_split{o}"].mean(),
                   Jf_null=q[f"{t}_Jf_split{o}"].mean(), Jf_excess=jx.mean(),
                   p_Jf_excess=stats.ttest_1samp(jx, 0).pvalue)
        row["r_age_Jf_excess"], row["p_age_Jf_excess"] = partial(jx.values, a, m)
        w = q.outlier_frac.notna().values
        row["r_outlier_Jf_excess"] = partial(jx.values[w], q.outlier_frac.values[w],
                                             [q.male.values[w], q.age.values[w]])[0]
        row["r_raise"] = q[f"{t}_r_raise"].mean() if t.endswith("c") else np.nan
        rows.append(row)
R = pd.DataFrame(rows)
R.to_csv(OUT / "floor_null_regions.csv", index=False)


def n_sig(T, p, x, sign):
    return int(((T[p] < thr) & (np.sign(T[x]) == sign)).sum())


summ = []
for t in NULLS:
    T = R[R.null == t].set_index("ROI")
    cr = T.loc["CR_all"]
    lo = T[(T.p_excess_exact < thr) & (T.excess_exact < 0)]
    summ.append(dict(null=t, floor_CR=cr.floor_exact, excess_CR=cr.excess_exact,
                     excess_neg=len(lo), excess_pos=n_sig(T, "p_excess_exact", "excess_exact", 1),
                     excess_min=-lo.excess_exact.max() if len(lo) else np.nan,
                     excess_max=-lo.excess_exact.min() if len(lo) else np.nan,
                     split_neg=n_sig(T, "p_excess_split", "excess_split", -1),
                     r_age_CR=cr.r_age_excess_exact,
                     age_pos=n_sig(T, "p_age_excess_exact", "r_age_excess_exact", 1),
                     age_neg=n_sig(T, "p_age_excess_exact", "r_age_excess_exact", -1),
                     Jf_above=n_sig(T, "p_Jf_excess", "Jf_excess", 1), Jf_null_CR=cr.Jf_null,
                     r_raise_CR=cr.r_raise))
S = pd.DataFrame(summ).set_index("null")
cr = R[(R.null == NULLS[0]) & (R.ROI == "CR_all")].iloc[0]
L += ["", "Nulls: T tensor, D directional diffusivities; c4 four-channel noncentral chi, a4 four averaged "
          "Rician, aN averaged with the session's n_eff (sigma from SH), aNv the same with the "
          "ventricle sigma, aNb the same with the b=0 sigma; u level from the measured ratio, c "
          "calibrated",
      f"Change of the diffusivity gap, % of its b=1500 cohort mean: corona radiata exact "
      f"{cr.change_exact:+.1f}, split half {cr.change_split:+.1f}",
      "floor_CR, excess_CR: points (exact); excess_neg/pos: regions with a significant excess "
      "decrease / increase (exact), by excess_min to excess_max points; split_neg: the same, split "
      "halves; r_age_CR, age_pos/neg: the excess's r with age; Jf_above: regions with split-half J_f "
      "above the null; r_raise_CR: calibrated level over the measured ratio, minus 1"]
L.append(S.to_string(float_format=lambda x: f"{x:.3g}"))
for t in ("DaNc", "DaNvc"):
    T = R[R.null == t].set_index("ROI")
    L += ["", f"{t} by region"]
    L.append(T[["change_exact", "floor_exact", "excess_exact", "p_excess_exact", "r_age_excess_exact",
                "excess_split", "p_excess_split", "Jf_null", "Jf_excess", "r_raise"]]
             .to_string(float_format=lambda x: f"{x:.3g}"))

# The numbers quoted in the paper (Sections 3.5.2, 4.4, S4, S5) ----------------------------
# PRIMARY: calibrated level, noise from n_eff averaged Rician images, both profiles, both
# sources of sigma. MAIN: the per-region null of Tables S4 and S5.
PRIMARY = ["TaNc", "TaNvc", "DaNc", "DaNvc"]
MAIN = {"tensor": "TaNc", "directional": "DaNc"}


def rng_(s, k=1):
    return f"{s.min():.{k}f} to {s.max():.{k}f}"


def regions(t, est, sign):
    T = R[R.null == t].set_index("ROI")
    return set(T.index[(T[f"p_excess_{est}"] < thr) & (np.sign(T[f"excess_{est}"]) == sign)])


P = S.loc[PRIMARY]
Rc = R[R.ROI == "CR_all"].set_index("null")
L += ["", "Paper (exact estimator unless stated; PRIMARY = " + ", ".join(PRIMARY) + ")"]
L.append(f"  corona radiata: measured change {cr.change_exact:+.1f}%, floor {rng_(-P.floor_CR)} points, "
         f"excess {rng_(-P.excess_CR)} points (split halves {rng_(-Rc.loc[PRIMARY, 'excess_split'])})")
L.append(f"  regions with a significant remaining decrease: {rng_(P.excess_neg, 0)} (split halves "
         f"{rng_(pd.Series([len(regions(t, 'split', -1)) for t in PRIMARY]), 0)}); reversed: "
         f"{rng_(P.excess_pos, 0)}, always including "
         f"{', '.join(sorted(set.intersection(*[regions(t, 'exact', 1) for t in PRIMARY])))}")
L.append(f"  excess r with age, corona radiata, all 18 nulls: exact {rng_(Rc.r_age_excess_exact, 2)}, split "
         f"{rng_(Rc.r_age_excess_split, 2)}; regions significant and positive, PRIMARY: {rng_(P.age_pos, 0)}")
U = S.loc[[t for t in NULLS if t.endswith("u")]]
L.append(f"  uncalibrated (9 nulls): remaining decrease significant in {rng_(U.excess_neg, 0)} regions, "
         f"corona radiata excess {rng_(-U.excess_CR)} points")
L.append(f"  four-channel noncentral chi, calibrated: corona radiata excess tensor {S.loc['Tc4c', 'excess_CR']:+.2f}, "
         f"directional {S.loc['Dc4c', 'excess_CR']:+.2f} points; regions with a remaining decrease "
         f"{S.loc['Tc4c', 'excess_neg']}, {S.loc['Dc4c', 'excess_neg']}")
cc = R[R.null.isin(PRIMARY) & R.ROI.isin(["Genu", "Body", "Splenium"])]
L.append(f"  calibration raised the b=3000 level by {rng_(100 * Rc.loc[PRIMARY, 'r_raise'])}% in the corona "
         f"radiata, {rng_(100 * cc.r_raise)}% in the corpus callosum (PRIMARY)")
dif = pd.concat([R[R.null == f"{p}aNc"].set_index("ROI").r_age_excess_exact
                 - R[R.null == f"{p}a4c"].set_index("ROI").r_age_excess_exact for p in "TD"], axis=1, keys=list("TD"))
L.append(f"  n_eff by session against a fixed 4 (aNc minus a4c): age r of the excess {rng_(dif.loc['CR_all'], 2)} "
         f"higher in the corona radiata, {rng_(dif.stack(), 2)} over the regions; with n_eff fixed at 4, "
         f"significant and positive in {S.loc['Ta4c', 'age_pos']} (tensor) and {S.loc['Da4c', 'age_pos']} "
         f"(directional) regions")
for name, t in MAIN.items():
    T = R[R.null == t].set_index("ROI")
    c0 = T.loc["CR_all"]
    jneg = (T.p_age_Jf_excess < thr) & (T.r_age_Jf_excess < 0)
    L.append(f"  {name} ({t}): split-half J_f above the null in {S.loc[t, 'Jf_above']}/16 (not: "
             f"{', '.join(T.index[~((T.p_Jf_excess < thr) & (T.Jf_excess > 0))]) or 'none'}); corona radiata null "
             f"{c0.Jf_null:.1e}, excess {c0.Jf_excess:.1e}; excess r with age {span(T.r_age_Jf_excess)} "
             f"(corona radiata {c0.r_age_Jf_excess:+.2f}), significant and negative in {int(jneg.sum())}/16; "
             f"partial r with the outlier fraction {span(T.r_outlier_Jf_excess)}")
    sn = T[(T.p_excess_split < thr) & (T.excess_split < 0)]
    L.append(f"    split halves, diffusivity gap: remaining decrease in {len(sn)}/16 (by "
             f"{-sn.excess_split.max():.1f} to {-sn.excess_split.min():.1f} points), excess r with age "
             f"{span(T.r_age_excess_split)}, significant and positive in "
             f"{n_sig(T, 'p_age_excess_split', 'r_age_excess_split', 1)}/16")

text = "\n".join(L)
print(text)
(OUT / "floor_null_stats.txt").write_text(text + "\n", encoding="utf-8")
