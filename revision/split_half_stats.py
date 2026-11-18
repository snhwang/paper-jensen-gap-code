"""The dependence of the diffusivity gap on b, and J_f, against the noise-floor null.

Reads out/split_half.csv (split_half_pass.py). By region, at SH order 6 (order 4
as a check), for the first-visit cohort:
  - the change of the diffusivity gap from b=1500 to b=3000, as % of its b=1500
    cohort mean: exact (the paper's estimator, noise variance included) and split
    half (noise variance removed), for the data and for the primary null nullC_
    (scaled b=3000 signal, four-channel noncentral chi), whose true change is 0.
    The Rician nulls are listed for comparison.
  - the null's change as % of the data's b=1500 cohort mean ("floor"), and the
    excess of the data's change over the null's, per participant, as % of the
    same mean, so that change = floor + excess. The excess is tested against 0
    (one-sample t, Bonferroni over 16 regions).
  - sex-adjusted partial r with age of the data's change, of the null's, and of
    the excess.
  - J_f from all the data and from split halves, the split-half J_f of the null,
    the excess over the null (tested as above) and its r with age. Also the
    smallest J_f that the excess change of the gap allows: if ln f_i = c - beta
    ln(D_i/<D>) in every direction, J_ln(3000) = (1-beta)^2 J_ln(1500) and
    J_f = beta^2 J_ln(1500), and any other form of f gives a larger J_f for the
    same change (as jf_stats.py). And the partial r (sex, age) of the J_f excess
    with eddy's outlier fraction (out/motion.csv), since outlier replacement
    predicts a slice from the other volumes and so could couple the two halves.
  - the share of each gap that is noise variance, 1 - split/full, for the data
    and the null.
  - the floor, excess and r with age again against the lighter null nullA_ (the
    mean of four Rician images), columns ending _avg4. The CSF floor lies between
    the two nulls' zero-signal floors, so the two bracket the floor's effect.
  - the CSF floor at b=3000: median and 5th to 95th percentile over sessions of
    mean/SD and of mean/sigma, against the zero-signal values of one Rician
    magnitude (1.91, 1.25 sigma), of the mean of four Rician images (3.83, 2.51
    sigma) and of a four-channel noncentral chi (3.94, 2.74 sigma).

Writes out/split_half_regions.csv (group-level) and out/split_half_stats.txt.
Usage:  python split_half_stats.py [--partial]
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from config import OUT

ROIS = ["CR_all", "ACR", "SCR", "PCR", "Genu", "Body", "Splenium", "alic", "plic", "rlic", "ec",
        "slf", "ptr", "ss", "cingulum_cc", "tapetum"]
NULLS = {"nullC_": "nc4", "nullA_": "avg4", "nullS_": "riceS", "null_": "rice"}
BRACKET = {"nullC_": "", "nullA_": "_avg4"}

coh = pd.read_csv(OUT / "v1_cohort_sessions.csv")
mot = pd.read_csv(OUT / "motion.csv")
src = OUT / ("split_half_partial.csv" if "--partial" in sys.argv else "split_half.csv")
d = pd.read_csv(src).merge(coh, on="Session_ID")
d = d.merge(mot, on="Subject_ID", how="left")
d["male"] = (d.sex == "M").astype(float)
N = d.Session_ID.nunique()


def partial(y, x, covs):
    X = np.column_stack([np.ones(len(y))] + covs)
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    r = np.corrcoef(ry, rx)[0, 1]
    df = len(y) - X.shape[1] - 1
    return r, 2 * stats.t.sf(abs(r) * np.sqrt(df / (1 - r * r)), df)


def region_table(k):
    o = f"_o{k}"
    e = d.copy()
    for t in ["", *NULLS]:
        e[f"{t}dB_exact"] = e[f"{t}Jln3000_exact"] - e[f"{t}Jln1500_exact"]
        e[f"{t}dB_split"] = e[f"{t}Jln3000_split{o}"] - e[f"{t}Jln1500_split{o}"]
    for t, x in BRACKET.items():
        e[f"ex_exact{x}"] = e.dB_exact - e[f"{t}dB_exact"]
        e[f"ex_split{x}"] = e.dB_split - e[f"{t}dB_split"]
        e[f"Jf_ex{x}"] = e[f"Jf_split{o}"] - e[f"{t}Jf_split{o}"]
    rows = []
    for roi in ROIS:
        q = e[e.ROI == roi]
        a, m = q.age.values, [q.male.values]
        J1x, J1s = q.Jln1500_exact.mean(), q[f"Jln1500_split{o}"].mean()
        row = dict(ROI=roi, n=len(q), Jln1500_exact=J1x, Jln3000_exact=q.Jln3000_exact.mean(),
                   Jln1500_split=J1s, Jln3000_split=q[f"Jln3000_split{o}"].mean())
        for t, name in (("", "data"), *NULLS.items()):
            row[f"change_exact_{name}"] = 100 * q[f"{t}dB_exact"].mean() / q[f"{t}Jln1500_exact"].mean()
            row[f"change_split_{name}"] = 100 * q[f"{t}dB_split"].mean() / q[f"{t}Jln1500_split{o}"].mean()
        for est, J1 in (("exact", J1x), ("split", J1s)):
            row[f"r_age_data_{est}"] = partial(q[f"dB_{est}"].values, a, m)[0]
            for t, x in BRACKET.items():
                ex = q[f"ex_{est}{x}"]
                row[f"floor_{est}{x}"] = 100 * q[f"{t}dB_{est}"].mean() / J1
                row[f"excess_{est}{x}"] = 100 * ex.mean() / J1
                row[f"p_excess_{est}{x}"] = stats.ttest_1samp(ex, 0).pvalue
                row[f"r_age_null_{est}{x}"] = partial(q[f"{t}dB_{est}"].values, a, m)[0]
                row[f"r_age_excess_{est}{x}"], row[f"p_age_excess_{est}{x}"] = partial(ex.values, a, m)
        J3 = J1s + q.ex_split.mean()
        beta = 1 - np.sqrt(J3 / J1s)
        row.update(Jf_full=q[f"Jf_full{o}"].mean(), Jf_split=q[f"Jf_split{o}"].mean(),
                   Jf_null=q[f"nullC_Jf_split{o}"].mean(), Jf_excess=q.Jf_ex.mean(),
                   p_Jf_excess=stats.ttest_1samp(q.Jf_ex, 0).pvalue, Jf_min=beta ** 2 * J1s)
        row["r_age_Jf_excess"], row["p_age_Jf_excess"] = partial(q.Jf_ex.values, a, m)
        row.update(Jf_null_avg4=q[f"nullA_Jf_split{o}"].mean(), Jf_excess_avg4=q.Jf_ex_avg4.mean(),
                   p_Jf_excess_avg4=stats.ttest_1samp(q.Jf_ex_avg4, 0).pvalue)
        row["r_age_Jf_excess_avg4"], row["p_age_Jf_excess_avg4"] = partial(q.Jf_ex_avg4.values, a, m)
        w = q.dropna(subset=["outlier_frac"])
        row["r_outlier_Jf_excess"] = partial(w.Jf_ex.values, w.outlier_frac.values,
                                             [w.male.values, w.age.values])[0]
        for s in (1500, 3000):
            row[f"noise_share_{s}"] = 100 * (1 - q[f"Jln{s}_split{o}"].mean() / q[f"Jln{s}_full{o}"].mean())
            row[f"noise_share_{s}_null"] = 100 * (1 - q[f"nullC_Jln{s}_split{o}"].mean()
                                                  / q[f"nullC_Jln{s}_full{o}"].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("ROI")


T = region_table(6)
T.to_csv(OUT / "split_half_regions.csv")
T4 = region_table(4)
thr = 0.05 / 16


def span(s):
    return f"{s.min():+.2f} to {s.max():+.2f}"


def n_sig(T, p, x, sign):
    return int(((T[p] < thr) & (np.sign(T[x]) == sign)).sum())


cr = T.loc["CR_all"]
L = [f"{src.name}: {N} sessions, SH order 6 unless stated; Bonferroni over 16 regions (p < {thr:.4f})", ""]
L.append("Change of the diffusivity gap from b=1500 to 3000, % of its b=1500 cohort mean")
L.append(T[["change_exact_data", "floor_exact", "excess_exact", "change_split_data", "floor_split",
            "excess_split", "change_split_nc4", "change_split_riceS", "change_split_rice"]].round(1).to_string())
for est in ("exact", "split"):
    lo = T[(T[f"p_excess_{est}"] < thr) & (T[f"excess_{est}"] < 0)]
    L.append(f"{est}: data's decrease beyond the null's, significant in {len(lo)}/16 "
             f"(by {-lo[f'excess_{est}'].max():.1f} to {-lo[f'excess_{est}'].min():.1f} points); "
             f"significant the other way in {n_sig(T, f'p_excess_{est}', f'excess_{est}', 1)}/16; "
             f"not significant: {', '.join(T.index[T[f'p_excess_{est}'] >= thr]) or 'none'}")
    L.append(f"  {est}: data {T[f'change_{est}_data'].min():+.1f}% to {T[f'change_{est}_data'].max():+.1f}%, "
             f"floor {T[f'floor_{est}'].min():+.1f} to {T[f'floor_{est}'].max():+.1f} points; corona radiata: "
             f"data {cr[f'change_{est}_data']:+.1f}%, floor {cr[f'floor_{est}']:+.1f}, "
             f"excess {cr[f'excess_{est}']:+.1f} points")
L.append(f"null's change as % of its own b=1500 gap, exact: {T.change_exact_nc4.min():+.1f}% to "
         f"{T.change_exact_nc4.max():+.1f}%; split half: {T.change_split_nc4.min():+.1f}% to "
         f"{T.change_split_nc4.max():+.1f}%")
L.append(f"order 4, split half: excess significant and negative in "
         f"{n_sig(T4, 'p_excess_split', 'excess_split', -1)}/16, positive in "
         f"{n_sig(T4, 'p_excess_split', 'excess_split', 1)}/16")
L += ["", "Sex-adjusted r with age of the change"]
L.append(T[["r_age_data_exact", "r_age_null_exact", "r_age_excess_exact", "r_age_data_split",
            "r_age_null_split", "r_age_excess_split"]].round(3).to_string())
for est in ("exact", "split"):
    L.append(f"{est}: data {span(T[f'r_age_data_{est}'])} (corona radiata {cr[f'r_age_data_{est}']:+.2f}); "
             f"null {span(T[f'r_age_null_{est}'])} (corona radiata {cr[f'r_age_null_{est}']:+.2f}); "
             f"excess {span(T[f'r_age_excess_{est}'])} (corona radiata {cr[f'r_age_excess_{est}']:+.2f}), "
             f"significant and positive in {n_sig(T, f'p_age_excess_{est}', f'r_age_excess_{est}', 1)}/16")
L.append(f"order 4, split half: excess r {span(T4.r_age_excess_split)}, significant and positive in "
         f"{n_sig(T4, 'p_age_excess_split', 'r_age_excess_split', 1)}/16")
L += ["", "J_f, cohort means"]
L.append(T[["Jf_full", "Jf_split", "Jf_null", "Jf_excess", "p_Jf_excess", "Jf_min", "r_age_Jf_excess",
            "r_outlier_Jf_excess"]].to_string(float_format=lambda x: f"{x:.3g}"))
pos = T[(T.p_Jf_excess < thr) & (T.Jf_excess > 0)]
L.append(f"split-half J_f above the null in {len(pos)}/16; share of the full-data J_f that is noise variance "
         f"{100 * (1 - T.Jf_split / T.Jf_full).min():.0f}% to {100 * (1 - T.Jf_split / T.Jf_full).max():.0f}%")
L.append(f"  corona radiata: full {cr.Jf_full:.1e}, split half {cr.Jf_split:.1e}, null {cr.Jf_null:.1e}, "
         f"excess {cr.Jf_excess:.1e}, smallest allowed by the change of the gap {cr.Jf_min:.1e} "
         f"(excess / smallest {cr.Jf_excess / cr.Jf_min:.1f})")
L.append(f"  excess r with age {span(T.r_age_Jf_excess)} (corona radiata {cr.r_age_Jf_excess:+.2f}), "
         f"significant and negative in {n_sig(T, 'p_age_Jf_excess', 'r_age_Jf_excess', -1)}/16; "
         f"partial r with eddy's outlier fraction {span(T.r_outlier_Jf_excess)}")
L.append(f"order 4: split-half J_f above the null in "
         f"{int(((T4.p_Jf_excess < thr) & (T4.Jf_excess > 0)).sum())}/16")
L += ["", "Against the lighter null nullA_ (mean of four Rician images); primary nullC_ beside it"]
L.append(T[["floor_exact", "floor_exact_avg4", "excess_exact", "excess_exact_avg4", "r_age_excess_exact",
            "r_age_excess_exact_avg4", "r_age_null_exact", "r_age_null_exact_avg4"]].round(2).to_string())
for est in ("exact", "split"):
    x = "_avg4"
    lo = T[(T[f"p_excess_{est}{x}"] < thr) & (T[f"excess_{est}{x}"] < 0)]
    L.append(f"{est}: floor {T[f'floor_{est}{x}'].min():+.1f} to {T[f'floor_{est}{x}'].max():+.1f} points "
             f"(primary {T[f'floor_{est}'].min():+.1f} to {T[f'floor_{est}'].max():+.1f}); excess negative and "
             f"significant in {len(lo)}/16 (by "
             f"{-lo[f'excess_{est}{x}'].max():.1f} to {-lo[f'excess_{est}{x}'].min():.1f} points), positive in "
             f"{n_sig(T, f'p_excess_{est}{x}', f'excess_{est}{x}', 1)}/16; corona radiata floor "
             f"{cr[f'floor_{est}{x}']:+.1f}, excess {cr[f'excess_{est}{x}']:+.1f}")
    L.append(f"  excess r with age {span(T[f'r_age_excess_{est}{x}'])} (corona radiata "
             f"{cr[f'r_age_excess_{est}{x}']:+.2f}), significant and positive in "
             f"{n_sig(T, f'p_age_excess_{est}{x}', f'r_age_excess_{est}{x}', 1)}/16; null r "
             f"{span(T[f'r_age_null_{est}{x}'])} (corona radiata {cr[f'r_age_null_{est}{x}']:+.2f})")
L.append(f"split-half J_f above nullA_ in {n_sig(T, 'p_Jf_excess_avg4', 'Jf_excess_avg4', 1)}/16 (corona radiata "
         f"null {cr.Jf_null_avg4:.1e}, excess {cr.Jf_excess_avg4:.1e}); excess r with age "
         f"{span(T.r_age_Jf_excess_avg4)}, significant and negative in "
         f"{n_sig(T, 'p_age_Jf_excess_avg4', 'r_age_Jf_excess_avg4', -1)}/16")
L += ["", "Share of the full-data gap that is noise variance, 1 - split/full, % (data, null)"]
for s in (1500, 3000):
    L.append(f"  b={s}: data {T[f'noise_share_{s}'].min():.1f} to {T[f'noise_share_{s}'].max():.1f} "
             f"(corona radiata {cr[f'noise_share_{s}']:.1f}), null {T[f'noise_share_{s}_null'].min():.1f} to "
             f"{T[f'noise_share_{s}_null'].max():.1f} (corona radiata {cr[f'noise_share_{s}_null']:.1f})")
f = d.drop_duplicates("Session_ID")
v = f.dropna(subset=["csf_mean_over_sd"])
L += ["", f"CSF floor at b=3000, {len(v)} of {N} sessions with at least 10 CSF voxels "
          f"(median {v.csf_n_voxels.median():.0f} voxels)"]
for c, ref in (("csf_mean_over_sd", "one Rician 1.91, four averaged Rician 3.83, four-channel noncentral chi 3.94"),
               ("csf_mean_over_sigma", "one Rician 1.25, four averaged Rician 2.51, four-channel noncentral chi 2.74")):
    L.append(f"  {c}: median {v[c].median():.2f}, 5th to 95th percentile {v[c].quantile(0.05):.2f} to "
             f"{v[c].quantile(0.95):.2f} ({ref})")
text = "\n".join(L)
print(text)
(OUT / "split_half_stats.txt").write_text(text + "\n", encoding="utf-8")
