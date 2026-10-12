"""Analyses that need the native pass (native_pass.py), first-visit cohort.

1. Check that the pass reproduces the gap table (cr_gaps_long.csv).
2. Across-participant Spearman correlation of each gap with FA, by region
   (Figure 3c,d; written to out/fa_correlations.csv for fig3_fa_relation.py).
3. Age effects not explained by the tensor and noise (Section 4.6, Table S4):
   the region mean of (measured gap - simulated gap of tensor plus noise),
   correlated with age, also without the noise term and with the b=0 noise
   estimate, and with the signal-to-noise ratio as a covariate.
3b. Dependence on b with the signal-to-noise ratio as a covariate (Section 4.4).
3c. Main age associations and Delta_b after subtracting the noise inflation
   (Sections 4.2, 4.4, Supplementary S3).
4. Summary numbers quoted in the text: ADC_1/MD ratio (Section 3.1), noise
   inflation of both gaps (Sections 4.2, 5.3), and how often the simulation
   over-predicts the measured gap (Sections 3.5.2, 4.6).

Inputs:  out/native_pass.csv, out/v1_cohort_sessions.csv, config.HCP tables
Outputs: out/fa_correlations.csv, out/beyond_tensor.csv, out/bdep_snr.csv,
         out/noise_corrected.csv, out/post_native_summary.txt
Usage:   python post_native.py [--partial]
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from config import HCP, OUT, ROIS

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def partial(y, age, covs):
    X = np.column_stack([np.ones(len(y))] + covs)
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    ra = age - X @ np.linalg.lstsq(X, age, rcond=None)[0]
    r = np.corrcoef(ry, ra)[0, 1]
    df = len(y) - X.shape[1] - 1
    t = r * np.sqrt(df / (1 - r * r))
    return r, 2 * stats.t.sf(abs(t), df)


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


src = OUT / ("native_pass_partial.csv" if "--partial" in sys.argv else "native_pass.csv")
nat = pd.read_csv(src)
coh = pd.read_csv(OUT / "v1_cohort_sessions.csv")
d = nat.merge(coh, on="Session_ID")
d["male"] = (d.sex == "M").astype(float)
say(f"{src.name}: {d.Session_ID.nunique()} sessions")

# 1. Reproduction of the gap table ----------------------------------------
g = pd.read_csv(HCP / "cr_gaps_long.csv", usecols=["Session_ID", "ROI", "shell_bval",
                                                    "J_ln_D_mean", "J_ln_S_mean", "n_voxels"])
j = d.merge(g, on=["Session_ID", "ROI", "shell_bval"])
say("vs gap table: max relative difference J_ln %.1e, J_lnS %.1e; voxel counts equal %.4f" % (
    (abs(j.J_ln / j.J_ln_D_mean - 1)).max(), (abs(j.J_lnS / j.J_ln_S_mean - 1)).max(),
    (j.n_voxels_x == j.n_voxels_y).mean()))

# 2. Correlation with FA across participants ------------------------------
rows = []
for (b, roi), s in d.groupby(["shell_bval", "ROI"]):
    rows.append(dict(shell=b, ROI=roi,
                     rho_Jln_FA=stats.spearmanr(s.J_ln, s.FA)[0],
                     rho_JlnS_FA=stats.spearmanr(s.J_lnS, s.FA)[0],
                     rho_JlnS_MD=stats.spearmanr(s.J_lnS, s.MD)[0]))
F = pd.DataFrame(rows)
F.to_csv(OUT / "fa_correlations.csv", index=False)
for b in (1500, 3000):
    q = F[F.shell == b]
    say(f"b={b}: rho(J_ln,FA) {q.rho_Jln_FA.min():.2f}-{q.rho_Jln_FA.max():.2f}, "
        f"rho(J_lnS,FA) {q.rho_JlnS_FA.min():.2f}-{q.rho_JlnS_FA.max():.2f}, "
        f"CR_all {q[q.ROI == 'CR_all'].rho_Jln_FA.item():.3f} / "
        f"{q[q.ROI == 'CR_all'].rho_JlnS_FA.item():.3f}")

# 3. Beyond the tensor and noise -----------------------------------------
say("\nSNR (sh) vs age, partial r, CR_all:",
    *[f"b={b} {partial(s.SNR.values, s.age.values, [s.male.values])[0]:+.3f}"
      for b, s in d[d.ROI == "CR_all"].groupby("shell_bval")])
rows = []
for var in ("T", "TN", "TNb0"):
    for gap in ("J_ln", "J_lnS"):
        for (b, roi), s in d.groupby(["shell_bval", "ROI"]):
            ex = (s[gap] - s[f"{gap}_{var}"]).values
            r, p = partial(ex, s.age.values, [s.male.values])
            r2, p2 = partial(ex, s.age.values, [s.male.values, s.SNR.values])
            rows.append(dict(variant=var, gap=gap, shell=b, ROI=roi,
                             excess_pct=100 * np.mean(ex) / s[gap].mean(),
                             r=r, p=p, r_snr=r2, p_snr=p2))
X = pd.DataFrame(rows)
for var in ("T", "TN", "TNb0"):
    m = X.variant == var
    X.loc[m, "q"] = bh(X.loc[m, "p"])
    X.loc[m, "q_snr"] = bh(X.loc[m, "p_snr"])
X.to_csv(OUT / "beyond_tensor.csv", index=False)
for var, name in (("T", "tensor only (no noise)"), ("TN", "tensor + noise (DW residual)"),
                  ("TNb0", "tensor + noise (b0 variance)")):
    say(f"\n{name}")
    for gap in ("J_ln", "J_lnS"):
        for b in (1500, 3000):
            q = X[(X.variant == var) & (X.gap == gap) & (X.shell == b)]
            c = q[q.ROI == "CR_all"].iloc[0]
            say(f"  {gap:5s} b={b}: excess {q.excess_pct.median():+5.1f}% (median over regions); "
                f"age r {q.r.min():+.2f}..{q.r.max():+.2f}, CR_all {c.r:+.3f}; "
                f"q<0.05: {int((q.q < 0.05).sum())}/16 "
                f"(+{int(((q.q < 0.05) & (q.r > 0)).sum())}, -{int(((q.q < 0.05) & (q.r < 0)).sum())}); "
                f"with SNR: CR_all {c.r_snr:+.3f}, q<0.05 {int((q.q_snr < 0.05).sum())}/16")

# 3b. Dependence on b, with and without SNR as a covariate ------------------
# Noise inflates J_f (both shells' noise enters ln f) and inflates J_ln more at
# b=3000 than at b=1500. SNR rises with age, so it is a candidate confound.
jf = pd.read_csv(HCP / "cr_jf_long.csv", usecols=["Session_ID", "ROI", "J_f_mean"])
snr = d[d.shell_bval == 1500][["Session_ID", "ROI", "SNR", "age", "male"]]
piv = d.pivot_table(index=["Session_ID", "ROI"], columns="shell_bval", values="J_ln").reset_index()
piv["delta"] = piv[3000] - piv[1500]
Q = snr.merge(jf, on=["Session_ID", "ROI"]).merge(piv[["Session_ID", "ROI", "delta"]],
                                                     on=["Session_ID", "ROI"])
rows = []
for roi, s in Q.groupby("ROI"):
    s = s.dropna()
    row = dict(ROI=roi, n=len(s))
    for k in ("J_f_mean", "delta"):
        row[f"{k}_r"], row[f"{k}_p"] = partial(s[k].values, s.age.values, [s.male.values])
        row[f"{k}_r_snr"], row[f"{k}_p_snr"] = partial(s[k].values, s.age.values,
                                                       [s.male.values, s.SNR.values])
    rows.append(row)
Bd = pd.DataFrame(rows).set_index("ROI").loc[ROIS]
Bd.to_csv(OUT / "bdep_snr.csv")
say("\nDependence on b, partial r with age (sex) and (sex + SNR):")
say(Bd[["n", "J_f_mean_r", "J_f_mean_r_snr", "delta_r", "delta_r_snr"]].round(3).to_string())

# 3c. Main age associations after removing the simulated noise inflation -----
# Inflation = gap(tensor + noise) - gap(tensor). Subtracting it from the
# measured gap removes what noise alone adds, session by session.
for g in ("J_ln", "J_lnS"):
    for v in ("TN", "TNb0"):
        d[f"{g}_nc_{v}"] = d[g] - (d[f"{g}_{v}"] - d[f"{g}_T"])
rows = []
for (b, roi), s in d.groupby(["shell_bval", "ROI"]):
    row = dict(shell=b, ROI=roi, infl_Jln_pct=100 * ((s.J_ln_TN - s.J_ln_T) / s.J_ln).median())
    for g in ("J_ln", "J_lnS"):
        row[f"{g}_raw"] = partial(s[g].values, s.age.values, [s.male.values])[0]
        row[f"{g}_nc"] = partial(s[f"{g}_nc_TN"].values, s.age.values, [s.male.values])[0]
        row[f"{g}_ncb0"] = partial(s[f"{g}_nc_TNb0"].values, s.age.values, [s.male.values])[0]
    rows.append(row)
NC = pd.DataFrame(rows)
p = d.pivot_table(index=["Session_ID", "ROI"], columns="shell_bval",
                  values=["J_ln", "J_ln_nc_TN"]).dropna()
p.columns = [f"{a}_{b}" for a, b in p.columns]
p = p.reset_index().merge(d[d.shell_bval == 1500][["Session_ID", "ROI", "age", "male"]],
                          on=["Session_ID", "ROI"])
rows = []
for roi, s in p.groupby("ROI"):
    dr = (s.J_ln_3000 - s.J_ln_1500).values
    dn = (s.J_ln_nc_TN_3000 - s.J_ln_nc_TN_1500).values
    rows.append(dict(ROI=roi, dB_raw=partial(dr, s.age.values, [s.male.values])[0],
                     dB_nc=partial(dn, s.age.values, [s.male.values])[0],
                     # change of the cohort mean, the same statistic as the
                     # measured change in C_bdependence.csv (v1_roi_stats.py)
                     dB_nc_shift_pct=100 * (s.J_ln_nc_TN_3000.mean()
                                            / s.J_ln_nc_TN_1500.mean() - 1),
                     dB_shift_pct=100 * (s.J_ln_3000.mean() / s.J_ln_1500.mean() - 1),
                     frac_lower_nc=np.mean(dn < 0)))
NC = NC.merge(pd.DataFrame(rows), on="ROI", how="left")
NC.to_csv(OUT / "noise_corrected.csv", index=False)
say("\nNoise-corrected age associations (DW-residual noise), CR_all:")
for b in (1500, 3000):
    c = NC[(NC.shell == b) & (NC.ROI == "CR_all")].iloc[0]
    q = NC[NC.shell == b]
    say(f"  b={b}: inflation {c.infl_Jln_pct:.1f}%; J_ln {c.J_ln_raw:+.3f} -> {c.J_ln_nc:+.3f} "
        f"(range {q.J_ln_nc.min():+.2f}..{q.J_ln_nc.max():+.2f}); "
        f"J_lnS {c.J_lnS_raw:+.3f} -> {c.J_lnS_nc:+.3f}; "
        f"sign of J_lnS kept in {int((np.sign(q.J_lnS_raw) == np.sign(q.J_lnS_nc)).sum())}/16")
q = NC[NC.shell == 1500]
say(f"  Delta_b: raw r {q.dB_raw.min():+.2f}..{q.dB_raw.max():+.2f}, noise-corrected "
    f"{q.dB_nc.min():+.2f}..{q.dB_nc.max():+.2f} (CR_all {q[q.ROI == 'CR_all'].dB_nc.item():+.3f}); "
    f"corrected shift {q.dB_nc_shift_pct.min():.0f}% to {q.dB_nc_shift_pct.max():.0f}%, "
    f"lower at b=3000 in {q.frac_lower_nc.min():.1%}-{q.frac_lower_nc.max():.1%} of participants")

# 4. Summary numbers quoted in the text -------------------------------------
say("\nSummary numbers")
say(f"  ADC_1 / tensor MD, median over sessions, regions and shells: "
    f"{(d.ADC_1 / d.MD).median():.3f}")
for b in (1500, 3000):
    s = d[(d.ROI == "CR_all") & (d.shell_bval == b)]
    say(f"  noise inflation, corona radiata b={b}: J_ln "
        f"{100 * ((s.J_ln_TN - s.J_ln_T) / s.J_ln).median():.1f}%, J_lnS "
        f"{100 * ((s.J_lnS_TN - s.J_lnS_T) / s.J_lnS).median():.1f}%")
tn = X[X.variant == "TN"]
say(f"  tensor+noise prediction exceeds the measured gap in "
    f"{int((tn.excess_pct < 0).sum())} of {len(tn)} region x gap x shell cells")
for v in ("TN", "TNb0"):
    q = X[(X.variant == v) & (X.gap == "J_ln") & (X.shell == 1500)]
    say(f"  {v}: J_ln b=1500 residual age r {q.r.min():+.2f} to {q.r.max():+.2f}, "
        f"median excess {q.excess_pct.median():+.1f}%")
say(f"  uncorrected (tensor only) J_ln b=1500 median excess "
    f"{X[(X.variant == 'T') & (X.gap == 'J_ln') & (X.shell == 1500)].excess_pct.median():+.1f}%")

(OUT / "post_native_summary.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
