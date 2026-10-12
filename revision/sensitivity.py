"""Supplementary Table S2: sensitivity of the corona radiata age correlations.

Rows (sex-adjusted partial r of each gap with age, corona radiata, both shells):
  1. main analysis (first-visit cohort, N = 1,354)
  2. the 42 participants recorded as "90 or older" given ages drawn from an
     exponential tail above 90 (mean +2.5 y), averaged over 200 draws
  3. site, mean framewise head motion and eddy outlier fraction as covariates
  4. the 16 sessions with a non-standard acquisition included (N = 1,370)
  5. the noise inflation of each gap (native_pass.py) subtracted per session

Inputs:  config.HCP tables (cr_gaps_long.csv, AABC2 demographics),
         out/v1_cohort_sessions.csv (v1_roi_stats.py), out/motion.csv
         (motion_covariates.py), out/native_pass.csv
Outputs: tables_supp/tab_sens.tex, out/sensitivity.csv
Usage:   python sensitivity.py
"""
import glob

import numpy as np
import pandas as pd

from config import HCP, OUT, TABLES


def clean_age(s):
    return pd.to_numeric(s.astype(str).str.replace("90 or older", "90", regex=False),
                         errors="coerce")


def partial_r(y, age, covs):
    X = np.column_stack([np.ones(len(y))] + covs)
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    ra = age - X @ np.linalg.lstsq(X, age, rcond=None)[0]
    return np.corrcoef(ry, ra)[0, 1]


dem = pd.read_csv(glob.glob(str(HCP / "AABC2_subjects_*.csv"))[0], low_memory=False)
dem = dem.dropna(subset=["id_event"]).copy()
dem["Subject_ID"] = dem.id_event.str.rsplit("_", n=1).str[0]
dem["Visit"] = dem.id_event.str.rsplit("_", n=1).str[1]
dem["age"] = clean_age(dem.age_open)
dem["top90"] = dem.age_open.astype(str).str.contains("90 or older")

g = pd.read_csv(HCP / "cr_gaps_long.csv", low_memory=False,
                usecols=["Subject_ID", "Visit", "Session_ID", "ROI", "shell_bval",
                         "J_ln_D_mean", "J_ln_S_mean"])
g = g[g.ROI == "CR_all"]
sess = (g[["Subject_ID", "Visit", "Session_ID"]].drop_duplicates()
        .sort_values("Session_ID").groupby(["Subject_ID", "Visit"]).tail(1))
g = g.merge(sess).query("Visit == 'V1'").merge(
    dem[["Subject_ID", "Visit", "age", "top90", "sex", "site"]], on=["Subject_ID", "Visit"])
g = g.dropna(subset=["age", "sex", "J_ln_D_mean", "J_ln_S_mean"])
cohort = set(pd.read_csv(OUT / "v1_cohort_sessions.csv").Session_ID)
mot = pd.read_csv(OUT / "motion.csv")   # the same for both shells
nat = pd.read_csv(OUT / "native_pass.csv")
nat = nat[nat.ROI == "CR_all"].copy()
for gap in ("J_ln", "J_lnS"):
    nat[f"{gap}_nc"] = nat[gap] - (nat[f"{gap}_TN"] - nat[f"{gap}_T"])

rows = {k: {} for k in ("main", "top", "cov", "all", "noise")}
n = {}
for b in (1500, 3000):
    s_all = g[g.shell_bval == b]
    s = s_all[s_all.Session_ID.isin(cohort)]
    male = (s.sex == "M").astype(float).values
    for gap, col in (("J_ln", "J_ln_D_mean"), ("J_lnS", "J_ln_S_mean")):
        y = s[col].values
        rows["main"][(b, gap)] = partial_r(y, s.age.values, [male])
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(200):
            a = s.age.values.copy()
            top = s.top90.values.astype(bool)
            a[top] = 90 + rng.exponential(2.5, top.sum())
            draws.append(partial_r(y, a, [male]))
        rows["top"][(b, gap)] = np.mean(draws)
        q = s.merge(mot, on="Subject_ID").dropna(
            subset=["motion_rel", "outlier_frac", "site"])
        Z = pd.get_dummies(q[["sex", "site"]], drop_first=True).astype(float)
        rows["cov"][(b, gap)] = partial_r(q[col].values, q.age.values,
                                          [Z.values, q.motion_rel.values,
                                           q.outlier_frac.values])
        rows["all"][(b, gap)] = partial_r(s_all[col].values, s_all.age.values,
                                          [(s_all.sex == "M").astype(float).values])
        v = nat[nat.shell_bval == b].merge(s[["Session_ID", "age", "sex"]], on="Session_ID")
        rows["noise"][(b, gap)] = partial_r(v[f"{gap}_nc"].values, v.age.values,
                                            [(v.sex == "M").astype(float).values])
    n.update(main=len(s), top=len(s), cov=len(q), all=len(s_all), noise=len(v))

labels = {"main": "Main analysis", "top": "Top-coded ages drawn above 90",
          "cov": "Site, motion and outliers as covariates",
          "all": "Non-standard acquisitions included",
          "noise": "Simulated noise inflation subtracted"}


def f(r):
    return f"${r:+.3f}$".replace("$-", "$-")


lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{2}{c}{$b=1500\,\mathrm{s/mm^2}$} & "
         r"\multicolumn{2}{c}{$b=3000\,\mathrm{s/mm^2}$}\\",
         r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
         r"Analysis & $N$ & $J_{\ln}$ & $J_{\ln}^{S}$ & $J_{\ln}$ & $J_{\ln}^{S}$\\", r"\midrule"]
out = []
for k, lab in labels.items():
    r = rows[k]
    lines.append(f"{lab} & {n[k]:,} & {f(r[(1500, 'J_ln')])} & {f(r[(1500, 'J_lnS')])} & "
                 f"{f(r[(3000, 'J_ln')])} & {f(r[(3000, 'J_lnS')])}" + r"\\")
    out.append(dict(analysis=k, N=n[k], **{f"{gap}_{b}": v for (b, gap), v in r.items()}))
lines += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "tab_sens.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
pd.DataFrame(out).to_csv(OUT / "sensitivity.csv", index=False)
print(pd.DataFrame(out).round(3).to_string(index=False))
main = rows["main"]
lines = [f"{k}: N={n[k]}, " + ", ".join(f"{gap} b{b} {v:+.3f}" for (b, gap), v in r.items())
         for k, r in rows.items()]
lines.append("largest change from main: " + ", ".join(
    f"{k} {max(abs(r[key] - main[key]) for key in main):.3f}" for k, r in rows.items()
    if k != "main"))
m15 = g[(g.shell_bval == 1500) & g.Session_ID.isin(cohort)].merge(
    mot, on="Subject_ID").dropna(subset=["motion_rel"])
lines.append(f"head motion vs age (b=1500 sessions): r = "
             f"{np.corrcoef(m15.age, m15.motion_rel)[0, 1]:+.3f}, n = {len(m15)}")
print("\n".join(lines[-2:]))
(OUT / "sensitivity.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
