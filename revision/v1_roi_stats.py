"""Region-level statistics for the revision, on the first-visit (V1) cohort.

The submitted analysis chose, per participant and per shell, the first visit
whose processing cache was complete on 24 May 2026. That mixed visits (301
participants used a later visit at b=1500) and could pair different sessions
across shells. Here every participant contributes one session, their first
visit, to both shells.

Everything below reads the cached per-session ROI tables only. No image data are
touched. Outputs are aggregate statistics, written to OUT.

Usage:  python v1_roi_stats.py
"""
import glob

import numpy as np
import pandas as pd
from scipy import stats

from config import DTI, HCP, OUT, ROIS

SHELLS = (1500, 3000)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def clean_age(s):
    return pd.to_numeric(s.astype(str).str.replace("90 or older", "90", regex=False),
                         errors="coerce")


def partial_r(y, age, sex):
    """Pearson r of y with age after both are residualized on [1, sex]."""
    X = np.column_stack([np.ones_like(age), (sex == "M").astype(float)])
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    ra = age - X @ np.linalg.lstsq(X, age, rcond=None)[0]
    r = np.corrcoef(ry, ra)[0, 1]
    n = len(y)
    t = r * np.sqrt((n - 3) / (1 - r * r))
    return r, 2 * stats.t.sf(abs(t), n - 3), n


def age_slope(y, age, sex):
    X = np.column_stack([np.ones_like(age), age, (sex == "M").astype(float)])
    return np.linalg.lstsq(X, y, rcond=None)[0][1]


# --------------------------------------------------------------------------
# Demographics
# --------------------------------------------------------------------------
dem = pd.read_csv(glob.glob(str(HCP / "AABC2_subjects_*.csv"))[0], low_memory=False)
dem = dem.dropna(subset=["id_event"]).copy()
dem["Subject_ID"] = dem.id_event.str.rsplit("_", n=1).str[0]
dem["Visit"] = dem.id_event.str.rsplit("_", n=1).str[1]
dem["age"] = clean_age(dem.age_open)
dem["top90"] = dem.age_open.astype(str).str.contains("90 or older")
say(f"release: {len(dem)} visits, {dem.Subject_ID.nunique()} participants, "
    f"{dem[dem.dMRI.astype(str).str.upper() == 'YES'].Subject_ID.nunique()} with "
    f"diffusion MRI at one or more visits")
dem = dem[["Subject_ID", "Visit", "age", "top90", "sex", "site", "scanner", "study"]]

# --------------------------------------------------------------------------
# Gap table, one session per subject-visit (latest session if duplicated)
# --------------------------------------------------------------------------
g = pd.read_csv(HCP / "cr_gaps_long.csv", low_memory=False)
num = [c for c in g.columns if c.endswith("_mean") or c.endswith("_sd")]
g[num] = g[num].apply(pd.to_numeric, errors="coerce")
sess = (g[["Subject_ID", "Visit", "Session_ID"]].drop_duplicates()
        .sort_values("Session_ID").groupby(["Subject_ID", "Visit"]).tail(1))
g = g.merge(sess, on=["Subject_ID", "Visit", "Session_ID"])
say(f"gap table: {g.Session_ID.nunique()} sessions after de-duplication "
    f"({g.Subject_ID.nunique()} subjects)")

# --------------------------------------------------------------------------
# V1 cohort: complete on both shells and all 16 ROIs
# --------------------------------------------------------------------------
v1 = g[g.Visit == "V1"]
ok = v1.dropna(subset=["J_ln_D_mean", "J_ln_S_mean", "ADC_1_mean"])
cnt = ok.groupby("Session_ID").size()
complete = cnt[cnt == 2 * len(ROIS)].index
v1 = v1[v1.Session_ID.isin(complete)].merge(dem, on=["Subject_ID", "Visit"], how="left")
v1 = v1.dropna(subset=["age", "sex"])
say(f"V1 cohort, complete: {v1.Subject_ID.nunique()} participants")


# Acquisition check from each session's b-values
def shell_counts(sid):
    f = DTI / sid / "inputs" / "dwi.bval"
    if not f.exists():
        return (np.nan, np.nan, np.nan)
    b = np.loadtxt(f).ravel()
    return (int((b < 50).sum()), int((abs(b - 1500) < 100).sum()),
            int((abs(b - 3000) < 100).sum()))


acq = pd.DataFrame([(s, *shell_counts(s)) for s in v1.Session_ID.unique()],
                   columns=["Session_ID", "n_b0", "n_b1500", "n_b3000"])
acq["standard"] = (acq.n_b0 == 14) & (acq.n_b1500 == 93) & (acq.n_b3000 == 92)
say("acquisitions (b0/b1500/b3000) among V1 sessions:")
say(acq.groupby(["n_b0", "n_b1500", "n_b3000"]).size().to_string())
v1 = v1.merge(acq[["Session_ID", "standard"]], on="Session_ID")
v1 = v1[v1.standard]
N = v1.Subject_ID.nunique()
say(f"V1 cohort, standard acquisition: N = {N}")
p = v1.drop_duplicates("Subject_ID")
say(f"  age {p.age.min():.0f}-{p.age.max():.0f}, mean {p.age.mean():.1f} "
    f"(SD {p.age.std():.1f}); top-coded 90: {int(p.top90.sum())}; "
    f"female {int((p.sex == 'F').sum())}, male {int((p.sex == 'M').sum())}")
say("  study era of V1 sessions: " + p.study.value_counts().to_dict().__repr__())
say("  sites: " + p.site.value_counts().to_dict().__repr__())
p[["Subject_ID", "Session_ID", "age", "sex", "site"]].to_csv(OUT / "v1_cohort_sessions.csv",
                                                              index=False)

# --------------------------------------------------------------------------
# A. Cross-sectional age associations
# --------------------------------------------------------------------------
say("\nA. Sex-adjusted partial r with age, V1 cohort")
rows = []
for b in SHELLS:
    for roi in ROIS:
        s = v1[(v1.shell_bval == b) & (v1.ROI == roi)]
        age, sex = s.age.values, s.sex.values
        mets = {
            "J_ln": s.J_ln_D_mean.values,
            "J_lnS": s.J_ln_S_mean.values,
            "ADC_1": s.ADC_1_mean.values,
            "ADC_0": s.ADC_0_mean.values,
            "G01_logratio": np.log(s.ADC_1_mean / s.ADC_0_mean).values,
        }
        for k, y in mets.items():
            r, pv, n = partial_r(y, age, sex)
            rows.append(dict(shell=b, ROI=roi, metric=k, r=r, p=pv, n=n))
A = pd.DataFrame(rows)
A.to_csv(OUT / "A_cross_sectional.csv", index=False)
piv = A.pivot_table(index=["ROI"], columns=["shell", "metric"], values="r").loc[ROIS]
say(piv.round(3).to_string())
ntest = len(A)
say(f"Bonferroni over {ntest} tests: max p among |r| reported = "
    f"{A.p.max():.2e}; n significant at 0.05/{ntest}: {(A.p < 0.05 / ntest).sum()}")

# --------------------------------------------------------------------------
# B. Decomposition  ln J_lnS  ~  2 ln(b ADC_1) + ln J_ln
# --------------------------------------------------------------------------
say("\nB. Tensor-model decomposition, per region")
rows = []
for b in SHELLS:
    for roi in ROIS:
        s = v1[(v1.shell_bval == b) & (v1.ROI == roi)]
        age, sex = s.age.values, s.sex.values
        lS = np.log(s.J_ln_S_mean.values)
        lA = np.log(b * s.ADC_1_mean.values)
        lJ = np.log(s.J_ln_D_mean.values)
        pred = 2 * lA + lJ
        rows.append(dict(
            shell=b, ROI=roi,
            ratio_median=np.median(np.exp(lS - pred)),
            r_subjects=np.corrcoef(lS, pred)[0, 1],
            slope_lnJS=age_slope(lS, age, sex),
            slope_2lnADC1=2 * age_slope(lA, age, sex),
            slope_lnJ=age_slope(lJ, age, sex),
        ))
B = pd.DataFrame(rows)
B["slope_pred"] = B.slope_2lnADC1 + B.slope_lnJ
B.to_csv(OUT / "B_decomposition.csv", index=False)
say((B.set_index(["shell", "ROI"]) * [1, 1, 100, 100, 100, 100]).round(3).to_string())
say("(slopes in % per year of each log quantity)")
for b in SHELLS:
    q = B[B.shell == b]
    say(f"b={b}: sign of predicted slope matches observed in "
        f"{int((np.sign(q.slope_pred) == np.sign(q.slope_lnJS)).sum())}/16 regions; "
        f"r(pred, observed) across regions = {np.corrcoef(q.slope_pred, q.slope_lnJS)[0, 1]:.3f}")

# --------------------------------------------------------------------------
# C. Dependence on b (same session at both shells by construction)
# --------------------------------------------------------------------------
say("\nC. Dependence on b")
rows = []
for roi in ROIS:
    s = v1[v1.ROI == roi].pivot_table(index=["Subject_ID", "age", "sex"],
                                      columns="shell_bval",
                                      values="J_ln_D_mean").dropna().reset_index()
    d = s[3000] - s[1500]
    r, pv, n = partial_r(d.values, s.age.values, s.sex.values)
    rows.append(dict(ROI=roi, Jln1500=s[1500].mean(), Jln3000=s[3000].mean(),
                     shift_pct=100 * (s[3000].mean() - s[1500].mean()) / s[1500].mean(),
                     frac_subjects_neg=(d < 0).mean(), r_age=r, p=pv, n=n))
C = pd.DataFrame(rows)
say(C.round(4).to_string())
say(f"regions with mean J_ln(3000) < J_ln(1500): {(C.Jln3000 < C.Jln1500).sum()}/16, "
    f"sign test p = {stats.binomtest(int((C.Jln3000 < C.Jln1500).sum()), 16).pvalue:.1e}")

jf = pd.read_csv(HCP / "cr_jf_long.csv", low_memory=False)
jf = jf.merge(v1[["Session_ID", "Subject_ID", "age", "sex"]].drop_duplicates(),
              on=["Session_ID", "Subject_ID"])
rows = []
for roi in ROIS:
    s = jf[jf.ROI == roi].dropna(subset=["J_f_mean"])
    r, pv, n = partial_r(s.J_f_mean.values, s.age.values, s.sex.values)
    rows.append(dict(ROI=roi, r_age=r, p=pv, n=n, n_dirs_median=s.n_dirs.median()))
Cf = pd.DataFrame(rows)
say("J_f vs age:")
say(Cf.round(4).to_string())
C.merge(Cf, on="ROI", suffixes=("_delta", "_jf")).to_csv(OUT / "C_bdependence.csv", index=False)

# --------------------------------------------------------------------------
# D. Longitudinal change: see longitudinal.py
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# E. Gap spectrum, voxelwise-gap aggregation (Supplementary S2)
# --------------------------------------------------------------------------
say("\nE. Rung-pair gaps, ROI mean of voxelwise gaps")
spec = {
    "G_-1,0": lambda s: s.J_harm_D_mean - s.J_ln_D_mean,
    "G_0,1": lambda s: s.J_ln_D_mean,
    "G_1,2": lambda s: s.J_quad_D_mean,
    "G_-1,2": lambda s: s.J_quad_D_mean + s.J_harm_D_mean,
    "GS_0,1": lambda s: s.J_ln_S_mean,
    "GS_1,2": lambda s: s.J_quad_S_mean,
}
rows = []
for b in SHELLS:
    for roi in ROIS:
        s = v1[(v1.shell_bval == b) & (v1.ROI == roi)]
        for k, f in spec.items():
            r, pv, n = partial_r(f(s).values, s.age.values, s.sex.values)
            rows.append(dict(shell=b, ROI=roi, gap=k, r=r, p=pv))
E = pd.DataFrame(rows)
E.to_csv(OUT / "E_spectrum_voxelwise.csv", index=False)
say(E.pivot_table(index="ROI", columns=["shell", "gap"], values="r").loc[ROIS].round(3).to_string())

# --------------------------------------------------------------------------
# F. Sensitivity analyses: see sensitivity.py
# --------------------------------------------------------------------------

(OUT / "summary.txt").write_text("\n".join(LOG), encoding="utf-8")
say(f"\nwrote {OUT}")
