"""Within-person change of the two gaps in the corona radiata (Section 4.5).

Participants with two or more visits (long_sessions.py), standard acquisitions
only. For each gap and shell, a linear mixed-effects model with a random
intercept per participant,
    gap ~ age_at_first_visit + time_since_first_visit + sex,
gives the within-person rate as the coefficient of time (Hoffman and Stawski
2009). Time comes from the exact number of days since the first visit
(days_from_V1 in the demographics), because the released ages are whole years
and top-coded at 90. Rates are in percent of the mean per year. The model is fitted to the
measured gap and to the gap with its noise inflation removed (native_pass.py:
gap - (gap of tensor + noise - gap of tensor)).

Whether the rate depends on age: a second model adds the interaction of time with
age at the first visit (centred at 60, in decades),
    gap ~ a60 + time + time:a60 + sex,
which gives the within-person rate and its 95% CI at each first-visit age from 40 to
85 (the text reports 45, 60 and 75), and the model's mean level at each age (for the
segments of Figure 7c, d). The
gap is first expressed in percent from its mean, so coefficients are in %/y.
A random-slope version of this model does not converge in statsmodels, so as a check
that lets rates differ between people, each participant's own least-squares slope is
regressed on a60 and sex, weighted by squared follow-up, with HC3 robust errors.

Inputs:  out/long_sessions.csv, out/native_pass.csv and out/native_pass_long.csv
Outputs: out/longitudinal.csv, out/longitudinal_by_age.csv, out/longitudinal.txt
Usage:   python longitudinal.py
"""
import glob

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

AGES = (45, 60, 75)                    # reported in the text and drawn as segments
CURVE = tuple(range(40, 86))           # the rate as a curve over first-visit age (Figure 7d inset)

from config import HCP, OUT

sess = pd.read_csv(OUT / "long_sessions.csv")
dem = pd.read_csv(glob.glob(str(HCP / "AABC2_subjects_*.csv"))[0], low_memory=False,
                  usecols=["id_event", "scanner"]).dropna(subset=["id_event"])
dem["Subject_ID"] = dem.id_event.str.rsplit("_", n=1).str[0]
dem["Visit"] = dem.id_event.str.rsplit("_", n=1).str[1]
sess = sess.merge(dem[["Subject_ID", "Visit", "scanner"]], on=["Subject_ID", "Visit"],
                  how="left")
nat = pd.concat([pd.read_csv(OUT / "native_pass.csv"),
                 pd.read_csv(OUT / "native_pass_long.csv")], ignore_index=True)
nat = nat[nat.ROI == "CR_all"].drop_duplicates(["Session_ID", "shell_bval"])
for gap in ("J_ln", "J_lnS"):
    nat[f"{gap}_nc"] = nat[gap] - (nat[f"{gap}_TN"] - nat[f"{gap}_T"])

d = sess[sess.standard].merge(nat, on="Session_ID")
lines, rows, by_age = [], [], []


def at_ages(params, cov, rate, inter):
    """Rate and 95% CI at each first-visit age, from a rate and its interaction with a60."""
    out = []
    V = cov.loc[[rate, inter], [rate, inter]].values
    for a in CURVE:
        w = np.array([1.0, (a - 60) / 10])
        r, se = w @ params[[rate, inter]].values, np.sqrt(w @ V @ w)
        out.append((a, r, r - 1.96 * se, r + 1.96 * se))
    return out


def by_age_rows(b, gap, kind, method):
    return [(r["age"], r["rate"], r["ci_lo"], r["ci_hi"]) for r in by_age
            if (r["shell"], r["gap"], r["kind"], r["method"]) == (b, gap, kind, method)
            and r["age"] in AGES]

for b in (1500, 3000):
    s = d[d.shell_bval == b].copy()
    s = s[s.groupby("Subject_ID").Visit.transform("nunique") >= 2]
    # Time from the exact interval since the first included visit, not from the
    # released ages, which are whole years and top-coded at 90.
    first = s.groupby("Subject_ID").days_from_V1.transform("min")
    s["t"] = (s.days_from_V1 - first) / 365.25
    s["age_bl"] = s.loc[s.days_from_V1 == first].set_index("Subject_ID").age.reindex(
        s.Subject_ID).values
    changed = s.groupby("Subject_ID").scanner.nunique() > 1
    lines.append(f"b={b}: {s.Subject_ID.nunique()} participants, {len(s)} sessions, "
                 f"median follow-up {s.groupby('Subject_ID').t.max().median():.1f} y, "
                 f"scanner changed between visits for {100 * changed.mean():.0f}%")
    # Visits per participant, and age at the last included visit (released ages:
    # whole years, "90 or older" coded as 90).
    visits = s.groupby("Subject_ID").Visit.nunique()
    last_age = s.loc[s.groupby("Subject_ID").days_from_V1.idxmax()].age
    lines.append(f"  visits per participant: " + ", ".join(
        f"{k} visits {n}" for k, n in visits.value_counts().sort_index().items())
        + f"; 3 or more {int((visits >= 3).sum())}. Age at last visit 75 or older "
        f"{int((last_age >= 75).sum())}, 85 or older {int((last_age >= 85).sum())}")
    for gap in ("J_ln", "J_lnS"):
        for kind, col in (("measured", gap), ("noise removed", f"{gap}_nc")):
            mu = s[col].mean()
            m = smf.mixedlm(f"{col} ~ age_bl + t + C(sex)", s,
                            groups=s.Subject_ID).fit(method="lbfgs")
            ci = 100 * m.conf_int().loc["t"].values / mu
            rate = 100 * m.params["t"] / mu
            rows.append(dict(shell=b, gap=gap, kind=kind, rate=rate, ci_lo=ci[0],
                             ci_hi=ci[1], p=m.pvalues["t"],
                             n_participants=s.Subject_ID.nunique(), n_sessions=len(s)))
            lines.append(f"  {gap:5s} {kind:13s}: {rate:+.3f} %/y "
                         f"[{ci[0]:+.3f}, {ci[1]:+.3f}], p={m.pvalues['t']:.1e}")

            # The rate as a function of age at the first visit.
            s["y"] = 100 * (s[col] / mu - 1)
            s["a60"] = (s.age_bl - 60) / 10
            s["male"] = (s.sex == "M").astype(float)
            m = smf.mixedlm("y ~ a60 + t + t:a60 + male", s, groups=s.Subject_ID).fit(method="lbfgs")
            for a, r, lo, hi in at_ages(m.params, m.cov_params(), "t", "t:a60"):
                level = m.params["Intercept"] + m.params["a60"] * (a - 60) / 10 + m.params["male"] * s.male.mean()
                by_age.append(dict(shell=b, gap=gap, kind=kind, method="mixed", age=a, rate=r, ci_lo=lo,
                                   ci_hi=hi, p_age=m.pvalues["t:a60"], level=level))
            lines.append(f"    by first-visit age (mixed): " + ", ".join(
                f"{a}: {r:+.2f} [{lo:+.2f}, {hi:+.2f}]" for a, r, lo, hi in by_age_rows(b, gap, kind, "mixed"))
                + f"; interaction p={m.pvalues['t:a60']:.1e}")

            # Check: each participant's own slope, regressed on age at the first visit.
            per = s.groupby("Subject_ID").apply(lambda q: pd.Series(dict(
                slope=np.polyfit(q.t, q.y, 1)[0], a60=q.a60.iloc[0], male=q.male.iloc[0], fu=q.t.max())),
                include_groups=False)
            X = sm.add_constant(per[["a60"]].assign(male=per.male - per.male.mean()))
            w = sm.WLS(per.slope, X, weights=per.fu ** 2).fit(cov_type="HC3")
            for a, r, lo, hi in at_ages(w.params, w.cov_params(), "const", "a60"):
                by_age.append(dict(shell=b, gap=gap, kind=kind, method="per-person", age=a, rate=r,
                                   ci_lo=lo, ci_hi=hi, p_age=w.pvalues["a60"], level=np.nan))
            lines.append(f"    by first-visit age (per-person slopes): " + ", ".join(
                f"{a}: {r:+.2f} [{lo:+.2f}, {hi:+.2f}]" for a, r, lo, hi in by_age_rows(b, gap, kind, "per-person"))
                + f"; age p={w.pvalues['a60']:.1e}")
pd.DataFrame(rows).to_csv(OUT / "longitudinal.csv", index=False)
pd.DataFrame(by_age).to_csv(OUT / "longitudinal_by_age.csv", index=False)
(OUT / "longitudinal.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
