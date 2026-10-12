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

Inputs:  out/long_sessions.csv, out/native_pass.csv and out/native_pass_long.csv
Outputs: out/longitudinal.csv, out/longitudinal.txt
Usage:   python longitudinal.py
"""
import glob

import pandas as pd
import statsmodels.formula.api as smf

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
lines, rows = [], []
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
pd.DataFrame(rows).to_csv(OUT / "longitudinal.csv", index=False)
(OUT / "longitudinal.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
