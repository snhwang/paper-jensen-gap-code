"""Does J_f survive once it compares identical directions and noise is removed?

Reads out/jf_pass.csv (or the partial file) and reports, by region:
  - the submitted J_f against its noise-plus-mismatch null
  - the corrected J_f (identical directions) against its noise null
  - the sex-adjusted age correlation of each, and of corrected minus null

Usage:  python jf_stats.py [--partial]
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from config import OUT

src = OUT / ("jf_pass_partial.csv" if "--partial" in sys.argv else "jf_pass.csv")
d = pd.read_csv(src).merge(pd.read_csv(OUT / "v1_cohort_sessions.csv"), on="Session_ID")
d["male"] = (d.sex == "M").astype(float)


def pr(y, a, m):
    X = np.column_stack([np.ones(len(y)), m])
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    r = np.corrcoef(ry, ra)[0, 1]
    t = r * np.sqrt((len(y) - 3) / (1 - r * r))
    return r, 2 * stats.t.sf(abs(t), len(y) - 3)


rows = []
for roi, s in d.groupby("ROI"):
    a, m = s.age.values, s.male.values
    ex_new = (s.Jf_new - s.Jf_new_null).values
    ex_old = (s.Jf_old - s.Jf_old_null).values
    rows.append(dict(
        ROI=roi, n=len(s),
        old=s.Jf_old.mean(), old_null=s.Jf_old_null.mean(),
        old_mismatch_pct=100 * (s.Jf_old_mismatch / s.Jf_old).median(),
        new=s.Jf_new.mean(), new_null=s.Jf_new_null.mean(),
        new_excess_pct=100 * np.median(ex_new / s.Jf_new),
        frac_new_above_null=np.mean(ex_new > 0),
        r_old=pr(s.Jf_old.values, a, m)[0], r_new=pr(s.Jf_new.values, a, m)[0],
        r_new_excess=pr(ex_new, a, m)[0], p_new_excess=pr(ex_new, a, m)[1],
        r_old_excess=pr(ex_old, a, m)[0]))
R = pd.DataFrame(rows).set_index("ROI")
R.to_csv(OUT / "jf_stats.csv")
pd.set_option("display.width", 250)
print(f"{src.name}: {d.Session_ID.nunique()} sessions")
print(R.round(4).to_string())
above = R[R.new > R.new_null]
lines = [f"{src.name}: {d.Session_ID.nunique()} sessions",
         f"paired directions: J_f below its noise-plus-mismatch null in "
         f"{(R.old < R.old_null).sum()}/16 regions; mismatch share "
         f"{R.old_mismatch_pct.min():.0f}% to {R.old_mismatch_pct.max():.0f}%",
         f"identical directions: J_f above its noise null in {len(above)}/16 regions "
         f"({', '.join(above.index)}), by at most "
         f"{100 * (above.new / above.new_null - 1).max():.0f}% (cohort means)"]

# Expected size of J_f from the measured shell difference of J_ln. If
# ln f_i = c - beta ln(D_i/<D>), then J_ln(b2) ~ (1-beta)^2 J_ln(b1) and
# J_f ~ beta^2 Var(ln D)/2 ~ beta^2 J_ln(b1).
C = pd.read_csv(OUT / "C_bdependence.csv").set_index("ROI").loc["CR_all"]
beta = 1 - np.sqrt(C.Jln3000 / C.Jln1500)
lines.append(f"corona radiata: beta {beta:.3f}; expected J_f from Delta_b J_ln about "
             f"{beta ** 2 * C.Jln1500:.1e}; corrected J_f noise level "
             f"{R.loc['CR_all', 'new_null']:.1e}")
print("\n" + "\n".join(lines))
(OUT / "jf_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
