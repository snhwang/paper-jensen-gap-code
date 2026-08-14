"""Significance (with multiple-comparison context) for every statistic the
jensen-gap paper cites in the text, so the manuscript can state them correctly.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

import os
BASE = Path(os.environ.get("JG_DATA", Path(__file__).resolve().parent))
g = pd.read_csv(BASE / "HCP" / "cr_gaps_long.csv")
mans = {b: pd.read_csv(BASE / "HCP" / f"manifest_n1379_b{b}.tsv", sep="\t") for b in (1500, 3000)}
for b in mans:
    mans[b]["age"] = pd.to_numeric(mans[b]["age"], errors="coerce")

M_FAMILY = 320  # gap-spectrum heatmap cells (Bonferroni family)


def corr(bval, roi, col_hi, col_lo=None):
    """Partial correlation of the metric vs age, controlling for sex."""
    gb = g[g.shell_bval == bval]
    j = gb.merge(mans[bval][["session_id", "age", "sex"]], left_on="Session_ID",
                 right_on="session_id", how="inner")
    s = j[j.ROI == roi]
    hi = pd.to_numeric(s[f"{col_hi}_mean"], errors="coerce")
    if col_lo is None:
        val = hi
    else:
        lo = pd.to_numeric(s[f"{col_lo}_mean"], errors="coerce").clip(lower=1e-12)
        val = np.log(hi.clip(lower=1e-12) / lo)
    df = pd.DataFrame({"v": val, "a": s["age"], "s": s["sex"]}).dropna()
    sd = pd.get_dummies(df["s"], drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(df)), sd.values])
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    r, _ = stats.pearsonr(res(df["v"].values), res(df["a"].values))
    dofp = max(len(df) - X.shape[1] - 1, 1)
    t = r * np.sqrt(dofp / max(1 - r * r, 1e-300))
    return r, 2 * stats.t.sf(abs(t), dofp), len(df)


def show(label, bval, roi, hi, lo=None):
    r, p, n = corr(bval, roi, hi, lo)
    bonf = p < 0.05 / M_FAMILY
    print(f"  {label:34s} r={r:+.3f}  p={p:.2e}  n={n}  Bonferroni(320)={'PASS' if bonf else 'FAIL'}")


print("=== Cross-sectional (section 7.1) cited correlations ===")
show("CR_all b1500 J_ln (G0,1)", 1500, "CR_all", "ADC_1", "ADC_0")
show("CR_all b1500 J_ln^S (G0,1)", 1500, "CR_all", "M1_S", "M0_S")
show("SCR    b1500 J_ln", 1500, "SCR", "ADC_1", "ADC_0")
show("SCR    b1500 J_ln^S", 1500, "SCR", "M1_S", "M0_S")
show("CR_all b3000 J_ln", 3000, "CR_all", "ADC_1", "ADC_0")
show("CR_all b3000 J_ln^S", 3000, "CR_all", "M1_S", "M0_S")
show("CR_all b1500 ADC_1 (trace)", 1500, "CR_all", "ADC_1")
print()
print("=== Spectrum extremes (section 7.2) ===")
show("CR_all b1500 G_min,max (diff)", 1500, "CR_all", "ADC_max", "ADC_min")
show("CR_all b1500 G^S_min,max (sig)", 1500, "CR_all", "M_max_S", "M_min_S")

print()
print("=== b-signature (panel a): are all 16 ROIs below y=x (J_ln falls b1500->b3000)? ===")
ROIS = ["CR_all", "ACR", "SCR", "PCR", "Body", "Genu", "Splenium",
        "alic", "plic", "rlic", "ec", "slf", "ptr", "cingulum_cc", "ss", "tapetum"]
def roi_mean_jln(bval, roi):
    gb = g[g.shell_bval == bval]
    j = gb.merge(mans[bval][["session_id"]], left_on="Session_ID", right_on="session_id", how="inner")
    s = j[j.ROI == roi]
    return pd.to_numeric(s["J_ln_D_mean"], errors="coerce").mean()
below = 0
drops = []
for roi in ROIS:
    a, b = roi_mean_jln(1500, roi), roi_mean_jln(3000, roi)
    if np.isfinite(a) and np.isfinite(b):
        drops.append((b - a) / a * 100)
        if b < a:
            below += 1
n_roi = len(drops)
sign_p = 2 * stats.binom.sf(below - 1, n_roi, 0.5) if below == n_roi else None
print(f"  {below}/{n_roi} ROIs have J_ln(b3000) < J_ln(b1500); mean change = {np.mean(drops):+.1f}%")
print(f"  sign test (all {n_roi} same direction): p = {2*0.5**n_roi:.2e}")
