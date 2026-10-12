"""LaTeX tables for the Supplementary Material, written from the analysis outputs.

  tab_age.tex       age correlations of the gaps and tensor metrics (Figure 4)
  tab_bdep.tex      dependence on b, by region, with and without noise
  tab_jf.tex        J_f with paired and with identical directions
  tab_beyond.tex    age effects not explained by the tensor and noise

Usage:  python supp_tables.py
"""

import numpy as np
import pandas as pd

from config import OUT
from config import TABLES as TAB

NAMES = {
    "CR_all": "Corona radiata (all)", "ACR": "Anterior corona radiata",
    "SCR": "Superior corona radiata", "PCR": "Posterior corona radiata",
    "Genu": "Genu of corpus callosum", "Body": "Body of corpus callosum",
    "Splenium": "Splenium of corpus callosum", "alic": "Internal capsule, anterior",
    "plic": "Internal capsule, posterior", "rlic": "Internal capsule, retrolenticular",
    "ec": "External capsule", "slf": "Sup.\\ longitudinal fasciculus",
    "ptr": "Posterior thalamic radiation", "ss": "Sagittal stratum",
    "cingulum_cc": "Cingulum (cingulate gyrus)", "tapetum": "Tapetum",
}
ROIS = list(NAMES)


def fmt(r, sig=True):
    s = f"{r:+.2f}".replace("-", "$-$")
    return s if sig else f"\\textit{{{s}}}"


# Age correlations -----------------------------------------------------------
A = pd.read_csv(OUT / "fig4_age_correlations.csv")
alpha = 0.05 / len(A)
mets = ["FA", "MD", "AD", "RD", "J_ln", "J_lnS"]
head = ["FA", "MD", "AD", "RD", r"$J_{\ln}$", r"$J_{\ln}^{S}$"]
lines = [r"\begin{tabular}{@{}l" + "r" * 12 + "@{}}", r"\toprule",
         r" & \multicolumn{6}{c}{$b=1500\,\mathrm{s/mm^2}$} & \multicolumn{6}{c}{$b=3000\,\mathrm{s/mm^2}$}\\",
         r"\cmidrule(lr){2-7}\cmidrule(l){8-13}",
         "Region & " + " & ".join(head * 2) + r"\\", r"\midrule"]
for roi in ROIS:
    cells = []
    for b in (1500, 3000):
        for m in mets:
            q = A[(A.shell == b) & (A.ROI == roi) & (A.metric == m)].iloc[0]
            cells.append(fmt(q.r, q.p < alpha))
    lines.append(NAMES[roi] + " & " + " & ".join(cells) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_age.tex").write_text("\n".join(lines), encoding="utf-8")

# Dependence on b ------------------------------------------------------------
C = pd.read_csv(OUT / "C_bdependence.csv").set_index("ROI")
NC = pd.read_csv(OUT / "noise_corrected.csv")
NC = NC[NC.shell == 1500].set_index("ROI")
lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & & & \multicolumn{2}{c}{change} & \multicolumn{2}{c}{$r(\Delta_b J_{\ln},\mathrm{age})$}\\",
         r"\cmidrule(lr){4-5}\cmidrule(l){6-7}",
         r"Region & $J_{\ln}(1500)$ & $J_{\ln}(3000)$ & measured & noise removed & measured & noise removed\\",
         r"\midrule"]
for roi in ROIS:
    q, n = C.loc[roi], NC.loc[roi]
    ch = f"{q.shift_pct:+.1f}".replace("-", "$-$")
    chn = f"{n.dB_nc_shift_pct:+.1f}".replace("-", "$-$")
    lines.append(f"{NAMES[roi]} & {q.Jln1500:.4f} & {q.Jln3000:.4f} & {ch}\\% & {chn}\\% & "
                 f"{fmt(q.r_age_delta)} & {fmt(n.dB_nc)}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_bdep.tex").write_text("\n".join(lines), encoding="utf-8")

# J_f, earlier preprint and corrected -------------------------------------------
J = pd.read_csv(OUT / "jf_stats.csv").set_index("ROI")
lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{paired directions (earlier preprint)} & \multicolumn{3}{c}{identical directions}\\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
         r"Region & $J_f$ & null & mismatch & $J_f$ & null & above null\\", r"\midrule"]
for roi in ROIS:
    q = J.loc[roi]
    lines.append(f"{NAMES[roi]} & {q.old:.4f} & {q.old_null:.4f} & {q.old_mismatch_pct:.0f}\\% & "
                 f"{q.new:.4f} & {q.new_null:.4f} & {100 * q.frac_new_above_null:.0f}\\%\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_jf.tex").write_text("\n".join(lines), encoding="utf-8")

# Beyond the tensor and noise -----------------------------------------------
X = pd.read_csv(OUT / "beyond_tensor.csv")
X = X[X.variant == "TN"]
lines = [r"\begin{tabular}{@{}l" + "rr" * 4 + "@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{$J_{\ln}$} & \multicolumn{4}{c}{$J_{\ln}^{S}$}\\",
         r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
         r" & \multicolumn{2}{c}{1500} & \multicolumn{2}{c}{3000} & \multicolumn{2}{c}{1500} & \multicolumn{2}{c}{3000}\\",
         r"Region & excess & $r$ & excess & $r$ & excess & $r$ & excess & $r$\\", r"\midrule"]
for roi in ROIS:
    cells = []
    for gap in ("J_ln", "J_lnS"):
        for b in (1500, 3000):
            q = X[(X.gap == gap) & (X.shell == b) & (X.ROI == roi)].iloc[0]
            cells += [f"{q.excess_pct:+.0f}\\%".replace("-", "$-$"), fmt(q.r, q.q < 0.05)]
    lines.append(NAMES[roi] + " & " + " & ".join(cells) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_beyond.tex").write_text("\n".join(lines), encoding="utf-8")
print("wrote", *sorted(p.name for p in TAB.glob("*.tex")))
