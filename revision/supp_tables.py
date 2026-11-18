"""LaTeX tables for the Supplementary Material, written from the analysis outputs.

  tab_age.tex       age correlations of the gaps and tensor metrics (Figure 4)
  tab_bdep.tex      dependence on b, by region, measured (v1_roi_stats.py) and against the
                    noise-floor nulls TaNc and DaNc (floor_null_stats.py)

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
F = pd.read_csv(OUT / "floor_null_regions.csv")
NT, ND = (F[F.null == t].set_index("ROI") for t in ("TaNc", "DaNc"))
a16 = 0.05 / 16


def pct(x, sig=True):
    s = f"{x:+.1f}".replace("-", "$-$")
    return s if sig else f"\\textit{{{s}}}"


lines = [r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
         r" & & & \multicolumn{5}{c}{change, \% of $J_{\ln}(1500)$} & \multicolumn{2}{c}{$r$ of excess with age}\\",
         r"\cmidrule(lr){4-8}\cmidrule(l){9-10}",
         r" & & & & \multicolumn{2}{c}{null} & \multicolumn{2}{c}{excess} & & \\",
         r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Region & $J_{\ln}(1500)$ & $J_{\ln}(3000)$ & measured & tensor & direct. & tensor & direct."
         r" & tensor & direct.\\",
         r"\midrule"]
for roi in ROIS:
    q, t, d = C.loc[roi], NT.loc[roi], ND.loc[roi]
    lines.append(f"{NAMES[roi]} & {q.Jln1500:.4f} & {q.Jln3000:.4f} & {pct(q.shift_pct)} & "
                 f"{pct(t.floor_exact)} & {pct(d.floor_exact)} & "
                 f"{pct(t.excess_exact, t.p_excess_exact < a16)} & {pct(d.excess_exact, d.p_excess_exact < a16)} & "
                 f"{fmt(t.r_age_excess_exact, t.p_age_excess_exact < a16)} & "
                 f"{fmt(d.r_age_excess_exact, d.p_age_excess_exact < a16)}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "tab_bdep.tex").write_text("\n".join(lines), encoding="utf-8")

print("wrote", *sorted(p.name for p in TAB.glob("*.tex")))
