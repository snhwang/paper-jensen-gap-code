"""Supplementary Table S1: accuracy of the quadratic approximations, by simulation.

For Gaussian tensors with MD = 0.75e-3 mm^2/s, sampled by 40,000 near-uniform
(Fibonacci) directions, at FA 0.2 to 0.6 and three shapes (prolate,
intermediate, planar), it computes
  - CV_D, checked against CV_D^2 = 4 FA^2 / (15 - 10 FA^2) (Equation 8)
  - the exact diffusivity gap and the relative error of 0.5 CV_D^2
  - the exact signal gap at b = 1500 and 3000 and the relative error of
    0.5 b^2 Var(D)
The theory section quotes this table (prolate within about 5%, planar up to 30%
at FA 0.6, CV_D 0.22 to 0.36 for FA 0.4 to 0.6).

Output:  tables_supp/tab_accuracy.tex
Usage:   python theory_table.py
"""
import numpy as np
from scipy.optimize import brentq

from config import TABLES

N = 40000
k = np.arange(N) + 0.5
phi = np.arccos(1 - 2 * k / N)
th = np.pi * (1 + 5 ** 0.5) * k
G2 = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1) ** 2
MD = 0.75e-3
SHAPES = {"prolate": (lambda t: np.array([1 + 2 * t, 1 - t, 1 - t]), 0.999),
          "intermediate": (lambda t: np.array([1 + 1.5 * t, 1, 1 - 1.5 * t]), 0.6666),
          "planar": (lambda t: np.array([1 + t, 1 + t, 1 - 2 * t]), 0.4999)}


def fa_of(lam):
    m = lam.mean()
    return np.sqrt(1.5 * ((lam - m) ** 2).sum() / (lam ** 2).sum())


def pct(x):
    return f"${x:+.1f}\\%$"


rows = []
for FA in (0.2, 0.3, 0.4, 0.5, 0.6):
    for i, (shape, (f, hi)) in enumerate(SHAPES.items()):
        t = brentq(lambda t: fa_of(f(t)) - FA, 1e-9, hi)
        D = G2 @ (f(t) * MD)
        cv2 = D.var() / D.mean() ** 2
        assert abs(cv2 - 4 * FA ** 2 / (15 - 10 * FA ** 2)) < 1e-3, "Equation 8 check"
        j = np.log(D.mean()) - np.log(D).mean()
        cells = [f"{FA:.1f}" if i == 0 else "", shape, f"{np.sqrt(cv2):.3f}",
                 f"{j:.4f}", pct(100 * (0.5 * cv2 - j) / j)]
        for b in (1500, 3000):
            S = np.exp(-b * D)
            js = np.log(S.mean()) - np.log(S).mean()
            cells += [f"{js:.4f}", pct(100 * (0.5 * b * b * D.var() - js) / js)]
        rows.append(" & ".join(cells) + r"\\")

lines = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
         r" & & & \multicolumn{2}{c}{$J_{\ln}$} & \multicolumn{2}{c}{$J_{\ln}^{S}$, $b=1500$}"
         r" & \multicolumn{2}{c}{$J_{\ln}^{S}$, $b=3000$}\\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
         r"FA & Shape & $\mathrm{CV}_D$ & exact & error & exact & error & exact & error\\",
         r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
(TABLES / "tab_accuracy.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(rows))
print("wrote", TABLES / "tab_accuracy.tex")
