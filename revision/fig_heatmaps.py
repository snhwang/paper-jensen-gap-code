"""Age-correlation heatmaps: Figure 4 (gaps and tensor metrics) and
Supplementary Figure S1 (rung-pair gap spectrum).

Both use the same layout: regions in rows, metrics in columns, one block per
shell, sex-adjusted partial r in colour, Bonferroni-non-significant cells hatched.

Inputs: out/E_spectrum_voxelwise.csv (v1_roi_stats.py) for S1, and
out/native_pass.csv with the cohort file for Figure 4.
Usage:  python fig_heatmaps.py [spectrum|main|both]
"""
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import FIGS, NAMES, OUT, ROIS
from config import SUPP_FIGS as SUPP

matplotlib.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "pdf.fonttype": 42,
                            "hatch.linewidth": 0.5})



def heatmap(R, P, cols, col_labels, title, path, alpha):
    """R, P: dicts shell -> DataFrame (rows ROIS, columns cols)."""
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 3.9), sharey=True,
                            gridspec_kw={"wspace": 0.04}, constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r")
    for ax, b in zip(axs, (1500, 3000)):
        r = R[b].loc[ROIS, cols].values
        p = P[b].loc[ROIS, cols].values
        im = ax.imshow(r, cmap=cmap, vmin=-0.75, vmax=0.75, aspect="auto")
        for i, j in zip(*np.where(p >= alpha)):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       hatch="////", edgecolor="0.45", lw=0))
        ax.set_xticks(range(len(cols)), col_labels, rotation=0)
        ax.set_title(f"$b={b}\\,\\mathrm{{s/mm^2}}$")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ROIS)), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="minor", length=0)
    axs[0].set_yticks(range(len(ROIS)), [NAMES[k] for k in ROIS])
    cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02)
    cb.set_label("partial $r$ with age (sex-adjusted)")
    cb.outline.set_linewidth(0.4)
    if title:
        fig.suptitle(title)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def spectrum():
    E = pd.read_csv(OUT / "E_spectrum_voxelwise.csv")
    cols = ["G_-1,0", "G_0,1", "G_1,2", "G_-1,2", "GS_0,1", "GS_1,2"]
    labels = [r"$G_{-1,0}$", r"$G_{0,1}$", r"$G_{1,2}$", r"$G_{-1,2}$",
              r"$G^{S}_{0,1}$", r"$G^{S}_{1,2}$"]
    R = {b: E[E.shell == b].pivot(index="ROI", columns="gap", values="r") for b in (1500, 3000)}
    P = {b: E[E.shell == b].pivot(index="ROI", columns="gap", values="p") for b in (1500, 3000)}
    n = len(E)
    heatmap(R, P, cols, labels, None, SUPP / "figS1_spectrum", 0.05 / n)
    print(f"spectrum: {n} tests, {(E.p < 0.05 / n).sum()} significant")


def partial_r(y, age, sex):
    X = np.column_stack([np.ones_like(age), (sex == "M").astype(float)])
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    ra = age - X @ np.linalg.lstsq(X, age, rcond=None)[0]
    r = np.corrcoef(ry, ra)[0, 1]
    t = r * np.sqrt((len(y) - 3) / (1 - r * r))
    return r, 2 * stats.t.sf(abs(t), len(y) - 3)


def main():
    nat = pd.read_csv(OUT / "native_pass.csv")
    coh = pd.read_csv(OUT / "v1_cohort_sessions.csv")
    d = nat.merge(coh, on="Session_ID")
    mets = ["FA", "MD", "AD", "RD", "J_ln", "J_lnS"]
    labels = ["FA", "MD", "AD", "RD", r"$J_{\ln}$", r"$J_{\ln}^{S}$"]
    rows = []
    for (b, roi), s in d.groupby(["shell_bval", "ROI"]):
        for m in mets:
            r, p = partial_r(s[m].values, s.age.values, s.sex.values)
            rows.append(dict(shell=b, ROI=roi, metric=m, r=r, p=p, n=len(s)))
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "fig4_age_correlations.csv", index=False)
    R = {b: A[A.shell == b].pivot(index="ROI", columns="metric", values="r") for b in (1500, 3000)}
    P = {b: A[A.shell == b].pivot(index="ROI", columns="metric", values="p") for b in (1500, 3000)}
    heatmap(R, P, mets, labels, None, FIGS / "fig4_age_heatmap", 0.05 / len(A))
    print(f"main: {len(A)} tests, {(A.p < 0.05 / len(A)).sum()} significant, "
          f"n per cell {A.n.min()}-{A.n.max()}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("spectrum", "both"):
        spectrum()
    if what in ("main", "both"):
        main()
