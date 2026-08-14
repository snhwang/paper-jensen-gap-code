"""Two multidimensional gap visualizations on the n=1379 HCP-A cohort.

(A) Spectrum heatmap: rung-pair gaps (diffusivity + signal domains, both shells)
    x ROIs (the 16 JHU white-matter ROIs from the skel analysis); cell = Pearson
    r vs age. Diverging colormap, red = +age, blue = -age.

(B) Per-subject SCR scatter: J_ln_D x J_ln_S (b=1500), colored by age. Shows
    the dissociation as an anti-diagonal aging trajectory.

Output: figures/gap_spectrum_heatmap.{png,pdf}, figures/gap_scr_scatter.{png,pdf}
"""
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"; FIG.mkdir(exist_ok=True)

ROIS = ["CR_all", "ACR", "SCR", "PCR", "Body", "Genu", "Splenium",
        "alic", "plic", "rlic", "ec", "slf", "ptr",
        "cingulum_cc", "ss", "tapetum"]

DIFF = [("ADC_m1","ADC_0",r"$G_{-1,0}$"),
        ("ADC_0","ADC_1",r"$G_{0,1}=J_{\ln}$"),
        ("ADC_1","ADC_2",r"$G_{1,2}$"),
        ("ADC_m1","ADC_2",r"$G_{-1,2}$"),
        ("ADC_min","ADC_max",r"$G_{\min,\max}$")]
SIG  = [("Mm1_S","M0_S",r"$G^{S}_{-1,0}$"),
        ("M0_S","M1_S",r"$G^{S}_{0,1}=J_{\ln}^{S}$"),
        ("M1_S","M2_S",r"$G^{S}_{1,2}$"),
        ("Mm1_S","M2_S",r"$G^{S}_{-1,2}$"),
        ("M_min_S","M_max_S",r"$G^{S}_{\min,\max}$")]


def partial_r_p(gap, age, sex):
    """Partial correlation (and its p) of gap vs age, controlling for binary sex."""
    df = pd.DataFrame({"g": gap, "a": age, "s": sex}).dropna()
    if len(df) <= 10:
        return np.nan, np.nan
    sd = pd.get_dummies(df["s"], drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(df)), sd.values])
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    r, _ = stats.pearsonr(res(df["g"].values), res(df["a"].values))
    dofp = max(len(df) - X.shape[1] - 1, 1)
    t = r * np.sqrt(dofp / max(1 - r * r, 1e-300))
    return r, 2 * stats.t.sf(abs(t), dofp)


def age_r_grid(g, man, bval, pairs):
    gb = g[g.shell_bval == bval]
    j = gb.merge(man[["session_id","age","sex"]], left_on="Session_ID", right_on="session_id", how="inner")
    rout = np.full((len(pairs), len(ROIS)), np.nan)
    pout = np.full((len(pairs), len(ROIS)), np.nan)
    for i, (lo, hi, _) in enumerate(pairs):
        for k, roi in enumerate(ROIS):
            s = j[j.ROI == roi]
            l = pd.to_numeric(s[f"{lo}_mean"], errors="coerce").clip(lower=1e-12)
            h = pd.to_numeric(s[f"{hi}_mean"], errors="coerce").clip(lower=1e-12)
            gap = np.log(h / l)
            rout[i, k], pout[i, k] = partial_r_p(gap.values, s["age"].values, s["sex"].values)
    return rout, pout


def main():
    g = pd.read_csv(BASE / "HCP" / "cr_gaps_long.csv")
    mans = {b: pd.read_csv(BASE / "HCP" / f"manifest_n1379_b{b}.tsv", sep="\t") for b in (1500, 3000)}
    for b in mans:
        mans[b]["age"] = pd.to_numeric(mans[b]["age"], errors="coerce")

    # ---------- (A) spectrum heatmap ----------
    # Build a stacked matrix: [diff_b1500 (5), diff_b3000 (5), sig_b1500 (5), sig_b3000 (5)] x ROIS
    blocks = [
        (age_r_grid(g, mans[1500], 1500, DIFF), [d[2] for d in DIFF], "D, $b{=}1500$"),
        (age_r_grid(g, mans[3000], 3000, DIFF), [d[2] for d in DIFF], "D, $b{=}3000$"),
        (age_r_grid(g, mans[1500], 1500, SIG),  [s[2] for s in SIG],  "S, $b{=}1500$"),
        (age_r_grid(g, mans[3000], 3000, SIG),  [s[2] for s in SIG],  "S, $b{=}3000$"),
    ]
    M = np.vstack([b[0][0] for b in blocks])
    P = np.vstack([b[0][1] for b in blocks])
    # Transpose to portrait: ROIs on y-axis (horizontal labels), rung-shell blocks on x-axis.
    MT = M.T
    PT = P.T
    # Bonferroni correction across all finite cells
    finite = np.isfinite(PT)
    n_tests = int(finite.sum())
    sig = finite & (PT < 0.05 / n_tests)
    xticklabs = sum([[f"{lbl} ({tag})" for lbl in lbls] for grid, lbls, tag in blocks], [])
    fig, ax = plt.subplots(figsize=(10, 12.5))
    im = ax.imshow(MT, cmap="RdBu_r", vmin=-0.75, vmax=0.75, aspect="auto")
    ax.set_xticks(range(MT.shape[1]))
    ax.set_xticklabels(xticklabs, rotation=90, ha="center", va="top", fontsize=12)
    ax.set_yticks(range(MT.shape[0])); ax.set_yticklabels(ROIS, fontsize=13)
    # block dividers between the four rung-shell blocks of 5 (now along x)
    for k in (5, 10, 15):
        ax.axvline(k - 0.5, color="black", lw=1.2)
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label(r"Pearson $r$ vs age", fontsize=13)
    cb.ax.tick_params(labelsize=12)
    # hatch cells that are NOT significant after Bonferroni correction
    for i in range(MT.shape[0]):
        for j in range(MT.shape[1]):
            if finite[i, j] and not sig[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                             hatch="////", edgecolor="black", lw=0, alpha=0.5))
    # annotate r-values on cells (rounded to 2)
    for i in range(MT.shape[0]):
        for j in range(MT.shape[1]):
            v = MT[i, j]
            if np.isfinite(v):
                color = "white" if abs(v) > 0.45 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        rotation=90, fontsize=10, color=color)
    ax.set_title("Gap-spectrum age sensitivity across white-matter ROIs\n"
                 "(HCP-A, $N=1{,}379$; cols: rung-pair gaps in diffusivity (D) and signal (S) domains, both shells; "
                 "rows: JHU WM ROIs)", fontsize=12)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(FIG / f"gap_spectrum_heatmap.{ext}", dpi=200,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {FIG / 'gap_spectrum_heatmap'}.{{png,pdf}}")
    print(f"  Bonferroni: {int(sig.sum())}/{n_tests} cells significant (p < 0.05/{n_tests})")
    paper_fig = BASE.parent / "paper-jensen-gap" / "figures"
    if paper_fig.exists():
        for ext in ("png", "pdf"):
            shutil.copy2(FIG / f"gap_spectrum_heatmap.{ext}", paper_fig / f"gap_spectrum_heatmap.{ext}")
        print("  copied heatmap into paper-jensen-gap/figures")

    # ---------- (B) SCR per-subject J_ln_D x J_ln_S scatter ----------
    gb = g[(g.shell_bval == 1500) & (g.ROI == "SCR")]
    j = gb.merge(mans[1500][["session_id","age"]], left_on="Session_ID", right_on="session_id", how="inner")
    j["J_ln_D_mean"] = pd.to_numeric(j["J_ln_D_mean"], errors="coerce")
    j["J_ln_S_mean"] = pd.to_numeric(j["J_ln_S_mean"], errors="coerce")
    j = j.dropna(subset=["J_ln_D_mean","J_ln_S_mean","age"])

    fig, ax = plt.subplots(figsize=(7.2, 6))
    sc = ax.scatter(j["J_ln_D_mean"], j["J_ln_S_mean"], c=j["age"],
                    cmap="viridis", s=14, alpha=0.85, edgecolors="none")
    cb = plt.colorbar(sc, ax=ax); cb.set_label("Age (y)", fontsize=10)
    # add a best-fit line per age tertile to show the anti-diagonal trajectory
    q1, q2 = j["age"].quantile([1/3, 2/3]).values
    bands = [("young ($<{:.0f}$)".format(q1), j[j["age"] < q1]),
             (f"mid ($\\sim${q1:.0f}--{q2:.0f})", j[(j["age"] >= q1) & (j["age"] <= q2)]),
             ("old ($>{:.0f}$)".format(q2), j[j["age"] > q2])]
    for label, b in bands:
        cx, cy = b["J_ln_D_mean"].mean(), b["J_ln_S_mean"].mean()
        ax.scatter([cx],[cy], s=160, marker="X", edgecolor="black", linewidth=1.4,
                   color="red" if "old" in label else ("orange" if "mid" in label else "white"),
                   zorder=10, label=f"{label} centroid")
    ax.set_xlabel(r"$J_{\ln}=\ln(\mathrm{ADC}_1/\mathrm{ADC}_0)$  (diffusivity gap)", fontsize=11)
    ax.set_ylabel(r"$J_{\ln}^{S}=\ln\langle S'\rangle-\langle\ln S'\rangle$  (signal gap)", fontsize=11)
    ax.set_title("Per-subject gap dissociation in SCR  (HCP-A, $b=1500$, $N=1{,}379$)\n"
                 r"Anti-diagonal age trajectory: aging moves subjects to lower $J_{\ln}$ AND higher $J_{\ln}^{S}$",
                 fontsize=10)
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9, framealpha=0.92)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(FIG / f"gap_scr_scatter.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG / 'gap_scr_scatter'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
