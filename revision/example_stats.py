"""Numbers reported for the example participant and for the direction sets.

Manuscript Section 3.1: eigenvalues of M = (1/N) sum g g^T for each shell.
Manuscript Section 4.1: in white matter of the example participant,
  - medians of ADC_0/ADC_1 and D_SM/ADC_1 ("about 5% below")
  - Spearman correlation of J_ln with FA at both shells
  - Spearman correlation of J_lnS with (b MD)^2 J_ln, and their median ratio
  - number of white-matter voxels (Figure 4a caption)
Manuscript Section 3.2: in the same white matter, how many directional signals the
  limits S' in [1e-2, 1] and the floor D >= 1e-6 change, and how many gaps are
  set to zero

Input:   example-participant maps (example_maps.py), the session's b-vectors and
         its DWI
Output:  out/example_stats.txt
Usage:   python example_stats.py
"""
import nibabel as nib
import numpy as np
from scipy.stats import spearmanr

from config import DTI, EXAMPLE_SESSION, MAPS, OUT

lines = []


def say(s):
    print(s)
    lines.append(s)


src = DTI / EXAMPLE_SESSION
bvals = np.loadtxt(src / "inputs" / "dwi.bval").ravel()
bvecs = np.loadtxt(src / "processed" / "dwi_raw.bvec")
if bvecs.shape[0] != 3:
    bvecs = bvecs.T
for b in (1500, 3000):
    g = bvecs[:, np.abs(bvals - b) < 100].T
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    ev = np.sort(np.linalg.eigvalsh(g.T @ g / len(g)))[::-1]
    say(f"direction set b={b}: {len(g)} directions, eigenvalues of M "
        f"{', '.join(f'{e:.3f}' for e in ev)}; max deviation from 1/3 "
        f"{100 * np.abs(ev - 1 / 3).max() / (1 / 3):.1f}%")


def L(name):
    return nib.load(MAPS / f"{name}.nii.gz").get_fdata()


wm = L("wm_jhu_mask") > 0
for b in (1500, 3000):
    m = {k: L(f"{k}_b{b}") for k in ("ADC_1", "ADC_0", "D_SM", "FA", "MD", "J_ln", "J_lnS")}
    w = wm & (m["FA"] > 0) & (m["J_ln"] > 0)
    fa, jl, js, md = (m[k][w] for k in ("FA", "J_ln", "J_lnS", "MD"))
    pred = (b * md) ** 2 * jl
    say(f"example b={b}: white-matter voxels {w.sum()}; median ADC_0/ADC_1 "
        f"{np.median(m['ADC_0'][w] / m['ADC_1'][w]):.3f}; median D_SM/ADC_1 "
        f"{np.median(m['D_SM'][w] / m['ADC_1'][w]):.3f}; Spearman J_ln~FA "
        f"{spearmanr(jl, fa)[0]:.3f}; Spearman J_lnS~(b MD)^2 J_ln "
        f"{spearmanr(js, pred)[0]:.3f}; median J_lnS/((b MD)^2 J_ln) "
        f"{np.median(js / pred):.3f}")

# The safeguards of Section 3.2, counted on the raw signals of the same voxels.
dwi = np.asarray(nib.load(src / "processed" / "dwi_raw.nii.gz").dataobj, dtype=np.float32)
X = dwi[wm].astype(np.float64)
del dwi
S0 = X[:, bvals < 50].mean(-1)
ok = S0 > 0
for b in (1500, 3000):
    sh = np.abs(bvals - b) < 100
    r = X[ok][:, sh] / S0[ok, None]
    Sp = np.clip(r, 1e-2, 1.0)
    D = -np.log(Sp) / bvals[sh].mean()
    lnD = np.log(np.maximum(D, 1e-6))
    jln = np.log(np.maximum(D.mean(-1), 1e-6)) - lnD.mean(-1)
    jls = np.log(Sp.mean(-1)) - np.log(Sp).mean(-1)
    say(f"example b={b} safeguards: {r.size} signals in {ok.sum()} voxels; "
        f"S' > 1 {100 * (r > 1).mean():.3f}%, S' < 1e-2 {100 * (r < 1e-2).mean():.3f}%, "
        f"D < 1e-6 {100 * (D < 1e-6).mean():.3f}%; gaps below zero J_ln {int((jln < 0).sum())}, "
        f"J_lnS {int((jls < 0).sum())}")

(OUT / "example_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
