"""How good is the small-gap approximation J_ln ~ (1/2) CV_D^2 in real data?

The manuscript states that with CV_D ~ 0.2-0.3 "the cubic correction enters at
the ~1% level". Analytically that looks too small. Expanding
ln(1+d) = d - d^2/2 + d^3/3 - ... and averaging (with <d> = 0) gives

    J_ln = <d^2>/2 - <d^3>/3 + ...

so the cubic term RELATIVE to the leading term is

    (<d^3>/3) / (<d^2>/2) = (2/3) * <d^3>/<d^2> = (2/3) * gamma * CV

with gamma the skewness of the directional diffusivities. At CV = 0.25 and
gamma = 1 that is ~17%, not ~1%. A ~1% relative correction would need
gamma*CV ~ 0.015, i.e. a nearly symmetric distribution. "~1%" looks like the
absolute size of CV^3 (0.016) rather than its size relative to (1/2)CV^2.

Rather than argue from assumed skewness, this measures the thing directly: per
voxel it compares the exact gap to the quadratic approximation, and reports the
observed CV, skewness and relative error.

Output: printed table.
"""
import numpy as np
import nibabel as nib

import sys
from pathlib import Path

# batch_jf_hcp lives in gap_computation/; reuse its constants and direction
# matching so this measures the pipeline that produces the published maps.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gap_computation"))
import batch_jf_hcp as bj  # noqa: E402

SESSION = "session_20260222_031250"
SHELL = bj.B1  # b = 1500, the shell the manuscript quotes CV_D for


def main():
    src = bj.OUTPUT_DIR / SESSION
    bvals = np.loadtxt(src / "inputs" / "dwi.bval").astype(float)
    img = nib.load(str(src / "processed" / "dwi_raw.nii.gz"))
    dwi = img.get_fdata().astype(np.float32)

    r = np.round(bvals / 500) * 500
    sel = r == SHELL
    b0 = dwi[..., bvals < 50].mean(-1)
    D = bj._shell_D(dwi, bvals, sel, b0)          # (X,Y,Z,ndir)
    print(f"directions at b={SHELL:.0f}: {D.shape[-1]}")

    mean = D.mean(-1)
    dev = D / np.maximum(mean[..., None], 1e-12) - 1.0   # delta_i
    m2 = (dev ** 2).mean(-1)
    m3 = (dev ** 3).mean(-1)
    cv = np.sqrt(np.maximum(m2, 0))
    skew = m3 / np.maximum(m2 ** 1.5, 1e-12)

    exact = np.maximum(np.log(np.clip(mean, 1e-12, None)) - np.log(D).mean(-1), 0.0)
    approx = 0.5 * m2

    mk = nib.load(str(bj.BASE / "HCP" / "voxel_maps" /
                      f"{SESSION}_DKI_MK.nii.gz")).get_fdata()
    wm = (mean > 0.2e-3) & (mean < 1.1e-3) & (mk > 0) & (exact > 1e-5)
    print(f"WM voxels: {int(wm.sum()):,}\n")

    def q(a, name, fmt="{:+.3f}"):
        v = a[wm]
        print(f"  {name:34s} median {fmt.format(np.median(v))}   "
              f"IQR [{fmt.format(np.percentile(v,25))}, {fmt.format(np.percentile(v,75))}]")

    q(cv, "CV_D")
    q(skew, "skewness of D across directions")
    q((2.0 / 3.0) * skew * cv, "predicted cubic term, relative")
    rel = (approx - exact) / np.maximum(exact, 1e-12)
    q(rel, "actual (approx - exact)/exact")
    q(np.abs(rel), "actual |relative error|")

    v = np.abs(rel)[wm]
    print(f"\n  fraction of WM voxels with |error| > 5%  : {100*(v>0.05).mean():.1f}%")
    print(f"  fraction of WM voxels with |error| > 10% : {100*(v>0.10).mean():.1f}%")
    print(f"  fraction of WM voxels with |error| > 1%  : {100*(v>0.01).mean():.1f}%")


if __name__ == "__main__":
    main()
