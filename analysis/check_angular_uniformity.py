"""Is the acquisition's angular coverage isotropic enough for uniform weights?

The gaps are Holder means over the sampled directions with weights w_i. The paper
uses uniform weights w_i = 1/N_dir, which Section 2.7 justifies for
"approximately isotropic angular coverage", and Limitations notes that the gap
reflects the true angular profile only when the sphere is well covered. This
checks that the acquisition actually meets that condition.

Two measures, both computed from the bvecs alone:

  Second-moment tensor  T = <u u^T> over the shell's unit directions. For
  perfectly isotropic sampling T = I/3, so all three eigenvalues are 1/3 and the
  deviation from 1/3 bounds how much a directional average is skewed by the
  scheme rather than by tissue. This is the measure that matters for the gaps,
  because it is exactly the first-order bias in a direction-averaged quantity.

  Nearest-neighbour angle, antipodally symmetric (|cos| is used, since a
  direction and its negative are the same measurement). Small scatter here means
  no clustered or bare patches.

For the HCP-A scheme used in the paper this gives eigenvalues within 0.008 of
1/3 at both shells, about 2.5% relative, so uniform weights are appropriate and
Voronoi-area weighting is unnecessary.

Usage:
    DTI_OUTPUT_DIR=/path/to/sessions python check_angular_uniformity.py [session]
"""
import os
import sys

import numpy as np

DEFAULT_SESSION = "session_20260222_031250"   # the paper's example participant
SHELLS = (1500, 3000)


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SESSION
    root = os.environ.get("DTI_OUTPUT_DIR", "Q:/dti_output")
    base = os.path.join(root, session, "inputs")

    bvals = np.loadtxt(os.path.join(base, "dwi.bval"))
    bvecs = np.loadtxt(os.path.join(base, "dwi.bvec"))
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T

    shell_of = np.round(bvals / 500) * 500
    print(f"session: {session}")
    for b in SHELLS:
        u = bvecs[shell_of == b]
        u = u / np.linalg.norm(u, axis=1, keepdims=True)

        T = (u[:, :, None] * u[:, None, :]).mean(0)
        ev = np.sort(np.linalg.eigvalsh(T))[::-1]

        cos = np.abs(u @ u.T)
        np.fill_diagonal(cos, 0.0)
        nn = np.degrees(np.arccos(np.clip(cos.max(1), 0, 1)))

        print(f"\n  b = {b:.0f}   N_dir = {len(u)}")
        print(f"    second-moment eigenvalues : {ev[0]:.4f}, {ev[1]:.4f}, {ev[2]:.4f}"
              "   (isotropic: 0.3333 each)")
        print(f"    max deviation from 1/3    : {np.abs(ev - 1/3).max():.4f}"
              f"  ({100*np.abs(ev - 1/3).max()/(1/3):.1f}% relative)")
        print(f"    nearest-neighbour angle   : median {np.median(nn):.1f} deg, "
              f"min {nn.min():.1f} deg, max {nn.max():.1f} deg")


if __name__ == "__main__":
    main()
