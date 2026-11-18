"""Split each HCP-A shell into two balanced halves of its gradient directions.

split_half_pass.py estimates each gap from two halves of the data whose noise is
independent. Each half must sample the sphere well enough for a sixth-order
spherical-harmonic fit. Here the directions of each shell (93 at b=1500, 92 at
b=3000, in the order of the protocol) are split into halves of 46/47 and 46/46
by a swap search that minimizes the larger of the two halves' condition numbers
of the order-6 basis. 20 random starts of 3,000 swaps each, generator seed 0,
directions of the example session (config.EXAMPLE_SESSION). The search is
deterministic, and the protocol order is the same in every session.

Writes split_half_masks.npz next to this script (boolean masks b1500 and b3000,
True for half A) and prints each half's size, the eigenvalues of its direction
matrix, and its order-4 and order-6 condition numbers.
Usage:  python split_half_masks.py
"""
from pathlib import Path

import numpy as np
from dipy.core.geometry import cart2sphere
from dipy.reconst.shm import real_sh_descoteaux

from config import DTI, EXAMPLE_SESSION

HERE = Path(__file__).resolve().parent


def cond(G, order):
    _, th, ph = cart2sphere(*G.T)
    return np.linalg.cond(real_sh_descoteaux(order, th, ph, legacy=False)[0])


src = DTI / EXAMPLE_SESSION
bvals = np.loadtxt(src / "inputs" / "dwi.bval").ravel()
bvecs = np.loadtxt(src / "processed" / "dwi_raw.bvec")
bvecs = bvecs if bvecs.shape[0] == 3 else bvecs.T
masks = {}
for b in (1500, 3000):
    i = np.where(np.abs(bvals - b) < 100)[0]
    G = bvecs[:, i].T
    G = G / np.linalg.norm(G, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    n = len(i)
    best = None
    for start in range(20):
        lab = np.zeros(n, bool)
        lab[rng.permutation(n)[:n // 2]] = True
        c = max(cond(G[lab], 6), cond(G[~lab], 6))
        for it in range(3000):
            a = rng.choice(np.where(lab)[0])
            z = rng.choice(np.where(~lab)[0])
            lab[a], lab[z] = False, True
            c2 = max(cond(G[lab], 6), cond(G[~lab], 6))
            if c2 <= c:
                c = c2
            else:
                lab[a], lab[z] = True, False
        if best is None or c < best[0]:
            best = (c, lab.copy())
    masks[f"b{b}"] = best[1]
    for name, h in (("A", best[1]), ("B", ~best[1])):
        H = G[h]
        ev = np.linalg.eigvalsh(H.T @ H / len(H))
        print(f"b={b} half {name}: {h.sum()} directions, eigenvalues {np.round(ev, 3)}, "
              f"condition number order 4 {cond(H, 4):.2f}, order 6 {cond(H, 6):.2f}")
np.savez(HERE / "split_half_masks.npz", **masks)
print("wrote split_half_masks.npz")
