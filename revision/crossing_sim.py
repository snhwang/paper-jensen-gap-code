"""Crossing fibres: the diffusivity gap against the gap of the fitted tensor (Section 2.5, added 2026-11-01).

Theory only, deterministic, no data. A voxel holds two equal populations of Gaussian fibres (axially symmetric
tensors, D_par = 1.7e-3 and D_perp = 0.3e-3 mm^2/s) whose axes cross at an angle theta. Its signal along each
direction is the mean of the two populations' signals. The directional ADCs are D_i = -ln(S_i) / b, sampled by 40,000
near-uniform (Fibonacci) directions as in theory_table.py, so the result reflects the profile, not a direction set.

  J_ln      the diffusivity gap of the directional ADCs, ln<D> - <ln D>
  tensor    the tensor fitted to the D_i by least squares (the log-linear fit), its FA, and the gap of its own
            directional ADCs on the same directions, J_ln_T
For one population the profile is that of a tensor, so J_ln = J_ln_T. When two populations cross, the profile has
angular structure a tensor cannot represent, and J_ln differs from J_ln_T. Reported for theta = 0, 30, 60 and 90
degrees at b = 1500 and 3000 s/mm^2.

With noise (one example, as in an HCP-A white-matter voxel): 93 and 92 near-uniform directions at b = 1500 and 3000,
14 b = 0 volumes, Rician noise at a baseline signal-to-noise ratio of 25, and the measured pipeline of
native_pass.py (S0 from the noisy b = 0 mean, S' clipped to [1e-2, 1]). 10,000 voxels for theta = 0 and 90, seed 0.
Noise adds angular structure of its own, so it raises J_ln above J_ln_T even for one population. The question is
how much more the crossing adds.

Output:  out/crossing_sim.txt
Usage:   python crossing_sim.py
"""
import numpy as np

from config import OUT

SNR0, NB0, NVOX, NDIR = 25.0, 14, 10000, {1500: 93, 3000: 92}


def fib(n):
    kk = np.arange(n) + 0.5
    ph, tt = np.arccos(1 - 2 * kk / n), np.pi * (1 + 5 ** 0.5) * kk
    return np.stack([np.cos(tt) * np.sin(ph), np.sin(tt) * np.sin(ph), np.cos(ph)], 1)

N = 40000
k = np.arange(N) + 0.5
phi = np.arccos(1 - 2 * k / N)
th = np.pi * (1 + 5 ** 0.5) * k
G = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
X = np.c_[G[:, 0] ** 2, G[:, 1] ** 2, G[:, 2] ** 2, 2 * G[:, 0] * G[:, 1], 2 * G[:, 0] * G[:, 2], 2 * G[:, 1] * G[:, 2]]
D_PAR, D_PERP = 1.7e-3, 0.3e-3


def fibre(theta_deg):
    t = np.radians(theta_deg)
    u = np.array([np.cos(t), np.sin(t), 0.0])
    return D_PERP * np.eye(3) + (D_PAR - D_PERP) * np.outer(u, u)


def proj(T):
    return np.einsum("ij,jk,ik->i", G, T, G)


def jln(D):
    return np.log(D.mean()) - np.log(D).mean()


lines = ["Two equal Gaussian fibre populations crossing at theta (crossing_sim.py)",
         "b       theta  FA_tensor  J_ln     J_ln_T   J_ln/J_ln_T"]
for b in (1500, 3000):
    for theta in (0, 30, 60, 90):
        S = 0.5 * np.exp(-b * proj(fibre(0))) + 0.5 * np.exp(-b * proj(fibre(theta)))
        D = -np.log(S) / b
        c = np.linalg.lstsq(X, D, rcond=None)[0]
        T = np.array([[c[0], c[3], c[4]], [c[3], c[1], c[5]], [c[4], c[5], c[2]]])
        lam = np.linalg.eigvalsh(T)
        fa = np.sqrt(1.5 * ((lam - lam.mean()) ** 2).sum() / (lam ** 2).sum())
        jt = jln(proj(T))
        lines.append(f"{b}  {theta:5d}  {fa:9.3f}  {jln(D):.5f}  {jt:.5f}  {jln(D) / jt:.3f}")

# One example with noise, as in an HCP-A white-matter voxel
rng = np.random.default_rng(0)
lines += ["", f"With Rician noise, baseline SNR {SNR0:.0f}, {NB0} b=0 volumes, {NVOX} voxels "
              "(means over voxels; J_ln/J_ln_T is the ratio of the means)",
          "b       theta  ndir  J_ln     J_ln_T   J_ln/J_ln_T"]
sig = 1.0 / SNR0
for b in (1500, 3000):
    g = fib(NDIR[b])
    Xg = np.c_[g[:, 0] ** 2, g[:, 1] ** 2, g[:, 2] ** 2, 2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2], 2 * g[:, 1] * g[:, 2]]
    pinv = np.linalg.pinv(Xg)
    for theta in (0, 90):
        clean = 0.5 * np.exp(-b * np.einsum("ij,jk,ik->i", g, fibre(0), g)) \
            + 0.5 * np.exp(-b * np.einsum("ij,jk,ik->i", g, fibre(theta), g))
        S = np.hypot(clean + sig * rng.standard_normal((NVOX, len(g))), sig * rng.standard_normal((NVOX, len(g))))
        S0 = np.hypot(1 + sig * rng.standard_normal((NVOX, NB0)), sig * rng.standard_normal((NVOX, NB0))).mean(1)
        Sp = np.clip(S / S0[:, None], 1e-2, 1.0)
        D = np.maximum(-np.log(Sp) / b, 1e-6)
        jl = np.log(D.mean(1)) - np.log(D).mean(1)
        Dt = np.maximum((D @ pinv.T) @ Xg.T, 1e-6)          # directional ADCs of the fitted tensor
        jt = np.log(Dt.mean(1)) - np.log(Dt).mean(1)
        lines.append(f"{b}  {theta:5d}  {len(g):4d}  {jl.mean():.5f}  {jt.mean():.5f}  {jl.mean() / jt.mean():.3f}")
txt = "\n".join(lines)
(OUT / "crossing_sim.txt").write_text(txt + "\n", encoding="utf-8")
print(txt)
