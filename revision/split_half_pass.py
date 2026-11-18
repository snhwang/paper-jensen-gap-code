"""Third native pass: gaps free of the noise variance, against a calibrated noise-floor null.

Noise acts on the diffusivity gap in two ways. Its variance raises J_ln, more at
b=3000. The rectified floor of the magnitude raises the low signals, which lowers
measured anisotropy at high b. This pass removes the first from the data and
measures the second with a null.

Split halves. Each gap is used in its second-order form, half the variance over
directions of a log profile: J_ln(b) ~ 1/2 Var ln D(b), and J_f ~ 1/2 Var ln f
with f = D(3000)/D(1500). The directions of each shell are split into two
balanced halves (split_half_masks.py), and the 14 b=0 volumes into two halves of
7. The directional ADCs of each half are fitted with real, symmetric spherical
harmonics (orders 4 and 6) and evaluated on one common, even set of 100
directions, so both shells are compared on identical directions (as jf_pass.py).
Half A and half B share no image, so their noise is independent, and half the
covariance of their two log profiles over the common directions estimates the
second-order gap without the noise variance. Biases of the mean, such as the
floor and the log of a noisy value, are shared by both halves and stay.
The same quantities from all the data ("full") keep the noise variance.
J_ln is also computed exactly from the acquired directions, as native_pass.py
does ("exact", the paper's diffusivity gap).

Null. In each voxel the tensor fitted to b=0 plus b=1500 predicts the signals of
both shells. At b=3000 its diffusivities are scaled by the voxel's measured
ADC_1(3000)/ADC_1(1500), so the b=3000 signal sits near its measured level, while
the profile keeps the same shape on both shells and the true Delta_b J_ln and J_f
stay zero. Noise at the voxel's level (sigma from the residual of an order-6 SH
fit to the b=1500 signals) is added to every volume, b=0 included, and the result
goes through the same steps. Four nulls:
  null_    Rician, b=3000 signal from the unscaled b=1500 tensor
  nullS_   Rician, scaled b=3000 signal
  nullC_   scaled, noncentral chi from four channels (signal split equally over
           the channels, per-channel SD sigma). Its
           zero-signal magnitude has mean 2.74 sigma and mean/SD 3.94, just above
           the CSF floor below. A Rician magnitude gives 1.25 sigma and 1.91.
  nullA_   scaled, the mean of four Rician images, each with SD 2 sigma, so that
           the mean has SD sigma at high signal. Single raw runs are Rician, and
           the processed images average magnitudes (the AP and PA runs combined
           after eddy, then resampled). Its zero-signal magnitude has mean 2.51
           sigma and mean/SD 3.83, just below the CSF floor. At intermediate
           signal it raises the magnitude less than nullC_, so the two bracket
           the floor's effect.
The paper's comparison with the floor now uses the calibrated nulls of
floor_null_pass.py, and these nulls remain for comparison.
The noise generator is seeded per session, so the pass is reproducible. nullA_
draws its noise last, so the other nulls are unchanged by it.

CSF floor. Ventricular CSF keeps almost no signal at b=3000 (about exp(-9) of
S0), so its magnitude there shows the floor directly. Voxels in the brain mask,
within the bounding box of the 16 regions, with S0 above twice the median S0 of
the one-voxel-eroded corona radiata and mean b=1500 S'/S0 below 0.05, eroded by
one voxel. Per voxel, the mean and SD over the b=3000 directions of the
magnitude. Per session,
the median over voxels of mean/SD, which needs no sigma, and the median mean over
sigma of the eroded corona radiata (sqrt of the median of sigma^2).

Region means of every per-voxel value, and the session's CSF floor, go to
out/split_half.csv (participant-level, not published).
Usage:  python split_half_pass.py [n_sessions_for_test]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from dipy.core.geometry import cart2sphere
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.reconst.shm import real_sh_descoteaux
from scipy import ndimage

from config import DTI, OUT, ROI_DEFS

HERE = Path(__file__).resolve().parent
MASKS = np.load(HERE / "split_half_masks.npz")
ALL_LABELS = sorted({l for v in ROI_DEFS.values() for l in v})
ORDERS = (4, 6)


def hemisphere(n):
    """Even directions on the upper hemisphere (ADC profiles are antipodal)."""
    k = np.arange(n) + 0.5
    z = 1 - k / n
    r = np.sqrt(1 - z * z)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([r * np.cos(th), r * np.sin(th), z], 1)


G_COMMON = hemisphere(100)


def basis(g, order):
    _, th, ph = cart2sphere(*np.asarray(g).T)
    return real_sh_descoteaux(order, th, ph, legacy=False)[0]


def to_common(g, order):
    """Matrix taking directional values on g to their SH fit on G_COMMON."""
    return np.linalg.pinv(basis(g, order)).T @ basis(G_COMMON, order).T


def adc(S, S0, b):
    return -np.log(np.clip(S / S0[:, None], 1e-2, 1.0)) / b


def logprofile(S, S0, b, P):
    return np.log(np.maximum(adc(S, S0, b) @ P, 1e-6))


def half_var(x):
    return 0.5 * x.var(-1)


def half_cov(x, y):
    return 0.5 * ((x - x.mean(-1, keepdims=True)) * (y - y.mean(-1, keepdims=True))).mean(-1)


def measures(X, i0, sh, g, b, tag, masks=MASKS):
    """Exact, full-data and split-half gaps for one data set (measured or simulated)."""
    out = {}
    S0 = X[:, i0].mean(-1)
    S0h = [X[:, i0[0::2]].mean(-1), X[:, i0[1::2]].mean(-1)]
    for s in (1500, 3000):
        D = adc(X[:, sh[s]], S0, b[s])
        out[f"{tag}Jln{s}_exact"] = np.maximum(
            np.log(np.maximum(D.mean(-1), 1e-6)) - np.log(np.maximum(D, 1e-6)).mean(-1), 0)
    for k in ORDERS:
        L, LA, LB = {}, {}, {}
        for s in (1500, 3000):
            idx, m = sh[s], masks[f"b{s}"]
            L[s] = logprofile(X[:, idx], S0, b[s], to_common(g[s], k))
            LA[s] = logprofile(X[:, idx[m]], S0h[0], b[s], to_common(g[s][m], k))
            LB[s] = logprofile(X[:, idx[~m]], S0h[1], b[s], to_common(g[s][~m], k))
            out[f"{tag}Jln{s}_full_o{k}"] = half_var(L[s])
            out[f"{tag}Jln{s}_split_o{k}"] = half_cov(LA[s], LB[s])
        out[f"{tag}Jf_full_o{k}"] = half_var(L[3000] - L[1500])
        out[f"{tag}Jf_split_o{k}"] = half_cov(LA[3000] - LA[1500], LB[3000] - LB[1500])
    return out


def ventricle_voxels(dwi, lab, brain, i0, sh):
    """Magnitudes at b=3000 (voxels x directions) of ventricular CSF, where only the floor remains."""
    xs, ys, zs = np.where(np.isin(lab, ALL_LABELS) & brain)
    box = tuple(slice(max(0, v.min() - 1), v.max() + 2) for v in (xs, ys, zs))
    d, lb, br = dwi[box], lab[box], brain[box]
    S0 = d[..., i0].mean(-1)
    core = ndimage.binary_erosion(np.isin(lb, ROI_DEFS["CR_all"]) & br, iterations=1)
    att = d[..., sh[1500]].mean(-1) / np.where(S0 > 0, S0, np.inf)
    csf = ndimage.binary_erosion(br & (S0 > 2 * np.median(S0[core])) & (att < 0.05), iterations=1)
    return d[csf][:, sh[3000]].astype(np.float64)


def csf_floor(dwi, lab, brain, bvals, i0, sh, sig2_cr_core):
    """Magnitude of ventricular CSF at b=3000, where only the floor remains."""
    M = ventricle_voxels(dwi, lab, brain, i0, sh)
    if len(M) < 10:
        return {"csf_n_voxels": int(len(M)), "csf_mean_over_sd": np.nan, "csf_mean_over_sigma": np.nan}
    mu, sd = M.mean(1), M.std(1, ddof=1)
    return {"csf_n_voxels": int(len(M)), "csf_mean_over_sd": float(np.median(mu / sd)),
            "csf_mean_over_sigma": float(np.median(mu) / np.sqrt(sig2_cr_core))}


def one(sid):
    t0 = time.time()
    try:
        proc = DTI / sid / "processed"
        bvals = np.loadtxt(DTI / sid / "inputs" / "dwi.bval").ravel()
        bvecs = np.loadtxt(proc / "dwi_raw.bvec")
        if bvecs.shape[0] != 3:
            bvecs = bvecs.T
        lab = np.asarray(nib.load(proc / "atlas" / "jhu_labels_registered.nii.gz").dataobj).round().astype(int)
        brain = np.asarray(nib.load(proc / "mask.nii.gz").dataobj) > 0
        U = np.isin(lab, ALL_LABELS) & brain
        i0 = np.where(bvals < 50)[0]
        sh = {s: np.where(np.abs(bvals - s) < 100)[0] for s in (1500, 3000)}
        assert len(i0) == 14 and len(sh[1500]) == 93 and len(sh[3000]) == 92
        g = {s: bvecs[:, sh[s]].T for s in sh}
        b = {s: float(bvals[sh[s]].mean()) for s in sh}
        B1 = basis(g[1500], 6)
        R1 = np.eye(len(sh[1500])) - np.linalg.pinv(B1).T @ B1.T

        def sigma2(S):
            return ((S @ R1) ** 2).sum(-1) / (len(sh[1500]) - B1.shape[1])

        dwi = np.asarray(nib.load(proc / "dwi_raw.nii.gz").dataobj, dtype=np.float32)
        core = ndimage.binary_erosion(np.isin(lab, ROI_DEFS["CR_all"]) & brain, iterations=1)
        floor = csf_floor(dwi, lab, brain, bvals, i0, sh,
                          float(np.median(sigma2(dwi[core][:, sh[1500]].astype(np.float64)))))
        X = dwi[U].astype(np.float64)
        del dwi
        labU = lab[U]
        S0 = X[:, i0].mean(-1)
        sig = np.sqrt(sigma2(X[:, sh[1500]]))
        ok = (S0 > 0) & (sig > 0) & (X[:, i0[0::2]].mean(-1) > 0) & (X[:, i0[1::2]].mean(-1) > 0)

        per = measures(X, i0, sh, g, b, "")

        keep = (bvals < 50) | (np.abs(bvals - 1500) < 100)
        fit = TensorModel(gradient_table(bvals[keep], bvecs=bvecs[:, keep].T,
                                         b0_threshold=50)).fit(X[:, keep])
        Q = fit.quadratic_form
        adc1 = {s: adc(X[:, sh[s]], S0, b[s]).mean(-1) for s in sh}
        ratio = np.clip(adc1[3000] / np.maximum(adc1[1500], 1e-6), 0.5, 1.2)
        used = np.concatenate([i0, sh[1500], sh[3000]])
        rng = np.random.default_rng(int(sid.replace("session_", "").replace("_", "")) + 11)
        s_ = sig[:, None]
        for tag, scaled, noise in (("null_", False, "rician"), ("nullS_", True, "rician"),
                                   ("nullC_", True, "ncchi4"), ("nullA_", True, "riceavg4")):
            clean = np.empty_like(X)
            clean[:, i0] = S0[:, None]
            for s in sh:
                Dt = np.maximum(np.einsum("ij,njk,ik->ni", g[s], Q, g[s]), 0)
                if scaled and s == 3000:
                    Dt = Dt * ratio[:, None]
                clean[:, sh[s]] = S0[:, None] * np.exp(-b[s] * Dt)
            c = clean[:, used]
            if noise == "rician":
                M = np.hypot(c + rng.normal(0, 1, c.shape) * s_, rng.normal(0, 1, c.shape) * s_)
            elif noise == "riceavg4":
                M = np.zeros_like(c)
                for _ in range(4):
                    M += np.hypot(c + 2 * s_ * rng.normal(size=c.shape), 2 * s_ * rng.normal(size=c.shape))
                M /= 4
            else:
                tot = np.zeros_like(c)
                for _ in range(4):
                    tot += (c / 2 + s_ * rng.normal(size=c.shape)) ** 2 + (s_ * rng.normal(size=c.shape)) ** 2
                M = np.sqrt(tot)
            Xn = np.zeros_like(X)
            Xn[:, used] = M
            per.update(measures(Xn, i0, sh, g, b, tag))

        rows = []
        for roi, ls in ROI_DEFS.items():
            mk = np.isin(labU, ls) & ok
            row = {"Session_ID": sid, "ROI": roi, "n_voxels": int(mk.sum()), **floor}
            for k, v in per.items():
                vv = v[mk]
                vv = vv[np.isfinite(vv)]
                row[k] = float(vv.mean()) if vv.size else np.nan
            rows.append(row)
        for r in rows:
            r["seconds"] = time.time() - t0
        return rows
    except Exception as e:  # noqa: BLE001
        return [{"Session_ID": sid, "error": repr(e)}]


if __name__ == "__main__":
    sessions = pd.read_csv(OUT / "v1_cohort_sessions.csv").Session_ID.tolist()
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        df = pd.DataFrame([r for s in sessions[: int(sys.argv[1])] for r in one(s)])
        df.to_csv(OUT / "split_half_test.csv", index=False)
        print(df[df.ROI == "CR_all"].T.to_string())
        sys.exit()
    part = OUT / "split_half_partial.csv"
    done = set(pd.read_csv(part).Session_ID) if part.exists() else set()
    todo = [s for s in sessions if s not in done]
    print(f"{len(todo)} sessions to process ({len(done)} already done)", flush=True)
    with Pool(6) as pool:
        for i, rows in enumerate(pool.imap_unordered(one, todo, chunksize=2), 1):
            dest = OUT / "split_half_errors.csv" if "error" in rows[0] else part
            pd.DataFrame(rows).to_csv(dest, mode="a", header=not dest.exists(), index=False)
            if i % 50 == 0:
                print(f"{i}/{len(todo)} at {time.strftime('%H:%M:%S')}", flush=True)
    df = pd.read_csv(part)
    df.to_csv(OUT / "split_half.csv", index=False)
    print(f"wrote split_half.csv: {df.Session_ID.nunique()} sessions")
