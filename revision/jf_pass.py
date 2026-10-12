"""Second native pass: J_f compared on identical directions, with a noise null.

The HCP-A shells do not share their gradient directions. The submitted J_f paired
each b=1500 direction with its nearest b=3000 direction, but the pairs are 2 to
10 degrees apart, so f_i compared different directions. Here, in every voxel, the
directional ADCs of each shell are fitted with real, symmetric spherical
harmonics of order 6, and both fits are evaluated on one common, even set of
directions. Under Gaussian diffusion the two ADC profiles are the same quadratic
form, which order 6 represents exactly, so f = 1 and J_f = 0.

Noise still raises J_f. A null is simulated per voxel: the tensor fitted to
b=0 plus b=1500 predicts the same Gaussian profile on both shells, Rician noise
at the voxel's level is added (sigma from the residual of an order-6 SH fit to
the b=1500 signals), and the result goes through the same pipeline. The noise-
corrected value is J_f minus this null.

For comparison the submitted pairing is also computed, with its noise-free
Gaussian null (the mismatch artifact alone).

Region means for the 16 JHU regions go to out/jf_pass.csv.
Usage:  python jf_pass.py [n_sessions_for_test | --sample N]
"""
import sys
import time
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
from dipy.core.geometry import cart2sphere
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.reconst.shm import real_sh_descoteaux

from config import DTI, OUT, ROI_DEFS

ALL_LABELS = sorted({l for v in ROI_DEFS.values() for l in v})
ORDER = 6
N_COMMON = 100


def hemisphere(n):
    """Even directions on the upper hemisphere (ADC profiles are antipodal)."""
    k = np.arange(n) + 0.5
    z = 1 - k / n
    r = np.sqrt(1 - z * z)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([r * np.cos(th), r * np.sin(th), z], 1)


G_COMMON = hemisphere(N_COMMON)


def basis(g, order):
    _, th, ph = cart2sphere(*np.asarray(g).T)
    return real_sh_descoteaux(order, th, ph, legacy=False)[0]


def jgap(v):
    lv = np.log(np.maximum(v, 1e-9))
    return np.maximum(np.log(np.maximum(v.mean(-1), 1e-9)) - lv.mean(-1), 0)


def directional(S, S0, b):
    return -np.log(np.clip(S / S0[:, None], 1e-2, 1.0)) / b


def one(sid):
    t0 = time.time()
    try:
        proc = DTI / sid / "processed"
        bvals = np.loadtxt(DTI / sid / "inputs" / "dwi.bval").ravel()
        bvecs = np.loadtxt(proc / "dwi_raw.bvec")
        if bvecs.shape[0] != 3:
            bvecs = bvecs.T
        lab = np.asarray(nib.load(proc / "atlas" / "jhu_labels_registered.nii.gz").dataobj)
        brain = np.asarray(nib.load(proc / "mask.nii.gz").dataobj) > 0
        U = np.isin(lab, ALL_LABELS) & brain
        dwi = np.asarray(nib.load(proc / "dwi_raw.nii.gz").dataobj, dtype=np.float32)
        X = dwi[U].astype(np.float64)
        del dwi
        labU = lab[U]
        is0 = bvals < 50
        S0 = X[:, is0].mean(-1)
        nb0 = int(is0.sum())
        i1 = np.where(np.abs(bvals - 1500) < 100)[0]
        i3 = np.where(np.abs(bvals - 3000) < 100)[0]
        g1, g3 = bvecs[:, i1].T, bvecs[:, i3].T
        b1, b3 = float(bvals[i1].mean()), float(bvals[i3].mean())

        B1, B3, Bc = basis(g1, ORDER), basis(g3, ORDER), basis(G_COMMON, ORDER)
        P1 = np.linalg.pinv(B1).T @ Bc.T            # samples on g1 -> common set
        P3 = np.linalg.pinv(B3).T @ Bc.T
        res = X[:, i1] - X[:, i1] @ np.linalg.pinv(B1).T @ B1.T
        sig = np.sqrt((res ** 2).sum(-1) / (len(i1) - B1.shape[1]))
        ok = (S0 > 0) & (sig > 0)

        def jf_common(S1, S3, s0):
            D1c = directional(S1, s0, b1) @ P1
            D3c = directional(S3, s0, b3) @ P3
            return jgap(np.maximum(D3c, 1e-6) / np.maximum(D1c, 1e-6))

        # pairing used in the submitted version
        C = np.abs(g1 @ g3.T)
        j3 = C.argmax(1)
        j1 = C.argmax(0)
        pr = np.array([(a, j3[a]) for a in range(len(i1))
                       if j1[j3[a]] == a and C[a, j3[a]] > 0.99])

        def jf_paired(S1, S3, s0):
            D1 = directional(S1, s0, b1)[:, pr[:, 0]]
            D3 = directional(S3, s0, b3)[:, pr[:, 1]]
            return jgap(D3 / np.maximum(D1, 1e-6))

        jf_new = jf_common(X[:, i1], X[:, i3], S0)
        jf_old = jf_paired(X[:, i1], X[:, i3], S0)

        keep = is0 | (np.abs(bvals - 1500) < 100)
        fit = TensorModel(gradient_table(bvals[keep], bvecs=bvecs[:, keep].T,
                                         b0_threshold=50)).fit(X[:, keep])
        Q = fit.quadratic_form
        Dt1 = np.maximum(np.einsum("ij,njk,ik->ni", g1, Q, g1), 0)
        Dt3 = np.maximum(np.einsum("ij,njk,ik->ni", g3, Q, g3), 0)
        clean1 = S0[:, None] * np.exp(-b1 * Dt1)
        clean3 = S0[:, None] * np.exp(-b3 * Dt3)
        mismatch_old = jgap(Dt3[:, pr[:, 1]] / np.maximum(Dt1[:, pr[:, 0]], 1e-9))

        rng = np.random.default_rng(int(sid.replace("session_", "").replace("_", "")) + 7)

        def rician(c):
            return np.hypot(c + rng.normal(0, 1, c.shape) * sig[:, None],
                            rng.normal(0, 1, c.shape) * sig[:, None])

        null_new = np.zeros_like(jf_new)
        null_old = np.zeros_like(jf_old)
        for _ in range(2):
            s0n = np.maximum(S0 + rng.normal(0, 1, S0.shape) * sig / np.sqrt(nb0), 1e-6)
            n1, n3 = rician(clean1), rician(clean3)
            null_new += jf_common(n1, n3, s0n) / 2
            null_old += jf_paired(n1, n3, s0n) / 2

        per = {"Jf_new": jf_new, "Jf_new_null": null_new, "Jf_old": jf_old,
               "Jf_old_null": null_old, "Jf_old_mismatch": mismatch_old}
        rows = []
        for roi, ls in ROI_DEFS.items():
            mk = np.isin(labU, ls) & ok
            row = {"Session_ID": sid, "ROI": roi, "n_voxels": int(mk.sum()),
                   "n_pairs_old": len(pr)}
            for k, v in per.items():
                vv = v[mk]
                vv = vv[np.isfinite(vv)]
                row[k] = float(vv.mean()) if vv.size else np.nan
            rows.append(row)
        for r in rows:
            r["seconds"] = time.time() - t0
        return rows
    except Exception as e:
        return [{"Session_ID": sid, "error": repr(e)}]


if __name__ == "__main__":
    sessions = pd.read_csv(OUT / "v1_cohort_sessions.csv").Session_ID.tolist()
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        res = [one(s) for s in sessions[: int(sys.argv[1])]]
        pd.DataFrame([r for rows in res for r in rows]).to_csv(OUT / "jf_pass_test.csv",
                                                               index=False)
        print(pd.DataFrame([r for rows in res for r in rows]).query("ROI == 'CR_all'")
              .round(4).to_string())
        sys.exit()
    if "--sample" in sys.argv:          # random subset, for a decisive first look
        n = int(sys.argv[sys.argv.index("--sample") + 1])
        sessions = list(np.random.default_rng(1).choice(sessions, n, replace=False))
    part = OUT / "jf_pass_partial.csv"
    done = set(pd.read_csv(part).Session_ID) if part.exists() else set()
    todo = [s for s in sessions if s not in done]
    print(f"{len(todo)} sessions to process ({len(done)} already done)", flush=True)
    with Pool(6) as pool:
        for i, rows in enumerate(pool.imap_unordered(one, todo, chunksize=2), 1):
            dest = OUT / "jf_pass_errors.csv" if "error" in rows[0] else part
            pd.DataFrame(rows).to_csv(dest, mode="a", header=not dest.exists(), index=False)
            if i % 50 == 0:
                print(f"{i}/{len(todo)} at {time.strftime('%H:%M:%S')}", flush=True)
    df = pd.read_csv(part)
    df.to_csv(OUT / "jf_pass.csv", index=False)
    print(f"wrote jf_pass.csv: {df.Session_ID.nunique()} sessions")
