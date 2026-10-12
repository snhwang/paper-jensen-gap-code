"""One pass over the raw DWI of every first-visit session, in native space.

For each session and shell (b = 1500, 3000), restricted to the 16 JHU regions:

  measured   J_ln, J_lnS from the directional signals (as in the gap table)
  tensor     DIPY TensorModel (WLS) on the b=0 volumes plus the shell, the same
             fit as the original pipeline. FA, MD, AD, RD, and the gaps of the
             tensor-predicted directional ADCs g_i^T D g_i on the same directions
  tensor+noise
             the tensor-predicted signals with simulated noise added, then put
             through exactly the measured pipeline (S0 from noisy b=0 mean,
             clip to [1e-2, 1], logarithms). Diffusion-weighted signals get
             Rician noise. Two draws are averaged. Two noise estimates are used.
               sh  (primary) residual of a real, symmetric spherical-harmonic
                   fit of order 6 to the b=1500 signals, RSS/(N-28). It
                   measures the noise of the diffusion-weighted images directly.
               b0  variance over the 14 b=0 volumes. It also contains CSF
                   pulsation and other fluctuation that diffusion weighting
                   suppresses, so it is expected to overstate the noise.

The excess measured - (tensor+noise) is what the tensor model and noise do not
explain. Region means of every quantity are written to out/native_pass.csv.

Usage:  python native_pass.py [n_sessions_for_test]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.core.geometry import cart2sphere
from dipy.reconst.dti import TensorModel
from dipy.reconst.shm import real_sh_descoteaux

from config import DTI, OUT, ROI_DEFS

ALL_LABELS = sorted({l for v in ROI_DEFS.values() for l in v})
N_DRAWS = 2


def gaps(D, S):
    lnD = np.log(np.maximum(D, 1e-6))
    jln = np.maximum(np.log(np.maximum(D.mean(-1), 1e-6)) - lnD.mean(-1), 0)
    jls = np.maximum(np.log(S.mean(-1)) - np.log(S).mean(-1), 0)
    return jln, jls


def pipeline_gaps(S_dw, S0, b):
    """Measured pipeline: normalize, clip, directional ADC, gaps."""
    Sp = np.clip(S_dw / S0[:, None], 1e-2, 1.0)
    return gaps(-np.log(Sp) / b, Sp)


def sh_sigma2(S, g, order):
    """Per-voxel noise variance from the residual of an SH fit of given order."""
    _, th, ph = cart2sphere(*g.T)
    Bm = real_sh_descoteaux(order, th, ph, legacy=False)[0]
    res = S - S @ np.linalg.pinv(Bm).T @ Bm.T
    return (res ** 2).sum(-1) / (S.shape[1] - Bm.shape[1])


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
        X = dwi[U].astype(np.float64)                 # (n_vox, n_vol)
        del dwi
        labU = lab[U]
        is_b0 = bvals < 50
        b0 = X[:, is_b0]
        S0 = b0.mean(-1)
        var0 = b0.var(-1, ddof=1)
        sh15 = np.abs(bvals - 1500) < 100
        sh30 = np.abs(bvals - 3000) < 100
        var_sh = sh_sigma2(X[:, sh15], bvecs[:, sh15].T, 6)
        var_sh3 = sh_sigma2(X[:, sh30], bvecs[:, sh30].T, 8)
        ok = (S0 > 0) & (var0 > 0) & (var_sh > 0)
        rng = np.random.default_rng(int(sid.replace("session_", "").replace("_", "")))
        rows = []
        for target in (1500, 3000):
            sh = np.abs(bvals - target) < 100
            b = float(bvals[sh].mean())
            g = bvecs[:, sh].T
            S_dw = X[:, sh]
            jln_m, jls_m = pipeline_gaps(S_dw, S0, b)
            adc1 = (-np.log(np.clip(S_dw / S0[:, None], 1e-2, 1)) / b).mean(-1)

            keep = is_b0 | sh
            gt = gradient_table(bvals[keep], bvecs=bvecs[:, keep].T, b0_threshold=50)
            fit = TensorModel(gt).fit(X[:, keep])
            q = fit.quadratic_form
            Dt = np.maximum(np.einsum("ij,njk,ik->ni", g, q, g), 0)
            jln_t, jls_t = gaps(Dt, np.exp(-b * Dt))

            nb0 = int(is_b0.sum())
            clean = S0[:, None] * np.exp(-b * Dt)
            sim = {}
            for tag, var in (("TN", var_sh), ("TNb0", var0)):
                sig = np.sqrt(var)
                a_sum = np.zeros_like(jln_m)
                c_sum = np.zeros_like(jls_m)
                for _ in range(N_DRAWS):
                    S0n = S0 + rng.normal(0, 1, S0.shape) * sig / np.sqrt(nb0)
                    Sn = np.hypot(clean + rng.normal(0, 1, clean.shape) * sig[:, None],
                                  rng.normal(0, 1, clean.shape) * sig[:, None])
                    a, c = pipeline_gaps(Sn, np.maximum(S0n, 1e-6), b)
                    a_sum += a / N_DRAWS
                    c_sum += c / N_DRAWS
                sim[tag] = (a_sum, c_sum)

            per_vox = {"J_ln": jln_m, "J_lnS": jls_m, "J_ln_T": jln_t, "J_lnS_T": jls_t,
                       "J_ln_TN": sim["TN"][0], "J_lnS_TN": sim["TN"][1],
                       "J_ln_TNb0": sim["TNb0"][0], "J_lnS_TNb0": sim["TNb0"][1],
                       "FA": fit.fa,
                       "MD": fit.md, "AD": fit.ad, "RD": fit.rd, "ADC_1": adc1}
            for roi, ls in ROI_DEFS.items():
                mk = np.isin(labU, ls) & ok
                row = {"Session_ID": sid, "shell_bval": target, "ROI": roi,
                       "n_voxels": int(mk.sum()),
                       "SNR": float(S0[mk].mean() / np.sqrt(var_sh[mk].mean())),
                       "SNR_b0": float(S0[mk].mean() / np.sqrt(var0[mk].mean())),
                       "SNR_sh3000": float(S0[mk].mean() / np.sqrt(var_sh3[mk].mean()))}
                for k, v in per_vox.items():
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
    # --sessions FILE.csv --tag NAME  processes another session list into
    # out/native_pass_NAME.csv (used for the longitudinal sessions).
    tag = ""
    src = OUT / "v1_cohort_sessions.csv"
    if "--sessions" in sys.argv:
        src = Path(sys.argv[sys.argv.index("--sessions") + 1])
        tag = "_" + sys.argv[sys.argv.index("--tag") + 1]
    sessions = pd.read_csv(src).Session_ID.tolist()
    test = len(sys.argv) > 1 and sys.argv[1].isdigit()
    if test:
        sessions = sessions[: int(sys.argv[1])]
        res = [one(s) for s in sessions]
        name = "native_pass_test.csv"
    else:
        done = set()
        part = OUT / f"native_pass{tag}_partial.csv"
        if part.exists():
            done = set(pd.read_csv(part).Session_ID)
        todo = [s for s in sessions if s not in done]
        print(f"{len(todo)} sessions to process ({len(done)} already done)", flush=True)
        res = []
        with Pool(6) as pool:
            for i, rows in enumerate(pool.imap_unordered(one, todo, chunksize=2), 1):
                dest = OUT / f"native_pass{tag}_errors.csv" if "error" in rows[0] else part
                pd.DataFrame(rows).to_csv(dest, mode="a", header=not dest.exists(),
                                          index=False)
                if i % 50 == 0:
                    print(f"{i}/{len(todo)} at {time.strftime('%H:%M:%S')}", flush=True)
        name = f"native_pass{tag}.csv"
        res = [pd.read_csv(part).to_dict("records")]
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT / name, index=False)
    nerr = df["error"].notna().sum() if "error" in df else 0
    print(f"wrote {OUT / name}: {df.Session_ID.nunique()} sessions, {nerr} errors")
