"""Fourth native pass: noise-floor nulls compared, and the noise measures behind them.

The diffusivity gap falls from b=1500 to b=3000 (Section 4.4), and the rectified noise floor
of magnitude images lowers it too. A null with the measured noise and no true change measures
the floor's part. split_half_pass.py builds one kind: a tensor fitted to b=0 plus b=1500, with
its b=3000 diffusivities scaled by the voxel's measured ADC_1(3000)/ADC_1(1500). This pass
varies the three choices behind such a null, and records the noise measures themselves.

Profile.
  T  the tensor, as split_half_pass.py.
  D  the measured b=1500 directional diffusivities D_1(g_i), with no fit. The null's b=1500
     shell is S0 exp(-b D_1), its b=3000 shell S0 exp(-b r D_1) on the same 93 directions.
     Scaling leaves the gap unchanged, so the true Delta_b J_ln and J_f are exactly zero. The
     noise of the measured D_1 is shared by both shells of the null, so it cancels from
     Delta_b to second order, and from the split halves, whose halves hold independent parts.
Level at b=3000.
  u  r = the measured ADC_1(3000)/ADC_1(1500), clipped 0.5 to 1.2, as split_half_pass.py. The
     floor has already raised the measured b=3000 signal, so this level is too high, and the
     null's noise puts the floor on it a second time.
  c  calibrated: r is raised until the null's mean ADC at b=3000, after noise, equals the
     measured one (three fixed-point steps on the b=0 and b=3000 images, clipped 0.5 to 1.5,
     then a full draw).
Noise. Single raw HCP-A runs are Rician, and the released images average magnitudes (the AP
and PA runs combined after eddy, then resampled). The mean of such an average is the
single-image Rician mean at the raw-image sigma_raw, so sigma_raw fixes the floor at every
signal level, and n_eff = (sigma_raw / sigma)^2 fixes the spread. Five models:
  c4   noncentral chi from four channels, per-channel SD sigma_SH (split_half_pass's nullC_)
  a4   mean of four Rician images, sigma_raw = 2 sigma_SH (split_half_pass's nullA_)
  aN   weighted mean of five Rician images with sum(w^2) = 1/n_eff, n_eff from the session's
       ventricle floor (below), sigma_raw = sqrt(n_eff) sigma_SH
  aNv  as aN, with one sigma for the session from the ventricle floor itself (no fit)
  aNb  as aN, with sigma from the SD of the 14 b=0 images in each voxel (no fit, but it also
       holds pulsation and motion between volumes)
Nulls (profile and noise): T with c4, a4, aN, aNv, and D with c4, a4, aN, aNv, aNb. Each gives
an uncalibrated and a calibrated set of measures, prefixed for example "Tc4u_" and "Tc4c_",
with the calibrated level's raise over r ("r_raise") and its remaining error ("level_error").
The noise generator is seeded per session, so the pass is reproducible.

Noise measures. sigma per voxel from the residual of order-4, -6 and -8 SH fits to the
b=1500 signals (sigma_sh4, 6, 8) and from the 14 b=0 images (sigma_b0). Ventricle floor at
b=3000, in the voxels of split_half_pass.ventricle_voxels: per voxel the mean and SD over the
92 directions, medians over voxels. At zero signal an averaged Rician magnitude has mean
1.2533 sigma_raw and mean/SD 1.9131 sqrt(n_eff), so n_eff = (mean/SD / 1.9131)^2,
sigma_raw = mean / 1.2533 and sigma_vent = sigma_raw / sqrt(n_eff). Background: voxels more
than 15 voxels outside the brain mask and nonzero in every volume, with the ratio of their
b=0 to b=3000 mean (1 for pure noise) and their b=3000 mean/SD. Session values are given
against sigma_cr, the median sigma_SH of the eroded corona radiata.

Each null goes through split_half_pass.measures (exact, full-data and split-half gaps, J_f).
Region means, with the session's values, go to out/floor_null.csv (participant-level, not
published).
Usage:  python floor_null_pass.py [n_sessions_for_test]
"""
import sys
import time
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from scipy import ndimage

from config import DTI, OUT, ROI_DEFS
from split_half_pass import ALL_LABELS, MASKS, adc, basis, measures, ventricle_voxels

NULLS = [("T", "c4"), ("T", "a4"), ("T", "aN"), ("T", "aNv"),
         ("D", "c4"), ("D", "a4"), ("D", "aN"), ("D", "aNv"), ("D", "aNb")]
CAL_STEPS = 3
MASKS_D = {"b1500": MASKS["b1500"], "b3000": MASKS["b1500"]}


def avg_weights(n_eff):
    """Weights (a, b, b, b, b) of five images with sum 1 and sum of squares 1/n_eff."""
    n = min(max(n_eff, 1.0), 5.0)
    b = (8 - np.sqrt(max(64 - 80 * (1 - 1 / n), 0.0))) / 40
    return np.array([1 - 4 * b] + [b] * 4)


def magnitude(c, s, rng, model, w):
    """Noisy magnitude of the clean signal c, with the processed-image noise SD s."""
    if model == "c4":
        tot = np.zeros_like(c)
        for _ in range(4):
            tot += (c / 2 + s * rng.normal(size=c.shape)) ** 2 + (s * rng.normal(size=c.shape)) ** 2
        return np.sqrt(tot)
    if model == "a4":
        w = np.full(4, 0.25)
    sr = s / np.sqrt((w ** 2).sum())
    M = np.zeros_like(c)
    for wk in w:
        M += wk * np.hypot(c + sr * rng.normal(size=c.shape), sr * rng.normal(size=c.shape))
    return M


def background(dwi, brain, i0, sh, sig_cr):
    """Voxels far outside the head that are nonzero in every volume."""
    nz = np.ones(brain.shape, bool)
    for k in range(dwi.shape[-1]):
        nz &= dwi[..., k] != 0
    M = dwi[nz & ~ndimage.binary_dilation(brain, iterations=15)].astype(np.float64)
    if len(M) < 10:
        return {"bg_n_voxels": int(len(M)), "bg_b0_over_b3000": np.nan, "bg_mean_over_sd": np.nan,
                "bg_mean_over_sigma": np.nan}
    m3, s3 = M[:, sh[3000]].mean(1), M[:, sh[3000]].std(1, ddof=1)
    return {"bg_n_voxels": int(len(M)), "bg_b0_over_b3000": float(np.median(M[:, i0].mean(1) / m3)),
            "bg_mean_over_sd": float(np.median(m3 / s3)), "bg_mean_over_sigma": float(np.median(m3) / sig_cr)}


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
        n1 = len(sh[1500])

        def sigma(S, order):
            B = basis(g[1500], order)
            R = np.eye(n1) - np.linalg.pinv(B).T @ B.T
            return np.sqrt(((S @ R) ** 2).sum(-1) / (n1 - B.shape[1]))

        dwi = np.asarray(nib.load(proc / "dwi_raw.nii.gz").dataobj, dtype=np.float32)
        core = ndimage.binary_erosion(np.isin(lab, ROI_DEFS["CR_all"]) & brain, iterations=1)
        sig_cr = float(np.median(sigma(dwi[core][:, sh[1500]].astype(np.float64), 6)))
        V = ventricle_voxels(dwi, lab, brain, i0, sh)
        assert len(V) >= 10, "fewer than 10 ventricle voxels"
        mu, sd = V.mean(1), V.std(1, ddof=1)
        mos = float(np.median(mu / sd))
        n_eff = (mos / 1.9131) ** 2
        s_raw = float(np.median(mu)) / 1.2533
        s_vent = s_raw / np.sqrt(n_eff)
        ses = {"vent_n_voxels": int(len(V)), "vent_mean_over_sd": mos, "n_eff": n_eff,
               "vent_mean_over_sigma": s_raw * 1.2533 / sig_cr, "sigma_vent_over_sigma": s_vent / sig_cr,
               "sigma_cr": sig_cr, **background(dwi, brain, i0, sh, sig_cr)}
        X = dwi[U].astype(np.float64)
        del dwi
        labU = lab[U]
        S0 = X[:, i0].mean(-1)
        sig = {k: sigma(X[:, sh[1500]], k) for k in (4, 6, 8)}
        sig_b0 = X[:, i0].std(-1, ddof=1)
        ok = (S0 > 0) & (sig[6] > 0) & (X[:, i0[0::2]].mean(-1) > 0) & (X[:, i0[1::2]].mean(-1) > 0)
        per = {"sigma_sh4": sig[4], "sigma_sh6": sig[6], "sigma_sh8": sig[8], "sigma_b0": sig_b0, "S0": S0}

        keep = (bvals < 50) | (np.abs(bvals - 1500) < 100)
        Q = TensorModel(gradient_table(bvals[keep], bvecs=bvecs[:, keep].T,
                                       b0_threshold=50)).fit(X[:, keep]).quadratic_form
        Dt = {s: np.maximum(np.einsum("ij,njk,ik->ni", g[s], Q, g[s]), 0) for s in sh}
        D1 = adc(X[:, sh[1500]], S0, b[1500])
        adc1 = {s: adc(X[:, sh[s]], S0, b[s]).mean(-1) for s in sh}
        r = np.clip(adc1[3000] / np.maximum(adc1[1500], 1e-6), 0.5, 1.2)
        # column layouts: T on the acquired directions, D with both shells on the b=1500 directions
        lay = {"T": dict(n=X.shape[1], i0=i0, sh=sh, g=g, masks=MASKS),
               "D": dict(n=len(i0) + 2 * n1, i0=np.arange(len(i0)),
                         sh={1500: len(i0) + np.arange(n1), 3000: len(i0) + n1 + np.arange(n1)},
                         g={1500: g[1500], 3000: g[1500]}, masks=MASKS_D)}

        def clean(prof, rc):
            L = lay[prof]
            C = np.empty((len(X), L["n"]))
            C[:, L["i0"]] = S0[:, None]
            if prof == "T":
                C[:, L["sh"][1500]] = S0[:, None] * np.exp(-b[1500] * Dt[1500])
                C[:, L["sh"][3000]] = S0[:, None] * np.exp(-b[3000] * Dt[3000] * rc[:, None])
            else:
                C[:, L["sh"][1500]] = S0[:, None] * np.exp(-b[1500] * D1)
                C[:, L["sh"][3000]] = S0[:, None] * np.exp(-b[3000] * D1 * rc[:, None])
            return C

        rng = np.random.default_rng(int(sid.replace("session_", "").replace("_", "")) + 47)
        w = avg_weights(n_eff)
        for prof, model in NULLS:
            L = lay[prof]
            s_ = s_vent if model == "aNv" else (sig_b0 if model == "aNb" else sig[6])[:, None]
            tag = f"{prof}{model}"
            M = magnitude(clean(prof, r), s_, rng, model, w)
            per.update(measures(M, L["i0"], L["sh"], L["g"], b, f"{tag}u_", L["masks"]))
            cols = np.concatenate([L["i0"], L["sh"][3000]])
            k0, k3 = np.arange(len(L["i0"])), len(L["i0"]) + np.arange(len(L["sh"][3000]))
            M3, rc = M[:, cols], r.copy()
            for _ in range(CAL_STEPS):
                lvl = adc(M3[:, k3], M3[:, k0].mean(-1), b[3000]).mean(-1)
                rc = np.clip(rc * adc1[3000] / np.maximum(lvl, 1e-6), 0.5, 1.5)
                M3 = magnitude(clean(prof, rc)[:, cols], s_, rng, model, w)
            M = magnitude(clean(prof, rc), s_, rng, model, w)
            per.update(measures(M, L["i0"], L["sh"], L["g"], b, f"{tag}c_", L["masks"]))
            per[f"{tag}c_r_raise"] = rc / r - 1
            per[f"{tag}c_level_error"] = adc(M[:, L["sh"][3000]], M[:, L["i0"]].mean(-1), b[3000]).mean(-1) \
                / np.maximum(adc1[3000], 1e-6) - 1
        per.update(measures(X, i0, sh, g, b, ""))

        rows = []
        for roi, ls in ROI_DEFS.items():
            mk = np.isin(labU, ls) & ok
            row = {"Session_ID": sid, "ROI": roi, "n_voxels": int(mk.sum()), **ses}
            for k, v in per.items():
                vv = v[mk]
                vv = vv[np.isfinite(vv)]
                row[k] = float(vv.mean()) if vv.size else np.nan
            rows.append(row)
        for row in rows:
            row["seconds"] = time.time() - t0
        return rows
    except Exception as e:  # noqa: BLE001
        return [{"Session_ID": sid, "error": repr(e)}]


if __name__ == "__main__":
    sessions = pd.read_csv(OUT / "v1_cohort_sessions.csv").Session_ID.tolist()
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        df = pd.DataFrame([r for s in sessions[: int(sys.argv[1])] for r in one(s)])
        df.to_csv(OUT / "floor_null_test.csv", index=False)
        print(df[df.ROI == "CR_all"].T.to_string())
        sys.exit()
    part = OUT / "floor_null_partial.csv"
    done = set(pd.read_csv(part).Session_ID) if part.exists() else set()
    todo = [s for s in sessions if s not in done]
    print(f"{len(todo)} sessions to process ({len(done)} already done)", flush=True)
    with Pool(8) as pool:
        for i, rows in enumerate(pool.imap_unordered(one, todo, chunksize=2), 1):
            dest = OUT / "floor_null_errors.csv" if "error" in rows[0] else part
            pd.DataFrame(rows).to_csv(dest, mode="a", header=not dest.exists(), index=False)
            if i % 50 == 0:
                print(f"{i}/{len(todo)} at {time.strftime('%H:%M:%S')}", flush=True)
    df = pd.read_csv(part)
    df.to_csv(OUT / "floor_null.csv", index=False)
    print(f"wrote floor_null.csv: {df.Session_ID.nunique()} sessions")
