"""Check of the two null profiles of floor_null_pass.py on simulated data with no true change.

For each of the first N cohort sessions, each voxel's tensor fitted to b=0 plus b=1500
predicts both shells on the 93 b=1500 directions, with its b=3000 diffusivities scaled by
the voxel's measured ADC_1(3000)/ADC_1(1500) (clipped 0.5 to 1.2). The true Delta_b J_ln
of these signals is zero. Noise is added (the mean of four Rician images, each with SD
2 sigma, sigma from the residual of an order-6 SH fit to the measured b=1500 signals), and
the result is treated as data. The two nulls are then built from the simulated data
exactly as floor_null_pass.py builds them from real data, at the true b=3000 level (no
calibration), each with fresh noise of the same kind:
  T  the tensor fitted to the simulated b=0 and b=1500 images
  D  the simulated b=1500 directional diffusivities, which carry that shell's noise and floor
The excess of the data's change over a null's should be zero. Exact and split-half
estimators (split_half_pass.measures, SH order 6), region means, then cohort means as % of
the simulated data's b=1500 gap. Because the nulls' bias may depend on the signal-to-noise
ratio, which changes with age, the sex-adjusted partial r of each session's spurious excess
with age is also given, with the region's mean b=0 signal-to-noise ratio S0/sigma.

Bias correction. For the simulated sessions, each session's spurious excess is subtracted
from its real excess over the nulls of floor_null_pass.py (out/floor_null.csv): over Ta4c and
Da4c, whose noise matches the simulation's, and over TaNc and DaNc. The corrected mean excess
and its r with age are compared with the uncorrected ones.

Writes out/floor_null_sim.csv (participant-level, not published) and out/floor_null_sim.txt.
Usage:  python floor_null_sim.py [n_sessions]          the first n cohort sessions (default 30)
        python floor_null_sim.py --sample N            a random sample of N (seed 0)
        python floor_null_sim.py --sample N --stored   the same report from the stored
                                                       out/floor_null_sim.csv, without simulating
"""
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel

from config import DTI, OUT, ROI_DEFS
from floor_null_pass import MASKS_D, magnitude
from split_half_pass import ALL_LABELS, adc, basis, measures

ROIS = ["CR_all", "ACR", "SCR", "PCR", "Genu", "Body", "Splenium", "alic", "plic", "rlic", "ec",
        "slf", "ptr", "ss", "cingulum_cc", "tapetum"]


def one(sid):
    proc = DTI / sid / "processed"
    bvals = np.loadtxt(DTI / sid / "inputs" / "dwi.bval").ravel()
    bvecs = np.loadtxt(proc / "dwi_raw.bvec")
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T
    lab = np.asarray(nib.load(proc / "atlas" / "jhu_labels_registered.nii.gz").dataobj).round().astype(int)
    brain = np.asarray(nib.load(proc / "mask.nii.gz").dataobj) > 0
    U = np.isin(lab, ALL_LABELS) & brain
    X = np.asarray(nib.load(proc / "dwi_raw.nii.gz").dataobj, dtype=np.float32)[U].astype(np.float64)
    labU = lab[U]
    i0 = np.where(bvals < 50)[0]
    s1, s3 = np.where(np.abs(bvals - 1500) < 100)[0], np.where(np.abs(bvals - 3000) < 100)[0]
    g1, b1, b3 = bvecs[:, s1].T, float(bvals[s1].mean()), float(bvals[s3].mean())
    S0 = X[:, i0].mean(-1)
    B = basis(g1, 6)
    R = np.eye(len(s1)) - np.linalg.pinv(B).T @ B.T
    sig = np.sqrt(((X[:, s1] @ R) ** 2).sum(-1) / (len(s1) - B.shape[1]))[:, None]
    r = np.clip(adc(X[:, s3], S0, b3).mean(-1) / np.maximum(adc(X[:, s1], S0, b1).mean(-1), 1e-6), 0.5, 1.2)
    n0, n1 = len(i0), len(s1)
    j0, j1 = np.arange(n0), n0 + np.arange(n1)
    sh, g, b = {1500: j1, 3000: n0 + n1 + np.arange(n1)}, {1500: g1, 3000: g1}, {1500: b1, 3000: b3}
    lay_b = np.concatenate([np.zeros(n0), np.full(n1, b1)])
    lay_g = np.concatenate([np.zeros((n0, 3)), g1])

    def tensor(Y):
        Q = TensorModel(gradient_table(lay_b, bvecs=lay_g, b0_threshold=50)).fit(Y[:, :n0 + n1]).quadratic_form
        return np.maximum(np.einsum("ij,njk,ik->ni", g1, Q, g1), 0)

    def signals(D, s0):
        return np.concatenate([np.repeat(s0[:, None], n0, 1), s0[:, None] * np.exp(-b1 * D),
                               s0[:, None] * np.exp(-b3 * r[:, None] * D)], 1)

    rng = np.random.default_rng(int(sid.replace("session_", "").replace("_", "")) + 83)
    Y0 = np.concatenate([X[:, i0], X[:, s1]], 1)
    data = magnitude(signals(tensor(Y0), S0), sig, rng, "a4", None)
    S0d = data[:, j0].mean(-1)
    nulls = {"T": magnitude(signals(tensor(data), S0d), sig, rng, "a4", None),
             "D": magnitude(signals(adc(data[:, j1], S0d, b1), S0d), sig, rng, "a4", None)}
    per = {k: measures(v, j0, sh, g, b, "", MASKS_D) for k, v in {"data": data, **nulls}.items()}
    ok = S0 > 0
    rows = []
    for roi in ROIS:
        mk = np.isin(labU, ROI_DEFS[roi]) & ok
        row = {"Session_ID": sid, "ROI": roi, "snr0": np.nanmean(S0[mk] / sig[mk, 0])}
        for est, key in (("exact", "exact"), ("split", "split_o6")):
            row[f"J1_{est}"] = np.nanmean(per["data"][f"Jln1500_{key}"][mk])
            for k, v in per.items():
                row[f"{k}_{est}"] = np.nanmean(v[f"Jln3000_{key}"][mk]) - np.nanmean(v[f"Jln1500_{key}"][mk])
        rows.append(row)
    return rows


def partial(y, x, covs):
    X = np.column_stack([np.ones(len(y))] + covs)
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    r = np.corrcoef(ry, rx)[0, 1]
    df = len(y) - X.shape[1] - 1
    from scipy import stats
    return r, 2 * stats.t.sf(abs(r) * np.sqrt(df / (1 - r * r)), df)


if __name__ == "__main__":
    coh = pd.read_csv(OUT / "v1_cohort_sessions.csv")
    if "--sample" in sys.argv:
        n = int(sys.argv[sys.argv.index("--sample") + 1])
        sessions = sorted(np.random.default_rng(0).choice(coh.Session_ID, n, replace=False))
        label = f"a random sample of {n} cohort sessions (seed 0)"
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
        sessions = coh.Session_ID.tolist()[:n]
        label = f"first {n} cohort sessions"
    if "--stored" in sys.argv:
        d = pd.read_csv(OUT / "floor_null_sim.csv")
        assert sorted(d.Session_ID.unique()) == sorted(sessions), "stored sessions differ from the request"
    else:
        with Pool(8) as pool:
            d = pd.DataFrame([r for rows in pool.map(one, sessions) for r in rows])
        d.to_csv(OUT / "floor_null_sim.csv", index=False)
    g = d.groupby("ROI").mean(numeric_only=True).loc[ROIS]
    T = pd.DataFrame(index=g.index)
    for est in ("exact", "split"):
        for k in ("data", "T", "D"):
            T[f"{k}_{est}"] = 100 * g[f"{k}_{est}"] / g[f"J1_{est}"]
        for k in ("T", "D"):
            T[f"excess_{k}_{est}"] = T[f"data_{est}"] - T[f"{k}_{est}"]
    cb = ["Genu", "Body", "Splenium"]
    L = [f"Simulated data with no true change, {label}; change of the diffusivity gap as % of "
         "the simulated b=1500 gap; excess = data minus null (should be 0)", "", T.round(2).to_string(), ""]
    e = d.merge(coh, on="Session_ID")
    e["male"] = (e.sex == "M").astype(float)
    A = []
    for roi in ROIS:
        q = e[e.ROI == roi]
        row = {"ROI": roi, "r_snr0_age": partial(q.snr0.values, q.age.values, [q.male.values])[0]}
        for est in ("exact", "split"):
            for k in ("T", "D"):
                ex = (q[f"data_{est}"] - q[f"{k}_{est}"]).values
                row[f"r_age_excess_{k}_{est}"], row[f"p_age_excess_{k}_{est}"] = \
                    partial(ex, q.age.values, [q.male.values])
        A.append(row)
    A = pd.DataFrame(A).set_index("ROI")
    L += ["Sex-adjusted partial r with age of each session's spurious excess (and of the b=0 SNR)",
          A.round(3).to_string(), ""]
    if (OUT / "floor_null.csv").exists():
        nl = {"Ta4c": "T", "TaNc": "T", "Da4c": "D", "DaNc": "D"}
        cols = ["Session_ID", "ROI", "Jln1500_exact", "Jln3000_exact"] + \
            [f"{t}_Jln{s}_exact" for t in nl for s in (1500, 3000)]
        f = e.merge(pd.read_csv(OUT / "floor_null.csv", usecols=cols), on=["Session_ID", "ROI"])
        C = []
        for roi in ROIS:
            q = f[f.ROI == roi]
            a, m = q.age.values, [q.male.values]
            row = {"ROI": roi}
            for t, p in nl.items():
                real = ((q.Jln3000_exact - q.Jln1500_exact)
                        - (q[f"{t}_Jln3000_exact"] - q[f"{t}_Jln1500_exact"])).values
                corr = real - (q.data_exact - q[f"{p}_exact"]).values
                row[f"{t}_r"], row[f"{t}_r_corr"] = partial(real, a, m)[0], partial(corr, a, m)[0]
                row[f"{t}_excess"] = 100 * real.mean() / q.Jln1500_exact.mean()
                row[f"{t}_excess_corr"] = 100 * corr.mean() / q.Jln1500_exact.mean()
            C.append(row)
        C = pd.DataFrame(C).set_index("ROI")
        L += ["Real excess (% of the real b=1500 gap) and its r with age, before and after subtracting each "
              "session's simulated spurious excess", C.round(2).to_string(), ""]
        for t in nl:
            dr = C[f"{t}_r_corr"] - C[f"{t}_r"]
            L.append(f"{t}: corona radiata excess {C.loc['CR_all', f'{t}_excess']:+.2f} -> "
                     f"{C.loc['CR_all', f'{t}_excess_corr']:+.2f} points, r with age "
                     f"{C.loc['CR_all', f'{t}_r']:+.2f} -> {C.loc['CR_all', f'{t}_r_corr']:+.2f}; "
                     f"change of r over the regions {dr.min():+.2f} to {dr.max():+.2f}")
        L.append("")
    for est in ("exact", "split"):
        L.append(f"{est}: excess over the tensor null {T[f'excess_T_{est}'].min():+.2f} to "
                 f"{T[f'excess_T_{est}'].max():+.2f}; over the directional null "
                 f"{T[f'excess_D_{est}'].min():+.2f} to {T[f'excess_D_{est}'].max():+.2f} (corona radiata "
                 f"{T.loc['CR_all', f'excess_D_{est}']:+.2f}, corpus callosum "
                 f"{T.loc[cb, f'excess_D_{est}'].min():+.2f} to {T.loc[cb, f'excess_D_{est}'].max():+.2f})")
    text = "\n".join(L)
    print(text)
    (OUT / "floor_null_sim.txt").write_text(text + "\n", encoding="utf-8")
