"""
Lightweight local pass: Hölder-mean ladder + Jensen gaps (diffusivity & signal
domains) for HCP-A sessions.

This does NOT redo FNIRT / SH / needlet.  It reuses each session's already-cached
JHU atlas registration (processed/atlas/jhu_labels_registered.nii.gz) and only
loads the DWI to form directional diffusivities, so it runs entirely locally and
fast (I/O-bound on Q:).

Per session and per shell (b1500, b3000) it computes, per voxel:
  Signal-domain Holder means of S'_i = S_i/S0:
    M0_S (geom), M1_S (arith), M2_S (quad), Mm1_S (harmonic)
  Diffusivity-domain Holder means of D_i = -(1/b) ln S'_i  (= ADC_r):
    ADC_0 (geom), ADC_1 (arith), ADC_2 (quad), ADC_m1 (harmonic)
  Gaps (all >= 0 by Holder/Jensen):
    J_ln_D   = ln(ADC_1/ADC_0)          diffusivity arith/geom  (the "pure" gap)
    J_quad_D = ln(ADC_2/ADC_1)          diffusivity quad/arith
    J_harm_D = ln(ADC_1/ADC_m1)         diffusivity arith/harm
    J_ln_S   = ln(M1_S/M0_S)            signal arith/geom  (== pipeline J_dir)
    J_quad_S = ln(M2_S/M1_S)            signal quad/arith
then ROI means over the 16 JHU ROIs.

Output: HCP/cr_gaps_long.csv  (one row per session x shell x ROI).
Resumable: sessions already present in the output CSV are skipped.

Usage:
  python batch_gaps_hcp.py --workers 6
  python batch_gaps_hcp.py --limit 3          # quick test
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import nibabel as nib

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import shell_utils as fdp  # standalone shell clustering (extracted from fastapi_diffusion_processor)

OUTPUT_DIR = Path(os.environ.get("DTI_OUTPUT_DIR", str(Path("Q:/dti_output"))))
OUT_CSV = BASE / "HCP" / "cr_gaps_long.csv"
TARGET_BVALS = [1500, 3000]
D_FLOOR = 1e-6      # floor on directional diffusivity for log/reciprocal means
# clamp the normalized signal S' = S/S0 to (0, 1]; the upper bound enforces
# D_i = -(1/b) ln S' >= 0 (introduces a small upward bias on the gaps at high b;
# see the paper's Limitations)
SP_CLIP = (1e-2, 1.0)

# Must mirror ROI_DEFS in batch_cr_metrics_hcp.py / extract_16roi_skel.py
ROI_DEFS = [
    ("CR_all",      [23, 24, 25, 26, 27, 28]),
    ("ACR",         [23, 24]),
    ("SCR",         [25, 26]),
    ("PCR",         [27, 28]),
    ("Genu",        [3]),
    ("Body",        [4]),
    ("Splenium",    [5]),
    ("alic",        [17, 18]),
    ("plic",        [19, 20]),
    ("rlic",        [21, 22]),
    ("ec",          [33, 34]),
    ("slf",         [41, 42]),
    ("ptr",         [29, 30]),
    ("ss",          [31, 32]),
    ("cingulum_cc", [35, 36]),
    ("tapetum",     [47, 48]),
]

GAP_METRICS = [
    "ADC_min", "ADC_m1", "ADC_0", "ADC_1", "ADC_2", "ADC_max",
    "M_min_S", "Mm1_S", "M0_S", "M1_S", "M2_S", "M_max_S",
    "J_ln_D", "J_quad_D", "J_harm_D",
    "J_ln_S", "J_quad_S",
]


def roi_stats(arr, mask):
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), int(vals.size)


def compute_session(session_id):
    """Return (session_id, rows, error). rows: list of dicts (one per shell x ROI)."""
    proc = OUTPUT_DIR / session_id / "processed"
    inputs = OUTPUT_DIR / session_id / "inputs"
    dwi_path = proc / "dwi_raw.nii.gz"
    bval_path = inputs / "dwi.bval"
    mask_path = proc / "mask.nii.gz"
    labels_path = proc / "atlas" / "jhu_labels_registered.nii.gz"
    for p in (dwi_path, bval_path, mask_path, labels_path):
        if not p.exists():
            return session_id, [], f"missing {p.name}"

    try:
        bvals = np.loadtxt(bval_path).astype(float)
        dwi = nib.load(str(dwi_path)).get_fdata().astype(np.float32)
        brain_mask = nib.load(str(mask_path)).get_fdata().astype(bool)
        labels = nib.load(str(labels_path)).get_fdata().astype(int)
    except Exception as e:
        return session_id, [], f"load: {type(e).__name__}: {e}"

    if labels.shape != brain_mask.shape or labels.shape != dwi.shape[:3]:
        return session_id, [], "shape mismatch (labels/mask/dwi)"
    if bvals.size != dwi.shape[-1]:
        return session_id, [], f"bval/dwi mismatch {bvals.size} vs {dwi.shape[-1]}"

    roi_masks = {name: np.isin(labels, lbls) & brain_mask for name, lbls in ROI_DEFS}

    shells = fdp.cluster_bvals_into_shells(bvals)
    if not shells:
        return session_id, [], "no shells"
    b0_mask = bvals < 50
    if not b0_mask.any():
        return session_id, [], "no b0 volumes"

    rows = []
    for target in TARGET_BVALS:
        best = min(shells, key=lambda s: abs(s[0] - target))
        nominal_b, shell_mask = best
        # require the chosen shell to actually be near the target
        if abs(nominal_b - target) > 400:
            continue
        keep = b0_mask | shell_mask
        dwi_s = dwi[..., keep]
        bvals_s = bvals[keep]
        shell_only = bvals_s >= 50
        if shell_only.sum() < 6:
            continue
        b0_mean = np.mean(dwi_s[..., bvals_s < 50], axis=-1)
        shell_b = float(np.mean(bvals_s[shell_only]))

        with np.errstate(divide="ignore", invalid="ignore"):
            Sp = dwi_s[..., shell_only] / (b0_mean[..., np.newaxis] + 1e-10)
        Sp = np.clip(Sp, SP_CLIP[0], SP_CLIP[1]).astype(np.float32)
        lnSp = np.log(Sp)

        # signal-domain Holder means
        M1_S = Sp.mean(-1)
        M0_S = np.exp(lnSp.mean(-1))
        M2_S = np.sqrt((Sp ** 2).mean(-1))
        Mm1_S = 1.0 / np.clip((1.0 / Sp).mean(-1), 1e-12, None)

        # diffusivity-domain
        D = -(1.0 / shell_b) * lnSp                       # >= 0
        Df = np.clip(D, D_FLOOR, None)
        lnD = np.log(Df)
        ADC_1 = D.mean(-1)
        ADC_0 = np.exp(lnD.mean(-1))
        ADC_2 = np.sqrt((D ** 2).mean(-1))
        ADC_m1 = 1.0 / np.clip((1.0 / Df).mean(-1), 1e-12, None)

        def safelog(x):
            return np.log(np.clip(x, 1e-12, None))

        maps = {
            "ADC_min": D.min(-1), "ADC_m1": ADC_m1, "ADC_0": ADC_0,
            "ADC_1": ADC_1, "ADC_2": ADC_2, "ADC_max": D.max(-1),
            "M_min_S": Sp.min(-1), "Mm1_S": Mm1_S, "M0_S": M0_S,
            "M1_S": M1_S, "M2_S": M2_S, "M_max_S": Sp.max(-1),
            # Jensen gaps = ln(arithmetic mean) - mean(ln); >= 0 by Jensen, clamp for roundoff
            "J_ln_D":   np.maximum(safelog(ADC_1) - lnD.mean(-1), 0.0),   # ln(ADC_1/ADC_0) = ln<D> - <ln D>
            "J_quad_D": np.maximum(safelog(ADC_2) - safelog(ADC_1), 0.0),
            "J_harm_D": np.maximum(safelog(ADC_1) - safelog(ADC_m1), 0.0),
            "J_ln_S":   np.maximum(safelog(M1_S) - lnSp.mean(-1), 0.0),   # ln<S'> - <ln S'>  (signal-domain gap)
            "J_quad_S": np.maximum(safelog(M2_S) - safelog(M1_S), 0.0),
        }

        for roi_name, _ in ROI_DEFS:
            m = roi_masks[roi_name]
            row = {"Session_ID": session_id, "ROI": roi_name,
                   "shell_bval": round(nominal_b), "n_voxels": int(m.sum())}
            for name in GAP_METRICS:
                mu, sd, _ = roi_stats(maps[name], m)
                row[f"{name}_mean"] = "" if mu is None else f"{mu:.6g}"
                row[f"{name}_sd"] = "" if sd is None else f"{sd:.6g}"
            rows.append(row)

        del dwi_s, Sp, lnSp, D, Df, lnD, maps
    return session_id, rows, ""


def discover_sessions():
    """All HCA sessions on disk with the 4 required inputs; map session->(sub,visit)."""
    out = {}
    for mf in glob.glob(str(OUTPUT_DIR / "*" / "metadata.json")):
        sess = Path(mf).parent.name
        try:
            nm = (json.load(open(mf)).get("name") or "").strip()
        except Exception:
            nm = ""
        if "HCA" not in nm:
            continue
        # name like HCA######_V#_MR
        sub = visit = ""
        parts = nm.split("_")
        if parts and parts[0].startswith("HCA"):
            sub = parts[0]
            for p in parts:
                if p.startswith("V") and p[1:].isdigit():
                    visit = p
                    break
        out[sess] = (sub, visit)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out_csv = Path(args.out) if args.out else OUT_CSV

    sess_map = discover_sessions()
    all_sessions = sorted(sess_map)

    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                done.add(r["Session_ID"])
    todo = [s for s in all_sessions if s not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"sessions on disk: {len(all_sessions)}, already done: {len(done)}, "
          f"todo: {len(todo)}, workers: {args.workers}", flush=True)
    if not todo:
        return

    fieldnames = ["Subject_ID", "Visit", "Session_ID", "ROI", "shell_bval", "n_voxels"]
    for m in GAP_METRICS:
        fieldnames += [f"{m}_mean", f"{m}_sd"]

    write_header = not out_csv.exists()
    t0 = time.time()
    n_ok = n_fail = 0
    with open(out_csv, "a", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        with Pool(args.workers) as pool:
            for i, (session_id, rows, err) in enumerate(
                    pool.imap_unordered(compute_session, todo), 1):
                if err:
                    n_fail += 1
                    print(f"  [{i}/{len(todo)}] FAIL {session_id}: {err}", flush=True)
                    continue
                sub, visit = sess_map.get(session_id, ("", ""))
                for row in rows:
                    row["Subject_ID"] = sub
                    row["Visit"] = visit
                    w.writerow(row)
                fout.flush()
                n_ok += 1
                if i % 25 == 0 or i == len(todo):
                    el = time.time() - t0
                    rate = i / el if el else 0
                    eta = (len(todo) - i) / rate if rate else 0
                    print(f"  [{i}/{len(todo)}] ok={n_ok} fail={n_fail} "
                          f"elapsed={el:.0f}s rate={rate:.2f}/s eta={eta:.0f}s", flush=True)

    print(f"\nDone. ok={n_ok} fail={n_fail} in {time.time()-t0:.0f}s -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
