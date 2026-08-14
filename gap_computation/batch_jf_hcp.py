"""
Cohort-scale attenuation gap J_f for HCP-A sessions.

J_f is a fit-free non-Gaussianity index built from the SAME Jensen construction
as the paper's other gaps, but applied to the per-direction attenuation ratios

    f_i = D_i(b2) / D_i(b1)          (b1 = 1500, b2 = 3000)
    J_f = ln<f> - <ln f>   >= 0      (Jensen, exactly as J_ln and J_lnS)

Why this and not Delta_b J_ln = J_ln(b2) - J_ln(b1):
  Both are zero under direction-independent attenuation, so both test the
  b-flatness of Prop. 6.1. But Delta_b contrasts two already-AGGREGATED gaps
  (~0.040 and ~0.033 -> ~-0.007), and the two shells are ~0.97 correlated, so
  the shared anatomy cancels and the residual ~24% of variance fights unchanged
  noise. J_f contrasts PER DIRECTION first and aggregates second, which avoids
  the cancellation entirely. Single-subject split-half reliability:

      Delta_b J_ln   r = 0.694,  Spearman 0.443
      J_f            r = 0.771,  Spearman 0.822

Note <ln f> = ln ADC_0(b2) - ln ADC_0(b1) is recoverable from the existing
aggregate tables, but ln<f> is a mean of RATIOS and is not. That is precisely
the contrast-first term, and it is why this needs the raw DWI.

DIRECTION MATCHING is essential: the b1500 and b3000 shells do NOT share a
gradient ordering. Index-paired vectors sit ~67 deg apart, so pairing by index
compares different axes. Directions are matched by nearest |cos| (antipodal
equivalent) and only near-exact pairs (|cos| > 0.99) are kept.

Clamps, floors and ROI definitions mirror batch_gaps_hcp.py exactly so J_f is
consistent with the manuscript's other quantities.

Output: HCP/cr_jf_long.csv (one row per session x ROI). Resumable.

Usage:
  python batch_jf_hcp.py --limit 3          # quick test
  python batch_jf_hcp.py --workers 6
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
import shell_utils as fdp  # standalone shell clustering (extracted from fastapi_diffusion_processor)  # noqa: E402


def _resolve_output_dir():
    """First candidate path that actually contains sessions.

    Set DTI_OUTPUT_DIR to the directory holding the per-session preprocessing
    output (one subdirectory per session, each with metadata.json, inputs/ and
    processed/). Candidates are checked for metadata.json rather than mere
    existence, because a mount point can exist while being empty and would
    otherwise yield zero sessions with no error.
    """
    cands = [os.environ.get("DTI_OUTPUT_DIR"), "Q:/dti_output"]
    for c in cands:
        if c and glob.glob(str(Path(c) / "*" / "metadata.json")):
            return Path(c)
    return Path(cands[1])


OUTPUT_DIR = _resolve_output_dir()
OUT_CSV = BASE / "HCP" / "cr_jf_long.csv"

B1, B2 = 1500, 3000
D_FLOOR = 1e-6          # mirrors batch_gaps_hcp.py
SP_CLIP = (1e-2, 1.0)   # mirrors batch_gaps_hcp.py
COS_MIN = 0.99          # near-exact direction match
MIN_PAIRS = 20          # refuse a session that cannot be matched properly

# Must mirror ROI_DEFS in batch_gaps_hcp.py / batch_cr_metrics_hcp.py
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

# J_f is primary. Delta_b J_ln is recomputed here on the IDENTICAL voxels and
# clamps so the two can be compared without any masking confound.
JF_METRICS = ["J_f", "Delta_b_J_ln", "J_ln_b1", "J_ln_b2", "f_mean"]


def roi_stats(arr, mask):
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None, 0
    return float(np.mean(vals)), float(np.std(vals)), int(vals.size)


def _shell_D(dwi, bvals_s, sel, b0_mean):
    """Per-direction apparent diffusivity D_i for one shell."""
    with np.errstate(divide="ignore", invalid="ignore"):
        Sp = dwi[..., sel] / (b0_mean[..., np.newaxis] + 1e-10)
    Sp = np.clip(Sp, SP_CLIP[0], SP_CLIP[1]).astype(np.float32)
    b = float(np.mean(bvals_s[sel]))
    D = -(1.0 / b) * np.log(Sp)
    return np.clip(D, D_FLOOR, None)


def compute_session(session_id):
    """Return (session_id, rows, error). rows: list of dicts (one per ROI)."""
    proc = OUTPUT_DIR / session_id / "processed"
    inputs = OUTPUT_DIR / session_id / "inputs"
    dwi_path = proc / "dwi_raw.nii.gz"
    bval_path = inputs / "dwi.bval"
    bvec_path = inputs / "dwi.bvec"
    mask_path = proc / "mask.nii.gz"
    labels_path = proc / "atlas" / "jhu_labels_registered.nii.gz"
    for p in (dwi_path, bval_path, bvec_path, mask_path, labels_path):
        if not p.exists():
            return session_id, [], f"missing {p.name}"

    try:
        bvals = np.loadtxt(bval_path).astype(float)
        bvecs = np.loadtxt(bvec_path).astype(float)
        if bvecs.shape[0] == 3:
            bvecs = bvecs.T
        dwi = nib.load(str(dwi_path)).get_fdata().astype(np.float32)
        brain_mask = nib.load(str(mask_path)).get_fdata().astype(bool)
        labels = nib.load(str(labels_path)).get_fdata().astype(int)
    except Exception as e:
        return session_id, [], f"load: {type(e).__name__}: {e}"

    if labels.shape != brain_mask.shape or labels.shape != dwi.shape[:3]:
        return session_id, [], "shape mismatch (labels/mask/dwi)"
    if bvals.size != dwi.shape[-1]:
        return session_id, [], f"bval/dwi mismatch {bvals.size} vs {dwi.shape[-1]}"
    if bvecs.shape[0] != bvals.size:
        return session_id, [], f"bvec/bval mismatch {bvecs.shape[0]} vs {bvals.size}"

    shells = fdp.cluster_bvals_into_shells(bvals)
    if not shells:
        return session_id, [], "no shells"
    b0_mask = bvals < 50
    if not b0_mask.any():
        return session_id, [], "no b0 volumes"

    # locate both shells; J_f needs them together
    sel = {}
    for target in (B1, B2):
        nominal_b, shell_mask = min(shells, key=lambda s: abs(s[0] - target))
        if abs(nominal_b - target) > 400 or shell_mask.sum() < 6:
            return session_id, [], f"no usable b{target} shell"
        sel[target] = shell_mask
    if (sel[B1] & sel[B2]).any():
        return session_id, [], "shell masks overlap"

    b0_mean = np.mean(dwi[..., b0_mask], axis=-1)

    # The shells do NOT share a gradient ordering: pair by nearest |cos|.
    v1, v2 = bvecs[sel[B1]], bvecs[sel[B2]]
    n1 = np.linalg.norm(v1, axis=1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=1, keepdims=True)
    if (n1 < 0.5).any() or (n2 < 0.5).any():
        return session_id, [], "degenerate bvec in a shell"
    cos = np.abs((v1 / n1) @ (v2 / n2).T)
    j = cos.argmax(1)          # best b3000 partner for each b1500 direction
    back = cos.argmax(0)       # best b1500 partner for each b3000 direction
    idx = np.arange(len(v1))
    # Mutual nearest neighbour, not just nearest: a plain argmax lets two b1500
    # directions claim the same b3000 direction (3 of 72 do), which would
    # double-count it. Requiring the match to be reciprocal makes the pairing
    # one-to-one by construction.
    keep = (cos[idx, j] > COS_MIN) & (back[j] == idx)
    if keep.sum() < MIN_PAIRS:
        return session_id, [], f"only {int(keep.sum())} matched directions"

    D1 = _shell_D(dwi, bvals, sel[B1], b0_mean)[..., keep]
    D2 = _shell_D(dwi, bvals, sel[B2], b0_mean)[..., j[keep]]

    # --- primary: contrast per direction, THEN aggregate --------------------
    f = D2 / D1
    lnf = np.log(f)
    f_mean = f.mean(-1)
    J_f = np.maximum(np.log(np.clip(f_mean, 1e-12, None)) - lnf.mean(-1), 0.0)

    # --- comparator: the manuscript's Delta_b, same voxels and clamps -------
    def gap(D):
        lnD = np.log(D)
        return np.maximum(np.log(np.clip(D.mean(-1), 1e-12, None)) - lnD.mean(-1), 0.0)

    J1, J2 = gap(D1), gap(D2)

    maps = {"J_f": J_f, "Delta_b_J_ln": J2 - J1, "J_ln_b1": J1, "J_ln_b2": J2,
            "f_mean": f_mean}

    roi_masks = {name: np.isin(labels, lbls) & brain_mask for name, lbls in ROI_DEFS}
    rows = []
    for roi_name, _ in ROI_DEFS:
        m = roi_masks[roi_name]
        row = {"Session_ID": session_id, "ROI": roi_name,
               "n_voxels": int(m.sum()), "n_dirs": int(keep.sum())}
        for name in JF_METRICS:
            mu, sd, _ = roi_stats(maps[name], m)
            row[f"{name}_mean"] = "" if mu is None else f"{mu:.6g}"
            row[f"{name}_sd"] = "" if sd is None else f"{sd:.6g}"
        rows.append(row)

    del dwi, D1, D2, f, lnf, maps
    return session_id, rows, ""


def discover_sessions():
    """All HCA sessions on disk; map session -> (subject, visit)."""
    out = {}
    for mf in glob.glob(str(OUTPUT_DIR / "*" / "metadata.json")):
        sess = Path(mf).parent.name
        try:
            nm = (json.load(open(mf)).get("name") or "").strip()
        except Exception:
            nm = ""
        if "HCA" not in nm:
            continue
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

    fieldnames = ["Subject_ID", "Visit", "Session_ID", "ROI", "n_voxels", "n_dirs"]
    for m in JF_METRICS:
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

    print(f"\nDone. ok={n_ok} fail={n_fail} in {time.time()-t0:.0f}s -> {out_csv}",
          flush=True)


if __name__ == "__main__":
    main()
