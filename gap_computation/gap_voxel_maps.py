"""
Save voxelwise diffusivity-/signal-domain Holder & Jensen-gap maps for one
session and shell. Same math as batch_gaps_hcp.py, but writes NIfTI maps
instead of ROI means.

Used for (a) the single-subject parameter-map figure, and (b) as the per-subject
building block for the TBSS pass (run over the cohort, then project + TFCE).

Maps written (masked, float32, DWI affine):
  ADC_1        arithmetic mean of directional diffusivities (= clinical trace ADC)
  ADC_0        geometric mean of directional diffusivities  = exp<ln D>   (corrected)
  ADC_0_sig    -(1/b) ln<S'>  (the OLD signal-proxy previously mislabeled ADC_0)
  J_ln_D       ln(ADC_1/ADC_0) = ln<D> - <ln D>             (diffusivity gap)
  J_ln_S       ln<S'> - <ln S'>                              (signal gap, = pipeline J_dir)

Usage:
  python gap_voxel_maps.py <session_id> --bval 1500 --out-dir HCP/voxel_maps
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import shell_utils as fdp  # standalone shell clustering (extracted from fastapi_diffusion_processor)

OUTPUT_DIR = Path(os.environ.get("DTI_OUTPUT_DIR", str(Path("Q:/dti_output"))))
D_FLOOR = 1e-6
SP_CLIP = (1e-2, 1.0)


def compute_maps(session_id, target_bval):
    proc = OUTPUT_DIR / session_id / "processed"
    inputs = OUTPUT_DIR / session_id / "inputs"
    dwi_img = nib.load(str(proc / "dwi_raw.nii.gz"))
    dwi = dwi_img.get_fdata().astype(np.float32)
    bvals = np.loadtxt(inputs / "dwi.bval").astype(float)
    mask = nib.load(str(proc / "mask.nii.gz")).get_fdata().astype(bool)

    shells = fdp.cluster_bvals_into_shells(bvals)
    nominal_b, shell_mask = min(shells, key=lambda s: abs(s[0] - target_bval))
    if abs(nominal_b - target_bval) > 400:
        raise SystemExit(f"no shell near b={target_bval} (nearest {nominal_b})")
    b0_mask = bvals < 50
    keep = b0_mask | shell_mask
    dwi_s = dwi[..., keep]
    bvals_s = bvals[keep]
    shell_only = bvals_s >= 50
    b0_mean = np.mean(dwi_s[..., bvals_s < 50], axis=-1)
    shell_b = float(np.mean(bvals_s[shell_only]))

    with np.errstate(divide="ignore", invalid="ignore"):
        Sp = dwi_s[..., shell_only] / (b0_mean[..., np.newaxis] + 1e-10)
    Sp = np.clip(Sp, *SP_CLIP).astype(np.float32)
    lnSp = np.log(Sp)

    D = -(1.0 / shell_b) * lnSp
    Df = np.clip(D, D_FLOOR, None)
    lnD = np.log(Df)

    def slog(x):
        return np.log(np.clip(x, 1e-12, None))

    ADC_1 = D.mean(-1)
    ADC_0 = np.exp(lnD.mean(-1))
    ADC_0_sig = -(1.0 / shell_b) * slog(Sp.mean(-1))
    J_ln_D = np.maximum(slog(ADC_1) - lnD.mean(-1), 0.0)
    J_ln_S = np.maximum(slog(Sp.mean(-1)) - lnSp.mean(-1), 0.0)

    m = mask.astype(np.float32)
    maps = {
        "ADC_1": ADC_1 * m,
        "ADC_0": ADC_0 * m,
        "ADC_0_sig": ADC_0_sig * m,
        "J_ln_D": J_ln_D * m,
        "J_ln_S": J_ln_S * m,
    }
    return maps, dwi_img.affine, dwi_img.header, round(nominal_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--bval", type=int, required=True, choices=[1500, 3000])
    ap.add_argument("--out-dir", type=str, default="HCP/voxel_maps")
    args = ap.parse_args()

    maps, affine, header, nominal_b = compute_maps(args.session_id, args.bval)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h = header.copy()
    h.set_data_dtype(np.float32)
    for name, arr in maps.items():
        p = out_dir / f"{args.session_id}_{name}_b{args.bval}.nii.gz"
        nib.save(nib.Nifti1Image(arr.astype(np.float32), affine, h), str(p))
        print(f"  wrote {p}  (range {np.nanmin(arr):.4g}..{np.nanmax(arr):.4g})")
    print(f"done: shell b={nominal_b}, {len(maps)} maps -> {out_dir}")


if __name__ == "__main__":
    main()
