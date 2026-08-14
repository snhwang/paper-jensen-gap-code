"""Quantify how the directional Jensen gaps correlate with FA on the HCP-A
(n=1379) WM skeleton at b=1500.

Two complementary correlations, computed in one streaming pass over the 4D
skeletonised TBSS data (no full load):
  (1) spatial: group-mean gap map vs group-mean FA map across skeleton voxels;
  (2) across-subject: per-subject skeleton-mean gap vs skeleton-mean FA.

Diffusivity gap J_ln = ln(ADC_lin / ADC_geo) = ln(ADC_1/ADC_0). Signal gap is
the stored J_dir. Writes results to gap_fa_correlation_b1500.txt.
"""
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import stats

import os
DATA = Path(os.environ.get("JG_DATA", Path(__file__).resolve().parent))
S = DATA / "tbss_age_b1500_n1379" / "stats"

mask = nib.load(str(S / "mean_FA_skeleton_mask.nii.gz")).get_fdata() > 0
idx = np.where(mask)
nvox = int(mask.sum())

fa = nib.load(str(S / "all_FA_skeletonised.nii.gz"))
jd = nib.load(str(S / "all_J_dir_skeletonised.nii.gz"))
al = nib.load(str(S / "all_ADC_lin_skeletonised.nii.gz"))
ag = nib.load(str(S / "all_ADC_geo_skeletonised.nii.gz"))
nsub = fa.shape[3]

sum_fa = np.zeros(nvox); sum_jd = np.zeros(nvox); sum_jl = np.zeros(nvox)
m_fa = np.empty(nsub); m_jd = np.empty(nsub); m_jl = np.empty(nsub)

for i in range(nsub):
    vfa = np.asarray(fa.dataobj[..., i])[idx]
    vjd = np.asarray(jd.dataobj[..., i])[idx]
    vlin = np.asarray(al.dataobj[..., i])[idx]
    vgeo = np.asarray(ag.dataobj[..., i])[idx]
    vjl = np.log(np.clip(vlin, 1e-12, None) / np.clip(vgeo, 1e-12, None))
    sum_fa += vfa; sum_jd += vjd; sum_jl += vjl
    m_fa[i] = vfa.mean(); m_jd[i] = vjd.mean(); m_jl[i] = vjl.mean()
    if i % 200 == 0:
        print(f"  {i}/{nsub}", flush=True)

mean_fa = sum_fa / nsub; mean_jd = sum_jd / nsub; mean_jl = sum_jl / nsub

# valid voxels (finite, FA in plausible range)
ok = np.isfinite(mean_fa) & np.isfinite(mean_jd) & np.isfinite(mean_jl) & (mean_fa > 0)

lines = []
lines.append(f"HCP-A n={nsub}, skeleton voxels={nvox} (valid={int(ok.sum())}), b=1500")
lines.append("")
lines.append("(1) SPATIAL  (group-mean gap vs group-mean FA, across skeleton voxels)")
for name, vec in [("J_ln (diffusivity)", mean_jl), ("J_dir (signal)", mean_jd)]:
    r, p = stats.pearsonr(vec[ok], mean_fa[ok])
    rho, _ = stats.spearmanr(vec[ok], mean_fa[ok])
    lines.append(f"    {name:20s} vs FA:  r={r:+.3f}  rho={rho:+.3f}  (n={int(ok.sum())} voxels)")
lines.append("")
lines.append("(2) ACROSS-SUBJECT  (per-subject skeleton-mean gap vs skeleton-mean FA)")
for name, vec in [("J_ln (diffusivity)", m_jl), ("J_dir (signal)", m_jd)]:
    r, p = stats.pearsonr(vec, m_fa)
    lines.append(f"    {name:20s} vs FA:  r={r:+.3f}  p={p:.2e}  (n={nsub} subjects)")

out = "\n".join(lines)
print(out)
(DATA / "gap_fa_correlation_b1500.txt").write_text(out)
print("\nwrote gap_fa_correlation_b1500.txt")
