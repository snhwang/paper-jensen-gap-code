#!/usr/bin/env bash
# Beyond-tensor TBSS launcher: does the metric carry an age effect the diffusion
# tensor cannot account for?
#
#   1. residualize all_<metric>_skeletonised on [1,FA,MD,AD,RD] voxelwise
#      (make_resid_tensor.py) over skeleton_good_mask
#   2. randomise the age contrast on the residual, full 10-EV design
#      (design_full: age + age^2 + sex + age*sex + site(3) + motion + outlier;
#       design.con tstat1 = +age, tstat2 = -age), TFCE, fixed seed
#   3. localize_beyondtensor.py -> FWE voxel counts / % usable / partial-r / JHU
#
# Metrics: J_ln_D (diffusivity gap), J_dir (signal gap, = J_ln_S), FA (control).
# The identical test is applied to the info-theory descriptors in the companion
# paper using the same skeleton_good_mask, so results are directly comparable.
#
# NOTE: FSLDIR is set BEFORE sourcing fslconf/fsl.sh -- that script references
# FSLDIR before defining it, so sourcing it under `set -u` without FSLDIR set
# aborts before the first echo and leaves a 0-byte log.
#
# Usage:  dgx_randomise_beyondtensor.sh <bval> <metric>
#         NPERM=5000 dgx_randomise_beyondtensor.sh 1500 J_ln_D
# Run detached so an SSH drop cannot SIGHUP a multi-hour job:
#         setsid nohup bash dgx_randomise_beyondtensor.sh 1500 J_ln_D > log 2>&1 < /dev/null &
set -u
BVAL="${1:?usage: $0 <bval> <metric>}"
METRIC="${2:?usage: $0 <bval> <metric>}"
NPERM="${NPERM:-5000}"
SEED="${SEED:-1}"
# canonical output prefix: lowercase, underscores stripped, + _rig
#   J_ln_D -> jlnd_rig,  J_dir -> jdir_rig  (matches build_gap_beyondtensor.py)
PREFIX="${PREFIX:-$(echo "$METRIC" | tr 'A-Z' 'a-z' | tr -d '_')_rig}"

FSLDIR="$HOME/fsl"
. "$FSLDIR/etc/fslconf/fsl.sh"
PATH="$FSLDIR/share/fsl/bin:$FSLDIR/bin:$PATH"
export FSLDIR PATH FSLOUTPUTTYPE=NIFTI_GZ

PY="${PY:-$HOME/diffusion-venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TBSS="$HOME/tbss_age_b${BVAL}_n1379"
cd "$TBSS" || { echo "no $TBSS"; exit 1; }

echo "==== beyond-tensor $METRIC b=$BVAL on $(hostname) $(date) ===="
for f in skeleton_good_mask.nii.gz design_full/design.mat design_full/design.con \
         stats/all_${METRIC}_skeletonised.nii.gz; do
  [ -s "$f" ] || { echo "MISSING $f"; exit 1; }
done

if [ ! -s "resid_${METRIC}_tensor.nii.gz" ]; then
  echo "[resid] $(date +%H:%M:%S)"
  "$PY" "$HERE/make_resid_tensor.py" "$TBSS" "$METRIC" || { echo "RESID FAILED"; exit 1; }
fi

if [ ! -s "${PREFIX}_tfce_corrp_tstat1.nii.gz" ]; then
  echo "[randomise $NPERM] $(date +%H:%M:%S)"; s=$(date +%s)
  randomise -i "resid_${METRIC}_tensor.nii.gz" -o "$PREFIX" -m skeleton_good_mask.nii.gz \
    -d design_full/design.mat -t design_full/design.con \
    -n "$NPERM" -T --seed="$SEED" > "${PREFIX}.log" 2>&1 || { echo "RANDOMISE FAILED"; exit 1; }
  echo "  randomise $(( $(date +%s) - s ))s"
fi

echo "[localize]"
"$PY" "$HERE/localize_beyondtensor.py" "$TBSS" "$PREFIX" "$METRIC"
echo "ALL DONE beyond-tensor $METRIC b=${BVAL}"
