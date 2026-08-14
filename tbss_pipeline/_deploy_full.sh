#!/usr/bin/env bash
# Deploy full confound-adjusted TBSS inputs + design + launcher to a Spark.
# Mirror of _deploy_agesex.sh for the 10-EV full design. Ships the design_full
# (converted from build_design_full.py output via Text2Vest) and the launcher;
# the skeletonised 4D is usually already on the host from the agesex deploy.
# usage (run inside WSL): _deploy_full.sh <bval> <dgxhost>
set -e
BVAL="${1:?usage: $0 <bval> <dgxhost>}"
HOST="${2:?usage: $0 <bval> <dgxhost>}"
# Local data-tree root (holds tbss_age_b*/stats and HCP/). Set JG_DATA to it.
L="${JG_DATA:?set JG_DATA to your local data-tree root (tbss_age_b*/stats, HCP/)}"
HERE="$(cd "$(dirname "$0")" && pwd)"   # tbss_pipeline/ in this repo (holds the launcher)
STATS="$L/tbss_age_b${BVAL}_n1379/stats"

echo "[$HOST b=$BVAL] ensure remote dirs"
ssh "$HOST" "mkdir -p tbss_age_b${BVAL}_n1379/stats tbss_age_b${BVAL}_n1379/design_full"

echo "[$HOST b=$BVAL] ship skeletonised 4D + mask (skip if already present)"
rsync -aP --inplace \
  "$STATS"/all_{FA,J_ln_D,J_dir}_skeletonised.nii.gz \
  "$STATS"/mean_FA_skeleton_mask.nii.gz \
  "$HOST:tbss_age_b${BVAL}_n1379/stats/"

echo "[$HOST b=$BVAL] ship design"
rsync -a "$L/HCP/design_full_b${BVAL}.mat" "$HOST:tbss_age_b${BVAL}_n1379/design_full/design.mat"
rsync -a "$L/HCP/contrast_full.con"        "$HOST:tbss_age_b${BVAL}_n1379/design_full/design.con"

echo "[$HOST b=$BVAL] ship launcher"
rsync -a "$HERE/dgx_randomise_full.sh" "$HOST:"

echo "[$HOST b=$BVAL] deploy done; remote skeletonised count:"
ssh "$HOST" "ls tbss_age_b${BVAL}_n1379/stats/all_*_skeletonised.nii.gz 2>/dev/null | wc -l"
