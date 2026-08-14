#!/usr/bin/env bash
# Deploy sex-adjusted TBSS inputs + design + launcher to a Spark.
# usage (run inside WSL): _deploy_agesex.sh <bval> <dgxhost>
set -e
BVAL="${1:?usage: $0 <bval> <dgxhost>}"
HOST="${2:?usage: $0 <bval> <dgxhost>}"
# Local data-tree root (holds tbss_age_b*/stats and HCP/). Set JG_DATA to it.
L="${JG_DATA:?set JG_DATA to your local data-tree root (tbss_age_b*/stats, HCP/)}"
HERE="$(cd "$(dirname "$0")" && pwd)"   # tbss_pipeline/ in this repo (holds the launcher)
STATS="$L/tbss_age_b${BVAL}_n1379/stats"

echo "[$HOST b=$BVAL] ensure remote dirs"
ssh "$HOST" "mkdir -p tbss_age_b${BVAL}_n1379/stats tbss_age_b${BVAL}_n1379/design_agesex"

echo "[$HOST b=$BVAL] ship skeletonised 4D + mask (~12GB)"
rsync -aP --inplace \
  "$STATS"/all_*_skeletonised.nii.gz \
  "$STATS"/mean_FA_skeleton_mask.nii.gz \
  "$HOST:tbss_age_b${BVAL}_n1379/stats/"

echo "[$HOST b=$BVAL] ship design"
rsync -a "$L/HCP/design_agesex_b${BVAL}.mat" "$HOST:tbss_age_b${BVAL}_n1379/design_agesex/design.mat"
rsync -a "$L/HCP/design_agesex.con"          "$HOST:tbss_age_b${BVAL}_n1379/design_agesex/design.con"

echo "[$HOST b=$BVAL] ship launcher"
rsync -a "$HERE/dgx_randomise_agesex_n1379.sh" "$HOST:"

echo "[$HOST b=$BVAL] deploy done; remote skeletonised count:"
ssh "$HOST" "ls tbss_age_b${BVAL}_n1379/stats/all_*_skeletonised.nii.gz 2>/dev/null | wc -l"
