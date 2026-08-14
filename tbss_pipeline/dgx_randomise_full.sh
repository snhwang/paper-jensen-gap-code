#!/usr/bin/env bash
# randomise launcher for the FULL confound-adjusted design (Figs 5-6):
#   age + age^2 + sex + age*sex + site(3 dummies) + motion + outlier  (10 EVs)
# Only the adjusted linear-age effect is contrasted (tstat1 +age, tstat2 -age).
# Reuses the existing all_${m}_skeletonised.nii.gz 4D in place; writes
#   ${m}_full_tfce_corrp_tstat1..2.nii.gz  (does NOT touch ${m}_agesex_* maps).
#
# Metrics are the three the manuscript's TBSS figures report:
#   J_dir (signal-domain gap), J_ln_D (diffusivity-domain gap), FA (DTI reference).
#
# Usage:  dgx_randomise_full.sh <bval> [--parallel]
set -u
PARALLEL="${PARALLEL:-0}"
BVAL=""
for a in "$@"; do
  case "$a" in
    --parallel) PARALLEL=1 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *)  if [ -z "$BVAL" ]; then BVAL="$a"; else echo "unexpected arg: $a" >&2; exit 2; fi ;;
  esac
done
[ -n "$BVAL" ] || { echo "usage: $0 <bval> [--parallel]" >&2; exit 1; }
NPERM="${NPERM:-5000}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

METRICS=(FA J_ln_D J_dir)

TBSS="$HOME/tbss_age_b${BVAL}_n1379"
cd "$TBSS/stats" || { echo "no $TBSS/stats"; exit 1; }

FSLDIR="$HOME/fsl"
. "$FSLDIR/etc/fslconf/fsl.sh"
PATH="$FSLDIR/share/fsl/bin:$FSLDIR/bin:$PATH"
export FSLDIR PATH

DESIGN="../design_full/design.mat"
CONTRAST="../design_full/design.con"
[ -s "$DESIGN" ]   || { echo "MISSING $DESIGN"; exit 1; }
[ -s "$CONTRAST" ] || { echo "MISSING $CONTRAST"; exit 1; }

LOGDIR="$HOME/randomise_full_logs_b${BVAL}"; mkdir -p "$LOGDIR"
echo "=== randomise FULL b=$BVAL  $(date) ==="
echo "  design=$DESIGN (10 EV)  contrasts=2 (+age,-age)  NPERM=$NPERM  parallel=$PARALLEL"
grep -E "NumWaves|NumPoints|NumContrasts" "$DESIGN" "$CONTRAST" 2>/dev/null

active_count() { ps -C randomise --no-headers 2>/dev/null | wc -l; }

for m in "${METRICS[@]}"; do
  if [ -s "${m}_full_tfce_corrp_tstat2.nii.gz" ]; then echo "[$m] done, skip"; continue; fi
  if [ ! -s "all_${m}_skeletonised.nii.gz" ]; then echo "[$m] NO 4D all_${m}_skeletonised, skip"; continue; fi
  rm -f "${m}_full"*.nii.gz

  if [ "$PARALLEL" = "1" ]; then
    echo "[$m] randomise_parallel (all cores) $(date +%H:%M:%S) free=$(free -g | awk '/Mem:/{print $4}')G"
    randomise_parallel \
      -i "all_${m}_skeletonised.nii.gz" -o "${m}_full" \
      -m mean_FA_skeleton_mask.nii.gz -d "$DESIGN" -t "$CONTRAST" \
      -n "$NPERM" --T2 --quiet > "$LOGDIR/${m}.log" 2>&1
    echo "[$m] done $(date +%H:%M:%S)"
  else
    while [ "$(active_count)" -ge "$MAX_PARALLEL" ]; do sleep 120; done
    echo "[$m] launch $(date +%H:%M:%S) free=$(free -g | awk '/Mem:/{print $4}')G"
    nohup "$FSLDIR/bin/randomise" \
      -i "all_${m}_skeletonised.nii.gz" -o "${m}_full" \
      -m mean_FA_skeleton_mask.nii.gz -d "$DESIGN" -t "$CONTRAST" \
      -n "$NPERM" --T2 --quiet > "$LOGDIR/${m}.log" 2>&1 < /dev/null &
    sleep 60
  fi
done

while [ "$(active_count)" -gt 0 ]; do echo "  $(active_count) randomise running $(date +%H:%M:%S)"; sleep 300; done
echo "=== done b=$BVAL $(date) ==="
ls -la *_full_tfce_corrp_tstat*.nii.gz 2>/dev/null
