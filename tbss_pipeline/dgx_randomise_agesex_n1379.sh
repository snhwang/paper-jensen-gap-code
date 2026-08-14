#!/usr/bin/env bash
# randomise launcher for the field-standard age + age^2 + sex + age*sex model
# (N=1379). Clone of dgx_randomise_n1379.sh.
#
# Differences from the age-only launcher:
#   - design:  ../design_agesex/design.mat  (5 EVs)  ../design_agesex/design.con (8 t-contrasts)
#   - output:  ${m}_agesex  ->  ${m}_agesex_tfce_corrp_tstat1..8.nii.gz
#              (does NOT touch the existing ${m}_age_* age-only maps)
#   - reuses the existing all_${m}_skeletonised.nii.gz 4D files in place
#   - MAX_PARALLEL default lowered (8 contrasts use more RAM/time than 2)
#
# Contrast -> tstat map:
#   1 +age   2 -age   3 +age^2   4 -age^2
#   5 +sex(M>F)  6 -sex(F>M)  7 +age*sex  8 -age*sex
#
# Usage:
#   dgx_randomise_agesex_n1379.sh <bval> [--parallel]
#
# Two execution modes:
#   (default) THROUGHPUT -- many metrics at once, each a plain single-core
#             `randomise`; up to MAX_PARALLEL (default 8) run concurrently in the
#             background with RAM-aware throttling. Best for sweeping the whole
#             METRICS list: cores stay busy across metrics, finishes the batch in
#             ~a day even though each metric takes ~a day on its own core.
#   --parallel LATENCY   -- one metric at a time, but `randomise_parallel` splits
#             the 5,000 permutations across all free cores, so a SINGLE job
#             finishes in hours instead of ~a day. Use this for ONE new parameter
#             or a lone rerun. Do NOT use --parallel with a long METRICS list:
#             cores x metrics oversubscribes and swaps (the trap we hit before).
#             MAX_PARALLEL is ignored in this mode.
#             (randomise_parallel uses fsl_sub; on a standalone box like the
#             Sparks it runs the permutation chunks locally across cores.)
#
# Env knobs: MAX_PARALLEL (throughput fan-out), NPERM (default 5000), WAIT_SEC.
set -u
PARALLEL="${PARALLEL:-0}"
BVAL=""
for a in "$@"; do
  case "$a" in
    --parallel) PARALLEL=1 ;;
    -*)         echo "unknown flag: $a  (usage: $0 <bval> [--parallel])" >&2; exit 2 ;;
    *)          if [ -z "$BVAL" ]; then BVAL="$a"; else echo "unexpected arg: $a" >&2; exit 2; fi ;;
  esac
done
[ -n "$BVAL" ] || { echo "usage: $0 <bval> [--parallel]" >&2; exit 1; }
MAX_PARALLEL="${MAX_PARALLEL:-8}"
WAIT_SEC="${WAIT_SEC:-300}"
NPERM="${NPERM:-5000}"

# Priority order: FA (shared ref) -> info-theory headline (PR_ang) -> jensen-gap
# -> info-theory rest -> model-light -> tensor rest
# PR_ang/N is the info-theory paper's headline TBSS metric; its skeletonised 4D
# (all_PR_ang_skeletonised.nii.gz) is built by dgx_prang_tbss.sh into the same
# stats dir, so the agesex run picks it up like any other metric.
METRICS=(FA PR_ang J_dir J_ln_D ADC_geo ADC_lin H_ang_norm R_aniso H_SH H_aniso CDCI H_W H_scale gamma_scale1 gamma_avg gamma_scale0 gamma_scale2 PR_pos_avg MD AD RD)

TBSS="$HOME/tbss_age_b${BVAL}_n1379"
cd "$TBSS/stats"

FSLDIR="$HOME/fsl"
. "$FSLDIR/etc/fslconf/fsl.sh"
PATH="$FSLDIR/share/fsl/bin:$FSLDIR/bin:$PATH"
export FSLDIR PATH

DESIGN="../design_agesex/design.mat"
CONTRAST="../design_agesex/design.con"
[ -s "$DESIGN" ]   || { echo "MISSING $DESIGN (rsync design_agesex_b${BVAL}.mat -> design_agesex/design.mat)"; exit 1; }
[ -s "$CONTRAST" ] || { echo "MISSING $CONTRAST (rsync design_agesex.con -> design_agesex/design.con)"; exit 1; }

LOGDIR="$HOME/randomise_agesex_logs_b${BVAL}"
mkdir -p "$LOGDIR"

echo "============================================"
echo "randomise AGESEX b=$BVAL  $(date)"
if [ "$PARALLEL" = "1" ]; then
  echo "  design=$DESIGN  contrasts=8  NPERM=$NPERM  mode=PARALLEL (randomise_parallel, all cores, 1 metric at a time)"
else
  echo "  design=$DESIGN  contrasts=8  NPERM=$NPERM  mode=THROUGHPUT (MAX_PARALLEL=$MAX_PARALLEL)"
fi
echo "  metrics: ${#METRICS[@]}"
echo "============================================"

active_count() { ps -C randomise --no-headers 2>/dev/null | wc -l; }

for m in "${METRICS[@]}"; do
  out="${m}_agesex_tfce_corrp_tstat8.nii.gz"   # last contrast == completion marker
  if [ -s "$out" ]; then
    echo "[$m] all 8 contrasts already done, skip"
    continue
  fi
  if pgrep -f "randomise.* -o ${m}_agesex" > /dev/null; then
    echo "[$m] already running, skip"
    continue
  fi
  if [ ! -s "all_${m}_skeletonised.nii.gz" ]; then
    echo "[$m] no input 4D (all_${m}_skeletonised.nii.gz), skip"
    continue
  fi

  # --parallel: run ONE metric at a time, but split its permutations across all
  # free cores via randomise_parallel (hours, not ~a day). Blocks until the
  # chunks finish and are merged, then moves to the next metric.
  if [ "$PARALLEL" = "1" ]; then
    echo
    echo "[$m] randomise_parallel (all cores)  $(date +%H:%M:%S)  free=$(free -g | awk '/Mem:/ {print $4}')G"
    rm -f "${m}_agesex"*.nii.gz
    randomise_parallel \
         -i "all_${m}_skeletonised.nii.gz" -o "${m}_agesex" \
         -m mean_FA_skeleton_mask.nii.gz \
         -d "$DESIGN" -t "$CONTRAST" \
         -n "$NPERM" --T2 --quiet \
         > "$LOGDIR/${m}.log" 2>&1
    echo "[$m] done  $(date +%H:%M:%S)"
    continue
  fi

  # Throttle: wait if too many running.
  while true; do
    n=$(active_count)
    if [ "$n" -lt "$MAX_PARALLEL" ]; then break; fi
    echo "  [$m] waiting: $n active >= $MAX_PARALLEL  $(date +%H:%M:%S)"
    sleep 120
  done

  fg=$(free -g | awk '/Mem:/ {print $4}')
  echo
  echo "[$m] launch $(date +%H:%M:%S)  active=$n  free=${fg}G"
  rm -f "${m}_agesex"*.nii.gz
  nohup "$FSLDIR/bin/randomise" \
       -i "all_${m}_skeletonised.nii.gz" -o "${m}_agesex" \
       -m mean_FA_skeleton_mask.nii.gz \
       -d "$DESIGN" -t "$CONTRAST" \
       -n "$NPERM" --T2 --quiet \
       > "$LOGDIR/${m}.log" 2>&1 < /dev/null &
  pid=$!
  echo "  PID $pid"

  start=$(date +%s)
  for t in 30 60 120 180 240 300; do
    now=$(date +%s)
    delta=$(( t - (now - start) ))
    [ "$delta" -gt 0 ] && sleep "$delta"
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "  [$m] DIED before t+${t}s  $(date +%H:%M:%S)"
      break
    fi
    rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d " ")
    fg_now=$(free -g | awk '/Mem:/ {print $4}')
    echo "  [$m] t+${t}s alive  RSS=$(( rss_kb / 1024 ))MB  free=${fg_now}G"
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "  [$m] OK past peak, moving on"
  fi
done

echo
echo "============================================"
echo "all launches done  $(date)  -- waiting for completion"
echo "============================================"
while true; do
  n=$(active_count)
  if [ "$n" -eq 0 ]; then break; fi
  echo "  $n randomise still running  $(date +%H:%M:%S)"
  sleep 600
done

echo
echo "============================================"
echo "done  $(date)"
echo "  *_agesex_tfce_corrp_tstat1.nii.gz (+age): $(ls *_agesex_tfce_corrp_tstat1.nii.gz 2>/dev/null | wc -l) / 20"
echo "  *_agesex_tfce_corrp_tstat8.nii.gz (-age*sex): $(ls *_agesex_tfce_corrp_tstat8.nii.gz 2>/dev/null | wc -l) / 20"
echo "============================================"
