#!/usr/bin/env bash
# tbss_fill the corrp maps used in the manuscript figures. Thickening follows
# tract structure (FSL's sanctioned method) rather than isotropic dilation.
# corrp is masked to the SCREENED skeleton first, so the displayed extent grows
# only from voxels that were actually part of the inference.
# Usage: run_tbss_fill.sh <bval>
B="${1:?usage: $0 <bval>}"
export FSLDIR=$HOME/fsl
. $FSLDIR/etc/fslconf/fsl.sh 2>/dev/null
export FSLDIR=$HOME/fsl PATH=$FSLDIR/share/fsl/bin:$FSLDIR/bin:$PATH FSLOUTPUTTYPE=NIFTI_GZ
D=$HOME/tbss_age_b${B}_n1379; S=$D/stats
OUT=$D/filled; mkdir -p "$OUT"; TMP=$(mktemp -d)
MEANFA=$S/mean_FA; SCREEN=$D/skeleton_screened_mask
echo "==== tbss_fill b=$B on $(hostname) $(date) ===="
for spec in "$S/J_dir_agesex_tfce_corrp_tstat1 J_dir_agesex_1" \
            "$S/J_dir_agesex_tfce_corrp_tstat2 J_dir_agesex_2" \
            "$S/J_ln_D_agesex_tfce_corrp_tstat1 J_ln_D_agesex_1" \
            "$S/J_ln_D_agesex_tfce_corrp_tstat2 J_ln_D_agesex_2" \
            "$S/FA_agesex_tfce_corrp_tstat1 FA_agesex_1" \
            "$S/FA_agesex_tfce_corrp_tstat2 FA_agesex_2" \
            "$D/jlnd_rig_tfce_corrp_tstat1 jlnd_rig_1" \
            "$D/jlnd_rig_tfce_corrp_tstat2 jlnd_rig_2" \
            "$D/jdir_rig_tfce_corrp_tstat1 jdir_rig_1" \
            "$D/jdir_rig_tfce_corrp_tstat2 jdir_rig_2"; do
  set -- $spec; src="$1"; name="$2"
  [ -s "${src}.nii.gz" ] || { echo "  [skip] ${src##*/}"; continue; }
  fslmaths "$src" -mas "$SCREEN" "$TMP/${name}_screened"
  tbss_fill "$TMP/${name}_screened" 0.95 "$MEANFA" "$OUT/${name}_fill" >/dev/null 2>&1 \
    && echo "  filled ${name}" || echo "  [FAIL] ${name}"
done
rm -rf "$TMP"
echo "==== done b=$B: $(ls $OUT/*_fill.nii.gz 2>/dev/null | wc -l) maps ===="
