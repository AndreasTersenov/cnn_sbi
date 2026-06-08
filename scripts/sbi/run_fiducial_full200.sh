#!/usr/bin/env bash
# Full-200 fiducial study orchestrator (overnight, autonomous).
# Pipeline: per-patch SUMMARIES (5 extractions) -> ANALYSIS (6 arms: step1 mean-dv +
# step2 per-patch distribution, each behind a G3 end-to-end gate) -> PLOTS -> SUMMARY.
# Each arm is isolated: a G1/G3 failure aborts THAT arm only (logged), never silent garbage.
set -u -o pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi
cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
DC="$REPO/scripts/sbi/results/exploratory/definitive_comparison"
FF="$DC/fiducial_full200"
SUM="$FF/summaries"; LOG="$FF/logs"
GRID="$REPO/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
FID="$REPO/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial"
mkdir -p "$SUM" "$LOG" "$FF/posteriors"
GPU=1; PERMS="0-199"; N2=300
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONUNBUFFERED=1
stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log(){ echo "[$(stamp)] $*" | tee -a "$FF/STATUS.log"; }
feltnote(){ felt history append definitive-l1-vs-cnn-2026-05/fiducial-full200-meandv --summary "$*" 2>/dev/null || true; }

CTF="$DC/phaseA_tfdata_2026_05_30/compressors"
MAFC="$DC/phaseA_maf_2026_05_31/compressors"
ac_p() { echo "$1/autocross_s41/vmim/nbody/sigma_0.26/gal_density_30/bin_4/harmonic_nobnt_ch10/$2"; }
ao_p() { echo "$1/autoonly_s41/vmim/nbody/sigma_0.26/gal_density_30/bin_4/harmonic_nobnt_ch4/$2"; }

log "=== FIDUCIAL FULL-200 START (GPU$GPU, perms $PERMS, step2 N=$N2) ==="
feltnote "Full-200 run STARTED: summaries (5) -> analysis (6 arms, G3-gated) -> plots."

# ---------------- 1. SUMMARIES ----------------
cnn_summ(){ # label params state nch out
  local lbl=$1 par=$2 st=$3 nch=$4 out=$5 obs=$6
  [ -f "$out" ] && { log "SKIP summary $lbl (exists)"; return 0; }
  log "SUMMARY(cnn) $lbl"
  $PY build_fiducial_summaries_cnn.py --arm-label "$lbl" --params-pkl "$par" --state-pkl "$st" \
    --n-channels "$nch" --perms "$PERMS" --out "$out" --g1-obs-npz "$obs" \
    --cuda-visible-devices "$GPU" > "$LOG/summary_${lbl}.log" 2>&1 \
    && log "  OK $lbl" || log "  FAIL $lbl (see summary_${lbl}.log)"
}
cnn_summ cnn_autocross "$(ac_p "$CTF" params_nd_compressor_best_val.pkl)" "$(ac_p "$CTF" opt_state_resnet_best_val.pkl)" 10 "$SUM/cnn_autocross_S.npz" "$DC/phaseA_tfdata_2026_05_30/compressed/autocross_s41/cnn_obs.npz"
cnn_summ cnn_autoonly  "$(ao_p "$CTF" params_nd_compressor_best_val.pkl)" "$(ao_p "$CTF" opt_state_resnet_best_val.pkl)" 4  "$SUM/cnn_autoonly_S.npz"  "$DC/phaseA_tfdata_2026_05_30/compressed/autoonly_s41/cnn_obs.npz"
cnn_summ cnn_maf_autocross "$(ac_p "$MAFC" params_nd_compressor_best_val.pkl)" "$(ac_p "$MAFC" opt_state_resnet_best_val.pkl)" 10 "$SUM/cnn_maf_autocross_S.npz" "$DC/phaseA_maf_2026_05_31/compressed/autocross_s41/cnn_obs.npz"

l1_summ(){ # label channelmode cachedir out
  local lbl=$1 cm=$2 cache=$3 out=$4
  [ -f "$out" ] && { log "SKIP summary $lbl (exists)"; return 0; }
  log "SUMMARY(l1) $lbl"
  local CM=(); [ "$cm" = "auto_only" ] && CM=(--channel-mode auto_only)
  $PY npe_l1norm_cross_jaxili_nbody_tomo.py \
    --full-sphere-cross-cache "$GRID" \
    --zero-mean-maps --map-kind nbody --field-size 20 --field-npix 160 \
    --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --pca-components 0 --l1-min-snr -13 --l1-max-snr 13 --cross-snr-percentile 1.0 \
    --batch-size 256 --learning-rate 0.0001 --npe-samples 100000 --no-wandb \
    --cross-noise-model channel_empirical_global --epochs 50000 --no-l1-train-flip \
    --nde-train-split "train[70%:]" --cache-dir "$cache" --seed 41 --harmonic-obs-perm 0 \
    "${CM[@]}" --cuda-visible-devices "$GPU" \
    --fiducial-summaries-out "$out" --fiducial-perms "$PERMS" --fiducial-obs-cache-dir "$FID" \
    > "$LOG/summary_${lbl}.log" 2>&1 \
    && log "  OK $lbl" || log "  FAIL $lbl (see summary_${lbl}.log)"
}
l1_summ l1_autocross auto_cross "$DC/compressed/l1_autocross_split70_dv" "$SUM/l1_autocross_S.npz"
l1_summ l1_autoonly  auto_only  "$DC/compressed/l1_autoonly_split70_dv"  "$SUM/l1_autoonly_S.npz"
feltnote "Summaries done. Files: $(ls "$SUM"/*.npz 2>/dev/null | wc -l)/5."

# ---------------- 2. ANALYSIS (6 arms, G3-gated) ----------------
analyze(){ # label cachedir prefix summaries transform clip minvar expfom3
  local lbl=$1 cache=$2 pref=$3 sm=$4 tr=$5 clip=$6 mv=$7 ef=$8
  [ -f "$FF/posteriors/$lbl/mean_dv.fom.json" ] && { log "SKIP analysis $lbl (done)"; return 0; }
  [ -f "$sm" ] || { log "SKIP analysis $lbl (no summaries $sm)"; return 0; }
  log "ANALYZE $lbl (transform=$tr clip=$clip expFoM3=$ef)"
  $PY fiducial_analyze.py --train-cache-dir "$cache" --cache-prefix "$pref" \
    --summaries-npz "$sm" --arm-label "$lbl" --output-dir "$FF/posteriors" \
    --preproc-transform "$tr" --clip-value "$clip" --min-feature-variance "$mv" \
    --n-step2-patches "$N2" --expected-fom3 "$ef" --g3-tol 0.20 \
    --cuda-visible-devices "$GPU" > "$LOG/analyze_${lbl}.log" 2>&1 \
    && { log "  OK $lbl"; feltnote "Arm $lbl analysis OK (G3 passed)."; } \
    || { log "  FAIL $lbl (see analyze_${lbl}.log)"; feltnote "Arm $lbl analysis FAILED (G3 or error)."; }
}
TF="$DC/phaseA_tfdata_2026_05_30/compressed"
analyze cnn_autocross       "$TF/autocross_s41"             cnn "$SUM/cnn_autocross_S.npz"     none         0   1e-12 26748
analyze cnn_autoonly        "$TF/autoonly_s41"              cnn "$SUM/cnn_autoonly_S.npz"      none         0   1e-12 9125
analyze cnn_autocross_std   "$TF/autocross_s41"             cnn "$SUM/cnn_autocross_S.npz"     zscore       0   1e-12 24281
analyze cnn_maf_autocross   "$DC/phaseA_maf_2026_05_31/compressed/autocross_s41" cnn "$SUM/cnn_maf_autocross_S.npz" none 0 1e-12 11984
analyze l1_autocross        "$DC/compressed/l1_autocross_split70_dv" l1 "$SUM/l1_autocross_S.npz" log1p-zscore 5.0 1e-5 34607
analyze l1_autoonly         "$DC/compressed/l1_autoonly_split70_dv"  l1 "$SUM/l1_autoonly_S.npz"  log1p-zscore 5.0 1e-5 10560

# ---------------- 3. PLOTS + SUMMARY ----------------
log "=== plotting + summary ==="
$PY plot_fiducial.py --root "$FF/posteriors" --out "$FF" > "$LOG/plot.log" 2>&1 \
  && log "  plots OK" || log "  plots FAIL (see plot.log)"

touch "$FF/.FIDUCIAL_FULL200_DONE"
log "=== FIDUCIAL FULL-200 COMPLETE ==="
feltnote "Full-200 run COMPLETE. See fiducial_full200/FIDUCIAL_FULL200_SUMMARY.md + overlays."
