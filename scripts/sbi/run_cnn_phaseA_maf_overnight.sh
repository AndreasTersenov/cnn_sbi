#!/usr/bin/env bash
# Phase A (arm b: MAF companion) — retrain the 2 CNN compressors with the
# conditional-MAF VMIM companion (8 transforms, hidden [256,256]) instead of
# the sbi_lens RealNVP companion. SAME fast tf.data route as 2026-05-30, so the
# MAF-vs-RealNVP companion delta is isolated (leakage identical on both, cancels).
#
# Tests: does the companion flow quality limit the CNN compressor? Compare these
# FoM to phaseB_tfdata_2026_05_30 (RealNVP companion). MAF validated by
# test_vmim_maf_companion.py (identity-init==N(0,I); log-det==autograd Jacobian).
set -u -o pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
NPZ="$REPO/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
TFDATA=/home/tersenov/tensorflow_datasets
TFDS_CROSS=nbody_cosmogrid_dataset_tomo_cross/grid_20deg_160px_nonoverlap48
TFDS_OBS=NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48

OUT="$REPO/scripts/sbi/results/exploratory/definitive_comparison/phaseA_maf_2026_05_31"
mkdir -p "$OUT/logs" "$OUT/compressors" "$OUT/compressed"
SEEDS=(41 42 43)
stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log(){ echo "[$(stamp)] $*" | tee -a "$OUT/orchestrator.log"; }

COMMON=(
  --train-compressor --exit-after-compress
  --map-kind nbody --tfds-name "$TFDS_OBS"
  --field-size 20 --field-npix 160 --nbins 4 --tomo-bin-indices 1,2,3,4
  --full-sphere-cross-cache "$NPZ"
  --cross-tfdata-dir "$TFDATA" --grain-tfds-name "$TFDS_CROSS"
  --harmonic-cache-regime nobnt
  --harmonic-normalize-input-channels --zero-mean-maps
  --compressor-arch plain --compressor-dim 10
  --compressor-conv-channels 64,128,256 --compressor-dense-width 256
  --compressor-train-split 'train[:70%]' --compressor-val-split val
  --nde-train-split 'train[70%:]' --nde-val-split val
  --compressor-lr 5e-4 --compressor-batch-size 128 --compressor-checkpoint-policy best_val
  --compressor-steps 80000 --compressor-save-every 1000
  --vmim-companion-backend maf --vmim-maf-transforms 8 --vmim-maf-hidden 256
)

run_one(){
  local arm="$1" seed="$2" gpu="$3"
  local mode; [ "$arm" = autocross ] && mode=auto_cross || mode=auto_only
  local sdir="$OUT/compressors/${arm}_s${seed}" cdir="$OUT/compressed/${arm}_s${seed}"
  local lg="$OUT/logs/${arm}_s${seed}.log"
  mkdir -p "$sdir" "$cdir"
  [ -f "$OUT/.done_${arm}_s${seed}" ] && { log "SKIP ${arm}_s${seed}"; return 0; }
  log "START ${arm}_s${seed} (mode=$mode, MAF companion) GPU$gpu -> $lg"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 PYTHONUNBUFFERED=1 \
  "$PY" -u "$CNN" "${COMMON[@]}" --channel-mode "$mode" \
    --cuda-visible-devices "$gpu" --seed "$seed" \
    --save-dir "$sdir" --cache-dir "$cdir" > "$lg" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$cdir/cnn_train.npz" ]; then
    touch "$OUT/.done_${arm}_s${seed}"; log "DONE  ${arm}_s${seed} rc=0"
  else log "FAIL  ${arm}_s${seed} rc=$rc (see $lg)"; fi
}

arm_loop(){ local arm="$1" gpu="$2"; for s in "${SEEDS[@]}"; do run_one "$arm" "$s" "$gpu"; done; touch "$OUT/.ARM_DONE_${arm}"; log "=== arm $arm done ==="; }

log "=== Phase A MAF START (autocross@GPU0, autoonly@GPU1) ==="
arm_loop autocross 0 & A0=$!
arm_loop autoonly  1 & A1=$!
wait $A0; wait $A1
log "=== Phase A MAF COMPLETE ==="; touch "$OUT/.PHASEA_MAF_DONE"
