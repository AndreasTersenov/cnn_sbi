#!/usr/bin/env bash
# Phase A (definitive L1-vs-CNN) — retrain the 2 RealNVP CNN compressors on the
# FAST tf.data cross route (Andreas decision 2026-05-30: "use the new route").
#
#   auto+cross : --channel-mode auto_cross  (10 ch)   GPU 0
#   auto-only  : --channel-mode auto_only   (4 ch, sliced from the SAME cross
#                dataset, so the auto-vs-cross gain is route-matched)  GPU 1
#
# Seeds 41,42,43. 80k steps, best-val. --exit-after-compress -> writes the
# compressed summary cache (cnn_train/val/obs.npz) into each run's --cache-dir,
# ready for Phase B (train_jaxili_from_compressed.py, jaxili MAF NDE).
#
# *** LEAKAGE FLAG (accepted by Andreas, "flag leakage") ***
# The tf.data cross dataset was built with pool.imap_unordered (builder
# tf_dataset_nbody_tomo_cross.py:144), so its example order is the workers'
# completion order, NOT sorted-file order. The compressor trains on tf.data
# train[:70%] (a ~random 70% of realizations) while the NDE reads .npz
# train[70%:] (sorted last 30%). These overlap ~70% -> compressor<->NDE
# leakage -> ABSOLUTE FoM inflated (~1.6x, per HANDOFF_PERF_REGRESSION_RESOLVED).
# => L1-vs-CNN ABSOLUTE numbers are NOT trustworthy from this batch.
# => auto-vs-cross RELATIVE CNN gain IS fair (both arms leak identically).
# See README_LEAKAGE.md in the output dir.
set -u -o pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
NPZ="$REPO/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
TFDATA=/home/tersenov/tensorflow_datasets
TFDS_CROSS=nbody_cosmogrid_dataset_tomo_cross/grid_20deg_160px_nonoverlap48
TFDS_OBS=NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48

OUT="$REPO/scripts/sbi/results/exploratory/definitive_comparison/phaseA_tfdata_2026_05_30"
mkdir -p "$OUT/logs" "$OUT/compressors" "$OUT/compressed"
STATUS="$OUT/STATUS.md"

SEEDS=(41 42 43)
stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log(){ echo "[$(stamp)] $*" | tee -a "$OUT/orchestrator.log"; }

COMMON=(
  --train-compressor --exit-after-compress
  --map-kind nbody
  --tfds-name "$TFDS_OBS"
  --field-size 20 --field-npix 160 --nbins 4 --tomo-bin-indices 1,2,3,4
  --full-sphere-cross-cache "$NPZ"
  --cross-tfdata-dir "$TFDATA"
  --grain-tfds-name "$TFDS_CROSS"
  --harmonic-cache-regime nobnt
  --harmonic-normalize-input-channels --zero-mean-maps
  --compressor-arch plain --compressor-dim 10
  --compressor-conv-channels 64,128,256 --compressor-dense-width 256
  --compressor-train-split 'train[:70%]' --compressor-val-split val
  --nde-train-split 'train[70%:]' --nde-val-split val
  --compressor-lr 5e-4 --compressor-batch-size 128
  --compressor-checkpoint-policy best_val
  --compressor-steps 80000 --compressor-save-every 1000
)

# run_one <arm:autocross|autoonly> <seed> <gpu>
run_one(){
  local arm="$1" seed="$2" gpu="$3"
  local mode; [ "$arm" = autocross ] && mode=auto_cross || mode=auto_only
  local sdir="$OUT/compressors/${arm}_s${seed}"
  local cdir="$OUT/compressed/${arm}_s${seed}"
  local lg="$OUT/logs/${arm}_s${seed}.log"
  mkdir -p "$sdir" "$cdir"
  if [ -f "$OUT/.done_${arm}_s${seed}" ]; then log "SKIP ${arm}_s${seed} (done)"; return 0; fi
  log "START ${arm}_s${seed} (mode=$mode) GPU$gpu -> $lg"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 PYTHONUNBUFFERED=1 \
  "$PY" -u "$CNN" \
    "${COMMON[@]}" \
    --channel-mode "$mode" \
    --cuda-visible-devices "$gpu" \
    --seed "$seed" \
    --save-dir "$sdir" --cache-dir "$cdir" \
    > "$lg" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$cdir/cnn_train.npz" ]; then
    touch "$OUT/.done_${arm}_s${seed}"; log "DONE  ${arm}_s${seed} rc=0"
  else
    log "FAIL  ${arm}_s${seed} rc=$rc (see $lg)"
  fi
  return $rc
}

# Independent per-arm loops: each GPU churns its 3 seeds back-to-back with no
# cross-arm wait, so neither GPU idles (autoonly's one-time RMS scan on seed 41
# doesn't stall GPU 0's autocross seeds).
arm_loop(){
  local arm="$1" gpu="$2"
  for s in "${SEEDS[@]}"; do run_one "$arm" "$s" "$gpu"; done
  touch "$OUT/.ARM_DONE_${arm}"
  log "=== arm '$arm' (GPU$gpu) all seeds complete ==="
}

log "=== Phase A tf.data overnight START (seeds=${SEEDS[*]}; autocross@GPU0, autoonly@GPU1) ==="
arm_loop autocross 0 &  A0=$!
arm_loop autoonly  1 &  A1=$!
wait $A0; wait $A1
log "=== Phase A tf.data overnight COMPLETE ==="
touch "$OUT/.PHASEA_TFDATA_DONE"
