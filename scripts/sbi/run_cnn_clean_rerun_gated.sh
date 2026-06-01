#!/usr/bin/env bash
# (a) Clean disjoint CNN rerun (RealNVP companion) — the leakage-free absolute
# L1-vs-CNN. Uses the .npz route (NO --cross-tfdata-dir) so the compressor trains
# from build_harmonic_batch_iterator (sorted-file slicing == NDE) => genuinely
# disjoint, audit-valid. GATED & self-validating so it never burns 80k on a slow
# loader:
#   1) wait for the MAF campaign to free GPU 1 (.ARM_DONE_autoonly).
#   2) 500-step loader-threads smoke (threads=24); measure it/s.
#   3) only if >= 6 it/s: launch clean RealNVP seed-41 auto+cross & auto-only
#      (80k, --exit-after-compress) + jaxili NDE (seeds 41,42,43, perm0).
#      else: log and STOP (don't waste compute) for review.
set -u -o pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi
cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
NDE="$REPO/scripts/sbi/train_jaxili_from_compressed.py"
NPZ="$REPO/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
MAFPA="$REPO/scripts/sbi/results/exploratory/definitive_comparison/phaseA_maf_2026_05_31"
OUT="$REPO/scripts/sbi/results/exploratory/definitive_comparison/clean_rerun_2026_05_31"
mkdir -p "$OUT/logs" "$OUT/compressors" "$OUT/compressed" "$OUT/posteriors"
GPU=1
stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }; log(){ echo "[$(stamp)] $*" | tee -a "$OUT/gated.log"; }

COMMON=(
  --train-compressor --exit-after-compress
  --map-kind nbody --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48
  --field-size 20 --field-npix 160 --nbins 4 --tomo-bin-indices 1,2,3,4
  --full-sphere-cross-cache "$NPZ"
  --harmonic-cache-regime nobnt --harmonic-normalize-input-channels --zero-mean-maps
  --compressor-arch plain --compressor-dim 10
  --compressor-conv-channels 64,128,256 --compressor-dense-width 256
  --compressor-train-split 'train[:70%]' --compressor-val-split val
  --nde-train-split 'train[70%:]' --nde-val-split val
  --compressor-lr 5e-4 --compressor-batch-size 128 --compressor-checkpoint-policy best_val
  --harmonic-loader-threads 24 --harmonic-loader-pool 12 --harmonic-loader-prefetch 12
)

log "=== clean-rerun gated waiter START; waiting for GPU1 free (.ARM_DONE_autoonly) ==="
until [ -f "$MAFPA/.ARM_DONE_autoonly" ]; do sleep 60; done
log "GPU1 freed (MAF auto-only arm done). Running loader-threads smoke (threads=24)."

SM="$OUT/smoke"; rm -rf "$SM"; mkdir -p "$SM"
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONUNBUFFERED=1 \
"$PY" -u "$CNN" "${COMMON[@]}" --channel-mode auto_cross --cuda-visible-devices "$GPU" \
  --seed 41 --compressor-steps 500 --compressor-save-every 500 \
  --save-dir "$SM/save" --cache-dir "$SM/cache" > "$SM/smoke.log" 2>&1 || true
RATE=$(grep -aoE "[0-9]+/500 \[[0-9:]+<[0-9:]+, +[0-9.]+it/s\]" "$SM/smoke.log" | tail -3 | grep -oE "[0-9.]+it/s" | grep -oE "[0-9.]+" | tail -1)
RATE=${RATE:-0}
log "loader-threads smoke steady-state: ${RATE} it/s (gate >= 4; MAF auto+cross may share CPU)"
rm -rf "$SM"
if ! awk -v r="$RATE" 'BEGIN{exit !(r+0>=4)}'; then
  log "GATE FAILED: clean .npz route still ${RATE} it/s (<4). NOT launching 80k. Needs review."
  touch "$OUT/.CLEAN_RERUN_ABORTED"; exit 0
fi
log "GATE PASSED (${RATE} it/s). Launching clean RealNVP seed-41 auto+cross then auto-only."

run_clean(){
  local arm="$1" mode; [ "$arm" = autocross ] && mode=auto_cross || mode=auto_only
  local sdir="$OUT/compressors/${arm}_s41" cdir="$OUT/compressed/${arm}_s41" lg="$OUT/logs/${arm}_s41.log"
  mkdir -p "$sdir" "$cdir"
  log "CLEAN compressor START $arm (mode=$mode, disjoint .npz)"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 PYTHONUNBUFFERED=1 \
  "$PY" -u "$CNN" "${COMMON[@]}" --channel-mode "$mode" --cuda-visible-devices "$GPU" \
    --seed 41 --compressor-steps 80000 --compressor-save-every 1000 \
    --save-dir "$sdir" --cache-dir "$cdir" > "$lg" 2>&1
  if [ -f "$cdir/cnn_train.npz" ]; then
    log "CLEAN compressor DONE $arm; running jaxili NDE (seeds 41,42,43)"
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 PYTHONUNBUFFERED=1 \
    "$PY" -u "$NDE" --compressed-dir "$cdir" --arm-label "${arm}_clean_cs41" \
      --output-dir "$OUT/posteriors/${arm}" --seeds 41,42,43 --cuda-visible-devices "$GPU" \
      > "$OUT/logs/nde_${arm}.log" 2>&1 && log "CLEAN NDE DONE $arm" || log "CLEAN NDE FAIL $arm"
  else
    log "CLEAN compressor FAIL $arm (see $lg)"
  fi
}
run_clean autocross
run_clean autoonly
log "=== clean-rerun gated waiter COMPLETE ==="; touch "$OUT/.CLEAN_RERUN_DONE"
