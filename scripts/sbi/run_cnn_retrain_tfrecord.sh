#!/usr/bin/env bash
# Retrain the CNN compressors on the FAST harmonic TFRecord path (yesterday's work),
# so both auto-only and auto+cross share one route+shuffle regime.
#
#   auto+cross : --channel-mode auto_cross (10ch)  -> compressors/autocross_tfrec_rnvp
#   auto-only  : --channel-mode auto_only  (4 auto)-> compressors/autoonly_tfrec_rnvp
#
# Config is the reconstructed-consistent set (Andreas-approved): recovered flags
# (plain, cdim10, 64/128/256, dense256, splits, 80k, save-every 1000, zero-mean,
# harmonic-normalize, best_val) + script defaults for the few unrecorded ones
# (compressor-lr 5e-4, compressor-batch-size 128, seed 41). RealNVP companion is the
# script default (no --vmim-companion-backend flag exists yet). --exit-after-compress
# trains the compressor AND writes compressed cnn_{train,val,obs}.npz to --cache-dir.
#
# GPU 0 only, maxed out. Smoke (500 steps) first, then full 80k.
# Usage: run_cnn_retrain_tfrecord.sh {smoke_autocross|smoke_autoonly|train_autocross|train_autoonly}
set -uo pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi; cd "$REPO"
CNN=scripts/sbi/npe_cnn_nbody_tomo.py
NPZ=scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid
TFREC=/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid
OUT=scripts/sbi/results/exploratory/definitive_comparison
LOGS=$OUT/logs

# Reconstructed-consistent compressor config (identical across all arms).
COMMON=(--cuda-visible-devices "${CNN_GPU:-1}"
        --train-compressor
        --map-kind nbody --field-size 20 --field-npix 160
        --nbins 4 --tomo-bin-indices 1,2,3,4
        --full-sphere-cross-cache "$NPZ"
        --harmonic-tfrecord-dir "$TFREC"
        --harmonic-normalize-input-channels
        --zero-mean-maps
        --compressor-arch plain --compressor-dim 10
        --compressor-conv-channels 64,128,256 --compressor-dense-width 256
        --compressor-train-split 'train[:70%]' --nde-train-split 'train[70%:]'
        --compressor-lr 5e-4 --compressor-batch-size 128
        --compressor-checkpoint-policy best_val
        --seed 41
        --exit-after-compress)

# run <channel_mode> <label> <steps> <save_every>
run() {
  local cm="$1" label="$2" steps="$3" save_every="$4"
  local sdir="$OUT/compressors/$label" cdir="$OUT/compressed/${label}_split70"
  local log="$LOGS/cnn_retrain_${label}.log"
  mkdir -p "$sdir" "$cdir"
  echo "[launch] $label  channel_mode=$cm steps=$steps  save=$sdir cache=$cdir -> $log"
  # TF threading is set IN-PROCESS by npe_cnn (commit 526b12e: tf.config.threading,
  # CNN_TF_THREADS=32 default) before any TF op, so the tf.data pipeline is multi-threaded
  # regardless of the shell's OMP_NUM_THREADS=1. No env prefix needed (env-var thread limits
  # do NOT govern TF's tf.data threadpool). Look for "[cnn-tf-threading] intra=32 ..." in stdout.
  # NOTE: throughput still sags under heavy node load (threads contend); run in a lighter window.
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n jaxili python "$CNN" "${COMMON[@]}" \
    --channel-mode "$cm" \
    --compressor-steps "$steps" --compressor-save-every "$save_every" \
    --save-dir "$sdir" --cache-dir "$cdir" \
    > "$log" 2>&1
}

case "${1:?mode}" in
  smoke_autocross) run auto_cross autocross_tfrec_rnvp_SMOKE 500 250 ;;
  smoke_autoonly)  run auto_only  autoonly_tfrec_rnvp_SMOKE  500 250 ;;
  train_autocross) run auto_cross autocross_tfrec_rnvp 80000 1000 ;;
  train_autoonly)  run auto_only  autoonly_tfrec_rnvp  80000 1000 ;;
  *) echo "unknown mode: $1"; exit 2 ;;
esac
