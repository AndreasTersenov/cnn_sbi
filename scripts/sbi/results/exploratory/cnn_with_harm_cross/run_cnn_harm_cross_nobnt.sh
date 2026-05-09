#!/usr/bin/env bash
set -euo pipefail

cd /mnt/home/tersenov/software/cnn_sbi

GPU="${1:-0}"
MODE="${2:-smoke}"    # smoke | full
SEED="${3:-42}"

case "$GPU" in
  0|1|2) ;;
  *)
    echo "GPU must be one of: 0, 1, 2 (policy lock)." >&2
    exit 2
    ;;
esac

case "$MODE" in
  smoke|full) ;;
  *)
    echo "MODE must be smoke or full." >&2
    exit 2
    ;;
esac

ROOT="scripts/sbi/results/exploratory/cnn_with_harm_cross"
RUN_DIR="$ROOT/nobnt/${MODE}_seed_${SEED}"
HARM_CACHE="scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"

mkdir -p "$RUN_DIR/cache" "$RUN_DIR/posteriors" "$RUN_DIR/figures"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_PYTHON_CLIENT_PREALLOCATE=false

COMMON=(
  --cuda-visible-devices "$GPU"
  --no-wandb
  --map-kind nbody
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48
  --field-size 20
  --field-npix 160
  --nbins 4
  --tomo-bin-indices 1,2,3,4
  --full-sphere-cross-cache "$HARM_CACHE"
  --cnn-map-route harmonic
  --harmonic-cache-regime nobnt
  --train-compressor
  --seed "$SEED"
  --save-dir "$RUN_DIR/save_params"
  --cache-dir "$RUN_DIR/cache"
  --posterior-out "$RUN_DIR/posteriors/cnn_harm_cross_nobnt_s${SEED}.npy"
  --figure-out "$RUN_DIR/figures/cnn_harm_cross_nobnt_s${SEED}.png"
)

if [[ "$MODE" == "smoke" ]]; then
  EXTRA=(
    --harmonic-train-realizations-limit 1
    --harmonic-val-realizations-limit 1
    --compressor-steps 2
    --compressor-save-every 1
    --compressor-batch-size 16
    --total-steps 2
    --save-every 1
    --batch-size 16
    --patience 0
    --ds-batch-size 16
    --npe-samples 64
    --no-sample
  )
else
  EXTRA=(
    --compressor-steps 150000
    --compressor-save-every 2000
    --compressor-batch-size 128
    --total-steps 50000
    --save-every 2000
    --batch-size 128
    --patience 20
    --ds-batch-size 500
    --npe-samples 100000
    --plot
  )
fi

echo "[$(date '+%F %T')] Starting CNN harmonic-cross run (mode=$MODE, gpu=$GPU, seed=$SEED)"
conda run --no-capture-output -n jaxili python -u \
  scripts/sbi/npe_cnn_nbody_tomo.py \
  "${COMMON[@]}" \
  "${EXTRA[@]}"
echo "[$(date '+%F %T')] Completed CNN harmonic-cross run."
