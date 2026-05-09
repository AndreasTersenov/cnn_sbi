#!/usr/bin/env bash
# A3-norm: harmonic CNN (4 auto + 6 cross channels) with per-channel RMS normalization.
#
# Identical to A3 (cnn_with_harm_cross) except for --harmonic-normalize-input-channels.
# Seeds 41, 42, 43 in parallel on GPUs 0, 1, 2.
#
# Usage:
#   bash run_a3_norm.sh            # 3 seeds, full run
#   bash run_a3_norm.sh --smoke    # smoke test, 1 realization, 2 steps

set -uo pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
ROOT="$REPO/scripts/sbi/results/exploratory/cnn_with_harm_cross_normalized"
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
HARM_CACHE="$REPO/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"

SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false

run_seed() {
  local seed=$1 gpu=$2 mem_frac=$3
  local run_dir="$ROOT/seed_${seed}"
  mkdir -p "$run_dir/save_params" "$run_dir/cache"

  local extra_args=()
  if [[ "$SMOKE" -eq 1 ]]; then
    extra_args+=(
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
    extra_args+=(
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

  XLA_PYTHON_CLIENT_MEM_FRACTION="$mem_frac" \
  conda run --no-capture-output -n jaxili python -u "$CNN" \
    --cuda-visible-devices "$gpu" \
    --no-wandb \
    --map-kind nbody \
    --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48 \
    --field-size 20 \
    --field-npix 160 \
    --nbins 4 \
    --tomo-bin-indices 1,2,3,4 \
    --full-sphere-cross-cache "$HARM_CACHE" \
    --cnn-map-route harmonic \
    --harmonic-cache-regime nobnt \
    --harmonic-normalize-input-channels \
    --train-compressor \
    --compressor-arch plain \
    --compressor-dim 10 \
    --compressor-dense-width 256 \
    --compressor-train-split train \
    --compressor-val-split test \
    --nde-train-split train \
    --nde-val-split test \
    --seed "$seed" \
    --save-dir "$run_dir/save_params" \
    --cache-dir "$run_dir/cache" \
    --posterior-out "$ROOT/posteriors/cnn_harm_cross_norm_nobnt_s${seed}.npy" \
    --figure-out "$ROOT/figures/cnn_harm_cross_norm_nobnt_s${seed}.pdf" \
    "${extra_args[@]}" \
    > "$ROOT/logs/s${seed}_gpu${gpu}.log" 2>&1
  local rc=$?
  echo "[$(date -u +%T)] seed=${seed} gpu=${gpu} rc=${rc}"
  return $rc
}

if [[ "$SMOKE" -eq 1 ]]; then
  echo "[$(date -u)] A3-norm smoke test (seed 41 only, GPU 0)"
  run_seed 41 0 0.48
  echo "[$(date -u)] Smoke done."
else
  # GPU 0: ~21 GB free (external PCA job + A4 job running)  → 0.48
  # GPU 1: ~24 GB free (A4 job only)                        → 0.55
  # GPU 2: ~19 GB free (external app.py + start_dec.py)     → 0.42
  echo "[$(date -u)] A3-norm: launching seeds 41, 42, 43 on GPUs 0, 1, 2"
  run_seed 41 0 0.48 & PID0=$!
  run_seed 42 1 0.55 & PID1=$!
  run_seed 43 2 0.42 & PID2=$!
  echo "[$(date -u)] PIDs: seed41=$PID0  seed42=$PID1  seed43=$PID2"
  wait $PID0 $PID1 $PID2
  echo "[$(date -u)] A3-norm complete."
fi
