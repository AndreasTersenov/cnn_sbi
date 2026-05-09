#!/usr/bin/env bash
# A3-norm resnet50: harmonic CNN (10 channels) with per-channel RMS normalization,
# resnet50 backbone, cdim=10.  Complements the plain-arch A3-norm run.
#
# Usage:
#   bash run_a3_norm_resnet50.sh          # seeds 41,42 on GPUs 0,1 + seed 43 on GPU 2 if room
#   bash run_a3_norm_resnet50.sh --smoke  # seed 41 only, 2 steps

set -uo pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
ROOT="$REPO/scripts/sbi/results/exploratory/cnn_with_harm_cross_normalized/resnet50"
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
    --compressor-arch resnet50 \
    --compressor-dim 10 \
    --compressor-dense-width 256 \
    --compressor-train-split train \
    --compressor-val-split test \
    --nde-train-split train \
    --nde-val-split test \
    --seed "$seed" \
    --save-dir "$run_dir/save_params" \
    --cache-dir "$run_dir/cache" \
    --posterior-out "$ROOT/posteriors/cnn_harm_cross_norm_resnet50_nobnt_s${seed}.npy" \
    --figure-out "$ROOT/figures/cnn_harm_cross_norm_resnet50_nobnt_s${seed}.pdf" \
    "${extra_args[@]}" \
    > "$ROOT/logs/s${seed}_gpu${gpu}.log" 2>&1
  local rc=$?
  echo "[$(date -u +%T)] seed=${seed} gpu=${gpu} rc=${rc}"
  return $rc
}

if [[ "$SMOKE" -eq 1 ]]; then
  echo "[$(date -u)] A3-norm resnet50 smoke (seed 41, GPU 0)"
  run_seed 41 0 0.78
  echo "[$(date -u)] Smoke done."
else
  # GPU 0: ~33 GB free after plain-CNN job peak  → 0.78
  # GPU 1: ~36 GB free (no external processes)   → 0.85
  # GPU 2: only ~14 GB free (external blocked)   → skipped; run seed 43 manually when GPU 2 clears
  echo "[$(date -u)] A3-norm resnet50: launching seeds 41, 42 on GPUs 0, 1"
  run_seed 41 0 0.78 & PID0=$!
  run_seed 42 1 0.85 & PID1=$!
  echo "[$(date -u)] PIDs: seed41=$PID0  seed42=$PID1"
  echo "[$(date -u)] NOTE: seed 43 skipped — GPU 2 blocked by external process (~18.5 GB)."
  echo "[$(date -u)]       Run manually when GPU 2 clears:  run_seed 43 2 0.80"
  wait $PID0 $PID1
  echo "[$(date -u)] A3-norm resnet50 (seeds 41+42) complete."
fi
