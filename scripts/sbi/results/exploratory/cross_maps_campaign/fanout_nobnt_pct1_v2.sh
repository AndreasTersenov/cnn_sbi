#!/usr/bin/env bash
# v2 fanout: cache is already built. Fire s42 on GPU 0 and s43 on GPU 1
# in parallel with the in-flight s41 NPE on GPU 2. Then kill the original
# serial loop (PID 662382) once s41's posterior lands so it doesn't re-run
# s42/s43.
set -euo pipefail

cd /mnt/home/tersenov/software/cnn_sbi
ROOT=scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_cross_nobnt_pct1
LOGS=scripts/sbi/results/exploratory/cross_maps_campaign/logs

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON=(
  --no-wandb --map-kind nbody
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48
  --field-size 20 --field-npix 160
  --nbins 4 --tomo-bin-indices 1,2,3,4
  --zero-mean-maps
  --cross-maps
  --cross-map-apodize cosine
  --cross-map-auto-calibrate-snr
  --cross-snr-percentile 1.0
  --n-scales 5 --l1-nbins 40
  --l1-min-snr -13 --l1-max-snr 13
  --pca-components 0
  --total-steps 5000 --save-every 500 --patience 30
  --batch-size 256 --npe-samples 100000
  --ds-batch-size 96
  --cache-dir "$ROOT/cache"
  --plot
)

run_seed() {
  local seed=$1 gpu=$2 logfile=$3
  local post="$ROOT/posteriors/l1cross_tomo4_20deg160mp_nobnt_p1_s${seed}.npy"
  if [ -f "$post" ]; then
    echo "[fanout v2] seed ${seed} posterior exists, skipping" | tee -a "$logfile"
    return 0
  fi
  {
    echo "===== JAXILI CROSS NOBNT pct1 seed ${seed} (fanout v2, GPU ${gpu}, started $(date '+%H:%M:%S')) ====="
    conda run --no-capture-output -n jaxili python -u \
      scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py \
      --cuda-visible-devices "$gpu" "${COMMON[@]}" \
      --seed "$seed" \
      --save-dir "$ROOT/seed_${seed}" \
      --posterior-out "$post" \
      --figure-out    "$ROOT/figures/l1cross_tomo4_20deg160mp_nobnt_p1_s${seed}.pdf"
    echo "===== JAXILI CROSS NOBNT pct1 seed ${seed} (fanout v2, finished $(date '+%H:%M:%S')) ====="
  } > "$logfile" 2>&1
}

run_seed 42 0 "$LOGS/cross_nobnt_pct1_s42_g0.log" &
PID42=$!
run_seed 43 1 "$LOGS/cross_nobnt_pct1_s43_g1.log" &
PID43=$!
echo "[fanout v2] launched s42 (PID $PID42, GPU 0) and s43 (PID $PID43, GPU 1)"

# When s41 finishes, kill the original serial loop so it doesn't re-fire s42/s43.
S41_POST="$ROOT/posteriors/l1cross_tomo4_20deg160mp_nobnt_p1_s41.npy"
echo "[fanout v2] waiting for s41 posterior: $S41_POST"
until [ -f "$S41_POST" ]; do sleep 10; done
echo "[fanout v2] s41 posterior detected at $(date '+%H:%M:%S'); killing serial loop"

PARENT_PID=662382
if kill -0 "$PARENT_PID" 2>/dev/null; then
  pkill -P "$PARENT_PID" 2>/dev/null || true
  kill "$PARENT_PID" 2>/dev/null || true
  echo "[fanout v2] killed serial loop PID $PARENT_PID"
else
  echo "[fanout v2] serial loop already exited"
fi

wait "$PID42" "$PID43"
echo "[fanout v2] DONE at $(date '+%H:%M:%S')"
