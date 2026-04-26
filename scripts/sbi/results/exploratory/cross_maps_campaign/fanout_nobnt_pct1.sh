#!/usr/bin/env bash
# Watcher: waits for the NOBNT-pct1 L1 cache to exist, then fans seeds
# 42 and 43 onto GPU 3 in parallel with the in-flight s41 NPE on GPU 2.
# Once both finish (or s41 finishes naturally), kill the original serial loop
# so it doesn't re-run s42/s43.
set -euo pipefail

cd /mnt/home/tersenov/software/cnn_sbi
ROOT=scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_cross_nobnt_pct1
LOGS=scripts/sbi/results/exploratory/cross_maps_campaign/logs
META="$ROOT/cache/l1_cache_meta.npz"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[fanout] waiting for cache: $META"
until [ -f "$META" ]; do sleep 10; done
echo "[fanout] cache ready at $(date '+%H:%M:%S'); launching s42 and s43 on GPU 3"

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

# s42 first on GPU 3 (background), then s43 on GPU 3 sequentially.
(
  for s in 42 43; do
    POST="$ROOT/posteriors/l1cross_tomo4_20deg160mp_nobnt_p1_s${s}.npy"
    if [ -f "$POST" ]; then
      echo "[fanout] seed ${s} posterior already exists, skipping"
      continue
    fi
    echo "===== JAXILI CROSS NOBNT pct1 seed ${s} (fanout, GPU 3, started $(date '+%H:%M:%S')) ====="
    conda run --no-capture-output -n jaxili python -u \
      scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py \
      --cuda-visible-devices 3 "${COMMON[@]}" \
      --seed $s \
      --save-dir "$ROOT/seed_${s}" \
      --posterior-out "$POST" \
      --figure-out    "$ROOT/figures/l1cross_tomo4_20deg160mp_nobnt_p1_s${s}.pdf"
    echo "===== JAXILI CROSS NOBNT pct1 seed ${s} (fanout, finished $(date '+%H:%M:%S')) ====="
  done
) > "$LOGS/cross_nobnt_pct1_fanout.log" 2>&1 &
FANOUT_PID=$!
echo "[fanout] fanout subshell PID=$FANOUT_PID"

# When s41's posterior lands, kill the original serial loop so it doesn't
# try to re-run s42/s43.
S41_POST="$ROOT/posteriors/l1cross_tomo4_20deg160mp_nobnt_p1_s41.npy"
echo "[fanout] waiting for s41 posterior: $S41_POST"
until [ -f "$S41_POST" ]; do sleep 10; done
echo "[fanout] s41 posterior detected at $(date '+%H:%M:%S'); killing serial loop"

# Kill the original serial bash script (PID 662382) and any python child.
PARENT_PID=662382
if kill -0 "$PARENT_PID" 2>/dev/null; then
  pkill -P "$PARENT_PID" 2>/dev/null || true
  kill "$PARENT_PID" 2>/dev/null || true
  echo "[fanout] killed original serial loop PID $PARENT_PID"
else
  echo "[fanout] serial loop already exited"
fi

wait "$FANOUT_PID"
echo "[fanout] DONE at $(date '+%H:%M:%S')"
