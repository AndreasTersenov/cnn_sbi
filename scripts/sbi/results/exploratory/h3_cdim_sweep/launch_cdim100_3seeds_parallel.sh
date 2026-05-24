#!/usr/bin/env bash
# H3 test: CNN auto+cross at cdim=100 — does summary-dim explain the gap?
#
# 3 seeds in parallel on GPU 1 (mem fraction 0.30 each — well below the
# iter-108-Q6ON-60k single-run footprint of 0.25). Production config
# matches iter-108-Q6ON-60k EXACTLY except --compressor-dim 10 → 100.
#
# Decision rule (3-seed pooled FoM3, primary metric):
#   pooled >= 28,000  → H3 confirmed (>=15% above the 23,986 anchor)
#   25k – 28k         → partial
#   <= 25,000         → H3 falsified at the summary-dim end
#
# Output: scripts/sbi/results/exploratory/h3_cdim_sweep/

set -uo pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
OUT="$REPO/scripts/sbi/results/exploratory/h3_cdim_sweep"
CACHE="/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"

run_seed() {
    local SEED=$1
    local NAME="cnn_cross_cdim100_s${SEED}"
    local LOG="$OUT/logs/${NAME}.log"

    echo "[start s${SEED}] $(date -u +%FT%TZ)"

    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
    conda run -n jaxili --no-capture-output python "$CNN" \
        --train-compressor \
        --zero-mean-maps \
        --standardize-summary \
        --map-kind nbody \
        --seed ${SEED} \
        --cnn-map-route harmonic \
        --full-sphere-cross-cache "$CACHE" \
        --harmonic-cache-regime nobnt \
        --harmonic-normalize-input-channels \
        --field-size 20 --field-npix 160 \
        --nbins 4 --tomo-bin-indices 1,2,3,4 \
        --compressor-arch plain \
        --compressor-dim 100 \
        --compressor-dense-width 256 \
        --compressor-conv-channels 64,128,256 \
        --compressor-steps 60000 \
        --compressor-batch-size 128 \
        --compressor-lr 0.0005 \
        --compressor-checkpoint-policy best_val \
        --compressor-train-split train \
        --compressor-val-split test \
        --nde-train-split train \
        --nde-val-split test \
        --total-steps 50000 \
        --save-every 500 \
        --batch-size 256 \
        --nvp-layers 8 --nvp-hidden 256 \
        --npe-samples 100000 \
        --save-dir "$OUT/train_s${SEED}" \
        --posterior-out "$OUT/posteriors/${NAME}.npy" \
        --figure-out "$OUT/figures/${NAME}.pdf" \
        --cuda-visible-devices 1 \
        --no-wandb \
        > "$LOG" 2>&1
    echo "[end s${SEED}] $(date -u +%FT%TZ)  rc=$?"
}

# Launch all 3 in parallel.
run_seed 41 &
PID41=$!
sleep 5   # stagger by 5s so JAX initialization doesn't all hit at once
run_seed 42 &
PID42=$!
sleep 5
run_seed 43 &
PID43=$!

echo "[parent] launched 3 seeds in parallel on GPU 1 (mem fraction 0.30 each)"
echo "  pids: s41=$PID41  s42=$PID42  s43=$PID43"

wait $PID41; rc41=$?
wait $PID42; rc42=$?
wait $PID43; rc43=$?

echo "[done] $(date -u +%FT%TZ)"
echo "  s41 rc=$rc41   posterior: $OUT/posteriors/cnn_cross_cdim100_s41.npy"
echo "  s42 rc=$rc42   posterior: $OUT/posteriors/cnn_cross_cdim100_s42.npy"
echo "  s43 rc=$rc43   posterior: $OUT/posteriors/cnn_cross_cdim100_s43.npy"

[[ $rc41 -eq 0 && $rc42 -eq 0 && $rc43 -eq 0 ]]
