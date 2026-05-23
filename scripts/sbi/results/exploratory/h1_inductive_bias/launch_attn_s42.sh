#!/usr/bin/env bash
# s42 of the H1 attention-arm 3-seed pool. Mirror of launch_attn_s41.sh.
set -uo pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
OUT="$REPO/scripts/sbi/results/exploratory/h1_inductive_bias"

mkdir -p "$OUT"/{posteriors,logs,figures,train_s42}

SEED=42
NAME_STEM="cnn_attn_auto_s${SEED}"
LOG="$OUT/logs/${NAME_STEM}.log"

echo "[start] $(date -u +%FT%TZ) — plain_attn (L=1, H=4) auto-only seed ${SEED} on GPU 1"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
conda run -n jaxili --no-capture-output python "$CNN" \
    --train-compressor \
    --zero-mean-maps \
    --standardize-summary \
    --map-kind nbody \
    --seed ${SEED} \
    --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px \
    --field-size 20 --field-npix 160 \
    --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --compressor-arch plain_attn \
    --compressor-dim 10 \
    --compressor-dense-width 256 \
    --compressor-conv-channels 64,128,256 \
    --compressor-steps 60000 \
    --compressor-batch-size 128 \
    --compressor-lr 0.0005 \
    --compressor-checkpoint-policy best_val \
    --attn-layers 1 \
    --attn-heads 4 \
    --attn-mlp-mult 4 \
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
    --posterior-out "$OUT/posteriors/${NAME_STEM}.npy" \
    --figure-out "$OUT/figures/${NAME_STEM}.pdf" \
    --cuda-visible-devices 1 \
    --no-wandb \
    > "$LOG" 2>&1

rc=$?
echo "[end] $(date -u +%FT%TZ) — rc=$rc"
exit $rc
