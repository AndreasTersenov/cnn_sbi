#!/usr/bin/env bash
# Add seed 41 to the May 8 resnet50 (stock-BN) cdim=20 auto-only sweep.
#
# Existing on disk:
#   - cnn_resnet50_zm_nobnt_cdim20_s42.npy (FoM3 31,684; bias w0 -1.01σ)
#   - cnn_resnet50_zm_nobnt_cdim20_s43.npy (FoM3 23,652; bias σ8 -1.53σ)
#
# Goal: 3-seed pooled FoM3 + bias-vs-truth pattern to disambiguate
#   (A) stock-BN running-stats leakage producing tight-but-biased posteriors
#   (B) genuine deeper-arch advantage on auto-only convergence (Q1 reopened).
#
# Config matches run_a4_resnet50_sweep.py exactly (resnet50 stock-BN, cdim=20,
# dense=256, 120k compressor steps, default LR/batch, val_split=test). The
# only addition vs May 8 is the explicit --compressor-checkpoint-policy
# last_step flag — equivalent to May 8 implicit behavior.
#
# Resource policy: GPU 1 sole tenant.

set -uo pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
OUT="$REPO/scripts/sbi/results/exploratory/cnn_resnet50_zm_sweep"

mkdir -p "$OUT"/{posteriors,logs,figures,resnet50_cdim20_s41}

echo "[start] $(date -u +%FT%TZ) — resnet50 stock-BN cdim=20 seed 41 @120k on GPU 1"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
conda run -n jaxili --no-capture-output python "$CNN" \
    --train-compressor \
    --zero-mean-maps \
    --map-kind nbody \
    --seed 41 \
    --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px \
    --field-size 20 --field-npix 160 \
    --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --compressor-arch resnet50 \
    --compressor-dim 20 \
    --compressor-dense-width 256 \
    --compressor-steps 120000 \
    --compressor-checkpoint-policy last_step \
    --compressor-train-split train \
    --compressor-val-split test \
    --nde-train-split train \
    --nde-val-split test \
    --total-steps 10000 \
    --save-every 500 \
    --npe-samples 100000 \
    --ds-batch-size 500 \
    --save-dir "$OUT/resnet50_cdim20_s41" \
    --posterior-out "$OUT/posteriors/cnn_resnet50_zm_nobnt_cdim20_s41.npy" \
    --figure-out "$OUT/figures/cnn_resnet50_zm_nobnt_cdim20_s41.pdf" \
    --cuda-visible-devices 1 \
    --no-wandb \
    > "$OUT/logs/resnet50_cdim20_s41.log" 2>&1

rc=$?
echo "[end] $(date -u +%FT%TZ) — rc=$rc"
echo "posterior at: $OUT/posteriors/cnn_resnet50_zm_nobnt_cdim20_s41.npy"
exit $rc
