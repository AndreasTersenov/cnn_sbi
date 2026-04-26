#!/usr/bin/env bash
# One-shot calibration probe: BNT seed 41 with --cross-map-auto-calibrate-snr.
# Produces a saved snr_calibration.npz with the empirical cross-channel SNR
# percentiles; the main BNT/no-BNT campaigns then consume that range
# explicitly via --cross-map-{min,max}-snr for apples-to-apples parity.
set -euo pipefail

cd /mnt/home/tersenov/software/cnn_sbi

ROOT=scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_cross_bnt
mkdir -p "$ROOT/cache" "$ROOT/posteriors" "$ROOT/figures"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda run --no-capture-output -n jaxili python -u \
  scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py \
  --cuda-visible-devices 0 \
  --no-wandb --map-kind nbody \
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48 \
  --field-size 20 --field-npix 160 \
  --nbins 4 --tomo-bin-indices 1,2,3,4 \
  --apply-bnt \
  --zero-mean-maps \
  --cross-maps \
  --cross-map-apodize cosine \
  --cross-map-auto-calibrate-snr \
  --n-scales 5 --l1-nbins 40 \
  --l1-min-snr -13 --l1-max-snr 13 \
  --pca-components 0 \
  --total-steps 5000 --save-every 500 --patience 30 \
  --batch-size 256 --npe-samples 100000 \
  --ds-batch-size 96 \
  --cache-dir "$ROOT/cache" \
  --plot \
  --seed 41 \
  --save-dir "$ROOT/seed_41" \
  --posterior-out "$ROOT/posteriors/l1cross_tomo4_20deg160mp_bnt_s41.npy" \
  --figure-out    "$ROOT/figures/l1cross_tomo4_20deg160mp_bnt_s41.pdf"
