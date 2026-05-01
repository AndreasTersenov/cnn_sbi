#!/usr/bin/env bash
# Single-arm runner for the harmonic-cross campaign.
# Usage: run_harmonic_arm.sh <gpu> <regime> <seed>
#   gpu     : CUDA_VISIBLE_DEVICES index (0/1/2)
#   regime  : "bnt" or "nobnt"
#   seed    : RNG seed (41/42/43)
#
# Mirrors jaxili_cross_*_pct1 settings exactly, but pulls 10-channel patches
# from the precomputed full-sphere harmonic cache instead of computing them
# on the flat-sky tomographic patches at runtime.
set -euo pipefail

cd /mnt/home/tersenov/software/cnn_sbi

GPU=$1
REGIME=$2
SEED=$3

CACHE_DIR=scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid
ROOT=scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_harm_cross_${REGIME}
RUN_DIR=$ROOT/seed_${SEED}
L1_CACHE=$ROOT/l1_cache_seed${SEED}

mkdir -p "$RUN_DIR" "$L1_CACHE" "$ROOT/posteriors" "$ROOT/figures"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXTRA=()
if [ "$REGIME" = "bnt" ]; then
  EXTRA=(--apply-bnt)
fi

conda run --no-capture-output -n jaxili python -u \
  scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py \
  --cuda-visible-devices "$GPU" \
  --no-wandb --map-kind nbody \
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48 \
  --field-size 20 --field-npix 160 \
  --nbins 4 --tomo-bin-indices 1,2,3,4 \
  "${EXTRA[@]}" \
  --full-sphere-cross-cache "$CACHE_DIR" \
  --cross-snr-percentile 1.0 \
  --n-scales 5 --l1-nbins 40 \
  --l1-min-snr -13 --l1-max-snr 13 \
  --pca-components 0 \
  --total-steps 5000 --save-every 500 --patience 30 \
  --batch-size 256 --npe-samples 100000 \
  --ds-batch-size 96 \
  --cache-dir "$L1_CACHE" \
  --plot \
  --seed "$SEED" \
  --save-dir "$RUN_DIR" \
  --posterior-out "$ROOT/posteriors/l1cross_tomo4_20deg160mp_harm_${REGIME}_p1_s${SEED}.npy" \
  --figure-out    "$ROOT/figures/l1cross_tomo4_20deg160mp_harm_${REGIME}_p1_s${SEED}.pdf"
