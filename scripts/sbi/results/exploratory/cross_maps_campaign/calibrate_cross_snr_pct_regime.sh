#!/usr/bin/env bash
# Generalized percentile probe for the cross SNR knob, BNT or no-BNT.
# Usage: calibrate_cross_snr_pct_regime.sh <gpu> <pct> <tag> <regime>
#   gpu    : CUDA_VISIBLE_DEVICES index
#   pct    : --cross-snr-percentile value (e.g. 0.1 or 0.5)
#   tag    : subdir suffix under jaxili_cross_<regime>_pct${tag}
#   regime : "bnt" or "nobnt"
#
# Single seed (41), end-to-end, with a fresh cache under
#   jaxili_cross_<regime>_pct${tag}/cache
set -euo pipefail

cd /mnt/home/tersenov/software/cnn_sbi

GPU=$1
PCT=$2
TAG=$3
REGIME=$4
ROOT=scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_cross_${REGIME}_pct${TAG}
mkdir -p "$ROOT/cache" "$ROOT/posteriors" "$ROOT/figures"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
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
  --zero-mean-maps \
  --cross-maps \
  --cross-map-apodize cosine \
  --cross-map-auto-calibrate-snr \
  --cross-snr-percentile "$PCT" \
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
  --posterior-out "$ROOT/posteriors/l1cross_tomo4_20deg160mp_${REGIME}_p${TAG}_s41.npy" \
  --figure-out    "$ROOT/figures/l1cross_tomo4_20deg160mp_${REGIME}_p${TAG}_s41.pdf"
