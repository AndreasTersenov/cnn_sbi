#!/usr/bin/env bash
# Auto + cross L1-norms, BNT, zero-mean-maps, jaxili NPE — pct=1 SNR knob.
#
# Same config as launch_jaxili_cross_bnt.sh, except:
#   --cross-map-auto-calibrate-snr --cross-snr-percentile 1.0
# Cache lives under jaxili_cross_bnt_pct1/ (separate from min/max-calibrated
# cache under jaxili_cross_bnt/).
#
# Seed 41 was completed by calibrate_cross_snr_pct.sh (FoM3=875.67); this
# script picks up seeds 42 and 43, reusing the cache built for seed 41.
set -euo pipefail

cd /mnt/home/tersenov/software/cnn_sbi

ROOT=scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_cross_bnt_pct1
mkdir -p "$ROOT/cache" "$ROOT/posteriors" "$ROOT/figures"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON=(
  --cuda-visible-devices 3
  --no-wandb --map-kind nbody
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48
  --field-size 20 --field-npix 160
  --nbins 4 --tomo-bin-indices 1,2,3,4
  --apply-bnt
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

for s in 42 43; do
  echo "===== JAXILI CROSS BNT pct1 seed ${s} (started $(date '+%Y-%m-%d %H:%M:%S')) ====="
  conda run --no-capture-output -n jaxili python -u \
    scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py "${COMMON[@]}" \
    --seed $s \
    --save-dir "$ROOT/seed_${s}" \
    --posterior-out "$ROOT/posteriors/l1cross_tomo4_20deg160mp_bnt_p1_s${s}.npy" \
    --figure-out    "$ROOT/figures/l1cross_tomo4_20deg160mp_bnt_p1_s${s}.pdf"
  echo "===== JAXILI CROSS BNT pct1 seed ${s} (finished $(date '+%Y-%m-%d %H:%M:%S')) ====="
done
echo "===== JAXILI CROSS BNT pct1 ALL SEEDS DONE ($(date '+%Y-%m-%d %H:%M:%S')) ====="
