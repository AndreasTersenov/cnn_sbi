#!/usr/bin/env bash
# Auto-only, no-BNT, jaxili NPE — exact reference config (run_bnt_tomo4_study.py).
# 3 seeds on GPU 1. Cache shared across seeds.
set -euo pipefail

cd /mnt/home/tersenov/software/cnn_sbi

ROOT=scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_auto_nobnt
mkdir -p "$ROOT/cache" "$ROOT/posteriors" "$ROOT/figures"

COMMON=(
  --cuda-visible-devices 3
  --no-wandb --map-kind nbody
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px
  --field-size 20 --field-npix 160
  --nbins 4 --tomo-bin-indices 1,2,3,4
  --n-scales 5 --l1-nbins 40
  --l1-min-snr -13 --l1-max-snr 13
  --pca-components 0
  --total-steps 5000 --save-every 500 --patience 30
  --batch-size 256 --npe-samples 100000
  --ds-batch-size 96
  --cache-dir "$ROOT/cache"
  --plot
)

for s in 41 42 43; do
  echo "===== JAXILI AUTO NOBNT seed ${s} (started $(date '+%Y-%m-%d %H:%M:%S')) ====="
  conda run --no-capture-output -n jaxili python -u \
    scripts/sbi/npe_l1norm_jaxili_nbody_tomo.py "${COMMON[@]}" \
    --seed $s \
    --save-dir "$ROOT/seed_${s}" \
    --posterior-out "$ROOT/posteriors/l1_tomo4_20deg160_nobnt_s${s}.npy" \
    --figure-out    "$ROOT/figures/l1_tomo4_20deg160_nobnt_s${s}.pdf"
  echo "===== JAXILI AUTO NOBNT seed ${s} (finished $(date '+%Y-%m-%d %H:%M:%S')) ====="
done
echo "===== JAXILI AUTO NOBNT ALL SEEDS DONE ($(date '+%Y-%m-%d %H:%M:%S')) ====="
