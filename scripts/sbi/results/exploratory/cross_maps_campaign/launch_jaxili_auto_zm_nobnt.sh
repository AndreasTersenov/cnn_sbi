#!/usr/bin/env bash
# Auto-only L1, NO BNT, --zero-mean-maps, multipatch — matched baseline for
# the cross-maps comparison (no-BNT regime).
# 3 seeds, GPU 3. Cache shared across seeds.
set -euo pipefail

cd /mnt/home/tersenov/software/cnn_sbi

ROOT=scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_auto_zm_nobnt
mkdir -p "$ROOT/cache" "$ROOT/posteriors" "$ROOT/figures"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON=(
  --cuda-visible-devices 3
  --no-wandb --map-kind nbody
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48
  --field-size 20 --field-npix 160
  --nbins 4 --tomo-bin-indices 1,2,3,4
  --zero-mean-maps
  --no-cross-maps
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
  echo "===== JAXILI AUTO-ZM NOBNT seed ${s} (started $(date '+%Y-%m-%d %H:%M:%S')) ====="
  conda run --no-capture-output -n jaxili python -u \
    scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py "${COMMON[@]}" \
    --seed $s \
    --save-dir "$ROOT/seed_${s}" \
    --posterior-out "$ROOT/posteriors/l1_tomo4_20deg160mp_zm_nobnt_s${s}.npy" \
    --figure-out    "$ROOT/figures/l1_tomo4_20deg160mp_zm_nobnt_s${s}.pdf"
  echo "===== JAXILI AUTO-ZM NOBNT seed ${s} (finished $(date '+%Y-%m-%d %H:%M:%S')) ====="
done
echo "===== JAXILI AUTO-ZM NOBNT ALL SEEDS DONE ($(date '+%Y-%m-%d %H:%M:%S')) ====="
