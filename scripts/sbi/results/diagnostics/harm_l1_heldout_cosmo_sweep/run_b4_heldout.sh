#!/usr/bin/env bash
# B4: held-out cosmology sweep for harmonic-L1 no-BNT calibration check.
#
# Steps:
#   1. Build the filtered L1 cache (5 cosmologies moved train → val).
#   2. Retrain one seed (41) using --precomputed-l1-cache-dir.
#   3. Run SBC rank test on ONLY the held-out cosmologies (targeted calibration).
#
# Usage: bash run_b4_heldout.sh [GPU]   (default GPU=0)
#
# GPU policy: single GPU, XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="${1:-0}"
XLA_FRAC="${2:-0.45}"
SEED=41
N_HELDOUT=5
HELDOUT_SEED=77

CACHE_SCRIPT="$THIS_DIR/build_heldout_cache.py"
HELDOUT_CACHE="$THIS_DIR/heldout_cache"
HELDOUT_EVAL_CACHE="$HELDOUT_CACHE/heldout_eval"
RETRAIN_OUT="$THIS_DIR/retrain_seed${SEED}"
SBC_OUT="$THIS_DIR/sbc_heldout_n200"

FULL_SPHERE_CACHE="$REPO_ROOT/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"

export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_FRAC"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Step 1: build filtered cache (skip if already built) ────────────────────
if [[ -f "$HELDOUT_CACHE/l1_train.npz" ]]; then
    echo "=== B4 Step 1: held-out cache already exists, skipping rebuild ==="
else
    echo "=== B4 Step 1: building held-out L1 cache ==="
    conda run -n jaxili python "$CACHE_SCRIPT" \
        --n-heldout "$N_HELDOUT" \
        --seed "$HELDOUT_SEED" \
        --output-dir "$HELDOUT_CACHE"
fi

# ── Step 2: retrain one seed ─────────────────────────────────────────────────
echo ""
echo "=== B4 Step 2: retraining harmonic-L1 no-BNT seed ${SEED} ==="
mkdir -p "$RETRAIN_OUT/seed_${SEED}" "$RETRAIN_OUT/posteriors"

conda run --no-capture-output -n jaxili python -u \
    "$REPO_ROOT/scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py" \
    --cuda-visible-devices "$GPU" \
    --no-wandb --map-kind nbody \
    --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48 \
    --field-size 20 --field-npix 160 \
    --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --full-sphere-cross-cache "$FULL_SPHERE_CACHE" \
    --precomputed-l1-cache-dir "$HELDOUT_CACHE" \
    --cross-snr-percentile 1.0 \
    --n-scales 5 --l1-nbins 40 \
    --l1-min-snr -13 --l1-max-snr 13 \
    --pca-components 0 \
    --total-steps 5000 --save-every 500 --patience 30 \
    --batch-size 256 --npe-samples 100000 \
    --ds-batch-size 96 \
    --plot \
    --seed "$SEED" \
    --save-dir "$RETRAIN_OUT/seed_${SEED}" \
    --posterior-out "$RETRAIN_OUT/posteriors/l1cross_harm_nobnt_heldout_s${SEED}.npy"

# ── Step 3: SBC on held-out cosmologies only ─────────────────────────────────
echo ""
echo "=== B4 Step 3: SBC rank test on held-out cosmologies (N=200) ==="
CHECKPOINT_ROOT="$RETRAIN_OUT/seed_${SEED}/l1norm_cross_jaxili/nbody/params_l1norm_cross_jaxili"
PREPROC_STATS="$RETRAIN_OUT/seed_${SEED}/l1norm_cross_jaxili/nbody/l1_cross_jaxili_standardization.npz"
FEATURE_MASK="$RETRAIN_OUT/seed_${SEED}/l1norm_cross_jaxili/nbody/l1_cross_jaxili_feature_mask.npz"

conda run -n jaxili python \
    "$REPO_ROOT/scripts/sbi/run_sbc_harm_l1_nobnt.py" \
    --cache-dir "$HELDOUT_EVAL_CACHE" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --preprocessing-stats "$PREPROC_STATS" \
    --feature-mask "$FEATURE_MASK" \
    --output-root "$SBC_OUT" \
    --n-ranks 200 \
    --posterior-samples 2000 \
    --rank-seed 20260507 \
    --cuda-visible-devices "$GPU" \
    --xla-mem-fraction "$XLA_FRAC"

echo ""
echo "=== B4 complete. SBC outputs in $SBC_OUT ==="
echo "  Compare sigma_8 rank histogram to sbc_cnn_nobnt for calibration check."
