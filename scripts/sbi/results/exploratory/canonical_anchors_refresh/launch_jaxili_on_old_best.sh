#!/usr/bin/env bash
# Run jaxili NPE on the old best CNN setups (train/train, no 70/30 split).
# Two arms:
#   1. Auto+cross (iter-108-Q6ON-60k compressor, harmonic route, 10 ch)
#   2. Auto-only (apples_v_iter108_autoonly compressor, TFDS route, 4 ch)
#
# Both use pre-cached compressed data (cnn_train.npz/cnn_val.npz).
# We only need to compress the obs map, then feed everything to jaxili.

set -uo pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
NDE_TEST="$REPO/scripts/sbi/results/exploratory/canonical_anchors_refresh/run_cnn_jaxili_nde_test.py"
OUT="$REPO/scripts/sbi/results/exploratory/canonical_anchors_refresh/nde_swap_test"
HARMONIC_CACHE="$REPO/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"

SEED=41
stamp() { date -u +%FT%TZ; }

echo "[start] $(stamp) — jaxili NPE on old-best CNN compressors"

############################################################
# Arm 1: iter-108 auto+cross (FoM3 ~24k with RealNVP)
############################################################
ITER108="/nas/tersenov/claude-notes/runs/cnn-auto-cross-push-18-20-2026/iter-108-Q6ON-60k"
ITER108_COMP="$ITER108/compressor/vmim/nbody/sigma_0.26/gal_density_30/bin_4/harmonic_nobnt_ch10"
ITER108_CACHE="$ITER108/cache/eval"
ARM1_OUT="$OUT/iter108_autocross_jaxili_s${SEED}"

echo
echo "[arm1: iter-108 auto+cross] $(stamp)"
echo "  Compressor: $ITER108_COMP/params_nd_compressor_batch60000.pkl (last_step)"
echo "  Cached data: $ITER108_CACHE (302k train, 134k val)"

# Step 1a: Compress just the obs map using the iter-108 compressor
# Use --exit-after-compress but point --cache-dir to a temp dir that has a copy
# of the existing cached data + will receive the obs.
CACHE_ARM1="$OUT/cache_iter108_autocross"
mkdir -p "$CACHE_ARM1"
# Symlink existing cached compressed data + meta
ln -sf "$ITER108_CACHE/cnn_train.npz" "$CACHE_ARM1/cnn_train.npz" 2>/dev/null
ln -sf "$ITER108_CACHE/cnn_val.npz" "$CACHE_ARM1/cnn_val.npz" 2>/dev/null
ln -sf "$ITER108_CACHE/cnn_cache_meta.npz" "$CACHE_ARM1/cnn_cache_meta.npz" 2>/dev/null

# Compress obs map
echo "  Compressing obs map..."
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
conda run -n jaxili --no-capture-output python "$CNN" \
    --exit-after-compress \
    --zero-mean-maps \
    --standardize-summary \
    --map-kind nbody \
    --seed ${SEED} \
    --cnn-map-route harmonic \
    --full-sphere-cross-cache "$HARMONIC_CACHE" \
    --harmonic-cache-regime nobnt \
    --harmonic-normalize-input-channels \
    --field-size 20 --field-npix 160 \
    --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --compressor-arch plain \
    --compressor-dim 10 \
    --compressor-dense-width 256 \
    --compressor-conv-channels 64,128,256 \
    --compressor-checkpoint-policy last_step \
    --compressor-params "$ITER108_COMP/params_nd_compressor_batch60000.pkl" \
    --compressor-state "$ITER108_COMP/opt_state_resnet_batch60000.pkl" \
    --compressor-train-split train \
    --compressor-val-split val \
    --nde-train-split train \
    --nde-val-split val \
    --save-dir "$ARM1_OUT/train" \
    --cache-dir "$CACHE_ARM1" \
    --cuda-visible-devices 1 --no-wandb \
    > "$OUT/logs/iter108_autocross_compress_s${SEED}.log" 2>&1
rc1a=$?
echo "  Compress rc=$rc1a"

if [ $rc1a -ne 0 ]; then
    echo "  FAILED — check $OUT/logs/iter108_autocross_compress_s${SEED}.log"
    # Continue to arm 2 anyway
else
    # Step 1b: Run jaxili NPE (default hidden [50,50] — same as L1)
    mkdir -p "$ARM1_OUT"
    echo "  Running jaxili NPE..."
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    conda run -n jaxili --no-capture-output python "$NDE_TEST" \
        --cache-dir "$CACHE_ARM1" \
        --seed ${SEED} \
        --out-dir "$ARM1_OUT" \
        --epochs 50000 \
        --batch-size 256 \
        --learning-rate 1e-4 \
        --maf-hidden 50,50 \
        --maf-layers 5 \
        --patience 100 \
        --min-delta 0.0005 \
        --npe-samples 100000 \
        --cuda-visible-devices 1 \
        > "$OUT/logs/iter108_autocross_jaxili_s${SEED}.log" 2>&1
    rc1b=$?
    echo "[arm1 done] $(stamp)  rc=$rc1b"
fi

############################################################
# Arm 2: auto-only (FoM3 ~18.6k with RealNVP)
############################################################
AUTOONLY="/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/apples_v_iter108_autoonly"
AUTOONLY_COMP="$AUTOONLY/compressor/vmim/nbody/sigma_0.26/gal_density_30/bin_4"
AUTOONLY_CACHE_SRC="$AUTOONLY/cache"
ARM2_OUT="$OUT/autoonly_jaxili_s${SEED}"

echo
echo "[arm2: auto-only] $(stamp)"
echo "  Compressor: $AUTOONLY_COMP/params_nd_compressor_best_val.pkl"
echo "  Cached data: $AUTOONLY_CACHE_SRC (157k train, 70k val)"

# Symlink cached data
CACHE_ARM2="$OUT/cache_autoonly"
mkdir -p "$CACHE_ARM2"
ln -sf "$AUTOONLY_CACHE_SRC/cnn_train.npz" "$CACHE_ARM2/cnn_train.npz" 2>/dev/null
ln -sf "$AUTOONLY_CACHE_SRC/cnn_val.npz" "$CACHE_ARM2/cnn_val.npz" 2>/dev/null
ln -sf "$AUTOONLY_CACHE_SRC/cnn_cache_meta.npz" "$CACHE_ARM2/cnn_cache_meta.npz" 2>/dev/null

# Compress obs map (TFDS route, 4 channels)
echo "  Compressing obs map..."
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
conda run -n jaxili --no-capture-output python "$CNN" \
    --exit-after-compress \
    --zero-mean-maps \
    --standardize-summary \
    --map-kind nbody \
    --seed ${SEED} \
    --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px \
    --field-size 20 --field-npix 160 \
    --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --compressor-arch plain \
    --compressor-dim 10 \
    --compressor-dense-width 256 \
    --compressor-conv-channels 64,128,256 \
    --compressor-checkpoint-policy best_val \
    --compressor-params "$AUTOONLY_COMP/params_nd_compressor_best_val.pkl" \
    --compressor-state "$AUTOONLY_COMP/opt_state_resnet_best_val.pkl" \
    --compressor-train-split train \
    --compressor-val-split test \
    --nde-train-split train \
    --nde-val-split test \
    --save-dir "$ARM2_OUT/train" \
    --cache-dir "$CACHE_ARM2" \
    --cuda-visible-devices 1 --no-wandb \
    > "$OUT/logs/autoonly_compress_s${SEED}.log" 2>&1
rc2a=$?
echo "  Compress rc=$rc2a"

if [ $rc2a -ne 0 ]; then
    echo "  FAILED — check $OUT/logs/autoonly_compress_s${SEED}.log"
else
    mkdir -p "$ARM2_OUT"
    echo "  Running jaxili NPE..."
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    conda run -n jaxili --no-capture-output python "$NDE_TEST" \
        --cache-dir "$CACHE_ARM2" \
        --seed ${SEED} \
        --out-dir "$ARM2_OUT" \
        --epochs 50000 \
        --batch-size 256 \
        --learning-rate 1e-4 \
        --maf-hidden 50,50 \
        --maf-layers 5 \
        --patience 100 \
        --min-delta 0.0005 \
        --npe-samples 100000 \
        --cuda-visible-devices 1 \
        > "$OUT/logs/autoonly_jaxili_s${SEED}.log" 2>&1
    rc2b=$?
    echo "[arm2 done] $(stamp)  rc=$rc2b"
fi

echo
echo "[ALL DONE] $(stamp)"
echo "Results:"
echo "  Arm 1 (iter-108 auto+cross): $ARM1_OUT"
echo "  Arm 2 (auto-only):           $ARM2_OUT"
