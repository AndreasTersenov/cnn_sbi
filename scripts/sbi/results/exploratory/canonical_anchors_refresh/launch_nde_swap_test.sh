#!/usr/bin/env bash
# NDE-swap test: CNN compressor + jaxili MAF NDE
# Tests whether the NDE architecture (RealNVP vs MAF) explains the CNN FoM3 gap.
#
# Phase 1: re-compress with the canonical compressor (no training, just forward pass)
# Phase 2: run jaxili NPE on the compressed summaries
#
# GPU 1 only. ~15-20 min total.

set -uo pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
NDE_TEST="$REPO/scripts/sbi/results/exploratory/canonical_anchors_refresh/run_cnn_jaxili_nde_test.py"
OUT="$REPO/scripts/sbi/results/exploratory/canonical_anchors_refresh"
CACHE="$REPO/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"

# Compressor checkpoint from the canonical CNN cross run (iter-1, completed before crash)
COMP_DIR="$OUT/train/cnn_cross_canon_s41/vmim/nbody/sigma_0.26/gal_density_30/bin_4/harmonic_nobnt_ch10"

SEED=41

stamp() { date -u +%FT%TZ; }
echo "[start] $(stamp) — NDE-swap test (CNN compressor + jaxili MAF)"

############################################################
# Phase 1: Compress dataset with existing compressor
############################################################
CACHE_S41="$OUT/nde_swap_test/cache_s${SEED}"
mkdir -p "$CACHE_S41"

echo
echo "[phase 1] $(stamp) — Compressing datasets with canonical compressor"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
conda run -n jaxili --no-capture-output python "$CNN" \
    --exit-after-compress \
    --zero-mean-maps \
    --standardize-summary \
    --map-kind nbody \
    --seed ${SEED} \
    --cnn-map-route harmonic \
    --full-sphere-cross-cache "$CACHE" \
    --harmonic-cache-regime nobnt \
    --harmonic-normalize-input-channels \
    --field-size 20 --field-npix 160 \
    --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --compressor-arch plain \
    --compressor-dim 10 \
    --compressor-dense-width 256 \
    --compressor-conv-channels 64,128,256 \
    --compressor-checkpoint-policy best_val \
    --compressor-params "$COMP_DIR/params_nd_compressor_best_val.pkl" \
    --compressor-state "$COMP_DIR/opt_state_resnet_best_val.pkl" \
    --compressor-train-split 'train[:70%]' \
    --compressor-val-split val \
    --nde-train-split 'train[70%:]' \
    --nde-val-split val \
    --require-disjoint-train-examples \
    --save-dir "$OUT/nde_swap_test/train_s${SEED}" \
    --cache-dir "$CACHE_S41" \
    --cuda-visible-devices 1 --no-wandb \
    > "$OUT/logs/nde_swap_compress_s${SEED}.log" 2>&1
rc_phase1=$?
echo "[phase 1 done] $(stamp)  rc=$rc_phase1"

if [ $rc_phase1 -ne 0 ]; then
    echo "Phase 1 FAILED. Check $OUT/logs/nde_swap_compress_s${SEED}.log"
    exit 1
fi

# Verify cache files exist
for f in cnn_train.npz cnn_val.npz cnn_obs.npz cnn_cache_meta.npz; do
    if [ ! -f "$CACHE_S41/$f" ]; then
        echo "FATAL: Missing $CACHE_S41/$f after Phase 1"
        exit 1
    fi
done
echo "  Cache verified: $(ls $CACHE_S41/*.npz | wc -l) NPZ files"

############################################################
# Phase 2: jaxili NPE on compressed summaries
############################################################
NDE_OUT="$OUT/nde_swap_test/results_s${SEED}"
mkdir -p "$NDE_OUT"

echo
echo "[phase 2] $(stamp) — jaxili MAF NDE on CNN compressed summaries"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
conda run -n jaxili --no-capture-output python "$NDE_TEST" \
    --cache-dir "$CACHE_S41" \
    --seed ${SEED} \
    --out-dir "$NDE_OUT" \
    --epochs 50000 \
    --batch-size 256 \
    --learning-rate 1e-4 \
    --npe-samples 100000 \
    --cuda-visible-devices 1 \
    > "$OUT/logs/nde_swap_jaxili_s${SEED}.log" 2>&1
rc_phase2=$?
echo "[phase 2 done] $(stamp)  rc=$rc_phase2"

if [ $rc_phase2 -ne 0 ]; then
    echo "Phase 2 FAILED. Check $OUT/logs/nde_swap_jaxili_s${SEED}.log"
    exit 1
fi

echo
echo "[ALL DONE] $(stamp)"
echo "Results in: $NDE_OUT"
echo "Logs:       $OUT/logs/nde_swap_*.log"
ls -la "$NDE_OUT/"
