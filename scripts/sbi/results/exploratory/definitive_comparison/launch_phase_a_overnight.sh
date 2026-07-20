#!/usr/bin/env bash
# Phase A (partial): Train 2 RealNVP-companion compressors overnight.
#
# Compressor 1: auto-only (TFDS route, 4 channels) — GPU 0
# Compressor 2: auto+cross (harmonic cache, 10 channels) — GPU 1
#
# Both use: plain CNN, 64,128,256, dense=256, cdim=10, 80k steps, best-val,
# save-every=1000, zero-mean-maps, compressor-train-split=train[:70%].
#
# GPU 0 + GPU 1 in parallel. ~3.5h each = ~3.5h wall time.
# Output: compressors/{auto_rnvp,autocross_rnvp}/
#
# After completion, the compressors are ready for Phase A.5 (compression)
# and Phase B (NDE training). The MAF-companion compressors (arms 7-8, 10)
# need code changes first and will be trained in a follow-up.

set -uo pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
OUT="$REPO/scripts/sbi/results/exploratory/definitive_comparison"
CACHE="$REPO/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
TFDS=NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48

# Common flags (GPU set per-compressor, not here)
COMMON_CNN_FLAGS=(
    --zero-mean-maps
    --map-kind nbody
    --seed 42
    --field-size 20 --field-npix 160
    --nbins 4 --tomo-bin-indices 1,2,3,4
    --compressor-arch plain
    --compressor-dim 10
    --compressor-dense-width 256
    --compressor-conv-channels 64,128,256
    --compressor-steps 80000
    --compressor-batch-size 128
    --compressor-lr 0.0005
    --compressor-save-every 2000
    --compressor-checkpoint-policy best_val
    --compressor-train-split 'train[:70%]'
    --compressor-val-split test
    --nde-train-split 'train[70%:]'
    --nde-val-split test
    --require-disjoint-train-examples
    --npe-samples 100000
    --no-wandb
)

stamp() { date -u +%FT%TZ; }
mkdir -p "$OUT/logs"

############################################################
# SMOKE TEST (500 steps, ~5 min, catches crashes early)
############################################################
echo "================================================================"
echo "SMOKE TEST: 500-step compressor training (both GPUs)"
echo "Started: $(stamp)"
echo "================================================================"

SMOKE_DIR="$OUT/smoke"
mkdir -p "$SMOKE_DIR"

SMOKE_CNN_FLAGS=(
    --zero-mean-maps
    --map-kind nbody
    --seed 42
    --field-size 20 --field-npix 160
    --nbins 4 --tomo-bin-indices 1,2,3,4
    --compressor-arch plain
    --compressor-dim 10
    --compressor-dense-width 256
    --compressor-conv-channels 64,128,256
    --compressor-steps 500
    --compressor-batch-size 128
    --compressor-lr 0.0005
    --compressor-save-every 500
    --compressor-checkpoint-policy best_val
    --compressor-train-split 'train[:70%]'
    --compressor-val-split test
    --nde-train-split 'train[70%:]'
    --nde-val-split test
    --require-disjoint-train-examples
    --npe-samples 2000
    --no-wandb
)

echo "[smoke: auto-only on GPU 0]"
(
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
conda run -n jaxili --no-capture-output python "$CNN" \
    --train-compressor \
    --exit-after-compress \
    --tfds-name "$TFDS" \
    --cuda-visible-devices 0 \
    "${SMOKE_CNN_FLAGS[@]}" \
    --save-dir "$SMOKE_DIR/auto_rnvp" \
    --cache-dir "$SMOKE_DIR/auto_rnvp_cache" \
    --posterior-out /dev/null \
    --figure-out /dev/null \
    > "$OUT/logs/smoke_auto_rnvp.log" 2>&1
) &
SPID1=$!

echo "[smoke: auto+cross on GPU 1]"
(
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
conda run -n jaxili --no-capture-output python "$CNN" \
    --train-compressor \
    --exit-after-compress \
    --cnn-map-route harmonic \
    --full-sphere-cross-cache "$CACHE" \
    --harmonic-cache-regime nobnt \
    --harmonic-normalize-input-channels \
    --cuda-visible-devices 1 \
    "${SMOKE_CNN_FLAGS[@]}" \
    --save-dir "$SMOKE_DIR/autocross_rnvp" \
    --cache-dir "$SMOKE_DIR/autocross_rnvp_cache" \
    --posterior-out /dev/null \
    --figure-out /dev/null \
    > "$OUT/logs/smoke_autocross_rnvp.log" 2>&1
) &
SPID2=$!

wait $SPID1; src1=$?
wait $SPID2; src2=$?
echo "[smoke done] $(stamp) auto_rc=$src1 autocross_rc=$src2"

if [ $src1 -ne 0 ] || [ $src2 -ne 0 ]; then
    echo "*** SMOKE TEST FAILED ***"
    echo "  auto-only  rc=$src1 — check $OUT/logs/smoke_auto_rnvp.log"
    echo "  auto+cross rc=$src2 — check $OUT/logs/smoke_autocross_rnvp.log"
    echo "  Aborting overnight run. Fix errors and retry."
    exit 1
fi

# Verify smoke outputs exist
if [ ! -f "$SMOKE_DIR/auto_rnvp_cache/cnn_train.npz" ]; then
    echo "*** SMOKE FAILED: missing auto_rnvp_cache/cnn_train.npz ***"; exit 1
fi
if [ ! -f "$SMOKE_DIR/autocross_rnvp_cache/cnn_train.npz" ]; then
    echo "*** SMOKE FAILED: missing autocross_rnvp_cache/cnn_train.npz ***"; exit 1
fi

# Check that compressor loss decreased (not just "didn't crash")
for log in "$OUT/logs/smoke_auto_rnvp.log" "$OUT/logs/smoke_autocross_rnvp.log"; do
    if ! grep -q "best_val_loss\|Best val" "$log" 2>/dev/null; then
        echo "*** WARNING: no best_val_loss found in $log — check manually ***"
    fi
done

echo "Smoke test PASSED. Proceeding to production runs."
echo

# Clean up smoke artifacts (keep logs)
rm -rf "$SMOKE_DIR"

############################################################
# PRODUCTION: Phase A compressor training
############################################################
echo "================================================================"
echo "Phase A (overnight): 2 RealNVP-companion compressor trainings"
echo "Started: $(stamp)"
echo "GPU 0: auto-only compressor"
echo "GPU 1: auto+cross compressor"
echo "Running in PARALLEL (~3.5h wall time)"
echo "================================================================"

############################################################
# Compressor 1: auto-only (TFDS route, 4 channels) — GPU 0
############################################################
COMP1_DIR="$OUT/compressors/auto_rnvp"
COMP1_CACHE="$OUT/compressed/auto_rnvp_split70"
mkdir -p "$COMP1_DIR" "$COMP1_CACHE"

echo
echo "[comp1: auto-only RealNVP on GPU 0] $(stamp)"

(
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
conda run -n jaxili --no-capture-output python "$CNN" \
    --train-compressor \
    --exit-after-compress \
    --tfds-name "$TFDS" \
    --cuda-visible-devices 0 \
    "${COMMON_CNN_FLAGS[@]}" \
    --save-dir "$COMP1_DIR" \
    --cache-dir "$COMP1_CACHE" \
    --posterior-out /dev/null \
    --figure-out /dev/null \
    > "$OUT/logs/phase_a_auto_rnvp.log" 2>&1
) &
PID1=$!
echo "  PID: $PID1, log: $OUT/logs/phase_a_auto_rnvp.log"

############################################################
# Compressor 2: auto+cross (harmonic cache, 10 channels) — GPU 1
############################################################
COMP2_DIR="$OUT/compressors/autocross_rnvp"
COMP2_CACHE="$OUT/compressed/autocross_rnvp_split70"
mkdir -p "$COMP2_DIR" "$COMP2_CACHE"

echo "[comp2: auto+cross RealNVP on GPU 1] $(stamp)"

(
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
conda run -n jaxili --no-capture-output python "$CNN" \
    --train-compressor \
    --exit-after-compress \
    --cnn-map-route harmonic \
    --full-sphere-cross-cache "$CACHE" \
    --harmonic-cache-regime nobnt \
    --harmonic-normalize-input-channels \
    --cuda-visible-devices 1 \
    "${COMMON_CNN_FLAGS[@]}" \
    --save-dir "$COMP2_DIR" \
    --cache-dir "$COMP2_CACHE" \
    --posterior-out /dev/null \
    --figure-out /dev/null \
    > "$OUT/logs/phase_a_autocross_rnvp.log" 2>&1
) &
PID2=$!
echo "  PID: $PID2, log: $OUT/logs/phase_a_autocross_rnvp.log"

############################################################
# Wait for both
############################################################
echo
echo "Waiting for both compressors to finish..."
echo "  comp1 (GPU 0, auto-only):  PID $PID1"
echo "  comp2 (GPU 1, auto+cross): PID $PID2"

wait $PID1
rc1=$?
echo "[comp1 done] $(stamp) rc=$rc1"
if [ $rc1 -ne 0 ]; then
    echo "  FAILED — check $OUT/logs/phase_a_auto_rnvp.log"
fi
if [ -f "$COMP1_CACHE/cnn_train.npz" ] && [ -f "$COMP1_CACHE/cnn_obs.npz" ]; then
    echo "  Cache verified: $(ls "$COMP1_CACHE"/*.npz | wc -l) NPZ files"
else
    echo "  WARNING: Missing cache files in $COMP1_CACHE"
fi

wait $PID2
rc2=$?
echo "[comp2 done] $(stamp) rc=$rc2"
if [ $rc2 -ne 0 ]; then
    echo "  FAILED — check $OUT/logs/phase_a_autocross_rnvp.log"
fi
if [ -f "$COMP2_CACHE/cnn_train.npz" ] && [ -f "$COMP2_CACHE/cnn_obs.npz" ]; then
    echo "  Cache verified: $(ls "$COMP2_CACHE"/*.npz | wc -l) NPZ files"
else
    echo "  WARNING: Missing cache files in $COMP2_CACHE"
fi

############################################################
# Phase A summary
############################################################
echo
echo "================================================================"
echo "Phase A (compressor training) complete: $(stamp)"
echo "  Compressor 1 (auto-only RealNVP, GPU 0):  rc=$rc1"
echo "  Compressor 2 (auto+cross RealNVP, GPU 1): rc=$rc2"
echo "================================================================"

echo "{\"phase_a_partial\": {\"auto_rnvp_rc\": $rc1, \"autocross_rnvp_rc\": $rc2, \"completed_utc\": \"$(stamp)\"}}" \
    > "$OUT/phase_a_status.json"

############################################################
# Phase B-partial: L1 arms (no compressor needed)
#
# Arm 1: L1 auto+cross, full-train NDE, harmonic cache, 3 seeds × 3 perms
# Arm 2: L1 auto+cross, 70/30 NDE split, harmonic cache, 3 seeds × 3 perms
# Arm 3: L1 auto-only, full-train NDE, TFDS, 3 seeds (seed=perm on TFDS)
#
# Arms 1-2 run in parallel on GPU 0 + GPU 1 (one seed at a time per GPU).
# Arm 3 runs after arms 1-2 finish (TFDS auto-only, fast).
############################################################

L1="$REPO/scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py"

COMMON_L1_FLAGS=(
    --zero-mean-maps
    --map-kind nbody
    --field-size 20 --field-npix 160
    --nbins 4 --tomo-bin-indices 1,2,3,4
    --pca-components 0
    --l1-min-snr -13 --l1-max-snr 13
    --cross-snr-percentile 1.0
    --batch-size 256 --learning-rate 0.0001
    --npe-samples 100000
    --no-wandb
)

echo
echo "================================================================"
echo "Phase B-partial: L1 arms (3 arms, no compressor needed)"
echo "Started: $(stamp)"
echo "================================================================"

run_l1_cross() {
    # $1=seed, $2=perm, $3=nde_split, $4=label, $5=gpu, $6=epochs
    local SEED=$1 PERM=$2 SPLIT=$3 LABEL=$4 GPU=$5 EPOCHS=$6
    local NAME="${LABEL}_s${SEED}_p${PERM}"
    local PDIR="$OUT/posteriors/${LABEL}"
    mkdir -p "$PDIR" "$OUT/logs"

    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    conda run -n jaxili --no-capture-output python "$L1" \
        --full-sphere-cross-cache "$CACHE" \
        --cross-noise-model channel_empirical_global \
        --harmonic-obs-perm "$PERM" \
        --nde-train-split "$SPLIT" \
        --epochs "$EPOCHS" \
        --seed "$SEED" \
        --cuda-visible-devices "$GPU" \
        "${COMMON_L1_FLAGS[@]}" \
        --save-dir "$PDIR/train_${NAME}" \
        --posterior-out "$PDIR/${NAME}.npy" \
        --figure-out "$PDIR/${NAME}.pdf" \
        > "$OUT/logs/${NAME}.log" 2>&1
    local rc=$?
    echo "  ${NAME}: rc=$rc ($(stamp))"
    return $rc
}

run_l1_auto() {
    # $1=seed, $2=label, $3=gpu, $4=epochs
    local SEED=$1 LABEL=$2 GPU=$3 EPOCHS=$4
    local NAME="${LABEL}_s${SEED}"
    local PDIR="$OUT/posteriors/${LABEL}"
    mkdir -p "$PDIR" "$OUT/logs"

    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    conda run -n jaxili --no-capture-output python "$L1" \
        --tfds-name "$TFDS" \
        --nde-train-split train \
        --epochs "$EPOCHS" \
        --seed "$SEED" \
        --cuda-visible-devices "$GPU" \
        "${COMMON_L1_FLAGS[@]}" \
        --save-dir "$PDIR/train_${NAME}" \
        --posterior-out "$PDIR/${NAME}.npy" \
        --figure-out "$PDIR/${NAME}.pdf" \
        > "$OUT/logs/${NAME}.log" 2>&1
    local rc=$?
    echo "  ${NAME}: rc=$rc ($(stamp))"
    return $rc
}

# Arms 1 + 2: L1 auto+cross on harmonic cache
# Run in parallel: arm 1 on GPU 0, arm 2 on GPU 1, one (seed,perm) at a time
echo
echo "[arms 1+2: L1 auto+cross] $(stamp)"
for SEED in 41 42 43; do
    for PERM in 0 1 2; do
        run_l1_cross "$SEED" "$PERM" "train" "l1_autocross_fulltrain" 0 50000 &
        P1=$!
        run_l1_cross "$SEED" "$PERM" "train[70%:]" "l1_autocross_split70" 1 50000 &
        P2=$!
        wait $P1 $P2
    done
done
echo "[arms 1+2 done] $(stamp)"

# Arm 3: L1 auto-only on TFDS (fast, ~1-2 min each)
# No perm concept on TFDS — seed controls both NDE init and obs noise.
# Run 3 seeds on GPU 0 sequentially (each is very fast).
echo
echo "[arm 3: L1 auto-only] $(stamp)"
for SEED in 41 42 43; do
    run_l1_auto "$SEED" "l1_auto_fulltrain" 0 5000
done
echo "[arm 3 done] $(stamp)"

############################################################
# Final summary
############################################################
echo
echo "================================================================"
echo "Overnight run complete: $(stamp)"
echo "  Phase A: compressor 1 rc=$rc1, compressor 2 rc=$rc2"
echo "  Phase B-partial: L1 arms (check logs for individual rc)"
echo
echo "Posteriors produced:"
ls "$OUT/posteriors/"*/*.npy 2>/dev/null | wc -l
echo " .npy files in posteriors/"
echo
echo "Next: Phase 0a/0b code, MAF compressors, CNN NDE runs"
echo "================================================================"

# Update status
echo "{\"phase_a_partial\": {\"auto_rnvp_rc\": $rc1, \"autocross_rnvp_rc\": $rc2}, \"l1_arms\": {\"completed_utc\": \"$(stamp)\"}}" \
    > "$OUT/phase_a_status.json"
