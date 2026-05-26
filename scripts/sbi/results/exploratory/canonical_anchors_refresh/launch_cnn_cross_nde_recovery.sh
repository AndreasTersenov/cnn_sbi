#!/usr/bin/env bash
# Arm 2 recovery: NDE-only re-run using the saved best_val compressor
# checkpoints (the compressor training was successful pre-crash; the bug was
# downstream of training in the wandb-stamp section).
#
# Same flags as the canonical arm 2 EXCEPT:
#   - drop --train-compressor (the inverse loads pretrained weights)
#   - add --compressor-params and --compressor-state pointing at the saved
#     best_val checkpoint pickles

set -uo pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
OUT="$REPO/scripts/sbi/results/exploratory/canonical_anchors_refresh"
CACHE="$REPO/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"

stamp() { date -u +%FT%TZ; }
echo "[recovery start] $(stamp) — arm 2 NDE-only re-run, 3 seeds parallel"

run_recovery_seed() {
    local SEED=$1
    local NAME="cnn_cross_canon_s${SEED}"
    local LOG="$OUT/logs/${NAME}_recovery.log"
    local CKPT_DIR="$OUT/train/${NAME}/vmim/nbody/sigma_0.26/gal_density_30/bin_4/harmonic_nobnt_ch10"
    local PARAMS="$CKPT_DIR/params_nd_compressor_best_val.pkl"
    local STATE="$CKPT_DIR/opt_state_resnet_best_val.pkl"

    if [ ! -f "$PARAMS" ] || [ ! -f "$STATE" ]; then
        echo "[s${SEED}] MISSING CHECKPOINT — params=$([ -f "$PARAMS" ] && echo OK || echo MISSING)  state=$([ -f "$STATE" ] && echo OK || echo MISSING)"
        return 1
    fi

    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
    conda run -n jaxili --no-capture-output python "$CNN" \
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
        --compressor-params "$PARAMS" \
        --compressor-state "$STATE" \
        --compressor-train-split 'train[:70%]' \
        --compressor-val-split   val \
        --nde-train-split        'train[70%:]' \
        --nde-val-split          val \
        --require-disjoint-train-examples \
        --total-steps 50000 --save-every 500 \
        --batch-size 256 --nvp-layers 8 --nvp-hidden 256 \
        --npe-samples 100000 \
        --save-dir "$OUT/train/${NAME}" \
        --posterior-out "$OUT/posteriors/${NAME}.npy" \
        --figure-out "$OUT/figures/${NAME}.pdf" \
        --cuda-visible-devices 1 --no-wandb \
        > "$LOG" 2>&1
}

run_recovery_seed 41 & P41=$!; sleep 5
run_recovery_seed 42 & P42=$!; sleep 5
run_recovery_seed 43 & P43=$!
wait $P41; r1=$?; wait $P42; r2=$?; wait $P43; r3=$?

echo "[recovery end] $(stamp)  rc=$r1,$r2,$r3"
ls -la "$OUT/posteriors/cnn_cross_canon_s4"*.npy 2>&1
