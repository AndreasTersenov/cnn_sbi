#!/usr/bin/env bash
# Canonical-anchors refresh — 4 arms × 3 seeds, on the canonical 70/30 split setup.
# See METHODOLOGY.md (will be written after the runs land) for the protocol.
#
# Canonical config (mirrors `cnn_bnt_losslessness_campaign_indep_split_*`):
#   --tfds-name grid_20deg_160px_nonoverlap48  (or matching harmonic cache)
#   --compressor-train-split train[:70%]       # 211k examples
#   --compressor-val-split   test              # held-out (cosmos 900-1300)
#   --nde-train-split        train[70%:]       # 90k examples, disjoint triples
#   --nde-val-split          test              # CNN: 'test', L1 harmonic: 'val'
#   --require-disjoint-train-examples          # zero-overlap audit
#   --zero-mean-maps                           # mandatory project rule
#
# Each arm uses the simplest-best-performing setup found in prior campaigns
# (CNN: iter-108-Q6ON-60k; L1: v2_chsigma).
#
# Resource: GPU 1 sole tenant. 4 arms run sequentially; within each arm,
# 3 seeds run in parallel with mem_fraction 0.30 each (the H1 exit-interview
# rule: parallel-3 works for small models, sequential for heavy ones).
# CNN auto+cross at cdim=10 (NOT cdim=100) — checking if parallel-3 holds;
# if it bogs down we switch to sequential mid-run.

set -uo pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
L1="$REPO/scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py"
OUT="$REPO/scripts/sbi/results/exploratory/canonical_anchors_refresh"
CACHE="$REPO/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"

TFDS=NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48
COMP_TRAIN='train[:70%]'
COMP_VAL='test'
NDE_TRAIN='train[70%:]'
NDE_VAL='test'    # for CNN auto-only TFDS route; harmonic route uses 'val' (auto-mapped)

# Harmonic-route splits for CNN auto+cross. The cache has its own
# {train, val, obs} basenames. The code extension lets us pass the same
# slicing notation; _normalize_harmonic_split maps test->val automatically.
HARM_COMP_TRAIN='train[:70%]'
HARM_COMP_VAL='val'
HARM_NDE_TRAIN='train[70%:]'
HARM_NDE_VAL='val'

mkdir -p "$OUT"/{logs,posteriors,figures,train,fom}

stamp() { date -u +%FT%TZ; }
echo "[start] $(stamp) — canonical-anchors-refresh on GPU 1"
echo "  TFDS: $TFDS"
echo "  Splits: comp=[$COMP_TRAIN/$COMP_VAL]  nde=[$NDE_TRAIN/$NDE_VAL]"

############################################################
# Arm 1: CNN auto-only (plain CNN, 4-channel, cdim=10)
############################################################
run_cnn_auto_seed() {
    local SEED=$1
    local NAME="cnn_auto_canon_s${SEED}"
    local LOG="$OUT/logs/${NAME}.log"
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
    conda run -n jaxili --no-capture-output python "$CNN" \
        --train-compressor \
        --zero-mean-maps \
        --standardize-summary \
        --map-kind nbody \
        --seed ${SEED} \
        --tfds-name "$TFDS" \
        --field-size 20 --field-npix 160 \
        --nbins 4 --tomo-bin-indices 1,2,3,4 \
        --compressor-arch plain \
        --compressor-dim 10 \
        --compressor-dense-width 256 \
        --compressor-conv-channels 64,128,256 \
        --compressor-steps 60000 \
        --compressor-batch-size 128 \
        --compressor-lr 0.0005 \
        --compressor-checkpoint-policy best_val \
        --compressor-train-split "$COMP_TRAIN" \
        --compressor-val-split   "$COMP_VAL" \
        --nde-train-split        "$NDE_TRAIN" \
        --nde-val-split          "$NDE_VAL" \
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

echo
echo "[arm 1 CNN auto-only] $(stamp) — 3 seeds parallel"
run_cnn_auto_seed 41 & P41=$!; sleep 5
run_cnn_auto_seed 42 & P42=$!; sleep 5
run_cnn_auto_seed 43 & P43=$!
wait $P41; r1=$?; wait $P42; r2=$?; wait $P43; r3=$?
echo "[arm 1 done] $(stamp)  rc=$r1,$r2,$r3"

############################################################
# Arm 2: CNN auto+cross (plain CNN, 10-channel harmonic, cdim=10)
############################################################
run_cnn_cross_seed() {
    local SEED=$1
    local NAME="cnn_cross_canon_s${SEED}"
    local LOG="$OUT/logs/${NAME}.log"
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
    conda run -n jaxili --no-capture-output python "$CNN" \
        --train-compressor \
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
        --compressor-steps 60000 \
        --compressor-batch-size 128 \
        --compressor-lr 0.0005 \
        --compressor-checkpoint-policy best_val \
        --compressor-train-split "$HARM_COMP_TRAIN" \
        --compressor-val-split   "$HARM_COMP_VAL" \
        --nde-train-split        "$HARM_NDE_TRAIN" \
        --nde-val-split          "$HARM_NDE_VAL" \
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

############################################################
# Arm 5 (NEW, sanity check): CNN on cache's AUTO-ONLY slice
# Tests whether the harmonic-cache auto channels (bandlimited to ell=1024
# via SHT/iSHT roundtrip in build_full_sphere_cross_cache.py) give
# significantly different FoM3 from the TFDS auto maps that the CNN
# auto-only canonical run uses. If within seed scatter (~10%), bandlimit
# is negligible and cross/auto comparison is clean. If significantly
# different, real protocol asymmetry to flag.
# Uses --channel-mode=auto_only to slice the 10-channel cache to its 4
# auto channels.
############################################################
run_cnn_cache_auto_only_seed() {
    local SEED=$1
    local NAME="cnn_cache_auto_sanity_s${SEED}"
    local LOG="$OUT/logs/${NAME}.log"
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
    conda run -n jaxili --no-capture-output python "$CNN" \
        --train-compressor \
        --zero-mean-maps \
        --standardize-summary \
        --map-kind nbody \
        --seed ${SEED} \
        --cnn-map-route harmonic \
        --full-sphere-cross-cache "$CACHE" \
        --harmonic-cache-regime nobnt \
        --harmonic-normalize-input-channels \
        --channel-mode auto_only \
        --field-size 20 --field-npix 160 \
        --nbins 4 --tomo-bin-indices 1,2,3,4 \
        --compressor-arch plain \
        --compressor-dim 10 \
        --compressor-dense-width 256 \
        --compressor-conv-channels 64,128,256 \
        --compressor-steps 60000 \
        --compressor-batch-size 128 \
        --compressor-lr 0.0005 \
        --compressor-checkpoint-policy best_val \
        --compressor-train-split "$HARM_COMP_TRAIN" \
        --compressor-val-split   "$HARM_COMP_VAL" \
        --nde-train-split        "$HARM_NDE_TRAIN" \
        --nde-val-split          "$HARM_NDE_VAL" \
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

echo
echo "[arm 2 CNN auto+cross] $(stamp) — 3 seeds parallel"
run_cnn_cross_seed 41 & P41=$!; sleep 5
run_cnn_cross_seed 42 & P42=$!; sleep 5
run_cnn_cross_seed 43 & P43=$!
wait $P41; r1=$?; wait $P42; r2=$?; wait $P43; r3=$?
echo "[arm 2 done] $(stamp)  rc=$r1,$r2,$r3"

############################################################
# Arm 3: L1 auto-only (4-channel, cross_maps off)
############################################################
run_l1_auto_seed() {
    local SEED=$1
    local NAME="l1_auto_canon_s${SEED}"
    local LOG="$OUT/logs/${NAME}.log"
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
    conda run -n jaxili --no-capture-output python "$L1" \
        --zero-mean-maps \
        --map-kind nbody \
        --seed ${SEED} \
        --tfds-name "$TFDS" \
        --field-size 20 --field-npix 160 \
        --nbins 4 --tomo-bin-indices 1,2,3,4 \
        --l1-min-snr -13 --l1-max-snr 13 \
        --pca-components 0 \
        --nde-train-split train \
        --epochs 5000 --batch-size 256 --learning-rate 0.0001 \
        --npe-samples 100000 \
        --save-dir "$OUT/train/${NAME}" \
        --posterior-out "$OUT/posteriors/${NAME}.npy" \
        --figure-out "$OUT/figures/${NAME}.pdf" \
        --cuda-visible-devices 1 --no-wandb \
        > "$LOG" 2>&1
}

echo
echo "[arm 3 L1 auto-only] $(stamp) — 3 seeds parallel"
run_l1_auto_seed 41 & P41=$!; sleep 5
run_l1_auto_seed 42 & P42=$!; sleep 5
run_l1_auto_seed 43 & P43=$!
wait $P41; r1=$?; wait $P42; r2=$?; wait $P43; r3=$?
echo "[arm 3 done] $(stamp)  rc=$r1,$r2,$r3"

############################################################
# Arm 4: L1 auto+cross (10-channel, cross_maps on, channel-aware noise)
############################################################
run_l1_cross_seed() {
    local SEED=$1
    local NAME="l1_cross_canon_s${SEED}"
    local LOG="$OUT/logs/${NAME}.log"
    # IMPORTANT: L1 cross MUST use the harmonic-cache route (--full-sphere-cross-cache).
    # The channel-aware cross-noise model is ONLY implemented for the harmonic route;
    # the TFDS route + --cross-maps internally computes flat-sky FFT cross-maps but
    # silently falls back to the broken auto_scalar noise model (warning printed,
    # behavior changed). See feedback_never_pca_l1.md and the related cross-route
    # memory for context.
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
    conda run -n jaxili --no-capture-output python "$L1" \
        --zero-mean-maps \
        --map-kind nbody \
        --seed ${SEED} \
        --tfds-name "$TFDS" \
        --full-sphere-cross-cache "$CACHE" \
        --field-size 20 --field-npix 160 \
        --nbins 4 --tomo-bin-indices 1,2,3,4 \
        --cross-noise-model channel_empirical_global \
        --l1-min-snr -13 --l1-max-snr 13 \
        --cross-snr-percentile 1.0 \
        --pca-components 0 \
        --nde-train-split train \
        --epochs 50000 --batch-size 256 --learning-rate 0.0001 \
        --npe-samples 100000 \
        --save-dir "$OUT/train/${NAME}" \
        --posterior-out "$OUT/posteriors/${NAME}.npy" \
        --figure-out "$OUT/figures/${NAME}.pdf" \
        --cuda-visible-devices 1 --no-wandb \
        > "$LOG" 2>&1
}

echo
echo "[arm 4 L1 auto+cross] $(stamp) — 3 seeds parallel"
run_l1_cross_seed 41 & P41=$!; sleep 5
run_l1_cross_seed 42 & P42=$!; sleep 5
run_l1_cross_seed 43 & P43=$!
wait $P41; r1=$?; wait $P42; r2=$?; wait $P43; r3=$?
echo "[arm 4 done] $(stamp)  rc=$r1,$r2,$r3"

echo
echo "[arm 5 SANITY: CNN cache auto-only] $(stamp) — 3 seeds parallel"
run_cnn_cache_auto_only_seed 41 & P41=$!; sleep 5
run_cnn_cache_auto_only_seed 42 & P42=$!; sleep 5
run_cnn_cache_auto_only_seed 43 & P43=$!
wait $P41; r1=$?; wait $P42; r2=$?; wait $P43; r3=$?
echo "[arm 5 done] $(stamp)  rc=$r1,$r2,$r3"

echo
echo "[ALL DONE] $(stamp)"
ls -la "$OUT/posteriors/" | tail -25
