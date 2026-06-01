#!/usr/bin/env bash
# L1 auto+cross relaunch (definitive L1-vs-CNN campaign, §6) — flip=False + dedup path.
#
# Plan landed by the Part-1 (audit/dedup) session: the L1 TFRecord port was DROPPED
# (L1 is wavelet-compute-bound, not read-bound); the speedup is cross-seed datavector
# dedup via a shared --cache-dir per arm + --no-l1-train-flip (seed-independent train
# datavector, validated reproducible to 2.7e-11). NO --harmonic-tfrecord-dir for L1.
#
# Two arms, 3 seeds × 3 perms = 18 posteriors:
#   arm 1  l1_autocross_fulltrain  --nde-train-split train
#   arm 2  l1_autocross_split70    --nde-train-split 'train[70%:]'
#
# Orchestration: run ONE (s41,p0) warm-up per arm FIRST to populate the shared
# datavector cache (the ~2-4 h summarization), confirm the harmonic route engaged,
# THEN fan out the remaining 8 (cache hit -> NDE+sampling only).
#
# Usage:
#   run_l1_autocross_definitive.sh warmup_arm1 <gpu>
#   run_l1_autocross_definitive.sh warmup_arm2 <gpu>
#   run_l1_autocross_definitive.sh fanout_arm1 <gpu_csv>   # e.g. 0,1
#   run_l1_autocross_definitive.sh fanout_arm2 <gpu_csv>
#   run_l1_autocross_definitive.sh one <arm> <split> <seed> <perm> <gpu>
set -uo pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
cd "$REPO"
L1=scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py
NPZ=scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid
OUT=scripts/sbi/results/exploratory/definitive_comparison
LOGS=$OUT/logs

COMMON=(--full-sphere-cross-cache "$NPZ"
        --zero-mean-maps --map-kind nbody --field-size 20 --field-npix 160
        --nbins 4 --tomo-bin-indices 1,2,3,4
        --pca-components 0 --l1-min-snr -13 --l1-max-snr 13 --cross-snr-percentile 1.0
        --batch-size 256 --learning-rate 0.0001 --npe-samples 100000 --no-wandb
        --cross-noise-model channel_empirical_global --epochs 50000
        --no-l1-train-flip)

SEEDS=(41 42 43)
PERMS=(0 1 2)

# run_one <label> <split> <seed> <perm> <gpu> [&]
run_one() {
  local label="$1" split="$2" seed="$3" perm="$4" gpu="$5"
  local cache="$OUT/compressed/${label}_dv"
  local pdir="$OUT/posteriors/$label"
  local tag="${label}_s${seed}_p${perm}"
  local log="$LOGS/${tag}.log"
  mkdir -p "$cache" "$pdir"
  echo "[launch] $tag  gpu=$gpu  split='$split'  cache=$cache  -> $log"
  # --no-capture-output + PYTHONUNBUFFERED so stdout (route/config markers) streams
  # live to the log instead of being block-buffered by `conda run` (defeats the
  # back-pressure check and made the F-leg log sit empty for hours).
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n jaxili python "$L1" "${COMMON[@]}" \
    --nde-train-split "$split" \
    --seed "$seed" --harmonic-obs-perm "$perm" \
    --cuda-visible-devices "$gpu" \
    --cache-dir "$cache" \
    --save-dir   "$pdir/train_${tag}" \
    --posterior-out "$pdir/${tag}.npy" \
    --figure-out "$pdir/${tag}.pdf" \
    > "$log" 2>&1
}

mode="${1:?mode required}"
case "$mode" in
  warmup_arm1) run_one l1_autocross_fulltrain "train"       41 0 "${2:?gpu}" ;;
  warmup_arm2) run_one l1_autocross_split70   "train[70%:]" 41 0 "${2:?gpu}" ;;
  one)         run_one "$2" "$3" "$4" "$5" "$6" ;;
  fanout_arm1|fanout_arm2)
    if [[ "$mode" == fanout_arm1 ]]; then label=l1_autocross_fulltrain; split="train";
    else label=l1_autocross_split70; split="train[70%:]"; fi
    gpus_csv="${2:?gpu_csv}"; IFS=',' read -r -a GPUS <<< "$gpus_csv"
    # fan out all (seed,perm) EXCEPT the (41,0) warm-up, round-robin across GPUS,
    # max 2 concurrent per GPU.
    declare -A running_per_gpu
    i=0
    for s in "${SEEDS[@]}"; do for p in "${PERMS[@]}"; do
      [[ "$s" == 41 && "$p" == 0 ]] && continue
      gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
      # throttle: wait if this gpu already has 2 running
      while [[ "$(jobs -rp | wc -l)" -ge $(( 2 * ${#GPUS[@]} )) ]]; do sleep 10; done
      run_one "$label" "$split" "$s" "$p" "$gpu" &
      i=$((i+1)); sleep 3
    done; done
    wait
    echo "[fanout done] $label"
    ;;
  *) echo "unknown mode: $mode"; exit 2 ;;
esac
