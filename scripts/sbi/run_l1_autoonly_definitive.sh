#!/usr/bin/env bash
# L1 auto-ONLY (route-matched, flip=False) re-run for the definitive campaign.
#
# Same harmonic-cache route / NDE / preprocessing as the auto+cross arms, but
# --channel-mode auto_only (slices to the 4 auto channels, auto-SNR). This is the
# apples-to-apples flip=False baseline for the cross-channel-gain comparison
# (the existing l1_auto_* baselines are flip=True AND on the TFDS route).
#
# Two arms, 3 seeds × 3 perms = 18 posteriors:
#   l1_autoonly_fulltrain  --nde-train-split train
#   l1_autoonly_split70    --nde-train-split 'train[70%:]'
# Shared per-arm --cache-dir (dedup); --no-l1-train-flip => seed-independent.
#
# Usage:
#   run_l1_autoonly_definitive.sh warmup_arm1 <gpu>
#   run_l1_autoonly_definitive.sh warmup_arm2 <gpu>
#   run_l1_autoonly_definitive.sh fanout_arm1 <gpu_csv>
#   run_l1_autoonly_definitive.sh fanout_arm2 <gpu_csv>
set -uo pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi; cd "$REPO"
L1=scripts/sbi/npe_l1norm_cross_jaxili_nbody_tomo.py
NPZ=scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid
OUT=scripts/sbi/results/exploratory/definitive_comparison
LOGS=$OUT/logs

COMMON=(--full-sphere-cross-cache "$NPZ"
        --channel-mode auto_only
        --zero-mean-maps --map-kind nbody --field-size 20 --field-npix 160
        --nbins 4 --tomo-bin-indices 1,2,3,4
        --pca-components 0 --l1-min-snr -13 --l1-max-snr 13 --cross-snr-percentile 1.0
        --batch-size 256 --learning-rate 0.0001 --npe-samples 100000 --no-wandb
        --cross-noise-model channel_empirical_global --epochs 50000
        --no-l1-train-flip)

SEEDS=(41 42 43)
PERMS=(0 1 2)

run_one() {
  local label="$1" split="$2" seed="$3" perm="$4" gpu="$5"
  local cache="$OUT/compressed/${label}_dv"
  local pdir="$OUT/posteriors/$label"
  local tag="${label}_s${seed}_p${perm}"
  local log="$LOGS/${tag}.log"
  mkdir -p "$cache" "$pdir"
  echo "[launch] $tag  gpu=$gpu  split='$split'  cache=$cache  -> $log"
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
  warmup_arm1) run_one l1_autoonly_fulltrain "train"       41 0 "${2:?gpu}" ;;
  warmup_arm2) run_one l1_autoonly_split70   "train[70%:]" 41 0 "${2:?gpu}" ;;
  fanout_arm1|fanout_arm2)
    if [[ "$mode" == fanout_arm1 ]]; then label=l1_autoonly_fulltrain; split="train";
    else label=l1_autoonly_split70; split="train[70%:]"; fi
    gpus_csv="${2:?gpu_csv}"; IFS=',' read -r -a GPUS <<< "$gpus_csv"
    i=0
    for s in "${SEEDS[@]}"; do for p in "${PERMS[@]}"; do
      [[ "$s" == 41 && "$p" == 0 ]] && continue
      gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
      while [[ "$(jobs -rp | wc -l)" -ge $(( 2 * ${#GPUS[@]} )) ]]; do sleep 10; done
      run_one "$label" "$split" "$s" "$p" "$gpu" &
      i=$((i+1)); sleep 3
    done; done
    wait
    echo "[fanout done] $label"
    ;;
  *) echo "unknown mode: $mode"; exit 2 ;;
esac
