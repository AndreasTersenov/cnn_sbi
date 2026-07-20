#!/usr/bin/env bash
# Stability re-run for L1 harmonic-cross no-BNT: seeds 44, 45, 46.
# Sequential on GPU 3 (only free GPU; 0/1/2 hold the resnet50_gn campaign).
#
# Uses the existing per-seed runner `run_harmonic_arm.sh`. Each seed builds
# its own L1 cache (~2h for the 6293-realization train walk) and trains its
# own NDE, so the 3 seeds are statistically independent.

set -uo pipefail

ARM=/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/cross_maps_campaign/run_harmonic_arm.sh
GPU=3
REGIME=nobnt
ROOT=/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_harm_cross_${REGIME}

for seed in 44 45 46; do
  seed_dir="$ROOT/seed_${seed}"
  mkdir -p "$seed_dir"
  log="$seed_dir/run.log"
  echo "[$(date '+%F %T')] === seed=${seed} (GPU ${GPU}) → ${log} ==="
  bash "$ARM" "$GPU" "$REGIME" "$seed" > "$log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] === seed=${seed} done rc=${rc} ==="
  if [ "$rc" -ne 0 ]; then
    echo "[$(date '+%F %T')] FAILED at seed=${seed} — stopping."
    exit "$rc"
  fi
done

echo "[$(date '+%F %T')] === stability run complete (seeds 44,45,46) ==="
