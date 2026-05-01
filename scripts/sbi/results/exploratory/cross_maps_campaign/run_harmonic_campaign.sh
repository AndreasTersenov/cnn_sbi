#!/usr/bin/env bash
# 3-seed harmonic-cross NPE campaign.
# Two waves; each wave runs 3 GPUs in parallel.
#   Wave 1: BNT regime, seeds 41/42/43 on GPUs 0/1/2.
#   Wave 2: no-BNT regime, seeds 41/42/43 on GPUs 0/1/2.
# Each arm logs to its own .log file under scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_harm_cross_<regime>/seed_<seed>/run.log
set -uo pipefail

ROOT_DIR=/mnt/home/tersenov/software/cnn_sbi
ARM=$ROOT_DIR/scripts/sbi/results/exploratory/cross_maps_campaign/run_harmonic_arm.sh

run_wave() {
  local regime=$1
  local out_root=$ROOT_DIR/scripts/sbi/results/exploratory/cross_maps_campaign/jaxili_harm_cross_${regime}
  echo "[$(date '+%F %T')] === wave: regime=${regime} ==="
  local pids=()
  for pair in "0:41" "1:42" "2:43"; do
    local gpu=${pair%%:*}
    local seed=${pair##*:}
    local seed_dir=$out_root/seed_${seed}
    mkdir -p "$seed_dir"
    local log=$seed_dir/run.log
    echo "[$(date '+%F %T')]   gpu=${gpu} seed=${seed}  log=${log}"
    bash "$ARM" "$gpu" "$regime" "$seed" >"$log" 2>&1 &
    pids+=("$!")
  done
  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "[$(date '+%F %T')]   pid=${pid} exited non-zero"
      fail=1
    fi
  done
  if [ "$fail" -ne 0 ]; then
    echo "[$(date '+%F %T')] === wave ${regime} had failure(s) ==="
    return 1
  fi
  echo "[$(date '+%F %T')] === wave ${regime} done ==="
  return 0
}

run_wave bnt   || exit 1
run_wave nobnt || exit 1

echo "[$(date '+%F %T')] === all 6 arms complete ==="
