#!/usr/bin/env bash
# Two-phase pipeline:
#   Phase 1: 4 missing A4-ext resnet50 jobs (GPUs 0,1,2,3 in parallel)
#   Phase 2: 5 missing A4 resnet50 jobs (GPUs 0,1,2 in parallel, then remainder)
# Logs written per-job; this script's own log goes to run_remaining_a4_pipeline.log

set -uo pipefail

REPO=/home/tersenov/software/cnn_sbi
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
A4EXT="$REPO/scripts/sbi/results/exploratory/cnn_resnet34_50_zm_cdim1224"
A4="$REPO/scripts/sbi/results/exploratory/cnn_resnet50_zm_sweep"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$A4EXT/logs" "$A4EXT/posteriors" "$A4EXT/figures"
mkdir -p "$A4/logs" "$A4/posteriors" "$A4/figures"

run_cnn() {
  # run_cnn <arch> <cdim> <seed> <gpu> <out_base> <log_dir> [mem_frac]
  local arch=$1 cdim=$2 seed=$3 gpu=$4 out_base=$5 log_dir=$6
  local mem_frac=${7:-0.80}
  local tag="${arch}_cdim${cdim}_s${seed}"
  local run_dir="$out_base/$tag"
  mkdir -p "$run_dir"
  XLA_PYTHON_CLIENT_MEM_FRACTION="$mem_frac" \
  conda run --no-capture-output -n jaxili python -u "$CNN" \
    --no-wandb \
    --train-compressor \
    --zero-mean-maps \
    --map-kind nbody \
    --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48 \
    --field-size 20 --field-npix 160 \
    --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --compressor-arch "$arch" \
    --compressor-dim "$cdim" \
    --compressor-dense-width 256 \
    --compressor-steps 120000 \
    --compressor-train-split train --compressor-val-split test \
    --nde-train-split train --nde-val-split test \
    --total-steps 10000 --save-every 500 \
    --npe-samples 100000 --ds-batch-size 500 \
    --plot \
    --seed "$seed" \
    --save-dir "$run_dir" \
    --posterior-out "$out_base/posteriors/cnn_${arch}_zm_nobnt_cdim${cdim}_s${seed}.npy" \
    --figure-out "$out_base/figures/cnn_${arch}_zm_nobnt_cdim${cdim}_s${seed}.pdf" \
    --cuda-visible-devices "$gpu" \
    > "$log_dir/${arch}_cdim${cdim}_s${seed}_gpu${gpu}.log" 2>&1
  local rc=$?
  echo "[$(date -u +%T)] ${tag} gpu=${gpu} rc=${rc}"
  return $rc
}

# ── Phase 1: A4-ext missing resnet50 ─────────────────────────────────────────
echo "[$(date -u)] Phase 1: launching 4 A4-ext resnet50 jobs"
run_cnn resnet50 12 43 0 "$A4EXT" "$A4EXT/logs" 0.80 &  PID0=$!
run_cnn resnet50 16 42 1 "$A4EXT" "$A4EXT/logs" 0.80 &  PID1=$!
run_cnn resnet50 16 43 2 "$A4EXT" "$A4EXT/logs" 0.80 &  PID2=$!
run_cnn resnet50 24 43 3 "$A4EXT" "$A4EXT/logs" 0.45 &  PID3=$!

echo "[$(date -u)] A4-ext PIDs: $PID0 $PID1 $PID2 $PID3"
wait $PID0 $PID1 $PID2 $PID3
echo "[$(date -u)] Phase 1 complete."

# ── Phase 2: A4 missing resnet50 ─────────────────────────────────────────────
# Missing: cdim10_s42, cdim10_s43, cdim20_s43, cdim50_s42, cdim50_s43
echo "[$(date -u)] Phase 2: launching 5 A4 resnet50 jobs (3+2)"

# Batch 2a: 3 in parallel on GPUs 0,1,2
run_cnn resnet50 10 42 0 "$A4" "$A4/logs" 0.80 &  PA=$!
run_cnn resnet50 10 43 1 "$A4" "$A4/logs" 0.80 &  PB=$!
run_cnn resnet50 20 43 2 "$A4" "$A4/logs" 0.80 &  PC=$!
echo "[$(date -u)] Batch 2a PIDs: $PA $PB $PC"
wait $PA $PB $PC
echo "[$(date -u)] Batch 2a complete."

# Batch 2b: 2 in parallel on GPUs 0,1
run_cnn resnet50 50 42 0 "$A4" "$A4/logs" 0.80 &  PD=$!
run_cnn resnet50 50 43 1 "$A4" "$A4/logs" 0.80 &  PE=$!
echo "[$(date -u)] Batch 2b PIDs: $PD $PE"
wait $PD $PE
echo "[$(date -u)] Phase 2 complete. All done."
