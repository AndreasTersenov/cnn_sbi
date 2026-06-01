#!/usr/bin/env bash
# Phase B (definitive L1-vs-CNN) — jaxili MAF NDE on the Phase A compressed caches.
# Self-driving waiter: polls Phase A .done markers; as each compressor completes,
# trains the jaxili MAF NDE (validated end-to-end 2026-05-30 — needs ABSOLUTE paths).
#
#   compressor seed 41 -> NDE seeds 41,42,43 (plan-faithful 3-NDE-seed headline, perm 0)
#   compressor seed 42 -> NDE seed 41        (compressor-seed-variance bonus, perm 0)
#   compressor seed 43 -> NDE seed 41        (compressor-seed-variance bonus, perm 0)
#
# Perm 0 only tonight (cnn_obs.npz holds one perm; multi-perm needs --obs-files,
# the plan's VERIFY item). FoM here inherits the Phase A leakage (README_LEAKAGE.md):
# absolute FoM inflated; auto-vs-cross RELATIVE gain fair.
set -u -o pipefail

REPO=/mnt/home/tersenov/software/cnn_sbi
cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
NDE="$REPO/scripts/sbi/train_jaxili_from_compressed.py"
PA="$REPO/scripts/sbi/results/exploratory/definitive_comparison/phaseA_tfdata_2026_05_30"
PB="$REPO/scripts/sbi/results/exploratory/definitive_comparison/phaseB_tfdata_2026_05_30"
mkdir -p "$PB/posteriors" "$PB/logs"
GPU=1   # NDE is light (~4-8 GB, fast); shares a card with the running compressors

stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log(){ echo "[$(stamp)] $*" | tee -a "$PB/waiter.log"; }

# run_nde <arm> <compressor_seed> <nde_seeds_csv>
run_nde(){
  local arm="$1" cs="$2" seeds="$3"
  local cdir="$PA/compressed/${arm}_s${cs}"
  local odir="$PB/posteriors/${arm}"
  local label="${arm}_cs${cs}"
  local lg="$PB/logs/nde_${label}.log"
  mkdir -p "$odir"
  if [ -f "$PB/.ndedone_${label}" ]; then return 0; fi
  if [ ! -f "$cdir/cnn_train.npz" ]; then log "WAIT cache missing for $label"; return 1; fi
  log "NDE START $label seeds=$seeds -> $lg"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 PYTHONUNBUFFERED=1 \
  "$PY" -u "$NDE" \
    --compressed-dir "$cdir" \
    --arm-label "$label" \
    --output-dir "$odir" \
    --seeds "$seeds" \
    --cuda-visible-devices "$GPU" \
    > "$lg" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then touch "$PB/.ndedone_${label}"; log "NDE DONE  $label rc=0"
  else log "NDE FAIL  $label rc=$rc (see $lg)"; fi
}

log "=== Phase B NDE waiter START (poll Phase A markers) ==="
# (arm, compressor_seed, nde_seeds)
JOBS=(
  "autocross 41 41,42,43"
  "autoonly  41 41,42,43"
  "autocross 42 41"
  "autoonly  42 41"
  "autocross 43 41"
  "autoonly  43 41"
)
while :; do
  pending=0
  for j in "${JOBS[@]}"; do
    set -- $j; arm="$1"; cs="$2"; seeds="$3"
    [ -f "$PB/.ndedone_${arm}_cs${cs}" ] && continue
    if [ -f "$PA/.done_${arm}_s${cs}" ]; then
      run_nde "$arm" "$cs" "$seeds"
    else
      pending=1
    fi
  done
  # exit when Phase A fully done AND nothing pending
  if [ -f "$PA/.PHASEA_TFDATA_DONE" ] && [ $pending -eq 0 ]; then
    # final sweep to catch any last completed marker
    for j in "${JOBS[@]}"; do set -- $j; [ -f "$PB/.ndedone_${1}_cs${2}" ] || run_nde "$1" "$2" "$3"; done
    break
  fi
  sleep 60
done
log "=== Phase B NDE waiter COMPLETE ==="
touch "$PB/.PHASEB_TFDATA_DONE"
