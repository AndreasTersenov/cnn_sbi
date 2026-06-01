#!/usr/bin/env bash
# TARP coverage for the 4 "final definitive" CNN arms (seed 41), then plot with
# the repo's existing run_tarp_coverage.py (3-D + 6-D, per-arm + overlay).
# Dumper = tarp_from_compressed.py (reuses train_jaxili_from_compressed NDE).
set -u -o pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi
cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
DC="$REPO/scripts/sbi/results/exploratory/definitive_comparison"
ROOT="$DC/tarp_2026_05_31"; DUMPS="$ROOT/dumps"
mkdir -p "$DUMPS" "$ROOT/logs"
GPU=1; N=200; M=2000
stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }; log(){ echo "[$(stamp)] $*" | tee -a "$ROOT/tarp.log"; }

# arm-label  ->  compressed-dir (seed-41 compressor)
declare -A CACHE=(
  [cnn_autocross_rnvp]="$DC/phaseA_tfdata_2026_05_30/compressed/autocross_s41"
  [cnn_autoonly_rnvp]="$DC/phaseA_tfdata_2026_05_30/compressed/autoonly_s41"
  [cnn_autocross_maf]="$DC/phaseA_maf_2026_05_31/compressed/autocross_s41"
  [cnn_autoonly_maf]="$DC/phaseA_maf_2026_05_31/compressed/autoonly_s41"
)
ORDER=(cnn_autocross_rnvp cnn_autoonly_rnvp cnn_autocross_maf cnn_autoonly_maf)

log "=== TARP dumps START (N=$N M=$M seed=41 GPU$GPU) ==="
for arm in "${ORDER[@]}"; do
  cdir="${CACHE[$arm]}"
  if [ ! -f "$cdir/cnn_train.npz" ]; then log "SKIP $arm (cache missing: $cdir)"; continue; fi
  if [ -f "$DUMPS/$arm/seed_41/n${N}_m${M}/posterior_samples.npz" ]; then log "SKIP $arm (dumped)"; continue; fi
  log "DUMP $arm <- $cdir"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONUNBUFFERED=1 \
  "$PY" -u tarp_from_compressed.py --compressed-dir "$cdir" --arm-label "$arm" \
    --dumps-root "$DUMPS" --seed 41 --n-sims "$N" --m-samples "$M" \
    --cuda-visible-devices "$GPU" > "$ROOT/logs/dump_${arm}.log" 2>&1 \
    && log "DUMP DONE $arm" || log "DUMP FAIL $arm (see logs/dump_${arm}.log)"
done

log "=== plotting with run_tarp_coverage.py (dim 3 + 6) ==="
"$PY" -u run_tarp_coverage.py --dumps-root "$DUMPS" --outdir "$ROOT" \
  > "$ROOT/logs/plot.log" 2>&1 && log "PLOT DONE" || log "PLOT FAIL (see logs/plot.log)"
log "=== TARP CNN arms COMPLETE ==="; touch "$ROOT/.TARP_CNN_DONE"
