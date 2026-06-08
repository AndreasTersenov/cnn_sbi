#!/usr/bin/env bash
# TARP coverage for the 2 NEW definitive arms not covered by run_tarp_cnn_arms.sh:
#   - cnn_autocross_rnvp_std   : same compressor cache as autocross, but the NDE
#                                trains on z-scored summaries (--standardize-summary)
#                                -> a genuinely different NDE -> needs its own TARP.
#   - cnn_auto_native_rnvp     : the native-TFDS auto-only compressor (different
#                                compressor entirely) -> needs its own TARP.
# Multi-perm arms are NOT here: they reuse the SAME compressed cache + NDE seeds as
# cnn_autocross_rnvp / cnn_autoonly_rnvp (the perm only changes the single obs map,
# which never enters TARP), so their coverage == the core RealNVP arms' (already done).
# Dumps land in the SAME tree as the core arms so the re-plot overlays all 8 arms.
set -u -o pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi
cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
DC="$REPO/scripts/sbi/results/exploratory/definitive_comparison"
ROOT="$DC/tarp_2026_05_31"; DUMPS="$ROOT/dumps"
mkdir -p "$DUMPS" "$ROOT/logs"
GPU=1; N=200; M=2000
SEEDS=(41 42 43)
stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log(){ echo "[$(stamp)] $*" | tee -a "$ROOT/tarp_new_arms.log"; }

# arm-label -> compressed-dir
declare -A CACHE=(
  [cnn_autocross_rnvp_std]="$DC/phaseA_tfdata_2026_05_30/compressed/autocross_s41"
  [cnn_auto_native_rnvp]="$DC/steps_overnight_2026_05_31/nativeauto/compressed/s41"
)
# arm-label -> extra flags (std arm z-scores the summary)
declare -A EXTRA=(
  [cnn_autocross_rnvp_std]="--standardize-summary"
  [cnn_auto_native_rnvp]=""
)
ORDER=(cnn_autocross_rnvp_std cnn_auto_native_rnvp)

log "=== TARP NEW arms START (N=$N M=$M seeds=${SEEDS[*]} GPU$GPU) ==="
for arm in "${ORDER[@]}"; do
  cdir="${CACHE[$arm]}"; extra="${EXTRA[$arm]}"
  if [ ! -f "$cdir/cnn_train.npz" ]; then log "SKIP $arm (cache missing: $cdir)"; continue; fi
  for s in "${SEEDS[@]}"; do
    if [ -f "$DUMPS/$arm/seed_${s}/n${N}_m${M}/posterior_samples.npz" ]; then
      log "SKIP $arm s$s (dumped)"; continue
    fi
    log "DUMP $arm s$s <- $cdir ${extra:+($extra)}"
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONUNBUFFERED=1 \
    "$PY" -u tarp_from_compressed.py --compressed-dir "$cdir" --arm-label "$arm" \
      --dumps-root "$DUMPS" --seed "$s" --n-sims "$N" --m-samples "$M" \
      --cuda-visible-devices "$GPU" $extra \
      > "$ROOT/logs/dump_${arm}_s${s}.log" 2>&1 \
      && log "DUMP DONE $arm s$s" || log "DUMP FAIL $arm s$s (see logs/dump_${arm}_s${s}.log)"
  done
done

log "=== re-plot ALL arms with run_tarp_coverage.py (dim 3 + 6) ==="
"$PY" -u run_tarp_coverage.py --dumps-root "$DUMPS" --outdir "$ROOT" \
  > "$ROOT/logs/plot_new_arms.log" 2>&1 && log "PLOT DONE" || log "PLOT FAIL (see logs/plot_new_arms.log)"
log "=== TARP NEW arms COMPLETE ==="; touch "$ROOT/.TARP_NEW_ARMS_DONE"
