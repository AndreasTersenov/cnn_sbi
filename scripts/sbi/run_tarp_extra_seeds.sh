#!/usr/bin/env bash
# Fill the idle window: TARP seeds 42 & 43 for all 6 arms (same cs41 compressors /
# L1 caches; the seed varies the NDE realization + test draw) -> proper 3-seed
# coverage bands via run_tarp_coverage.py. Runs on GPU 1 (free) in parallel with
# the steps block on GPU 0.
set -u -o pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi
cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
DC="$REPO/scripts/sbi/results/exploratory/definitive_comparison"
DUMPS="$DC/tarp_2026_05_31/dumps"; ROOT="$DC/tarp_2026_05_31"
mkdir -p "$ROOT/logs"; GPU=1; N=200; M=2000
stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }; log(){ echo "[$(stamp)] $*" | tee -a "$ROOT/extra_seeds.log"; }

declare -A CACHE=(
  [cnn_autocross_rnvp]="$DC/phaseA_tfdata_2026_05_30/compressed/autocross_s41"
  [cnn_autoonly_rnvp]="$DC/phaseA_tfdata_2026_05_30/compressed/autoonly_s41"
  [cnn_autocross_maf]="$DC/phaseA_maf_2026_05_31/compressed/autocross_s41"
  [cnn_autoonly_maf]="$DC/phaseA_maf_2026_05_31/compressed/autoonly_s41"
  [l1_autocross]="$DC/compressed/l1_autocross_split70_dv"
  [l1_autoonly]="$DC/compressed/l1_autoonly_split70_dv"
)
log "=== TARP extra seeds (42,43) START on GPU$GPU ==="
for seed in 42 43; do
  for arm in "${!CACHE[@]}"; do
    cdir="${CACHE[$arm]}"
    [ -f "$cdir/cnn_train.npz" ] || [ -f "$cdir/l1_train.npz" ] || { log "SKIP $arm s$seed (no cache)"; continue; }
    [ -f "$DUMPS/$arm/seed_${seed}/n${N}_m${M}/posterior_samples.npz" ] && { log "SKIP $arm s$seed (dumped)"; continue; }
    log "DUMP $arm seed=$seed"
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONUNBUFFERED=1 \
    "$PY" -u tarp_from_compressed.py --compressed-dir "$cdir" --arm-label "$arm" \
      --dumps-root "$DUMPS" --seed "$seed" --n-sims "$N" --m-samples "$M" \
      --cuda-visible-devices "$GPU" > "$ROOT/logs/dump_${arm}_s${seed}.log" 2>&1 \
      && log "DUMP DONE $arm s$seed" || log "DUMP FAIL $arm s$seed"
  done
done
log "=== re-plot all arms with 3-seed bands ==="
"$PY" -u run_tarp_coverage.py --dumps-root "$DUMPS" --outdir "$ROOT" --recompute \
  > "$ROOT/logs/plot_3seed.log" 2>&1 && log "PLOT DONE (3-seed bands)" || log "PLOT FAIL"
log "=== TARP extra seeds COMPLETE ==="; touch "$ROOT/.TARP_3SEED_DONE"
