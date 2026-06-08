#!/usr/bin/env bash
set -u -o pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi; cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
DC="$REPO/scripts/sbi/results/exploratory/definitive_comparison"; FF="$DC/fiducial_full200"
TPP="$FF/tarp_per_patch"; mkdir -p "$TPP/logs"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 PYTHONUNBUFFERED=1
GPU=1; N=200; M=2000
stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }; log(){ echo "[$(stamp)] $*" | tee -a "$TPP/STATUS.log"; }
feltnote(){ felt history append definitive-l1-vs-cnn-2026-05/finding-l1-patch-sensitivity-full200 --summary "$*" 2>/dev/null||true; }
log "=== TARP per-patch fiducial START (GPU$GPU N=$N M=$M) ==="
run(){ # label cache prefix summ transform clip minvar
  log "ARM $1"
  $PY tarp_per_patch_fiducial.py --train-cache-dir "$2" --cache-prefix "$3" \
    --summaries-npz "$FF/summaries/$4" --arm-label "$1" \
    --dumps-root "$TPP/dumps" --output-dir "$TPP/coverage" \
    --preproc-transform "$5" --clip-value "$6" --min-feature-variance "$7" \
    --seeds 41,42,43 --n-patches "$N" --m-samples "$M" --cuda-visible-devices "$GPU" \
    > "$TPP/logs/$1.log" 2>&1 && log "  OK $1" || log "  FAIL $1"
  grep -E "\[coverage\]|\[patch0\]" "$TPP/logs/$1.log" | while read -r l; do log "    $l"; done
}
run l1_autocross  "$DC/compressed/l1_autocross_split70_dv" l1  l1_autocross_S.npz  log1p-zscore 5.0 1e-5
run cnn_autocross "$DC/phaseA_tfdata_2026_05_30/compressed/autocross_s41" cnn cnn_autocross_S.npz none 0 1e-12
log "=== plotting TARP (run_tarp_coverage) ==="
$PY run_tarp_coverage.py --dumps-root "$TPP/dumps" --outdir "$TPP" > "$TPP/logs/plot.log" 2>&1 && log "  plot OK" || log "  plot FAIL"
touch "$TPP/.TARP_PERPATCH_DONE"; log "=== TARP per-patch COMPLETE ==="
feltnote "Per-patch coverage done: L1 + CNN. See tarp_per_patch/coverage/*/coverage.json + STATUS.log."
