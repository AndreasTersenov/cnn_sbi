#!/usr/bin/env bash
# #2: stratified varied-theta TARP — is the HIGH-FoM3 tercile calibrated? (L1 vs CNN)
set -u -o pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi; cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
DC="$REPO/scripts/sbi/results/exploratory/definitive_comparison"; FF="$DC/fiducial_full200"
ST="$FF/tarp_stratified"; mkdir -p "$ST/logs"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 PYTHONUNBUFFERED=1
GPU=1; N=600; M=2000
stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }; log(){ echo "[$(stamp)] $*" | tee -a "$ST/STATUS.log"; }
feltnote(){ felt history append definitive-l1-vs-cnn-2026-05/finding-l1-patch-sensitivity-full200 --summary "$*" 2>/dev/null||true; }
log "=== STRATIFIED VAL TARP START (GPU$GPU N=$N M=$M) ==="
run(){ # label cache prefix transform clip minvar
  log "ARM $1"
  $PY tarp_stratified_val.py --train-cache-dir "$2" --cache-prefix "$3" --arm-label "$1" \
    --dumps-root "$ST/dumps" --preproc-transform "$4" --clip-value "$5" --min-feature-variance "$6" \
    --seeds 41,42,43 --n-points "$N" --m-samples "$M" --cuda-visible-devices "$GPU" \
    > "$ST/logs/$1.log" 2>&1 && log "  OK $1" || log "  FAIL $1"
  grep -E "terciles:|LOW:|MID:|HIGH:" "$ST/logs/$1.log" | while read -r l; do log "    $l"; done
}
run l1_autocross  "$DC/compressed/l1_autocross_split70_dv" l1  log1p-zscore 5.0 1e-5
run cnn_autocross "$DC/phaseA_tfdata_2026_05_30/compressed/autocross_s41" cnn none 0 1e-12
log "=== plotting TARP per tercile ==="
$PY run_tarp_coverage.py --dumps-root "$ST/dumps" --outdir "$ST" > "$ST/logs/plot.log" 2>&1 && log "  plot OK" || log "  plot FAIL"
touch "$ST/.DONE"; log "=== STRATIFIED VAL TARP COMPLETE ==="
feltnote "Stratified varied-theta TARP done (L1+CNN, FoM3 terciles). See tarp_stratified/figures + STATUS.log for HIGH-tercile calibration verdict."
