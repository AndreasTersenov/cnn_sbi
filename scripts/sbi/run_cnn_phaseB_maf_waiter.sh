#!/usr/bin/env bash
# Phase B for the MAF-companion arms — jaxili MAF NDE on the Phase A MAF caches.
# Mirrors run_cnn_phaseB_nde_waiter.sh. Perm 0; compressor s41 -> NDE 41,42,43
# (headline), s42/s43 -> NDE 41 (compressor-seed variance).
set -u -o pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi
cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
NDE="$REPO/scripts/sbi/train_jaxili_from_compressed.py"
PA="$REPO/scripts/sbi/results/exploratory/definitive_comparison/phaseA_maf_2026_05_31"
PB="$REPO/scripts/sbi/results/exploratory/definitive_comparison/phaseB_maf_2026_05_31"
mkdir -p "$PB/posteriors" "$PB/logs"; GPU=1
stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }; log(){ echo "[$(stamp)] $*" | tee -a "$PB/waiter.log"; }
run_nde(){
  local arm="$1" cs="$2" seeds="$3"
  local cdir="$PA/compressed/${arm}_s${cs}" odir="$PB/posteriors/${arm}" label="${arm}_cs${cs}"
  mkdir -p "$odir"; [ -f "$PB/.ndedone_${label}" ] && return 0
  [ -f "$cdir/cnn_train.npz" ] || { log "WAIT cache $label"; return 1; }
  log "NDE START $label seeds=$seeds"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 PYTHONUNBUFFERED=1 \
  "$PY" -u "$NDE" --compressed-dir "$cdir" --arm-label "$label" --output-dir "$odir" \
    --seeds "$seeds" --cuda-visible-devices "$GPU" > "$PB/logs/nde_${label}.log" 2>&1
  local rc=$?; if [ $rc -eq 0 ]; then touch "$PB/.ndedone_${label}"; log "NDE DONE $label"; else log "NDE FAIL $label rc=$rc"; fi
}
log "=== Phase B MAF waiter START ==="
JOBS=("autocross 41 41,42,43" "autoonly 41 41,42,43" "autocross 42 41" "autoonly 42 41" "autocross 43 41" "autoonly 43 41")
while :; do
  pending=0
  for j in "${JOBS[@]}"; do set -- $j; [ -f "$PB/.ndedone_${1}_cs${2}" ] && continue
    if [ -f "$PA/.done_${1}_s${2}" ]; then run_nde "$1" "$2" "$3"; else pending=1; fi; done
  if [ -f "$PA/.PHASEA_MAF_DONE" ] && [ $pending -eq 0 ]; then
    for j in "${JOBS[@]}"; do set -- $j; [ -f "$PB/.ndedone_${1}_cs${2}" ] || run_nde "$1" "$2" "$3"; done; break; fi
  sleep 60
done
log "=== Phase B MAF waiter COMPLETE ==="; touch "$PB/.PHASEB_MAF_DONE"
