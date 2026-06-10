#!/bin/bash
# Autonomous overnight pipeline for the flat-local CNN phase.
# Waits for the population sweep, then: headline table -> SBC -> L-C2ST -> representative
# corners -> full consolidation (table + overlays). Each step logged; PASS/FAIL appended to
# STATUS_OVERNIGHT.md. Launch detached: setsid nohup bash overnight_cnn_pipeline.sh &
set -u
SBI=/mnt/home/tersenov/software/cnn_sbi/scripts/sbi
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
CNNP=$SBI/results/exploratory/flatsky_cross_2026_06/cnn_phase
LOGD=$CNNP/overnight_logs
STATUS=$CNNP/STATUS_OVERNIGHT.md
mkdir -p "$LOGD"
cd "$SBI" || exit 1
export PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$STATUS"; }
step() {  # step "name" "logfile" command...
  local name="$1"; local lf="$2"; shift 2
  log "START $name"
  if "$@" > "$LOGD/$lf" 2>&1; then log "PASS  $name"; else log "FAIL  $name (rc=$?, see $LOGD/$lf)"; fi
}

echo "# Overnight CNN pipeline status" > "$STATUS"
log "PIPELINE START (pid $$)"

# 1. Wait for the population sweep to finish.
log "waiting for population sweep (run_flatsky_cnn_population_sweep.py) ..."
while pgrep -f "[r]un_flatsky_cnn_population_sweep.py" >/dev/null 2>&1; do sleep 60; done
log "population sweep no longer running"

# 2. Headline FoM table ASAP (consolidate is defensive; overlays skipped until repr corners exist).
step "headline-consolidate" "consolidate_early.log" $PY consolidate_cnn_vs_l1.py
log "headline table -> FLATSKY_CNN_RESULT.md (overlays added after representative corners)"

# 3. GATE C SBC (CPU, reuses TARP dumps).
step "sbc" "sbc.log" $PY compute_sbc_from_tarp_dumps_cnn.py

# 4. GATE C L-C2ST (GPU 1+2).
step "lc2st" "lc2st_orch.log" $PY run_flatsky_cnn_gate_c_lc2st.py --gpus 1,2 --mem-fraction 0.5

# 5. Representative corners 3-seed, all arms (GPU 1+2).
step "repr-corners" "repr_orch.log" $PY run_flatsky_cnn_repr_corners.py --gpus 1,2 --mem-fraction 0.45

# 6. Final consolidation (now with overlays).
step "final-consolidate" "consolidate_final.log" $PY consolidate_cnn_vs_l1.py

log "PIPELINE DONE"
echo "" >> "$STATUS"
echo "Deliverables: FLATSKY_CNN_RESULT.md (root), cnn_phase/figs/ (overlays + bars), " >> "$STATUS"
echo "cnn_phase/gate_c/{sbc,lc2st,tarp_drp}/ (calibration)." >> "$STATUS"
