#!/usr/bin/env bash
# Master overnight orchestrator for the L1 auto+cross definitive campaign.
#
# Assumes arm-1 warm-up (l1_autocross_fulltrain s41/p0, flip=False) is ALREADY
# running on GPU 0 (launched separately, route confirmed). This script:
#   1. snapshots the flip A/B verdict from the F-leg the moment it lands
#   2. launches arm-2 warm-up (l1_autocross_split70 s41/p0) on GPU 1 when free
#      (falls back to GPU 0 if arm-1 finishes first)
#   3. waits for BOTH datavector caches + warm-up posteriors
#   4. fans out the 16 remaining (seed,perm) runs (cache hits, fast)
#   5. runs analyze_l1_autocross_definitive.py
# File-based polling only (no pkill, no PID assumptions). Logs to logs/orchestrator.log
# and writes progress to OVERNIGHT_STATUS.md.
set -uo pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi; cd "$REPO"
RUN=scripts/sbi/run_l1_autocross_definitive.sh
OUT=scripts/sbi/results/exploratory/definitive_comparison
LOGS=$OUT/logs
OLOG=$LOGS/orchestrator.log
STAT=$OUT/OVERNIGHT_STATUS.md

A1_CACHE=$OUT/compressed/l1_autocross_fulltrain_dv/l1_train.npz
A2_CACHE=$OUT/compressed/l1_autocross_split70_dv/l1_train.npz
A1_POST=$OUT/posteriors/l1_autocross_fulltrain/l1_autocross_fulltrain_s41_p0.npy
A2_POST=$OUT/posteriors/l1_autocross_split70/l1_autocross_split70_s41_p0.npy
VERDICT=$OUT/flip_ab_Fleg.fom.json
FLEG=/tmp/l1_flip_ab/F/post.fom.json

mkdir -p "$LOGS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$OLOG"; }
stat(){ printf '%s\n\nUpdated: %s\n' "$1" "$(date)" > "$STAT"; }
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null || echo 99999; }
snapshot_verdict(){
  if [ ! -f "$VERDICT" ] && [ -f "$FLEG" ]; then
    cp "$FLEG" "$VERDICT" 2>/dev/null && log "flip A/B verdict snapshotted: $(cat "$VERDICT" | tr -d '\n')"
  fi
}

log "orchestrator start (PID $$). arm-1 warm-up assumed running on GPU 0."
stat "# Overnight L1 auto+cross — RUNNING

Phase: waiting to launch arm-2 warm-up (GPU 1 free or GPU 0 frees)."

# ---- Phase 1: launch arm-2 warm-up when a GPU frees ----
A2GPU=""
for i in $(seq 1 480); do            # up to 8 h
  snapshot_verdict
  if [ "$(gpu_used 1)" -lt 3000 ]; then A2GPU=1; break; fi
  if [ -f "$A1_POST" ]; then A2GPU=0; break; fi   # arm-1 done -> GPU 0 free
  sleep 60
done
[ -z "$A2GPU" ] && { log "ERROR: no GPU freed in 8h; aborting arm-2 launch"; stat "# ABORTED: no GPU for arm-2"; exit 1; }
snapshot_verdict
log "launching arm-2 warm-up on GPU $A2GPU"
setsid nohup bash "$RUN" warmup_arm2 "$A2GPU" > "$LOGS/warmup_arm2_driver.log" 2>&1 < /dev/null &
stat "# Overnight L1 auto+cross — RUNNING

Phase: both warm-ups running (arm1 GPU0, arm2 GPU$A2GPU). Computing datavectors (~2-4h)."

# ---- Phase 2: wait for both datavector caches + warm-up posteriors ----
log "waiting for both datavector caches..."
while [ ! -f "$A1_CACHE" ] || [ ! -f "$A2_CACHE" ]; do snapshot_verdict; sleep 60; done
log "both datavector caches present."
log "waiting for both warm-up posteriors..."
while [ ! -f "$A1_POST" ] || [ ! -f "$A2_POST" ]; do snapshot_verdict; sleep 60; done
log "both warm-up posteriors present."
stat "# Overnight L1 auto+cross — RUNNING

Phase: warm-ups done. Fanning out 16 remaining runs (cache hits)."

# ---- Phase 3: fan out the remaining 8/arm (cache hits -> fast NDE) ----
log "fan-out arm1 (GPU 0) + arm2 (GPU 1) in parallel..."
setsid nohup bash "$RUN" fanout_arm1 0 > "$LOGS/fanout_arm1_driver.log" 2>&1 < /dev/null &
F1=$!
setsid nohup bash "$RUN" fanout_arm2 1 > "$LOGS/fanout_arm2_driver.log" 2>&1 < /dev/null &
F2=$!
wait "$F1" "$F2" 2>/dev/null || true
# belt-and-braces: wait until 9 posteriors per arm exist (or 20 min idle)
for i in $(seq 1 40); do
  n1=$(ls "$OUT/posteriors/l1_autocross_fulltrain"/l1_autocross_fulltrain_s*_p*.npy 2>/dev/null | wc -l)
  n2=$(ls "$OUT/posteriors/l1_autocross_split70"/l1_autocross_split70_s*_p*.npy 2>/dev/null | wc -l)
  log "posteriors so far: arm1=$n1/9 arm2=$n2/9"
  [ "$n1" -ge 9 ] && [ "$n2" -ge 9 ] && break
  sleep 30
done

# ---- Phase 4: analysis ----
log "running analysis..."
stat "# Overnight L1 auto+cross — ANALYSIS

Fan-out complete. Computing FoM3 + corners."
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n jaxili python scripts/sbi/analyze_l1_autocross_definitive.py \
  > "$LOGS/analysis.log" 2>&1 || log "WARN analysis returned nonzero"
log "DONE. summary: $OUT/DEFINITIVE_L1_SUMMARY.md ; figures: $OUT/figures/definitive_l1/"
touch "$OUT/.OVERNIGHT_L1_DONE"
stat "# Overnight L1 auto+cross — DONE

See DEFINITIVE_L1_SUMMARY.md, definitive_l1_fom3.csv, figures/definitive_l1/.
flip A/B verdict: $( [ -f "$VERDICT" ] && cat "$VERDICT" | tr -d '\n' || echo 'F-leg not captured' )"
