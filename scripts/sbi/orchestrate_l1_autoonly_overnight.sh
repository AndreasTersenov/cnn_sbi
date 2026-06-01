#!/usr/bin/env bash
# Orchestrator for the L1 auto-only (route-matched, flip=False) re-run.
# Assumes arm-1 warm-up (l1_autoonly_fulltrain s41/p0) already running on GPU 0.
# Both GPUs free => launches arm-2 (l1_autoonly_split70 s41/p0) on GPU 1 immediately,
# waits for both datavector caches + warm-up posteriors, fans out the 16 remaining
# (cache hits), then re-runs analyze_l1_autocross_definitive.py (now includes the
# auto-only arms). File-based polling only.
set -uo pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi; cd "$REPO"
RUN=scripts/sbi/run_l1_autoonly_definitive.sh
OUT=scripts/sbi/results/exploratory/definitive_comparison
LOGS=$OUT/logs
OLOG=$LOGS/orchestrator_autoonly.log
STAT=$OUT/OVERNIGHT_STATUS_AUTOONLY.md

A1_CACHE=$OUT/compressed/l1_autoonly_fulltrain_dv/l1_train.npz
A2_CACHE=$OUT/compressed/l1_autoonly_split70_dv/l1_train.npz
A1_POST=$OUT/posteriors/l1_autoonly_fulltrain/l1_autoonly_fulltrain_s41_p0.npy
A2_POST=$OUT/posteriors/l1_autoonly_split70/l1_autoonly_split70_s41_p0.npy

mkdir -p "$LOGS"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$OLOG"; }
stat(){ printf '%s\n\nUpdated: %s\n' "$1" "$(date)" > "$STAT"; }

log "orchestrator_autoonly start (PID $$). arm-1 warm-up assumed running on GPU 0."
stat "# L1 auto-only re-run — RUNNING (both warm-ups)"
log "launching arm-2 warm-up on GPU 1"
setsid nohup bash "$RUN" warmup_arm2 1 > "$LOGS/autoonly_warmup_arm2_driver.log" 2>&1 < /dev/null &

log "waiting for both datavector caches..."
while [ ! -f "$A1_CACHE" ] || [ ! -f "$A2_CACHE" ]; do sleep 30; done
log "both caches present; waiting for both warm-up posteriors..."
while [ ! -f "$A1_POST" ] || [ ! -f "$A2_POST" ]; do sleep 30; done
log "both warm-up posteriors present. Fanning out."
stat "# L1 auto-only re-run — fan-out (16 runs, cache hits)"

setsid nohup bash "$RUN" fanout_arm1 0 > "$LOGS/autoonly_fanout_arm1_driver.log" 2>&1 < /dev/null & F1=$!
setsid nohup bash "$RUN" fanout_arm2 1 > "$LOGS/autoonly_fanout_arm2_driver.log" 2>&1 < /dev/null & F2=$!
wait "$F1" "$F2" 2>/dev/null || true
for i in $(seq 1 40); do
  n1=$(ls "$OUT/posteriors/l1_autoonly_fulltrain"/l1_autoonly_fulltrain_s*_p*.npy 2>/dev/null | wc -l)
  n2=$(ls "$OUT/posteriors/l1_autoonly_split70"/l1_autoonly_split70_s*_p*.npy 2>/dev/null | wc -l)
  log "posteriors so far: arm1=$n1/9 arm2=$n2/9"
  [ "$n1" -ge 9 ] && [ "$n2" -ge 9 ] && break
  sleep 20
done

log "running analysis (now includes auto-only route-matched arms)..."
stat "# L1 auto-only re-run — ANALYSIS"
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n jaxili python scripts/sbi/analyze_l1_autocross_definitive.py \
  > "$LOGS/analysis_autoonly.log" 2>&1 || log "WARN analysis nonzero"
log "DONE."
touch "$OUT/.OVERNIGHT_AUTOONLY_DONE"
stat "# L1 auto-only re-run — DONE
See DEFINITIVE_L1_SUMMARY.md (route-matched gain section)."
