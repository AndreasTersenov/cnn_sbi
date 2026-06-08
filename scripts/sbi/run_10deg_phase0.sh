#!/usr/bin/env bash
# Phase 0 (autonomous): wait for the 20deg archival rsyncs -> VERIFY (rsync itemize shows 0 files
# differing) -> delete local 20deg (recoverable on /nas) -> launch run_10deg_build.sh.
# HARD GATE: never deletes unless the /nas copy is byte/size/mtime-identical. Self-contained so it
# survives context compaction. CPU/IO only.
set -u -o pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi; cd "$REPO/scripts/sbi"
CMP=results/exploratory/cross_maps_campaign
CACHE_SRC="$CMP/full_sphere_cache_grid"
CACHE_DST=/nas/tersenov/archive_20deg/full_sphere_cache_grid
TFDS_SRC=/home/tersenov/tensorflow_datasets
TFDS_DST=/nas/tersenov/archive_20deg/tensorflow_datasets
TFDS_LOG="$CMP/rsync_archive_tfds.log"
LOG="$CMP/run_10deg_phase0.log"
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
abort(){ log "ABORT: $*"; exit 1; }

log "=== PHASE 0 START (wait archival -> verify -> delete -> build) ==="

# 1. wait for BOTH archivals (cache rsync proc gone + 'ALL TFDS ARCHIVED' marker). Max ~16h.
tfds_done=0
for i in $(seq 1 192); do  # 192 * 5min = 16h
  cache_running=$(pgrep -f "rsync.*full_sphere_cache_grid" | wc -l)
  tfds_done=$(grep -c "ALL TFDS ARCHIVED" "$TFDS_LOG" 2>/dev/null || echo 0)
  if [ "$cache_running" -eq 0 ] && [ "$tfds_done" -ge 1 ]; then
    log "both archivals complete (iter $i)"; break
  fi
  [ $((i % 6)) -eq 0 ] && log "waiting... cache_running=$cache_running tfds_done=$tfds_done free=$(df -h /mnt|tail -1|awk '{print $4}')"
  sleep 300
done
[ "$tfds_done" -ge 1 ] && [ "$(pgrep -f 'rsync.*full_sphere_cache_grid' | wc -l)" -eq 0 ] \
  || abort "archival not complete after 16h (cache_running / tfds marker)"

# 2. VERIFY each archive: rsync dry-run itemize -> regular files still needing transfer (^>f). "0" =
# byte/size/mtime-identical. FAIL-SAFE: if the verify-rsync itself errors, return RSYNC_ERR (never "0").
verify(){ local s=$1 d=$2 out
  out=$(rsync -ani "$s/" "$d/" 2>/dev/null) || { echo "RSYNC_ERR"; return; }
  printf '%s\n' "$out" | grep -c '^>f'
}
log "verifying cache archive..."
nc=$(verify "$CACHE_SRC" "$CACHE_DST"); log "  cache files-differing: $nc"
[ "$nc" = "0" ] || abort "cache archive INCOMPLETE/ERROR ($nc) -> NOT deleting anything"
for dd in nbody_cosmogrid_dataset_tomo nbody_cosmogrid_dataset_tomo_cross; do
  n=$(verify "$TFDS_SRC/$dd" "$TFDS_DST/$dd"); log "  tfds/$dd files-differing: $n"
  [ "$n" = "0" ] || abort "tfds/$dd archive INCOMPLETE/ERROR ($n) -> NOT deleting anything"
done

# 3. delete local 20deg (verified identical on /nas; recoverable)
log "ARCHIVES VERIFIED (0 diffs). Deleting local 20deg..."
rm -rf "$CACHE_SRC"
for dd in nbody_cosmogrid_dataset_tomo nbody_cosmogrid_dataset_tomo_cross; do rm -rf "$TFDS_SRC/$dd"; done
log "deleted. free now: $(df -h /mnt|tail -1|awk '{print $4}')"

# 4. launch the build (smoke-gated; writes HANDOFF_10DEG.md)
log "launching run_10deg_build.sh ..."
bash run_10deg_build.sh >> "$CMP/run_10deg_build_outer.log" 2>&1
rc=$?
log "=== PHASE 0 DONE (build rc=$rc; see HANDOFF_10DEG.md) ==="
