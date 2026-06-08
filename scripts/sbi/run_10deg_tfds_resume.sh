#!/usr/bin/env bash
# RESUME the 10deg build from the intact grid cache: build the cross TFDS programmatically
# (beam-free, bypasses the broken `tfds` CLI) -> verify (count + bit-exact) -> delete transient
# grid cache. Phases 3-4 of run_10deg_build.sh; the SHT cache (fiducial+grid) is already built.
set -u -o pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi; cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
CMP=results/exploratory/cross_maps_campaign
GRID10="$CMP/full_sphere_cache_grid_10deg"          # TRANSIENT
TFDS_DIR=/home/tersenov/tensorflow_datasets
HANDOFF="$REPO/HANDOFF_10DEG.md"
LOG="$CMP/run_10deg_tfds_resume.log"
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
hand(){ echo "$*" >> "$HANDOFF"; }
abort(){ log "ABORT: $*"; hand ""; hand "## ❌ TFDS RESUME ABORTED $(date -u +%H:%M:%SZ): $*"; exit 1; }

log "=== TFDS RESUME (programmatic, beam-free) ==="
hand ""; hand "## TFDS resume (programmatic, beam-free; tfds CLI needs apache_beam which is absent) $(date -u +%H:%M:%SZ)"

# Phase 3 — build TFRecord from the grid cache.
# LEAN + POLITE: CPU-only TF (no GPU context per worker), 1 thread/worker, few workers — so it
# coexists with other users' jobs on a saturated node (50 GPU-grabbing thread-heavy workers
# crashed/starved here). 8 single-threaded workers fit the spare cores and won't thrash.
CROSS_TFDS_CACHE_DIR="$REPO/scripts/sbi/$GRID10" CROSS_TFDS_BUILD_WORKERS=8 TFDS_DATA_DIR="$TFDS_DIR" \
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1 \
  $PY build_10deg_tfds.py >> "$LOG" 2>&1 || abort "programmatic TFDS build crashed (see $LOG)"

# verify: count + bit-exact one example vs the cache
$PY - "$GRID10" "$TFDS_DIR" <<'PY' >> "$LOG" 2>&1 || abort "TFDS verify FAILED -> keeping grid cache, not deleting."
import sys, glob, numpy as np
sys.path.insert(0, ".")
import tf_dataset_nbody_tomo_cross  # noqa
import tensorflow_datasets as tfds
gdir, ddir = sys.argv[1], sys.argv[2]
b = tfds.builder("nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180", data_dir=ddir)
n = sum(s.num_examples for s in b.info.splits.values())
ncache = len(glob.glob(gdir+"/nobnt/**/*.npz", recursive=True)) * 180
assert n == ncache, f"TFDS count {n} != cache {ncache}"
ex = next(iter(b.as_dataset(split="train").take(1)))
ci, pm, pk, m = int(ex["cosmo_idx"]), int(ex["perm"]), int(ex["patch"]), ex["map_nbody"].numpy()
hit = None
for f in glob.glob(gdir+"/nobnt/**/*.npz", recursive=True):
    if f"_perm{pm}.npz" not in f: continue
    z = np.load(f, allow_pickle=True)
    if int(z["cosmo_idx"]) == ci: hit = np.asarray(z["patches"])[pk]; break
assert hit is not None and np.array_equal(hit, m), "TFDS map not bit-exact vs cache"
print(f"TFDS OK: {n} examples == cache; bit-exact verified (ci={ci},perm={pm},patch={pk})")
PY
log "  TFDS verified"; hand "- ✅ Phase 3 TFDS built (programmatic) + verified (count + bit-exact)."

# Phase 4 — delete transient grid cache
log "PHASE 4: deleting transient grid cache"
rm -rf "$GRID10"
hand "- ✅ Phase 4 transient grid cache deleted."
hand ""; hand "## ✅ 10deg DATASET READY (resumed) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
hand "- TFDS: nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180 @ $TFDS_DIR"
hand "- Fiducial obs cache (kept): $CMP/full_sphere_cache_fiducial_10deg"
hand "- Free disk: $(df -h /mnt|tail -1|awk '{print $4}')"
hand ""; hand "## NEXT (morning, with Andreas): L1-reads-TFDS loader (channel_empirical_global, PCA OFF),"
hand "CNN read_config retune, clean split by cosmo_idx, run 4 arms + diagnostics. See PLAN_10DEG_CAMPAIGN.md."
log "=== TFDS RESUME COMPLETE ==="
