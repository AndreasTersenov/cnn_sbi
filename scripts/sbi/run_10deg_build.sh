#!/usr/bin/env bash
# Autonomous 10deg dataset build (Phase 1-4 of PLAN_10DEG_CAMPAIGN.md).
# Runs AFTER the 20deg archival is verified+deleted (disk freed). CPU-only (healpy SHT, 50 workers);
# touches no GPU. HARD GATES abort-and-log on any failure -> never produce/keep garbage.
# Stops after a VERIFIED 10deg TFDS + small fiducial cache; leaves loaders+campaign for Andreas.
set -u -o pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi
cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
CMP=results/exploratory/cross_maps_campaign
GRID10="$CMP/full_sphere_cache_grid_10deg"          # TRANSIENT (deleted after TFDS)
FID10="$CMP/full_sphere_cache_fiducial_10deg"       # KEPT (obs source for diagnostics)
TFDS_DIR=/home/tersenov/tensorflow_datasets         # local XFS
HANDOFF="$REPO/HANDOFF_10DEG.md"
LOG="$CMP/run_10deg_build.log"
GEO="--field-size 10 --field-npix 80 --n-centers 180 --center-nside 64 --min-separation-deg 14.2 --max-abs-lat 75"
COMMON="--regime nobnt --num-workers 50"
FIDPERMS=$(seq -s, 0 199); GRIDPERMS=0,1,2,3,4,5,6

stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log(){ echo "[$(stamp)] $*" | tee -a "$LOG"; }
hand(){ echo "$*" >> "$HANDOFF"; }
abort(){ log "ABORT: $*"; hand ""; hand "## ❌ ABORTED $(stamp)"; hand "$*"; hand "Disk: $(df -h /mnt|tail -1|awk '{print $4" free"}')"; exit 1; }

: > "$HANDOFF"
hand "# 10deg build — overnight handoff ($(stamp))"
hand "Autonomous build per PLAN_10DEG_CAMPAIGN.md. CPU-only. Gates abort-and-log."
hand ""
log "=== 10deg BUILD START === free: $(df -h /mnt|tail -1|awk '{print $4}')"

# Disk guard: need >=1.5TB free for the grid cache (~600G) + TFDS (~700G) peak.
FREE_G=$(df --output=avail -BG /mnt | tail -1 | tr -dc '0-9')
[ "$FREE_G" -ge 1400 ] || abort "Only ${FREE_G}G free (<1400G). Archival/delete likely not done. Not building."

# ---- helper: sanity-check a freshly built cache dir ----
verify_cache(){  # $1=dir $2=expected_npatch $3=label
  $PY - "$1" "$2" "$3" <<'PY' || return 1
import sys, glob, numpy as np
d, npat, lab = sys.argv[1], int(sys.argv[2]), sys.argv[3]
fs = sorted(glob.glob(d+"/nobnt/**/*.npz", recursive=True))
assert fs, f"[{lab}] no .npz produced in {d}"
z = np.load(fs[0], allow_pickle=True); p = np.asarray(z["patches"]); c = np.asarray(z["patch_centers"])
assert p.shape == (npat,80,80,10), f"[{lab}] patch shape {p.shape} != ({npat},80,80,10)"
assert np.isfinite(p).all(), f"[{lab}] non-finite patches"
assert np.abs(c[:,1]).max() < 75, f"[{lab}] polar leak max|lat|={np.abs(c[:,1]).max():.1f}"
auto = np.abs(p[...,:4]).mean(); cross = np.abs(p[...,4:]).mean()
assert auto > 0 and cross > 0 and cross < auto, f"[{lab}] channel scales off auto={auto:.2e} cross={cross:.2e}"
print(f"[{lab}] OK: {len(fs)} files, patches{p.shape}, max|lat|={np.abs(c[:,1]).max():.1f}, auto~{auto:.2e} cross~{cross:.2e}")
PY
}

# ================= PHASE 2a — SMOKE (fiducial, 3 perms) =================
log "PHASE 2a SMOKE: fiducial 10deg, 3 perms"
$PY build_full_sphere_cross_cache.py --cosmo-subset fiducial --cosmo-id cosmo_fiducial \
    --realizations 0,1,2 $GEO $COMMON --out-dir "${FID10}_smoke" >> "$LOG" 2>&1 \
    || abort "smoke build crashed (see $LOG)"
verify_cache "${FID10}_smoke" 180 "SMOKE" >> "$LOG" 2>&1 || abort "SMOKE verify failed (see $LOG)"
log "  SMOKE PASSED"; hand "- ✅ Phase 2a SMOKE passed (fiducial 3-perm, patches 180x80x80x10, no polar leak)."
rm -rf "${FID10}_smoke"

# ================= PHASE 2b — FIDUCIAL full (200 perms, KEPT) =================
log "PHASE 2b: fiducial 10deg, 200 perms (kept = obs source)"
$PY build_full_sphere_cross_cache.py --cosmo-subset fiducial --cosmo-id cosmo_fiducial \
    --realizations "$FIDPERMS" $GEO $COMMON --out-dir "$FID10" >> "$LOG" 2>&1 \
    || abort "fiducial build crashed"
verify_cache "$FID10" 180 "FIDUCIAL" >> "$LOG" 2>&1 || abort "FIDUCIAL verify failed"
NFID=$(find "$FID10" -name '*.npz' | wc -l)
log "  FIDUCIAL ok ($NFID npz)"; hand "- ✅ Phase 2b fiducial cache: $NFID npz (200 perms), kept at $FID10."

# ================= PHASE 2c — GRID full (TRANSIENT) =================
log "PHASE 2c: GRID 10deg (train+val, perms $GRIDPERMS) — the big one"
$PY build_full_sphere_cross_cache.py --cosmo-subset grid \
    --realizations "$GRIDPERMS" $GEO $COMMON --out-dir "$GRID10" >> "$LOG" 2>&1 \
    || abort "GRID build crashed"
verify_cache "$GRID10" 180 "GRID" >> "$LOG" 2>&1 || abort "GRID verify failed"
NGRID=$(find "$GRID10" -name '*.npz' | wc -l)
log "  GRID ok ($NGRID npz); free: $(df -h /mnt|tail -1|awk '{print $4}')"
hand "- ✅ Phase 2c grid cache (transient): $NGRID npz."

# ================= PHASE 3 — TFDS (cross, ordered) =================
log "PHASE 3: build 10deg cross TFDS from grid cache"
TFDS_BIN=/home/tersenov/anaconda3/envs/jaxili/bin/tfds
CROSS_TFDS_CACHE_DIR="$REPO/scripts/sbi/$GRID10" CROSS_TFDS_BUILD_WORKERS=50 \
  "$TFDS_BIN" build tf_dataset_nbody_tomo_cross.py \
    --config grid_10deg_80px_nonoverlap180 --file_format=tfrecord --data_dir "$TFDS_DIR" \
    >> "$LOG" 2>&1 || abort "TFDS build crashed (see $LOG)"
# verify: count + bit-exact one example vs the cache
$PY - "$GRID10" "$TFDS_DIR" <<'PY' >> "$LOG" 2>&1 || abort "TFDS verify FAILED -> keeping grid cache, not deleting."
import sys, glob, numpy as np, tensorflow_datasets as tfds, tensorflow as tf
gdir, ddir = sys.argv[1], sys.argv[2]
b = tfds.builder("nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180", data_dir=ddir)
info = b.info; n = sum(s.num_examples for s in info.splits.values())
ncache = len(glob.glob(gdir+"/nobnt/**/*.npz", recursive=True)) * 180
assert n == ncache, f"TFDS count {n} != cache {ncache}"
# bit-exact: pull one example, find its cache patch by (cosmo_idx,perm,patch)
ds = b.as_dataset(split="train").take(1)
for ex in ds:
    ci,pm,pk = int(ex["cosmo_idx"]), int(ex["perm"]), int(ex["patch"])
    m = ex["map_nbody"].numpy()
import glob as g
# locate the cache file for ci/pm
cand = [f for f in glob.glob(gdir+"/nobnt/**/*.npz", recursive=True) if f"_perm{pm}.npz" in f]
hit=None
for f in cand:
    z=np.load(f, allow_pickle=True)
    if int(z["cosmo_idx"])==ci: hit=np.asarray(z["patches"])[pk]; break
assert hit is not None and np.array_equal(hit, m), "TFDS map not bit-exact vs cache"
print(f"TFDS OK: {n} examples == cache; bit-exact patch verified (ci={ci},perm={pm},patch={pk})")
PY
log "  TFDS verified"; hand "- ✅ Phase 3 TFDS built + verified (count match + bit-exact)."

# ================= PHASE 4 — delete transient grid cache =================
log "PHASE 4: deleting transient grid cache $GRID10"
rm -rf "$GRID10"
log "  done; free: $(df -h /mnt|tail -1|awk '{print $4}')"
hand "- ✅ Phase 4 transient grid cache deleted."
hand ""
hand "## ✅ 10deg DATASET READY $(stamp)"
hand "- TFDS: nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180 @ $TFDS_DIR"
hand "- Fiducial obs cache (kept): $FID10"
hand "- Free disk: $(df -h /mnt|tail -1|awk '{print $4}')"
hand ""
hand "## NEXT (morning, with Andreas) — NOT done autonomously:"
hand "1. L1-reads-TFDS loader (science-critical: channel_empirical_global noise, pca OFF, no PCA;"
hand "   verify datavector vs a reference). 2. CNN loader read_config retune. 3. Clean compressor/NDE"
hand "   split by cosmo_idx (1-630 vs 631-899). 4. Run 4 arms + diagnostics. 5. Headline test: does"
hand "   L1's -0.37sigma fiducial w0 offset shrink at 10deg (flat-sky)? See PLAN_10DEG_CAMPAIGN.md."
log "=== 10deg BUILD COMPLETE ==="
