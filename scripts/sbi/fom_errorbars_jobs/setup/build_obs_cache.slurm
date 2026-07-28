#!/bin/bash
#SBATCH --job-name=obs_cache_10deg
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/obs_cache_%j.out

# Rebuild the destroyed 10deg fiducial observation cache -- the observation set the
# 9000 mocks are drawn from. The original directory survived with ZERO files.
#
# Geometry and flags are NOT reconstructed here: they are copied verbatim from
# scripts/sbi/run_10deg_build.sh (Phase 2b), the surviving build script, and the
# 180 centres it implies were confirmed independently by bit-exact reprojection
# against the cross TFDS at 22 patch indices (centers_PAPER_180.json).
#
#   GEO      = --field-size 10 --field-npix 80 --n-centers 180 --center-nside 64 \
#              --min-separation-deg 14.2 --max-abs-lat 75
#   COMMON   = --regime nobnt --num-workers 50
#   FIDPERMS = 0..199
#
# BNT is deliberately absent: the cache is nobnt only, and BNT is applied later
# on-device as a channel mix (--flatsky-channel-mix bnt).
#
# Remaining defaults (sigma_e 0.26, galaxy_density 10.0, nside 512, lmax 1024,
# noise_seed_base 12345, map_label nobaryons) come from the builder itself and were
# each verified against the intact _bnt frozen sigma table -- sigma_pix matches to
# six decimals, and the reprojection match could not have happened at 1e-8 unless
# the noise seed formula (base + 100*cosmo_idx + perm) were exactly right.
#
# Source data: the read-only IDRIS shared copy of CosmoGridV1 in $DSDIR. The builder's
# _remap_path_par was patched for it (Titan hardcoded /home/tersenov + a grid->new_grid
# rename that does not apply here); all 200 perms of cosmo_fiducial were confirmed present.
# Output goes to $SCRATCH per guide section 4 (bulk data, not $WORK's inode-capped space).
#
# Centres are PINNED via --centers-npy rather than regenerated. _build_non_overlapping_centers
# orders candidates with np.argsort(|lat|), whose default quicksort is NOT stable, and every
# HEALPix ring is a block of exact ties -- so identical arguments give different tilings under
# different numpy versions. Measured directly: numpy 1.24.4 vs 1.26.4 share only 2 of 180
# centres, and only the 1.24.4 result matches the paper (22/22 measured). Pinning the verified
# file removes that hidden dependence.

set -uo pipefail
module purge
module load python/3.11.5

N=/lustre/fswork/projects/rech/nzu/ulx34io
S=/lustre/fsn1/projects/rech/nzu/ulx34io
PY=$N/envs/jaxili/bin/python
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
OUT=$S/recovery/full_sphere_cache_fiducial_10deg
CG=/lustre/fsmisc/dataset/CosmoGridV1

export TMPDIR=$S/recovery/tmp; mkdir -p "$TMPDIR"
export CG_PARENT=/lustre/fsmisc/dataset
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$SRC:${PYTHONPATH:-}

GEO="--field-size 10 --field-npix 80 --n-centers 180 --center-nside 64 --min-separation-deg 14.2 --max-abs-lat 75"
FIDPERMS=$(seq -s, 0 199)

echo "=== 10deg fiducial obs cache rebuild ==="
$PY -V
echo "out : $OUT"
echo "geo : $GEO"
echo "perms: 0..199 (the eval uses perm<50 for its 9000 obs; the full 200 matches the paper)"
mkdir -p "$OUT"

cd "$SRC"
$PY -u build_full_sphere_cross_cache.py \
    --cosmo-meta $CG/CosmoGridV1_metainfo.h5 \
    --cosmogrid-root $CG \
    --cosmo-subset fiducial --cosmo-id cosmo_fiducial \
    --realizations "$FIDPERMS" \
    $GEO \
    --centers-npy /lustre/fswork/projects/rech/prk/ulx34io/recovery/centers_PAPER_180.npy \
    --regime nobnt --num-workers 40 \
    --out-dir "$OUT"
rc=$?
echo "builder exit=$rc"
[ $rc -eq 0 ] || { echo "BUILD FAILED"; exit 1; }

echo; echo "=== verify_cache gate (ported from run_10deg_build.sh) ==="
$PY - "$OUT" 180 FIDUCIAL <<'PY'
import sys, glob, numpy as np
d, npat, lab = sys.argv[1], int(sys.argv[2]), sys.argv[3]
fs = sorted(glob.glob(d + "/nobnt/**/*.npz", recursive=True))
assert fs, f"[{lab}] no .npz produced in {d}"
z = np.load(fs[0], allow_pickle=True)
p = np.asarray(z["patches"]); c = np.asarray(z["patch_centers"])
assert p.shape == (npat, 80, 80, 10), f"[{lab}] patch shape {p.shape} != ({npat},80,80,10)"
assert np.isfinite(p).all(), f"[{lab}] non-finite patches"
assert np.abs(c[:, 1]).max() < 75, f"[{lab}] polar leak max|lat|={np.abs(c[:,1]).max():.1f}"
auto = np.abs(p[..., :4]).mean(); cross = np.abs(p[..., 4:]).mean()
assert auto > 0 and cross > 0 and cross < auto, \
    f"[{lab}] channel scales off auto={auto:.2e} cross={cross:.2e}"
print(f"[{lab}] OK: {len(fs)} files, patches{p.shape}, "
      f"max|lat|={np.abs(c[:,1]).max():.1f}, auto~{auto:.2e} cross~{cross:.2e}")

# extra gate: the centres written into the cache must equal the recovered geometry
ref = np.load("/lustre/fswork/projects/rech/prk/ulx34io/recovery/centers_PAPER_180.npy")
assert c.shape == ref.shape, f"centre shape {c.shape} != {ref.shape}"
dmax = float(np.abs(c.astype(np.float64) - ref.astype(np.float64)).max())
print(f"[{lab}] centres vs centers_PAPER_180.npy: max|diff| = {dmax:.3e}")
assert dmax < 1e-4, "cache centres do not match the recovered geometry"
print(f"[{lab}] GEOMETRY CONFIRMED")
PY

echo; echo "=== summary ==="
echo "npz files : $(find $OUT -name '*.npz' | wc -l)  (expect 200)"
echo "size      : $(du -sh $OUT | cut -f1)"
echo "manifest  : $OUT/manifest.json"
