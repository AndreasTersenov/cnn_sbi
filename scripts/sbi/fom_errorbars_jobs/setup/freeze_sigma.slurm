#!/bin/bash
#SBATCH --job-name=freeze_sigma_nobnt
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/freeze_sigma_%j.out

# Regenerate flatsky_cross_noise_sigma.npz -- the NO-BNT frozen per-(channel,scale)
# noise sigma table, which is the SNR denominator for the flat-local L1 binning.
#
# The original survived as 5462 bytes of pure zeros (a RAID0 hole that was NOT in the
# 30,488-entry damaged inventory). Its _bnt and _whiten twins are intact, which is how
# every parameter below was pinned; the defaults here match that twin exactly:
#     n_real 48, n_scales 5, seed_base 987654321, op both (16 channels)
# The BNT rows of the table are unaffected -- they use the intact _bnt table -- so only
# the four no-BNT rows depend on this file.
#
# The script reads its physical constants straight out of the fiducial cache npz
# ("so they cannot drift"), which is why it had to wait for the cache rebuild.

set -uo pipefail
module purge
module load python/3.11.5

N=/lustre/fswork/projects/rech/nzu/ulx34io
S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
OUT=$S/recovery/sigma_tables

export TMPDIR=$S/recovery/tmp; mkdir -p "$TMPDIR" "$OUT"
export CG_PARENT=/lustre/fsmisc/dataset
export FLATSKY_FID_CACHE=$S/recovery/full_sphere_cache_fiducial_10deg/nobnt/obs
export FLATSKY_FID_H5=/lustre/fsmisc/dataset/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/perm_0000/projected_probes_maps_nobaryons512.h5
export FLATSKY_OUT_DIR=$OUT
export PYTHONPATH=$SRC:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== regenerate no-BNT frozen sigma table ==="
echo "cache: $FLATSKY_FID_CACHE  ($(ls $FLATSKY_FID_CACHE/*.npz 2>/dev/null | wc -l) npz)"
echo "out  : $OUT"

cd "$SRC"
$N/envs/jaxili/bin/python -u freeze_flatsky_cross_noise.py \
    --op both --n-real 48 --n-scales 5 --seed-base 987654321 \
    --workers 32 --out-dir "$OUT"
rc=$?
echo "exit=$rc"
[ $rc -eq 0 ] || { echo "FREEZE FAILED"; exit 1; }

echo; echo "=== cross-check the new no-BNT table against the intact _bnt twin ==="
$N/envs/jaxili/bin/python - <<'PY'
import numpy as np, glob, os
NEW = glob.glob("/lustre/fsn1/projects/rech/nzu/ulx34io/recovery/sigma_tables/"
                "flatsky_cross_noise_sigma*.npz")
TWIN = ("/lustre/fsn1/projects/rech/prk/ulx34io/cnn_sbi/scripts/sbi/results/exploratory/"
        "flatsky_cross_2026_06/flatsky_cross_noise_sigma_bnt.npz")
assert NEW, "no sigma table produced"
new = [p for p in NEW if "_bnt" not in p and "_whiten" not in p]
assert new, f"no NO-BNT table among {[os.path.basename(p) for p in NEW]}"
a = np.load(new[0], allow_pickle=True); b = np.load(TWIN, allow_pickle=True)
print(f"  new : {os.path.basename(new[0])}")
# metadata must match the twin exactly (BNT only mixes channels; it cannot change these)
same = True
for k in ("n_scales","n_real","nside","lmax","sigma_e","galaxy_density",
          "reso_arcmin","field_npix","seed_base"):
    va, vb = a[k], b[k]
    ok = np.allclose(np.asarray(va, float), np.asarray(vb, float))
    same &= ok
    print(f"  {k:16s} new={va}  twin={vb}   {'OK' if ok else 'MISMATCH'}")
print(f"  channel_names match: {list(a['channel_names'])==list(b['channel_names'])}")
print(f"  sigma shape: new={a['sigma'].shape} twin={b['sigma'].shape}")
print(f"  bnt flag   : new={a['bnt']} (expect False)  twin={b['bnt']} (expect True)")
s = np.asarray(a["sigma"], float)
print(f"  sigma finite/positive: {np.isfinite(s).all()} / {(s>0).all()}")
assert same, "metadata disagrees with the intact twin"
assert np.isfinite(s).all() and (s > 0).all(), "bad sigma values"
assert not bool(a["bnt"]), "new table is flagged bnt=True"
print("\n  NO-BNT SIGMA TABLE REGENERATED AND CONSISTENT")
PY
