#!/bin/bash
#SBATCH --job-name=joint_build
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/joint_build_%j.out
#
# Usage: sbatch build_joint_arm.slurm <nobnt|bnt>
#
# Build the RAW joint-ell1 datavector for one basis, using the ACTUAL
# flatsky_joint_stats.py recovered from GitHub on 2026-07-28 (branch
# analytical-nde-match-2026-06, sha256 bd17b539..., 11710 B = exactly the size of the
# destroyed local file). This is the real module, not a reimplementation.
#
# Config is NOT guessed -- it is the surviving jointl1_bnt_raw/cache/l1_cache_meta.npz:
#     stat=jointl1  k=10  adaptive_ranges=True  dequantize=False  snr_range=5.0  append_to=None
# NOT --rotated-binning: that was a later shear-aware experiment that FAILED its BNT gate
# (3221, SBC 0.338) and is not the published lineage.
#
# Loader params are baked into the driver and match every campaign arm:
#   train = split 'train' perms 5-6 flip seed 1001 | val = split 'test' perms 0-1 noflip 2001
#   fiducial = all 200 perm files of the nobnt obs cache (BNT is applied on-device via basis)
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5

BASIS=$1
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
DEAD=/lustre/fsn1/projects/rech/prk/ulx34io/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06

export PYTHONPATH=$SRC:${PYTHONPATH:-} TMPDIR=$S/recovery/tmp
export TFDS_DATA_DIR=/lustre/fsn1/projects/rech/prk/ulx34io/tensorflow_datasets
export JOINT_TFDS_DIR=$TFDS_DATA_DIR
export JOINT_FID_OBS=$S/recovery/full_sphere_cache_fiducial_10deg/nobnt/obs
export JOINT_SIGMA_NOBNT=$S/recovery/sigma_tables/flatsky_cross_noise_sigma.npz
export JOINT_SIGMA_BNT=$DEAD/flatsky_cross_noise_sigma_bnt.npz
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false
export CG_PARENT=/lustre/fsmisc/dataset
# This driver computes in TORCH but loads through tf.data. Unlike npe_cnn_nbody_tomo.py it
# never calls tf.config.set_memory_growth, so TF grabbed all 79 GB of the A100 and torch
# OOM'd with only 900 MB of its own allocations (job 330761). TF is a pure data loader here.
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=$S/recovery/joint/jointl1_${BASIS}_raw
mkdir -p "$OUT" "$TMPDIR"
echo "=== joint l1 raw build: basis=$BASIS -> $OUT ==="
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null

cd "$SRC"
srun $N/envs/jaxili/bin/python -u build_flatsky_joint_arm.py \
    --stat jointl1 --basis "$BASIS" --k 10 --adaptive-ranges \
    --out-cache "$OUT/cache" --out-fid "$OUT/fid.npz"
rc=$?
echo "driver exit=$rc"
[ $rc -eq 0 ] || exit $rc

$N/envs/jaxili/bin/python - <<PY
import numpy as np
o="$OUT"
tr=np.load(o+"/cache/l1_train.npz"); va=np.load(o+"/cache/l1_val.npz"); fd=np.load(o+"/fid.npz")
print("  train x",tr["x"].shape,"theta",tr["theta"].shape)
print("  val   x",va["x"].shape,"theta",va["theta"].shape)
print("  fid   S",fd["S"].shape,"perm",fd["perm"].shape,"patch",fd["patch"].shape)
# k=10 -> 6 pairs x 5 scales x 10x10 cells = 3000
exp=6*5*10*10
assert tr["x"].shape[1]==exp, f'train dim {tr["x"].shape[1]} != expected {exp}'
assert fd["S"].shape[1]==exp, f'fid dim {fd["S"].shape[1]} != expected {exp}'
assert np.isfinite(tr["x"]).all() and np.isfinite(fd["S"]).all(), "non-finite features"
assert (tr["x"]>=0).all(), "jointl1 cells must be non-negative (they are sums of |u|)"
print(f"  feature dim {exp} OK, finite, non-negative")
print(f"  perms {fd['perm'].min()}..{fd['perm'].max()} x patches 0..{fd['patch'].max()}")
PY
echo "BUILD+CHECK OK for $BASIS"
