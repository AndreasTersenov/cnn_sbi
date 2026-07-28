#!/bin/bash
#SBATCH --job-name=precompute_obs
#SBATCH --account=nzu@a100
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/precompute_obs_%j.out
# Observation datavectors for ALL L1 rows in one pass over the small fiducial cache
# (explicitly NOT the 290GB TFDS). 16ch 'both' output is column-sliced per arm:
# none=4ch/800-D, product=10ch/2000-D, both=16ch/3200-D.
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-}
export TMPDIR=$S/recovery/tmp
export FLATSKY_FID_CACHE=$S/recovery/full_sphere_cache_fiducial_10deg/nobnt/obs
export TF_CPP_MIN_LOG_LEVEL=3
cd "$SRC"
srun $N/envs/jaxili/bin/python -u precompute_fiducial_both_datavectors.py \
  --both-cache $S/recovery/l1_datavector_cache/flat_local_both \
  --sigma $S/recovery/sigma_tables/flatsky_cross_noise_sigma.npz \
  --perms 50 \
  --out $S/recovery/fiducial_both_datavectors_nobnt.npz
