#!/bin/bash
#SBATCH --job-name=sw_joint_bnt_s41
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/sw_joint_bnt_s41_%j.out
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} CNN_SBI_SRC=$SRC TMPDIR=$S/recovery/tmp
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /lustre/fswork/projects/rech/prk/ulx34io/recovery
srun $N/envs/jaxili/bin/python -u population_sweep.py --arm-dir /lustre/fsn1/projects/rech/nzu/ulx34io/recovery/vmim/both_bnt_s41 \
  --arm-label joint_bnt_s41 --out $S/recovery/sweeps/joint_bnt_s41 \
  --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000
