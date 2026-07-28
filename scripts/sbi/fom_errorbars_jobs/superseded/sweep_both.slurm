#!/bin/bash
#SBATCH --job-name=sweep_joint_nobnt
#SBATCH --account=nzu@a100
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/sweep_joint_%j.out
# Population sweep for joint l1 no-BNT (op=both, 16ch). Paper value: 3371.
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} CNN_SBI_SRC=$SRC
export TMPDIR=$S/recovery/tmp TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /lustre/fswork/projects/rech/prk/ulx34io/recovery
srun $N/envs/jaxili/bin/python -u population_sweep.py \
  --arm-dir $S/recovery/arms/l1both_nobnt_s41 \
  --fiducial-datavectors $S/recovery/fiducial_both_datavectors_nobnt.npz \
  --op both --arm-label joint_l1_nobnt \
  --out $S/recovery/sweeps/joint_l1_nobnt \
  --n-obs 9000 --max-perm 50 --m-samples 2000
