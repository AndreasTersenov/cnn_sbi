#!/bin/bash
#SBATCH --job-name=sweep_joint_ens
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/sweep_joint_ens_%j.out
# joint l1 no-BNT is an ENSEMBLE row (paper 3371): pool the 3 VMIM compressor seeds,
# each NDE-pooled over 3 flow seeds = 9 flows. The per-member moments this writes
# (arm_mean/arm_cov) are exactly what the leave-one-out jackknife bar needs.
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} CNN_SBI_SRC=$SRC TMPDIR=$S/recovery/tmp
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /lustre/fswork/projects/rech/prk/ulx34io/recovery
srun $N/envs/jaxili/bin/python -u population_sweep.py \
  --arm-dir $S/recovery/vmim/both_nobnt_s41 \
  --arm-dir $S/recovery/vmim/both_nobnt_s42 \
  --arm-dir $S/recovery/vmim/both_nobnt_s43 \
  --arm-label joint_l1_nobnt_ensemble \
  --out $S/recovery/sweeps/joint_l1_nobnt_ensemble \
  --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000
