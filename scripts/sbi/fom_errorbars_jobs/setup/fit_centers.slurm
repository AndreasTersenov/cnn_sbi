#!/bin/bash
#SBATCH --job-name=fit_centers
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/fit_centers_%j.out
set -uo pipefail
N=/lustre/fswork/projects/rech/nzu/ulx34io
S=/lustre/fsn1/projects/rech/nzu/ulx34io
export TMPDIR=$S/recovery/tmp; mkdir -p $TMPDIR
export TFDS_DATA_DIR=/lustre/fsn1/projects/rech/prk/ulx34io/tensorflow_datasets
export TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=10 N_GROUPS=12
cd /lustre/fswork/projects/rech/prk/ulx34io/recovery
$N/envs/jaxili/bin/python -u fit_centers.py
