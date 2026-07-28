#!/bin/bash
#SBATCH --job-name=confirm_centers
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/confirm_centers_%j.out

set -uo pipefail
module purge
module load python/3.11.5
export PYTHONPATH=/lustre/fswork/projects/rech/prk/ulx34io/.local/lib/python3.11/site-packages:${PYTHONPATH:-}
export TFDS_DATA_DIR=/lustre/fsn1/projects/rech/prk/ulx34io/tensorflow_datasets
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=8

cd /lustre/fswork/projects/rech/prk/ulx34io/recovery
python3 -u confirm_centers.py
