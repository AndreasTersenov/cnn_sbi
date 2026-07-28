#!/bin/bash
#SBATCH --job-name=find_centers
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/find_centers_%j.out

set -uo pipefail
module purge
module load python/3.11.5
export PYTHONPATH=/lustre/fswork/projects/rech/prk/ulx34io/.local/lib/python3.11/site-packages:${PYTHONPATH:-}
export OMP_NUM_THREADS=4

cd /lustre/fswork/projects/rech/prk/ulx34io/recovery
python3 -u find_centers.py
