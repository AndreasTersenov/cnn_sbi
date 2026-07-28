#!/bin/bash
#SBATCH --job-name=vmim_none_bnt
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/vmim_none_bnt_%j.out
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} TMPDIR=$S/recovery/tmp
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$SRC"
for SEED in 41 42 43; do
  echo "######## VMIM none BNT seed $SEED ########"
  $N/envs/jaxili/bin/python -u vmim_from_cache.py \
    --cache-dir $S/recovery/l1_datavector_cache_bnt/flat_local_none_bnt \
    --fid-npz $S/recovery/fid_raw_none_bnt.npz \
    --out-cache $S/recovery/vmim/none_bnt_s${SEED} \
    --out-fid $S/recovery/vmim/none_bnt_s${SEED}/fiducial_summaries.npz \
    --summary-dim 10 --hidden 256,256 --nf-layers 4 --nf-hidden 128 \
    --steps 30000 --seed $SEED \
    --preproc-transform log1p-zscore --clip-value 5 --min-feature-variance 1e-5 \
    --max-minutes 35 --cuda-visible-devices 0 || echo "SEED $SEED FAILED"
done
