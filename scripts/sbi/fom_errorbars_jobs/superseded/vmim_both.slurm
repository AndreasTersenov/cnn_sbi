#!/bin/bash
#SBATCH --job-name=vmim_both_nobnt
#SBATCH --account=nzu@a100
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/vmim_both_%j.out
# VMIM MLP compressor (256,256 -> 10D) on the raw 'both' L1 cache, RealNVP companion.
# Defaults of vmim_from_cache.py reproduce the surviving l1_cache_meta.npz exactly:
#   summary_dim 10, hidden 256,256, nf_layers 4, nf_hidden 128, steps 30000, seed 41,
#   preproc log1p-zscore / clip 5 / min-var 1e-5 (downstream then uses none/0/1e-12).
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} TMPDIR=$S/recovery/tmp TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$N/envs/jaxili/bin/python

# adapter: vmim_from_cache expects the fiducial datavectors under key "S"
FID=$S/recovery/fid_raw_both.npz
$PY - <<PY
import numpy as np
d=np.load("$S/recovery/fiducial_both_datavectors_nobnt.npz")
np.savez("$FID", S=d["x"], perm=d["perm"], patch=d["patch"])
print("  adapter: S", d["x"].shape, "perm", d["perm"].shape)
PY

cd "$SRC"
srun $PY -u vmim_from_cache.py \
  --cache-dir $S/recovery/l1_datavector_cache/flat_local_both \
  --fid-npz $FID \
  --out-cache $S/recovery/vmim/both_nobnt_s41 \
  --out-fid $S/recovery/vmim/both_nobnt_s41/fiducial_summaries.npz \
  --summary-dim 10 --hidden 256,256 --nf-layers 4 --nf-hidden 128 \
  --steps 30000 --seed 41 \
  --preproc-transform log1p-zscore --clip-value 5 --min-feature-variance 1e-5 \
  --max-minutes 100 --cuda-visible-devices 0
echo "exit=$?"
ls -la $S/recovery/vmim/both_nobnt_s41 2>/dev/null
