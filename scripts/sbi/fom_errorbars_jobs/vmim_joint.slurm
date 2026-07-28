#!/bin/bash
#SBATCH --job-name=vmim_joint
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/vmim_joint_%j.out
#
# Usage: sbatch vmim_joint.slurm <nobnt|bnt>
#
# 3 VMIM compressor seeds on the raw joint-l1 datavector (3000-D).
# Flags are exactly the published recipe from the surviving run_jointl1_seedcheck.py:
#     vmim_from_cache.py --cache-dir <RAW>/cache --fid-npz <RAW>/fid.npz \
#                        --out-cache ... --out-fid ... --summary-dim 10 --seed s
# i.e. everything else at vmim_from_cache defaults, which are
#     hidden 256,256 | nf 4x128 | steps 30000 | log1p-zscore | clip 5 | min-var 1e-5
# Stated explicitly here so the recipe is auditable from the job script alone.
#
# Seed 41's arm is named plain "jointl1_<basis>" upstream (not _s41) -- the same naming
# trap as the auto arm's ens_nobnt_auto_s4{2,3}. Using explicit _s41/_s42/_s43 here.
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} TMPDIR=$S/recovery/tmp
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

BASIS=$1
RAW=$S/recovery/joint/jointl1_${BASIS}_raw
[ -f "$RAW/cache/l1_train.npz" ] || { echo "missing $RAW/cache/l1_train.npz"; exit 1; }
[ -f "$RAW/fid.npz" ] || { echo "missing $RAW/fid.npz"; exit 1; }

cd "$SRC"
rc_all=0
for SEED in 41 42 43; do
  OUT=$S/recovery/joint/jointl1_${BASIS}_s${SEED}
  echo "######## VMIM jointl1 ${BASIS} seed $SEED ########"
  $N/envs/jaxili/bin/python -u vmim_from_cache.py \
    --cache-dir "$RAW/cache" --fid-npz "$RAW/fid.npz" \
    --out-cache "$OUT" --out-fid "$OUT/fiducial_summaries.npz" \
    --summary-dim 10 --hidden 256,256 --nf-layers 4 --nf-hidden 128 \
    --steps 30000 --seed $SEED \
    --preproc-transform log1p-zscore --clip-value 5 --min-feature-variance 1e-5 \
    --max-minutes 35 --cuda-visible-devices 0 || { echo "SEED $SEED FAILED"; rc_all=1; }
done
echo; echo "=== summary ($BASIS) ==="
for SEED in 41 42 43; do
  f=$S/recovery/joint/jointl1_${BASIS}_s${SEED}/fiducial_summaries.npz
  [ -s "$f" ] && echo "  OK   s$SEED ($(stat -c%s "$f") B)" || echo "  MISS s$SEED"
done
exit $rc_all
