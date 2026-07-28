#!/bin/bash
#SBATCH --job-name=sw_cnn
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/sw_cnn_%j.out
#
# Usage: sbatch sw_cnn.slurm <label> <preproc> <clip> <arm> [<arm> <arm>]
#   one <arm> = single-quoted row; three = ensemble row (9 pooled flows).
#
# PREPROC — this is the one place the CNN sweep differs from the L1 sweeps:
#   the L1 driver has no --standardize-summary at all, so its compressed summaries went
#   into the flow raw ("none"); the CNN driver standardizes by default
#   (--standardize-summary, --summary-clip-value 5.0) and that is what produced the
#   paper's 3326. So CNN rows use "zscore 5.0". population_sweep's preprocess_summaries
#   fits mean/std on the arm's train x and clips — identical to the driver's
#   fit_and_apply_summary_standardization.
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} CNN_SBI_SRC=$SRC TMPDIR=$S/recovery/tmp
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false

LABEL=$1; PREPROC=$2; CLIP=$3; shift 3
ARGS=(); for A in "$@"; do ARGS+=(--arm-dir "$S/recovery/cnn/$A"); done
echo "=== sweep $LABEL | preproc=$PREPROC clip=$CLIP | arms: $* ==="

cd /lustre/fswork/projects/rech/prk/ulx34io/recovery
srun $N/envs/jaxili/bin/python -u population_sweep.py "${ARGS[@]}" \
  --arm-label "$LABEL" --out "$S/recovery/sweeps/$LABEL" \
  --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000 \
  --preproc-transform "$PREPROC" --clip-value "$CLIP" --min-feature-variance 1e-12
