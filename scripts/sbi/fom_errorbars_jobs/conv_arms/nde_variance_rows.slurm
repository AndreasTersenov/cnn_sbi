#!/bin/bash
#SBATCH --job-name=nde_rows
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/nde_rows_%j.out
#
# Usage: sbatch nde_variance_rows.slurm <seedset> <label>:<armdir> [<label>:<armdir> ...]
#   e.g. sbatch nde_variance_rows.slurm 51,52,53 auto_nobnt:vmim/none_nobnt_s41
#
# Measures the NDE(flow)-seed contribution for a row by re-sweeping the SAME trained
# compressor with a different set of pooled flow seeds. Everything else identical, so the
# paired difference against the s41 baseline isolates the NDE term.
#
# Motivation: the Table-1 +/- varies only the COMPRESSOR seed. For the CNN that term is
# 0.41%/0.59% while swapping the flow-seed set moved the answer 1.19%/2.36% -- i.e. the
# quoted bar was measuring the smaller contribution. NOTE_FOM_ERROR_BARS.md sec 2 bounds
# the NDE term at "<1.2%"; CNN BNT measured 2.36%, so the bound needs checking per row.
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} CNN_SBI_SRC=$SRC TMPDIR=$S/recovery/tmp
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true

SEEDSET=$1; shift
TAG=nde$(echo "$SEEDSET" | tr -d ',')
cd /lustre/fswork/projects/rech/prk/ulx34io/recovery
for spec in "$@"; do
  LBL=${spec%%:*}; ARM=${spec#*:}
  OUT=$S/recovery/sweeps/${LBL}_s41_${TAG}
  [ -f "$OUT/median_summary.json" ] && { echo "### $LBL done"; continue; }
  echo "######## $LBL  arm=$ARM  flow seeds $SEEDSET ########"
  $N/envs/jaxili/bin/python -u population_sweep.py \
     --arm-dir "$S/recovery/$ARM" --arm-label "${LBL}_s41_${TAG}" --out "$OUT" \
     --seeds "$SEEDSET" --n-obs 9000 --max-perm 50 --m-samples 2000 \
     --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 \
     || echo "$LBL FAILED"
done
echo; echo "=== baseline (flow 41,42,43) vs $SEEDSET ==="
for spec in "$@"; do
  LBL=${spec%%:*}
  a=$S/recovery/sweeps/${LBL}_s41/median_summary.json
  b=$S/recovery/sweeps/${LBL}_s41_${TAG}/median_summary.json
  [ -f "$a" ] && [ -f "$b" ] && $N/envs/jaxili/bin/python -c "
import json
x=json.load(open('$a'))['fom3']; y=json.load(open('$b'))['fom3']
print(f'  {\"$LBL\":22s} {x:8.1f} -> {y:8.1f}   delta {y-x:+8.1f}  ({100*(y-x)/x:+.2f}%)')"
done
