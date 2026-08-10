#!/bin/bash
#SBATCH --job-name=conv_arm
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/conv_arm_%j.out
#
# Usage: sbatch conv_arm.slurm <nobnt|bnt>
#
# The l1+conv arm (10 ch = 4 autos + 6 conv), end to end: 3 VMIM compressor seeds then
# 3 single sweeps + the 3-pooled ensemble sweep. Raw datavector was column-sliced from the
# existing 16-ch 'both' cache by slice_conv.slurm (auto block verified identical to the
# 'none' arm in both frames), so no TFDS pass was needed.
#
# Recipe identical to every other l1 arm -- only the input cache differs:
#   VMIM  : summary_dim 10, hidden 256,256, nf 4x128, 30000 steps, log1p-zscore/5/1e-5
#   sweep : 3 pooled flow seeds 41,42,43, 9000 mocks, preproc none/0/1e-12
#
# Validation target (no-BNT): operator_realnvp/conv/n9000 = 2671.1, a SINGLE-compressor
# run whose sibling 'product' arm reproduced the committed 3045 to 0.4% (3057.9).
# GATE C on that campaign: PASS-with-caveat. There is NO published BNT counterpart.
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} CNN_SBI_SRC=$SRC TMPDIR=$S/recovery/tmp
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true

B=$1
if [ "$B" = "bnt" ]; then
  RAW=$S/recovery/l1_datavector_cache_bnt/flat_local_conv_bnt; FID=$S/recovery/fid_raw_conv_bnt.npz
else
  RAW=$S/recovery/l1_datavector_cache/flat_local_conv;         FID=$S/recovery/fid_raw_conv.npz
fi
[ -s "$RAW/l1_train.npz" ] || { echo "missing $RAW/l1_train.npz"; exit 1; }
[ -s "$FID" ] || { echo "missing $FID"; exit 1; }

cd "$SRC"
for SEED in 41 42 43; do
  OUT=$S/recovery/vmim/conv_${B}_s${SEED}
  [ -s "$OUT/fiducial_summaries.npz" ] && { echo "### vmim s$SEED done"; continue; }
  echo "######## VMIM conv $B seed $SEED ########"
  $N/envs/jaxili/bin/python -u vmim_from_cache.py \
    --cache-dir "$RAW" --fid-npz "$FID" \
    --out-cache "$OUT" --out-fid "$OUT/fiducial_summaries.npz" \
    --summary-dim 10 --hidden 256,256 --nf-layers 4 --nf-hidden 128 \
    --steps 30000 --seed $SEED \
    --preproc-transform log1p-zscore --clip-value 5 --min-feature-variance 1e-5 \
    --max-minutes 35 --cuda-visible-devices 0 || echo "VMIM s$SEED FAILED"
done

cd /lustre/fswork/projects/rech/prk/ulx34io/recovery
for SEED in 41 42 43; do
  OUT=$S/recovery/sweeps/conv_${B}_s${SEED}
  [ -f "$OUT/median_summary.json" ] && { echo "### sweep s$SEED done"; continue; }
  echo "######## sweep conv $B s$SEED ########"
  $N/envs/jaxili/bin/python -u population_sweep.py \
     --arm-dir "$S/recovery/vmim/conv_${B}_s${SEED}" \
     --arm-label "conv_${B}_s${SEED}" --out "$OUT" \
     --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000 \
     --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 || echo "sweep s$SEED FAILED"
done
OUT=$S/recovery/sweeps/conv_${B}_ensemble
if [ ! -f "$OUT/median_summary.json" ]; then
  echo "######## sweep conv $B ENSEMBLE (3 pooled) ########"
  $N/envs/jaxili/bin/python -u population_sweep.py \
     --arm-dir "$S/recovery/vmim/conv_${B}_s41" --arm-dir "$S/recovery/vmim/conv_${B}_s42" \
     --arm-dir "$S/recovery/vmim/conv_${B}_s43" \
     --arm-label "conv_${B}_ensemble" --out "$OUT" \
     --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000 \
     --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 || echo "ENSEMBLE FAILED"
fi
echo; echo "=== conv $B ==="
for t in s41 s42 s43 ensemble; do
  f=$S/recovery/sweeps/conv_${B}_${t}/median_summary.json
  [ -f "$f" ] && printf "  %-10s %s\n" "$t" "$($N/envs/jaxili/bin/python -c "import json;print(round(json.load(open('$f'))['fom3'],1))")" \
              || printf "  %-10s MISSING\n" "$t"
done
