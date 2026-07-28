#!/bin/bash
#SBATCH --job-name=sw_joint
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/sw_joint_%j.out
#
# Usage: sbatch sw_joint_chain.slurm <nobnt|bnt>
#
# All four sweeps for one basis in ONE job (3 singles + the 3-pooled ensemble),
# run sequentially. Chained rather than fanned out because qos_gpu_a100-dev caps
# submitted jobs at 10 per user and each sweep is only ~5-13 min.
#
# Preproc = none/0/1e-12: these are VMIM-COMPRESSED arms, and the compressed-arm
# path is train_nde_from_compressed.py whose defaults are exactly that (the same
# settings run_jointl1_seedcheck.py relied on for the published 3754/3761/4034).
# The log1p-zscore/5/1e-5 defaults in population_sweep_flatsky.py apply to RAW
# datavectors, not to 10-D VMIM summaries.
#
# Validation targets (surviving docs):
#   nobnt singles 3754 / 3761 / 4034   ensemble 3371 (published)
#   bnt   singles 3232 / 3235 / 2708   ensemble 2424.3 (surviving per-obs data)
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} CNN_SBI_SRC=$SRC TMPDIR=$S/recovery/tmp
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true

B=$1
J=$S/recovery/joint
cd /lustre/fswork/projects/rech/prk/ulx34io/recovery

for SEED in 41 42 43; do
  OUT=$S/recovery/sweeps/jointl1_${B}_s${SEED}
  [ -f "$OUT/median_summary.json" ] && { echo "### s$SEED done, skip"; continue; }
  echo "######## sweep jointl1 $B s$SEED ########"
  $N/envs/jaxili/bin/python -u population_sweep.py \
     --arm-dir "$J/jointl1_${B}_s${SEED}" \
     --arm-label "jointl1_${B}_s${SEED}" --out "$OUT" \
     --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000 \
     --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 \
     || echo "s$SEED FAILED"
done

OUT=$S/recovery/sweeps/jointl1_${B}_ensemble
if [ ! -f "$OUT/median_summary.json" ]; then
  echo "######## sweep jointl1 $B ENSEMBLE (3 pooled) ########"
  $N/envs/jaxili/bin/python -u population_sweep.py \
     --arm-dir "$J/jointl1_${B}_s41" --arm-dir "$J/jointl1_${B}_s42" --arm-dir "$J/jointl1_${B}_s43" \
     --arm-label "jointl1_${B}_ensemble" --out "$OUT" \
     --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000 \
     --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 \
     || echo "ENSEMBLE FAILED"
fi

echo; echo "=== jointl1 $B summary ==="
for t in s41 s42 s43 ensemble; do
  f=$S/recovery/sweeps/jointl1_${B}_${t}/median_summary.json
  [ -f "$f" ] && printf "  %-10s %s\n" "$t" "$($N/envs/jaxili/bin/python -c "import json;print(round(json.load(open('$f'))['fom3'],1))")" \
              || printf "  %-10s MISSING\n" "$t"
done
