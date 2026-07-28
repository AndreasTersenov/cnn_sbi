#!/bin/bash
#SBATCH --job-name=extra_seeds
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/extra_seeds_%j.out
#
# Usage: sbatch extra_seeds.slurm <jointl1_bnt|auto_bnt>
#
# WHY: the two double-digit rows (joint l1 BNT 8.6%, l1 auto BNT 11.1%) each have their
# spread driven by ONE deviant compressor seed. That seed is NOT an unlucky draw --
# the per-seed deviation pattern reproduces across two independent campaigns
# (joint l1 no-BNT r=+0.98, BNT r=+0.94 vs the published singles), so re-running the
# same seeds is uninformative. What is unknown is whether such a draw is RARE or TYPICAL,
# i.e. the shape of the seed population. Only more DISTINCT seeds answer that.
# n=3 -> n=6 drops the relative uncertainty on sigma from ~46% to ~30%.
#
# Same recipe as the original three seeds (vmim_from_cache defaults: hidden 256,256,
# nf 4x128, 30000 steps, log1p-zscore/5/1e-5), then the compressed-arm sweep preproc
# (none/0/1e-12). Nothing varies except --seed.
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
export PYTHONPATH=$SRC:${PYTHONPATH:-} CNN_SBI_SRC=$SRC TMPDIR=$S/recovery/tmp
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true

ARM=$1
case "$ARM" in
  jointl1_bnt) RAWC=$S/recovery/joint/jointl1_bnt_raw/cache
               RAWF=$S/recovery/joint/jointl1_bnt_raw/fid.npz
               ADIR=$S/recovery/joint/jointl1_bnt ;;
  auto_bnt)    RAWC=$S/recovery/l1_datavector_cache_bnt/flat_local_none_bnt
               RAWF=$S/recovery/fid_raw_none_bnt.npz
               ADIR=$S/recovery/vmim/none_bnt ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
[ -s "$RAWC/l1_train.npz" ] || { echo "missing $RAWC/l1_train.npz"; exit 1; }
[ -s "$RAWF" ] || { echo "missing $RAWF"; exit 1; }

cd "$SRC"
for SEED in 44 45 46; do
  OUT=${ADIR}_s${SEED}
  if [ ! -s "$OUT/fiducial_summaries.npz" ]; then
    echo "######## VMIM $ARM seed $SEED ########"
    $N/envs/jaxili/bin/python -u vmim_from_cache.py \
      --cache-dir "$RAWC" --fid-npz "$RAWF" \
      --out-cache "$OUT" --out-fid "$OUT/fiducial_summaries.npz" \
      --summary-dim 10 --hidden 256,256 --nf-layers 4 --nf-hidden 128 \
      --steps 30000 --seed $SEED \
      --preproc-transform log1p-zscore --clip-value 5 --min-feature-variance 1e-5 \
      --max-minutes 35 --cuda-visible-devices 0 || { echo "VMIM s$SEED FAILED"; continue; }
  fi
done

cd /lustre/fswork/projects/rech/prk/ulx34io/recovery
LBL=$([ "$ARM" = "jointl1_bnt" ] && echo jointl1_bnt || echo auto_bnt)
for SEED in 44 45 46; do
  OUT=$S/recovery/sweeps/${LBL}_s${SEED}
  [ -f "$OUT/median_summary.json" ] && { echo "### sweep s$SEED done"; continue; }
  [ -s "${ADIR}_s${SEED}/fiducial_summaries.npz" ] || { echo "### s$SEED has no arm, skip"; continue; }
  echo "######## sweep $LBL s$SEED ########"
  $N/envs/jaxili/bin/python -u population_sweep.py \
     --arm-dir "${ADIR}_s${SEED}" --arm-label "${LBL}_s${SEED}" --out "$OUT" \
     --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000 \
     --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 \
     || echo "sweep s$SEED FAILED"
done

echo; echo "=== $LBL : all six seeds ==="
for SEED in 41 42 43 44 45 46; do
  f=$S/recovery/sweeps/${LBL}_s${SEED}/median_summary.json
  [ -f "$f" ] && printf "  s%-3s %s\n" "$SEED" "$($N/envs/jaxili/bin/python -c "import json;print(round(json.load(open('$f'))['fom3'],1))")" \
              || printf "  s%-3s MISSING\n" "$SEED"
done
