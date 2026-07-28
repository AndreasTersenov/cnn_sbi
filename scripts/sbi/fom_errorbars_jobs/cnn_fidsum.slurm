#!/bin/bash
#SBATCH --job-name=cnn_fidsum
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/cnn_fidsum_%j.out
#
# Compress the 9000 fiducial obs through each finished CNN arm.
# Arms are passed as $1.. (arm dir names under $S/recovery/cnn); with no args it
# processes every arm whose cache/cnn_obs.npz exists, skipping ones already built.
#
# Each arm runs its own G1 gate: recompute the driver's persisted observed vector
# (cache/cnn_obs.npz, perm 0 / patch 0) and require max|delta| <= 9e-4. A failure is
# fatal for that arm -- the summaries would be wrong and the FoM3 meaningless.
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5

N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
W=/lustre/fswork/projects/rech/prk/ulx34io/recovery
export PYTHONPATH=$SRC:${PYTHONPATH:-} TMPDIR=$S/recovery/tmp
export TFDS_DATA_DIR=/lustre/fsn1/projects/rech/prk/ulx34io/tensorflow_datasets
export CG_PARENT=/lustre/fsmisc/dataset TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=()
  for d in "$S"/recovery/cnn/*/; do
    [ -f "$d/cache/cnn_obs.npz" ] && ARMS+=("$(basename "$d")")
  done
fi
echo "arms: ${ARMS[*]:-<none>}"

fail=0
for A in "${ARMS[@]}"; do
  ARM=$S/recovery/cnn/$A
  if [ -f "$ARM/fiducial_summaries.npz" ]; then
    echo "### $A: already built, skipping"; continue
  fi
  echo; echo "############################## $A ##############################"
  $N/envs/jaxili/bin/python -u $W/build_fiducial_summaries_cnn.py \
      --arm-dir "$ARM" \
      --obs-cache $S/recovery/full_sphere_cache_fiducial_10deg \
      --max-perm 50 --g1-tol 9e-4 --cuda-visible-devices 0
  rc=$?
  [ $rc -eq 0 ] || { echo "### $A FAILED (rc=$rc)"; fail=1; }
done
echo; echo "=== summary ==="
for A in "${ARMS[@]}"; do
  f=$S/recovery/cnn/$A/fiducial_summaries.npz
  [ -f "$f" ] && echo "  OK   $A  ($(stat -c%s "$f") B)" || echo "  MISS $A"
done
exit $fail
