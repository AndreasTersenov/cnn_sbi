#!/bin/bash
#SBATCH --job-name=cnn_timing
#SBATCH --account=nzu@a100
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/cnn_timing_%j.out

# TIMING RUN ONLY -- 2000 of the 80000 compressor steps, to measure the step rate so the
# real job can be sized from data instead of padded. (Padding a 20h request last time put
# us behind ~2380 queued jobs for no reason.)
#
# Recipe from cnn_phase/arch_sweep_2026_06_13/SUMMARY_ARCH.md, which records the exact
# setup that produced the paper's CNN number:
#   "auto-only flat-local, seed 41, dim 10, 80k steps, batch 128, lr 5e-4, best_val,
#    only --compressor-arch varies"   ->  resnet18 = FoM3 3326
# The readout is the same 3-pooled sbi_lens RealNVP 4x128 already working for the L1 rows.
#
# No frozen sigma / SNR ranges here: the CNN reads the auto MAPS directly (no L1 statistic).
# Known risk recorded in that handoff: "resnet18 (BatchNorm) may collapse (BN-on-cosmology-
# mixed-batches) -- if its FoM3 craters, that's expected, not a bug."

set -uo pipefail
module purge
module load arch/a100
module load python/3.11.5

N=/lustre/fswork/projects/rech/nzu/ulx34io
S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
TFDS=/lustre/fsn1/projects/rech/prk/ulx34io/tensorflow_datasets
OUT=$S/recovery/cnn_timing

export PYTHONPATH=$SRC:${PYTHONPATH:-} TMPDIR=$S/recovery/tmp
export TFDS_DATA_DIR=$TFDS CG_PARENT=/lustre/fsmisc/dataset
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$OUT"

cd "$SRC"
srun $N/envs/jaxili/bin/python -u npe_cnn_nbody_tomo.py \
    --train-compressor \
    --cnn-map-route flat_local --cross-op none \
    --compressor-arch resnet18 --compressor-dim 10 \
    --compressor-steps 2000 \
    --compressor-batch-size 128 --compressor-lr 5e-4 \
    --compressor-checkpoint-policy best_val \
    --compressor-save-every 500 \
    --cross-tfds-name nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180 \
    --cross-tfds-data-dir "$TFDS" \
    --fiducial-obs-cache $S/recovery/full_sphere_cache_fiducial_10deg \
    --cosmogrid-meta /lustre/fsmisc/dataset/CosmoGridV1/CosmoGridV1_metainfo.h5 \
    --field-size 10 --field-npix 80 \
    --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --map-kind nbody --zero-mean-maps \
    --seed 41 \
    --cache-dir "$OUT/cache" \
    --save-dir "$OUT" \
    --total-steps 2 \
    --exit-after-compress \
    --cuda-visible-devices 0 \
    --no-wandb
echo "driver exit=$?"
