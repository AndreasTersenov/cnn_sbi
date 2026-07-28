#!/bin/bash
#SBATCH --job-name=cnn_bnt
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/prk/ulx34io/recovery/logs/cnn_bnt_%j.out
# 3 ResNet-18 VMIM compressors (seeds 41/42/43), bnt. Recipe verbatim from
# cnn_phase/arch_sweep_2026_06_13/SUMMARY_ARCH.md: auto-only flat-local, dim 10,
# 80k steps, batch 128, lr 5e-4, best_val -> resnet18 = FoM3 3326.
# Measured rate: ~70 steps/s => ~19 min train + ~2 min compress per seed.
set -uo pipefail
module purge; module load arch/a100; module load python/3.11.5
N=/lustre/fswork/projects/rech/nzu/ulx34io; S=/lustre/fsn1/projects/rech/nzu/ulx34io
SRC=/lustre/fswork/projects/rech/prk/ulx34io/recovery/rescued_scripts
TFDS=/lustre/fsn1/projects/rech/prk/ulx34io/tensorflow_datasets
export PYTHONPATH=$SRC:${PYTHONPATH:-} TMPDIR=$S/recovery/tmp TFDS_DATA_DIR=$TFDS
export CG_PARENT=/lustre/fsmisc/dataset TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$SRC"
for SEED in 41 42 43; do
  OUT=$S/recovery/cnn/bnt_s${SEED}
  echo "######## CNN resnet18 bnt seed $SEED ########"
  $N/envs/jaxili/bin/python -u npe_cnn_nbody_tomo.py \
    --train-compressor --exit-after-compress \
    --cnn-map-route flat_local --cross-op none --flatsky-bnt \
    --compressor-arch resnet18 --compressor-dim 10 \
    --compressor-steps 80000 --compressor-batch-size 128 --compressor-lr 5e-4 \
    --compressor-checkpoint-policy best_val --compressor-save-every 5000 \
    --cross-tfds-name nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180 \
    --cross-tfds-data-dir "$TFDS" \
    --fiducial-obs-cache $S/recovery/full_sphere_cache_fiducial_10deg \
    --cosmogrid-meta /lustre/fsmisc/dataset/CosmoGridV1/CosmoGridV1_metainfo.h5 \
    --field-size 10 --field-npix 80 --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --map-kind nbody --zero-mean-maps --seed $SEED \
    --cache-dir "$OUT/cache" --save-dir "$OUT" --total-steps 2 \
    --cuda-visible-devices 0 --no-wandb || echo "SEED $SEED FAILED"
done
