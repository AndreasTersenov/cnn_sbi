#!/usr/bin/env bash
# BNT test for the BEST pipeline (resnet18 compressor + sbi_lens RealNVP NDE).
# Recipe = the exact no-BNT arch-sweep resnet18 command + --flatsky-bnt (only difference = BNT),
# so the BNT/no-BNT ratio is clean. Per seed: train BNT compressor -> BNT fiducial summaries
# (G1-checked) -> RealNVP readout (3-seed/9000) -> BNT FoM3. Compare vs no-BNT 3326/3314/3273.
set -u -o pipefail
SBI=/mnt/home/tersenov/software/cnn_sbi/scripts/sbi
OUT=$SBI/results/exploratory/flatsky_cross_2026_06/cnn_phase/bnt_resnet18_2026_06_14
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
TFDS="nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR=/home/tersenov/tensorflow_datasets
FID=$SBI/results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg
mkdir -p "$OUT"
declare -A NOBNT=([41]=3326 [42]=3314 [43]=3273)

do_bnt(){
  local s=$1 gpu=$2 d="$OUT/cnn_resnet18_bnt_s$1"; mkdir -p "$d"
  # 1) BNT compressor (= arch-sweep resnet18 + --flatsky-bnt)
  if [ ! -f "$d/cache/cnn_train.npz" ]; then
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 PYTHONUNBUFFERED=1 \
    "$PY" -u "$SBI/npe_cnn_nbody_tomo.py" --train-compressor --exit-after-compress \
      --cnn-map-route flat_local --cross-op none --flatsky-bnt \
      --cross-tfds-name "$TFDS" --cross-tfds-data-dir "$DDIR" --fiducial-obs-cache "$FID" \
      --harmonic-cache-regime nobnt --harmonic-normalize-input-channels --cnn-perm-split 0-4:5-6 \
      --zero-mean-maps --map-kind nbody --seed "$s" --field-size 10 --field-npix 80 \
      --nbins 4 --tomo-bin-indices 1,2,3,4 --compressor-arch resnet18 --compressor-dim 10 \
      --compressor-dense-width 256 --compressor-conv-channels 64,128,256 --compressor-steps 80000 \
      --compressor-batch-size 128 --compressor-lr 0.0005 --compressor-checkpoint-policy best_val --no-wandb \
      --harmonic-obs-perm 0 --harmonic-obs-patch-idx 90 --cuda-visible-devices "$gpu" \
      --save-dir "$d" --cache-dir "$d/cache" --posterior-out "$d/posterior.npy" --figure-out "$d/corner.pdf" \
      > "$d/train.log" 2>&1
  fi
  [ -f "$d/cache/cnn_train.npz" ] || { echo "s$s TRAIN FAIL"; return 1; }
  # 2) BNT fiducial summaries (arch-aware + --flatsky-bnt, G1-checked)
  local fid="$OUT/fidsumm_bnt_resnet18_s$s.npz"
  if [ ! -f "$fid" ]; then
    read -r PP SP PSHA SSHA NCH < <("$PY" -c "
import numpy as np;m=np.load('$d/cache/cnn_cache_meta.npz',allow_pickle=True)
print(m['compressor_params_path'],m['compressor_state_path'],m['compressor_params_sha256'],m['compressor_state_sha256'],int(m['cnn_input_channels']))")
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 PYTHONUNBUFFERED=1 \
    "$PY" -u "$SBI/build_fiducial_summaries_cnn.py" --arm-label bnt_resnet18_s$s \
      --params-pkl "$PP" --state-pkl "$SP" --expect-params-sha "$PSHA" --expect-state-sha "$SSHA" \
      --n-channels "$NCH" --dim 10 --conv-channels 64,128,256 --dense-width 256 --pool-window 16 --pool-stride 8 \
      --compressor-arch resnet18 --flatsky-bnt --cross-op none --nbins 4 --flatsky-roll-frac 0.10 \
      --cross-tfds-name "$TFDS" --cross-tfds-data-dir "$DDIR" --channel-rms-nsample 8000 \
      --fid-cache-dir "$FID" --regime nobnt --cosmo-id cosmo_fiducial --perms 0-49 \
      --g1-obs-npz "$d/cache/cnn_obs.npz" --g1-perm 0 --g1-patch 90 --out "$fid" --cuda-visible-devices "$gpu" \
      > "$OUT/fidsumm_s$s.log" 2>&1
  fi
  grep -aq "PASS" "$OUT/fidsumm_s$s.log" || { echo "s$s FIDSUMM G1 FAIL"; return 1; }
  # 3) RealNVP readout
  local o="$d/readout"; mkdir -p "$o"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 PYTHONUNBUFFERED=1 \
  "$PY" -u "$SBI/train_nde_from_compressed.py" --train-cache-dir "$d/cache" --cache-prefix cnn \
    --fiducial-summaries-npz "$fid" --arm-label bnt_resnet18_s$s --output-dir "$o" \
    --nde-family sbilens_realnvp --nde-layers 4 --nde-hidden 128 \
    --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 \
    --seeds 41,42,43 --n-obs 9000 --m-samples 2000 --cuda-visible-devices "$gpu" > "$o/run.log" 2>&1
  local bnt; bnt=$(grep -o '"fom3":[^,}]*' "$o/median_summary.json" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
  echo "s$s: BNT ${bnt:-FAIL} / noBNT ${NOBNT[$s]}"
}

do_bnt 41 0 & do_bnt 42 2 & wait
do_bnt 43 0
echo "=== BNT resnet18 (RealNVP) — FINAL ==="
for s in 41 42 43; do
  b=$(grep -o '"fom3":[^,}]*' "$OUT/cnn_resnet18_bnt_s$s/readout/median_summary.json" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
  echo "s$s: BNT=${b:-FAIL}  noBNT=${NOBNT[$s]}"
done
