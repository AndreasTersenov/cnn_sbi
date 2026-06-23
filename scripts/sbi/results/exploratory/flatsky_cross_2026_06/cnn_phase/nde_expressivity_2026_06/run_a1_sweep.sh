#!/usr/bin/env bash
# Route A1: deeper/wider sbi_lens RealNVP on the FROZEN resnet18 s41 summaries (no code change).
# Per config: screen FoM3 (n_obs=1000) + gate dumps (tarp_stratified, n_points=600, 3 NDE seeds pooled).
# Two parallel streams: GPU 0 (depth sweep @128) and GPU 2 (wide @256). Baseline = 4x128 (FoM3 3326).
set -u -o pipefail
SBI=/mnt/home/tersenov/software/cnn_sbi/scripts/sbi
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
ROOT=$SBI/results/exploratory/flatsky_cross_2026_06/cnn_phase
CACHE=$ROOT/arch_sweep_2026_06_13/cnn_resnet18_s41/cache
FID=$ROOT/arch_sweep_2026_06_13/fidsumm_resnet18.npz
OUT=$ROOT/nde_expressivity_2026_06
COM="--nde-family sbilens_realnvp --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 --seeds 41,42,43 --m-samples 2000"

do_config(){
  local L=$1 H=$2 gpu=$3
  local d="$OUT/L${L}_H${H}" arm="realnvp_L${L}_H${H}"
  mkdir -p "$d/screen" "$d/gate"
  # 1) screen FoM3 on 1000 fiducial obs
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 PYTHONUNBUFFERED=1 \
  "$PY" -u "$SBI/train_nde_from_compressed.py" --train-cache-dir "$CACHE" --cache-prefix cnn \
    --arm-label "$arm" --fiducial-summaries-npz "$FID" --output-dir "$d/screen" \
    --nde-layers "$L" --nde-hidden "$H" $COM --n-obs 1000 --cuda-visible-devices "$gpu" \
    > "$d/screen.log" 2>&1
  # 2) gate dumps (varied-theta val, 600 points)
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 PYTHONUNBUFFERED=1 \
  "$PY" -u "$SBI/tarp_stratified_val_nde.py" --train-cache-dir "$CACHE" --cache-prefix cnn \
    --arm-label "$arm" --dumps-root "$d/gate/dumps" \
    --nde-layers "$L" --nde-hidden "$H" $COM --n-points 600 --cuda-visible-devices "$gpu" \
    > "$d/gate.log" 2>&1
  local f; f=$(grep -o '"fom3":[^,}]*' "$d/screen/median_summary.json" 2>/dev/null | grep -oE '[0-9.]+' | head -1)
  echo "DONE L${L}_H${H} (gpu$gpu): screen FoM3=${f:-FAIL}"
}

# stream A: depth @ width 128 (GPU 0)
( do_config 8 128 0; do_config 12 128 0; do_config 16 128 0 ) &
# stream B: wider (GPU 2)
( do_config 8 256 2; do_config 12 256 2 ) &
wait
echo "=== A1 SWEEP COMPLETE ==="
ls -1 "$OUT"