#!/usr/bin/env bash
# Autonomous block (Andreas away ~8h): steps 1-4 of the definitive comparison +
# finish TARP. Two parallel GPU branches + Phase C at the end. Each step is
# independent (failures logged, never cascade). Markers under $ROOT.
set -u -o pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi
cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
NDE="$REPO/scripts/sbi/train_jaxili_from_compressed.py"
NPZ="$REPO/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
DC="$REPO/scripts/sbi/results/exploratory/definitive_comparison"
ROOT="$DC/steps_overnight_2026_05_31"; mkdir -p "$ROOT/logs"
stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }; log(){ echo "[$(stamp)] $*" | tee -a "$ROOT/steps.log"; }
RNVP_AC="$DC/phaseA_tfdata_2026_05_30/compressed/autocross_s41"
RNVP_AO="$DC/phaseA_tfdata_2026_05_30/compressed/autoonly_s41"

# ---------- GPU 1 branch: L1 TARP, then re-plot CNN+L1 together ----------
gpu1_branch(){
  local G=1
  log "[gpu1] L1 TARP dumps (autocross+autoonly split70)"
  for arm in l1_autocross l1_autoonly; do
    cdir="$DC/compressed/${arm}_split70_dv"
    [ -f "$cdir/l1_train.npz" ] || { log "[gpu1] SKIP $arm (no $cdir)"; continue; }
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONUNBUFFERED=1 \
    "$PY" -u tarp_from_compressed.py --compressed-dir "$cdir" --arm-label "$arm" \
      --dumps-root "$DC/tarp_2026_05_31/dumps" --seed 41 --n-sims 200 --m-samples 2000 \
      --cuda-visible-devices "$G" > "$ROOT/logs/tarp_${arm}.log" 2>&1 \
      && log "[gpu1] TARP DONE $arm" || log "[gpu1] TARP FAIL $arm"
  done
  log "[gpu1] re-plotting all TARP arms (CNN+L1, dim 3+6)"
  "$PY" -u run_tarp_coverage.py --dumps-root "$DC/tarp_2026_05_31/dumps" \
    --outdir "$DC/tarp_2026_05_31" > "$ROOT/logs/tarp_replot.log" 2>&1 \
    && log "[gpu1] TARP replot DONE" || log "[gpu1] TARP replot FAIL"
  touch "$ROOT/.GPU1_DONE"
}

# ---------- GPU 0 branch: step3 (native-auto), step2 (std), step1 (multi-perm) ----------
COMMON_TFDS=(--map-kind nbody --cnn-map-route tfds
  --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48
  --field-size 20 --field-npix 160 --nbins 4 --tomo-bin-indices 1,2,3,4 --zero-mean-maps
  --compressor-arch plain --compressor-dim 10 --compressor-conv-channels 64,128,256 --compressor-dense-width 256
  --compressor-train-split 'train[:70%]' --compressor-val-split test
  --nde-train-split 'train[70%:]' --nde-val-split test
  --compressor-lr 5e-4 --compressor-batch-size 128 --compressor-checkpoint-policy best_val)

step3_native_auto(){
  local G=0 sdir="$ROOT/nativeauto/compressors/s41" cdir="$ROOT/nativeauto/compressed/s41"
  mkdir -p "$sdir" "$cdir"
  log "[step3] native-TFDS-auto compressor (RealNVP, 80k)"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 PYTHONUNBUFFERED=1 \
  "$PY" -u "$CNN" --train-compressor --exit-after-compress "${COMMON_TFDS[@]}" \
    --seed 41 --compressor-steps 80000 --compressor-save-every 2000 \
    --cuda-visible-devices "$G" --save-dir "$sdir" --cache-dir "$cdir" \
    > "$ROOT/logs/step3_compressor.log" 2>&1
  if [ -f "$cdir/cnn_train.npz" ]; then
    log "[step3] compressor DONE; NDE (seeds 41,42,43)"
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONUNBUFFERED=1 \
    "$PY" -u "$NDE" --compressed-dir "$cdir" --arm-label cnn_auto_native_rnvp \
      --output-dir "$DC/phaseB_nativeauto_2026_05_31/posteriors" --seeds 41,42,43 \
      --cuda-visible-devices "$G" > "$ROOT/logs/step3_nde.log" 2>&1 \
      && log "[step3] NDE DONE" || log "[step3] NDE FAIL"
  else log "[step3] compressor FAIL (see step3_compressor.log)"; fi
}

step2_standardization(){
  local G=0
  log "[step2] standardization NDE on existing RealNVP auto+cross cache"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONUNBUFFERED=1 \
  "$PY" -u "$NDE" --compressed-dir "$RNVP_AC" --arm-label cnn_autocross_rnvp_std \
    --output-dir "$DC/phaseB_std_2026_05_31/posteriors" --seeds 41,42,43 \
    --standardize-summary --cuda-visible-devices "$G" \
    > "$ROOT/logs/step2_std.log" 2>&1 && log "[step2] DONE" || log "[step2] FAIL"
}

# multi-perm: cheap obs-only recompress (limit train/val to 1 realization) at perm p
recompress_obs(){  # <arm> <compressor_save_dir> <perm> -> echoes obs npz path or empty
  local arm="$1" save="$2" p="$3"
  local tmp="$ROOT/multiperm/${arm}_p${p}_tmp"; mkdir -p "$tmp"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONUNBUFFERED=1 \
  "$PY" -u "$CNN" --no-train --exit-after-compress \
    --map-kind nbody --cnn-map-route harmonic --full-sphere-cross-cache "$NPZ" \
    --harmonic-cache-regime nobnt --harmonic-normalize-input-channels --zero-mean-maps \
    --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48 \
    --field-size 20 --field-npix 160 --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --compressor-arch plain --compressor-dim 10 --compressor-conv-channels 64,128,256 --compressor-dense-width 256 \
    --channel-mode "$4" --harmonic-obs-perm "$p" \
    --harmonic-train-realizations-limit 1 --harmonic-val-realizations-limit 1 \
    --compressor-train-split 'train[:70%]' --compressor-val-split val \
    --nde-train-split 'train[70%:]' --nde-val-split val \
    --cuda-visible-devices 0 --save-dir "$save" --cache-dir "$tmp" \
    > "$ROOT/logs/multiperm_${arm}_p${p}.log" 2>&1 || true
  [ -f "$tmp/cnn_obs.npz" ] && echo "$tmp/cnn_obs.npz" || echo ""
}

step1_multiperm(){
  log "[step1] multi-perm: validate obs recompress on (autocross, perm1) first"
  local save_ac="$DC/phaseA_tfdata_2026_05_30/compressors/autocross_s41"
  local o1; o1=$(recompress_obs autocross "$save_ac" 1 auto_cross)
  if [ -z "$o1" ]; then log "[step1] GATE FAIL: obs recompress produced no cnn_obs.npz; SKIPPING multi-perm (CNN stays perm0)"; return 0; fi
  log "[step1] GATE PASS ($o1). Fanning out perms 1,2 for autocross+autoonly."
  declare -A SAVE=( [autocross]="$save_ac" [autoonly]="$DC/phaseA_tfdata_2026_05_30/compressors/autoonly_s41" )
  declare -A MODE=( [autocross]=auto_cross [autoonly]=auto_only )
  declare -A FULLCACHE=( [autocross]="$RNVP_AC" [autoonly]="$RNVP_AO" )
  for arm in autocross autoonly; do
    obs="$ROOT/multiperm/${arm}_obs"; mkdir -p "$obs"
    cp "${FULLCACHE[$arm]}/cnn_obs.npz" "$obs/p0.npz" 2>/dev/null || true
    ok=1
    for p in 1 2; do
      f=$(recompress_obs "$arm" "${SAVE[$arm]}" "$p" "${MODE[$arm]}")
      [ -n "$f" ] && cp "$f" "$obs/p${p}.npz" || ok=0
    done
    if [ -f "$obs/p0.npz" ] && [ -f "$obs/p1.npz" ] && [ -f "$obs/p2.npz" ]; then
      log "[step1] $arm: 3 obs perms ready; NDE multi-perm"
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONUNBUFFERED=1 \
      "$PY" -u "$NDE" --compressed-dir "${FULLCACHE[$arm]}" --arm-label "${arm}_multiperm" \
        --output-dir "$DC/phaseB_multiperm_2026_05_31/posteriors" --seeds 41,42,43 \
        --obs-files "$obs/p0.npz,$obs/p1.npz,$obs/p2.npz" --cuda-visible-devices 0 \
        > "$ROOT/logs/step1_nde_${arm}.log" 2>&1 && log "[step1] $arm NDE DONE" || log "[step1] $arm NDE FAIL"
    else log "[step1] $arm: missing obs perms (ok=$ok); skipping"; fi
  done
}

gpu0_branch(){ step3_native_auto; step2_standardization; step1_multiperm; touch "$ROOT/.GPU0_DONE"; }

log "=== STEPS 1-4 autonomous block START ==="
gpu1_branch & G1=$!
gpu0_branch & G0=$!
wait $G1; wait $G0
log "=== branches done; Phase C aggregation ==="
"$PY" -u aggregate_all_arms.py > "$ROOT/logs/phase_c.log" 2>&1 && log "[phase-c] DONE" || log "[phase-c] FAIL"
log "=== STEPS 1-4 autonomous block COMPLETE ==="; touch "$ROOT/.STEPS_DONE"
