#!/usr/bin/env bash
# Step 1 (multi-perm) — FIXED. The orchestrator's gate failed because it used
# --no-train (that's for the flow) + --save-dir, which fell back to a default
# checkpoint path. Correct way: OMIT --train-compressor and pass
# --compressor-params/--compressor-state pointing at the actual best_val.pkl.
# Recompress the fiducial obs at perms 1,2 (train/val limited to 1 realization
# => fast, obs-only), then NDE with --obs-files p0,p1,p2 (3 seeds × 3 perms).
set -u -o pipefail
REPO=/mnt/home/tersenov/software/cnn_sbi
cd "$REPO/scripts/sbi"
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
CNN="$REPO/scripts/sbi/npe_cnn_nbody_tomo.py"
NDE="$REPO/scripts/sbi/train_jaxili_from_compressed.py"
NPZ="$REPO/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid"
DC="$REPO/scripts/sbi/results/exploratory/definitive_comparison"
CMP="$DC/phaseA_tfdata_2026_05_30/compressors"
ROOT="$DC/multiperm_fixed_2026_05_31"; mkdir -p "$ROOT/logs"
GPU=0
stamp(){ date -u +'%Y-%m-%dT%H:%M:%SZ'; }; log(){ echo "[$(stamp)] $*" | tee -a "$ROOT/mp.log"; }

declare -A FULLCACHE=( [autocross]="$DC/phaseA_tfdata_2026_05_30/compressed/autocross_s41"
                       [autoonly]="$DC/phaseA_tfdata_2026_05_30/compressed/autoonly_s41" )
declare -A MODE=( [autocross]=auto_cross [autoonly]=auto_only )
declare -A CHDIR=( [autocross]=harmonic_nobnt_ch10 [autoonly]=harmonic_nobnt_ch4 )

recompress_obs(){  # <arm> <perm> -> echoes obs npz path or empty
  local arm="$1" p="$2"
  local base="$CMP/${arm}_s41/vmim/nbody/sigma_0.26/gal_density_30/bin_4/${CHDIR[$arm]}"
  local pp="$base/params_nd_compressor_best_val.pkl" sp="$base/opt_state_resnet_best_val.pkl"
  local tmp="$ROOT/${arm}_p${p}_tmp"; mkdir -p "$tmp"
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONUNBUFFERED=1 \
  "$PY" -u "$CNN" --exit-after-compress \
    --map-kind nbody --cnn-map-route harmonic --full-sphere-cross-cache "$NPZ" \
    --harmonic-cache-regime nobnt --harmonic-normalize-input-channels --zero-mean-maps \
    --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48 \
    --field-size 20 --field-npix 160 --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --compressor-arch plain --compressor-dim 10 --compressor-conv-channels 64,128,256 --compressor-dense-width 256 \
    --channel-mode "${MODE[$arm]}" --harmonic-obs-perm "$p" \
    --harmonic-train-realizations-limit 1 --harmonic-val-realizations-limit 1 \
    --compressor-train-split 'train[:70%]' --compressor-val-split val \
    --nde-train-split 'train[70%:]' --nde-val-split val \
    --compressor-params "$pp" --compressor-state "$sp" \
    --cuda-visible-devices "$GPU" --cache-dir "$tmp" \
    > "$ROOT/logs/recompress_${arm}_p${p}.log" 2>&1 || true
  [ -f "$tmp/cnn_obs.npz" ] && echo "$tmp/cnn_obs.npz" || echo ""
}

log "=== multi-perm FIXED START; gate on (autocross, perm1) ==="
g=$(recompress_obs autocross 1)
if [ -z "$g" ]; then log "GATE STILL FAILS (see logs/recompress_autocross_p1.log) — aborting"; touch "$ROOT/.MP_ABORTED"; exit 0; fi
log "GATE PASS ($g). Fanning out."
for arm in autocross autoonly; do
  obs="$ROOT/${arm}_obs"; mkdir -p "$obs"
  cp "${FULLCACHE[$arm]}/cnn_obs.npz" "$obs/p0.npz" 2>/dev/null || true
  for p in 1 2; do f=$(recompress_obs "$arm" "$p"); [ -n "$f" ] && cp "$f" "$obs/p${p}.npz"; done
  if [ -f "$obs/p0.npz" ] && [ -f "$obs/p1.npz" ] && [ -f "$obs/p2.npz" ]; then
    log "$arm: 3 obs perms ready; NDE multi-perm (3 seeds × 3 perms)"
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 PYTHONUNBUFFERED=1 \
    "$PY" -u "$NDE" --compressed-dir "${FULLCACHE[$arm]}" --arm-label "${arm}_multiperm" \
      --output-dir "$DC/phaseB_multiperm_2026_05_31/posteriors" --seeds 41,42,43 \
      --obs-files "$obs/p0.npz,$obs/p1.npz,$obs/p2.npz" --cuda-visible-devices "$GPU" \
      > "$ROOT/logs/nde_${arm}.log" 2>&1 && log "$arm NDE DONE" || log "$arm NDE FAIL"
  else log "$arm: missing obs perms; skipping"; fi
done
log "=== multi-perm FIXED COMPLETE; refreshing Phase C ==="
"$PY" -u aggregate_all_arms.py > "$ROOT/logs/phase_c.log" 2>&1 || true
touch "$ROOT/.MP_DONE"
