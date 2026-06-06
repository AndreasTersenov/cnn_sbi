# PLAN — Phase B-1: CNN auto+cross loader on the unified 10° TFDS

Status: **PLAN (awaiting Andreas sign-off).** Created 2026-06-05.
Campaign: `definitive-l1-vs-cnn-10deg-2026-06`. Prereq: Phase A PASSED (dataset
validated — `…/validate_10deg/report_{scales,disjoint,bitmatch}.json`).

## Goal

Make `npe_cnn_nbody_tomo.py` train the CNN-VMIM compressor on the **10-channel
auto+cross maps read directly from the unified TFDS** (`nbody_cosmogrid_dataset_tomo_cross/
grid_10deg_80px_nonoverlap180`), with **no grid `.npz` cache** (deleted by design)
and a **clean compressor↔NDE split by `cosmo_idx`**. Unblocks the A4 smoke and all
CNN Phase-C arms. The L1 loader (B-2) and the auto-only channel slice (B-3) follow.

## Why this is needed (route reality, confirmed by reading the code)

The CNN has two existing routes, neither of which reads our dataset out of the box:
- **`tfds` route** (`load_observed_map` / `:782`): 4-channel **auto** only (stacks
  `stage3_lensing{1..4}`) — the separate `NbodyCosmogridDatasetTomo` the 20° auto-only
  arm used. Does not read cross channels.
- **`harmonic` route** (`--full-sphere-cross-cache`, `:3673`): 10 channels but
  **requires the grid `.npz` cache** for the train stream, channel-RMS, obs, and the
  split audit (`:3667`). The 10° grid cache was deleted in Phase 4.

So reading the unified 10-ch TFDS is genuinely new wiring — there is **no zero-code
smoke**. (This corrects the earlier "free CNN smoke parallel to Phase B" framing.)

## What already exists (so the change is small)

- **`tfds_cross_tfdata_loader.build_tfds_tfdata_iterator`** (`:28`) already reads our
  schema (`map_nbody` `[H,W,10]`, `theta`) via `tfds.load(name, data_dir, read_config)`,
  applies `channel_slice` `[lo:hi]`, **divides** by `channel_scale`, flips, and does
  `theta[3]·0.01 → h0`. It even carries the **read_config retune** the handoff asked for
  (`interleave_cycle_length=8, block_length=16` — fixes the 2112-shard throughput
  collapse). It is already wired into `main()` at `:4109`.
- **`load_observed_from_harmonic_cache`** (`:1213`) reads the obs patch from a
  `…/<regime>/obs/<cosmo_id>_perm<perm>.npz` — i.e. **the fiducial cache we KEPT**
  serves obs unchanged, with `channel_slice` + `channel_scale` already applied (`:1240`).
- **`--channel-mode {auto_cross,cross_only,auto_only}`** (`:726`) + `cnn_channel_slice`
  already resolve the slice; the loader honors it. (B-3 = just allow it on this route.)

## The 4 gaps to close (B-1)

1. **Split compressor↔NDE by `perm` (example-disjoint, cosmology-SHARED).**
   `build_tfds_tfdata_iterator` currently selects only the TFDS split. The compressor and
   NDE both draw from the **`train`** split (cosmo 1–899); they must read **disjoint
   examples**. Per Andreas (2026-06-05): the demonstrated 20° leak (`imap_unordered` →
   ~1.6× FoM, `project_tfdata_cross_route_leakage`) was the *same* (cosmo,perm,patch) map
   in both — that is what must not happen. Sharing *cosmologies* is fine and better (both
   keep all 899 cosmos → denser θ coverage for the flow, cleaner L1-vs-CNN comparison; the
   residual "compressor memorizes a cosmology" risk is second-order and is what SBC/L-C2ST
   catch, with the fiducial held out from both). Split: **compressor perms 0–4 (71%), NDE
   perms 5–6 (29%)** — the train split is uniform 899 cosmos × 7 perms × 180 patches, so
   this is example-disjoint, balanced ~70/30 (matches the 20° ratio), and is the *clean*
   version of what 20° intended.
2. **`channel_scale` from a TFDS sample** (no grid cache to read it from).
3. **obs from the fiducial cache** (path-only change; loader reused as-is).
4. **Split audit by cosmo_idx** (replaces the cache file-set audit).

## Design — a new `tfds_cross` route (keeps the cache guards untangled)

Add `--cnn-map-route tfds_cross` as a third route so we never trip the
`--full-sphere-cross-cache`-required guards. New/changed surfaces, all in
`npe_cnn_nbody_tomo.py` + the small loader module:

### New args
- `--cnn-map-route tfds_cross` (extend the existing choices).
- `--cross-tfds-name` (default `nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180`).
- `--cross-tfds-data-dir` (default `/home/tersenov/tensorflow_datasets`).
- `--fiducial-obs-cache <dir>` (the kept fiducial cache; obs source).
- `--cnn-perm-split "0-4:5-6"` → compressor perms : NDE perms, both on the **`train`**
  split (all 899 cosmos). val/NDE-val = the TFDS **`test`** split (cosmo 900–1299), all perms.
- Reuse: `--harmonic-cache-regime`, `--harmonic-normalize-input-channels`,
  `--harmonic-obs-perm`, `--harmonic-obs-patch-idx`, `--channel-mode`.

### Loader change (`tfds_cross_tfdata_loader.build_tfds_tfdata_iterator`)
Add `perm_lo:int|None, perm_hi:int|None`. When set, insert **before**
`repeat()/shuffle()`:
```python
ds = ds.filter(lambda ex: tf.logical_and(ex["perm"] >= lo, ex["perm"] <= hi))
```
(`perm` is already in the TFDS; filter is a cheap int compare; drops ~3/7 or 4/7 of
records but reads all shards — fine.) `_transform` then strips the scalar keys.

### New helper — `compute_cross_tfds_channel_rms(name, data_dir, regime, n_sample=8000)`
Stream `n_sample` **train-split** examples (no shuffle needed), accumulate per-channel
`sqrt(mean(x²))` over the 10 channels → `(10,)` float32; **JSON-cache** it next to the
results dir (keyed by name+regime+n_sample). Sanity-assert against the fiducial-cache
bounds (auto∈[3e-3,2e-2], cross∈[5e-8,2e-6]) — these are the Phase-A-measured scales.
This is the `--harmonic-normalize-input-channels` source.

### New helper — `compress_dataset_from_cross_tfds(...)`
Analogue of `compress_dataset_from_harmonic_cache` (`:1570`): iterate the TFDS
**filtered to a perm range**, deterministic order, **no flip/shuffle**, apply
`channel_slice`+`channel_scale`, run the trained compressor → `(summary, theta)`.
Produces the NDE train/val compressed arrays (the `compress_dataset_from_harmonic_cache`
call sites at `:4293/:4307` get a route branch).

### New audit — `audit_cross_perm_split(comp_perms, nde_perms)`
Assert the two perm sets are disjoint (structural; logged to meta) and, as a belt-and-
braces check on a sample, that no `(cosmo_idx, perm, patch)` tuple appears in both
streams. Replaces `audit_harmonic_split_overlap` for this route.
(`--require-disjoint-train-examples` is satisfied by construction.)

### `main()` branch
A `cnn_map_route == "tfds_cross"` block that: resolves `channel_scale` via the new RMS
helper; builds train/val iterators (compressor: `train`/perms 0–4; compressor-val:
`test`; NDE-train: `train`/perms 5–6; NDE-val: `test`); loads obs from
`--fiducial-obs-cache`; runs the perm audit; routes compression through the new compress
fn. No `--full-sphere-cross-cache` required.

## Decisions to confirm
- **Compressor↔NDE split:** **by perm** — compressor perms **0–4** (71%), NDE perms
  **5–6** (29%), both on `train` (all 899 cosmos); val = `test` (900–1299, all perms).
  Example-disjoint, cosmology-shared (Andreas 2026-06-05).
- **channel_scale source:** a **grid TFDS train sample** (principled — matches training
  data), not the fiducial cache. Sanity-bounded against the fiducial scales.
- **Compressor arch for the smoke:** **`plain`** (64,128,256 / dense 256 / cdim 10) —
  matches the 20° auto+cross arm exactly; CLAUDE.md permits `plain` for 10-ch.

## Verification oracle (back-pressure)

**Unit checks (CPU, fast), gate the smoke:**
1. `channel_scale` from TFDS sample within the fiducial bounds (auto 7e-3–1e-2, cross
   2e-7–7e-7); print the 10 values.
2. The compressor and NDE **perm** sets, read back from their iterators on a few batches,
   are **disjoint** ({0–4} vs {5–6}); both span all 899 cosmos; and no shared
   `(cosmo,perm,patch)` tuple across the two streams (sampled check).
3. One `compress_dataset_from_cross_tfds` batch → summary shape `(B, cdim)`, finite.

**A4 smoke (GPU 1, the deferred sanity gate):** reduced CNN auto+cross run.
- **Decision oracle:** typical-patch posterior **~2× wider** than the 20° CNN auto+cross
  (20°: σ(w0) 0.167, σ(Ωm) 0.027, FoM3 24453). Expect σ(w0)~0.30–0.40, σ(Ωm)~0.05–0.06,
  **FoM3 ~3k** (≈ 24k ÷ ~8 from 2× σ widening). **FAIL** = NaN / degenerate / FoM3 > 10k.
- **Exact command (for sign-off before launch):**
```bash
cd /mnt/home/tersenov/software/cnn_sbi/scripts/sbi
FID=results/exploratory/cross_maps_campaign/full_sphere_cache_fiducial_10deg
OUT=results/exploratory/definitive_comparison_10deg/smoke_autocross
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
PYTHONUNBUFFERED=1 /home/tersenov/anaconda3/envs/jaxili/bin/python npe_cnn_nbody_tomo.py \
  --train-compressor \
  --cnn-map-route tfds_cross \
  --cross-tfds-name nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180 \
  --cross-tfds-data-dir /home/tersenov/tensorflow_datasets \
  --fiducial-obs-cache "$FID" --harmonic-cache-regime nobnt \
  --harmonic-normalize-input-channels --channel-mode auto_cross \
  --cnn-perm-split 0-4:5-6 \
  --harmonic-obs-perm 0 --harmonic-obs-patch-idx 90 \
  --zero-mean-maps --map-kind nbody --seed 42 \
  --field-size 10 --field-npix 80 --nbins 4 --tomo-bin-indices 1,2,3,4 \
  --compressor-arch plain --compressor-dim 10 --compressor-dense-width 256 \
  --compressor-conv-channels 64,128,256 \
  --compressor-steps 8000 --compressor-batch-size 128 --compressor-lr 0.0005 \
  --compressor-checkpoint-policy best_val \
  --npe-samples 100000 --no-wandb \
  --cuda-visible-devices 1 \
  --save-dir "$OUT" --cache-dir "$OUT/cache" \
  --posterior-out "$OUT/posterior.npy" --figure-out "$OUT/corner.pdf"
```
(`--compressor-steps 8000` ≈ 25–35 min for the smoke; Phase C uses the 20° budget 80000.
`--harmonic-obs-patch-idx 90` = a mid-latitude non-polar tile; final typical-patch stats
median over many patches as at 20°.)

## Risks / non-goals
- **Risk:** `ds.filter` on perm after the read_config interleave could interact with
  the shuffle-buffer starvation fix. Mitigate: filter before `repeat().shuffle()`; the
  unit check #2 (disjoint perm sets) + a throughput print catch it.
- **Risk:** channel-RMS from a finite sample vs the (gone) full-grid cache RMS. Mitigate:
  n_sample=8000 (~5 cosmos × … patches), JSON-cached, bounded vs fiducial scales; RMS is
  a stable population stat (Phase A sample of 400 already matched the cache to ~2%).
- **Non-goal (B-1):** L1 loader, auto-only slice, the 3-seed Phase-C run, diagnostics.
- **Non-goal:** touching the harmonic-cache or tfds(auto) routes (left intact for 20°
  reproducibility).

## After sign-off
Implement B-1 → run the CPU unit checks → run the A4 smoke (GPU 1) → report FoM3 vs the
20° expectation. Then B-2 (L1 TFDS loader; `channel_empirical_global`, PCA off, datavector
verified vs a cache-route reference) and B-3 (auto-only slice = `--channel-mode auto_only`
on this route + `channel_scale[:4]`). No `git add` / commit without Andreas's OK.
