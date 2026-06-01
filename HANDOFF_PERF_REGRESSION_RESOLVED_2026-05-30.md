# HANDOFF — CNN perf regression RESOLVED + dataset/loader updated (2026-05-30)

**For:** the "definitive L1-vs-CNN" session (the one stuck at ~1.1 it/s / ~0% GPU,
reading `HANDOFF_CNN_TFRECORD_PERF_REGRESSION.md`).
**From:** the dataset-format / loader-rebuild session, 2026-05-30.
**Status:** the perf regression is **solved**, the loader was **rebuilt + a real bug
fixed**, and the **old path you were using has been deleted**. Read this before you fire
any retrain — your last plan would have hit the same ~1 it/s wall.

---

## 0. TL;DR (the one thing that matters)

The ~1.1 it/s / ~0% GPU was **NOT** DLPack, jax 0.5.0, mem-fraction, or thread count. The
root cause is **storage**: the cross dataset lived on **`/nas`, a FUSE `mergerfs` mount**
that delivers only ~100 MB/s for tf.data's many-shard random-read pattern → the GPU starves.
(Your "1237 threads / blocked" finding was a real but secondary symptom: those threads were
*asleep waiting on FUSE reads*, which is why CPU looked idle and iowait was ~0.)

**Fix: read the dataset from local xfs, not `/nas`.** The full 421 GB cross dataset has been
copied (byte-exact) to:

```
/home/tersenov/tensorflow_datasets/nbody_cosmogrid_dataset_tomo_cross/
```

On local xfs the standard `tfds.load`+tf.data loader runs **~22 it/s (pure)** and full
compressor **training ~15–19 it/s (cross_only) / ~12–13 it/s (auto_cross)** — measured,
no special thread tuning, GPU fed. An 80k-step compressor is **~1.5 h, not ~20 h**.

**If you point `--cross-tfdata-dir` at `/nas/...` you will get ~1 it/s again.** Point it at
the local copy above.

---

## 1. What changed since your last state (the deltas)

1. **Root cause = storage (above).** `HANDOFF_CNN_TFRECORD_PERF_REGRESSION.md` is now
   **historical** — its DLPack / thread-oversubscription diagnoses were about the old
   hand-rolled path and are superseded.

2. **A real normalization bug in the new loader was found and FIXED** (commit `ad75511`).
   `tfds_cross_tfdata_loader.py` was **multiplying** maps by the per-channel RMS instead of
   **dividing** — collapsing the ~1e-7 cross channels to ~1e-14 (cross signal numerically
   invisible). **Any `--cross-tfdata-dir` run started before `ad75511` is scientifically
   invalid and must be redone.** Make sure you are on current `HEAD` (≥ `676d407`).

3. **The old loader paths are DELETED** (commit `676d407`):
   - `--harmonic-tfrecord-dir` (hand-rolled TFRecord reader), `--grain-tfds-dir`,
     `--grain-num-workers`, `--harmonic-tfrecord-compression` flags — **gone**.
   - `grain_loader.py`, `build_harmonic_tfrecord.py`, `tests/test_tfrecord_*.py` — **gone**.
   - The old TFRecord at `/nas/tersenov/harmonic_tfrecord/full_sphere_cache_grid` is
     orphaned. Do **not** reference any of these.
   - **Kept:** `--cross-tfdata-dir` (the winner) and `--grain-tfds-name` (it just names the
     cross TFDS dataset for the tf.data path).

4. **The loader decision is settled:** standard `tfds.load` + tf.data on the TFRecord cross
   dataset (mirrors the fast auto-only route). Validated bit-exact + cross_only FoM3 matches
   the legacy `.npz` baseline. Full detail:
   `scripts/sbi/results/exploratory/loader_validation_2026_05_30/VALIDATION_STATUS.md`.

---

## 2. Corrected, ready-to-run command (your Arm 1, auto+cross)

Two changes from your last plan: `--cross-tfdata-dir` → **local**, and you must be on
current HEAD. Everything else of your heavy config is unchanged.

```bash
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
NPZ=/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/cross_maps_campaign/full_sphere_cache_grid
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
"$PY" -u scripts/sbi/npe_cnn_nbody_tomo.py \
  --cuda-visible-devices 0 --train-compressor \
  --map-kind nbody --tfds-name NbodyCosmogridDatasetTomo/grid_20deg_160px_nonoverlap48 \
  --field-size 20 --field-npix 160 --nbins 4 --tomo-bin-indices 1,2,3,4 \
  --full-sphere-cross-cache "$NPZ" \
  --cross-tfdata-dir /home/tersenov/tensorflow_datasets \
  --grain-tfds-name nbody_cosmogrid_dataset_tomo_cross/grid_20deg_160px_nonoverlap48 \
  --channel-mode auto_cross --harmonic-cache-regime nobnt \
  --harmonic-normalize-input-channels --zero-mean-maps \
  --compressor-arch plain --compressor-dim 10 \
  --compressor-conv-channels 64,128,256 --compressor-dense-width 256 \
  --compressor-train-split 'train[:70%]' --compressor-val-split val \
  --nde-train-split 'train[70%:]' --nde-val-split val \
  --compressor-lr 5e-4 --compressor-batch-size 128 --compressor-checkpoint-policy best_val \
  --seed 41 --compressor-steps 80000 --compressor-save-every 1000 --exit-after-compress \
  --save-dir <out>/save_params --cache-dir <out>/cache
```

**Arm 2 (auto-only):** same command, `--channel-mode auto_only`. As you noted, the first
auto_only run pays a one-time `.npz` channel-RMS scan for the `[0:4]` slice (cache miss),
then it's fast. Your "shared route+shuffle regime so the auto-vs-cross gain isn't
confounded" rationale is sound — keep it.

`--full-sphere-cross-cache <NPZ>` is still required (channel-RMS, observed data, split
audit). The `.npz` cache is local already, so those are fast.

---

## 3. Sanity check (do your 500-step smoke first — "benchmark, don't assume")

Your plan to smoke 500 steps and measure it/s on your actual GPU/MEM setup is exactly
right. Expected on the **local** path: **~15–19 it/s**, GPU fed. If you see **~1 it/s /
~0% GPU**, you are still reading `/nas` (or on a pre-fix checkout) — stop and fix the
data dir / pull HEAD. `MEM_FRACTION=0.30` is plenty: the model only uses ~3–5 GB.

You do **not** need the `OMP/TF_NUM_*` thread prefix on the local path; it ran ~15–19 it/s
with defaults. (If under very heavy node load you still see thread thrash, `CNN_TF_THREADS=8`
was the lever last time — but storage was the real wall, not threads.)

---

## 4. ⚠️ Methodological warning for the definitive comparison (train[:70%] split)

Your config uses `--compressor-train-split train[:70%]` + `--nde-train-split train[70%:]`
(disjoint compressor/NDE). On the **new TFRecord path this disjointness is NOT preserved**:
the loader slices `train[:70%]` by **example** (the build's arbitrary realization order),
while `compress_dataset`/NDE reads `.npz train[70%:]` by **file** order. The two slices
overlap in realization space → the compressor trains on realizations the NDE then uses →
**leakage → over-confident / inflated FoM** (I measured an auto_cross FoM3 inflated ~1.6×
from exactly this).

- For your **auto-vs-cross relative** comparison it may be acceptable (both arms share the
  same artifact if run identically), but **absolute** FoM will be inflated and **not
  comparable to historical `.npz`-path numbers**.
- Cleanest fixes: use **full `train`** for the compressor on both arms (no 70/30 slice), or
  match the realization subsets explicitly. Your call scientifically — but be aware of it
  before you read the contours as truth. (The in-run "split audit overlap=0" is computed on
  `.npz` files, NOT on what the compressor actually trained from, so it will falsely report
  clean.)

---

## 5. GPU / coordination

GPU situation on this node is shared and fluid (I've seen `gkogkou`/`alahiry` holding
~25–26 GB on GPU 0 at various times; GPU 1 has been the L1 card). **Check `nvidia-smi`
yourself and coordinate the card with Andreas** — I'm not dictating GPU policy for your
run. The model is small (~3–5 GB), so sharing a card at `MEM_FRACTION=0.30` is fine if the
co-tenant leaves headroom.

---

## 6. Pointers
- Resolution + numbers: `scripts/sbi/results/exploratory/loader_validation_2026_05_30/VALIDATION_STATUS.md`
- Commits: `ad75511` (normalization fix + `tests/gate1_batch_parity.py`), `676d407` (dead-path removal).
- The local dataset: `/home/tersenov/tensorflow_datasets/nbody_cosmogrid_dataset_tomo_cross/` (byte-exact copy of the `/nas` one; `/nas` copy kept as backup).
- This supersedes `HANDOFF_CNN_TFRECORD_PERF_REGRESSION.md`.
